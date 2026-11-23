"""Exercise online generation and embedding through mocked OpenAI HTTP protocols."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from openai import OpenAI
import yaml

from scripts import generate_answer, query_settings, retrieve_skills


class TimedSSE(httpx.SyncByteStream):
    def __init__(self, events, clock):
        self.events = events
        self.clock = clock

    def __iter__(self):
        for at, delta, finish in self.events:
            self.clock[0] = at
            event = {
                "id": "chat-test", "object": "chat.completion.chunk", "created": 1,
                "model": "qwen-api", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            yield f"data: {json.dumps(event)}\n\n".encode()
        self.clock[0] = 17.0
        yield b"data: [DONE]\n\n"


class OnlineAPITests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="persona-online-api-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.embedding_settings = {
            "api_base": "http://embeddings.example.test/v1", "api_key": "test-embedding-key",
            "model": "bge-api", "request_timeout": 30.0,
        }
        self.llm_settings = {
            "api_base": "http://chat.example.test/v1", "api_key": "test-chat-key",
            "model": "qwen-api", "request_timeout": 45.0,
        }
        self.profile = {
            "explicit": {"age_group": "older adult", "gender_presentation": None, "role": None},
            "implicit": {"personality": None, "values": None, "communication_preferences": None},
        }

    def set_client(self, model, settings, handler):
        model.client.close()
        client = OpenAI(
            base_url=settings["api_base"], api_key=settings["api_key"], max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        model.client = client
        self.addCleanup(client.close)
        return model

    def skill_files(self):
        directory = self.directory / "skills"
        for name in ("alpha", "beta", "boundary", "low"):
            path = directory / name / "SKILL.md"
            path.parent.mkdir(parents=True)
            path.write_text(
                f"---\nname: {name}\nmetadata:\n  source-rule: '01'\n---\n\n"
                f"# {name}\n\n## When\n\nWhen {name} applies.\n\n## Act\n\nOffer clear guidance.\n",
                encoding="utf-8",
            )
        return directory

    @staticmethod
    def embedding_response(vectors):
        return httpx.Response(200, json={
            "object": "list", "model": "bge-api",
            "data": [{"object": "embedding", "index": index, "embedding": vector}
                     for index, vector in reversed(list(enumerate(vectors)))],
            "usage": {"prompt_tokens": len(vectors), "total_tokens": len(vectors)},
        })

    def test_settings_keep_distinct_llm_and_embedding_endpoints(self):
        path = self.directory / "query_settings.yaml"
        expected = {"k": 1, "epsilon": 60, "llm": self.llm_settings, "embeddings": self.embedding_settings}
        path.write_text(yaml.safe_dump(expected), encoding="utf-8")
        actual = query_settings.load_query_settings(path)
        self.assertEqual(actual, expected)
        self.assertNotEqual(actual["llm"]["api_base"], actual["embeddings"]["api_base"])

    def test_embeddings_submit_model_and_text_and_restore_response_order(self):
        requests = []

        def handle(request):
            requests.append(request)
            return self.embedding_response([[3, 0, 0], [0, 4, 0]])

        model = self.set_client(retrieve_skills.BGEM3(self.embedding_settings), self.embedding_settings, handle)
        vectors = model.encode(["first persona", "second persona"])
        self.assertEqual(len(requests), 1)
        self.assertEqual(str(requests[0].url), "http://embeddings.example.test/v1/embeddings")
        payload = json.loads(requests[0].content)
        self.assertEqual(payload["model"], "bge-api")
        self.assertEqual(payload["input"], ["first persona", "second persona"])
        np.testing.assert_allclose(vectors, [[1, 0, 0], [0, 1, 0]], atol=1e-7)

    def test_online_query_requests_one_embedding_and_preserves_threshold_and_top_k(self):
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            return self.embedding_response([[5, 0, 0]])

        model = self.set_client(retrieve_skills.BGEM3(self.embedding_settings), self.embedding_settings, handle)
        skills = retrieve_skills.load_skills(self.skill_files())
        by_name = {"alpha": [1, 0, 0], "beta": [.8, .6, 0], "boundary": [.6, .8, 0], "low": [-1, 0, 0]}
        matrix = np.asarray([by_name[skill.name] for skill in skills], dtype=np.float64)
        first = retrieve_skills.retrieve_profile(model, skills, matrix, self.profile, 1, 60)
        wider = retrieve_skills.retrieve_profile(model, skills, matrix, self.profile, 3, 60)
        self.assertEqual([row["name"] for row in first["selected"]], ["alpha"])
        self.assertEqual([row["name"] for row in wider["selected"]], ["alpha", "beta"])
        self.assertEqual([row["name"] for row in wider["ranked"]], ["alpha", "beta", "boundary", "low"])
        np.testing.assert_allclose([row["score"] for row in wider["ranked"]], [100, 80, 60, -100], atol=1e-6)
        query = retrieve_skills.profile_query(self.profile)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(row["input"] == [query] and row["model"] == "bge-api" for row in requests))

    def test_offline_cache_reuses_matrix_and_invalidates_api_identity_and_old_signature(self):
        requests = []

        def handle(request):
            payload = json.loads(request.content)
            requests.append((str(request.url), payload["model"], payload["input"]))
            return self.embedding_response([[index + 1, 1, 0] for index, _ in enumerate(payload["input"])])

        directory = self.skill_files()
        index = self.directory / "when.npz"
        np.savez(index, signature="bge-m3-cls-l2-v1-old-local-signature", embeddings=np.zeros((4, 1024)))
        model = self.set_client(retrieve_skills.BGEM3(self.embedding_settings), self.embedding_settings, handle)
        skills, first = retrieve_skills.prepare_skill_index(model, directory, index)
        _, cached = retrieve_skills.prepare_skill_index(model, directory, index)
        self.assertEqual(len(requests), 1)
        np.testing.assert_array_equal(first, cached)
        self.assertEqual(first.shape, (4, 3))
        initial_signature = retrieve_skills.index_signature(skills, self.embedding_settings)
        changed_model = {**self.embedding_settings, "model": "new-bge-api"}
        changed_url = {**changed_model, "api_base": "http://second-embeddings.example.test/v1"}
        for settings in (changed_model, changed_url):
            replacement = self.set_client(retrieve_skills.BGEM3(settings), settings, handle)
            retrieve_skills.prepare_skill_index(replacement, directory, index)
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[1][1], "new-bge-api")
        self.assertEqual(requests[2][0], "http://second-embeddings.example.test/v1/embeddings")
        self.assertEqual(len({initial_signature, retrieve_skills.index_signature(skills, changed_model),
                              retrieve_skills.index_signature(skills, changed_url)}), 3)
        with np.load(index, allow_pickle=False) as saved:
            self.assertEqual(saved["signature"].item(), retrieve_skills.index_signature(skills, changed_url))

    def test_query_and_cached_skill_dimension_mismatch_is_reported(self):
        model = self.set_client(
            retrieve_skills.BGEM3(self.embedding_settings), self.embedding_settings,
            lambda request: self.embedding_response([[1, 0]]),
        )
        skills = retrieve_skills.load_skills(self.skill_files())
        with self.assertRaisesRegex(ValueError, "[Dd]imension"):
            retrieve_skills.retrieve_profile(model, skills, np.ones((4, 3)), self.profile, 1, 60)

    def test_stream_timing_starts_at_first_content_and_ends_after_stream(self):
        clock = [10.0]
        requests = []
        events = [(11.0, {"role": "assistant", "content": ""}, None),
                  (12.0, {"content": ""}, None), (14.0, {"content": " Hello"}, None),
                  (15.0, {"content": " world."}, None), (16.0, {}, "stop")]

        def handle(request):
            requests.append(request)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=TimedSSE(events, clock))

        model = self.set_client(generate_answer.AnswerModel(self.llm_settings), self.llm_settings, handle)
        with patch.object(generate_answer, "perf_counter", side_effect=lambda: clock[0]):
            answer, limited, timing = model.generate("Give concise advice.", 256, "English")
        self.assertEqual(answer, "Hello world.")
        self.assertFalse(limited)
        self.assertEqual(timing["ttrt_seconds"], 4.0)
        self.assertEqual(timing["generation_seconds"], 7.0)
        self.assertEqual(str(requests[0].url), "http://chat.example.test/v1/chat/completions")
        payload = json.loads(requests[0].content)
        self.assertEqual(payload["model"], "qwen-api")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": "Give concise advice."})

    def test_stream_length_finish_is_preserved(self):
        clock = [10.0]
        events = [(11.0, {"role": "assistant"}, None), (12.0, {"content": "Partial reply"}, None), (16.0, {}, "length")]
        model = self.set_client(
            generate_answer.AnswerModel(self.llm_settings), self.llm_settings,
            lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=TimedSSE(events, clock)),
        )
        answer, limited, _ = model.generate("A question", 256, "English")
        self.assertEqual(answer, "Partial reply")
        self.assertTrue(limited)

    def test_stream_empty_answer_raises(self):
        clock = [10.0]
        events = [(11.0, {"role": "assistant", "content": ""}, None), (16.0, {}, "stop")]
        model = self.set_client(
            generate_answer.AnswerModel(self.llm_settings), self.llm_settings,
            lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=TimedSSE(events, clock)),
        )
        with self.assertRaisesRegex(RuntimeError, "[Ee]mpty"):
            model.generate("A question", 256, "English")


if __name__ == "__main__":
    unittest.main()
