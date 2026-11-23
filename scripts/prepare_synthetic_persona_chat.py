"""Convert Synthetic-Persona-Chat to conversational examples, optionally using user-side personas."""

import argparse
import csv
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("D:/Data/Synthetic-Persona-Chat")
OUTPUT = ROOT / "data/synthetic_persona_chat"
USER_SIDE_OUTPUT = ROOT / "data/synthetic_persona_chat_user_side"
SPLITS = {
    "train": "Synthetic-Persona-Chat_train.csv",
    "validation": "Synthetic-Persona-Chat_valid.csv",
    "test": "Synthetic-Persona-Chat_test.csv",
    "new_persona_unsplit": "New-Persona-New-Conversations.csv",
}
PERSONA_COLUMNS = {"1": "user 1 personas", "2": "user 2 personas"}
CONVERSATION_COLUMN = "Best Generated Conversation"
SPEAKER = re.compile(
    r"^\s*(?P<markup>(?:[*_>]+\s*)*)User\s*(?P<speaker>\d+)\s*"
    r"(?:[*_]+\s*)*[:：]\s*(?:[*_]+\s+)*",
    re.IGNORECASE,
)
LETTER_SPEAKER = re.compile(r"^([AB])[.:]\s+", re.IGNORECASE)


def parse_dialogue(text):
    """Return the labelled turns, merging consecutive messages by the same speaker."""
    lines = text.splitlines()
    letter_format = not any(SPEAKER.match(line) for line in lines)
    turns = []
    leading_context = []
    for line in lines:
        match = LETTER_SPEAKER.match(line) if letter_format else SPEAKER.match(line)
        if match:
            speaker = {"A": "1", "B": "2"}[match[1].upper()] if letter_format else match["speaker"]
            content = line[match.end():].strip()
            if not letter_format and match["markup"].strip():
                content = re.sub(r"\s+[*_](?:\s*[*_])*\s*$", "", content)
            turns.append((speaker, [content]))
        elif line.strip():
            if turns:
                turns[-1][1].append(line.strip())
            else:
                leading_context.append(line.strip())

    merged = []
    for speaker, parts in turns:
        content = "\n".join(parts).strip()
        if not content:
            continue
        if merged and speaker == merged[-1][0]:
            merged[-1] = (speaker, f"{merged[-1][1]}\n{content}")
        else:
            merged.append((speaker, content))
    return "\n".join(leading_context), merged, letter_format


def make_examples(split, source_row, personas, context, turns):
    history = [f"Context: {context}"] if context else []
    for turn, (speaker, content) in enumerate(turns, 1):
        if turn > 1:
            persona = personas[speaker]
            system = (
                f"You are User {speaker} in a conversation. Reply as this speaker using the persona below "
                "and the conversation so far. Write only the next reply without a speaker label.\n\n"
                f"Your persona:\n{persona}"
            )
            conversation = "\n".join(history)
            yield {
                "id": f"{split}-{source_row:05d}-{turn:02d}",
                "source_row": source_row,
                "target_turn": turn,
                "target_speaker": f"User {speaker}",
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Conversation so far:\n{conversation}\n\nWrite your next reply."},
                ],
                "completion": [{"role": "assistant", "content": content}],
            }
        history.append(f"User {speaker}: {content}")


def make_user_side_examples(split, source_row, persona, context, turns):
    """Use User 1 as the user and User 2 as the assistant throughout each dialogue."""
    instruction = (
        "You are an assistant talking with a user. Use the user's persona and the conversation "
        "history to give an appropriate next reply. Write only your reply without a speaker label."
        f"\n\nUser persona:\n{persona}"
    )
    if context:
        instruction += f"\n\nConversation context:\n{context}"
    history = []
    for turn, (speaker, content) in enumerate(turns, 1):
        role = "user" if speaker == "1" else "assistant"
        if role == "assistant" and history and history[-1]["role"] == "user":
            prior_turns = [message.copy() for message in history]
            yield {
                "id": f"{split}-{source_row:05d}-{turn:02d}",
                "split": split,
                "source_row": source_row,
                "target_turn": turn,
                "user_speaker": "User 1",
                "assistant_speaker": "User 2",
                "user_persona": persona,
                "leading_context": context,
                "history": prior_turns,
                "user_query": prior_turns[-1]["content"],
                "target_response": content,
                "prompt": [{"role": "system", "content": instruction}, *prior_turns],
                "completion": [{"role": "assistant", "content": content}],
            }
        history.append({"role": role, "content": content})


def convert_split(source, output, split, user_side=False):
    valid_columns = {*PERSONA_COLUMNS.values(), CONVERSATION_COLUMN}
    stats = {
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_dialogues": 0,
        "converted_dialogues": 0,
        "letter_label_dialogues": 0,
        "examples": 0,
        "excluded": [],
    }
    if user_side:
        stats["assistant_opening_turns_without_user_history"] = 0
    fingerprints = set()
    with (
        source.open(encoding="utf-8-sig", newline="") as inp,
        output.open("w", encoding="utf-8", newline="\n") as stream,
    ):
        reader = csv.DictReader(inp)
        if set(reader.fieldnames or []) != valid_columns:
            raise ValueError(f"Unexpected CSV header in {source}")
        for source_row, row in enumerate(reader, 1):
            stats["source_dialogues"] += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV record {source_row} in {source}")
            personas = {speaker: row[column].strip() for speaker, column in PERSONA_COLUMNS.items()}
            text = row[CONVERSATION_COLUMN].strip()
            context, turns, letter_format = parse_dialogue(text)
            if not text:
                reason = "empty_dialogue"
            elif not (personas["1"] if user_side else all(personas.values())):
                reason = "missing_persona"
            elif {speaker for speaker, _ in turns} - set(personas):
                reason = "speaker_without_persona"
            elif {speaker for speaker, _ in turns} != set(personas):
                reason = "missing_two_identifiable_speakers"
            else:
                reason = None
            if reason:
                stats["excluded"].append({"source_row": source_row, "reason": reason})
                continue

            normalized = [(speaker, " ".join(content.split())) for speaker, content in turns]
            fingerprints.add(hashlib.sha256(json.dumps(normalized).encode("utf-8")).digest())
            stats["converted_dialogues"] += 1
            stats["letter_label_dialogues"] += int(letter_format)
            if user_side:
                stats["assistant_opening_turns_without_user_history"] += int(turns[0][0] == "2")
                examples = make_user_side_examples(split, source_row, personas["1"], context, turns)
            else:
                examples = make_examples(split, source_row, personas, context, turns)
            for example in examples:
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
                stats["examples"] += 1
    if not stats["examples"]:
        raise ValueError(f"No usable dialogue in {source}")
    return stats, fingerprints


def prepare(source, output, user_side=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": "TRL conversational prompt-completion JSONL",
        "modality": "text",
        "source": "https://github.com/google-research-datasets/Synthetic-Persona-Chat/tree/main/data",
        "splits": {},
    }
    splits = {split: filename for split, filename in SPLITS.items() if not user_side or split != "new_persona_unsplit"}
    if user_side:
        summary.update({
            "format": "User-side persona assistant-turn JSONL with TRL prompt-completion fields",
            "schema_version": 1,
            "persona_owner": "user",
            "speaker_mapping": {"User 1": "user", "User 2": "assistant"},
            "example_unit": "One User 2 reply following user history, conditioned on User 1 persona",
            "assistant_persona_included": False,
            "role_swapping": False,
            "field_definitions": {
                "id": "split-source_record_1based-target_merged_turn_1based",
                "source_row": "1-based CSV data record, excluding the header",
                "target_turn": "1-based turn after parsing labels and merging consecutive turns by the same speaker",
                "user_persona": "Original user 1 personas cell",
                "leading_context": "Source text preceding the first identifiable speaker turn",
                "history": "All parsed turns before the target, mapped to fixed user/assistant roles",
                "user_query": "Last user message in history",
                "target_response": "Current User 2 reply, excluded from history and prompt",
                "prompt": "Assistant instructions containing user persona and optional leading context, followed by history",
                "completion": "Single assistant message containing target_response",
            },
            "construction_split": "train",
            "id_evaluation_split": "test",
            "new_persona_unsplit_included": False,
        })
    fingerprints = {}
    with TemporaryDirectory(prefix=".synthetic-persona-chat-", dir=output.parent) as directory:
        staging = Path(directory)
        for split, file in splits.items():
            summary["splits"][split], fingerprints[split] = convert_split(
                source / file, staging / f"{split}.jsonl", split, user_side=user_side,
            )
        summary["cross_split_identical_dialogues"] = {
            f"{left}/{right}": len(fingerprints[left] & fingerprints[right])
            for left, right in combinations(splits, 2)
        }
        (staging / "conversion_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        output.mkdir(exist_ok=True)
        for file in staging.iterdir():
            file.replace(output / file.name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--user-side", action="store_true", help="Fix User 1 as user and User 2 as assistant, using only User 1's persona.")
    args = parser.parse_args()
    output = args.output_dir or (USER_SIDE_OUTPUT if args.user_side else OUTPUT)
    summary = prepare(args.source_dir, output, user_side=args.user_side)
    for split, stats in summary["splits"].items():
        print(f"{split}: {stats['converted_dialogues']}/{stats['source_dialogues']} dialogues, {stats['examples']} examples")
    print(f"Saved to {output.resolve()}")


if __name__ == "__main__":
    main()
