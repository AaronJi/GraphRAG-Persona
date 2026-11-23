"""Read shared retrieval and OpenAI-compatible endpoint settings."""

import math
from pathlib import Path
from urllib.parse import urlsplit

import yaml


SETTINGS = Path(__file__).resolve().parents[1] / "query_settings.yaml"


def load_query_settings(path=SETTINGS):
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError("Query settings must be a mapping")
    k, epsilon = settings.get("k"), settings.get("epsilon")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or not -100 <= epsilon <= 100:
        raise ValueError("epsilon must be a finite score between -100 and 100")
    for name in ("llm", "embeddings"):
        service = settings.get(name)
        if not isinstance(service, dict) or any(
            not isinstance(service.get(key), str) or not service[key].strip()
            for key in ("api_base", "api_key", "model")
        ):
            raise ValueError(f"{name} requires nonempty api_base, api_key, and model strings")
        address = urlsplit(service["api_base"])
        if address.scheme not in {"http", "https"} or not address.netloc or address.query or address.fragment:
            raise ValueError(f"{name}.api_base must be an HTTP(S) base URL")
        timeout = service.get("request_timeout")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"{name}.request_timeout must be a finite positive number")
    return settings
