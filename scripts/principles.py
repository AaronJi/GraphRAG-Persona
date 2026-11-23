"""Read the shared 43 English principles using only the standard library."""

import json
from pathlib import Path


PRINCIPLES = Path(__file__).resolve().parents[1] / "principles.json"


def read_principles(path=PRINCIPLES):
    principles = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(principles, dict) or set(principles) != {f"{n:02d}" for n in range(1, 44)}:
        raise ValueError("Expected the 43 principles numbered 01–43")
    fields = {"category", "label", "when", "act"}
    for identifier, principle in principles.items():
        if not isinstance(principle, dict) or set(principle) != fields or any(
            not isinstance(value, str) or not value.strip() for value in principle.values()
        ):
            raise ValueError(f"Expected four nonempty text fields for principle {identifier}")
    return principles
