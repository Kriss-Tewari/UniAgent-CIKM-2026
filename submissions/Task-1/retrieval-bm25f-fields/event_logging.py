"""Event log for the UniAgent'26 event-logging contract.

See ../../event-logging-contract/README.md. Every event is one JSON object per
line in a gzip-compressed `run-trace.jsonl.log.gz`, with the required fields
case_id, event_id, parent_event_id, timestamp, event_type, model, tool, input,
output, status and error.

Mirrors the event layout of ../retrieval-baseline-pyterrier-smolagent/event_logging.py,
but without its smolagents dependency: this submission calls no model, so
`model` is "-" on every event, as the contract prescribes.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

NO_MODEL = "-"
TRACE_FILE = "run-trace.jsonl.log.gz"
EVENT_TYPES = ("input", "model_call", "tool_call", "observation", "decision", "error")
# Long strings (e.g. query descriptions) are whitespace-normalised and cut to
# this many characters; lists/dicts keep their JSON structure.
PREVIEW_LIMIT = 1000


def preview(value: Any, limit: int = PREVIEW_LIMIT) -> Any:
    if isinstance(value, str):
        text = " ".join(value.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"
    if isinstance(value, dict):
        return {str(key): preview(item, limit) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [preview(item, limit) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return preview(str(value), limit)


class EventLog:
    """Writes contract-compliant events; use as `with EventLog(path) as log:`."""

    def __init__(self, path: Path, model: str = NO_MODEL):
        if not model:
            raise ValueError("The contract requires a model on every event ('-' if none).")
        self.model = model
        self._file = gzip.open(path, "wt", encoding="utf-8")
        self._count = 0

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc_info) -> None:
        self._file.close()

    def log(self, case_id: str, event_type: str, *, parent: Optional[str] = None,
            tool: Optional[str] = None, input: Any = None, output: Any = None,
            status: str = "ok", error: Optional[str] = None) -> str:
        """Write one event and return its event_id (to pass as `parent` of the next one)."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event type: {event_type}")
        if (event_type == "tool_call") != (tool is not None):
            raise ValueError("`tool` is required for tool_call events and must be null otherwise.")
        if status not in ("ok", "error") or (status == "error") != (error is not None):
            raise ValueError("status must be 'ok' with error=None, or 'error' with an error message.")
        self._count += 1
        event_id = f"evt-{self._count:04d}"
        entry = {
            "case_id": str(case_id),
            "event_id": event_id,
            "parent_event_id": parent,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event_type": event_type,
            "model": self.model,
            "tool": tool,
            "input": preview(input),
            "output": preview(output),
            "status": status,
            "error": error,
        }
        print(json.dumps(entry, ensure_ascii=False), file=self._file, flush=True)
        return event_id
