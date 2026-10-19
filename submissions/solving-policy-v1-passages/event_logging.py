"""Event-logging for the UniAgent'26 event-logging contract.

See ../event-logging-contract/README.md for the full contract. This module
writes one gzip-compressed JSON-Lines event per line to `run-trace.jsonl.log.gz`,
and gives every event the required `case_id`, `event_id`, `parent_event_id`,
`timestamp`, `event_type`, `model`, `tool`, `input`, `output`, `status`, and
`error` fields.

Causal linkage (`parent_event_id`) is derived automatically: within a
`case_context()`, each logged event becomes the parent of the next one,
forming a chain from the case's first tool/model call through to its final
`decision` event. `parent_event()` can override this default chain for
events whose true cause is not simply "the previous event" (for instance a
retrieval `tool_call` triggered by a model's follow-up query rather than by
the immediately preceding event).
"""

import contextlib
import contextvars
import functools
import gzip
import inspect
import itertools
import json
import sys
from datetime import datetime, timezone
from typing import Any, IO, Iterator, Optional

from smolagents import Tool

_current_case: contextvars.ContextVar[str] = contextvars.ContextVar(
    "event_logging_case", default="-"
)
_current_model: contextvars.ContextVar[str] = contextvars.ContextVar(
    "event_logging_model", default="-"
)
_current_parent: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "event_logging_parent", default=None
)
_current_destination: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "event_logging_destination", default=None
)
_event_counter = itertools.count(1)

# Tracks, per case_id, the event_id that the next event logged for that case
# should chain onto. A case is typically processed in separate phases (e.g.
# evidence gathering, then retrieval, then the final decision), each entering
# case_context() again; this lets the causal chain resume across phases
# instead of restarting (parent_event_id=null) every time the same case is
# re-entered.
_case_chain_tail: dict[str, Optional[str]] = {}


def reset_state() -> None:
    """Clear all module-level chain state (not the shared event-id counter).

    For tests only: independent test cases that reuse the same case_id (e.g.
    a fixture case like "dienstreiseantrag-01") would otherwise chain onto
    whatever an earlier, unrelated test already logged for that case_id in
    the same process. Production runs process each case exactly once per
    process and never need this.
    """
    _case_chain_tail.clear()

# Inputs/outputs are previewed (not dumped in full) to keep log lines
# greppable even when a tool reads a multi-page PDF or a model returns a long
# response; see _preview()'s docstring for the exact truncation rule.
PREVIEW_LIMIT = 200


@contextlib.contextmanager
def case_context(case_id: str) -> Iterator[None]:
    """Tag every event logged while active with `case_id`, chaining onto
    this case's existing event chain if one exists (or starting a fresh one,
    with `parent_event_id: null`, on the case's first use).

    Usage: `with case_context("dienstreiseantrag-01"): ...` around the code
    that processes one case, regardless of how many tools/models it calls,
    and regardless of whether the case is processed in one call or several
    separate phases (each re-entering this context manager); the chain
    resumes across phases instead of restarting.
    """
    case_token = _current_case.set(case_id)
    parent_token = _current_parent.set(_case_chain_tail.get(case_id))
    try:
        yield
    finally:
        _case_chain_tail[case_id] = _current_parent.get()
        _current_case.reset(case_token)
        _current_parent.reset(parent_token)


@contextlib.contextmanager
def model_context(model_id: str) -> Iterator[None]:
    """Tag every event logged while active with the LLM model identifier.

    Per the contract, `model` is required on every event, including
    `tool_call`/`observation` events that do not themselves call a model, so
    the model backing a run is never ambiguous.
    """
    token = _current_model.set(model_id)
    try:
        yield
    finally:
        _current_model.reset(token)


@contextlib.contextmanager
def parent_event(event_id: Optional[str]) -> Iterator[None]:
    """Temporarily override the causal parent for events logged in this block.

    Useful when an event's true cause is not simply "the previous event",
    e.g. a retrieval `tool_call` triggered by a model's follow-up query
    rather than by whatever was logged immediately before it.
    """
    token = _current_parent.set(event_id)
    try:
        yield
    finally:
        _current_parent.reset(token)


@contextlib.contextmanager
def log_to_file(path: Any) -> Iterator[None]:
    """Write every event as one gzip-compressed JSONL line to `path`.

    Usage: `with log_to_file(output_dir / "run-trace.jsonl.log.gz"): ...` around
    the code whose events should be captured; the file is opened once
    (truncating any previous contents) and closed on exit.
    """
    with gzip.open(path, "wt", encoding="utf-8") as destination:
        token = _current_destination.set(destination)
        try:
            yield
        finally:
            _current_destination.reset(token)


def _write(entry: dict[str, Any]) -> None:
    destination: IO[str] = _current_destination.get() or sys.stdout
    print(json.dumps(entry, ensure_ascii=False), file=destination, flush=True)


def _preview(value: Any, limit: int = PREVIEW_LIMIT) -> Any:
    """Return value unchanged if it is a short JSON value, else a truncated preview string.

    Strings are whitespace-normalised and cut to `limit` characters. Other
    JSON-serialisable values (numbers, booleans, null, lists, dicts) are kept
    as-is when their JSON encoding is at most `limit` characters, so small
    structured inputs/outputs stay queryable as native JSON; larger ones fall
    back to a truncated string preview of their JSON encoding.
    Non-JSON-serialisable values fall back to str().
    """
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
        else:
            if len(text) <= limit:
                return value
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _preview_all(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _preview(item) for key, item in value.items()}
    return _preview(value)


def _bound_arguments(forward: Any, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Map positional/keyword call arguments to parameter names by name."""
    try:
        bound = inspect.signature(forward).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except TypeError:
        # Fall back to positional indices if the signature does not match
        # (should not happen for the tools defined in this baseline).
        arguments = {f"arg{index}": value for index, value in enumerate(args)}
        arguments.update(kwargs)
        return arguments


def log_event(
    event_type: str,
    *,
    tool: Optional[str] = None,
    input: Any = None,
    output: Any = None,
    status: str = "ok",
    error: Optional[str] = None,
) -> str:
    """Log one contract-compliant event and return its `event_id`.

    The event is chained onto the previously logged event in the active
    `case_context()` (its `parent_event_id`), and that chain pointer is then
    advanced to this event, so the next call in the same case chains onto
    this one by default. Use `parent_event()` to attach an event to a
    specific earlier event instead.
    """
    event_id = f"evt-{next(_event_counter):04d}"
    entry: dict[str, Any] = {
        "case_id": _current_case.get(),
        "event_id": event_id,
        "parent_event_id": _current_parent.get(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event_type": event_type,
        "model": _current_model.get(),
        "tool": tool,
        "input": _preview_all(input),
        "output": _preview_all(output),
        "status": status,
        "error": error,
    }
    _write(entry)
    _current_parent.set(event_id)
    return event_id


def log_tool_calls(tool: Tool) -> Tool:
    """Wrap a Tool instance so every call logs one `tool_call` event.

    Wraps forward() in place (not __call__) so smolagents' own argument
    validation still runs first. Both success (with the tool's return value
    as `output`) and failure (with `status: "error"` and the exception
    message) are logged; a failing call is re-raised after logging, never
    swallowed.
    """
    original_forward = tool.forward

    @functools.wraps(original_forward)
    def logged_forward(*args: Any, **kwargs: Any) -> Any:
        arguments = _bound_arguments(original_forward, args, kwargs)
        try:
            result = original_forward(*args, **kwargs)
        except Exception as error:
            log_event(
                "tool_call",
                tool=tool.name,
                input=arguments,
                output=None,
                status="error",
                error=str(error),
            )
            raise
        log_event(
            "tool_call",
            tool=tool.name,
            input=arguments,
            output=result,
            status="ok",
            error=None,
        )
        return result

    tool.forward = logged_forward
    return tool
