"""Mini-parser for the ESPHome `!lambda` C++ patterns our firmware actually uses.

Real ESPHome compiles lambda bodies as C++ via the ESPHome compile pipeline.
Our simulator can't compile C++; instead we recognize the specific lambda
patterns the firmware uses and map them to Python operations against the
simulator's GlobalsRegistry + state.

Patterns we recognize (assignment / method-call / condition):

  Assignment:
    id(NAME) = INT_LITERAL;            -> globals[NAME] = INT
    id(NAME) = "STRING";               -> globals[NAME] = STRING
    id(NAME) = true|false;             -> globals[NAME] = bool

  Method calls:
    id(va).set_use_wake_word(true);    -> mark voice_assistant.use_wake_word = True
    id(va).set_use_wake_word(false);   -> ditto False
    (others currently unused; can be extended)

  Conditions (return ...; pattern):
    return id(NAME) == VALUE;          -> bool
    return id(NAME) != VALUE;          -> bool
    return !id(NAME);                  -> bool (negation of bool-ish global)
    return id(NAME);                   -> bool (truthiness)
    return !x.empty();                 -> uses captured `x` argument (from event handler)
    return code == "STRING";           -> uses captured `code` argument
    return code != "STRING";           -> ditto

Unparseable lambdas:
  - Assignments: silently no-op (warned via logger)
  - Conditions: default to True (safe-fail toward "action chain continues" so
    we don't accidentally break tests; warning logged so we know to extend
    the parser when needed)

The parser is regex-based. ESPHome lambdas are tiny snippets in this firmware
so this is sufficient. A real C++ parser would be heavyweight overkill.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

_LOGGER = logging.getLogger(__name__)


_RE_ID_ASSIGN_INT = re.compile(r"id\(([A-Za-z_]\w*)\)\s*=\s*(-?\d+)\s*;")
_RE_ID_INCREMENT = re.compile(
    r"id\(([A-Za-z_]\w*)\)\s*=\s*id\(\1\)\s*([+\-])\s*(-?\d+)\s*;"
)
_RE_ID_ASSIGN_BOOL = re.compile(r"id\(([A-Za-z_]\w*)\)\s*=\s*(true|false)\s*;")
_RE_ID_ASSIGN_STR = re.compile(r"id\(([A-Za-z_]\w*)\)\s*=\s*\"([^\"]*)\"\s*;")
_RE_ID_METHOD_BOOL = re.compile(
    r"id\(([A-Za-z_]\w*)\)\.([A-Za-z_]\w*)\(\s*(true|false)\s*\)\s*;"
)

_RE_COND_EQ_INT = re.compile(
    r"return\s+id\(([A-Za-z_]\w*)\)\s*(==|!=)\s*(-?\d+)\s*;"
)
_RE_COND_EQ_STR = re.compile(
    r"return\s+([A-Za-z_]\w*)\s*(==|!=)\s*\"([^\"]*)\"\s*;"
)
_RE_COND_BANG_EMPTY = re.compile(r"return\s+!([A-Za-z_]\w*)\.empty\(\)\s*;")
_RE_COND_TRUTHY_BANG = re.compile(r"return\s+!id\(([A-Za-z_]\w*)\)\s*;")
_RE_COND_TRUTHY = re.compile(r"return\s+id\(([A-Za-z_]\w*)\)\s*;")


@dataclass
class LambdaContext:
    """Caller-provided context for lambda execution.

    For action-position lambdas there's no `x` or `code` variable; pass
    `event_args={}`. For event-handler lambdas like `on_error`, the YAML's
    runtime sets `code` to the error string — pass that in here.
    """

    globals_get: Callable[[str], Any]
    globals_set: Callable[[str, Any], None]
    on_method_call: Callable[[str, str, list[Any]], None] | None = None
    event_args: dict[str, Any] | None = None


def execute_lambda_assignment(body: str, ctx: LambdaContext) -> bool:
    """Apply an assignment/method-call lambda to the simulator state.

    Returns True if any known pattern matched (one or more substitutions).
    Returns False if the body had no recognized pattern; caller may then
    log a warning. Multiple matches in one body are all applied (the
    firmware sometimes packs `id(a) = 1; id(b) = 2;` in one lambda).
    """
    matched = False
    body = body.strip()

    for m in _RE_ID_INCREMENT.finditer(body):
        name, op, literal = m.group(1), m.group(2), int(m.group(3))
        delta = literal if op == "+" else -literal
        cur = ctx.globals_get(name)
        try:
            cur_int = int(cur) if cur is not None else 0
        except (TypeError, ValueError):
            cur_int = 0
        ctx.globals_set(name, cur_int + delta)
        matched = True

    for m in _RE_ID_ASSIGN_INT.finditer(body):
        if _RE_ID_INCREMENT.search(body) and m.group(1) in {im.group(1) for im in _RE_ID_INCREMENT.finditer(body)}:
            continue
        ctx.globals_set(m.group(1), int(m.group(2)))
        matched = True

    for m in _RE_ID_ASSIGN_BOOL.finditer(body):
        ctx.globals_set(m.group(1), m.group(2) == "true")
        matched = True

    for m in _RE_ID_ASSIGN_STR.finditer(body):
        ctx.globals_set(m.group(1), m.group(2))
        matched = True

    if ctx.on_method_call is not None:
        for m in _RE_ID_METHOD_BOOL.finditer(body):
            obj_id, method, val = m.group(1), m.group(2), (m.group(3) == "true")
            ctx.on_method_call(obj_id, method, [val])
            matched = True

    if not matched:
        _LOGGER.debug("lambda_parser: assignment pattern not recognized: %s", body)
    return matched


def evaluate_lambda_condition(body: str, ctx: LambdaContext) -> bool:
    """Evaluate a condition lambda. Returns True on parse failure (safe-fail).

    Safe-fail toward True means action chains continue executing even when
    we can't parse the condition. Tests catch the real behavioral failures
    (script not called, phase not set) rather than getting blocked by
    parsing limitations.
    """
    body = body.strip()
    args = ctx.event_args or {}

    m = _RE_COND_EQ_INT.search(body)
    if m:
        global_name, op, literal = m.group(1), m.group(2), int(m.group(3))
        cur = ctx.globals_get(global_name)
        try:
            cur_int = int(cur)
        except (TypeError, ValueError):
            cur_int = 0
        return (cur_int == literal) if op == "==" else (cur_int != literal)

    m = _RE_COND_EQ_STR.search(body)
    if m:
        var_name, op, literal = m.group(1), m.group(2), m.group(3)
        cur = args.get(var_name, ctx.globals_get(var_name))
        return (cur == literal) if op == "==" else (cur != literal)

    m = _RE_COND_BANG_EMPTY.search(body)
    if m:
        var_name = m.group(1)
        val = args.get(var_name, ctx.globals_get(var_name))
        if val is None:
            return False
        try:
            return len(val) > 0
        except TypeError:
            return bool(val)

    m = _RE_COND_TRUTHY_BANG.search(body)
    if m:
        return not bool(ctx.globals_get(m.group(1)))

    m = _RE_COND_TRUTHY.search(body)
    if m:
        return bool(ctx.globals_get(m.group(1)))

    _LOGGER.warning(
        "lambda_parser: condition pattern not recognized, defaulting to True: %s",
        body,
    )
    return True
