"""ESPHome `script:` engine for the simulator.

ESPHome scripts are reusable action sequences with mode semantics governing
what happens when a script is invoked while already running:

| Mode      | Behavior when re-invoked                                   |
|-----------|------------------------------------------------------------|
| single    | Drop the new invocation silently (default)                 |
| restart   | Cancel current run, start fresh from the top               |
| queued    | Run sequentially, queuing additional calls                 |
| parallel  | Each invocation runs concurrently                          |

For the simulator we don't have true concurrency — every action is synchronous.
We model mode behavior by tracking `is_running` and a re-entry token that
increments each `restart`. Tests assert script invocation counts + that
re-entrant calls behave per-mode.

The simulator's Script class stores raw action dicts (as they came out of the
firmware adapter). Execution dispatches each action via the Simulator's
existing `_execute_single_action` after applying lambda parsing for the
`lambda` action type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

ScriptMode = Literal["single", "restart", "queued", "parallel"]


@dataclass
class Script:
    """Definition of an ESPHome script.

    `actions` is the raw list from the YAML's `then:` block. The simulator
    executes them by passing each to its own action dispatcher; no parsing
    happens here.
    """

    id: str
    mode: ScriptMode = "single"
    actions: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, str] = field(default_factory=dict)


@dataclass
class ScriptInvocation:
    """A single execution of a script. Tracks invocation order + status."""

    script_id: str
    token: int
    args: dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    cancelled: bool = False


@dataclass
class ScriptRegistry:
    """Holds script definitions and tracks runtime invocation state.

    Tests can read `invocations` to assert script call sequences and verify
    mode semantics (e.g., 'restart' produces fresh tokens, 'single' drops).
    """

    _scripts: dict[str, Script] = field(default_factory=dict)
    invocations: list[ScriptInvocation] = field(default_factory=list)
    _running: dict[str, list[ScriptInvocation]] = field(default_factory=dict)
    _queued: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _next_token: int = 0

    def register(self, script: Script) -> None:
        self._scripts[script.id] = script

    def get(self, script_id: str) -> Script | None:
        return self._scripts.get(script_id)

    def is_running(self, script_id: str) -> bool:
        return bool(self._running.get(script_id))

    def begin_invocation(
        self, script_id: str, args: dict[str, Any] | None = None
    ) -> ScriptInvocation | None:
        """Decide whether a new invocation should run, per the script's mode.

        Returns the ScriptInvocation if the action dispatcher should proceed.
        Returns None if the call was dropped (single mode while running).

        For restart mode, marks any currently-running invocation as cancelled
        (the action dispatcher must check `invocation.cancelled` between
        actions). The simulator core also marks any pending delayed tails as
        cancelled so they don't fire after the new run starts.

        An invocation stays in `_running` (and counts as active for mode checks)
        until all its scheduled tails have either executed or been cancelled —
        the simulator's advance_time + _execute_script_action coordinate this
        via `end_invocation` calls.
        """
        script = self._scripts.get(script_id)
        if script is None:
            return None
        args = args or {}

        running = self._running.setdefault(script_id, [])

        if script.mode == "single" and running:
            return None
        if script.mode == "queued" and running:
            self._queued.setdefault(script_id, []).append(args)
            return None
        if script.mode == "restart" and running:
            for inv in running:
                inv.cancelled = True
            running.clear()

        self._next_token += 1
        inv = ScriptInvocation(script_id=script_id, token=self._next_token, args=args)
        self.invocations.append(inv)
        running.append(inv)
        return inv

    def pop_next_queued(self, script_id: str) -> dict[str, Any] | None:
        """Return next queued args for `mode: queued` scripts, or None if empty.

        Caller (simulator) is responsible for invoking the script with these
        args after the current invocation fully completes (including all
        delayed tails).
        """
        queue = self._queued.get(script_id, [])
        if not queue:
            return None
        return queue.pop(0)

    def end_invocation(self, inv: ScriptInvocation) -> None:
        inv.completed = True
        running = self._running.get(inv.script_id, [])
        if inv in running:
            running.remove(inv)

    def stop(self, script_id: str) -> int:
        """Cancel all running invocations of a script. Returns count cancelled."""
        running = self._running.get(script_id, [])
        cancelled = 0
        for inv in running:
            inv.cancelled = True
            cancelled += 1
        running.clear()
        return cancelled

    def reset(self) -> None:
        """Test isolation: clear all invocation history + queued args.

        Note: `_queued` MUST be cleared along with `_running`, or stale queued
        args survive into the next test and a single post-reset invocation
        starts two scripts (the live one + the stale queued tail). Oracle
        round-5 found this leak; regression test in TestScriptRegistryReset.
        """
        self.invocations.clear()
        self._running.clear()
        self._queued.clear()
        self._next_token = 0
