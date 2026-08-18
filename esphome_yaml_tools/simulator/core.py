"""Core state machine simulator for ESPHome configurations.

Architecture:
- GlobalsRegistry tracks ESPHome `globals:` (voice_assistant_phase, init_in_progress)
- ScriptRegistry handles `script:` definitions + mode semantics (single/restart/queued/parallel)
- Lambda mini-parser handles `id(X) = N;` assignments and `id(X) == N` conditions
- ConditionTree replaces flat conditions list with nested and/or/not + lambda + leaf
- Conditional action `{if: {condition, then, else}}` supported in _execute_actions
- New action handlers: script.execute, script.stop, lambda, wait_until,
  voice_assistant.{start, set_use_wake_word}, micro_wake_word.{start, stop, enable_model, disable_model}
- Base script stubs (control_leds, activate_stop_word_once, play_sound) auto-registered

Compatibility API: Condition(type, target) dataclass, flat _evaluate_conditions
still work for existing test_simulator_core.py tests.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from esphome_yaml_tools.simulator.base_stubs import (
    BaseScriptsState,
    activate_stop_word_once_stub,
    base_voice_assistant_automations,
    control_leds_stub,
    play_sound_stub,
)
from esphome_yaml_tools.simulator.condition_tree import (
    ConditionEvaluator,
    ConditionNode,
)
from esphome_yaml_tools.simulator.globals_ import GlobalsRegistry
from esphome_yaml_tools.simulator.lambda_parser import (
    LambdaContext,
    evaluate_lambda_condition,
    execute_lambda_assignment,
)
from esphome_yaml_tools.simulator.scripts import Script, ScriptRegistry
from esphome_yaml_tools.simulator.states import (
    ComponentState,
    LightState,
    MediaPlayerState,
    MicroWakeWordState,
    SensorState,
    SwitchState,
    TimerState,
    VoiceAssistantPhase,
    VoiceAssistantState,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class SimulatorEvent:
    name: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Translated firmware action.

    For most action types, `type` is the ESPHome action name (e.g. "light.turn_on"),
    `target` is "domain.id" (e.g. "light.led_ring"), `params` carries the rest.

    For conditional actions (`{if: ...}` in YAML), `type="if"` and params hold:
      params["condition"]: ConditionNode
      params["then"]:      list[Action]
      params["else"]:      list[Action]

    For lambda actions, params["body"] is the raw lambda C++ source.

    For wait_until, params["condition"] is ConditionNode + params["timeout"] float seconds (0 = none).

    For script.execute, params["id"] is script id + params["args"] is kwargs dict.
    """

    type: str
    target: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Condition:
    """Flat-condition compatibility dataclass.

    New code should use ConditionNode in condition_tree.py. The legacy
    _evaluate_conditions still walks Condition[] (treating them as AND).
    """

    type: str
    target: str = ""


@dataclass
class Automation:
    trigger: str
    actions: list[Action] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)
    condition_tree: ConditionNode | None = None


@dataclass
class _PendingActions:
    """Actions queued for future execution by a `delay` action.

    `owning_invocation` is the ScriptInvocation that scheduled these (None if
    queued outside any script, e.g. from an event-handler automation directly).
    When the owning invocation is cancelled (via `script.stop` or `mode: restart`
    re-entry), these tails are skipped in advance_time — preventing the
    "restart cancels active frame but stale tails still fire" bug.
    """

    execute_at: float
    actions: list[Action]
    owning_invocation: Any = None


EventHandler = Callable[[SimulatorEvent], None]

_COMPONENT_DOMAIN_TO_STATE: dict[str, type[ComponentState]] = {
    "light": LightState,
    "switch": SwitchState,
    "binary_sensor": SwitchState,
    "sensor": SensorState,
    "voice_assistant": VoiceAssistantState,
    "micro_wake_word": MicroWakeWordState,
    "media_player": MediaPlayerState,
}


class Simulator:
    def __init__(self) -> None:
        self._states: dict[str, ComponentState] = {}
        self._default_states: dict[str, ComponentState] = {}
        self._event_handlers: dict[str, list[EventHandler]] = {}
        self._automations: list[Automation] = []
        self._event_history: list[SimulatorEvent] = []
        self._timers: dict[str, TimerState] = {}
        self._pending: list[_PendingActions] = []
        self._current_time: float = 0.0

        self.globals = GlobalsRegistry()
        self.scripts = ScriptRegistry()
        self.base_scripts_state = BaseScriptsState()
        self._va_id: str = "va"
        self._mww_id: str = "mww"
        self._invocation_stack: list = []

        self._register_base_stubs()

    @property
    def current_time(self) -> float:
        return self._current_time

    def _register_base_stubs(self) -> None:
        """Pre-register base-package scripts AND voice_assistant automations.

        Scripts: control_leds, activate_stop_word_once, play_sound — invoked
        via `script.execute:` from overlay firmware. Modeled as Python
        callbacks (see base_stubs.py) rather than YAML action lists.

        Automations: the base voice_assistant block's event handlers (on_listening,
        on_stt_vad_*, on_intent_progress, on_tts_start, on_end, on_error,
        on_timer_*, on_client_*). The simulator concatenates these with any
        overlay-defined handlers for the same event, matching ESPHome's
        package list-concat merge semantics. This lets minimal overlay firmware
        add ONLY new behaviors without reproducing base behavior.
        """
        for sid in ("control_leds", "activate_stop_word_once", "play_sound"):
            self.scripts.register(Script(id=sid, mode="single", actions=[]))
        for auto in base_voice_assistant_automations(va_id=self._va_id, mww_id=self._mww_id):
            self.register_automation(auto)

    def load_config(self, config: dict[str, Any]) -> None:
        self._states.clear()
        self._default_states.clear()
        self._automations.clear()
        self._event_history.clear()
        self._timers.clear()
        self._pending.clear()
        self._current_time = 0.0
        self.globals = GlobalsRegistry()
        self.scripts = ScriptRegistry()
        self.base_scripts_state = BaseScriptsState()
        self._register_base_stubs()

        if not config:
            return

        for domain, state_cls in _COMPONENT_DOMAIN_TO_STATE.items():
            raw = config.get(domain)
            if raw is None:
                continue

            components = raw if isinstance(raw, list) else [raw]
            for comp in components:
                comp_id = comp.get("id") if isinstance(comp, dict) else None
                if comp_id:
                    full_id = f"{domain}.{comp_id}"
                    state = state_cls()
                    self._states[full_id] = state
                    self._default_states[full_id] = copy.deepcopy(state)
                    if domain == "voice_assistant":
                        self._va_id = comp_id
                    elif domain == "micro_wake_word":
                        self._mww_id = comp_id

        for g in config.get("globals", []):
            if isinstance(g, dict) and "id" in g:
                self.globals.declare(g["id"], g.get("type", "any"), g.get("initial_value"))

        for s in config.get("script", []):
            if isinstance(s, dict) and "id" in s:
                self.scripts.register(
                    Script(
                        id=s["id"],
                        mode=s.get("mode", "single"),
                        actions=s.get("actions", []),
                        parameters=s.get("parameters", {}),
                    )
                )

        for auto_def in config.get("automation", []):
            conditions = [
                Condition(type=c["type"], target=c.get("target", ""))
                for c in auto_def.get("conditions", [])
            ]
            actions = self._coerce_actions(auto_def.get("actions", []))
            self.register_automation(
                Automation(
                    trigger=auto_def["trigger"],
                    conditions=conditions,
                    actions=actions,
                    condition_tree=auto_def.get("condition_tree"),
                )
            )

    def _coerce_actions(self, raw_actions: list[Any]) -> list[Action]:
        """Accept either Action objects or {type, target, params, condition?, then?, else?} dicts."""
        out: list[Action] = []
        for a in raw_actions:
            if isinstance(a, Action):
                out.append(a)
            elif isinstance(a, dict):
                out.append(
                    Action(
                        type=a["type"],
                        target=a.get("target", ""),
                        params=a.get("params", {}),
                    )
                )
        return out

    def has_component(self, component_id: str) -> bool:
        return component_id in self._states

    def get_state(self, component_id: str) -> ComponentState | None:
        return self._states.get(component_id)

    def set_state(self, component_id: str, state: ComponentState) -> None:
        self._states[component_id] = state

    def get_all_states(self) -> dict[str, ComponentState]:
        return dict(self._states)

    def register_event_handler(self, event_name: str, handler: EventHandler) -> None:
        self._event_handlers.setdefault(event_name, []).append(handler)

    def dispatch_event(self, event_name: str, data: dict[str, Any] | None = None) -> None:
        event = SimulatorEvent(name=event_name, data=data or {})
        self._event_history.append(event)

        for handler in self._event_handlers.get(event_name, []):
            handler(event)

        matching = [a for a in self._automations if a.trigger == event_name]
        for auto in matching:
            if auto.condition_tree is not None:
                evaluator = self._make_evaluator(event.data)
                if not evaluator.evaluate(auto.condition_tree):
                    continue
            elif not self._evaluate_conditions(auto.conditions):
                continue
            self._execute_actions(auto.actions, event_args=event.data)

    def get_event_history(self) -> list[SimulatorEvent]:
        return list(self._event_history)

    def register_automation(self, automation: Automation) -> None:
        self._automations.append(automation)

    def advance_time(self, seconds: float) -> None:
        target_time = self._current_time + seconds

        for timer_id in list(self._timers):
            timer = self._timers[timer_id]
            if not timer.active:
                continue
            new_remaining = timer.remaining - seconds
            if new_remaining <= 0.0:
                timer.remaining = 0.0
                timer.active = False
                self._timers[timer_id] = timer
                self.dispatch_event("timer_finished", data={"timer_id": timer_id})
            else:
                timer.remaining = new_remaining
                self._timers[timer_id] = timer
                self.dispatch_event(
                    "timer_tick",
                    data={"timer_id": timer_id, "seconds_left": timer.remaining},
                )

        while True:
            ready = sorted(
                (p for p in self._pending if p.execute_at <= target_time),
                key=lambda p: p.execute_at,
            )
            if not ready:
                break
            next_pending = ready[0]
            self._pending.remove(next_pending)
            owner = next_pending.owning_invocation
            if owner is not None and owner.cancelled:
                continue
            self._current_time = max(self._current_time, next_pending.execute_at)
            if owner is not None:
                self._invocation_stack.append(owner)
                try:
                    self._execute_actions(next_pending.actions)
                finally:
                    if self._invocation_stack and self._invocation_stack[-1] is owner:
                        self._invocation_stack.pop()
                    if not any(p.owning_invocation is owner for p in self._pending):
                        self.scripts.end_invocation(owner)
                        self._drain_queued(owner.script_id)
            else:
                self._execute_actions(next_pending.actions)

        self._current_time = target_time

    def start_timer(self, timer_id: str, duration: float) -> None:
        self._timers[timer_id] = TimerState(active=True, remaining=duration, total=duration)
        self.dispatch_event("timer_started", data={"timer_id": timer_id, "duration": duration})

    def cancel_timer(self, timer_id: str) -> None:
        if timer_id in self._timers:
            self._timers[timer_id].active = False
        self.dispatch_event("timer_cancelled", data={"timer_id": timer_id})

    def get_timer(self, timer_id: str) -> TimerState:
        return self._timers.get(timer_id, TimerState())

    def set_voice_assistant_phase(self, component_id: str, phase: VoiceAssistantPhase) -> None:
        current = self._states.get(component_id)
        old_phase = (
            current.phase if isinstance(current, VoiceAssistantState) else VoiceAssistantPhase.IDLE
        )
        self._states[component_id] = VoiceAssistantState(phase=phase)
        self.dispatch_event(
            "voice_assistant_phase_changed",
            data={"component_id": component_id, "old_phase": old_phase, "new_phase": phase},
        )

    def reset(self) -> None:
        self._states = {k: copy.deepcopy(v) for k, v in self._default_states.items()}
        self._event_history.clear()
        self._timers.clear()
        self._pending.clear()
        self._current_time = 0.0
        self.globals = GlobalsRegistry()
        self.scripts.reset()
        self.base_scripts_state = BaseScriptsState()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {k: v.to_dict() for k, v in self._states.items()}

    def _make_evaluator(self, event_args: dict[str, Any] | None = None) -> ConditionEvaluator:
        lambda_ctx = LambdaContext(
            globals_get=self.globals.get,
            globals_set=self.globals.set,
            on_method_call=self._handle_lambda_method_call,
            event_args=event_args or {},
        )
        return ConditionEvaluator(
            state_lookup=self._states.get,
            lambda_eval=lambda body: evaluate_lambda_condition(body, lambda_ctx),
        )

    def _handle_lambda_method_call(self, obj_id: str, method: str, args: list[Any]) -> None:
        """Apply a lambda method call like `id(va).set_use_wake_word(true)` to state."""
        if obj_id == self._va_id and method == "set_use_wake_word" and args:
            state = self._states.get(f"voice_assistant.{self._va_id}")
            if isinstance(state, VoiceAssistantState):
                state.use_wake_word = bool(args[0])
            else:
                self._states[f"voice_assistant.{self._va_id}"] = VoiceAssistantState(
                    use_wake_word=bool(args[0])
                )

    def _evaluate_conditions(self, conditions: list[Condition]) -> bool:
        for cond in conditions:
            state = self._states.get(cond.target)
            if cond.type == "switch.is_on":
                if not isinstance(state, SwitchState) or not state.on:
                    return False
            elif cond.type == "switch.is_off":
                if not isinstance(state, SwitchState) or state.on:
                    return False
        return True

    def _execute_actions(
        self, actions: list[Action], event_args: dict[str, Any] | None = None
    ) -> None:
        for i, action in enumerate(actions):
            if self._invocation_stack and self._invocation_stack[-1].cancelled:
                return

            if action.type == "delay":
                delay_seconds = action.params.get("seconds", 0.0)
                remaining_actions = actions[i + 1 :]
                if remaining_actions:
                    owning_inv = self._invocation_stack[-1] if self._invocation_stack else None
                    self._pending.append(
                        _PendingActions(
                            execute_at=self._current_time + delay_seconds,
                            actions=remaining_actions,
                            owning_invocation=owning_inv,
                        )
                    )
                return

            if action.type == "if":
                self._execute_conditional(action, event_args or {})
                continue

            self._execute_single_action(action, event_args=event_args or {})

    def _execute_conditional(self, action: Action, event_args: dict[str, Any]) -> None:
        """Evaluate {if: {condition, then, else}} and run the matching branch."""
        node = action.params.get("condition")
        evaluator = self._make_evaluator(event_args)
        passed = True
        if isinstance(node, ConditionNode):
            passed = evaluator.evaluate(node)
        then_actions = self._coerce_actions(action.params.get("then", []))
        else_actions = self._coerce_actions(action.params.get("else", []))
        self._execute_actions(then_actions if passed else else_actions, event_args=event_args)

    def _execute_single_action(self, action: Action, event_args: dict[str, Any] | None = None) -> None:
        target = action.target

        if action.type == "switch.turn_on":
            self._states[target] = SwitchState(on=True)
            return

        if action.type == "switch.turn_off":
            self._states[target] = SwitchState(on=False)
            return

        if action.type == "light.turn_on":
            params = action.params
            self._states[target] = LightState(
                on=True,
                brightness=params.get("brightness", 1.0),
                color=params.get("color", (255, 255, 255)),
                effect=params.get("effect", ""),
            )
            return

        if action.type == "light.turn_off":
            current = self._states.get(target)
            if isinstance(current, LightState):
                self._states[target] = LightState(
                    on=False,
                    brightness=current.brightness,
                    color=current.color,
                    effect=current.effect,
                )
            else:
                self._states[target] = LightState(on=False)
            return

        if action.type == "voice_assistant.start_continuous":
            self._ensure_va_state().is_streaming = True
            self._ensure_va_state().is_in_session = True
            return

        if action.type == "voice_assistant.start":
            va = self._ensure_va_state()
            va.is_streaming = False
            va.is_in_session = True
            return

        if action.type == "voice_assistant.stop":
            va = self._ensure_va_state()
            va.is_streaming = False
            va.is_in_session = False
            return

        if action.type == "micro_wake_word.start":
            mww = self._ensure_mww_state()
            mww.running = True
            return

        if action.type == "micro_wake_word.stop":
            mww = self._ensure_mww_state()
            mww.running = False
            return

        if action.type == "micro_wake_word.enable_model":
            mww = self._ensure_mww_state()
            model_id = action.params.get("model_id") or target
            mww.enabled_models[model_id] = True
            return

        if action.type == "micro_wake_word.disable_model":
            mww = self._ensure_mww_state()
            model_id = action.params.get("model_id") or target
            mww.enabled_models[model_id] = False
            return

        if action.type == "lambda":
            body = action.params.get("body", "")
            lambda_ctx = LambdaContext(
                globals_get=self.globals.get,
                globals_set=self.globals.set,
                on_method_call=self._handle_lambda_method_call,
                event_args=event_args or {},
            )
            execute_lambda_assignment(body, lambda_ctx)
            return

        if action.type == "script.execute":
            self._execute_script_action(action)
            return

        if action.type == "script.stop":
            script_id = action.params.get("id") or target
            self.scripts.stop(script_id)
            return

        if action.type == "wait_until":
            # Evaluate the condition once and continue. ESPHome's real semantic
            # is "block until condition true or timeout, then continue regardless".
            # In tests we don't model real-time blocking; we just record whether
            # the condition was true at the moment of evaluation. If False at
            # eval time, we proceed as if timeout fired (per ESPHome semantic
            # "on timeout, continue with remaining actions").
            node = action.params.get("condition")
            if isinstance(node, ConditionNode):
                evaluator = self._make_evaluator(event_args)
                evaluator.evaluate(node)
            return

        if action.type == "mixer_speaker.apply_ducking":
            return

        if action.type == "logger.log":
            return

        _LOGGER.debug("Simulator: unhandled action type %r target=%r", action.type, target)

    def _ensure_va_state(self) -> VoiceAssistantState:
        key = f"voice_assistant.{self._va_id}"
        state = self._states.get(key)
        if not isinstance(state, VoiceAssistantState):
            state = VoiceAssistantState()
            self._states[key] = state
        return state

    def _ensure_mww_state(self) -> MicroWakeWordState:
        key = f"micro_wake_word.{self._mww_id}"
        state = self._states.get(key)
        if not isinstance(state, MicroWakeWordState):
            state = MicroWakeWordState()
            self._states[key] = state
        return state

    def _execute_script_action(self, action: Action) -> None:
        """Run a script.execute action.

        Dispatches to base stubs (control_leds, activate_stop_word_once,
        play_sound) by name. Other scripts are firmware-defined; we look up
        their action list in self.scripts and execute it respecting the
        script's mode (single / restart / queued / parallel).
        """
        script_id = action.params.get("id") or action.target
        args = {k: v for k, v in action.params.items() if k != "id"}

        if script_id == "control_leds":
            control_leds_stub(
                self.globals.get,
                self._states.get,
                self._states.__setitem__,
                self.base_scripts_state.call_history,
            )
            return

        if script_id == "activate_stop_word_once":
            activate_stop_word_once_stub(self.base_scripts_state)
            return

        if script_id == "play_sound":
            sound_file = args.get("sound_file", "")
            priority = bool(args.get("priority", False))
            play_sound_stub(self.base_scripts_state, sound_file, priority)
            return

        script = self.scripts.get(script_id)
        if script is None:
            _LOGGER.debug("Simulator: unknown script.execute target %r", script_id)
            return

        invocation = self.scripts.begin_invocation(script_id, args=args)
        if invocation is None:
            return

        self._invocation_stack.append(invocation)
        try:
            actions = self._coerce_actions(script.actions)
            self._execute_actions(actions, event_args=args)
        finally:
            self._invocation_stack.pop()
            if not any(
                p.owning_invocation is invocation for p in self._pending
            ):
                self.scripts.end_invocation(invocation)
                self._drain_queued(script_id)

    def _drain_queued(self, script_id: str) -> None:
        """After an invocation truly completes, start the next queued invocation if any."""
        next_args = self.scripts.pop_next_queued(script_id)
        if next_args is None:
            return
        self._execute_script_action(
            Action(type="script.execute", target=script_id, params={"id": script_id, **next_args})
        )
