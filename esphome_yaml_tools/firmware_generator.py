"""Firmware generator: merges a pinned upstream ESPHome base package with an
overlay and applies safety-checked patches, producing a fully-flat YAML.

ESPHome `packages:` cannot surgically override list items without `id:` fields.
This generator runs the merge locally so additional patch ops can fix what
package-merge can't.

Workflow when upstream bumps:
    1. Update `revision` + `sha256` of the patch module's `BasePackage`.
    2. Re-run generator. Any patch whose `expected` no longer matches raises
       `PatchError` with path + diff -> the upstream-changed early warning.
    3. Review upstream diff, update `expected`, re-run.
    4. Run firmware tests against the generated YAML.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .parser import Extend, Lambda, Remove, load_yaml


def merge_config(old: Any, new: Any) -> Any:
    """Deep-merge ESPHome-style. Dicts merge recursively. Lists CONCATENATE
    (no dedup, no replace). `Remove` sentinel propagates. `None` keeps old.
    Matches `esphome/config_helpers.py::merge_config`."""
    if isinstance(new, Remove):
        return new
    if isinstance(new, dict):
        if not isinstance(old, dict):
            return new
        res: dict[Any, Any]
        if isinstance(old, OrderedDict) or isinstance(new, OrderedDict):
            res = OrderedDict(old)
        else:
            res = dict(old)
        for k, v in new.items():
            res[k] = merge_config(old.get(k), v)
        return res
    if isinstance(new, list):
        if not isinstance(old, list):
            return new
        return list(old) + list(new)
    if new is None:
        return old
    return new


def _get_item_id(item: Any) -> str | Extend | Remove | None:
    """Match ESPHome's _get_item_id: returns id string or Extend/Remove sentinel.
    Strips the Extend marker from the item's id field (in-place mutation) so it
    can later be merged into the base item by id."""
    if not isinstance(item, dict):
        return None
    item_id = item.get("id")
    if item_id is None and len(item) == 1:
        item = next(iter(item.values()))
        if isinstance(item, dict):
            item_id = item.get("id")
    if isinstance(item_id, Extend):
        del item["id"]
        return item_id
    if not isinstance(item_id, (str, Remove)):
        return None
    return item_id


def resolve_extend_remove(value: Any) -> None:
    """Match `esphome/config.py::resolve_extend_remove`. In-place: extensions
    merge into matched-by-id base items; removals drop matched items; dict keys
    with Remove value are dropped. Recurses into children."""
    if isinstance(value, list):
        index: OrderedDict[Any, Any] = OrderedDict()
        extensions: list[tuple[int, Any, Any]] = []
        removals: set[Any] = set()
        for pos, item in enumerate(value):
            if item is None:
                removals.add(None)
                continue
            item_id = _get_item_id(item)
            if isinstance(item_id, Extend):
                extensions.append((pos, item_id.value, item))
                continue
            if isinstance(item_id, Remove):
                removals.add(item_id.value)
                continue
            if not item_id or item_id in index:
                item_id = id(item)
            index[item_id] = item
        if extensions or removals:
            for pos, item_id, item in extensions:
                if item_id in removals:
                    continue
                old = index.get(item_id)
                if old is None:
                    raise PatchError(
                        f"!extend at list position {pos}: source id "
                        f"{item_id!r} not found in base list"
                    )
                index[item_id] = merge_config(old, item)
            for item_id in removals:
                index.pop(item_id, None)
            value[:] = index.values()
        for item in value:
            resolve_extend_remove(item)
        return
    if isinstance(value, dict):
        to_drop: list[Any] = []
        for k, v in value.items():
            if isinstance(v, Remove):
                to_drop.append(k)
                continue
            resolve_extend_remove(v)
        for k in to_drop:
            value.pop(k, None)


class PatchError(Exception):
    pass


class _Missing:
    """Sentinel for set_key/merge_dict patches that target a key absent from base.

    Distinguishable from None (which is a legitimate YAML value the user might
    write as `key:` with no value). `expected=MISSING` means the patch asserts
    the key does NOT exist; `expected=None` means it asserts the key exists
    with value None.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __reduce__(self):
        return (_Missing, ())


MISSING = _Missing()


def _split_path(path: str) -> list[str]:
    return [p for p in path.strip("/").split("/") if p != ""]


def _traverse(config: Any, parts: list[str], pretty: str) -> Any:
    cur = config
    for i, part in enumerate(parts):
        if isinstance(cur, dict):
            if part not in cur:
                raise PatchError(
                    f"Path {pretty!r}: key {part!r} not found at depth {i}"
                )
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                raise PatchError(
                    f"Path {pretty!r}: expected list index at depth {i}, got {part!r}"
                )
            if not -len(cur) <= idx < len(cur):
                raise PatchError(
                    f"Path {pretty!r}: index {idx} out of range at depth {i}"
                )
            cur = cur[idx]
        else:
            raise PatchError(
                f"Path {pretty!r}: cannot traverse {type(cur).__name__} at depth {i}"
            )
    return cur


def _assign(config: Any, parts: list[str], value: Any, pretty: str) -> None:
    if not parts:
        raise PatchError("Cannot patch the root config")
    parent = _traverse(config, parts[:-1], pretty)
    last = parts[-1]
    if isinstance(parent, dict):
        parent[last] = value
    elif isinstance(parent, list):
        parent[int(last)] = value
    else:
        raise PatchError(
            f"Path {pretty!r}: cannot assign to {type(parent).__name__}"
        )


_DEFAULT = object()


@dataclass(frozen=True)
class Patch:
    """Declarative patch applied AFTER merge.

    Ops:
    - 'replace': asserts current value at path equals `expected` before
      setting `value`. Use for surgical value swaps with upstream-change detection.
    - 'transform': applies `transform(current_value)` and writes the result.
      Use for "filter the list" / "compute from current" — no expected check.
    - 'merge_dict': asserts current value at path is a dict (and equals `expected`
      if expected is not _DEFAULT), then deep-merges `value` into it. Use for
      adding keys to a dict (e.g., substitutions, esphome metadata).
    - 'append_list': asserts current value at path is a list (and equals
      `expected` if expected is not _DEFAULT), then appends `value` items to it.
      Use for adding to an existing event-handler list.
    - 'set_key': asserts the LAST PATH ELEMENT is a key in the parent dict whose
      current value equals `expected` (or doesn't exist when `expected=MISSING`),
      then sets it to `value`. Use for adding a NEW handler key that base doesn't
      define (e.g., voice_assistant.on_wake_word_detected).

    `rationale` is printed on failure to make the upstream-change signal actionable.
    """

    op: str
    path: str
    rationale: str
    expected: Any = _DEFAULT
    value: Any = None
    transform: Callable[[Any], Any] | None = None


def _check_expected(current: Any, expected: Any) -> None:
    if expected is _DEFAULT:
        return
    if current != expected:
        raise PatchError(
            f"expected:\n    {expected!r}\n"
            f"got:\n    {current!r}\n"
            f"Upstream likely changed at this path; review and update."
        )


def apply_patches(config: Any, patches: list[Patch]) -> None:
    for i, patch in enumerate(patches):
        parts = _split_path(patch.path)
        try:
            if patch.op == "replace":
                current = _traverse(config, parts, patch.path)
                _check_expected(current, patch.expected)
                _assign(config, parts, patch.value, patch.path)
            elif patch.op == "transform":
                assert patch.transform is not None, "transform op requires .transform"
                current = _traverse(config, parts, patch.path)
                _assign(config, parts, patch.transform(current), patch.path)
            elif patch.op == "merge_dict":
                current = _traverse(config, parts, patch.path)
                if not isinstance(current, dict):
                    raise PatchError(
                        f"merge_dict requires dict at path, got {type(current).__name__}"
                    )
                _check_expected(current, patch.expected)
                if not isinstance(patch.value, dict):
                    raise PatchError(
                        f"merge_dict requires dict value, got {type(patch.value).__name__}"
                    )
                _assign(config, parts, merge_config(current, patch.value), patch.path)
            elif patch.op == "append_list":
                current = _traverse(config, parts, patch.path)
                if not isinstance(current, list):
                    raise PatchError(
                        f"append_list requires list at path, got {type(current).__name__}"
                    )
                _check_expected(current, patch.expected)
                if not isinstance(patch.value, list):
                    raise PatchError(
                        f"append_list requires list value, got {type(patch.value).__name__}"
                    )
                _assign(config, parts, list(current) + list(patch.value), patch.path)
            elif patch.op == "set_key":
                if not parts:
                    raise PatchError("set_key requires a non-root path")
                parent = _traverse(config, parts[:-1], patch.path)
                if not isinstance(parent, dict):
                    raise PatchError(
                        f"set_key requires parent dict at {'/'.join(parts[:-1])!r}, "
                        f"got {type(parent).__name__}"
                    )
                key = parts[-1]
                if patch.expected is MISSING:
                    if key in parent:
                        raise PatchError(
                            f"expected key {key!r} to be ABSENT but found value "
                            f"{parent[key]!r}; upstream may have added it"
                        )
                elif patch.expected is not _DEFAULT:
                    if key not in parent:
                        raise PatchError(
                            f"expected key {key!r} with value {patch.expected!r} but "
                            f"key is absent; upstream may have removed it"
                        )
                    _check_expected(parent[key], patch.expected)
                parent[key] = patch.value
            else:
                raise PatchError(f"unknown op: {patch.op!r}")
        except PatchError as e:
            raise PatchError(
                f"Patch {i + 1}/{len(patches)} failed at {patch.path!r} "
                f"({patch.rationale[:120]}):\n  {e}"
            ) from None


@dataclass(frozen=True)
class BasePackage:
    """Pinned upstream base. `sha256` is verified on every load; mismatch is
    the intentional fail-loud signal that upstream changed and the patches
    must be reviewed."""

    url: str
    revision: str
    sha256: str
    cache_path: Path

    def load(self, *, refresh: bool = False) -> str:
        if refresh or not self.cache_path.exists():
            req = urllib.request.Request(
                self.url, headers={"User-Agent": "voice-pe-firmware-generator/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8")
        else:
            content = self.cache_path.read_text(encoding="utf-8")
        actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual != self.sha256:
            raise PatchError(
                f"Base package SHA-256 mismatch.\n"
                f"  URL:      {self.url}\n"
                f"  revision: {self.revision}\n"
                f"  expected: {self.sha256}\n"
                f"  actual:   {actual}\n"
                f"Upstream changed. Review diff and update revision + sha256."
            )
        if refresh:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(content, encoding="utf-8")
        return content


class _FirmwareDumper(yaml.SafeDumper):
    """Emits Lambda as !lambda; raises on unresolved Extend/Remove (safety)."""


def _lambda_representer(dumper: yaml.SafeDumper, data: Lambda) -> yaml.Node:
    style = "|" if "\n" in data.code else None
    return dumper.represent_scalar("!lambda", data.code, style=style)


def _refuse_marker(name: str) -> Callable[[yaml.SafeDumper, Any], yaml.Node]:
    def _refuse(dumper: yaml.SafeDumper, data: Any) -> yaml.Node:
        raise PatchError(
            f"Unresolved {name} marker leaked into output; generator bug. "
            f"resolve_extend_remove must run before dumping."
        )

    return _refuse


def _ordered_dict_representer(
    dumper: yaml.SafeDumper, data: OrderedDict
) -> yaml.Node:
    return dumper.represent_dict(data.items())


_FirmwareDumper.add_representer(Lambda, _lambda_representer)
_FirmwareDumper.add_representer(Extend, _refuse_marker("!extend"))
_FirmwareDumper.add_representer(Remove, _refuse_marker("!remove"))
_FirmwareDumper.add_representer(OrderedDict, _ordered_dict_representer)


def dump_yaml(config: Any) -> str:
    return yaml.dump(
        config,
        Dumper=_FirmwareDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=120,
    )


@dataclass
class GenerationResult:
    config: dict
    yaml_text: str
    output_path: Path
    base_revision: str
    base_sha256: str


def generate_firmware(
    base: BasePackage,
    patches: list[Patch],
    output_path: Path,
    *,
    overlay_path: Path | None = None,
    refresh_base: bool = False,
    header: str | None = None,
) -> GenerationResult:
    """Pipeline: load pinned base (sha256 verified) -> [optional overlay merge +
    !extend/!remove resolve] -> apply patches with safety checks -> dump.

    `overlay_path` is optional; when None the patches list is the SOLE source
    of overrides on top of base. This is the patches-only architecture.
    Validation against the simulator is the caller's responsibility (firmware
    tests run against the produced YAML)."""
    base_text = base.load(refresh=refresh_base)
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(base_text)
        base_tmp = Path(f.name)
    try:
        base_config = load_yaml(base_tmp)
    finally:
        base_tmp.unlink(missing_ok=True)

    if overlay_path is not None:
        overlay_config = load_yaml(overlay_path)
        merged = merge_config(base_config, overlay_config)
        resolve_extend_remove(merged)
    else:
        merged = base_config
    apply_patches(merged, patches)

    body = dump_yaml(merged)
    text = (header + "\n" if header else "") + body
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    return GenerationResult(
        config=merged,
        yaml_text=text,
        output_path=output_path,
        base_revision=base.revision,
        base_sha256=base.sha256,
    )
