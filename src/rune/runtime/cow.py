"""Copy-on-write wrappers for Rune function parameters."""
from __future__ import annotations

import copy
from collections.abc import MutableMapping, MutableSequence, MutableSet
from typing import Any, Callable

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.object_builder import ObjectBuilder

__all__ = ["rune_cow", "rune_unwrap"]


def _clone_for_write(value: Any) -> Any:
    """Create a detached copy when a mutable value is first written."""
    if isinstance(value, BaseDataClass):
        return value.model_copy(deep=True)
    return copy.deepcopy(value)


def _wrap_child(value: Any, on_write: Callable[[Any], None] | None) -> Any:
    """Wrap mutable child values and pass through immutable values."""
    if isinstance(value, _COWBase):
        return value
    if isinstance(value, list):
        return _COWList(value, on_write=on_write)
    if isinstance(value, dict):
        return _COWDict(value, on_write=on_write)
    if isinstance(value, set):
        return _COWSet(value, on_write=on_write)
    if isinstance(value, BaseDataClass):
        return _COWObject(value, on_write=on_write)
    return value


class _COWBase:
    """Shared mechanics for COW wrappers."""

    _original: Any
    _shadow: Any | None
    _on_write: Callable[[Any], None] | None

    def __init__(self, original: Any, on_write: Callable[[Any], None] | None = None):
        self._original = original
        self._shadow = None
        self._on_write = on_write

    def _current(self) -> Any:
        return self._shadow if self._shadow is not None else self._original

    def _ensure_shadow(self) -> Any:
        if self._shadow is None:
            self._shadow = _clone_for_write(self._original)
            if self._on_write is not None:
                self._on_write(self._shadow)
        return self._shadow

    def _mark_written(self) -> None:
        if self._on_write is not None:
            self._on_write(self._current())

    def _unwrap(self) -> Any:
        return self._current()

    def __bool__(self) -> bool:
        return bool(self._current())


class _COWObject(_COWBase):
    """Proxy for mutable model/object instances."""

    def __getattr__(self, name: str) -> Any:
        target = self._current()
        value = getattr(target, name)
        return _wrap_child(value, lambda new_value: self._set_attr(name, new_value))

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._set_attr(name, rune_unwrap(value))

    def __delattr__(self, name: str) -> None:
        shadow = self._ensure_shadow()
        delattr(shadow, name)
        self._mark_written()

    def _set_attr(self, name: str, value: Any) -> None:
        shadow = self._ensure_shadow()
        setattr(shadow, name, value)
        self._mark_written()

    def _rune_get_attr(self, attrib: str) -> Any:
        """Resolver hook used by rune_resolve_attr."""
        target = self._current()
        value = getattr(target, attrib, None)
        return _wrap_child(value, lambda new_value: self._set_attr(attrib, new_value))

    def __repr__(self) -> str:
        return f"COWObject({self._current()!r})"


class _COWList(_COWBase, MutableSequence):
    """List wrapper with copy-on-write behavior."""

    def __len__(self) -> int:
        return len(self._current())

    def __getitem__(self, index):
        current = self._current()
        value = current[index]
        if isinstance(index, slice):
            return value
        return _wrap_child(value, lambda new_value: self._set_item(index, new_value))

    def __setitem__(self, index, value) -> None:
        self._set_item(index, rune_unwrap(value))

    def __delitem__(self, index) -> None:
        shadow = self._ensure_shadow()
        del shadow[index]
        self._mark_written()

    def insert(self, index: int, value: Any) -> None:
        shadow = self._ensure_shadow()
        shadow.insert(index, rune_unwrap(value))
        self._mark_written()

    def __iter__(self):
        current = self._current()
        for idx, value in enumerate(current):
            yield _wrap_child(value, lambda new_value, idx=idx: self._set_item(idx, new_value))

    def _set_item(self, index, value: Any) -> None:
        shadow = self._ensure_shadow()
        shadow[index] = value
        self._mark_written()

    def _rune_get_attr(self, attrib: str) -> Any:
        # Mirrors rune_resolve_attr list flattening semantics.
        res = []
        for elem in self._current():
            elem_value = rune_resolve_attr(elem, attrib)
            if elem_value is None:
                continue
            if isinstance(elem_value, (list, tuple)):
                res.extend(elem_value)
            else:
                res.append(elem_value)
        return res if res else None

    def __repr__(self) -> str:
        return f"COWList({self._current()!r})"


class _COWDict(_COWBase, MutableMapping):
    """Dict wrapper with copy-on-write behavior."""

    def __getitem__(self, key):
        current = self._current()
        value = current[key]
        return _wrap_child(value, lambda new_value: self._set_item(key, new_value))

    def __setitem__(self, key, value) -> None:
        self._set_item(key, rune_unwrap(value))

    def __delitem__(self, key) -> None:
        shadow = self._ensure_shadow()
        del shadow[key]
        self._mark_written()

    def __iter__(self):
        return iter(self._current())

    def __len__(self) -> int:
        return len(self._current())

    def _set_item(self, key, value: Any) -> None:
        shadow = self._ensure_shadow()
        shadow[key] = value
        self._mark_written()

    def _rune_get_attr(self, attrib: str) -> Any:
        return self._current().get(attrib)

    def __repr__(self) -> str:
        return f"COWDict({self._current()!r})"


class _COWSet(_COWBase, MutableSet):
    """Set wrapper with copy-on-write behavior."""

    def __contains__(self, value: Any) -> bool:
        return value in self._current()

    def __iter__(self):
        return iter(self._current())

    def __len__(self) -> int:
        return len(self._current())

    def add(self, value: Any) -> None:
        shadow = self._ensure_shadow()
        shadow.add(rune_unwrap(value))
        self._mark_written()

    def discard(self, value: Any) -> None:
        shadow = self._ensure_shadow()
        shadow.discard(value)
        self._mark_written()

    def _rune_get_attr(self, attrib: str) -> Any:
        res = []
        for elem in self._current():
            elem_value = rune_resolve_attr(elem, attrib)
            if elem_value is None:
                continue
            if isinstance(elem_value, (list, tuple)):
                res.extend(elem_value)
            else:
                res.append(elem_value)
        return res if res else None

    def __repr__(self) -> str:
        return f"COWSet({self._current()!r})"


def rune_cow(value: Any) -> Any:
    """Wrap mutable input values in a copy-on-write proxy."""
    return _wrap_child(value, None)


def rune_unwrap(value: Any) -> Any:
    """Unwrap COW proxies and recursively unwrap nested values."""

    memo: dict[int, Any] = {}

    def _unwrap_inner(item: Any) -> tuple[Any, bool]:
        item_id = id(item)
        if item_id in memo:
            return memo[item_id], True

        if isinstance(item, _COWBase):
            unwrapped, _ = _unwrap_inner(item._unwrap())
            return unwrapped, True

        if isinstance(item, list):
            out = []
            changed = False
            memo[item_id] = out
            for value in item:
                value_unwrapped, value_changed = _unwrap_inner(value)
                out.append(value_unwrapped)
                changed = changed or value_changed
            if changed:
                return out, True
            memo[item_id] = item
            return item, False

        if isinstance(item, tuple):
            out = []
            changed = False
            for value in item:
                value_unwrapped, value_changed = _unwrap_inner(value)
                out.append(value_unwrapped)
                changed = changed or value_changed
            if changed:
                return tuple(out), True
            return item, False

        if isinstance(item, set):
            out = set()
            changed = False
            memo[item_id] = out
            for value in item:
                value_unwrapped, value_changed = _unwrap_inner(value)
                out.add(value_unwrapped)
                changed = changed or value_changed
            if changed:
                return out, True
            memo[item_id] = item
            return item, False

        if isinstance(item, dict):
            out: dict[Any, Any] = {}
            changed = False
            memo[item_id] = out
            for key, value in item.items():
                key_unwrapped, key_changed = _unwrap_inner(key)
                value_unwrapped, value_changed = _unwrap_inner(value)
                out[key_unwrapped] = value_unwrapped
                changed = changed or key_changed or value_changed
            if changed:
                return out, True
            memo[item_id] = item
            return item, False

        if isinstance(item, ObjectBuilder):
            changed = False
            clone = ObjectBuilder(getattr(item, "_model_cls", None))
            memo[item_id] = clone
            for key, current in item._data.items():  # pylint: disable=protected-access
                unwrapped, field_changed = _unwrap_inner(current)
                clone._data[key] = unwrapped  # pylint: disable=protected-access
                changed = changed or field_changed
            if changed:
                return clone, True
            memo[item_id] = item
            return item, False

        if isinstance(item, BaseDataClass):
            field_values: dict[str, Any] = {}
            changed = False
            field_names = type(item).__pydantic_fields__.keys()
            for name in field_names:
                current = getattr(item, name)
                unwrapped, field_changed = _unwrap_inner(current)
                field_values[name] = unwrapped
                changed = changed or field_changed
            if not changed:
                return item, False

            clone = item.model_copy(deep=True)
            memo[item_id] = clone
            for name, unwrapped in field_values.items():
                setattr(clone, name, unwrapped)
            return clone, True

        return item, False

    unwrapped, _ = _unwrap_inner(value)
    return unwrapped

# EOF
