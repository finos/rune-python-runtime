"""Copy-on-write wrappers for Rune function parameters."""
from __future__ import annotations

import copy
from collections.abc import MutableMapping, MutableSequence, MutableSet
from typing import Any, Callable

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.object_builder import ObjectBuilder
from rune.runtime.reference_list import _ReferenceList
from rune.runtime.utils import rune_mangle_name, rune_resolve_attr

__all__ = ["rune_cow", "rune_unwrap"]


def _clone_for_write(value: Any) -> Any:
    """Create a detached copy when a mutable value is first written."""
    if isinstance(value, BaseDataClass):
        return value.model_copy(deep=True)
    return copy.deepcopy(value)


def _wrap_child(value: Any, on_write: Callable[[Any], Any] | None) -> Any:
    """Wrap mutable child values and pass through immutable values."""
    if isinstance(value, _COWBase):
        return value
    if isinstance(value, list):
        return _COWList(value, on_write=on_write)
    if isinstance(value, dict):
        return _COWDict(value, on_write=on_write)
    if isinstance(value, set):
        return _COWSet(value, on_write=on_write)
    if isinstance(value, ObjectBuilder):
        return _COWBuilder(value, on_write=on_write)
    if isinstance(value, BaseDataClass):
        return _COWObject(value, on_write=on_write)
    return value


class _COWBase:
    """Shared mechanics for COW wrappers."""

    _original: Any
    _shadow: Any | None
    _on_write: Callable[[Any], Any] | None

    def __init__(self, original: Any, on_write: Callable[[Any], Any] | None = None):
        self._original = original
        self._shadow = None
        self._on_write = on_write

    def _current(self) -> Any:
        return self._shadow if self._shadow is not None else self._original

    def _ensure_shadow(self) -> Any:
        if self._shadow is None:
            self._shadow = _clone_for_write(self._original)
            if self._on_write is not None:
                attached = self._on_write(self._shadow)
                if isinstance(self._shadow, _ReferenceList) and isinstance(attached, _ReferenceList):
                    self._shadow = attached
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

    def __init__(self, original: Any,
                 on_write: Callable[[Any], Any] | None = None,
                 writable_target: Callable[[], Any] | None = None) -> None:
        super().__init__(original, on_write)
        self._writable_target = writable_target

    def _ensure_shadow(self) -> Any:
        if self._shadow is None and self._writable_target is not None:
            self._shadow = self._writable_target()
        return super()._ensure_shadow()

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

    def _set_attr(self, name: str, value: Any) -> Any:
        shadow = self._ensure_shadow()
        if isinstance(value, _ReferenceList):
            value = value.copy()
            for index in range(len(value)):
                if reference := value.reference_at(index):
                    try:
                        target = shadow.get_object_by_key(*reference)
                    except KeyError:
                        continue
                    value.bind(index, target, *reference)
        setattr(shadow, name, value)
        self._mark_written()
        return getattr(shadow, name)

    def _rune_get_attr(self, attrib: str) -> Any:
        """Resolver hook used by rune_resolve_attr."""
        target = self._current()
        value = getattr(target, attrib, None)
        return _wrap_child(value, lambda new_value: self._set_attr(attrib, new_value))

    def __repr__(self) -> str:
        return f"COWObject({self._current()!r})"


class _COWBuilder(_COWBase):
    """Proxy for ObjectBuilder instances with copy-on-write semantics."""

    def __getattr__(self, name: str) -> Any:
        key = rune_mangle_name(name)
        target = self._current()
        if key in target._data:  # pylint: disable=protected-access
            value = target._data[key]  # pylint: disable=protected-access
            return _wrap_child(
                value, lambda new_value, key=key: self._set_key(key, new_value)
            )
        return _COWBuilderPath(self, [key])

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._set_key(rune_mangle_name(name), rune_unwrap(value))

    def __getitem__(self, key):
        current = self._current()
        value = current[key]
        return _wrap_child(value, lambda new_value: self._set_item(key, new_value))

    def __setitem__(self, key, value) -> None:
        self._set_item(key, rune_unwrap(value))

    def _set_item(self, key: str, value: Any) -> None:
        shadow = self._ensure_shadow()
        shadow[key] = value
        self._mark_written()

    def _set_key(self, key: str, value: Any) -> None:
        shadow = self._ensure_shadow()
        shadow._data[key] = value  # pylint: disable=protected-access
        self._mark_written()

    def _set_path_value(self, path: list[str], key: str, value: Any) -> None:
        shadow = self._ensure_shadow()
        node = shadow
        for segment in path:
            child = node._data.get(segment)  # pylint: disable=protected-access
            if not isinstance(child, ObjectBuilder):
                child = ObjectBuilder()
                node._data[segment] = child  # pylint: disable=protected-access
            node = child
        node._data[key] = value  # pylint: disable=protected-access
        self._mark_written()

    def _rune_get_attr(self, attrib: str) -> Any:
        target = self._current()
        value = target._data.get(attrib)  # pylint: disable=protected-access
        return _wrap_child(
            value, lambda new_value, key=attrib: self._set_key(key, new_value)
        )

    def to_dict(self) -> dict[str, Any]:
        return rune_unwrap(self).to_dict()

    def to_model(self) -> BaseDataClass:
        return rune_unwrap(self).to_model()

    def __repr__(self) -> str:
        return f"COWBuilder({self._current()!r})"


class _COWBuilderPath:
    """Autovivifying path helper for missing ObjectBuilder attributes."""

    __slots__ = ("_root", "_path")

    _root: _COWBuilder
    _path: list[str]

    def __init__(self, root: _COWBuilder, path: list[str]):
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str):
        key = rune_mangle_name(name)
        return _COWBuilderPath(self._root, self._path + [key])

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        key = rune_mangle_name(name)
        self._root._set_path_value(self._path, key, rune_unwrap(value))


class _COWList(_COWBase, MutableSequence):
    """List wrapper with copy-on-write behavior."""

    def __len__(self) -> int:
        return len(self._current())

    def __getitem__(self, index):
        current = self._current()
        value = current[index]
        return self._wrap_item(index, value)

    def _wrap_item(self, index: int | slice, value: Any) -> Any:
        def callback(new_value: Any) -> None:
            self._set_child(index, new_value)

        current = self._current()
        if (isinstance(current, _ReferenceList) and not isinstance(index, slice)
                and current.reference_at(index) and isinstance(value, BaseDataClass)):
            # Edit the canonical target in the detached graph, so a reference
            # and its containment occurrence still expose the same object.
            return _COWObject(value, on_write=callback,
                              writable_target=lambda: self._ensure_shadow()[index])
        return _wrap_child(value, callback)

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
            yield self._wrap_item(idx, value)

    def _set_child(self, index, value: Any) -> None:
        '''Keep reference intent when a contained object's COW proxy changes.'''
        shadow = self._ensure_shadow()
        reference = (shadow.reference_at(index)
                     if isinstance(shadow, _ReferenceList) and not isinstance(index, slice)
                     else None)
        if reference is not None:
            shadow.bind(index, rune_unwrap(value), *reference)
        else:
            shadow[index] = rune_unwrap(value)
        self._mark_written()

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

        if isinstance(item, _ReferenceList):
            out = item.copy()
            changed = False
            memo[item_id] = out
            for index, value in enumerate(item):
                value_unwrapped, value_changed = _unwrap_inner(value)
                list.__setitem__(out, index, value_unwrapped)
                changed = changed or value_changed
            if changed:
                return out, True
            memo[item_id] = item
            return item, False

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
