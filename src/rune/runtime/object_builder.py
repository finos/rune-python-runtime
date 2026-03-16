'''Draft helpers for dynamic property access.'''
from __future__ import annotations

import copy
from typing import Any

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.utils import rune_mangle_name

__all__ = ['ObjectBuilder']


class ObjectBuilder:
    '''Autovivifying object for dynamic property access.'''
    _data: dict[str, Any]
    _model_cls: type[BaseDataClass] | None

    def __init__(self,
                 model_cls: type[BaseDataClass] | None = None,
                 /,
                 **initial):
        object.__setattr__(self, '_data', {})
        object.__setattr__(self, '_model_cls', model_cls)
        if initial:
            for k, v in initial.items():
                self._data[rune_mangle_name(k)] = v

    def __getattr__(self, name):
        name = rune_mangle_name(name)
        if name in self._data:
            return self._data[name]
        return _PathDraft(self, [name])

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        name = rune_mangle_name(name)
        self._data[name] = value

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def _rune_get_attr(self, attrib: str):
        # "attrib" is already mangled by rune_resolve_attr.
        return self._data.get(attrib)

    def to_dict(self) -> dict[str, Any]:
        '''Return a plain dict, converting nested Draft values recursively.'''
        return {k: self._normalize(v) for k, v in self._data.items()}

    def to_model(self) -> BaseDataClass:
        '''Create a BaseDataClass instance from collected data.'''
        if self._model_cls is None:
            raise ValueError('Draft has no model class configured.')
        data = self.to_dict()
        if hasattr(self._model_cls, 'model_validate'):
            return self._model_cls.model_validate(data)
        return self._model_cls(**data)

    def __deepcopy__(self, memo):
        clone = ObjectBuilder(self._model_cls)
        memo[id(self)] = clone
        clone._data.update({
            key: copy.deepcopy(value, memo)
            for key, value in self._data.items()
        })
        return clone

    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, ObjectBuilder):
            return value.to_dict()
        if isinstance(value, dict):
            return {k: cls._normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._normalize(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._normalize(v) for v in value)
        return value

    def __repr__(self):
        data = ', '.join(f'{k}={v!r}' for k, v in self._data.items())
        return f'Draft({data})'


class _PathDraft:
    __slots__ = ('_root', '_path')
    _root: ObjectBuilder
    _path: list[str]

    def __init__(self, root: ObjectBuilder, path: list[str]):
        object.__setattr__(self, '_root', root)
        object.__setattr__(self, '_path', path)

    def __getattr__(self, name):
        name = rune_mangle_name(name)
        return _PathDraft(self._root, self._path + [name])

    def __setattr__(self, name, value):
        if name in ('_root', '_path'):
            object.__setattr__(self, name, value)
            return
        name = rune_mangle_name(name)
        node = self._root
        for seg in self._path:
            child = node._data.get(seg)
            if not isinstance(child, ObjectBuilder):
                child = ObjectBuilder()
                node._data[seg] = child
            node = child
        node._data[name] = value
