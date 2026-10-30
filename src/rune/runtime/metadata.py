# Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

'''Classes representing annotated basic Rune types'''
import copy
import datetime
import importlib
import uuid
import weakref
from collections.abc import Iterable, MutableSequence
from decimal import Decimal
from enum import Enum
from functools import lru_cache, partial
from typing import Any, Never, Self, get_args

from pydantic import (PlainSerializer, PlainValidator,
                      ValidatorFunctionWrapHandler, WrapSerializer, WrapValidator)
from pydantic_core import PydanticCustomError

# from rune.runtime.object_registry import get_object

DEFAULT_META = '_ALLOWED_METADATA'
META_CONTAINER = '__rune_metadata'
REFS_CONTAINER = '__rune_references'
PARENT_PROP = '__rune_parent'
RUNE_OBJ_MAPS = '__rune_object_maps'


def _replaceable(prop):
    return isinstance(prop, (BaseMetaDataMixin, UnresolvedReference, Reference))


def _py_to_ser_key(key: str) -> str:
    if key[0] == '@':
        return key
    return '@' + key.replace('_', ':')


def _get_basic_type(annotated_type):
    embedded_type = get_args(annotated_type)
    if embedded_type:
        for arg in embedded_type:
            if arg is type(None) or (
                isinstance(arg, type) and issubclass(arg, BaseReference)
            ):
                continue
            return _get_basic_type(arg)
        return _get_basic_type(embedded_type[0])
    return annotated_type


def _extract_allowed_refs(tp_or_meta: Any) -> set[str]:
    '''Extract allowed @ref tags from field validator metadata or annotations.'''
    allowed: set[str] = set()
    func = getattr(tp_or_meta, 'func', tp_or_meta)
    meta = getattr(func, '_allowed_meta', None)
    if meta is None and isinstance(func, partial):
        meta = func.keywords.get('allowed_meta')
    if meta:
        allowed.update(m for m in meta if m.startswith('@ref'))
    for arg in get_args(tp_or_meta):
        allowed.update(_extract_allowed_refs(arg))
    return allowed


def _unwrap_cow_proxy(value: Any) -> Any:
    '''Return the current value behind a copy-on-write proxy, if present.'''
    unwrap = getattr(value, '_unwrap', None)
    if callable(unwrap):
        return unwrap()
    return value


class KeyType(Enum):
    '''Enum for the currently supported by Rune external keys/refs'''
    INTERNAL = 'internal'
    EXTERNAL = 'external'
    SCOPED = 'scoped'

    @property
    def key_tag(self):
        '''the key tag as used internally'''
        if self == KeyType.INTERNAL:
            return 'key'
        return f'key_{self.value}'

    @property
    def rune_key_tag(self):
        '''the key tag as represented in rune'''
        if self == KeyType.INTERNAL:
            return '@key'
        return f'@key:{self.value}'

    @property
    def ref_tag(self):
        '''the ref tag as used internally'''
        if self == KeyType.INTERNAL:
            return 'ref'
        return f'ref_{self.value}'

    @property
    def rune_ref_tag(self):
        '''the ref tag as represented in rune'''
        if self == KeyType.INTERNAL:
            return '@ref'
        return f'@ref:{self.value}'

    @classmethod
    def from_rune(cls, rune_item: str):
        '''returns an enum instance for the passed in rune key/ref'''
        rune_consts = rune_item.split(':')
        rune_type = rune_consts[-1] if len(rune_consts) > 1 else 'internal'
        return KeyType(rune_type)


class BaseReference:
    ''' A base class allowing for Refrence and UnresolvedReference to be used 
        in model construction
    '''


class Reference(BaseReference):
    '''manages a reference to a object with a key'''
    def __init__(self,
                 target: str | Any,
                 ext_key: str | None = None,
                 key_type: KeyType | None = None,
                 parent=None):
        target = _unwrap_cow_proxy(target)
        parent = _unwrap_cow_proxy(parent)
        if not isinstance(target, BaseMetaDataMixin) and ext_key:
            raise ValueError('Need to pass an object as target when specifying '
                             'an external key!')
        if ext_key:
            key_type = key_type or KeyType.EXTERNAL
            target.set_external_key(ext_key, key_type)  # type: ignore
            self.target = target
            self.target_key = ext_key
            self.key_type = key_type
        elif isinstance(target, BaseMetaDataMixin):
            if key_type and key_type != KeyType.INTERNAL:
                raise ValueError('key_type should be None or INTERNAL when '
                                 'passing in an object without a key!')
            self.target = target
            self.target_key = target.get_or_create_key()
            self.key_type = KeyType.INTERNAL
        else:
            key_type = key_type or KeyType.EXTERNAL
            self.target_key = target
            self.key_type = key_type
            if not parent:
                raise ValueError('When creating a reference only with an '
                                 'external key, a parent object must be '
                                 'specified!')
            self.target = parent.get_object_by_key(target, key_type)

    def get_reference(self, _):
        '''returns itself reference'''
        return self


class UnresolvedReference(BaseReference):
    '''used by the deserialization to hold temporarily unresolved references'''
    def __init__(self, key):
        rune_type, self.key = next(iter(key.items()))
        self.key_type = KeyType.from_rune(rune_type)

    def get_reference(self, parent):
        '''convert to a resolved reference'''
        return Reference(self.key, key_type=self.key_type, parent=parent)


_MISSING = object()


class _ReferenceList(list):
    '''List subclass that keeps per-element Rune reference bindings in sync.'''

    def __init__(
        self,
        iterable: Iterable[Any] = (),
        *,
        ref_entries: (
            Iterable[tuple[str, KeyType] | None]
            | dict[int, tuple[str, KeyType]]
            | None
        ) = None,
        owner: Any | None = None,
        prop_name: str | None = None,
    ):
        items = list(iterable)
        super().__init__(items)
        if isinstance(ref_entries, dict):
            self._ref_entries: list[tuple[str, KeyType] | None] = [
                ref_entries.get(idx) for idx in range(len(items))
            ]
        elif ref_entries is not None:
            entries = list(ref_entries)
            if len(entries) < len(items):
                entries.extend([None] * (len(items) - len(entries)))
            self._ref_entries = entries[:len(items)]
        elif isinstance(iterable, _ReferenceList):
            self._ref_entries = list(iterable._ref_entries[:len(items)])
        else:
            self._ref_entries = [None] * len(items)
        self._owner_ref = weakref.ref(owner) if owner is not None else None
        self._prop_name = prop_name

    def _owner(self) -> Any | None:
        return self._owner_ref() if self._owner_ref is not None else None

    def _bind_owner(self, owner: Any, prop_name: str) -> None:
        self._owner_ref = weakref.ref(owner)
        self._prop_name = prop_name
        self._sync_owner_refs()

    def _sync_owner_refs(self) -> None:
        owner = self._owner()
        if owner is None or not self._prop_name:
            return
        refs = owner.__dict__.setdefault(REFS_CONTAINER, {})
        mapping = {
            idx: entry
            for idx, entry in enumerate(self._ref_entries)
            if entry is not None
        }
        if mapping:
            refs[self._prop_name] = mapping
        else:
            refs.pop(self._prop_name, None)

    def _prepare_element(
        self,
        item: Any,
        existing_entry: tuple[str, KeyType] | None = None,
        current_item: Any = _MISSING,
        is_already_ref: bool = False,
        index: int | None = None,
    ) -> tuple[Any, tuple[str, KeyType] | None]:
        item = _unwrap_cow_proxy(item)
        owner = self._owner()
        prop_name = self._prop_name
        if isinstance(item, Reference):
            if owner is not None and prop_name:
                # pylint: disable=protected-access
                owner._validate_ref_for_property(
                    prop_name,
                    item,
                    current_item=current_item,
                    is_already_ref=is_already_ref,
                    index=index,
                )
            return item.target, (item.target_key, item.key_type)
        if existing_entry is not None:
            key, key_type = existing_entry
            if owner is not None:
                try:
                    registered = owner.get_object_by_key(key, key_type)
                except KeyError:
                    registered = None
                if registered is not None:
                    if item is registered or (
                        isinstance(item, BaseMetaDataMixin)
                        and item.get_meta(key_type.key_tag) == key
                    ):
                        return registered, existing_entry
                    return item, None
            if (
                isinstance(item, BaseMetaDataMixin)
                and item.get_meta(key_type.key_tag) == key
            ):
                return item, existing_entry
        return item, None

    def _attach_inline_parent(
        self,
        item: Any,
        entry: tuple[str, KeyType] | None,
    ) -> None:
        owner = self._owner()
        if (
            owner is not None
            and entry is None
            and isinstance(item, BaseMetaDataMixin)
        ):
            # pylint: disable=protected-access
            item._set_rune_parent(owner)

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if isinstance(index, slice):
            return _ReferenceList(
                result,
                ref_entries=self._ref_entries[index],
            )
        return result

    def __setitem__(self, index, value) -> None:
        if isinstance(index, slice):
            if isinstance(value, _ReferenceList):
                raw_items = list(value)
                raw_entries = list(value._ref_entries)
            else:
                raw_items = list(value)
                raw_entries = [None] * len(raw_items)
            prepared = [
                self._prepare_element(v, existing_entry=e)
                for v, e in zip(raw_items, raw_entries)
            ]
            new_items = [p[0] for p in prepared]
            new_entries = [p[1] for p in prepared]
            super().__setitem__(index, new_items)
            self._ref_entries[index] = new_entries
            for item, entry in prepared:
                self._attach_inline_parent(item, entry)
            self._sync_owner_refs()
            return

        old_item = super().__getitem__(index)
        old_entry = self._ref_entries[index]
        val, entry = self._prepare_element(
            value,
            current_item=old_item,
            is_already_ref=old_entry is not None,
            index=index,
        )
        super().__setitem__(index, val)
        self._ref_entries[index] = entry
        self._attach_inline_parent(val, entry)
        self._sync_owner_refs()

    def __delitem__(self, index) -> None:
        super().__delitem__(index)
        del self._ref_entries[index]
        self._sync_owner_refs()

    def pop(self, index: int = -1) -> Any:
        val = super().pop(index)
        self._ref_entries.pop(index)
        self._sync_owner_refs()
        return val

    def append(self, value: Any) -> None:
        val, entry = self._prepare_element(value)
        super().append(val)
        self._ref_entries.append(entry)
        self._attach_inline_parent(val, entry)
        self._sync_owner_refs()

    def insert(self, index: int, value: Any) -> None:
        val, entry = self._prepare_element(value)
        super().insert(index, val)
        self._ref_entries.insert(index, entry)
        self._attach_inline_parent(val, entry)
        self._sync_owner_refs()

    def extend(self, iterable: Iterable[Any]) -> None:
        if isinstance(iterable, _ReferenceList):
            raw_items = list(iterable)
            raw_entries = list(iterable._ref_entries)
        else:
            raw_items = list(iterable)
            raw_entries = [None] * len(raw_items)
        prepared = [
            self._prepare_element(v, existing_entry=e)
            for v, e in zip(raw_items, raw_entries)
        ]
        super().extend(p[0] for p in prepared)
        self._ref_entries.extend(p[1] for p in prepared)
        for item, entry in prepared:
            self._attach_inline_parent(item, entry)
        self._sync_owner_refs()

    def __iadd__(self, iterable: Iterable[Any]) -> Self:
        self.extend(iterable)
        return self

    def __add__(self, other: Any) -> _ReferenceList:
        if not isinstance(other, list):
            return super().__add__(other)  # type: ignore[return-value]
        other_entries = (
            list(other._ref_entries)
            if isinstance(other, _ReferenceList)
            else [None] * len(other)
        )
        return _ReferenceList(
            super().__add__(other),
            ref_entries=self._ref_entries + other_entries,
        )

    def __radd__(self, other: Any) -> _ReferenceList:
        if not isinstance(other, list):
            return NotImplemented
        other_entries = (
            list(other._ref_entries)
            if isinstance(other, _ReferenceList)
            else [None] * len(other)
        )
        return _ReferenceList(
            list(other) + list(self),
            ref_entries=other_entries + self._ref_entries,
        )

    def reverse(self) -> None:
        super().reverse()
        self._ref_entries.reverse()
        self._sync_owner_refs()

    def clear(self) -> None:
        super().clear()
        self._ref_entries.clear()
        self._sync_owner_refs()

    def remove(self, value: Any) -> None:
        idx = self.index(value)
        del self[idx]

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        pairs = list(zip(self, self._ref_entries))
        if key is None:
            pairs.sort(key=lambda pair: pair[0], reverse=reverse)
        else:
            pairs.sort(key=lambda pair: key(pair[0]), reverse=reverse)
        super().__setitem__(slice(None), [p[0] for p in pairs])
        self._ref_entries[:] = [p[1] for p in pairs]
        self._sync_owner_refs()

    def copy(self) -> _ReferenceList:
        return _ReferenceList(self, ref_entries=self._ref_entries)

    def __copy__(self) -> _ReferenceList:
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> _ReferenceList:
        clone = _ReferenceList()
        memo[id(self)] = clone
        super(_ReferenceList, clone).extend(
            copy.deepcopy(item, memo) for item in self
        )
        clone._ref_entries = list(self._ref_entries)
        return clone


class BaseMetaDataMixin:
    '''Base class for the meta data support of basic amd complex types'''
    _DEFAULT_SCOPE_TYPE = 'cdm.event.common.TradeState'
    __meta_check_disabled = False

    @classmethod
    def _fqrtn(cls):
        '''returns the fully qualified'''
        if explicit_fqrtn := getattr(cls, '_FQRTN', None):
            return explicit_fqrtn

        module_name = cls.__module__
        if prefix := cls._get_rune_namespace_prefix():
            prefixed_module = prefix + '.'
            if module_name.startswith(prefixed_module):
                return module_name[len(prefixed_module):]
        return module_name

    @classmethod
    def _get_rune_namespace_prefix(cls) -> str | None:
        try:
            module = importlib.import_module(
                cls.__module__.split('.', maxsplit=1)[0])
            return getattr(module, 'rune_namespace_prefix', None)
        # pylint: disable=bare-except
        except:  # noqa
            return None

    @classmethod
    def enable_meta_checks(cls):
        '''enables the metadata checks in deserialize'''
        BaseMetaDataMixin.__meta_check_disabled = False

    @classmethod
    def disable_meta_checks(cls):
        '''disables the metadata checks in deserialize'''
        BaseMetaDataMixin.__meta_check_disabled = True

    @classmethod
    def meta_checks_enabled(cls):
        '''is metadata checked during deserialize'''
        return not BaseMetaDataMixin.__meta_check_disabled

    def is_scope_instance(self):
        '''is this object a scope for `scoped` keys/references'''
        if not (scope := self._get_rune_scope_type()):
            scope = self._DEFAULT_SCOPE_TYPE
        # if not (fqcn := getattr(self, '_FQRTN', None)):
        #     fqcn = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
        # return fqcn == scope
        return self._fqrtn() == scope

    def set_meta(self, check_allowed=True, **kwds):
        '''set some/all metadata properties'''
        props = {_py_to_ser_key(k): v for k, v in kwds.items()}
        if check_allowed:
            self._check_props_allowed(props)
        meta = self.__dict__.setdefault(META_CONTAINER, {})
        meta |= props

    def get_meta(self, name: str):
        '''get a metadata property'''
        return self._get_meta_container().get(_py_to_ser_key(name))

    def serialise_meta(self) -> dict:
        '''used as serialisation method with pydantic'''
        metadata = self._get_meta_container()
        return {key: value for key, value in metadata.items() if value}

    def get_or_create_key(self) -> str:
        '''gets or creates the key associated with this object'''
        if not (key := self.get_meta('key')):
            key = str(uuid.uuid4())
            self.set_meta(key=key)
            try:
                self._get_object_map(KeyType.INTERNAL)[key] = self
            except:
                self.set_meta(key=None)
                raise
        return key

    def set_external_key(self,
                         key: str,
                         key_type: KeyType = KeyType.EXTERNAL) -> Self:
        '''registers this object under the provided external key'''
        aux = self.get_meta(key_type.key_tag)
        if aux and aux != key:
            raise ValueError(f'This object already has an external key {aux}!'
                             f'Can\'t change it to {key}')
        if aux == key:
            return self

        self.set_meta(check_allowed=True, **{key_type.key_tag: key})
        try:
            self._get_object_map(key_type)[key] = self
        except:
            self.set_meta(check_allowed=True, **{key_type.key_tag: None})
            raise
        return self

    def get_object_by_key(self, key: str, key_type: KeyType):
        '''retrieve an object with a key an key type'''
        return self._get_object_map(key_type)[key]

    def get_rune_parent(self) -> Self | None:
        '''the parent object'''
        return self.__dict__.get(PARENT_PROP)

    def _copy_rune_dict(self) -> dict[str, Any]:
        '''Shallow-copy instance attributes without retaining the parent link.'''
        copied = copy.copy(self.__dict__)
        copied.pop(PARENT_PROP, None)
        return copied

    def _deepcopy_rune_dict(self, memo: dict[int, Any]) -> dict[str, Any]:
        '''Deep-copy instance attributes without walking up the parent chain.'''
        parent = self.__dict__.get(PARENT_PROP)
        cloned_parent = memo.get(id(parent)) if parent is not None else None
        copied = {
            k: copy.deepcopy(v, memo)
            for k, v in self.__dict__.items()
            if k != PARENT_PROP
        }
        if cloned_parent is not None:
            copied[PARENT_PROP] = cloned_parent
        return copied

    def resolve_ref_key(self,
                        property_name: str,
                        index: int | None = None) -> Any:
        ''' if the property exists and it is a reference, return the reference,
            otherwise return None
        '''
        refs = self._get_rune_refs_container()
        entry = refs.get(property_name)
        if entry is None:
            return None
        if isinstance(entry, dict):
            if index is not None:
                item = entry.get(index)
                return item[0] if item is not None else None
            return [item[0] for _, item in sorted(entry.items())]
        return entry[0]

    def _get_allowed_ref_types(self, property_nm: str) -> set[str]:
        '''Return the allowed @ref tags for property_nm.'''
        if hasattr(self, '_KEY_REF_CONSTRAINTS'):
            constraints = getattr(self, '_KEY_REF_CONSTRAINTS') or {}
            return set(constraints.get(property_nm, ()))
        model_fields = getattr(type(self), 'model_fields', {})
        field_info = model_fields.get(property_nm)
        if field_info is None:
            return set()
        allowed = _extract_allowed_refs(field_info.annotation)
        for item in getattr(field_info, 'metadata', ()):
            allowed.update(_extract_allowed_refs(item))
        return allowed

    def _get_meta_container(self) -> dict[str, Any]:
        return self.__dict__.get(META_CONTAINER, {})

    def _merged_allowed_meta(
            self, allowed_meta: set[str] | Iterable[str]) -> set[str]:
        default_meta: set[str] = getattr(self, DEFAULT_META, set())
        return set(allowed_meta) | default_meta

    def _check_props_allowed(self, props: dict[str, Any]):
        if not props:
            return
        allowed = self._merged_allowed_meta(self._get_meta_container().keys())
        prop_keys = set(props.keys())
        if not prop_keys.issubset(allowed):
            raise ValueError('Not allowed metadata provided: '
                             f'{prop_keys - allowed}')

    def _init_meta(self, allowed_meta: set[str]):
        ''' if not initialised, just creates empty meta slots. If the metadata
            container is not empty, it will check if the already present keys
            are conform to the allowed keys.
        '''
        allowed_meta = self._merged_allowed_meta(allowed_meta)
        meta = self.__dict__.setdefault(META_CONTAINER, {})
        current_meta = set(meta.keys())
        if not current_meta.issubset(allowed_meta):
            raise ValueError(f'Allowed meta {allowed_meta} differs from the '
                             f'currently existing meta slots: {current_meta}')
        meta |= {k: None for k in allowed_meta - current_meta}

    def _validate_ref_for_property(
        self,
        property_nm: str,
        ref: Reference,
        *,
        current_item: Any = _MISSING,
        is_already_ref: bool = False,
        index: int | None = None,
    ) -> None:
        '''Validate that ref is permitted and type-compatible for property_nm.'''
        allowed_refs = self._get_allowed_ref_types(property_nm)
        if ref.key_type.rune_ref_tag not in allowed_refs:
            raise ValueError(f'Ref of type {ref.key_type} '
                             f'not allowed for {property_nm}. Allowed types '
                             f'are: {allowed_refs}')

        # Pydantic includes inherited fields and resolves their annotations.
        field_type = type(self).model_fields[property_nm].annotation  # type: ignore
        allowed_type = _get_basic_type(field_type)
        target_obj = (ref.target.enum_instance
                      if isinstance(ref.target, _EnumWrapper) else ref.target)
        if not (isinstance(allowed_type, str)
                or isinstance(target_obj, allowed_type)):
            raise TypeError("Can't set reference. Incompatible types: "
                            f"expected {allowed_type}, "
                            f"got {ref.target.__class__}")

        if current_item is not _MISSING and not is_already_ref:
            if not _replaceable(current_item):
                if index is not None:
                    raise ValueError(f'Property {property_nm}[{index}] of type '
                                     f"{type(current_item)} can't be a reference")
                raise ValueError(f'Property {property_nm} of type '
                                 f"{type(current_item)} can't be a reference")
            # pylint: disable=protected-access
            if isinstance(current_item, BaseMetaDataMixin):
                current_item._check_props_allowed({ref.key_type.rune_ref_tag: ''})

    def _bind_property_to(self,
                          property_nm: str,
                          ref: Reference,
                          index: int | None = None):
        '''set the property to reference the object referenced by the key'''
        old_val = getattr(self, property_nm)
        refs = self.__dict__.setdefault(REFS_CONTAINER, {})
        if index is not None:
            if isinstance(old_val, tuple):
                old_val = list(old_val)
            if not isinstance(old_val, MutableSequence):
                raise ValueError(f'Property {property_nm} of type '
                                 f"{type(old_val)} is not a sequence")
            old_item = old_val[index]
            prop_refs = refs.get(property_nm)
            is_already_ref = isinstance(prop_refs, dict) and index in prop_refs
            self._validate_ref_for_property(
                property_nm,
                ref,
                current_item=old_item,
                is_already_ref=is_already_ref,
                index=index,
            )
            if not isinstance(old_val, _ReferenceList):
                old_val = _ReferenceList(
                    old_val,
                    ref_entries=prop_refs if isinstance(prop_refs, dict) else None,
                    owner=self,
                    prop_name=property_nm,
                )
                self.__dict__[property_nm] = old_val
            elif old_val._owner() is not self or old_val._prop_name != property_nm:
                old_val._bind_owner(self, property_nm)
            super(_ReferenceList, old_val).__setitem__(index, ref.target)
            old_val._ref_entries[index] = (ref.target_key, ref.key_type)
            old_val._sync_owner_refs()
            if hasattr(self, '__pydantic_fields_set__'):
                self.__pydantic_fields_set__.add(property_nm)  # type: ignore
            return

        self._validate_ref_for_property(
            property_nm,
            ref,
            current_item=old_val,
            is_already_ref=property_nm in refs,
        )

        # setattr(self, property_nm, ref.target)  # nope - need to avoid here!
        self.__dict__[property_nm] = ref.target  # NOTE: avoid here setattr
        refs[property_nm] = (ref.target_key, ref.key_type)
        if hasattr(self, '__pydantic_fields_set__'):
            self.__pydantic_fields_set__.add(property_nm)  # type: ignore

    def _register_keys(self, metadata):
        keys = {
            k: v
            for k, v in metadata.items() if k.startswith('@key') and v
        }
        for key_t, key_v in keys.items():
            self._get_object_map(KeyType.from_rune(key_t))[key_v] = self

    def _get_object_map(self, key_type: KeyType) -> dict[str, Any]:
        if not self.get_rune_parent():
            object_maps = self.__dict__.setdefault(RUNE_OBJ_MAPS, {})
            return object_maps.setdefault(key_type, {})
        if key_type == KeyType.SCOPED and self.is_scope_instance():
            object_maps = self.__dict__.setdefault(RUNE_OBJ_MAPS, {})
            return object_maps.setdefault(key_type, {})
        if local_map := self.__dict__.get(RUNE_OBJ_MAPS, {}).get(key_type):
            return local_map
        # pylint: disable=protected-access
        return self.get_rune_parent()._get_object_map(key_type)  # type:ignore

    def _set_rune_parent(self, parent: Self):
        '''sets the parent object'''
        self.__dict__[PARENT_PROP] = parent
        if obj_maps := self.__dict__.pop(RUNE_OBJ_MAPS, None):
            # pylint: disable=protected-access
            self._update_object_maps(obj_maps)

    def _extract_scoped_map(self, maps):
        scoped = None
        if self.is_scope_instance():
            scoped = maps.pop(KeyType.SCOPED, None)
        return scoped, maps

    def _update_object_maps(self, new_maps):
        if parent := self.get_rune_parent():
            scoped, reduced_maps = self._extract_scoped_map(new_maps)
            # pylint: disable=protected-access
            parent._update_object_maps(reduced_maps)
            if not scoped:
                return
            new_maps = {KeyType.SCOPED: scoped}

        obj_maps = self.__dict__.setdefault(RUNE_OBJ_MAPS, {})
        for map_type, new_map in new_maps.items():
            local_map = obj_maps.setdefault(map_type, {})
            if dup_keys := set(local_map.keys()).intersection(
                    set(new_map.keys())):
                raise ValueError('Duplicated keys detected in updating the '
                                 f'object map {map_type}. '
                                 f'Duplicated keys {dup_keys}')
            local_map |= new_map

    def _get_rune_refs_container(self):
        '''return the dictionary of the refs held'''
        return self.__dict__.get(REFS_CONTAINER, {})

    def _remove_rune_ref(self, name):
        '''remove a reference'''
        return self.__dict__[REFS_CONTAINER].pop(name)

    @classmethod
    def _create_unresolved_ref(cls, metadata) -> UnresolvedReference | None:
        if ref := {k: v for k, v in metadata.items() if k.startswith('@ref')}:
            if len(ref) != 1:
                ref.pop(KeyType.INTERNAL.rune_ref_tag, None)
                if len(ref) != 1:
                    ref.pop(KeyType.EXTERNAL.rune_ref_tag, None)
                    if len(ref) != 1:
                        raise ValueError(f'Multiple references found: {ref}!')
            return UnresolvedReference(ref)
        return None

    @classmethod
    @lru_cache
    def _get_rune_scope_type(cls):
        ''' Attempt to obtain the name of the rune scoping type,
            in case of a failure, None will be returned.
        '''
        try:
            module = importlib.import_module(
                cls.__module__.split('.', maxsplit=1)[0])
            return getattr(module, 'rune_scope_type', None)
        # pylint: disable=bare-except
        except:  # noqa
            return None


class ComplexTypeMetaDataMixin(BaseMetaDataMixin):
    '''metadata support for complex types'''
    @classmethod
    def _type_to_cls(cls,
                     metadata:dict[str, Any],
                     namespace_prefix: str | None = None):
        if rune_type:= metadata.pop('@type', None):
            rune_class_name = rune_type.rsplit('.', maxsplit=1)[-1]
            import_path = rune_type
            prefix = namespace_prefix
            if prefix is None:
                prefix = cls._get_rune_namespace_prefix()
            if prefix and not rune_type.startswith(prefix + '.'):
                import_path = prefix + '.' + rune_type
            rune_module = importlib.import_module(import_path)
            return getattr(rune_module, rune_class_name)
        return cls  # support for legacy json

    @classmethod
    def serialise(cls, obj) -> dict | list | None:
        '''used as serialisation method with pydantic'''
        obj = _unwrap_cow_proxy(obj)
        if obj is None:
            return None
        if isinstance(obj, list):
            return [cls.serialise(item) for item in obj]
        if isinstance(obj, UnresolvedReference):
            return {obj.key_type.rune_ref_tag: obj.key}
        if isinstance(obj, Reference):
            return {obj.key_type.rune_ref_tag: obj.target_key}
        res = obj.serialise_meta()
        res |= obj.model_dump(exclude_unset=True, exclude_defaults=True)
        if cls != obj.__class__:
            # pylint: disable=protected-access
            # res = {'@type': obj._FQRTN} | res
            res = {'@type': obj._fqrtn()} | res
        return res

    @classmethod
    def deserialize(cls, obj, allowed_meta: set[str]):
        '''method used as pydantic `validator`'''
        obj = _unwrap_cow_proxy(obj)
        if isinstance(obj, list):
            return [cls.deserialize(item, allowed_meta) for item in obj]

        if isinstance(obj, cls):
            if cls.meta_checks_enabled():
                obj._init_meta(allowed_meta)  # pylint: disable=protected-access
            return obj

        if isinstance(obj, (Reference, UnresolvedReference)):
            return obj

        if not isinstance(obj, dict):
            raise PydanticCustomError('Input Validation Error',
                                      'Expected either {my_type} or dict but '
                                      'got {type}.',
                                      {'type': type(obj), 'my_type': cls})
        metadata, fields = {}, {}
        for k, v in obj.items():
            (metadata if k.startswith('@') else fields)[k] = v

        # References deserialization treatment
        if aux := cls._create_unresolved_ref(metadata):
            return aux

        # Model creation
        rune_cls = cls._type_to_cls(metadata)
        if rune_cls != cls and not issubclass(rune_cls, cls):
            raise ValueError(f'{rune_cls} has to be a child class of {cls}!')
        model = rune_cls.model_validate(fields)  # type: ignore
        model.__dict__[META_CONTAINER] = metadata
        if cls.meta_checks_enabled():
            model._init_meta(allowed_meta)  # pylint: disable=protected-access

        # Keys deserialization treatment
        model._register_keys(metadata)  # pylint: disable=protected-access
        return model

    @classmethod
    @lru_cache
    def serializer(cls):
        '''should return the validator for the specific class'''
        return PlainSerializer(cls.serialise, return_type=dict | list | None)

    @classmethod
    @lru_cache
    def validator(cls, allowed_meta: tuple[str, ...] = ()) -> WrapValidator:
        '''Let the field schema validate None and deserialize Rune values.'''
        allowed = set(allowed_meta)

        def validate(obj: Any, handler: ValidatorFunctionWrapHandler) -> Any:
            if obj is None:
                return handler(None)
            return cls.deserialize(obj, allowed)

        validate._allowed_meta = allowed  # type: ignore[attr-defined]  # pylint: disable=protected-access
        return WrapValidator(validate, json_schema_input_type=dict)


class BasicTypeMetaDataMixin(BaseMetaDataMixin):
    '''holds the metadata associated with an instance'''
    _INPUT_TYPES: Any | tuple[Any, ...] = str  # to be overridden by subclasses
    _OUTPUT_TYPE: Any = str  # to be overridden by subclasses
    _JSON_OUTPUT = str | dict

    @classmethod
    def _check_type(cls, value):
        if not isinstance(value, cls._INPUT_TYPES):
            raise TypeError(f'{cls.__name__} can be instantiated only with '
                            f'one of the following type(s): {cls._INPUT_TYPES},'
                            f' however the value is of type {type(value)}')

    @classmethod
    def serialise(cls, obj, base_type) -> dict | list | None:
        '''used as serialisation method with pydantic'''
        obj = _unwrap_cow_proxy(obj)
        if obj is None:
            return None
        if isinstance(obj, list):
            return [cls.serialise(item, base_type) for item in obj]
        if isinstance(obj, UnresolvedReference):
            return {obj.key_type.rune_ref_tag: obj.key}
        if isinstance(obj, Reference):
            return {obj.key_type.rune_ref_tag: obj.target_key}
        res = obj.serialise_meta() if isinstance(obj, BaseMetaDataMixin) else {}
        res['@data'] = base_type(obj)
        return res

    @classmethod
    def deserialize(cls, obj, handler, base_types, allowed_meta: set[str]):
        '''method used as pydantic `validator`'''
        obj = _unwrap_cow_proxy(obj)
        if obj is None:
            return handler(None)
        if isinstance(obj, list):
            identity = lambda x: x
            processed = [cls.deserialize(item, identity, base_types, allowed_meta) for item in obj]
            return handler(processed)
        if isinstance(obj, (Reference, UnresolvedReference)):
            return obj
        model = obj
        if isinstance(obj, base_types) and not isinstance(obj, cls):
            model = cls(obj)  # type: ignore
        elif isinstance(obj, dict):
            if ref := cls._create_unresolved_ref(obj):
                return ref
            data = obj['@data']
            metadata = {k: v for k, v in obj.items() if k != '@data'}
            model = cls(data, **metadata)  # type: ignore
            model._register_keys(metadata)
        if cls.meta_checks_enabled():
            model._init_meta(allowed_meta)  # pylint: disable=protected-access
        return handler(model)

    @classmethod
    @lru_cache
    def serializer(cls):
        '''should return the validator for the specific class'''
        ser_fn = partial(cls.serialise, base_type=cls._OUTPUT_TYPE)
        return PlainSerializer(ser_fn, return_type=dict | list | None)

    @classmethod
    @lru_cache
    def validator(cls, allowed_meta: tuple[str]):
        '''default validator for the specific class'''
        allowed = set(allowed_meta)
        return WrapValidator(partial(cls.deserialize,
                                     base_types=cls._INPUT_TYPES,
                                     allowed_meta=allowed),
                             json_schema_input_type=cls._JSON_OUTPUT)

    def __copy__(self) -> Self:
        '''Copy the value and share its attribute values without parent links.'''
        clone = type(self)(self)  # type: ignore
        clone.__dict__.update(self._copy_rune_dict())
        clone.__dict__.pop(RUNE_OBJ_MAPS, None)
        clone._register_keys(clone._get_meta_container())
        return clone

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        '''Deep-copy attributes while preserving cycles and in-subtree parents.'''
        clone = type(self)(self)  # type: ignore
        memo[id(self)] = clone
        clone.__dict__.update(self._deepcopy_rune_dict(memo))
        if not clone.get_rune_parent():
            clone.__dict__.pop(RUNE_OBJ_MAPS, None)
            clone._register_keys(clone._get_meta_container())
        return clone


class _DateTimeCopyMixin:
    '''Copy temporal values without using their bytes-based pickle protocol.'''

    def __copy__(self) -> Self:
        '''Copy the value and share its attribute values.'''
        clone = type(self)(self)  # type: ignore
        clone.__dict__.update(self._copy_rune_dict())  # type: ignore
        clone.__dict__.pop(RUNE_OBJ_MAPS, None)
        clone._register_keys(clone._get_meta_container())  # type: ignore
        return clone

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        '''Deep-copy attributes while preserving cycles and shared values.'''
        clone = type(self)(self)  # type: ignore
        memo[id(self)] = clone
        clone.__dict__.update(self._deepcopy_rune_dict(memo))  # type: ignore
        if not clone.get_rune_parent():  # type: ignore
            clone.__dict__.pop(RUNE_OBJ_MAPS, None)
            clone._register_keys(clone._get_meta_container())  # type: ignore
        return clone


class DateWithMeta(datetime.date, _DateTimeCopyMixin, BasicTypeMetaDataMixin):
    '''date with metadata'''
    _INPUT_TYPES = (datetime.date, str)

    def __new__(cls, value, **kwds):  # pylint: disable=signature-differs
        cls._check_type(value)
        if isinstance(value, str):
            value = datetime.date.fromisoformat(value)
        ymd = value.timetuple()[:3]
        obj = datetime.date.__new__(cls, *ymd)
        obj.set_meta(check_allowed=False, **kwds)
        return obj


class TimeWithMeta(datetime.time, _DateTimeCopyMixin, BasicTypeMetaDataMixin):
    '''annotated time'''
    _INPUT_TYPES = (datetime.time, str)

    def __new__(cls, value, **kwds):  # pylint: disable=signature-differs
        cls._check_type(value)
        if isinstance(value, str):
            value = datetime.time.fromisoformat(value)
        obj = datetime.time.__new__(cls,
                                    value.hour,
                                    value.minute,
                                    value.second,
                                    value.microsecond,
                                    value.tzinfo,
                                    fold=value.fold)
        obj.set_meta(check_allowed=False, **kwds)
        return obj


class DateTimeWithMeta(datetime.datetime, _DateTimeCopyMixin, BasicTypeMetaDataMixin):
    '''annotated datetime'''
    _INPUT_TYPES = (datetime.datetime, str)

    def __new__(cls, value, **kwds):  # pylint: disable=signature-differs
        cls._check_type(value)
        if isinstance(value, str):
            value = datetime.datetime.fromisoformat(value)
        obj = datetime.datetime.__new__(cls,
                                        value.year,
                                        value.month,
                                        value.day,
                                        value.hour,
                                        value.minute,
                                        value.second,
                                        value.microsecond,
                                        value.tzinfo,
                                        fold=value.fold)
        obj.set_meta(check_allowed=False, **kwds)
        return obj

    def __str__(self):
        return self.isoformat()


class StrWithMeta(str, BasicTypeMetaDataMixin):
    '''string with metadata'''
    def __new__(cls, value, **kwds):
        obj = str.__new__(cls, value)
        obj.set_meta(check_allowed=False, **kwds)
        return obj


class IntWithMeta(int, BasicTypeMetaDataMixin):
    '''annotated integer'''
    _INPUT_TYPES = int
    _OUTPUT_TYPE = int
    _JSON_OUTPUT = int | dict

    def __new__(cls, value, **kwds):
        obj = int.__new__(cls, value)
        obj.set_meta(check_allowed=False, **kwds)
        return obj


class NumberWithMeta(_DateTimeCopyMixin, Decimal, BasicTypeMetaDataMixin):
    '''annotated number'''
    _INPUT_TYPES = (Decimal, float, int, str)
    _OUTPUT_TYPE = Decimal
    _JSON_OUTPUT = float | int | str | dict

    def __new__(cls, value, **kwds):
        # NOTE: it could be necessary to convert the value to str if it is a
        # float
        obj = Decimal.__new__(cls, value)
        obj.set_meta(check_allowed=False, **kwds)
        return obj


class _EnumWrapperDefaultVal(Enum):
    '''marker for not set value in enum wrapper'''
    NOT_SET = "NOT_SET"


class _EnumWrapper(BaseMetaDataMixin):
    '''wrapper for enums with metadata'''
    def __init__(self, enum_instance=_EnumWrapperDefaultVal.NOT_SET):
        if not isinstance(enum_instance, Enum):
            raise TypeError("enum_instance must be an instance of an Enum")
        self._enum_instance = enum_instance

    @property
    def enum_instance(self):
        '''the actual enum instance'''
        return self._enum_instance

    @property
    def name(self):
        '''enum name - pass through'''
        return self._enum_instance.name

    @property
    def value(self):
        '''enum value  - pass through'''
        return self._enum_instance.value

    def __str__(self):
        return str(self._enum_instance)

    def __repr__(self):
        return repr(self._enum_instance)

    def __eq__(self, other):
        if isinstance(other, _EnumWrapper):
            return self._enum_instance == other._enum_instance
        return self._enum_instance == other

    def __hash__(self):
        return hash(self._enum_instance)

    def __copy__(self) -> Self:
        '''Copy the enum wrapper without retaining its parent link.'''
        clone = type(self)(self._enum_instance)
        clone.__dict__.update(self._copy_rune_dict())
        clone.__dict__.pop(RUNE_OBJ_MAPS, None)
        clone._register_keys(clone._get_meta_container())
        return clone

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        '''Deep-copy the enum wrapper without walking up the parent chain.'''
        clone = type(self)(self._enum_instance)
        memo[id(self)] = clone
        clone.__dict__.update(self._deepcopy_rune_dict(memo))
        if not clone.get_rune_parent():
            clone.__dict__.pop(RUNE_OBJ_MAPS, None)
            clone._register_keys(clone._get_meta_container())
        return clone


class EnumWithMetaMixin:
    '''holds the metadata associated with a Rune Enum'''
    @classmethod
    def serialise(cls, obj, handler, info) -> dict | list | None:
        '''used as serialisation method with pydantic'''
        obj = _unwrap_cow_proxy(obj)
        if obj is None:
            return None
        if isinstance(obj, list):
            return [cls.serialise(item, handler, info) for item in obj]
        if isinstance(obj, UnresolvedReference):
            return {obj.key_type.rune_ref_tag: obj.key}
        if isinstance(obj, Reference):
            return {obj.key_type.rune_ref_tag: obj.target_key}
        if isinstance(obj, _EnumWrapper):
            res = obj.serialise_meta()
            res['@data'] = obj.value
            return res
        if isinstance(obj, Enum):
            return {'@data': obj.value}
        return handler(obj)

    @classmethod
    def deserialize(cls, obj, allowed_meta: set[str]):
        '''method used as pydantic `validator`'''
        obj = _unwrap_cow_proxy(obj)
        if isinstance(obj, list):
            return [cls.deserialize(item, allowed_meta) for item in obj]
        if isinstance(obj, (Reference, UnresolvedReference)):
            return obj
        model = obj
        if (isinstance(obj, str)
                and not isinstance(obj, _EnumWrapper)):
            model = _EnumWrapper(cls(obj))  # type: ignore
        if (isinstance(obj, EnumWithMetaMixin)
                and not isinstance(obj, _EnumWrapper)):
            model = _EnumWrapper(obj)  # type: ignore
        elif isinstance(obj, dict):
            # pylint: disable=protected-access
            if ref := _EnumWrapper._create_unresolved_ref(obj):
                return ref
            data = obj['@data']
            metadata = {k: v for k, v in obj.items() if k != '@data'}
            model = _EnumWrapper(cls(data))  # type: ignore
            model.set_meta(check_allowed=False, **metadata)
            model._register_keys(metadata)  # pylint: disable=protected-access
        if _EnumWrapper.meta_checks_enabled():
            model._init_meta(allowed_meta)  # pylint: disable=protected-access
        return model

    @classmethod
    @lru_cache
    def serializer(cls):
        '''should return the validator for the specific class'''
        return WrapSerializer(cls.serialise, return_type=dict | list | None)

    @classmethod
    @lru_cache
    def validator(cls, allowed_meta: tuple[str] | tuple[Never, ...] = ()):
        '''default validator for the specific class'''
        allowed = set(allowed_meta)
        return PlainValidator(partial(cls.deserialize, allowed_meta=allowed),
                              json_schema_input_type=str | dict)

# EOF
