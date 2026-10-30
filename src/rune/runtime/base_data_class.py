'''Base class for all Rune type classes'''
from collections.abc import MutableSequence
import logging
import importlib
import copy
import json
from typing import get_args, get_origin, Any, Literal
from typing_extensions import Self
from pydantic import (BaseModel, ValidationError, ConfigDict, model_serializer,
                      model_validator, ModelWrapValidatorHandler)
from pydantic.main import IncEx
from rune.runtime.conditions import ConditionViolationError
from rune.runtime.conditions import get_conditions
from rune.runtime.metadata import (META_CONTAINER, PARENT_PROP, REFS_CONTAINER,
                                   ComplexTypeMetaDataMixin, KeyType, Reference,
                                   UnresolvedReference, BaseMetaDataMixin,
                                   _EnumWrapper, _ReferenceList,
                                   _unwrap_cow_proxy, RUNE_OBJ_MAPS)

ROOT_CONTAINER = '__rune_root_metadata'
_DEEPCOPY_ROOT_KEY = '_rune_deepcopy_root'


def _collect_inline_subtree_ids(
    node: Any,
    visited: set[int] | None = None,
) -> set[int]:
    '''Return object ids of BaseMetaDataMixin nodes in node's inline subtree.'''
    if visited is None:
        visited = set()
    if isinstance(node, (MutableSequence, tuple)):
        ref_entries = (
            node._ref_entries  # pylint: disable=protected-access
            if isinstance(node, _ReferenceList)
            else None
        )
        for idx, item in enumerate(node):
            if ref_entries is not None and idx < len(ref_entries) and ref_entries[idx] is not None:
                continue
            if isinstance(item, BaseMetaDataMixin):
                _collect_inline_subtree_ids(item, visited)
        return visited

    if not isinstance(node, BaseMetaDataMixin) or id(node) in visited:
        return visited
    visited.add(id(node))
    if isinstance(node, BaseDataClass):
        refs = node._get_rune_refs_container()  # pylint: disable=protected-access
        for prop_nm, child in node.__dict__.items():
            if prop_nm.startswith('__'):
                continue
            prop_refs = refs.get(prop_nm)
            if prop_refs is not None and not isinstance(prop_refs, dict):
                continue
            if isinstance(child, (MutableSequence, tuple)):
                ref_indices = prop_refs if isinstance(prop_refs, dict) else ()
                for idx, item in enumerate(child):
                    if idx not in ref_indices and isinstance(item, BaseMetaDataMixin):
                        _collect_inline_subtree_ids(item, visited)
            elif isinstance(child, BaseMetaDataMixin):
                _collect_inline_subtree_ids(child, visited)
    return visited


class BaseDataClass(BaseModel, ComplexTypeMetaDataMixin):
    ''' A base class for all cdm generated classes. It is derived from
        `pydantic.BaseModel` which provides type checking at object creation
        for all cdm classes. It provides as well the `validate_model`,
        `validate_conditions` and `validate_attribs` methods which perform the
        conditions, cardinality and type checks as specified in the rune
        type model. The method `validate_model` is not invoked automatically,
        but is left to the user to determine when to check the validity of the
        cdm model.
    '''
    model_config = ConfigDict(extra='ignore',
                              revalidate_instances='always',
                              arbitrary_types_allowed=True)

    def _unregister_replaced_subtree_keys(
        self,
        name: str,
        new_value: Any,
    ) -> None:
        '''Remove object-map entries owned by an inline property being replaced.'''
        refs = self._get_rune_refs_container()
        prop_refs = refs.get(name)
        if prop_refs is not None and not isinstance(prop_refs, dict):
            return
        old_val = self.__dict__.get(name)
        if old_val is None or old_val is new_value:
            return
        old_ids = _collect_inline_subtree_ids(old_val)
        if not old_ids:
            return
        new_ids = _collect_inline_subtree_ids(new_value)
        removed_ids = old_ids - new_ids
        if not removed_ids:
            return
        curr: BaseMetaDataMixin | None = self
        while curr is not None:
            if obj_maps := curr.__dict__.get(RUNE_OBJ_MAPS):
                for key_map in obj_maps.values():
                    stale = [k for k, v in key_map.items() if id(v) in removed_ids]
                    for k in stale:
                        del key_map[k]
            curr = curr.get_rune_parent()

    def _assign_sequence_property(self, name: str, value: Any) -> None:
        '''Atomically validate and assign a sequence property with references.'''
        raw_items = list(value)
        refs_container = self._get_rune_refs_container()
        existing_refs = refs_container.get(name)
        existing_seq = self.__dict__.get(name)

        if isinstance(value, _ReferenceList):
            incoming_entries: list[tuple[str, KeyType] | None] = list(
                value._ref_entries  # pylint: disable=protected-access
            )
            if len(incoming_entries) < len(raw_items):
                incoming_entries.extend(
                    [None] * (len(raw_items) - len(incoming_entries))
                )
        else:
            existing_queues: dict[int, list[tuple[str, KeyType] | None]] = {}
            if isinstance(existing_refs, dict) and isinstance(
                existing_seq, (MutableSequence, tuple)
            ):
                for old_idx, old_item in enumerate(existing_seq):
                    existing_queues.setdefault(id(old_item), []).append(
                        existing_refs.get(old_idx)
                    )
            incoming_entries = []
            for item in raw_items:
                if isinstance(item, Reference):
                    incoming_entries.append(None)
                else:
                    unwrapped_item = _unwrap_cow_proxy(item)
                    queue = existing_queues.get(id(unwrapped_item))
                    if queue:
                        incoming_entries.append(
                            queue.pop(0) if len(queue) > 1 else queue[0]
                        )
                    else:
                        incoming_entries.append(None)

        old_items = (
            list(existing_seq)
            if isinstance(existing_seq, (MutableSequence, tuple))
            else []
        )
        old_ref_indices = (
            set(existing_refs.keys()) if isinstance(existing_refs, dict) else set()
        )

        # Step 1: Validate all Reference items before mutating anything on self
        staged_items: list[Any] = []
        staged_entries: list[tuple[str, KeyType] | None] = []
        for idx, raw_item in enumerate(raw_items):
            item = _unwrap_cow_proxy(raw_item)
            if isinstance(item, Reference):
                if idx < len(old_items):
                    self._validate_ref_for_property(
                        name,
                        item,
                        current_item=old_items[idx],
                        is_already_ref=idx in old_ref_indices,
                        index=idx,
                    )
                else:
                    self._validate_ref_for_property(name, item, index=idx)
                staged_items.append(item.target)
                staged_entries.append((item.target_key, item.key_type))
                continue

            entry = incoming_entries[idx] if idx < len(incoming_entries) else None
            if entry is not None:
                key, key_type = entry
                if (
                    isinstance(item, BaseMetaDataMixin)
                    and item.get_meta(key_type.key_tag) == key
                ):
                    staged_items.append(item)
                    staged_entries.append(entry)
                    continue

            staged_items.append(item)
            staged_entries.append(None)

        # Step 2: Unregister keys owned by replaced inline items
        if isinstance(value, _ReferenceList) and (
            value._owner() is None or value._owner() is self  # pylint: disable=protected-access
        ):
            ref_list = value
            super(_ReferenceList, ref_list).__setitem__(slice(None), staged_items)
            ref_list._ref_entries = staged_entries  # pylint: disable=protected-access
        else:
            ref_list = _ReferenceList(staged_items, ref_entries=staged_entries)

        self._unregister_replaced_subtree_keys(name, ref_list)

        # Step 3: Attach inline items first so their keys are registered on self
        for item, entry in zip(ref_list, staged_entries):
            if entry is None and isinstance(item, BaseMetaDataMixin):
                item._set_rune_parent(self)  # pylint: disable=protected-access

        # Step 4: Re-alias preserved reference slots to canonical targets on self
        for idx, (item, entry) in enumerate(zip(list(ref_list), staged_entries)):
            if entry is not None and not isinstance(raw_items[idx], Reference):
                key, key_type = entry
                try:
                    registered = self.get_object_by_key(key, key_type)
                except KeyError:
                    registered = None
                if registered is not None and registered is not item:
                    super(_ReferenceList, ref_list).__setitem__(idx, registered)

        # Step 5: Commit property and synchronize reference container
        super().__setattr__(name, ref_list)
        ref_list._bind_owner(self, name)  # pylint: disable=protected-access

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Reference):
            self._bind_property_to(name, value)
        elif isinstance(value, (MutableSequence, tuple)) and (
            not name.startswith('_')
        ):
            self._assign_sequence_property(name, value)
        else:
            self._unregister_replaced_subtree_keys(name, value)
            # replace reference with an object
            if name in self._get_rune_refs_container():
                self._remove_rune_ref(name)
                if isinstance(self.__dict__.get(name), _EnumWrapper):
                    self.__dict__[name] = _EnumWrapper()
            # if the value is an enum, pass it to the EnumWrapper
            if (isinstance(self.__dict__.get(name), _EnumWrapper)
                    and not isinstance(value, _EnumWrapper)):
                value = _EnumWrapper(value)
            # if the value is a "model", register as rune_parent
            if isinstance(value, BaseMetaDataMixin):
                value._set_rune_parent(self)
            super().__setattr__(name, value)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, BaseModel):
            # Align with Pydantic's equality semantics but avoid parent recursion
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__

            if not (
                self_type == other_type
                and getattr(self, '__pydantic_private__', None)
                == getattr(other, '__pydantic_private__', None)
                and self.__pydantic_extra__ == other.__pydantic_extra__
            ):
                return False

            model_fields = type(self).__pydantic_fields__.keys()
            if not model_fields:
                return True
            _missing = object()
            self_vals = tuple(self.__dict__.get(k, _missing) for k in model_fields)
            other_vals = tuple(other.__dict__.get(k, _missing) for k in model_fields)
            return self_vals == other_vals

        return NotImplemented

    def __copy__(self) -> Self:
        '''Return a detached shallow copy without retaining the parent link.'''
        clone = super().__copy__()
        clone.__dict__.pop(PARENT_PROP, None)
        if META_CONTAINER in self.__dict__:
            clone.__dict__[META_CONTAINER] = copy.copy(self.__dict__[META_CONTAINER])
        if REFS_CONTAINER in self.__dict__:
            clone.__dict__[REFS_CONTAINER] = copy.deepcopy(self.__dict__[REFS_CONTAINER])
        for prop_nm, val in list(clone.__dict__.items()):
            if prop_nm.startswith('__'):
                continue
            if isinstance(val, _ReferenceList):
                ref_copy = val.copy()
                ref_copy._bind_owner(clone, prop_nm)  # pylint: disable=protected-access
                clone.__dict__[prop_nm] = ref_copy
        clone._rebuild_shallow_object_maps()
        return clone

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        '''Return a deep copy of the subtree without following parent links.'''
        if memo is None:
            memo = {}
        is_deepcopy_root = _DEEPCOPY_ROOT_KEY not in memo
        if is_deepcopy_root:
            memo[_DEEPCOPY_ROOT_KEY] = id(self)
        try:
            clone = super().__copy__()
            clone.__dict__.pop(PARENT_PROP, None)
            memo[id(self)] = clone
            object.__setattr__(clone, '__dict__', self._deepcopy_rune_dict(memo))
            if clone.__pydantic_extra__ is not None:
                object.__setattr__(
                    clone,
                    '__pydantic_extra__',
                    copy.deepcopy(clone.__pydantic_extra__, memo),
                )
            if clone.__pydantic_private__ is not None:
                object.__setattr__(
                    clone,
                    '__pydantic_private__',
                    copy.deepcopy(clone.__pydantic_private__, memo),
                )
            self._restore_child_parents(clone, memo)
            if is_deepcopy_root:
                clone._rebuild_subtree_object_maps()
            return clone
        finally:
            if is_deepcopy_root:
                memo.pop(_DEEPCOPY_ROOT_KEY, None)

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        '''Return a detached model copy with a rebuilt subtree key registry.'''
        clone = self.__deepcopy__() if deep else self.__copy__()
        if update:
            for name, val in update.items():
                setattr(clone, name, val)
            if deep:
                clone._rebuild_subtree_object_maps()
            else:
                clone._rebuild_shallow_object_maps()
        return clone

    def _restore_child_parents(self, clone: Self, memo: dict[int, Any]) -> None:
        '''Ensure direct children whose parent was self point to clone.'''
        refs = clone._get_rune_refs_container()
        for prop_nm, obj in self.__dict__.items():
            if prop_nm.startswith('__'):
                continue
            clone_obj = clone.__dict__.get(prop_nm)
            if isinstance(clone_obj, _ReferenceList):
                clone_obj._bind_owner(clone, prop_nm)  # pylint: disable=protected-access
            prop_refs = refs.get(prop_nm)
            if prop_refs is not None and not isinstance(prop_refs, dict):
                continue
            children = obj if isinstance(obj, (MutableSequence, tuple)) else (obj,)
            for child in children:
                if (isinstance(child, BaseMetaDataMixin)
                        and child.get_rune_parent() is self):
                    child_clone = memo.get(id(child))
                    if isinstance(child_clone, BaseMetaDataMixin):
                        child_clone.__dict__[PARENT_PROP] = clone

    def _rebuild_subtree_object_maps(self) -> None:
        '''Rebuild key registries from a detached deep-copied subtree.'''
        self.__dict__.pop(PARENT_PROP, None)
        self.__dict__[RUNE_OBJ_MAPS] = {}
        visited: set[int] = set()
        containment_models: list[BaseDataClass] = []

        def _walk(node: BaseDataClass) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            containment_models.append(node)
            if node is not self:
                node.__dict__.pop(RUNE_OBJ_MAPS, None)
            node._register_keys(node._get_meta_container())

            refs = node._get_rune_refs_container()
            for prop_nm, obj in list(node.__dict__.items()):
                if prop_nm.startswith('__'):
                    continue
                prop_refs = refs.get(prop_nm)
                if isinstance(obj, list) and not isinstance(obj, _ReferenceList):
                    obj = _ReferenceList(
                        obj,
                        ref_entries=prop_refs if isinstance(prop_refs, dict) else None,
                        owner=node,
                        prop_name=prop_nm,
                    )
                    node.__dict__[prop_nm] = obj
                elif isinstance(obj, _ReferenceList):
                    obj._bind_owner(node, prop_nm)  # pylint: disable=protected-access

                if prop_refs is not None and not isinstance(prop_refs, dict):
                    continue
                if isinstance(obj, (MutableSequence, tuple)):
                    ref_indices = prop_refs if isinstance(prop_refs, dict) else ()
                    for idx, child in enumerate(obj):
                        if idx not in ref_indices and isinstance(child, BaseMetaDataMixin):
                            child.__dict__[PARENT_PROP] = node
                            if isinstance(child, BaseDataClass):
                                _walk(child)
                            else:
                                visited.add(id(child))
                                child.__dict__.pop(RUNE_OBJ_MAPS, None)
                                child._register_keys(child._get_meta_container())
                elif isinstance(obj, BaseMetaDataMixin):
                    obj.__dict__[PARENT_PROP] = node
                    if isinstance(obj, BaseDataClass):
                        _walk(obj)
                    else:
                        visited.add(id(obj))
                        obj.__dict__.pop(RUNE_OBJ_MAPS, None)
                        obj._register_keys(obj._get_meta_container())

        _walk(self)

        # Ensure any resolved references in the detached subtree alias to the
        # canonical cloned target registered in the subtree key map.
        for node in containment_models:
            refs = node._get_rune_refs_container()
            for prop_nm, ref_entry in list(refs.items()):
                if isinstance(ref_entry, dict):
                    node._reconcile_list_refs(prop_nm)
                else:
                    key, key_type = ref_entry
                    try:
                        registered = node.get_object_by_key(key, key_type)
                    except KeyError:
                        registered = None
                    if registered is not None:
                        node.__dict__[prop_nm] = registered

    def _rebuild_shallow_object_maps(self) -> None:
        '''Populate RUNE_OBJ_MAPS on a detached shallow copy without mutating shared children.'''
        self.__dict__.pop(PARENT_PROP, None)
        obj_maps: dict[KeyType, dict[str, Any]] = {}
        visited: set[int] = set()

        def _collect_node_keys(node: BaseMetaDataMixin, inside_nested_scope: bool) -> None:
            for key_t, key_v in node._get_meta_container().items():
                if not (key_t.startswith('@key') and key_v):
                    continue
                kt = KeyType.from_rune(key_t)
                if kt == KeyType.SCOPED and inside_nested_scope:
                    continue
                obj_maps.setdefault(kt, {})[key_v] = node

        def _walk_shallow(node: BaseDataClass, inside_nested_scope: bool) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            _collect_node_keys(node, inside_nested_scope)
            next_scope = inside_nested_scope or (
                node is not self and node.is_scope_instance()
            )
            refs = node._get_rune_refs_container()
            for prop_nm, obj in node.__dict__.items():
                if prop_nm.startswith('__'):
                    continue
                prop_refs = refs.get(prop_nm)
                if prop_refs is not None and not isinstance(prop_refs, dict):
                    continue
                if isinstance(obj, (MutableSequence, tuple)):
                    ref_indices = prop_refs if isinstance(prop_refs, dict) else ()
                    for idx, child in enumerate(obj):
                        if idx not in ref_indices and isinstance(child, BaseMetaDataMixin):
                            if isinstance(child, BaseDataClass):
                                _walk_shallow(child, next_scope)
                            else:
                                _collect_node_keys(child, next_scope)
                elif isinstance(obj, BaseMetaDataMixin):
                    if isinstance(obj, BaseDataClass):
                        _walk_shallow(obj, next_scope)
                    else:
                        _collect_node_keys(obj, next_scope)

        _walk_shallow(self, False)
        self.__dict__[RUNE_OBJ_MAPS] = obj_maps

    def _reconcile_list_refs(self, prop_nm: str) -> None:
        '''Ensure a list property is a _ReferenceList with reconciled bindings.'''
        seq = self.__dict__.get(prop_nm)
        if not isinstance(seq, list):
            return
        refs = self.__dict__.setdefault(REFS_CONTAINER, {})
        prop_refs = refs.get(prop_nm)
        if not isinstance(seq, _ReferenceList):
            seq = _ReferenceList(
                seq,
                ref_entries=prop_refs if isinstance(prop_refs, dict) else None,
                owner=self,
                prop_name=prop_nm,
            )
            self.__dict__[prop_nm] = seq
        elif seq._owner() is not self or seq._prop_name != prop_nm:  # pylint: disable=protected-access
            if seq._owner() is not None and seq._owner() is not self:  # pylint: disable=protected-access
                seq = _ReferenceList(
                    seq,
                    ref_entries=seq._ref_entries,  # pylint: disable=protected-access
                    owner=self,
                    prop_name=prop_nm,
                )
                self.__dict__[prop_nm] = seq
            else:
                seq._bind_owner(self, prop_nm)  # pylint: disable=protected-access

        if len(seq._ref_entries) < len(seq):  # pylint: disable=protected-access
            seq._ref_entries.extend(  # pylint: disable=protected-access
                [None] * (len(seq) - len(seq._ref_entries))  # pylint: disable=protected-access
            )
        elif len(seq._ref_entries) > len(seq):  # pylint: disable=protected-access
            del seq._ref_entries[len(seq):]  # pylint: disable=protected-access

        for idx, (item, entry) in enumerate(
            zip(list(seq), list(seq._ref_entries))  # pylint: disable=protected-access
        ):
            if entry is None:
                continue
            key, key_type = entry
            if isinstance(item, (Reference, UnresolvedReference)):
                continue
            if isinstance(item, BaseMetaDataMixin):
                item_key = item.get_meta(key_type.key_tag)
                if item_key != key:
                    seq._ref_entries[idx] = None  # pylint: disable=protected-access
                    item._set_rune_parent(self)  # pylint: disable=protected-access
                else:
                    try:
                        registered = self.get_object_by_key(key, key_type)
                    except KeyError:
                        registered = None
                    if registered is not None and registered is not item:
                        super(_ReferenceList, seq).__setitem__(idx, registered)
            else:
                seq._ref_entries[idx] = None  # pylint: disable=protected-access

        seq._sync_owner_refs()  # pylint: disable=protected-access

    def _reconcile_all_list_refs(self) -> None:
        '''Reconcile all list properties and their reference bindings on self.'''
        for prop_nm, obj in list(self.__dict__.items()):
            if prop_nm.startswith('__'):
                continue
            if isinstance(obj, list):
                self._reconcile_list_refs(prop_nm)

    @classmethod
    def _unwrap_cow_validation_input(cls, data: Any) -> Any:
        '''Prepare model inputs by removing COW proxies.

        This intentionally leaves already-materialized BaseDataClass instances
        untouched so model graphs and parent/reference wiring are not walked
        during validation. Nested model fields are handled by their own
        validators when pydantic descends into them.
        '''
        if isinstance(data, BaseDataClass):
            return data

        unwrap = getattr(data, '_unwrap', None)
        if callable(unwrap):
            data = unwrap()
            if isinstance(data, BaseDataClass):
                return {
                    name: getattr(data, name)
                    for name in type(data).__pydantic_fields__.keys()
                }
            return data

        return data

    @model_validator(mode='before')
    @classmethod
    def _unwrap_cow_inputs(cls, data: Any) -> Any:
        '''Unwrap copy-on-write proxies before pydantic validation.'''
        return cls._unwrap_cow_validation_input(data)

    @model_serializer(mode='wrap')
    def _serialize_refs(self, serializer, info):
        '''should replace objects with refs while serializing'''
        self._reconcile_all_list_refs()
        res = serializer(self)
        refs = self._get_rune_refs_container()
        for property_nm, ref_entry in refs.items():
            if isinstance(ref_entry, dict):
                seq = res.get(property_nm)
                if isinstance(seq, MutableSequence):
                    seq = list(seq)
                    for idx, (key, ref_type) in ref_entry.items():
                        if 0 <= idx < len(seq):
                            seq[idx] = {ref_type.rune_ref_tag: key}
                    res[property_nm] = seq
            else:
                key, ref_type = ref_entry
                res[property_nm] = {ref_type.rune_ref_tag: key}
        res = self.__dict__.get(ROOT_CONTAINER, {}) | res
        return res

    @model_validator(mode='wrap')
    @classmethod
    def _deserialize_refs(cls, data: Any,
                          handler: ModelWrapValidatorHandler[Self]) -> Self:
        '''should resolve refs after creation'''
        if isinstance(data, dict):
            metadata = {k: v for k, v in data.items() if k.startswith('@')}
            if aux := cls._create_unresolved_ref(metadata):
                return aux
        elif isinstance(data, BaseDataClass):
            data._reconcile_all_list_refs()  # pylint: disable=protected-access
        obj = handler(data)
        # transfer refs that were established on the original before
        # re-validation wiped them, and before _init_rune_parent runs
        if isinstance(data, BaseDataClass) and not obj._get_rune_refs_container():
            if original_refs := data._get_rune_refs_container():
                obj.__dict__[REFS_CONTAINER] = copy.deepcopy(original_refs)
        obj._init_rune_parent()  # pylint: disable=protected-access
        if isinstance(data, dict):
            # Deferred annotations can bypass the field metadata validator.
            # Retain target keys before resolving references to this model.
            keys = {k: v for k, v in metadata.items() if k.startswith('@key')}
            if keys:
                # Field-level validators handle allowed metadata, including
                # scoped keys that are not declared on the target class.
                obj.set_meta(check_allowed=False, **keys)
                obj._register_keys(keys)  # pylint: disable=protected-access
        obj.resolve_references(ignore_dangling=True, recurse=False)
        return obj

    def _init_rune_parent(self, recurse: bool = False):
        '''sets the rune parent in all properties'''
        if not self.get_rune_parent() and RUNE_OBJ_MAPS not in self.__dict__:
            self.__dict__[RUNE_OBJ_MAPS] = {}

        self._reconcile_all_list_refs()
        refs = self._get_rune_refs_container()

        for prop_nm, obj in list(self.__dict__.items()):
            if prop_nm.startswith('__'):
                continue
            prop_refs = refs.get(prop_nm)
            if prop_refs is not None and not isinstance(prop_refs, dict):
                continue
            if isinstance(obj, (MutableSequence, tuple)):
                ref_indices = prop_refs if isinstance(prop_refs, dict) else ()
                for idx, child in enumerate(obj):
                    if idx not in ref_indices and isinstance(child, BaseMetaDataMixin):
                        child._set_rune_parent(self)  # pylint: disable=protected-access
                        if recurse and isinstance(child, BaseDataClass):
                            child._init_rune_parent(recurse=True)  # pylint: disable=protected-access
            elif isinstance(obj, BaseMetaDataMixin):
                obj._set_rune_parent(self)  # pylint: disable=protected-access
                if recurse and isinstance(obj, BaseDataClass):
                    obj._init_rune_parent(recurse=True)  # pylint: disable=protected-access

    def rune_serialize(
        self,
        *,
        validate_model: bool = True,
        check_rune_constraints: bool = True,
        strict: bool = True,
        raise_validation_errors: bool = True,
        indent: int | None = None,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        exclude_unset: bool = True,
        exclude_defaults: bool = True,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal['none', 'warn', 'error'] = True,
        serialize_as_any: bool = False,
    ) -> str:
        '''Rune conform serialization to json string. To be invoked on the model
        root.

        #### Args:
            `validate_model (bool, optional):` Validate the model prior
            serialization. It checks also all Rune type constraints.
            Defaults to True.

            `check_rune_constraints (bool, optional):` If `validate_model` is
            set to `True`, executes all model defined Rune constraints after
            deserialization. Defaults to True.

            `strict (bool, optional):` Perform strict attribute validation. 
            Defaults to True.

            `raise_validation_errors (bool, optional):` Raise an exception in
            case a validation error has occurred. Defaults to True.

            `indent (int | None, optional):` Indentation to use in the JSON
            output. If None is passed, the output will be compact. Defaults to
            None.

            `include (IncEx | None, optional):` Field(s) to include in the JSON
            output. Defaults to None.

            `exclude (IncEx | None, optional):` Field(s) to exclude from the
            JSON output. Defaults to None.

            `exclude_unset (bool, optional):` Whether to exclude fields that
            have not been explicitly set. Defaults to True.

            `exclude_defaults (bool, optional):` Whether to exclude fields that
            are set to their default value. Defaults to True.

            `exclude_none (bool, optional):` Whether to exclude fields that have
            a value of `None`. Defaults to False.

            `round_trip (bool, optional):` If True, dumped values should be
            valid as input for non-idempotent types such as Json[T]. Defaults to
            False.

            `warnings (bool | Literal['none', 'warn', 'error'], optional):` How
            to handle serialization errors. False/"none" ignores them,
            True/"warn" logs errors, "error" raises a
            `PydanticSerializationError`. Defaults to True.

            `serialize_as_any (bool, optional):` Whether to serialize fields
            with duck-typing serialization behavior. Defaults to False.

        #### Returns:
            `str:` A Rune conforming JSON string representation of the model.
        '''
        try:
            if validate_model:
                self.validate_model(
                    check_rune_constraints=check_rune_constraints,
                    strict=strict,
                    raise_exc=raise_validation_errors)

            root_meta = self.__dict__.setdefault(ROOT_CONTAINER, {})  # type: ignore
            # root_meta['@type'] = self._FQRTN
            # root_meta['@model'] = self._FQRTN.split('.', maxsplit=1)[0]
            root_meta['@type'] = self._fqrtn()
            root_meta['@model'] = self._fqrtn().split('.', maxsplit=1)[0]
            root_meta['@version'] = self.get_model_version()

            return self.model_dump_json(indent=indent,
                                        include=include,
                                        exclude=exclude,
                                        exclude_unset=exclude_unset,
                                        exclude_defaults=exclude_defaults,
                                        exclude_none=exclude_none,
                                        round_trip=round_trip,
                                        warnings=warnings,
                                        serialize_as_any=serialize_as_any)
        finally:
            self.__dict__.pop(ROOT_CONTAINER)

    @classmethod
    def rune_deserialize(cls,
                         rune_data: str | dict[str, Any],
                         validate_model: bool = True,
                         check_rune_constraints: bool = True,
                         strict: bool = False,
                         raise_validation_errors: bool = True,
                         namespace_prefix: str | None = None) -> BaseModel:
        # pylint: disable=line-too-long
        '''Rune compliant deserialization

        #### Args:
            `rune_json (str):` A JSON string.

            `validate_model (bool, optional):` Validate the model after
            deserialization. It checks also all Rune type constraints. Defaults
            to True.

            `check_rune_constraints (bool, optional):` If `validate_model` is
            set to `True`, executes all model defined Rune constraints after
            deserialization. Defaults to True.

            `strict (bool, optional):` Perform strict attribute validation.
            Defaults to False.

            `raise_validation_errors (bool, optional):` Raise an exception in
            case a validation error has occurred. Defaults to True.

            `namespace_prefix (str | None, optional):` Prefix to prepend to
            the Rune `@type` value when resolving Python import paths during
            deserialization when called via `BaseDataClass`. Concrete
            generated root classes use the prefix from their own package.
            Defaults to None.

        #### Returns:
            `BaseModel:` The Rune model.
        '''
        if isinstance(rune_data, str):
            rune_dict = json.loads(rune_data)
            # NOTE: json.loads will not create the right atomic types
            # (e.g. date, datetime, time etc) and the strict model validate
            # will not attempt to convert them.
            strict = False
        elif not isinstance(rune_data, dict):
            raise ValueError(f'rune_data is of type {type(rune_data)}, '
                             'alas it has to be either dict or str!')
        else:
            rune_dict = copy.deepcopy(rune_data)
        rune_dict.pop('@version', None)
        rune_dict.pop('@model', None)
        rune_prefix = cls._get_rune_namespace_prefix()
        if rune_prefix is None and cls is BaseDataClass:
            rune_prefix = namespace_prefix
        rune_cls = cls._type_to_cls(rune_dict,
                                    namespace_prefix=rune_prefix)
        model = rune_cls.model_validate(rune_dict, strict=strict)
        model.resolve_references(ignore_dangling=False, recurse=True)
        if validate_model:
            model.validate_model(
                check_rune_constraints=check_rune_constraints,
                strict=strict,
                raise_exc=raise_validation_errors)
        return model

    def resolve_references(self, ignore_dangling=False, recurse=True):
        '''resolves all attributes which are references'''
        self._reconcile_all_list_refs()
        if recurse:
            resolved_refs = self._get_rune_refs_container()
            for prop_nm, obj in self.__dict__.items():
                # Follow containment only; resolved references can point back
                # to an ancestor and must not be traversed again.
                if prop_nm.startswith('__'):
                    continue
                prop_refs = resolved_refs.get(prop_nm)
                if prop_refs is not None and not isinstance(prop_refs, dict):
                    continue
                if isinstance(obj, (MutableSequence, tuple)):
                    ref_indices = prop_refs if isinstance(prop_refs, dict) else ()
                    for idx, child in enumerate(obj):
                        if idx not in ref_indices and isinstance(child, BaseDataClass):
                            child.resolve_references(ignore_dangling=ignore_dangling,
                                                     recurse=recurse)
                elif isinstance(obj, BaseDataClass):
                    obj.resolve_references(ignore_dangling=ignore_dangling,
                                           recurse=recurse)

        refs: list[tuple[str, Reference, int | None]] = []
        for prop_nm, obj in self.__dict__.items():
            if prop_nm.startswith('__'):
                continue
            if isinstance(obj, (UnresolvedReference, Reference)):
                try:
                    refs.append((prop_nm, obj.get_reference(self), None))
                except KeyError:
                    if not ignore_dangling:
                        raise
            elif isinstance(obj, (MutableSequence, tuple)):
                for idx, item in enumerate(obj):
                    if isinstance(item, (UnresolvedReference, Reference)):
                        try:
                            refs.append((prop_nm, item.get_reference(self), idx))
                        except KeyError:
                            if not ignore_dangling:
                                raise

        for prop_nm, ref, idx in refs:
            self._bind_property_to(prop_nm, ref, index=idx)

    def validate_model(self,
                       check_rune_constraints=True,
                       recursively: bool = True,
                       raise_exc: bool = True,
                       strict: bool = True) -> list:
        ''' This method performs full model validation. It will validate all
            attributes and it will also invoke `validate_conditions` to check
            all conditions and the cardinality of all attributes of this object.
            The parameter `raise_exc` controls whether an exception should be
            thrown if a validation or condition is violated or if a list with
            all encountered violations should be returned instead.
        '''
        try:
            self.disable_meta_checks()
            att_errors = self.validate_attribs(raise_exc=raise_exc,
                                               strict=strict)
            if check_rune_constraints:
                att_errors.extend(
                    self.validate_conditions(recursively=recursively,
                                             raise_exc=raise_exc))
            return att_errors
        finally:
            self.enable_meta_checks()

    def validate_attribs(self,
                         raise_exc: bool = True,
                         strict: bool = True) -> list:
        ''' This method performs attribute type validation.
            The parameter `raise_exc` controls whether an exception should be
            thrown if a validation or condition is violated or if a list with
            all encountered violations should be returned instead.
        '''
        try:
            self.model_validate(self, strict=strict)
        except ValidationError as validation_error:
            if raise_exc:
                raise validation_error
            return [validation_error]
        finally:
            self._init_rune_parent(recurse=True)
        return []

    def validate_conditions(self,
                            recursively: bool = True,
                            raise_exc: bool = True) -> list:
        ''' This method will check all conditions and the cardinality of all
            attributes of this object. This includes conditions and cardinality
            of properties specified in the base classes. If the parameter
            `recursively` is set to `True`, it will invoke the validation on the
            rune defined attributes of this object too.
            The parameter `raise_exc` controls whether an exception should be
            thrown if a condition is not met or if a list with all encountered
            condition violations should be returned instead.
        '''
        self_rep = object.__repr__(self)
        logging.info('Checking conditions for %s ...', self_rep)
        exceptions = []
        for name, condition in get_conditions(self.__class__, BaseDataClass):
            logging.info('Checking condition %s for %s...', name, self_rep)
            if not condition(self):
                msg = f'Condition "{name}" for {repr(self)} failed!'
                logging.error(msg)
                exc = ConditionViolationError(msg)
                if raise_exc:
                    raise exc
                exceptions.append(exc)
            else:
                logging.info('Condition %s for %s satisfied.', name, self_rep)
        if recursively:
            for k, v in self.__dict__.items():
                if k.startswith('__'):  # ignore *all* private vars!
                    continue
                logging.info('Validating conditions of property %s', k)
                exceptions += _validate_conditions_recursively(
                    v, raise_exc=raise_exc)
        err = f'with {len(exceptions)}' if exceptions else 'without'
        logging.info('Done conditions checking for %s %s errors.', self_rep,
                     err)
        return exceptions

    @classmethod
    def get_model_version(cls):
        ''' Attempt to obtain the Rune model version, in case of a failure,
            0.0.0 will be returned
        '''
        try:
            module = importlib.import_module(
                cls.__module__.split('.', maxsplit=1)[0])
            return getattr(module, 'rune_model_version', '0.0.0')
        # pylint: disable=bare-except
        except:  # noqa
            return '0.0.0'


def _validate_conditions_recursively(obj, raise_exc=True):
    '''Helper to execute conditions recursively on a model.'''
    unwrap = getattr(obj, '_unwrap', None)
    if callable(unwrap):
        obj = unwrap()
    if not obj:
        return []
    if isinstance(obj, BaseDataClass):
        return obj.validate_conditions(
            recursively=True,  # type:ignore
            raise_exc=raise_exc)
    if isinstance(obj, (tuple, MutableSequence)):
        exc = []
        for item in obj:
            exc += _validate_conditions_recursively(item, raise_exc=raise_exc)
        return exc
    return []

# EOF
