'''Base class for all Rune type classes'''
import contextvars
import copy
import importlib
import json
import logging
from collections.abc import Mapping, MutableSequence
from decimal import Decimal
from enum import Enum
from functools import cache
from typing import Any, ForwardRef, Literal, Self, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    ModelWrapValidatorHandler,
    TypeAdapter,
    ValidationError,
    model_serializer,
    model_validator,
)
from pydantic.main import IncEx

from rune.runtime.conditions import ConditionViolationError, get_conditions
from rune.runtime.metadata import (
    REFS_CONTAINER,
    RUNE_OBJ_MAPS,
    BaseMetaDataMixin,
    BaseReference,
    ComplexTypeMetaDataMixin,
    Reference,
    UnresolvedReference,
    _EnumWrapper,
)
from rune.runtime.reference_list import _ReferenceList
from rune.runtime.utils import rune_unwrap_proxy

logger = logging.getLogger(__name__)

ROOT_CONTAINER = '__rune_root_metadata'
_LIST_INDICES = TypeAdapter(list[int])


def _annotation_contains(annotation: Any, base_types: tuple[type, ...], *, unknown: bool = False) -> bool:
    '''Inspect union and collection annotations, treating unknown types conservatively.'''
    if annotation is Any or annotation is object or isinstance(annotation, (str, ForwardRef)):
        return unknown
    if isinstance(annotation, type) and issubclass(annotation, base_types):
        return True
    return any(_annotation_contains(arg, base_types, unknown=unknown) for arg in get_args(annotation))
_CONSTRUCTING = contextvars.ContextVar('rune_constructing', default=False)


class _RuneModelMeta(type(BaseModel)):
    '''Records that an object is being built directly in Python.

    The validator of a choice can't otherwise tell `ChoiceBasic(string='x')`
    from the JSON `{"string": "x"}`. This has to be done here: defining
    `__init__` on the model would make pydantic build nested models through
    it, so that JSON would look like direct construction as well.
    '''
    def __call__(cls, *args, **kwargs):
        token = _CONSTRUCTING.set(True)
        try:
            return super().__call__(*args, **kwargs)
        finally:
            _CONSTRUCTING.reset(token)


class BaseDataClass(BaseModel, ComplexTypeMetaDataMixin,
                    metaclass=_RuneModelMeta):
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

    @classmethod
    @cache
    def _reference_fields(cls) -> tuple[str, ...]:
        '''Cache fields that declare reference permissions or reference tokens.'''
        return tuple(name for name, field in cls.model_fields.items()
                     if any(tag.startswith('@ref') for tag in cls._reference_constraints(name))
                     or _annotation_contains(field.annotation, (BaseReference,), unknown=True))

    @classmethod
    @cache
    def _containment_fields(cls) -> tuple[str, ...]:
        '''Skip primitive-only fields during parent wiring and graph traversal.'''
        return tuple(name for name, field in cls.model_fields.items()
                     if _annotation_contains(field.annotation, (BaseMetaDataMixin, BaseReference, Enum),
                                             unknown=True))

    @classmethod
    def model_rebuild(cls, *, force: bool = False, raise_errors: bool = True,
                      _parent_namespace_depth: int = 2,
                      _types_namespace: Mapping[str, Any] | None = None) -> bool | None:
        '''Invalidate field caches when deferred annotations or constraints change.'''
        cls._reference_fields.cache_clear()
        cls._containment_fields.cache_clear()
        cls._reference_constraints.cache_clear()
        return super().model_rebuild(force=force, raise_errors=raise_errors,
                                     _parent_namespace_depth=_parent_namespace_depth + 1,
                                     _types_namespace=_types_namespace)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Reference):
            self._bind_property_to(name, value)
        else:
            reference_list = self._prepare_reference_list(name, value, copy_existing=True)
            if reference_list is not None:
                value = reference_list
            # replace reference with an object
            if name in self._get_rune_refs_container():
                self._remove_rune_ref(name)
                if isinstance(self.__dict__[name], _EnumWrapper):
                    self.__dict__[name] = _EnumWrapper()  # type: ignore
            # if the value is an enum, pass it to the EnumWrapper
            if (isinstance(self.__dict__[name], _EnumWrapper)
                    and not isinstance(value, _EnumWrapper)):
                value = _EnumWrapper(value)
            # if the value is a "model", register as rune_parent
            if isinstance(value, BaseMetaDataMixin):
                value._set_rune_parent(self)  # type: ignore
            super().__setattr__(name, value)
            if reference_list is not None:
                for index, child in enumerate(reference_list):
                    if (reference_list.reference_at(index) is None
                            and isinstance(child, BaseMetaDataMixin)):
                        child._set_rune_parent(self)
                self._resolve_references(ignore_dangling=True, recurse=False)

    def _prepare_reference_list(self, name: str, value: Any,
                                 *, copy_existing: bool = False) -> _ReferenceList | None:
        '''Prepare reference lists, copying existing storage when taking ownership.'''
        value = rune_unwrap_proxy(value)
        if isinstance(value, _ReferenceList):
            if value.has_references():
                for child, reference in value.reference_items():
                    self._check_reference(name, child, reference[1], child, already_bound=True)
            return value.copy() if copy_existing else value
        if not isinstance(value, (MutableSequence, tuple)):
            return None
        constraints = self._reference_constraints(name)
        if (any(tag.startswith('@ref') for tag in constraints)
                or any(isinstance(item, (Reference, UnresolvedReference)) for item in value)):
            # A newly created wrapper already owns its storage.
            return self._prepare_reference_list(name, _ReferenceList(value))
        return None

    def _contained_values(self):
        '''Yield containment children, skipping scalar and list reference edges.'''
        refs = self._get_rune_refs_container()
        for name in type(self)._containment_fields():
            if name in refs:
                continue
            value = self.__dict__.get(name)
            if isinstance(value, _ReferenceList):
                yield from value.contained_values()
            elif isinstance(value, (MutableSequence, tuple)):
                yield from value
            else:
                yield value

    def __eq__(self, other: object) -> bool:
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

        data = rune_unwrap_proxy(data)
        if isinstance(data, BaseDataClass):
            return {
                name: getattr(data, name)
                for name in type(data).__pydantic_fields__
            }
        return data

    @model_validator(mode='before')
    @classmethod
    def _unwrap_cow_inputs(cls, data: Any) -> Any:
        '''Unwrap copy-on-write proxies before pydantic validation.'''
        return cls._unwrap_cow_validation_input(data)

    @model_serializer(mode='wrap')
    def _serialize_refs(self, serializer, info):
        '''should replace objects with refs while serializing'''
        refs = self._get_rune_refs_container()
        reference_fields = type(self)._reference_fields()
        root_metadata = self.__dict__.get(ROOT_CONTAINER)
        if not refs and not reference_fields:
            result = serializer(self)
            if self._is_rune_choice():
                result = self._flatten_choice(result)
            return root_metadata | result if root_metadata else result
        updates = {}
        list_fields = {}
        for name in dict.fromkeys((*reference_fields, *refs)):
            if name not in self.__dict__:
                continue
            if info.include is not None and name not in info.include:
                continue
            if (info.exclude is not None and name in info.exclude
                    and (not isinstance(info.exclude, dict)
                         or info.exclude[name] is True or info.exclude[name] is ...)):
                continue
            value = getattr(self, name)
            if isinstance(value, _ReferenceList):
                pending = any(isinstance(child, (Reference, UnresolvedReference)) for child in value)
                if not value.has_references() and not pending and info.include is None and info.exclude is None:
                    continue
                self._prepare_reference_list(name, value)
                indices = self._serialized_list_indices(name, len(value), info)
                selected = set(indices)
                list_fields[name] = (value, indices)
                updates[name] = [None if index not in selected or value.reference_at(index) else child
                                 for index, child in enumerate(value)]
                pending = (value[index] for index in indices)
            else:
                pending = (value,)
                if name in refs and isinstance(value, BaseDataClass):
                    # Avoid traversing the target before replacing it with @ref.
                    updates[name] = type(value).model_construct()
            if any(isinstance(child, (Reference, UnresolvedReference)) for child in pending):
                raise ValueError(f'Unbound reference in {name!r}; '
                                 'call resolve_references() before serialization')

        projected = self.model_copy(update=updates) if updates else self
        if updates:
            object.__setattr__(projected, '__pydantic_fields_set__',
                               self.__pydantic_fields_set__.copy())
        res = serializer(projected)
        for property_nm, (key, ref_type) in refs.items():
            output_name = self._serialized_field_name(property_nm, info)
            if output_name in res:
                res[output_name] = {ref_type.rune_ref_tag: key}
        for name, (values, indices) in list_fields.items():
            output_name = self._serialized_field_name(name, info)
            if output_name not in res:
                continue
            written = res[output_name]
            # Whole-list plain serializers bypass Pydantic's item selection.
            if len(written) == len(values):
                written = [written[index] for index in indices]
            for position, index in enumerate(indices):
                if reference := values.reference_at(index):
                    key, key_type = reference
                    written[position] = {key_type.rune_ref_tag: key}
            res[output_name] = written
        if self._is_rune_choice():
            res = self._flatten_choice(res)
        return root_metadata | res if root_metadata else res

    @staticmethod
    def _serialized_list_indices(name: str, length: int, info) -> list[int]:
        '''Map filtered output positions back to their original list indices.'''
        include = info.include.get(name) if isinstance(info.include, dict) else None
        exclude = info.exclude.get(name) if isinstance(info.exclude, dict) else None

        if include is True or include is ...:
            include = None
        return _LIST_INDICES.dump_python(list(range(length)), include=include,
                                         exclude=exclude)

    def _serialized_field_name(self, name: str, info) -> str:
        '''Use the same field alias as the wrapped Pydantic serializer.'''
        by_alias = info.by_alias
        if by_alias is None:
            by_alias = self.model_config.get('serialize_by_alias', False)
        field = type(self).model_fields[name]
        return (field.serialization_alias or name) if by_alias else name

    @classmethod
    def _is_rune_choice(cls) -> bool:
        '''True for a class generated from a Rune `choice`.'''
        return getattr(cls, '_RUNE_CHOICE', False)

    @classmethod
    def _option_class(cls, name: str):
        '''The data class declared for a choice option. None if the option
        is a basic type or an enum.'''
        def find(tp):
            if isinstance(tp, type):
                return tp if issubclass(tp, BaseDataClass) else None
            for arg in get_args(tp):
                if found := find(arg):
                    return found
            return None
        return find(cls.model_fields[name].annotation)

    def _flatten_choice(self, res: dict) -> dict:
        '''Rewrites a serialized choice in the Rune JSON format: `@type` names
        the selected option and its fields sit directly on the choice.'''
        selected = [name for name in type(self).model_fields
                    if self.__dict__.get(name) is not None]
        if not selected:
            return {}
        if len(selected) > 1:
            raise ValueError(f'Multiple choice options selected for '
                             f'{self._fqrtn()}: {selected}')
        name = selected[0]
        value = self.__dict__[name]
        option = res[name]
        declared = self._option_class(name)
        if (isinstance(value, BaseDataClass) and declared is not None
                and type(value) is not declared):
            raise ValueError(f'Choice option {name} of {self._fqrtn()} holds '
                             f'{type(value)}, but must be exactly {declared}')
        option_cls = type(value) if isinstance(value, BaseDataClass) else declared
        if option_cls is None:  # basic type or enum
            if isinstance(option, dict) and '@data' in option:
                return {'@type': name} | option
            return {'@type': name, '@data': option}
        # pylint: disable=protected-access
        if option_cls._is_rune_choice():
            # A nested choice is already flat, unless the option has metadata
            # or is a reference
            if '@data' in option or any(k.startswith('@ref') for k in option):
                return {'@type': option_cls._fqrtn()} | option
            return option
        return {'@type': option_cls._fqrtn()} | {
            k: v for k, v in option.items() if k != '@type'}

    @classmethod
    def _choice_has_option(cls, rune_type: str) -> bool:
        '''True if `rune_type` is an option of this choice, or of a choice
        nested in it.'''
        # pylint: disable=protected-access
        for name in cls.model_fields:
            option = cls._option_class(name)
            if option is None:
                if name == rune_type:
                    return True
            elif option._fqrtn() == rune_type or (
                    option._is_rune_choice() and option._choice_has_option(rune_type)):
                return True
        return False

    @classmethod
    def _choice_to_fields(cls, data: dict) -> dict:
        '''Converts a choice in the Rune JSON format to this class's fields,
        with the option selected by `@type`.'''
        if '@type' not in data:
            if any(k.startswith('@ref') for k in data):
                return data  # a reference to a choice
            if '@data' in data:
                metadata = {k: v for k, v in data.items() if k != '@data'}
                return metadata | cls._choice_to_fields(data['@data'])
            raise ValueError(f"Missing required '@type' for choice "
                             f'{cls._fqrtn()}')
        rune_type = data['@type']
        body = {k: v for k, v in data.items() if k != '@type'}
        # pylint: disable=protected-access
        for name in cls.model_fields:
            option = cls._option_class(name)
            if option is None:
                if name == rune_type:
                    return {name: body['@data']}
            elif option._fqrtn() == rune_type:
                return {name: body}
            elif option._is_rune_choice() and option._choice_has_option(rune_type):
                return {name: data}
        raise ValueError(f"'{rune_type}' is not an option of choice "
                         f'{cls._fqrtn()}')

    @model_validator(mode='wrap')
    @classmethod
    def _deserialize_refs(cls, data: Any,
                          handler: ModelWrapValidatorHandler[Self]) -> Self:
        '''should resolve refs after creation'''
        metadata = {}
        if isinstance(data, dict):
            if cls._is_rune_choice() and not _CONSTRUCTING.get():
                data = cls._choice_to_fields(data)
            metadata = {k: v for k, v in data.items() if k.startswith('@')}
            if aux := cls._create_unresolved_ref(metadata):
                return aux  # type: ignore
        obj = handler(data)
        # Pydantic instances have writable dictionaries; Pyright can infer the
        # metaclass's MappingProxyType instead (microsoft/pyright#11548).
        obj_state = cast(dict[str, Any], obj.__dict__)
        # Pydantic may replace a list during revalidation. Keep occurrence
        # metadata before wiring parents so reference targets are not reparented.
        source = data if isinstance(data, (dict, BaseDataClass)) else cls._unwrap_cow_validation_input(data)
        if isinstance(source, dict) and cls._containment_fields():
            original_lists = ((name, value) for name, value in source.items()
                              if name in obj_state and isinstance(value, _ReferenceList))
        elif isinstance(source, BaseDataClass):
            original_lists = ((name, source.__dict__.get(name)) for name in cls._reference_fields())
        else:
            original_lists = ()
        for name, original in original_lists:
            if isinstance(original, _ReferenceList):
                obj_state[name] = obj._prepare_reference_list(
                    name, original.with_values(getattr(obj, name)))
        if (isinstance(source, BaseDataClass)
                and (original_refs := source._get_rune_refs_container())):
            obj_state[REFS_CONTAINER] = dict(original_refs)
        if isinstance(source, BaseDataClass) and source is not obj:
            obj_state[RUNE_OBJ_MAPS] = {
                key_type: dict(objects)
                for key_type, objects in source.__dict__.get(RUNE_OBJ_MAPS, {}).items()
            }
        obj._init_rune_parent(source if isinstance(source, BaseDataClass) else None)
        if isinstance(data, dict):
            # Deferred annotations can bypass the field metadata validator.
            # Retain target keys before resolving references to this model.
            keys = {k: v for k, v in metadata.items() if k.startswith('@key')}
            if keys:
                # Field-level validators handle allowed metadata, including
                # scoped keys that are not declared on the target class.
                obj.set_meta(check_allowed=False, **keys)
                obj._register_keys(keys)  # pylint: disable=protected-access
        if cls._reference_fields():
            obj._resolve_references(ignore_dangling=True, recurse=False)
        return obj

    def _init_rune_parent(self, original: 'BaseDataClass | None' = None):
        '''sets the rune parent in all properties'''
        state = cast(dict[str, Any], self.__dict__)
        if not self.get_rune_parent() and RUNE_OBJ_MAPS not in state:
            state[RUNE_OBJ_MAPS] = {}

        for name in type(self)._reference_fields():
            values = self._prepare_reference_list(name, getattr(self, name))
            if values is not None:
                state[name] = values
        if not type(self)._containment_fields():
            return
        original_children = {id(child) for child in original._contained_values()} if original else set()
        for child in self._contained_values():
            if isinstance(child, BaseMetaDataMixin):
                # Field validators can retain existing instances on revalidation.
                # Do not move those shared children out of the original graph.
                if id(child) in original_children:
                    continue
                child._set_rune_parent(self)  # pylint: disable=protected-access

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
            # The top-level object isn't held in a field, so no field
            # serializer writes its own metadata
            root_meta |= self.serialise_meta()  # type: ignore

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
            self.__dict__.pop(ROOT_CONTAINER, None)  # type: ignore

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
            # Decimal keeps every digit; a float would round long numbers
            rune_dict = json.loads(rune_data, parse_float=Decimal)
            # NOTE: json.loads will not create the right atomic types
            # (e.g. date, datetime, time etc) and the strict model validate
            # will not attempt to convert them.
            strict = False
        elif not isinstance(rune_data, dict):
            raise TypeError(f'rune_data is of type {type(rune_data)}, '
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

    @classmethod
    def model_validate_json(cls,
                            json_data: str | bytes | bytearray,
                            *,
                            strict: bool | None = None,
                            extra: Any = None,
                            context: Any | None = None,
                            by_alias: bool | None = None,
                            by_name: bool | None = None) -> Self:
        '''Like pydantic's `model_validate_json`, but JSON numbers are read
        as `Decimal`, so none are rounded.

        Pydantic parses JSON numbers with a fraction or exponent as binary
        floats, which round values beyond about 15 to 17 significant digits.

        With `strict=True`, pydantic's own JSON parsing is kept, so strict
        validation behaves as before. Validating parsed values strictly would
        reject some JSON forms that are accepted today, such as an enum's
        value or a UUID string.
        '''
        if strict:
            return super().model_validate_json(json_data,
                                               strict=strict,
                                               extra=extra,
                                               context=context,
                                               by_alias=by_alias,
                                               by_name=by_name)
        # Decimal keeps every digit; a float would round long numbers
        data = json.loads(json_data, parse_float=Decimal)
        return cls.model_validate(data,
                                  strict=strict,
                                  extra=extra,
                                  context=context,
                                  by_alias=by_alias,
                                  by_name=by_name)

    def resolve_references(self, ignore_dangling=False, recurse=True):
        '''resolves all attributes which are references'''
        self._refresh_rune_parents(recurse=recurse)
        self._resolve_references(ignore_dangling=ignore_dangling, recurse=recurse)

    def _refresh_rune_parents(self, *, recurse: bool) -> None:
        '''Register newly added containment before resolving any forward references.'''
        for child in self._contained_values():
            if (isinstance(child, BaseMetaDataMixin) and child is not self
                    and child.get_rune_parent() is None):
                child._set_rune_parent(self)
            if recurse and isinstance(child, BaseDataClass) and type(child)._containment_fields():
                child._refresh_rune_parents(recurse=True)

    def _resolve_references(self, *, ignore_dangling: bool, recurse: bool) -> None:
        '''Resolve tokens without repeating parent wiring during model validation.'''
        if recurse and type(self)._containment_fields():
            for child in self._contained_values():
                if isinstance(child, BaseDataClass):
                    child._resolve_references(ignore_dangling=ignore_dangling, recurse=True)

        refs = []
        for prop_nm in type(self)._reference_fields():
            obj = self.__dict__.get(prop_nm)
            candidates = enumerate(obj) if isinstance(obj, _ReferenceList) else ((None, obj),)
            for index, candidate in candidates:
                if isinstance(candidate, (UnresolvedReference, Reference)):
                    try:
                        refs.append((prop_nm, index, candidate.get_reference(self)))
                    except KeyError:
                        if not ignore_dangling:
                            raise

        for prop_nm, index, ref in refs:
            self._bind_property_to(prop_nm, ref, index)

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
                raise
            return [validation_error]
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
        logger.info('Checking conditions for %s ...', self_rep)
        exceptions = []
        for name, condition in get_conditions(self.__class__, BaseDataClass):
            logger.info('Checking condition %s for %s...', name, self_rep)
            if not condition(self):
                msg = f'Condition "{name}" for {self!r} failed!'
                logger.error(msg)
                exc = ConditionViolationError(msg)
                if raise_exc:
                    raise exc
                exceptions.append(exc)
            else:
                logger.info('Condition %s for %s satisfied.', name, self_rep)
        if recursively:
            for value in self._contained_values():
                exceptions += _validate_conditions_recursively(
                    value, raise_exc=raise_exc)
        err = f'with {len(exceptions)}' if exceptions else 'without'
        logger.info('Done conditions checking for %s %s errors.', self_rep, err)
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
    obj = rune_unwrap_proxy(obj)
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
