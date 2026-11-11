'''Base class for all Rune type classes'''
import copy
import importlib
import json
import logging
from collections.abc import MutableSequence
from decimal import Decimal
from typing import Any, Literal, Self, cast

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
    ComplexTypeMetaDataMixin,
    Reference,
    UnresolvedReference,
    _EnumWrapper,
)
from rune.runtime.reference_list import _ReferenceList

logger = logging.getLogger(__name__)

ROOT_CONTAINER = '__rune_root_metadata'
_LIST_INDICES = TypeAdapter(list[int])


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
                self.resolve_references(ignore_dangling=True, recurse=False)

    def _prepare_reference_list(self, name: str, value: Any,
                                 *, copy_existing: bool = False) -> _ReferenceList | None:
        '''Prepare reference lists, copying existing storage when taking ownership.'''
        unwrap = getattr(value, '_unwrap', None)
        if callable(unwrap):
            value = unwrap()
        if isinstance(value, _ReferenceList):
            for index, child in enumerate(value):
                if reference := value.reference_at(index):
                    self._check_reference(name, child, reference[1], child,
                                          already_bound=True)
                elif isinstance(child, Reference):
                    self._check_reference(name, child.target, child.key_type, child)
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
        for name, value in self.__dict__.items():
            if name.startswith('__') or name in refs:
                continue
            if isinstance(value, _ReferenceList):
                for index, child in enumerate(value):
                    if value.reference_at(index) is None:
                        yield child
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

        unwrap = getattr(data, '_unwrap', None)
        if callable(unwrap):
            data = unwrap()
            if isinstance(data, BaseDataClass):
                return {
                    name: getattr(data, name)
                    for name in type(data).__pydantic_fields__
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
        refs = self._get_rune_refs_container()
        updates = {}
        list_fields = {}
        for name in type(self).model_fields:
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
        res = self.__dict__.get(ROOT_CONTAINER, {}) | res
        return res

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

    @model_validator(mode='wrap')
    @classmethod
    def _deserialize_refs(cls, data: Any,
                          handler: ModelWrapValidatorHandler[Self]) -> Self:
        '''should resolve refs after creation'''
        metadata = {}
        if isinstance(data, dict):
            metadata = {k: v for k, v in data.items() if k.startswith('@')}
            if aux := cls._create_unresolved_ref(metadata):
                return aux  # type: ignore
        obj = handler(data)
        # Pydantic instances have writable dictionaries; Pyright can infer the
        # metaclass's MappingProxyType instead (microsoft/pyright#11548).
        obj_state = cast(dict[str, Any], obj.__dict__)
        # Pydantic may replace a list during revalidation. Keep occurrence
        # metadata before wiring parents so reference targets are not reparented.
        source = cls._unwrap_cow_validation_input(data)
        for name in cls.model_fields:
            original = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
            if isinstance(original, _ReferenceList):
                obj_state[name] = original.with_values(getattr(obj, name))
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
        obj.resolve_references(ignore_dangling=True, recurse=False)
        return obj

    def _init_rune_parent(self, original: 'BaseDataClass | None' = None):
        '''sets the rune parent in all properties'''
        state = cast(dict[str, Any], self.__dict__)
        if not self.get_rune_parent() and RUNE_OBJ_MAPS not in state:
            state[RUNE_OBJ_MAPS] = {}

        for name in type(self).model_fields:
            values = self._prepare_reference_list(name, getattr(self, name))
            if values is not None:
                state[name] = values
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
        # List mutations can introduce new containment children after construction.
        for child in self._contained_values():
            if (isinstance(child, BaseMetaDataMixin) and child is not self
                    and child.get_rune_parent() is None):
                child._set_rune_parent(self)
        if recurse:
            for child in self._contained_values():
                if isinstance(child, BaseDataClass):
                    child.resolve_references(ignore_dangling=ignore_dangling,
                                             recurse=recurse)

        refs = []
        for prop_nm, obj in self.__dict__.items():
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
