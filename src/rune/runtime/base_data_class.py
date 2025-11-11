'''Base class for all Rune type classes'''
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
from rune.runtime.metadata import (ComplexTypeMetaDataMixin, Reference,
                                   UnresolvedReference, BaseMetaDataMixin,
                                   _EnumWrapper, RUNE_OBJ_MAPS)

ROOT_CONTAINER = '__rune_root_metadata'


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
            # replace reference with an object
            if name in self._get_rune_refs_container():
                self._remove_rune_ref(name)
                if isinstance(self.__dict__[name], _EnumWrapper):
                    self.__dict__[name] = _EnumWrapper()
            # if the value is an enum, pass it to the EnumWrapper
            if (isinstance(self.__dict__[name], _EnumWrapper)
                    and not isinstance(value, _EnumWrapper)):
                value = _EnumWrapper(value)
            # if the value is a "model", register as rune_parent
            if isinstance(value, BaseMetaDataMixin):
                value._set_rune_parent(self)
            super().__setattr__(name, value)

    @model_serializer(mode='wrap')
    def _serialize_refs(self, serializer, info):
        '''should replace objects with refs while serializing'''
        res = serializer(self, info)
        refs = self._get_rune_refs_container()
        for property_nm, (key, ref_type) in refs.items():
            res[property_nm] = {ref_type.rune_ref_tag: key}
        res = self.__dict__.get(ROOT_CONTAINER, {}) | res
        return res

    @model_validator(mode='wrap')
    @classmethod
    def _deserialize_refs(cls, data: Any,
                          handler: ModelWrapValidatorHandler[Self]) -> Self:
        '''should resolve refs after creation'''
        obj = handler(data)
        obj._init_rune_parent()  # pylint: disable=protected-access
        obj.resolve_references(ignore_dangling=True, recurse=False)
        return obj

    def _init_rune_parent(self):
        '''sets the rune parent in all properties'''
        refs = self._get_rune_refs_container()
        if not self.get_rune_parent() and RUNE_OBJ_MAPS not in self.__dict__:
            self.__dict__[RUNE_OBJ_MAPS] = {}

        for prop_nm, obj in self.__dict__.items():
            if (isinstance(obj, BaseMetaDataMixin)
                    and not prop_nm.startswith('__') and prop_nm not in refs):
                obj._set_rune_parent(self)  # pylint: disable=protected-access

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

            root_meta = self.__dict__.setdefault(ROOT_CONTAINER, {})
            root_meta['@type'] = self._FQRTN
            root_meta['@model'] = self._FQRTN.split('.', maxsplit=1)[0]
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
                         raise_validation_errors: bool = True) -> BaseModel:
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
        rune_cls = cls._type_to_cls(rune_dict)
        model = rune_cls.model_validate(rune_dict, strict=strict)
        model.resolve_references(ignore_dangling=False, recurse=True)
        if validate_model:
            model.validate_model(check_rune_constraints=check_rune_constraints,
                                 strict=strict,
                                 raise_exc=raise_validation_errors)
        return model

    def resolve_references(self, ignore_dangling=False, recurse=True):
        '''resolves all attributes which are references'''
        if recurse:
            for prop_nm, obj in self.__dict__.items():
                if (isinstance(obj, BaseDataClass)
                        and not prop_nm.startswith('__')):
                    obj.resolve_references(ignore_dangling=ignore_dangling,
                                           recurse=recurse)

        refs = []
        for prop_nm, obj in self.__dict__.items():
            if isinstance(obj, (UnresolvedReference, Reference)):
                try:
                    refs.append((prop_nm, obj.get_reference(self)))
                except KeyError:
                    if not ignore_dangling:
                        raise

        for prop_nm, ref in refs:
            self._bind_property_to(prop_nm, ref)

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

    def add_to_list_attribute(self, attr_name: str, value) -> None:
        '''
        Adds a value to a list attribute, ensuring the value is of an allowed
        type.

        Parameters:
        attr_name (str): Name of the list attribute.
        value: Value to add to the list.

        Raises:
        AttributeError: If the attribute name is not found or not a list.
        TypeError: If the value type is not one of the allowed types.
        '''
        if not hasattr(self, attr_name):
            raise AttributeError(f"Attribute {attr_name} not found.")

        attr = getattr(self, attr_name)
        if not isinstance(attr, list):
            raise AttributeError(f"Attribute {attr_name} is not a list.")

        # Get allowed types for the list elements
        allowed_types = self.get_allowed_types_for_list_field(attr_name)

        # Check if value is an instance of one of the allowed types
        if not isinstance(value, allowed_types):
            raise TypeError(f"Value must be an instance of {allowed_types}, "
                            f"not {type(value)}")

        attr.append(value)

    @classmethod
    def get_allowed_types_for_list_field(cls, field_name: str):
        '''
        Gets the allowed types for a list field in a Pydantic model, supporting
        both Union and | operator.

        Parameters:
        cls (type): The Pydantic model class.
        field_name (str): The field name.

        Returns:
        tuple: A tuple of allowed types.
        '''
        field_type = cls.__annotations__.get(field_name)
        if field_type and get_origin(field_type) is list:
            list_elem_type = get_args(field_type)[0]
            if get_origin(list_elem_type):
                return get_args(list_elem_type)
            return (list_elem_type, )  # Single type or | operator used
        return ()

    @classmethod
    def get_model_version(cls):
        ''' Attempt to obtain the Rune model version, in case of a failure,
            0.0.0 will be returned
        '''
        try:
            module = importlib.import_module(
                cls.__module__.split('.', maxsplit=1)[0])
            return getattr(module, 'rune_model_version', default='0.0.0')
        # pylint: disable=bare-except
        except:  # noqa
            return '0.0.0'


def _validate_conditions_recursively(obj, raise_exc=True):
    '''Helper to execute conditions recursively on a model.'''
    if not obj:
        return []
    if isinstance(obj, BaseDataClass):
        return obj.validate_conditions(
            recursively=True,  # type:ignore
            raise_exc=raise_exc)
    if isinstance(obj, (list, tuple)):
        exc = []
        for item in obj:
            exc += _validate_conditions_recursively(item, raise_exc=raise_exc)
        return exc
    return []

# EOF
