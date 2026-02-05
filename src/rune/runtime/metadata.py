'''Classes representing annotated basic Rune types'''
import uuid
import datetime
import importlib
from enum import Enum
from functools import partial, lru_cache
from decimal import Decimal
from typing import Any, Never, get_args, Iterable
from typing_extensions import Self, Tuple
from pydantic import (PlainSerializer, PlainValidator, WrapValidator,
                      WrapSerializer)
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
        return _get_basic_type(embedded_type[0])
    return annotated_type


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


class Reference:
    '''manages a reference to a object with a key'''
    def __init__(self,
                 target: str | Any,
                 ext_key: str | None = None,
                 key_type: KeyType | None = None,
                 parent=None):
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


class UnresolvedReference:
    '''used by the deserialization to hold temporarily unresolved references'''
    def __init__(self, key):
        rune_type, self.key = list(key.items())[0]
        self.key_type = KeyType.from_rune(rune_type)

    def get_reference(self, parent):
        '''convert to a resolved reference'''
        return Reference(self.key, key_type=self.key_type, parent=parent)


class BaseMetaDataMixin:
    '''Base class for the meta data support of basic amd complex types'''
    _DEFAULT_SCOPE_TYPE = 'cdm.event.common.TradeState'
    __meta_check_disabled = False

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
        if not (fqcn := getattr(self, '_FQRTN', None)):
            fqcn = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
        return fqcn == scope

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
            except:  # noqa
                self.set_meta(key=None)
                raise
        return key

    def set_external_key(self, key: str, key_type: KeyType):
        '''registers this object under the provided external key'''
        aux = self.get_meta(key_type.key_tag)
        if aux and aux != key:
            raise ValueError(f'This object already has an external key {aux}!'
                             f'Can\'t change it to {key}')
        if aux == key:
            return

        self.set_meta(check_allowed=True, **{key_type.key_tag: key})
        try:
            self._get_object_map(key_type)[key] = self
        except:  # noqa
            self.set_meta(check_allowed=True, **{key_type.key_tag: None})
            raise

    def get_object_by_key(self, key: str, key_type: KeyType):
        '''retrieve an object with a key an key type'''
        return self._get_object_map(key_type)[key]

    def get_rune_parent(self) -> Self | None:
        '''the parent object'''
        return self.__dict__.get(PARENT_PROP)

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

    def _bind_property_to(self, property_nm: str, ref: Reference):
        '''set the property to reference the object referenced by the key'''
        old_val = getattr(self, property_nm)
        allowed_ref_types = getattr(self, '_KEY_REF_CONSTRAINTS', {})
        if (ref.key_type.rune_ref_tag not in allowed_ref_types.get(
                property_nm, {})): # and not _replaceable(old_val)):
            raise ValueError(f'Ref of type {ref.key_type} '
                             f'not allowed for {property_nm}. Allowed types '
                             f'are: {allowed_ref_types.get(property_nm, {})}')

        field_type = self.__class__.__annotations__.get(property_nm)
        allowed_type = _get_basic_type(field_type)
        if not (isinstance(allowed_type, str)
                or isinstance(ref.target, allowed_type)):
            raise ValueError("Can't set reference. Incompatible types: "
                             f"expected {allowed_type}, "
                             f"got {ref.target.__class__}")

        refs = self.__dict__.setdefault(REFS_CONTAINER, {})
        if property_nm not in refs:
            # not a reference - check if allowed to replace with one
            if not _replaceable(old_val):
                raise ValueError(f'Property {property_nm} of type '
                                 f"{type(old_val)} can't be a reference")
            # pylint: disable=protected-access
            if isinstance(old_val, BaseMetaDataMixin):
                old_val._check_props_allowed({ref.key_type.rune_ref_tag: ''})

        # setattr(self, property_nm, ref.target)  # nope - need to avoid here!
        self.__dict__[property_nm] = ref.target  # NOTE: avoid here setattr
        refs[property_nm] = (ref.target_key, ref.key_type)

    def _register_keys(self, metadata):
        keys = {k: v for k, v in metadata.items() if k.startswith('@key') and v}
        for key_t, key_v in keys.items():
            self._get_object_map(KeyType.from_rune(key_t))[key_v] = self

    def _get_object_map(self, key_type: KeyType) -> dict[str, Any]:
        if not self.get_rune_parent():
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
    def _type_to_cls(cls, metadata:dict[str, Any]):
        if rune_type:= metadata.pop('@type', None):
            rune_class_name = rune_type.rsplit('.', maxsplit=1)[-1]
            rune_module = importlib.import_module(rune_type)
            return getattr(rune_module, rune_class_name)
        return cls  # support for legacy json

    @classmethod
    def serialise(cls, obj) -> dict:
        '''used as serialisation method with pydantic'''
        res = obj.serialise_meta()
        res |= obj.model_dump(exclude_unset=True, exclude_defaults=True)
        if cls != obj.__class__:
            # pylint: disable=protected-access
            res = {'@type': obj._FQRTN} | res
        return res

    @classmethod
    def deserialize(cls, obj, allowed_meta: set[str]):
        '''method used as pydantic `validator`'''
        if isinstance(obj, cls):
            if cls.meta_checks_enabled():
                obj._init_meta(allowed_meta)  # pylint: disable=protected-access
            return obj

        if isinstance(obj, Reference):
            return obj

        if not isinstance(obj, dict):
            raise PydanticCustomError('Input Validation Error',
                                      'Expected either {my_type} or dict but '
                                      'got {type}.',
                                      {'type': type(obj), 'my_type': cls})
        metadata = {k: obj[k] for k in obj.keys() if k.startswith('@')}

        # References deserialization treatment
        if aux := cls._create_unresolved_ref(metadata):
            return aux

        # Model creation
        for k in metadata.keys():
            obj.pop(k)

        rune_cls = cls._type_to_cls(metadata)
        if rune_cls != cls and not issubclass(rune_cls, cls):
            raise ValueError(f'{rune_cls} has to be a child class of {cls}!')
        model = rune_cls.model_validate(obj)  # type: ignore
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
        return PlainSerializer(cls.serialise, return_type=dict)

    @classmethod
    @lru_cache
    def validator(cls, allowed_meta: tuple[str] | tuple[Never, ...] = tuple()):
        '''default validator for the specific class'''
        allowed = set(allowed_meta)
        return PlainValidator(partial(cls.deserialize, allowed_meta=allowed),
                              json_schema_input_type=dict)


class BasicTypeMetaDataMixin(BaseMetaDataMixin):
    '''holds the metadata associated with an instance'''
    _INPUT_TYPES: Any | Tuple[Any, ...] = str  # to be overridden by subclasses
    _OUTPUT_TYPE: Any = str  # to be overridden by subclasses
    _JSON_OUTPUT = str | dict

    @classmethod
    def _check_type(cls, value):
        if not isinstance(value, cls._INPUT_TYPES):
            raise ValueError(f'{cls.__name__} can be instantiated only with '
                             f'one of the following type(s): {cls._INPUT_TYPES},'
                             f' however the value is of type {type(value)}')

    @classmethod
    def serialise(cls, obj, base_type) -> dict:
        '''used as serialisation method with pydantic'''
        res = obj.serialise_meta()
        res['@data'] = base_type(obj)
        return res

    @classmethod
    def deserialize(cls, obj, handler, base_types, allowed_meta: set[str]):
        '''method used as pydantic `validator`'''
        model = obj
        if isinstance(obj, base_types) and not isinstance(obj, cls):
            model = cls(obj)  # type: ignore
        elif isinstance(obj, dict):
            if ref := cls._create_unresolved_ref(obj):
                return ref
            data = obj.pop('@data')
            model = cls(data, **obj)  # type: ignore
            model._register_keys(obj)
        if cls.meta_checks_enabled():
            model._init_meta(allowed_meta)  # pylint: disable=protected-access
        return handler(model)

    @classmethod
    @lru_cache
    def serializer(cls):
        '''should return the validator for the specific class'''
        ser_fn = partial(cls.serialise, base_type=cls._OUTPUT_TYPE)
        return PlainSerializer(ser_fn, return_type=dict)

    @classmethod
    @lru_cache
    def validator(cls, allowed_meta: tuple[str]):
        '''default validator for the specific class'''
        allowed = set(allowed_meta)
        return WrapValidator(partial(cls.deserialize,
                                     base_types=cls._INPUT_TYPES,
                                     allowed_meta=allowed),
                             json_schema_input_type=cls._JSON_OUTPUT)


class DateWithMeta(datetime.date, BasicTypeMetaDataMixin):
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


class TimeWithMeta(datetime.time, BasicTypeMetaDataMixin):
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


class DateTimeWithMeta(datetime.datetime, BasicTypeMetaDataMixin):
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


class NumberWithMeta(Decimal, BasicTypeMetaDataMixin):
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
            raise ValueError("enum_instance must be an instance of an Enum")
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


class EnumWithMetaMixin:
    '''holds the metadata associated with a Rune Enum'''
    @classmethod
    def serialise(cls, obj, handler, info) -> dict:
        '''used as serialisation method with pydantic'''
        res = obj.serialise_meta()
        res['@data'] = handler(obj.enum_instance, info)
        return res

    @classmethod
    def deserialize(cls, obj, allowed_meta: set[str]):
        '''method used as pydantic `validator`'''
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
            data = obj.pop('@data')
            model = _EnumWrapper(cls(data))  # type: ignore
            model.set_meta(check_allowed=False, **obj)
            model._register_keys(obj)  # pylint: disable=protected-access
        if _EnumWrapper.meta_checks_enabled():
            model._init_meta(allowed_meta)  # pylint: disable=protected-access
        return model

    @classmethod
    @lru_cache
    def serializer(cls):
        '''should return the validator for the specific class'''
        return WrapSerializer(cls.serialise, return_type=dict)

    @classmethod
    @lru_cache
    def validator(cls, allowed_meta: tuple[str] | tuple[Never, ...] = tuple()):
        '''default validator for the specific class'''
        allowed = set(allowed_meta)
        return PlainValidator(partial(cls.deserialize, allowed_meta=allowed),
                              json_schema_input_type=str | dict)

# EOF
