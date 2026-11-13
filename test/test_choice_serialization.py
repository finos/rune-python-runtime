# Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

"""Wire-format choices use generated dispatch; Python construction uses fields."""
import datetime
from decimal import Decimal
from enum import Enum
import json
import sys
import types
from typing import Annotated, ClassVar

import pytest
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.conditions import rune_condition
from rune.runtime.cow import rune_cow
from rune.runtime.func_proxy import rune_finalize_return
from rune.runtime.metadata import BaseReference, EnumWithMetaMixin, StrWithMeta
from rune.runtime.object_builder import ObjectBuilder
from rune.runtime.utils import rune_check_one_of

NS = 'test_choice_serialization'


class Keyed(BaseDataClass):
    _FQRTN: ClassVar[str] = f'{NS}.Keyed'
    _ALLOWED_METADATA: ClassVar[set[str]] = {'@key', '@key:external'}
    value: str


class A(BaseDataClass):
    _FQRTN: ClassVar[str] = f'{NS}.A'
    value: str
    keyed: Annotated[Keyed | None, Keyed.serializer(), Keyed.validator()] = None


class B(BaseDataClass):
    _FQRTN: ClassVar[str] = f'{NS}.B'
    value: str


class C(BaseDataClass):
    _FQRTN: ClassVar[str] = f'{NS}.C'
    value: str


class D(BaseDataClass):
    _FQRTN: ClassVar[str] = f'{NS}.D'
    value: str


class Choice(BaseDataClass):
    _RUNE_CHOICE: ClassVar[bool] = True
    _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {}

    @rune_condition
    def exactly_one_option(self):
        return rune_check_one_of(self, *type(self).model_fields, necessity=True)


_A, _B, _C, _D, _Keyed = A, B, C, D, Keyed


class ChoiceAB(Choice):
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceAB'
    _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {f'{NS}.A': 'A', f'{NS}.B': 'B'}
    A: _A | None = None
    B: _B | None = None


class ChoiceCD(Choice):
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceCD'
    _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {f'{NS}.C': 'C', f'{NS}.D': 'D'}
    C: _C | None = None
    D: _D | None = None


_ChoiceAB, _ChoiceCD = ChoiceAB, ChoiceCD


class ChoiceTree(Choice):
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceTree'
    _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {
        f'{NS}.A': 'ChoiceAB', f'{NS}.B': 'ChoiceAB',
        f'{NS}.C': 'ChoiceCD', f'{NS}.D': 'ChoiceCD',
    }
    ChoiceAB: _ChoiceAB | None = None
    ChoiceCD: _ChoiceCD | None = None


class ChoiceDeep(Choice):
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceDeep'
    _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {
        tag: 'tree' for tag in ChoiceTree._CHOICE_DISPATCH
    }
    tree: ChoiceTree | None = None


class ChoiceBasic(Choice):
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceBasic'
    _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {
        name: name for name in ('string', 'integer', 'boolean', 'number', 'date')
    }
    string: str | None = None
    integer: int | None = None
    boolean: bool | None = None
    number: Decimal | None = None
    date: datetime.date | None = None


class Currency(EnumWithMetaMixin, Enum):
    USD = 'USD'
    EUR = 'EUR'


class ChoiceBasicMeta(Choice):
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceBasicMeta'
    _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {
        'java.lang.String': 'string', f'{NS}.Currency': 'currency',
    }
    string: Annotated[StrWithMeta | None, StrWithMeta.serializer(),
                      StrWithMeta.validator(('@scheme', '@key:scoped'))] = None
    currency: Annotated[Currency | None, Currency.serializer(),
                        Currency.validator(('@scheme',))] = None


class ChoiceLocated(Choice):
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceLocated'
    _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {
        f'{NS}.ChoiceAB': 'ChoiceAB', f'{NS}.A': 'ChoiceAB',
        f'{NS}.B': 'ChoiceAB', 'string': 'string',
    }
    ChoiceAB: Annotated[_ChoiceAB | None, _ChoiceAB.serializer(),
                        _ChoiceAB.validator(('@key:scoped',))] = None
    string: str | None = None
    _KEY_REF_CONSTRAINTS: ClassVar[dict[str, set[str]]] = {'ChoiceAB': {'@key:scoped'}}


class ChoiceReference(Choice):
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceReference'
    _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {f'{NS}.Keyed': 'Keyed'}
    Keyed: Annotated[_Keyed | BaseReference | None, _Keyed.serializer(),
                     _Keyed.validator(('@ref', '@ref:external'))] = None
    _KEY_REF_CONSTRAINTS: ClassVar[dict[str, set[str]]] = {'Keyed': {'@ref', '@ref:external'}}


class ChoiceChoiceReference(Choice):
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceChoiceReference'
    _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {
        f'{NS}.ChoiceAB': 'ChoiceAB', f'{NS}.A': 'ChoiceAB', f'{NS}.B': 'ChoiceAB',
    }
    ChoiceAB: Annotated[_ChoiceAB | BaseReference | None, _ChoiceAB.serializer(),
                        _ChoiceAB.validator(('@ref:scoped',))] = None
    _KEY_REF_CONSTRAINTS: ClassVar[dict[str, set[str]]] = {'ChoiceAB': {'@ref:scoped'}}


class Root(BaseDataClass):
    _FQRTN: ClassVar[str] = f'{NS}.Root'
    data: ChoiceAB | None = None
    hooked: Annotated[ChoiceAB | None, ChoiceAB.serializer(), ChoiceAB.validator()] = None
    tree: ChoiceTree | None = None
    deep: ChoiceDeep | None = None
    basic: ChoiceBasic | None = None
    basic_meta: ChoiceBasicMeta | None = None
    entries: Annotated[list[ChoiceAB | BaseReference | None] | None,
                       ChoiceAB.serializer(), ChoiceAB.validator(('@ref:scoped',))] = None
    located: Annotated[ChoiceAB | None, ChoiceAB.serializer(),
                       ChoiceAB.validator(('@key:scoped',))] = None
    located_option: ChoiceLocated | None = None
    referenced_option: Annotated[ChoiceReference | None, ChoiceReference.serializer(),
                                 ChoiceReference.validator()] = None
    referenced_choice_option: ChoiceChoiceReference | None = None
    choice_ref: Annotated[ChoiceAB | BaseReference | None, ChoiceAB.serializer(),
                          ChoiceAB.validator(('@ref:scoped',))] = None
    keyed: Annotated[Keyed | None, Keyed.serializer(), Keyed.validator()] = None
    keyed_ref: Annotated[Keyed | BaseReference | None, Keyed.serializer(),
                         Keyed.validator(('@ref', '@ref:external'))] = None
    _KEY_REF_CONSTRAINTS: ClassVar[dict[str, set[str]]] = {
        'entries': {'@ref:scoped'}, 'located': {'@key:scoped'},
        'choice_ref': {'@ref:scoped'}, 'keyed_ref': {'@ref', '@ref:external'},
    }


@pytest.fixture(autouse=True)
def register_root_modules(monkeypatch):
    for cls in (Root, ChoiceAB, ChoiceBasic, ChoiceDeep, ChoiceLocated, ChoiceBasicMeta):
        module = types.ModuleType(cls._fqrtn())
        setattr(module, cls.__name__, cls)
        monkeypatch.setitem(sys.modules, module.__name__, module)


def content(model):
    return {key: value for key, value in json.loads(model.rune_serialize()).items()
            if key not in ('@type', '@model', '@version')}


@pytest.mark.parametrize('name, leaf, branch', [
    ('A', A, ChoiceAB), ('B', B, ChoiceAB), ('C', C, ChoiceCD), ('D', D, ChoiceCD),
])
def test_disjoint_nested_choices_round_trip_all_paths(name, leaf, branch):
    nested = branch(**{name: leaf(value='foo')})
    tree = ChoiceTree(**{branch.__name__: nested})
    original = Root(tree=tree, deep=ChoiceDeep(tree=tree))
    expected = {'@type': f'{NS}.{name}', 'value': 'foo'}
    assert content(original) == {'tree': expected, 'deep': expected}
    restored = Root.rune_deserialize(original.rune_serialize())
    assert getattr(getattr(restored.tree, branch.__name__), name).value == 'foo'
    assert content(restored) == content(original)


@pytest.mark.parametrize('field, value, wire_value', [
    ('string', 'foo', 'foo'), ('integer', 0, 0), ('boolean', False, False),
    ('number', Decimal('1.234567890123456789'), '1.234567890123456789'),
    ('date', datetime.date(2026, 11, 13), '2026-11-13'),
])
def test_basic_options_preserve_values(field, value, wire_value):
    original = Root(basic=ChoiceBasic(**{field: value}))
    written = json.loads(original.rune_serialize())
    assert written['basic'] == {'@type': field, '@data': wire_value}
    restored = Root.rune_deserialize(written)
    assert getattr(restored.basic, field) == value


@pytest.mark.parametrize('field, tag, value, metadata', [
    ('string', 'java.lang.String', 'foo', {'@scheme': 'urn:test', '@key:scoped': 'k1'}),
    ('currency', f'{NS}.Currency', 'USD', {'@scheme': 'urn:currencies'}),
])
def test_basic_and_enum_option_metadata_is_retained(field, tag, value, metadata):
    expected = {'basic_meta': {'@type': tag, '@data': value, **metadata}}
    restored = Root.rune_deserialize(expected)
    option = getattr(restored.basic_meta, field)
    assert all(option.get_meta(key) == val for key, val in metadata.items())
    assert content(restored) == expected


@pytest.mark.parametrize('field, value', [
    ('located', {'@key:scoped': 'outer', '@data': {'@type': f'{NS}.A', 'value': 'foo'}}),
    ('located_option', {'@type': f'{NS}.ChoiceAB', '@key:scoped': 'inner',
                        '@data': {'@type': f'{NS}.B', 'value': 'foo'}}),
])
def test_valid_choice_field_and_option_metadata_wrappers(field, value):
    model = Root.rune_deserialize({field: value})
    assert content(model) == {field: value}


def test_references_select_the_option_and_preserve_target_identity():
    data = {
        'keyed': {'@key:external': 'k1', 'value': 'target'},
        'referenced_option': {'@type': f'{NS}.Keyed', '@ref:external': 'k1'},
        'located': {'@key:scoped': 'c1', '@data': {'@type': f'{NS}.A', 'value': 'foo'}},
        'referenced_choice_option': {'@type': f'{NS}.ChoiceAB', '@ref:scoped': 'c1'},
        'choice_ref': {'@ref:scoped': 'c1'},
        'entries': [{'@ref:scoped': 'c1'}, {'@type': f'{NS}.B', 'value': 'inline'}],
    }
    model = Root.rune_deserialize(data)
    assert model.referenced_option.Keyed is model.keyed
    assert model.referenced_choice_option.ChoiceAB is model.located
    assert model.choice_ref is model.located
    assert model.entries[0] is model.located
    assert content(model) == data


def test_key_inside_selected_leaf_is_registered():
    data = {
        'data': {'@type': f'{NS}.A', 'value': 'foo',
                 'keyed': {'@key:external': 'k1', 'value': 'target'}},
        'keyed_ref': {'@ref:external': 'k1'},
    }
    model = Root.rune_deserialize(data)
    assert model.keyed_ref is model.data.A.keyed
    assert content(model) == data


@pytest.mark.parametrize('cls, initial', [
    (ChoiceAB, {'A': A(value='foo')}),
    (ChoiceBasic, {'string': 'foo'}),
    (ChoiceDeep, {'tree': ChoiceTree(ChoiceCD=ChoiceCD(D=D(value='foo')))}),
])
def test_concrete_choice_roots_round_trip(cls, initial):
    original = cls(**initial)
    restored = cls.rune_deserialize(original.rune_serialize())
    assert type(restored) is cls
    assert restored == original


@pytest.mark.parametrize('load', [
    lambda cls, data: cls(**data),
    lambda cls, data: cls.model_validate(data),
])
def test_python_construction_keeps_native_field_dictionaries(load):
    model = load(Root, {'data': {'A': {'value': 'foo'}}})
    assert model.data.A.value == 'foo'


def test_python_construction_also_accepts_tagged_values():
    model = Root(hooked={'@type': f'{NS}.A', 'value': 'foo'})
    assert model.hooked.A.value == 'foo'


def test_builders_and_function_returns_materialize_choices():
    direct = ObjectBuilder(ChoiceAB)
    direct.A.value = 'foo'
    assert direct.to_model().A.value == 'foo'
    parent = ObjectBuilder(Root)
    parent.deep.tree.ChoiceCD.D.value = 'bar'
    result = rune_finalize_return(parent)
    assert isinstance(result, Root)
    assert result.deep.tree.ChoiceCD.D.value == 'bar'


@pytest.mark.parametrize('load', [
    lambda data: Root.rune_deserialize(data, validate_model=False),
    lambda data: Root.model_validate_json(json.dumps(data)),
    lambda data: Root.model_validate_json(json.dumps(data), strict=True),
    lambda data: TypeAdapter(Root).validate_json(json.dumps(data)),
])
@pytest.mark.parametrize('value, error, message', [
    ({'A': {'value': 'foo'}}, ValidationError, "Missing required '@type'"),
    ({}, ValidationError, "Missing required '@type'"),
    ({'@type': f'{NS}.C', 'value': 'foo'}, ValidationError, 'not an option'),
    ({'@type': f'{NS}.ChoiceAB', 'A': {'value': 'foo'}}, ValidationError, 'not an option'),
    ({'@type': None}, TypeError, "Choice '@type' must be a string"),
    ({'@type': 5}, TypeError, "Choice '@type' must be a string"),
    ({'@type': []}, TypeError, "Choice '@type' must be a string"),
    ({'@data': None}, TypeError, "Choice '@data' must contain an object"),
    ({'@data': 'foo'}, TypeError, "Choice '@data' must contain an object"),
    ({'@data': 5}, TypeError, "Choice '@data' must contain an object"),
    ({'@data': []}, TypeError, "Choice '@data' must contain an object"),
    ({'@data': {}}, ValidationError, "Missing required '@type'"),
    ({'@data': {'@type': 5}}, TypeError, "Choice '@type' must be a string"),
])
def test_wire_readers_reject_invalid_choice_inputs(load, value, error, message):
    with pytest.raises(error, match=message):
        load({'data': value})


@pytest.mark.parametrize('value', [{'@type': 'string'}, {'@type': 'string', '@data': None}])
def test_basic_wire_option_requires_non_null_data(value):
    with pytest.raises(ValidationError):
        Root.rune_deserialize({'basic': value}, validate_model=False)


def test_wrapped_basic_wire_option_requires_data():
    with pytest.raises(ValidationError, match='Missing required'):
        Root.rune_deserialize({'basic_meta': {'@type': 'java.lang.String'}}, validate_model=False)


@pytest.mark.parametrize('load', [
    lambda cls: cls.model_validate_json('{}'),
    lambda cls: cls.model_validate_json('{}', strict=True),
    lambda cls: TypeAdapter(cls).validate_json('{}'),
])
def test_native_choice_default_factory_works_during_json_validation(load):
    class FactoryRoot(BaseDataClass):
        data: ChoiceAB = Field(default_factory=lambda: ChoiceAB(A=A(value='default')))

    assert load(FactoryRoot).data.A.value == 'default'


@pytest.mark.parametrize('load', [
    lambda cls: cls.model_validate_json('{}'),
    lambda cls: cls.model_validate_json('{}', strict=True),
    lambda cls: TypeAdapter(cls).validate_json('{}'),
])
def test_native_builder_default_factory_works_during_json_validation(load):
    def make_choice():
        builder = ObjectBuilder(ChoiceAB)
        builder.A.value = 'default'
        return builder.to_model()

    class FactoryRoot(BaseDataClass):
        data: ChoiceAB = Field(default_factory=make_choice)

    assert load(FactoryRoot).data.A.value == 'default'


def test_choice_option_cache_is_invalidated_by_model_rebuild():
    class Deferred(Choice):
        model_config = ConfigDict(defer_build=True)
        _FQRTN: ClassVar[str] = f'{NS}.Deferred'
        _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {f'{NS}.A': 'value'}
        value: 'Late | None' = None

    Deferred.model_rebuild(_types_namespace={'Late': A})
    assert Deferred(value=A(value='a')).model_dump(exclude_none=True)['@type'] == f'{NS}.A'
    Deferred.model_fields['value'].annotation = B | None
    Deferred._CHOICE_DISPATCH = {f'{NS}.B': 'value'}
    Deferred.model_rebuild(force=True)
    rebuilt = Deferred.model_validate_json(json.dumps({'@type': f'{NS}.B', 'value': 'b'}))
    assert type(rebuilt.value) is B
    assert rebuilt.model_dump(exclude_none=True)['@type'] == f'{NS}.B'


def test_failed_wire_read_does_not_change_python_construction_mode():
    with pytest.raises(ValidationError):
        Root.rune_deserialize({'data': {'A': {'value': 'foo'}}})
    assert Root.model_validate({'data': {'A': {'value': 'foo'}}}).data.A.value == 'foo'


@pytest.mark.parametrize('options', [
    {'exclude': {'A'}}, {'include': set()}, {'exclude': {'A': True}},
])
def test_filtering_out_selected_option_does_not_raise(options):
    assert ChoiceAB(A=A(value='foo')).model_dump(**options) == {}


def test_filtering_leaf_fields_keeps_discriminator():
    assert ChoiceAB(A=A(value='foo')).model_dump(
        exclude={'A': {'value'}}, exclude_none=True) == {'@type': f'{NS}.A'}


def test_serialization_alias_uses_serialized_field_name():
    class Aliased(Choice):
        _FQRTN: ClassVar[str] = f'{NS}.Aliased'
        _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {f'{NS}.A': 'a'}
        a: A | None = Field(None, serialization_alias='selected')

    assert Aliased(a=A(value='foo')).model_dump(
        by_alias=True, exclude_none=True) == {'@type': f'{NS}.A', 'value': 'foo'}


def test_mangled_basic_option_uses_original_wire_name():
    class Mangled(Choice):
        _FQRTN: ClassVar[str] = f'{NS}.Mangled'
        _CHOICE_DISPATCH: ClassVar[dict[str, str]] = {'class': 'rune_attr_class'}
        rune_attr_class: str | None = None

    value = Mangled(rune_attr_class='foo')
    assert value.model_dump() == {'@type': 'class', '@data': 'foo'}
    assert Mangled.model_validate_json(value.model_dump_json()).rune_attr_class == 'foo'


def test_multiple_options_and_subtypes_are_rejected_on_write():
    with pytest.raises(PydanticSerializationError, match='Multiple choice options'):
        ChoiceAB(A=A(value='a'), B=B(value='b')).model_dump()

    class SubA(A):
        extra: str = 'subtype'

    choice = ChoiceAB()
    choice.A = SubA(value='foo')
    with pytest.raises(PydanticSerializationError, match='must be exactly'):
        choice.model_dump()


def test_no_option_matches_java_empty_root_but_not_empty_nested_choice():
    assert ChoiceAB().model_dump() == {}
    with pytest.raises(PydanticSerializationError, match='has no selected option'):
        ChoiceTree(ChoiceAB=ChoiceAB()).model_dump()


def test_choice_field_metadata_stays_in_its_field_wrapper():
    model = Root(located=ChoiceAB(A=A(value='foo')))
    model.located.set_meta(**{'@key:scoped': 'location'})
    written = json.loads(model.rune_serialize())
    assert written['located'] == {
        '@key:scoped': 'location', '@data': {'@type': f'{NS}.A', 'value': 'foo'},
    }
    root_written = json.loads(model.located.rune_serialize())
    assert not any(key.startswith('@key') for key in root_written)
    assert model.located.get_meta('@key:scoped') == 'location'


def test_copy_on_write_choices_still_construct_without_mutating_source():
    original = ChoiceAB(A=A(value='original'))
    proxy = rune_cow(original)
    proxy.A.value = 'changed'
    restored = ChoiceAB.model_validate(proxy)
    assert restored.A.value == 'changed'
    assert original.A.value == 'original'


def test_ordinary_optional_fields_are_not_treated_as_choices():
    class Ordinary(BaseDataClass):
        a: A | None = None
        b: B | None = None

    value = Ordinary(a=A(value='foo'))
    assert value.model_dump(exclude_none=True) == {'a': {'value': 'foo'}}
