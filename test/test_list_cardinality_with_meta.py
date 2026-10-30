'''Tests for (0..*) cardinality fields whose element types carry metadata.

Regression tests for the bug where EnumWithMetaMixin.deserialize and
BasicTypeMetaDataMixin.deserialize called model._init_meta(allowed_meta)
unconditionally, without guarding against the case where model is a list.

When Pydantic's validator is placed on the outer Annotated wrapper of a
list-cardinality field (the pattern the Python generator uses for deferred
bundled types), Pydantic passes the entire list to the validator in a single
call instead of calling it per-element.  The existing code then does:

    model = obj          # obj is a list
    ...                  # no isinstance branch matches a list
    model._init_meta(…)  # AttributeError: 'list' has no _init_meta

These tests call deserialize directly so they are Pydantic-version-independent.
'''
from enum import Enum
from typing import Annotated, Optional

import pytest

from rune.runtime.metadata import (
    BaseReference, ComplexTypeMetaDataMixin, EnumWithMetaMixin, KeyType,
    Reference, StrWithMeta, UnresolvedReference, _EnumWrapper,
)
from rune.runtime.base_data_class import BaseDataClass
from pydantic import Field


class Color(EnumWithMetaMixin, Enum):
    '''test enum with metadata support'''
    RED = 'RED'
    BLUE = 'BLUE'
    GREEN = 'GREEN'


# ---------------------------------------------------------------------------
# EnumWithMetaMixin.deserialize — line 706 in metadata.py
# ---------------------------------------------------------------------------

def test_enum_deserialize_list_does_not_raise():
    '''deserialize must not raise when obj is a list (Pydantic list-level call)'''
    result = Color.deserialize(['RED', 'BLUE'], allowed_meta=set())
    assert isinstance(result, list)
    assert len(result) == 2


def test_enum_deserialize_list_elements_wrapped():
    '''each element in the returned list must be an _EnumWrapper'''
    result = Color.deserialize(['RED', 'GREEN'], allowed_meta=set())
    assert all(isinstance(item, _EnumWrapper) for item in result)
    assert result[0] == Color.RED
    assert result[1] == Color.GREEN


def test_enum_deserialize_list_single_element():
    '''single-element list works correctly'''
    result = Color.deserialize(['BLUE'], allowed_meta=set())
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == Color.BLUE


def test_enum_deserialize_empty_list():
    '''empty list returns an empty list without error'''
    result = Color.deserialize([], allowed_meta=set())
    assert result == []


def test_enum_deserialize_scalar_still_works():
    '''scalar string input is unaffected by the fix'''
    result = Color.deserialize('RED', allowed_meta=set())
    assert isinstance(result, _EnumWrapper)
    assert result == Color.RED


def test_enum_deserialize_dict_still_works():
    '''dict input (with @data key) is unaffected by the fix'''
    result = Color.deserialize({'@data': 'BLUE'}, allowed_meta=set())
    assert isinstance(result, _EnumWrapper)
    assert result == Color.BLUE


# ---------------------------------------------------------------------------
# BasicTypeMetaDataMixin.deserialize — line 523 in metadata.py
# ---------------------------------------------------------------------------

def test_strwithmeta_deserialize_list_does_not_raise():
    '''StrWithMeta.deserialize must not raise when obj is a list'''
    handler = lambda x: x  # identity handler; Pydantic provides this in practice
    result = StrWithMeta.deserialize(
        [{'@data': 'alpha'}, {'@data': 'beta'}],
        handler=handler,
        base_types=StrWithMeta._INPUT_TYPES,
        allowed_meta=set(),
    )
    assert isinstance(result, list)
    assert len(result) == 2


def test_strwithmeta_deserialize_list_elements_deserialized():
    '''each element in the list must be a StrWithMeta after deserialization'''
    handler = lambda x: x
    result = StrWithMeta.deserialize(
        [{'@data': 'hello'}, {'@data': 'world'}],
        handler=handler,
        base_types=StrWithMeta._INPUT_TYPES,
        allowed_meta=set(),
    )
    assert all(isinstance(item, StrWithMeta) for item in result)
    assert result[0] == 'hello'
    assert result[1] == 'world'


def test_strwithmeta_deserialize_scalar_still_works():
    '''scalar string input is unaffected by the fix'''
    handler = lambda x: x
    result = StrWithMeta.deserialize(
        'EUR',
        handler=handler,
        base_types=StrWithMeta._INPUT_TYPES,
        allowed_meta=set(),
    )
    assert isinstance(result, StrWithMeta)
    assert result == 'EUR'


# ---------------------------------------------------------------------------
# ComplexTypeMetaDataMixin.deserialize — the "Expected ... or dict but got list" error
# ---------------------------------------------------------------------------

class SimpleItem(BaseDataClass, ComplexTypeMetaDataMixin):
    '''minimal complex type with metadata support for testing'''
    value: str = Field(...)


def test_complex_deserialize_list_does_not_raise():
    '''ComplexTypeMetaDataMixin.deserialize must not raise when obj is a list'''
    result = SimpleItem.deserialize(
        [{'value': 'a'}, {'value': 'b'}], allowed_meta=set()
    )
    assert isinstance(result, list)
    assert len(result) == 2


def test_complex_deserialize_list_elements_are_instances():
    '''each element in the returned list must be a SimpleItem'''
    result = SimpleItem.deserialize(
        [{'value': 'x'}, {'value': 'y'}], allowed_meta=set()
    )
    assert all(isinstance(item, SimpleItem) for item in result)
    assert result[0].value == 'x'
    assert result[1].value == 'y'


def test_complex_deserialize_empty_list():
    '''empty list returns an empty list without error'''
    result = SimpleItem.deserialize([], allowed_meta=set())
    assert result == []


def test_complex_deserialize_scalar_dict_still_works():
    '''single dict input is unaffected by the fix'''
    result = SimpleItem.deserialize({'value': 'z'}, allowed_meta=set())
    assert isinstance(result, SimpleItem)
    assert result.value == 'z'


# ---------------------------------------------------------------------------
# serialise() on list, None, and reference values (Issue #42 Bug 4)
# ---------------------------------------------------------------------------

class ListSerializationContainer(BaseDataClass):
    '''Model with list-cardinality and optional fields using outer serializers.'''
    items: Optional[Annotated[
        list[SimpleItem | BaseReference],
        SimpleItem.serializer(),
        SimpleItem.validator(('@key', '@ref')),
    ]] = Field(None)
    strings: Optional[Annotated[
        list[StrWithMeta | BaseReference],
        StrWithMeta.serializer(),
        StrWithMeta.validator(('@scheme', '@key', '@ref')),
    ]] = Field(None)
    colors: Optional[Annotated[
        list[Color | BaseReference],
        Color.serializer(),
        Color.validator(('@scheme', '@key', '@ref')),
    ]] = Field(None)


def test_complex_serialise_handles_none_list_and_references():
    '''ComplexTypeMetaDataMixin.serialise handles None, lists, and references.'''
    assert SimpleItem.serialise(None) is None
    assert SimpleItem.serialise([]) == []

    item = SimpleItem(value='alpha')
    item.set_meta(check_allowed=False, key='k1')
    unresolved = UnresolvedReference({'@ref': 'k1'})
    resolved = Reference(item)

    assert SimpleItem.serialise([item, unresolved, resolved]) == [
        {'@key': 'k1', 'value': 'alpha'},
        {'@ref': 'k1'},
        {'@ref': 'k1'},
    ]


def test_basic_serialise_handles_none_list_and_references():
    '''BasicTypeMetaDataMixin.serialise handles None, lists, and references.'''
    assert StrWithMeta.serialise(None, str) is None
    assert StrWithMeta.serialise([], str) == []

    s = StrWithMeta('EUR', scheme='iso4217', key='cur-1')
    unresolved = UnresolvedReference({'@ref': 'cur-1'})
    resolved = Reference(s)

    assert StrWithMeta.serialise([s, unresolved, resolved], str) == [
        {'@scheme': 'iso4217', '@key': 'cur-1', '@data': 'EUR'},
        {'@ref': 'cur-1'},
        {'@ref': 'cur-1'},
    ]


def test_enum_serialise_handles_none_list_and_references():
    '''EnumWithMetaMixin.serialise handles None, lists, and references.'''
    handler = lambda x: x.value
    assert Color.serialise(None, handler, None) is None
    assert Color.serialise([], handler, None) == []

    wrapped = Color.deserialize({'@data': 'RED', '@scheme': 'rgb', '@key': 'c1'},
                                allowed_meta={'@scheme', '@key'})
    unresolved = UnresolvedReference({'@ref': 'c1'})
    resolved = Reference(wrapped)

    assert Color.serialise([wrapped, Color.BLUE, unresolved, resolved], handler, None) == [
        {'@scheme': 'rgb', '@key': 'c1', '@data': 'RED'},
        {'@data': 'BLUE'},
        {'@ref': 'c1'},
        {'@ref': 'c1'},
    ]


def test_model_rune_serialize_with_list_and_optional_none_fields():
    '''rune_serialize() succeeds on models with list-typed and None fields.'''
    empty_model = ListSerializationContainer(items=None, strings=None, colors=None)
    dumped_empty = empty_model.model_dump()
    assert dumped_empty == {'items': None, 'strings': None, 'colors': None}

    populated = ListSerializationContainer.rune_deserialize({
        'items': [{'@key': 'i1', 'value': 'one'}, {'@ref': 'i1'}],
        'strings': [{'@key': 's1', '@data': 'hello'}, {'@ref': 's1'}],
        'colors': [{'@key': 'c1', '@data': 'GREEN', '@scheme': 'rgb'}, {'@ref': 'c1'}],
    })
    assert populated.items[1] is populated.items[0]
    assert populated.strings[1] is populated.strings[0]
    assert populated.colors[1] is populated.colors[0]

    dumped = populated.model_dump(exclude_unset=True)
    assert dumped == {
        'items': [{'@key': 'i1', 'value': 'one'}, {'@ref': 'i1'}],
        'strings': [{'@key': 's1', '@data': 'hello'}, {'@ref': 's1'}],
        'colors': [{'@key': 'c1', '@scheme': 'rgb', '@data': 'GREEN'}, {'@ref': 'c1'}],
    }

# EOF
