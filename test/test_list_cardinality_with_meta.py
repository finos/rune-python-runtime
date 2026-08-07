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

import pytest

from rune.runtime.metadata import ComplexTypeMetaDataMixin, EnumWithMetaMixin, StrWithMeta, _EnumWrapper
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

# EOF
