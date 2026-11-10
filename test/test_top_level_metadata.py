'''The top-level object's metadata is written by rune_serialize.

Metadata is normally written by the serializer of the field holding an object.
The top-level object isn't held in a field, so rune_serialize has to write it.
'''
import json
from typing import Annotated, Optional

import pytest
from pydantic import ConfigDict, Field

from rune.runtime.base_data_class import BaseDataClass

# The classes below are what rune-python-generator produces from this model:
#
#   type Party:
#       [metadata key]
#       name string (0..1)
#
#   type Trade:
#       [metadata key]
#       party Party (0..1)
#
#   type Plain:
#       name string (0..1)


class Party(BaseDataClass):
    '''no doc'''
    model_config = ConfigDict(defer_build=True)
    _ALLOWED_METADATA = {'@key', '@key:external'}
    name: Optional[str] = Field(None, description='')


class Trade(BaseDataClass):
    '''no doc'''
    model_config = ConfigDict(defer_build=True)
    _ALLOWED_METADATA = {'@key', '@key:external'}
    party: Annotated[Optional[Party], Party.serializer(), Party.validator(('@key', '@key:external'))] = Field(None, description='')


class Plain(BaseDataClass):
    '''a type without metadata'''
    model_config = ConfigDict(defer_build=True)
    name: Optional[str] = Field(None, description='')


def _round_trip(data: dict) -> dict:
    obj = Trade.rune_deserialize(data, validate_model=False)
    return json.loads(obj.rune_serialize(validate_model=False))


@pytest.mark.parametrize('metadata', [
    pytest.param({'@key': 'k1'}, id='key'),
    pytest.param({'@key:external': 'e1'}, id='external-key'),
    pytest.param({'@key': 'k1', '@key:external': 'e1'}, id='both'),
])
def test_top_level_metadata_is_written(metadata):
    '''a key on the top-level object is written back'''
    written = _round_trip(metadata | {'party': {'name': 'A'}})
    for key, value in metadata.items():
        assert written.get(key) == value


def test_top_level_and_field_metadata_are_both_written():
    '''writing the top-level key doesn't affect a key on a field'''
    written = _round_trip({'@key': 'k1', 'party': {'@key:external': 'e1', 'name': 'A'}})
    assert written.get('@key') == 'k1'
    assert written['party'] == {'@key:external': 'e1', 'name': 'A'}


def test_top_level_metadata_with_validation():
    '''the key is also written with validation on'''
    obj = Trade.rune_deserialize({'@key': 'k1', 'party': {'name': 'A'}})
    assert json.loads(obj.rune_serialize()).get('@key') == 'k1'


def test_rune_type_fields_are_kept():
    '''@type, @model and @version are still written'''
    written = _round_trip({'@key': 'k1', 'party': {'name': 'A'}})
    assert {'@type', '@model', '@version'} <= set(written)


def test_object_without_metadata_writes_no_key():
    '''nothing is added when there is no metadata'''
    written = json.loads(Plain(name='A').rune_serialize(validate_model=False))
    assert not [k for k in written if k.startswith('@key')]


def test_writing_is_repeatable():
    '''writing twice gives the same result and keeps the key on the object'''
    obj = Trade.rune_deserialize({'@key': 'k1', 'party': {'name': 'A'}}, validate_model=False)
    first = obj.rune_serialize(validate_model=False)
    assert obj.rune_serialize(validate_model=False) == first
    assert obj.get_meta('@key') == 'k1'
