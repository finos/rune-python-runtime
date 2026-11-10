'''Writing (0..*) attributes whose serializer wraps the whole list.

The generator puts the serializer and validator around the whole list, so the
list reaches `serialise` in a single call: for a list attribute with metadata,
and for every list attribute in a generated bundle. `deserialize` already
handles this (see test_list_cardinality_with_meta.py); these tests cover the
writing side.
'''
import json
from enum import Enum
from typing import Annotated, ClassVar, Optional

from pydantic import ConfigDict, Field

import pytest

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.metadata import BaseReference, EnumWithMetaMixin, StrWithMeta

NS = 'test_list_cardinality_serialise'

# The classes below are what rune-python-generator produces from this model,
# in the form used in a generated bundle, where each class sets _FQRTN and every
# attribute has the runtime's serializer and validator:
#
#   enum Colour:
#       Red
#       Blue
#
#   type Item:
#       name string (0..1)
#
#   type SpecialItem extends Item:
#       extra string (0..1)
#
#   type Party:
#       [metadata key]
#       name string (0..1)
#
#   type Holder:
#       items Item (0..*)
#       parties Party (0..*)
#       refs Party (0..*)
#           [metadata reference]
#       names string (0..*)
#           [metadata scheme]
#       colours Colour (0..*)
#           [metadata scheme]


class Colour(EnumWithMetaMixin, Enum):
    BLUE = "Blue"
    RED = "Red"


class Item(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.Item'
    name: Optional[str] = Field(None, description='')


class SpecialItem(Item):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.SpecialItem'
    extra: Optional[str] = Field(None, description='')


class Party(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.Party'
    _ALLOWED_METADATA = {'@key', '@key:external'}
    name: Optional[str] = Field(None, description='')


class Holder(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.Holder'
    items: Annotated[Optional[list[Item | None]], Item.serializer(), Item.validator()] = Field(None, description='')
    parties: Annotated[Optional[list[Party | None]], Party.serializer(), Party.validator(('@key', '@key:external'))] = Field(None, description='')
    refs: Annotated[Optional[list[Party | BaseReference | None]], Party.serializer(), Party.validator(('@key', '@key:external', '@ref', '@ref:external'))] = Field(None, description='')
    names: Annotated[Optional[list[StrWithMeta | None]], StrWithMeta.serializer(), StrWithMeta.validator(('@scheme', ))] = Field(None, description='')
    colours: Annotated[Optional[list[Colour | None]], Colour.serializer(), Colour.validator(('@scheme', ))] = Field(None, description='')

    _KEY_REF_CONSTRAINTS = {
        'refs': {'@key', '@key:external', '@ref', '@ref:external'}
    }


@pytest.mark.parametrize('attribute, value', [
    pytest.param('items', [{'name': 'A'}, {'name': 'B'}], id='objects'),
    pytest.param('parties', [{'@key:external': 'p1', 'name': 'A'}], id='objects-with-keys'),
    pytest.param('names',
                 [{'@scheme': 'http://a', '@data': 'x'}, {'@scheme': 'http://b', '@data': 'y'}],
                 id='basic-with-metadata'),
    pytest.param('colours',
                 [{'@scheme': 'http://a', '@data': 'Red'}, {'@data': 'Blue'}],
                 id='enum-with-metadata'),
])
def test_write_list(attribute, value):
    '''a list is written item by item, each with its own metadata'''
    model = Holder.rune_deserialize({attribute: value}, validate_model=False)
    written = json.loads(model.rune_serialize(validate_model=False))
    assert written.get(attribute) == value


def test_write_empty_list():
    '''an empty list is written as an empty list'''
    written = json.loads(Holder(items=[]).rune_serialize(validate_model=False))
    assert written.get('items') == []


def test_write_absent_list():
    '''an absent list is not written'''
    written = json.loads(Holder().rune_serialize(validate_model=False))
    assert 'items' not in written


def test_write_list_built_in_python():
    '''a list built in Python is written the same way'''
    written = json.loads(Holder(items=[Item(name='A')]).rune_serialize(validate_model=False))
    assert written.get('items') == [{'name': 'A'}]


def test_write_list_keeps_subtype():
    '''an item of a subtype is written with its @type'''
    model = Holder(items=[SpecialItem(name='A', extra='x')])
    written = json.loads(model.rune_serialize(validate_model=False))
    assert written.get('items') == [{'@type': f'{NS}.SpecialItem', 'name': 'A', 'extra': 'x'}]


def test_write_list_of_unresolved_references():
    '''a reference in a list is written back as the reference it was read as'''
    data = {'parties': [{'@key:external': 'p1', 'name': 'A'}],
            'refs': [{'@ref:external': 'p1'}]}
    model = Holder.rune_deserialize(data, validate_model=False)
    assert json.loads(model.rune_serialize(validate_model=False)).get('refs') == data['refs']
