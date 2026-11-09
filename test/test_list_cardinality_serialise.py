'''Writing (0..*) fields whose serializer wraps the whole list.

The generator puts the serializer and validator around the whole list, so the
list reaches `serialise` in a single call. `deserialize` already handles this
(see test_list_cardinality_with_meta.py); these tests cover the writing side.
'''
import json
from enum import Enum
from typing import Annotated, ClassVar, Optional

import pytest
from pydantic import Field

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.metadata import BaseReference, EnumWithMetaMixin, StrWithMeta

NS = 'test_list_cardinality_serialise'
KEYS = ('@key', '@key:external')
REFS = ('@key', '@key:external', '@ref', '@ref:external')


class Colour(EnumWithMetaMixin, Enum):
    '''no doc'''
    RED = 'Red'
    BLUE = 'Blue'


class Item(BaseDataClass):
    '''no doc'''
    _FQRTN: ClassVar[str] = f'{NS}.Item'
    name: Optional[str] = Field(None)


class SpecialItem(Item):
    '''no doc'''
    _FQRTN: ClassVar[str] = f'{NS}.SpecialItem'
    extra: Optional[str] = Field(None)


class Party(BaseDataClass):
    '''no doc'''
    _FQRTN: ClassVar[str] = f'{NS}.Party'
    _ALLOWED_METADATA = {'@key', '@key:external'}
    name: Optional[str] = Field(None)


# As in generated code: an attribute named after its type needs an alias.
_Item, _Party = Item, Party


class ChoiceData(BaseDataClass):
    '''no doc'''
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceData'
    _RUNE_CHOICE: ClassVar[bool] = True
    Item: Optional[_Item] = Field(None)
    Party: Optional[_Party] = Field(None)


class Holder(BaseDataClass):
    '''no doc'''
    _FQRTN: ClassVar[str] = f'{NS}.Holder'
    items: Annotated[Optional[list[Item | None]], Item.serializer(),
                     Item.validator()] = Field(None)
    parties: Annotated[Optional[list[Party | None]], Party.serializer(),
                       Party.validator(KEYS)] = Field(None)
    refs: Annotated[Optional[list[Party | BaseReference | None]],
                    Party.serializer(), Party.validator(REFS)] = Field(None)
    choices: Annotated[Optional[list[ChoiceData | None]], ChoiceData.serializer(),
                       ChoiceData.validator()] = Field(None)
    names: Annotated[Optional[list[StrWithMeta | None]], StrWithMeta.serializer(),
                     StrWithMeta.validator(('@scheme', ))] = Field(None)
    colours: Annotated[Optional[list[Colour | None]], Colour.serializer(),
                       Colour.validator(('@scheme', ))] = Field(None)
    _KEY_REF_CONSTRAINTS = {'refs': {'@ref', '@ref:external'}}


@pytest.mark.parametrize('attribute, value', [
    pytest.param('items', [{'name': 'A'}, {'name': 'B'}], id='objects'),
    pytest.param('parties', [{'@key:external': 'p1', 'name': 'A'}], id='objects-with-keys'),
    pytest.param('choices',
                 [{'@type': f'{NS}.Item', 'name': 'A'}, {'@type': f'{NS}.Party', 'name': 'B'}],
                 id='choices'),
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
    assert written[attribute] == value


def test_write_empty_list():
    '''an empty list is written as an empty list'''
    written = json.loads(Holder(items=[]).rune_serialize(validate_model=False))
    assert written['items'] == []


def test_write_absent_list():
    '''an absent list is not written'''
    written = json.loads(Holder().rune_serialize(validate_model=False))
    assert 'items' not in written


def test_write_list_built_in_python():
    '''a list built in Python is written the same way'''
    written = json.loads(Holder(items=[Item(name='A')]).rune_serialize(validate_model=False))
    assert written['items'] == [{'name': 'A'}]


def test_write_list_keeps_subtype():
    '''an item of a subtype is written with its @type'''
    model = Holder(items=[SpecialItem(name='A', extra='x')])
    written = json.loads(model.rune_serialize(validate_model=False))
    assert written['items'] == [{'@type': f'{NS}.SpecialItem', 'name': 'A', 'extra': 'x'}]


def test_write_list_of_unresolved_references():
    '''a reference in a list is written back as the reference it was read as'''
    data = {'parties': [{'@key:external': 'p1', 'name': 'A'}],
            'refs': [{'@ref:external': 'p1'}]}
    model = Holder.rune_deserialize(data, validate_model=False)
    assert json.loads(model.rune_serialize(validate_model=False))['refs'] == data['refs']
