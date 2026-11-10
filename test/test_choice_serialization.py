'''Rune choice types are read and written in the Rune JSON format.

The selected option is named by `@type` and its fields sit directly on the
choice, as in rune-common's `RuneChoiceTypeSerializer` and
`RuneChoiceTypeDeserializer`. The nested form, `{"A": {...}}`, is not accepted.

Testing the following Rune definitions:

type A:
    fieldA string (1..1)

type ExtA extends A:
    fieldExt string (1..1)

type B:
    fieldB string (1..1)

type Located:
    fieldL string (1..1)

type Keyed:
    [metadata key]
    fieldK string (1..1)

type PaymentDate:
    [metadata key]
    day string (1..1)

type WithDate:
    paymentDate PaymentDate (0..1)

type OneOfHolder:
    A A (0..1)
    B B (0..1)

choice ChoiceData:
    A
    B

choice ChoiceBasic:
    string
    int

choice ChoiceNested:
    ChoiceData
    ChoiceBasic

choice ChoiceDeep:
    ChoiceNested

choice ChoiceWithMeta:
    Located
        [metadata location]
    Keyed
        [metadata reference]

choice ChoiceLocatedOption:
    ChoiceData
        [metadata location]
    string

choice ChoiceRefOption:
    ChoiceData
        [metadata address "pointsTo"=Root->located]
    string

choice ChoiceDated:
    WithDate
    B

type Root:
    data ChoiceData (0..1)
    basic ChoiceBasic (0..1)
    nested ChoiceNested (0..1)
    deep ChoiceDeep (0..1)
    entries ChoiceData (0..*)
    located ChoiceData (0..1)
        [metadata location]
    withMeta ChoiceWithMeta (0..1)
    locatedOption ChoiceLocatedOption (0..1)
    dated ChoiceDated (0..1)
    dateRef PaymentDate (0..1)
        [metadata reference]
    keyed Keyed (0..1)
    refOption ChoiceRefOption (0..1)
    withMetaHooked ChoiceWithMeta (0..1)
    choiceRef ChoiceData (0..1)
        [metadata address "pointsTo"=Root->located]
    holder OneOfHolder (0..1)

refOption and withMetaHooked have the runtime's serializer and validator
although they have no metadata, as every attribute of a class in a generated
bundle does.
'''
import json
from typing import Annotated, ClassVar, Optional

import pytest
from pydantic import ConfigDict, Field

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.conditions import rune_condition
from rune.runtime.metadata import BaseReference
from rune.runtime.utils import rune_check_one_of, rune_resolve_attr

NS = 'test_choice_serialization'

# The classes below are what rune-python-generator produces from the model
# above, in the form used in a generated bundle, where each class sets _FQRTN.


class A(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.A'
    fieldA: str = Field(..., description='')


class ExtA(A):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.ExtA'
    fieldExt: str = Field(..., description='')


class B(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.B'
    fieldB: str = Field(..., description='')


class Located(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.Located'
    fieldL: str = Field(..., description='')


class Keyed(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.Keyed'
    _ALLOWED_METADATA = {'@key', '@key:external'}
    fieldK: str = Field(..., description='')


class PaymentDate(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.PaymentDate'
    _ALLOWED_METADATA = {'@key', '@key:external'}
    day: str = Field(..., description='')


# Generated modules import a type used by an attribute of the same name with an
# alias, such as `from ...A import A as _A`, or the annotation would resolve to
# the attribute itself. The generator doesn't do this for `int` (an option of
# ChoiceBasic), so on Python 3.14 and later its generated annotation resolves to
# the attribute and isn't validated; the alias here gives the intended type.
_A, _B, _Located, _Keyed, _int = A, B, Located, Keyed, int


class WithDate(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.WithDate'
    paymentDate: Annotated[Optional[PaymentDate], PaymentDate.serializer(), PaymentDate.validator(('@key', '@key:external'))] = Field(None, description='')


_WithDate = WithDate


class OneOfHolder(BaseDataClass):
    '''A type with optional attributes is not a choice: it has no marker.'''
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.OneOfHolder'
    A: Optional[_A] = Field(None, description='')
    B: Optional[_B] = Field(None, description='')


class ChoiceData(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _RUNE_CHOICE: ClassVar[bool] = True
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceData'
    A: Optional[_A] = Field(None, description='')
    B: Optional[_B] = Field(None, description='')

    @rune_condition
    def condition_0_Choice(self):
        item = self
        return rune_check_one_of(item, 'A', 'B', necessity=True)


_ChoiceData = ChoiceData


class ChoiceBasic(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _RUNE_CHOICE: ClassVar[bool] = True
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceBasic'
    string: Optional[str] = Field(None, description='')
    int: Optional[_int] = Field(None, description='')

    @rune_condition
    def condition_0_Choice(self):
        item = self
        return rune_check_one_of(item, 'string', 'int', necessity=True)


_ChoiceBasic = ChoiceBasic


class ChoiceNested(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _RUNE_CHOICE: ClassVar[bool] = True
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceNested'
    ChoiceData: Optional[_ChoiceData] = Field(None, description='')
    ChoiceBasic: Optional[_ChoiceBasic] = Field(None, description='')

    @rune_condition
    def condition_0_Choice(self):
        item = self
        return rune_check_one_of(item, 'ChoiceData', 'ChoiceBasic', necessity=True)


_ChoiceNested = ChoiceNested


class ChoiceDeep(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _RUNE_CHOICE: ClassVar[bool] = True
    _CHOICE_ALIAS_MAP ={"ChoiceNested":[("ChoiceNested", rune_resolve_attr)],"ChoiceBasic":[("ChoiceNested", rune_resolve_attr)],"ChoiceData":[("ChoiceNested", rune_resolve_attr)]}
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceDeep'
    ChoiceNested: Optional[_ChoiceNested] = Field(None, description='')

    @rune_condition
    def condition_0_Choice(self):
        item = self
        return rune_check_one_of(item, 'ChoiceNested', necessity=True)


class ChoiceWithMeta(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _RUNE_CHOICE: ClassVar[bool] = True
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceWithMeta'
    Located: Annotated[Optional[_Located], _Located.serializer(), _Located.validator(('@key:scoped', ))] = Field(None, description='')
    Keyed: Annotated[Optional[_Keyed | BaseReference], _Keyed.serializer(), _Keyed.validator(('@key', '@key:external', '@ref', '@ref:external'))] = Field(None, description='')

    _KEY_REF_CONSTRAINTS = {
        'Located': {'@key:scoped'},
        'Keyed': {'@key', '@key:external', '@ref', '@ref:external'}
    }

    @rune_condition
    def condition_0_Choice(self):
        item = self
        return rune_check_one_of(item, 'Located', 'Keyed', necessity=True)


class ChoiceLocatedOption(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _RUNE_CHOICE: ClassVar[bool] = True
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceLocatedOption'
    ChoiceData: Annotated[Optional[_ChoiceData], _ChoiceData.serializer(), _ChoiceData.validator(('@key:scoped', ))] = Field(None, description='')
    string: Optional[str] = Field(None, description='')

    _KEY_REF_CONSTRAINTS = {
        'ChoiceData': {'@key:scoped'}
    }

    @rune_condition
    def condition_0_Choice(self):
        item = self
        return rune_check_one_of(item, 'ChoiceData', 'string', necessity=True)


class ChoiceRefOption(BaseDataClass):
    '''An option that is itself a choice, and is a reference.'''
    model_config = ConfigDict(defer_build=True)
    _RUNE_CHOICE: ClassVar[bool] = True
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceRefOption'
    ChoiceData: Annotated[Optional[_ChoiceData | BaseReference], _ChoiceData.serializer(), _ChoiceData.validator(('@ref:scoped', ))] = Field(None, description='')
    string: Optional[str] = Field(None, description='')

    _KEY_REF_CONSTRAINTS = {
        'ChoiceData': {'@ref:scoped'}
    }

    @rune_condition
    def condition_0_Choice(self):
        item = self
        return rune_check_one_of(item, 'ChoiceData', 'string', necessity=True)


class ChoiceDated(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _RUNE_CHOICE: ClassVar[bool] = True
    _FQRTN: ClassVar[str] = f'{NS}.ChoiceDated'
    WithDate: Optional[_WithDate] = Field(None, description='')
    B: Optional[_B] = Field(None, description='')

    @rune_condition
    def condition_0_Choice(self):
        item = self
        return rune_check_one_of(item, 'WithDate', 'B', necessity=True)


class Root(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.Root'
    data: Optional[ChoiceData] = Field(None, description='')
    basic: Optional[ChoiceBasic] = Field(None, description='')
    nested: Optional[ChoiceNested] = Field(None, description='')
    deep: Optional[ChoiceDeep] = Field(None, description='')
    entries: Optional[list[ChoiceData | None]] = Field(None, description='')
    located: Annotated[Optional[ChoiceData], ChoiceData.serializer(), ChoiceData.validator(('@key:scoped', ))] = Field(None, description='')
    withMeta: Optional[ChoiceWithMeta] = Field(None, description='')
    locatedOption: Optional[ChoiceLocatedOption] = Field(None, description='')
    dated: Optional[ChoiceDated] = Field(None, description='')
    dateRef: Annotated[Optional[PaymentDate | BaseReference], PaymentDate.serializer(), PaymentDate.validator(('@key', '@key:external', '@ref', '@ref:external'))] = Field(None, description='')
    keyed: Annotated[Optional[Keyed], Keyed.serializer(), Keyed.validator(('@key', '@key:external'))] = Field(None, description='')
    # In a generated bundle, every attribute has the runtime's serializer and
    # validator, including refOption and withMetaHooked, which have no metadata.
    refOption: Annotated[Optional[ChoiceRefOption], ChoiceRefOption.serializer(), ChoiceRefOption.validator()] = Field(None, description='')
    withMetaHooked: Annotated[Optional[ChoiceWithMeta], ChoiceWithMeta.serializer(), ChoiceWithMeta.validator()] = Field(None, description='')
    choiceRef: Annotated[Optional[ChoiceData | BaseReference], ChoiceData.serializer(), ChoiceData.validator(('@ref:scoped', ))] = Field(None, description='')
    holder: Optional[OneOfHolder] = Field(None, description='')

    _KEY_REF_CONSTRAINTS = {
        'choiceRef': {'@ref:scoped'},
        'dateRef': {'@key', '@key:external', '@ref', '@ref:external'},
        'located': {'@key:scoped'}
    }


def _written(model: BaseDataClass, attribute: str):
    '''what a model writes for one of its attributes'''
    return json.loads(model.rune_serialize(validate_model=False))[attribute]


A_JSON = {'@type': f'{NS}.A', 'fieldA': 'foo'}
B_JSON = {'@type': f'{NS}.B', 'fieldB': 'bar'}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('model, attribute, expected', [
    pytest.param(Root(data=ChoiceData(A=A(fieldA='foo'))),
                 'data', A_JSON, id='data-option'),
    pytest.param(Root(data=ChoiceData(B=B(fieldB='bar'))),
                 'data', B_JSON, id='other-data-option'),
    pytest.param(Root(basic=ChoiceBasic(string='foo')),
                 'basic', {'@type': 'string', '@data': 'foo'}, id='string'),
    pytest.param(Root(basic=ChoiceBasic(int=10)),
                 'basic', {'@type': 'int', '@data': 10}, id='int'),
    pytest.param(Root(nested=ChoiceNested(
        ChoiceData=ChoiceData(A=A(fieldA='foo')))),
                 'nested', A_JSON, id='nested-choice'),
    pytest.param(Root(nested=ChoiceNested(
        ChoiceBasic=ChoiceBasic(string='foo'))),
                 'nested', {'@type': 'string', '@data': 'foo'},
                 id='nested-basic'),
    pytest.param(Root(deep=ChoiceDeep(ChoiceNested=ChoiceNested(
        ChoiceData=ChoiceData(B=B(fieldB='bar'))))),
                 'deep', B_JSON, id='deep-nested-choice'),
    pytest.param(Root(entries=[ChoiceData(A=A(fieldA='foo')),
                               ChoiceData(B=B(fieldB='bar'))]),
                 'entries', [A_JSON, B_JSON], id='list'),
])
def test_write_choice(model, attribute, expected):
    '''the selected option is written with @type, on the choice itself'''
    assert _written(model, attribute) == expected


@pytest.mark.parametrize('attribute, value', [
    pytest.param('located',
                 {'@key:scoped': 'outer', '@data': A_JSON},
                 id='choice-attribute-with-metadata'),
    pytest.param('withMeta',
                 {'@type': f'{NS}.Located', '@key:scoped': 'loc', 'fieldL': 'x'},
                 id='option-with-location'),
    pytest.param('locatedOption',
                 {'@type': f'{NS}.ChoiceData', '@key:scoped': 'inner',
                  '@data': A_JSON},
                 id='nested-choice-option-with-metadata'),
])
def test_round_trip_with_metadata(attribute, value):
    '''metadata is kept, and written where Java writes it'''
    model = Root.rune_deserialize({attribute: value}, validate_model=False)
    assert _written(model, attribute) == value


def test_round_trip_option_is_a_reference():
    '''an option that is a reference is written with @type and the reference'''
    data = {
        'withMeta': {'@type': f'{NS}.Keyed', '@ref:external': 'k1'},
        'keyed': {'@key:external': 'k1', 'fieldK': 'x'},
    }
    model = Root.rune_deserialize(data, validate_model=False)
    assert _written(model, 'withMeta') == data['withMeta']


def test_round_trip_option_is_a_reference_on_field_with_hooks():
    '''a reference beside @type belongs to the option, not to the field'''
    data = {
        'withMetaHooked': {'@type': f'{NS}.Keyed', '@ref:external': 'k1'},
        'keyed': {'@key:external': 'k1', 'fieldK': 'x'},
    }
    model = Root.rune_deserialize(data, validate_model=False)
    assert model.withMetaHooked.Keyed is model.keyed
    assert _written(model, 'withMetaHooked') == data['withMetaHooked']


def test_round_trip_nested_choice_option_is_a_reference():
    '''an option that is a choice, and a reference, keeps its @type'''
    data = {
        'located': {'@key:scoped': 'outer', '@data': A_JSON},
        'refOption': {'@type': f'{NS}.ChoiceData', '@ref:scoped': 'outer'},
    }
    model = Root.rune_deserialize(data, validate_model=False)
    assert model.refOption.ChoiceData is model.located
    assert _written(model, 'refOption') == data['refOption']


def test_reference_without_type_refers_to_the_whole_choice():
    '''a reference with no @type is a reference to the choice itself'''
    data = {
        'located': {'@key:scoped': 'outer', '@data': A_JSON},
        'choiceRef': {'@ref:scoped': 'outer'},
    }
    model = Root.rune_deserialize(data, validate_model=False)
    assert model.choiceRef is model.located
    assert _written(model, 'choiceRef') == data['choiceRef']


def test_write_no_option_selected():
    '''a choice with no option set is written as an empty object'''
    assert _written(Root(data=ChoiceData()), 'data') == {}


def test_write_two_options_selected():
    '''a choice must have exactly one option'''
    model = Root(data=ChoiceData(A=A(fieldA='foo'), B=B(fieldB='bar')))
    with pytest.raises(Exception, match='[Mm]ultiple'):
        model.rune_serialize(validate_model=False)


def test_write_choice_at_root():
    '''a choice at the root is a value of its option'''
    written = json.loads(ChoiceData(A=A(fieldA='foo')).rune_serialize(
        validate_model=False))
    assert written['@type'] == f'{NS}.A'
    assert written['fieldA'] == 'foo'
    assert 'A' not in written


def test_write_type_without_marker_is_not_flattened():
    '''a type with optional attributes keeps its attribute names'''
    assert _written(Root(holder=OneOfHolder(A=A(fieldA='foo'))),
                    'holder') == {'A': {'fieldA': 'foo'}}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_read_data_option():
    '''@type selects the option'''
    model = Root.rune_deserialize({'data': A_JSON})
    assert model.data.A.fieldA == 'foo'
    assert model.data.B is None


def test_read_basic_option():
    '''@type names a basic option, and its value is in @data'''
    model = Root.rune_deserialize(
        {'basic': {'@type': 'string', '@data': 'foo'}})
    assert model.basic.string == 'foo'


def test_read_nested_choice():
    '''@type is found through nested choices'''
    model = Root.rune_deserialize({'nested': A_JSON})
    assert model.nested.ChoiceData.A.fieldA == 'foo'


def test_read_deep_nested_choice():
    '''@type is found through several levels of nested choices'''
    model = Root.rune_deserialize({'deep': B_JSON})
    assert model.deep.ChoiceNested.ChoiceData.B.fieldB == 'bar'


def test_read_list():
    '''each item of a list selects its own option'''
    model = Root.rune_deserialize({'entries': [A_JSON, B_JSON]})
    assert model.entries[0].A.fieldA == 'foo'
    assert model.entries[1].B.fieldB == 'bar'


def test_read_choice_attribute_with_metadata():
    '''metadata beside @data belongs to the attribute'''
    model = Root.rune_deserialize(
        {'located': {'@key:scoped': 'outer', '@data': A_JSON}},
        validate_model=False)
    assert model.located.A.fieldA == 'foo'
    assert model.located.get_meta('@key:scoped') == 'outer'


def test_read_key_inside_choice_is_registered():
    '''a key inside the selected option can be referenced from elsewhere'''
    data = {
        'dated': {'@type': f'{NS}.WithDate',
                  'paymentDate': {'@key:external': 'k1', 'day': 'Monday'}},
        'dateRef': {'@ref:external': 'k1'},
    }
    model = Root.rune_deserialize(data, validate_model=False)
    assert model.dateRef is model.dated.WithDate.paymentDate


def test_python_construction_is_not_affected():
    '''building a choice in Python still names the option'''
    assert ChoiceData(A=A(fieldA='foo')).A.fieldA == 'foo'
    assert ChoiceBasic(string='foo').string == 'foo'


@pytest.mark.parametrize('attribute, value', [
    pytest.param('data', {'@type': f'{NS}.Located', 'fieldL': 'x'},
                 id='not-an-option'),
    pytest.param('nested', {'@type': f'{NS}.Located', 'fieldL': 'x'},
                 id='not-an-option-of-nested-choice'),
    pytest.param('data', {'@type': f'{NS}.ExtA', 'fieldA': 'a', 'fieldExt': 'b'},
                 id='subtype-of-an-option'),
    pytest.param('basic', {'@type': 'date', '@data': '2024-01-01'},
                 id='not-a-basic-option'),
])
@pytest.mark.parametrize('validate', [True, False])
def test_read_rejects_invalid_option(attribute, value, validate):
    '''@type must exactly match a declared option, even with validation off'''
    with pytest.raises(Exception):
        Root.rune_deserialize({attribute: value}, validate_model=validate)


NESTED_FORM = [
    pytest.param('data', {'A': {'fieldA': 'foo'}}, id='data'),
    pytest.param('basic', {'string': 'foo'}, id='basic'),
    pytest.param('nested', {'ChoiceData': {'A': {'fieldA': 'foo'}}}, id='nested'),
]


@pytest.mark.parametrize('attribute, value', NESTED_FORM)
@pytest.mark.parametrize('validate', [True, False])
def test_rune_deserialize_rejects_nested_form(attribute, value, validate):
    '''a choice without @type is an error and is never silently dropped'''
    with pytest.raises(Exception):
        Root.rune_deserialize({attribute: value}, validate_model=validate)


@pytest.mark.parametrize('attribute, value', NESTED_FORM)
def test_model_validate_rejects_nested_form(attribute, value):
    '''the nested form can't get in through model_validate'''
    with pytest.raises(Exception):
        Root.model_validate({attribute: value})


@pytest.mark.parametrize('attribute, value', NESTED_FORM)
def test_model_validate_json_rejects_nested_form(attribute, value):
    '''the nested form can't get in through model_validate_json'''
    with pytest.raises(Exception):
        Root.model_validate_json(json.dumps({attribute: value}))


def test_read_ignores_unknown_field():
    '''an unknown field in the selected option is ignored, as Java does by default'''
    model = Root.rune_deserialize(
        {'data': {'@type': f'{NS}.A', 'fieldA': 'foo', 'notAField': 'baz'}})
    assert model.data.A.fieldA == 'foo'


def test_read_type_without_marker_uses_attribute_names():
    '''a type with optional attributes is read by attribute name'''
    model = Root.rune_deserialize({'holder': {'A': {'fieldA': 'foo'}}})
    assert model.holder.A.fieldA == 'foo'
