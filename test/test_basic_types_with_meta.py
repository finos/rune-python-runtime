# Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

'''test module for the annotated base rune types'''
import json
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Annotated

import pytest
from pydantic import Field, ValidationError

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.metadata import (
    DateTimeWithMeta,
    DateWithMeta,
    EnumWithMetaMixin,
    NumberWithMeta,
    StrWithMeta,
    TimeWithMeta,
    _EnumWrapper,
)


class AnnotatedStringModel(BaseDataClass):
    '''string test class'''
    currency: Annotated[StrWithMeta,
                        StrWithMeta.serializer(),
                        StrWithMeta.validator(('@scheme',))
    ] = Field(..., description="Test currency")


class AnnotatedNumberModel(BaseDataClass):
    '''number test class'''
    amount: Annotated[NumberWithMeta,
                      NumberWithMeta.serializer(),
                      NumberWithMeta.validator(('@scheme',))
    ] = Field(..., description="Test amount", decimal_places=3)


class AnnotatedDateModel(BaseDataClass):
    '''date test class'''
    date: Annotated[DateWithMeta,
                    DateWithMeta.serializer(),
                    DateWithMeta.validator(('@scheme',))
    ] = Field(..., description="Test date")


class OptionalAnnotatedDateModel(BaseDataClass):
    '''Optional date with the validator wrapping the nullable type.'''
    date: Annotated[DateWithMeta | None,
                    DateWithMeta.serializer(),
                    DateWithMeta.validator(('@scheme',))
    ] = Field(None, description="Optional test date", validate_default=True)


class AnnotatedDateTimeModel(BaseDataClass):
    '''datetime test class'''
    datetime: Annotated[DateTimeWithMeta,
                        DateTimeWithMeta.serializer(),
                        DateTimeWithMeta.validator(('@scheme',))
    ] = Field(..., description="Test datetime")


class AnnotatedTimeModel(BaseDataClass):
    '''datetime test class'''
    time: Annotated[TimeWithMeta,
                    TimeWithMeta.serializer(),
                    TimeWithMeta.validator(('@scheme',))
    ] = Field(..., description="Test time")


class StrWithMetaModel(BaseDataClass):
    '''generic meta support test case'''
    currency: Annotated[StrWithMeta,
                        StrWithMeta.serializer(),
                        StrWithMeta.validator(('@scheme', '@key'))
    ] = Field(..., description="Test currency")


class StrWithMetaAndConstraintsModel(BaseDataClass):
    '''meta and string constraints test case'''
    currency: Annotated[StrWithMeta,
                        StrWithMeta.serializer(),
                        StrWithMeta.validator(('@scheme', '@key'))
    ] = Field(..., description="Test currency",
              min_length=3, max_length=5, pattern=r'^[A-Z]*$')


class ConstrainedNumberModel(BaseDataClass):
    '''number test class'''
    amount: Annotated[NumberWithMeta,
                      NumberWithMeta.serializer(),
                      NumberWithMeta.validator(('@scheme',))
    ] = Field(..., description="Test amount", decimal_places=3, ge=0)


class EnumType(EnumWithMetaMixin, Enum):
    '''test rune enum'''
    A = "A"
    B = "B"
    C = "C"


class AnnotatedEnumModel(BaseDataClass):
    '''string test class'''
    enum_t: Annotated[EnumType,
                      EnumType.serializer(),
                      EnumType.validator(
                          ('@scheme', ))] = Field(..., description='')


class OptionalAnnotatedEnumModel(BaseDataClass):
    '''Optional enum with metadata around the entire nullable type.'''
    enum_t: Annotated[EnumType | None,
                      EnumType.serializer(),
                      EnumType.validator(('@scheme',))] = Field(
                          None, validate_default=True)


@pytest.fixture(params=[
    pytest.param(lambda cls, data: cls(**data), id='constructor'),
    pytest.param(lambda cls, data: cls.model_validate(data), id='model_validate'),
    pytest.param(lambda cls, data: cls.model_validate_json(json.dumps(data)),
                 id='model_validate_json'),
    pytest.param(lambda cls, data: cls.rune_deserialize(data),
                 id='rune_deserialize'),
])
def load_enum_model(request):
    '''Exercise the public construction and deserialization entry points.'''
    return request.param


def test_dump_annotated_string_simple():
    '''test the annotated string'''
    model = AnnotatedStringModel(currency='EUR')
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"currency":{"@data":"EUR"}}', 'explicit string failed'

    model = AnnotatedStringModel(currency=StrWithMeta('EUR'))
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"currency":{"@data":"EUR"}}', 'annotated string failed'


def test_annotated_string_with_forbidden_meta():
    '''test exception when extra meta is passed'''
    with pytest.raises(ValidationError):
        AnnotatedStringModel(currency=StrWithMeta(
            'EUR', scheme='http://fpml.org', key='currency-1'))


def test_dump_annotated_string_scheme():
    '''test the scheme treatment'''
    model = AnnotatedStringModel(
        currency=StrWithMeta('EUR', scheme='http://fpml.org'))
    json_str = model.model_dump_json(exclude_unset=True)
    assert (
        json_str == '{"currency":{"@scheme":"http://fpml.org","@data":"EUR"}}')


def test_load_annotated_string_simple():
    '''test the loading of annotated strings'''
    simple_json = '{"currency":{"@data":"EUR"}}'
    model = AnnotatedStringModel.model_validate_json(simple_json)
    assert model.currency == 'EUR', 'currency differs'
    assert model.currency.get_meta('@scheme') is None, 'scheme is not None'


def test_load_annotated_string_scheme():
    '''test the loading of annotated with a scheme strings'''
    scheme_json = '{"currency":{"@data":"EUR","@scheme":"http://fpml.org"}}'
    model = AnnotatedStringModel.model_validate_json(scheme_json)
    assert model.currency == 'EUR', 'currency differs'
    assert model.currency.get_meta('@scheme') == 'http://fpml.org'


def test_dump_annotated_number_simple():
    '''test the annotated string'''
    model = AnnotatedNumberModel(amount=10)
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"amount":{"@data":"10"}}', 'explicit int failed'

    model = AnnotatedNumberModel(amount="10.344")
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"amount":{"@data":"10.344"}}', 'explicit string failed'

    model = AnnotatedNumberModel(amount=NumberWithMeta(10))
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"amount":{"@data":"10"}}', 'annotated number failed'


def test_dump_annotated_number_scheme():
    '''test the annotated string'''
    model = AnnotatedNumberModel(
        amount=NumberWithMeta(10, scheme='http://fpml.org'))
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"amount":{"@scheme":"http://fpml.org","@data":"10"}}'


def test_load_annotated_number():
    '''test the loading of annotated with a scheme strings'''
    scheme_json = '{"amount":{"@data":"10"}}'
    model = AnnotatedNumberModel.model_validate_json(scheme_json)
    assert model.amount == 10, 'string amount differs'

    scheme_json = '{"amount":{"@data":10}}'
    model = AnnotatedNumberModel.model_validate_json(scheme_json)
    assert model.amount == 10, 'int amount differs'

    scheme_json = '{"amount":{"@data":"10.3"}}'
    model = AnnotatedNumberModel.model_validate_json(scheme_json)
    assert model.amount == Decimal("10.3"), 'float amount differs'


def test_load_annotated_number_scheme():
    '''test the loading of annotated with a scheme strings'''
    scheme_json = '{"amount":{"@data":"10","@scheme":"http://fpml.org"}}'
    model = AnnotatedNumberModel.model_validate_json(scheme_json)
    assert model.amount == 10, 'amount differs'
    assert model.amount.get_meta('@scheme') == 'http://fpml.org'


def test_fail_load_annotated_number():
    '''test the loading of annotated with a scheme strings'''
    scheme_json = '{"amount":{"@data":"10.1234","@scheme":"http://fpml.org"}}'
    with pytest.raises(ValidationError):
        AnnotatedNumberModel.model_validate_json(scheme_json)


def test_fail_create_annotated_number():
    '''test the loading of annotated with a scheme strings'''
    with pytest.raises(ValidationError):
        AnnotatedNumberModel(amount=NumberWithMeta("1.1234"))


def test_fail_create_annotated_number_():
    '''test the loading of annotated with a scheme strings'''
    with pytest.raises(ValidationError):
        AnnotatedNumberModel(amount="10.1234")


def test_dump_annotated_date_simple():
    '''test the annotated string'''
    model = AnnotatedDateModel(date="2024-10-10")
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"date":{"@data":"2024-10-10"}}'

    model = AnnotatedDateModel(date=DateWithMeta("2024-10-10"))
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"date":{"@data":"2024-10-10"}}'


def test_dump_annotated_date_date():
    '''test the annotated string'''
    model = AnnotatedDateModel(date=date(2024, 10, 10))
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"date":{"@data":"2024-10-10"}}'

    model = AnnotatedDateModel(date=DateWithMeta(date(2024, 10, 10)))
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"date":{"@data":"2024-10-10"}}'


@pytest.mark.parametrize('data', [{}, {'date': None}])
def test_create_optional_annotated_date_with_none(data):
    '''Accept omitted and explicit None values during construction.'''
    model = OptionalAnnotatedDateModel(**data)
    assert model.date is None


@pytest.mark.parametrize('data', [{}, {'date': None}])
@pytest.mark.parametrize('method', ['model_validate', 'rune_deserialize'])
def test_deserialize_optional_annotated_date_with_none(data, method):
    '''Accept absent optional dates through the public dictionary APIs.'''
    model = getattr(OptionalAnnotatedDateModel, method)(data)
    assert model.date is None


@pytest.mark.parametrize('json_str', ['{}', '{"date":null}'])
def test_load_optional_annotated_date_with_none(json_str):
    '''Accept omitted dates and JSON null with default validation enabled.'''
    model = OptionalAnnotatedDateModel.model_validate_json(json_str)
    assert model.date is None


@pytest.mark.parametrize('options, expected', [
    ({}, {'date': None}),
    ({'exclude_none': True}, {}),
    ({'exclude_defaults': True}, {}),
])
def test_optional_annotated_date_serializes_none(options, expected):
    '''Null basic values honor the public dump options.'''
    model = OptionalAnnotatedDateModel(date=None)

    assert model.model_dump(**options) == expected
    assert json.loads(model.model_dump_json(**options)) == expected
    written = json.loads(model.rune_serialize(**({'exclude_defaults': False} | options)))
    assert ('date' in written) == ('date' in expected)


@pytest.mark.parametrize('value', [
    '2024-10-10',
    {'@data': '2024-10-10', '@scheme': 'http://fpml.org'},
])
def test_create_optional_annotated_date_with_value(value):
    '''Continue to convert date values and retain their metadata.'''
    model = OptionalAnnotatedDateModel(date=value)
    assert isinstance(model.date, DateWithMeta)
    assert model.date == date(2024, 10, 10)
    expected_scheme = 'http://fpml.org' if isinstance(value, dict) else None
    assert model.date.get_meta('scheme') == expected_scheme


def test_create_required_annotated_date_with_none():
    '''Reject None for a required date with a Pydantic validation error.'''
    with pytest.raises(ValidationError):
        AnnotatedDateModel(date=None)


@pytest.mark.parametrize('method', ['model_validate', 'rune_deserialize'])
def test_deserialize_required_annotated_date_with_none(method):
    '''Dictionary APIs must still reject None for a required date.'''
    with pytest.raises(ValidationError):
        getattr(AnnotatedDateModel, method)({'date': None})


def test_load_required_annotated_date_with_none():
    '''JSON null must still be invalid for a required date.'''
    with pytest.raises(ValidationError):
        AnnotatedDateModel.model_validate_json('{"date":null}')


def test_annotated_date_fail():
    '''test instantiation failure with an incorrect type'''
    with pytest.raises(AttributeError):
        AnnotatedDateModel(date=10)


def test_date_with_meta_fail():
    '''test instantiation failure with an incorrect type'''
    with pytest.raises(TypeError):
        DateWithMeta(10)


def test_load_annotated_date_scheme():
    '''test the loading of annotated with a scheme strings'''
    scheme_json = '{"date":{"@data":"2024-10-10","@scheme":"http://fpml.org"}}'
    model = AnnotatedDateModel.model_validate_json(scheme_json)
    assert model.date == date(2024, 10, 10), 'date differs'
    assert model.date.get_meta('scheme') == 'http://fpml.org', 'scheme differs'


def test_dump_annotated_datetime_simple():
    '''test the annotated string'''
    model = AnnotatedDateTimeModel(datetime="2024-10-10T01:01:01")
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"datetime":{"@data":"2024-10-10T01:01:01"}}'

    model = AnnotatedDateTimeModel(
        datetime=DateTimeWithMeta("2024-10-10T01:01:01"))
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"datetime":{"@data":"2024-10-10T01:01:01"}}'


def test_load_annotated_datetime_scheme():
    '''test the loading of annotated with a scheme strings'''
    scheme_json = ('{"datetime":{"@data":"2024-10-10T01:01:01",'
                   '"@scheme":"http://fpml.org"}}')
    model = AnnotatedDateTimeModel.model_validate_json(scheme_json)
    assert model.datetime == datetime(2024, 10, 10, 1, 1, 1), 'datetime differs'
    assert model.datetime.get_meta('scheme') == 'http://fpml.org'


def test_dump_annotated_time_simple():
    '''test the annotated string'''
    model = AnnotatedTimeModel(time="01:01:01.000087")
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"time":{"@data":"01:01:01.000087"}}'

    model = AnnotatedTimeModel(time=TimeWithMeta("01:01:01"))
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str == '{"time":{"@data":"01:01:01"}}'


def test_load_annotated_time_scheme():
    '''test the loading of annotated with a scheme strings'''
    scheme_json = (
        '{"time":{"@data":"01:01:01.000087","@scheme":"http://fpml.org"}}')
    model = AnnotatedTimeModel.model_validate_json(scheme_json)
    assert model.time == time(1, 1, 1, 87), 'time differs'
    assert model.time.get_meta('scheme') == 'http://fpml.org', 'scheme differs'


def test_generic_string_with_meta():
    '''generic meta support'''
    model = StrWithMetaModel(currency='EUR')
    assert model.currency == 'EUR'

    model = StrWithMetaModel(currency=StrWithMeta(
        'EUR', scheme='http://fpml.org', key='currency-1'))
    assert model.currency == 'EUR'
    # pylint: disable=no-member
    assert model.currency.get_meta('@scheme') == 'http://fpml.org'
    assert model.currency.get_meta('@key') == 'currency-1'

    json_str = model.model_dump_json(exclude_unset=True)
    j_dict = json.loads(json_str)
    assert j_dict['currency']['@data'] == 'EUR'
    assert j_dict['currency']['@scheme'] == 'http://fpml.org'
    assert j_dict['currency']['@key'] == 'currency-1'


def test_load_generic_string_with_meta():
    '''load json with generic meta support'''
    simple_json = (
        '{"currency":{"@data":"EUR","@scheme":"http://fpml.org",'
        '"@key":"currency-1"}}'
    )
    model = StrWithMetaModel.model_validate_json(simple_json)
    assert model.currency == 'EUR'
    assert model.currency.get_meta('@scheme') == 'http://fpml.org'
    assert model.currency.get_meta('@key') == 'currency-1'


def test_generic_string_with_forbidden_meta():
    '''test exception when extra meta is passed'''
    with pytest.raises(ValidationError):
        StrWithMetaModel(currency=StrWithMeta(
            'EUR', scheme='http://fpml.org', key='currency-1', ref='blah'))


def test_create_constrained_str_model():
    '''test the creation of the constrained str model'''
    model = StrWithMetaAndConstraintsModel(currency='EUR')
    assert model.currency == 'EUR'


def test_fail_min_create_constrained_str_model():
    '''test the creation of the constrained str model'''
    with pytest.raises(ValidationError):
        StrWithMetaAndConstraintsModel(currency='EU')


def test_fail_max_create_constrained_str_model():
    '''test the creation of the constrained str model'''
    with pytest.raises(ValidationError):
        StrWithMetaAndConstraintsModel(currency='EUROOO')


def test_fail_pattern_create_constrained_str_model():
    '''test the creation of the constrained str model'''
    with pytest.raises(ValidationError):
        StrWithMetaAndConstraintsModel(currency='EUR1')


def test_fail_create_constrained_num_model():
    '''test the creation of the constrained str model'''
    with pytest.raises(ValidationError):
        ConstrainedNumberModel(amount=NumberWithMeta(-1))


def test_annotated_enum_assignment():
    '''test assignment'''
    model = AnnotatedEnumModel(enum_t=EnumType.B)
    assert model.enum_t == EnumType.B
    assert isinstance(model.enum_t, _EnumWrapper)

    model.enum_t = EnumType.C
    assert model.enum_t != EnumType.B
    assert isinstance(model.enum_t, _EnumWrapper)


@pytest.mark.parametrize('data', [{}, {'enum_t': None}])
def test_optional_annotated_enum_accepts_none(load_enum_model, data):
    '''Accept omitted values and explicit None, including validated defaults.'''
    model = load_enum_model(OptionalAnnotatedEnumModel, data)
    assert model.enum_t is None


@pytest.mark.parametrize('data', [{}, {'enum_t': None}])
def test_required_annotated_enum_rejects_none(load_enum_model, data):
    '''Missing and null non-nullable enums must raise ValidationError.'''
    with pytest.raises(ValidationError) as error:
        load_enum_model(AnnotatedEnumModel, data)
    assert error.value.errors()[0]['loc'] == ('enum_t',)


@pytest.mark.parametrize('value', [
    'A',
    {'@data': 'A', '@scheme': 'http://fpml.org'},
])
def test_optional_annotated_enum_retains_value_and_metadata(load_enum_model, value):
    '''Continue to wrap enum values and round-trip their metadata.'''
    model = load_enum_model(OptionalAnnotatedEnumModel, {'enum_t': value})
    assert isinstance(model.enum_t, _EnumWrapper)
    assert model.enum_t.enum_instance is EnumType.A
    expected_scheme = value.get('@scheme') if isinstance(value, dict) else None
    assert model.enum_t.get_meta('scheme') == expected_scheme

    expected = value if isinstance(value, dict) else {'@data': value}
    serialized = model.model_dump_json()
    assert json.loads(serialized) == {'enum_t': expected}
    restored = OptionalAnnotatedEnumModel.model_validate_json(serialized)
    assert restored.enum_t.enum_instance is EnumType.A
    assert restored.enum_t.get_meta('scheme') == expected_scheme


def test_optional_annotated_enum_accepts_enum_instance():
    '''Keep accepting enum instances as well as strings and metadata objects.'''
    model = OptionalAnnotatedEnumModel(enum_t=EnumType.A)
    assert isinstance(model.enum_t, _EnumWrapper)
    assert model.enum_t.enum_instance is EnumType.A


@pytest.mark.parametrize('options, expected', [
    ({}, {'enum_t': None}),
    ({'exclude_none': True}, {}),
    ({'exclude_defaults': True}, {}),
])
def test_optional_annotated_enum_serializes_none(options, expected):
    '''Serialize None as null or omit it according to the dump options.'''
    model = OptionalAnnotatedEnumModel(enum_t=None)
    assert model.model_dump(**options) == expected
    assert json.loads(model.model_dump_json(**options)) == expected


@pytest.mark.parametrize('options, include_enum', [
    ({}, False),
    ({'exclude_defaults': False}, True),
    ({'exclude_defaults': False, 'exclude_none': True}, False),
])
def test_optional_annotated_enum_rune_serializes_none(options, include_enum):
    '''Rune serialization must honor null handling while adding root metadata.'''
    model = OptionalAnnotatedEnumModel(enum_t=None)
    serialized = json.loads(model.rune_serialize(**options))
    assert '@type' in serialized
    if include_enum:
        assert serialized['enum_t'] is None
    else:
        assert 'enum_t' not in serialized

# EOF
