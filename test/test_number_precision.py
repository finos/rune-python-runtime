# Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

'''JSON numbers are read by rune_deserialize without losing precision.

A number with a fraction or exponent must not pass through a binary float on
its way to a Decimal, which would change values beyond about 15 to 17
significant digits without an error.
'''
from decimal import Decimal
import json
from typing import Annotated, Optional

import pytest
from pydantic import ConfigDict, Field

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.metadata import NumberWithMeta

# The classes below are what rune-python-generator produces from this model:
#
#   type PlainNumber:
#       amount number (0..1)
#
#   type NumberList:
#       amounts number (0..*)
#
#   type NumberWithScheme:
#       amount number (1..1)
#           [metadata scheme]
#
#   type IntAttribute:
#       total int (0..1)

_LONG_PRECISION = [
    '12345678901234567.89',
    '0.1000000000000000000001',
    '1234567890.123456789012345678',
]


class PlainNumber(BaseDataClass):
    '''number attribute'''
    model_config = ConfigDict(defer_build=True)
    amount: Optional[Decimal] = Field(None, description='')


class NumberList(BaseDataClass):
    '''multi-cardinality number attribute'''
    model_config = ConfigDict(defer_build=True)
    amounts: Optional[list[Decimal | None]] = Field(None, description='')


class NumberWithScheme(BaseDataClass):
    '''number attribute with metadata'''
    model_config = ConfigDict(defer_build=True)
    amount: Annotated[NumberWithMeta, NumberWithMeta.serializer(), NumberWithMeta.validator(('@scheme', ))] = Field(..., description='')


class IntAttribute(BaseDataClass):
    '''int attribute'''
    model_config = ConfigDict(defer_build=True)
    total: Optional[int] = Field(None, description='')


def _read(cls, document: str):
    return cls.rune_deserialize(document, validate_model=False)


@pytest.mark.parametrize('text', _LONG_PRECISION)
def test_json_number_keeps_every_digit(text):
    '''An unquoted number reads exactly.'''
    assert _read(PlainNumber, f'{{"amount": {text}}}').amount == Decimal(text)


@pytest.mark.parametrize('text', _LONG_PRECISION)
def test_quoted_number_keeps_every_digit(text):
    '''A quoted number reads exactly.'''
    assert _read(PlainNumber, f'{{"amount": "{text}"}}').amount == Decimal(text)


def test_list_of_json_numbers_keeps_every_digit():
    '''Each number in a list reads exactly.'''
    document = f'{{"amounts": [{", ".join(_LONG_PRECISION)}]}}'
    assert _read(NumberList, document).amounts == [Decimal(t) for t in _LONG_PRECISION]


@pytest.mark.parametrize('text', _LONG_PRECISION)
def test_number_with_metadata_keeps_every_digit(text):
    '''The @data of a number with metadata reads exactly.'''
    model = _read(NumberWithScheme, f'{{"amount": {{"@data": {text}, "@scheme": "s"}}}}')
    assert Decimal(model.amount) == Decimal(text)
    assert model.amount.get_meta('@scheme') == 's'


def test_trailing_zero_is_kept():
    '''The digits of the JSON number are kept as written.'''
    assert str(_read(PlainNumber, '{"amount": 50000.10}').amount) == '50000.10'


def test_exponent_form_is_exact():
    '''A number in exponent form reads exactly.'''
    model = _read(PlainNumber, '{"amount": 1.2345678901234567890123e5}')
    assert model.amount == Decimal('123456.78901234567890123')


def test_large_whole_number_keeps_every_digit():
    '''Whole numbers are not rounded.'''
    text = '123456789012345678901234567890'
    assert _read(PlainNumber, f'{{"amount": {text}}}').amount == Decimal(text)


def test_long_precision_survives_writing():
    '''The exact value read is the value written.'''
    text = _LONG_PRECISION[0]
    written = _read(PlainNumber, f'{{"amount": {text}}}').rune_serialize(validate_model=False)
    assert Decimal(json.loads(written).get('amount')) == Decimal(text)


def test_int_attribute_accepts_whole_number_with_fraction():
    '''A whole value such as 10.0 is still accepted by an int attribute.'''
    assert _read(IntAttribute, '{"total": 10.0}').total == 10
