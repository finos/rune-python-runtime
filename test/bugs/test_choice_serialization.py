# Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for choice validation during Rune serialization."""
import json
from typing import ClassVar

import pytest
from pydantic import ValidationError

from rune.runtime.base_data_class import BaseDataClass, ROOT_CONTAINER
from rune.runtime.conditions import ConditionViolationError, rune_condition
from rune.runtime.utils import rune_check_one_of


class _ChoiceA(BaseDataClass):
    value: int


class _ChoiceB(BaseDataClass):
    value: int


class _ChoiceSerializationTest(BaseDataClass):
    _FQRTN: ClassVar[str] = 'rosetta_dsl.test.choice_serialization.ChoiceSerializatonTest'
    A: _ChoiceA | None = None
    B: _ChoiceB | None = None

    @rune_condition
    def condition_0_Choice(self) -> bool:
        return rune_check_one_of(self, 'A', 'B', necessity=True)


@pytest.fixture(params=[
    pytest.param({}, id='no-options'),
    pytest.param({'A': {'value': 1}, 'B': {'value': 2}}, id='both-options'),
])
def invalid_choice_data(request):
    return request.param


def test_invalid_choice_serialization_raises_condition_violation(invalid_choice_data):
    model = _ChoiceSerializationTest(**invalid_choice_data)

    with pytest.raises(ConditionViolationError, match='condition_0_Choice'):
        model.rune_serialize()

    assert ROOT_CONTAINER not in model.__dict__


def test_invalid_choice_deserialization_raises_condition_violation(invalid_choice_data):
    with pytest.raises(ConditionViolationError, match='condition_0_Choice'):
        _ChoiceSerializationTest.rune_deserialize(invalid_choice_data)


@pytest.mark.parametrize('data', [
    pytest.param({'A': {'value': 1}}, id='A'),
    pytest.param({'B': {'value': 2}}, id='B'),
])
def test_valid_choice_serialization(data):
    model = _ChoiceSerializationTest(**data)

    assert json.loads(model.rune_serialize()) == {
        '@type': 'rosetta_dsl.test.choice_serialization.ChoiceSerializatonTest',
        '@model': 'rosetta_dsl',
        '@version': '0.0.0',
        **data,
    }
    assert ROOT_CONTAINER not in model.__dict__
    assert model.model_dump(exclude_unset=True) == data


@pytest.mark.parametrize('options', [
    {'validate_model': False},
    {'check_rune_constraints': False},
    {'raise_validation_errors': False},
])
def test_invalid_choice_serialization_respects_validation_options(invalid_choice_data, options):
    model = _ChoiceSerializationTest(**invalid_choice_data)

    serialized = json.loads(model.rune_serialize(**options))

    assert {key: value for key, value in serialized.items() if not key.startswith('@')} == invalid_choice_data
    assert ROOT_CONTAINER not in model.__dict__


def test_serialization_preserves_attribute_validation_error():
    model = _ChoiceA(value=1)
    model.value = 'invalid'

    with pytest.raises(ValidationError, match='value'):
        model.rune_serialize()

    assert ROOT_CONTAINER not in model.__dict__
