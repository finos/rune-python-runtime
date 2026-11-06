# pylint: disable=invalid-name
'''testing basic conditions compliance'''
import datetime
import pytest
from pydantic import Field, ValidationError
from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.conditions import ConditionViolationError, rune_condition


class cdm_base_datetime_DateList(BaseDataClass):
    """
    List of dates.
    """
    _FQRTN = 'cdm.base.datetime.DateList'
    date: list[datetime.date] = Field(..., description='', min_length=1)


def test_min_list_length_all_defaults():
    '''no doc'''
    with pytest.raises(ValidationError):
        cdm_base_datetime_DateList()


def test_min_list_length_empty_list():
    '''no doc'''
    with pytest.raises(ValidationError):
        cdm_base_datetime_DateList(date=[])


class _PositiveVal(BaseDataClass):
    '''Model with a condition that fails when val <= 0.'''
    val: int = Field(...)

    @rune_condition
    def c_Positive(self):
        '''val must be positive.'''
        return self.val > 0


def test_rune_serialize_propagates_condition_violation_without_key_error():
    '''rune_serialize must raise ConditionViolationError, not KeyError.'''
    model = _PositiveVal(val=-1)
    with pytest.raises(ConditionViolationError):
        model.rune_serialize()


def test_rune_serialize_propagates_validation_error_without_key_error():
    '''rune_serialize must raise ValidationError when attribute validation fails.'''
    model = _PositiveVal.model_construct(val='not-an-int')
    with pytest.raises(ValidationError):
        model.rune_serialize()

# EOF
