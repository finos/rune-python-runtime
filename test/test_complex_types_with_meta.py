# Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

'''complex types tests'''
from decimal import Decimal
import json
from typing_extensions import Annotated
import pytest
from pydantic import Field, ValidationError

from rune.runtime.base_data_class import BaseDataClass


class CashFlow(BaseDataClass):
    '''test cashflow'''
    currency: str = Field(...,
                          description='currency',
                          min_length=3,
                          max_length=3)
    amount: Decimal = Field(..., description='payment amount', ge=0)


class DummyLoan(BaseDataClass):
    '''some more complex data structure'''
    loan: CashFlow = Field(..., description='loaned amount')
    repayment: CashFlow = Field(..., description='repaid amount')


class DummyLoan2(BaseDataClass):
    '''some more complex data structure'''
    loan: Annotated[CashFlow,
                    CashFlow.serializer(),
                    CashFlow.validator(allowed_meta=('@key', '@ref'))] = Field(
                        ..., description='loaned amount')
    repayment: Annotated[CashFlow,
                         CashFlow.serializer(),
                         CashFlow.validator(
                             allowed_meta=('@key', '@ref'))] = Field(
                                 ..., description='repaid amount')


class OptionalLoan(BaseDataClass):
    '''Optional complex type with the validator outside the nullable type.'''
    loan: Annotated[CashFlow | None,
                    CashFlow.serializer(),
                    CashFlow.validator(('@key',))] = Field(
                        None, validate_default=True)


class OptionalLoanBook(BaseDataClass):
    '''Optional list with a validator around the entire field type.'''
    loans: Annotated[list[CashFlow] | None,
                     CashFlow.serializer(),
                     CashFlow.validator()] = Field(None, validate_default=True)


@pytest.fixture(params=[
    pytest.param(lambda cls, data: cls(**data), id='constructor'),
    pytest.param(lambda cls, data: cls.model_validate(data), id='model_validate'),
    pytest.param(lambda cls, data: cls.model_validate_json(json.dumps(data)),
                 id='model_validate_json'),
    pytest.param(lambda cls, data: cls.rune_deserialize(data),
                 id='rune_deserialize'),
])
def load_model(request):
    '''Exercise the public construction and deserialization entry points.'''
    return request.param


@pytest.mark.parametrize('data', [{}, {'loan': None}])
def test_optional_complex_type_accepts_none(load_model, data):
    '''Accept None and validated defaults for nullable complex fields.'''
    model = load_model(OptionalLoan, data)
    assert model.loan is None


@pytest.mark.parametrize('data', [{}, {'loan': None}])
def test_required_complex_type_rejects_none(load_model, data):
    '''Required complex fields must reject both missing values and None.'''
    data = {'repayment': {'currency': 'EUR', 'amount': '101'}} | data
    with pytest.raises(ValidationError) as error:
        load_model(DummyLoan2, data)
    assert error.value.errors()[0]['loc'] == ('loan',)


def test_optional_complex_type_retains_metadata(load_model):
    '''Non-null complex values must retain their values and metadata.'''
    model = load_model(OptionalLoan, {
        'loan': {'@key': 'loan-1', 'currency': 'EUR', 'amount': '100'},
    })
    assert isinstance(model.loan, CashFlow)
    assert model.loan.currency == 'EUR'
    assert model.loan.amount == Decimal('100')
    assert model.loan.get_meta('key') == 'loan-1'


@pytest.mark.parametrize('data', [{}, {'loans': None}])
def test_optional_complex_list_accepts_none(load_model, data):
    '''Accept None for an optional list without accepting nullable elements.'''
    model = load_model(OptionalLoanBook, data)
    assert model.loans is None


@pytest.mark.parametrize('values', [
    [],
    [{'currency': 'EUR', 'amount': '100'}],
])
def test_optional_complex_list_accepts_values(load_model, values):
    '''Continue to deserialize empty and populated lists of complex types.'''
    model = load_model(OptionalLoanBook, {'loans': values})
    assert isinstance(model.loans, list)
    assert len(model.loans) == len(values)
    for loan in model.loans:
        assert isinstance(loan, CashFlow)
        assert loan.currency == 'EUR'
        assert loan.amount == Decimal('100')


def test_complex_list_rejects_none_elements(load_model):
    '''A nullable list of non-nullable objects must reject None elements.'''
    with pytest.raises(ValidationError):
        load_model(OptionalLoanBook, {'loans': [None]})


def test_complex_type_preserves_subtype_identity():
    '''The adapter must not revalidate and replace existing subtype objects.'''
    class DetailedCashFlow(CashFlow):
        '''Cash flow with an additional subtype field.'''
        detail: str

    loan = DetailedCashFlow(currency='EUR', amount=100, detail='subtype value')
    model = OptionalLoan(loan=loan)
    assert model.loan is loan
    assert model.loan.detail == 'subtype value'
    assert model.loan.get_rune_parent() is model


def test_create_loan_no_meta():
    '''tests the creation of a simple model - no meta'''
    model = DummyLoan(loan=CashFlow(currency='EUR', amount=100),
                      repayment=CashFlow(currency='EUR', amount=101))
    json_str = model.model_dump_json(exclude_unset=True)
    assert json_str


def test_create_loan_no_meta_exc():
    '''tests the creation of a simple model - no meta'''
    model = DummyLoan(loan=CashFlow(currency='EUR', amount=100),
                      repayment=CashFlow(currency='EUR', amount=101))
    with pytest.raises(ValueError):
        model.loan.set_meta(key='cf-1-1')  # pylint: disable=no-member


def test_create_loan_with_meta():
    '''tests the creation of a simple model with some meta'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    model.loan.set_meta(key='cf-1-1')  # pylint: disable=no-member
    assert model.loan.get_meta('key') == 'cf-1-1'  # pylint: disable=no-member


def test_load_loan_with_meta():
    '''test load a simple model with json with some meta'''
    json_str = ('{"loan":{"@key":"cf-1-1","currency":"EUR","amount":"100"},'
                '"repayment":{"currency":"EUR","amount":"101"}}')
    model = DummyLoan2.model_validate_json(json_str)
    assert model.loan.get_meta('key') == 'cf-1-1'

# EOF
