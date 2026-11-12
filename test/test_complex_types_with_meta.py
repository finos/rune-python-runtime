# Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

'''complex types tests'''
from collections.abc import Callable
from decimal import Decimal
import json
from typing_extensions import Annotated
import pytest
from pydantic import ConfigDict, Field, ValidationError

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.metadata import BaseReference, Reference
from rune.runtime.object_builder import ObjectBuilder


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


class KeyedEntity(BaseDataClass):
    '''Key-bearing value from the metadata reuse reproduction in issue #15.'''
    model_config = ConfigDict(defer_build=True)
    _ALLOWED_METADATA = {'@key', '@key:external'}
    number: int


class ReferenceEntityHolder(BaseDataClass):
    '''A field accepting an inline entity or an explicit reference.'''
    model_config = ConfigDict(defer_build=True)
    bar: Annotated[
        KeyedEntity | BaseReference,
        KeyedEntity.serializer(),
        KeyedEntity.validator(('@key', '@key:external', '@ref', '@ref:external')),
    ]
    _KEY_REF_CONSTRAINTS = {'bar': {'@ref', '@ref:external'}}


class PlainEntityHolder(BaseDataClass):
    '''A metadata-validated field that does not allow references.'''
    model_config = ConfigDict(defer_build=True)
    bar: Annotated[
        KeyedEntity,
        KeyedEntity.serializer(),
        KeyedEntity.validator(('@key', '@key:external')),
    ]


class PlainEntityList(BaseDataClass):
    '''A list with the same metadata restrictions as the plain scalar field.'''
    model_config = ConfigDict(defer_build=True)
    bars: Annotated[
        list[KeyedEntity],
        KeyedEntity.serializer(),
        KeyedEntity.validator(('@key', '@key:external')),
    ]


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


@pytest.mark.parametrize('model_type, field_name', [
    (OptionalLoan, 'loan'),
    (OptionalLoanBook, 'loans'),
])
@pytest.mark.parametrize('options, include_null', [
    ({}, True),
    ({'exclude_none': True}, False),
    ({'exclude_defaults': True}, False),
])
def test_optional_complex_values_serialize_none(model_type, field_name,
                                               options, include_null):
    '''Null complex values and lists honor the public dump options.'''
    model = model_type(**{field_name: None})
    expected = {field_name: None} if include_null else {}

    assert model.model_dump(**options) == expected
    assert json.loads(model.model_dump_json(**options)) == expected
    written = json.loads(model.rune_serialize(**({'exclude_defaults': False} | options)))
    assert (field_name in written) == include_null


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


def _assign_and_validate_entity(entity: KeyedEntity) -> KeyedEntity:
    holder = PlainEntityHolder(bar=KeyedEntity(number=2))
    holder.bar = entity
    holder.validate_attribs()
    return holder.bar


@pytest.mark.parametrize('load', [
    pytest.param(lambda entity: PlainEntityHolder(bar=entity).bar,
                 id='constructor'),
    pytest.param(lambda entity: PlainEntityHolder.model_validate({'bar': entity}).bar,
                 id='model_validate'),
    pytest.param(lambda entity: ObjectBuilder(PlainEntityHolder, bar=entity).to_model().bar,
                 id='object_builder'),
    pytest.param(lambda entity: PlainEntityList(bars=[entity]).bars[0],
                 id='list'),
    pytest.param(_assign_and_validate_entity, id='assignment_then_validation'),
])
def test_reuse_entity_after_reference_capable_field(
    load: Callable[[KeyedEntity], KeyedEntity],
) -> None:
    '''Empty reference slots must not invalidate reuse in a plain field (#15).'''
    entity = KeyedEntity(number=1)
    ReferenceEntityHolder(bar=entity)

    assert load(entity) is entity


def test_entity_reuse_in_reverse_order_remains_valid() -> None:
    '''Later reference-capable use must not invalidate an earlier holder.'''
    entity = KeyedEntity(number=1)
    plain = PlainEntityHolder(bar=entity)
    ReferenceEntityHolder(bar=entity)

    assert plain.validate_attribs() == []


def test_entity_reuse_retains_populated_allowed_metadata() -> None:
    '''Accept empty reference slots without discarding the entity's real key.'''
    entity = KeyedEntity(number=1)
    entity.set_external_key('entity-1')
    ReferenceEntityHolder(bar=entity)

    plain = PlainEntityHolder(bar=entity)

    assert plain.bar is entity
    assert plain.model_dump()['bar'] == {'@key:external': 'entity-1', 'number': 1}


def test_entity_reuse_retains_reference_permissions() -> None:
    '''A plain-field validation must keep an existing field reference-capable.'''
    entity = KeyedEntity(number=1)
    holder = ReferenceEntityHolder(bar=entity)
    PlainEntityHolder(bar=entity)

    reference = Reference(entity)
    holder.bar = reference

    assert holder.bar is entity
    assert holder.model_dump()['bar'] == {'@ref': reference.target_key}


def test_explicit_reference_reuse_preserves_target_identity() -> None:
    '''Explicit references must continue to expose their existing targets.'''
    entity = KeyedEntity(number=1)
    entity.set_external_key('entity-1')

    referenced = ReferenceEntityHolder(bar=Reference(entity))
    plain = PlainEntityHolder(bar=entity)

    assert referenced.bar is entity
    assert plain.bar is entity


def test_unset_metadata_is_compatible_with_a_plain_field() -> None:
    '''Treat None-valued metadata as unset even if its name is not allowed.'''
    entity = KeyedEntity(number=1)
    entity.set_meta(check_allowed=False, scheme=None)

    assert PlainEntityHolder(bar=entity).model_dump()['bar'] == {'number': 1}


@pytest.mark.parametrize('value', ['unexpected', '', 0, False])
def test_entity_reuse_rejects_disallowed_populated_metadata(value: object) -> None:
    '''Continue rejecting forbidden metadata values, including falsey values.'''
    entity = KeyedEntity(number=1)
    entity.set_meta(check_allowed=False, scheme=value)

    with pytest.raises(ValidationError, match='Allowed meta'):
        PlainEntityHolder(bar=entity)

# EOF
