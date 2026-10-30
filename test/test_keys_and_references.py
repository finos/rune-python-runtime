'''test key generation/retrieval runtime functions'''
import json
from decimal import Decimal
from typing import ClassVar, Optional
from typing_extensions import Annotated
import pytest
from pydantic import Field, ValidationError

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.cow import rune_cow
from rune.runtime.metadata import BaseReference, Reference, KeyType, UnresolvedReference
from rune.runtime.metadata import NumberWithMeta, StrWithMeta


class CashFlow(BaseDataClass):
    '''test cashflow'''
    _ALLOWED_METADATA = {'@key', '@key:external'}
    currency: str = Field(...,
                          description='currency',
                          min_length=3,
                          max_length=3)
    amount: Decimal = Field(..., description='payment amount', ge=0)


class CashFlowNoKey(BaseDataClass):
    '''test cashflow'''
    currency: str = Field(...,
                          description='currency',
                          min_length=3,
                          max_length=3)
    amount: Decimal = Field(..., description='payment amount', ge=0)


class DummyLoanNoKey(BaseDataClass):
    '''some more complex data structure'''
    loan: CashFlowNoKey = Field(..., description='loaned amount')
    repayment: CashFlowNoKey = Field(..., description='repaid amount')


class DummyLoan(BaseDataClass):
    '''some more complex data structure'''
    loan: CashFlow = Field(..., description='loaned amount')
    repayment: CashFlow = Field(..., description='repaid amount')


class DummyLoan2(BaseDataClass):
    '''some more complex data structure'''
    loan: Annotated[CashFlow,
                    CashFlow.serializer(),
                    CashFlow.validator(
                        allowed_meta=('@key', '@key:external', '@ref:external',
                                      '@ref'))] = Field(
                                          ..., description='loaned amount')
    repayment: Annotated[CashFlow,
                         CashFlow.serializer(),
                         CashFlow.validator(
                             allowed_meta=('@key', '@key:external',
                                           '@ref:external', '@ref'))] = Field(
                                               ...,
                                               description='repaid amount')

    _KEY_REF_CONSTRAINTS = {
        'loan': {'@ref', '@ref:external'},
        'repayment': {'@ref', '@ref:external'}
    }


class NamedLoan(DummyLoan2):
    '''Inherit both reference fields without redeclaring their annotations.'''
    label: str = 'loan'


class ExtendedNamedLoan(NamedLoan):
    '''Reference fields can be inherited through multiple levels.'''
    pass


class LoanBook(BaseDataClass):
    '''Key targets and inherited references in separate list fields.'''
    cashflows: list[CashFlow]
    loans: list[NamedLoan]


class DummyLoan3(BaseDataClass):
    '''number test class'''
    loan: Annotated[NumberWithMeta,
                    NumberWithMeta.serializer(),
                    NumberWithMeta.validator(
                        ('@key', '@key:external'))] = Field(...,
                                             description="Test amount",
                                             decimal_places=3)
    repayment: Annotated[NumberWithMeta,
                         NumberWithMeta.serializer(),
                         NumberWithMeta.validator(
                             ('@ref', '@ref:external'))] = Field(...,
                                                  description="Test amount",
                                                  decimal_places=3)

    _KEY_REF_CONSTRAINTS = {
        'loan': {'@ref', '@ref:external'},
        'repayment': {'@ref', '@ref:external'}
    }


class DummyLoan4(BaseDataClass):
    '''number test class'''
    loan: Annotated[NumberWithMeta,
                    NumberWithMeta.serializer(),
                    NumberWithMeta.validator(
                        ('@key', '@key:external'))] = Field(...,
                                             description="Test amount",
                                             decimal_places=3)
    repayment: Annotated[NumberWithMeta,
                         NumberWithMeta.serializer(),
                         NumberWithMeta.validator(
                             ('@ref', '@ref:external'))] = Field(...,
                                                  description="Test amount",
                                                  decimal_places=3, gt=0)

    _KEY_REF_CONSTRAINTS = {
        'loan': {'@ref', '@ref:external'},
        'repayment': {'@ref', '@ref:external'}
    }

class DummyTradeParties(BaseDataClass):
    '''number test class'''
    party1: Annotated[StrWithMeta,
                      StrWithMeta.serializer(),
                      StrWithMeta.validator(
                          ('@key',
                           '@key:external'))] = Field(..., description="cpty1")
    party2: Annotated[StrWithMeta,
                      StrWithMeta.serializer(),
                      StrWithMeta.validator(
                          ('@ref',
                           '@ref:external'))] = Field(..., description="cpty2")

    _KEY_REF_CONSTRAINTS = {
        'party2': {'@ref', '@ref:external'}
    }


class OtherThing(BaseDataClass):
    '''helper class for incompatible type tests'''
    _ALLOWED_METADATA = {'@key'}
    name: str = Field(..., description='name')


class ObjectHolder(BaseDataClass):
    '''helper class for non-replaceable property tests'''
    payload: object = Field(..., description='payload')

    _KEY_REF_CONSTRAINTS = {
        'payload': {'@ref'}
    }


class DummyLoanInternalOnly(BaseDataClass):
    '''reference constraints limited to internal refs'''
    loan: Annotated[CashFlow,
                    CashFlow.serializer(),
                    CashFlow.validator(allowed_meta=('@ref', ))] = Field(
                        ..., description='loaned amount')
    repayment: Annotated[CashFlow,
                         CashFlow.serializer(),
                         CashFlow.validator(allowed_meta=('@ref', ))] = Field(
                             ..., description='repaid amount')

    _KEY_REF_CONSTRAINTS = {
        'repayment': {'@ref'}
    }


class MultiKeyThing(BaseDataClass):
    '''helper class for key registration tests'''
    _ALLOWED_METADATA = {'@key', '@key:external', '@key:scoped'}
    name: str = Field(..., description='name')


class DummyBiLoan(BaseDataClass):
    '''more complex model'''
    loan1: DummyLoan2
    loan2: DummyLoan2


def test_key_generation():
    '''generate a key for an object'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    key = model.loan.get_or_create_key()  # pylint: disable=no-member
    assert key


def test_use_ref_from_key():
    '''test use a ref'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    key = model.loan.get_or_create_key()  # pylint: disable=no-member
    ref = Reference(key, key_type=KeyType.INTERNAL, parent=model)
    # pylint: disable=protected-access
    model._bind_property_to('repayment', ref)
    assert id(model.loan) == id(model.repayment)


def test_use_ref_from_object():
    '''test use a ref'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    # pylint: disable=protected-access
    model._bind_property_to('repayment', Reference(model.loan))
    assert id(model.loan) == id(model.repayment)


def test_bad_key_generation():
    '''generate a key for an object which can't be referenced'''
    model = DummyLoanNoKey(loan=CashFlowNoKey(currency='EUR', amount=100),
                           repayment=CashFlowNoKey(currency='EUR', amount=101))
    with pytest.raises(ValueError):
        model.loan.get_or_create_key()  # pylint: disable=no-member


def test_invalid_property():
    '''Attempts to bind a property when not allowed'''
    model = DummyLoan(loan=CashFlow(currency='EUR', amount=100),
                      repayment=CashFlow(currency='EUR', amount=101))
    model2 = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                        repayment=CashFlow(currency='EUR', amount=101))

    with pytest.raises(ValueError):
        # pylint: disable=protected-access
        model._bind_property_to('repayment', Reference(model2.loan))


def test_ref_assign():
    '''test use a ref'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    model.repayment = Reference(model.loan)
    assert id(model.loan) == id(model.repayment)


@pytest.mark.parametrize('model_type', [NamedLoan, ExtendedNamedLoan])
def test_ref_assign_to_inherited_field(model_type):
    '''Inherited reference fields retain their declared target type.'''
    model = model_type(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))

    model.repayment = Reference(model.loan)

    assert model.repayment is model.loan
    assert model.resolve_ref_key('repayment') == model.loan.get_meta('@key')


@pytest.mark.parametrize('validate_model', [False, True])
def test_deserialize_inherited_references_in_lists(validate_model):
    '''Inherited references resolve before optional model validation.'''
    data = json.dumps({
        'cashflows': [{
            '@key:external': 'cashflow1', 'currency': 'EUR', 'amount': '100',
        }],
        'loans': [{
            'loan': {'@ref:external': 'cashflow1'},
            'repayment': {'@ref:external': 'cashflow1'},
        }],
    })

    book = LoanBook.rune_deserialize(data, validate_model=validate_model)

    assert book.loans[0].loan is book.cashflows[0]
    assert book.loans[0].repayment is book.cashflows[0]
    assert book.loans[0].resolve_ref_key('repayment') == 'cashflow1'


def test_ref_assign_from_cow_wrapped_object():
    '''test use a ref from a COW-wrapped object'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    model.repayment = Reference(rune_cow(model.loan))
    assert id(model.loan) == id(model.repayment)


def test_ref_in_constructor():
    '''test use a ref'''
    cf = CashFlow(currency='EUR', amount=100)
    model = DummyLoan2(loan=cf, repayment=Reference(cf))
    assert id(model.loan) == id(model.repayment)


def test_ref_re_assign():
    '''test use a ref'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    old_cf = model.repayment
    model.repayment = Reference(model.loan)
    assert id(model.loan) == id(model.repayment)
    model.repayment = old_cf
    assert 'repayment' not in model.__dict__['__rune_references']
    assert id(model.repayment) == id(old_cf)


def test_ref_ext_assign():
    '''test use a ext key and ref'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    model.repayment = Reference(model.loan, 'ext_key1')
    assert id(model.loan) == id(model.repayment)


def test_ref_ext_assign_from_cow_wrapped_object():
    '''test use an external-key ref from a COW-wrapped object'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    model.repayment = Reference(rune_cow(model.loan), 'ext_key1')
    assert id(model.loan) == id(model.repayment)


def test_ref_ext_assign_2():
    '''test use a ext key and ref'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    # pylint: disable=no-member
    model.loan.set_external_key('ext_key3', KeyType.EXTERNAL)
    model.repayment = Reference('ext_key3',
                                key_type=KeyType.EXTERNAL,
                                parent=model)
    assert id(model.loan) == id(model.repayment)


def test_resolve_ref_key_returns_reference_key():
    '''resolve_ref returns the stored reference key'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    assert model.resolve_ref_key('repayment') is None

    model.repayment = Reference(model.loan, 'loan-ref-key')

    assert model.resolve_ref_key('repayment') == 'loan-ref-key'
    assert model.resolve_ref_key('loan') is None


def test_ref_ext_assign_with_cow_wrapped_parent():
    '''test external-key lookup through a COW-wrapped parent'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    # pylint: disable=no-member
    model.loan.set_external_key('ext_key3', KeyType.EXTERNAL)
    model.repayment = Reference('ext_key3',
                                key_type=KeyType.EXTERNAL,
                                parent=rune_cow(model))
    assert id(model.loan) == id(model.repayment)


def test_init_ref_assign():
    '''test use a ref'''
    loan = CashFlow(currency='EUR', amount=100)
    # repayment = Reference(loan, True)
    model = DummyLoan2(loan=loan, repayment=loan)
    assert id(model.loan) == id(model.repayment)


def test_basic_ref_assign():
    '''test use a ref'''
    model = DummyLoan3(loan=100, repayment=101)
    model.repayment = Reference(model.loan)
    assert id(model.loan) == id(model.repayment)


def test_basic_str_ref_assign():
    '''test use a ref'''
    model = DummyTradeParties(party1='p1', party2='p2')
    model.party2 = Reference(model.party1)
    assert id(model.party1) == id(model.party2)


def test_dump_key_ref():
    '''test dump a ref'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    model.repayment = Reference(model.loan)
    dict_ = model.model_dump(exclude_unset=True)
    assert dict_['loan']['@key'] == dict_['repayment']['@ref']
    assert len(dict_['repayment']) == 1


def test_dump_ref_ext():
    '''test use a ext key and ref'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    model.repayment = Reference(model.loan, 'ext_key2')
    dict_ = model.model_dump(exclude_unset=True)
    assert dict_['loan']['@key:external'] == dict_['repayment']['@ref:external']


def test_dump_key_ref_2():
    '''test dump a ref'''
    model = DummyBiLoan(loan1=DummyLoan2(loan=CashFlow(currency='EUR',
                                                       amount=100),
                                         repayment=CashFlow(currency='EUR',
                                                            amount=101)),
                        loan2=DummyLoan2(loan=CashFlow(currency='EUR',
                                                       amount=100),
                                         repayment=CashFlow(currency='EUR',
                                                            amount=101)))
    model.loan1.repayment = Reference(model.loan1.loan)
    dict_ = model.model_dump(exclude_unset=True)
    assert dict_['loan1']['loan']['@key'] == dict_['loan1']['repayment']['@ref']
    assert len(dict_['loan1']['repayment']) == 1


def test_load_loan_with_key_ref():
    '''test load a simple model with json with some meta'''
    json_str = '''{
        "loan":{"@key":"cf-1-2","currency":"EUR","amount":"100"},
        "repayment":{"@ref":"cf-1-2"}
    }'''
    model = DummyLoan2.model_validate_json(json_str)
    assert id(model.loan) == id(model.repayment)


def test_load_basic_type_loan_with_key_ref():
    '''test load a simple model with json with some meta'''
    json_str = '''{
        "loan": {"@key":"8e50b68b-6426-44a8-bbfd-cbe3b833131c","@data":"100"},
        "repayment":{"@ref":"8e50b68b-6426-44a8-bbfd-cbe3b833131c"}
    }'''
    model = DummyLoan3.model_validate_json(json_str)
    assert id(model.loan) == id(model.repayment)


def test_load_basic_type_loan_with_key_ref_and_constraints():
    '''test load a simple model with json with some meta'''
    json_str = '''{
        "loan": {"@key":"8e50b68b-6426-44a8-bbfd-cbe3b833131a","@data":"100"},
        "repayment":{"@ref":"8e50b68b-6426-44a8-bbfd-cbe3b833131a"}
    }'''
    model = DummyLoan4.model_validate_json(json_str)
    model.validate_model()
    assert id(model.loan) == id(model.repayment)


def test_load_basic_type_loan_with_key_ref_and_broken_constraints():
    '''test load a simple model with json with some meta'''
    json_str = '''{
        "loan": {"@key":"8e50b68b-6426-44a8-bbfd-cbe3b833131b","@data":"-100"},
        "repayment":{"@ref":"8e50b68b-6426-44a8-bbfd-cbe3b833131b"}
    }'''
    model = DummyLoan4.model_validate_json(json_str)
    with pytest.raises(ValidationError):
        model.validate_model()


def test_reference_object_with_ext_key_defaults_external():
    '''reference defaults to EXTERNAL when ext_key provided'''
    cf = CashFlow(currency='EUR', amount=100)
    ref = Reference(cf, 'ext_key1')
    assert ref.key_type == KeyType.EXTERNAL
    assert ref.target is cf
    assert cf.get_meta('@key:external') == 'ext_key1'
    assert cf.get_object_by_key('ext_key1', KeyType.EXTERNAL) is cf


def test_reference_cow_object_with_ext_key_defaults_external():
    '''COW-wrapped object refs should still register external keys'''
    cf = CashFlow(currency='EUR', amount=100)
    ref = Reference(rune_cow(cf), 'ext_key1')
    assert ref.key_type == KeyType.EXTERNAL
    assert ref.target is cf
    assert cf.get_meta('@key:external') == 'ext_key1'
    assert cf.get_object_by_key('ext_key1', KeyType.EXTERNAL) is cf


def test_reference_object_with_non_internal_key_type_raises():
    '''object ref without ext_key must be INTERNAL'''
    cf = CashFlow(currency='EUR', amount=100)
    with pytest.raises(ValueError):
        Reference(cf, key_type=KeyType.EXTERNAL)


def test_reference_string_without_parent_raises():
    '''string ref without parent is invalid'''
    with pytest.raises(ValueError):
        Reference('missing_key')


def test_reference_string_missing_key_raises_keyerror():
    '''missing key should bubble up as KeyError'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    with pytest.raises(KeyError):
        Reference('missing_key', key_type=KeyType.INTERNAL, parent=model)


def test_reference_non_metadata_target_with_ext_key_raises():
    '''ext_key requires metadata-capable target'''
    with pytest.raises(ValueError):
        Reference('not_an_object', 'ext_key')


@pytest.mark.parametrize('model_type', [DummyLoan2, NamedLoan, ExtendedNamedLoan])
def test_bind_property_rejects_wrong_type(model_type):
    '''reject refs that don't match field type'''
    model = model_type(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    original_repayment = model.repayment
    other = OtherThing(name='other')
    with pytest.raises(TypeError, match='Incompatible types'):
        model.repayment = Reference(other)
    assert model.repayment is original_repayment
    assert model.resolve_ref_key('repayment') is None


def test_bind_property_rejects_non_replaceable():
    '''reject refs when current value isn't replaceable'''
    holder = ObjectHolder(payload=123)
    with pytest.raises(ValueError):
        holder.payload = Reference(CashFlow(currency='EUR', amount=100))


def test_bind_property_rejects_ref_tag_not_allowed():
    '''reject ref types not allowed by constraints'''
    model = DummyLoanInternalOnly(
        loan=CashFlow(currency='EUR', amount=100),
        repayment=CashFlow(currency='EUR', amount=101))
    with pytest.raises(ValueError):
        model.repayment = Reference(model.loan, 'ext_key1')


def test_rebind_updates_reference_map():
    '''re-binding should update the stored ref mapping'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    model.repayment = Reference(model.loan)
    first = model.__dict__['__rune_references']['repayment']
    other = CashFlow(currency='EUR', amount=102)
    model.repayment = Reference(other)
    second = model.__dict__['__rune_references']['repayment']
    assert first != second
    assert model.repayment is other


def test_set_external_key_cannot_change():
    '''external keys are immutable once set'''
    cf = CashFlow(currency='EUR', amount=100)
    cf.set_external_key('key1', KeyType.EXTERNAL)
    with pytest.raises(ValueError):
        cf.set_external_key('key2', KeyType.EXTERNAL)
    assert cf.get_meta('@key:external') == 'key1'


def test_set_external_key_idempotent():
    '''setting same external key twice is a no-op'''
    cf = CashFlow(currency='EUR', amount=100)
    cf.set_external_key('key1', KeyType.EXTERNAL)
    cf.set_external_key('key1', KeyType.EXTERNAL)
    assert cf.get_meta('@key:external') == 'key1'


def test_register_keys_multiple_tags():
    '''registers internal/external/scoped keys'''
    obj = MultiKeyThing(name='multi')
    obj._register_keys({
        '@key': 'k1',
        '@key:external': 'k2',
        '@key:scoped': 'k3'
    })
    assert obj.get_object_by_key('k1', KeyType.INTERNAL) is obj
    assert obj.get_object_by_key('k2', KeyType.EXTERNAL) is obj
    assert obj.get_object_by_key('k3', KeyType.SCOPED) is obj


def test_ref_prefers_external_over_internal():
    '''external ref should be preferred over internal when both provided'''
    json_str = '''{
        "loan":{
            "@key":"cf-1-2",
            "@key:external":"cf-ext-1",
            "currency":"EUR",
            "amount":"100"
        },
        "repayment":{
            "@ref":"cf-1-2",
            "@ref:external":"cf-ext-1"
        }
    }'''
    model = DummyLoan2.model_validate_json(json_str)
    assert model.repayment is model.loan
    assert model.__dict__['__rune_references']['repayment'][1] == KeyType.EXTERNAL


def test_unresolved_ref_dangling_behavior():
    '''dangling refs should raise when not ignored'''
    json_str = '''{
        "loan":{"@key":"cf-1-2","currency":"EUR","amount":"100"},
        "repayment":{"@ref":"missing-key"}
    }'''
    model = DummyLoan2.model_validate_json(json_str)
    with pytest.raises(KeyError):
        model.resolve_references(ignore_dangling=False, recurse=False)


class Observation(BaseDataClass):
    '''minimal observation model for list-reference tests (Issue #43)'''
    _ALLOWED_METADATA: ClassVar[set[str]] = {'@key', '@key:external'}
    value: str = Field(...)


class Reset(BaseDataClass):
    '''model with both single-cardinality and multi-cardinality reference fields'''
    primaryObservation: Optional[Annotated[
        Observation,
        Observation.serializer(),
        Observation.validator(('@key', '@key:external', '@ref', '@ref:external')),
    ]] = Field(None)

    observations: Annotated[
        list[Observation | BaseReference | None],
        Observation.serializer(),
        Observation.validator(('@key', '@key:external', '@ref', '@ref:external')),
    ] = Field(default_factory=list)


class BasicListRefModel(BaseDataClass):
    '''model with a list of basic-type references'''
    primaryTag: Annotated[
        StrWithMeta,
        StrWithMeta.serializer(),
        StrWithMeta.validator(('@key', '@key:external')),
    ] = Field(...)
    tags: Annotated[
        list[StrWithMeta | BaseReference],
        StrWithMeta.serializer(),
        StrWithMeta.validator(('@key', '@ref', '@ref:external')),
    ] = Field(default_factory=list)


@pytest.mark.parametrize('validate_model', [False, True])
def test_resolve_references_in_list_fields_and_roundtrip(validate_model):
    '''Issue #43: references held in list fields must resolve to target objects
    and serialize back as reference dicts.'''
    data = {
        'primaryObservation': {'@key': 'obs-1', 'value': '1.234'},
        'observations': [
            {'@ref': 'obs-1'},
        ],
    }

    model = Reset.rune_deserialize(data, validate_model=validate_model)

    assert isinstance(model.primaryObservation, Observation)
    assert isinstance(model.observations[0], Observation)
    assert model.observations[0] is model.primaryObservation
    assert model.resolve_ref_key('observations') == ['obs-1']
    assert model.resolve_ref_key('observations', 0) == 'obs-1'
    assert model.resolve_ref_key('observations', 1) is None

    serialized = json.loads(model.rune_serialize(validate_model=validate_model))
    assert serialized['primaryObservation'] == {'@key': 'obs-1', 'value': '1.234'}
    assert serialized['observations'] == [{'@ref': 'obs-1'}]


def test_mixed_list_of_inline_objects_and_references():
    '''Lists containing both inline keyed objects and references must resolve,
    preserve parent links, and roundtrip cleanly.'''
    data = {
        'primaryObservation': {'@key': 'obs-1', 'value': '1.234'},
        'observations': [
            {'@ref': 'obs-1'},
            {'@key': 'obs-2', 'value': '2.500'},
            {'@ref': 'obs-2'},
        ],
    }

    model = Reset.rune_deserialize(data)

    assert model.observations[0] is model.primaryObservation
    assert isinstance(model.observations[1], Observation)
    assert model.observations[1].value == '2.500'
    assert model.observations[2] is model.observations[1]
    assert model.primaryObservation.get_rune_parent() is model
    assert model.observations[1].get_rune_parent() is model
    assert model.resolve_ref_key('observations') == ['obs-1', 'obs-2']
    assert model.resolve_ref_key('observations', 0) == 'obs-1'
    assert model.resolve_ref_key('observations', 1) is None
    assert model.resolve_ref_key('observations', 2) == 'obs-2'

    dumped = model.model_dump(exclude_unset=True)
    assert dumped['observations'] == [
        {'@ref': 'obs-1'},
        {'@key': 'obs-2', 'value': '2.500'},
        {'@ref': 'obs-2'},
    ]


def test_assign_and_replace_list_of_references():
    '''Assigning a list containing Reference objects binds each reference index,
    and reassigning inline objects clears the reference bindings.'''
    obs1 = Observation(value='1.0')
    obs2 = Observation(value='2.0')
    model = Reset(primaryObservation=obs1)

    model.observations = [Reference(model.primaryObservation), obs2]

    assert model.observations[0] is model.primaryObservation
    assert model.observations[1] is obs2
    assert obs2.get_rune_parent() is model
    assert model.resolve_ref_key('observations') == [model.primaryObservation.get_meta('key')]

    dumped = model.model_dump(exclude_unset=True)
    assert dumped['observations'][0] == {'@ref': model.primaryObservation.get_meta('key')}
    assert dumped['observations'][1] == {'value': '2.0'}

    replacement = Observation(value='3.0')
    model.observations = [replacement]
    assert model.resolve_ref_key('observations') is None
    assert replacement.get_rune_parent() is model


def test_list_reference_dangling_behavior():
    '''Dangling references inside list fields respect ignore_dangling.'''
    data = {
        'primaryObservation': {'@key': 'obs-1', 'value': '1.234'},
        'observations': [{'@ref': 'obs-1'}, {'@ref': 'missing-obs'}],
    }
    model = Reset.model_validate(data)
    assert model.observations[0] is model.primaryObservation
    assert isinstance(model.observations[1], UnresolvedReference)

    with pytest.raises(KeyError, match='missing-obs'):
        model.resolve_references(ignore_dangling=False, recurse=False)


def test_basic_type_list_reference_resolution_and_roundtrip():
    '''List fields of annotated basic types must also resolve and serialize refs.'''
    data = {
        'primaryTag': {'@key:external': 'tag-1', '@data': 'PRIME'},
        'tags': [
            {'@ref:external': 'tag-1'},
            {'@data': 'SECONDARY'},
        ],
    }
    model = BasicListRefModel.rune_deserialize(data)
    assert model.tags[0] is model.primaryTag
    assert model.tags[1] == 'SECONDARY'
    assert model.resolve_ref_key('tags') == ['tag-1']

    dumped = model.model_dump(exclude_unset=True)
    assert dumped['tags'] == [
        {'@ref:external': 'tag-1'},
        {'@data': 'SECONDARY'},
    ]


def test_list_in_place_mutations_keep_reference_bindings_in_sync():
    '''Replacing elements, reversing, popping, inserting, or deleting on a list
    with resolved references updates reference bindings in lockstep.'''
    data = {
        'primaryObservation': {'@key:external': 'obs-1', 'value': '1.0'},
        'observations': [
            {'@ref:external': 'obs-1'},
            {'value': '2.0'},
        ],
    }

    # 1. Replacing observations[0] with an inline object clears the ref at index 0
    model = Reset.rune_deserialize(data)
    inline_replacement = Observation(value='9.9')
    model.observations[0] = inline_replacement
    assert model.resolve_ref_key('observations') is None
    assert model. resolve_ref_key('observations', 0) is None
    assert inline_replacement.get_rune_parent() is model
    model.resolve_references()
    model.validate_model()
    dumped = json.loads(model.rune_serialize())
    assert dumped['observations'] == [
        {'value': '9.9'},
        {'value': '2.0'},
    ]

    # 2. reverse() moves the reference binding to its new index
    model = Reset.rune_deserialize(data)
    model.observations.reverse()
    assert model.resolve_ref_key('observations', 0) is None
    assert model.resolve_ref_key('observations', 1) == 'obs-1'
    assert model.resolve_ref_key('observations') == ['obs-1']
    model.resolve_references()
    model.validate_model()
    dumped = json.loads(model.rune_serialize())
    assert dumped['observations'] == [
        {'value': '2.0'},
        {'@ref:external': 'obs-1'},
    ]

    # 3. pop(0) removes the reference binding and shifts remaining items
    model = Reset.rune_deserialize(data)
    popped = model.observations.pop(0)
    assert popped is model.primaryObservation
    assert model.resolve_ref_key('observations') is None
    model.resolve_references()
    model.validate_model()
    dumped = json.loads(model.rune_serialize())
    assert dumped['observations'] == [{'value': '2.0'}]

    # 4. insert(0, inline_obs) shifts the reference binding from index 0 to index 1
    model = Reset.rune_deserialize(data)
    inserted = Observation(value='0.5')
    model.observations.insert(0, inserted)
    assert inserted.get_rune_parent() is model
    assert model.resolve_ref_key('observations', 0) is None
    assert model.resolve_ref_key('observations', 1) == 'obs-1'
    dumped = json.loads(model.rune_serialize())
    assert dumped['observations'] == [
        {'value': '0.5'},
        {'@ref:external': 'obs-1'},
        {'value': '2.0'},
    ]


def test_list_iadd_concat_and_cow_preserve_reference_bindings():
    '''+=, list concatenation, and rune_cow updates on list fields preserve
    existing reference bindings and round-trip without duplicate keys.'''
    from rune.runtime.cow import rune_cow, rune_unwrap

    data = {
        'primaryObservation': {'@key:external': 'obs-1', 'value': '1.0'},
        'observations': [
            {'@ref:external': 'obs-1'},
            {'value': '2.0'},
        ],
    }

    # 1. += in-place augmentation
    model = Reset.rune_deserialize(data)
    model.observations += [Observation(value='3.0')]
    assert model.resolve_ref_key('observations') == ['obs-1']
    assert model.resolve_ref_key('observations', 0) == 'obs-1'
    serialized = json.loads(model.rune_serialize())
    serialized.pop('@type', None)
    round_tripped = Reset.rune_deserialize(serialized)
    assert round_tripped.observations[0] is round_tripped.primaryObservation
    assert [o.value for o in round_tripped.observations] == ['1.0', '2.0', '3.0']

    # 2. List concatenation (observations = observations + [...])
    model2 = Reset.rune_deserialize(data)
    model2.observations = model2.observations + [Observation(value='4.0')]
    assert model2.resolve_ref_key('observations') == ['obs-1']
    serialized2 = json.loads(model2.rune_serialize())
    serialized2.pop('@type', None)
    round_tripped2 = Reset.rune_deserialize(serialized2)
    assert round_tripped2.observations[0] is round_tripped2.primaryObservation
    assert [o.value for o in round_tripped2.observations] == ['1.0', '2.0', '4.0']

    # 3. Copy-on-write list append / += / reverse / nested edit
    base = Reset.rune_deserialize(data)
    cow = rune_cow(base)
    cow.observations.append(Observation(value='5.0'))
    cow.observations += [Observation(value='6.0')]
    cow.observations[1].value = '2.5'
    updated = rune_unwrap(cow)

    assert [o.value for o in base.observations] == ['1.0', '2.0']
    assert base.resolve_ref_key('observations') == ['obs-1']
    assert updated.observations[0] is updated.primaryObservation
    assert updated.resolve_ref_key('observations') == ['obs-1']
    assert [o.value for o in updated.observations] == ['1.0', '2.5', '5.0', '6.0']

    cow_serialized = json.loads(updated.rune_serialize())
    cow_serialized.pop('@type', None)
    cow_round_tripped = Reset.rune_deserialize(cow_serialized)
    assert cow_round_tripped.observations[0] is cow_round_tripped.primaryObservation
    assert [o.value for o in cow_round_tripped.observations] == ['1.0', '2.5', '5.0', '6.0']

    # 4. Plain-list copy-on-write and intra-list inline + reference bindings
    intra_data = {
        'observations': [
            {'@key': 'obs-inline', 'value': '10.0'},
            {'@ref': 'obs-inline'},
        ],
    }
    intra = Reset.rune_deserialize(intra_data)
    intra.observations = [*intra.observations, Observation(value='20.0')]
    assert intra.resolve_ref_key('observations', 0) is None
    assert intra.resolve_ref_key('observations', 1) == 'obs-inline'
    assert intra.resolve_ref_key('observations', 2) is None
    intra_dumped = json.loads(intra.rune_serialize())
    intra_dumped.pop('@type', None)
    intra_round_tripped = Reset.rune_deserialize(intra_dumped)
    assert intra_round_tripped.observations[1] is intra_round_tripped.observations[0]


def test_rejected_list_reference_assignment_is_atomic():
    '''Assigning a list containing an invalid Reference must raise before
    mutating existing list items or reference bindings.'''
    obs1 = Observation(value='1.0').set_external_key('obs-1')
    obs2 = Observation(value='2.0').set_external_key('obs-2')
    model = Reset(
        primaryObservation=obs1,
        observations=[Reference('obs-1', key_type=KeyType.EXTERNAL, parent=obs1)],
    )
    original_list = list(model.observations)
    original_refs = model.resolve_ref_key('observations')

    wrong_target = OtherThing(name='wrong')
    valid_ref = Reference(obs2)
    wrong_type_ref = Reference(wrong_target)

    with pytest.raises(TypeError, match='Incompatible types'):
        model.observations = [valid_ref, wrong_type_ref]

    assert list(model.observations) == original_list
    assert model.observations[0] is obs1
    assert model.resolve_ref_key('observations') == original_refs

    scoped_target = MultiKeyThing(name='scoped').set_external_key(
        'scoped-obs', KeyType.SCOPED
    )
    disallowed_scoped_ref = Reference(
        'scoped-obs', key_type=KeyType.SCOPED, parent=scoped_target
    )
    with pytest.raises(ValueError, match='not allowed for observations'):
        model.observations = [valid_ref, disallowed_scoped_ref]

    assert list(model.observations) == original_list
    assert model.observations[0] is obs1
    assert model.resolve_ref_key('observations') == original_refs

# EOF
