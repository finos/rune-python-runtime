'''test key generation/retrieval runtime functions'''
from decimal import Decimal
from typing_extensions import Annotated
import pytest
from pydantic import Field, ValidationError

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.metadata import Reference, KeyType
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


def test_bind_property_rejects_wrong_type():
    '''reject refs that don't match field type'''
    model = DummyLoan2(loan=CashFlow(currency='EUR', amount=100),
                       repayment=CashFlow(currency='EUR', amount=101))
    other = OtherThing(name='other')
    with pytest.raises(ValueError):
        model.repayment = Reference(other)


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

# EOF
