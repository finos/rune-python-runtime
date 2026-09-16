'''Test that reference dicts deserialize correctly for fields set up via the
deferred Phase 1/2/3 annotation pattern used by the bundle generator.

The bundle generator defers field annotations for bundled (cyclic) classes:
  Phase 1: field uses None placeholder so Pydantic skips type resolution
  Phase 2: model_fields["f"].annotation and __annotations__["f"] are set
  Phase 3: model_rebuild(force=True) is called

Pydantic's model_rebuild reads model_fields["f"].annotation (the bare type,
e.g. Optional[Party | BaseReference]) to build the core schema.  The
PlainValidator placed in __annotations__["f"] is ignored.  The resulting
schema for the Party union arm is function-wrap[_deserialize_refs()], which
calls handler(data) before checking whether data is a reference dict.  When
data is {'@ref:external': 'some-id'}, handler fails trying to construct Party
because its required fields are absent.

The fix is to have _deserialize_refs check for reference dicts BEFORE calling
handler, consistent with ComplexTypeMetaDataMixin.deserialize.
Target keys must also be retained when the deferred schema bypasses the field
metadata validator, so references in sibling lists can find their targets.

Note on placement: this test lives in rune-runtime even though the Phase 1/2/3
pattern originates in the rune-python-generator's bundle generator.  The runtime
has no awareness of bundles; however, _deserialize_refs is a runtime method and
the fix must be applied there.  The test simulates the bundle generator's output
in plain Python so that the runtime can own a regression guard for this behaviour
without depending on the generator.
'''
from typing import Annotated, Optional
import pytest
from pydantic import Field

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.metadata import BaseReference, KeyType, UnresolvedReference


class Issuer(BaseDataClass):
    '''Represents the issuing party; requires at minimum one identifier.'''
    _ALLOWED_METADATA = {'@key', '@key:external'}
    partyId: list[str] = Field(..., description='Party identifiers', min_length=1)


class Contract(BaseDataClass):
    '''A contract that references an issuer via a key/ref constraint.'''
    _ALLOWED_METADATA = {'@key', '@key:external'}
    issuerReference: None = Field(None, description='Reference to the issuing party')
    name: str = Field(..., description='Contract name')

    _KEY_REF_CONSTRAINTS = {
        'issuerReference': {'@key', '@key:external', '@ref', '@ref:external', '@ref:scoped'}
    }


def _apply_deferred_annotations():
    '''Simulate what the bundle generator Phase 2 + Phase 3 does.'''
    Contract.model_fields['issuerReference'].annotation = Optional[Issuer | BaseReference]
    Contract.__annotations__['issuerReference'] = Annotated[
        Optional[Issuer | BaseReference],
        Issuer.serializer(),
        Issuer.validator(('@key', '@key:external', '@ref', '@ref:external', '@ref:scoped')),
    ]
    Contract.model_rebuild(force=True)


_apply_deferred_annotations()


class Portfolio(BaseDataClass):
    '''Bare field types reproduce schemas rebuilt from deferred annotations.'''
    contracts: list[Contract]
    issuers: list[Issuer]


@pytest.mark.parametrize('key_type', list(KeyType))
def test_deferred_list_targets_retain_keys_and_resolve(key_type):
    '''Target keys must survive even when field metadata validators are absent.'''
    data = {
        'contracts': [{
            'name': 'test-contract',
            'issuerReference': {key_type.rune_ref_tag: 'party1'},
        }],
        'issuers': [{key_type.rune_key_tag: 'party1', 'partyId': ['LEI-001']}],
    }

    portfolio = Portfolio.rune_deserialize(data)

    assert portfolio.issuers[0].get_meta(key_type.key_tag) == 'party1'
    assert portfolio.contracts[0].issuerReference is portfolio.issuers[0]
    assert portfolio.get_object_by_key('party1', key_type) is portfolio.issuers[0]
    assert portfolio.validate_model() == []


def test_ref_external_deserializes_to_unresolved_reference():
    '''A ref dict in a deferred field should produce an UnresolvedReference,
    not a validation error about missing required fields on the target type.'''
    data = {
        'name': 'test-contract',
        'issuerReference': {'@ref:external': 'party1'},
    }
    contract = Contract.model_validate(data)
    assert isinstance(contract.issuerReference, UnresolvedReference)


def test_ref_internal_deserializes_to_unresolved_reference():
    '''Internal @ref should also produce an UnresolvedReference.'''
    data = {
        'name': 'test-contract',
        'issuerReference': {'@ref': 'party1'},
    }
    contract = Contract.model_validate(data)
    assert isinstance(contract.issuerReference, UnresolvedReference)


def test_none_field_accepted():
    '''Optional reference field set to None should remain None.'''
    contract = Contract.model_validate({'name': 'test-contract'})
    assert contract.issuerReference is None


def test_inline_object_still_validates():
    '''A normal object dict should still deserialize as the target type.'''
    data = {
        'name': 'test-contract',
        'issuerReference': {'partyId': ['LEI-001']},
    }
    contract = Contract.model_validate(data)
    assert isinstance(contract.issuerReference, Issuer)
    assert contract.issuerReference.partyId == ['LEI-001']
