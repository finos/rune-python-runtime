"""Demonstrates a runtime polymorphism issue independent of any Rune-DSL feature.

`BaseDataClass.model_config` sets `revalidate_instances='always'`. Combined
with `@validate_call` on every generated function, this means: passing a
subtype instance (`_Bar`) into a parameter declared as its supertype
(`_Foo`) causes Pydantic to rebuild the value strictly from `_Foo`'s own
schema, discarding the subtype's identity and its extra fields - before the
function body ever runs.

"""
from typing import Optional
import json
import pytest
from pydantic import Field, validate_call
from pydantic import InstanceOf

from rune.runtime.base_data_class import BaseDataClass


class _Foo(BaseDataClass):
    pass


class _Bar(_Foo):
    bar_attr: int = 0


@validate_call
def _identity_foo(foo: InstanceOf[_Foo]) -> _Foo:
    """Mirrors a generated Rune function: declared parameter/return type is
    the supertype, exactly as a function taking `foo Foo (1..1)` would be
    generated."""
    return foo


def test_subtype_survives_plain_python_assignment():
    """Sanity check: outside of Pydantic's validate_call, identity is preserved."""
    bar = _Bar(bar_attr=42)
    held = bar
    assert isinstance(held, _Bar)
    assert held.bar_attr == 42


def test_subtype_identity_lost_across_validate_call_boundary():
    """A _Bar passed into a _Foo-typed parameter should still be a _Bar on return."""
    bar = _Bar(bar_attr=42)
    assert isinstance(bar, _Bar)  # true before the call

    result = _identity_foo(foo=bar)

    # Expected under standard OOP/Liskov semantics: the identity function
    # returns the same kind of object it was given.
    assert isinstance(result, _Bar)
    assert result.bar_attr == 42


class Person(BaseDataClass):
    """Base model class for subtype attribute tests (Issue #58)."""
    name: Optional[str] = Field(None, description='')


class Contractor(Person):
    """Subtype of Person with an extra attribute."""
    hourlyRate: Optional[int] = Field(None, description='')


class Payroll(BaseDataClass):
    """Model holding a parent-typed attribute and list attribute."""
    person: Optional[Person] = Field(None, description='')
    team: list[Person] = Field(default_factory=list, description='')


def test_subtype_preserved_on_attribute_construction():
    """Constructing Payroll with Contractor preserves Contractor and its fields."""
    payroll = Payroll(person=Contractor(hourlyRate=5))
    assert isinstance(payroll.person, Contractor)
    assert payroll.person.hourlyRate == 5
    assert payroll.person.get_rune_parent() is payroll


def test_subtype_preserved_on_model_validate():
    """model_validate with a Contractor instance preserves Contractor and its fields."""
    payroll = Payroll.model_validate({'person': Contractor(hourlyRate=5)})
    assert isinstance(payroll.person, Contractor)
    assert payroll.person.hourlyRate == 5


def test_subtype_preserved_across_validate_model():
    """validate_model() re-validates without stripping subtype identity or fields."""
    payroll = Payroll(person=Contractor(name='Alice', hourlyRate=5))
    errors = payroll.validate_model()
    assert errors == []
    assert isinstance(payroll.person, Contractor)
    assert payroll.person.hourlyRate == 5


def test_subtype_attribute_round_trip_serialize_and_deserialize():
    """rune_serialize writes @type and subtype fields; rune_deserialize restores the subtype."""
    assigned = Payroll()
    assigned.person = Contractor(hourlyRate=5)
    assert isinstance(assigned.person, Contractor)
    assert assigned.person.hourlyRate == 5

    raw = assigned.rune_serialize(validate_model=False)
    data = json.loads(raw)
    assert data['person']['@type'].endswith('Contractor')
    assert data['person']['hourlyRate'] == 5

    restored = Payroll.rune_deserialize(raw, validate_model=False)
    assert isinstance(restored.person, Contractor)
    assert restored.person.hourlyRate == 5

    restored_from_json = Payroll.model_validate_json(raw)
    assert isinstance(restored_from_json.person, Contractor)
    assert restored_from_json.person.hourlyRate == 5


def test_subtype_list_attribute_round_trip():
    """Lists declared with a parent type preserve subtype elements across JSON round-trip."""
    payroll = Payroll(team=[Person(name='Bob'), Contractor(name='Carol', hourlyRate=10)])
    assert isinstance(payroll.team[0], Person)
    assert not isinstance(payroll.team[0], Contractor)
    assert isinstance(payroll.team[1], Contractor)
    assert payroll.team[1].hourlyRate == 10

    raw = payroll.rune_serialize()
    restored = Payroll.rune_deserialize(raw)
    assert type(restored.team[0]) is Person
    assert restored.team[0].name == 'Bob'
    assert isinstance(restored.team[1], Contractor)
    assert restored.team[1].name == 'Carol'
    assert restored.team[1].hourlyRate == 10
