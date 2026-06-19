"""Demonstrates a runtime polymorphism issue independent of any Rune-DSL feature.

`BaseDataClass.model_config` sets `revalidate_instances='always'`. Combined
with `@validate_call` on every generated function, this means: passing a
subtype instance (`_Bar`) into a parameter declared as its supertype
(`_Foo`) causes Pydantic to rebuild the value strictly from `_Foo`'s own
schema, discarding the subtype's identity and its extra fields - before the
function body ever runs.

"""
from pydantic import validate_call
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