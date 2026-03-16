"""Tests for runtime utility helpers."""

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.cow import rune_unwrap
from rune.runtime.func_proxy import (
    replaceable,
    rune_call_unchecked,
    rune_call_unchecked_raw,
)


class _UncheckedChild(BaseDataClass):
    x: int


class _UncheckedParent(BaseDataClass):
    child: _UncheckedChild


def test_rune_call_unchecked_wraps_plain_callable_return():
    original = _UncheckedParent(child=_UncheckedChild(x=1))

    def _plain_builder():
        return original

    wrapped = rune_call_unchecked(_plain_builder)
    wrapped.child.x = 9

    assert original.child.x == 1

    result = rune_unwrap(wrapped)
    assert result.child.x == 9


def test_rune_call_unchecked_raw_returns_original_result():
    original = _UncheckedParent(child=_UncheckedChild(x=3))

    @replaceable
    def _replaceable_builder():
        return original

    result = rune_call_unchecked_raw(_replaceable_builder)
    result.child.x = 5

    assert result is original
    assert original.child.x == 5


def test_rune_call_unchecked_wraps_replaceable_callable_return():
    original = _UncheckedParent(child=_UncheckedChild(x=2))

    @replaceable
    def _replaceable_builder():
        return original

    wrapped = rune_call_unchecked(_replaceable_builder)
    wrapped.child.x = 7

    assert original.child.x == 2

    result = rune_unwrap(wrapped)
    assert isinstance(result, _UncheckedParent)
    assert result.child.x == 7
