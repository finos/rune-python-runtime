"""Tests for runtime utility helpers."""
import inspect

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.cow import rune_cow, rune_unwrap
from rune.runtime.func_proxy import (
    replaceable,
    rune_call_unchecked,
    rune_call_unchecked_raw,
)
from rune.runtime.utils import (
    rune_any_elements,
    rune_all_elements,
    rune_check_cardinality,
    rune_check_one_of,
    rune_flatten_list,
    rune_get_only_element,
)


class _UncheckedChild(BaseDataClass):
    x: int


class _UncheckedParent(BaseDataClass):
    child: _UncheckedChild


class _OneOfHolder(BaseDataClass):
    items: list[int] | None = None
    label: str | None = None


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


def test_list_helpers_accept_cow_wrapped_lists():
    wrapped = rune_cow([1, 2, 3])

    assert rune_any_elements(wrapped, "=", 2) is True
    assert rune_check_cardinality(wrapped, 3, 3) is True
    assert rune_get_only_element(rune_cow([9])) == 9


def test_rune_flatten_list_accepts_nested_cow_wrapped_lists():
    wrapped = rune_cow([[1, 2], [3]])

    assert rune_flatten_list(wrapped) == [1, 2, 3]


def test_rune_all_elements_treats_none_as_never_equal():
    assert rune_all_elements(None, "=", None) is False
    assert rune_all_elements(None, "=", 0) is False


def test_direct_not_equals_treats_none_as_not_equal():
    assert (not rune_all_elements(None, "=", None)) is True
    assert (not rune_all_elements(None, "=", 0)) is True


def test_direct_not_equals_uses_pairwise_list_semantics():
    assert (not rune_all_elements([1, 2], "=", [1, 2])) is False
    assert (not rune_all_elements([1, 2], "=", [2, 1])) is True
    assert (not rune_all_elements([1], "=", [1, 2])) is True
    assert (not rune_all_elements([], "=", [])) is False


def test_direct_not_equals_uses_pairwise_list_semantics_for_cow_lists():
    assert (not rune_all_elements(rune_cow([1, 2]), "=", rune_cow([1, 2]))) is False
    assert (not rune_all_elements(rune_cow([1, 2]), "=", rune_cow([2, 1]))) is True


def test_rune_any_elements_keeps_cartesian_not_equals_semantics():
    assert rune_any_elements([1, 2], "<>", [1, 2]) is True
    assert rune_any_elements([], "<>", []) is False


def test_rune_check_one_of_treats_empty_cow_list_as_absent_in_frame():
    items = rune_cow([])
    label = "x"
    frame = inspect.currentframe()

    assert frame is not None
    assert rune_check_one_of(frame, "items", "label") is True


def test_rune_check_one_of_treats_empty_cow_list_as_absent_on_model():
    holder = _OneOfHolder()
    holder.items = rune_cow([])
    holder.label = "x"

    assert rune_check_one_of(holder, "items", "label") is True
