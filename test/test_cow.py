"""Tests for runtime copy-on-write wrappers."""
from typing import Any

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.cow import rune_cow, rune_unwrap
from rune.runtime.object_builder import ObjectBuilder


class _Child(BaseDataClass):
    x: int


class _Parent(BaseDataClass):
    child: _Child
    ys: list[int]


class _Holder(BaseDataClass):
    node: Any


class _BuilderModel(BaseDataClass):
    child: _Child


def test_rune_cow_model_and_list_are_isolated_on_write():
    original = _Parent(child=_Child(x=1), ys=[1, 2])
    wrapped = rune_cow(original)

    wrapped.child.x = 9
    wrapped.ys.append(3)

    assert original.child.x == 1
    assert original.ys == [1, 2]

    result = rune_unwrap(wrapped)
    assert isinstance(result, _Parent)
    assert result.child.x == 9
    assert result.ys == [1, 2, 3]


def test_rune_cow_nested_list_element_write_does_not_mutate_original():
    original = [_Child(x=1), _Child(x=2)]
    wrapped = rune_cow(original)

    wrapped[0].x = 7

    assert original[0].x == 1
    result = rune_unwrap(wrapped)
    assert result[0].x == 7
    assert result[1].x == 2


def test_rune_cow_dict_value_write_does_not_mutate_original():
    original = {"node": _Child(x=3)}
    wrapped = rune_cow(original)

    wrapped["node"].x = 5

    assert original["node"].x == 3
    result = rune_unwrap(wrapped)
    assert result["node"].x == 5


def test_rune_cow_object_builder_is_isolated_on_write():
    original = ObjectBuilder(_BuilderModel)
    original.child.x = 1

    wrapped = rune_cow(original)
    wrapped.child.x = 9

    assert original.child.x == 1

    result = rune_unwrap(wrapped)
    assert isinstance(result, ObjectBuilder)
    assert result.child.x == 9

    model = wrapped.to_model()
    assert isinstance(model, _BuilderModel)
    assert model.child.x == 9


def test_rune_cow_object_builder_missing_path_write_does_not_mutate_original():
    original = ObjectBuilder()
    wrapped = rune_cow(original)

    wrapped.a.b = 7

    assert original.to_dict() == {}

    result = rune_unwrap(wrapped)
    assert isinstance(result, ObjectBuilder)
    assert result.to_dict() == {"a": {"b": 7}}


def test_rune_unwrap_without_write_returns_original_instance():
    original = _Parent(child=_Child(x=4), ys=[8])
    wrapped = rune_cow(original)

    assert rune_unwrap(wrapped) is original


def test_rune_unwrap_deep_unwraps_basedataclass_fields():
    original = _Parent(child=_Child(x=1), ys=[2])
    wrapped = rune_cow(original)

    holder = _Holder(node=wrapped.child)
    holder.node.x = 9

    assert original.child.x == 1

    unwrapped_holder = rune_unwrap(holder)
    assert isinstance(unwrapped_holder, _Holder)
    assert isinstance(unwrapped_holder.node, _Child)
    assert unwrapped_holder.node.x == 9


def test_rune_unwrap_deep_unwraps_object_builder_payload():
    original = _Parent(child=_Child(x=2), ys=[7])
    wrapped = rune_cow(original)

    draft = ObjectBuilder(_BuilderModel)
    draft.child = wrapped.child
    draft.child.x = 11

    assert original.child.x == 2

    unwrapped_draft = rune_unwrap(draft)
    assert isinstance(unwrapped_draft, ObjectBuilder)
    assert isinstance(unwrapped_draft.child, _Child)
    assert unwrapped_draft.child.x == 11

    model = unwrapped_draft.to_model()
    assert isinstance(model, _BuilderModel)
    assert model.child.x == 11
