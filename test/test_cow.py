"""Tests for runtime copy-on-write wrappers."""
from typing import Any

import pytest
from pydantic import ValidationError

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.conditions import rune_condition
from rune.runtime.cow import rune_cow, rune_unwrap
from rune.runtime.func_proxy import rune_finalize_return
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


class _BuilderValueModel(BaseDataClass):
    value: int


class _ConditionalChild(BaseDataClass):
    x: int

    @rune_condition
    def positive(self):
        return self.x > 0


class _ConditionalParent(BaseDataClass):
    children: list[_ConditionalChild]


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


def test_rune_cow_list_slice_nested_write_does_not_mutate_original():
    original = [_Child(x=1), _Child(x=2)]
    wrapped = rune_cow(original)

    sliced = wrapped[:1]
    sliced[0].x = 9

    assert original[0].x == 1
    result = rune_unwrap(wrapped)
    assert [child.x for child in result] == [9, 2]


def test_rune_cow_list_slice_append_updates_parent_copy_only():
    original = [_Child(x=1), _Child(x=2)]
    wrapped = rune_cow(original)

    sliced = wrapped[:1]
    sliced.append(_Child(x=3))

    assert [child.x for child in original] == [1, 2]
    result = rune_unwrap(wrapped)
    assert [child.x for child in result] == [1, 3, 2]


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


def test_rune_finalize_return_materializes_valid_object_builder():
    draft = ObjectBuilder(_BuilderValueModel)
    draft.value = 7

    result = rune_finalize_return(draft)

    assert isinstance(result, _BuilderValueModel)
    assert result.value == 7


def test_rune_finalize_return_falls_back_to_builder_on_validation_error():
    draft = ObjectBuilder(_BuilderValueModel)

    result = rune_finalize_return(draft)

    assert isinstance(result, ObjectBuilder)
    with pytest.raises(ValidationError):
        result.to_model()


def test_validate_conditions_recurses_into_cow_list_fields():
    parent = _ConditionalParent(children=[_ConditionalChild(x=1)])
    parent.children = rune_cow([_ConditionalChild(x=-1)])

    errors = parent.validate_conditions(raise_exc=False)

    assert len(errors) == 1
