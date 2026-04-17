"""Tests for replaceable function proxy behavior."""

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.func_proxy import replaceable
from rune.runtime.object_builder import ObjectBuilder


class _ReturnModel(BaseDataClass):
    value: int


@replaceable
def _build_valid():
    draft = ObjectBuilder(_ReturnModel)
    draft.value = 4
    return draft


@replaceable
def _build_invalid():
    return ObjectBuilder(_ReturnModel)


def test_replaceable_function_materializes_valid_builder_return():
    result = _build_valid()

    assert isinstance(result, _ReturnModel)
    assert result.value == 4


def test_replaceable_function_falls_back_to_builder_on_validation_error():
    result = _build_invalid()

    assert isinstance(result, ObjectBuilder)

