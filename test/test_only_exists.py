# Copyright (c) 2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Rune only exists operator."""

import inspect
import logging

import pytest

from rune.runtime import utils
from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.cow import rune_cow
from rune.runtime.object_builder import ObjectBuilder
from rune.runtime.utils import rune_attr_exists, rune_only_exists, rune_resolve_attr


class _BaseState(BaseDataClass):
    before: int | None = None


class _State(_BaseState):
    after: int | None = None
    other: int | None = None


class _ExtendedState(_State):
    extra: int | None = None


class _StateHolder(BaseDataClass):
    bar: _State | None = None


_STATE_ATTRIBUTES = ('before', 'after', 'other')


def test_only_exists_is_exported_for_generated_code():
    assert 'rune_only_exists' in utils.__all__


@pytest.mark.parametrize(('values', 'required', 'expected'), [
    ({'before': 1}, ('before',), True),
    ({'before': 1, 'other': 2}, ('before',), False),
    ({'after': 1}, ('before',), False),
    ({}, ('before',), False),
    ({'before': 1, 'after': 2}, ('before', 'after'), True),
    ({'before': 1}, ('before', 'after'), False),
    ({'before': 1}, ('missing',), False),
    ({}, (), True),
    ({'before': 1}, (), False),
])
def test_only_exists_requires_exactly_the_requested_attributes(values, required, expected):
    holder = _StateHolder(bar=_State(**values))

    assert rune_only_exists(rune_resolve_attr(holder, 'bar'), _STATE_ATTRIBUTES, required) is expected


def test_only_exists_ignores_subtype_fields_and_checks_inherited_fields():
    state = _ExtendedState(before=1, extra=2)

    assert rune_only_exists(state, _STATE_ATTRIBUTES, ('before',)) is True
    assert rune_only_exists(state, _STATE_ATTRIBUTES, ('before', 'extra')) is False


def test_only_exists_resolves_mangled_model_attributes():
    class _MangledState(BaseDataClass):
        rune_attr_class: int | None = None
        after: int | None = None

    assert rune_only_exists(_MangledState(rune_attr_class=0), ('class', 'after'), ('class',)) is True


def test_only_exists_resolves_mangled_function_frame_locals():
    def condition(rune_attr_class, after=None):
        return rune_only_exists(inspect.currentframe(), ('class', 'after'), ('class',))

    assert condition(0) is True
    assert condition(0, 1) is False


@pytest.mark.parametrize(('states', 'expected'), [
    (({'before': 1, 'after': 2}, {'before': 3, 'after': 4}), True),
    (({'before': 1}, {'after': 2}), False),
    (({'before': 1, 'after': 2}, {'before': 3, 'after': 4, 'other': 5}), False),
])
def test_only_exists_checks_each_list_item_using_the_same_name_sets(states, expected):
    parents = [_State(**values) for values in states]
    all_attrs = iter(('before', 'after', 'other', 'before'))
    required = iter(('after', 'before', 'before'))

    assert rune_only_exists(parents, all_attrs, required) is expected


@pytest.mark.parametrize('parent', [None, [], [None]])
def test_only_exists_rejects_absent_parents_even_with_no_required_fields(parent):
    assert rune_only_exists(parent, _STATE_ATTRIBUTES, ()) is False


@pytest.mark.parametrize(('value', 'expected'), [
    (None, False),
    ([], False),
    ([None], False),
    ([None, None], False),
    ([None, 0], True),
    ([_State()], True),
    (False, True),
    (0, True),
    ('', True),
])
def test_only_exists_checks_list_contents_and_scalar_presence(value, expected):
    assert rune_only_exists({'value': value}, ('value',), ('value',)) is expected


def test_only_exists_uses_model_data_and_keeps_exists_behavior():
    empty_holder = _StateHolder(bar=_State())
    populated_holder = _StateHolder()
    populated_holder.bar = _ExtendedState(extra=0)

    assert rune_only_exists({'state': empty_holder, 'items': [None]}, ('state', 'items'), ()) is True
    assert rune_only_exists(rune_cow(populated_holder), ('bar',), ('bar',)) is True
    assert rune_attr_exists(empty_holder) is True
    assert rune_attr_exists([None]) is True


def test_only_exists_uses_draft_field_values():
    draft = ObjectBuilder(bar=ObjectBuilder(before=None))

    assert rune_only_exists(draft, ('bar',), ()) is True
    draft.bar.before = 0
    assert rune_only_exists(draft, ('bar',), ('bar',)) is True


def test_only_exists_handles_cycles_when_checking_nested_models_for_data():
    class _Node(BaseDataClass):
        child: '_Node | None' = None
        value: int | None = None

    node = _Node()
    node.child = node

    assert rune_only_exists({'node': node}, ('node',), ()) is True
    node.value = 0
    assert rune_only_exists({'node': node}, ('node',), ('node',)) is True


@pytest.mark.parametrize('parent', [None, [], _State(), _State(before=1, other=2)])
def test_only_exists_returns_false_without_logging(parent, caplog):
    with caplog.at_level(logging.DEBUG, logger=utils.__name__):
        assert rune_only_exists(parent, _STATE_ATTRIBUTES, ('before',)) is False

    assert not caplog.records
