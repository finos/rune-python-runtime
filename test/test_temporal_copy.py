# Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

"""Copying date/time metadata wrappers directly and through Pydantic and COW."""

from copy import copy, deepcopy
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.cow import rune_cow, rune_unwrap
from rune.runtime.metadata import DateTimeWithMeta, DateWithMeta, TimeWithMeta


TemporalValue = DateWithMeta | TimeWithMeta | DateTimeWithMeta


class TemporalModel(BaseDataClass):
    """Model with two fields sharing the same metadata-bearing value."""

    first: TemporalValue
    second: TemporalValue
    counter: int = 1


class TemporalParent(BaseDataClass):
    """Parent model used to exercise a nested COW write."""

    child: TemporalModel


@pytest.fixture(params=[
    pytest.param(lambda: DateWithMeta('2026-10-29'), id='date'),
    pytest.param(lambda: TimeWithMeta(time(1, 30, 42, 123456, fold=1)),
                 id='naive-time'),
    pytest.param(lambda: TimeWithMeta(time(
        1, 30, 42, 123456, tzinfo=timezone(timedelta(hours=5, minutes=30)), fold=1)),
        id='offset-time'),
    pytest.param(lambda: TimeWithMeta(time(
        1, 30, 42, 123456, tzinfo=ZoneInfo('Europe/London'), fold=1)),
        id='zoned-time'),
    pytest.param(lambda: DateTimeWithMeta(datetime(
        2026, 10, 25, 1, 30, 42, 123456, fold=1)), id='naive-datetime'),
    pytest.param(lambda: DateTimeWithMeta(datetime(
        2026, 10, 25, 1, 30, 42, 123456,
        tzinfo=timezone(timedelta(hours=5, minutes=30)), fold=1)),
        id='offset-datetime'),
    pytest.param(lambda: DateTimeWithMeta(datetime(
        2026, 10, 25, 1, 30, 42, 123456,
        tzinfo=ZoneInfo('Europe/London'), fold=1)), id='zoned-datetime'),
])
def temporal_value(request: pytest.FixtureRequest) -> TemporalValue:
    """Create a fresh wrapper with metadata that serialization would filter."""
    value = request.param()
    nested = {'items': [1]}
    value.set_meta(check_allowed=False, scheme='test-scheme', zero=0,
                   disabled=False, empty='', unset=None, nested=nested)
    value.extra = nested
    return value


def _assert_same_value(clone: TemporalValue, original: TemporalValue) -> None:
    assert clone is not original
    assert type(clone) is type(original)
    assert clone == original
    if isinstance(original, (TimeWithMeta, DateTimeWithMeta)):
        assert clone.tzinfo is original.tzinfo
        assert clone.fold == original.fold
        assert clone.microsecond == original.microsecond


def test_shallow_copy_shares_attribute_values(temporal_value: TemporalValue) -> None:
    """A shallow copy has its own instance dictionary and shared attributes."""
    clone = copy(temporal_value)

    _assert_same_value(clone, temporal_value)
    assert clone.__dict__ == temporal_value.__dict__
    assert clone.__dict__ is not temporal_value.__dict__
    assert clone.extra is temporal_value.extra
    assert clone._get_meta_container() is temporal_value._get_meta_container()


def test_deep_copy_preserves_and_isolates_metadata(temporal_value: TemporalValue) -> None:
    """Deep copying must retain false-valued metadata and isolate mutable data."""
    clone = deepcopy(temporal_value)

    _assert_same_value(clone, temporal_value)
    assert clone.__dict__ == temporal_value.__dict__
    assert clone._get_meta_container() is not temporal_value._get_meta_container()
    assert clone.extra is clone.get_meta('nested')
    assert clone.extra is not temporal_value.extra

    clone.extra['items'].append(2)
    assert temporal_value.extra['items'] == [1]


def test_deep_copy_preserves_cycles_and_aliases(temporal_value: TemporalValue) -> None:
    """The deepcopy memo must retain self-references and shared values."""
    temporal_value.loop = temporal_value
    temporal_value.extra['owner'] = temporal_value

    first, second = deepcopy([temporal_value, temporal_value])

    _assert_same_value(first, temporal_value)
    assert first is second
    assert first.loop is first
    assert first.extra['owner'] is first
    assert first.extra is first.get_meta('nested')


@pytest.mark.parametrize('wrapper, value', [
    (DateWithMeta, date(2026, 10, 29)),
    (TimeWithMeta, time(1, 30)),
    (DateTimeWithMeta, datetime(2026, 10, 29, 1, 30)),
])
@pytest.mark.parametrize('copier', [copy, deepcopy], ids=['copy', 'deepcopy'])
def test_copy_preserves_wrapper_subclasses(wrapper: type, value: Any, copier: Any) -> None:
    """Copy hooks inherited by subclasses must construct the actual subtype."""
    subtype = type(f'Special{wrapper.__name__}', (wrapper,), {})
    original = subtype(value, scheme='subtype-scheme')

    clone = copier(original)

    _assert_same_value(clone, original)
    assert clone.get_meta('scheme') == 'subtype-scheme'


def test_pydantic_deep_copy_preserves_shared_values(temporal_value: TemporalValue) -> None:
    """Pydantic deep copies must succeed and preserve aliases between fields."""
    original = TemporalModel(first=temporal_value, second=temporal_value)

    clone = original.model_copy(deep=True)

    assert clone is not original
    _assert_same_value(clone.first, temporal_value)
    assert clone.first is clone.second
    clone.first.extra['items'].append(2)
    assert temporal_value.extra['items'] == [1]


def test_cow_write_copies_temporal_values(temporal_value: TemporalValue) -> None:
    """The first COW write must clone a model containing temporal wrappers."""
    original = TemporalModel(first=temporal_value, second=temporal_value)
    proxy = rune_cow(original)

    proxy.counter = 2
    result = rune_unwrap(proxy)

    assert result.counter == 2
    assert original.counter == 1
    _assert_same_value(result.first, temporal_value)
    assert result.first is result.second
    result.first.extra['items'].append(2)
    assert temporal_value.extra['items'] == [1]


def test_nested_cow_write_copies_temporal_values(temporal_value: TemporalValue) -> None:
    """Cloning a child and then its parent must also handle temporal metadata."""
    original = TemporalParent(child=TemporalModel(
        first=temporal_value, second=temporal_value))
    proxy = rune_cow(original)

    proxy.child.counter = 2
    result = rune_unwrap(proxy)

    assert result.child.counter == 2
    assert original.child.counter == 1
    _assert_same_value(result.child.first, temporal_value)
    assert result.child.first is result.child.second
