# Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

"""COW inputs and non-destructive extraction of nested Rune metadata."""

from collections.abc import Callable
from copy import deepcopy
from datetime import date
from enum import Enum
from typing import Annotated, Any

import pytest
from pydantic import ValidationError, validate_call

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.cow import rune_cow, rune_unwrap
from rune.runtime.func_proxy import replaceable
from rune.runtime.metadata import DateWithMeta, EnumWithMetaMixin
from rune.runtime.object_builder import ObjectBuilder


class Currency(EnumWithMetaMixin, Enum):
    """Enum with metadata nested inside a complex model."""

    USD = 'USD'


class Leaf(BaseDataClass):
    """Complex value containing basic-type and enum metadata."""

    value: int
    date: Annotated[DateWithMeta, DateWithMeta.validator(('@scheme',))]
    currency: Annotated[Currency, Currency.validator(('@scheme',))]


class Branch(BaseDataClass):
    """Intermediate complex value with metadata on its child."""

    leaf: Annotated[Leaf, Leaf.validator(('@key',))]


class DetailedValue(BaseDataClass):
    """Subtype that must survive proxy unwrapping."""

    value: int
    detail: str


class ModelRoot(BaseDataClass):
    """Root accepting a model instance with additional subtype fields."""

    item: Annotated[BaseDataClass, BaseDataClass.validator()]


class Root(BaseDataClass):
    """Root with a metadata-annotated complex field."""

    branch: Annotated[Branch, Branch.validator(('@key',))]


class ListedRoot(BaseDataClass):
    """Root with a metadata validator around the entire list."""

    branches: Annotated[list[Branch], Branch.validator(('@key',))]


ModelLoader = Callable[[type[BaseDataClass], dict[str, Any]], BaseDataClass]


@pytest.fixture(params=[
    pytest.param(lambda cls, data: cls(**data), id='constructor'),
    pytest.param(lambda cls, data: cls.model_validate(data), id='model_validate'),
    pytest.param(lambda cls, data: ObjectBuilder(cls, **data).to_model(),
                 id='object_builder'),
])
def load_model(request: pytest.FixtureRequest) -> ModelLoader:
    """Load values through the public APIs used by generated functions."""
    return request.param


def _branch_data() -> dict[str, Any]:
    return {
        '@key': 'branch-1',
        'leaf': {
            '@key': 'leaf-1',
            'value': 1,
            'date': {'@data': '2024-01-15', '@scheme': 'date-scheme'},
            'currency': {'@data': 'USD', '@scheme': 'currency-scheme'},
        },
    }


@pytest.mark.parametrize('wrapped', [False, True], ids=['dict', 'cow_dict'])
def test_nested_metadata_survives_repeated_validation(
    load_model: ModelLoader, wrapped: bool,
) -> None:
    """Every nested validator must preserve its input dictionary."""
    data = _branch_data()
    original = deepcopy(data)
    value = rune_cow(data) if wrapped else data

    for _ in range(2):
        result = load_model(Root, {'branch': value})
        assert result.branch.get_meta('key') == 'branch-1'
        assert result.branch.leaf.get_meta('key') == 'leaf-1'
        assert result.branch.leaf.date == date(2024, 1, 15)
        assert result.branch.leaf.date.get_meta('scheme') == 'date-scheme'
        assert result.branch.leaf.currency == Currency.USD
        assert result.branch.leaf.currency.get_meta('scheme') == 'currency-scheme'
        assert data == original
        assert rune_unwrap(value) == original


def test_cow_dictionary_uses_current_shadow(load_model: ModelLoader) -> None:
    """Use changes made through the proxy without consuming either payload."""
    data = _branch_data()
    original = deepcopy(data)
    proxy = rune_cow(data)
    proxy['leaf']['value'] = 2
    shadow = deepcopy(rune_unwrap(proxy))

    result = load_model(Root, {'branch': proxy})

    assert result.branch.leaf.value == 2
    assert data == original
    assert rune_unwrap(proxy) == shadow


@pytest.mark.parametrize('wrapped', [False, True], ids=['list', 'cow_list'])
def test_list_preserves_nested_metadata(
    load_model: ModelLoader, wrapped: bool,
) -> None:
    """List traversal must apply non-destructive extraction at every level."""
    data = [_branch_data()]
    original = deepcopy(data)
    value = rune_cow(data) if wrapped else data

    for _ in range(2):
        result = load_model(ListedRoot, {'branches': value})
        assert result.branches[0].leaf.value == 1
        assert result.branches[0].get_meta('key') == 'branch-1'
        assert result.branches[0].leaf.get_meta('key') == 'leaf-1'
        assert data == original
        assert rune_unwrap(value) == original


@pytest.mark.parametrize('changed', [False, True], ids=['original', 'shadow'])
def test_cow_model_retains_subtype_and_identity(
    load_model: ModelLoader, changed: bool,
) -> None:
    """Unwrap model proxies without copying or rebuilding their targets."""
    original = DetailedValue(value=1, detail='subtype value')
    proxy = rune_cow(original)
    if changed:
        proxy.value = 2
    current = rune_unwrap(proxy)

    result = load_model(ModelRoot, {'item': proxy})

    assert result.item is current
    assert isinstance(result.item, DetailedValue)
    assert result.item.detail == 'subtype value'
    assert result.item.value == (2 if changed else 1)
    assert original.value == 1


def test_cow_dictionary_does_not_require_deepcopy() -> None:
    """Metadata extraction must accept data that cannot be deep-copied."""
    class NoDeepCopyDict(dict):
        """Dictionary that rejects an unnecessary deep copy."""

        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            raise AssertionError('Deserialization must not deep-copy this input')

    data = NoDeepCopyDict(_branch_data())

    result = Root(branch=rune_cow(data))

    assert result.branch.leaf.value == 1
    assert data == _branch_data()


@pytest.mark.parametrize('wrapped', [False, True], ids=['dict', 'cow_dict'])
def test_validation_failure_preserves_metadata(
    load_model: ModelLoader, wrapped: bool,
) -> None:
    """Validation errors must also leave nested input metadata intact."""
    data = _branch_data()
    data['leaf']['value'] = 'invalid'
    original = deepcopy(data)
    value = rune_cow(data) if wrapped else data

    with pytest.raises(ValidationError):
        load_model(Root, {'branch': value})

    assert data == original
    assert rune_unwrap(value) == original


@pytest.mark.parametrize('field', ['date', 'currency'])
def test_missing_data_key_still_raises(field: str) -> None:
    """Missing @data must retain its existing error without consuming input."""
    data = _branch_data()
    data['leaf'][field].pop('@data')
    original = deepcopy(data)

    with pytest.raises(KeyError, match='@data'):
        Root(branch=data)

    assert data == original


def test_basic_type_and_enum_accept_cow_dictionaries() -> None:
    """Basic and enum dictionary proxies must use the same extraction rules."""
    date_data = {'@data': '2024-01-15', '@scheme': 'date-scheme'}
    currency_data = {'@data': 'USD', '@scheme': 'currency-scheme'}
    original_date = dict(date_data)
    original_currency = dict(currency_data)

    result = Leaf(value=1, date=rune_cow(date_data),
                  currency=rune_cow(currency_data))

    assert result.date == date(2024, 1, 15)
    assert result.date.get_meta('scheme') == 'date-scheme'
    assert result.currency == Currency.USD
    assert result.currency.get_meta('scheme') == 'currency-scheme'
    assert date_data == original_date
    assert currency_data == original_currency


@replaceable
@validate_call
def _construct_root(branch: Branch) -> Root:
    """Reproduce a generated Rune function constructing a complex result."""
    branch = rune_cow(branch)
    return Root(branch=branch)


def test_generated_constructor_accepts_cow_parameter() -> None:
    """A generated-style function must accept its COW-wrapped parameter."""
    branch = Branch.model_validate(_branch_data())

    result = _construct_root(branch)

    assert isinstance(result, Root)
    assert result.branch.leaf.value == 1
