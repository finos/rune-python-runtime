''' Signature validation, and replaceable function overrides tests.'''
from decimal import Decimal
import pytest
from pydantic import ValidationError
from rune.runtime.func_proxy import scoped_replace
from rune.runtime.native_registry import rune_register_native
from rune.runtime.native_registry import rune_deregister_native
from test.functions.specs.cdm.base.math.RoundToNearest import RoundToNearest
from test.functions.specs.cdm.base.math.RoundingModeEnum import RoundingModeEnum

import test.functions.specs.cdm.base.math.RoundToNearest as mod_RoundToNearest


def _echo_fn(value, nearest, roundingMode):
    return value, nearest, roundingMode


def _native_round(value, nearest, roundingMode):
    return Decimal(round(value, int(nearest)))


@pytest.fixture
def setup_native():
    rune_register_native('cdm.base.math.RoundToNearest', _native_round)
    yield
    rune_deregister_native('cdm.base.math.RoundToNearest')


def test_register_native():
    try:
        rune_register_native('cdm.base.math.RoundToNearest', _native_round)
        x = RoundToNearest(Decimal(4.59989), Decimal(1), RoundingModeEnum.UP)
        assert x == Decimal('4.6')
    finally:
        rune_deregister_native('cdm.base.math.RoundToNearest')


def test_no_native():
    with pytest.raises(NotImplementedError):
        RoundToNearest(Decimal(4.59989), Decimal(1), RoundingModeEnum.UP)


def test_bad_native():
    with pytest.raises(TypeError):
        rune_register_native('cdm.base.math.RoundToNearest',
                             'just a string')  # type: ignore


@pytest.mark.usefixtures("setup_native")
def test_simple_types():
    ''' Test that valid parameters are accepted and the return contains the
        expected rounded value.
     '''
    x = RoundToNearest(Decimal(4.59989), Decimal(1), RoundingModeEnum.UP)
    assert x == Decimal('4.6')


@pytest.mark.usefixtures("setup_native")
def test_validation_failure():
    '''Test that invalid parameter types raise ValidationError.'''
    with pytest.raises(ValidationError):
        RoundToNearest(
            'some text',  # type: ignore
            Decimal(1),
            RoundingModeEnum.UP)


def test_direct_repalce():
    '''Test direct replacement of the replaceable target.'''
    proxy = mod_RoundToNearest.RoundToNearest.__proxy__  # type: ignore
    original = proxy.func
    try:
        mod_RoundToNearest.RoundToNearest = _echo_fn
        par = 'a', 'b', 'c'
        assert par == RoundToNearest(*par)  # type: ignore
    finally:
        proxy.func = original


@pytest.mark.usefixtures("setup_native")
def test_scoped_replace():
    ''' Test scoped_replace temporarily overrides and then restores the
        replaceable function.
    '''
    with scoped_replace(RoundToNearest, _echo_fn):
        # test function was replaced
        par = 'a', 'b', 'c'
        assert par == RoundToNearest(*par)  # type: ignore
    # test the function was restored
    test_simple_types()

# EOF
