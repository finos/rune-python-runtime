''' Signature validation, and replaceable function overrides tests.'''
from decimal import Decimal
import pytest
from pydantic import ValidationError
from rune.runtime.func_proxy import scoped_replace
from test.functions.specs.RoundToNearest import RoundToNearest
from test.functions.specs.RoundingModeEnum import RoundingModeEnum
import test.functions.specs.RoundToNearest as mod_RoundToNearest


def test_simple_types():
    ''' Test that valid parameters are accepted and the return contains the
        expected rounded value.
     '''
    x = RoundToNearest(Decimal(4.59989), Decimal(1), RoundingModeEnum.UP)
    assert x == Decimal('4.6')


def test_validation_failure():
    '''Test that invalid parameter types raise ValidationError.'''
    with pytest.raises(ValidationError):
        RoundToNearest(
            'some text',  # type: ignore
            Decimal(1),
            RoundingModeEnum.UP)


def _echo_fn(value, nearest, roundingMode):
    return value, nearest, roundingMode


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
