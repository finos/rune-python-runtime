# pylint: disable=line-too-long, invalid-name, missing-function-docstring, missing-module-docstring, superfluous-parens
# pylint: disable=wrong-import-position, unused-import, unused-wildcard-import, wildcard-import, wrong-import-order, missing-class-docstring
from __future__ import annotations
import sys
import inspect
from decimal import Decimal
from pydantic import validate_call
from rune.runtime.utils import if_cond_fn, rune_all_elements, rune_resolve_attr
from rune.runtime.conditions import rune_execute_local_conditions, rune_local_condition
from rune.runtime.func_proxy import replaceable, create_module_attr_guardian
from rune.runtime.native_registry import rune_execute_native
# from .RoundingModeEnum import RoundingModeEnum
import test.functions.specs.cdm.base.math.RoundingModeEnum


@replaceable
@validate_call
def cdm_base_math_RoundToNearest(value: Decimal, nearest: Decimal, roundingMode: test.functions.specs.cdm.base.math.RoundingModeEnum.RoundingModeEnum) -> Decimal:
    """
    Round a number to the supplied nearest, using the supplied rounding mode.
    
    Parameters 
    ----------
    value : number
    The original (unrounded) number.
    
    nearest : number
    The nearest number to round to.
    
    roundingMode : RoundingModeEnum
    The method of rounding (up to nearest/down to nearest).
    
    Returns
    -------
    roundedValue : number
    
    """
    _pre_registry = {}
    self = inspect.currentframe()
    
    # conditions
    
    @rune_local_condition(_pre_registry)
    def condition_0_PositiveNearest():  #FIXME: here we should NOT generate cond_name(self) as it is a closure function
        return rune_all_elements(rune_resolve_attr(self, "nearest"), ">", 0)
    # Execute all registered conditions
    rune_execute_local_conditions(_pre_registry, 'Pre-condition')  #FIXME: the generator needs to add the rine_ prefix
    
    roundedValue = rune_execute_native('cdm.base.math.RoundToNearest', value, nearest, roundingMode)
    
    return roundedValue



@replaceable
@validate_call
def cdm_base_math_Abs(arg: Decimal) -> Decimal:
    """
    Returns the absolute value of a number. If the argument is not negative, the argument is returned. If the argument is negative, the negation of the argument is returned.

    Parameters 
    ----------
    arg : number

    Returns
    -------
    result : number

    """
    self = inspect.currentframe()


    def _then_fn0():
        return (-1 * rune_resolve_attr(self, "arg"))

    def _else_fn0():
        return rune_resolve_attr(self, "arg")

    result =  if_cond_fn(rune_all_elements(rune_resolve_attr(self, "arg"), "<", 0), _then_fn0, _else_fn0)


    return result


sys.modules[__name__].__class__ = create_module_attr_guardian(sys.modules[__name__].__class__)
