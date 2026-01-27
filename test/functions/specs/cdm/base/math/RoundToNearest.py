# pylint: disable=missing-module-docstring, invalid-name, line-too-long
import sys

from rune.runtime.func_proxy import create_module_attr_guardian
from test.functions.specs.cdm._bundle import cdm_base_math_RoundToNearest as RoundToNearest

sys.modules[__name__].__class__ = create_module_attr_guardian(sys.modules[__name__].__class__)
