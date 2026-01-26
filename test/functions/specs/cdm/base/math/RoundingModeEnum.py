# pylint: disable=missing-module-docstring, invalid-name, line-too-long
from enum import Enum
import rune.runtime.metadata


class RoundingModeEnum(rune.runtime.metadata.EnumWithMetaMixin, Enum):
    """
    The enumerated values to specify the rounding direction when rounding of a number to nearest.  Used by function cdm.base.math.RoundToNearest.
    """
    DOWN = "Down"
    """
    A number will be rounded down to the specified nearest number. For example, 529 rounded down to the nearest 10 is 520.
    """
    UP = "Up"
    """
    A number will be rounded up to the specified nearest number. For example, 521 rounded up to the nearest 10 is 530.
    """
