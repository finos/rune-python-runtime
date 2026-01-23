from decimal import Decimal
import pytest
from pydantic import ValidationError
from test.functions.specs.RoundToNearest import RoundToNearest
from test.functions.specs.RoundingModeEnum import RoundingModeEnum


def test_simple_types():
    x = RoundToNearest(Decimal(4.59989), Decimal(1), RoundingModeEnum.UP)
    assert x == Decimal('4.6')

    with pytest.raises(ValidationError):
        x = RoundToNearest(
            'some text',  # type: ignore
            Decimal(1),
            RoundingModeEnum.UP)

# EOF
