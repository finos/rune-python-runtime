'''Full attribute validation - pydantic and constraints'''
from pathlib import Path
import pytest
from pydantic import ValidationError
from rune.runtime.base_data_class import BaseDataClass
try:
    # pylint: disable=unused-import
    # type: ignore
    from cdm.base.math.NonNegativeQuantity import NonNegativeQuantity
    from cdm.base.math.UnitType import UnitType
    NO_SER_TEST_MOD = False
except ImportError:
    NO_SER_TEST_MOD = True


@pytest.mark.skipif(NO_SER_TEST_MOD, reason='CDM package not found')
def test_bad_attrib_validation():
    '''Invalid attribute assigned'''
    unit = UnitType(currency='EUR')
    mq = NonNegativeQuantity(value=10, unit=unit)
    mq.frequency = 'Blah'
    with pytest.raises(ValidationError):
        mq.validate_model()


def test_vanilla_swap_list_reference_validation():
    '''Validate the reported sample when local data and the CDM bundle exist.'''
    sample = Path(__file__).resolve().parents[1] / 'local_data' / 'ird-ex01-vanilla-swap-versioned.json'
    if not sample.is_file():
        pytest.skip('Local CDM sample not found')
    pytest.importorskip('finos')

    model = BaseDataClass.rune_deserialize(
        sample.read_text(encoding='utf-8'),
        namespace_prefix='finos',
        check_rune_constraints=False,
    )

    assert len(model.trade.counterparty) == 2
    parties = {party.get_meta('@key:external'): party for party in model.trade.party}
    for counterparty in model.trade.counterparty:
        key = counterparty.resolve_ref_key('partyReference')
        assert counterparty.partyReference is parties[key]
    assert model.validate_model(check_rune_constraints=False) == []

# EOF
