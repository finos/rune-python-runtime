# Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Annotated
from pydantic import Field, validate_call
from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.func_proxy import replaceable
from rune.runtime.metadata import IntWithMeta, Reference
from rune.runtime.utils import rune_resolve_attr

class rosetta_dsl_test_functions_KeyEntity(BaseDataClass):
    _ALLOWED_METADATA = {'@key', '@key:external'}
    _FQRTN = 'rosetta_dsl.test.functions.KeyEntity'
    value: int = Field(..., description='')


KeyEntity = rosetta_dsl_test_functions_KeyEntity


# class rosetta_dsl_test_functions_RefEntity(BaseDataClass):
#     _FQRTN = 'rosetta_dsl.test.functions.RefEntity'
#     ke: Annotated[rosetta_dsl_test_functions_KeyEntity, rosetta_dsl_test_functions_KeyEntity.serializer(), rosetta_dsl_test_functions_KeyEntity.validator()] = Field(..., description='')

class rosetta_dsl_test_functions_RefEntity(BaseDataClass):
    _FQRTN = 'rosetta_dsl.test.functions.RefEntity'
    ke: Annotated[rosetta_dsl_test_functions_KeyEntity, rosetta_dsl_test_functions_KeyEntity.serializer(), rosetta_dsl_test_functions_KeyEntity.validator(('@key', '@key:external', '@ref', '@ref:external'))] = Field(..., description='')

    _KEY_REF_CONSTRAINTS = {
        'ke': {'@key', '@key:external', '@ref', '@ref:external'}
    }

RefEntity = rosetta_dsl_test_functions_RefEntity

@replaceable
@validate_call
def MetadataFunction(ref: rosetta_dsl_test_functions_RefEntity) -> int:
    """

    Parameters
    ----------
    ref : rosetta_dsl.test.functions.RefEntity

    Returns
    -------
    result : int

    """
    self = inspect.currentframe()
    result = rune_resolve_attr(rune_resolve_attr(rune_resolve_attr(self, "ref"), "ke"), "value")
    return result  # type: ignore


def test_extra_meta():
    key_entity = KeyEntity(value=IntWithMeta(value=5))
    key_entity.validate_attribs()
    ref_entity = RefEntity(ke=Reference(target=key_entity, ext_key="key-123"))  # type: ignore
    ref_entity.validate_attribs()
    assert MetadataFunction(ref=ref_entity) == 5
