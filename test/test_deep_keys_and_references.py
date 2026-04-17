'''test module for ref lifecycle'''
from typing import Optional, Annotated
from pydantic import Field
import pytest
from rune.runtime.metadata import Reference, KeyType
from rune.runtime.base_data_class import BaseDataClass


class B(BaseDataClass):
    '''no doc'''
    fieldB: str = Field(..., description='')


class A(BaseDataClass):
    '''no doc'''
    b: Annotated[B, B.serializer(),
                 B.validator(('@key:scoped', ))] = Field(..., description='')


class Root(BaseDataClass):
    '''no doc'''
    typeA: Optional[Annotated[A, A.serializer(),
                              A.validator()]] = Field(None, description='')
    bAddress: Optional[Annotated[B,
                                 B.serializer(),
                                 B.validator(('@ref:scoped', ))]] = Field(
                                     None, description='')
    _KEY_REF_CONSTRAINTS = {
        'bAddress': {'@ref:scoped'}
    }

class DeepRef(BaseDataClass):
    '''no doc'''
    root: Annotated[Root, Root.serializer(),
                    Root.validator()] = Field(..., description='')


def test_ref_creation():
    '''no doc'''
    b = B(fieldB='some b content')
    a = A(b=b)
    root = Root(typeA=a, bAddress=Reference(a.b, 'aKey', KeyType.SCOPED))
    # pylint: disable=no-member
    assert id(root.typeA.b) == id(root.bAddress)


def test_deep_ref_creation():
    '''no doc'''
    b = B(fieldB='some b content')
    a = A(b=b)
    root = Root(typeA=a, bAddress=Reference(a.b, 'aKey2', KeyType.SCOPED))
    deep_ref = DeepRef(root=root)
    # pylint: disable=no-member
    assert id(deep_ref.root.typeA.b) == id(deep_ref.root.bAddress)


def test_fail_wrong_key_ext():
    '''no doc'''
    b = B(fieldB='some b content')
    a = A(b=b)
    with pytest.raises(ValueError):
        Root(typeA=a, bAddress=Reference(a.b, 'aKey', KeyType.EXTERNAL))


def test_fail_wrong_key_int():
    '''no doc'''
    b = B(fieldB='some b content')
    a = A(b=b)
    with pytest.raises(ValueError):
        Root(typeA=a, bAddress=Reference(a.b))


def test_scoped_reference_metadata_and_type():
    '''scoped refs should store key type and metadata'''
    b = B(fieldB='some b content')
    a = A(b=b)
    root = Root(typeA=a, bAddress=Reference(a.b, 'aKeyScoped', KeyType.SCOPED))
    assert b.get_meta('@key:scoped') == 'aKeyScoped'
    assert root.__dict__['__rune_references']['bAddress'][1] == KeyType.SCOPED


def test_ref_prefers_scoped_over_internal():
    '''scoped ref should be preferred when both tags are provided'''
    rune_dict = {
        "bAddress": {
            "@ref": "internalKey",
            "@ref:scoped": "scopedKey"
        },
        "typeA": {
            "b": {
                "@key:scoped": "scopedKey",
                "fieldB": "some b content"
            }
        },
    }
    root = Root.model_validate(rune_dict)
    assert root.bAddress == root.typeA.b
    assert root.__dict__['__rune_references']['bAddress'][1] == KeyType.SCOPED


def test_invalid_multiple_ref_tags_raise():
    '''unknown multiple ref tags should raise'''
    rune_dict = {
        "bAddress": {
            "@ref:foo": "key1",
            "@ref:bar": "key2"
        }
    }
    with pytest.raises(ValueError):
        Root.model_validate(rune_dict)


def test_scoped_key_not_visible_outside_scope(mocker):
    '''scoped keys should not leak across scope instances'''
    mocker.patch('rune.runtime.metadata.BaseMetaDataMixin._DEFAULT_SCOPE_TYPE',
                 'test_deep_keys_and_references.Root')
    b = B(fieldB='some b content')
    a = A(b=b)
    Root(typeA=a, bAddress=Reference(a.b, 'aKeyScoped', KeyType.SCOPED))
    root2 = Root.model_validate({"bAddress": {"@ref:scoped": "aKeyScoped"}})
    with pytest.raises(KeyError):
        root2.resolve_references(ignore_dangling=False, recurse=False)

# EOF
