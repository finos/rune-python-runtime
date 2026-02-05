'''test module for rune root lifecycle'''
from typing import Optional, Annotated
from pydantic import Field
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


class Bplus(BaseDataClass):
    '''no doc'''
    bAddress: Optional[Annotated[B,
                                 B.serializer(),
                                 B.validator(('@ref:scoped', ))]] = Field(
                                     None, description='')

    _KEY_REF_CONSTRAINTS = {
        'bAddress': {'@ref:scoped'}
    }


class RootDeep(BaseDataClass):
    '''no doc'''
    typeA: Optional[Annotated[A, A.serializer(),
                              A.validator()]] = Field(None, description='')
    bplus: Optional[Annotated[Bplus,
                              Bplus.serializer(),
                              Bplus.validator()]] = Field(None, description='')


class DeepRef(BaseDataClass):
    '''no doc'''
    _FQRTN = 'test_rune_parent.DeepRef'
    root: Annotated[Root, Root.serializer(),
                    Root.validator()] = Field(..., description='')


class DeepRef2(BaseDataClass):
    '''no doc'''
    root: Annotated[Root, Root.serializer(),
                    Root.validator()] = Field(..., description='')
    root2: Annotated[Root, Root.serializer(),
                    Root.validator()] = Field(..., description='')


def test_root_creation():
    '''no doc'''
    b = B(fieldB='some b content')
    a = A(b=b)
    root = Root(typeA=a, bAddress=Reference(a.b, 'aKey3', KeyType.SCOPED))
    # pylint: disable=no-member
    assert root.get_rune_parent() is None
    assert root == root.typeA.get_rune_parent()
    assert root.typeA == root.typeA.b.get_rune_parent()
    assert root.typeA == root.bAddress.get_rune_parent()
    assert root.typeA.b == root.bAddress


def test_deep_creation():
    '''no doc'''
    b = B(fieldB='some b content')
    a = A(b=b)
    root = Root(typeA=a, bAddress=Reference(a.b, 'aKey3', KeyType.SCOPED))
    deep = DeepRef(root=root)
    # pylint: disable=no-member
    assert deep.get_rune_parent() is None
    assert deep == deep.root.get_rune_parent()
    assert deep.root == deep.root.typeA.get_rune_parent()
    assert deep.root.typeA == deep.root.typeA.b.get_rune_parent()
    assert deep.root.typeA == deep.root.bAddress.get_rune_parent()
    assert deep.root.typeA.b == deep.root.bAddress


def test_deep2_creation(mocker):
    '''no doc'''
    mocker.patch('rune.runtime.metadata.BaseMetaDataMixin._DEFAULT_SCOPE_TYPE',
                 'test.test_rune_parent.Root')
    b = B(fieldB='some b content')
    a = A(b=b)
    b2 = B(fieldB='2 some other b content')
    a2 = A(b=b2)
    root = Root(typeA=a, bAddress=Reference(a.b, 'aKey3', KeyType.SCOPED))
    root2 = Root(typeA=a2, bAddress=Reference(a2.b, 'aKey3', KeyType.SCOPED))
    deep = DeepRef2(root=root, root2=root2)
    # pylint: disable=no-member
    assert deep.get_rune_parent() is None
    assert deep == deep.root.get_rune_parent()
    assert deep.root == deep.root.typeA.get_rune_parent()
    assert deep.root.typeA == deep.root.typeA.b.get_rune_parent()
    assert deep.root.typeA == deep.root.bAddress.get_rune_parent()
    assert deep.root.typeA.b == deep.root.bAddress

    assert deep == deep.root2.get_rune_parent()
    assert deep.root2 == deep.root2.typeA.get_rune_parent()
    assert deep.root2.typeA == deep.root2.typeA.b.get_rune_parent()
    assert deep.root2.typeA == deep.root2.bAddress.get_rune_parent()
    assert deep.root2.typeA.b == deep.root2.bAddress


def test_root_deserialization():
    '''no doc'''
    rune_dict = {
        "bAddress": {
            "@ref:scoped": "aKey3"
        },
        "typeA": {
            "b": {
                "@key:scoped": "aKey3",
                "fieldB": "some b content"
            }
        },
    }
    root = Root.model_validate(rune_dict)
    assert root.get_rune_parent() is None
    assert root == root.typeA.get_rune_parent()
    assert root.typeA == root.typeA.b.get_rune_parent()
    assert root.typeA == root.bAddress.get_rune_parent()
    assert root.typeA.b == root.bAddress


def test_root_deep_deserialization():
    '''no doc'''
    rune_dict = {
        "bplus": {
            "bAddress": {
                "@ref:scoped": "aKey3"
            }
        },
        "typeA": {
            "b": {
                "@key:scoped": "aKey3",
                "fieldB": "some b content"
            }
        },
        # "bplus": {
        #     "bAddress": {
        #         "@ref:scoped": "aKey3"
        #     }
        # },
    }
    root = RootDeep.rune_deserialize(rune_dict)
    assert root.get_rune_parent() is None
    assert root == root.typeA.get_rune_parent()
    assert root.typeA == root.typeA.b.get_rune_parent()
    assert root.typeA == root.bplus.bAddress.get_rune_parent()
    assert root.typeA.b == root.bplus.bAddress


def test_deep_deserialization():
    '''no doc'''
    rune_dict = {
        "root": {
            "bAddress": {
                "@ref:scoped": "aKey3"
            },
            "typeA": {
                "b": {
                    "@key:scoped": "aKey3",
                    "fieldB": "some b content"
                }
            },
        }
    }
    deep = DeepRef.model_validate(rune_dict)
    assert deep.get_rune_parent() is None
    assert deep == deep.root.get_rune_parent()
    assert deep.root == deep.root.typeA.get_rune_parent()
    assert deep.root.typeA == deep.root.typeA.b.get_rune_parent()
    assert deep.root.typeA == deep.root.bAddress.get_rune_parent()
    assert deep.root.typeA.b == deep.root.bAddress

# EOF
