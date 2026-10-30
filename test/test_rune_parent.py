'''test module for rune root lifecycle'''
import copy
from enum import Enum
from typing import Optional, Annotated, ClassVar
from pydantic import Field
from rune.runtime.cow import _clone_for_write, rune_cow, rune_unwrap
from rune.runtime.metadata import (
    DateTimeWithMeta, DateWithMeta, EnumWithMetaMixin, IntWithMeta, KeyType,
    NumberWithMeta, Reference, StrWithMeta, TimeWithMeta, _EnumWrapper,
)
from rune.runtime.base_data_class import BaseDataClass


class B(BaseDataClass):
    '''no doc'''
    _FQRTN: ClassVar[str] = 'test_rune_parent.B'
    fieldB: str = Field(..., description='')


class A(BaseDataClass):
    '''no doc'''
    _FQRTN: ClassVar[str] = 'test_rune_parent.A'
    b: Annotated[B, B.serializer(),
                 B.validator(('@key:scoped', ))] = Field(..., description='')


class Root(BaseDataClass):
    '''no doc'''
    _FQRTN: ClassVar[str] = 'test.test_rune_parent.Root'

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
    _FQRTN: ClassVar[str] = 'test_rune_parent.RootDeep'
    typeA: Optional[Annotated[A, A.serializer(),
                              A.validator()]] = Field(None, description='')
    bplus: Optional[Annotated[Bplus,
                              Bplus.serializer(),
                              Bplus.validator()]] = Field(None, description='')


class DeepRef(BaseDataClass):
    '''no doc'''
    _FQRTN: ClassVar[str] = 'test_rune_parent.DeepRef'
    root: Annotated[Root, Root.serializer(),
                    Root.validator()] = Field(..., description='')


class DeepRef2(BaseDataClass):
    '''no doc'''
    _FQRTN: ClassVar[str] = 'test_rune_parent.DeepRef2'
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


class ChainNode(BaseDataClass):
    '''Recursive chain node for testing parent-detachment on deep graphs.'''
    _FQRTN: ClassVar[str] = 'test_rune_parent.ChainNode'
    label: str = Field(..., description='')
    child: Optional['ChainNode'] = Field(None, description='')


ChainNode.model_rebuild()


class SampleEnum(EnumWithMetaMixin, Enum):
    '''Enum with metadata for parent-detachment tests.'''
    ALPHA = 'ALPHA'


def test_clone_for_write_detaches_rune_parent():
    '''_clone_for_write(child) should produce a detached copy with no
    rune_parent.'''
    parent = ChainNode.model_construct(label='parent', child=None)
    child = ChainNode.model_construct(label='child', child=None)
    child._set_rune_parent(parent)  # pylint: disable=protected-access

    assert child.get_rune_parent() is parent

    clone = _clone_for_write(child)

    assert clone is not child
    assert clone.get_rune_parent() is None
    assert child.get_rune_parent() is parent


def test_model_copy_and_deepcopy_detach_subtree_root_and_preserve_descendants():
    '''Copying a subtree detaches its root while re-parenting its descendants
    to the cloned subtree.'''
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
    deep = DeepRef.model_validate({"root": rune_dict})

    # Deep copy of intermediate subtree (root inside deep)
    root_clone = deep.root.model_copy(deep=True)
    assert root_clone is not deep.root
    assert root_clone.get_rune_parent() is None
    assert root_clone.typeA is not deep.root.typeA
    assert root_clone.typeA.get_rune_parent() is root_clone
    assert root_clone.typeA.b is not deep.root.typeA.b
    assert root_clone.typeA.b.get_rune_parent() is root_clone.typeA
    assert root_clone.bAddress is root_clone.typeA.b
    assert root_clone.bAddress.get_rune_parent() is root_clone.typeA

    # Shallow copy also detaches the copied root from its parent
    shallow_clone = deep.root.model_copy(deep=False)
    assert shallow_clone.get_rune_parent() is None
    assert copy.copy(deep.root.typeA).get_rune_parent() is None

    # model_copy(deep=True, update=...) detaches root and preserves child parent
    updated_clone = deep.root.model_copy(deep=True, update={'bAddress': None})
    assert updated_clone.get_rune_parent() is None
    assert updated_clone.bAddress is None
    assert updated_clone.typeA.get_rune_parent() is updated_clone
    assert updated_clone.typeA.b.get_rune_parent() is updated_clone.typeA


def test_deep_chain_clone_for_write_does_not_walk_ancestors():
    '''Cloning a leaf in a deep parent chain must not walk up __rune_parent or
    hit RecursionError.'''
    depth = 200
    nodes: list[ChainNode] = []
    current = ChainNode(label=f'node-{depth}', child=None)
    nodes.append(current)
    for idx in range(depth - 1, -1, -1):
        parent = ChainNode.model_construct(label=f'node-{idx}', child=current)
        current._set_rune_parent(parent)  # pylint: disable=protected-access
        nodes.append(parent)
        current = parent

    leaf = nodes[0]
    assert leaf.get_rune_parent() is nodes[1]

    copied_ids: set[int] = set()
    orig_deepcopy = ChainNode.__deepcopy__

    def tracked_deepcopy(self, memo=None):
        copied_ids.add(id(self))
        return orig_deepcopy(self, memo)

    ChainNode.__deepcopy__ = tracked_deepcopy  # type: ignore[method-assign]
    try:
        leaf_clone = _clone_for_write(leaf)
    finally:
        ChainNode.__deepcopy__ = orig_deepcopy  # type: ignore[method-assign]

    assert leaf_clone.get_rune_parent() is None
    assert leaf_clone.label == f'node-{depth}'
    # Only the leaf itself should have been deep-copied, never its 200 ancestors
    assert copied_ids == {id(leaf)}


def test_metadata_wrappers_detach_parent_on_direct_copy():
    '''Direct copy and deepcopy of basic and enum metadata wrappers must not
    retain an uncopied parent reference.'''
    parent = B(fieldB='parent')
    wrappers = [
        StrWithMeta('hello', scheme='s'),
        IntWithMeta(7, scheme='s'),
        NumberWithMeta('3.14', scheme='s'),
        DateWithMeta('2026-10-30', scheme='s'),
        TimeWithMeta('12:30:00', scheme='s'),
        DateTimeWithMeta('2026-10-30T12:30:00', scheme='s'),
        _EnumWrapper(SampleEnum.ALPHA),
    ]
    for wrapper in wrappers:
        wrapper._set_rune_parent(parent)  # pylint: disable=protected-access
        assert wrapper.get_rune_parent() is parent

        shallow = copy.copy(wrapper)
        deep = copy.deepcopy(wrapper)
        assert shallow.get_rune_parent() is None
        assert deep.get_rune_parent() is None


def test_cow_write_on_child_produces_detached_shadow_rebound_to_parent():
    '''Mutating a nested child through rune_cow detaches from the original
    parent and binds the shadow child to the shadow parent.'''
    root = Root(typeA=A(b=B(fieldB='initial')))
    proxy = rune_cow(root)
    proxy.typeA.b.fieldB = 'updated'
    result = rune_unwrap(proxy)

    assert root.typeA.b.fieldB == 'initial'
    assert root.typeA.get_rune_parent() is root
    assert root.typeA.b.get_rune_parent() is root.typeA

    assert result.get_rune_parent() is None
    assert result.typeA.get_rune_parent() is result
    assert result.typeA.b.get_rune_parent() is result.typeA
    assert result.typeA.b.fieldB == 'updated'

# EOF
