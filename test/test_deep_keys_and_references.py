'''test module for ref lifecycle'''
from collections import UserList
from typing import Optional, Annotated
from pydantic import Field
import pytest
from rune.runtime.metadata import Reference, KeyType, UnresolvedReference
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


class ListedReferences(BaseDataClass):
    '''References and their targets occur in separate list fields.'''
    references: list[Annotated[Root, Root.validator()]]
    targets: list[Annotated[A, A.validator()]]


class NestedListedReferences(BaseDataClass):
    '''Exercise lists below both model fields and other lists.'''
    groups: list[Annotated[ListedReferences, ListedReferences.validator()]]
    labels: list[str] = Field(default_factory=list)


class SequenceContainer(BaseDataClass):
    '''Keep sequence implementations intact during validation.'''
    entries: object


class CyclicReference(BaseDataClass):
    '''A resolved reference can point back to its containing object.'''
    _ALLOWED_METADATA = {'@key'}
    target: Annotated[BaseDataClass, BaseDataClass.validator(('@ref', ))]
    _KEY_REF_CONSTRAINTS = {'target': {'@ref'}}


@pytest.fixture
def listed_reference_data():
    '''A forward reference to a target in a sibling list.'''
    return {
        'groups': [{
            'references': [
                {'bAddress': {'@ref:scoped': 'first'}},
                {'bAddress': {'@ref:scoped': 'second'}},
            ],
            'targets': [
                {'b': {'@key:scoped': 'first', 'fieldB': 'first target'}},
                {'b': {'@key:scoped': 'second', 'fieldB': 'second target'}},
            ],
        }],
        'labels': ['primitive list entries are ignored'],
    }


def test_list_children_share_parent_and_key_maps(listed_reference_data):
    '''Keys from list children must be visible to sibling list entries.'''
    model = NestedListedReferences.model_validate(listed_reference_data)
    group = model.groups[0]

    assert group.get_rune_parent() is model
    for reference, target in zip(group.references, group.targets):
        assert reference.get_rune_parent() is group
        assert target.get_rune_parent() is group
        key = target.b.get_meta('@key:scoped')
        assert reference.get_object_by_key(key, KeyType.SCOPED) is target.b


def test_deserialize_and_validate_references_in_nested_lists(listed_reference_data):
    '''Rune deserialization resolves every list entry before validation.'''
    model = NestedListedReferences.rune_deserialize(listed_reference_data)
    group = model.groups[0]

    for reference, target in zip(group.references, group.targets):
        assert reference.bAddress is target.b
        assert reference.resolve_ref_key('bAddress') == target.b.get_meta('@key:scoped')
    assert model.validate_model() == []

    model.resolve_references()
    for reference, target in zip(group.references, group.targets):
        assert reference.bAddress is target.b


def test_resolve_list_references_respects_recurse_false(listed_reference_data):
    '''Nonrecursive resolution leaves child references untouched.'''
    model = NestedListedReferences.model_validate(listed_reference_data)

    model.resolve_references(recurse=False)

    assert all(isinstance(ref.bAddress, UnresolvedReference)
               for ref in model.groups[0].references)


def test_resolve_list_references_respects_ignore_dangling(listed_reference_data):
    '''Missing list references may be ignored or reported by the caller.'''
    listed_reference_data['groups'][0]['references'][1]['bAddress'] = {
        '@ref:scoped': 'missing-key'
    }
    model = NestedListedReferences.model_validate(listed_reference_data)

    model.resolve_references(ignore_dangling=True)

    group = model.groups[0]
    assert group.references[0].bAddress is group.targets[0].b
    assert isinstance(group.references[1].bAddress, UnresolvedReference)
    with pytest.raises(KeyError, match='missing-key'):
        model.resolve_references(ignore_dangling=False)


@pytest.mark.parametrize('sequence_type', [list, tuple, UserList])
def test_resolve_references_in_sequence_types(sequence_type):
    '''Supported sequences provide both parent wiring and reference traversal.'''
    reference = Root.model_validate({'bAddress': {'@ref:scoped': 'target'}})
    target = A.model_validate({'b': {'@key:scoped': 'target', 'fieldB': 'value'}})
    model = SequenceContainer(entries=sequence_type([reference, target, 'label', None]))

    model.resolve_references()

    assert reference.get_rune_parent() is model
    assert target.get_rune_parent() is model
    assert reference.bAddress is target.b


def test_resolve_references_skips_resolved_cycles_in_lists():
    '''Repeated resolution must not recurse through an already-bound reference.'''
    child = CyclicReference.model_validate({'target': {'@ref': 'self'}})
    child.target = Reference(child)
    model = SequenceContainer(entries=[child])

    model.resolve_references()
    model.resolve_references()

    assert child.target is child


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
