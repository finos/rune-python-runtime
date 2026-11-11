'''Writing (0..*) attributes whose serializer wraps the whole list.

The generator puts the serializer and validator around the whole list, so the
list reaches `serialise` in a single call: for a list attribute with metadata,
and for every list attribute in a generated bundle. `deserialize` already
handles this (see test_list_cardinality_with_meta.py); these tests cover the
writing side.
'''
import json
from enum import Enum
from typing import Annotated, ClassVar, Optional

import pytest
from pydantic import ConfigDict, Field, ValidationError
from pydantic_core import PydanticSerializationError

from rune.runtime.base_data_class import BaseDataClass
from rune.runtime.cow import rune_cow, rune_unwrap
from rune.runtime.metadata import (
    BaseReference,
    EnumWithMetaMixin,
    KeyType,
    Reference,
    StrWithMeta,
    UnresolvedReference,
)
from rune.runtime.utils import rune_add_to_list, rune_sorted

NS = 'test_list_cardinality_serialise'

# The classes below are what rune-python-generator produces from this model,
# in the form used in a generated bundle, where each class sets _FQRTN and every
# attribute has the runtime's serializer and validator:
#
#   enum Colour:
#       Red
#       Blue
#
#   type Item:
#       name string (0..1)
#
#   type SpecialItem extends Item:
#       extra string (0..1)
#
#   type Party:
#       [metadata key]
#       name string (0..1)
#
#   type Holder:
#       items Item (0..*)
#       parties Party (0..*)
#       refs Party (0..*)
#           [metadata reference]
#       names string (0..*)
#           [metadata scheme]
#       colours Colour (0..*)
#           [metadata scheme]


class Colour(EnumWithMetaMixin, Enum):
    BLUE = "Blue"
    RED = "Red"


class Item(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.Item'
    name: Optional[str] = Field(None, description='')


class SpecialItem(Item):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.SpecialItem'
    extra: Optional[str] = Field(None, description='')


class Party(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.Party'
    _ALLOWED_METADATA = {'@key', '@key:external'}
    name: Optional[str] = Field(None, description='')


class Holder(BaseDataClass):
    model_config = ConfigDict(defer_build=True)
    _FQRTN: ClassVar[str] = f'{NS}.Holder'
    items: Annotated[Optional[list[Item | None]], Item.serializer(), Item.validator()] = Field(None, description='')
    parties: Annotated[Optional[list[Party | None]], Party.serializer(), Party.validator(('@key', '@key:external', '@key:scoped'))] = Field(None, description='')
    refs: Annotated[Optional[list[Party | BaseReference | None]], Party.serializer(), Party.validator(('@key', '@key:external', '@key:scoped', '@ref', '@ref:external', '@ref:scoped'))] = Field(None, description='')
    names: Annotated[Optional[list[StrWithMeta | None]], StrWithMeta.serializer(), StrWithMeta.validator(('@scheme', ))] = Field(None, description='')
    colours: Annotated[Optional[list[Colour | None]], Colour.serializer(), Colour.validator(('@scheme', ))] = Field(None, description='')

    _KEY_REF_CONSTRAINTS = {
        'refs': {'@key', '@key:external', '@ref', '@ref:external', '@ref:scoped'}
    }


class NestedHolder(BaseDataClass):
    '''Reference lists can point to targets in a later sibling field.'''
    holders: list[Holder]
    parties: list[Annotated[Party, Party.serializer(), Party.validator(('@key:external',))]]


class BasicReferenceHolder(BaseDataClass):
    '''Primitive and enum lists use the same reference bookkeeping as objects.'''
    names: Annotated[list[StrWithMeta | BaseReference], StrWithMeta.serializer(), StrWithMeta.validator(('@key', '@key:external', '@ref', '@ref:external'))] = Field(default_factory=list)
    colours: Annotated[list[Colour | BaseReference], Colour.serializer(), Colour.validator(('@key', '@key:external', '@ref', '@ref:external'))] = Field(default_factory=list)
    _KEY_REF_CONSTRAINTS = {
        'names': {'@ref', '@ref:external'},
        'colours': {'@ref', '@ref:external'},
    }


@pytest.mark.parametrize('attribute, value', [
    pytest.param('items', [{'name': 'A'}, {'name': 'B'}], id='objects'),
    pytest.param('parties', [{'@key:external': 'p1', 'name': 'A'}], id='objects-with-keys'),
    pytest.param('names',
                 [{'@scheme': 'http://a', '@data': 'x'}, {'@scheme': 'http://b', '@data': 'y'}],
                 id='basic-with-metadata'),
    pytest.param('colours',
                 [{'@scheme': 'http://a', '@data': 'Red'}, {'@data': 'Blue'}],
                 id='enum-with-metadata'),
])
def test_write_list(attribute, value):
    '''a list is written item by item, each with its own metadata'''
    model = Holder.rune_deserialize({attribute: value}, validate_model=False)
    written = json.loads(model.rune_serialize(validate_model=False))
    assert written.get(attribute) == value


def test_write_empty_list():
    '''an empty list is written as an empty list'''
    written = json.loads(Holder(items=[]).rune_serialize(validate_model=False))
    assert written.get('items') == []


def test_write_absent_list():
    '''an absent list is not written'''
    written = json.loads(Holder().rune_serialize(validate_model=False))
    assert 'items' not in written


def test_write_null_lists():
    '''Pydantic skips the serializers for null whole-list values.'''
    expected = dict.fromkeys(('items', 'parties', 'refs', 'names', 'colours'))
    model = Holder(**expected)

    assert model.model_dump() == expected
    assert json.loads(model.model_dump_json()) == expected
    assert model.model_dump(exclude_none=True) == {}


def test_write_null_basic_list_item():
    '''Direct recursion preserves null list items without invoking a serializer.'''
    model = Holder(names=['x', None])

    assert json.loads(model.rune_serialize())['names'] == [{'@data': 'x'}, None]


def test_write_list_built_in_python():
    '''a list built in Python is written the same way'''
    written = json.loads(Holder(items=[Item(name='A')]).rune_serialize(validate_model=False))
    assert written.get('items') == [{'name': 'A'}]


def test_write_list_keeps_subtype():
    '''an item of a subtype is written with its @type'''
    model = Holder(items=[SpecialItem(name='A', extra='x')])
    written = json.loads(model.rune_serialize(validate_model=False))
    assert written.get('items') == [{'@type': f'{NS}.SpecialItem', 'name': 'A', 'extra': 'x'}]


@pytest.mark.parametrize('key_type', list(KeyType))
def test_resolved_list_references_round_trip(key_type):
    '''Resolve list entries before validation and preserve their reference tags.'''
    data = {'parties': [{key_type.rune_key_tag: 'p1', 'name': 'A'}],
            'refs': [{key_type.rune_ref_tag: 'p1'}]}

    model = Holder.rune_deserialize(data)

    assert model.refs[0] is model.parties[0]
    assert model.resolve_ref_key('refs', 0) == 'p1'
    model.resolve_references()
    assert model.refs[0] is model.parties[0]
    assert json.loads(model.rune_serialize())['refs'] == data['refs']


def test_nested_forward_list_reference_preserves_parent():
    '''Binding and revalidation must not turn reference edges into containment.'''
    model = NestedHolder.rune_deserialize({
        'holders': [{'refs': [{'@ref:external': 'p1'}, {'name': 'inline'}]}],
        'parties': [{'@key:external': 'p1', 'name': 'A'}],
    })

    assert model.holders[0].refs[0] is model.parties[0]
    assert model.parties[0].get_rune_parent() is model
    assert model.holders[0].refs[1].get_rune_parent() is model.holders[0]
    assert model.validate_model() == []
    assert model.parties[0].get_rune_parent() is model
    assert json.loads(model.rune_serialize())['holders'][0]['refs'] == [
        {'@ref:external': 'p1'}, {'name': 'inline'},
    ]


def test_python_list_keeps_reference_and_inline_occurrences_distinct():
    '''Two occurrences of the same target can have different serialization.'''
    party = Party(name='A')
    model = Holder(parties=[party], refs=[Reference(party, 'p1'), party])

    assert model.refs[0] is model.refs[1] is party
    assert json.loads(model.rune_serialize())['refs'] == [
        {'@ref:external': 'p1'}, {'@key:external': 'p1', 'name': 'A'},
    ]


def test_unresolved_list_references_are_rejected():
    '''Staged dangling references may be retained, but cannot be serialized.'''
    model = Holder.model_validate({'refs': [{'@ref:external': 'missing'}]})
    model.resolve_references(ignore_dangling=True)

    assert isinstance(model.refs[0], UnresolvedReference)
    with pytest.raises(KeyError, match='missing'):
        model.resolve_references()
    with pytest.raises(PydanticSerializationError, match='Unbound reference'):
        model.rune_serialize(validate_model=False)
    with pytest.raises(KeyError, match='missing'):
        Holder.rune_deserialize({'refs': [{'@ref:external': 'missing'}]},
                                validate_model=False)


def test_excluded_unresolved_reference_does_not_block_serialization():
    '''Only selected list entries must be serializable.'''
    model = Holder.model_validate({
        'refs': [{'@ref:external': 'missing'}, {'name': 'inline'}],
    })

    assert model.model_dump(include={'refs': {1}}) == {'refs': [{'name': 'inline'}]}
    assert model.model_dump(exclude={'refs'}) == {
        'items': None, 'parties': None, 'names': None, 'colours': None,
    }


def test_list_reference_checks_target_type_and_permissions():
    '''Reference tokens cannot bypass the field's type or reference permissions.'''
    class OtherParty(BaseDataClass):
        _ALLOWED_METADATA = {'@key'}

    class InternalHolder(Holder):
        _KEY_REF_CONSTRAINTS = {'refs': {'@ref'}}

    with pytest.raises(TypeError, match='Incompatible types'):
        Holder(refs=[Reference(OtherParty())])
    with pytest.raises(ValidationError, match='not allowed'):
        InternalHolder(refs=[Reference(Party(name='A'), 'p1')])


def test_reference_list_mutations_preserve_occurrence_metadata():
    '''Ordinary list operations move metadata with the original occurrence.'''
    party = Party(name='A')
    model = Holder(parties=[party], refs=[Reference(party, 'p1'), party])

    model.refs.reverse()
    model.refs.insert(0, Party(name='B'))
    del model.refs[0]
    model.refs.extend(model.refs[-1:])
    model.refs.sort(key=lambda item: item.name)
    assert json.loads(model.rune_serialize())['refs'] == [
        {'@key:external': 'p1', 'name': 'A'},
        {'@ref:external': 'p1'}, {'@ref:external': 'p1'},
    ]

    model.refs[1] = party
    assert model.resolve_ref_key('refs', 1) is None
    assert json.loads(model.rune_serialize())['refs'][1] == {
        '@key:external': 'p1', 'name': 'A',
    }


@pytest.mark.parametrize('reverse', [False, True])
def test_rune_sorted_preserves_reference_occurrences_and_input_order(reverse):
    '''Sort a copy while retaining distinct inline/reference occurrences of one target.'''
    first = Party(name='A')
    second = Party(name='B')
    model = Holder(parties=[first, second], refs=[Reference(second, 'p-b'), first, Reference(first, 'p-a')])
    before = json.loads(model.rune_serialize())['refs']

    result = rune_sorted(model.refs, key=lambda party: party.name, reverse=reverse)

    assert type(result) is type(model.refs)
    assert result is not model.refs
    assert result[1 if reverse else 0] is first
    ascending = [
        {'@key:external': 'p-a', 'name': 'A'},
        {'@ref:external': 'p-a'},
        {'@ref:external': 'p-b'},
    ]
    expected = [ascending[2], ascending[0], ascending[1]] if reverse else ascending
    sorted_model = model.model_copy(update={'refs': result})
    assert json.loads(sorted_model.rune_serialize())['refs'] == expected
    assert json.loads(model.rune_serialize())['refs'] == before
    result.pop()
    assert len(model.refs) == 3


def test_rune_sorted_key_error_keeps_source_reference_metadata():
    '''A failing key function does not change the input collection or its intent.'''
    party = Party(name='A')
    model = Holder(refs=[Reference(party, 'p1'), party])

    def failing_key(_):
        raise ValueError('cannot sort')

    with pytest.raises(ValueError, match='cannot sort'):
        rune_sorted(model.refs, key=failing_key)

    assert model.refs[0] is model.refs[1] is party
    assert json.loads(model.rune_serialize())['refs'] == [
        {'@ref:external': 'p1'}, {'@key:external': 'p1', 'name': 'A'},
    ]


@pytest.mark.parametrize('mutate_draft', [False, True])
def test_rune_sorted_cow_list_preserves_metadata_and_object_isolation(mutate_draft):
    '''Sorted COW results keep proxies and reference flags from the current list.'''
    first = Party(name='A')
    second = Party(name='B')
    model = Holder(parties=[first, second], refs=[Reference(second, 'p-b'), first, Reference(first, 'p-a')])
    draft = rune_cow(model)
    if mutate_draft:
        draft.refs.append(Party(name='C'))

    result = rune_sorted(draft.refs, key=lambda party: party.name)

    assert type(result) is type(model.refs)
    assert result.reference_at(0) is None
    assert result.reference_at(1) == ('p-a', KeyType.EXTERNAL)
    assert result.reference_at(2) == ('p-b', KeyType.EXTERNAL)
    assert [party.name for party in draft.refs] == ['B', 'A', 'A'] + (['C'] if mutate_draft else [])
    result[1].name = 'Edited'
    unwrapped = rune_unwrap(result)
    assert unwrapped[1].name == 'Edited'
    assert unwrapped.reference_at(1) == ('p-a', KeyType.EXTERNAL)
    assert first.name == 'A'
    assert len(model.refs) == 3


def test_reference_list_assignment_and_addition_preserve_metadata():
    '''Assignment owns its list storage; slices and runtime additions keep intent.'''
    party = Party(name='A')
    incoming = [Reference(party, 'p1')]
    model = Holder(parties=[party], refs=incoming)
    incoming.clear()
    model.refs = model.refs[:]
    rune_add_to_list(model.refs, model.refs)
    model.refs.append(Reference(party, 'p1'))

    assert model.refs == [party, party, party]
    assert json.loads(model.rune_serialize())['refs'] == [{'@ref:external': 'p1'}] * 3
    model.refs = list(model.refs)
    assert all(model.resolve_ref_key('refs', index) is None for index in range(3))


def test_reference_list_repetition_and_slice_replacement():
    '''Repetition, concatenation and slice replacement preserve each occurrence.'''
    party = Party(name='A')
    model = Holder(parties=[party], refs=[Reference(party, 'p1'), party])
    model.refs *= 2
    model.refs.pop(1)
    model.refs[1:] = [] + model.refs[:1]
    model.refs += model.refs[:1]

    assert json.loads(model.rune_serialize())['refs'] == [{'@ref:external': 'p1'}] * 3
    model.refs.clear()
    assert json.loads(model.rune_serialize())['refs'] == []


@pytest.mark.parametrize('operation', ['insert', 'pop', 'imul', 'mul', 'rmul'])
def test_reference_list_accepts_index_protocol(operation):
    '''Index-like arguments must keep values and occurrence metadata aligned.'''
    class IndexValue:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def __index__(self):
            # A second conversion would select a different position or count.
            result = self.value + self.calls
            self.calls += 1
            return result

    party = Party(name='A')
    model = Holder(parties=[party], refs=[Reference(party, 'p1'), party])
    reference = {'@ref:external': 'p1'}
    inline = {'@key:external': 'p1', 'name': 'A'}
    index = IndexValue(0 if operation in ('insert', 'pop') else 2)

    if operation == 'insert':
        model.refs.insert(index, party)
        expected = [inline, reference, inline]
    elif operation == 'pop':
        assert model.refs.pop(index) is party
        expected = [inline]
    else:
        if operation == 'imul':
            model.refs *= index
        elif operation == 'mul':
            model.refs = model.refs * index
        else:
            model.refs = index * model.refs
        expected = [reference, inline] * 2

    assert index.calls == 1
    assert json.loads(model.rune_serialize())['refs'] == expected


@pytest.mark.parametrize('options, expected', [
    ({'exclude': {'refs'}}, None),
    ({'include': {'refs': {1}}}, [{'@ref:external': 'p1'}]),
    ({'exclude': {'refs': {0}}}, [{'@ref:external': 'p1'}]),
    ({'include': {'refs': {-1}}}, [{'@ref:external': 'p1'}]),
    ({'exclude_none': True}, [{'name': 'inline'}, {'@ref:external': 'p1'}]),
])
def test_list_reference_serialization_options(options, expected):
    '''Output filtering must retain the reference intent of the selected positions.'''
    party = Party(name='A')
    model = Holder(parties=[party], refs=[Party(name='inline'), Reference(party, 'p1')])

    assert json.loads(model.rune_serialize(**options)).get('refs') == expected


def test_list_references_survive_copy_and_cow():
    '''Copying and mutating a COW list must not change the source metadata.'''
    party = Party(name='A')
    model = Holder(parties=[party], refs=[Reference(party, 'p1')])
    clone = model.model_copy(deep=True)

    assert clone.refs[0] is clone.parties[0]
    assert clone.refs[0] is not party
    assert json.loads(clone.rune_serialize())['refs'] == [{'@ref:external': 'p1'}]

    draft = rune_cow(model)
    draft.refs.append(Party(name='B'))
    written = json.loads(rune_unwrap(draft).rune_serialize())
    assert written['refs'] == [{'@ref:external': 'p1'}, {'name': 'B'}]
    assert len(model.refs) == 1
    assert json.loads(model.rune_serialize())['refs'] == [{'@ref:external': 'p1'}]


def test_cow_target_edit_preserves_reference_occurrence():
    '''An internal COW child update retains reference intent and isolates the source.'''
    party = Party(name='A')
    model = Holder(parties=[party], refs=[Reference(party, 'p1')])
    draft = rune_cow(model)

    draft.refs[0].name = 'B'

    assert party.name == 'A'
    assert rune_unwrap(draft).refs[0].name == 'B'
    assert rune_unwrap(draft).refs[0] is rune_unwrap(draft).parties[0]
    assert json.loads(rune_unwrap(draft).rune_serialize())['refs'] == [{'@ref:external': 'p1'}]


@pytest.mark.parametrize('attribute, value', [('names', 'A'), ('colours', 'Red')])
def test_basic_and_enum_reference_lists_round_trip(attribute, value):
    '''Reference entries are resolved before the basic or enum serializer runs.'''
    data = {attribute: [
        {'@key:external': 'target', '@data': value},
        {'@ref:external': 'target'},
    ]}

    model = BasicReferenceHolder.rune_deserialize(data)

    values = getattr(model, attribute)
    assert values[0] is values[1]
    assert model.model_dump(mode='json', exclude_unset=True) == data
    assert json.loads(model.rune_serialize())[attribute] == data[attribute]


def test_inherited_list_reference_constraints_and_overrides():
    '''New subclass constraints do not hide inherited fields' permissions.'''
    class ExtendedHolder(Holder):
        _KEY_REF_CONSTRAINTS = {'other': {'@ref'}}

    class OverriddenHolder(Holder):
        refs: list[Party | BaseReference]
        _KEY_REF_CONSTRAINTS = {}

    party = Party(name='A')
    model = ExtendedHolder(parties=[party], refs=[Reference(party, 'p1')])
    assert model.refs[0] is party
    assert json.loads(model.rune_serialize())['refs'] == [{'@ref:external': 'p1'}]

    with pytest.raises(ValidationError, match='not allowed'):
        OverriddenHolder(parties=[party], refs=model.refs)


def test_list_reference_cycles_are_not_traversed():
    '''A list reference to its owner is neither containment nor embedded JSON.'''
    class CyclicHolder(BaseDataClass):
        _ALLOWED_METADATA = {'@key'}
        refs: Annotated[list[BaseDataClass | BaseReference], BaseDataClass.serializer(), BaseDataClass.validator(('@ref',))]
        _KEY_REF_CONSTRAINTS = {'refs': {'@ref'}}

    model = CyclicHolder(refs=[])
    model.refs = [Reference(model)]
    model.resolve_references()
    model.resolve_references()

    assert model.refs[0] is model
    assert model.get_rune_parent() is None
    assert json.loads(model.rune_serialize())['refs'] == [{'@ref': model.get_meta('key')}]


def test_list_reference_aliases_and_bare_list_schema():
    '''Aliased fields with ordinary Pydantic list schemas retain their references.'''
    class BareHolder(BaseDataClass):
        parties: list[Party]
        refs: list[Party | BaseReference] = Field(serialization_alias='references')
        _KEY_REF_CONSTRAINTS = {'refs': {'@ref:external'}}

    model = BareHolder.rune_deserialize({
        'parties': [{'@key:external': 'p1', 'name': 'A'}],
        'refs': [{'@ref:external': 'p1'}],
    })

    assert model.refs[0] is model.parties[0]
    assert model.model_dump(by_alias=True)['references'] == [{'@ref:external': 'p1'}]
    assert model.model_dump(exclude={'refs'}) == {'parties': [{'name': 'A'}]}


def test_reference_field_caches_follow_deferred_model_rebuild():
    '''A cached placeholder must not hide a reference field added by the generator.'''
    class DeferredHolder(BaseDataClass):
        parties: list[Party]
        refs: None = None
        _KEY_REF_CONSTRAINTS = {}

    DeferredHolder(parties=[])
    DeferredHolder.model_fields['refs'].annotation = list[Party | BaseReference]
    DeferredHolder.__annotations__['refs'] = list[Party | BaseReference]
    constraints = DeferredHolder.__private_attributes__['_KEY_REF_CONSTRAINTS'].default
    constraints['refs'] = {'@ref:external'}
    DeferredHolder.model_rebuild(force=True)

    model = DeferredHolder.rune_deserialize({
        'parties': [{'@key:external': 'p1', 'name': 'A'}],
        'refs': [{'@ref:external': 'p1'}],
    })
    assert model.refs[0] is model.parties[0]
    assert model.model_dump()['refs'] == [{'@ref:external': 'p1'}]

    # Rebuilding must invalidate cached permissions as well as field shapes.
    constraints.clear()
    DeferredHolder.model_rebuild(force=True)
    with pytest.raises(ValidationError, match='not allowed'):
        DeferredHolder(parties=[], refs=[Reference(Party(name='B'), 'p2')])


def test_rebuild_preserves_local_forward_reference_resolution():
    '''The cache-invalidating wrapper must preserve Pydantic's caller namespace.'''
    class DeferredContainer(BaseDataClass):
        child: 'LaterChild'

    class LaterChild(BaseDataClass):
        name: str

    DeferredContainer.model_rebuild()
    model = DeferredContainer(child={'name': 'A'})

    assert isinstance(model.child, LaterChild)
    assert model.child.get_rune_parent() is model


def test_public_resolution_registers_mutated_sibling_subtrees_first():
    '''New targets in a later branch must be registered before resolving earlier refs.'''
    class Groups(BaseDataClass):
        groups: list[NestedHolder]

    model = Groups.model_validate({'groups': [
        {'holders': [{'refs': [{'@ref:external': 'later'}]}], 'parties': []},
        {'holders': [], 'parties': []},
    ]})
    party = Party(name='Later')
    party.set_external_key('later')
    model.groups[1].parties.append(party)

    model.resolve_references()

    assert model.groups[0].holders[0].refs[0] is party
    assert party.get_rune_parent() is model.groups[1]
    model.resolve_references()
    assert model.validate_model() == []


def test_revalidation_preserves_shared_parents_and_copies_registry_dictionaries():
    '''Revalidation shares retained children without sharing registry dictionaries.'''
    model = Holder.rune_deserialize({
        'parties': [{'@key:external': 'p1', 'name': 'A'}],
        'refs': [{'@ref:external': 'p1'}],
    })

    validated = Holder.model_validate(model)

    assert validated is not model
    assert validated.parties[0] is model.parties[0]
    assert validated.refs[0] is model.parties[0]
    assert model.parties[0].get_rune_parent() is model
    source_keys = model._get_object_map(KeyType.EXTERNAL)
    validated_keys = validated._get_object_map(KeyType.EXTERNAL)
    assert validated_keys is not source_keys
    assert validated_keys['p1'] is source_keys['p1']
    validated_keys['p2'] = Party(name='B')
    assert 'p2' not in source_keys


def test_reference_lists_keep_value_equality_and_first_equal_remove():
    '''Equality and remove follow normal list semantics, independently of reference intent.'''
    first = Party(name='A')
    second = Party(name='A')
    model = Holder(refs=[Reference(first, 'p1'), second])

    assert first is not second
    assert model.refs == [second, second]
    assert model.refs[:1] == model.refs[1:]

    model.refs.remove(second)

    assert model.refs[0] is second
    assert model.resolve_ref_key('refs', 0) is None
    assert json.loads(model.rune_serialize())['refs'] == [{'name': 'A'}]
