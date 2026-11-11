# Object identity and references

Rune objects have an identity independently of the keys used to reference them.
`as-key` requests the identity of its operand. Allocating a missing key records
that identity; ordinary data writes are governed by ownership and copy-on-write
(COW) isolation.

## Principles

1. Every object has an identity, even when it has no key.
2. Distinct domain objects have distinct identities. For a given key type and
   resolution scope, a key identifies at most one target.
3. Keys represent identity. Multiple keys of different types can identify the
   same object.
4. Creating a key preserves the object's identity and ordinary data. It records
   an identifier for the existing object.
5. `as-key` requests the identity of its operand; it does not create a new target.
6. A reference into the caller's graph grants read access and identity access.
   Ordinary data writes require ownership.

## Identity and keys

Every object has an identity, including an object with no serialized key. Multiple
keys can identify the same object, using different key types and resolution
scopes. Allocating an additional key preserves the object's identity and ordinary
data.

Key uniqueness is evaluated using both the key type and the resolution scope.
The same key text can occur under different types or in different scopes without
implying that the targets share an identity.

Key allocation can change the object's key metadata and register the key in a
lookup table. This is intentional identity bookkeeping. A serialized object can
therefore gain an `@key` without a change to its ordinary fields. Key allocation
remains subject to the model's metadata permissions; having an identity does not
make every type eligible for every kind of serialized key.

The key identifies the existing target. Creating a key must not construct a
different target solely to hold that key. A reference to a newly keyed object and
a reference to an already keyed object have the same identity semantics.

## Key selection for as-key

Generated code implements `as-key` with `Reference(target)`. With no explicit key
or key type, selection follows this order:

1. Use the existing external key, if present.
2. Otherwise reuse the existing internal key.
3. Otherwise allocate an internal key for the existing target.

| Existing keys | Serialized reference | Key allocation |
| --- | --- | --- |
| External `p1` | `{"@ref:external": "p1"}` | None |
| Internal `i1` | `{"@ref": "i1"}` | None |
| External `p1` and internal `i1` | `{"@ref:external": "p1"}` | None |
| No external or internal key | `{"@ref": "<allocated key>"}` | Internal key |

An explicit `key_type=KeyType.INTERNAL` selects an internal reference even when
an external key exists. Supplying an explicit key selects that key using the
constructor's existing key-type rules.

Local keys are represented by `@key:scoped`. They require the appropriate
resolution scope, so automatic selection currently excludes them. Creating a
scoped reference remains explicit. Automatic selection would need a rule for
checking the receiving scope.

## Reference edges and key registries

A reference occurrence records its key and key type on the receiving field or
list entry. It exposes the target object without making that target a containment
child of the receiver. The same target can occur both inline and by reference.

The target holds its own key metadata. Key registries map keys to targets and are
normally reached through the containment parent chain, with separate handling for
scoped keys. An empty registry does not establish that the target has no key;
key selection reads the target's metadata.

Serialization writes a reference occurrence as its reference key, without
embedding the target's data. The target's containment graph or external context
provides the corresponding keyed object for resolution.

## Ownership and COW

Allocating an identity key alone does not require COW detachment. It may extend
the existing target's key metadata and registry while leaving ordinary data
unchanged. This exception is specific to identity keys and registration; it does
not grant unrestricted access to other metadata or mutable values.

The agreed access rule for generated functions is:

> A reference into the caller's graph grants read access and identity access;
> data writes require ownership.

**Enforcing this access rule is follow-up work.** Current reference access can
expose mutable targets; the external-key selection fix does not generally prevent
data writes through borrowed references.

The follow-up must protect nested objects, mutable collections and mutating
methods reached through a borrowed reference. Replacing a reference field on an
owned receiver is distinct from modifying the referenced target. References to
contained targets within a graph managed by COW must retain the supported
behavior of editing the detached graph while preserving its internal references.

## Code and regression coverage

Key selection and registration are implemented in
[metadata.py](../src/rune/runtime/metadata.py). COW wrappers are implemented in
[cow.py](../src/rune/runtime/cow.py).

[test_keys_and_references.py](../test/test_keys_and_references.py) includes both
generated-function reproductions from [issue #73](https://github.com/finos/rune-python-runtime/issues/73),
along with coverage for existing keys and explicit internal selection.
[test_list_cardinality_serialise.py](../test/test_list_cardinality_serialise.py)
covers reference occurrences and the list form of `as-key`.
