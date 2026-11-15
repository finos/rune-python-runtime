# Choice serialization

## Principles

1. Python objects retain their choice option fields. JSON flattens nested choices
   to the selected leaf, preserving required metadata and reference wrappers.
2. Discriminators use Rune identifiers: data types and enums use their Rune FQN;
   primitive options use their option name. Python package prefixes and Java
   implementation class names are excluded.
3. The generator supplies `_RUNE_CHOICE` and `_CHOICE_DISPATCH`. Runtime dispatch
   follows that table without searching the complete choice tree.
4. The DSL prohibits metadata on choice types. Metadata on choice-valued fields
   and on individual options remains valid and must survive serialization.
5. References remain reference occurrences, preserving their keys and targets.
   Flattening must respect the distinction between a reference to a choice and
   a reference to its selected option.
6. JSON readers require the flat format. Native Python constructors and builders
   continue to use option field names. Ordinary types with `one-of` conditions
   retain their existing representation.

## Wire format

Examples omit the document's `@model` and `@version` envelope fields.

| Selected option | JSON choice value |
| --- | --- |
| Data type | `{"@type": "example.A", "value": "foo"}` |
| Primitive | `{"@type": "string", "@data": "foo"}` |
| Enum | `{"@type": "example.Currency", "@data": "USD"}` |
| Primitive with field metadata | `{"@type": "string", "@scheme": "urn:test", "@data": "foo"}` |

Metadata does not turn a primitive discriminator into a Java class name such as
`java.lang.String`. Enums use their Rune FQN with or without metadata. Their
Python field names and annotations remain unchanged.

**Java interoperability check:** the [Java enum example](examples/java-enum-choice/README.md)
records Rune Common 12.29.0 emitting an option name for a plain enum and a Rune FQN
for a metadata-bearing enum. That differs from the uniform enum-FQN generator
rule above and requires alignment with the intended interoperability contract.
The [atomic-choice example](examples/java-atomic-choices/README.md) records Java's
numeric and temporal representations and its unsupported `pattern` option.

## Generator contract

Each generated choice declares a marker and a discriminator-to-field map:

```python
_RUNE_CHOICE: ClassVar[bool] = True
_CHOICE_DISPATCH: ClassVar[dict[str, str]] = {
    "example.A": "ChoiceAB",
    "example.B": "ChoiceAB",
    "example.C": "ChoiceCD",
    "example.D": "ChoiceCD",
}
```

Values are immediate Python field names, including any name mangling. Each nested
choice repeats the lookup at its own level. For example, a deeper path
`Outer.ChoiceABCD.ChoiceAB.A` is reconstructed one field at a time; the table
does not store that complete path.

The generator strips type aliases, propagates descendant discriminators, excludes
Python namespace prefixes, and rejects ambiguous routes to the same key. It adds
a nested choice's own FQN only when metadata on that option requires a wrapper or
reference discriminator. Plain nested choices contribute their leaves only.

Discriminator-based reads use the map. Primitive and enum writes reverse it to obtain
`@type`; data writes use the declared class's `_fqrtn()`. Nested writes forward
the selected leaf representation. Generated enums currently have no `_fqrtn()`.
The generator therefore owns the correctness of their dispatch keys.

## Metadata and references

Metadata permissions are additive: field permissions union type permissions.
For a valid generated choice, the type contribution is empty, so permitted
metadata comes from its containing field. The inherited `serialise_meta` handles
it; no choice-specific metadata blacklist is needed. Manually bypassing metadata
validation is outside this contract.

A choice-valued field with location metadata retains a wrapper:

```json
{"@key:scoped": "c1", "@data": {"@type": "example.A", "value": "foo"}}
```

When that field is itself a selected choice option, the enclosing choice adds
the option's own type:

```json
{"@type": "example.Inner", "@key:scoped": "c1", "@data": {"@type": "example.A", "value": "foo"}}
```

Likewise, `{"@ref:scoped": "c1"}` references the whole choice, while
`{"@type": "example.Inner", "@ref:scoped": "c1"}` selects a referenced option.
Keys inside selected leaves remain available to graph reference resolution.
Lists retain reference occurrence metadata, and COW proxies retain their normal
behavior; see [Object identity and references](object-identity.md).

A standalone choice has no containing-field metadata. The root serializer omits
any such metadata retained on a reused choice instance, while preserving metadata
on its selected leaf or option, including a primitive's `@scheme`.

## Runtime flow and entry points

- **Read:** unwrap field metadata, dispatch `@type` into native option fields,
  validate those fields, then initialize containment parents, keys and references.
- **Write:** serialize fields, substitute reference tokens, flatten the selected
  option, then merge the document envelope.

The shared Pydantic hooks are `_deserialize_rune_model` and
`_serialize_rune_model`. Choice handling lives in `_choice_to_fields` and
`_flatten_choice`; field wrappers are handled by `ComplexTypeMetaDataMixin`.

`rune_deserialize`, `model_validate_json` and `TypeAdapter.validate_json` enter
wire-validation mode. Constructors, ordinary `model_validate` calls and
`ObjectBuilder` accept native field dictionaries. A scoped context flag, the model
metaclass and the builder's explicit native scope keep constructors and builders
working inside JSON-validation default factories.

Inside a graph, the declared field type supplies the enclosing choice class.
For standalone round trips, call `KnownChoice.rune_deserialize(payload)`. The
generic `BaseDataClass` reader cannot infer the enclosing choice from a leaf tag:
a data FQN identifies the leaf class; primitive and enum tags require choice
context. `rune_serialize()` adds the normal document envelope in either case.

## Errors and maintenance

- Missing `_CHOICE_DISPATCH`: `_choice_dispatch()` raises `RuntimeError` with
  regeneration guidance on read and write. An empty table is distinct from a
  missing table. The lookup is deliberately uncached.
- Wrong discriminator or wrapper Python type: `TypeError`. Missing or unsupported
  wire values: `ValueError`, which Pydantic wraps as `ValidationError` when reading.
- Serializer failures are wrapped in `PydanticSerializationError`. Multiple
  selected options and data values whose runtime type differs from the declared
  option class are rejected on write. Rune constraint validation remains separate
  from a bare `model_dump()` call.
- Option-class inspection is cached; `model_rebuild()` clears that cache.
- Regenerate older models to obtain dispatch tables. Older JSON using nested
  option field names must be regenerated into the flat format.

Implementation: [base_data_class.py](../src/rune/runtime/base_data_class.py),
[metadata.py](../src/rune/runtime/metadata.py), and
[object_builder.py](../src/rune/runtime/object_builder.py). Generator counterpart:
`PythonChoiceDispatchProcessor` in `rune-python-generator`.

Regression coverage: [test_choice_serialization.py](../test/test_choice_serialization.py)
covers nesting, root and field use, metadata, references and lists, builders, COW,
filtering, aliases, rebuilding, and error diagnostics.
