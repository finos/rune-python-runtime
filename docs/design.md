# Rune Python Runtime design

This document describes the shared runtime architecture and the boundary between
generated Python code and runtime support. The detailed design documents linked
below define the contracts for individual areas.

## Generator and runtime boundary

The [Rune Python Code Generator](https://github.com/finos/rune-python-generator)
translates Rune DSL models into Python classes, enums, constraints, and functions.
Generated packages depend on the
[Rune Python Runtime](https://github.com/finos/rune-python-runtime) for common
behavior; model-specific definitions and business logic remain in generated code.

The generator supplies field annotations, metadata permissions, Rune type
identifiers, and choice dispatch tables. The runtime uses those declarations to
construct and validate objects, execute shared operations, manage references, and
read and write Rune JSON. Changes to this contract may require corresponding
generator changes and regeneration of model packages.

## Main components

| Component | Responsibility |
| --- | --- |
| [BaseDataClass](../src/rune/runtime/base_data_class.py) | Pydantic-based foundation for generated models, graph initialization, explicit Rune validation, and JSON serialization and deserialization. |
| [Metadata and references](../src/rune/runtime/metadata.py) | Metadata-bearing values, identity keys, reference tokens, key registries, and containment parent links. |
| [Reference lists](../src/rune/runtime/reference_list.py) | Mutable lists that expose targets while retaining reference metadata for each occurrence. |
| [ObjectBuilder](../src/rune/runtime/object_builder.py) | Incremental construction of nested values before materializing a generated model. |
| [Copy-on-write wrappers](../src/rune/runtime/cow.py) | Deferred copying for supported mutations of function inputs and intermediate values. |
| [Expression helpers](../src/rune/runtime/utils.py) | Shared Rune operations for attribute access, collections, comparisons, cardinality, and sorting. |
| [Conditions](../src/rune/runtime/conditions.py) | Registration and execution support for generated constraints. |
| [Function proxies](../src/rune/runtime/func_proxy.py) and [native registry](../src/rune/runtime/native_registry.py) | Replaceable function calls, return-value finalization, and registration and dispatch of native implementations. |

## Model lifecycle

Python constructors and `ObjectBuilder` work with native option field names.
JSON entry points interpret the wire representation, including metadata wrappers
and choice discriminators. Pydantic validates field structure and types; runtime
initialization establishes containment parents, registers keys, and resolves
references.

Generated functions use the shared expression helpers, builders, and copy-on-write
support. Copy-on-write wrappers defer copying until a supported mutation requires
it. The object identity design below describes the ownership rules and remaining
limitations around writes through borrowed references.

Rune model validation is explicit: `validate_model()` performs attribute,
cardinality, and condition checks. Ordinary construction or a Pydantic validation
call does not replace this step.

`rune_serialize()` writes the Rune JSON representation and its document envelope.
Reference occurrences serialize as keys rather than embedded targets; choices
serialize as the selected leaf while preserving required metadata and wrappers.
`rune_deserialize()` reads that representation into generated model classes.

## Detailed designs

| Design document | Scope |
| --- | --- |
| [Object identity and references](object-identity.md) | Identity independently of keys, `as-key` selection, reference occurrences, resolution scopes, ownership, and copy-on-write boundaries. It also identifies follow-up ownership enforcement work. |
| [Choice serialization](choice-serialization.md) | Flat JSON choices, Rune discriminators, generator dispatch tables, metadata and reference wrappers, native versus JSON entry points, errors, and interoperability considerations. |

These contracts interact: choice flattening must preserve reference occurrences,
and builders and collection helpers must preserve occurrence metadata. A single
target can appear inline and by reference in the same list, so reference intent
cannot be inferred from object identity. Use `rune_sorted` when sorting must retain
that metadata; built-in `sorted` returns a plain list.

Keep detailed rules and implementation status in the individual design documents.
Use this overview as the entry point when adding further design topics.
