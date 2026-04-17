'''tests for namespace-prefix-aware Rune deserialization'''
import json
import sys
import types

from rune.runtime.base_data_class import BaseDataClass


def _register_prefixed_root(monkeypatch, prefix: str):
    package_names = (
        prefix,
        f'{prefix}.rosetta_dsl',
        f'{prefix}.rosetta_dsl.test',
    )
    for package_name in package_names:
        module = types.ModuleType(package_name)
        module.__path__ = []
        if package_name == prefix:
            module.rune_namespace_prefix = prefix
        monkeypatch.setitem(sys.modules, package_name, module)

    module_name = f'{prefix}.rosetta_dsl.test.Root'
    root_module = types.ModuleType(module_name)
    root_cls = type(
        'Root',
        (BaseDataClass,),
        {
            '__module__': module_name,
            '__annotations__': {},
        },
    )
    root_module.Root = root_cls
    monkeypatch.setitem(sys.modules, module_name, root_module)
    return root_cls


def test_base_deserialize_uses_explicit_namespace_prefix(monkeypatch):
    '''the root deserializer should resolve unprefixed @type via the argument'''
    prefix_a_root = _register_prefixed_root(monkeypatch, 'prefix_a')
    prefix_b_root = _register_prefixed_root(monkeypatch, 'prefix_b')
    rune_data = {'@type': 'rosetta_dsl.test.Root'}

    model_a = BaseDataClass.rune_deserialize(
        rune_data,
        validate_model=False,
        namespace_prefix='prefix_a')
    model_b = BaseDataClass.rune_deserialize(
        rune_data,
        validate_model=False,
        namespace_prefix='prefix_b')

    assert type(model_a) is prefix_a_root
    assert type(model_b) is prefix_b_root


def test_root_deserialize_uses_package_namespace_prefix(monkeypatch):
    '''generated package metadata should drive import resolution when present'''
    prefixed_root = _register_prefixed_root(monkeypatch, 'prefixed_model')
    rune_data = {'@type': 'rosetta_dsl.test.Root'}

    model = prefixed_root.rune_deserialize(rune_data, validate_model=False)

    assert type(model) is prefixed_root


def test_root_deserialize_prefers_its_own_package_prefix(monkeypatch):
    '''a concrete root class should use the prefix from its own package'''
    prefix_a_root = _register_prefixed_root(monkeypatch, 'prefix_a')
    _register_prefixed_root(monkeypatch, 'prefix_b')
    rune_data = {'@type': 'rosetta_dsl.test.Root'}

    model = prefix_a_root.rune_deserialize(
        rune_data,
        validate_model=False,
        namespace_prefix='prefix_b')

    assert type(model) is prefix_a_root


def test_standalone_serialization_strips_package_namespace_prefix(monkeypatch):
    '''standalone classes should serialize native Rune type names'''
    prefixed_root = _register_prefixed_root(monkeypatch, 'prefixed_model')

    serialized = json.loads(
        prefixed_root().rune_serialize(validate_model=False))

    assert serialized['@type'] == 'rosetta_dsl.test.Root'
    assert serialized['@model'] == 'rosetta_dsl'
