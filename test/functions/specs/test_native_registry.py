"""Native registry tests."""
import logging
import sys
import types

from rune.runtime.native_registry import (
    rune_attempt_register_native_functions,
    rune_deregister_native,
    rune_execute_native,
)


def test_attempt_register_native_functions_custom_package():
    registered = rune_attempt_register_native_functions(
        ["native_registry.rune_register_native"],
        native_pacakge="rune.runtime",
    )
    assert registered == ["native_registry.rune_register_native"]

    def _sample():
        return "ok"

    try:
        rune_execute_native(
            "native_registry.rune_register_native",
            "test.custom.register",
            _sample,
        )
        assert rune_execute_native("test.custom.register") == "ok"
    finally:
        rune_deregister_native("native_registry.rune_register_native")
        rune_deregister_native("test.custom.register")


def test_attempt_register_native_functions_logs(caplog):
    with caplog.at_level(logging.WARNING):
        rune_attempt_register_native_functions(
            [
                "missingmodule.Foo",
                "native_registry._NATIVE_REGISTRY",
            ],
            native_pacakge="rune.runtime",
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Native function module import failed" in msg for msg in messages)
    assert any("Ignored native function native_registry._NATIVE_REGISTRY" in msg
               for msg in messages)


def test_attempt_register_native_functions_custom_package_short_native():
    registered = rune_attempt_register_native_functions(
        ["rune_register_native"],
        native_pacakge="rune.runtime.native_registry",
    )
    assert registered == ["rune_register_native"]

    def _sample():
        return "ok"

    try:
        rune_execute_native(
            "rune_register_native",
            "test.custom.register",
            _sample,
        )
        assert rune_execute_native("test.custom.register") == "ok"
    finally:
        rune_deregister_native("rune_register_native")
        rune_deregister_native("test.custom.register")


def test_attempt_register_native_functions_with_namespace_prefix(monkeypatch):
    prefix = "prefixed_model"
    package_names = (
        prefix,
        f"{prefix}.rune",
        f"{prefix}.rune.native",
        f"{prefix}.rune.native.cdm",
        f"{prefix}.rune.native.cdm.event",
        f"{prefix}.rune.native.cdm.event.common",
        f"{prefix}.rune.native.cdm.event.common.functions",
    )
    for package_name in package_names:
        module = types.ModuleType(package_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, module)

    module_name = f"{prefix}.rune.native.cdm.event.common.functions.NativeFunc"
    native_module = types.ModuleType(module_name)
    native_module.NativeFunc = lambda n: n + 1
    monkeypatch.setitem(sys.modules, module_name, native_module)

    registered = rune_attempt_register_native_functions(
        ["cdm.event.common.functions.NativeFunc"],
        rune_namespace_prefix=prefix,
    )

    assert registered == ["cdm.event.common.functions.NativeFunc"]
    try:
        assert rune_execute_native("cdm.event.common.functions.NativeFunc", 3) == 4
    finally:
        rune_deregister_native("cdm.event.common.functions.NativeFunc")
