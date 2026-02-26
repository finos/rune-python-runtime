"""module for managing functions with "native" implementation"""
from importlib import import_module
from typing import Any, Callable
import logging

_NATIVE_REGISTRY: dict[str, Callable] = {}


def rune_execute_native(function_name: str, *args, **kwargs) -> Any:
    """Execute a registered native function by name."""
    if function := _NATIVE_REGISTRY.get(function_name):
        return function(*args, **kwargs)
    available = ", ".join(sorted(_NATIVE_REGISTRY))
    raise NotImplementedError(
        f"Function {function_name} doesn't have an implementation! "
        f"Available: {available or '<none>'}"
    )

def rune_register_native(function_name: str, function: Callable) -> None:
    """Register a native function implementation under the given name."""
    if not callable(function):
        raise TypeError(f"{function_name} must be callable, "
                        f"got {type(function).__name__}")
    _NATIVE_REGISTRY[function_name] = function


def rune_deregister_native(function_name: str) -> None:
    """Removes a native function implementation from the registry."""
    if function_name in _NATIVE_REGISTRY:
        del _NATIVE_REGISTRY[function_name]


def rune_attempt_register_native_functions(
    function_names: list[str], native_pacakge="rune.native"
) -> list[str]:
    """Attempt to import and register native implementations for named functions."""
    registered: list[str] = []
    for function_name in function_names:
        parts = function_name.split(".")
        attr_name = parts[-1]
        module_name = '.'.join([native_pacakge]+parts[:-1])
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            logging.warning('Native function module import failed: %s', exc)
            if exc.name == module_name:
                continue
            raise

        native_impl = getattr(module, attr_name, None)
        if native_impl is None or not callable(native_impl):
            logging.warning(
                'Ignored native function %s. It was either not found or it is '
                'not a callable', function_name)
            continue

        rune_register_native(function_name, native_impl)
        registered.append(function_name)

    return registered

# EOF
