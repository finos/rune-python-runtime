"""module for managing functions with "native" implementation"""
from typing import Any, Callable

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

# EOF
