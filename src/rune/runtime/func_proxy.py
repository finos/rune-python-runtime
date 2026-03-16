'''func proxy'''
import functools
import inspect
from contextlib import contextmanager
from typing import Any, Callable

from pydantic import ValidationError

from rune.runtime.cow import rune_cow, rune_unwrap
from rune.runtime.object_builder import ObjectBuilder

__all__ = [
    'FuncProxy', 'replaceable', 'create_module_attr_guardian',
    'rune_finalize_return', 'rune_call_unchecked_raw', 'rune_call_unchecked'
]


def rune_finalize_return(value: Any) -> Any:
    '''Finalize function return values by unwrapping and materializing drafts.'''
    unwrapped = rune_unwrap(value)
    if isinstance(unwrapped, ObjectBuilder):
        try:
            return unwrapped.to_model()
        except ValidationError:
            return unwrapped
    return unwrapped


def rune_call_unchecked_raw(function: Callable[..., Any], /, *args,
                            **kwargs) -> Any:
    '''Invoke a callable without pydantic `validate_call` checks when possible.

    If `function` is a replaceable Rune function, the current proxy target is
    used. If that target is a pydantic `validate_call` wrapper, its
    `raw_function` is invoked instead. The raw result is returned unchanged.
    '''
    if not callable(function):
        raise TypeError(
            f'Expected a callable, got {type(function).__name__}')

    proxy = getattr(function, '__proxy__', None)
    target = proxy.func if proxy is not None else function
    target = getattr(target, 'raw_function', target)

    return target(*args, **kwargs)


def rune_call_unchecked(function: Callable[..., Any], /, *args, **kwargs) -> Any:
    '''Invoke a callable unchecked and return a COW-wrapped result.'''
    return rune_cow(rune_call_unchecked_raw(function, *args, **kwargs))


class FuncProxy:
    '''A callable proxy allowing functions to be replaced at runtime'''
    __slots__ = ('_func',)

    def __init__(self, func):
        self._func = func

    def __call__(self, *args, **kwargs):
        '''pass the call to the current function'''
        return rune_finalize_return(self._func(*args, **kwargs))

    @property
    def func(self):
        '''current function'''
        return self._func

    @func.setter
    def func(self, func):
        '''replace the current function with a new one'''
        self.__assign__(func)

    def __assign__(self, func):
        '''assigns the new function and checks parameter list compatibility'''
        if not callable(func):
            raise ValueError(f'Need a callable, but got {str(func)}')

        curr_params = inspect.signature(self._func).parameters
        new_params = inspect.signature(func).parameters
        if curr_params.keys() != new_params.keys():
            raise ValueError(
                'Replacement function parameter list do not match the current '
                f'parameter list of {str(self._func)}'
            )
        self._func = func


def replaceable(func):
    '''wrapper for a function which can be replaced at runtime'''
    proxy = FuncProxy(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return proxy(*args, **kwargs)

    wrapper.__assign__ = proxy.__assign__  # type: ignore
    wrapper.__proxy__ = proxy   # type: ignore
    return wrapper


@contextmanager
def scoped_replace(proxy_func, replacement):
    ''' Temporarily replace a replaceable function and restore after use.
        Raises TypeError if not replaceable.
    '''
    if not (hasattr(proxy_func, "__assign__")
            and hasattr(proxy_func, "__proxy__")):
        raise TypeError(
            "Function is not replaceable (missing __assign__/__proxy__).")

    proxy = proxy_func.__proxy__
    original = proxy.func
    proxy.func = replacement
    try:
        yield
    finally:
        proxy.func = original


def create_module_attr_guardian(module):
    '''Returns a module setter class derived from the invoking module'''
    # pylint: disable=too-few-public-methods
    class ModuleAttrSetter(module):
        ''' Redirects the assignment of an attribute to its __assign__ method
            if defined, otherwise the default functionality is used and the
            attribute is just replaced.
        '''
        def __setattr__(self, attr, val):
            exists = getattr(self, attr, None)
            if exists is not None and hasattr(exists, '__assign__'):
                exists.__assign__(val)
            else:
                super().__setattr__(attr, val)
    return ModuleAttrSetter

# EOF
