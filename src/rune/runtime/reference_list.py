# Copyright (c) 2023-2026 CLOUDRISK Limited and FT Advisory LLC
# SPDX-License-Identifier: Apache-2.0

'''List storage that keeps reference metadata with each occurrence.'''
import copy
import operator
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, Self, SupportsIndex

if TYPE_CHECKING:
    from rune.runtime.metadata import KeyType


class _ReferenceList(list):
    '''Expose targets as list items while retaining their reference key/type.

    Bare items inserted into the list are inline occurrences. Resolved references
    expose their targets immediately; unresolved tokens are bound by the model's
    normal reference-resolution lifecycle.
    '''

    def __init__(self, values: Iterable[Any] = ()) -> None:
        super().__init__(values)
        self._references: list[tuple[str, KeyType] | None]
        if isinstance(values, _ReferenceList):
            self._references = values._references.copy()
        else:
            entries = [self._entry(value) for value in self]
            super().__setitem__(slice(None), [value for value, _ in entries])
            self._references = [reference for _, reference in entries]

    @staticmethod
    def _entry(value: Any) -> tuple[Any, tuple[str, 'KeyType'] | None]:
        # Metadata imports this collection; defer the reverse import until use.
        from rune.runtime.metadata import Reference

        if isinstance(value, Reference):
            return value.target, (value.target_key, value.key_type)
        return value, None

    def reference_at(self, index: int) -> tuple[str, 'KeyType'] | None:
        '''Return the reference recorded for one occurrence, if any.'''
        return self._references[index]

    def has_references(self) -> bool:
        '''Return whether any occurrence has a resolved reference key.'''
        return any(self._references)

    def reference_items(self) -> Iterator[tuple[Any, tuple[str, 'KeyType']]]:
        '''Yield only targets with reference metadata.'''
        return ((value, reference) for value, reference in zip(self, self._references)
                if reference is not None)

    def contained_values(self) -> Iterator[Any]:
        '''Yield inline entries without following reference edges.'''
        if not self.has_references():
            yield from self
        else:
            for value, reference in zip(self, self._references):
                if reference is None:
                    yield value

    def bind(self, index: int, target: Any, key: str, key_type: 'KeyType') -> None:
        '''Replace a reference token with its target without losing its key.'''
        super().__setitem__(index, target)
        self._references[index] = (key, key_type)

    def with_values(self, values: Iterable[Any]) -> Self:
        '''Keep occurrence metadata when validation replaces the item values.'''
        result = type(self)(values)
        if len(result) != len(self):
            raise ValueError('Validation changed the length of a reference list')
        result._references = self._references.copy()
        return result

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if isinstance(index, slice):
            result = type(self)(result)
            result._references = self._references[index]
        return result

    def __setitem__(self, index, value) -> None:
        if isinstance(index, slice):
            value = type(self)(value)
            super().__setitem__(index, value)
            self._references[index] = value._references
        else:
            value, reference = self._entry(value)
            super().__setitem__(index, value)
            self._references[index] = reference

    def __delitem__(self, index) -> None:
        super().__delitem__(index)
        del self._references[index]

    def append(self, value: Any) -> None:
        '''Append an inline value or reference occurrence.'''
        value, reference = self._entry(value)
        super().append(value)
        self._references.append(reference)

    def extend(self, values: Iterable[Any]) -> None:
        '''Append occurrences, preserving metadata from another reference list.'''
        values = type(self)(values)
        super().extend(values)
        self._references.extend(values._references)

    def insert(self, index: SupportsIndex, value: Any) -> None:
        '''Insert an inline value or reference occurrence.'''
        index = operator.index(index)
        value, reference = self._entry(value)
        super().insert(index, value)
        self._references.insert(index, reference)

    def pop(self, index: SupportsIndex = -1) -> Any:
        '''Remove and return a target as a bare value.'''
        index = operator.index(index)
        result = super().pop(index)
        self._references.pop(index)
        return result

    def remove(self, value: Any) -> None:
        '''Remove the first matching occurrence and its metadata.'''
        del self[self.index(value)]

    def clear(self) -> None:
        '''Remove every occurrence.'''
        super().clear()
        self._references.clear()

    def reverse(self) -> None:
        '''Reverse values and their reference metadata together.'''
        super().reverse()
        self._references.reverse()

    def sort(self, *, key=None, reverse: bool = False) -> None:
        '''Sort occurrences without separating values from their metadata.'''
        entries = list(zip(self, self._references))
        entries.sort(key=lambda entry: key(entry[0]) if key is not None else entry[0],
                     reverse=reverse)
        super().__setitem__(slice(None), [value for value, _ in entries])
        self._references[:] = [ref for _, ref in entries]

    def copy(self) -> Self:
        '''Copy the collection without copying its target objects.'''
        return self[:]  # type: ignore

    def __copy__(self) -> Self:
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        result = type(self)()
        memo[id(self)] = result
        list.extend(result, (copy.deepcopy(value, memo) for value in self))
        result._references = self._references.copy()
        return result

    def __iadd__(self, values: Iterable[Any]) -> Self:
        self.extend(values)
        return self

    def __add__(self, values: list) -> Self:
        if not isinstance(values, list):
            return NotImplemented
        result = self.copy()
        result.extend(values)
        return result

    def __radd__(self, values: list) -> Self:
        if not isinstance(values, list):
            return NotImplemented
        result = type(self)(values)
        result.extend(self)
        return result

    def __imul__(self, count: SupportsIndex) -> Self:
        count = operator.index(count)
        super().__imul__(count)
        self._references *= count
        return self

    def __mul__(self, count: SupportsIndex) -> Self:
        result = self.copy()
        result *= count
        return result

    __rmul__ = __mul__
