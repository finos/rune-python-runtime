'''Utility functions (runtime) for rune models.'''
from __future__ import annotations
import logging
import keyword
import inspect
import datetime
from collections.abc import MutableSequence
from enum import Enum
from typing import Callable, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from dateutil import parser

__all__ = [
    'if_cond_fn', 'Multiprop', 'rune_any_elements', 'rune_get_only_element',
    'rune_filter', 'rune_all_elements', 'rune_contains', 'rune_disjoint',
    'rune_join', 'rune_flatten_list', 'rune_resolve_attr',
    'rune_resolve_deep_attr', 'rune_count', 'rune_attr_exists',
    'rune_add_to_list', 'rune_check_cardinality', 'rune_str',
    'rune_check_one_of', 'rune_zoned_date_time', 'rune_wrap_none'
]


def if_cond_fn(ifexpr: bool, thenexpr: Callable, elseexpr: Callable) -> Any:
    ''' A helper to return the value of the ternary operator
        (functional version).
    '''
    expr = thenexpr if ifexpr else elseexpr
    return expr()


def _is_rune_list_like(obj: Any) -> bool:
    '''Return whether an object should follow Rune list semantics.'''
    return isinstance(obj, (tuple, MutableSequence))


def _to_list(obj: Any) -> tuple[Any, ...] | MutableSequence[Any]:
    if _is_rune_list_like(obj):
        return obj
    return (obj, )


def rune_mangle_name(attrib: str) -> str:
    ''' Mangle any attrib that is a Python keyword, is a Python soft keyword
        or begins with _
    '''
    if (keyword.iskeyword(attrib) or keyword.issoftkeyword(attrib)
            or attrib.startswith('_')):
        return 'rune_attr_' + attrib
    return attrib


def rune_resolve_attr(obj: Any | None, attrib: str) -> Any | list[Any] | None:
    ''' rune semantics compliant attribute resolver.
        Lists and mangled attributes are treated as defined by
        the rune definition (list flattening).
    '''
    if obj is None:
        return None

    if isinstance(obj, (list, tuple)):
        res = [
            item for elem in obj
            for item in _to_list(rune_resolve_attr(elem, attrib))
            if item is not None
        ]
        return res if res else None

    attrib = rune_mangle_name(attrib)

    if inspect.isframe(obj):
        return obj.f_locals.get(attrib)
    elif isinstance(obj, dict):
        return obj.get(attrib)
    elif hasattr(obj, '_rune_get_attr'):
        # Custom resolver hook for draft-like objects.
        return obj._rune_get_attr(attrib)

    return getattr(obj, attrib, None)


def rune_resolve_deep_attr(obj: Any | None,
                              attrib: str) -> Any | list[Any] | None:
    ''' Resolves a "deep path" attribute. If the attribute or the object is
        not a "deep path" one, the function falls back to the regular
        `rosetta_resolve_attr`.
    '''
    # pylint: disable=protected-access
    if obj is None:
        return None
    # if not a "deep path" object or attribute, fall back to the std function
    if (not hasattr(obj, '_CHOICE_ALIAS_MAP')
            or attrib not in obj._CHOICE_ALIAS_MAP):
        return rune_resolve_attr(obj, attrib)

    for container_nm, getter_fn in obj._CHOICE_ALIAS_MAP[attrib]:
        if container_obj := rune_resolve_attr(obj, container_nm):
            return getter_fn(container_obj, attrib)
    return None


def rune_check_one_of(obj, *attr_names, necessity=True) -> bool:
    '''Checks that one and only one attribute is set.'''
    if inspect.isframe(obj):
        vals = [obj.f_locals.get(n) for n in attr_names]
    else:
        vals = [getattr(obj, n, None) for n in attr_names]
    n_attr = sum(1 for v in vals if rune_attr_exists(v))
    if necessity and n_attr != 1:
        logging.error('One and only one of %s should be set!', attr_names)
        return False
    if not necessity and n_attr > 1:
        logging.error('Only one of %s can be set!', attr_names)
        return False
    return True


def rune_count(obj: Any | None) -> int:
    '''Implements the lose count semantics of the rune DSL'''
    if not obj:
        return 0
    try:
        return len(obj)
    except TypeError:
        return 1


def rune_attr_exists(val: Any) -> bool:
    '''Implements the rune semantics of property existence'''
    if val is None:
        return False
    if isinstance(val, MutableSequence) and len(val) == 0:
        return False
    return True


def rune_str(x: Any) -> str:
    '''Returns a rune conform string representation'''
    if isinstance(x, Enum):
        x = x.value
    return str(x)


def rune_zoned_date_time(dt_str: str) -> datetime.datetime:
    """
    Return a `datetime` parsed from *dt_str*.

    The input may contain:
      • a bare date-time,  
      • a numeric UTC offset (e.g. “+02:00”),  
      • an IANA time-zone name (e.g. “Europe/Paris”), or  
      • both offset **and** zone (the offset is validated against the zone).

    Parameters
    ----------
    dt_str : str
        Date/time string such as  
        “2024-06-01 12:34:56 +05:30 Asia/Kolkata”.

    Returns
    -------
    datetime.datetime
        Time-zone-aware when an offset or zone is supplied, otherwise naive.

    Raises
    ------
    ValueError
        If the supplied offset contradicts the IANA zone.

    Notes
    -----
    Parsing is delegated to `dateutil.parser.parse`; any of its
    `ParserError`s propagate unchanged for malformed date fragments.
    """
    dt: datetime.datetime
    extras: list[Any]
    dt, extras = parser.parse(dt_str, fuzzy_with_tokens=True)  # type: ignore

    # Try every leftover token (strip commas/periods) for a ZoneInfo name
    zone = None
    for tok in extras:
        tok = tok.strip(' ,.;')
        if not tok:
            continue
        try:
            if tok:
                zone = ZoneInfo(tok)
                break
        except ZoneInfoNotFoundError:
            continue

    if zone:
        # Same offset-matching logic as before …
        if dt.tzinfo and dt.utcoffset() != dt.astimezone(zone).utcoffset():
            raise ValueError("Offset mismatch …")
        dt = dt.astimezone(zone) if dt.tzinfo else dt.replace(tzinfo=zone)

    return dt


class Multiprop(list):
    ''' A class allowing for dot access to a attribute of all elements of a
        list.
    '''
    def __getattr__(self, attr):
        # return multiprop(getattr(x, attr) for x in self)
        res = Multiprop()
        for x in self:
            if isinstance(x, Multiprop):
                res.extend(x.__getattr__(attr))
            else:
                res.append(getattr(x, attr))
        return res


def _ntoz(v):
    '''Support the lose rune treatment of None in comparisons'''
    if v is None:
        return 0
    unwrap = getattr(v, '_unwrap', None)
    if callable(unwrap):
        v = unwrap()
    return v


_cmp = {
    '=': lambda x, y: _ntoz(x) == _ntoz(y),
    '<>': lambda x, y: _ntoz(x) != _ntoz(y),
    '>=': lambda x, y: _ntoz(x) >= _ntoz(y),
    '<=': lambda x, y: _ntoz(x) <= _ntoz(y),
    '>': lambda x, y: _ntoz(x) > _ntoz(y),
    '<': lambda x, y: _ntoz(x) < _ntoz(y)
}


def rune_all_elements(lhs, op, rhs) -> bool:
    '''Checks that two lists have the same elements'''
    cmp = _cmp[op]
    op1 = _to_list(lhs)
    op2 = _to_list(rhs)

    return all(
        cmp(el1, el2)
        for el1, el2 in zip(op1, op2)) if len(op1) == len(op2) else False


def rune_disjoint(op1, op2):
    '''Checks if two lists have no common elements'''
    op1 = set(_to_list(op1))
    op2 = set(_to_list(op2))
    return not op1 & op2


def rune_contains(op1, op2):
    ''' Checks if op2 is contained in op1
        (e.g. every element of op2 is in op1)
    '''
    op1 = set(_to_list(op1))
    op2 = set(_to_list(op2))

    return op2.issubset(op1)


def rune_join(lst, sep=''):
    ''' Joins the string representation of the list elements, optionally
        separated.
    '''
    return sep.join([str(el) for el in lst])


def rune_any_elements(lhs, op, rhs) -> bool:
    '''Checks if to lists have any common element(s)'''
    cmp = _cmp[op]
    op1 = _to_list(lhs)
    op2 = _to_list(rhs)

    return any(cmp(el1, el2) for el1 in op1 for el2 in op2)


def rune_check_cardinality(prop, inf: int, sup: int | None = None) -> bool:
    ''' If the supremum is not supplied (e.g. is None), the property is
        unbounded (e.g. it corresponds to (x..*) in rune).
    '''
    if not prop:
        prop_card = 0
    elif _is_rune_list_like(prop):
        prop_card = len(prop)
    else:
        prop_card = 1

    if sup is None:
        sup = prop_card

    return inf <= prop_card <= sup


def rune_get_only_element(collection):
    ''' Returns the single element of a collection, if the list contains more
        more than one element or is empty, None is returned.
    '''
    if _is_rune_list_like(collection) and len(collection) == 1:
        return collection[0]
    return None


def rune_flatten_list(nested_list):
    '''flattens the list of lists (no-recursively)'''
    return [
        item for sublist in _to_list(nested_list) for item in _to_list(sublist)
    ]


def rune_filter(items, filter_func):
    '''
    Filters a list of items based on a specified filtering criteria provided as
    a boolean lambda function.

    :param items: List of items to be filtered.
    :param filter_func: A lambda function representing the boolean expression
        for filtering.
    :param item_name: The name used to refer to each item in the boolean
        expression.
    :return: Filtered list.
    '''
    return [item for item in (items or []) if filter_func(item)]


def rune_add_to_list(rune_list: list, value: Any) -> None:
    '''
    Adds a value to a list-like attribute of a rune model object.

    Parameters:
    obj (Any): The object whose attribute is to be modified.
    attrib (str): The list-like attribute to add the value to.
    value (Any): The value to add to the attribute.
    '''
    if isinstance(rune_list, MutableSequence):
        if isinstance(value, MutableSequence):
            rune_list.extend(value)
        else:
            rune_list.append(value)
    else:
        raise ValueError(
            "The rune_list for rune_add_to_list has to be an isntance of list "
            "and cannot be None.")


class _NullObject:
    '''
    Replicates the rune null comparison behaviour as defined in:
    https://github.com/finos/rune-dsl/blob/main/docs/rune-modelling-component.md#comparison-operator-and-null
    '''
    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __lt__(self, other):
        return True


_NULLOBJECT = _NullObject()


def rune_wrap_none(item):
    """
    Return a comparison-safe sentinel for `None` values.

    If `item` is `None`, this returns `NULLOBJECT`; otherwise it returns
    `item` unchanged.
    """
    if item is None:
        return _NULLOBJECT
    return item
# EOF
