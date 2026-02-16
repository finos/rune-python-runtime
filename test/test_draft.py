import pytest
from typing import Any
from rune.runtime.draft import Draft
from rune.runtime.base_data_class import BaseDataClass


def test_basic_initialisation():
    p = Draft()
    p.a = 9
    assert p.a == 9

    p = Draft(a=8)
    assert p.a == 8

    p = Draft(model_cls='Class')
    assert p.model_cls == 'Class'

    with pytest.raises(ValueError):
        p.to_model()

    p = Draft(_attrib=7)
    assert p.rune_attr__attrib == 7

    p = Draft()
    p.a.b.c1.d = 7
    p.a.b.c2 = 8
    assert p.a.b.c1.d == 7
    assert p.a.b.c2 == 8

    aux = p.to_dict()
    assert aux == {'a':{'b':{'c1':{'d':7},'c2':8}}}


def test_to_model():
    class DraftModel(BaseDataClass):
        a: dict[str, Any]
        value: int

    p = Draft(DraftModel)
    p.value = 3
    p.a.b = 5

    obj = p.to_model()
    assert isinstance(obj, DraftModel)
    assert obj.value == 3
    assert obj.a == {'b': 5}


def test_to_model_with_basedataclass_property():
    class Child(BaseDataClass):
        x: int

    class Parent(BaseDataClass):
        child: Child
        value: int

    p = Draft(Parent)
    p.value = 11
    p.child.x = 7

    obj = p.to_model()
    assert isinstance(obj, Parent)
    assert obj.value == 11
    assert isinstance(obj.child, Child)
    assert obj.child.x == 7

# EOF
