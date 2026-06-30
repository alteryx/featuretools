"""Tests that confirm primitive input_types lists have the expected shapes.

Every Primitive's input_types must satisfy:
  1. It is either a flat list of ColumnSchema, or a list of lists of ColumnSchema
     (never a mix of the two).
  2. When it is a list of lists, all sub-lists have the same length.
  3. Neither the outer list nor any inner list is empty.
"""
import pytest
from woodwork.column_schema import ColumnSchema

from featuretools.primitives import get_aggregation_primitives, get_transform_primitives


def _all_primitives():
    prims = {}
    prims.update(get_aggregation_primitives())
    prims.update(get_transform_primitives())
    return prims


ALL_PRIMITIVES = _all_primitives()


@pytest.mark.parametrize("name,cls", list(ALL_PRIMITIVES.items()))
def test_input_types_not_empty(name, cls):
    """input_types must not be an empty list."""
    assert len(cls.input_types) > 0, (
        f"{name}.input_types is empty"
    )


@pytest.mark.parametrize("name,cls", list(ALL_PRIMITIVES.items()))
def test_input_types_consistent_nesting(name, cls):
    """input_types must be either all ColumnSchema (flat) or all lists (nested), not a mix."""
    input_types = cls.input_types
    kinds = {type(item).__name__ for item in input_types}
    assert len(kinds) == 1, (
        f"{name}.input_types mixes ColumnSchema and list: found types {kinds}"
    )
    kind = type(input_types[0])
    assert kind in (ColumnSchema, list), (
        f"{name}.input_types items must be ColumnSchema or list, got {kind}"
    )


@pytest.mark.parametrize("name,cls", list(ALL_PRIMITIVES.items()))
def test_input_types_sublists_same_length(name, cls):
    """When input_types is a list of lists, all sub-lists must have the same length."""
    input_types = cls.input_types
    if not isinstance(input_types[0], list):
        return  # flat list — nothing to check

    lengths = {len(sub) for sub in input_types}
    assert len(lengths) == 1, (
        f"{name}.input_types has sub-lists of unequal lengths: {lengths}"
    )


@pytest.mark.parametrize("name,cls", list(ALL_PRIMITIVES.items()))
def test_input_types_sublists_not_empty(name, cls):
    """When input_types is a list of lists, no sub-list may be empty."""
    input_types = cls.input_types
    if not isinstance(input_types[0], list):
        return  # flat list — nothing to check

    for i, sub in enumerate(input_types):
        assert len(sub) > 0, (
            f"{name}.input_types sub-list at index {i} is empty"
        )
