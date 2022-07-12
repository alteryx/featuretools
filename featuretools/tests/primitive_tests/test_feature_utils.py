from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, Integer

from featuretools.feature_base.utils import is_valid_input


def test_is_valid_input():
    assert is_valid_input(candidate=ColumnSchema(), template=ColumnSchema())

    assert is_valid_input(
        candidate=ColumnSchema(logical_type=Integer, semantic_tags={"index"}),
        template=ColumnSchema(logical_type=Integer, semantic_tags={"index"}),
    )

    assert is_valid_input(
        candidate=ColumnSchema(
            logical_type=Integer,
            semantic_tags={"index", "numeric"},
        ),
        template=ColumnSchema(semantic_tags={"index"}),
    )

    assert is_valid_input(
        candidate=ColumnSchema(semantic_tags={"index"}),
        template=ColumnSchema(semantic_tags={"index"}),
    )

    assert is_valid_input(
        candidate=ColumnSchema(logical_type=Integer, semantic_tags={"index"}),
        template=ColumnSchema(),
    )

    assert is_valid_input(
        candidate=ColumnSchema(logical_type=Integer),
        template=ColumnSchema(logical_type=Integer),
    )

    assert is_valid_input(
        candidate=ColumnSchema(logical_type=Integer, semantic_tags={"numeric"}),
        template=ColumnSchema(logical_type=Integer),
    )

    assert not is_valid_input(
        candidate=ColumnSchema(logical_type=Integer, semantic_tags={"index"}),
        template=ColumnSchema(logical_type=Double, semantic_tags={"index"}),
    )

    assert not is_valid_input(
        candidate=ColumnSchema(logical_type=Integer, semantic_tags={}),
        template=ColumnSchema(logical_type=Integer, semantic_tags={"index"}),
    )

    assert not is_valid_input(
        candidate=ColumnSchema(),
        template=ColumnSchema(logical_type=Integer, semantic_tags={"index"}),
    )

    assert not is_valid_input(
        candidate=ColumnSchema(),
        template=ColumnSchema(logical_type=Integer),
    )

    assert not is_valid_input(
        candidate=ColumnSchema(),
        template=ColumnSchema(semantic_tags={"index"}),
    )
