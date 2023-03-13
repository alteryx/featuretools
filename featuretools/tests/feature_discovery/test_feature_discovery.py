import pytest
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean

from featuretools.feature_discovery.feature_discovery import index_input_set


@pytest.mark.parametrize(
    "column_list, expected",
    [
        ([ColumnSchema(logical_type=Boolean)], {"Boolean": 1}),
        ([ColumnSchema()], {"ANY": 1}),
        (
            [
                ColumnSchema(logical_type=Boolean),
                ColumnSchema(logical_type=Boolean),
            ],
            {"Boolean": 2},
        ),
    ],
)
def test_index_input_set(column_list, expected):
    actual = index_input_set(column_list)

    assert actual == expected
    # raise Exception("dave")
