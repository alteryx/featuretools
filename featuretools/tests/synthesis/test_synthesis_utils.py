from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, Integer

from featuretools.synthesis.utils import _schemas_equal


def test_schemas_equal():
    assert _schemas_equal(feature_schema=ColumnSchema(), primitive_schema=ColumnSchema())

    assert _schemas_equal(feature_schema=ColumnSchema(logical_type=Integer, semantic_tags={'index'}),
                          primitive_schema=ColumnSchema(logical_type=Integer, semantic_tags={'index'}))

    assert _schemas_equal(feature_schema=ColumnSchema(logical_type=Integer, semantic_tags={'index', 'numeric'}),
                          primitive_schema=ColumnSchema(semantic_tags={'index'}))

    assert _schemas_equal(feature_schema=ColumnSchema(semantic_tags={'index'}),
                          primitive_schema=ColumnSchema(semantic_tags={'index'}))

    assert _schemas_equal(feature_schema=ColumnSchema(logical_type=Integer, semantic_tags={'index'}),
                          primitive_schema=ColumnSchema())

    assert _schemas_equal(feature_schema=ColumnSchema(logical_type=Integer),
                          primitive_schema=ColumnSchema(logical_type=Integer))

    assert _schemas_equal(feature_schema=ColumnSchema(logical_type=Integer, semantic_tags={'numeric'}),
                          primitive_schema=ColumnSchema(logical_type=Integer))

    assert not _schemas_equal(feature_schema=ColumnSchema(logical_type=Integer, semantic_tags={'index'}),
                              primitive_schema=ColumnSchema(logical_type=Double, semantic_tags={'index'}))

    assert not _schemas_equal(feature_schema=ColumnSchema(logical_type=Integer, semantic_tags={}),
                              primitive_schema=ColumnSchema(logical_type=Integer, semantic_tags={'index'}))

    assert not _schemas_equal(feature_schema=ColumnSchema(),
                              primitive_schema=ColumnSchema(logical_type=Integer, semantic_tags={'index'}))
