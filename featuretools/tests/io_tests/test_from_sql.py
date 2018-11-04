import pytest

import featuretools as ft

from ..testing_utils import sqlite, sqlite_composite_pk

def test_creates_entity_set(sqlite):
    es = ft.io.from_sql(id="widget_es", connection=sqlite)
    assert isinstance(es, ft.EntitySet)


def test_loads_tables(sqlite):
    es = ft.io.from_sql(id="widget_es", connection=sqlite)
    assert(len(es.entities) == 4)


def test_infers_index_column(sqlite):
    es = ft.io.from_sql(id="widget_es", connection=sqlite)

    assert(es.entity_dict['widgets'].index == 'my_id')
    assert(es.entity_dict['factories'].index == 'id')
    assert(es.entity_dict['customers'].index == 'id')


def test_generates_index_column(sqlite):
    es = ft.io.from_sql(id="widget_es", connection=sqlite)
    assert(es.entity_dict['customers_widgets'].index == 'featuretools_sql_import_id')


def test_infers_relationships(sqlite):
    es = ft.io.from_sql(id="widget_es", connection=sqlite)
    assert(len(es.relationships) == 3)


def test_throws_error_on_composite_primary_key(sqlite_composite_pk):
    with pytest.raises(RuntimeError, match="Composite primary key detected"):
        ft.io.from_sql(id="composite_pk", connection=sqlite_composite_pk)
