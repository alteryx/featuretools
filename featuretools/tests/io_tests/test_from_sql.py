# flake8: noqa: F811
from unittest.mock import patch

import pytest

import featuretools as ft

from ..testing_utils import sqlite, sqlite_composite_pk  # noqa: F401; pylint: disable=unused-variable


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


def test_passes_through_parameters(sqlite):
    es = ft.io.from_sql(id="widget_es", connection=sqlite, variable_types={"widgets": {'name': ft.variable_types.Categorical}})
    assert(es.entity_dict['widgets']['name'].dtype == 'categorical')


def test_throws_an_error_without_connection_or_connection_string(sqlite):
    with pytest.raises(ValueError, match="connection or connection_string is required"):
        ft.io.from_sql(id="no_connection")

def test_loads_only_specified_tables(sqlite):
    es = ft.io.from_sql(id="widget_limited_tables", connection=sqlite, tables=['widgets', 'factories'])
    assert(len(es.entities) == 2)
    assert(len(es.relationships) == 1)
