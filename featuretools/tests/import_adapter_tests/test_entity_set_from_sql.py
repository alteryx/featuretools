import featuretools as ft

import sqlite3

import pytest

from sqlalchemy import Table, Column, Integer, Float, String, MetaData, ForeignKey, create_engine

@pytest.fixture
def sqlite_db_conn():
    engine = create_engine('sqlite:///:memory:', echo=True)
    metadata = MetaData()
    widgets = Table('widgets', metadata,
        Column('my_id', Integer, primary_key=True),
        Column('name', String),
        Column('quantity', Integer),
        Column('price', Float)
    )
    metadata.create_all(engine)

    conn = engine.connect()
    widgets = Table('widgets', metadata, autoload=True)
    insert = widgets.insert()
    conn.execute(insert,name="whatsit", quantity="8", price="0.57" )
    return conn

def test_creates_entity_set(sqlite_db_conn):
    es = ft.entity_set_from_sql(id="widgets", connection=sqlite_db_conn)
    assert isinstance(es, ft.EntitySet)

def test_loads_single_table(sqlite_db_conn):
    es = ft.entity_set_from_sql(id="widgets", connection=sqlite_db_conn)
    assert(len(es.entities) == 1)

def test_infers_index_column(sqlite_db_conn):
    es = ft.entity_set_from_sql(id="widgets", connection=sqlite_db_conn)
    assert(es.entities[0].index == 'my_id')
