import pytest
import sqlalchemy as sa

import featuretools as ft


@pytest.fixture
def sqlite_db_conn():
    engine = sa.create_engine('sqlite:///:memory:', echo=True)
    metadata = sa.MetaData()
    widgets = sa.Table('widgets', metadata,
                       sa.Column('my_id', sa.Integer, primary_key=True),
                       sa.Column('name', sa.String),
                       sa.Column('quantity', sa.Integer),
                       sa.Column('price', sa.Float))
    metadata.create_all(engine)

    conn = engine.connect()
    widgets = sa.Table('widgets', metadata, autoload=True)
    insert = widgets.insert()
    conn.execute(insert, name="whatsit", quantity="8", price="0.57")
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
