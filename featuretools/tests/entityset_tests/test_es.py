import os
import shutil

import pytest

from ..testing_utils import make_ecommerce_entityset

from featuretools import EntitySet, Relationship, variable_types
from featuretools.tests import integration_data


@pytest.fixture
def es():
    return make_ecommerce_entityset()


def test_cannot_readd_relationships_that_already_exists(es):
    before_len = len(es.relationships)
    es.add_relationship(es.relationships[0])
    after_len = len(es.relationships)
    assert before_len == after_len


def test_add_relationships_convert_type(es):
    for r in es.relationships:
        parent_e = es[r.parent_entity.id]
        child_e = es[r.child_entity.id]
        assert type(r.parent_variable) == variable_types.Index
        assert type(r.child_variable) == variable_types.Id
        assert parent_e.df[r.parent_variable.id].dtype == child_e.df[r.child_variable.id].dtype


def test_get_forward_entities(es):
    entities = es.get_forward_entities('log')
    assert entities == set(['sessions', 'products'])


def test_get_backward_entities(es):
    entities = es.get_backward_entities('sessions')
    assert entities == set(['log'])


def test_get_forward_entities_deep(es):
    entities = es.get_forward_entities('log', 'deep')
    assert entities == set(['sessions', 'customers', 'products', 'regions', 'cohorts'])


def test_get_backward_entities_deep(es):
    entities = es.get_backward_entities('customers', deep=True)
    assert entities == set(['log', 'sessions'])


def test_get_forward_relationships(es):
    relationships = es.get_forward_relationships('log')
    assert len(relationships) == 2
    assert relationships[0].parent_entity.id == 'sessions'
    assert relationships[0].child_entity.id == 'log'
    assert relationships[1].parent_entity.id == 'products'
    assert relationships[1].child_entity.id == 'log'

    relationships = es.get_forward_relationships('sessions')
    assert len(relationships) == 1
    assert relationships[0].parent_entity.id == 'customers'
    assert relationships[0].child_entity.id == 'sessions'


def test_get_backward_relationships(es):
    relationships = es.get_backward_relationships('sessions')
    assert len(relationships) == 1
    assert relationships[0].parent_entity.id == 'sessions'
    assert relationships[0].child_entity.id == 'log'

    relationships = es.get_backward_relationships('customers')
    assert len(relationships) == 1
    assert relationships[0].parent_entity.id == 'customers'
    assert relationships[0].child_entity.id == 'sessions'


def test_find_forward_path(es):
    path = es.find_forward_path('log', 'customers')

    assert len(path) == 2
    assert path[0].child_entity.id == 'log'
    assert path[0].parent_entity.id == 'sessions'
    assert path[1].child_entity.id == 'sessions'
    assert path[1].parent_entity.id == 'customers'


def test_find_backward_path(es):
    path = es.find_backward_path('customers', 'log')

    assert len(path) == 2
    assert path[0].child_entity.id == 'sessions'
    assert path[0].parent_entity.id == 'customers'
    assert path[1].child_entity.id == 'log'
    assert path[1].parent_entity.id == 'sessions'


def test_find_path(es):
    path, forward = es.find_path('products', 'customers',
                                 include_num_forward=True)

    assert len(path) == 3
    assert forward == 2
    assert path[0].child_entity.id == 'log'
    assert path[0].parent_entity.id == 'products'
    assert path[1].child_entity.id == 'log'
    assert path[1].parent_entity.id == 'sessions'
    assert path[2].child_entity.id == 'sessions'
    assert path[2].parent_entity.id == 'customers'


def test_raise_key_error_missing_entity(es):
    with pytest.raises(KeyError):
        es["this entity doesn't exist"]


def test_add_parent_not_index_varible(es):
    with pytest.raises(AttributeError):
        es.add_relationship(Relationship(es['regions']['language'],
                                         es['customers']['region_id']))


def test_serialization(es):
    dirname = os.path.dirname(integration_data.__file__)
    path = os.path.join(dirname, 'test_entityset.p')
    if os.path.exists(path):
        shutil.rmtree(path)
    es.to_pickle(path)
    new_es = EntitySet.read_pickle(path)
    assert es.__eq__(new_es, deep=True)
    shutil.rmtree(path)
