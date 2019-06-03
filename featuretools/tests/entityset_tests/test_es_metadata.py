# -*- coding: utf-8 -*-
import pandas as pd
import pytest

import featuretools as ft
from featuretools import EntitySet, Relationship, variable_types


def test_cannot_re_add_relationships_that_already_exists(es):
    before_len = len(es.relationships)
    es.add_relationship(es.relationships[0])
    after_len = len(es.relationships)
    assert before_len == after_len


def test_add_relationships_convert_type(es):
    for r in es.relationships:
        try:
            assert type(r.parent_variable) == variable_types.Index
            assert type(r.child_variable) == variable_types.Id
        except Exception:
            assert type(r.parent_variable) == variable_types.Index
            assert type(r.child_variable) == variable_types.Id


def test_get_forward_entities(es):
    entities = es.get_forward_entities('log')
    assert entities == set(['sessions', 'products'])


def test_get_backward_entities(es):
    entities = set(e for e, _ in es.get_backward_entities('sessions'))
    assert entities == set(['log'])


def test_get_forward_entities_deep(es):
    entities = es.get_forward_entities('log', 'deep')
    assert entities == set(['sessions', 'customers', 'products', u'régions', 'cohorts'])


def test_get_backward_entities_deep(es):
    entities = set(e for e, _ in es.get_backward_entities('customers', depth=None))
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


def test_find_forward_paths(es):
    paths = list(es.find_forward_paths('log', 'customers'))
    assert len(paths) == 1

    path = paths[0]

    assert len(path) == 2
    assert path[0].child_entity.id == 'log'
    assert path[0].parent_entity.id == 'sessions'
    assert path[1].child_entity.id == 'sessions'
    assert path[1].parent_entity.id == 'customers'


def test_find_forward_paths_multiple_paths(diamond_es):
    paths = list(diamond_es.find_forward_paths('transactions', 'regions'))
    assert len(paths) == 2

    path1, path2 = paths

    r1, r2 = path1
    assert r1.child_entity.id == 'transactions'
    assert r1.parent_entity.id == 'stores'
    assert r2.child_entity.id == 'stores'
    assert r2.parent_entity.id == 'regions'

    r1, r2 = path2
    assert r1.child_entity.id == 'transactions'
    assert r1.parent_entity.id == 'customers'
    assert r2.child_entity.id == 'customers'
    assert r2.parent_entity.id == 'regions'


def test_find_forward_paths_multiple_relationships(games_es):
    paths = list(games_es.find_forward_paths('games', 'teams'))
    assert len(paths) == 2

    path1, path2 = paths
    assert len(path1) == 1
    assert len(path2) == 1
    r1 = path1[0]
    r2 = path2[0]

    assert r1.child_entity.id == 'games'
    assert r2.child_entity.id == 'games'
    assert r1.parent_entity.id == 'teams'
    assert r2.parent_entity.id == 'teams'

    assert r1.child_variable.id == 'home_team_id'
    assert r2.child_variable.id == 'away_team_id'
    assert r1.parent_variable.id == 'id'
    assert r2.parent_variable.id == 'id'


def test_find_forward_paths_ignores_loops():
    employee_df = pd.DataFrame({'id': [0], 'manager_id': [0]})
    entities = {'employees': (employee_df, 'id')}
    relationships = [('employees', 'id', 'employees', 'manager_id')]
    es = ft.EntitySet(entities=entities, relationships=relationships)

    paths = list(es.find_forward_paths('employees', 'employees'))
    assert len(paths) == 1
    assert paths[0] == []


def test_find_backward_paths(es):
    paths = list(es.find_backward_paths('customers', 'log'))
    assert len(paths) == 1

    path = paths[0]

    assert len(path) == 2
    assert path[0].child_entity.id == 'sessions'
    assert path[0].parent_entity.id == 'customers'
    assert path[1].child_entity.id == 'log'
    assert path[1].parent_entity.id == 'sessions'


def test_find_backward_paths_multiple_paths(diamond_es):
    paths = list(diamond_es.find_backward_paths('regions', 'transactions'))
    assert len(paths) == 2

    path1, path2 = paths

    r1, r2 = path1
    assert r1.child_entity.id == 'stores'
    assert r1.parent_entity.id == 'regions'
    assert r2.child_entity.id == 'transactions'
    assert r2.parent_entity.id == 'stores'

    r1, r2 = path2
    assert r1.child_entity.id == 'customers'
    assert r1.parent_entity.id == 'regions'
    assert r2.child_entity.id == 'transactions'
    assert r2.parent_entity.id == 'customers'


def test_find_backward_paths_multiple_relationships(games_es):
    paths = list(games_es.find_backward_paths('teams', 'games'))
    assert len(paths) == 2

    path1, path2 = paths
    assert len(path1) == 1
    assert len(path2) == 1
    r1 = path1[0]
    r2 = path2[0]

    assert r1.child_entity.id == 'games'
    assert r2.child_entity.id == 'games'
    assert r1.parent_entity.id == 'teams'
    assert r2.parent_entity.id == 'teams'

    assert r1.child_variable.id == 'home_team_id'
    assert r2.child_variable.id == 'away_team_id'
    assert r1.parent_variable.id == 'id'
    assert r2.parent_variable.id == 'id'


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


def test_find_path_same_entity(es):
    path, forward = es.find_path('products', 'products',
                                 include_num_forward=True)
    assert len(path) == 0
    assert forward == 0

    # also test include_num_forward==False
    path = es.find_path('products', 'products',
                        include_num_forward=False)
    assert len(path) == 0


def test_find_path_no_path_found(es):
    es.relationships = []
    error_text = "No path from products to customers. Check that all entities are connected by relationships"
    with pytest.raises(ValueError, match=error_text):
        es.find_path('products', 'customers')


def test_has_unique_path(diamond_es):
    assert diamond_es.has_unique_forward_path('customers', 'regions')
    assert not diamond_es.has_unique_forward_path('transactions', 'regions')


def test_raise_key_error_missing_entity(es):
    error_text = "Entity this entity doesn't exist does not exist in ecommerce"
    with pytest.raises(KeyError, match=error_text):
        es["this entity doesn't exist"]

    es_without_id = EntitySet()
    error_text = "Entity this entity doesn't exist does not exist in entity set"
    with pytest.raises(KeyError, match=error_text):
        es_without_id["this entity doesn't exist"]


def test_add_parent_not_index_variable(es):
    error_text = "Parent variable.*is not the index of entity Entity.*"
    with pytest.raises(AttributeError, match=error_text):
        es.add_relationship(Relationship(es[u'régions']['language'],
                                         es['customers'][u'région_id']))
