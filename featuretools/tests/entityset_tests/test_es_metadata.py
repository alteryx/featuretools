import pandas as pd
import pytest

import featuretools as ft
from featuretools import EntitySet, Relationship, variable_types
from featuretools.tests.testing_utils import backward_path, forward_path


def test_cannot_re_add_relationships_that_already_exists(pd_es):
    before_len = len(pd_es.relationships)
    pd_es.add_relationship(pd_es.relationships[0])
    after_len = len(pd_es.relationships)
    assert before_len == after_len


def test_add_relationships_convert_type(pd_es):
    for r in pd_es.relationships:
        assert type(r.parent_variable) == variable_types.Index
        assert type(r.child_variable) == variable_types.Id


def test_get_forward_entities(pd_es):
    entities = pd_es.get_forward_entities('log')
    path_to_sessions = forward_path(pd_es, ['log', 'sessions'])
    path_to_products = forward_path(pd_es, ['log', 'products'])
    assert list(entities) == [('sessions', path_to_sessions), ('products', path_to_products)]


def test_get_backward_entities(pd_es):
    entities = pd_es.get_backward_entities('customers')
    path_to_sessions = backward_path(pd_es, ['customers', 'sessions'])
    assert list(entities) == [('sessions', path_to_sessions)]


def test_get_forward_entities_deep(pd_es):
    entities = pd_es.get_forward_entities('log', deep=True)
    path_to_sessions = forward_path(pd_es, ['log', 'sessions'])
    path_to_products = forward_path(pd_es, ['log', 'products'])
    path_to_customers = forward_path(pd_es, ['log', 'sessions', 'customers'])
    path_to_regions = forward_path(pd_es, ['log', 'sessions', 'customers', u'régions'])
    path_to_cohorts = forward_path(pd_es, ['log', 'sessions', 'customers', 'cohorts'])
    assert list(entities) == [
        ('sessions', path_to_sessions),
        ('customers', path_to_customers),
        ('cohorts', path_to_cohorts),
        (u'régions', path_to_regions),
        ('products', path_to_products),
    ]


def test_get_backward_entities_deep(pd_es):
    entities = pd_es.get_backward_entities('customers', deep=True)
    path_to_log = backward_path(pd_es, ['customers', 'sessions', 'log'])
    path_to_sessions = backward_path(pd_es, ['customers', 'sessions'])
    assert list(entities) == [('sessions', path_to_sessions), ('log', path_to_log)]


def test_get_forward_relationships(pd_es):
    relationships = pd_es.get_forward_relationships('log')
    assert len(relationships) == 2
    assert relationships[0].parent_entity.id == 'sessions'
    assert relationships[0].child_entity.id == 'log'
    assert relationships[1].parent_entity.id == 'products'
    assert relationships[1].child_entity.id == 'log'

    relationships = pd_es.get_forward_relationships('sessions')
    assert len(relationships) == 1
    assert relationships[0].parent_entity.id == 'customers'
    assert relationships[0].child_entity.id == 'sessions'


def test_get_backward_relationships(pd_es):
    relationships = pd_es.get_backward_relationships('sessions')
    assert len(relationships) == 1
    assert relationships[0].parent_entity.id == 'sessions'
    assert relationships[0].child_entity.id == 'log'

    relationships = pd_es.get_backward_relationships('customers')
    assert len(relationships) == 1
    assert relationships[0].parent_entity.id == 'customers'
    assert relationships[0].child_entity.id == 'sessions'


def test_find_forward_paths(pd_es):
    paths = list(pd_es.find_forward_paths('log', 'customers'))
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
    pd_es = ft.EntitySet(entities=entities, relationships=relationships)

    paths = list(pd_es.find_forward_paths('employees', 'employees'))
    assert len(paths) == 1
    assert paths[0] == []


def test_find_backward_paths(pd_es):
    paths = list(pd_es.find_backward_paths('customers', 'log'))
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


def test_has_unique_path(diamond_es):
    assert diamond_es.has_unique_forward_path('customers', 'regions')
    assert not diamond_es.has_unique_forward_path('transactions', 'regions')


def test_raise_key_error_missing_entity(pd_es):
    error_text = "Entity this entity doesn't exist does not exist in ecommerce"
    with pytest.raises(KeyError, match=error_text):
        pd_es["this entity doesn't exist"]

    es_without_id = EntitySet()
    error_text = "Entity this entity doesn't exist does not exist in entity set"
    with pytest.raises(KeyError, match=error_text):
        es_without_id["this entity doesn't exist"]


def test_add_parent_not_index_variable(pd_es):
    error_text = "Parent variable.*is not the index of entity Entity.*"
    with pytest.raises(AttributeError, match=error_text):
        pd_es.add_relationship(Relationship(pd_es[u'régions']['language'],
                                            pd_es['customers'][u'région_id']))
