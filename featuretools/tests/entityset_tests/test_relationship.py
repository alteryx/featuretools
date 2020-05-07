from featuretools.entityset.relationship import Relationship, RelationshipPath


def test_relationship_path(pd_es):
    log_to_sessions = Relationship(pd_es['sessions']['id'],
                                   pd_es['log']['session_id'])
    sessions_to_customers = Relationship(pd_es['customers']['id'],
                                         pd_es['sessions']['customer_id'])
    path_list = [(True, log_to_sessions),
                 (True, sessions_to_customers),
                 (False, sessions_to_customers)]
    path = RelationshipPath(path_list)

    for i, edge in enumerate(path_list):
        assert path[i] == edge

    assert [edge for edge in path] == path_list


def test_relationship_path_name(pd_es):
    assert RelationshipPath([]).name == ''

    log_to_sessions = Relationship(pd_es['sessions']['id'],
                                   pd_es['log']['session_id'])
    sessions_to_customers = Relationship(pd_es['customers']['id'],
                                         pd_es['sessions']['customer_id'])

    forward_path = [(True, log_to_sessions), (True, sessions_to_customers)]
    assert RelationshipPath(forward_path).name == 'sessions.customers'

    backward_path = [(False, sessions_to_customers), (False, log_to_sessions)]
    assert RelationshipPath(backward_path).name == 'sessions.log'

    mixed_path = [(True, log_to_sessions), (False, log_to_sessions)]
    assert RelationshipPath(mixed_path).name == 'sessions.log'


def test_relationship_path_entities(pd_es):
    assert list(RelationshipPath([]).entities()) == []

    log_to_sessions = Relationship(pd_es['sessions']['id'],
                                   pd_es['log']['session_id'])
    sessions_to_customers = Relationship(pd_es['customers']['id'],
                                         pd_es['sessions']['customer_id'])

    forward_path = [(True, log_to_sessions), (True, sessions_to_customers)]
    assert list(RelationshipPath(forward_path).entities()) == ['log', 'sessions', 'customers']

    backward_path = [(False, sessions_to_customers), (False, log_to_sessions)]
    assert list(RelationshipPath(backward_path).entities()) == ['customers', 'sessions', 'log']

    mixed_path = [(True, log_to_sessions), (False, log_to_sessions)]
    assert list(RelationshipPath(mixed_path).entities()) == ['log', 'sessions', 'log']


def test_names_when_multiple_relationships_between_entities(games_es):
    relationship = Relationship(games_es['teams']['id'],
                                games_es['games']['home_team_id'])
    assert relationship.child_name == 'games[home_team_id]'
    assert relationship.parent_name == 'teams[home_team_id]'


def test_names_when_no_other_relationship_between_entities(home_games_es):
    relationship = Relationship(home_games_es['teams']['id'],
                                home_games_es['games']['home_team_id'])
    assert relationship.child_name == 'games'
    assert relationship.parent_name == 'teams'


def test_relationship_serialization(pd_es):
    relationship = Relationship(pd_es['sessions']['id'], pd_es['log']['session_id'])

    dictionary = {
        'parent_entity_id': 'sessions',
        'parent_variable_id': 'id',
        'child_entity_id': 'log',
        'child_variable_id': 'session_id',
    }
    assert relationship.to_dictionary() == dictionary
    assert Relationship.from_dictionary(dictionary, pd_es) == relationship
