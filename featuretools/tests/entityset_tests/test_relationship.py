import featuretools as ft


def test_relationship_path_name(es):
    assert ft.relationship_path_name([], True) == ''
    assert ft.relationship_path_name([], False) == ''

    log_to_sessions = ft.Relationship(es['sessions']['id'],
                                      es['log']['session_id'])
    sessions_to_customers = ft.Relationship(es['customers']['id'],
                                            es['sessions']['customer_id'])

    forward_path = [log_to_sessions, sessions_to_customers]
    assert ft.relationship_path_name(forward_path, True) == 'sessions.customers'

    backward_path = [sessions_to_customers, log_to_sessions]
    assert ft.relationship_path_name(backward_path, False) == 'sessions.log'


def test_names_when_multiple_relationships_between_entities(games_es):
    relationship = ft.Relationship(games_es['teams']['id'],
                                   games_es['games']['home_team_id'])
    assert relationship.child_name == 'games[home_team_id]'
    assert relationship.parent_name == 'teams[home_team_id]'


def test_names_when_no_other_relationship_between_entities(home_games_es):
    relationship = ft.Relationship(home_games_es['teams']['id'],
                                   home_games_es['games']['home_team_id'])
    assert relationship.child_name == 'games'
    assert relationship.parent_name == 'teams'


def test_serialization(es):
    relationship = ft.Relationship(es['sessions']['id'], es['log']['session_id'])

    dictionary = {
        'parent_entity_id': 'sessions',
        'parent_variable_id': 'id',
        'child_entity_id': 'log',
        'child_variable_id': 'session_id',
    }
    assert relationship.to_dictionary() == dictionary
    assert ft.Relationship.from_dictionary(dictionary, es) == relationship
