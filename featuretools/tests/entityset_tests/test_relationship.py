from featuretools.entityset.relationship import Relationship, RelationshipPath


def test_relationship_path(es):
    log_to_sessions = Relationship(es, "sessions", "id", "log", "session_id")
    sessions_to_customers = Relationship(
        es,
        "customers",
        "id",
        "sessions",
        "customer_id",
    )
    path_list = [
        (True, log_to_sessions),
        (True, sessions_to_customers),
        (False, sessions_to_customers),
    ]
    path = RelationshipPath(path_list)

    for i, edge in enumerate(path_list):
        assert path[i] == edge

    assert [edge for edge in path] == path_list


def test_relationship_path_name(es):
    assert RelationshipPath([]).name == ""

    log_to_sessions = Relationship(es, "sessions", "id", "log", "session_id")
    sessions_to_customers = Relationship(
        es,
        "customers",
        "id",
        "sessions",
        "customer_id",
    )

    forward_path = [(True, log_to_sessions), (True, sessions_to_customers)]
    assert RelationshipPath(forward_path).name == "sessions.customers"

    backward_path = [(False, sessions_to_customers), (False, log_to_sessions)]
    assert RelationshipPath(backward_path).name == "sessions.log"

    mixed_path = [(True, log_to_sessions), (False, log_to_sessions)]
    assert RelationshipPath(mixed_path).name == "sessions.log"


def test_relationship_path_dataframes(es):
    assert list(RelationshipPath([]).dataframes()) == []

    log_to_sessions = Relationship(es, "sessions", "id", "log", "session_id")
    sessions_to_customers = Relationship(
        es,
        "customers",
        "id",
        "sessions",
        "customer_id",
    )

    forward_path = [(True, log_to_sessions), (True, sessions_to_customers)]
    assert list(RelationshipPath(forward_path).dataframes()) == [
        "log",
        "sessions",
        "customers",
    ]

    backward_path = [(False, sessions_to_customers), (False, log_to_sessions)]
    assert list(RelationshipPath(backward_path).dataframes()) == [
        "customers",
        "sessions",
        "log",
    ]

    mixed_path = [(True, log_to_sessions), (False, log_to_sessions)]
    assert list(RelationshipPath(mixed_path).dataframes()) == ["log", "sessions", "log"]


def test_names_when_multiple_relationships_between_dataframes(games_es):
    relationship = Relationship(games_es, "teams", "id", "games", "home_team_id")
    assert relationship.child_name == "games[home_team_id]"
    assert relationship.parent_name == "teams[home_team_id]"


def test_names_when_no_other_relationship_between_dataframes(home_games_es):
    relationship = Relationship(home_games_es, "teams", "id", "games", "home_team_id")
    assert relationship.child_name == "games"
    assert relationship.parent_name == "teams"


def test_relationship_serialization(es):
    relationship = Relationship(es, "sessions", "id", "log", "session_id")

    dictionary = {
        "parent_dataframe_name": "sessions",
        "parent_column_name": "id",
        "child_dataframe_name": "log",
        "child_column_name": "session_id",
    }
    assert relationship.to_dictionary() == dictionary
    assert Relationship.from_dictionary(dictionary, es) == relationship
