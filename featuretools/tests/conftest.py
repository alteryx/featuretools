import pandas as pd
import pytest

import featuretools as ft
from featuretools.tests.testing_utils import make_ecommerce_entityset


@pytest.fixture
def es():
    return make_ecommerce_entityset()


@pytest.fixture
def int_es():
    return make_ecommerce_entityset(with_integer_time_index=True)


@pytest.fixture
def diamond_es():
    regions_df = pd.DataFrame({
        'id': range(3),
        'name': ['Northeast', 'Midwest', 'South'],
    })
    stores_df = pd.DataFrame({
        'id': range(5),
        'region_id': [0, 1, 2, 2, 1],
    })
    customers_df = pd.DataFrame({
        'id': range(5),
        'region_id': [1, 0, 0, 1, 1],
        'name': ['A', 'B', 'C', 'D', 'E'],
    })
    transactions_df = pd.DataFrame({
        'id': range(8),
        'store_id': [4, 4, 2, 3, 4, 0, 1, 1],
        'customer_id': [3, 0, 2, 4, 3, 3, 2, 3],
        'amount': [100, 40, 45, 83, 13, 94, 27, 81],
    })

    entities = {
        'regions': (regions_df, 'id'),
        'stores': (stores_df, 'id'),
        'customers': (customers_df, 'id'),
        'transactions': (transactions_df, 'id'),
    }
    relationships = [
        ('regions', 'id', 'stores', 'region_id'),
        ('regions', 'id', 'customers', 'region_id'),
        ('stores', 'id', 'transactions', 'store_id'),
        ('customers', 'id', 'transactions', 'customer_id'),
    ]
    return ft.EntitySet(id='ecommerce_diamond',
                        entities=entities,
                        relationships=relationships)


@pytest.fixture
def home_games_es():
    teams = pd.DataFrame({
        'id': range(3),
        'name': ['Breakers', 'Spirit', 'Thorns']
    })
    games = pd.DataFrame({
        'id': range(5),
        'home_team_id': [2, 2, 1, 0, 1],
        'away_team_id': [1, 0, 2, 1, 0],
        'home_team_score': [3, 0, 1, 0, 4],
        'away_team_score': [2, 1, 2, 0, 0]
    })
    entities = {'teams': (teams, 'id'), 'games': (games, 'id')}
    relationships = [('teams', 'id', 'games', 'home_team_id')]
    return ft.EntitySet(entities=entities,
                        relationships=relationships)


@pytest.fixture
def games_es(home_games_es):
    away_team = ft.Relationship(home_games_es['teams']['id'],
                                home_games_es['games']['away_team_id'])
    return home_games_es.add_relationship(away_team)
