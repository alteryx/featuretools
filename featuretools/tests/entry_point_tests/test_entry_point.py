import pandas as pd
import os
import site
import pandas as pd
import pytest
import json
from distributed.utils_test import cluster

from featuretools.tests.testing_utils import make_ecommerce_entityset

from featuretools.primitives import Max, Mean, Min, Sum
from featuretools.synthesis import dfs
from ..primitive_tests.test_install_primitives import pip_freeze


@pytest.fixture(scope='module')
def es():
    return make_ecommerce_entityset()


@pytest.fixture(scope='module')
def entities():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "card_id": [1, 2, 1, 3, 4, 5],
                                    "transaction_time": [10, 12, 13, 20, 21, 20],
                                    "fraud": [True, False, False, False, True, True]})
    entities = {
        "cards": (cards_df, "id"),
        "transactions": (transactions_df, "id", "transaction_time")
    }
    return entities


@pytest.fixture(scope='module')
def relationships():
    return [("cards", "id", "transactions", "card_id")]


@pytest.fixture(scope='module')
def check_plugin():
    assert "featuretools-plugin-tester" in pip_freeze()
    return


def test_entry_point_primitive(check_plugin):
    from featuretools.primitives import CustomMinPlusOne
    input_array = pd.Series([0, 1, 2, 3, 4])
    primitive_func = CustomMinPlusOne().get_function()
    # CustomMinPlusOne is actually min + plus 1
    assert primitive_func(input_array) == 1
    assert CustomMinPlusOne.name == 'custom_min_plus_one'


def test_entry_point_initialize(check_plugin):
    file_path = os.path.join(site.getsitepackages()[0],
                             'featuretools_plugin_tester',
                             'initialize',
                             'initialize_plugin_called.txt')
    import featuretools
    assert os.path.isfile(file_path)


def test_dfs_plugin(entities, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3],
                                    "time": [10, 12, 15]})
    feature_matrix, features = dfs(entities=entities,
                                   relationships=relationships,
                                   target_entity="transactions",
                                   cutoff_time=cutoff_times_df)
    assert len(feature_matrix.index) == 3
    assert len(feature_matrix.columns) == len(features)

    file_path = os.path.join(site.getsitepackages()[0],
                             'featuretools_plugin_tester',
                             'dfs',
                             'dfs_plugin_tester.json')
    assert os.path.isfile(file_path)
    with open(file_path) as f:
        data = json.load(f)
    assert 'entityset_name' in data