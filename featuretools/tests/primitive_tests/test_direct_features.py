import numpy as np
import pandas as pd
import pytest

import featuretools as ft
from featuretools.computational_backends import PandasBackend
from featuretools.feature_base import DirectFeature, Feature
from featuretools.feature_base.features_deserializer import (
    FeaturesDeserializer
)
from featuretools.feature_base.features_serializer import FeaturesSerializer
from featuretools.primitives import (
    AggregationPrimitive,
    Day,
    Hour,
    Minute,
    Month,
    Second,
    TransformPrimitive,
    Year
)
from featuretools.primitives.utils import PrimitivesDeserializer
from featuretools.synthesis import dfs
from featuretools.variable_types import Categorical, Datetime, Numeric


def test_direct_from_identity(es):
    device = es['sessions']['device_type']
    d = DirectFeature(base_feature=device, child_entity=es['log'])

    pandas_backend = PandasBackend(es, [d])
    df = pandas_backend.calculate_all_features(instance_ids=[0, 5],
                                               time_last=None)
    v = df[d.get_name()].tolist()
    assert v == [0, 1]


def test_direct_from_variable(es):
    # should be same behavior as test_direct_from_identity
    device = es['sessions']['device_type']
    d = DirectFeature(base_feature=device,
                      child_entity=es['log'])

    pandas_backend = PandasBackend(es, [d])
    df = pandas_backend.calculate_all_features(instance_ids=[0, 5],
                                               time_last=None)
    v = df[d.get_name()].tolist()
    assert v == [0, 1]


def test_direct_rename(es):
    # should be same behavior as test_direct_from_identity
    feat = DirectFeature(base_feature=es['sessions']['device_type'],
                         child_entity=es['log'])
    copy_feat = feat.rename("session_test")
    assert feat.hash() != copy_feat.hash()
    assert feat.get_name() != copy_feat.get_name()
    assert feat.base_features[0].generate_name() == copy_feat.base_features[0].generate_name()
    assert feat.entity == copy_feat.entity


def test_direct_copy(games_es):
    home_team = next(r for r in games_es.relationships
                     if r.child_variable.id == 'home_team_id')
    feat = DirectFeature(games_es['teams']['name'], games_es['games'],
                         relationship_path=[home_team])
    copied = feat.copy()
    assert copied.entity == feat.entity
    assert copied.base_features == feat.base_features
    assert copied.relationship_path == feat.relationship_path


def test_direct_of_multi_output_transform_feat(es):
    class TestTime(TransformPrimitive):
        name = "test_time"
        input_types = [Datetime]
        return_type = Numeric
        number_output_features = 6

        def get_function(self):
            def test_f(x):
                times = pd.Series(x)
                units = ["year", "month", "day", "hour", "minute", "second"]
                return [times.apply(lambda x: getattr(x, unit)) for unit in units]
            return test_f

    join_time_split = Feature(es["customers"]["signup_date"],
                              primitive=TestTime)
    alt_features = [Feature(es["customers"]["signup_date"], primitive=Year),
                    Feature(es["customers"]["signup_date"], primitive=Month),
                    Feature(es["customers"]["signup_date"], primitive=Day),
                    Feature(es["customers"]["signup_date"], primitive=Hour),
                    Feature(es["customers"]["signup_date"], primitive=Minute),
                    Feature(es["customers"]["signup_date"], primitive=Second)]
    fm, fl = dfs(
        entityset=es,
        target_entity="sessions",
        trans_primitives=[TestTime, Year, Month, Day, Hour, Minute, Second])

    # Get column names of for multi feature and normal features
    subnames = DirectFeature(join_time_split, es["sessions"]).get_feature_names()
    altnames = [DirectFeature(f, es["sessions"]).get_name() for f in alt_features]

    # Check values are equal between
    for col1, col2 in zip(subnames, altnames):
        assert (fm[col1] == fm[col2]).all()


def test_direct_features_of_multi_output_agg_primitives(es):
    class ThreeMostCommonCat(AggregationPrimitive):
        name = "n_most_common_categorical"
        input_types = [Categorical]
        return_type = Categorical
        number_output_features = 3

        def get_function(self):
            def pd_top3(x):
                array = np.array(x.value_counts()[:3].index)
                if len(array) < 3:
                    filler = np.full(3 - len(array), np.nan)
                    array = np.append(array, filler)
                return array
            return pd_top3

    fm, fl = dfs(entityset=es,
                 target_entity="log",
                 agg_primitives=[ThreeMostCommonCat],
                 trans_primitives=[],
                 max_depth=3)

    has_nmost_as_base = []
    for feature in fl:
        is_base = False
        if (len(feature.base_features) > 0 and
                isinstance(feature.base_features[0].primitive, ThreeMostCommonCat)):
            is_base = True
        has_nmost_as_base.append(is_base)
    assert any(has_nmost_as_base)

    true_result_rows = []
    session_data = {
        0: ['coke zero', "car", np.nan],
        1: ['toothpaste', 'brown bag', np.nan],
        2: ['brown bag', np.nan, np.nan],
        3: set(['Haribo sugar-free gummy bears', 'coke zero', np.nan]),
        4: ['coke zero', np.nan, np.nan],
        5: ['taco clock', np.nan, np.nan]
    }
    for i, count in enumerate([5, 4, 1, 2, 3, 2]):
        while count > 0:
            true_result_rows.append(session_data[i])
            count -= 1

    tempname = "sessions.N_MOST_COMMON_CATEGORICAL(log.product_id)__%s"
    for i, row in enumerate(true_result_rows):
        for j in range(3):
            value = fm[tempname % (j)][i]
            if isinstance(row, set):
                assert pd.isnull(value) or value in row
            else:
                assert ((pd.isnull(value) and pd.isnull(row[j])) or
                        value == row[j])


def test_direct_with_invalid_init_args(diamond_es):
    customer_to_region = diamond_es.get_forward_relationships('customers')[0]
    error_text = 'child_entity must match the first relationship'
    with pytest.raises(AssertionError, match=error_text):
        ft.DirectFeature(diamond_es['regions']['name'], diamond_es['stores'],
                         relationship_path=[customer_to_region])

    transaction_relationships = diamond_es.get_forward_relationships('transactions')
    transaction_to_store = next(r for r in transaction_relationships
                                if r.parent_entity.id == 'stores')
    error_text = 'Base feature must be defined on the entity at the end of relationship_path'
    with pytest.raises(AssertionError, match=error_text):
        ft.DirectFeature(diamond_es['regions']['name'], diamond_es['transactions'],
                         relationship_path=[transaction_to_store])


def test_direct_with_multiple_possible_paths(diamond_es):
    error_text = "There are multiple possible paths to the base entity. " \
                 "You must specify a relationship path."
    with pytest.raises(RuntimeError, match=error_text):
        ft.DirectFeature(diamond_es['regions']['name'], diamond_es['transactions'])

    transaction_relationships = diamond_es.get_forward_relationships('transactions')
    transaction_to_customer = next(r for r in transaction_relationships
                                   if r.parent_entity.id == 'customers')
    customer_to_region = diamond_es.get_forward_relationships('customers')[0]
    # Does not raise if path specified.
    feat = ft.DirectFeature(diamond_es['regions']['name'], diamond_es['transactions'],
                            relationship_path=[transaction_to_customer, customer_to_region])
    assert feat.get_name() == 'customers.regions.name'


def test_direct_with_single_possible_path(diamond_es):
    # This uses diamond_es to test that there being a cycle somewhere in the
    # graph doesn't cause an error.
    feat = ft.DirectFeature(diamond_es['customers']['name'], diamond_es['transactions'])
    relationships = diamond_es.get_forward_relationships('transactions')
    relationship = next(r for r in relationships if r.parent_entity.id == 'customers')
    assert feat.relationship_path == [relationship]


def test_get_name_skips_relationships_when_single_possible_path(es):
    feat = ft.DirectFeature(es['customers']['age'], es['log'])
    assert feat.get_name() == 'customers.age'


def test_direct_with_no_path(diamond_es):
    error_text = 'No path from "regions" to "customers" found.'
    with pytest.raises(RuntimeError, match=error_text):
        ft.DirectFeature(diamond_es['customers']['name'], diamond_es['regions'])

    error_text = 'No path from "customers" to "customers" found.'
    with pytest.raises(RuntimeError, match=error_text):
        ft.DirectFeature(diamond_es['customers']['name'], diamond_es['customers'])


def test_serialization(es):
    value = ft.IdentityFeature(es['products']['rating'])
    direct = ft.DirectFeature(value, es['log'])

    log_to_products = next(r for r in es.get_forward_relationships('log')
                           if r.parent_entity.id == 'products')
    dictionary = {
        'name': None,
        'base_feature': value.unique_name(),
        'relationship_path': [log_to_products.to_dictionary()],
    }

    assert dictionary == direct.get_arguments()
    assert direct == \
        ft.DirectFeature.from_dictionary(dictionary, es,
                                         {value.unique_name(): value},
                                         PrimitivesDeserializer())
