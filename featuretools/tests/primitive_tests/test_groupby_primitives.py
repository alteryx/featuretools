import numpy as np
import pandas as pd
import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools.computational_backends import PandasBackend
from featuretools.primitives import CumCount, CumMax, CumMean, CumMin, CumSum


@pytest.fixture
def es():
    return make_ecommerce_entityset()


class TestCumCount:

    primitive = CumCount

    def test_order(self):
        g = pd.Series(["a", "b", "a"])

        answer = [1, 1, 2]

        function = self.primitive().get_function()
        np.testing.assert_array_equal(function(g.groupby(g)), answer)

    def test_regular(self):
        g = pd.Series(["a", "b", "a", "c", "d", "b"])
        answer = [1, 1, 2, 1, 1, 2]

        function = self.primitive().get_function()
        np.testing.assert_array_equal(function(g.groupby(g)), answer)

    def test_discrete(self):
        g = pd.Series(["a", "b", "a", "c", "d", "b"])
        answer = [1, 1, 2, 1, 1, 2]

        function = self.primitive().get_function()
        np.testing.assert_array_equal(function(g.groupby(g)), answer)


class TestCumSum:

    primitive = CumSum

    def test_order(self):
        v = pd.Series([1, 2, 2])
        g = pd.Series(["a", "b", "a"])

        answer = [1, 2, 3]

        function = self.primitive().get_function()
        np.testing.assert_array_equal(function(v.groupby(g)), answer)

    def test_regular(self):
        v = pd.Series([101, 102, 103, 104, 105, 106])
        g = pd.Series(["a", "b", "a", "c", "d", "b"])
        answer = [101, 102, 204, 104, 105, 208]

        function = self.primitive().get_function()
        np.testing.assert_array_equal(function(v.groupby(g)), answer)


class TestCumMean:
    primitive = CumMean

    def test_order(self):
        v = pd.Series([1, 2, 2])
        g = pd.Series(["a", "b", "a"])

        answer = [1, 2, 1.5]

        function = self.primitive().get_function()
        np.testing.assert_array_equal(function(v.groupby(g)), answer)

    def test_regular(self):
        v = pd.Series([101, 102, 103, 104, 105, 106])
        g = pd.Series(["a", "b", "a", "c", "d", "b"])
        answer = [101, 102, 102, 104, 105, 104]

        function = self.primitive().get_function()
        np.testing.assert_array_equal(function(v.groupby(g)), answer)


class TestCumMax:

    primitive = CumMax

    def test_order(self):
        v = pd.Series([1, 2, 2])
        g = pd.Series(["a", "b", "a"])

        answer = [1, 2, 2]

        function = self.primitive().get_function()
        np.testing.assert_array_equal(function(v.groupby(g)), answer)

    def test_regular(self):
        v = pd.Series([101, 102, 103, 104, 105, 106])
        g = pd.Series(["a", "b", "a", "c", "d", "b"])
        answer = [101, 102, 103, 104, 105, 106]

        function = self.primitive().get_function()
        np.testing.assert_array_equal(function(v.groupby(g)), answer)


class TestCumMin:

    primitive = CumMin

    def test_order(self):
        v = pd.Series([1, 2, 2])
        g = pd.Series(["a", "b", "a"])

        answer = [1, 2, 1]

        function = self.primitive().get_function()
        np.testing.assert_array_equal(function(v.groupby(g)), answer)

    def test_regular(self):
        v = pd.Series([101, 102, 103, 104, 105, 106, 100])
        g = pd.Series(["a", "b", "a", "c", "d", "b", "a"])
        answer = [101, 102, 101, 104, 105, 102, 100]

        function = self.primitive().get_function()
        np.testing.assert_array_equal(function(v.groupby(g)), answer)


def test_cum_sum(es):
    log_value_feat = es['log']['value']

    cum_sum = ft.Feature(log_value_feat, groupby=es['log']['session_id'], primitive=CumSum)
    features = [cum_sum]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(15))
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 15
    cum_sum_values = [0, 5, 15, 30, 50, 0, 1, 3, 6, 0, 0, 5, 0, 7, 21]
    for i, v in enumerate(cum_sum_values):
        assert v == cvalues[i]


def test_cum_min(es):
    log_value_feat = es['log']['value']
    cum_min = ft.Feature(log_value_feat, groupby=es['log']['session_id'], primitive=CumMin)
    features = [cum_min]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(15))
    cvalues = df[cum_min.get_name()].values
    assert len(cvalues) == 15
    cum_min_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, v in enumerate(cum_min_values):
        assert v == cvalues[i]


def test_cum_max(es):
    log_value_feat = es['log']['value']
    cum_max = ft.Feature(log_value_feat, groupby=es['log']['session_id'], primitive=CumMax)
    features = [cum_max]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(15))
    cvalues = df[cum_max.get_name()].values
    assert len(cvalues) == 15
    cum_max_values = [0, 5, 10, 15, 20, 0, 1, 2, 3, 0, 0, 5, 0, 7, 14]
    for i, v in enumerate(cum_max_values):
        assert v == cvalues[i]


def test_cum_sum_group_on_nan(es):
    log_value_feat = es['log']['value']
    es['log'].df['product_id'] = (['coke zero'] * 3 + ['car'] * 2 +
                                  ['toothpaste'] * 3 + ['brown bag'] * 2 +
                                  ['shoes'] +
                                  [np.nan] * 4 +
                                  ['coke_zero'] * 2)
    cum_sum = ft.Feature(log_value_feat, groupby=es['log']['product_id'], primitive=CumSum)
    features = [cum_sum]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(15))
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 15
    cum_sum_values = [0, 5, 15,
                      15, 35,
                      0, 1, 3,
                      3, 3,
                      0,
                      np.nan, np.nan, np.nan, np.nan]
    for i, v in enumerate(cum_sum_values):
        if np.isnan(v):
            assert (np.isnan(cvalues[i]))
        else:
            assert v == cvalues[i]


def test_cum_handles_uses_full_entity(es):
    def check(feature):
        pandas_backend = PandasBackend(es, [feature])
        df_1 = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2], time_last=None)
        df_2 = pandas_backend.calculate_all_features(instance_ids=[2, 4], time_last=None)

        # check that the value for instance id 2 matches
        assert (df_2.loc[2] == df_1.loc[2]).all()

    for primitive in [CumSum, CumMean, CumMax, CumMin]:
        check(ft.Feature(es['log']['value'], groupby=es['log']['session_id'], primitive=primitive))

    check(ft.Feature(es['log']['session_id'], groupby=es['log']['session_id'], primitive=CumCount))


def test_cum_mean(es):
    log_value_feat = es['log']['value']
    cum_mean = ft.Feature(log_value_feat, groupby=es['log']['session_id'], primitive=CumMean)
    features = [cum_mean]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(15))
    cvalues = df[cum_mean.get_name()].values
    assert len(cvalues) == 15
    cum_mean_values = [0, 2.5, 5, 7.5, 10, 0, .5, 1, 1.5, 0, 0, 2.5, 0, 3.5, 7]
    for i, v in enumerate(cum_mean_values):
        assert v == cvalues[i]


def test_cum_count(es):
    cum_count = ft.Feature(es['log']['session_id'], groupby=es['log']['session_id'], primitive=CumCount)
    features = [cum_count]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(15))
    cvalues = df[cum_count.get_name()].values
    assert len(cvalues) == 15
    cum_count_values = [1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 1, 2, 1, 2, 3]
    for i, v in enumerate(cum_count_values):
        assert v == cvalues[i]
