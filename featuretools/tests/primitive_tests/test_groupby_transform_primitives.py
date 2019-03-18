import numpy as np
import pandas as pd
import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools.computational_backends import PandasBackend
from featuretools.primitives import (
    CumCount,
    CumMax,
    CumMean,
    CumMin,
    CumSum,
    Last,
    TransformPrimitive
)
from featuretools.variable_types import DatetimeTimeIndex, Numeric


@pytest.fixture
def es():
    return make_ecommerce_entityset()


class TestCumCount:

    primitive = CumCount

    def test_order(self):
        g = pd.Series(["a", "b", "a"])

        answers = ([1, 2], [1])

        function = self.primitive().get_function()
        for (_, group), answer in zip(g.groupby(g), answers):
            np.testing.assert_array_equal(function(group), answer)

    def test_regular(self):
        g = pd.Series(["a", "b", "a", "c", "d", "b"])
        answers = ([1, 2], [1, 2], [1], [1])

        function = self.primitive().get_function()
        for (_, group), answer in zip(g.groupby(g), answers):
            np.testing.assert_array_equal(function(group), answer)

    def test_discrete(self):
        g = pd.Series(["a", "b", "a", "c", "d", "b"])
        answers = ([1, 2], [1, 2], [1], [1])

        function = self.primitive().get_function()
        for (_, group), answer in zip(g.groupby(g), answers):
            np.testing.assert_array_equal(function(group), answer)


class TestCumSum:

    primitive = CumSum

    def test_order(self):
        v = pd.Series([1, 2, 2])
        g = pd.Series(["a", "b", "a"])

        answers = ([1, 3], [2])

        function = self.primitive().get_function()
        for (_, group), answer in zip(v.groupby(g), answers):
            np.testing.assert_array_equal(function(group), answer)

    def test_regular(self):
        v = pd.Series([101, 102, 103, 104, 105, 106])
        g = pd.Series(["a", "b", "a", "c", "d", "b"])
        answers = ([101, 204], [102, 208], [104], [105])

        function = self.primitive().get_function()
        for (_, group), answer in zip(v.groupby(g), answers):
            np.testing.assert_array_equal(function(group), answer)


class TestCumMean:
    primitive = CumMean

    def test_order(self):
        v = pd.Series([1, 2, 2])
        g = pd.Series(["a", "b", "a"])

        answers = ([1, 1.5], [2])

        function = self.primitive().get_function()
        for (_, group), answer in zip(v.groupby(g), answers):
            np.testing.assert_array_equal(function(group), answer)

    def test_regular(self):
        v = pd.Series([101, 102, 103, 104, 105, 106])
        g = pd.Series(["a", "b", "a", "c", "d", "b"])
        answers = ([101, 102], [102, 104], [104], [105])

        function = self.primitive().get_function()
        for (_, group), answer in zip(v.groupby(g), answers):
            np.testing.assert_array_equal(function(group), answer)


class TestCumMax:

    primitive = CumMax

    def test_order(self):
        v = pd.Series([1, 2, 2])
        g = pd.Series(["a", "b", "a"])

        answers = ([1, 2], [2])

        function = self.primitive().get_function()
        for (_, group), answer in zip(v.groupby(g), answers):
            np.testing.assert_array_equal(function(group), answer)

    def test_regular(self):
        v = pd.Series([101, 102, 103, 104, 105, 106])
        g = pd.Series(["a", "b", "a", "c", "d", "b"])
        answers = ([101, 103], [102, 106], [104], [105])

        function = self.primitive().get_function()
        for (_, group), answer in zip(v.groupby(g), answers):
            np.testing.assert_array_equal(function(group), answer)


class TestCumMin:

    primitive = CumMin

    def test_order(self):
        v = pd.Series([1, 2, 2])
        g = pd.Series(["a", "b", "a"])

        answers = ([1, 1], [2])

        function = self.primitive().get_function()
        for (_, group), answer in zip(v.groupby(g), answers):
            np.testing.assert_array_equal(function(group), answer)

    def test_regular(self):
        v = pd.Series([101, 102, 103, 104, 105, 106, 100])
        g = pd.Series(["a", "b", "a", "c", "d", "b", "a"])
        answers = ([101, 101, 100], [102, 102], [104], [105])

        function = self.primitive().get_function()
        for (_, group), answer in zip(v.groupby(g), answers):
            np.testing.assert_array_equal(function(group), answer)


def test_cum_sum(es):
    log_value_feat = es['log']['value']
    dfeat = ft.Feature(es['sessions']['device_type'], entity=es['log'])
    cum_sum = ft.Feature(log_value_feat, groupby=dfeat, primitive=CumSum)
    features = [cum_sum]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(15))
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 15
    cum_sum_values = [0, 5, 15, 30, 50, 0, 1, 3, 6, 6, 50, 55, 55, 62, 76]
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
    es['log'].df['value'][16] = 10
    cum_sum = ft.Feature(log_value_feat, groupby=es['log']['product_id'], primitive=CumSum)
    features = [cum_sum]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(17))
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 17
    cum_sum_values = [0, 5, 15,
                      15, 35,
                      0, 1, 3,
                      3, 3,
                      0,
                      np.nan, np.nan, np.nan, np.nan, np.nan, 10]

    assert len(cvalues) == len(cum_sum_values)
    for i, v in enumerate(cum_sum_values):
        if np.isnan(v):
            assert (np.isnan(cvalues[i]))
        else:
            assert v == cvalues[i]


def test_cum_sum_numpy_group_on_nan(es):
    class CumSumNumpy(TransformPrimitive):
        """Returns the cumulative sum after grouping"""

        name = "cum_sum"
        input_types = [Numeric]
        return_type = Numeric
        uses_full_entity = True

        def get_function(self):
            def cum_sum(values):
                return values.cumsum().values
            return cum_sum

    log_value_feat = es['log']['value']
    es['log'].df['product_id'] = (['coke zero'] * 3 + ['car'] * 2 +
                                  ['toothpaste'] * 3 + ['brown bag'] * 2 +
                                  ['shoes'] +
                                  [np.nan] * 4 +
                                  ['coke_zero'] * 2)
    es['log'].df['value'][16] = 10
    cum_sum = ft.Feature(log_value_feat, groupby=es['log']['product_id'], primitive=CumSumNumpy)
    assert cum_sum.get_name() == "CUM_SUM(value) by product_id"
    features = [cum_sum]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(17))
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 17
    cum_sum_values = [0, 5, 15,
                      15, 35,
                      0, 1, 3,
                      3, 3,
                      0,
                      np.nan, np.nan, np.nan, np.nan, np.nan, 10]

    assert len(cvalues) == len(cum_sum_values)
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
    cum_count = ft.Feature(es['log']['session_id'],
                           groupby=es['log']['session_id'],
                           primitive=CumCount)
    features = [cum_count]
    df = ft.calculate_feature_matrix(entityset=es,
                                     features=features,
                                     instance_ids=range(15))
    cvalues = df[cum_count.get_name()].values
    assert len(cvalues) == 15
    cum_count_values = [1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 1, 2, 1, 2, 3]
    for i, v in enumerate(cum_count_values):
        assert v == cvalues[i]


def test_rename(es):
    cum_count = ft.Feature(es['log']['session_id'],
                           groupby=es['log']['session_id'],
                           primitive=CumCount)
    copy_feat = cum_count.rename("rename_test")
    assert cum_count.hash() != copy_feat.hash()
    assert cum_count.get_name() != copy_feat.get_name()
    assert all([x.generate_name() == y.generate_name() for x, y
                in zip(cum_count.base_features, copy_feat.base_features)])
    assert cum_count.entity == copy_feat.entity


def test_groupby_no_data(es):
    cum_count = ft.Feature(es['log']['session_id'],
                           groupby=es['log']['session_id'],
                           primitive=CumCount)
    last_feat = ft.Feature(cum_count, parent_entity=es['customers'], primitive=Last)
    df = ft.calculate_feature_matrix(entityset=es,
                                     features=[last_feat],
                                     cutoff_time=pd.Timestamp("2011-04-08"))
    cvalues = df[last_feat.get_name()].values
    assert len(cvalues) == 3
    assert all([pd.isnull(value) for value in cvalues])


def test_groupby_uses_calc_time(es):
    def projected_amount_left(amount, timestamp, time=None):
        # cumulative sum of amout, with timedelta *  constant subtracted
        delta = time - timestamp
        delta_seconds = delta / np.timedelta64(1, 's')
        return amount.cumsum() - (delta_seconds)

    class ProjectedAmountRemaining(TransformPrimitive):
        name = "projected_amount_remaining"
        uses_calc_time = True
        input_types = [Numeric, DatetimeTimeIndex]
        return_type = Numeric
        uses_full_entity = True

        def get_function(self):
            return projected_amount_left

    time_since_product = ft.Feature([es['log']['value'], es['log']['datetime']],
                                    groupby=es['log']['product_id'],
                                    primitive=ProjectedAmountRemaining)
    df = ft.calculate_feature_matrix(entityset=es,
                                     features=[time_since_product],
                                     cutoff_time=pd.Timestamp("2011-04-10 11:10:30"))
    answers = [-88830, -88819, -88803, -88797, -88771, -88770, -88760, -88749,
               -88740, -88227, -1830, -1809, -1750, -1740, -1723, np.nan, np.nan]

    for x, y in zip(df[time_since_product.get_name()], answers):
        assert ((pd.isnull(x) and pd.isnull(y)) or x == y)
