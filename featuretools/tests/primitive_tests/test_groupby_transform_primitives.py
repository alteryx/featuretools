import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime

from featuretools import (
    Feature,
    GroupByTransformFeature,
    IdentityFeature,
    calculate_feature_matrix,
    feature_base,
)
from featuretools.computational_backends.feature_set import FeatureSet
from featuretools.computational_backends.feature_set_calculator import (
    FeatureSetCalculator,
)
from featuretools.primitives import CumCount, CumMax, CumMean, CumMin, CumSum, Last
from featuretools.primitives.base import TransformPrimitive
from featuretools.synthesis import dfs
from featuretools.tests.testing_utils import feature_with_name


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


def test_cum_sum(pd_es):
    log_value_feat = IdentityFeature(pd_es["log"].ww["value"])
    dfeat = Feature(
        IdentityFeature(pd_es["sessions"].ww["device_type"]),
        dataframe_name="log",
    )
    cum_sum = Feature(log_value_feat, groupby=dfeat, primitive=CumSum)
    features = [cum_sum]
    df = calculate_feature_matrix(
        entityset=pd_es,
        features=features,
        instance_ids=range(15),
    )
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 15
    cum_sum_values = [0, 5, 15, 30, 50, 0, 1, 3, 6, 6, 50, 55, 55, 62, 76]
    for i, v in enumerate(cum_sum_values):
        assert v == cvalues[i]


def test_cum_min(pd_es):
    log_value_feat = IdentityFeature(pd_es["log"].ww["value"])
    cum_min = Feature(
        log_value_feat,
        groupby=IdentityFeature(pd_es["log"].ww["session_id"]),
        primitive=CumMin,
    )
    features = [cum_min]
    df = calculate_feature_matrix(
        entityset=pd_es,
        features=features,
        instance_ids=range(15),
    )
    cvalues = df[cum_min.get_name()].values
    assert len(cvalues) == 15
    cum_min_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, v in enumerate(cum_min_values):
        assert v == cvalues[i]


def test_cum_max(pd_es):
    log_value_feat = IdentityFeature(pd_es["log"].ww["value"])
    cum_max = Feature(
        log_value_feat,
        groupby=IdentityFeature(pd_es["log"].ww["session_id"]),
        primitive=CumMax,
    )
    features = [cum_max]
    df = calculate_feature_matrix(
        entityset=pd_es,
        features=features,
        instance_ids=range(15),
    )
    cvalues = df[cum_max.get_name()].values
    assert len(cvalues) == 15
    cum_max_values = [0, 5, 10, 15, 20, 0, 1, 2, 3, 0, 0, 5, 0, 7, 14]
    for i, v in enumerate(cum_max_values):
        assert v == cvalues[i]


def test_cum_sum_group_on_nan(pd_es):
    log_value_feat = IdentityFeature(pd_es["log"].ww["value"])
    pd_es["log"]["product_id"] = (
        ["coke zero"] * 3
        + ["car"] * 2
        + ["toothpaste"] * 3
        + ["brown bag"] * 2
        + ["shoes"]
        + [np.nan] * 4
        + ["coke_zero"] * 2
    )
    pd_es["log"]["value"][16] = 10
    cum_sum = Feature(
        log_value_feat,
        groupby=IdentityFeature(pd_es["log"].ww["product_id"]),
        primitive=CumSum,
    )
    features = [cum_sum]
    df = calculate_feature_matrix(
        entityset=pd_es,
        features=features,
        instance_ids=range(17),
    )
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 17
    cum_sum_values = [
        0,
        5,
        15,
        15,
        35,
        0,
        1,
        3,
        3,
        3,
        0,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        10,
    ]

    assert len(cvalues) == len(cum_sum_values)
    for i, v in enumerate(cum_sum_values):
        if np.isnan(v):
            assert np.isnan(cvalues[i])
        else:
            assert v == cvalues[i]


def test_cum_sum_numpy_group_on_nan(pd_es):
    class CumSumNumpy(TransformPrimitive):
        """Returns the cumulative sum after grouping"""

        name = "cum_sum"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        uses_full_dataframe = True

        def get_function(self):
            def cum_sum(values):
                return values.cumsum().values

            return cum_sum

    log_value_feat = IdentityFeature(pd_es["log"].ww["value"])
    pd_es["log"]["product_id"] = (
        ["coke zero"] * 3
        + ["car"] * 2
        + ["toothpaste"] * 3
        + ["brown bag"] * 2
        + ["shoes"]
        + [np.nan] * 4
        + ["coke_zero"] * 2
    )
    pd_es["log"]["value"][16] = 10
    cum_sum = Feature(
        log_value_feat,
        groupby=IdentityFeature(pd_es["log"].ww["product_id"]),
        primitive=CumSumNumpy,
    )
    assert cum_sum.get_name() == "CUM_SUM(value) by product_id"
    features = [cum_sum]
    df = calculate_feature_matrix(
        entityset=pd_es,
        features=features,
        instance_ids=range(17),
    )
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 17
    cum_sum_values = [
        0,
        5,
        15,
        15,
        35,
        0,
        1,
        3,
        3,
        3,
        0,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        10,
    ]

    assert len(cvalues) == len(cum_sum_values)
    for i, v in enumerate(cum_sum_values):
        if np.isnan(v):
            assert np.isnan(cvalues[i])
        else:
            assert v == cvalues[i]


def test_cum_handles_uses_full_dataframe(pd_es):
    def check(feature):
        feature_set = FeatureSet([feature])
        calculator = FeatureSetCalculator(
            pd_es,
            feature_set=feature_set,
            time_last=None,
        )
        df_1 = calculator.run(np.array([0, 1, 2]))
        df_2 = calculator.run(np.array([2, 4]))

        # check that the value for instance id 2 matches
        assert (df_2.loc[2] == df_1.loc[2]).all()

    for primitive in [CumSum, CumMean, CumMax, CumMin]:
        check(
            Feature(
                pd_es["log"].ww["value"],
                groupby=IdentityFeature(pd_es["log"].ww["session_id"]),
                primitive=primitive,
            ),
        )

    check(
        Feature(
            pd_es["log"].ww["product_id"],
            groupby=Feature(pd_es["log"].ww["product_id"]),
            primitive=CumCount,
        ),
    )


def test_cum_mean(pd_es):
    log_value_feat = IdentityFeature(pd_es["log"].ww["value"])
    cum_mean = Feature(
        log_value_feat,
        groupby=IdentityFeature(pd_es["log"].ww["session_id"]),
        primitive=CumMean,
    )
    features = [cum_mean]
    df = calculate_feature_matrix(
        entityset=pd_es,
        features=features,
        instance_ids=range(15),
    )
    cvalues = df[cum_mean.get_name()].values
    assert len(cvalues) == 15
    cum_mean_values = [0, 2.5, 5, 7.5, 10, 0, 0.5, 1, 1.5, 0, 0, 2.5, 0, 3.5, 7]
    for i, v in enumerate(cum_mean_values):
        assert v == cvalues[i]


def test_cum_count(pd_es):
    cum_count = Feature(
        IdentityFeature(pd_es["log"].ww["product_id"]),
        groupby=IdentityFeature(pd_es["log"].ww["product_id"]),
        primitive=CumCount,
    )
    features = [cum_count]
    df = calculate_feature_matrix(
        entityset=pd_es,
        features=features,
        instance_ids=range(15),
    )
    cvalues = df[cum_count.get_name()].values
    assert len(cvalues) == 15
    cum_count_values = [1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 1, 4, 5, 6, 7]
    for i, v in enumerate(cum_count_values):
        assert v == cvalues[i]


def test_rename(pd_es):
    cum_count = Feature(
        IdentityFeature(pd_es["log"].ww["product_id"]),
        groupby=IdentityFeature(pd_es["log"].ww["product_id"]),
        primitive=CumCount,
    )
    copy_feat = cum_count.rename("rename_test")
    assert cum_count.unique_name() != copy_feat.unique_name()
    assert cum_count.get_name() != copy_feat.get_name()
    assert all(
        [
            x.generate_name() == y.generate_name()
            for x, y in zip(cum_count.base_features, copy_feat.base_features)
        ],
    )
    assert cum_count.dataframe_name == copy_feat.dataframe_name


def test_groupby_no_data(pd_es):
    cum_count = Feature(
        IdentityFeature(pd_es["log"].ww["product_id"]),
        groupby=IdentityFeature(pd_es["log"].ww["product_id"]),
        primitive=CumCount,
    )
    last_feat = Feature(cum_count, parent_dataframe_name="customers", primitive=Last)
    df = calculate_feature_matrix(
        entityset=pd_es,
        features=[last_feat],
        cutoff_time=pd.Timestamp("2011-04-08"),
    )
    cvalues = df[last_feat.get_name()].values
    assert len(cvalues) == 2
    assert all([pd.isnull(value) for value in cvalues])


def test_groupby_uses_calc_time(pd_es):
    def projected_amount_left(amount, timestamp, time=None):
        # cumulative sum of amount, with timedelta *  constant subtracted
        delta = time - timestamp
        delta_seconds = delta / np.timedelta64(1, "s")
        return amount.cumsum() - (delta_seconds)

    class ProjectedAmountRemaining(TransformPrimitive):
        name = "projected_amount_remaining"
        uses_calc_time = True
        input_types = [
            ColumnSchema(semantic_tags={"numeric"}),
            ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
        ]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        uses_full_dataframe = True

        def get_function(self):
            return projected_amount_left

    time_since_product = GroupByTransformFeature(
        [
            IdentityFeature(pd_es["log"].ww["value"]),
            IdentityFeature(pd_es["log"].ww["datetime"]),
        ],
        groupby=IdentityFeature(pd_es["log"].ww["product_id"]),
        primitive=ProjectedAmountRemaining,
    )
    df = calculate_feature_matrix(
        entityset=pd_es,
        features=[time_since_product],
        cutoff_time=pd.Timestamp("2011-04-10 11:10:30"),
    )
    answers = [
        -88830,
        -88819,
        -88803,
        -88797,
        -88771,
        -88770,
        -88760,
        -88749,
        -88740,
        -88227,
        -1830,
        -1809,
        -1750,
        -1740,
        -1723,
        np.nan,
        np.nan,
    ]

    for x, y in zip(df[time_since_product.get_name()], answers):
        assert (pd.isnull(x) and pd.isnull(y)) or x == y


def test_groupby_multi_output_stacking(pd_es):
    class TestTime(TransformPrimitive):
        name = "test_time"
        input_types = [ColumnSchema(logical_type=Datetime)]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        number_output_features = 6

    fl = dfs(
        entityset=pd_es,
        target_dataframe_name="sessions",
        agg_primitives=["sum"],
        groupby_trans_primitives=[TestTime],
        features_only=True,
        max_depth=4,
    )

    for i in range(6):
        f = "SUM(log.TEST_TIME(datetime)[%d] by product_id)" % i
        assert feature_with_name(fl, f)
        assert ("customers.SUM(log.TEST_TIME(datetime)[%d] by session_id)" % i) in fl


def test_serialization(pd_es):
    value = IdentityFeature(pd_es["log"].ww["value"])
    zipcode = IdentityFeature(pd_es["log"].ww["zipcode"])
    primitive = CumSum()
    groupby = feature_base.GroupByTransformFeature(value, primitive, zipcode)

    dictionary = {
        "name": "CUM_SUM(value) by zipcode",
        "base_features": [value.unique_name()],
        "primitive": primitive,
        "groupby": zipcode.unique_name(),
    }

    assert dictionary == groupby.get_arguments()
    dependencies = {
        value.unique_name(): value,
        zipcode.unique_name(): zipcode,
    }
    assert groupby == feature_base.GroupByTransformFeature.from_dictionary(
        dictionary,
        pd_es,
        dependencies,
        primitive,
    )


def test_groupby_with_multioutput_primitive(pd_es):
    class MultiCumSum(TransformPrimitive):
        name = "multi_cum_sum"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        number_output_features = 3

        def get_function(self):
            def multi_cum_sum(x):
                return x.cumsum(), x.cummax(), x.cummin()

            return multi_cum_sum

    fm, _ = dfs(
        entityset=pd_es,
        target_dataframe_name="customers",
        trans_primitives=[],
        agg_primitives=[],
        groupby_trans_primitives=[MultiCumSum, CumSum, CumMax, CumMin],
    )

    # Calculate output in a separate DFS call to make sure the multi-output code
    # does not alter any values
    fm2, _ = dfs(
        entityset=pd_es,
        target_dataframe_name="customers",
        trans_primitives=[],
        agg_primitives=[],
        groupby_trans_primitives=[CumSum, CumMax, CumMin],
    )

    answer_cols = [
        ["CUM_SUM(age) by cohort", "CUM_SUM(age) by région_id"],
        ["CUM_MAX(age) by cohort", "CUM_MAX(age) by région_id"],
        ["CUM_MIN(age) by cohort", "CUM_MIN(age) by région_id"],
    ]

    for i in range(3):
        # Check that multi-output gives correct answers
        f = "MULTI_CUM_SUM(age)[%d] by cohort" % i
        assert f in fm.columns
        for x, y in zip(fm[f].values, fm[answer_cols[i][0]].values):
            assert x == y
        f = "MULTI_CUM_SUM(age)[%d] by région_id" % i
        assert f in fm.columns
        for x, y in zip(fm[f].values, fm[answer_cols[i][1]].values):
            assert x == y
        # Verify single output results are unchanged by inclusion of
        # multi-output primitive
        for x, y in zip(fm[answer_cols[i][0]], fm2[answer_cols[i][0]]):
            assert x == y
        for x, y in zip(fm[answer_cols[i][1]], fm2[answer_cols[i][1]]):
            assert x == y


def test_groupby_with_multioutput_primitive_custom_names(pd_es):
    class MultiCumSum(TransformPrimitive):
        name = "multi_cum_sum"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        number_output_features = 3

        def get_function(self):
            def multi_cum_sum(x):
                return x.cumsum(), x.cummax(), x.cummin()

            return multi_cum_sum

        def generate_names(primitive, base_feature_names):
            return ["CUSTOM_SUM", "CUSTOM_MAX", "CUSTOM_MIN"]

    fm, _ = dfs(
        entityset=pd_es,
        target_dataframe_name="customers",
        trans_primitives=[],
        agg_primitives=[],
        groupby_trans_primitives=[MultiCumSum, CumSum, CumMax, CumMin],
    )

    answer_cols = [
        ["CUM_SUM(age) by cohort", "CUM_SUM(age) by région_id"],
        ["CUM_MAX(age) by cohort", "CUM_MAX(age) by région_id"],
        ["CUM_MIN(age) by cohort", "CUM_MIN(age) by région_id"],
    ]

    expected_names = [
        ["CUSTOM_SUM by cohort", "CUSTOM_SUM by région_id"],
        ["CUSTOM_MAX by cohort", "CUSTOM_MAX by région_id"],
        ["CUSTOM_MIN by cohort", "CUSTOM_MIN by région_id"],
    ]

    for i in range(3):
        f = expected_names[i][0]
        assert f in fm.columns
        for x, y in zip(fm[f].values, fm[answer_cols[i][0]].values):
            assert x == y
        f = expected_names[i][1]
        assert f in fm.columns
        for x, y in zip(fm[f].values, fm[answer_cols[i][1]].values):
            assert x == y
