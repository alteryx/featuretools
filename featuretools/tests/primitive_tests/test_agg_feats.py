from datetime import datetime
from inspect import isclass
from math import isnan

import numpy as np
import pandas as pd
import pytest
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime

from featuretools import (
    AggregationFeature,
    Feature,
    IdentityFeature,
    Timedelta,
    calculate_feature_matrix,
    dfs,
    primitives,
)
from featuretools.entityset.relationship import RelationshipPath
from featuretools.feature_base.cache import feature_cache
from featuretools.primitives import (
    Count,
    Max,
    Mean,
    Median,
    NMostCommon,
    NumTrue,
    NumUnique,
    Sum,
    TimeSinceFirst,
    TimeSinceLast,
    get_aggregation_primitives,
)
from featuretools.primitives.base import AggregationPrimitive
from featuretools.synthesis.deep_feature_synthesis import DeepFeatureSynthesis, match
from featuretools.tests.testing_utils import backward_path, feature_with_name, to_pandas
from featuretools.utils.gen_utils import Library


@pytest.fixture(autouse=True)
def reset_dfs_cache():
    feature_cache.enabled = False
    feature_cache.clear_all()


def test_get_depth(es):
    log_id_feat = IdentityFeature(es["log"].ww["id"])
    customer_id_feat = IdentityFeature(es["customers"].ww["id"])
    count_logs = Feature(log_id_feat, parent_dataframe_name="sessions", primitive=Count)
    sum_count_logs = Feature(
        count_logs,
        parent_dataframe_name="customers",
        primitive=Sum,
    )
    num_logs_greater_than_5 = sum_count_logs > 5
    count_customers = Feature(
        customer_id_feat,
        parent_dataframe_name="régions",
        where=num_logs_greater_than_5,
        primitive=Count,
    )
    num_customers_region = Feature(count_customers, dataframe_name="customers")

    depth = num_customers_region.get_depth()
    assert depth == 5


def test_makes_count(es):
    dfs = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[Count],
        trans_primitives=[],
    )

    features = dfs.build_features()
    assert feature_with_name(features, "device_type")
    assert feature_with_name(features, "customer_id")
    assert feature_with_name(features, "customers.région_id")
    assert feature_with_name(features, "customers.age")
    assert feature_with_name(features, "COUNT(log)")
    assert feature_with_name(features, "customers.COUNT(sessions)")
    assert feature_with_name(features, "customers.régions.language")
    assert feature_with_name(features, "customers.COUNT(log)")


def test_count_null(pd_es):
    class Count(AggregationPrimitive):
        name = "count"
        input_types = [[ColumnSchema(semantic_tags={"foreign_key"})], [ColumnSchema()]]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        stack_on_self = False

        def __init__(self, count_null=True):
            self.count_null = count_null

        def get_function(self):
            def count_func(values):
                if self.count_null:
                    values = values.fillna(0)

                return values.count()

            return count_func

        def generate_name(
            self,
            base_feature_names,
            relationship_path_name,
            parent_dataframe_name,
            where_str,
            use_prev_str,
        ):
            return "COUNT(%s%s%s)" % (relationship_path_name, where_str, use_prev_str)

    count_null = Feature(
        pd_es["log"].ww["value"],
        parent_dataframe_name="sessions",
        primitive=Count(count_null=True),
    )
    feature_matrix = calculate_feature_matrix([count_null], entityset=pd_es)
    values = [5, 4, 1, 2, 3, 2]
    assert (values == feature_matrix[count_null.get_name()]).all()


def test_check_input_types(es):
    count = Feature(
        es["sessions"].ww["id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    mean = Feature(count, parent_dataframe_name="régions", primitive=Mean)
    assert mean._check_input_types()

    boolean = count > 3
    mean = Feature(
        count,
        parent_dataframe_name="régions",
        where=boolean,
        primitive=Mean,
    )
    assert mean._check_input_types()


def test_mean_nan(es):
    array = pd.Series([5, 5, 5, 5, 5])
    mean_func_nans_default = Mean().get_function()
    mean_func_nans_false = Mean(skipna=False).get_function()
    mean_func_nans_true = Mean(skipna=True).get_function()
    assert mean_func_nans_default(array) == 5
    assert mean_func_nans_false(array) == 5
    assert mean_func_nans_true(array) == 5
    array = pd.Series([5, np.nan, np.nan, np.nan, np.nan, 10])
    assert mean_func_nans_default(array) == 7.5
    assert isnan(mean_func_nans_false(array))
    assert mean_func_nans_true(array) == 7.5
    array_nans = pd.Series([np.nan, np.nan, np.nan, np.nan])
    assert isnan(mean_func_nans_default(array_nans))
    assert isnan(mean_func_nans_false(array_nans))
    assert isnan(mean_func_nans_true(array_nans))

    # test naming
    default_feat = Feature(
        es["log"].ww["value"],
        parent_dataframe_name="customers",
        primitive=Mean,
    )
    assert default_feat.get_name() == "MEAN(log.value)"
    ignore_nan_feat = Feature(
        es["log"].ww["value"],
        parent_dataframe_name="customers",
        primitive=Mean(skipna=True),
    )
    assert ignore_nan_feat.get_name() == "MEAN(log.value)"
    include_nan_feat = Feature(
        es["log"].ww["value"],
        parent_dataframe_name="customers",
        primitive=Mean(skipna=False),
    )
    assert include_nan_feat.get_name() == "MEAN(log.value, skipna=False)"


def test_init_and_name(es):
    log = es["log"]

    # Add a BooleanNullable column so primitives with that input type get tested
    boolean_nullable = log.ww["purchased"]
    boolean_nullable = boolean_nullable.ww.set_logical_type("BooleanNullable")
    log.ww["boolean_nullable"] = boolean_nullable

    features = [Feature(es["log"].ww[col]) for col in log.columns]

    # check all primitives have name
    for attribute_string in dir(primitives):
        attr = getattr(primitives, attribute_string)
        if isclass(attr):
            if issubclass(attr, AggregationPrimitive) and attr != AggregationPrimitive:
                assert getattr(attr, "name") is not None

    agg_primitives = get_aggregation_primitives().values()
    # If Dask EntitySet use only Dask compatible primitives
    if es.dataframe_type == Library.DASK:
        agg_primitives = [
            prim for prim in agg_primitives if Library.DASK in prim.compatibility
        ]
    if es.dataframe_type == Library.SPARK:
        agg_primitives = [
            prim for prim in agg_primitives if Library.SPARK in prim.compatibility
        ]

    for agg_prim in agg_primitives:
        input_types = agg_prim.input_types
        if not isinstance(input_types[0], list):
            input_types = [input_types]

        # test each allowed input_types for this primitive
        for it in input_types:
            # use the input_types matching function from DFS
            matching_types = match(it, features)
            if len(matching_types) == 0:
                raise Exception("Agg Primitive %s not tested" % agg_prim.name)
            for t in matching_types:
                instance = Feature(
                    t,
                    parent_dataframe_name="sessions",
                    primitive=agg_prim,
                )

                # try to get name and calculate
                instance.get_name()
                calculate_feature_matrix([instance], entityset=es)


def test_invalid_init_args(diamond_es):
    error_text = "parent_dataframe must match first relationship in path"
    with pytest.raises(AssertionError, match=error_text):
        path = backward_path(diamond_es, ["stores", "transactions"])
        AggregationFeature(
            IdentityFeature(diamond_es["transactions"].ww["amount"]),
            "customers",
            Mean,
            relationship_path=path,
        )

    error_text = (
        "Base feature must be defined on the dataframe at the end of relationship_path"
    )
    with pytest.raises(AssertionError, match=error_text):
        path = backward_path(diamond_es, ["regions", "stores"])
        AggregationFeature(
            IdentityFeature(diamond_es["transactions"].ww["amount"]),
            "regions",
            Mean,
            relationship_path=path,
        )

    error_text = "All relationships in path must be backward"
    with pytest.raises(AssertionError, match=error_text):
        backward = backward_path(diamond_es, ["customers", "transactions"])
        forward = RelationshipPath([(True, r) for _, r in backward])
        path = RelationshipPath(list(forward) + list(backward))
        AggregationFeature(
            IdentityFeature(diamond_es["transactions"].ww["amount"]),
            "transactions",
            Mean,
            relationship_path=path,
        )


def test_init_with_multiple_possible_paths(diamond_es):
    error_text = (
        "There are multiple possible paths to the base dataframe. "
        "You must specify a relationship path."
    )
    with pytest.raises(RuntimeError, match=error_text):
        AggregationFeature(
            IdentityFeature(diamond_es["transactions"].ww["amount"]),
            "regions",
            Mean,
        )

    # Does not raise if path specified.
    path = backward_path(diamond_es, ["regions", "customers", "transactions"])
    AggregationFeature(
        IdentityFeature(diamond_es["transactions"].ww["amount"]),
        "regions",
        Mean,
        relationship_path=path,
    )


def test_init_with_single_possible_path(diamond_es):
    # This uses diamond_es to test that there being a cycle somewhere in the
    # graph doesn't cause an error.
    feat = AggregationFeature(
        IdentityFeature(diamond_es["transactions"].ww["amount"]),
        "customers",
        Mean,
    )
    expected_path = backward_path(diamond_es, ["customers", "transactions"])
    assert feat.relationship_path == expected_path


def test_init_with_no_path(diamond_es):
    error_text = 'No backward path from "transactions" to "customers" found.'
    with pytest.raises(RuntimeError, match=error_text):
        AggregationFeature(
            IdentityFeature(diamond_es["customers"].ww["name"]),
            "transactions",
            Count,
        )

    error_text = 'No backward path from "transactions" to "transactions" found.'
    with pytest.raises(RuntimeError, match=error_text):
        AggregationFeature(
            IdentityFeature(diamond_es["transactions"].ww["amount"]),
            "transactions",
            Mean,
        )


def test_name_with_multiple_possible_paths(diamond_es):
    path = backward_path(diamond_es, ["regions", "customers", "transactions"])
    feat = AggregationFeature(
        IdentityFeature(diamond_es["transactions"].ww["amount"]),
        "regions",
        Mean,
        relationship_path=path,
    )

    assert feat.get_name() == "MEAN(customers.transactions.amount)"
    assert feat.relationship_path_name() == "customers.transactions"


def test_copy(games_es):
    home_games = next(
        r for r in games_es.relationships if r._child_column_name == "home_team_id"
    )
    path = RelationshipPath([(False, home_games)])
    feat = AggregationFeature(
        IdentityFeature(games_es["games"].ww["home_team_score"]),
        "teams",
        relationship_path=path,
        primitive=Mean,
    )
    copied = feat.copy()
    assert copied.dataframe_name == feat.dataframe_name
    assert copied.base_features == feat.base_features
    assert copied.relationship_path == feat.relationship_path
    assert copied.primitive == feat.primitive


def test_serialization(es):
    value = IdentityFeature(es["log"].ww["value"])
    primitive = Max()
    max1 = AggregationFeature(value, "customers", primitive)

    path = next(es.find_backward_paths("customers", "log"))
    dictionary = {
        "name": max1.get_name(),
        "base_features": [value.unique_name()],
        "relationship_path": [r.to_dictionary() for r in path],
        "primitive": primitive,
        "where": None,
        "use_previous": None,
    }

    assert dictionary == max1.get_arguments()
    deserialized = AggregationFeature.from_dictionary(
        dictionary,
        es,
        {value.unique_name(): value},
        primitive,
    )
    _assert_agg_feats_equal(max1, deserialized)

    is_purchased = IdentityFeature(es["log"].ww["purchased"])
    use_previous = Timedelta(3, "d")
    max2 = AggregationFeature(
        value,
        "customers",
        primitive,
        where=is_purchased,
        use_previous=use_previous,
    )

    dictionary = {
        "name": max2.get_name(),
        "base_features": [value.unique_name()],
        "relationship_path": [r.to_dictionary() for r in path],
        "primitive": primitive,
        "where": is_purchased.unique_name(),
        "use_previous": use_previous.get_arguments(),
    }

    assert dictionary == max2.get_arguments()
    dependencies = {
        value.unique_name(): value,
        is_purchased.unique_name(): is_purchased,
    }
    deserialized = AggregationFeature.from_dictionary(
        dictionary,
        es,
        dependencies,
        primitive,
    )
    _assert_agg_feats_equal(max2, deserialized)


def test_time_since_last(pd_es):
    f = Feature(
        pd_es["log"].ww["datetime"],
        parent_dataframe_name="customers",
        primitive=TimeSinceLast,
    )
    fm = calculate_feature_matrix(
        [f],
        entityset=pd_es,
        instance_ids=[0, 1, 2],
        cutoff_time=datetime(2015, 6, 8),
    )

    correct = [131376000.0, 131289534.0, 131287797.0]
    # note: must round to nearest second
    assert all(fm[f.get_name()].round().values == correct)


def test_time_since_first(pd_es):
    f = Feature(
        pd_es["log"].ww["datetime"],
        parent_dataframe_name="customers",
        primitive=TimeSinceFirst,
    )
    fm = calculate_feature_matrix(
        [f],
        entityset=pd_es,
        instance_ids=[0, 1, 2],
        cutoff_time=datetime(2015, 6, 8),
    )

    correct = [131376600.0, 131289600.0, 131287800.0]
    # note: must round to nearest second
    assert all(fm[f.get_name()].round().values == correct)


def test_median(pd_es):
    f = Feature(
        pd_es["log"].ww["value_many_nans"],
        parent_dataframe_name="customers",
        primitive=Median,
    )
    fm = calculate_feature_matrix(
        [f],
        entityset=pd_es,
        instance_ids=[0, 1, 2],
        cutoff_time=datetime(2015, 6, 8),
    )

    correct = [1, 3, np.nan]
    np.testing.assert_equal(fm[f.get_name()].values, correct)


def test_agg_same_method_name(es):
    """
    Pandas relies on the function name when calculating aggregations. This means if a two
    primitives with the same function name are applied to the same column, pandas
    can't differentiate them. We have a work around to this based on the name property
    that we test here.
    """
    # TODO: Update to work with Dask and Spark
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Need to update to work with Dask and Spark EntitySets")

    # test with normally defined functions
    class Sum(AggregationPrimitive):
        name = "sum"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})

        def get_function(self):
            def custom_primitive(x):
                return x.sum()

            return custom_primitive

    class Max(AggregationPrimitive):
        name = "max"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})

        def get_function(self):
            def custom_primitive(x):
                return x.max()

            return custom_primitive

    f_sum = Feature(
        es["log"].ww["value"],
        parent_dataframe_name="customers",
        primitive=Sum,
    )
    f_max = Feature(
        es["log"].ww["value"],
        parent_dataframe_name="customers",
        primitive=Max,
    )

    fm = calculate_feature_matrix([f_sum, f_max], entityset=es)
    assert fm.columns.tolist() == [f_sum.get_name(), f_max.get_name()]

    # test with lambdas
    class Sum(AggregationPrimitive):
        name = "sum"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})

        def get_function(self):
            return lambda x: x.sum()

    class Max(AggregationPrimitive):
        name = "max"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})

        def get_function(self):
            return lambda x: x.max()

    f_sum = Feature(
        es["log"].ww["value"],
        parent_dataframe_name="customers",
        primitive=Sum,
    )
    f_max = Feature(
        es["log"].ww["value"],
        parent_dataframe_name="customers",
        primitive=Max,
    )
    fm = calculate_feature_matrix([f_sum, f_max], entityset=es)
    assert fm.columns.tolist() == [f_sum.get_name(), f_max.get_name()]


def test_time_since_last_custom(pd_es):
    class TimeSinceLast(AggregationPrimitive):
        name = "time_since_last"
        input_types = [
            ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
        ]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        uses_calc_time = True

        def get_function(self):
            def time_since_last(values, time):
                time_since = time - values.iloc[0]
                return time_since.total_seconds()

            return time_since_last

    f = Feature(
        pd_es["log"].ww["datetime"],
        parent_dataframe_name="customers",
        primitive=TimeSinceLast,
    )
    fm = calculate_feature_matrix(
        [f],
        entityset=pd_es,
        instance_ids=[0, 1, 2],
        cutoff_time=datetime(2015, 6, 8),
    )

    correct = [131376600, 131289600, 131287800]
    # note: must round to nearest second
    assert all(fm[f.get_name()].round().values == correct)


def test_custom_primitive_multiple_inputs(pd_es):
    class MeanSunday(AggregationPrimitive):
        name = "mean_sunday"
        input_types = [
            ColumnSchema(semantic_tags={"numeric"}),
            ColumnSchema(logical_type=Datetime),
        ]
        return_type = ColumnSchema(semantic_tags={"numeric"})

        def get_function(self):
            def mean_sunday(numeric, datetime):
                """
                Finds the mean of non-null values of a feature that occurred on Sundays
                """
                days = pd.DatetimeIndex(datetime).weekday.values
                df = pd.DataFrame({"numeric": numeric, "time": days})
                return df[df["time"] == 6]["numeric"].mean()

            return mean_sunday

    fm, features = dfs(
        entityset=pd_es,
        target_dataframe_name="sessions",
        agg_primitives=[MeanSunday],
        trans_primitives=[],
    )
    mean_sunday_value = pd.Series([None, None, None, 2.5, 7, None])
    iterator = zip(fm["MEAN_SUNDAY(log.value, datetime)"], mean_sunday_value)
    for x, y in iterator:
        assert (pd.isnull(x) and pd.isnull(y)) or (x == y)

    pd_es.add_interesting_values()
    mean_sunday_value_priority_0 = pd.Series([None, None, None, 2.5, 0, None])
    fm, features = dfs(
        entityset=pd_es,
        target_dataframe_name="sessions",
        agg_primitives=[MeanSunday],
        trans_primitives=[],
        where_primitives=[MeanSunday],
    )
    where_feat = "MEAN_SUNDAY(log.value, datetime WHERE priority_level = 0)"
    for x, y in zip(fm[where_feat], mean_sunday_value_priority_0):
        assert (pd.isnull(x) and pd.isnull(y)) or (x == y)


def test_custom_primitive_default_kwargs(es):
    class SumNTimes(AggregationPrimitive):
        name = "sum_n_times"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})

        def __init__(self, n=1):
            self.n = n

    sum_n_1_n = 1
    sum_n_1_base_f = Feature(es["log"].ww["value"])
    sum_n_1 = Feature(
        [sum_n_1_base_f],
        parent_dataframe_name="sessions",
        primitive=SumNTimes(n=sum_n_1_n),
    )
    sum_n_2_n = 2
    sum_n_2_base_f = Feature(es["log"].ww["value_2"])
    sum_n_2 = Feature(
        [sum_n_2_base_f],
        parent_dataframe_name="sessions",
        primitive=SumNTimes(n=sum_n_2_n),
    )
    assert sum_n_1_base_f == sum_n_1.base_features[0]
    assert sum_n_1_n == sum_n_1.primitive.n
    assert sum_n_2_base_f == sum_n_2.base_features[0]
    assert sum_n_2_n == sum_n_2.primitive.n


def test_makes_numtrue(es):
    if es.dataframe_type == Library.SPARK:
        pytest.xfail("Spark EntitySets do not support NumTrue primitive")
    dfs = DeepFeatureSynthesis(
        target_dataframe_name="sessions",
        entityset=es,
        agg_primitives=[NumTrue],
        trans_primitives=[],
    )
    features = dfs.build_features()
    assert feature_with_name(features, "customers.NUM_TRUE(log.purchased)")
    assert feature_with_name(features, "NUM_TRUE(log.purchased)")


def test_make_three_most_common(pd_es):
    class NMostCommoner(AggregationPrimitive):
        name = "pd_top3"
        input_types = ([ColumnSchema(semantic_tags={"category"})],)
        return_type = None
        number_output_features = 3

        def get_function(self):
            def pd_top3(x):
                counts = x.value_counts()
                counts = counts[counts > 0]
                array = np.array(counts[:3].index)
                if len(array) < 3:
                    filler = np.full(3 - len(array), np.nan)
                    array = np.append(array, filler)
                return array

            return pd_top3

    fm, features = dfs(
        entityset=pd_es,
        target_dataframe_name="customers",
        instance_ids=[0, 1, 2],
        agg_primitives=[NMostCommoner],
        trans_primitives=[],
    )

    df = fm[["PD_TOP3(log.product_id)[%s]" % i for i in range(3)]]

    assert set(df.iloc[0].values[:2]) == set(
        ["coke zero", "toothpaste"],
    )  # coke zero and toothpaste have same number of occurrences
    assert df.iloc[0].values[2] in [
        "car",
        "brown bag",
    ]  # so just check that the top two match

    assert (
        df.iloc[1]
        .reset_index(drop=True)
        .equals(pd.Series(["coke zero", "Haribo sugar-free gummy bears", np.nan]))
    )
    assert (
        df.iloc[2]
        .reset_index(drop=True)
        .equals(pd.Series(["taco clock", np.nan, np.nan]))
    )


def test_stacking_multi(pd_es):
    threecommon = NMostCommon(3)
    tc = Feature(
        pd_es["log"].ww["product_id"],
        parent_dataframe_name="sessions",
        primitive=threecommon,
    )

    stacked = []
    for i in range(3):
        stacked.append(
            Feature(tc[i], parent_dataframe_name="customers", primitive=NumUnique),
        )

    fm = calculate_feature_matrix(stacked, entityset=pd_es, instance_ids=[0, 1, 2])

    correct_vals = [[3, 2, 1], [2, 1, 0], [0, 0, 0]]
    correct_vals1 = [[3, 1, 1], [2, 1, 0], [0, 0, 0]]
    # either of the above can be correct, and the outcome depends on the sorting of
    # two values in the initial n most common function, which changes arbitrarily.

    for i in range(3):
        f = "NUM_UNIQUE(sessions.N_MOST_COMMON(log.product_id)[%d])" % i
        cols = fm.columns
        assert f in cols
        assert (
            fm[cols[i]].tolist() == correct_vals[i]
            or fm[cols[i]].tolist() == correct_vals1[i]
        )


def test_use_previous_pd_dateoffset(es):
    total_events_pd = Feature(
        es["log"].ww["id"],
        parent_dataframe_name="customers",
        use_previous=pd.DateOffset(hours=47, minutes=60),
        primitive=Count,
    )

    feature_matrix = calculate_feature_matrix(
        [total_events_pd],
        es,
        cutoff_time=pd.Timestamp("2011-04-11 10:31:30"),
        instance_ids=[0, 1, 2],
    )
    feature_matrix = to_pandas(feature_matrix, index="id", sort_index=True)
    col_name = list(feature_matrix.head().keys())[0]
    assert (feature_matrix[col_name] == [1, 5, 2]).all()


def _assert_agg_feats_equal(f1, f2):
    assert f1.unique_name() == f2.unique_name()
    assert f1.child_dataframe_name == f2.child_dataframe_name
    assert f1.parent_dataframe_name == f2.parent_dataframe_name
    assert f1.relationship_path == f2.relationship_path
    assert f1.use_previous == f2.use_previous


def test_override_multi_feature_names(pd_es):
    def gen_custom_names(
        primitive,
        base_feature_names,
        relationship_path_name,
        parent_dataframe_name,
        where_str,
        use_prev_str,
    ):
        base_string = "Custom_%s({}.{})".format(
            parent_dataframe_name,
            base_feature_names,
        )
        return [base_string % i for i in range(primitive.number_output_features)]

    class NMostCommoner(AggregationPrimitive):
        name = "pd_top3"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"category"})
        number_output_features = 3

        def generate_names(
            self,
            base_feature_names,
            relationship_path_name,
            parent_dataframe_name,
            where_str,
            use_prev_str,
        ):
            return gen_custom_names(
                self,
                base_feature_names,
                relationship_path_name,
                parent_dataframe_name,
                where_str,
                use_prev_str,
            )

    fm, features = dfs(
        entityset=pd_es,
        target_dataframe_name="products",
        instance_ids=[0, 1, 2],
        agg_primitives=[NMostCommoner],
        trans_primitives=[],
    )

    expected_names = []
    base_names = [["value"], ["value_2"], ["value_many_nans"]]
    for name in base_names:
        expected_names += gen_custom_names(
            NMostCommoner,
            name,
            None,
            "products",
            None,
            None,
        )

    for name in expected_names:
        assert name in fm.columns
