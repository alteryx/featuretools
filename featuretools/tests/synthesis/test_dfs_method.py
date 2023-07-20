from unittest.mock import patch

import composeml as cp
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import NaturalLanguage

from featuretools.computational_backends.calculate_feature_matrix import (
    FEATURE_CALCULATION_PERCENTAGE,
)
from featuretools.entityset import EntitySet, Timedelta
from featuretools.exceptions import UnusedPrimitiveWarning
from featuretools.primitives import GreaterThanScalar, Max, Mean, Min, Sum
from featuretools.primitives.base import AggregationPrimitive, TransformPrimitive
from featuretools.synthesis import dfs
from featuretools.synthesis.deep_feature_synthesis import DeepFeatureSynthesis
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import Library, import_or_none, is_instance

dd = import_or_none("dask.dataframe")


@pytest.fixture
def datetime_es():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "card_id": [1, 1, 5, 1, 5],
            "transaction_time": pd.to_datetime(
                [
                    "2011-2-28 04:00",
                    "2012-2-28 05:00",
                    "2012-2-29 06:00",
                    "2012-3-1 08:00",
                    "2014-4-1 10:00",
                ],
            ),
            "fraud": [True, False, False, False, True],
        },
    )

    datetime_es = EntitySet(id="fraud_data")
    datetime_es = datetime_es.add_dataframe(
        dataframe_name="transactions",
        dataframe=transactions_df,
        index="id",
        time_index="transaction_time",
    )

    datetime_es = datetime_es.add_dataframe(
        dataframe_name="cards",
        dataframe=cards_df,
        index="id",
    )

    datetime_es = datetime_es.add_relationship("cards", "id", "transactions", "card_id")
    datetime_es.add_last_time_indexes()
    return datetime_es


def test_dfs_empty_features():
    error_text = "No features can be generated from the specified primitives. Please make sure the primitives you are using are compatible with the variable types in your data."
    teams = pd.DataFrame({"id": range(3), "name": ["Breakers", "Spirit", "Thorns"]})
    games = pd.DataFrame(
        {
            "id": range(5),
            "home_team_id": [2, 2, 1, 0, 1],
            "away_team_id": [1, 0, 2, 1, 0],
            "home_team_score": [3, 0, 1, 0, 4],
            "away_team_score": [2, 1, 2, 0, 0],
        },
    )
    dataframes = {
        "teams": (teams, "id", None, {"name": "natural_language"}),
        "games": (games, "id"),
    }
    relationships = [("teams", "id", "games", "home_team_id")]
    with patch.object(DeepFeatureSynthesis, "build_features", return_value=[]):
        features = dfs(
            dataframes,
            relationships,
            target_dataframe_name="teams",
            features_only=True,
        )
        assert features == []
    with pytest.raises(AssertionError, match=error_text), patch.object(
        DeepFeatureSynthesis,
        "build_features",
        return_value=[],
    ):
        dfs(
            dataframes,
            relationships,
            target_dataframe_name="teams",
            features_only=False,
        )


def test_passing_strings_to_logical_types_dfs():
    teams = pd.DataFrame({"id": range(3), "name": ["Breakers", "Spirit", "Thorns"]})
    games = pd.DataFrame(
        {
            "id": range(5),
            "home_team_id": [2, 2, 1, 0, 1],
            "away_team_id": [1, 0, 2, 1, 0],
            "home_team_score": [3, 0, 1, 0, 4],
            "away_team_score": [2, 1, 2, 0, 0],
        },
    )
    dataframes = {
        "teams": (teams, "id", None, {"name": "natural_language"}),
        "games": (games, "id"),
    }
    relationships = [("teams", "id", "games", "home_team_id")]

    features = dfs(
        dataframes,
        relationships,
        target_dataframe_name="teams",
        features_only=True,
    )

    name_logical_type = features[0].dataframe["name"].ww.logical_type
    assert isinstance(name_logical_type, NaturalLanguage)


def test_accepts_cutoff_time_df(dataframes, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3], "time": [10, 12, 15]})
    feature_matrix, features = dfs(
        dataframes=dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
        cutoff_time=cutoff_times_df,
    )
    feature_matrix = to_pandas(feature_matrix, index="id", sort_index=True)
    assert len(feature_matrix.index) == 3
    assert len(feature_matrix.columns) == len(features)


@pytest.mark.skipif("not dd")
def test_warns_cutoff_time_dask(dataframes, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3], "time": [10, 12, 15]})
    cutoff_times_df = dd.from_pandas(cutoff_times_df, npartitions=2)
    match = (
        "cutoff_time should be a Pandas DataFrame: "
        "computing cutoff_time, this may take a while"
    )
    with pytest.warns(UserWarning, match=match):
        dfs(
            dataframes=dataframes,
            relationships=relationships,
            target_dataframe_name="transactions",
            cutoff_time=cutoff_times_df,
        )


def test_accepts_cutoff_time_compose(dataframes, relationships):
    def fraud_occured(df):
        return df["fraud"].any()

    kwargs = {
        "time_index": "transaction_time",
        "labeling_function": fraud_occured,
        "window_size": 1,
    }
    if parse(cp.__version__) >= parse("0.10.0"):
        kwargs["target_dataframe_index"] = "card_id"
    else:
        kwargs["target_dataframe_name"] = "card_id"  # pragma: no cover

    lm = cp.LabelMaker(**kwargs)

    transactions_df = to_pandas(dataframes["transactions"][0])

    labels = lm.search(transactions_df, num_examples_per_instance=-1)

    labels["time"] = pd.to_numeric(labels["time"])
    labels.rename({"card_id": "id"}, axis=1, inplace=True)

    feature_matrix, features = dfs(
        dataframes=dataframes,
        relationships=relationships,
        target_dataframe_name="cards",
        cutoff_time=labels,
    )
    feature_matrix = to_pandas(feature_matrix, index="id")
    assert len(feature_matrix.index) == 6
    assert len(feature_matrix.columns) == len(features) + 1


def test_accepts_single_cutoff_time(dataframes, relationships):
    feature_matrix, features = dfs(
        dataframes=dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
        cutoff_time=20,
    )
    feature_matrix = to_pandas(feature_matrix, index="id")
    assert len(feature_matrix.index) == 5
    assert len(feature_matrix.columns) == len(features)


def test_accepts_no_cutoff_time(dataframes, relationships):
    feature_matrix, features = dfs(
        dataframes=dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
        instance_ids=[1, 2, 3, 5, 6],
    )
    feature_matrix = to_pandas(feature_matrix, index="id")
    assert len(feature_matrix.index) == 5
    assert len(feature_matrix.columns) == len(features)


def test_ignores_instance_ids_if_cutoff_df(dataframes, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3], "time": [10, 12, 15]})
    instance_ids = [1, 2, 3, 4, 5]
    feature_matrix, features = dfs(
        dataframes=dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
        cutoff_time=cutoff_times_df,
        instance_ids=instance_ids,
    )
    feature_matrix = to_pandas(feature_matrix, index="id")
    assert len(feature_matrix.index) == 3
    assert len(feature_matrix.columns) == len(features)


def test_approximate_features(pd_dataframes, relationships):
    # TODO: Update to use Dask dataframes when issue #985 is closed
    cutoff_times_df = pd.DataFrame(
        {"instance_id": [1, 3, 1, 5, 3, 6], "time": [11, 16, 16, 26, 17, 22]},
    )
    # force column to BooleanNullable
    pd_dataframes["transactions"] += ({"fraud": "BooleanNullable"},)
    feature_matrix, features = dfs(
        dataframes=pd_dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
        cutoff_time=cutoff_times_df,
        approximate=5,
        cutoff_time_in_index=True,
    )
    direct_agg_feat_name = "cards.PERCENT_TRUE(transactions.fraud)"
    assert len(feature_matrix.index) == 6
    assert len(feature_matrix.columns) == len(features)

    truth_values = pd.Series(data=[1.0, 0.5, 0.5, 1.0, 0.5, 1.0])

    assert (feature_matrix[direct_agg_feat_name] == truth_values.values).all()


def test_all_columns(pd_dataframes, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3], "time": [10, 12, 15]})
    feature_matrix, features = dfs(
        dataframes=pd_dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
        cutoff_time=cutoff_times_df,
        agg_primitives=[Max, Mean, Min, Sum],
        trans_primitives=[],
        groupby_trans_primitives=["cum_sum"],
        max_depth=3,
        allowed_paths=None,
        ignore_dataframes=None,
        ignore_columns=None,
        seed_features=None,
    )
    assert len(feature_matrix.index) == 3
    assert len(feature_matrix.columns) == len(features)


def test_features_only(dataframes, relationships):
    if len(dataframes["transactions"]) > 3:
        dataframes["transactions"][3]["fraud"] = "BooleanNullable"
    else:
        dataframes["transactions"] += ({"fraud": "BooleanNullable"},)
    features = dfs(
        dataframes=dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
        features_only=True,
    )

    # pandas creates 11 features
    # dask creates 10 features (no skew)
    # spark creates 9 features (no skew, no percent_true)
    if isinstance(dataframes["transactions"][0], pd.DataFrame):
        expected_features = 11
    elif is_instance(dataframes["transactions"][0], dd, "DataFrame"):
        expected_features = 10
    else:
        expected_features = 9
    assert len(features) == expected_features


def test_accepts_relative_training_window(datetime_es):
    # TODO: Update to use Dask dataframes when issue #882 is closed
    feature_matrix, _ = dfs(entityset=datetime_es, target_dataframe_name="transactions")

    feature_matrix_2, _ = dfs(
        entityset=datetime_es,
        target_dataframe_name="transactions",
        cutoff_time=pd.Timestamp("2012-4-1 04:00"),
    )

    feature_matrix_3, _ = dfs(
        entityset=datetime_es,
        target_dataframe_name="transactions",
        cutoff_time=pd.Timestamp("2012-4-1 04:00"),
        training_window=Timedelta("3 months"),
    )

    feature_matrix_4, _ = dfs(
        entityset=datetime_es,
        target_dataframe_name="transactions",
        cutoff_time=pd.Timestamp("2012-4-1 04:00"),
        training_window="3 months",
    )

    assert (feature_matrix.index == [1, 2, 3, 4, 5]).all()
    assert (feature_matrix_2.index == [1, 2, 3, 4]).all()
    assert (feature_matrix_3.index == [2, 3, 4]).all()
    assert (feature_matrix_4.index == [2, 3, 4]).all()

    # Test case for leap years
    feature_matrix_5, _ = dfs(
        entityset=datetime_es,
        target_dataframe_name="transactions",
        cutoff_time=pd.Timestamp("2012-2-29 04:00"),
        training_window=Timedelta("1 year"),
        include_cutoff_time=True,
    )
    assert (feature_matrix_5.index == [2]).all()

    feature_matrix_5, _ = dfs(
        entityset=datetime_es,
        target_dataframe_name="transactions",
        cutoff_time=pd.Timestamp("2012-2-29 04:00"),
        training_window=Timedelta("1 year"),
        include_cutoff_time=False,
    )
    assert (feature_matrix_5.index == [1, 2]).all()


def test_accepts_pd_timedelta_training_window(datetime_es):
    # TODO: Update to use Dask dataframes when issue #882 is closed
    feature_matrix, _ = dfs(
        entityset=datetime_es,
        target_dataframe_name="transactions",
        cutoff_time=pd.Timestamp("2012-3-31 04:00"),
        training_window=pd.Timedelta(61, "D"),
    )

    assert (feature_matrix.index == [2, 3, 4]).all()


def test_accepts_pd_dateoffset_training_window(datetime_es):
    # TODO: Update to use Dask dataframes when issue #882 is closed
    feature_matrix, _ = dfs(
        entityset=datetime_es,
        target_dataframe_name="transactions",
        cutoff_time=pd.Timestamp("2012-3-31 04:00"),
        training_window=pd.DateOffset(months=2),
    )

    feature_matrix_2, _ = dfs(
        entityset=datetime_es,
        target_dataframe_name="transactions",
        cutoff_time=pd.Timestamp("2012-3-31 04:00"),
        training_window=pd.offsets.BDay(44),
    )

    assert (feature_matrix.index == [2, 3, 4]).all()
    assert (feature_matrix.index == feature_matrix_2.index).all()


def test_accepts_datetime_and_string_offset(datetime_es):
    feature_matrix, _ = dfs(
        entityset=datetime_es,
        target_dataframe_name="transactions",
        cutoff_time=pd.to_datetime("2012-3-31 04:00"),
        training_window=pd.DateOffset(months=2),
    )

    feature_matrix_2, _ = dfs(
        entityset=datetime_es,
        target_dataframe_name="transactions",
        cutoff_time="2012-3-31 04:00",
        training_window=pd.offsets.BDay(44),
    )

    assert (feature_matrix.index == [2, 3, 4]).all()
    assert (feature_matrix.index == feature_matrix_2.index).all()


def test_handles_pandas_parser_error(datetime_es):
    with pytest.raises(ValueError):
        _, _ = dfs(
            entityset=datetime_es,
            target_dataframe_name="transactions",
            cutoff_time="2--012-----3-----31 04:00",
            training_window=pd.DateOffset(months=2),
        )


def test_handles_pandas_overflow_error(datetime_es):
    # pandas 1.5.0 raises ValueError, older versions raised OverflowError
    with pytest.raises((OverflowError, ValueError)):
        _, _ = dfs(
            entityset=datetime_es,
            target_dataframe_name="transactions",
            cutoff_time="200000000000000000000000000000000000000000000000000000000000000000-3-31 04:00",
            training_window=pd.DateOffset(months=2),
        )


def test_warns_with_unused_primitives(es):
    if es.dataframe_type == Library.SPARK:
        pytest.skip("Spark throws extra warnings")
    trans_primitives = ["num_characters", "num_words", "add_numeric"]
    agg_primitives = [Max, "min"]

    warning_text = (
        "Some specified primitives were not used during DFS:\n"
        + "  trans_primitives: ['add_numeric']\n  agg_primitives: ['max', 'min']\n"
        + "This may be caused by a using a value of max_depth that is too small, not setting interesting values, "
        + "or it may indicate no compatible columns for the primitive were found in the data. If the DFS call "
        + "contained multiple instances of a primitive in the list above, none of them were used."
    )

    with pytest.warns(UnusedPrimitiveWarning) as record:
        dfs(
            entityset=es,
            target_dataframe_name="customers",
            trans_primitives=trans_primitives,
            agg_primitives=agg_primitives,
            max_depth=1,
            features_only=True,
        )

    assert record[0].message.args[0] == warning_text

    # Should not raise a warning
    with pytest.warns(None) as record:
        dfs(
            entityset=es,
            target_dataframe_name="customers",
            trans_primitives=trans_primitives,
            agg_primitives=agg_primitives,
            max_depth=2,
            features_only=True,
        )

    assert not record


def test_no_warns_with_camel_and_title_case(es):
    for trans_primitive in ["isNull", "IsNull"]:
        # Should not raise a UnusedPrimitiveWarning warning
        with pytest.warns(None) as record:
            dfs(
                entityset=es,
                target_dataframe_name="customers",
                trans_primitives=[trans_primitive],
                max_depth=1,
                features_only=True,
            )

        assert not record

    for agg_primitive in ["numUnique", "NumUnique"]:
        # Should not raise a UnusedPrimitiveWarning warning
        with pytest.warns(None) as record:
            dfs(
                entityset=es,
                target_dataframe_name="customers",
                agg_primitives=[agg_primitive],
                max_depth=2,
                features_only=True,
            )

        assert not record


def test_does_not_warn_with_stacking_feature(pd_es):
    with pytest.warns(None) as record:
        dfs(
            entityset=pd_es,
            target_dataframe_name="régions",
            agg_primitives=["percent_true"],
            trans_primitives=[GreaterThanScalar(5)],
            primitive_options={
                "greater_than_scalar": {"include_dataframes": ["stores"]},
            },
            features_only=True,
        )

    assert not record


def test_warns_with_unused_where_primitives(es):
    if es.dataframe_type == Library.SPARK:
        pytest.skip("Spark throws extra warnings")
    warning_text = (
        "Some specified primitives were not used during DFS:\n"
        + "  where_primitives: ['count', 'sum']\n"
        + "This may be caused by a using a value of max_depth that is too small, not setting interesting values, "
        + "or it may indicate no compatible columns for the primitive were found in the data. If the DFS call "
        + "contained multiple instances of a primitive in the list above, none of them were used."
    )

    with pytest.warns(UnusedPrimitiveWarning) as record:
        dfs(
            entityset=es,
            target_dataframe_name="customers",
            agg_primitives=["count"],
            where_primitives=["sum", "count"],
            max_depth=1,
            features_only=True,
        )

    assert record[0].message.args[0] == warning_text


def test_warns_with_unused_groupby_primitives(pd_es):
    warning_text = (
        "Some specified primitives were not used during DFS:\n"
        + "  groupby_trans_primitives: ['cum_sum']\n"
        + "This may be caused by a using a value of max_depth that is too small, not setting interesting values, "
        + "or it may indicate no compatible columns for the primitive were found in the data. If the DFS call "
        + "contained multiple instances of a primitive in the list above, none of them were used."
    )

    with pytest.warns(UnusedPrimitiveWarning) as record:
        dfs(
            entityset=pd_es,
            target_dataframe_name="sessions",
            groupby_trans_primitives=["cum_sum"],
            max_depth=1,
            features_only=True,
        )

    assert record[0].message.args[0] == warning_text

    # Should not raise a warning
    with pytest.warns(None) as record:
        dfs(
            entityset=pd_es,
            target_dataframe_name="customers",
            groupby_trans_primitives=["cum_sum"],
            max_depth=1,
            features_only=True,
        )

    assert not record


def test_warns_with_unused_custom_primitives(pd_es):
    class AboveTen(TransformPrimitive):
        name = "above_ten"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})

    trans_primitives = [AboveTen]

    warning_text = (
        "Some specified primitives were not used during DFS:\n"
        + "  trans_primitives: ['above_ten']\n"
        + "This may be caused by a using a value of max_depth that is too small, not setting interesting values, "
        + "or it may indicate no compatible columns for the primitive were found in the data. If the DFS call "
        + "contained multiple instances of a primitive in the list above, none of them were used."
    )

    with pytest.warns(UnusedPrimitiveWarning) as record:
        dfs(
            entityset=pd_es,
            target_dataframe_name="sessions",
            trans_primitives=trans_primitives,
            max_depth=1,
            features_only=True,
        )

    assert record[0].message.args[0] == warning_text

    # Should not raise a warning
    with pytest.warns(None) as record:
        dfs(
            entityset=pd_es,
            target_dataframe_name="customers",
            trans_primitives=trans_primitives,
            max_depth=1,
            features_only=True,
        )

    class MaxAboveTen(AggregationPrimitive):
        name = "max_above_ten"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})

    agg_primitives = [MaxAboveTen]

    warning_text = (
        "Some specified primitives were not used during DFS:\n"
        + "  agg_primitives: ['max_above_ten']\n"
        + "This may be caused by a using a value of max_depth that is too small, not setting interesting values, "
        + "or it may indicate no compatible columns for the primitive were found in the data. If the DFS call "
        + "contained multiple instances of a primitive in the list above, none of them were used."
    )

    with pytest.warns(UnusedPrimitiveWarning) as record:
        dfs(
            entityset=pd_es,
            target_dataframe_name="stores",
            agg_primitives=agg_primitives,
            max_depth=1,
            features_only=True,
        )

    assert record[0].message.args[0] == warning_text

    # Should not raise a warning
    with pytest.warns(None) as record:
        dfs(
            entityset=pd_es,
            target_dataframe_name="sessions",
            agg_primitives=agg_primitives,
            max_depth=1,
            features_only=True,
        )


def test_calls_progress_callback(dataframes, relationships):
    class MockProgressCallback:
        def __init__(self):
            self.progress_history = []
            self.total_update = 0
            self.total_progress_percent = 0

        def __call__(self, update, progress_percent, time_elapsed):
            self.total_update += update
            self.total_progress_percent = progress_percent
            self.progress_history.append(progress_percent)

    mock_progress_callback = MockProgressCallback()

    dfs(
        dataframes=dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
        progress_callback=mock_progress_callback,
    )

    # second to last entry is the last update from feature calculation
    assert np.isclose(
        mock_progress_callback.progress_history[-2],
        FEATURE_CALCULATION_PERCENTAGE * 100,
    )
    assert np.isclose(mock_progress_callback.total_update, 100.0)
    assert np.isclose(mock_progress_callback.total_progress_percent, 100.0)


def test_calls_progress_callback_cluster(pd_dataframes, relationships, dask_cluster):
    class MockProgressCallback:
        def __init__(self):
            self.progress_history = []
            self.total_update = 0
            self.total_progress_percent = 0

        def __call__(self, update, progress_percent, time_elapsed):
            self.total_update += update
            self.total_progress_percent = progress_percent
            self.progress_history.append(progress_percent)

    mock_progress_callback = MockProgressCallback()

    dkwargs = {"cluster": dask_cluster.scheduler.address}
    dfs(
        dataframes=pd_dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
        progress_callback=mock_progress_callback,
        dask_kwargs=dkwargs,
    )

    assert np.isclose(mock_progress_callback.total_update, 100.0)
    assert np.isclose(mock_progress_callback.total_progress_percent, 100.0)


def test_dask_kwargs(pd_dataframes, relationships, dask_cluster):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3], "time": [10, 12, 15]})
    feature_matrix, features = dfs(
        dataframes=pd_dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
        cutoff_time=cutoff_times_df,
    )

    dask_kwargs = {"cluster": dask_cluster.scheduler.address}
    feature_matrix_2, features_2 = dfs(
        dataframes=pd_dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
        cutoff_time=cutoff_times_df,
        dask_kwargs=dask_kwargs,
    )

    assert all(
        f1.unique_name() == f2.unique_name() for f1, f2 in zip(features, features_2)
    )
    for column in feature_matrix:
        for x, y in zip(feature_matrix[column], feature_matrix_2[column]):
            assert (pd.isnull(x) and pd.isnull(y)) or (x == y)
