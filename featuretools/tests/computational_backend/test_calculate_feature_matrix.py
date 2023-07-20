import logging
import os
import re
import shutil
from datetime import datetime
from itertools import combinations
from random import randint

import numpy as np
import pandas as pd
import psutil
import pytest
from tqdm import tqdm
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    Age,
    AgeNullable,
    Boolean,
    BooleanNullable,
    Integer,
    IntegerNullable,
)

from featuretools import (
    EntitySet,
    Feature,
    GroupByTransformFeature,
    Timedelta,
    calculate_feature_matrix,
    dfs,
)
from featuretools.computational_backends import utils
from featuretools.computational_backends.calculate_feature_matrix import (
    FEATURE_CALCULATION_PERCENTAGE,
    _chunk_dataframe_groups,
    _handle_chunk_size,
    scatter_warning,
)
from featuretools.computational_backends.utils import (
    bin_cutoff_times,
    create_client_and_cluster,
    n_jobs_to_workers,
)
from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    FeatureOutputSlice,
    IdentityFeature,
)
from featuretools.primitives import (
    Count,
    Max,
    Min,
    Negate,
    NMostCommon,
    Percentile,
    Sum,
    TransformPrimitive,
)
from featuretools.tests.testing_utils import (
    backward_path,
    get_mock_client_cluster,
    to_pandas,
)
from featuretools.utils.gen_utils import Library, import_or_none

dd = import_or_none("dask.dataframe")


def test_scatter_warning(caplog):
    logger = logging.getLogger("featuretools")
    match = "EntitySet was only scattered to {} out of {} workers"
    warning_message = match.format(1, 2)
    logger.propagate = True
    scatter_warning(1, 2)
    logger.propagate = False
    assert warning_message in caplog.text


# TODO: final assert fails w/ Dask
def test_calc_feature_matrix(es):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Distributed dataframe result not ordered")
    times = list(
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)]
        + [datetime(2011, 4, 9, 10, 40, 0)]
        + [datetime(2011, 4, 10, 10, 40, i) for i in range(2)]
        + [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)]
        + [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)],
    )
    instances = range(17)
    cutoff_time = pd.DataFrame({"time": times, es["log"].ww.index: instances})
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [True] + [False] * 2

    property_feature = Feature(es["log"].ww["value"]) > 10

    feature_matrix = calculate_feature_matrix(
        [property_feature],
        es,
        cutoff_time=cutoff_time,
        verbose=True,
    )

    assert (feature_matrix[property_feature.get_name()] == labels).values.all()

    error_text = "features must be a non-empty list of features"
    with pytest.raises(AssertionError, match=error_text):
        feature_matrix = calculate_feature_matrix(
            "features",
            es,
            cutoff_time=cutoff_time,
        )

    with pytest.raises(AssertionError, match=error_text):
        feature_matrix = calculate_feature_matrix([], es, cutoff_time=cutoff_time)

    with pytest.raises(AssertionError, match=error_text):
        feature_matrix = calculate_feature_matrix(
            [1, 2, 3],
            es,
            cutoff_time=cutoff_time,
        )

    error_text = (
        "cutoff_time times must be datetime type: try casting via "
        "pd\\.to_datetime\\(\\)"
    )
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix(
            [property_feature],
            es,
            instance_ids=range(17),
            cutoff_time=17,
        )

    error_text = "cutoff_time must be a single value or DataFrame"
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix(
            [property_feature],
            es,
            instance_ids=range(17),
            cutoff_time=times,
        )

    cutoff_times_dup = pd.DataFrame(
        {
            "time": [datetime(2018, 3, 1), datetime(2018, 3, 1)],
            es["log"].ww.index: [1, 1],
        },
    )

    error_text = "Duplicated rows in cutoff time dataframe."
    with pytest.raises(AssertionError, match=error_text):
        feature_matrix = calculate_feature_matrix(
            [property_feature],
            entityset=es,
            cutoff_time=cutoff_times_dup,
        )

    cutoff_reordered = cutoff_time.iloc[[-1, 10, 1]]  # 3 ids not ordered by cutoff time
    feature_matrix = calculate_feature_matrix(
        [property_feature],
        es,
        cutoff_time=cutoff_reordered,
        verbose=True,
    )

    assert all(feature_matrix.index == cutoff_reordered["id"].values)
    # fails with Dask and Spark entitysets, cutoff time not reordered; cannot verify out of order
    # - can't tell if wrong/different all are false so can't check positional


def test_cfm_warns_dask_cutoff_time(es):
    dd = pytest.importorskip("dask.dataframe", reason="Dask not installed, skipping")
    times = list(
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)]
        + [datetime(2011, 4, 9, 10, 40, 0)]
        + [datetime(2011, 4, 10, 10, 40, i) for i in range(2)]
        + [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)]
        + [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)],
    )
    instances = range(17)
    cutoff_time = pd.DataFrame({"time": times, es["log"].ww.index: instances})
    cutoff_time = dd.from_pandas(cutoff_time, npartitions=4)

    property_feature = Feature(es["log"].ww["value"]) > 10

    match = (
        "cutoff_time should be a Pandas DataFrame: "
        "computing cutoff_time, this may take a while"
    )
    with pytest.warns(UserWarning, match=match):
        calculate_feature_matrix([property_feature], es, cutoff_time=cutoff_time)


def test_cfm_compose(es, lt):
    property_feature = Feature(es["log"].ww["value"]) > 10

    feature_matrix = calculate_feature_matrix(
        [property_feature],
        es,
        cutoff_time=lt,
        verbose=True,
    )
    feature_matrix = to_pandas(feature_matrix, index="id", sort_index=True)

    assert (
        feature_matrix[property_feature.get_name()] == feature_matrix["label_func"]
    ).values.all()


def test_cfm_compose_approximate(es, lt):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("dask does not support approximate")

    property_feature = Feature(es["log"].ww["value"]) > 10

    feature_matrix = calculate_feature_matrix(
        [property_feature],
        es,
        cutoff_time=lt,
        approximate="1s",
        verbose=True,
    )
    assert type(feature_matrix) == pd.core.frame.DataFrame
    feature_matrix = to_pandas(feature_matrix, index="id", sort_index=True)

    assert (
        feature_matrix[property_feature.get_name()] == feature_matrix["label_func"]
    ).values.all()


def test_cfm_dask_compose(dask_es, lt):
    property_feature = Feature(dask_es["log"].ww["value"]) > 10

    feature_matrix = calculate_feature_matrix(
        [property_feature],
        dask_es,
        cutoff_time=lt,
        verbose=True,
    )
    feature_matrix = feature_matrix.compute()

    assert (
        feature_matrix[property_feature.get_name()] == feature_matrix["label_func"]
    ).values.all()


# tests approximate, skip for dask/spark
def test_cfm_approximate_correct_ordering():
    trips = {
        "trip_id": [i for i in range(1000)],
        "flight_time": [datetime(1998, 4, 2) for i in range(350)]
        + [datetime(1997, 4, 3) for i in range(650)],
        "flight_id": [randint(1, 25) for i in range(1000)],
        "trip_duration": [randint(1, 999) for i in range(1000)],
    }
    df = pd.DataFrame.from_dict(trips)
    es = EntitySet("flights")
    es.add_dataframe(
        dataframe_name="trips",
        dataframe=df,
        index="trip_id",
        time_index="flight_time",
    )
    es.normalize_dataframe(
        base_dataframe_name="trips",
        new_dataframe_name="flights",
        index="flight_id",
        make_time_index=True,
    )
    features = dfs(entityset=es, target_dataframe_name="trips", features_only=True)
    flight_features = [
        feature
        for feature in features
        if isinstance(feature, DirectFeature)
        and isinstance(feature.base_features[0], AggregationFeature)
    ]
    property_feature = IdentityFeature(es["trips"].ww["trip_id"])

    cutoff_time = pd.DataFrame.from_dict(
        {"instance_id": df["trip_id"], "time": df["flight_time"]},
    )
    time_feature = IdentityFeature(es["trips"].ww["flight_time"])
    feature_matrix = calculate_feature_matrix(
        flight_features + [property_feature, time_feature],
        es,
        cutoff_time_in_index=True,
        cutoff_time=cutoff_time,
    )
    feature_matrix.index.names = ["instance", "time"]
    assert np.all(
        feature_matrix.reset_index("time").reset_index()[["instance", "time"]].values
        == feature_matrix[["trip_id", "flight_time"]].values,
    )
    feature_matrix_2 = calculate_feature_matrix(
        flight_features + [property_feature, time_feature],
        es,
        cutoff_time=cutoff_time,
        cutoff_time_in_index=True,
        approximate=Timedelta(2, "d"),
    )
    feature_matrix_2.index.names = ["instance", "time"]
    assert np.all(
        feature_matrix_2.reset_index("time").reset_index()[["instance", "time"]].values
        == feature_matrix_2[["trip_id", "flight_time"]].values,
    )
    for column in feature_matrix:
        for x, y in zip(feature_matrix[column], feature_matrix_2[column]):
            assert (pd.isnull(x) and pd.isnull(y)) or (x == y)


# uses approximate, skip for dask/spark entitysets
def test_cfm_no_cutoff_time_index(pd_es):
    agg_feat = Feature(
        pd_es["log"].ww["id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    agg_feat4 = Feature(agg_feat, parent_dataframe_name="customers", primitive=Sum)
    dfeat = DirectFeature(agg_feat4, "sessions")
    cutoff_time = pd.DataFrame(
        {
            "time": [datetime(2013, 4, 9, 10, 31, 19), datetime(2013, 4, 9, 11, 0, 0)],
            "instance_id": [0, 2],
        },
    )
    feature_matrix = calculate_feature_matrix(
        [dfeat, agg_feat],
        pd_es,
        cutoff_time_in_index=False,
        approximate=Timedelta(12, "s"),
        cutoff_time=cutoff_time,
    )
    assert feature_matrix.index.name == "id"
    assert feature_matrix.index.tolist() == [0, 2]
    assert feature_matrix[dfeat.get_name()].tolist() == [10, 10]
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]

    cutoff_time = pd.DataFrame(
        {
            "time": [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)],
            "instance_id": [0, 2],
        },
    )
    feature_matrix_2 = calculate_feature_matrix(
        [dfeat, agg_feat],
        pd_es,
        cutoff_time_in_index=False,
        approximate=Timedelta(10, "s"),
        cutoff_time=cutoff_time,
    )
    assert feature_matrix_2.index.name == "id"
    assert feature_matrix_2.index.tolist() == [0, 2]
    assert feature_matrix_2[dfeat.get_name()].tolist() == [7, 10]
    assert feature_matrix_2[agg_feat.get_name()].tolist() == [5, 1]


# TODO: fails with dask entitysets
# TODO: fails with spark entitysets
def test_cfm_duplicated_index_in_cutoff_time(es):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Distributed results not ordered, missing duplicates")
    times = [
        datetime(2011, 4, 1),
        datetime(2011, 5, 1),
        datetime(2011, 4, 1),
        datetime(2011, 5, 1),
    ]

    instances = [1, 1, 2, 2]
    property_feature = Feature(es["log"].ww["value"]) > 10
    cutoff_time = pd.DataFrame({"id": instances, "time": times}, index=[1, 1, 1, 1])

    feature_matrix = calculate_feature_matrix(
        [property_feature],
        es,
        cutoff_time=cutoff_time,
        chunk_size=1,
    )
    assert feature_matrix.shape[0] == cutoff_time.shape[0]


# TODO: fails with Dask, Spark
def test_saveprogress(es, tmp_path):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("saveprogress fails with distributed entitysets")
    times = list(
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)]
        + [datetime(2011, 4, 9, 10, 40, 0)]
        + [datetime(2011, 4, 10, 10, 40, i) for i in range(2)]
        + [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)]
        + [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)],
    )
    cutoff_time = pd.DataFrame({"time": times, "instance_id": range(17)})
    property_feature = Feature(es["log"].ww["value"]) > 10
    save_progress = str(tmp_path)
    fm_save = calculate_feature_matrix(
        [property_feature],
        es,
        cutoff_time=cutoff_time,
        save_progress=save_progress,
    )
    _, _, files = next(os.walk(save_progress))
    files = [os.path.join(save_progress, file) for file in files]
    # there are 17 datetime files created above
    assert len(files) == 17
    list_df = []
    for file_ in files:
        df = pd.read_csv(file_, index_col="id", header=0)
        list_df.append(df)
    merged_df = pd.concat(list_df)
    merged_df.set_index(pd.DatetimeIndex(times), inplace=True, append=True)
    fm_no_save = calculate_feature_matrix(
        [property_feature],
        es,
        cutoff_time=cutoff_time,
    )
    assert np.all((merged_df.sort_index().values) == (fm_save.sort_index().values))
    assert np.all((fm_no_save.sort_index().values) == (fm_save.sort_index().values))
    assert np.all((fm_no_save.sort_index().values) == (merged_df.sort_index().values))
    shutil.rmtree(save_progress)


def test_cutoff_time_correctly(es):
    property_feature = Feature(
        es["log"].ww["id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    times = [datetime(2011, 4, 10), datetime(2011, 4, 11), datetime(2011, 4, 7)]
    cutoff_time = pd.DataFrame({"time": times, "instance_id": [0, 1, 2]})
    feature_matrix = calculate_feature_matrix(
        [property_feature],
        es,
        cutoff_time=cutoff_time,
    )
    feature_matrix = to_pandas(feature_matrix, index="id", sort_index=True)
    labels = [10, 5, 0]
    assert (feature_matrix[property_feature.get_name()] == labels).values.all()


def test_cutoff_time_binning():
    cutoff_time = pd.DataFrame(
        {
            "time": [
                datetime(2011, 4, 9, 12, 31),
                datetime(2011, 4, 10, 11),
                datetime(2011, 4, 10, 13, 10, 1),
            ],
            "instance_id": [1, 2, 3],
        },
    )
    cutoff_time.ww.init()
    binned_cutoff_times = bin_cutoff_times(cutoff_time, Timedelta(4, "h"))
    labels = [
        datetime(2011, 4, 9, 12),
        datetime(2011, 4, 10, 8),
        datetime(2011, 4, 10, 12),
    ]
    for i in binned_cutoff_times.index:
        assert binned_cutoff_times["time"][i] == labels[i]

    binned_cutoff_times = bin_cutoff_times(cutoff_time, Timedelta(25, "h"))
    labels = [
        datetime(2011, 4, 8, 22),
        datetime(2011, 4, 9, 23),
        datetime(2011, 4, 9, 23),
    ]
    for i in binned_cutoff_times.index:
        assert binned_cutoff_times["time"][i] == labels[i]

    error_text = "Unit is relative"
    with pytest.raises(ValueError, match=error_text):
        binned_cutoff_times = bin_cutoff_times(cutoff_time, Timedelta(1, "mo"))


def test_training_window_fails_dask(dask_es):
    property_feature = Feature(
        dask_es["log"].ww["id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )

    error_text = "Using training_window is not supported with Dask dataframes"
    with pytest.raises(ValueError, match=error_text):
        calculate_feature_matrix([property_feature], dask_es, training_window="2 hours")


def test_cutoff_time_columns_order(es):
    property_feature = Feature(
        es["log"].ww["id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    times = [datetime(2011, 4, 10), datetime(2011, 4, 11), datetime(2011, 4, 7)]
    id_col_names = ["instance_id", es["customers"].ww.index]
    time_col_names = ["time", es["customers"].ww.time_index]
    for id_col in id_col_names:
        for time_col in time_col_names:
            cutoff_time = pd.DataFrame(
                {
                    "dummy_col_1": [1, 2, 3],
                    id_col: [0, 1, 2],
                    "dummy_col_2": [True, False, False],
                    time_col: times,
                },
            )
            feature_matrix = calculate_feature_matrix(
                [property_feature],
                es,
                cutoff_time=cutoff_time,
            )

            labels = [10, 5, 0]
            feature_matrix = to_pandas(feature_matrix, index="id", sort_index=True)
            assert (feature_matrix[property_feature.get_name()] == labels).values.all()


def test_cutoff_time_df_redundant_column_names(es):
    property_feature = Feature(
        es["log"].ww["id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    times = [datetime(2011, 4, 10), datetime(2011, 4, 11), datetime(2011, 4, 7)]

    cutoff_time = pd.DataFrame(
        {
            es["customers"].ww.index: [0, 1, 2],
            "instance_id": [0, 1, 2],
            "dummy_col": [True, False, False],
            "time": times,
        },
    )
    err_msg = (
        'Cutoff time DataFrame cannot contain both a column named "instance_id" and a column'
        " with the same name as the target dataframe index"
    )
    with pytest.raises(AttributeError, match=err_msg):
        calculate_feature_matrix([property_feature], es, cutoff_time=cutoff_time)

    cutoff_time = pd.DataFrame(
        {
            es["customers"].ww.time_index: [0, 1, 2],
            "instance_id": [0, 1, 2],
            "dummy_col": [True, False, False],
            "time": times,
        },
    )
    err_msg = (
        'Cutoff time DataFrame cannot contain both a column named "time" and a column'
        " with the same name as the target dataframe time index"
    )
    with pytest.raises(AttributeError, match=err_msg):
        calculate_feature_matrix([property_feature], es, cutoff_time=cutoff_time)


def test_training_window(pd_es):
    property_feature = Feature(
        pd_es["log"].ww["id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    top_level_agg = Feature(
        pd_es["customers"].ww["id"],
        parent_dataframe_name="régions",
        primitive=Count,
    )

    # make sure features that have a direct to a higher level agg
    # so we have multiple "filter eids" in get_pandas_data_slice,
    # and we go through the loop to pull data with a training_window param more than once
    dagg = DirectFeature(top_level_agg, "customers")

    # for now, warns if last_time_index not present
    times = [
        datetime(2011, 4, 9, 12, 31),
        datetime(2011, 4, 10, 11),
        datetime(2011, 4, 10, 13, 10),
    ]
    cutoff_time = pd.DataFrame({"time": times, "instance_id": [0, 1, 2]})
    warn_text = (
        "Using training_window but last_time_index is not set for dataframe customers"
    )
    with pytest.warns(UserWarning, match=warn_text):
        feature_matrix = calculate_feature_matrix(
            [property_feature, dagg],
            pd_es,
            cutoff_time=cutoff_time,
            training_window="2 hours",
        )

    pd_es.add_last_time_indexes()

    error_text = "Training window cannot be in observations"
    with pytest.raises(AssertionError, match=error_text):
        feature_matrix = calculate_feature_matrix(
            [property_feature],
            pd_es,
            cutoff_time=cutoff_time,
            training_window=Timedelta(2, "observations"),
        )

    # Case1. include_cutoff_time = True
    feature_matrix = calculate_feature_matrix(
        [property_feature, dagg],
        pd_es,
        cutoff_time=cutoff_time,
        training_window="2 hours",
        include_cutoff_time=True,
    )
    prop_values = [4, 5, 1]
    dagg_values = [3, 2, 1]
    assert (feature_matrix[property_feature.get_name()] == prop_values).values.all()
    assert (feature_matrix[dagg.get_name()] == dagg_values).values.all()

    # Case2. include_cutoff_time = False
    feature_matrix = calculate_feature_matrix(
        [property_feature, dagg],
        pd_es,
        cutoff_time=cutoff_time,
        training_window="2 hours",
        include_cutoff_time=False,
    )
    prop_values = [5, 5, 2]
    dagg_values = [3, 2, 1]

    assert (feature_matrix[property_feature.get_name()] == prop_values).values.all()
    assert (feature_matrix[dagg.get_name()] == dagg_values).values.all()

    # Case3. include_cutoff_time = False with single cutoff time value
    feature_matrix = calculate_feature_matrix(
        [property_feature, dagg],
        pd_es,
        cutoff_time=pd.to_datetime("2011-04-09 10:40:00"),
        training_window="9 minutes",
        include_cutoff_time=False,
    )
    prop_values = [0, 4, 0]
    dagg_values = [3, 3, 3]
    assert (feature_matrix[property_feature.get_name()] == prop_values).values.all()
    assert (feature_matrix[dagg.get_name()] == dagg_values).values.all()

    # Case4. include_cutoff_time = True with single cutoff time value
    feature_matrix = calculate_feature_matrix(
        [property_feature, dagg],
        pd_es,
        cutoff_time=pd.to_datetime("2011-04-10 10:40:00"),
        training_window="2 days",
        include_cutoff_time=True,
    )
    prop_values = [0, 10, 1]
    dagg_values = [3, 3, 3]
    assert (feature_matrix[property_feature.get_name()] == prop_values).values.all()
    assert (feature_matrix[dagg.get_name()] == dagg_values).values.all()


def test_training_window_overlap(pd_es):
    pd_es.add_last_time_indexes()

    count_log = Feature(
        Feature(pd_es["log"].ww["id"]),
        parent_dataframe_name="customers",
        primitive=Count,
    )

    cutoff_time = pd.DataFrame(
        {
            "id": [0, 0],
            "time": ["2011-04-09 10:30:00", "2011-04-09 10:40:00"],
        },
    ).astype({"time": "datetime64[ns]"})

    # Case1. include_cutoff_time = True
    actual = calculate_feature_matrix(
        features=[count_log],
        entityset=pd_es,
        cutoff_time=cutoff_time,
        cutoff_time_in_index=True,
        training_window="10 minutes",
        include_cutoff_time=True,
    )
    actual = actual["COUNT(log)"]
    np.testing.assert_array_equal(actual.values, [1, 9])

    # Case2. include_cutoff_time = False
    actual = calculate_feature_matrix(
        features=[count_log],
        entityset=pd_es,
        cutoff_time=cutoff_time,
        cutoff_time_in_index=True,
        training_window="10 minutes",
        include_cutoff_time=False,
    )
    actual = actual["COUNT(log)"]
    np.testing.assert_array_equal(actual.values, [0, 9])


def test_include_cutoff_time_without_training_window(es):
    es.add_last_time_indexes()

    count_log = Feature(
        base=Feature(es["log"].ww["id"]),
        parent_dataframe_name="customers",
        primitive=Count,
    )

    cutoff_time = pd.DataFrame(
        {
            "id": [0, 0],
            "time": ["2011-04-09 10:30:00", "2011-04-09 10:31:00"],
        },
    ).astype({"time": "datetime64[ns]"})

    # Case1. include_cutoff_time = True
    actual = calculate_feature_matrix(
        features=[count_log],
        entityset=es,
        cutoff_time=cutoff_time,
        cutoff_time_in_index=True,
        include_cutoff_time=True,
    )
    actual = to_pandas(actual)["COUNT(log)"]
    np.testing.assert_array_equal(actual.values, [1, 6])

    # Case2. include_cutoff_time = False
    actual = calculate_feature_matrix(
        features=[count_log],
        entityset=es,
        cutoff_time=cutoff_time,
        cutoff_time_in_index=True,
        include_cutoff_time=False,
    )
    actual = to_pandas(actual)["COUNT(log)"]
    np.testing.assert_array_equal(actual.values, [0, 5])

    # Case3. include_cutoff_time = True with single cutoff time value
    actual = calculate_feature_matrix(
        features=[count_log],
        entityset=es,
        cutoff_time=pd.to_datetime("2011-04-09 10:31:00"),
        instance_ids=[0],
        cutoff_time_in_index=True,
        include_cutoff_time=True,
    )
    actual = to_pandas(actual)["COUNT(log)"]
    np.testing.assert_array_equal(actual.values, [6])

    # Case4. include_cutoff_time = False with single cutoff time value
    actual = calculate_feature_matrix(
        features=[count_log],
        entityset=es,
        cutoff_time=pd.to_datetime("2011-04-09 10:31:00"),
        instance_ids=[0],
        cutoff_time_in_index=True,
        include_cutoff_time=False,
    )
    actual = to_pandas(actual)["COUNT(log)"]
    np.testing.assert_array_equal(actual.values, [5])


def test_approximate_dfeat_of_agg_on_target_include_cutoff_time(pd_es):
    agg_feat = Feature(
        pd_es["log"].ww["id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    agg_feat2 = Feature(agg_feat, parent_dataframe_name="customers", primitive=Sum)
    dfeat = DirectFeature(agg_feat2, "sessions")

    cutoff_time = pd.DataFrame(
        {"time": [datetime(2011, 4, 9, 10, 31, 19)], "instance_id": [0]},
    )
    feature_matrix = calculate_feature_matrix(
        [dfeat, agg_feat2, agg_feat],
        pd_es,
        approximate=Timedelta(20, "s"),
        cutoff_time=cutoff_time,
        include_cutoff_time=False,
    )

    # binned cutoff_time will be datetime(2011, 4, 9, 10, 31, 0) and
    # log event 5 at datetime(2011, 4, 9, 10, 31, 0) will be
    # excluded due to approximate cutoff time point
    assert feature_matrix[dfeat.get_name()].tolist() == [5]
    assert feature_matrix[agg_feat.get_name()].tolist() == [5]

    feature_matrix = calculate_feature_matrix(
        [dfeat, agg_feat],
        pd_es,
        approximate=Timedelta(20, "s"),
        cutoff_time=cutoff_time,
        include_cutoff_time=True,
    )

    # binned cutoff_time will be datetime(2011, 4, 9, 10, 31, 0) and
    # log event 5 at datetime(2011, 4, 9, 10, 31, 0) will be
    # included due to approximate cutoff time point
    assert feature_matrix[dfeat.get_name()].tolist() == [6]
    assert feature_matrix[agg_feat.get_name()].tolist() == [5]


def test_training_window_recent_time_index(pd_es):
    # customer with no sessions
    row = {
        "id": [3],
        "age": [73],
        "région_id": ["United States"],
        "cohort": [1],
        "cancel_reason": ["Lost interest"],
        "loves_ice_cream": [True],
        "favorite_quote": ["Don't look back. Something might be gaining on you."],
        "signup_date": [datetime(2011, 4, 10)],
        "upgrade_date": [datetime(2011, 4, 12)],
        "cancel_date": [datetime(2011, 5, 13)],
        "birthday": [datetime(1938, 2, 1)],
        "engagement_level": [2],
    }
    to_add_df = pd.DataFrame(row)
    to_add_df.index = range(3, 4)

    # have to convert category to int in order to concat
    old_df = pd_es["customers"]
    old_df.index = old_df.index.astype("int")
    old_df["id"] = old_df["id"].astype(int)

    df = pd.concat([old_df, to_add_df], sort=True)

    # convert back after
    df.index = df.index.astype("category")
    df["id"] = df["id"].astype("category")

    pd_es.replace_dataframe(
        dataframe_name="customers",
        df=df,
        recalculate_last_time_indexes=False,
    )
    pd_es.add_last_time_indexes()

    property_feature = Feature(
        pd_es["log"].ww["id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    top_level_agg = Feature(
        pd_es["customers"].ww["id"],
        parent_dataframe_name="régions",
        primitive=Count,
    )
    dagg = DirectFeature(top_level_agg, "customers")
    instance_ids = [0, 1, 2, 3]
    times = [
        datetime(2011, 4, 9, 12, 31),
        datetime(2011, 4, 10, 11),
        datetime(2011, 4, 10, 13, 10, 1),
        datetime(2011, 4, 10, 1, 59, 59),
    ]
    cutoff_time = pd.DataFrame({"time": times, "instance_id": instance_ids})

    # Case1. include_cutoff_time = True
    feature_matrix = calculate_feature_matrix(
        [property_feature, dagg],
        pd_es,
        cutoff_time=cutoff_time,
        training_window="2 hours",
        include_cutoff_time=True,
    )
    prop_values = [4, 5, 1, 0]
    assert (feature_matrix[property_feature.get_name()] == prop_values).values.all()

    dagg_values = [3, 2, 1, 3]
    feature_matrix.sort_index(inplace=True)
    assert (feature_matrix[dagg.get_name()] == dagg_values).values.all()

    # Case2. include_cutoff_time = False
    feature_matrix = calculate_feature_matrix(
        [property_feature, dagg],
        pd_es,
        cutoff_time=cutoff_time,
        training_window="2 hours",
        include_cutoff_time=False,
    )
    prop_values = [5, 5, 1, 0]
    assert (feature_matrix[property_feature.get_name()] == prop_values).values.all()

    dagg_values = [3, 2, 1, 3]
    feature_matrix.sort_index(inplace=True)
    assert (feature_matrix[dagg.get_name()] == dagg_values).values.all()


# TODO: add test to fail w/ spark
def test_approximate_fails_dask(dask_es):
    agg_feat = Feature(
        dask_es["log"].ww["id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    error_text = "Using approximate is not supported with Dask dataframes"
    with pytest.raises(ValueError, match=error_text):
        calculate_feature_matrix([agg_feat], dask_es, approximate=Timedelta(1, "week"))


def test_approximate_multiple_instances_per_cutoff_time(pd_es):
    agg_feat = Feature(
        pd_es["log"].ww["id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    agg_feat2 = Feature(agg_feat, parent_dataframe_name="customers", primitive=Sum)
    dfeat = DirectFeature(agg_feat2, "sessions")
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({"time": times, "instance_id": [0, 2]})
    feature_matrix = calculate_feature_matrix(
        [dfeat, agg_feat],
        pd_es,
        approximate=Timedelta(1, "week"),
        cutoff_time=cutoff_time,
    )
    assert feature_matrix.shape[0] == 2
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]


def test_approximate_with_multiple_paths(pd_diamond_es):
    pd_es = pd_diamond_es
    path = backward_path(pd_es, ["regions", "customers", "transactions"])
    agg_feat = AggregationFeature(
        Feature(pd_es["transactions"].ww["id"]),
        parent_dataframe_name="regions",
        relationship_path=path,
        primitive=Count,
    )
    dfeat = DirectFeature(agg_feat, "customers")
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({"time": times, "instance_id": [0, 2]})
    feature_matrix = calculate_feature_matrix(
        [dfeat],
        pd_es,
        approximate=Timedelta(1, "week"),
        cutoff_time=cutoff_time,
    )
    assert feature_matrix[dfeat.get_name()].tolist() == [6, 2]


def test_approximate_dfeat_of_agg_on_target(pd_es):
    agg_feat = Feature(
        pd_es["log"].ww["id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    agg_feat2 = Feature(agg_feat, parent_dataframe_name="customers", primitive=Sum)
    dfeat = DirectFeature(agg_feat2, "sessions")
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({"time": times, "instance_id": [0, 2]})
    feature_matrix = calculate_feature_matrix(
        [dfeat, agg_feat],
        pd_es,
        approximate=Timedelta(10, "s"),
        cutoff_time=cutoff_time,
    )
    assert feature_matrix[dfeat.get_name()].tolist() == [7, 10]
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]


def test_approximate_dfeat_of_need_all_values(pd_es):
    p = Feature(pd_es["log"].ww["value"], primitive=Percentile)
    agg_feat = Feature(p, parent_dataframe_name="sessions", primitive=Sum)
    agg_feat2 = Feature(agg_feat, parent_dataframe_name="customers", primitive=Sum)
    dfeat = DirectFeature(agg_feat2, "sessions")
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({"time": times, "instance_id": [0, 2]})
    feature_matrix = calculate_feature_matrix(
        [dfeat, agg_feat],
        pd_es,
        approximate=Timedelta(10, "s"),
        cutoff_time_in_index=True,
        cutoff_time=cutoff_time,
    )
    log_df = pd_es["log"]
    instances = [0, 2]
    cutoffs = [pd.Timestamp("2011-04-09 10:31:19"), pd.Timestamp("2011-04-09 11:00:00")]
    approxes = [
        pd.Timestamp("2011-04-09 10:31:10"),
        pd.Timestamp("2011-04-09 11:00:00"),
    ]
    true_vals = []
    true_vals_approx = []
    for instance, cutoff, approx in zip(instances, cutoffs, approxes):
        log_data_cutoff = log_df[log_df["datetime"] < cutoff]
        log_data_cutoff["percentile"] = log_data_cutoff["value"].rank(pct=True)
        true_agg = (
            log_data_cutoff.loc[log_data_cutoff["session_id"] == instance, "percentile"]
            .fillna(0)
            .sum()
        )
        true_vals.append(round(true_agg, 3))

        log_data_approx = log_df[log_df["datetime"] < approx]
        log_data_approx["percentile"] = log_data_approx["value"].rank(pct=True)
        true_agg_approx = (
            log_data_approx.loc[
                log_data_approx["session_id"].isin([0, 1, 2]),
                "percentile",
            ]
            .fillna(0)
            .sum()
        )
        true_vals_approx.append(round(true_agg_approx, 3))
    lapprox = [round(x, 3) for x in feature_matrix[dfeat.get_name()].tolist()]
    test_list = [round(x, 3) for x in feature_matrix[agg_feat.get_name()].tolist()]
    assert lapprox == true_vals_approx
    assert test_list == true_vals


def test_uses_full_dataframe_feat_of_approximate(pd_es):
    agg_feat = Feature(
        pd_es["log"].ww["value"],
        parent_dataframe_name="sessions",
        primitive=Sum,
    )
    agg_feat2 = Feature(agg_feat, parent_dataframe_name="customers", primitive=Sum)
    agg_feat3 = Feature(agg_feat, parent_dataframe_name="customers", primitive=Max)
    dfeat = DirectFeature(agg_feat2, "sessions")
    dfeat2 = DirectFeature(agg_feat3, "sessions")
    p = Feature(dfeat, primitive=Percentile)
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({"time": times, "instance_id": [0, 2]})
    # only dfeat2 should be approximated
    # because Percentile needs all values

    feature_matrix_only_dfeat2 = calculate_feature_matrix(
        [dfeat2],
        pd_es,
        approximate=Timedelta(10, "s"),
        cutoff_time_in_index=True,
        cutoff_time=cutoff_time,
    )
    assert feature_matrix_only_dfeat2[dfeat2.get_name()].tolist() == [50, 50]

    feature_matrix_approx = calculate_feature_matrix(
        [p, dfeat, dfeat2, agg_feat],
        pd_es,
        approximate=Timedelta(10, "s"),
        cutoff_time_in_index=True,
        cutoff_time=cutoff_time,
    )
    assert (
        feature_matrix_only_dfeat2[dfeat2.get_name()].tolist()
        == feature_matrix_approx[dfeat2.get_name()].tolist()
    )

    feature_matrix_small_approx = calculate_feature_matrix(
        [p, dfeat, dfeat2, agg_feat],
        pd_es,
        approximate=Timedelta(10, "ms"),
        cutoff_time_in_index=True,
        cutoff_time=cutoff_time,
    )

    feature_matrix_no_approx = calculate_feature_matrix(
        [p, dfeat, dfeat2, agg_feat],
        pd_es,
        cutoff_time_in_index=True,
        cutoff_time=cutoff_time,
    )
    for f in [p, dfeat, agg_feat]:
        for fm1, fm2 in combinations(
            [
                feature_matrix_approx,
                feature_matrix_small_approx,
                feature_matrix_no_approx,
            ],
            2,
        ):
            assert fm1[f.get_name()].tolist() == fm2[f.get_name()].tolist()


def test_approximate_dfeat_of_dfeat_of_agg_on_target(pd_es):
    agg_feat = Feature(
        pd_es["log"].ww["id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    agg_feat2 = Feature(agg_feat, parent_dataframe_name="customers", primitive=Sum)
    dfeat = DirectFeature(Feature(agg_feat2, "sessions"), "log")
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({"time": times, "instance_id": [0, 2]})
    feature_matrix = calculate_feature_matrix(
        [dfeat],
        pd_es,
        approximate=Timedelta(10, "s"),
        cutoff_time=cutoff_time,
    )
    assert feature_matrix[dfeat.get_name()].tolist() == [7, 10]


def test_empty_path_approximate_full(pd_es):
    pd_es["sessions"].ww["customer_id"] = pd.Series(
        [np.nan, np.nan, np.nan, 1, 1, 2],
        dtype="category",
    )
    # Need to reassign the `foreign_key` tag as the column reassignment above removes it
    pd_es["sessions"].ww.set_types(semantic_tags={"customer_id": "foreign_key"})
    agg_feat = Feature(
        pd_es["log"].ww["id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    agg_feat2 = Feature(agg_feat, parent_dataframe_name="customers", primitive=Sum)
    dfeat = DirectFeature(agg_feat2, "sessions")
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({"time": times, "instance_id": [0, 2]})
    feature_matrix = calculate_feature_matrix(
        [dfeat, agg_feat],
        pd_es,
        approximate=Timedelta(10, "s"),
        cutoff_time=cutoff_time,
    )
    vals1 = feature_matrix[dfeat.get_name()].tolist()

    assert vals1[0] == 0
    assert vals1[1] == 0
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]


def test_approx_base_feature_is_also_first_class_feature(pd_es):
    log_to_products = DirectFeature(Feature(pd_es["products"].ww["rating"]), "log")
    # This should still be computed properly
    agg_feat = Feature(log_to_products, parent_dataframe_name="sessions", primitive=Min)
    customer_agg_feat = Feature(
        agg_feat,
        parent_dataframe_name="customers",
        primitive=Sum,
    )
    # This is to be approximated
    sess_to_cust = DirectFeature(customer_agg_feat, "sessions")
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({"time": times, "instance_id": [0, 2]})
    feature_matrix = calculate_feature_matrix(
        [sess_to_cust, agg_feat],
        pd_es,
        approximate=Timedelta(10, "s"),
        cutoff_time=cutoff_time,
    )

    vals1 = feature_matrix[sess_to_cust.get_name()].tolist()
    assert vals1 == [8.5, 7]
    vals2 = feature_matrix[agg_feat.get_name()].tolist()
    assert vals2 == [4, 1.5]


def test_approximate_time_split_returns_the_same_result(pd_es):
    agg_feat = Feature(
        pd_es["log"].ww["id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    agg_feat2 = Feature(agg_feat, parent_dataframe_name="customers", primitive=Sum)
    dfeat = DirectFeature(agg_feat2, "sessions")

    cutoff_df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2011-04-09 10:07:30"),
                pd.Timestamp("2011-04-09 10:07:40"),
            ],
            "instance_id": [0, 0],
        },
    )

    feature_matrix_at_once = calculate_feature_matrix(
        [dfeat, agg_feat],
        pd_es,
        approximate=Timedelta(10, "s"),
        cutoff_time=cutoff_df,
    )
    divided_matrices = []
    separate_cutoff = [cutoff_df.iloc[0:1], cutoff_df.iloc[1:]]
    # Make sure indexes are different
    # Note that this step is unnecessary and done to showcase the issue here
    separate_cutoff[0].index = [0]
    separate_cutoff[1].index = [1]
    for ct in separate_cutoff:
        fm = calculate_feature_matrix(
            [dfeat, agg_feat],
            pd_es,
            approximate=Timedelta(10, "s"),
            cutoff_time=ct,
        )
        divided_matrices.append(fm)
    feature_matrix_from_split = pd.concat(divided_matrices)
    assert feature_matrix_from_split.shape == feature_matrix_at_once.shape
    for i1, i2 in zip(feature_matrix_at_once.index, feature_matrix_from_split.index):
        assert (pd.isnull(i1) and pd.isnull(i2)) or (i1 == i2)
    for c in feature_matrix_from_split:
        for i1, i2 in zip(feature_matrix_at_once[c], feature_matrix_from_split[c]):
            assert (pd.isnull(i1) and pd.isnull(i2)) or (i1 == i2)


def test_approximate_returns_correct_empty_default_values(pd_es):
    agg_feat = Feature(
        pd_es["log"].ww["id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    dfeat = DirectFeature(agg_feat, "sessions")

    cutoff_df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2011-04-08 11:00:00"),
                pd.Timestamp("2011-04-09 11:00:00"),
            ],
            "instance_id": [0, 0],
        },
    )

    fm = calculate_feature_matrix(
        [dfeat],
        pd_es,
        approximate=Timedelta(10, "s"),
        cutoff_time=cutoff_df,
    )
    assert fm[dfeat.get_name()].tolist() == [0, 10]


def test_approximate_child_aggs_handled_correctly(pd_es):
    agg_feat = Feature(
        pd_es["customers"].ww["id"],
        parent_dataframe_name="régions",
        primitive=Count,
    )
    dfeat = DirectFeature(agg_feat, "customers")
    agg_feat_2 = Feature(
        pd_es["log"].ww["value"],
        parent_dataframe_name="customers",
        primitive=Sum,
    )
    cutoff_df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2011-04-08 10:30:00"),
                pd.Timestamp("2011-04-09 10:30:06"),
            ],
            "instance_id": [0, 0],
        },
    )

    fm = calculate_feature_matrix(
        [dfeat],
        pd_es,
        approximate=Timedelta(10, "s"),
        cutoff_time=cutoff_df,
    )
    fm_2 = calculate_feature_matrix(
        [dfeat, agg_feat_2],
        pd_es,
        approximate=Timedelta(10, "s"),
        cutoff_time=cutoff_df,
    )
    assert fm[dfeat.get_name()].tolist() == [2, 3]
    assert fm_2[agg_feat_2.get_name()].tolist() == [0, 5]


def test_cutoff_time_naming(es):
    agg_feat = Feature(
        es["customers"].ww["id"],
        parent_dataframe_name="régions",
        primitive=Count,
    )
    dfeat = DirectFeature(agg_feat, "customers")
    cutoff_df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2011-04-08 10:30:00"),
                pd.Timestamp("2011-04-09 10:30:06"),
            ],
            "instance_id": [0, 0],
        },
    )
    cutoff_df_index_name = cutoff_df.rename(columns={"instance_id": "id"})
    cutoff_df_wrong_index_name = cutoff_df.rename(columns={"instance_id": "wrong_id"})
    cutoff_df_wrong_time_name = cutoff_df.rename(columns={"time": "cutoff_time"})

    fm1 = calculate_feature_matrix([dfeat], es, cutoff_time=cutoff_df)
    fm1 = to_pandas(fm1, index="id", sort_index=True)
    fm2 = calculate_feature_matrix([dfeat], es, cutoff_time=cutoff_df_index_name)
    fm2 = to_pandas(fm2, index="id", sort_index=True)
    assert all((fm1 == fm2.values).values)

    error_text = (
        "Cutoff time DataFrame must contain a column with either the same name"
        ' as the target dataframe index or a column named "instance_id"'
    )
    with pytest.raises(AttributeError, match=error_text):
        calculate_feature_matrix([dfeat], es, cutoff_time=cutoff_df_wrong_index_name)

    time_error_text = (
        "Cutoff time DataFrame must contain a column with either the same name"
        ' as the target dataframe time_index or a column named "time"'
    )
    with pytest.raises(AttributeError, match=time_error_text):
        calculate_feature_matrix([dfeat], es, cutoff_time=cutoff_df_wrong_time_name)


# TODO: order doesn't match, but output matches
def test_cutoff_time_extra_columns(es):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Distributed result not ordered")
    agg_feat = Feature(
        es["customers"].ww["id"],
        parent_dataframe_name="régions",
        primitive=Count,
    )
    dfeat = DirectFeature(agg_feat, "customers")

    cutoff_df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2011-04-09 10:30:06"),
                pd.Timestamp("2011-04-09 10:30:03"),
                pd.Timestamp("2011-04-08 10:30:00"),
            ],
            "instance_id": [0, 1, 0],
            "label": [True, True, False],
        },
        columns=["time", "instance_id", "label"],
    )
    fm = calculate_feature_matrix([dfeat], es, cutoff_time=cutoff_df)
    # check column was added to end of matrix
    assert "label" == fm.columns[-1]

    assert (fm["label"].values == cutoff_df["label"].values).all()


def test_cutoff_time_extra_columns_approximate(pd_es):
    agg_feat = Feature(
        pd_es["customers"].ww["id"],
        parent_dataframe_name="régions",
        primitive=Count,
    )
    dfeat = DirectFeature(agg_feat, "customers")

    cutoff_df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2011-04-09 10:30:06"),
                pd.Timestamp("2011-04-09 10:30:03"),
                pd.Timestamp("2011-04-08 10:30:00"),
            ],
            "instance_id": [0, 1, 0],
            "label": [True, True, False],
        },
        columns=["time", "instance_id", "label"],
    )
    fm = calculate_feature_matrix(
        [dfeat],
        pd_es,
        cutoff_time=cutoff_df,
        approximate="2 days",
    )
    # check column was added to end of matrix
    assert "label" in fm.columns

    assert (fm["label"].values == cutoff_df["label"].values).all()


def test_cutoff_time_extra_columns_same_name(es):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Distributed result not ordered")
    agg_feat = Feature(
        es["customers"].ww["id"],
        parent_dataframe_name="régions",
        primitive=Count,
    )
    dfeat = DirectFeature(agg_feat, "customers")

    cutoff_df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2011-04-09 10:30:06"),
                pd.Timestamp("2011-04-09 10:30:03"),
                pd.Timestamp("2011-04-08 10:30:00"),
            ],
            "instance_id": [0, 1, 0],
            "régions.COUNT(customers)": [False, False, True],
        },
        columns=["time", "instance_id", "régions.COUNT(customers)"],
    )
    fm = calculate_feature_matrix([dfeat], es, cutoff_time=cutoff_df)

    assert (
        fm["régions.COUNT(customers)"].values
        == cutoff_df["régions.COUNT(customers)"].values
    ).all()


def test_cutoff_time_extra_columns_same_name_approximate(pd_es):
    agg_feat = Feature(
        pd_es["customers"].ww["id"],
        parent_dataframe_name="régions",
        primitive=Count,
    )
    dfeat = DirectFeature(agg_feat, "customers")

    cutoff_df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2011-04-09 10:30:06"),
                pd.Timestamp("2011-04-09 10:30:03"),
                pd.Timestamp("2011-04-08 10:30:00"),
            ],
            "instance_id": [0, 1, 0],
            "régions.COUNT(customers)": [False, False, True],
        },
        columns=["time", "instance_id", "régions.COUNT(customers)"],
    )
    fm = calculate_feature_matrix(
        [dfeat],
        pd_es,
        cutoff_time=cutoff_df,
        approximate="2 days",
    )

    assert (
        fm["régions.COUNT(customers)"].values
        == cutoff_df["régions.COUNT(customers)"].values
    ).all()


def test_instances_after_cutoff_time_removed(es):
    property_feature = Feature(
        es["log"].ww["id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    cutoff_time = datetime(2011, 4, 8)
    fm = calculate_feature_matrix(
        [property_feature],
        es,
        cutoff_time=cutoff_time,
        cutoff_time_in_index=True,
    )
    fm = to_pandas(fm, index="id", sort_index=True)
    actual_ids = (
        [id for (id, _) in fm.index]
        if isinstance(fm.index, pd.MultiIndex)
        else fm.index
    )

    # Customer with id 1 should be removed
    assert set(actual_ids) == set([2, 0])


# TODO: Dask and Spark do not keep instance_id after cutoff
def test_instances_with_id_kept_after_cutoff(es):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail("Distributed result not ordered, missing extra instances")
    property_feature = Feature(
        es["log"].ww["id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    cutoff_time = datetime(2011, 4, 8)
    fm = calculate_feature_matrix(
        [property_feature],
        es,
        instance_ids=[0, 1, 2],
        cutoff_time=cutoff_time,
        cutoff_time_in_index=True,
    )

    # Customer #1 is after cutoff, but since it is included in instance_ids it
    # should be kept.
    actual_ids = (
        [id for (id, _) in fm.index]
        if isinstance(fm.index, pd.MultiIndex)
        else fm.index
    )
    assert set(actual_ids) == set([0, 1, 2])


# TODO: Fails with Dask
# TODO: Fails with Spark
def test_cfm_returns_original_time_indexes(es):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail(
            "Distributed result not ordered, indexes are lost due to not multiindexing",
        )
    agg_feat = Feature(
        es["customers"].ww["id"],
        parent_dataframe_name="régions",
        primitive=Count,
    )
    dfeat = DirectFeature(agg_feat, "customers")
    cutoff_df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2011-04-09 10:30:06"),
                pd.Timestamp("2011-04-09 10:30:03"),
                pd.Timestamp("2011-04-08 10:30:00"),
            ],
            "instance_id": [0, 1, 0],
        },
    )

    fm = calculate_feature_matrix(
        [dfeat],
        es,
        cutoff_time=cutoff_df,
        cutoff_time_in_index=True,
    )

    instance_level_vals = fm.index.get_level_values(0).values
    time_level_vals = fm.index.get_level_values(1).values

    assert (instance_level_vals == cutoff_df["instance_id"].values).all()
    assert (time_level_vals == cutoff_df["time"].values).all()


def test_cfm_returns_original_time_indexes_approximate(pd_es):
    agg_feat = Feature(
        pd_es["customers"].ww["id"],
        parent_dataframe_name="régions",
        primitive=Count,
    )
    dfeat = DirectFeature(agg_feat, "customers")
    agg_feat_2 = Feature(
        pd_es["sessions"].ww["id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    cutoff_df = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2011-04-09 10:30:06"),
                pd.Timestamp("2011-04-09 10:30:03"),
                pd.Timestamp("2011-04-08 10:30:00"),
            ],
            "instance_id": [0, 1, 0],
        },
    )
    # approximate, in different windows, no unapproximated aggs
    fm = calculate_feature_matrix(
        [dfeat],
        pd_es,
        cutoff_time=cutoff_df,
        cutoff_time_in_index=True,
        approximate="1 m",
    )
    instance_level_vals = fm.index.get_level_values(0).values
    time_level_vals = fm.index.get_level_values(1).values
    assert (instance_level_vals == cutoff_df["instance_id"].values).all()
    assert (time_level_vals == cutoff_df["time"].values).all()

    # approximate, in different windows, unapproximated aggs
    fm = calculate_feature_matrix(
        [dfeat, agg_feat_2],
        pd_es,
        cutoff_time=cutoff_df,
        cutoff_time_in_index=True,
        approximate="1 m",
    )
    instance_level_vals = fm.index.get_level_values(0).values
    time_level_vals = fm.index.get_level_values(1).values
    assert (instance_level_vals == cutoff_df["instance_id"].values).all()
    assert (time_level_vals == cutoff_df["time"].values).all()

    # approximate, in same window, no unapproximated aggs
    fm2 = calculate_feature_matrix(
        [dfeat],
        pd_es,
        cutoff_time=cutoff_df,
        cutoff_time_in_index=True,
        approximate="2 d",
    )
    instance_level_vals = fm2.index.get_level_values(0).values
    time_level_vals = fm2.index.get_level_values(1).values
    assert (instance_level_vals == cutoff_df["instance_id"].values).all()
    assert (time_level_vals == cutoff_df["time"].values).all()

    # approximate, in same window, unapproximated aggs
    fm3 = calculate_feature_matrix(
        [dfeat, agg_feat_2],
        pd_es,
        cutoff_time=cutoff_df,
        cutoff_time_in_index=True,
        approximate="2 d",
    )
    instance_level_vals = fm3.index.get_level_values(0).values
    time_level_vals = fm3.index.get_level_values(1).values
    assert (instance_level_vals == cutoff_df["instance_id"].values).all()
    assert (time_level_vals == cutoff_df["time"].values).all()


def test_dask_kwargs(pd_es, dask_cluster):
    times = (
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)]
        + [datetime(2011, 4, 9, 10, 40, 0)]
        + [datetime(2011, 4, 10, 10, 40, i) for i in range(2)]
        + [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)]
        + [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)]
    )
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [True] + [False] * 2
    cutoff_time = pd.DataFrame({"time": times, "instance_id": range(17)})
    property_feature = IdentityFeature(pd_es["log"].ww["value"]) > 10

    dkwargs = {"cluster": dask_cluster.scheduler.address}
    feature_matrix = calculate_feature_matrix(
        [property_feature],
        entityset=pd_es,
        cutoff_time=cutoff_time,
        verbose=True,
        chunk_size=0.13,
        dask_kwargs=dkwargs,
        approximate="1 hour",
    )

    assert (feature_matrix[property_feature.get_name()] == labels).values.all()


def test_dask_persisted_es(pd_es, capsys, dask_cluster):
    times = (
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)]
        + [datetime(2011, 4, 9, 10, 40, 0)]
        + [datetime(2011, 4, 10, 10, 40, i) for i in range(2)]
        + [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)]
        + [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)]
    )
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [True] + [False] * 2
    cutoff_time = pd.DataFrame({"time": times, "instance_id": range(17)})
    property_feature = IdentityFeature(pd_es["log"].ww["value"]) > 10

    dkwargs = {"cluster": dask_cluster.scheduler.address}
    feature_matrix = calculate_feature_matrix(
        [property_feature],
        entityset=pd_es,
        cutoff_time=cutoff_time,
        verbose=True,
        chunk_size=0.13,
        dask_kwargs=dkwargs,
        approximate="1 hour",
    )
    assert (feature_matrix[property_feature.get_name()] == labels).values.all()
    feature_matrix = calculate_feature_matrix(
        [property_feature],
        entityset=pd_es,
        cutoff_time=cutoff_time,
        verbose=True,
        chunk_size=0.13,
        dask_kwargs=dkwargs,
        approximate="1 hour",
    )
    captured = capsys.readouterr()
    assert "Using EntitySet persisted on the cluster as dataset " in captured[0]
    assert (feature_matrix[property_feature.get_name()] == labels).values.all()


class TestCreateClientAndCluster(object):
    def test_user_cluster_as_string(self, monkeypatch):
        monkeypatch.setattr(utils, "get_client_cluster", get_mock_client_cluster)
        # cluster in dask_kwargs case
        client, cluster = create_client_and_cluster(
            n_jobs=2,
            dask_kwargs={"cluster": "tcp://127.0.0.1:54321"},
            entityset_size=1,
        )
        assert cluster == "tcp://127.0.0.1:54321"

    def test_cluster_creation(self, monkeypatch):
        total_memory = psutil.virtual_memory().total
        monkeypatch.setattr(utils, "get_client_cluster", get_mock_client_cluster)
        try:
            cpus = len(psutil.Process().cpu_affinity())
        except AttributeError:  # pragma: no cover
            cpus = psutil.cpu_count()

        # jobs < tasks case
        client, cluster = create_client_and_cluster(
            n_jobs=2,
            dask_kwargs={},
            entityset_size=1,
        )
        num_workers = min(cpus, 2)
        memory_limit = int(total_memory / float(num_workers))
        assert cluster == (min(cpus, 2), 1, None, memory_limit)
        # jobs > tasks case
        match = r".*workers requested, but only .* workers created"
        with pytest.warns(UserWarning, match=match) as record:
            client, cluster = create_client_and_cluster(
                n_jobs=1000,
                dask_kwargs={"diagnostics_port": 8789},
                entityset_size=1,
            )
        assert len(record) == 1

        num_workers = cpus
        memory_limit = int(total_memory / float(num_workers))
        assert cluster == (num_workers, 1, 8789, memory_limit)

        # dask_kwargs sets memory limit
        client, cluster = create_client_and_cluster(
            n_jobs=2,
            dask_kwargs={"diagnostics_port": 8789, "memory_limit": 1000},
            entityset_size=1,
        )
        num_workers = min(cpus, 2)
        assert cluster == (num_workers, 1, 8789, 1000)

    def test_not_enough_memory(self, monkeypatch):
        total_memory = psutil.virtual_memory().total
        monkeypatch.setattr(utils, "get_client_cluster", get_mock_client_cluster)
        # errors if not enough memory for each worker to store the entityset
        with pytest.raises(ValueError, match=""):
            create_client_and_cluster(
                n_jobs=1,
                dask_kwargs={},
                entityset_size=total_memory * 2,
            )

        # does not error even if worker memory is less than 2x entityset size
        create_client_and_cluster(
            n_jobs=1,
            dask_kwargs={},
            entityset_size=total_memory * 0.75,
        )


@pytest.mark.skipif("not dd")
def test_parallel_failure_raises_correct_error(pd_es):
    times = (
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)]
        + [datetime(2011, 4, 9, 10, 40, 0)]
        + [datetime(2011, 4, 10, 10, 40, i) for i in range(2)]
        + [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)]
        + [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)]
    )
    cutoff_time = pd.DataFrame({"time": times, "instance_id": range(17)})
    property_feature = IdentityFeature(pd_es["log"].ww["value"]) > 10

    error_text = "Need at least one worker"
    with pytest.raises(AssertionError, match=error_text):
        calculate_feature_matrix(
            [property_feature],
            entityset=pd_es,
            cutoff_time=cutoff_time,
            verbose=True,
            chunk_size=0.13,
            n_jobs=0,
            approximate="1 hour",
        )


def test_warning_not_enough_chunks(
    pd_es,
    capsys,
    three_worker_dask_cluster,
):  # pragma: no cover
    property_feature = IdentityFeature(pd_es["log"].ww["value"]) > 10

    dkwargs = {"cluster": three_worker_dask_cluster.scheduler.address}
    calculate_feature_matrix(
        [property_feature],
        entityset=pd_es,
        chunk_size=0.5,
        verbose=True,
        dask_kwargs=dkwargs,
    )

    captured = capsys.readouterr()
    pattern = r"Fewer chunks \([0-9]+\), than workers \([0-9]+\) consider reducing the chunk size"
    assert re.search(pattern, captured.out) is not None


def test_n_jobs():
    try:
        cpus = len(psutil.Process().cpu_affinity())
    except AttributeError:  # pragma: no cover
        cpus = psutil.cpu_count()

    assert n_jobs_to_workers(1) == 1
    assert n_jobs_to_workers(-1) == cpus
    assert n_jobs_to_workers(cpus) == cpus
    assert n_jobs_to_workers((cpus + 1) * -1) == 1
    if cpus > 1:
        assert n_jobs_to_workers(-2) == cpus - 1

    error_text = "Need at least one worker"
    with pytest.raises(AssertionError, match=error_text):
        n_jobs_to_workers(0)


def test_parallel_cutoff_time_column_pass_through(pd_es, dask_cluster):
    times = (
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)]
        + [datetime(2011, 4, 9, 10, 40, 0)]
        + [datetime(2011, 4, 10, 10, 40, i) for i in range(2)]
        + [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)]
        + [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)]
    )
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [True] + [False] * 2
    cutoff_time = pd.DataFrame(
        {"time": times, "instance_id": range(17), "labels": labels},
    )
    property_feature = IdentityFeature(pd_es["log"].ww["value"]) > 10

    dkwargs = {"cluster": dask_cluster.scheduler.address}
    feature_matrix = calculate_feature_matrix(
        [property_feature],
        entityset=pd_es,
        cutoff_time=cutoff_time,
        verbose=True,
        dask_kwargs=dkwargs,
        approximate="1 hour",
    )

    assert (
        feature_matrix[property_feature.get_name()] == feature_matrix["labels"]
    ).values.all()


def test_integer_time_index(int_es):
    if int_es.dataframe_type != Library.PANDAS:
        pytest.xfail("Dask and Spark do not retain time column")
    times = list(range(8, 18)) + list(range(19, 26))
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [True] + [False] * 2
    cutoff_df = pd.DataFrame({"time": times, "instance_id": range(17)})
    property_feature = IdentityFeature(int_es["log"].ww["value"]) > 10

    feature_matrix = calculate_feature_matrix(
        [property_feature],
        int_es,
        cutoff_time=cutoff_df,
        cutoff_time_in_index=True,
    )

    time_level_vals = feature_matrix.index.get_level_values(1).values
    sorted_df = cutoff_df.sort_values(["time", "instance_id"], kind="mergesort")
    assert (time_level_vals == sorted_df["time"].values).all()
    assert (feature_matrix[property_feature.get_name()] == labels).values.all()


def test_integer_time_index_single_cutoff_value(int_es):
    if int_es.dataframe_type != Library.PANDAS:
        pytest.xfail("Dask and Spark do not retain time column")
    labels = [False] * 3 + [True] * 2 + [False] * 4
    property_feature = IdentityFeature(int_es["log"].ww["value"]) > 10

    cutoff_times = [16, pd.Series([16])[0], 16.0, pd.Series([16.0])[0]]
    for cutoff_time in cutoff_times:
        feature_matrix = calculate_feature_matrix(
            [property_feature],
            int_es,
            cutoff_time=cutoff_time,
            cutoff_time_in_index=True,
        )
        time_level_vals = feature_matrix.index.get_level_values(1).values
        assert (time_level_vals == [16] * 9).all()
        assert (feature_matrix[property_feature.get_name()] == labels).values.all()


def test_integer_time_index_datetime_cutoffs(int_es):
    times = [datetime.now()] * 17
    cutoff_df = pd.DataFrame({"time": times, "instance_id": range(17)})
    property_feature = IdentityFeature(int_es["log"].ww["value"]) > 10

    error_text = (
        "cutoff_time times must be numeric: try casting via pd\\.to_numeric\\(\\)"
    )
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix(
            [property_feature],
            int_es,
            cutoff_time=cutoff_df,
            cutoff_time_in_index=True,
        )


def test_integer_time_index_passes_extra_columns(int_es):
    times = list(range(8, 18)) + list(range(19, 23)) + [25, 24, 23]
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [False] * 2 + [True]
    instances = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 15, 14]
    cutoff_df = pd.DataFrame(
        {"time": times, "instance_id": instances, "labels": labels},
    )
    cutoff_df = cutoff_df[["time", "instance_id", "labels"]]
    property_feature = IdentityFeature(int_es["log"].ww["value"]) > 10

    fm = calculate_feature_matrix(
        [property_feature],
        int_es,
        cutoff_time=cutoff_df,
        cutoff_time_in_index=True,
    )
    fm = to_pandas(fm)
    assert (fm[property_feature.get_name()] == fm["labels"]).all()


def test_integer_time_index_mixed_cutoff(int_es):
    times_dt = list(range(8, 17)) + [datetime(2011, 1, 1), 19, 20, 21, 22, 25, 24, 23]
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [False] * 2 + [True]
    instances = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 15, 14]
    cutoff_df = pd.DataFrame(
        {"time": times_dt, "instance_id": instances, "labels": labels},
    )
    cutoff_df = cutoff_df[["time", "instance_id", "labels"]]
    property_feature = IdentityFeature(int_es["log"].ww["value"]) > 10

    error_text = "cutoff_time times must be.*try casting via.*"
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature], int_es, cutoff_time=cutoff_df)

    times_str = list(range(8, 17)) + ["foobar", 19, 20, 21, 22, 25, 24, 23]
    cutoff_df["time"] = times_str
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature], int_es, cutoff_time=cutoff_df)

    times_date_str = list(range(8, 17)) + ["2018-04-02", 19, 20, 21, 22, 25, 24, 23]
    cutoff_df["time"] = times_date_str
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature], int_es, cutoff_time=cutoff_df)

    times_int_str = [0, 1, 2, 3, 4, 5, "6", 7, 8, 9, 9, 10, 11, 12, 15, 14, 13]
    times_int_str = list(range(8, 17)) + ["17", 19, 20, 21, 22, 25, 24, 23]
    cutoff_df["time"] = times_int_str
    # calculate_feature_matrix should convert time column to ints successfully here
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature], int_es, cutoff_time=cutoff_df)


def test_datetime_index_mixed_cutoff(es):
    times = list(
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)]
        + [17]
        + [datetime(2011, 4, 10, 10, 40, i) for i in range(2)]
        + [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)]
        + [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)],
    )
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [False] * 2 + [True]
    instances = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 15, 14]
    cutoff_df = pd.DataFrame(
        {"time": times, "instance_id": instances, "labels": labels},
    )
    cutoff_df = cutoff_df[["time", "instance_id", "labels"]]
    property_feature = IdentityFeature(es["log"].ww["value"]) > 10

    error_text = "cutoff_time times must be.*try casting via.*"
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature], es, cutoff_time=cutoff_df)

    times[9] = "foobar"
    cutoff_df["time"] = times
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature], es, cutoff_time=cutoff_df)

    times[9] = "17"
    cutoff_df["time"] = times
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature], es, cutoff_time=cutoff_df)


# TODO: Dask version fails (feature matrix is empty)
# TODO: Spark version fails (spark groupby agg doesn't support custom functions)
def test_no_data_for_cutoff_time(mock_customer):
    if mock_customer.dataframe_type != Library.PANDAS:
        pytest.xfail(
            "Dask fails because returned feature matrix is empty; Spark doesn't support custom agg functions",
        )
    es = mock_customer
    cutoff_times = pd.DataFrame(
        {"customer_id": [4], "time": pd.Timestamp("2011-04-08 20:08:13")},
    )

    trans_per_session = Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    trans_per_customer = Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    max_count = Feature(
        trans_per_session,
        parent_dataframe_name="customers",
        primitive=Max,
    )
    features = [trans_per_customer, max_count]

    fm = calculate_feature_matrix(features, entityset=es, cutoff_time=cutoff_times)

    # due to default values for each primitive
    # count will be 0, but max will nan
    answer = pd.DataFrame(
        {
            trans_per_customer.get_name(): pd.Series([0], dtype="Int64"),
            max_count.get_name(): pd.Series([np.nan], dtype="float"),
        },
    )
    for column in fm.columns:
        pd.testing.assert_series_equal(
            fm[column],
            answer[column],
            check_index=False,
            check_names=False,
        )


# adding missing instances not supported in Dask or Spark
def test_instances_not_in_data(pd_es):
    last_instance = max(pd_es["log"].index.values)
    instances = list(range(last_instance + 1, last_instance + 11))
    identity_feature = IdentityFeature(pd_es["log"].ww["value"])
    property_feature = identity_feature > 10
    agg_feat = AggregationFeature(
        Feature(pd_es["log"].ww["value"]),
        parent_dataframe_name="sessions",
        primitive=Max,
    )
    direct_feature = DirectFeature(agg_feat, "log")
    features = [identity_feature, property_feature, direct_feature]
    fm = calculate_feature_matrix(features, entityset=pd_es, instance_ids=instances)
    assert all(fm.index.values == instances)
    for column in fm.columns:
        assert fm[column].isnull().all()

    fm = calculate_feature_matrix(
        features,
        entityset=pd_es,
        instance_ids=instances,
        approximate="730 days",
    )
    assert all(fm.index.values == instances)
    for column in fm.columns:
        assert fm[column].isnull().all()


def test_some_instances_not_in_data(pd_es):
    a_time = datetime(2011, 4, 10, 10, 41, 9)  # only valid data
    b_time = datetime(2011, 4, 10, 11, 10, 5)  # some missing data
    c_time = datetime(2011, 4, 10, 12, 0, 0)  # all missing data

    times = [a_time, b_time, a_time, a_time, b_time, b_time] + [c_time] * 4
    cutoff_time = pd.DataFrame({"instance_id": list(range(12, 22)), "time": times})
    identity_feature = IdentityFeature(pd_es["log"].ww["value"])
    property_feature = identity_feature > 10
    agg_feat = AggregationFeature(
        Feature(pd_es["log"].ww["value"]),
        parent_dataframe_name="sessions",
        primitive=Max,
    )
    direct_feature = DirectFeature(agg_feat, "log")
    features = [identity_feature, property_feature, direct_feature]
    fm = calculate_feature_matrix(features, entityset=pd_es, cutoff_time=cutoff_time)
    ifeat_answer = pd.Series([0, 7, 14, np.nan] + [np.nan] * 6)
    prop_answer = pd.Series([0, 0, 1, pd.NA, 0] + [pd.NA] * 5, dtype="boolean")
    dfeat_answer = pd.Series([14, 14, 14, np.nan] + [np.nan] * 6)

    assert all(fm.index.values == cutoff_time["instance_id"].values)
    for x, y in zip(fm.columns, [ifeat_answer, prop_answer, dfeat_answer]):
        pd.testing.assert_series_equal(fm[x], y, check_index=False, check_names=False)

    fm = calculate_feature_matrix(
        features,
        entityset=pd_es,
        cutoff_time=cutoff_time,
        approximate="5 seconds",
    )

    dfeat_answer[0] = 7  # approximate calculated before 14 appears
    dfeat_answer[2] = 7  # approximate calculated before 14 appears
    prop_answer[3] = False  # no_unapproximated_aggs code ignores cutoff time

    assert all(fm.index.values == cutoff_time["instance_id"].values)
    for x, y in zip(fm.columns, [ifeat_answer, prop_answer, dfeat_answer]):
        pd.testing.assert_series_equal(fm[x], y, check_index=False, check_names=False)


def test_missing_instances_with_categorical_index(pd_es):
    instance_ids = ["coke zero", "car", 3, "taco clock"]
    features = dfs(
        entityset=pd_es,
        target_dataframe_name="products",
        features_only=True,
    )

    fm = calculate_feature_matrix(
        entityset=pd_es,
        features=features,
        instance_ids=instance_ids,
    )
    assert fm.index.values.to_list() == instance_ids
    assert isinstance(fm.index, pd.CategoricalIndex)


def test_handle_chunk_size():
    total_size = 100

    # user provides no chunk size
    assert _handle_chunk_size(None, total_size) is None

    # user provides fractional size
    assert _handle_chunk_size(0.1, total_size) == total_size * 0.1
    assert _handle_chunk_size(0.001, total_size) == 1  # rounds up
    assert _handle_chunk_size(0.345, total_size) == 35  # rounds up

    # user provides absolute size
    assert _handle_chunk_size(1, total_size) == 1
    assert _handle_chunk_size(100, total_size) == 100
    assert isinstance(_handle_chunk_size(100.0, total_size), int)

    # test invalid cases
    with pytest.raises(AssertionError, match="Chunk size must be greater than 0"):
        _handle_chunk_size(0, total_size)

    with pytest.raises(AssertionError, match="Chunk size must be greater than 0"):
        _handle_chunk_size(-1, total_size)


def test_chunk_dataframe_groups():
    df = pd.DataFrame({"group": [1, 1, 1, 1, 2, 2, 3]})

    grouped = df.groupby("group")
    chunked_grouped = _chunk_dataframe_groups(grouped, 2)

    # test group larger than chunk size gets split up
    first = next(chunked_grouped)
    assert first[0] == 1 and first[1].shape[0] == 2
    second = next(chunked_grouped)
    assert second[0] == 1 and second[1].shape[0] == 2

    # test that equal to and less than chunk size stays together
    third = next(chunked_grouped)
    assert third[0] == 2 and third[1].shape[0] == 2
    fourth = next(chunked_grouped)
    assert fourth[0] == 3 and fourth[1].shape[0] == 1


def test_calls_progress_callback(mock_customer):
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

    es = mock_customer

    # make sure to calculate features that have different paths to same base feature
    trans_per_session = Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    trans_per_customer = Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    features = [trans_per_session, Feature(trans_per_customer, "sessions")]
    calculate_feature_matrix(
        features,
        entityset=es,
        progress_callback=mock_progress_callback,
    )

    # second to last entry is the last update from feature calculation
    assert np.isclose(
        mock_progress_callback.progress_history[-2],
        FEATURE_CALCULATION_PERCENTAGE * 100,
    )
    assert np.isclose(mock_progress_callback.total_update, 100.0)
    assert np.isclose(mock_progress_callback.total_progress_percent, 100.0)

    # test with cutoff time dataframe
    mock_progress_callback = MockProgressCallback()
    cutoff_time = pd.DataFrame(
        {
            "instance_id": [1, 2, 3],
            "time": [
                pd.to_datetime("2014-01-01 01:00:00"),
                pd.to_datetime("2014-01-01 02:00:00"),
                pd.to_datetime("2014-01-01 03:00:00"),
            ],
        },
    )

    calculate_feature_matrix(
        features,
        entityset=es,
        cutoff_time=cutoff_time,
        progress_callback=mock_progress_callback,
    )
    assert np.isclose(
        mock_progress_callback.progress_history[-2],
        FEATURE_CALCULATION_PERCENTAGE * 100,
    )
    assert np.isclose(mock_progress_callback.total_update, 100.0)
    assert np.isclose(mock_progress_callback.total_progress_percent, 100.0)


def test_calls_progress_callback_cluster(pd_mock_customer, dask_cluster):
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

    trans_per_session = Feature(
        pd_mock_customer["transactions"].ww["transaction_id"],
        parent_dataframe_name="sessions",
        primitive=Count,
    )
    trans_per_customer = Feature(
        pd_mock_customer["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    features = [trans_per_session, Feature(trans_per_customer, "sessions")]

    dkwargs = {"cluster": dask_cluster.scheduler.address}
    calculate_feature_matrix(
        features,
        entityset=pd_mock_customer,
        progress_callback=mock_progress_callback,
        dask_kwargs=dkwargs,
    )

    assert np.isclose(mock_progress_callback.total_update, 100.0)
    assert np.isclose(mock_progress_callback.total_progress_percent, 100.0)


def test_closes_tqdm(es):
    class ErrorPrim(TransformPrimitive):
        """A primitive whose function raises an error"""

        name = "error_prim"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = "Numeric"
        compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

        def get_function(self):
            def error(s):
                raise RuntimeError("This primitive has errored")

            return error

    value = Feature(es["log"].ww["value"])
    property_feature = value > 10
    error_feature = Feature(value, primitive=ErrorPrim)

    calculate_feature_matrix([property_feature], es, verbose=True)

    assert len(tqdm._instances) == 0

    match = "This primitive has errored"
    with pytest.raises(RuntimeError, match=match):
        calculate_feature_matrix([value, error_feature], es, verbose=True)
    assert len(tqdm._instances) == 0


def test_approximate_with_single_cutoff_warns(pd_es):
    features = dfs(
        entityset=pd_es,
        target_dataframe_name="customers",
        features_only=True,
        ignore_dataframes=["cohorts"],
        agg_primitives=["sum"],
    )

    match = (
        "Using approximate with a single cutoff_time value or no cutoff_time "
        "provides no computational efficiency benefit"
    )
    # test warning with single cutoff time
    with pytest.warns(UserWarning, match=match):
        calculate_feature_matrix(
            features,
            pd_es,
            cutoff_time=pd.to_datetime("2020-01-01"),
            approximate="1 day",
        )
    # test warning with no cutoff time
    with pytest.warns(UserWarning, match=match):
        calculate_feature_matrix(features, pd_es, approximate="1 day")

    # check proper handling of approximate
    feature_matrix = calculate_feature_matrix(
        features,
        pd_es,
        cutoff_time=pd.to_datetime("2011-04-09 10:31:30"),
        approximate="1 minute",
    )

    expected_values = [50, 50, 50]
    assert (feature_matrix["régions.SUM(log.value)"] == expected_values).values.all()


def test_calc_feature_matrix_with_cutoff_df_and_instance_ids(es):
    times = list(
        [datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)]
        + [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)]
        + [datetime(2011, 4, 9, 10, 40, 0)]
        + [datetime(2011, 4, 10, 10, 40, i) for i in range(2)]
        + [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)]
        + [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)],
    )
    instances = range(17)
    cutoff_time = pd.DataFrame({"time": times, es["log"].ww.index: instances})
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [True] + [False] * 2

    property_feature = Feature(es["log"].ww["value"]) > 10

    match = "Passing 'instance_ids' is valid only if 'cutoff_time' is a single value or None - ignoring"
    with pytest.warns(UserWarning, match=match):
        feature_matrix = calculate_feature_matrix(
            [property_feature],
            es,
            cutoff_time=cutoff_time,
            instance_ids=[1, 3, 5],
            verbose=True,
        )

    feature_matrix = to_pandas(feature_matrix)
    assert (feature_matrix[property_feature.get_name()] == labels).values.all()


def test_calculate_feature_matrix_returns_default_values(default_value_es):
    sum_features = Feature(
        default_value_es["transactions"].ww["value"],
        parent_dataframe_name="sessions",
        primitive=Sum,
    )
    sessions_sum = Feature(sum_features, "transactions")

    feature_matrix = calculate_feature_matrix(
        features=[sessions_sum],
        entityset=default_value_es,
    )

    feature_matrix = to_pandas(feature_matrix, index="id", sort_index=True)
    expected_values = [2.0, 2.0, 1.0, 0.0]

    assert (feature_matrix[sessions_sum.get_name()] == expected_values).values.all()


def test_dataframes_relationships(dataframes, relationships):
    fm_1, features = dfs(
        dataframes=dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
    )

    fm_2 = calculate_feature_matrix(
        features=features,
        dataframes=dataframes,
        relationships=relationships,
    )

    fm_1 = to_pandas(fm_1, index="id", sort_index=True)
    fm_2 = to_pandas(fm_2, index="id", sort_index=True)
    assert fm_1.equals(fm_2)


def test_no_dataframes(dataframes, relationships):
    features = dfs(
        dataframes=dataframes,
        relationships=relationships,
        target_dataframe_name="transactions",
        features_only=True,
    )

    msg = "No dataframes or valid EntitySet provided"
    with pytest.raises(TypeError, match=msg):
        calculate_feature_matrix(features=features, dataframes=None, relationships=None)


def test_no_relationships(dataframes):
    fm_1, features = dfs(
        dataframes=dataframes,
        relationships=None,
        target_dataframe_name="transactions",
    )

    fm_2 = calculate_feature_matrix(
        features=features,
        dataframes=dataframes,
        relationships=None,
    )

    fm_1 = to_pandas(fm_1, index="id")
    fm_2 = to_pandas(fm_2, index="id")
    assert fm_1.equals(fm_2)


def test_cfm_with_invalid_time_index(es):
    features = dfs(entityset=es, target_dataframe_name="customers", features_only=True)
    es["customers"].ww.set_types(logical_types={"signup_date": "integer"})
    match = "customers time index is numeric type "
    match += "which differs from other entityset time indexes"
    with pytest.raises(TypeError, match=match):
        calculate_feature_matrix(features=features, entityset=es)


def test_cfm_introduces_nan_values_in_direct_feats(es):
    es["customers"].ww.set_types(
        logical_types={"age": "Age", "engagement_level": "Integer"},
    )
    age_feat = Feature(es["customers"].ww["age"])
    engagement_feat = Feature(es["customers"].ww["engagement_level"])
    loves_ice_cream_feat = Feature(es["customers"].ww["loves_ice_cream"])
    features = [age_feat, engagement_feat, loves_ice_cream_feat]
    fm = calculate_feature_matrix(
        features=features,
        entityset=es,
        cutoff_time=pd.Timestamp("2010-04-08 04:00"),
        instance_ids=[1],
    )

    assert isinstance(es["customers"].ww.logical_types["age"], Age)
    assert isinstance(es["customers"].ww.logical_types["engagement_level"], Integer)
    assert isinstance(es["customers"].ww.logical_types["loves_ice_cream"], Boolean)

    assert isinstance(fm.ww.logical_types["age"], AgeNullable)
    assert isinstance(fm.ww.logical_types["engagement_level"], IntegerNullable)
    assert isinstance(fm.ww.logical_types["loves_ice_cream"], BooleanNullable)


def test_feature_origins_present_on_all_fm_cols(pd_es):
    class MultiCumSum(TransformPrimitive):
        name = "multi_cum_sum"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        number_output_features = 3

        def get_function(self):
            def multi_cum_sum(x):
                return x.cumsum(), x.cummax(), x.cummin()

            return multi_cum_sum

    feature_matrix, _ = dfs(
        entityset=pd_es,
        target_dataframe_name="log",
        trans_primitives=[MultiCumSum],
    )

    for col in feature_matrix.columns:
        origin = feature_matrix.ww[col].ww.origin
        assert origin in ["base", "engineered"]


def test_renamed_features_have_expected_column_names_in_feature_matrix(pd_es):
    class MultiCumulative(TransformPrimitive):
        name = "multi_cum_sum"
        input_types = [ColumnSchema(semantic_tags={"numeric"})]
        return_type = ColumnSchema(semantic_tags={"numeric"})
        number_output_features = 3

        def get_function(self):
            def multi_cum_sum(x):
                return x.cumsum(), x.cummax(), x.cummin()

            return multi_cum_sum

    multi_output_trans_feat = Feature(
        pd_es["log"].ww["value"],
        primitive=MultiCumulative,
    )
    groupby_trans_feat = GroupByTransformFeature(
        pd_es["log"].ww["value"],
        primitive=MultiCumulative,
        groupby=pd_es["log"].ww["product_id"],
    )
    multi_output_agg_feat = Feature(
        pd_es["log"].ww["product_id"],
        parent_dataframe_name="customers",
        primitive=NMostCommon(n=2),
    )
    slice = FeatureOutputSlice(multi_output_trans_feat, 1)
    stacked_feat = Feature(slice, primitive=Negate)

    multi_output_trans_names = ["cumulative_sum", "cumulative_max", "cumulative_min"]
    multi_output_trans_feat.set_feature_names(multi_output_trans_names)
    groupby_trans_feat_names = ["grouped_sum", "grouped_max", "grouped_min"]
    groupby_trans_feat.set_feature_names(groupby_trans_feat_names)
    agg_names = ["first_most_common", "second_most_common"]
    multi_output_agg_feat.set_feature_names(agg_names)

    features = [
        multi_output_trans_feat,
        multi_output_agg_feat,
        stacked_feat,
        groupby_trans_feat,
    ]
    feature_matrix = calculate_feature_matrix(entityset=pd_es, features=features)
    expected_names = multi_output_trans_names + agg_names + groupby_trans_feat_names
    for renamed_col in expected_names:
        assert renamed_col in feature_matrix.columns

    expected_stacked_name = "-(cumulative_max)"
    assert expected_stacked_name in feature_matrix.columns
