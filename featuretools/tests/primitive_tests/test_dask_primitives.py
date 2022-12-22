import pandas as pd
import pytest

from featuretools import calculate_feature_matrix, dfs, list_primitives
from featuretools.feature_base.cache import feature_cache
from featuretools.primitives import get_aggregation_primitives, get_transform_primitives
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import Library

UNSUPPORTED = [
    p.name
    for p in get_transform_primitives().values()
    if Library.DASK not in p.compatibility
]
UNSUPPORTED += [
    p.name
    for p in get_aggregation_primitives().values()
    if Library.DASK not in p.compatibility
]


@pytest.fixture(autouse=True)
def reset_dfs_cache():
    feature_cache.enabled = False
    feature_cache.clear_all()


def test_transform(pd_es, dask_es):
    pytest.skip(
        "TODO: Dask issue with `series.eq`. Fix once Dask Issue #7957 is closed.",
    )
    primitives = list_primitives()
    trans_list = primitives[primitives["type"] == "transform"]["name"].tolist()
    trans_primitives = [prim for prim in trans_list if prim not in UNSUPPORTED]
    agg_primitives = []
    cutoff_time = pd.Timestamp("2019-01-05 04:00")

    assert pd_es == dask_es

    # Run DFS using each dataframe as a target and confirm results match
    for df in pd_es.dataframes:
        features = dfs(
            entityset=pd_es,
            target_dataframe_name=df.ww.name,
            trans_primitives=trans_primitives,
            agg_primitives=agg_primitives,
            max_depth=2,
            features_only=True,
        )

        dask_features = dfs(
            entityset=dask_es,
            target_dataframe_name=df.ww.name,
            trans_primitives=trans_primitives,
            agg_primitives=agg_primitives,
            max_depth=2,
            features_only=True,
        )
        assert features == dask_features

        # Calculate feature matrix values to confirm output is the same between dask and pandas.
        # Not testing on all returned features due to long run times.
        fm = calculate_feature_matrix(
            features=features[:100],
            entityset=pd_es,
            cutoff_time=cutoff_time,
        )
        dask_fm = calculate_feature_matrix(
            features=dask_features[:100],
            entityset=dask_es,
            cutoff_time=cutoff_time,
        )

        # Categorical categories can be ordered differently, this makes sure they are the same
        dask_fm = dask_fm.astype(fm.dtypes)

        # Use the same columns and make sure both indexes are sorted the same
        dask_computed_fm = (
            dask_fm.compute().set_index(df.ww.index).loc[fm.index][fm.columns]
        )
        pd.testing.assert_frame_equal(fm, dask_computed_fm)


def test_aggregation(pd_es, dask_es):
    primitives = list_primitives()
    trans_primitives = []
    agg_list = primitives[primitives["type"] == "aggregation"]["name"].tolist()
    agg_primitives = [prim for prim in agg_list if prim not in UNSUPPORTED]

    assert pd_es == dask_es

    # Run DFS using each dataframe as a target and confirm results match
    for df in pd_es.dataframes:
        fm, _ = dfs(
            entityset=pd_es,
            target_dataframe_name=df.ww.name,
            trans_primitives=trans_primitives,
            agg_primitives=agg_primitives,
            cutoff_time=pd.Timestamp("2019-01-05 04:00"),
            max_depth=2,
        )

        dask_fm, _ = dfs(
            entityset=dask_es,
            target_dataframe_name=df.ww.name,
            trans_primitives=trans_primitives,
            agg_primitives=agg_primitives,
            cutoff_time=pd.Timestamp("2019-01-05 04:00"),
            max_depth=2,
        )

        # Categorical categories can be ordered differently, this makes sure they
        # are the same, including the index column
        index_col = df.ww.index
        fm = fm.reset_index()
        dask_fm = dask_fm.astype(fm.dtypes)
        fm = fm.set_index(index_col)

        pd.testing.assert_frame_equal(
            fm.sort_index(),
            to_pandas(dask_fm, index=index_col, sort_index=True),
        )
