import numpy as np
import pandas as pd
import pytest

import featuretools as ft
from featuretools.primitives import Count, PercentChange


@pytest.fixture
def issue_2634_entityset():
    customers_df = pd.DataFrame({"customer_id": ["abc"]})
    transactions_df = pd.DataFrame(
        {
            "transaction_id": list(range(20)),
            "customer_id": ["abc"] * 20,
            "transaction_time": pd.date_range(
                start="2020-01-01",
                periods=20,
                freq="D",
            ),
            "amount": [10.0] * 20,
        },
    )

    es = ft.EntitySet(id="issue_2634")
    es.add_dataframe(
        dataframe_name="customers",
        dataframe=customers_df,
        index="customer_id",
    )
    es.add_dataframe(
        dataframe_name="transactions",
        dataframe=transactions_df,
        index="transaction_id",
        time_index="transaction_time",
    )
    es.add_relationship("customers", "customer_id", "transactions", "customer_id")
    return es


@pytest.fixture
def multi_customer_entityset():
    customers_df = pd.DataFrame({"customer_id": ["c1", "c2"]})
    transactions_df = pd.DataFrame(
        {
            "transaction_id": list(range(10)),
            "customer_id": [
                "c1",
                "c1",
                "c1",
                "c1",
                "c1",
                "c2",
                "c2",
                "c2",
                "c2",
                "c2",
            ],
            "transaction_time": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-03"),
                pd.Timestamp("2020-01-04"),
                pd.Timestamp("2020-01-05"),
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-03"),
                pd.Timestamp("2020-01-04"),
                pd.Timestamp("2020-01-05"),
            ],
            "amount": [10.0] * 10,
        },
    )

    es = ft.EntitySet(id="multi_customer")
    es.add_dataframe(
        dataframe_name="customers",
        dataframe=customers_df,
        index="customer_id",
    )
    es.add_dataframe(
        dataframe_name="transactions",
        dataframe=transactions_df,
        index="transaction_id",
        time_index="transaction_time",
    )
    es.add_relationship("customers", "customer_id", "transactions", "customer_id")
    return es


def test_percent_change_on_count_single_customer(issue_2634_entityset):
    es = issue_2634_entityset
    count_f = ft.Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    pct_f = ft.Feature(count_f, primitive=PercentChange)

    cutoff_times = pd.DataFrame(
        {
            "customer_id": ["abc", "abc", "abc", "abc"],
            "time": pd.to_datetime(
                ["2020-01-07", "2020-01-09", "2020-01-13", "2020-01-19"],
            ),
        },
    )

    fm = ft.calculate_feature_matrix(
        [pct_f, count_f],
        entityset=es,
        cutoff_time=cutoff_times,
    )

    expected_counts = [7, 9, 13, 19]
    expected_pct = [
        np.nan,
        (9 - 7) / 7,
        (13 - 9) / 9,
        (19 - 13) / 13,
    ]

    assert list(fm[count_f.get_name()]) == expected_counts
    np.testing.assert_allclose(
        fm[pct_f.get_name()].values,
        expected_pct,
        rtol=1e-5,
    )
    assert np.isnan(fm[pct_f.get_name()].iloc[0])


def test_percent_change_multi_customer_isolation(multi_customer_entityset):
    es = multi_customer_entityset
    count_f = ft.Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    pct_f = ft.Feature(count_f, primitive=PercentChange)

    cutoff_times = pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c2", "c2"],
            "time": pd.to_datetime(
                ["2020-01-01", "2020-01-03", "2020-01-02", "2020-01-04"],
            ),
        },
    )

    fm = ft.calculate_feature_matrix(
        [pct_f, count_f],
        entityset=es,
        cutoff_time=cutoff_times,
    )

    # For c1: count goes 1 -> 3; pct is [NaN, (3-1)/1 = 2.0]
    # For c2: count goes 2 -> 4; pct is [NaN, (4-2)/2 = 1.0]
    # No calculation should leak between c1 and c2
    expected_pct = [np.nan, 2.0, np.nan, 1.0]
    np.testing.assert_allclose(
        fm[pct_f.get_name()].values,
        expected_pct,
        rtol=1e-5,
    )
    assert np.isnan(fm[pct_f.get_name()].iloc[0])
    assert np.isnan(fm[pct_f.get_name()].iloc[2])


def test_percent_change_out_of_order_cutoff_times(multi_customer_entityset):
    es = multi_customer_entityset
    count_f = ft.Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    pct_f = ft.Feature(count_f, primitive=PercentChange)

    # Scrambled order:
    # Row 0: c1 at 2020-01-03 (count 3, prior count 1 -> pct 2.0)
    # Row 1: c2 at 2020-01-04 (count 4, prior count 2 -> pct 1.0)
    # Row 2: c1 at 2020-01-01 (count 1, first obs -> pct NaN)
    # Row 3: c2 at 2020-01-02 (count 2, first obs -> pct NaN)
    cutoff_times = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c1", "c2"],
            "time": pd.to_datetime(
                ["2020-01-03", "2020-01-04", "2020-01-01", "2020-01-02"],
            ),
        },
    )

    fm = ft.calculate_feature_matrix(
        [pct_f, count_f],
        entityset=es,
        cutoff_time=cutoff_times,
        cutoff_time_in_index=True,
    )

    # Verify input row order is preserved
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("c1", pd.Timestamp("2020-01-03")),
            ("c2", pd.Timestamp("2020-01-04")),
            ("c1", pd.Timestamp("2020-01-01")),
            ("c2", pd.Timestamp("2020-01-02")),
        ],
        names=["customer_id", "time"],
    )
    assert (fm.index == expected_index).all()

    expected_pct = [2.0, 1.0, np.nan, np.nan]
    np.testing.assert_allclose(
        fm[pct_f.get_name()].values,
        expected_pct,
        rtol=1e-5,
    )
    assert np.isnan(fm[pct_f.get_name()].iloc[2])
    assert np.isnan(fm[pct_f.get_name()].iloc[3])


def test_percent_change_cutoff_time_in_index(issue_2634_entityset):
    es = issue_2634_entityset
    count_f = ft.Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    pct_f = ft.Feature(count_f, primitive=PercentChange)

    cutoff_times = pd.DataFrame(
        {
            "customer_id": ["abc", "abc"],
            "time": pd.to_datetime(["2020-01-07", "2020-01-09"]),
        },
    )

    fm_with_time = ft.calculate_feature_matrix(
        [pct_f],
        entityset=es,
        cutoff_time=cutoff_times,
        cutoff_time_in_index=True,
    )
    fm_without_time = ft.calculate_feature_matrix(
        [pct_f],
        entityset=es,
        cutoff_time=cutoff_times,
        cutoff_time_in_index=False,
    )

    assert isinstance(fm_with_time.index, pd.MultiIndex)
    assert fm_with_time.index.names == ["customer_id", "time"]

    assert isinstance(fm_without_time.index, pd.Index)
    assert fm_without_time.index.name == "customer_id"

    np.testing.assert_allclose(
        fm_with_time[pct_f.get_name()].values,
        fm_without_time[pct_f.get_name()].values,
    )


def test_percent_change_chunk_size(issue_2634_entityset):
    es = issue_2634_entityset
    count_f = ft.Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    pct_f = ft.Feature(count_f, primitive=PercentChange)

    cutoff_times = pd.DataFrame(
        {
            "customer_id": ["abc", "abc", "abc", "abc"],
            "time": pd.to_datetime(
                ["2020-01-07", "2020-01-09", "2020-01-13", "2020-01-19"],
            ),
        },
    )

    fm_chunked = ft.calculate_feature_matrix(
        [pct_f],
        entityset=es,
        cutoff_time=cutoff_times,
        chunk_size=1,
    )
    fm_unchunked = ft.calculate_feature_matrix(
        [pct_f],
        entityset=es,
        cutoff_time=cutoff_times,
        chunk_size=None,
    )

    np.testing.assert_allclose(
        fm_chunked[pct_f.get_name()].values,
        fm_unchunked[pct_f.get_name()].values,
    )


def test_percent_change_parallel(issue_2634_entityset):
    pytest.importorskip("distributed")
    es = issue_2634_entityset
    count_f = ft.Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    pct_f = ft.Feature(count_f, primitive=PercentChange)

    cutoff_times = pd.DataFrame(
        {
            "customer_id": ["abc", "abc", "abc", "abc"],
            "time": pd.to_datetime(
                ["2020-01-07", "2020-01-09", "2020-01-13", "2020-01-19"],
            ),
        },
    )

    fm_serial = ft.calculate_feature_matrix(
        [pct_f],
        entityset=es,
        cutoff_time=cutoff_times,
        n_jobs=1,
    )
    fm_parallel = ft.calculate_feature_matrix(
        [pct_f],
        entityset=es,
        cutoff_time=cutoff_times,
        n_jobs=2,
    )

    np.testing.assert_allclose(
        fm_parallel[pct_f.get_name()].values,
        fm_serial[pct_f.get_name()].values,
    )


def test_only_percent_change_requested(issue_2634_entityset):
    es = issue_2634_entityset
    count_f = ft.Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    pct_f = ft.Feature(count_f, primitive=PercentChange)

    cutoff_times = pd.DataFrame(
        {
            "customer_id": ["abc", "abc"],
            "time": pd.to_datetime(["2020-01-07", "2020-01-09"]),
        },
    )

    fm = ft.calculate_feature_matrix(
        [pct_f],
        entityset=es,
        cutoff_time=cutoff_times,
    )

    # Base feature COUNT(transactions) must not be exposed in the output
    assert list(fm.columns) == [pct_f.get_name()]
    # Woodwork schema should be valid and typed
    assert fm.ww.schema is not None
    assert fm.ww.logical_types[pct_f.get_name()].type_string == "double"


def test_column_ordering_preserved(issue_2634_entityset):
    es = issue_2634_entityset
    count_f = ft.Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    pct_f = ft.Feature(count_f, primitive=PercentChange)

    cutoff_times = pd.DataFrame(
        {
            "customer_id": ["abc", "abc"],
            "time": pd.to_datetime(["2020-01-07", "2020-01-09"]),
        },
    )

    # Order 1: [pct_f, count_f]
    fm1 = ft.calculate_feature_matrix(
        [pct_f, count_f],
        entityset=es,
        cutoff_time=cutoff_times,
    )
    assert list(fm1.columns) == [pct_f.get_name(), count_f.get_name()]

    # Order 2: [count_f, pct_f]
    fm2 = ft.calculate_feature_matrix(
        [count_f, pct_f],
        entityset=es,
        cutoff_time=cutoff_times,
    )
    assert list(fm2.columns) == [count_f.get_name(), pct_f.get_name()]


def test_pass_through_columns_preserved(issue_2634_entityset):
    es = issue_2634_entityset
    count_f = ft.Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    pct_f = ft.Feature(count_f, primitive=PercentChange)

    cutoff_times = pd.DataFrame(
        {
            "customer_id": ["abc", "abc"],
            "time": pd.to_datetime(["2020-01-07", "2020-01-09"]),
            "target_label": [0, 1],
            "extra_str": ["x", "y"],
        },
    )

    fm = ft.calculate_feature_matrix(
        [pct_f],
        entityset=es,
        cutoff_time=cutoff_times,
    )

    assert list(fm.columns) == [pct_f.get_name(), "target_label", "extra_str"]
    assert list(fm["target_label"]) == [0, 1]
    assert list(fm["extra_str"]) == ["x", "y"]


def test_percent_change_with_custom_parameters(issue_2634_entityset):
    es = issue_2634_entityset
    count_f = ft.Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    # periods = 2
    pct_f_2 = ft.Feature(count_f, primitive=PercentChange(periods=2))

    cutoff_times = pd.DataFrame(
        {
            "customer_id": ["abc", "abc", "abc", "abc"],
            "time": pd.to_datetime(
                ["2020-01-07", "2020-01-09", "2020-01-13", "2020-01-19"],
            ),
        },
    )

    fm = ft.calculate_feature_matrix(
        [pct_f_2, count_f],
        entityset=es,
        cutoff_time=cutoff_times,
    )

    # counts: [7, 9, 13, 19]
    # periods=2:
    # obs 0: nan
    # obs 1: nan
    # obs 2: (13 - 7) / 7 = 6/7
    # obs 3: (19 - 9) / 9 = 10/9
    expected_pct = [np.nan, np.nan, 6 / 7, 10 / 9]
    np.testing.assert_allclose(
        fm[pct_f_2.get_name()].values,
        expected_pct,
        rtol=1e-5,
    )
    assert np.isnan(fm[pct_f_2.get_name()].iloc[0])
    assert np.isnan(fm[pct_f_2.get_name()].iloc[1])


def test_single_cutoff_time_per_entity_returns_nan(multi_customer_entityset):
    es = multi_customer_entityset
    count_f = ft.Feature(
        es["transactions"].ww["transaction_id"],
        parent_dataframe_name="customers",
        primitive=Count,
    )
    pct_f = ft.Feature(count_f, primitive=PercentChange)

    # Each customer has only 1 cutoff time
    cutoff_times = pd.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "time": pd.to_datetime(["2020-01-02", "2020-01-03"]),
        },
    )

    fm = ft.calculate_feature_matrix(
        [pct_f],
        entityset=es,
        cutoff_time=cutoff_times,
    )

    # Both must be NaN because each customer has only 1 observation
    assert np.isnan(fm[pct_f.get_name()].iloc[0])
    assert np.isnan(fm[pct_f.get_name()].iloc[1])
