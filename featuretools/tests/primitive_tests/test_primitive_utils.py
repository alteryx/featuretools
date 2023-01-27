import os

import pytest

from featuretools import list_primitives, summarize_primitives
from featuretools.primitives import (
    AddNumericScalar,
    Age,
    Count,
    Day,
    Diff,
    GreaterThan,
    Haversine,
    IsFreeEmailDomain,
    IsNull,
    Last,
    Max,
    Mean,
    Min,
    Mode,
    Month,
    MultiplyBoolean,
    NMostCommon,
    NumCharacters,
    NumericLag,
    NumUnique,
    NumWords,
    PercentTrue,
    Skew,
    Std,
    Sum,
    Weekday,
    Year,
    get_aggregation_primitives,
    get_default_aggregation_primitives,
    get_default_transform_primitives,
    get_transform_primitives,
)
from featuretools.primitives.base import PrimitiveBase
from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.primitives.utils import (
    _check_input_types,
    _get_descriptions,
    _get_summary_primitives,
    _get_unique_input_types,
    list_primitive_files,
    load_primitive_from_file,
)
from featuretools.utils.gen_utils import Library


def test_list_primitives_order():
    df = list_primitives()
    all_primitives = get_transform_primitives()
    all_primitives.update(get_aggregation_primitives())

    for name, primitive in all_primitives.items():
        assert name in df["name"].values
        row = df.loc[df["name"] == name].iloc[0]
        actual_desc = _get_descriptions([primitive])[0]
        if actual_desc:
            assert actual_desc == row["description"]
        assert row["dask_compatible"] == (Library.DASK in primitive.compatibility)
        assert row["valid_inputs"] == ", ".join(
            _get_unique_input_types(primitive.input_types),
        )
        expected_return_type = (
            str(primitive.return_type) if primitive.return_type is not None else None
        )
        assert row["return_type"] == expected_return_type

    types = df["type"].values
    assert "aggregation" in types
    assert "transform" in types


def test_valid_input_types():
    actual = _get_unique_input_types(Haversine.input_types)
    assert actual == {"<ColumnSchema (Logical Type = LatLong)>"}
    actual = _get_unique_input_types(MultiplyBoolean.input_types)
    assert actual == {
        "<ColumnSchema (Logical Type = Boolean)>",
        "<ColumnSchema (Logical Type = BooleanNullable)>",
    }
    actual = _get_unique_input_types(Sum.input_types)
    assert actual == {"<ColumnSchema (Semantic Tags = ['numeric'])>"}


def test_descriptions():
    primitives = {
        NumCharacters: "Calculates the number of characters in a given string, including whitespace and punctuation.",
        Day: "Determines the day of the month from a datetime.",
        Last: "Determines the last value in a list.",
        GreaterThan: "Determines if values in one list are greater than another list.",
    }
    assert _get_descriptions(list(primitives.keys())) == list(primitives.values())


def test_get_descriptions_doesnt_truncate_primitive_description():
    # single line
    descr = _get_descriptions([IsNull])
    assert descr[0] == "Determines if a value is null."

    # multiple line; one sentence
    descr = _get_descriptions([Diff])
    assert (
        descr[0]
        == "Computes the difference between the value in a list and the previous value in that list."
    )

    # multiple lines; multiple sentences
    class TestPrimitive(TransformPrimitive):
        """This is text that continues on after the line break
            and ends in a period.
            This is text on one line without a period

        Examples:
            >>> absolute = Absolute()
            >>> absolute([3.0, -5.0, -2.4]).tolist()
            [3.0, 5.0, 2.4]
        """

        name = "test_primitive"

    descr = _get_descriptions([TestPrimitive])
    assert (
        descr[0]
        == "This is text that continues on after the line break and ends in a period. This is text on one line without a period"
    )

    # docstring ends after description
    class TestPrimitive2(TransformPrimitive):
        """This is text that continues on after the line break
        and ends in a period.
        This is text on one line without a period
        """

        name = "test_primitive"

    descr = _get_descriptions([TestPrimitive2])
    assert (
        descr[0]
        == "This is text that continues on after the line break and ends in a period. This is text on one line without a period"
    )


def test_get_default_aggregation_primitives():
    primitives = get_default_aggregation_primitives()
    expected_primitives = [
        Sum,
        Std,
        Max,
        Skew,
        Min,
        Mean,
        Count,
        PercentTrue,
        NumUnique,
        Mode,
    ]
    assert set(primitives) == set(expected_primitives)


def test_get_default_transform_primitives():
    primitives = get_default_transform_primitives()
    expected_primitives = [
        Age,
        Day,
        Year,
        Month,
        Weekday,
        Haversine,
        NumWords,
        NumCharacters,
    ]
    assert set(primitives) == set(expected_primitives)


@pytest.fixture
def this_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def primitives_to_install_dir(this_dir):
    return os.path.join(this_dir, "primitives_to_install")


@pytest.fixture
def bad_primitives_files_dir(this_dir):
    return os.path.join(this_dir, "bad_primitive_files")


def test_list_primitive_files(primitives_to_install_dir):
    files = list_primitive_files(primitives_to_install_dir)
    custom_max_file = os.path.join(primitives_to_install_dir, "custom_max.py")
    custom_mean_file = os.path.join(primitives_to_install_dir, "custom_mean.py")
    custom_sum_file = os.path.join(primitives_to_install_dir, "custom_sum.py")
    assert {custom_max_file, custom_mean_file, custom_sum_file}.issubset(set(files))


def test_load_primitive_from_file(primitives_to_install_dir):
    primitve_file = os.path.join(primitives_to_install_dir, "custom_max.py")
    primitive_name, primitive_obj = load_primitive_from_file(primitve_file)
    assert issubclass(primitive_obj, PrimitiveBase)


def test_errors_more_than_one_primitive_in_file(bad_primitives_files_dir):
    primitive_file = os.path.join(bad_primitives_files_dir, "multiple_primitives.py")
    error_text = "More than one primitive defined in file {}".format(primitive_file)
    with pytest.raises(RuntimeError) as excinfo:
        load_primitive_from_file(primitive_file)
    assert str(excinfo.value) == error_text


def test_errors_no_primitive_in_file(bad_primitives_files_dir):
    primitive_file = os.path.join(bad_primitives_files_dir, "no_primitives.py")
    error_text = "No primitive defined in file {}".format(primitive_file)
    with pytest.raises(RuntimeError) as excinfo:
        load_primitive_from_file(primitive_file)
    assert str(excinfo.value) == error_text


def test_check_input_types():
    primitives = [Sum, Weekday, PercentTrue, Day, Std, NumericLag]
    log_in_type_checks = set()
    sem_tag_type_checks = set()
    unique_input_types = set()
    expected_log_in_check = {
        "boolean_nullable",
        "boolean",
        "datetime",
    }
    expected_sem_tag_type_check = {"numeric", "time_index"}
    expected_unique_input_types = {
        "<ColumnSchema (Logical Type = BooleanNullable)>",
        "<ColumnSchema (Semantic Tags = ['numeric'])>",
        "<ColumnSchema (Logical Type = Boolean)>",
        "<ColumnSchema (Logical Type = Datetime)>",
        "<ColumnSchema (Semantic Tags = ['time_index'])>",
    }
    for prim in primitives:
        input_types_flattened = prim.flatten_nested_input_types(prim.input_types)
        _check_input_types(
            input_types_flattened,
            log_in_type_checks,
            sem_tag_type_checks,
            unique_input_types,
        )

    assert log_in_type_checks == expected_log_in_check
    assert sem_tag_type_checks == expected_sem_tag_type_check
    assert unique_input_types == expected_unique_input_types


def test_get_summary_primitives():
    primitives = [
        Sum,
        Weekday,
        PercentTrue,
        Day,
        Std,
        NumericLag,
        AddNumericScalar,
        IsFreeEmailDomain,
        NMostCommon,
    ]
    primitives_summary = _get_summary_primitives(primitives)
    expected_unique_input_types = 7
    expected_unique_output_types = 6
    expected_uses_multi_input = 2
    expected_uses_multi_output = 1
    expected_uses_external_data = 1
    expected_controllable = 3
    expected_datetime_inputs = 2
    expected_bool = 1
    expected_bool_nullable = 1
    expected_time_index_tag = 1

    assert (
        primitives_summary["general_metrics"]["unique_input_types"]
        == expected_unique_input_types
    )
    assert (
        primitives_summary["general_metrics"]["unique_output_types"]
        == expected_unique_output_types
    )
    assert (
        primitives_summary["general_metrics"]["uses_multi_input"]
        == expected_uses_multi_input
    )
    assert (
        primitives_summary["general_metrics"]["uses_multi_output"]
        == expected_uses_multi_output
    )
    assert (
        primitives_summary["general_metrics"]["uses_external_data"]
        == expected_uses_external_data
    )
    assert (
        primitives_summary["general_metrics"]["are_controllable"]
        == expected_controllable
    )
    assert (
        primitives_summary["semantic_tag_metrics"]["time_index"]
        == expected_time_index_tag
    )
    assert (
        primitives_summary["logical_type_input_metrics"]["datetime"]
        == expected_datetime_inputs
    )
    assert primitives_summary["logical_type_input_metrics"]["boolean"] == expected_bool
    assert (
        primitives_summary["logical_type_input_metrics"]["boolean_nullable"]
        == expected_bool_nullable
    )


def test_summarize_primitives():
    df = summarize_primitives()
    trans_prims = get_transform_primitives()
    agg_prims = get_aggregation_primitives()
    tot_trans = len(trans_prims)
    tot_agg = len(agg_prims)
    tot_prims = tot_trans + tot_agg

    assert df["Count"].iloc[0] == tot_prims
    assert df["Count"].iloc[1] == tot_agg
    assert df["Count"].iloc[2] == tot_trans
