import os

import pytest

from featuretools import list_primitives
from featuretools.primitives import (
    Age,
    Count,
    Day,
    GreaterThan,
    Haversine,
    Last,
    Max,
    Mean,
    Min,
    Mode,
    Month,
    NumCharacters,
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
    get_transform_primitives
)
from featuretools.primitives.base import PrimitiveBase
from featuretools.primitives.utils import (
    _get_descriptions,
    _get_names_valid_inputs,
    list_primitive_files,
    load_primitive_from_file
)
from featuretools.utils.gen_utils import Library


def test_list_primitives_order():
    df = list_primitives()
    all_primitives = get_transform_primitives()
    all_primitives.update(get_aggregation_primitives())

    for name, primitive in all_primitives.items():
        assert name in df['name'].values
        row = df.loc[df['name'] == name].iloc[0]
        actual_desc = _get_descriptions([primitive])[0]
        if actual_desc:
            assert actual_desc == row['description']
        assert row['dask_compatible'] == (Library.DASK in primitive.compatibility)
        assert row['valid_inputs'] == ', '.join(_get_names_valid_inputs(primitive.input_types))
        assert row['return_type'] == getattr(primitive.return_type, '__name__', None)

    types = df['type'].values
    assert 'aggregation' in types
    assert 'transform' in types


def test_valid_input_types():
    actual = _get_names_valid_inputs(Haversine.input_types)
    assert actual == {'LatLong'}
    actual = _get_names_valid_inputs(GreaterThan.input_types)
    assert actual == {'Datetime', 'Numeric', 'Ordinal'}
    actual = _get_names_valid_inputs(Sum.input_types)
    assert actual == {'Numeric'}


def test_descriptions():
    primitives = {NumCharacters: 'Calculates the number of characters in a string.',
                  Day: 'Determines the day of the month from a datetime.',
                  Last: 'Determines the last value in a list.',
                  GreaterThan: 'Determines if values in one list are greater than another list.'}
    assert _get_descriptions(list(primitives.keys())) == list(primitives.values())


def test_get_default_aggregation_primitives():
    primitives = get_default_aggregation_primitives()
    expected_primitives = [Sum, Std, Max, Skew, Min, Mean, Count, PercentTrue,
                           NumUnique, Mode]
    assert set(primitives) == set(expected_primitives)


def test_get_default_transform_primitives():
    primitives = get_default_transform_primitives()
    expected_primitives = [Age, Day, Year, Month, Weekday, Haversine, NumWords,
                           NumCharacters]
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
