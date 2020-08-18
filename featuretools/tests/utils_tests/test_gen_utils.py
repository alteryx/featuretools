import dask.dataframe as dd
import pandas as pd
import pytest

from featuretools.utils.gen_utils import (
    import_or_none,
    import_or_raise,
    is_instance
)


def test_import_or_raise_errors():
    with pytest.raises(ImportError, match="error message"):
        import_or_raise("_featuretools", "error message")


def test_import_or_raise_imports():
    math = import_or_raise("math", "error message")
    assert math.ceil(0.1) == 1


def test_import_or_none():
    math = import_or_none('math')
    assert math.ceil(0.1) == 1

    bad_lib = import_or_none('_featuretools')
    assert bad_lib is None


@pytest.fixture
def df():
    return pd.DataFrame({'id': range(5)})


def test_is_instance_single_module(df):
    assert is_instance(df, pd, 'DataFrame')


def test_is_instance_multiple_modules(df):
    df2 = dd.from_pandas(df, npartitions=2)
    assert is_instance(df, (dd, pd), 'DataFrame')
    assert is_instance(df2, (dd, pd), 'DataFrame')
    assert is_instance(df2['id'], (dd, pd), ('Series', 'DataFrame'))
    assert not is_instance(df2['id'], (dd, pd), ('DataFrame', 'Series'))


def test_is_instance_errors_mismatch():
    msg = 'Number of modules does not match number of classnames'
    with pytest.raises(ValueError, match=msg):
        is_instance('abc', pd, ('DataFrame', 'Series'))


def test_is_instance_none_module(df):
    assert not is_instance(df, None, 'DataFrame')
    assert is_instance(df, (None, pd), 'DataFrame')
    assert is_instance(df, (None, pd), ('Series', 'DataFrame'))
