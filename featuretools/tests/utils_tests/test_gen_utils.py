import pytest

from featuretools.utils.gen_utils import import_or_raise


def test_import_or_raise_errors():
    with pytest.raises(ImportError, match="error message"):
        import_or_raise("_featuretools", "error message")


def test_import_or_raise_imports():
    math = import_or_raise("math", "error message")
    assert math.ceil(0.1) == 1
