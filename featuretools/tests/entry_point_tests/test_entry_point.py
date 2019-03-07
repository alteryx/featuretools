import pandas as pd

from ..primitive_tests.test_install_primitives import pip_freeze


def test_entry_point_plugin():
    assert "featuretools-plugin-tester" in pip_freeze()
    from featuretools.primitives import CustomMinPlusOne
    input_array = pd.Series([0, 1, 2, 3, 4])
    primitive_func = CustomMinPlusOne().get_function()
    # CustomMinPlusOne is actually min + plus 1
    assert primitive_func(input_array) == 1
    assert CustomMinPlusOne.name == 'custom_min_plus_one'
