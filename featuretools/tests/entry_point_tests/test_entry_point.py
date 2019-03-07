import pandas as pd
import pytest
import os

from ..primitive_tests.test_install_primitives import pip_freeze, remove_test_files


@pytest.fixture(scope='module')
def this_dir():
    return os.path.dirname(os.path.abspath(__file__))


def test_entry_point_plugin(this_dir):
    print(this_dir)
    file_path = os.path.join(this_dir, 'featuretools_plugin_tester',
                             'featuretools_plugin_tester', 'initialize',
                             'initialize_plugin_called.txt')
    import featuretools
    assert os.path.isfile(file_path)
    assert "featuretools-plugin-tester" in pip_freeze()
    from featuretools.primitives import CustomMinPlusOne
    input_array = pd.Series([0, 1, 2, 3, 4])
    primitive_func = CustomMinPlusOne().get_function()
    # CustomMinPlusOne is actually min + plus 1
    assert primitive_func(input_array) == 1
    assert CustomMinPlusOne.name == 'custom_min_plus_one'
