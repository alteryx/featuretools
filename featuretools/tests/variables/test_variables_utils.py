from pytest import warns
from woodwork import list_logical_types

from featuretools.variable_types import list_variable_types


def test_list_variables():
    match = 'list_variable_types has been deprecated. Please use featuretools.list_logical_types instead.'
    with warns(FutureWarning, match=match):
        vtypes = list_variable_types()
    ltypes = list_logical_types()
    assert vtypes.equals(ltypes)
