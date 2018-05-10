import pytest

from ..testing_utils import make_ecommerce_entityset

from featuretools import variable_types


@pytest.fixture
def es():
    return make_ecommerce_entityset()


def test_enforces_variable_id_is_str(es):
    assert variable_types.Categorical("1", es["customers"])
    with pytest.raises(AssertionError):
        variable_types.Categorical(1, es["customers"])


def test_get_all_instances(es):
    realvals = [2, 0, 1]
    for i, x in enumerate(es['customers'].get_all_instances()):
        assert x == realvals[i]


def test_is_index_column(es):
    assert es['cohorts'].index == 'cohort'


def test_sample(es):
    assert es['customers'].sample(3).shape[0] == 3


def test_eq(es):

    es['log'].id = 'customers'
    es['log'].index = 'notid'
    assert not es['customers'].__eq__(es['log'], deep=True)

    es['log'].index = 'id'
    assert not es['customers'].__eq__(es['log'], deep=True)

    es['log'].time_index = 'signup_date'
    assert not es['customers'].__eq__(es['log'], deep=True)

    es['log'].secondary_time_index = {
        'cancel_date': ['cancel_reason', 'cancel_date']}
    assert not es['customers'].__eq__(es['log'], deep=True)

    es['log'].indexed_by = None
    assert not es['log'].__eq__(es['customers'], deep=True)
