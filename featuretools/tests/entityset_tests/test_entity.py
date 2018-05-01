import numpy as np
import pandas as pd
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


def test_get_column_count(es):
    assert es['customers'].get_column_count('age') == 3


def test_get_column_max(es):
    assert es['customers'].get_column_max('age') == 56


def test_get_column_min(es):
    assert es['customers'].get_column_min('age') == 25


def test_get_column_mean(es):
    assert es['customers'].get_column_mean('age') == 38.0


def test_get_column_std(es):
    assert np.abs(es['customers'].get_column_std('age') - 16.093) < .001


def test_get_column_nunique(es):
    assert es['customers'].get_column_nunique('age') == 3


def test_sample_instances(es):
    assert es['customers'].sample_instances(2).shape[0] == 2


def test_get_top_n_instances(es):
    realvals = [2, 0]
    for i, x in enumerate(es['customers'].get_top_n_instances(2)):
        assert x == realvals[i]


def test_get_all_instances(es):
    realvals = [2, 0, 1]
    for i, x in enumerate(es['customers'].get_all_instances()):
        assert x == realvals[i]


def test_num_instances(es):
    assert es['customers'].num_instances == 3


def test_is_index_column(es):
    assert es['cohorts'].is_index_column('cohort')


def test_get_sample(es):
    assert es['customers'].get_sample(3).shape[0] == 3


def test_get_sliced_instance_ids(es):
    realvals = [1, 2, 3]
    for i, x in enumerate(es['log'].get_sliced_instance_ids(1, 4)):
        assert x == realvals[i]


def test_entity_head(es):
    ct = pd.Timestamp('1990-01-01')
    assert es['customers'].head(n=1).shape[0] == 1
    assert es['customers'].head(n=1, cutoff_time=ct).shape[0] == 0


def test_eq(es):
    assert not es['customers'].__eq__(es['log'])

    es['log'].id = 'customers'
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
