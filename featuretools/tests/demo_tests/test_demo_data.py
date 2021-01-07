import urllib.request

import pytest

from featuretools.demo import load_flight, load_mock_customer, load_retail


@pytest.fixture(autouse=True)
def set_testing_headers():
    opener = urllib.request.build_opener()
    opener.addheaders = [('Testing', 'True')]
    urllib.request.install_opener(opener)


def test_load_retail_diff():
    nrows = 10
    es_first = load_retail(nrows=nrows)
    assert es_first['order_products'].df.shape[0] == nrows
    nrows_second = 11
    es_second = load_retail(nrows=nrows_second)
    assert es_second['order_products'].df.shape[0] == nrows_second


def test_mock_customer():
    n_customers = 4
    n_products = 3
    n_sessions = 30
    n_transactions = 400
    es = load_mock_customer(n_customers=n_customers, n_products=n_products, n_sessions=n_sessions,
                            n_transactions=n_transactions, random_seed=0, return_entityset=True)
    entity_names = [entity.id for entity in es.entities]
    expected_names = ['transactions', 'products', 'sessions', 'customers']
    assert set(expected_names) == set(entity_names)
    assert len(es['customers'].df) == 4
    assert len(es['products'].df) == 3
    assert len(es['sessions'].df) == 30
    assert len(es['transactions'].df) == 400


def test_load_flight():
    es = load_flight(month_filter=[1],
                     categorical_filter={'origin_city': ['Charlotte, NC']},
                     return_single_table=False, nrows=1000)
    entity_names = ['airports', 'flights', 'trip_logs', 'airlines']
    realvals = [(11, 3), (13, 9), (103, 21), (1, 1)]
    for i, name in enumerate(entity_names):
        assert es[name].shape == realvals[i]
