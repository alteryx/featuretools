import pytest

from featuretools.demo import load_flight, load_mock_customer, load_retail
from featuretools.synthesis import dfs
from featuretools.utils.gen_utils import is_python_2

if is_python_2():
    import urllib2
else:
    import urllib.request as urllib2


@pytest.fixture(autouse=True)
def set_testing_headers():
    opener = urllib2.build_opener()
    opener.addheaders = [('Testing', 'True')]
    urllib2.install_opener(opener)


def test_load_retail_diff():
    nrows = 10
    es_first = load_retail(nrows=nrows)
    assert es_first['order_products'].df.shape[0] == nrows
    nrows_second = 11
    es_second = load_retail(nrows=nrows_second)
    assert es_second['order_products'].df.shape[0] == nrows_second


def test_mock_customer():
    es = load_mock_customer(return_entityset=True)
    fm, fl = dfs(entityset=es, target_entity="customers", max_depth=3)
    for feature in fl:
        assert feature.get_name() in fm.columns


def test_load_flight():
    es = load_flight(month_filter=[1],
                     categorical_filter={'origin_city': ['Charlotte, NC']},
                     return_single_table=False, nrows=1000)
    entity_names = ['airports', 'flights', 'trip_logs', 'airlines']
    realvals = [(11, 3), (13, 9), (103, 21), (1, 1)]
    for i, name in enumerate(entity_names):
        assert es[name].shape == realvals[i]
