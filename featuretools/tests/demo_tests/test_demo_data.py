import os

from featuretools.demo import load_flight, load_mock_customer, load_retail
from featuretools.demo.flight import make_flight_pathname
from featuretools.demo.retail import RETAIL_CSV, make_retail_pathname
from featuretools.synthesis import dfs


def test_load_retail_save():
    nrows = 10

    load_retail(nrows=nrows, return_single_table=True)
    assert os.path.isfile(make_retail_pathname(nrows, RETAIL_CSV))
    assert os.path.getsize(make_retail_pathname(nrows, RETAIL_CSV)) < 45580670
    os.remove(make_retail_pathname(nrows, RETAIL_CSV))


def test_load_retail_diff():
    nrows = 10
    es_first = load_retail(nrows=nrows)
    assert es_first['order_products'].df.shape[0] == nrows

    nrows_second = 11
    es_second = load_retail(nrows=nrows_second)
    assert es_second['order_products'].df.shape[0] == nrows_second

    os.remove(make_retail_pathname(nrows, RETAIL_CSV))
    os.remove(make_retail_pathname(nrows_second, RETAIL_CSV))


def test_mock_customer():
    es = load_mock_customer(return_entityset=True)
    fm, fl = dfs(entityset=es, target_entity="customers", max_depth=3)
    for feature in fl:
        assert feature.get_name() in fm.columns


def test_load_flight():
    demo_path, _, _ = make_flight_pathname(demo=True)
    es = load_flight(month_filter=[1],
                     categorical_filter={'origin_city': ['Charlotte, NC']},
                     return_single_table=False, nrows=1000)

    assert os.path.isfile(demo_path)
    entity_names = ['airports', 'flights', 'trip_logs', 'airlines']
    realvals = [(11, 3), (13, 9), (103, 22), (1, 1)]
    for i, name in enumerate(entity_names):
        assert es[name].shape == realvals[i]
