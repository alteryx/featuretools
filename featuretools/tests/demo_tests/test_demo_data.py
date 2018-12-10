from featuretools.demo import load_flight, load_mock_customer, load_retail
from featuretools.synthesis import dfs


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
    realvals = [(11, 3), (13, 9), (103, 22), (1, 1)]
    for i, name in enumerate(entity_names):
        assert es[name].shape == realvals[i]
