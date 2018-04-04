import pandas as pd
from featuretools.utils import make_temporal_cutoffs
from itertools import chain


def test_make_temporal_cutoffs():
    instance_ids = pd.Series(range(10))
    cutoffs = pd.DatetimeIndex(start='1/2/2015',
                               periods=10,
                               freq='1d')
    temporal_cutoffs_by_nwindows = make_temporal_cutoffs(instance_ids,
                                                         cutoffs,
                                                         window_size='1h',
                                                         num_windows=2)

    assert temporal_cutoffs_by_nwindows.shape[0] == 20
    actual_instances = chain.from_iterable([[i, i] for i in range(10)])
    actual_times = [
        '1/1/2015 23:00:00',
        '1/2/2015 00:00:00',
        '1/2/2015 23:00:00',
        '1/3/2015 00:00:00',
        '1/3/2015 23:00:00',
        '1/4/2015 00:00:00',
        '1/4/2015 23:00:00',
        '1/5/2015 00:00:00',
        '1/5/2015 23:00:00',
        '1/6/2015 00:00:00',
        '1/6/2015 23:00:00',
        '1/7/2015 00:00:00',
        '1/7/2015 23:00:00',
        '1/8/2015 00:00:00',
        '1/8/2015 23:00:00',
        '1/9/2015 00:00:00',
        '1/9/2015 23:00:00',
        '1/10/2015 00:00:00',
        '1/10/2015 23:00:00',
        '1/11/2015 00:00:00',
        '1/11/2015 23:00:00'
    ]
    actual_times = [pd.Timestamp(c) for c in actual_times]

    for computed, actual in zip(temporal_cutoffs_by_nwindows['instance_id'], actual_instances):
        assert computed == actual
    for computed, actual in zip(temporal_cutoffs_by_nwindows['time'], actual_times):
        assert computed == actual
    cutoffs = [pd.Timestamp('1/2/2015')] * 10
    actual_times = ['1/1/2015 00:00:00', '1/2/2015 00:00:00'] * 10
    actual_times = [pd.Timestamp(c) for c in actual_times]
    temporal_cutoffs_by_start = make_temporal_cutoffs(instance_ids,
                                                      cutoffs,
                                                      window_size='1d',
                                                      start=pd.Timestamp('1/1/2015'))

    for computed, actual in zip(temporal_cutoffs_by_start['instance_id'], actual_instances):
        assert computed == actual
    for computed, actual in zip(temporal_cutoffs_by_start['time'], actual_times):
        assert computed == actual
