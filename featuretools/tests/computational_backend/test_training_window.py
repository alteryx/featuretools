import featuretools as ft
import pandas as pd
from featuretools.primitives import Count


def test_training_window_overlap(es):
    es.add_last_time_indexes()

    count_log = ft.Feature(
        base=es['log']['id'],
        parent_entity=es['customers'],
        primitive=Count,
    )

    cutoff_time = pd.DataFrame({
        'id': [0, 0],
        'time': ['2011-04-09 10:30:00', '2011-04-09 10:40:00'],
    }).astype({'time': 'datetime64[ns]'})

    actual = ft.calculate_feature_matrix(
        features=[count_log],
        entityset=es,
        cutoff_time=cutoff_time,
        cutoff_time_in_index=True,
        training_window='10 minutes',
    )

    expected = pd.DataFrame({
        'COUNT(log)': {
            (0, pd.Timestamp('2011-04-09 10:30:00')): 1,
            (0, pd.Timestamp('2011-04-09 10:40:00')): 9,
        },
    })

    return actual.equals(expected)
