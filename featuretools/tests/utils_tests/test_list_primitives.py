from featuretools import list_primitives
from featuretools.primitives import (
    Day,
    GreaterThan,
    Last,
    NumCharacters,
    get_aggregation_primitives,
    get_transform_primitives
)
from featuretools.primitives.utils import _get_descriptions


def test_list_primitives_order():
    df = list_primitives()
    all_primitives = get_transform_primitives()
    all_primitives.update(get_aggregation_primitives())

    for name, primitive in all_primitives.items():
        assert name in df['name'].values
        row = df.loc[df['name'] == name].iloc[0]
        actual_desc = _get_descriptions([primitive])[0]
        if actual_desc:
            assert actual_desc == row['description']

    types = df['type'].values
    assert 'aggregation' in types
    assert 'transform' in types


def test_descriptions():
    primitives = {NumCharacters: 'Calculates the number of characters in a string.',
                  Day: 'Determines the day of the month from a datetime.',
                  Last: 'Determines the last value in a list.',
                  GreaterThan: 'Determines if values in one list are greater than another list.'}
    assert _get_descriptions(list(primitives.keys())) == list(primitives.values())
