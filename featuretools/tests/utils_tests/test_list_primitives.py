from featuretools import list_primitives
from featuretools.primitives import (
    get_aggregation_primitives,
    get_transform_primitives
)


def test_list_primitives():
    primitives_to_check = ['year', 'mean', 'count', 'characters', 'day',
                           'last']
    df = list_primitives()
    all_primitives = get_transform_primitives()
    all_primitives.update(get_aggregation_primitives())

    def get_description(primitive):
        return primitive.__doc__.split("\n")[0]
    for x in primitives_to_check:
        assert x in df['name'].values.tolist()
        row = df.loc[df['name'] == x]
        actual_desc = get_description(all_primitives[x])
        assert actual_desc in row['description'].values
    types = df['type'].values.tolist()
    assert 'aggregation' in types
    assert 'transform' in types
