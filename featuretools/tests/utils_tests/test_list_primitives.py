from featuretools import list_primitives


def test_list_primitives():
    df = list_primitives()
    for x in ['year', 'mean', 'count']:
        assert x in df['name'].values.tolist()
    for x in ['aggregation', 'transform']:
        assert x in df['type'].values.tolist()
    chars_desc = 'Return the characters in a given string.'
    day_desc = 'Transform a Datetime feature into the day.'
    last_desc = 'Returns the last value.'
    for x in [chars_desc] + [day_desc] + [last_desc]:
        assert x in df['description'].values.tolist()
