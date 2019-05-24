from featuretools.utils import Trie


def test_get_node():
    t = Trie(default=lambda: 'default')
    assert t.get_node([1, 2, 3])[[]] == 'default'

    t[[1, 2, 3]] = '123'
    t[[1, 2, 4]] = '124'
    sub = t.get_node([1, 2])
    assert sub[[3]] == '123'
    assert sub[[4]] == '124'

    sub[[4, 5]] = '1245'
    assert t[[1, 2, 4, 5]] == '1245'


def test_setting_and_getting():
    t = Trie(default=lambda: 'default')

    t[[1, 2, 3]] = '123'
    t[[1, 2, 4]] = '124'
    assert t[[1, 2, 3]] == '123'
    assert t[[1, 2, 4]] == '124'

    assert t[[1]] == 'default'
    t[[1]] = '1'
    assert t[[1]] == '1'

    t[[1, 2, 3]] = 'updated'
    assert t[[1, 2, 3]] == 'updated'


def test_iteration():
    t = Trie(default=lambda: 'default')

    t[[1, 2, 3]] = '123'
    t[[1, 2, 4]] = '124'
    expected = [
        ([], 'default'),
        ([1], 'default'),
        ([1, 2], 'default'),
        ([1, 2, 3], '123'),
        ([1, 2, 4], '124'),
    ]

    for i, value in enumerate(t):
        assert value == expected[i]
