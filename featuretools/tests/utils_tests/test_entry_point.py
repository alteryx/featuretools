import pandas as pd
import pytest

from ..testing_utils import make_ecommerce_entityset

from featuretools import dfs


@pytest.fixture()
def entityset():
    return make_ecommerce_entityset()


class MockEntryPoint(object):
    def on_call(self, kwargs):
        self.kwargs = kwargs

    def on_error(self, error, runtime):
        self.error = error

    def on_return(self, return_value, runtime):
        self.return_value = return_value

    def load(self):
        return self

    def __call__(self):
        return self


class MockPkgResources(object):
    def __init__(self, entry_point):
        self.entry_point = entry_point

    def iter_entry_points(self, name):
        return [self.entry_point]


def test_entry_point(entityset, monkeypatch):
    entry_point = MockEntryPoint()
    # overrides a module used in the entry_point decorator for dfs
    # so the decorator will use this mock entry point
    monkeypatch.setitem(dfs.__globals__['entry_point'].__globals__,
                        "pkg_resources",
                        MockPkgResources(entry_point))
    fm, fl = dfs(entityset=entityset, target_entity='customers')
    assert "entityset" in entry_point.kwargs.keys()
    assert "target_entity" in entry_point.kwargs.keys()
    assert (fm, fl) == entry_point.return_value


def test_entry_point_error(entityset, monkeypatch):
    entry_point = MockEntryPoint()
    monkeypatch.setitem(dfs.__globals__['entry_point'].__globals__,
                        "pkg_resources",
                        MockPkgResources(entry_point))
    with pytest.raises(KeyError):
        dfs(entityset=entityset, target_entity='missing_entity')

    assert isinstance(entry_point.error, KeyError)


def test_entry_point_detect_arg(monkeypatch):
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "card_id": [1, 2, 1, 3, 4, 5],
        "transaction_time": [10, 12, 13, 20, 21, 20],
        "fraud": [True, False, True, False, True, True]
    })
    entities = {
        "cards": (cards_df, "id"),
        "transactions": (transactions_df, "id", "transaction_time")
    }
    relationships = [("cards", "id", "transactions", "card_id")]
    entry_point = MockEntryPoint()
    monkeypatch.setitem(dfs.__globals__['entry_point'].__globals__,
                        "pkg_resources",
                        MockPkgResources(entry_point))
    fm, fl = dfs(entities,
                 relationships,
                 target_entity='cards')
    assert "entities" in entry_point.kwargs.keys()
    assert "relationships" in entry_point.kwargs.keys()
    assert "target_entity" in entry_point.kwargs.keys()
