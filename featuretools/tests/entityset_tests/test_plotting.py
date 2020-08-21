import os
import re
import sys

import graphviz
import pandas as pd
import pytest
from dask import dataframe as dd

import featuretools as ft
from featuretools.utils.gen_utils import import_or_none

ks = import_or_none('databricks.koalas')


@pytest.fixture
def pd_simple():
    es = ft.EntitySet("test")
    df = pd.DataFrame({"foo": [1]})
    es.entity_from_dataframe("test", df)
    return es


@pytest.fixture
def dd_simple():
    es = ft.EntitySet("test")
    df = pd.DataFrame({"foo": [1]})
    df = dd.from_pandas(df, npartitions=2)
    es.entity_from_dataframe("test", df)
    return es


@pytest.fixture
def ks_simple():
    if sys.platform.startswith('win'):
        pytest.skip('skipping Koalas tests for Windows')
    if not ks:
        pytest.skip('Koalas not installed, skipping')
    es = ft.EntitySet("test")
    df = ks.DataFrame({'foo': [1]})
    es.entity_from_dataframe('test', df)
    return es


@pytest.fixture(params=['pd_simple', 'dd_simple', 'ks_simple'])
def simple_es(request):
    return request.getfixturevalue(request.param)


def test_returns_digraph_object(es):
    graph = es.plot()

    assert isinstance(graph, graphviz.Digraph)


def test_saving_png_file(es, tmpdir):
    output_path = str(tmpdir.join("test1.png"))

    es.plot(to_file=output_path)

    assert os.path.isfile(output_path)
    os.remove(output_path)


def test_missing_file_extension(es):
    output_path = "test1"

    with pytest.raises(ValueError) as excinfo:
        es.plot(to_file=output_path)

    assert str(excinfo.value).startswith("Please use a file extension")


def test_invalid_format(es):
    output_path = "test1.xzy"

    with pytest.raises(ValueError) as excinfo:
        es.plot(to_file=output_path)

    assert str(excinfo.value).startswith("Unknown format")


def test_multiple_rows(pd_es):
    plot_ = pd_es.plot()
    result = re.findall(r"\((\d+\srows?)\)", plot_.source)
    expected = ["{} rows".format(str(i.shape[0])) for i in pd_es.entities]
    assert result == expected


def test_single_row(pd_simple):
    plot_ = pd_simple.plot()
    result = re.findall(r"\((\d+\srows?)\)", plot_.source)
    expected = ["1 row"]
    assert result == expected
