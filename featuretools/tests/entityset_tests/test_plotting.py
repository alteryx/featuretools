import os
import re

import graphviz
import pandas as pd
import pytest

import featuretools as ft


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


def test_multiple_rows(es):
    plot_ = es.plot()
    result = re.findall(r"\((\d+\srows?)\)", plot_.source)
    expected = ["{} rows".format(str(i.shape[0])) for i in es.entities]
    assert result == expected


def test_single_row():
    es = ft.EntitySet("test")
    df = pd.DataFrame({"foo": [1]})
    es.entity_from_dataframe("test", df)
    plot_ = es.plot()
    result = re.findall(r"\((\d+\srows?)\)", plot_.source)
    expected = ["1 row"]
    assert result == expected
