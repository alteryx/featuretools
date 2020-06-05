import os

import graphviz
import pytest

from featuretools.variable_types.utils import (
    find_variable_types,
    graph_variable_types,
    list_variable_types
)


def test_find_variable_types():
    for type_string, v_type in find_variable_types().items():
        assert v_type.type_string == type_string


def test_list_variables():
    df = list_variable_types()
    for v_name, v_type in find_variable_types().items():
        assert v_type.__name__ in df['name'].values
        assert v_type.__doc__ in df['description'].values
        assert v_type.type_string in df['type_string'].values


def test_returns_digraph_object():
    graph = graph_variable_types()
    assert isinstance(graph, graphviz.Digraph)


def test_saving_png_file(tmpdir):
    output_path = str(tmpdir.join("test1.png"))

    graph_variable_types(to_file=output_path)

    assert os.path.isfile(output_path)
    os.remove(output_path)


def test_missing_file_extension():
    output_path = "test1"

    with pytest.raises(ValueError) as excinfo:
        graph_variable_types(to_file=output_path)

    assert str(excinfo.value).startswith("Please use a file extension")


def test_invalid_format():
    output_path = "test1.xzy"

    with pytest.raises(ValueError) as excinfo:
        graph_variable_types(to_file=output_path)

    assert str(excinfo.value).startswith("Unknown format")
