from pytest import warns
from woodwork import list_logical_types

from featuretools.variable_types import (  # graph_variable_types,
    list_variable_types
)


def test_list_variables():
    match = 'list_variable_types has been deprecated. Please use featuretools.list_logical_types instead.'
    with warns(FutureWarning, match=match):
        vtypes = list_variable_types()
    ltypes = list_logical_types()
    assert vtypes.equals(ltypes)


# def test_returns_digraph_object():
#     graph = graph_variable_types()
#     assert isinstance(graph, graphviz.Digraph)


# def test_saving_png_file(tmpdir):
#     output_path = str(tmpdir.join("test1.png"))

#     graph_variable_types(to_file=output_path)

#     assert os.path.isfile(output_path)
#     os.remove(output_path)


# def test_missing_file_extension():
#     output_path = "test1"

#     with pytest.raises(ValueError) as excinfo:
#         graph_variable_types(to_file=output_path)

#     assert str(excinfo.value).startswith("Please use a file extension")


# def test_invalid_format():
#     output_path = "test1.xzy"

#     with pytest.raises(ValueError) as excinfo:
#         graph_variable_types(to_file=output_path)

#     assert str(excinfo.value).startswith("Unknown format")
