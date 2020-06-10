import pandas as pd

from featuretools.utils.gen_utils import find_descendents
from featuretools.utils.plot_utils import (
    check_graphviz,
    get_graphviz_format,
    save_graph
)
from featuretools.variable_types.variable import Variable


def find_variable_types():
    """
    Retrieves all Variable types as a dictionary where key is type_string
        of Variable, and value is a Variable object.

    Args:
        None

    Returns:
        variable_types (dict):

    """
    return {vtype.type_string: vtype for vtype in find_descendents(Variable)
            if vtype != Variable}


def list_variable_types():
    """
    Retrieves all Variable types as a dataframe, with the column headers
        of name, type_string, and description.

    Args:
        None

    Returns:
        variable_types (pd.DataFrame): a DataFrame with column headers of
            name, type_strings, and description.
    """
    v_types = list(find_variable_types().values())
    v_type_strings = list(find_variable_types().keys())
    v_names = [x.__name__ for x in v_types]
    descriptions = [v.__doc__ for v in v_types]
    return pd.DataFrame({'name': v_names,
                         'type_string': v_type_strings,
                         'description': descriptions})


def graph_variable_types(to_file=None):
    """
    Create a UML diagram-ish graph of all the Variables.

    Args:
        to_file (str, optional) : Path to where the plot should be saved.
            If set to None (as by default), the plot will not be saved.

    Returns:
        graphviz.Digraph : Graph object that can directly be displayed in
            Jupyter notebooks.
    """
    graphviz = check_graphviz()
    format_ = get_graphviz_format(graphviz=graphviz,
                                  to_file=to_file)

    # Initialize a new directed graph
    graph = graphviz.Digraph('variables', format=format_)
    graph.attr(rankdir="LR")
    graph.node(Variable.__name__, shape='Mdiamond')

    all_variables_types = list(find_variable_types().values())
    all_variables_types.sort(key=lambda x: x.__name__)

    for node in all_variables_types:
        for parent in node.__bases__:
            graph.edge(parent.__name__, node.__name__)

    if to_file:
        save_graph(graph, to_file, format_)
    return graph
