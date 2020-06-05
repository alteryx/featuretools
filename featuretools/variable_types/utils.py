import inspect

import pandas as pd

from featuretools.utils.gen_utils import find_descendents
from featuretools.utils.plot_utils import (
    check_graphviz,
    get_graphviz_format,
    save_graph
)
from featuretools.variable_types import (
    DatetimeTimeIndex,
    NumericTimeIndex,
    Variable
)


def find_variable_types():
    """
    Retrieves all Variable Types as a dictionary where key is type_string
        of Variable, and value is Variable object

    Args:
        None

    Returns:
        variable_types (dict):

    """
    return {vtype.type_string: vtype for vtype in find_descendents(Variable)
            if vtype != Variable}


def list_variable_types():
    """
    Retrieves all Variable Types as a dataframe, with the columns
        of name, and

    Args:
        None

    Returns:
        variable_types (pd.DataFrame):
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

    v_types = list(find_variable_types().values())
    v_types.remove(NumericTimeIndex)
    v_types.remove(DatetimeTimeIndex)
    v_types.sort(key=lambda x: x.__name__)

    graph.node(Variable.__name__, shape='Mdiamond')
    for x in v_types:
        parents = [y for y in inspect.getmro(x) if y not in [object, x]]
        if parents == [Variable]:
            # a direct child of Variable
            graph.edge(Variable.__name__, x.__name__)
        else:
            # a descent of Variable, only plot the parent - child relation
            graph.edge(parents[0].__name__, x.__name__)

    # NumericTimeIndex and DatetimeTimeIndex are different since they subclassed under 2 variables
    for x in [NumericTimeIndex, DatetimeTimeIndex]:
        parents = [y for y in inspect.getmro(x) if y not in [object, x]]
        for y in parents[:-1]:
            graph.edge(y.__name__, x.__name__)

    if to_file:
        save_graph(graph, to_file, format_)
    return graph
