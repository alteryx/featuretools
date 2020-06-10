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
    v_types.sort(key=lambda x: x.__name__)

    from collections import defaultdict
    adjacency_list = defaultdict(list)

    graph.node(Variable.__name__, shape='Mdiamond')
    for x in v_types:
        parents = [y for y in inspect.getmro(x) if y not in [object, x]]
        subclasses = x.__subclasses__()

        print('---')
        print(x.__name__)
        print('parents -> ', parents)
        print('subclasses -> ', subclasses)
        print('---')
        # if len(subclasses) == 0:
        #     graph.edge(Variable.__name__, x.__name__)
        if parents == [Variable]:
            # a direct child of Variable
            adjacency_list[Variable].append(x)
            graph.edge(Variable.__name__, x.__name__)
        # else:
        #     # a descent of Variable, only plot the parent - child relation
        #     graph.edge(parents[0].__name__, x.__name__)
    import pprint
    pprint(dict(adjacency_list))
    if to_file:
        save_graph(graph, to_file, format_)
    return graph
