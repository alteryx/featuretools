from warnings import warn

from woodwork import list_logical_types


def list_variable_types():
    """
    Retrieves all logical types as a dataframe

    Args:
        None

    Returns:
        logical_types (pd.DataFrame): a DataFrame with all logical types
    """
    message = 'list_variable_types has been deprecated. Please use featuretools.list_logical_types instead.'
    warn(message=message, category=FutureWarning)
    return list_logical_types()


# TODO: decide if this should be adapted for woodwork
# def graph_variable_types(to_file=None):
#     """
#     Create a UML diagram-ish graph of all the Variables.

#     Args:
#         to_file (str, optional) : Path to where the plot should be saved.
#             If set to None (as by default), the plot will not be saved.

#     Returns:
#         graphviz.Digraph : Graph object that can directly be displayed in
#             Jupyter notebooks.
#     """
#     graphviz = check_graphviz()
#     format_ = get_graphviz_format(graphviz=graphviz,
#                                   to_file=to_file)

#     # Initialize a new directed graph
#     graph = graphviz.Digraph('variables', format=format_)
#     graph.attr(rankdir="LR")
#     graph.node(Variable.__name__, shape='Mdiamond')

#     all_variables_types = list(find_variable_types().values())
#     all_variables_types.sort(key=lambda x: x.__name__)

#     for node in all_variables_types:
#         for parent in node.__bases__:
#             graph.edge(parent.__name__, node.__name__)

#     if to_file:
#         save_graph(graph, to_file, format_)
#     return graph
