import inspect


from featuretools.variable_types.variable import find_variable_types
from featuretools.utils.entity_utils import check_graphviz
from featuretools.variable_types import (
    Variable,
    NumericTimeIndex,
    DatetimeTimeIndex
)


def graph_variables(to_file=None):
    graphviz = check_graphviz()
    graph = graphviz.Digraph(id='variables')
    graph.attr(rankdir="LR", fixedsize='true')

    v_types = list(find_variable_types().values())

    v_types.remove(NumericTimeIndex)
    v_types.remove(DatetimeTimeIndex)

    graph.node(Variable.__name__, shape='Mdiamond')
    for x in v_types:
        parents = [y for y in inspect.getmro(x) if y not in [object, x]]
        if parents == [Variable]:
            graph.edge(Variable.__name__, x.__name__)
        else:
            graph.edge(parents[0].__name__, x.__name__)

    for x in [NumericTimeIndex, DatetimeTimeIndex]:
        parents = [y for y in inspect.getmro(x) if y not in [object, x]]
        for y in parents[:-1]:
            graph.edge(y.__name__, x.__name__)

    if to_file:
        # Graphviz always appends the format to the file name, so we need to
        # remove it manually to avoid file names like 'file_name.pdf.pdf'
        offset = len(format) + 1  # Add 1 for the dot
        output_path = to_file[:-offset]
        graph.render(output_path, cleanup=True)
    return graph
