.. _variables:

Variables
---------
.. **Future Release**
    * Enhancements
    * Fixes
    * Changes
    * Documentation Changes
    * Testing Changes
    Thanks to the following people for contributing to this release:


.. ipython:: python

    from graphviz import Digraph
    from featuretools.variable_types.variable import find_variable_types
    import inspect
    from featuretools.variable_types import (
        Variable,
        NumericTimeIndex,
        DatetimeTimeIndex
    )


    g = Digraph('Variables', filename='cluster.gv')
    g.attr(rankdir="LR", fixedsize='true')

    v_types = list(find_variable_types().values())
    v_types.remove(NumericTimeIndex)
    v_types.remove(DatetimeTimeIndex)
    for x in v_types:
        parents = [y for y in inspect.getmro(x) if y not in [object, x]]
        if parents == [Variable]:
            g.edge('Variable', x.__name__)
        else:
            g.edge(parents[0].__name__, x.__name__)

    for x in [NumericTimeIndex, DatetimeTimeIndex]:
        parents = [y for y in inspect.getmro(x) if y not in [object, x]]
        for y in parents[:-1]:
            g.edge(y.__name__, x.__name__)
    g.node('Variable', shape='Mdiamond')
    g
