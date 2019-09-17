Specifying Primitive Options
============================

By default, DFS will apply primitives across all entities and columns. This behavior can be altered through a few different
parameters. Entities and variables can be optionally ignored or included for an entire DFS run or on a per-primitive basis,
enabling greater control over features and less run time overhead.


Specifying Options for an Entire Run
************************************
The ``ignore_entities`` and ``ignore_variables`` parameters of DFS control entities and variables (columns) that should be
ignored for all primitives::

    # ignore the 'sessions' entity and the 'session_id' variable in 'transactions'
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           ignore_entities=['sessions'],
                                           ignore_variables={'transactions': ['session_id']})

DFS will then ignore the ``sessions`` entity and the variable ``session_id`` in ``transactions`` when creating features.
However, both of these options can be overridden by individual primitive options in the ``primitive_options`` parameter.

Specifying for Individual Primitives
************************************
Options for individual primitives or groups of primitives are set by the ``primitive_options`` parameter of DFS. This parameter
maps any desired options to specific primitives. In the case of conflicting options, options set at this level will override
options set at the entire DFS run level, and the include options will always take priority over any ignore options.

Specifying Entities for Individual Primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Which entities to include/ignore can also be specified for a single primitive or a group of primitives. Entities can be
ignored using the ``ignore_entities`` option in ``primitive_options``, while entities to explicitly include are set by
the ``include_entities`` option. When ``include_entities`` is given, all entities not listed are ignored by the primitive::

    # ignore the 'sessions' entity, but only for the primitive 'mean'
    # include only the 'sessions' and 'transactions' entity for the primitive 'mode'
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           primitive_options={
                                                'mean': {'ignore_entities': ['sessions']},
                                                'mode': {'include_entities': ['sessions', 'transactions']}
                                           })

In this example, DFS would only use the ``sessions`` and ``transactions`` entities for ``mode``, and would use all entities
except ``sessions`` for ``mean``.

Specifying Columns for Individual Primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Specific variables (columns) can also be explicitly included/ignored for a primitive or group of primitives. Variables to
ignore is set by the ``ignore_variables`` option, while variables to include is set by ``include_variables``::

    # ignore the variable 'amount' in 'transactions' for the primitive 'mean'
    # include all variables for all entities except 'sessions' where all variables except 'device' are ignored for the primitive 'mode'
    only the variable 'device' in the entity 'sessions' all other variables for all other entities for the primitive 'mode'
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           agg_primitives=['count', 'mode']
                                           primitive_options={
                                                'mean': {'ignore_variables': {'transactions': ['amount']}},
                                                'mode': {'include_variables': {'sessions': ['device']}}
                                           })



Specifying GroupBy Options
~~~~~~~~~~~~~~~~~~~~~~~~~~
GroupBy Transform Primitives can also have additional options ``include_groupby_entities``, ``ignore_groupby_entities``,
``include_groupby_variables``, and ``ignore_groupby_variables``. These options are used to specify entities and columns
to include/ignore to use to group inputs. By default, DFS only groups by ID columns. Specifying ``include_groupby_variables``
overrides this default, and will only group by variables given. On the other hand, ``ignore_groupby_variables`` will
continue to use the ID columns, ignoring any variables specified that are also ID columns.
