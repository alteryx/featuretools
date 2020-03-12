Specifying Primitive Options
============================

.. ipython:: python
    :suppress:

    import featuretools as ft



By default, DFS will apply primitives across all entities and columns. This behavior can be altered through a few different
parameters. Entities and variables can be optionally ignored or included for an entire DFS run or on a per-primitive basis,
enabling greater control over features and less run time overhead.

.. ipython:: python

    from featuretools.tests.testing_utils import make_ecommerce_entityset

    es = make_ecommerce_entityset()

    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity='customers',
                                           agg_primitives=['mode'],
                                           trans_primitives=['weekday'])
    features_list


Specifying Options for an Entire Run
************************************
The ``ignore_entities`` and ``ignore_variables`` parameters of DFS control entities and variables (columns) that should be
ignored for all primitives. This is useful for ignoring columns or entities that don't relate to the problem or otherwise
shouldn't be included in the DFS run.

.. ipython:: python

    # ignore the 'log' and 'cohorts' entities entirely
    # ignore the 'date_of_birth' variable in 'customers' and the 'device_name' variable in 'sessions'
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity='customers',
                                           agg_primitives=['mode'],
                                           trans_primitives=['weekday'],
                                           ignore_entities=['log', 'cohorts'],
                                           ignore_variables={
                                               'sessions': ['device_name'],
                                               'customers': ['date_of_birth']})
    features_list

DFS completely ignores the ``'log'`` and ``'cohorts'`` entities when creating features. It also ignores the variables
``'device_name'`` and ``'date_of_birth'`` in ``'sessions'`` and ``'customers'`` respectively.
However, both of these options can be overridden by individual primitive options in the ``primitive_options`` parameter.

Specifying for Individual Primitives
************************************
Options for individual primitives or groups of primitives are set by the ``primitive_options`` parameter of DFS. This parameter
maps any desired options to specific primitives. In the case of conflicting options, options set at this level will override
options set at the entire DFS run level, and the include options will always take priority over their ignore counterparts.

Specifying Entities for Individual Primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Which entities to include/ignore can also be specified for a single primitive or a group of primitives. Entities can be
ignored using the ``ignore_entities`` option in ``primitive_options``, while entities to explicitly include are set by
the ``include_entities`` option. When ``include_entities`` is given, all entities not listed are ignored by the primitive.
No variables from any excluded entity will be used to generate features with the given primitive.

.. ipython:: python

    # ignore the 'cohorts' and 'log' entities, but only for the primitive 'mode'
    # include only the 'customers' entity for the primitives 'weekday' and 'day'
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity='customers',
                                           agg_primitives=['mode'],
                                           trans_primitives=['weekday', 'day'],
                                           primitive_options={
                                               'mode': {'ignore_entities': ['cohorts', 'log']},
                                               ('weekday', 'day'): {'include_entities': ['customers']}
                                           })
    features_list

In this example, DFS would only use the ``'customers'`` entity for both ``weekday`` and ``day``, and would use all entities
except ``'cohorts'`` and ``'log'`` for ``mode``.

Specifying Columns for Individual Primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Specific variables (columns) can also be explicitly included/ignored for a primitive or group of primitives. Variables to
ignore is set by the ``ignore_variables`` option, while variables to include is set by ``include_variables``. When the
``include_variables`` option is set, no other variables from that entity will be used to make features with the given primitive.

.. ipython:: python

    # Include the variables 'product_id' and 'zipcode', 'device_type', and 'cancel_reason' for 'mean'
    # Ignore the variables 'signup_date' and 'cancel_date' for 'weekday'
    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity='customers',
                                           agg_primitives=['mode'],
                                           trans_primitives=['weekday'],
                                           primitive_options={
                                               'mode': {'include_variables': {'log': ['product_id', 'zipcode'],
                                                                              'sessions': ['device_type'],
                                                                              'customers': ['cancel_reason']}},
                                               'weekday': {'ignore_variables': {'customers':
                                                                                    ['signup_date',
                                                                                     'cancel_date']}}})
    features_list

Here, ``mode`` will only use the variables ``'product_id'`` and ``'zipcode'`` from the entity ``'log'``, ``'device_type'``
from the entity ``'sessions'``, and ``'cancel_reason'`` from ``'customers'``. For any other entity, ``mode`` will use all
variables. The ``weekday`` primitive will use all variables in all entities except for ``'signup_date'`` and ``'cancel_date'``
from the ``'customers'`` entity.


Specifying GroupBy Options
~~~~~~~~~~~~~~~~~~~~~~~~~~
GroupBy Transform Primitives also have the additional options ``include_groupby_entities``, ``ignore_groupby_entities``,
``include_groupby_variables``, and ``ignore_groupby_variables``. These options are used to specify entities and columns
to include/ignore as groupings for inputs. By default, DFS only groups by ID columns. Specifying ``include_groupby_variables``
overrides this default, and will only group by variables given. On the other hand, ``ignore_groupby_variables`` will
continue to use only the ID columns, ignoring any variables specified that are also ID columns. Note that if including 
non-ID columns to group by, the included columns must also be a discrete type. 

.. ipython:: python

    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity='log',
                                           agg_primitives=[],
                                           trans_primitives=[],
                                           groupby_trans_primitives=['cum_sum',
                                                                     'cum_count'],
                                           primitive_options={
                                                 'cum_sum': {'ignore_groupby_variables': {'log': ['product_id']}},
                                                 'cum_count': {'include_groupby_variables': {'log': ['product_id',
                                                                                                     'priority_level']},
                                                               'ignore_groupby_entities': ['sessions']}})
    features_list

We ignore ``'product_id'`` as a groupby for ``cum_sum`` but still use any other ID columns in that or any other entity. For
'cum_count', we use only ``'product_id'`` and ``'priority_level'`` as groupbys. Note that ``cum_sum`` doesn't use
``'priority_level'`` because it's not an ID column, but we explicitly include it for ``cum_count``. Finally, note that specifying
groupby options doesn't affect what features the primitive is applied to. For example, ``cum_count`` ignores the entity ``sessions`` 
for groupbys, but the feature ``<Feature: CUM_COUNT(sessions.customer_id) by product_id>`` is still made. The groupby is from
the target entity ``log``, so the feature is valid given the associated options. To ignore the sessions entity for ``cum_count``, 
the ``ignore_entities`` option for ``cum_count`` would need to include ``sessions``.


Specifying for each Input for Multiple Input Primitives
*******************************************************
For primitives that take multiple columns as input, such as ``Trend``, the above options can be specified for each input by
passing them in as a list. If only one option dictionary is given, it is used for all inputs. The length of the list provided
must match the number of inputs the primitive takes.

.. ipython:: python

    feature_matrix, features_list = ft.dfs(entityset=es,
                                           target_entity='customers',
                                           agg_primitives=['trend'],
                                           trans_primitives=[],
                                           primitive_options={
                                                 'trend': [{'ignore_variables': {'log': ['value_many_nans']}},
                                                           {'include_variables': {'customers': ['signup_date'],
                                                                                  'log': ['datetime']}}]})
    features_list

Here, we pass in a list of primitive options for trend.  We ignore the variable ``'value_many_nans'`` for the first input
to ``trend``, and include the variables ``'signup_date'`` from ``'customers'`` for the second input.
