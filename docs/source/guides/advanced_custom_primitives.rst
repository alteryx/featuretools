Advanced Custom Primitives Guide
--------------------------------

Functions With Additional Arguments
===================================
.. ipython:: python
    :suppress:

    import featuretools as ft
    from featuretools.primitives import make_trans_primitive
    from featuretools.variable_types import Text, Numeric

One caveat with the make\_primitive functions is that the required arguments of ``function`` must be input features.  Here we create a function for ``StringCount``, a primitive which counts the number of occurrences of a string in a ``Text`` input.  Since ``string`` is not a feature, it needs to be a keyword argument to ``string_count``.

.. ipython:: python

    def string_count(column, string=None):
        '''Count the number of times the value string occurs'''
        assert string is not None, "string to count needs to be defined"
        # this is a naive implementation used for clarity
        counts = [element.lower().count(string) for element in column]
        return counts

In order to have features defined using the primitive reflect what string is being counted, we define a custom ``generate_name`` function.

.. ipython:: python

    def string_count_generate_name(self, base_feature_names):
      return u'STRING_COUNT(%s, "%s")' % (base_feature_names[0], self.kwargs['string'])


Now that we have the function, we create the primitive using the ``make_trans_primitive`` function.

.. ipython:: python

    StringCount = make_trans_primitive(function=string_count,
                                       input_types=[Text],
                                       return_type=Numeric,
                                       cls_attributes={"generate_name": string_count_generate_name})


Passing in ``string="test"`` as a keyword argument when initializing the `StringCount` primitive will make "test" the value used for string when ``string_count`` is called to calculate the feature values.  Now we use this primitive to define features and calculate the feature values.

.. ipython:: python

    from featuretools.tests.testing_utils import make_ecommerce_entityset

    es = make_ecommerce_entityset()

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="sessions",
                                      agg_primitives=["sum", "mean", "std"],
                                      trans_primitives=[StringCount(string="the")])
    feature_matrix.columns
    feature_matrix[['STD(log.STRING_COUNT(comments, "the"))', 'SUM(log.STRING_COUNT(comments, "the"))', 'MEAN(log.STRING_COUNT(comments, "the"))']]

