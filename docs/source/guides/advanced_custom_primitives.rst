Advanced Custom Primitives Guide
--------------------------------

Functions With Additonal Arguments
==================================
.. ipython:: python
    :suppress:

    import featuretools as ft
    from featuretools.primitives import make_trans_primitive, Sum, Mean, Std
    from featuretools.variable_types import Text, Numeric

One caveat with the make\_primitive functions is that the required arguments of ``function`` must be input features.  Here we create a function for ``StringCount``, a primitive which counts the number of occurrences of a string in a ``Text`` input.  Since ``string`` is not a feature, it needs to be a keyword argument to ``string_count``.

.. ipython:: python

    def string_count(column, string=None):
        '''
        ..note:: this is a naive implementation used for clarity
        '''
        assert string is not None, "string to count needs to be defined"
        counts = [element.lower().count(string) for element in column]
        return counts

In order to have features defined using the primitive reflect what string is being counted, we define a custom ``generate_name`` function.

.. ipython:: python

    def string_count_generate_name(self):
        return u"STRING_COUNT(%s, %s)" % (self.base_features[0].get_name(),
                                          '"'+str(self.kwargs['string']+'"'))


Now that we have the function, we create the primitive using the ``make_trans_primitive`` function.

.. ipython:: python

    StringCount = make_trans_primitive(function=string_count,
                                       input_types=[Text],
                                       return_type=Numeric,
                                       cls_attributes={"generate_name": string_count_generate_name})

Passing in ``string="test"`` as a keyword argument when creating a StringCount feature will make "test" the value used for string when ``string_count`` is called to calculate the feature values.  Now we use this primitive to create a feature and calculate the feature values.

.. ipython:: python

    from featuretools.tests.testing_utils import make_ecommerce_entityset

    es = make_ecommerce_entityset()
    count_the_feat = StringCount(es['log']['comments'], string="the")

Since ``string`` is a non-feature input Deep Feature Synthesis cannot automatically stack ``StringCount`` on other primitives to create more features.  However, a user-defined ``StringCount`` feature can be used by DFS as a seed feature that DFS can stack on top of.

.. ipython:: python

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="sessions",
                                      agg_primitives=[Sum, Mean, Std],
                                      seed_features=[count_the_feat])
    feature_matrix[['STD(log.STRING_COUNT(comments, "the"))', 'SUM(log.STRING_COUNT(comments, "the"))', 'MEAN(log.STRING_COUNT(comments, "the"))']]
