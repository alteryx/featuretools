Advanced Custom Primitives Guide
--------------------------------

Functions With Additional Arguments
===================================
.. ipython:: python
    :suppress:

    import featuretools as ft
    from featuretools.primitives import make_trans_primitive
    from featuretools.variable_types import Text, Numeric, Ordinal

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

Primitives That Use External Data Files
=======================================
Some primitives require external data files in order to perform their computation. For example, imagine a primitive that uses a pre-trained sentiment classifier to classify text. Here is how that would be implemented

.. ipython:: python

    from featuretools.primitives import TransformPrimitive

    class Sentiment(TransformPrimitive):
        '''Reads in a text field and returns "negative", "neutral", or "positive"'''
        name = "sentiment"
        input_types = [Text]
        return_type = Categorical
        def get_function(self):
            filepath = self.get_data_path('sentiment_model.pickle') # returns absolute path to the file
            import pickle
            with open(filepath, 'r') as f:
                model = pickle.load(f)
            def predict(x):
                return model.predict(x)
            return predict


The ``get_data_path`` method is used to find the location of the trained model.

.. note::

    The primitive loads the model within the `get_function` method, but outside of the `score` function.  This way the model is loaded from disk only once when the Featuretools backend requests the primitive function instead of every time `score` is called.
