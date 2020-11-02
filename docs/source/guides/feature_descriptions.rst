Generating Feature Descriptions
================================
As features become more complicated, their names can become harder to understand. Both the :func:`featuretools.describe_feature` function and the :func:`featuretools.graph_feature` function can help explain what a feature is and the steps Featuretools took to generate it. Additionally, the ``describe_feature`` function can be augmented by providing custom definitions and templates to improve the resulting descriptions. 

.. ipython:: python
    :suppress:

    import featuretools as ft
    es = ft.demo.load_mock_customer(return_entityset=True)

    feature_defs = ft.dfs(entityset=es,
                          target_entity="customers",
                          agg_primitives=["mean", "sum", "mode", "n_most_common"],
                          trans_primitives=["month", "hour"],
                          max_depth=2,
                          features_only=True)

By default, ``describe_feature`` uses the existing variable and entity names and the default primitive description templates to generate feature descriptions. 

.. ipython:: python

    feature_defs[8]
    ft.describe_feature(feature_defs[8])

.. ipython:: python

    feature_defs[12]
    ft.describe_feature(feature_defs[12])

Improving Descriptions
~~~~~~~~~~~~~~~~~~~~~~~
While the default descriptions can be helpful, they can also be further improved by providing custom definitions of variables and features, and by providing alternative templates for primitive descriptions. 

Feature Descriptions
---------------------
Custom feature definitions will get used in the description in place of the automatically generated description. This can be used to better explain what a variable or feature is, or to provide descriptions that take advantage of a user's existing knowledge about the data or domain. 

.. ipython:: python

    feature_descriptions = {'customers: join_date': 'the date the customer joined'}

    ft.describe_feature(feature_defs[8], feature_descriptions=feature_descriptions)

For example, the above replaces the variable name ``"join_date"`` with a more descriptive definition of what that variable represents in the dataset. Variable descriptions can also be set directly on the variable through the ``description`` attribute:

.. ipython:: python

    es['customers']['join_date'].description = 'the date the customer joined'
    feature = ft.TransformFeature(es['customers']['join_date'], ft.primitives.Hour)
    feature
    ft.describe_feature(feature)

Variable descriptions must be set on the variable before the feature is created in order for descriptions to propagate. Note that if a description is set directly on a variable and a description is passed to ``describe_feature`` with ``feature_descriptions``, ``describe_feature`` will use the description found in ``feature_descriptions``. Feature descriptions can also be provided for generated features.

.. ipython:: python

    feature_descriptions = {
        'sessions: SUM(transactions.amount)': 'the total transaction amount for a session'}

    ft.describe_feature(feature_defs[12], feature_descriptions=feature_descriptions)


Here, we create and pass in a custom description of the intermediate feature ``SUM(transactions.amount)``. The description for ``MEAN(sessions.SUM(transactions.amount))``, which is built on top of ``SUM(transactions.amount)``, uses the custom description in place of the automatically generated one. Feature descriptions can be passed in as a dictionary that maps the custom descriptions to either the feature object itself or the unique feature name in the form ``"[entity_name]: [feature_name]"``, as shown above.

Primitive Templates
--------------------
Primitives descriptions are generated using primitive templates. By default, these are defined using the ``description_template`` attribute on the primitive. Primitives without a template default to using the ``name`` attribute of the primitive if it is defined, or the class name if it is not. Primitive description templates are string templates that take input feature descriptions as the positional arguments. These can be overwritten by mapping primitive instances or primitive names to custom templates and passing them into ``describe_feature`` through the ``primitive_templates`` argument. 

.. ipython:: python

    primitive_templates = {'sum': 'the total of {}'}

    feature_defs[6]
    ft.describe_feature(feature_defs[6], primitive_templates=primitive_templates)

In this example, we override the default template of ``'the sum of {}'`` with our custom template ``'the total of {}'``. The description uses our custom template instead of the default.

Multi-output primitives can use a list of primitive description templates to differentiate between the generic multi-output feature description and the feature slice descriptions. The first primitive template is always the generic overall feature. If only one other template is provided, it is used as the template for all slices. The slice number converted to the "nth" form is available through the ``nth_slice`` keyword.

.. ipython:: python

    feature = feature_defs[5]
    feature

    primitive_templates = {
        'n_most_common': [
            'the 3 most common elements of {}', # generic multi-output feature
            'the {nth_slice} most common element of {}']} # template for each slice 

    ft.describe_feature(feature, primitive_templates=primitive_templates)

Notice how the multi-output feature uses the first template for its description. Each slice of this feature will use the second slice template:

.. ipython:: python

    ft.describe_feature(feature[0], primitive_templates=primitive_templates)

    ft.describe_feature(feature[1], primitive_templates=primitive_templates)

    ft.describe_feature(feature[2], primitive_templates=primitive_templates)


Alternatively, instead of supplying a single template for all slices, templates can be provided for each slice to further customize the output. Note that in this case, each slice must get its own template.

.. ipython:: python



    primitive_templates = {
        'n_most_common': [
            'the 3 most common elements of {}',
            'the most common element of {}',
            'the second most common element of {}',
            'the third most common element of {}']}

    ft.describe_feature(feature, primitive_templates=primitive_templates)

    ft.describe_feature(feature[0], primitive_templates=primitive_templates)

    ft.describe_feature(feature[1], primitive_templates=primitive_templates)

    ft.describe_feature(feature[2], primitive_templates=primitive_templates)


Custom feature descriptions and primitive templates can also be seperately defined in a JSON file and passed to the ``describe_feature`` function using the ``metadata_file`` keyword argument. Descriptions passed in directly through the ``feature_descriptions`` and ``primitive_templates`` keyword arguments will take precedence over any descriptions provided in the JSON metadata file.