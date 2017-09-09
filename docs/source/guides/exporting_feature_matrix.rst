Exporting Feature Matrix
=========================


In this example, we're working with a mock customer behavior dataset

.. ipython:: python

    import featuretools as ft
    es = ft.demo.load_mock_customer(return_entityset=True)
    es




Run Deep Feature Synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~

A minimal input to DFS is a set of entities and a list of relationships and the "target_entity" to calculate features for. The output of DFS is a feature matrix and the corresponding list of feature definitions

.. ipython:: python

    feature_matrix, features = ft.dfs(entityset=es,
                                      target_entity="customers",
                                      verbose=True)
    feature_matrix

Save as csv
~~~~~~~~~~~
The feature matrix is a pandas dataframe that we can save to disk

.. ipython:: python

    feature_matrix.to_csv("feature_matrix.csv")

We can also read it back in as follows:

.. ipython:: python

    saved_fm = pd.read_csv("feature_matrix.csv", index_col="customer_id")
    saved_fm


.. ipython:: python
    :suppress:

    import os
    os.remove("feature_matrix.csv")
