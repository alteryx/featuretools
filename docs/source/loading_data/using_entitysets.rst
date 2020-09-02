.. _using-entitysets:

Representing Data with EntitySets
=================================
.. currentmodule:: featuretools

An ``EntitySet`` is a collection of entities and the relationships between them. They are useful for preparing raw, structured datasets for feature engineering. While many functions in Featuretools  take ``entities`` and ``relationships`` as separate arguments, it is recommended to create an ``EntitySet``, so you can more easily manipulate your data as needed.


The Raw Data
~~~~~~~~~~~~

Below we have a two tables of data (represented as Pandas DataFrames) related to customer transactions. The first is a merge of transactions, sessions, and customers so that the result looks like something you might see in a log file:

.. ipython:: python

    import featuretools as ft
    data = ft.demo.load_mock_customer()
    transactions_df = data["transactions"].merge(data["sessions"]).merge(data["customers"])

    transactions_df.sample(10)

And the second dataframe is a list of products involved in those transactions.

.. ipython:: python

    products_df = data["products"]
    products_df


Creating an EntitySet
~~~~~~~~~~~~~~~~~~~~~

First, we initialize an EntitySet. If you'd like to give it name, you can optionally provide an ``id`` to the constructor.

.. ipython:: python

    es = ft.EntitySet(id="customer_data")


Adding entities
~~~~~~~~~~~~~~~

To get started, we load the transactions dataframe as an entity.

.. ipython:: python

    es = es.entity_from_dataframe(entity_id="transactions",
                                  dataframe=transactions_df,
                                  index="transaction_id",
                                  time_index="transaction_time",
                                  variable_types={"product_id": ft.variable_types.Categorical,
                                                  "zip_code": ft.variable_types.ZIPCode})
    es

.. note ::

    You can also display your entity set structure graphically by calling :meth:`.EntitySet.plot`.

This method loads each column in the dataframe in as a variable. We can see the variables in an entity using the code below.

.. ipython:: python

    es["transactions"].variables

In the call to ``entity_from_dataframe``, we specified three important parameters

* The ``index`` parameter specifies the column that uniquely identifies rows in the dataframe
* The ``time_index`` parameter tells Featuretools when the data was created.
* The ``variable_types`` parameter indicates that "product_id" should be interpreted as a Categorical variable, even though it just an integer in the underlying data.


Now, we can do that same thing with our products dataframe

.. ipython:: python

    es = es.entity_from_dataframe(entity_id="products",
                                  dataframe=products_df,
                                  index="product_id")

    es

With two entities in our entity set, we can add a relationship between them.

Adding a Relationship
~~~~~~~~~~~~~~~~~~~~~
We want to relate these two entities by the columns called "product_id" in each entity. Each product has multiple transactions associated with it, so it is called it the **parent entity**, while the transactions entity is known as the **child entity**. When specifying relationships we list the variable in the parent entity first. Note that each `ft.Relationship` must denote a one-to-many relationship rather than a relationship which is one-to-one or many-to-many.

.. ipython:: python

    new_relationship = ft.Relationship(es["products"]["product_id"],
                                       es["transactions"]["product_id"])
    es = es.add_relationship(new_relationship)
    es

Now, we see the relationship has been added to our entity set.


Creating entity from existing table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When working with raw data, it is common to have sufficient information to justify the creation of new entities. In order to create a new entity and relationship for sessions, we "normalize" the transaction entity.

.. ipython:: python

    es = es.normalize_entity(base_entity_id="transactions",
                             new_entity_id="sessions",
                             index="session_id",
                             make_time_index="session_start",
                             additional_variables=["device", "customer_id", "zip_code", "session_start", "join_date"])
                             
    es

Looking at the output above, we see this method did two operations

1. It created a new entity called "sessions" based on the "session_id" and "session_start" variables in "transactions"
2. It added a relationship connecting "transactions" and "sessions".

If we look at the variables in transactions and the new sessions entity, we see two more operations that were performed automatically.

.. ipython:: python

    es["transactions"].variables
    es["sessions"].variables

1. It removed "device", "customer_id", "zip_code" and "join_date" from "transactions" and created a new variables in the sessions entity. This reduces redundant information as the those properties of a session don't change between transactions.
2. It copied and marked "session_start" as a time index variable into the new sessions entity to indicate the beginning of a session. If the base entity has a time index and ``make_time_index`` is not set, ``normalize entity`` will create a time index for the new entity.  In this case it would create a new time index called "first_transactions_time" using the time of the first transaction of each session. If we don't want this time index to be created, we can set ``make_time_index=False``.

If we look at the dataframes, can see what the ``normalize_entity`` did to the actual data.

.. ipython:: python

    es["sessions"].df.head(5)
    es["transactions"].df.head(5)


To finish preparing this dataset, create a "customers" entity using the same method call.

.. ipython:: python

    es = es.normalize_entity(base_entity_id="sessions",
                             new_entity_id="customers",
                             index="customer_id",
                             make_time_index="join_date",
                             additional_variables=["zip_code", "join_date"])
                             
    es


Using the EntitySet
~~~~~~~~~~~~~~~~~~~

Finally, we are ready to use this EntitySet with any functionality within Featuretools. For example, let's build a feature matrix for each product in our dataset.

.. ipython:: python

    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                          target_entity="products")
    feature_matrix

As we can see, the features from DFS use the relational structure of our entity set. Therefore it is important to think carefully about the entities that we create.

Dask and Koalas EntitySets
~~~~~~~~~~~~~~~~~~~~~~~~~~

EntitySets can also be created using Dask dataframes or Koalas dataframes. For more information refer to :doc:`../guides/using_dask_entitysets` and :doc:`../guides/using_koalas_entitysets`.
