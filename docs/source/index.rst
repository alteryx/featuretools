.. featuretools documentation main file, created by
   sphinx-quickstart on Thu May 19 20:40:30 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. currentmodule:: featuretools


What is Featuretools?
---------------------


.. image:: images/featuretools_nav2.svg
   :width: 500 px
   :alt: Featuretools
   :align: center

**Featuretools** is a framework to perform automated feature engineering. It excels at transforming temporal and relational datasets into feature matrices for machine learning.


.. _quick-start:

5 Minute Quick Start
====================

Below is an example of using Deep Feature Synthesis (DFS) to perform automated feature engineering. In this example, we apply DFS to a multi-table dataset consisting of timestamped customer transactions.

.. ipython:: python

    import featuretools as ft


Load Mock Data
~~~~~~~~~~~~~~
.. ipython:: python

    data = ft.demo.load_mock_customer()


Prepare data
~~~~~~~~~~~~

In this toy dataset, there are 3 tables. Each table is called an ``entity`` in Featuretools.

* **customers**: unique customers who had sessions
* **sessions**: unique sessions and associated attributes
* **transactions**: list of events in this session

.. ipython:: python

    customers_df = data["customers"]
    customers_df

    sessions_df = data["sessions"]
    sessions_df.sample(5)

    transactions_df = data["transactions"]
    transactions_df.sample(5)

First, we specify a dictionary with all the entities in our dataset.

.. ipython:: python

    entities = {
       "customers" : (customers_df, "customer_id"),
       "sessions" : (sessions_df, "session_id", "session_start"),
       "transactions" : (transactions_df, "transaction_id", "transaction_time")
    }


Second, we specify how the entities are related. When two entities have a one-to-many relationship, we call the "one" enitity, the "parent entity". A relationship between a parent and child is defined like this:

.. code-block:: python

    (parent_entity, parent_variable, child_entity, child_variable)

In this dataset we have two relationships

.. ipython:: python

    relationships = [("sessions", "session_id", "transactions", "session_id"),
                     ("customers", "customer_id", "sessions", "customer_id")]


.. note::

    To manage setting up entities and relationships, we recommend using the :class:`EntitySet <featuretools.EntitySet>` class which offers convenient APIs for managing data like this. See :doc:`getting_started/using_entitysets` for more information.


Run Deep Feature Synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~

A minimal input to DFS is a set of entities, a list of relationships, and the "target_entity" to calculate features for. The ouput of DFS is a feature matrix and the corresponding list of feature definitions.

Let's first create a feature matrix for each customer in the data

.. ipython:: python

    feature_matrix_customers, features_defs = ft.dfs(entities=entities,
                                                     relationships=relationships,
                                                     target_entity="customers")
    feature_matrix_customers

We now have dozens of new features to describe a customer's behavior.


Change target entity
~~~~~~~~~~~~~~~~~~~~
One of the reasons DFS is so powerful is that it can create a feature matrix for *any* entity in our data. For example, if we wanted to build features for sessions.


.. ipython:: python

    feature_matrix_sessions, features_defs = ft.dfs(entities=entities,
                                                    relationships=relationships,
                                                    target_entity="sessions")
    feature_matrix_sessions.head(5)


.. Technical problems it solves
.. ----------------------------

.. * Automatically creates features that require human intuition and expertise. Read more in :ref:`deep-feature-synthesis`
.. * Carefully handles time for predictive analytics use cases. Read more in :ref:`handling-time`.
.. * Creating feature engineering primitives that can be reused across datasets. Read more in :ref:`primitives`.





.. Featuretools can automatically identifying the best transformations, as well as dealing with time.
.. * It can be customized to address feature engineering use cases and is general enough to work across domains. It structures the process of transforming raw data into feature vectors ready for machine learning.
.. * It enables quick iteration through a unified interface to define prediction problems and feature transformations. It supports binary, multi-class, and regression predictions, as well as unsupervised learning approaches such as clustering or anomaly detection.
.. with and without experience building predictive models. Most functions in the library have sensible defaults to make it easy to run end to end with little configuration, but it does not intend to be a black box. The lower level API of Featuretools is a great way for those new to predictive modeling to learn how to build models from raw data, while enabling experts to maintain full control of how the framework handles their data.


What's next?
------------

* Learn about :doc:`getting_started/using_entitysets`
* Apply automated feature engineering with :doc:`getting_started/afe`
* Explore `runnable demos <https://www.featuretools.com/demos>`__ based on real world use cases
* Can't find what you're looking for? Ask for :doc:`resources/help`




Table of contents
-----------------

.. toctree::
   :maxdepth: 1

   install

.. toctree::
   :maxdepth: 2

   getting_started/getting_started_index
   guides/guides_index

.. toctree::
   :maxdepth: 1
   :caption: Resources and References

   resources/resources_index
   api_reference
   Primitives <https://primitives.featurelabs.com/>
   release_notes

Other links
------------
* :ref:`genindex`
* :ref:`search`
