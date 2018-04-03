.. _connect-to-data:
.. currentmodule:: featuretools

Connecting to Data
==================

Featuretools includes a standard format for representing data that is used to set up predictions and build features.
The first step of integrating with Featuretools is to import your data into the :class:`.Dataset` format.


Datasets
~~~~~~~~~~~~~~~~~~~~~~~
A :class:`Dataset` stores information about :term:`entities <entity>` (analogous to database tables)
and :term:`variables <variable>` (columns in a table), including
:term:`relationships <relationship>` and data types -- more on that later.


Creating a Dataset
~~~~~~~~~~~~~~~~~~
To begin with Featuretools, you need a :class:`Dataset`. You can add entities and :term:`relationships <relationship>` (like primary key - foreign key relationships in an RDBMS) using methods on the Dataset once it's initialized.

To initialize a dataset

.. code-block:: python

  import featuretools as ft

  dataset = ft.Dataset(id="my_data")



Creating entities
~~~~~~~~~~~~~~~~~
An :class:`Entity` corresponds to a single table-like collection of data in a Dataset. Entities must be initialized by an existing Dataset. An entity can be created from a pandas DataFrame or a CSV file using the :class:`Dataset` API:

.. code-block:: python

  import pandas as pd
  data = pd.read_csv("data/events.csv")
  dataset.entity_from_dataframe(entity_id="events",
                          csv="data/events.csv",
                          index="id",
                          time_index="event_timestamp")


Entities are accessable on the Dataset like a dict, where keys are entity IDs.

>>> ds["events"]
<Entity: events>

Variables can be accessed via entities in a similar manner.

>>> ds["events"]["id"]
<Variable: id (dtype: categorical)>

A few things to note:

* The ``index`` parameter specifies which variable (column) in the data
  source to use as the primary ID of the entity. If there is no variable in the
  source that acts as an ID, the ``make_index=True`` flag can be passed. A
  new variable named ``index`` will be created and set as the ID.

* The ``time_index`` parameter specifies the id of the variable in the entity
  that indictates when that that information became known. If left empty, each
  event will get interpretted as always being known for prediction. By
  setting ``infer_time_index=True``, the method will infer the format of that
  variable and automatically parse it to a pd.Timestamp object.

* When loading data, the Dataset will try to interpret the types of variables. You may also pass the argument
  ``variable_types``, a dict mapping variable ID to subclasses of
  :class:`Variable`, to override this interpretation.

>>> ds.add_entity(entity_id="customers",
                  variable_types={"id": ft.variable_types.Categorical,
                                  "signup_date": ft.variable_types.Datetime})


Creating relationships
~~~~~~~~~~~~~~~~~~~~~~

A :class:`Relationship` describes how two entities are releated. In Featuretools, relationships always follow the "one to many" pattern, with one "parent" entity and one "child". In these terms, an instance of the parent entity is the "one" and instances of the child are the "many": that is, many instances of the child entity may refer to a single instance of the parent entity, but not vice versa.

Each relationship is defined by a variable on the parent entity and the variable
on the child entity which refers to it. Both variables must be instances of
:class:`.variable_types.Discrete`: that is, not floating-point numbers or other
continuous types.

.. code-block:: python

  customer_to_event = ft.Relationship(parent_variable=dataset["customers"]["id"],
                                      child_variable=dataset["events"]["customer_id"])
  dataset.add_relationships([customer_to_event])


Bringing everything together
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

And that's it. Once you've created a Dataset, imported data, added entities,
and defined relationships between them, you are ready to start :ref:`building
features <defining-features>` and making predictions.


.. Some more advanced options for importing data are described below.


.. Data wrangling
.. ~~~~~~~~~~~~~~
.. featuretools comes with many utilities to help with preparing you data for the prediction


.. Normalizing data
.. ****************
.. Todo...

.. Converting variable types
.. *************************
.. Todo...

.. Combining variables
.. *******************
.. Todo...

.. Full example
.. ~~~~~~~~~~~~
.. Todo...
