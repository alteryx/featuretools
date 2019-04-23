.. _handling-time:

.. currentmodule:: featuretools

Handling Time
=============

Time is a naturally relevant factor in many predictive modeling problems. Consider the following questions:

1. How much revenue will I bring in next month?
2. What is the expected delay on my next flight?
3. Will my user purchase an upgrade to their membership?

A good way to estimate how effective you are at predicting revenue would be to see how you would have done predicting it last month or the month before. You would similarly be interested in checking if you were able to predict the delay on your previous flight, or how good you are historically at detecting customers who would upgrade.

However, it is immensely tricky to make a feature matrix by hand for those predictions. To create historical predictions you need to set a **time** to make a prediction for every row and then **cut off** any data in your dataset that happens after that time. Then, you can use the remaining valid data to make any features you like.

Some of the most powerful functionality in Featuretools is the ability to accurately and precisely handle time. To make the most of that functionality, it is necessary to understand how :ref:`provided times <representing-time>` will be used.

Outline
---------
This page is the answer to the question *why should I pay attention to datetimes in my data?* There are two interconnected parts to that answer:

1. What are the implications of setting a :ref:`time index <representing-time>`?
2. How does Featuretools take in :ref:`predictions <cutoff-time>`?

The first section shows explains how to handle the complexities that can come up when assigning times to your data and the second shows how to use those times to make rows of a feature matrix. While time can be a sticking point for our users, we have found that it's often a useful construct in utilizing data from the real world.

.. _representing-time:

Introduction to the Time Index
----------------------------------------------------------
We'll start with the :func:`Mock Customer <demo.load_mock_customer>` entityset.

.. ipython:: python
    :suppress:

    pd.options.display.max_columns = 200

.. ipython:: python

    import featuretools as ft
    es_mc = ft.demo.load_mock_customer(return_entityset=True, random_seed=0)
    es_mc['transactions'].df.head()

The ``transactions`` entity has one row for every transaction and a ``transaction_time`` for every row. The user has an option to set a **time index** for any entity they create, representing the first time information from the row can be used. In this example, most people would make the reasonable choice to set the ``transaction_time`` as the time index for the ``transactions`` entity. Not every datetime column is a time index, so the choice is not always straightforward. Consider the ``customers`` entity:

.. ipython:: python

    es_mc['customers'].df

Here we have two time columns ``join_date`` and ``date_of_birth``. While either column might be useful for making features, the ``join_date`` should be used as the time index. It represents when the data owner learns about the existence of a given customer. Generically: *the time index is the first time anything from a row can be known to the dataset owner*. Rows are treated as non-existent prior to the time index. 

.. important::

    The **time index** is defined as the first time information from a row can be used. It represents the first time anything from a row can be known to the dataset owner.

In databases, information tends to be written after an event has passed. This can be problematic on the machine learning side -- it's often necessary to ignore entire columns to avoid leaking labels. If you're interested in how to safely use those columns, the :ref:`advanced time index <flight-ti>` section below explores how time can used with a dataset from the US Department of Transportation. Before we get there, we're going to show how to make predictions using these time indices.

.. _cutoff-time:

Introduction to Cutoff Times
--------------------------------------------

For a given :class:`EntitySet <EntitySet>`, there are many possible prediction problems that you might want to solve. Trying to predict customer purchases an hour in advance uses different data than trying to predict purchases a day in advance. Often, it's desirable to test multiple questions and explore which one you want to use. Featuretools makes that process easier through cutoff times.

A **cutoff_time** specifies the last valid time for data. This allows you to give a cutoff time of 11:59pm on December 31st, and then make predictions for January 1st without leaking data. Since no data from January is available, we can be confident that our training data won't leak data from the future.

Let's do a short example with some code. We want to predict whether customers ``1``, ``2`` and ``3`` will spend $500 after ``04:00`` on January 1 by the end of the day. The ``time`` column emulates the way a human would make a historical prediction. It is an instruction to not use any future information constructing that row even if we have it in our entityset. Since our cutoff time is ``04:00`` DFS will use data that happened before, or at ``04:00``.

In this case, we're making predictions for all three customers at the same time, ``2014-1-1 04:00`` so we set that as the second column. We have also checked that ``1`` and ``2`` will spend $500 while customer ``3`` will not, so we include those labels as a third column.

.. image:: ../images/retail_ct.png
   :width: 400 px
   :alt: retail cutoff time diagram
   :align: center

We will use all of the information between the ``time_index`` of rows ``1``, ``2`` and ``3`` and the prediction time ``04:00 2014-1-1`` to make predictions about what will happen for the rest of the day.

.. ipython:: python

    ct = pd.DataFrame()
    ct['customer_id'] = [1, 2, 3]
    ct['time'] = pd.to_datetime(['2014-1-1 04:00', 
                                 '2014-1-1 04:00',
                                 '2014-1-1 04:00'])
    ct['label'] = [True, True, False]
    ct
    fm, features = ft.dfs(entityset=es_mc, 
                          target_entity='customers', 
                          cutoff_time=ct, 
                          cutoff_time_in_index=True)
    fm

We made 74 features for the three customers using only data whose time index was before the cutoff time. Since you can specify the prediction time for every row, you have a lot of control over which data will be used for a given row of your feature matrix. An advanced use of cutoff times can be found in the :ref:`second part <flight-ct>` of the next section.

.. _flight-ti:

Advanced Scenarios
-------------------------
The :func:`Flights <demo.load_flight>` entityset is a prototypical example of a dataset where an individual row can happen over time. Each trip is recorded in a ``trip_logs`` entity, and has many times associated to it.

.. ipython:: python

    es_flight = ft.demo.load_flight(nrows=100)
    es_flight
    es_flight['trip_logs'].df.head(3)


For every trip we have real arrival and departure times and scheduled arrival and departure times.

With the columns we have, it would be problematic for the ``scheduled_dep_time``, to be the time index: flights are scheduled far in advance!  If the time index were set to the scheduled departure time, we wouldn't be able to know anything about the flight at all until it was boarded. 

However, it's possible to know many things about a trip six months or more before it takes off; the trip distance, carrier, flight number and even when a flight is supposed to leave and land are always known before we buy a ticket. Our ``time_index`` exists to reflect the reality that those can be known much before the scheduled departure time.

That being said, not all columns can be known at our time index six months in advance. If we were able to know the real arrival time of the plane before we booked, we would have great success in predicting delays! 

.. image:: ../images/flight_ti_1.png
   :width: 400 px
   :alt: flight time index diagram
   :align: center

In this diagram of a row, we have set the ``time_index`` to the time the flight was scheduled. However, any information about what happens to the flight after it departs is **invalid** for use at that time. If we were to use any of that information prior to when the flight lands, we would be leaking labels. 

While one option would be to remove that data from the entityset, a better option would be to use that data somehow. To that end, it's possible to set a ``secondary_time_index`` which can mark specific columns as available at a later date. The ``secondary_time_index`` of this row is set to the arrival time. 

.. image:: ../images/flight_ti_2.png
   :width: 400 px
   :alt: flight secondary time index diagram
   :align: center

By setting a ``secondary_time_index``, we can still use the delay information from a row, but only when they would become known. It's possible to know everything about how a trip went after it has arrived, so we can happily use that information at any time after the flight lands.

.. hint::

    It's often a good idea to use a secondary time index if your entityset has inline labels. If you know when the label would be valid for use, it's possible to automatically create very predictive features using historical labels.


As an exercise, take a minute to think about which of the twenty two columns here can be known at each time index. Which can be known 6 months in advance and which would be better to only learn after the flight lands?

.. ipython:: python

    es_flight['trip_logs']

+ These columns can be known at the ``time_index`` months before the flight: ``trip_log_id``, ``flight_date``, ``scheduled_dep_time``, ``scheduled_elapsed_time``, ``distance``, ``scheduled_arr_time``, ``time_index``, ``flight_id``

+ These only be known at the ``secondary_time_index``, after the flight has completed: ``dep_delay``, ``taxi_out``, ``taxi_in``, ``arr_delay``, ``air_time``, ``carrier_delay``, ``weather_delay``, ``national_airspace_delay``, ``security_delay``, ``late_aircraft_delay``, ``dep_time``, ``arr_time``, ``cancelled``, ``diverted``

An entity can have a third, hidden, time index called the ``last_time_index``. More details for that can be found in the `other temporal workflows <#training-window-and-the-last-time-index>`_ section.

.. _flight-ct:

Flight Predictions
~~~~~~~~~~~~~~~~~~~

Let's make features at some varying times in the flight example. Trip ``14`` is a flight from CLT to PHX on January 31 2017 and trip ``92`` is a flight from PIT to DFW on January 1. We can set any cutoff time before the flight is scheduled to depart, emulating how we would make the prediction at that point in time. 

We set two cutoff times for trip ``14`` at two different times: one which is more than a month before the flight and another which is only 5 days before. For trip ``92``, we'll only set one cutoff time three days before it is scheduled to leave. 

.. image:: ../images/flight_ct.png
   :width: 500 px
   :alt: flight cutoff time diagram
   :align: center

Our cutoff time dataframe looks like this:

.. ipython:: python

    ct_flight = pd.DataFrame()
    ct_flight['trip_log_id'] = [14, 14, 92]
    ct_flight['time'] = pd.to_datetime(['2016-12-28', 
                                        '2017-1-25',
                                        '2016-12-28'])
    ct_flight['label'] = [True, True, False]
    ct_flight

These instructions say to build two rows for trip ``14`` using data from different times and one row for trip ``92``. Here's how DFS handles those instructions:

.. ipython:: python

    fm, features = ft.dfs(entityset=es_flight, 
                          target_entity='trip_logs', 
                          cutoff_time=ct_flight, 
                          cutoff_time_in_index=True)
    fm[['label', 'flight_id', 'flights.MAX(trip_logs.arr_delay)', 'MONTH(scheduled_dep_time)', 'DAY(scheduled_dep_time)']]

There is a lot to unpack from this output:

1. Even though one id showed up twice, a row was made for every id-time pair in ``ct_flight``. The id and cutoff time were returned as the index of the feature matrix.
2. The output, and label, were sorted by the passed in ``time`` column. Because of the sorting, it's often helpful to pass in a label with the cutoff time dataframe so that it will remain sorted in the same way as the feature matrix. Any additional columns past the ``id`` and ``cutoff_time`` will not be used for making features.
3. The column ``flights.MAX(trip_logs.arr_delay)`` is not always defined. It can only have real values when there are historical flights to aggregate. Since we excluded the arrival delay of this particular flight, there are no values to use!

Notice that for trip ``14``, there wasn't historical data when we made the feature a month in advance, but there were flights from Charlotte to Phoenix before January 25 whose delay could be validly used. These are powerful features that are often excluded in manual processes because of how hard they are to make.

Other Settings
-------------------------

Training Window and the Last Time Index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training window in DFS limits the amount of past data that can be used while calculating a particular feature vector. In the same way that a cutoff time filters out data which appears after it, a training window will filter out data that appears too much earlier. Here's an example of a two hour training window:

.. ipython:: python

    window_fm, window_features = ft.dfs(entityset=es_mc,
                                        target_entity="customers",
                                        cutoff_time=ct,
                                        cutoff_time_in_index=True,
                                        training_window="2 hours")
    window_fm.head()

This works well for :class:`entities <Entity>` where an instance occurs at a single point in time. However, sometimes an instance can happen at many points in time.

For example, suppose a customer’s session has multiple transactions which can happen at different points in time. If we are trying to count the number of sessions a user had in a given time period, we often want to count all sessions that were active during the training window. To accomplish this, we need to not only know when a session starts, but when it ends. The last time that an instance appears in the data is stored as the ``last_time_index`` of an entity. We can compare the time index and the last time index of ``sessions``: 

.. ipython:: python

    es_mc['sessions'].df['session_start'].head()
    es_mc['sessions'].last_time_index.head()

It is possible to automatically add last time indexes to every entity in an :class:`EntitySet` by running :func:`EntitySet.add_last_time_indexes`. If a ``last_time_index`` has been set, Featuretools will check to see if the ``last_time_index`` is after the start of the training window. That, combined with the cutoff time, allows Deep Feature Synthesis to discover which data is relevant for a given training window.

.. _approximate:

Approximating features by rounding cutoff time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If there are a large number of unique cutoff times relative to the number of instances for which we are calculating features, this overhead can outweigh the time needed to calculate the features. Therefore, by reducing the number of unique cutoff times, we minimize the overhead from searching for and extracting data for feature calculations.

One way to decrease the number of unique cutoff times is to round cutoff times to an nearby earlier point in time. An earlier cutoff time is always valid for predictive modeling — it just means we’re not using some of the data we could potentially use while calculating that feature. In that way, we gain computational speed by losing some information.

To understand when approximation is useful, consider calculating features for a model to predict fraudulent credit card transactions. In this case, an important feature might be, "the average transaction amount for this card in the past". While this value can change every time there is a new transaction, updating it less frequently might not impact accuracy.

.. note::

    The bank BBVA used approximation when building a predictive model for credit card fraud using Featuretools. For more details, see the "Real-time deployment considerations" section of the `white paper <https://arxiv.org/pdf/1710.07709.pdf>`_ describing the work.

The frequency of approximation is controlled using the ``approximate`` parameter to DFS or :func:`calculate_feature_matrix`. For example, the following code would approximate aggregation features at 1 day intervals::

    fm = ft.calculate_feature_matrix(features=features,
                                     entityset=es_flight,
                                     cutoff_time=ct_flight,
                                     approximate="1 day")

In this computation, features that can be approximated will be calculated at 1 day intervals, while features that cannot be approximated (e.g "what is the destination of this flight?") will be calculated at the exact cutoff time.

Creating and Flattening a Feature Tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`make_temporal_cutoffs` function generates a series of equally spaced cutoff times from a given set of cutoff times and instance ids.
This function can be paired with DFS to create and flatten a feature tensor rather than making multiple feature matrices at different delays.

The function
takes in the the following parameters:

 * ``instance_ids (list, pd.Series, or np.ndarray)``: A list of instances.
 * ``cutoffs (list, pd.Series, or np.ndarray)``: An associated list of cutoff times.
 * ``window_size (str or pandas.DateOffset)``: The amount of time between each cutoff time in the created time series.
 * ``start (datetime.datetime or pd.Timestamp)``: The first cutoff time in the created time series.
 * ``num_windows (int)``: The number of cutoff times to create in the created time series.

Only two of the three options ``window_size``, ``start``, and ``num_windows`` need to be specified to uninquely determine an equally-spaced set of cutoff times at which to compute each instance.

If your cutoff times are the ones used above:

.. ipython:: python

    ct_flight

Then passing in ``window_size='1h'`` and ``num_windows=2`` makes one row an hour over the last two hours to produce the following new dataframe. The result can be directly passed into DFS to make features at the different time points. 

.. ipython:: python

    temporal_cutoffs = ft.make_temporal_cutoffs(ct['customer_id'],
                                                ct['time'],
                                                window_size='1h',
                                                num_windows=2)
    temporal_cutoffs
    fm, features = ft.dfs(entityset=es_mc,
                          target_entity='customers',
                          cutoff_time=temporal_cutoffs,
                          cutoff_time_in_index=True)
    fm

    
