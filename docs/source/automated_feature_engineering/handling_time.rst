.. _handling-time:

.. currentmodule:: featuretools

Handling Time
=============

Time is a naturally relevant factor in many data science questions. Consider the following questions:

1. How much revenue will I bring in next month?
2. What is the expected delay on my next flight?
3. Will my user purchase an upgrade to their membership?

A good way to estimate how effective you are at predicting revenue would be to see how you would have done predicting it last month or the month before. You would similarly be interested in checking if you were able to predict the delay on your previous flight, or how good you are historically at detecting customers who would upgrade.

However, it would be immensely tricky to make features by hand for those predictions using only past data. For every row, you would need to find out the **prediction time** and then **cut off** any data in your dataset that happens after that time. Then, you could use the remaining valid data to make any features you like.

Some of the most powerful functionality in Featuretools is the ability to do all of that automatically if you pass in a :ref:`cutoff_time <cutoff-time>` dataframe to :func:`Deep Feature Synthesis <dfs>`. In order for that to work though, the user will always have to explain the meaning behind the datetime columns you've provided.


Representing time in an EntitySet
----------------------------------------------------------
Let's look at how time works in the :func:`Mock Customer <demo.load_mock_customer>` entityset.

.. ipython:: python

    import featuretools as ft
    es_mc = ft.demo.load_mock_customer(return_entityset=True, random_seed=0)
    es_mc['transactions'].df.head()

The ``transactions`` entity has one row for every transaction and a ``transaction_time`` for every row. The **time index** is the first time information from the row can be used. Most people would make the reasonable choice to set the ``transaction_time`` as the time index for the ``transactions`` entity. However, finding the time index is not always obvious. Let's look at the ``customers`` entity:

.. ipython:: python

    es_mc['customers'].df

Here we have two time columns ``join_date`` and ``date_of_birth``. While either column would be useful for making features, the ``join_date`` is the date when we learn that the customer exists, so it should be used as the time index. Specifically: *the time index is the first time anything from a row can be known to the dataset owner*. Rows are treated as non-existant prior to the time index. 

.. _cutoff_time:

Running DFS with a cutoff time dataframe
-----------------------------------------

A **cutoff_time** dataframe is a concise way of passing complicated instructions to :func:`Deep Feature Synthesis <dfs>` (DFS). Each row contains a reference id, a time and optionally a label. For every unique id-time pair, we will create a row of the feature matrix.

Let's do a short example. We can make features for customers ``1``, ``2`` and ``3``. We can set the cutoff time any time after the time index, emulating how you'd make a prediction at that point in time. In this case, we're making predictions for all three customers at the same time, 12/1/2014.

.. ipython:: python

    ct = pd.DataFrame({'customer_id': [1, 2, 3], 
                       'time': pd.to_datetime(['2014-1-2', 
                                               '2014-1-2',
                                               '2014-1-2']),
                       'label': [True, True, False]})
    fm, features = ft.dfs(entityset=es_mc, 
                          target_entity='customers', 
                          cutoff_time=ct, 
                          cutoff_time_in_index=True)
    fm

We made 74 features for the three customers using only data whose time index was before December 1st 2014. When the cutoff times are changed, the row will be made as if we don't know anything after that point in time. 


Making features from your labels
-------------------------------------------------------
The :func:`Flights <demo.load_flight>` entityset gives a more pressing example of why users need to specify the differences between different datetime columns. By paying attention to those time columns, we can make powerful features using historical delays.

.. ipython:: python

    import featuretools as ft
    es_flight = ft.demo.load_flight(nrows=100)
    es_flight

An individual trip is recorded in the ``trip_logs`` entity, and has many times associated to it.

.. ipython:: python

    es_flight['trip_logs'].df.head(3)

We have arrival and departure times, scheduled arrival and departure times and a mysterious ``time index``.

For every :class:`Entity <Entity>` we make in a time-based problem, we should tell Featuretools when it is allowed to use the data from a certain row. The time index column should provide that information. 

Let's look at the example. For the existing columns, it would be problematic for the ``scheduled_dep_time``, to be the time index. For one, sometimes the actual departure time is *before* it, so it is not always the earliest time in our dataframe. However, there's a reality-based reason we would want an earlier time: flights are scheduled far in advance! We know many things about a trip six months or more before it takes off; the trip distance, carrier, flight number and even when a flight is supposed to leave and land are always known before we buy a ticket. The point of the time index construct is to reflect the reality that those can be known much before the scheduled departure time.

However, not all columns can be known six months in advance. If we were able to know the real arrival time of the plane before we booked, we would have great success in predicting delays! For this purpose, it's possible to set a ``secondary_time_index`` which can mark specific columns as not available until a later date. The ``secondary_time_index`` of this entity is set to the arrival time. As an exercise, take a minute to think about which of the twenty two columns here should be known at each time index.

.. ipython:: python

    es_flight['trip_logs']

+ These columns can be known at the ``time_index`` months before the flight: ``trip_log_id``, ``flight_date``, ``scheduled_dep_time``, ``scheduled_elapsed_time``, ``distance``, ``scheduled_arr_time``, ``time_index``, ``flight_id``

+ These only be known at the ``secondary_time_index``, after the flight has completed: ``dep_delay``, ``taxi_out``, ``taxi_in``, ``arr_delay``, ``air_time``, ``carrier_delay``, ``weather_delay``, ``national_airspace_delay``, ``security_delay``, ``late_aircraft_delay``, ``dep_time``, ``arr_time``, ``cancelled``, ``diverted``

An Entity can have a third, hidden, time index called the ``last_time_index``. More details for that can be found in the `other temporal workflows <#training-window-and-the-last-time-index>`_ section.


Let's do a short example. Trip ``14`` is a flight from CLT to PHX on Janulary 31 2017 and trip ``92`` is a flight from PIT to DFW on January 1. We can set any cutoff time before the flight is scheduled to depart, emulating how we would make the prediction at that point in time. We set two cutoff times for trip ``14``, one which is more than a month before the flight and another which is only 5 days before. For trip ``92``, we'll only set one cutoff time three days before it is scheduled to leave.

These instructions say to build two rows for trip ``14`` using data from different times and one row for trip ``92``. Let's see what happens when we run DFS:

.. ipython:: python

    ct_flight = pd.DataFrame({'trip_log_id': [14, 14, 92], 
                              'time': pd.to_datetime(['2016-12-28', 
                                                      '2017-1-25',
                                                      '2016-12-28']),
                              'label': [True, True, False]})
    fm, features = ft.dfs(entityset=es_flight, 
                          target_entity='trip_logs', 
                          cutoff_time=ct_flight, 
                          cutoff_time_in_index=True)
    fm[['label', 'flight_id', 'flights.MAX(trip_logs.arr_delay)', 'MONTH(scheduled_dep_time)', 'DAY(scheduled_dep_time)']]

There is a lot to unpack from this output. Here are some highlights:

1. Even though one id showed up twice, one row was made for every id-time pair in ``ct_flight``. The id and cutoff time were returned as the index of the feature matrix.
2. The output, and label, were sorted by the passed in ``time`` column. Because of the sorting, it's often helpful to pass in a label with the cutoff time dataframe so that it will remain sorted in the same way as the feature matrix as it was here. Any additional columns past the ``id`` and ``cutoff_time`` will not be used for making features.
3. The column ``flights.MAX(trip_logs.arr_delay)`` is not always defined. It can only have real values when there are historical flights to aggregate. Since we excluded the arrival delay of this particular flight, there are no values to use!

Notice that for trip ``14``, there wasn't historical data when we made the feature a month in advance, but there were flights from Charlotte to Phoenix before January 25 whose delay could be validly used. These are powerful features that are often excluded in manual processes because of how hard they are to make.

Other Temporal Workflows
-------------------------

Training Window and the Last Time Index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training window in DFS limits the amount of past data that can be used while calculating a particular feature vector. In the same way that a cutoff time filters out data which appears after it, a training window will filter out data that appears too much earlier. Here's an example of a one hour training window:

.. ipython:: python

    window_fm, window_features = ft.dfs(entityset=es_mc,
                                        target_entity="customers",
                                        cutoff_time=ct,
                                        cutoff_time_in_index=True,
                                        training_window="2 hours")
    window_fm.head()

This works well for :class:`entities <Entity>` where an instance occurs at a single point in time. However, sometimes an instance can happen at many points in time.

For example, suppose a customer’s session has multiple transactions which can happen at different points in time. If we are trying to count the number of sessions a user had in a given time period, we often want to count all sessions that had *any* transaction during the training window. To accomplish this, we need to not only know when a session starts, but when it ends. The last time that an instance appears in the data is stored as the ``last_time_index`` of an entity. We can compare the time index and the last time index of ``sessions``: 

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

    fm = ft.calculate_feature_matrix(entityset=es_mc
                                     features=feature_list,
                                     cutoff_time=ct,
                                     approximate="1 hour")

In this computation, features that can be approximated will be calculated at 1 day intervals, while features that cannot be approximated (e.g "is the current transaction > $50") will be calculated at the exact cutoff time.

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

If your cutoff times (which could be directly passed into DFS) look like this:

.. ipython:: python

    cutoffs = pd.DataFrame({
      'customer_id': [13458, 13602, 15222],
      'cutoff_time': [pd.Timestamp('2011/12/15'), pd.Timestamp('2012/10/05'), pd.Timestamp('2012/01/25')]
    })

Then passing in ``window_size='3d'`` and ``num_windows=2`` makes two rows for each instance over the last 3 days to produce the following new dataframe. The result can then be directly passed into DFS to make features at the different time points. 

.. ipython:: python

    temporal_cutoffs = ft.make_temporal_cutoffs(cutoffs['customer_id'],
                                                cutoffs['cutoff_time'],
                                                window_size='3d',
                                                num_windows=2)
    temporal_cutoffs

    entityset = ft.demo.load_retail()
    feature_tensor, feature_defs = ft.dfs(entityset=entityset,
                                          target_entity='customers',
                                          cutoff_time=temporal_cutoffs,
                                          cutoff_time_in_index=True,
                                          max_features=4)
    feature_tensor
