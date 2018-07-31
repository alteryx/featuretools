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

Some of the most powerful functionality in Featuretools is the ability to do all of that automatically if you pass in a ``cutoff_time`` dataframe to :func:`Deep Feature Synthesis <dfs>`. In order for that to work though, the user will always have to explain the meaning behind the datetime columns you've provided.


Representing time in an EntitySet
----------------------------------------------------------
The :func:`Flights <demo.load_flight>` entityset gives a good example of why users need to specify the differences between different datetime columns.

.. ipython:: python

    import featuretools as ft
    es = ft.demo.load_flight(nrows=100)
    es

An individual trip is recorded in the ``trip_logs`` entity, and has many times associated to it.

.. ipython:: python

    es['trip_logs'].df.head(3)

We have arrival and departure times, scheduled arrival and departure times and a mysterious ``time index``.

For every :class:`Entity <Entity>` we make in a time-based problem, we should tell Featuretools when it is allowed to use the data from a certain row. The **time index** column should provide that information. Specifically: *the values of the time index are the first time anything from a row can be known*. Rows are treated as non-existent prior to the ``time index``.

Let's look at the example. The ``time index`` needs to be prior to any departure time if we want to make predictions about a specific trip, because it's not much good to tell a user if their flight is delayed as they're supposed to be boarding it. In reality, we know many things about a trip six months or more before it takes off. We always know the trip distance, carrier, flight number and when a flight is supposed to leave and land before we buy a ticket. The ``time index`` should reflect the reality that those can be known much before the scheduled departure time.

However, there are some columns that necessarily can not be known until much later. If we were able to know the real arrival time of the plane 6 months ahead of time, we would have great success in predicting delays! For this purpose, it's possible to set a ``secondary_time_index`` which can mark specific columns as not available until a later date. The ``secondary_time_index`` of this entity is set to the arrival time. As an exercise, take a minute to think about which of the twenty two columns here should be known at each time index.

.. ipython:: python

    es['trip_logs']

+ These columns can be known at the ``time_index`` months before the flight: ``trip_log_id``, ``flight_date``, ``scheduled_dep_time``, ``scheduled_elapsed_time``, ``distance``, ``scheduled_arr_time``, ``time_index``, ``flight_id``

+ These only be known at the ``secondary_time_index``, after the flight has completed: ``dep_delay``, ``taxi_out``, ``taxi_in``, ``arr_delay``, ``air_time``, ``carrier_delay``, ``weather_delay``, ``national_airspace_delay``, ``security_delay``, ``late_aircraft_delay``, ``dep_time``, ``arr_time``, ``cancelled``, ``diverted``

An :class:`Entity <Entity>` can have a third, hidden, time index called the ``last_time_index``. More details for that can be found in the `other temporal workflows <#training-window-and-the-last-time-index>`_ section.

Running DFS with a cutoff time dataframe
-----------------------------------------

A ``cutoff_time`` dataframe is a concise way of passing complicated instructions to :func:`Deep Feature Synthesis <dfs>`. Each row contains a reference id, a time and optionally a label. For every unique id-time pair, we will create a row of the feature matrix.

Let's do a short example. Trip ``14`` is a flight from CLT to PHX on Janulary 31 2017 and trip ``92`` is a flight from PIT to DFW on January 1. We can set any cutoff time before the actual flight, emulating how we would make the prediction at that point in time. We set two cutoff times for trip ``14``, one which is more than a month before the flight and another which is only 5 days before. For trip ``92``, we'll only set one cutoff time three days before it is scheduled to leave.

.. ipython:: python

    ct = pd.DataFrame({'trip_log_id': [14, 14, 92], 
                       'cutoff_time': pd.to_datetime(['2016-12-28', 
                                                      '2017-1-25',
                                                      '2016-12-28']),
                       'label': [True, True, False]})
    fm, features = ft.dfs(entityset=es, 
                          target_entity='trip_logs', 
                          cutoff_time=ct, 
                          cutoff_time_in_index=True)
    fm[['label', 'flight_id', 'flights.MAX(trip_logs.arr_delay)', 'MONTH(scheduled_dep_time)', 'DAY(scheduled_dep_time)']]

There is a lot to unpack from this output. Here are some highlights:

1. One row was made for every id-time pair in ``ct``. The id and cutoff time were returned as the index of the feature matrix.
2. The output, and label, were sorted by the passed in ``cutoff_time``. Because of the sorting, it's often helpful to pass in a label with the cutoff time dataframe so that it will remain sorted in the same way as the feature matrix. Any additional columns past the ``id`` and ``cutoff_time`` will not be used for making features.
3. The column ``flights.MAX(trip_logs.arr_delay)`` is not always defined. It can only have real values when there are historical flights to aggregate because we've excluded the arrival delay of this particular flight from use!

Notice that for trip ``14``, there wasn't historical data when we made the feature a month in advance, but there were flights from Charlotte to Phoenix before January 25 whose delay could be validly used. These are powerful features that are often excluded in manual processes because of how hard they are to make.

Other Temporal Workflows
-------------------------

Training Window and the Last Time Index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training window in DFS limits the amount of past data that can be used while calculating a particular feature vector. In the same way that a cutoff time filters out data which appears after it, a training window will filter out data that appears too much earlier. Here's an example of a one hour training window:

.. ipython:: python

    es_customer = ft.demo.load_mock_customer(return_entityset=True)
    window_fm, window_features = ft.dfs(entityset=es_customer,
                                        target_entity="customers",
                                        cutoff_time=cutoff_times,
                                        cutoff_time_in_index=True,
                                        training_window="1 hour")
    window_fm.head()


This works well for entities where an instance occurs at a single point in time. However, sometimes an instance can happen at many points in time.

For example, a customer’s session has multiple transactions which can happen at different points in time. If we are trying to count the number of sessions a user had in a given time period, we often want to count all sessions that had *any* transaction during the training window. To accomplish this, we need to not only know when a session starts, but when it ends. The last time that an instance appears in the data is stored as the ``last_time_index`` of an :class:`Entity`. We can compare the time index and the last time index of the ``sessions`` entity above:

.. ipython:: python

    es['sessions'].df['session_start'].head()
    es['sessions'].last_time_index.head()

Featuretools can automatically add last time indexes to every :class:`Entity` in an :class:`Entityset` by running ``EntitySet.add_last_time_indexes()``. If a ``last_time_index`` has been set, Featuretools will check to see if the ``last_time_index`` is after the start of the training window. That, combined with the cutoff time, allows DFS to discover which data is relevant for a given training window.


Flattening a Feature Tensor with make_temporal_cutoffs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`make_temporal_cutoffs` function generates a series of equally spaced cutoff times from a given set of cutoff times and instance ids.
This function can be paired with Deep Feature Synthesis to create and flatten a feature tensor rather than feature matrix and list of Featuretools feature definitions.

The function
takes in the the following parameters:

 * ``instance_ids (list, pd.Series, or np.ndarray)``: list of instances
 * ``cutoffs (list, pd.Series, or np.ndarray)``: associated list of cutoff times
 * ``window_size (str or pandas.DateOffset)``: amount of time between each cutoff time in the created time series
 * ``start (datetime.datetime or pd.Timestamp)``: first cutoff time in the created time series
 * ``num_windows (int)``: number of cutoff times in the created time series

Only two of the three options ``window_size``, ``start``, and ``num_windows`` need to be specified to uninquely determine an equally-spaced set of cutoff times at which to compute each instance.

Let's say the final cutoff times (which could be directly passed into :func:`dfs`) look like this:

.. ipython:: python

    cutoffs = pd.DataFrame({
      'customer_id': [13458, 13602, 15222],
      'cutoff_time': [pd.Timestamp('2011/12/15'), pd.Timestamp('2012/10/05'), pd.Timestamp('2012/01/25')]
    })

Then passing in ``window_size='3d'`` and ``num_windows=2`` produces the following cutoff times to be passed into DFS.

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
