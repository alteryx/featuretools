from datetime import datetime

import pytest

from featuretools.primitives import TimeSince


def test_time_since():
    time_since = TimeSince()
    # class datetime.datetime(year, month, day[, hour[, minute[, second[, microsecond[,
    times = [datetime(2019, 3, 1, 0, 0, 0, 1),
             datetime(2019, 3, 1, 0, 0, 1, 0),
             datetime(2019, 3, 1, 0, 2, 0, 0)]
    cutoff_time = datetime(2019, 3, 1, 0, 0, 0, 0)
    values = time_since(array=times, time=cutoff_time)

    assert(list(map(int, values)) == [0, -1, -120])

    time_since = TimeSince(unit='nanoseconds')
    values = time_since(array=times, time=cutoff_time)
    assert(list(map(round, values)) == [-1000, -1000000000, -120000000000])

    time_since = TimeSince(unit='milliseconds')
    values = time_since(array=times, time=cutoff_time)
    assert(list(map(int, values)) == [0, -1000, -120000])

    time_since = TimeSince(unit='Milliseconds')
    values = time_since(array=times, time=cutoff_time)
    assert(list(map(int, values)) == [0, -1000, -120000])

    time_since = TimeSince(unit='Years')
    values = time_since(array=times, time=cutoff_time)
    assert(list(map(int, values)) == [0, 0, 0])

    times_y = [datetime(2019, 3, 1, 0, 0, 0, 1),
               datetime(2020, 3, 1, 0, 0, 1, 0),
               datetime(2017, 3, 1, 0, 0, 0, 0)]

    time_since = TimeSince(unit='Years')
    values = time_since(array=times_y, time=cutoff_time)
    assert(list(map(int, values)) == [0, -1, 1])

    error_text = 'Invalid unit given, make sure it is plural'
    with pytest.raises(ValueError, match=error_text):
        time_since = TimeSince(unit='na')
        time_since(array=times, time=cutoff_time)
