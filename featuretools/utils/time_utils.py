import pandas as pd


def make_temporal_cutoffs(instance_ids,
                          cutoffs,
                          window_size=None,
                          num_windows=None,
                          start=None):
    '''Makes a set of equally spaced cutoff times prior to a set of input cutoffs and instance ids.

    If window_size and num_windows are provided, then num_windows of size window_size will be created
    prior to each cutoff time

    If window_size and a start list is provided, then a variable number of windows will be created prior
    to each cutoff time, with the corresponding start time as the first cutoff.

    If num_windows and a start list is provided, then num_windows of variable size will be created prior
    to each cutoff time, with the corresponding start time as the first cutoff

    Args:
        instance_ids (list, np.ndarray, or pd.Series): list of instance ids. This function will make a
            new datetime series of multiple cutoff times for each value in this array.
        cutoffs (list, np.ndarray, or pd.Series): list of datetime objects associated with each instance id.
            Each one of these will be the last time in the new datetime series for each instance id
        window_size (pd.Timedelta, optional): amount of time between each datetime in each new cutoff series
        num_windows (int, optional): number of windows in each new cutoff series
        start (list, optional): list of start times for each instance id
    '''
    if (window_size is not None and
            num_windows is not None and
            start is not None):
        raise ValueError("Only supply 2 of the 3 optional args, window_size, num_windows and start")
    out = []
    for i, id_time in enumerate(zip(instance_ids, cutoffs)):
        _id, time = id_time
        _window_size = window_size
        _start = None
        if start is not None:
            if window_size is None:
                _window_size = (time - start[i]) / (num_windows - 1)
            else:
                _start = start[i]
        to_add = pd.DataFrame()
        to_add["time"] = pd.date_range(end=time,
                                       periods=num_windows,
                                       freq=_window_size,
                                       start=_start)
        to_add['instance_id'] = [_id] * len(to_add['time'])
        out.append(to_add)
    return pd.concat(out).reset_index(drop=True)


def convert_time_units(secs,
                       unit):
    '''
    Converts a time specified in seconds to a time in the given units

    Args:
        secs (integer): number of seconds. This function will convert the units of this number.
        unit(str): units to be converted to.
            acceptable values: years, months, days, hours, minutes, seconds, milliseconds, nanoseconds
    '''
    unit_divs = {'years': 31540000,
                 'months': 2628000,
                 'days': 86400,
                 'hours': 3600,
                 'minutes': 60,
                 'seconds': 1,
                 'milliseconds': 0.001,
                 'nanoseconds': 0.000000001}
    if unit not in unit_divs:
        raise ValueError("Invalid unit given, make sure it is plural")

    return secs / (unit_divs[unit])
