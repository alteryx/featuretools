import pandas as pd


def make_temporal_cutoffs(instance_ids,
                          cutoffs,
                          window_size=None,
                          num_windows=None,
                          start=None):
    '''Makes a set of equally spaced cutoff times prior to a set of input cutoffs and instance ids.

    Must specify 2 of the optional args:
    - window_size and num_windows
    - window_size and start

    Args:
        instance_ids (list, np.ndarray, or pd.Series): list of instance ids. This function will make a
            new datetime series of multiple cutoff times for each value in this array.
        cutoffs (list, np.ndarray, or pd.Series): list of datetime objects associated with each instance id.
            Each one of these will be the last time in the new datetime series for each instance id
        window_size (pd.Timedelta, optional): amount of time between each datetime in each new cutoff series
        num_windows (int, optional): number of windows in each new cutoff series
        start (Datetime, optional): first time in each new cutoff series
    '''
    out = []
    for _id, time in zip(instance_ids, cutoffs):
        to_add = pd.DataFrame()
        to_add["time"] = pd.date_range(end=time,
                                       periods=num_windows,
                                       freq=window_size,
                                       start=start)
        to_add['instance_id'] = [_id] * len(to_add['time'])
        out.append(to_add)
    return pd.concat(out).reset_index(drop=True)
