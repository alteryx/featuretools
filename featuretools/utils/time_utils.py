import pandas as pd


def make_temporal_cutoffs(instance_ids,
                          cutoffs,
                          window_size=None,
                          num_windows=None,
                          start=None):
    '''
    Must specify 2 of the optional args:
    - window_size and num_windows
    - window_size and start
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
