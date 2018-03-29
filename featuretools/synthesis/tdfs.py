import featuretools as ft
from featuretools.entityset import EntitySet
import pandas as pd


def tdfs(entities=None,
         relationships=None,
         entityset=None,
         target_entity=None,
         cutoffs=None,
         features_only=False,
         window_size=None,
         num_windows=None,
         start=None,
         **kwargs):
    '''
    Must specify 2 of the following optional args:
    - window_size and num_windows
    - window_size and start
    - num_windows and start

    **kwargs will be passed to underlying featuretools.dfs() call.
    Refer to featuretools documentation for a listing and explanation of
    all possible optional arguments.

    Args:
        entities (dict[str -> tuple(pd.DataFrame, str, str)]): Dictionary of
            entities. Entries take the format
            {entity id -> (dataframe, id column, (time_column))}.

        relationships (list[(str, str, str, str)]): List of relationships
            between entities. List items are a tuple with the format
            (parent entity id, parent variable, child entity id, child variable).

        entityset (EntitySet): An already initialized entityset. Required if
            entities and relationships are not defined.

        target_entity (str): Entity id of entity on which to make predictions.

        cutoffs (pd.DataFrame or Datetime): Specifies times at which to
            calculate each instance. Can either be a DataFrame with
            'instance_id' and 'time' columns, or a DataFrame with the name of the
            index variable in the target entity and a time column
            If the dataframe has more than two columns, any additional columns will be added to the resulting
            feature matrix.

        features_only (bool, optional): If True, returns the list of
            features without calculating the feature matrix. If False,
            cutoffs must be provided

        window_size (str or pd.Timedelta or pd.DateOffset, optional): amount of time between each cutoff time in the created time series

        start (datetime.datetime or pd.Timestamp, optional): first cutoff time in the created time series

        num_windows (int, optional): number of cutoff times in the created time series
    '''
    temporal_cutoffs = None
    if not features_only:
        if not isinstance(entityset, EntitySet):
            entityset = EntitySet("dfs", entities, relationships)
        index = entityset[target_entity].index
        instance_id_column = 'instance_id'
        if 'instance_id' in cutoffs.columns:
            instance_ids = cutoffs['instance_id']
        elif index in cutoffs:
            instance_ids = cutoffs[index]
            instance_id_column = index
        else:
            instance_ids = cutoffs.iloc[:, 0]
            instance_id_column = cutoffs.columns[0]
        time_column = 'time'
        if time_column not in cutoffs:
            not_instance_id = [c for c in cutoffs.columns
                               if c != instance_id_column]
            time_column = not_instance_id[0]
        times = cutoffs[time_column]
        temporal_cutoffs = make_temporal_cutoffs(instance_ids,
                                                 times,
                                                 window_size,
                                                 num_windows,
                                                 start)
    result = ft.dfs(entityset=entityset,
                    features_only=features_only,
                    cutoff_time=temporal_cutoffs,
                    target_entity=target_entity,
                    cutoff_time_in_index=True,
                    **kwargs)
    if not features_only:
        fm, fl = result
        return fm.sort_index(level=[entityset[target_entity].index,
                                    'time']), fl
    return result


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
