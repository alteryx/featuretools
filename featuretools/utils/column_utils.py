import pandas as pd


def get_top_values(column, num_x, dropna=False):
    """Get the top, most frequency values, according to a given integer.

    Args:
        column (pd.Series): data to use find most frequent

        num_x (int): the number of top values to retrieve.

        dropna (bool): determines whether to remove NaN values when
            finding frequency. Defaults to False

    Returns:
        top_list (list(dict)): a list of dictionary with keys `count` and
            `value`
    """
    frequencies = column.value_counts(dropna=dropna)
    df = frequencies[:num_x].reset_index()
    df.columns = ["value", "count"]
    top_lt = list(df.to_dict(orient="index").values())
    top_lt = sorted(top_lt, key=lambda i: (i["count"], i["value"]))
    return top_lt


def get_time_values(column, num_x, clip_by="date", dropna=False, ascending=True):
    """Get the most frequent, recent values for a given datetime column.

    Args:
        column (pd.Series): data to use find most frequent

        num_x (int): the number of recent values to retrieve.

        clip_by (str): the recent values are bucketed. This value determines
            how to bucket them. Defaults to `date`. Valid values are attributes
            of datetime

        dropna (bool): determines whether to remove NaN values when
            finding recent datetimes (used for frequency count).
            Defaults to False

        ascending (bool): specify return order of the recent values. True means recent values,
            False means by oldest values.

    Returns:
        recent_list (list(dict)): a list of dictionary with keys `count` and
            `value`
    """
    datetimes = pd.to_datetime(column, infer_datetime_format=True, errors="coerce")
    datetimes = getattr(datetimes.dt, "date")
    frequencies = datetimes.value_counts(dropna=dropna)
    values = frequencies.sort_index(ascending=ascending)[:num_x]
    df = values.reset_index()
    df.columns = ["value", "count"]
    return list(df.to_dict(orient="index").values())
