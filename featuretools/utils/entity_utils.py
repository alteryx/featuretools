import warnings
from datetime import datetime

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes

from featuretools import variable_types as vtypes
from featuretools.utils.gen_utils import import_or_none, is_instance

ks = import_or_none('databricks.koalas')


def infer_variable_types(df, link_vars, variable_types, time_index, secondary_time_index):
    '''Infer variable types from dataframe

    Args:
        df (DataFrame): Input DataFrame
        link_vars (list[]): Linked variables
        variable_types (dict[str -> dict[str -> type]]) : An entity's
            variable_types dict maps string variable ids to types (:class:`.Variable`)
            or (type, kwargs) to pass keyword arguments to the Variable.
        time_index (str or None): Name of time_index column
        secondary_time_index (dict[str: [str]]): Dictionary of secondary time columns
            that each map to a list of columns that depend on that secondary time
    '''
    # TODO: set pk and pk types here
    inferred_types = {}
    vids_to_assume_datetime = [time_index]
    if len(list(secondary_time_index.keys())):
        vids_to_assume_datetime.append(list(secondary_time_index.keys())[0])
    inferred_type = vtypes.Unknown
    for variable in df.columns:
        if variable in variable_types:
            continue
        elif isinstance(df, dd.DataFrame):
            msg = 'Variable types cannot be inferred from Dask DataFrames, ' \
                  'use variable_types to provide type metadata for entity'
            raise ValueError(msg)
        elif is_instance(df, ks, 'DataFrame'):
            msg = 'Variable types cannot be inferred from Koalas DataFrames, ' \
                  'use variable_types to provide type metadata for entity'
            raise ValueError(msg)
        elif variable in vids_to_assume_datetime:
            if col_is_datetime(df[variable]):
                inferred_type = vtypes.Datetime
            else:
                inferred_type = vtypes.Numeric

        elif variable in link_vars:
            inferred_type = vtypes.Categorical

        elif df[variable].dtype == "object":
            if not len(df[variable]):
                inferred_type = vtypes.Categorical
            elif col_is_datetime(df[variable]):
                inferred_type = vtypes.Datetime
            else:
                inferred_type = vtypes.Categorical

                # heuristics to predict this some other than categorical
                sample = df[variable].sample(min(10000, len(df[variable])))

                # catch cases where object dtype cannot be interpreted as a string
                try:
                    avg_length = sample.str.len().mean()
                    if avg_length > 50:
                        inferred_type = vtypes.NaturalLanguage
                except AttributeError:
                    pass

        elif df[variable].dtype == "bool":
            inferred_type = vtypes.Boolean

        elif pdtypes.is_categorical_dtype(df[variable].dtype):
            inferred_type = vtypes.Categorical

        elif pdtypes.is_numeric_dtype(df[variable].dtype):
            inferred_type = vtypes.Numeric

        elif col_is_datetime(df[variable]):
            inferred_type = vtypes.Datetime

        elif len(df[variable]):
            n = min(10000, len(df[variable]))
            sample = df[variable].sample(n=n)
            n_unique = len(sample.unique())
            percent_unique = n_unique / sample.size

            if percent_unique < .05:
                inferred_type = vtypes.Categorical
            else:
                inferred_type = vtypes.Numeric

        inferred_types[variable] = inferred_type

    return inferred_types


def convert_all_variable_data(df, variable_types):
    """Convert all dataframes' variables to different types.
    """
    for var_id, desired_type in variable_types.items():
        type_args = {}
        if isinstance(desired_type, tuple):
            # grab args before assigning type
            type_args = desired_type[1]
            desired_type = desired_type[0]

        if var_id not in df.columns:
            raise LookupError("Variable ID %s not in DataFrame" % (var_id))
        current_type = df[var_id].dtype.name

        if issubclass(desired_type, vtypes.Numeric) and \
                current_type not in vtypes.PandasTypes._pandas_numerics:
            df = convert_variable_data(df=df,
                                       column_id=var_id,
                                       new_type=desired_type,
                                       **type_args)

        if issubclass(desired_type, vtypes.Discrete) and \
                current_type not in [vtypes.PandasTypes._categorical]:
            df = convert_variable_data(df=df,
                                       column_id=var_id,
                                       new_type=desired_type,
                                       **type_args)

        if issubclass(desired_type, vtypes.Datetime) and \
                current_type not in vtypes.PandasTypes._pandas_datetimes:
            df = convert_variable_data(df=df,
                                       column_id=var_id,
                                       new_type=desired_type,
                                       **type_args)

        # Fill in any single `NaN` values in LatLong variables with a tuple
        if issubclass(desired_type, vtypes.LatLong) and isinstance(df[var_id], pd.Series) and df[var_id].hasnans:
            df[var_id] = replace_latlong_nan(df[var_id])
            warnings.warn("LatLong columns should contain only tuples. All single 'NaN' values in column '{}' have been replaced with '(NaN, NaN)'.".format(var_id))

    return df


def convert_variable_data(df, column_id, new_type, **kwargs):
    """Convert dataframe's variable to different type.
    """
    empty = df[column_id].empty if isinstance(df, pd.DataFrame) else False
    if empty:
        return df
    if new_type == vtypes.Numeric:
        if isinstance(df, dd.DataFrame):
            df[column_id] = dd.to_numeric(df[column_id], errors='coerce')
        elif is_instance(df, ks, 'DataFrame'):
            df[column_id] = ks.to_numeric(df[column_id])
        else:
            orig_nonnull = df[column_id].dropna().shape[0]
            df[column_id] = pd.to_numeric(df[column_id], errors='coerce')
            # This will convert strings to nans
            # If column contained all strings, then we should
            # just raise an error, because that shouldn't have
            # been converted to numeric
            nonnull = df[column_id].dropna().shape[0]
            if nonnull == 0 and orig_nonnull != 0:
                raise TypeError("Attempted to convert all string column {} to numeric".format(column_id))
    elif issubclass(new_type, vtypes.Datetime):
        format = kwargs.get("format", None)
        # TODO: if float convert to int?
        if isinstance(df, dd.DataFrame):
            df[column_id] = dd.to_datetime(df[column_id], format=format,
                                           infer_datetime_format=True)
        elif is_instance(df, ks, 'DataFrame'):
            df[column_id] = ks.to_datetime(df[column_id], format=format,
                                           infer_datetime_format=True)
        else:
            df[column_id] = pd.to_datetime(df[column_id], format=format,
                                           infer_datetime_format=True)
    elif new_type == vtypes.Boolean:
        map_dict = {kwargs.get("true_val", True): True,
                    kwargs.get("false_val", False): False,
                    True: True,
                    False: False}
        # TODO: what happens to nans?
        df[column_id] = df[column_id].map(map_dict).astype(np.bool)
    elif not issubclass(new_type, vtypes.Discrete):
        raise Exception("Cannot convert column %s to %s" %
                        (column_id, new_type))
    return df


def get_linked_vars(entity):
    """Return a list with the entity linked variables.
    """
    link_relationships = [r for r in entity.entityset.relationships
                          if r.parent_entity.id == entity.id or
                          r.child_entity.id == entity.id]
    link_vars = [v.id for rel in link_relationships
                 for v in [rel.parent_variable, rel.child_variable]
                 if v.entity.id == entity.id]
    return link_vars


def col_is_datetime(col):
    # check if dtype is datetime - use .head() when getting first value
    # in case column is a dask Series
    if (col.dtype.name.find('datetime') > -1 or
            (len(col) and isinstance(col.head(1).iloc[0], datetime))):
        return True

    # if it can be casted to numeric, it's not a datetime
    dropped_na = col.dropna()
    try:
        pd.to_numeric(dropped_na, errors='raise')
    except (ValueError, TypeError):
        # finally, try to cast to datetime
        if col.dtype.name.find('str') > -1 or col.dtype.name.find('object') > -1:
            try:
                pd.to_datetime(dropped_na, errors='raise')
            except Exception:
                return False
            else:
                return True

    return False


def replace_latlong_nan(values):
    """replace a single `NaN` value with a tuple: `(np.nan, np.nan)`"""
    return values.where(values.notnull(), pd.Series([(np.nan, np.nan)] * len(values)))
