import importlib.util
import os
from inspect import isclass

import holidays
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

import featuretools
from featuretools.primitives.base import (
    AggregationPrimitive,
    PrimitiveBase,
    TransformPrimitive
)
from featuretools.utils.gen_utils import Library, find_descendents


# returns all aggregation primitives, regardless of compatibility
def get_aggregation_primitives():
    aggregation_primitives = set([])
    for attribute_string in dir(featuretools.primitives):
        attribute = getattr(featuretools.primitives, attribute_string)
        if isclass(attribute):
            if issubclass(attribute,
                          featuretools.primitives.AggregationPrimitive):
                if attribute.name:
                    aggregation_primitives.add(attribute)
    return {prim.name.lower(): prim for prim in aggregation_primitives}


# returns all transform primitives, regardless of compatibility
def get_transform_primitives():
    transform_primitives = set([])
    for attribute_string in dir(featuretools.primitives):
        attribute = getattr(featuretools.primitives, attribute_string)
        if isclass(attribute):
            if issubclass(attribute,
                          featuretools.primitives.TransformPrimitive):
                if attribute.name:
                    transform_primitives.add(attribute)
    return {prim.name.lower(): prim for prim in transform_primitives}


def list_primitives():
    trans_names, trans_primitives, valid_inputs, return_type = _get_names_primitives(get_transform_primitives)
    trans_dask = [Library.DASK in primitive.compatibility for primitive in trans_primitives]
    trans_spark = [Library.SPARK in primitive.compatibility for primitive in trans_primitives]
    transform_df = pd.DataFrame({'name': trans_names,
                                 'description': _get_descriptions(trans_primitives),
                                 'dask_compatible': trans_dask,
                                 'spark_compatible': trans_spark,
                                 'valid_inputs': valid_inputs,
                                 'return_type': return_type})
    transform_df['type'] = 'transform'

    agg_names, agg_primitives, valid_inputs, return_type = _get_names_primitives(get_aggregation_primitives)
    agg_dask = [Library.DASK in primitive.compatibility for primitive in agg_primitives]
    agg_spark = [Library.SPARK in primitive.compatibility for primitive in agg_primitives]
    agg_df = pd.DataFrame({'name': agg_names,
                           'description': _get_descriptions(agg_primitives),
                           'dask_compatible': agg_dask,
                           'spark_compatible': agg_spark,
                           'valid_inputs': valid_inputs,
                           'return_type': return_type})
    agg_df['type'] = 'aggregation'

    columns = ['name', 'type', 'dask_compatible', 'spark_compatible', 'description', 'valid_inputs', 'return_type']
    return pd.concat([agg_df, transform_df], ignore_index=True)[columns]


def get_default_aggregation_primitives():
    agg_primitives = [featuretools.primitives.Sum,
                      featuretools.primitives.Std,
                      featuretools.primitives.Max,
                      featuretools.primitives.Skew,
                      featuretools.primitives.Min,
                      featuretools.primitives.Mean,
                      featuretools.primitives.Count,
                      featuretools.primitives.PercentTrue,
                      featuretools.primitives.NumUnique,
                      featuretools.primitives.Mode]
    return agg_primitives


def get_default_transform_primitives():
    # featuretools.primitives.TimeSince
    trans_primitives = [featuretools.primitives.Age,
                        featuretools.primitives.Day,
                        featuretools.primitives.Year,
                        featuretools.primitives.Month,
                        featuretools.primitives.Weekday,
                        featuretools.primitives.Haversine,
                        featuretools.primitives.NumWords,
                        featuretools.primitives.NumCharacters]
    return trans_primitives


def _get_descriptions(primitives):
    descriptions = []
    for prim in primitives:
        description = ''
        if prim.__doc__ is not None:
            description = prim.__doc__.split("\n")[0]
        descriptions.append(description)
    return descriptions


def _get_names_primitives(primitive_func):
    names = []
    primitives = []
    valid_inputs = []
    return_type = []
    for name, primitive in primitive_func().items():
        names.append(name)
        primitives.append(primitive)
        input_types = _get_unique_input_types(primitive.input_types)
        valid_inputs.append(', '.join(input_types))
        return_type.append(getattr(primitive.return_type, '__name__', None))
    return names, primitives, valid_inputs, return_type


def _get_unique_input_types(input_types):
    types = set()
    for input_type in input_types:
        if isinstance(input_type, list):
            types |= _get_unique_input_types(input_type)
        else:
            types.add(str(input_type))
    return types


def list_primitive_files(directory):
    """returns list of files in directory that might contain primitives"""
    files = os.listdir(directory)
    keep = []
    for path in files:
        if not check_valid_primitive_path(path):
            continue
        keep.append(os.path.join(directory, path))
    return keep


def check_valid_primitive_path(path):
    if os.path.isdir(path):
        return False

    filename = os.path.basename(path)

    if filename[:2] == "__" or filename[0] == "." or filename[-3:] != ".py":
        return False

    return True


def load_primitive_from_file(filepath):
    """load primitive objects in a file"""
    module = os.path.basename(filepath)[:-3]
    # TODO: what is the first argument"?
    spec = importlib.util.spec_from_file_location(module, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    primitives = []
    for primitive_name in vars(module):
        primitive_class = getattr(module, primitive_name)
        if (isclass(primitive_class) and
                issubclass(primitive_class, PrimitiveBase) and
                primitive_class not in (AggregationPrimitive,
                                        TransformPrimitive)):
            primitives.append((primitive_name, primitive_class))

    if len(primitives) == 0:
        raise RuntimeError("No primitive defined in file %s" % filepath)
    elif len(primitives) > 1:
        raise RuntimeError("More than one primitive defined in file %s" % filepath)

    return primitives[0]


def serialize_primitive(primitive):
    """build a dictionary with the data necessary to construct the given primitive"""
    args_dict = {name: val for name, val in primitive.get_arguments()}
    cls = type(primitive)
    return {
        'type': cls.__name__,
        'module': cls.__module__,
        'arguments': args_dict,
    }


class PrimitivesDeserializer(object):
    """
    This class wraps a cache and a generator which iterates over all primitive
    classes. When deserializing a primitive if it is not in the cache then we
    iterate until it is found, adding every seen class to the cache. When
    deseriazing the next primitive the iteration resumes where it left off. This
    means that we never visit a class more than once.
    """

    def __init__(self):
        self.class_cache = {}  # (class_name, module_name) -> class
        self.primitive_classes = find_descendents(PrimitiveBase)

    def deserialize_primitive(self, primitive_dict):
        """
        Construct a primitive from the given dictionary (output from
        serialize_primitive).
        """
        class_name = primitive_dict['type']
        module_name = primitive_dict['module']
        cache_key = (class_name, module_name)

        if cache_key in self.class_cache:
            cls = self.class_cache[cache_key]
        else:
            cls = self._find_class_in_descendants(cache_key)

            if not cls:
                raise RuntimeError('Primitive "%s" in module "%s" not found' %
                                   (class_name, module_name))

        arguments = primitive_dict['arguments']
        return cls(**arguments)

    def _find_class_in_descendants(self, search_key):
        for cls in self.primitive_classes:
            cls_key = (cls.__name__, cls.__module__)
            self.class_cache[cls_key] = cls

            if cls_key == search_key:
                return cls


def _roll_series_with_gap(series, window_size, gap=0, min_periods=1):
    """Provide rolling window calculations where the windows are determined using both a gap parameter
    that indicates the amount of time between each instance and its window and a window length parameter
    that determines the amount of data in each window.

    Args:
        series (Series): The series over which rolling windows will be created. Must be numeric in nature
            and have a DatetimeIndex.
        window_size (int, string): Specifies the amount of data included in each window.
            If an integer is provided, will correspond to a number of rows. For data with a uniform sampling frequency,
            for example of one day, the window_length will correspond to a period of time, in this case,
            7 days for a window_length of 7.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time that each window should span.
            The list of available offset aliases, can be found at
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        gap (int, string, optional): Specifies a gap backwards from each instance before the
            window of usable data begins. If an integer is provided, will correspond to a number of rows.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time between a target instance and the beginning of its window.
            Defaults to 0, which will include the target instance in the window.
        min_periods (int, optional): Minimum number of observations required for performing calculations
            over the window. Can only be as large as window_length when window_length is an integer.
            When window_length is an offset alias string, this limitation does not exist, but care should be taken
            to not choose a min_periods that will always be larger than the number of observations in a window.
            Defaults to 1.

    Returns:
        pandas.core.window.rolling.Rolling: The Rolling object for the series passed in.

    Note:
        Certain operations, like `pandas.core.window.rolling.Rolling.count` that can be performed
        on the Rolling object returned here may treat NaNs as periods to include in window calculations.
        So a window [NaN, 1, 3]  when `min_periods=3` will proceed with count, saying there are three periods
        but only two values and would return count=2. The calculation `max` on the other hand,
        would say that there are not three periods in that window and would return max=NaN.
        Most rolling calculations act this way. The implication of that here is that in order to
        achieve the gap, we insert NaNs at the beinning of the series, which would cause `count` to calculate
        on windows that technically should not have the correct number of periods. In the RollingCount primitive,
        we handle this case manually, replacing those values with NaNs. Any primitive that uses this function
        should determine whether this kind of handling is also necessary.

    Note:
        Only offset aliases with fixed frequencies can be used when defining gap and window_length.
        This means that aliases such as `M` or `W` cannot be used, as they can indicate different
        numbers of days. ('M', because different months are different numbers of days;
        'W' because week will indicate a certain day of the week, like W-Wed, so that will
        indicate a different number of days depending on the anchoring date.)

    Note:
        When using an offset alias to define `gap`, an offset alias must also be used to define `window_size`.
        This limitation does not exist when using an offset alias to define `window_size`. In fact,
        if the data has a uniform sampling frequency, it is preferable to use a numeric `gap` as it is more
        efficient.

    """
    _check_window_size(window_size)
    _check_gap(window_size, gap)

    # Workaround for pandas' bug: https://github.com/pandas-dev/pandas/issues/43016
    # Can remove when upgraded to pandas 1.4.0
    if str(series.dtype) == 'Int64':
        series = series.astype('float64')

    functional_window_length = window_size
    if isinstance(gap, str):
        # Add the window_size and gap so that the rolling operation correctly takes gap into account.
        # That way, we can later remove the gap rows in order to apply the primitive function
        # to the correct window
        functional_window_length = to_offset(window_size) + to_offset(gap)
    elif gap > 0:
        # When gap is numeric, we can apply a shift to incorporate gap right now
        # since the gap will be the same number of rows for the whole dataset
        series = series.shift(gap)

    return series.rolling(functional_window_length, min_periods)


def _get_rolled_series_without_gap(window, gap_offset):
    """Applies the gap offset_string to the rolled window, returning a window
    that is the correct length of time away from the original instance.

     Args:
        window (Series): A rolling window that includes both the window length and gap spans of time.
        gap_offset (string): The pandas offset alias that determines how much time at the end of the window
            should be removed.

    Returns:
        Series: The window with gap rows removed
    """
    if not len(window):
        return window

    window_start_date = window.index[0]
    window_end_date = window.index[-1]

    gap_bound = window_end_date - to_offset(gap_offset)

    # If the gap is larger than the series, no rows are left in the window
    if gap_bound < window_start_date:
        return pd.Series()

    # Only return the rows that are within the offset's bounds
    return window[window.index <= gap_bound]


def _apply_roll_with_offset_gap(window, gap_offset, reducer_fn, min_periods):
    """Takes in a series to which an offset gap will be applied, removing however many
    rows fall under the gap before applying the reducing function.

    Args:
        window (Series):  A rolling window that includes both the window length and gap spans of time.
        gap_offset (string): The pandas offset alias that determines how much time at the end of the window
            should be removed.
        reducer_fn (callable[Series -> float]): The function to be applied to the window in order to produce
            the aggregate that will be included in the resulting feature.
        min_periods (int): Minimum number of observations required for performing calculations
            over the window.

    Returns:
        float: The aggregate value to be used as a feature value.
    """
    window = _get_rolled_series_without_gap(window, gap_offset)

    if min_periods is None:
        min_periods = 1

    if len(window) < min_periods or not len(window):
        return np.nan

    return reducer_fn(window)


def _check_window_size(window_size):
    # Window length must either be a valid offset alias
    if isinstance(window_size, str):
        try:
            to_offset(window_size)
        except ValueError:
            raise ValueError(f"Cannot roll series. The specified window length, {window_size}, is not a valid offset alias.")
    # Or an integer greater than zero
    elif isinstance(window_size, int):
        if window_size <= 0:
            raise ValueError("Window length must be greater than zero.")
    else:
        raise TypeError("Window length must be either an offset string or an integer.")


def _check_gap(window_size, gap):
    # Gap must either be a valid offset string that also has an offset string window length
    if isinstance(gap, str):
        if not isinstance(window_size, str):
            raise TypeError(f"Cannot roll series with offset gap, {gap}, and numeric window length, {window_size}. "
                            "If an offset alias is used for gap, the window length must also be defined as an offset alias. "
                            "Please either change gap to be numeric or change window length to be an offset alias.")
        try:
            to_offset(gap)
        except ValueError:
            raise ValueError(f"Cannot roll series. The specified gap, {gap}, is not a valid offset alias.")
    # Or an integer greater than or equal to zero
    elif isinstance(gap, int):
        if gap < 0:
            raise ValueError("Gap must be greater than or equal to zero.")
    else:
        raise TypeError("Gap must be either an offset string or an integer.")


def _deconstrct_latlongs(latlongs):
    lats = np.array([x[0] if isinstance(x, tuple) else np.nan for x in latlongs])
    longs = np.array([x[1] if isinstance(x, tuple) else np.nan for x in latlongs])
    return lats, longs


def _haversine_calculate(lat_1s, lon_1s, lat_2s, lon_2s, unit):
    # https://stackoverflow.com/a/29546836/2512385
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon_1s, lat_1s, lon_2s, lat_2s])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2.0)**2
    radius_earth = 3958.7613
    if unit == 'kilometers':
        radius_earth = 6371.0088
    distances = radius_earth * 2 * np.arcsin(np.sqrt(a))
    return distances


class HolidayUtil:
    def __init__(self, country='US'):
        try:
            holidays.country_holidays(country=country)
        except NotImplementedError:
            available_countries = 'https://github.com/dr-prodigy/python-holidays#available-countries'
            error = 'must be one of the available countries:\n%s' % available_countries
            raise ValueError(error)

        self.federal_holidays = getattr(holidays, country)(years=range(1950, 2100))

    def to_df(self):
        holidays_df = pd.DataFrame(sorted(self.federal_holidays.items()),
                                   columns=['holiday_date', 'names'])
        holidays_df.holiday_date = holidays_df.holiday_date.astype('datetime64')
        return holidays_df
