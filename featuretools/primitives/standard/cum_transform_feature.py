import uuid
from builtins import str

import numpy as np
import pandas as pd

from ..base.primitive_base import IdentityFeature, PrimitiveBase
from ..base.transform_primitive_base import TransformPrimitive
from .aggregation_primitives import Count, Max, Mean, Min, Sum
from .utils import apply_dual_op_from_feat

from featuretools.utils import is_string
from featuretools.utils.wrangle import _check_timedelta
from featuretools.variable_types import Id, Index, Numeric, TimeIndex
from featuretools.variable_types.variable import Discrete


class CumFeature(TransformPrimitive):
    allow_where = True
    agg_feature = None
    rolling_function = True
    uses_full_entity = True

    # Note: Any row with a nan value in the group by feature will have a
    # NaN value in the cumfeat

    # Todo: also passing the parent entity instead of the group_feat
    def __init__(self, base_feature, group_feature, time_index=None,
                 where=None, use_previous=None):
        """Summary

        Args:
            agg_feature (type): subclass of :class:`.AggregationPrimitive`;
                aggregation method being used.  This is passed by the
                constructors of the cumfeat subclasses
            base_feature (:class:`.PrimitiveBase` or :class:`.Variable`): Feature
                or variable calculated on
            group_feature (:class:`.PrimitiveBase` or :class:`.Variable`): Feature
                or variable used to group the rows before computation
            where (optional[:class:`.PrimitiveBase`]):
            use_previous (optional[:class:`.Timedelta`):
        """
        self.return_type = self.agg_feature.return_type

        base_feature = self._check_feature(base_feature)

        td_entity_id = None
        if is_string(use_previous):
            td_entity_id = base_feature.entity.id
        self.use_previous = _check_timedelta(
            use_previous, entity_id=td_entity_id)

        group_feature = self._check_feature(group_feature)
        self.group_feature = group_feature

        self.base_features = [base_feature, group_feature]

        if time_index is None:
            entity = base_feature.entity
            time_index = IdentityFeature(entity[entity.time_index])
        self.base_features += [time_index]

        if where is not None:
            self.where = where

        super(CumFeature, self).__init__(*self.base_features)

    def generate_name(self):
        where_str = u""
        use_prev_str = u""

        if self.where is not None:
            where_str = u" WHERE " + self.where.get_name()

        if self.use_previous is not None:
            use_prev_str = u", Last %s" % (self.use_previous.get_name())

        base_features_str = u"%s by %s" % \
            (self.base_features[0].get_name(), self.group_feature.get_name())

        return u"%s(%s%s%s)" % (self.name.upper(), base_features_str,
                                where_str, use_prev_str)

    def get_function(self):
        return pd_rolling_outer(self.rolling_func_name, self)


class CumSum(CumFeature):
    """Calculates the sum of previous values of an instance for each value in a time-dependent entity.
    """
    name = "cum_sum"
    rolling_func_name = "sum"
    default_value = 0
    agg_feature = Sum
    input_types = [[Numeric, Id, TimeIndex],
                   [Numeric, Discrete, TimeIndex]]


class CumMean(CumFeature):
    """Calculates the mean of previous values of an instance for each value in a time-dependent entity.
    """
    name = "cum_mean"
    rolling_func_name = "mean"
    default_value = 0
    agg_feature = Mean
    input_types = [[Numeric, Id, TimeIndex],
                   [Numeric, Discrete, TimeIndex]]


class CumCount(CumFeature):
    """Calculates the number of previous values of an instance for each value in a time-dependent entity.
    """
    name = "cum_count"
    rolling_func_name = "count"
    default_value = 0
    agg_feature = Count
    input_types = [Index, Discrete, TimeIndex]


class CumMax(CumFeature):
    """Calculates the max of previous values of an instance for each value in a time-dependent entity.
    """
    name = "cum_max"
    rolling_func_name = "max"
    default_value = 0
    agg_feature = Max
    input_types = [[Numeric, Id, TimeIndex],
                   [Numeric, Discrete, TimeIndex]]


class CumMin(CumFeature):
    """Calculates the min of previous values of an instance for each value in a time-dependent entity.
    """
    name = "cum_min"
    rolling_func_name = "min"
    default_value = 0
    agg_feature = Min
    input_types = [[Numeric, Id, TimeIndex],
                   [Numeric, Discrete, TimeIndex]]


def pd_rolling_outer(rolling_func, f):
    def pd_rolling(base_array, group_array, values_3=None, values_4=None):
        bf_name = f.base_features[0].get_name()
        entity = f.base_features[0].entity
        time_index = entity.time_index
        groupby = f.group_feature.get_name()
        timedelta = f.use_previous
        if timedelta is not None:
            if timedelta.is_absolute():
                timedelta = f.use_previous.get_pandas_timedelta()
                absolute = True
            else:
                timedelta = f.use_previous.value
                absolute = False
        df_dict = {bf_name: base_array, groupby: group_array}
        if timedelta:
            df_dict[time_index] = values_3
            if f.where:
                df_dict[f.where.get_name()] = values_4
        elif f.where:
            df_dict[f.where.get_name()] = values_3

        df = pd.DataFrame.from_dict(df_dict)

        if f.use_previous and not f.where:
            def apply_rolling(group):
                to_roll = group
                kwargs = {'window': timedelta,
                          'min_periods': 1}
                if absolute:
                    to_roll = to_roll[[bf_name, time_index]].sort_values(
                        time_index, kind='mergesort')
                    kwargs['on'] = time_index
                else:
                    to_roll = to_roll[bf_name]
                rolled = to_roll.rolling(**kwargs)
                rolled = getattr(rolled, rolling_func)()
                if absolute:
                    rolled = rolled[bf_name]
                return rolled
        elif not f.where:
            cumfuncs = {"count": "cumcount",
                        "sum": "cumsum",
                        "max": "cummax",
                        "min": "cummin",
                        "prod": "cumprod",
                        }
            if rolling_func in ["count", "sum", "max", "min"]:
                cumfunc = cumfuncs[rolling_func]
                grouped = df.groupby(groupby, sort=False, observed=True)[bf_name]
                applied = getattr(grouped, cumfunc)()
                # TODO: to produce same functionality as the rolling cases already
                # implemented, we add 1
                # We may want to consider changing this functionality to instead
                # return count of the *previous* events
                if rolling_func == "count":
                    applied += 1
                return applied
            else:
                def apply_rolling(group):
                    rolled = group[bf_name].expanding(min_periods=1)
                    return getattr(rolled, rolling_func)()
        elif f.use_previous and f.where:
            def apply_rolling(group):
                variable_data = [group[base.get_name()]
                                 for base in [f.where.left, f.where.right]
                                 if isinstance(base, PrimitiveBase)]
                mask = apply_dual_op_from_feat(f.where, *variable_data)
                to_roll = group[mask]
                kwargs = {'window': timedelta,
                          'min_periods': 1}
                if absolute:
                    output = pd.Series(f.default_value, index=group.index)
                    # mergesort is stable
                    to_roll = to_roll[[bf_name, time_index]].sort_values(
                        time_index, kind='mergesort')
                    kwargs['on'] = time_index
                else:
                    output = pd.Series(np.nan, index=group.index)
                    to_roll = to_roll[bf_name]
                rolled = to_roll.rolling(**kwargs)
                rolled = getattr(rolled, rolling_func)()
                if absolute:
                    rolled = rolled[bf_name]
                    output[mask] = rolled
                else:
                    output[mask] = rolled
                    # values filtered out by the Where statement
                    # should have their values be w
                    output.fillna(method='ffill', inplace=True)
                    # first value might still be nan
                    if pd.isnull(output.iloc[0]):
                        output.fillna(0, inplace=True)
                return output
        elif f.where:
            def apply_rolling(group):
                variable_data = [group[base.get_name()]
                                 for base in [f.where.left, f.where.right]
                                 if isinstance(base, PrimitiveBase)]
                mask = apply_dual_op_from_feat(f.where, *variable_data)
                output = pd.Series(np.nan, index=group.index)
                rolled = group[mask][bf_name].expanding(min_periods=1)
                rolled = getattr(rolled, rolling_func)()
                output[mask] = rolled
                # values filtered out by the Where statement
                # should have their values be w
                output.fillna(method='ffill', inplace=True)
                # first value might still be nan
                if pd.isnull(output.iloc[0]):
                    output.fillna(0, inplace=True)
                return output

        new_index_name = str(uuid.uuid1())
        new_index = pd.RangeIndex(len(df), name=new_index_name)
        df.set_index(new_index, append=True, inplace=True)
        grouped = df.groupby(groupby, observed=True).apply(apply_rolling)
        original_index = pd.Series(np.nan, index=df.index)
        if isinstance(grouped, pd.DataFrame):
            if grouped.shape[0] == 0 or grouped.empty:
                return original_index.values
            else:
                grouped = pd.Series(grouped.values[0], index=grouped.columns)

        df.reset_index(new_index_name, inplace=True, drop=True)
        # case where some values of df[groupby] are nan
        # pandas groupby().apply() filters those out
        # and returns a series that's shorter than the original
        # we need to add these values to the original index to
        # preserve the length and these nan values
        grouped_index = grouped.index.get_level_values(new_index_name)
        original_index[grouped_index] = grouped.values
        return original_index.values
    return pd_rolling
