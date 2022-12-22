import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Categorical, Datetime

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class PartOfDay(TransformPrimitive):
    """Determines the part of day of a datetime.

    Description:
        For a list of datetimes, determines the part of day the datetime
        falls into, based on the hour.
        If the hour falls from 4 to 5, the part of day is 'dawn'.
        If the hour falls from 6 to 7, the part of day is 'early morning'.
        If the hour falls from 8 to 10, the part of day is 'late morning'.
        If the hour falls from 11 to 13, the part of day is 'noon'.
        If the hour falls from 14 to 16, the part of day is 'afternoon'.
        If the hour falls from 17 to 19, the part of day is 'evening'.
        If the hour falls from 20 to 22, the part of day is 'night'.
        If the hour falls into 23, 24, or 1 to 3, the part of day is 'midnight'.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2020, 1, 11, 6, 2, 1),
        ...          datetime(2021, 3, 31, 4, 2, 1),
        ...          datetime(2020, 3, 4, 9, 2, 1)]
        >>> part_of_day = PartOfDay()
        >>> part_of_day(dates).tolist()
        ['early morning', 'dawn', 'late morning']
    """

    name = "part_of_day"
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={"category"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the part of day {} falls in"

    @staticmethod
    def construct_replacement_dict():
        tdict = dict()
        tdict[pd.NaT] = np.nan
        for hour in [4, 5]:
            tdict[hour] = "dawn"
        for hour in [6, 7]:
            tdict[hour] = "early morning"
        for hour in [8, 9, 10]:
            tdict[hour] = "late morning"
        for hour in [11, 12, 13]:
            tdict[hour] = "noon"
        for hour in [14, 15, 16]:
            tdict[hour] = "afternoon"
        for hour in [17, 18, 19]:
            tdict[hour] = "evening"
        for hour in [20, 21, 22]:
            tdict[hour] = "night"
        for hour in [23, 0, 1, 2, 3]:
            tdict[hour] = "midnight"
        return tdict

    def get_function(self):
        replacement_dict = self.construct_replacement_dict()

        def part_of_day(vals):
            ans = vals.dt.hour.replace(replacement_dict)
            return ans

        return part_of_day
