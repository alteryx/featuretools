from featuretools.primitives.standard.aggregation.all_primitive import All
from featuretools.primitives.standard.aggregation.any_primitive import Any
from featuretools.primitives.standard.aggregation.avg_time_between import AvgTimeBetween
from featuretools.primitives.standard.aggregation.average_count_per_unique import (
    AverageCountPerUnique,
)
from featuretools.primitives.standard.aggregation.count import Count
from featuretools.primitives.standard.aggregation.count_above_mean import CountAboveMean
from featuretools.primitives.standard.aggregation.count_below_mean import CountBelowMean
from featuretools.primitives.standard.aggregation.count_greater_than import (
    CountGreaterThan,
)
from featuretools.primitives.standard.aggregation.count_inside_nth_std import (
    CountInsideNthSTD,
)
from featuretools.primitives.standard.aggregation.count_inside_range import (
    CountInsideRange,
)
from featuretools.primitives.standard.aggregation.count_less_than import CountLessThan
from featuretools.primitives.standard.aggregation.count_outside_nth_std import (
    CountOutsideNthSTD,
)
from featuretools.primitives.standard.aggregation.count_outside_range import (
    CountOutsideRange,
)
from featuretools.primitives.standard.aggregation.date_first_event import DateFirstEvent
from featuretools.primitives.standard.aggregation.entropy import Entropy
from featuretools.primitives.standard.aggregation.first import First
from featuretools.primitives.standard.aggregation.first_last_time_delta import (
    FirstLastTimeDelta,
)
from featuretools.primitives.standard.aggregation.kurtosis import Kurtosis
from featuretools.primitives.standard.aggregation.is_unique import IsUnique
from featuretools.primitives.standard.aggregation.last import Last
from featuretools.primitives.standard.aggregation.max_primitive import Max
from featuretools.primitives.standard.aggregation.max_consecutive_false import (
    MaxConsecutiveFalse,
)
from featuretools.primitives.standard.aggregation.max_consecutive_negatives import (
    MaxConsecutiveNegatives,
)
from featuretools.primitives.standard.aggregation.max_consecutive_positives import (
    MaxConsecutivePositives,
)
from featuretools.primitives.standard.aggregation.max_consecutive_true import (
    MaxConsecutiveTrue,
)
from featuretools.primitives.standard.aggregation.max_consecutive_zeros import (
    MaxConsecutiveZeros,
)
from featuretools.primitives.standard.aggregation.mean import Mean
from featuretools.primitives.standard.aggregation.median import Median
from featuretools.primitives.standard.aggregation.max_count import MaxCount
from featuretools.primitives.standard.aggregation.median_count import MedianCount
from featuretools.primitives.standard.aggregation.max_min_delta import MaxMinDelta
from featuretools.primitives.standard.aggregation.min_count import MinCount
from featuretools.primitives.standard.aggregation.min_primitive import Min
from featuretools.primitives.standard.aggregation.mode import Mode
from featuretools.primitives.standard.aggregation.n_unique_days import NUniqueDays
from featuretools.primitives.standard.aggregation.n_unique_days_of_calendar_year import (
    NUniqueDaysOfCalendarYear,
)
from featuretools.primitives.standard.aggregation.n_unique_days_of_month import (
    NUniqueDaysOfMonth,
)
from featuretools.primitives.standard.aggregation.has_no_duplicates import (
    HasNoDuplicates,
)
from featuretools.primitives.standard.aggregation.is_monotonically_decreasing import (
    IsMonotonicallyDecreasing,
)
from featuretools.primitives.standard.aggregation.is_monotonically_increasing import (
    IsMonotonicallyIncreasing,
)
from featuretools.primitives.standard.aggregation.n_unique_months import NUniqueMonths
from featuretools.primitives.standard.aggregation.n_unique_weeks import NUniqueWeeks
from featuretools.primitives.standard.aggregation.n_most_common import NMostCommon
from featuretools.primitives.standard.aggregation.n_most_common_frequency import (
    NMostCommonFrequency,
)
from featuretools.primitives.standard.aggregation.num_true import NumTrue
from featuretools.primitives.standard.aggregation.num_peaks import NumPeaks
from featuretools.primitives.standard.aggregation.num_zero_crossings import (
    NumZeroCrossings,
)
from featuretools.primitives.standard.aggregation.num_true_since_last_false import (
    NumTrueSinceLastFalse,
)
from featuretools.primitives.standard.aggregation.num_false_since_last_true import (
    NumFalseSinceLastTrue,
)
from featuretools.primitives.standard.aggregation.num_consecutive_greater_mean import (
    NumConsecutiveGreaterMean,
)
from featuretools.primitives.standard.aggregation.num_consecutive_less_mean import (
    NumConsecutiveLessMean,
)
from featuretools.primitives.standard.aggregation.num_unique import NumUnique
from featuretools.primitives.standard.aggregation.percent_unique import PercentUnique
from featuretools.primitives.standard.aggregation.percent_true import PercentTrue
from featuretools.primitives.standard.aggregation.skew import Skew
from featuretools.primitives.standard.aggregation.std import Std
from featuretools.primitives.standard.aggregation.sum_primitive import Sum
from featuretools.primitives.standard.aggregation.time_since_first import TimeSinceFirst
from featuretools.primitives.standard.aggregation.time_since_last import TimeSinceLast
from featuretools.primitives.standard.aggregation.time_since_last_true import (
    TimeSinceLastTrue,
)
from featuretools.primitives.standard.aggregation.time_since_last_min import (
    TimeSinceLastMin,
)
from featuretools.primitives.standard.aggregation.time_since_last_max import (
    TimeSinceLastMax,
)
from featuretools.primitives.standard.aggregation.time_since_last_false import (
    TimeSinceLastFalse,
)
from featuretools.primitives.standard.aggregation.trend import Trend
from featuretools.primitives.standard.aggregation.variance import Variance
