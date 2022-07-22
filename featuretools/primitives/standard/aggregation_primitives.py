from featuretools.primitives.aggregation import (
    Count,
    Sum,
    Mean,
    Mode, 
    Min,
    Max,
    NumUnique,
    NumTrue,
    PercentTrue,
    NMostCommon,
    AvgTimeBetween,
    Median,
    Skew,
    Std,
    First,
    Last,
    Any,
    All,
    TimeSinceLast,
    TimeSinceFirst,
    Trend,
    Entropy,
)

from warnings import warn

warn(
    "featuretools.primitives.standard module will become deprecated. Use featuretools.primitives or featuretools.primitives.aggregation instead",
    Warning,
)