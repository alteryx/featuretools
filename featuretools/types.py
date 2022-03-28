from enum import Enum
from typing import Dict, Tuple

import pandas as pd

T_LogicalType = Dict[str, str]
T_SemanticTag = Dict[str, str]

# tuple(DataFrame, str, str, dict[str -> str/Woodwork.LogicalType], dict[str->str/set], boolean)

T_ES_Dataframe = Dict[
    str, Tuple[pd.DataFrame, str, str, T_LogicalType, T_SemanticTag, bool]
]


class Library(Enum):
    PANDAS = "pandas"
    DASK = "Dask"
    SPARK = "Spark"


class PrimitiveTypes(Enum):
    AGGREGATION = "aggregation"
    TRANSFORM = "transform"
    WHERE = "where"
    GROUPBY_TRANSFORM = "groupby transform"
