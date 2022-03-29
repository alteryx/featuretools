from enum import Enum
from typing import Dict, Tuple

import pandas as pd

T_LogicalType = Dict[str, str]
T_SemanticTag = Dict[str, str]


T_ES_Dataframe = Dict[
    str, Tuple[pd.DataFrame, str, str, T_LogicalType, T_SemanticTag, bool]
]


class Library(Enum):
    PANDAS = "pandas"
    DASK = "Dask"
    SPARK = "Spark"
