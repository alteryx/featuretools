from typing import Dict
from typing import Tuple
import pandas as pd

T_LogicalType = Dict[str, str]
T_SemanticTag = Dict[str, str] 

# tuple(DataFrame, str, str, dict[str -> str/Woodwork.LogicalType], dict[str->str/set], boolean)

T_ES_Dataframe = Dict[str, Tuple[pd.DataFrame, str, str, T_LogicalType, T_SemanticTag, bool]]










a:T_ES_Dataframe = {'d': (pd.DataFrame(),"d", "c", {"a": "b"},{'a': "b"}, True)}