import databricks.koalas as ks
import numpy as np
import pandas as pd

def replace_tuple_columns(pdf):
    new_df = pd.DataFrame()
    for c in pdf.columns:
        if isinstance(pdf[c].iloc[0], tuple):
            new_df[c] = pdf[c].map(lambda x: list(x))
        else:
            new_df[c] = pdf[c]
    return new_df

def replace_nan_with_flag(pdf, flag=-1):
    new_df = pd.DataFrame()
    for c in pdf.columns:
        if isinstance(pdf[c].iloc[0], list):
            new_df[c] = pdf[c].map(lambda l: [flag if np.isnan(x) else x for x in l])
        else:
            new_df[c] = pdf[c]
    return new_df

def pandas_to_koalas_clean(pdf):
    steps = [replace_tuple_columns, replace_nan_with_flag]
    intermediate_df = pdf
    for f in steps:
        intermediate_df = f(intermediate_df)
    return intermediate_df