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


def replace_categorical_columns(pdf):
    new_df = pd.DataFrame()
    for c in pdf.columns:
        col = pdf[c]
        if col.dtype.name == 'category':
            new_df[c] = col.astype(col.dtype.categories.dtype)
        else:
            new_df[c] = pdf[c]
    return new_df


def pd_to_ks_clean(pdf):
    steps = [replace_tuple_columns, replace_nan_with_flag, replace_categorical_columns]
    intermediate_df = pdf
    for f in steps:
        intermediate_df = f(intermediate_df)
    return intermediate_df
