import pandas as pd


def replace_tuple_columns(pdf):
    new_df = pd.DataFrame()
    for c in pdf.columns:
        if isinstance(pdf[c].iloc[0], tuple):
            new_df[c] = pdf[c].map(lambda x: list(x) if isinstance(x, tuple) else x)
        else:
            new_df[c] = pdf[c]
    return new_df


def replace_nan_with_None(df):
    def replace(val):
        if isinstance(val, (tuple, list)):
            return [None if pd.isna(x) else x for x in val]
        elif pd.isna(val):
            return None
        else:
            return val

    return df.copy().applymap(replace)


def replace_categorical_columns(pdf):
    new_df = pd.DataFrame()
    for c in pdf.columns:
        col = pdf[c]
        if col.dtype.name == "category":
            new_df[c] = col.astype("string")
        else:
            new_df[c] = pdf[c]
    return new_df


def pd_to_spark_clean(pdf):
    steps = [replace_tuple_columns, replace_nan_with_None, replace_categorical_columns]
    intermediate_df = pdf
    for f in steps:
        intermediate_df = f(intermediate_df)
    return intermediate_df
