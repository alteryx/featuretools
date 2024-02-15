import random
from datetime import datetime as dt

import pandas as pd
import pytest
import woodwork.type_sys.type_system as ww_type_system
from woodwork import logical_types

logical_type_mapping = {
    logical_types.Boolean.__name__: [True, False],
    logical_types.BooleanNullable.__name__: [True, False, pd.NA],
    logical_types.Categorical.__name__: ["A", "B", "C"],
    logical_types.Datetime.__name__: [
        dt(2020, 1, 1, 12, 0, 0),
        dt(2020, 6, 1, 12, 0, 0),
    ],
    logical_types.Double.__name__: [1.2, 2.3, 3.4],
    logical_types.Integer.__name__: [1, 2, 3],
    logical_types.IntegerNullable.__name__: [1, 2, 3, pd.NA],
    logical_types.EmailAddress.__name__: [
        "john.smith@example.com",
        "sally.jones@example.com",
    ],
    logical_types.LatLong.__name__: [(1, 2), (3, 4)],
    logical_types.NaturalLanguage.__name__: [
        "This is sentence 1",
        "This is sentence 2",
    ],
    logical_types.Ordinal.__name__: [1, 2, 3],
    logical_types.URL.__name__: ["https://www.example.com", "https://www.example2.com"],
    logical_types.PostalCode.__name__: ["60018", "60018-0123"],
}


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def generate_fake_dataframe(
    col_defs=[("f_1", "Numeric"), ("f_2", "Datetime", "time_index")],
    n_rows=10,
    df_name="df",
):
    dask = pytest.importorskip("dask", reason="Dask not installed, skipping")
    dask.config.set({"dataframe.convert-string": False})

    def randomize(values_):
        random.seed(10)
        values = values_.copy()
        random.shuffle(values)
        return values

    def gen_series(values):
        values = [values] * n_rows
        if isinstance(values, list):
            values = flatten_list(values)

        return randomize(values)[:n_rows]

    def get_tags(lt, tags=set()):
        inferred_tags = ww_type_system.str_to_logical_type(lt).standard_tags
        assert isinstance(inferred_tags, set)
        return inferred_tags.union(tags) - {"index", "time_index"}

    other_kwargs = {}

    df = pd.DataFrame()
    lt_dict = {}
    tags_dict = {}
    for name, lt_name, *rest in col_defs:
        if lt_name in logical_type_mapping:
            values = logical_type_mapping[lt_name]
            if lt_name == logical_types.Ordinal.__name__:
                lt = logical_types.Ordinal(order=values)
            else:
                lt = lt_name
            values = gen_series(values)
        else:
            raise Exception(f"Unknown logical type {lt_name}")

        lt_dict[name] = lt

        if len(rest):
            tags = rest[0]
            if "index" in tags:
                other_kwargs["index"] = name
                values = range(n_rows)
            if "time_index" in tags:
                other_kwargs["time_index"] = name
                values = pd.date_range("2000-01-01", periods=n_rows)
            tags_dict[name] = get_tags(lt_name, tags)
        else:
            tags_dict[name] = get_tags(lt_name)

        s = pd.Series(values, name=name)
        df = pd.concat([df, s], axis=1)

    df.ww.init(
        name=df_name,
        logical_types=lt_dict,
        semantic_tags=tags_dict,
        **other_kwargs,
    )

    return df
