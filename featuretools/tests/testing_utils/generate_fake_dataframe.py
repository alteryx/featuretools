import random
from datetime import datetime as dt
from typing import Sequence, TypeVar, Union

import numpy as np
import pandas as pd
import woodwork.type_sys.type_system as ww_type_system
from typing_extensions import TypeGuard
from woodwork import logical_types

T = TypeVar("T")


def is_list_of_lists(
    val: Sequence[Union[T, Sequence[T]]],
) -> TypeGuard[Sequence[Sequence[T]]]:
    """Determines if input is list of lists"""
    return isinstance(val[0], list)


def is_list(val: Sequence[Union[T, Sequence[T]]]) -> TypeGuard[Sequence[T]]:
    """Determines if input is list"""
    return not isinstance(val[0], list)


def flatten_list(input_list: Sequence[Union[T, Sequence[T]]]) -> Sequence[T]:
    """Flatten a list of lists into a single list"""
    if is_list_of_lists(input_list):
        a = [y for x in input_list for y in x]
        return a
    elif is_list(input_list):
        # this is a function because mypy needs help to successfully narrow type to Sequence[T]
        return input_list
    else:
        raise ValueError("Input must be a list or list of lists")


logical_type_mapping = {
    logical_types.Boolean.__name__: [True, False],
    logical_types.BooleanNullable.__name__: [True, False, np.nan],
    logical_types.Categorical.__name__: ["A", "B", "C"],
    logical_types.Datetime.__name__: [
        dt(2020, 1, 1, 12, 0, 0),
        dt(2020, 6, 1, 12, 0, 0),
    ],
    logical_types.Double.__name__: [1.2, 2.3, 3.4],
    logical_types.Integer.__name__: [1, 2, 3],
    logical_types.IntegerNullable.__name__: [1, 2, 3, np.nan],
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


def generate_fake_dataframe(
    col_defs=[("f_1", "Numeric"), ("f_2", "Datetime", "time_index")],
    n_rows=10,
):
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
        return inferred_tags.union(tags) - {"index"}

    other_kwargs = {}

    # if include_index:
    #     df = pd.DataFrame({"idx": range(n_rows)})
    #     other_kwargs["index"] = "idx"
    # else:
    df = pd.DataFrame()
    lt_dict = {}
    tags_dict = {}
    for name, lt, *rest in col_defs:
        if lt in logical_type_mapping:
            values = logical_type_mapping[lt]
            if lt == logical_types.Ordinal.__name__:
                lt = logical_types.Ordinal(order=values)
            values = gen_series(values)
        else:
            raise Exception(f"Unknown logical type {lt}")

        lt_dict[name] = lt

        if len(rest):
            tags = rest[0]
            if "index" in tags:
                other_kwargs["index"] = name
                values = range(n_rows)
            tags_dict[name] = get_tags(lt, tags)
        else:
            tags_dict[name] = get_tags(lt)

        s = pd.Series(values, name=name)
        df = pd.concat([df, s], axis=1)

    df.ww.init(
        name="nums",
        logical_types=lt_dict,
        semantic_tags=tags_dict,
        **other_kwargs,
    )

    return df
