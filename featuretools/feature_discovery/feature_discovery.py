from typing import List

from woodwork.column_schema import ColumnSchema


def index_input_set(input_set: List[ColumnSchema]):
    out = {}
    for c in input_set:
        lt = type(c.logical_type).__name__

        if lt != "NoneType":
            if lt in out:
                out[lt] += 1
            else:
                out[lt] = 1

        tags = c.semantic_tags
        for tag in tags:
            if tag is not None:
                if tag in out:
                    out[tag] += 1
                else:
                    out[tag] = 1

        if lt == "NoneType" and len(tags) == 0:
            if "ANY" in out:
                out["ANY"] += 1
            else:
                out["ANY"] = 1

    return out
