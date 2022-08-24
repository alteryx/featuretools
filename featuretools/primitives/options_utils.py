import logging
import warnings
from itertools import permutations

from featuretools import primitives
from featuretools.feature_base import IdentityFeature

logger = logging.getLogger("featuretools")


def _get_primitive_options():
    # all possible option keys: function that verifies value type
    return {
        "ignore_dataframes": list_dataframe_check,
        "include_dataframes": list_dataframe_check,
        "ignore_columns": dict_to_list_column_check,
        "include_columns": dict_to_list_column_check,
        "ignore_groupby_dataframes": list_dataframe_check,
        "include_groupby_dataframes": list_dataframe_check,
        "ignore_groupby_columns": dict_to_list_column_check,
        "include_groupby_columns": dict_to_list_column_check,
    }


def dict_to_list_column_check(option, es):
    if not (
        isinstance(option, dict)
        and all([isinstance(option_val, list) for option_val in option.values()])
    ):
        return False
    else:
        for dataframe, columns in option.items():
            if dataframe not in es:
                warnings.warn("Dataframe '%s' not in entityset" % (dataframe))
            else:
                for invalid_col in [
                    column for column in columns if column not in es[dataframe]
                ]:
                    warnings.warn(
                        "Column '%s' not in dataframe '%s'" % (invalid_col, dataframe),
                    )
        return True


def list_dataframe_check(option, es):
    if not isinstance(option, list):
        return False
    else:
        for invalid_dataframe in [
            dataframe for dataframe in option if dataframe not in es
        ]:
            warnings.warn("Dataframe '%s' not in entityset" % (invalid_dataframe))
        return True


def generate_all_primitive_options(
    all_primitives,
    primitive_options,
    ignore_dataframes,
    ignore_columns,
    es,
):
    dataframe_dict = {
        dataframe.ww.name: [col for col in dataframe.columns]
        for dataframe in es.dataframes
    }

    primitive_options = _init_primitive_options(primitive_options, dataframe_dict)
    global_ignore_dataframes = ignore_dataframes
    global_ignore_columns = ignore_columns.copy()
    # for now, only use primitive names as option keys
    for primitive in all_primitives:
        if primitive in primitive_options and primitive.name in primitive_options:
            msg = (
                "Options present for primitive instance and generic "
                "primitive class (%s), primitive instance will not use generic "
                "options" % (primitive.name)
            )
            warnings.warn(msg)
        if primitive in primitive_options or primitive.name in primitive_options:
            options = primitive_options.get(
                primitive,
                primitive_options.get(primitive.name),
            )
            # Reconcile global options with individually-specified options
            included_dataframes = set().union(
                *[
                    option.get("include_dataframes", set()).union(
                        option.get("include_columns", {}).keys(),
                    )
                    for option in options
                ]
            )
            global_ignore_dataframes = global_ignore_dataframes.difference(
                included_dataframes,
            )
            for option in options:
                # don't globally ignore a column if it's included for a primitive
                if "include_columns" in option:
                    for dataframe, include_cols in option["include_columns"].items():
                        global_ignore_columns[dataframe] = global_ignore_columns[
                            dataframe
                        ].difference(include_cols)
                option["ignore_dataframes"] = option["ignore_dataframes"].union(
                    ignore_dataframes.difference(included_dataframes),
                )
            for dataframe, ignore_cols in ignore_columns.items():
                # if already ignoring columns for this dataframe, add globals
                for option in options:
                    if dataframe in option["ignore_columns"]:
                        option["ignore_columns"][dataframe] = option["ignore_columns"][
                            dataframe
                        ].union(ignore_cols)
                    # if no ignore_columns and dataframe is explicitly included, don't ignore the column
                    elif dataframe in included_dataframes:
                        continue
                    # Otherwise, keep the global option
                    else:
                        option["ignore_columns"][dataframe] = ignore_cols
        else:
            # no user specified options, just use global defaults
            primitive_options[primitive] = [
                {
                    "ignore_dataframes": ignore_dataframes,
                    "ignore_columns": ignore_columns,
                },
            ]
    return primitive_options, global_ignore_dataframes, global_ignore_columns


def _init_primitive_options(primitive_options, es):
    # Flatten all tuple keys, convert value lists into sets, check for
    # conflicting keys
    flattened_options = {}
    for primitive_keys, options in primitive_options.items():
        if not isinstance(primitive_keys, tuple):
            primitive_keys = (primitive_keys,)
        if isinstance(options, list):
            for primitive_key in primitive_keys:
                if isinstance(primitive_key, str):
                    primitive = primitives.get_aggregation_primitives().get(
                        primitive_key,
                    ) or primitives.get_transform_primitives().get(primitive_key)
                    if not primitive:
                        msg = "Unknown primitive with name '{}'".format(primitive_key)
                        raise ValueError(msg)
                else:
                    primitive = primitive_key
                assert (
                    len(primitive.input_types[0]) == len(options)
                    if isinstance(primitive.input_types[0], list)
                    else len(primitive.input_types) == len(options)
                ), (
                    "Number of options does not match number of inputs for primitive %s"
                    % (primitive_key)
                )
            options = [
                _init_option_dict(primitive_keys, option, es) for option in options
            ]
        else:
            options = [_init_option_dict(primitive_keys, options, es)]

        for primitive in primitive_keys:
            if isinstance(primitive, type):
                primitive = primitive.name

            # if primitive is specified more than once, raise error
            if primitive in flattened_options:
                raise KeyError("Multiple options found for primitive %s" % (primitive))

            flattened_options[primitive] = options
    return flattened_options


def _init_option_dict(key, option_dict, es):
    initialized_option_dict = {}
    primitive_options = _get_primitive_options()
    # verify all keys are valid and match expected type, convert lists to sets
    for option_key, option in option_dict.items():
        if option_key not in primitive_options:
            raise KeyError(
                "Unrecognized primitive option '%s' for %s"
                % (option_key, ",".join(key)),
            )
        if not primitive_options[option_key](option, es):
            raise TypeError(
                "Incorrect type formatting for '%s' for %s"
                % (option_key, ",".join(key)),
            )
        if isinstance(option, list):
            initialized_option_dict[option_key] = set(option)
        elif isinstance(option, dict):
            initialized_option_dict[option_key] = {
                key: set(option[key]) for key in option
            }
    # initialize ignore_dataframes and ignore_columns to empty sets if not present
    if "ignore_columns" not in initialized_option_dict:
        initialized_option_dict["ignore_columns"] = dict()
    if "ignore_dataframes" not in initialized_option_dict:
        initialized_option_dict["ignore_dataframes"] = set()
    return initialized_option_dict


def column_filter(f, options, groupby=False):
    if groupby and not f.column_schema.semantic_tags.intersection(
        {"category", "foreign_key"},
    ):
        return False
    include_cols = "include_groupby_columns" if groupby else "include_columns"
    ignore_cols = "ignore_groupby_columns" if groupby else "ignore_columns"
    include_dataframes = (
        "include_groupby_dataframes" if groupby else "include_dataframes"
    )
    ignore_dataframes = "ignore_groupby_dataframes" if groupby else "ignore_dataframes"

    dependencies = f.get_dependencies(deep=True) + [f]
    for base_f in dependencies:
        if isinstance(base_f, IdentityFeature):
            if (
                include_cols in options
                and base_f.dataframe_name in options[include_cols]
            ):
                if base_f.get_name() in options[include_cols][base_f.dataframe_name]:
                    continue  # this is a valid feature, go to next
                else:
                    return False  # this is not an included feature
            if ignore_cols in options and base_f.dataframe_name in options[ignore_cols]:
                if base_f.get_name() in options[ignore_cols][base_f.dataframe_name]:
                    return False  # ignore this feature
        if include_dataframes in options:
            return base_f.dataframe_name in options[include_dataframes]
        elif (
            ignore_dataframes in options
            and base_f.dataframe_name in options[ignore_dataframes]
        ):
            return False  # ignore the dataframe
    return True


def ignore_dataframe_for_primitive(options, dataframe, groupby=False):
    # This logic handles whether given options ignore an dataframe or not
    def should_ignore_dataframe(option):
        if groupby:
            if (
                "include_groupby_columns" not in option
                or dataframe.ww.name not in option["include_groupby_columns"]
            ):
                if (
                    "include_groupby_dataframes" in option
                    and dataframe.ww.name not in option["include_groupby_dataframes"]
                ):
                    return True
                elif (
                    "ignore_groupby_dataframes" in option
                    and dataframe.ww.name in option["ignore_groupby_dataframes"]
                ):
                    return True
        if (
            "include_columns" in option
            and dataframe.ww.name in option["include_columns"]
        ):
            return False
        elif "include_dataframes" in option:
            return dataframe.ww.name not in option["include_dataframes"]
        elif dataframe.ww.name in option["ignore_dataframes"]:
            return True
        else:
            return False

    return any([should_ignore_dataframe(option) for option in options])


def filter_groupby_matches_by_options(groupby_matches, options):
    return filter_matches_by_options(
        [(groupby_match,) for groupby_match in groupby_matches],
        options,
        groupby=True,
    )


def filter_matches_by_options(matches, options, groupby=False, commutative=False):
    # If more than one option, than need to handle each for each input
    if len(options) > 1:

        def is_valid_match(match):
            if all(
                [
                    column_filter(m, option, groupby)
                    for m, option in zip(match, options)
                ],
            ):
                return True
            else:
                return False

    else:

        def is_valid_match(match):
            if all([column_filter(f, options[0], groupby) for f in match]):
                return True
            else:
                return False

    valid_matches = set()
    for match in matches:
        if is_valid_match(match):
            valid_matches.add(match)
        elif commutative:
            for order in permutations(match):
                if is_valid_match(order):
                    valid_matches.add(order)
                    break

    return sorted(
        valid_matches,
        key=lambda features: ([feature.unique_name() for feature in features]),
    )
