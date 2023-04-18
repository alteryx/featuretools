import importlib.util
import os
from inspect import getfullargspec, getsource, isclass
from typing import Dict, List

import pandas as pd
from woodwork import list_logical_types, list_semantic_tags, type_system
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import NaturalLanguage

import featuretools
from featuretools.primitives import NumberOfCommonWords
from featuretools.primitives.base import (
    AggregationPrimitive,
    PrimitiveBase,
    TransformPrimitive,
)
from featuretools.utils.gen_utils import Library, find_descendents


def _get_primitives(primitive_kind):
    """Helper function that selects all primitives
    that are instances of `primitive_kind`
    """
    primitives = set()
    for attribute_string in dir(featuretools.primitives):
        attribute = getattr(featuretools.primitives, attribute_string)
        if isclass(attribute):
            if issubclass(attribute, primitive_kind) and attribute.name:
                primitives.add(attribute)
    return {prim.name.lower(): prim for prim in primitives}


def get_aggregation_primitives():
    """Returns all aggregation primitives, regardless
    of compatibility
    """
    return _get_primitives(featuretools.primitives.AggregationPrimitive)


def get_transform_primitives():
    """Returns all transform primitives, regardless
    of compatibility
    """
    return _get_primitives(featuretools.primitives.TransformPrimitive)


def get_all_primitives():
    """Helper function to return all primitives"""
    primitives = set()
    for attribute_string in dir(featuretools.primitives):
        attribute = getattr(featuretools.primitives, attribute_string)
        if isclass(attribute):
            if issubclass(attribute, PrimitiveBase) and attribute.name:
                primitives.add(attribute)
    return {prim.__name__: prim for prim in primitives}


def _get_natural_language_primitives():
    """Returns all Natural Language transform primitives,
    regardless of compatibility
    """
    transform_primitives = get_transform_primitives()

    def _natural_language_in_input_type(primitive):
        for input_type in primitive.input_types:
            if isinstance(input_type, list):
                if any(
                    isinstance(column_schema.logical_type, NaturalLanguage)
                    for column_schema in input_type
                ):
                    return True
            else:
                if isinstance(input_type.logical_type, NaturalLanguage):
                    return True
        return False

    return {
        name: primitive
        for name, primitive in transform_primitives.items()
        if _natural_language_in_input_type(primitive)
    }


def list_primitives():
    """Returns a DataFrame that lists and describes each built-in primitive."""
    trans_names, trans_primitives, valid_inputs, return_type = _get_names_primitives(
        get_transform_primitives,
    )
    trans_dask = [
        Library.DASK in primitive.compatibility for primitive in trans_primitives
    ]
    trans_spark = [
        Library.SPARK in primitive.compatibility for primitive in trans_primitives
    ]
    transform_df = pd.DataFrame(
        {
            "name": trans_names,
            "description": _get_descriptions(trans_primitives),
            "dask_compatible": trans_dask,
            "spark_compatible": trans_spark,
            "valid_inputs": valid_inputs,
            "return_type": return_type,
        },
    )
    transform_df["type"] = "transform"

    agg_names, agg_primitives, valid_inputs, return_type = _get_names_primitives(
        get_aggregation_primitives,
    )
    agg_dask = [Library.DASK in primitive.compatibility for primitive in agg_primitives]
    agg_spark = [
        Library.SPARK in primitive.compatibility for primitive in agg_primitives
    ]
    agg_df = pd.DataFrame(
        {
            "name": agg_names,
            "description": _get_descriptions(agg_primitives),
            "dask_compatible": agg_dask,
            "spark_compatible": agg_spark,
            "valid_inputs": valid_inputs,
            "return_type": return_type,
        },
    )
    agg_df["type"] = "aggregation"

    columns = [
        "name",
        "type",
        "dask_compatible",
        "spark_compatible",
        "description",
        "valid_inputs",
        "return_type",
    ]
    return pd.concat([agg_df, transform_df], ignore_index=True)[columns]


def summarize_primitives() -> pd.DataFrame:
    """Returns a metrics summary DataFrame of all primitives found in list_primitives."""
    (
        trans_names,
        trans_primitives,
        trans_valid_inputs,
        trans_return_type,
    ) = _get_names_primitives(get_transform_primitives)

    (
        agg_names,
        agg_primitives,
        agg_valid_inputs,
        agg_return_type,
    ) = _get_names_primitives(get_aggregation_primitives)

    tot_trans = len(trans_names)
    tot_agg = len(agg_names)
    tot_prims = tot_trans + tot_agg
    all_primitives = trans_primitives + agg_primitives
    primitives_summary = _get_summary_primitives(all_primitives)
    summary_dict = {
        "total_primitives": tot_prims,
        "aggregation_primitives": tot_agg,
        "transform_primitives": tot_trans,
        **primitives_summary["general_metrics"],
    }
    summary_dict.update(
        {
            f"uses_{ltype}_input": count
            for ltype, count in primitives_summary["logical_type_input_metrics"].items()
        },
    )
    summary_dict.update(
        {
            f"uses_{tag}_tag_input": count
            for tag, count in primitives_summary["semantic_tag_metrics"].items()
        },
    )
    summary_df = pd.DataFrame(
        [{"Metric": k, "Count": v} for k, v in summary_dict.items()],
    )
    return summary_df


def get_default_aggregation_primitives():
    agg_primitives = [
        featuretools.primitives.Sum,
        featuretools.primitives.Std,
        featuretools.primitives.Max,
        featuretools.primitives.Skew,
        featuretools.primitives.Min,
        featuretools.primitives.Mean,
        featuretools.primitives.Count,
        featuretools.primitives.PercentTrue,
        featuretools.primitives.NumUnique,
        featuretools.primitives.Mode,
    ]
    return agg_primitives


def get_default_transform_primitives():
    # featuretools.primitives.TimeSince
    trans_primitives = [
        featuretools.primitives.Age,
        featuretools.primitives.Day,
        featuretools.primitives.Year,
        featuretools.primitives.Month,
        featuretools.primitives.Weekday,
        featuretools.primitives.Haversine,
        featuretools.primitives.NumWords,
        featuretools.primitives.NumCharacters,
    ]
    return trans_primitives


def _get_descriptions(primitives):
    descriptions = []
    for prim in primitives:
        description = ""
        if prim.__doc__ is not None:
            # Break on the empty line between the docstring description and the remainder of the docstring
            description = prim.__doc__.split("\n\n")[0]
            # remove any excess whitespace from line breaks
            description = " ".join(description.split())
        descriptions.append(description)
    return descriptions


def _get_summary_primitives(primitives: List) -> Dict[str, int]:
    """Provides metrics for a list of primitives."""
    unique_input_types = set()
    unique_output_types = set()
    uses_multi_input = 0
    uses_multi_output = 0
    uses_external_data = 0
    are_controllable = 0
    logical_type_metrics = {
        log_type: 0 for log_type in list(list_logical_types()["type_string"])
    }
    semantic_tag_metrics = {
        sem_tag: 0 for sem_tag in list(list_semantic_tags()["name"])
    }
    semantic_tag_metrics.update(
        {"foreign_key": 0},
    )  # not currently in list_semantic_tags()

    for prim in primitives:
        log_in_type_checks = set()
        sem_tag_type_checks = set()
        input_types = prim.flatten_nested_input_types(prim.input_types)
        _check_input_types(
            input_types,
            log_in_type_checks,
            sem_tag_type_checks,
            unique_input_types,
        )
        for ltype in list(log_in_type_checks):
            logical_type_metrics[ltype] += 1

        for sem_tag in list(sem_tag_type_checks):
            semantic_tag_metrics[sem_tag] += 1

        if len(prim.input_types) > 1:
            uses_multi_input += 1

        # checks if number_output_features is set as an instance variable or set as a constant
        if (
            "self.number_output_features =" in getsource(prim.__init__)
            or prim.number_output_features > 1
        ):
            uses_multi_output += 1
        unique_output_types.add(str(prim.return_type))

        if hasattr(prim, "filename"):
            uses_external_data += 1

        if len(getfullargspec(prim.__init__).args) > 1:
            are_controllable += 1

    return {
        "general_metrics": {
            "unique_input_types": len(unique_input_types),
            "unique_output_types": len(unique_output_types),
            "uses_multi_input": uses_multi_input,
            "uses_multi_output": uses_multi_output,
            "uses_external_data": uses_external_data,
            "are_controllable": are_controllable,
        },
        "logical_type_input_metrics": logical_type_metrics,
        "semantic_tag_metrics": semantic_tag_metrics,
    }


def _check_input_types(
    input_types: List[ColumnSchema],
    log_in_type_checks: set,
    sem_tag_type_checks: set,
    unique_input_types: set,
):
    """Checks if any logical types or semantic tags occur in a list of Woodwork input types and keeps track of unique input types."""
    for in_type in input_types:
        if in_type.semantic_tags:
            for sem_tag in in_type.semantic_tags:
                sem_tag_type_checks.add(sem_tag)
        if in_type.logical_type:
            log_in_type_checks.add(in_type.logical_type.type_string)
        unique_input_types.add(str(in_type))


def _get_names_primitives(primitive_func):
    names = []
    primitives = []
    valid_inputs = []
    return_type = []
    for name, primitive in primitive_func().items():
        names.append(name)
        primitives.append(primitive)
        input_types = _get_unique_input_types(primitive.input_types)
        valid_inputs.append(", ".join(input_types))
        return_type.append(
            str(primitive.return_type),
        ) if primitive.return_type is not None else return_type.append(None)
    return names, primitives, valid_inputs, return_type


def _get_unique_input_types(input_types):
    types = set()
    for input_type in input_types:
        if isinstance(input_type, list):
            types |= _get_unique_input_types(input_type)
        else:
            types.add(str(input_type))
    return types


def list_primitive_files(directory):
    """returns list of files in directory that might contain primitives"""
    files = os.listdir(directory)
    keep = []
    for path in files:
        if not check_valid_primitive_path(path):
            continue
        keep.append(os.path.join(directory, path))
    return keep


def check_valid_primitive_path(path):
    if os.path.isdir(path):
        return False

    filename = os.path.basename(path)

    if filename[:2] == "__" or filename[0] == "." or filename[-3:] != ".py":
        return False

    return True


def load_primitive_from_file(filepath):
    """load primitive objects in a file"""
    module = os.path.basename(filepath)[:-3]
    # TODO: what is the first argument"?
    spec = importlib.util.spec_from_file_location(module, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    primitives = []
    for primitive_name in vars(module):
        primitive_class = getattr(module, primitive_name)
        if (
            isclass(primitive_class)
            and issubclass(primitive_class, PrimitiveBase)
            and primitive_class not in (AggregationPrimitive, TransformPrimitive)
        ):
            primitives.append((primitive_name, primitive_class))

    if len(primitives) == 0:
        raise RuntimeError("No primitive defined in file %s" % filepath)
    elif len(primitives) > 1:
        raise RuntimeError("More than one primitive defined in file %s" % filepath)

    return primitives[0]


def serialize_primitive(primitive: PrimitiveBase):
    """build a dictionary with the data necessary to construct the given primitive"""
    args_dict = {name: val for name, val in primitive.get_arguments()}
    cls = type(primitive)
    if cls == NumberOfCommonWords and "word_set" in args_dict:
        args_dict["word_set"] = list(args_dict["word_set"])
    return {
        "type": cls.__name__,
        "module": cls.__module__,
        "arguments": args_dict,
    }


class PrimitivesDeserializer(object):
    """
    This class wraps a cache and a generator which iterates over all primitive
    classes. When deserializing a primitive if it is not in the cache then we
    iterate until it is found, adding every seen class to the cache. When
    deserializing the next primitive the iteration resumes where it left off. This
    means that we never visit a class more than once.
    """

    def __init__(self):
        # Cache to avoid repeatedly searching for primitive class
        # (class_name, module_name) -> class
        self.class_cache = {}

        self.primitive_classes = find_descendents(PrimitiveBase)

    def deserialize_primitive(self, primitive_dict):
        """
        Construct a primitive from the given dictionary (output from
        serialize_primitive).
        """
        class_name = primitive_dict["type"]
        module_name = primitive_dict["module"]
        class_cache_key = (class_name, module_name.split(".")[0])

        if class_cache_key in self.class_cache:
            cls = self.class_cache[class_cache_key]
        else:
            cls = self._find_class_in_descendants(class_cache_key)

        if not cls:
            raise RuntimeError(
                'Primitive "%s" in module "%s" not found' % (class_name, module_name),
            )
        arguments = primitive_dict["arguments"]
        if cls == NumberOfCommonWords and "word_set" in arguments:
            # We converted word_set from a set to a list to make it serializable,
            # we should convert it back now.
            arguments["word_set"] = set(arguments["word_set"])
        primitive_instance = cls(**arguments)

        return primitive_instance

    def _find_class_in_descendants(self, search_key):
        for cls in self.primitive_classes:
            cls_key = (cls.__name__, cls.__module__.split(".")[0])
            self.class_cache[cls_key] = cls

            if cls_key == search_key:
                return cls


def get_all_logical_type_names():
    """Helper function that returns all registered woodwork logical types"""
    return {lt.__name__: lt for lt in type_system.registered_types}
