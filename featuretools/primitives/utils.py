from inspect import isclass

import pandas as pd

import featuretools


def get_aggregation_primitives():
    aggregation_primitives = set([])
    for attribute_string in dir(featuretools.primitives):
        attribute = getattr(featuretools.primitives, attribute_string)
        if isclass(attribute):
            if issubclass(attribute,
                          featuretools.primitives.AggregationPrimitive):
                if attribute.name:
                    aggregation_primitives.add(attribute)
    return {prim.name.lower(): prim for prim in aggregation_primitives}


def get_transform_primitives():
    transform_primitives = set([])
    for attribute_string in dir(featuretools.primitives):
        attribute = getattr(featuretools.primitives, attribute_string)
        if isclass(attribute):
            if issubclass(attribute,
                          featuretools.primitives.TransformPrimitive):
                if attribute.name:
                    transform_primitives.add(attribute)
    return {prim.name.lower(): prim for prim in transform_primitives}


def list_primitives():
    transform_primitives = get_transform_primitives()
    agg_primitives = get_aggregation_primitives()

    transform_df = pd.DataFrame({'name': list(transform_primitives.keys()),
                                 'description': _get_descriptions(transform_primitives)})
    transform_df['type'] = 'transform'

    agg_df = pd.DataFrame({'name': list(agg_primitives.keys()),
                           'description': _get_descriptions(agg_primitives)})
    agg_df['type'] = 'aggregation'

    return pd.concat([agg_df, transform_df], ignore_index=True)[['name', 'type', 'description']]


def _get_descriptions(primitives):
    descriptions = []
    for prim in primitives.values():
        description = ''
        if prim.__doc__ is not None:
            description = prim.__doc__.split("\n")[0]
        descriptions.append(description)
    return description
