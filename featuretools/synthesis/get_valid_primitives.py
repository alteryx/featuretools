from featuretools.primitives import AggregationPrimitive, TransformPrimitive
from featuretools.primitives.utils import (
    get_aggregation_primitives,
    get_transform_primitives
)
from featuretools.synthesis.deep_feature_synthesis import DeepFeatureSynthesis
from featuretools.synthesis.utils import (
    _categorize_features,
    get_entityset_type,
    get_unused_primitives
)


def get_valid_primitives(entityset, target_entity, max_depth=2, selected_primitives=None):
    """
    Returns two lists of primitives (transform and aggregation) containing
    primitives that can be applied to the specific target entity to create
    features.  If the optional 'selected_primitives' parameter is not used,
    all discoverable primitives will be considered.

    Args:
        entityset (EntitySet): An already initialized entityset
        target_entity (str): Entity id of entity to create features for.
        max_depth (int, optional): Maximum allowed depth of features.
        selected_primitives(list[str or AggregationPrimitive/TransformPrimitive], optional):
            list of primitives to consider when looking for valid primitives.
            If None, all primitives will be considered
    Returns:
       list[AggregationPrimitive], list[TransformPrimitive]:
           The list of valid aggregation primitives and the list of valid
           transform primitives.
    """
    agg_primitives = []
    trans_primitives = []
    available_aggs = get_aggregation_primitives()
    available_trans = get_transform_primitives()
    entityset_type = get_entityset_type(entityset)

    if selected_primitives:
        for prim in selected_primitives:
            if not isinstance(prim, str) and not issubclass(prim, (AggregationPrimitive, TransformPrimitive)):
                raise ValueError(f"Selected primitive {prim} is not an "
                                 "AggergationPrimitive, TransformPrimitive, or str")
            if not isinstance(prim, str) and issubclass(prim, AggregationPrimitive):
                agg_primitives.append(prim)
            elif not isinstance(prim, str) and issubclass(prim, TransformPrimitive):
                trans_primitives.append(prim)
            elif prim in available_aggs:
                agg_primitives.append(available_aggs[prim])
            elif prim in available_trans:
                trans_primitives.append(available_trans[prim])
            else:
                raise ValueError(f"'{prim}' is not a recognized primitive name")
    else:
        agg_primitives = list(available_aggs.values())
        trans_primitives = list(available_trans.values())

    agg_primitives = [agg for agg in agg_primitives if entityset_type in agg.compatibility]
    trans_primitives = [trans for trans in trans_primitives if entityset_type in trans.compatibility]

    dfs_object = DeepFeatureSynthesis(target_entity, entityset,
                                      agg_primitives=agg_primitives,
                                      trans_primitives=trans_primitives,
                                      max_depth=max_depth)

    features = dfs_object.build_features()

    trans, agg, _, _ = _categorize_features(features)

    trans_unused = get_unused_primitives(trans_primitives, trans)
    agg_unused = get_unused_primitives(agg_primitives, agg)

    # switch from str to class
    agg_unused = [available_aggs[name] for name in agg_unused]
    trans_unused = [available_trans[name] for name in trans_unused]

    used_agg_prims = set(agg_primitives).difference(set(agg_unused))
    used_trans_prims = set(trans_primitives).difference(set(trans_unused))
    return list(used_agg_prims), list(used_trans_prims)
