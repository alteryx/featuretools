from featuretools.primitives import AggregationPrimitive, TransformPrimitive
from featuretools.primitives.utils import (
    get_aggregation_primitives,
    get_transform_primitives,
)
from featuretools.synthesis.deep_feature_synthesis import DeepFeatureSynthesis
from featuretools.synthesis.utils import _categorize_features, get_unused_primitives
from featuretools.utils.gen_utils import Library


def get_valid_primitives(
    entityset,
    target_dataframe_name,
    max_depth=2,
    selected_primitives=None,
    **dfs_kwargs,
):
    """
    Returns two lists of primitives (transform and aggregation) containing
    primitives that can be applied to the specific target dataframe to create
    features.  If the optional 'selected_primitives' parameter is not used,
    all discoverable primitives will be considered.

    Note:
        When using a ``max_depth`` greater than 1, some primitives returned by
        this function may not create any features if passed to DFS alone.  These
        primitives relied on features created by other primitives as input
        (primitive stacking).

    Args:
        entityset (EntitySet): An already initialized entityset
        target_dataframe_name (str): Name of dataframe to create features for.
        max_depth (int, optional): Maximum allowed depth of features.
        selected_primitives(list[str or AggregationPrimitive/TransformPrimitive], optional):
            list of primitives to consider when looking for valid primitives.
            If None, all primitives will be considered
        dfs_kwargs (keywords): Additional keyword arguments to pass as keyword arguments to
            the DeepFeatureSynthesis object. Should not include ``max_depth``, ``agg_primitives``,
            or ``trans_primitives``, as those are passed in explicity.
    Returns:
       list[AggregationPrimitive], list[TransformPrimitive]:
           The list of valid aggregation primitives and the list of valid
           transform primitives.
    """
    agg_primitives = []
    trans_primitives = []
    available_aggs = get_aggregation_primitives()
    available_trans = get_transform_primitives()

    for library in Library:
        if library.value == entityset.dataframe_type:
            df_library = library
            break

    if selected_primitives:
        for prim in selected_primitives:
            if not isinstance(prim, str):
                if issubclass(prim, AggregationPrimitive):
                    prim_list = agg_primitives
                elif issubclass(prim, TransformPrimitive):
                    prim_list = trans_primitives
                else:
                    raise ValueError(
                        f"Selected primitive {prim} is not an "
                        "AggregationPrimitive, TransformPrimitive, or str",
                    )
            elif prim in available_aggs:
                prim = available_aggs[prim]
                prim_list = agg_primitives
            elif prim in available_trans:
                prim = available_trans[prim]
                prim_list = trans_primitives
            else:
                raise ValueError(f"'{prim}' is not a recognized primitive name")
            if df_library in prim.compatibility:
                prim_list.append(prim)
    else:
        agg_primitives = [
            agg for agg in available_aggs.values() if df_library in agg.compatibility
        ]
        trans_primitives = [
            trans
            for trans in available_trans.values()
            if df_library in trans.compatibility
        ]

    dfs_object = DeepFeatureSynthesis(
        target_dataframe_name,
        entityset,
        agg_primitives=agg_primitives,
        trans_primitives=trans_primitives,
        max_depth=max_depth,
        **dfs_kwargs,
    )

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
