import json

import featuretools as ft


def describe_feature(
    feature,
    feature_descriptions=None,
    primitive_templates=None,
    metadata_file=None,
):
    """Generates an English language description of a feature.

    Args:
        feature (FeatureBase) : Feature to describe
        feature_descriptions (dict, optional) : dictionary mapping features or unique
            feature names to custom descriptions
        primitive_templates (dict, optional) : dictionary mapping primitives or
            primitive names to description templates
        metadata_file (str, optional) : path to json metadata file

    Returns:
        str : English description of the feature
    """
    feature_descriptions = feature_descriptions or {}
    primitive_templates = primitive_templates or {}

    if metadata_file:
        file_feature_descriptions, file_primitive_templates = parse_json_metadata(
            metadata_file,
        )
        feature_descriptions = {**file_feature_descriptions, **feature_descriptions}
        primitive_templates = {**file_primitive_templates, **primitive_templates}

    description = generate_description(
        feature,
        feature_descriptions,
        primitive_templates,
    )
    return description[:1].upper() + description[1:] + "."


def generate_description(feature, feature_descriptions, primitive_templates):
    # Check if feature has custom description
    if feature in feature_descriptions or feature.unique_name() in feature_descriptions:
        description = feature_descriptions.get(feature) or feature_descriptions.get(
            feature.unique_name(),
        )
        return description

    # Check if identity feature:
    if isinstance(feature, ft.IdentityFeature):
        description = feature.column_schema.description
        if description is None:
            description = 'the "{}"'.format(feature.column_name)
        return description

    # Handle direct features
    if isinstance(feature, ft.DirectFeature):
        base_feature, direct_description = get_direct_description(feature)
        direct_base = generate_description(
            base_feature,
            feature_descriptions,
            primitive_templates,
        )
        return direct_base + direct_description

    # Get input descriptions
    input_descriptions = []
    input_columns = feature.base_features
    if isinstance(feature, ft.feature_base.FeatureOutputSlice):
        input_columns = feature.base_feature.base_features

    for input_col in input_columns:
        col_description = generate_description(
            input_col,
            feature_descriptions,
            primitive_templates,
        )
        input_descriptions.append(col_description)

    # Remove groupby description from input columns
    groupby_description = None
    if isinstance(feature, ft.GroupByTransformFeature):
        groupby_description = input_descriptions.pop()

    # Generate primitive description
    template_override = None
    if (
        feature.primitive in primitive_templates
        or feature.primitive.name in primitive_templates
    ):
        template_override = primitive_templates.get(
            feature.primitive,
        ) or primitive_templates.get(feature.primitive.name)
    slice_num = feature.n if hasattr(feature, "n") else None
    primitive_description = feature.primitive.get_description(
        input_descriptions,
        slice_num=slice_num,
        template_override=template_override,
    )
    if isinstance(feature, ft.feature_base.FeatureOutputSlice):
        feature = feature.base_feature

    # Generate groupby phrase if applicable
    groupby = ""
    if isinstance(feature, ft.AggregationFeature):
        groupby_description = get_aggregation_groupby(feature, feature_descriptions)
    if groupby_description is not None:
        if groupby_description.startswith("the "):
            groupby_description = groupby_description[4:]
        groupby = "for each {}".format(groupby_description)

    # Generate aggregation dataframe phrase with use_previous
    dataframe_description = ""
    if isinstance(feature, ft.AggregationFeature):
        if feature.use_previous:
            dataframe_description = "of the previous {} of ".format(
                feature.use_previous.get_name().lower(),
            )
        else:
            dataframe_description = "of all instances of "
        dataframe_description += '"{}"'.format(
            feature.relationship_path[-1][1].child_dataframe.ww.name,
        )

    # Generate where phrase
    where = ""
    if hasattr(feature, "where") and feature.where:
        where_col = generate_description(
            feature.where.base_features[0],
            feature_descriptions,
            primitive_templates,
        )
        where = "where {} is {}".format(where_col, feature.where.primitive.value)

    # Join all parts of template
    description_template = [
        primitive_description,
        dataframe_description,
        where,
        groupby,
    ]
    description = " ".join([phrase for phrase in description_template if phrase != ""])

    return description


def get_direct_description(feature):
    direct_description = (
        ' the instance of "{}" associated with this ' 'instance of "{}"'.format(
            feature.relationship_path[-1][1].parent_dataframe.ww.name,
            feature.dataframe_name,
        )
    )
    base_features = feature.base_features
    # shortens stacked direct features to make it easier to understand
    while isinstance(base_features[0], ft.DirectFeature):
        base_feat = base_features[0]
        base_feat_description = ' the instance of "{}" associated ' "with".format(
            base_feat.relationship_path[-1][1].parent_dataframe.ww.name,
        )
        direct_description = base_feat_description + direct_description
        base_features = base_feat.base_features
    direct_description = " for" + direct_description

    return base_features[0], direct_description


def get_aggregation_groupby(feature, feature_descriptions=None):
    if feature_descriptions is None:
        feature_descriptions = {}
    groupby_name = feature.dataframe.ww.index
    groupby = ft.IdentityFeature(
        feature.entityset[feature.dataframe_name].ww[groupby_name],
    )
    if groupby in feature_descriptions or groupby.unique_name() in feature_descriptions:
        return feature_descriptions.get(groupby) or feature_descriptions.get(
            groupby.unique_name(),
        )
    else:
        return '"{}" in "{}"'.format(groupby_name, feature.dataframe_name)


def parse_json_metadata(file):
    with open(file) as f:
        json_metadata = json.load(f)

    return (
        json_metadata.get("feature_descriptions", {}),
        json_metadata.get("primitive_templates", {}),
    )
