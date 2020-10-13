import featuretools as ft


def describe_feature(feature, feature_descriptions={}, primitive_templates={}, metadata_file=None):
    '''Generates an English language description of a feature.

    Args:
        feature (FeatureBase) : Feature to describe
        feature_descriptions (dict, optional) : dictionary mapping features or unique feature names to custom descriptions
        primitive_templates (dict, optional) : dictionary mapping primitives or primitive names to description templates
        metadata_file (str, optional) : path to metadata json

    Returns:
        str : English description of the feature
    '''
    if metadata_file:
        file_feature_descriptions = parse_json_metadata(metadata_file)
        feature_descriptions = {**file_feature_descriptions, **feature_descriptions}

    description = generate_description(feature, feature_descriptions)
    return description[:1].upper() + description[1:] + '.'


def generate_description(feature, feature_descriptions={}):
    # 1) Check if has its own description
    if feature in feature_descriptions or feature.unique_name() in feature_descriptions:
        return (feature_descriptions.get(feature) or feature_descriptions.get(feature.unique_name()))

    # 2) Check if identity feature:
    if isinstance(feature, ft.IdentityFeature):
        return 'the "{}"'.format(feature.get_name())

    # 3) Deal with direct features
    if isinstance(feature, ft.DirectFeature):
        return get_direct_description(feature, feature_descriptions)

    # Get input descriptions -OR- feature names + adding to list of to explore
    to_describe = []
    input_descriptions = []
    input_columns = feature.base_features
    if isinstance(feature, ft.feature_base.FeatureOutputSlice):
        input_columns = feature.base_feature.base_features

    for input_col in input_columns:
        if input_col in feature_descriptions or input_col.unique_name() in feature_descriptions:
            col_description = feature_descriptions.get(input_col) or feature_descriptions.get(input_col.unique_name())
        else:
            if isinstance(input_col, ft.IdentityFeature):
                col_description = '"{}"'.format(input_col.get_name())
            else:
                col_description = generate_description(input_col, feature_descriptions)
            # col_description = '"{}"'.format(input_col.get_name())
            # if not isinstance(input_col, ft.IdentityFeature):
            #    to_describe.append(input_col)
        input_descriptions.append(col_description)

    if isinstance(feature, ft.GroupByTransformFeature):
        groupby_description = input_descriptions.pop()

    # Generate primitive description
    slice_num = None
    if isinstance(feature, ft.feature_base.FeatureOutputSlice):
        slice_num = feature.n
    primitive_description = feature.primitive.get_description(input_descriptions, slice_num=slice_num)
    if isinstance(feature, ft.feature_base.FeatureOutputSlice):
        feature = feature.base_feature

    # Generate groupby phrase if applicable
    groupby = ''
    if isinstance(feature, ft.AggregationFeature):
        groupby_name = get_aggregation_groupby(feature, feature_descriptions)
        groupby = "for each {}".format(groupby_name)
    elif isinstance(feature, ft.GroupByTransformFeature):
        groupby = "for each {}".format(groupby_description)

    # 6) generate aggregation entity phrase w/ use_previous
    entity_description = ''
    if isinstance(feature, ft.AggregationFeature):
        if feature.use_previous:
            entity_description = "of the previous {} of ".format(feature.use_previous.get_name().lower())
        else:
            entity_description = "of all instances of "
        entity_description += '"{}"'.format(feature.relationship_path[-1][1].child_entity.id)

    # 7) generate where phrase
    where = ''
    if hasattr(feature, 'where') and feature.where:
        where_value = feature.where.primitive.value
        if feature.where in feature_descriptions or feature.where.unique_name() in feature_descriptions:
            where_col = feature_descriptions.get(feature.where) or feature_descriptions.get(feature.where.unique_name())
        else:
            where_col = generate_description(feature.where.base_features[0], feature_descriptions)[:-1]
        where = 'where {} is {}'.format(where_col, where_value)

    # 8) join all parts of template
    description_template = [primitive_description, entity_description, where, groupby]
    description = " ".join([phrase for phrase in description_template if phrase != ''])

    # 9) attach any descriptions of input columns that need to be defined
    for feat in to_describe:
        next_description = ' ' + '"{}" is {}'.format(feat.get_name(), generate_description(feat, feature_descriptions))
        description += next_description

    return description


def get_direct_description(feature, feature_descriptions):
    direct_base = generate_description(feature.base_features[0], feature_descriptions)
    if direct_base.endswith(' of the instance of "{}"'.format(feature.relationship_path[-1][1].parent_entity.id)):
        return direct_base + ' associated with this instance of "{}"'.format(feature.entity_id)
    else:
        return direct_base + ' of the instance of "{}" associated with this instance of "{}"'.format(feature.relationship_path[-1][1].parent_entity.id,
                                                                                                     feature.entity_id)


def get_aggregation_groupby(feature, feature_descriptions={}):
    groupby_name = feature.entity.index
    groupby_feature = ft.IdentityFeature(feature.entity[groupby_name])
    if groupby_feature in feature_descriptions or groupby_feature.unique_name() in feature_descriptions:
        return feature_descriptions.get(groupby_feature) or feature_descriptions.get(groupby_feature.unique_name())
    else:
        return groupby_name


def parse_json_metadata(file):
    pass
