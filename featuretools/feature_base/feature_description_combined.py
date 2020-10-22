import json

import featuretools as ft


def describe_feature(feature, feature_descriptions=None, primitive_templates=None,
                     metadata_file=None):
    '''Generates an English language description of a feature.

    Args:
        feature (FeatureBase) : Feature to describe
        feature_descriptions (dict, optional) : dictionary mapping features or unique
            feature names to custom descriptions
        primitive_templates (dict, optional) : dictionary mapping primitives or
            primitive names to description templates
        metadata_file (str, optional) : path to metadata json

    Returns:
        str : English description of the feature
    '''
    if not feature_descriptions:
        feature_descriptions = {}
    if not primitive_templates:
        primitive_templates = {}

    if metadata_file:
        file_feature_descriptions, file_primitive_templates = parse_json_metadata(metadata_file)
        feature_descriptions = {**file_feature_descriptions, **feature_descriptions}
        primitive_templates = {**file_primitive_templates, **primitive_templates}

    descriptions, _ = generate_description(feature, feature_descriptions, primitive_templates)
    descriptions[0] = descriptions[0][:1].upper() + descriptions[0][1:] + '.'
    return ' '.join(descriptions)


def generate_description(feature,
                         feature_descriptions,
                         primitive_templates,
                         already_described=None,
                         split_at_complex=False,
                         current_depth=0):
    '''Recursively builds a feature description. Complex inputs are split out from
    the base description instead of being defined in-line.

    Args:
        feature (FeatureBase) : feature to generate a description list of
        feature_description (dict) : dictionary mapping features or unique feature names to
            custom descriptions
        primitive_templates (dict) : dictionary mapping primitives or primitive names to
            string description templates
        already_described (set, optional) : a set of features that already have already
            seperately been described
        split_at_complex (boolean, optional) : whether description should split complex input
            into seperate sentence. Defaults to False.
        current_depth (int, optional) : how many features are currently described in the
            working description

    Returns:
        list[str] : A list of descriptions. The first element is the working description of the
            given feature, any following elements are completed descriptions of inputs.
    '''
    if not already_described:
        already_described = set()
    described = already_described.copy()

    # Check if has custom description
    if feature in feature_descriptions or feature.unique_name() in feature_descriptions:
        description = (feature_descriptions.get(feature) or
                       feature_descriptions.get(feature.unique_name()))
        return [description], described

    # Check if identity feature:
    if isinstance(feature, ft.IdentityFeature):
        return ['the "{}"'.format(feature.get_name())], described

    # Handle direct features:
    if isinstance(feature, ft.DirectFeature):
        base_feature, direct_description = get_direct_description(feature)
        direct_bases, base_described = generate_description(base_feature,
                                                            feature_descriptions,
                                                            primitive_templates,
                                                            described)
        described.update(base_described)
        if split_at_complex and is_complicated_feature(base_feature, feature_descriptions.keys()):
            return ([base_feature.get_name() + direct_description] +
                    ['"{}" is {}'.format(base_feature.get_name(), direct_bases[0])] +
                    direct_bases[1:]), described
        else:
            return [direct_bases[0] + direct_description] + direct_bases[1:], described

    # Get input descriptions -OR- feature names + adding to list of to explore
    descriptions = []
    to_describe = []
    input_descriptions = []
    input_columns = feature.base_features

    split_next_complex = split_at_complex or is_complicated_feature(feature, feature_descriptions.keys())
    if isinstance(feature, ft.feature_base.FeatureOutputSlice):
        input_columns = feature.base_feature.base_features

    for input_col in input_columns:
        if is_complicated_feature(input_col, feature_descriptions.keys()) and split_next_complex:
            col_description = '"{}"'.format(input_col.get_name())
            to_describe.append(input_col)
        else:
            col_description, base_described = generate_description(input_col,
                                                                   feature_descriptions,
                                                                   primitive_templates,
                                                                   described,
                                                                   split_next_complex)
            described.update(base_described)
            descriptions += col_description[1:]
            col_description = col_description[0]
        input_descriptions.append(col_description)

    # Remove groupby description from input columns
    groupby_description = None
    if isinstance(feature, ft.GroupByTransformFeature):
        groupby_description = input_descriptions.pop()

    # Generate primitive description
    primitive_description = get_primitive_description(feature,
                                                      input_descriptions,
                                                      primitive_templates)
    if isinstance(feature, ft.feature_base.FeatureOutputSlice):
        feature = feature.base_feature

    # Generate groupby phrase if applicable
    groupby = ''
    if isinstance(feature, ft.AggregationFeature):
        groupby_description = get_aggregation_groupby(feature, feature_descriptions)
    if groupby_description is not None:
        if groupby_description.startswith('the '):
            groupby_description = groupby_description[4:]
        groupby = "for each {}".format(groupby_description)

    # Generate aggregation entity phrase w/ use_previous
    entity_description = ''
    if isinstance(feature, ft.AggregationFeature):
        if feature.use_previous:
            entity_description = "of the previous {} of ".format(feature.use_previous.get_name().lower())
        else:
            entity_description = "of all instances of "
        entity_description += '"{}"'.format(feature.relationship_path[-1][1].child_entity.id)

    # Generate where phrase
    where = ''
    if hasattr(feature, 'where') and feature.where:
        where_col, base_described = generate_description(feature.where.base_features[0],
                                                         feature_descriptions,
                                                         primitive_templates,
                                                         described)
        described.update(base_described)
        descriptions += where_col[1:]
        where_col = where_col[0]
        where = 'where {} is {}'.format(where_col, feature.where.primitive.value)

    # Join all parts of template
    description_template = [primitive_description, entity_description, where, groupby]
    description = " ".join([phrase for phrase in description_template if phrase != ''])

    descriptions = [description] + descriptions

    # Attach any descriptions of input columns that need to be defined
    for feat in to_describe:
        if feat in described:
            continue
        feat_name = feat.get_name()
        feat_name = feat_name[:1].upper() + feat_name[1:]
        next_descriptions, child_described = generate_description(feat,
                                                                  feature_descriptions,
                                                                  primitive_templates,
                                                                  described)
        next_descriptions[0] = '"{}" is {}.'.format(feat_name, next_descriptions[0])
        descriptions += next_descriptions
        described.update(child_described)
        described.add(feat)

    return descriptions, described


def get_direct_description(feature):
    direct_description = (' the instance of "{}" associated with '
                          'this instance of "{}"'.format(feature.relationship_path[-1][1].parent_entity.id,
                                                         feature.entity_id))
    base_features = feature.base_features
    while isinstance(base_features[0], ft.DirectFeature):
        base_feat = base_features[0]
        base_feat_description = (' the instance of "{}" associated '
                                 'with'.format(base_feat.relationship_path[-1][1].parent_entity.id))
        direct_description = base_feat_description + direct_description
        base_features = base_feat.base_features
    direct_description = ' of' + direct_description

    return base_features[0], direct_description


def get_primitive_description(feature, input_descriptions, primitive_templates):
    slice_num = None
    template_override = None
    if isinstance(feature, ft.feature_base.FeatureOutputSlice):
        slice_num = feature.n
    if feature.primitive in primitive_templates or feature.primitive.name in primitive_templates:
        template_override = (primitive_templates.get(feature.primitive) or
                             primitive_templates.get(feature.primitive.name))
    primitive_description = feature.primitive.get_description(input_descriptions,
                                                              slice_num=slice_num,
                                                              template_override=template_override)
    return primitive_description


def get_aggregation_groupby(feature, feature_descriptions={}):
    groupby_name = feature.entity.index
    groupby_feature = ft.IdentityFeature(feature.entity[groupby_name])
    if groupby_feature in feature_descriptions or groupby_feature.unique_name() in feature_descriptions:
        return(feature_descriptions.get(groupby_feature) or
               feature_descriptions.get(groupby_feature.unique_name()))
    else:
        return '"{}" in "{}"'.format(groupby_name, feature.entity.id)


def is_complicated_feature(feature, custom_descriptions):
    if feature in custom_descriptions or feature.unique_name() in custom_descriptions:
        return False
    complex_input = False
    if feature.primitive.input_types and isinstance(feature.primitive.input_types[0], list):
        complex_input = len(feature.primitive.input_types[0]) > 1
    elif feature.primitive.input_types:
        complex_input = len(feature.primitive.input_types) > 1
    if isinstance(feature, ft.AggregationFeature) or complex_input:
        return True
    return False


def parse_json_metadata(file):
    with open(file) as f:
        json_metadata = json.load(f)

    return json_metadata.get('feature_descriptions', {}), json_metadata.get('primitive_templates', {})
