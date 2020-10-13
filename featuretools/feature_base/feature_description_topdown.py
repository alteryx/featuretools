import featuretools as ft
from featuretools.feature_base.utils import PRIMITIVE_TEMPLATES, convert_to_nth


def describe_feature(feature, feature_descriptions={}, primitive_templates={},
                     metadata_file=None):
    '''Generates an English language description of a feature.

    Args:
        feature (FeatureBase) : Feature to describe
        feature_descriptions (dict, optional) : dictionary mapping features or unique feature names to custom descriptions
        primitive_templates (dict, optional) : dictionary mapping primitives or primitive names to description templates
        metadata_file (str, optional) : path to metadata json

    Returns:
        str : English description of the feature
    '''
    templates = PRIMITIVE_TEMPLATES.copy()
    if metadata_file:
        file_feature_descriptions, file_primitive_templates = parse_json_metadata(metadata_file)
        feature_descriptions = {**file_feature_descriptions, **feature_descriptions}
        primitive_templates = {**file_primitive_templates, **primitive_templates}
    metadata = {
        'feature_descriptions': feature_descriptions,
        'primitive_templates': {**templates, **primitive_templates}
    }

    description = generate_description(feature, metadata)
    return description[:1].upper() + description[1:]


def generate_description(feature, metadata=None):
    feature_descriptions = metadata['feature_descriptions'] if metadata else {}

    # 1) Check if has its own description
    if feature in feature_descriptions or feature.unique_name() in feature_descriptions:
        return (feature_descriptions.get(feature) or feature_descriptions.get(feature.unique_name())) + '.'

    # 2) Check if identity feature:
    if isinstance(feature, ft.IdentityFeature):
        return get_identity_description(feature, metadata) + '.'

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
            col_description = '"{}"'.format(input_col.get_name())
            if not isinstance(input_col, ft.IdentityFeature):
                to_describe.append(input_col)
        input_descriptions.append(col_description)

    if isinstance(feature, ft.GroupByTransformFeature):
        groupby_description = input_descriptions.pop()

    # 3) Deal with direct features  
    # Generate primitive description
    if isinstance(feature, ft.DirectFeature):
        primitive_description = get_direct_description(feature, input_descriptions, metadata)
    else:
        primitive_description = get_primitive_description(feature, input_descriptions, metadata)
        if isinstance(feature, ft.feature_base.FeatureOutputSlice):
            feature = feature.base_feature

    # Generate groupby phrase if applicable
    groupby = ''
    if isinstance(feature, ft.AggregationFeature):
        groupby_name = get_aggregation_groupby(feature, metadata)
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
            if not isinstance(feature.where.base_features[0], ft.IdentityFeature):
                to_describe.append(feature.where.base_features[0])
            where_col = feature.where.base_features[0].get_name()
            # where_col = generate_description(feature.where.base_features[0], metadata)[:-1]
        where = 'where "{}" is {}'.format(where_col, where_value)

    # 8) join all parts of template
    description_template = [primitive_description, entity_description, where, groupby]
    description = " ".join([phrase for phrase in description_template if phrase != '']) + '.'

    # 9) attach any descriptions of input columns that need to be defined
    for feat in to_describe:
        next_description = ' ' + '"{}" is {}'.format(feat.get_name(), generate_description(feat, metadata))
        description += next_description

    return description


def get_identity_description(feature, metadata=None):
    feature_descriptions = metadata['feature_descriptions'] if metadata else {}
    if feature in feature_descriptions or feature.unique_name() in feature_descriptions:
        return feature_descriptions.get(feature) or feature_descriptions.get(feature.unique_name())
    else:
        return 'the "{}"'.format(feature.get_name())


def get_direct_description(feature, input_columns, metadata=None):
    direct_base = input_columns[0]
    if direct_base.endswith(' of the instance of "{}"'.format(feature.relationship_path[-1][1].parent_entity.id)):
        return 'the ' + direct_base + ' associated with this instance of "{}"'.format(feature.entity_id)
    else:
        return 'the ' + direct_base + ' of the instance of "{}" associated with this instance of "{}"'.format(feature.relationship_path[-1][1].parent_entity.id,
                                                                                                              feature.entity_id)


def get_aggregation_groupby(feature, metadata=None):
    feature_descriptions = metadata['feature_descriptions'] if metadata else {}
    groupby_name = feature.entity.index
    groupby_feature = ft.IdentityFeature(feature.entity[groupby_name])
    if groupby_feature in feature_descriptions or groupby_feature.unique_name() in feature_descriptions:
        return feature_descriptions.get(groupby_feature) or feature_descriptions.get(groupby_feature.unique_name())
    else:
        return groupby_name


def get_primitive_description(feature, input_columns, metadata=None):
    '''Returns a completed primitive description from a template by inputting the given input column descriptions
    Args:
        feature (FeatureBase) : Feature whose primitive is to be described
        input_columns (list[str]) : Descriptions of the input columns
        metadata (dict, optional) : Metadata to use to describe primitive
    '''
    primitive_templates = metadata['primitive_templates'] if metadata else {}

    # 1) check if this feature's primitive has a description template
    if feature.primitive in primitive_templates or feature.primitive.name in primitive_templates:
        kwargs = {'feature': feature}
        template = primitive_templates.get(feature.primitive) or primitive_templates.get(feature.primitive.name)
        if isinstance(template, list):
            # this template is for a multi-output/slice feature
            if isinstance(feature, ft.feature_base.FeatureOutputSlice):
                # add assert that length is either 2 or matches the number of outputs?
                kwargs['feature_slice'] = convert_to_nth(feature.n + 1)
                if len(template) == feature.num_output_parent + 1:
                    template = template[feature.n + 1]
                elif len(template) == 2:
                    template = template[1]
                else:
                    raise ValueError("Number of output features does not match multi-output primitive description templates")
            else:
                # this is the primary feature
                template = template[0]
        description = template.format(*input_columns, **kwargs)

    # 2) If this feature has no template, use the default:
    else:
        if isinstance(feature, ft.feature_base.FeatureOutputSlice):
            # deal with the fact this is an output slice
            nth_slice = convert_to_nth(feature.n)
            description = "the {} ouput from applying {} to {}".format(nth_slice,
                                                                       feature.primitive.name,
                                                                       ', '.join(input_columns))

        else:
            description = "the result of applying {} to {}".format(feature.primitive.name, ', '.join(input_columns))
    return description


def parse_json_metadata(file):
    pass
