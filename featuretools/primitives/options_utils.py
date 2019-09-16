from featuretools.feature_base import IdentityFeature


def _get_primitive_options():
    # all possible option keys: function that verifies value type
    return {'ignore_entities': list_entity_check,
            'include_entities': list_entity_check,
            'ignore_variables': dict_to_list_variable_check,
            'include_variables': dict_to_list_variable_check,
            'ignore_groupby_entities': list_entity_check,
            'include_groupby_entities': list_entity_check,
            'ignore_groupby_variables': dict_to_list_variable_check,
            'include_groupby_variables': dict_to_list_variable_check}


def dict_to_list_variable_check(option):
    return (isinstance(option, dict) and
            all([isinstance(option_val, list) for option_val in option.values()]))


def list_entity_check(option):
    return (isinstance(option, list))


def generate_all_primitive_options(all_primitives,
                                   primitive_options,
                                   ignore_entities,
                                   ignore_variables):
    primitive_options = _init_primitive_options(primitive_options)
    global_ignore_entities = ignore_entities
    # for now, only use primitive names as option keys
    for primitive in all_primitives:
        if not isinstance(primitive, str):
            primitive = primitive.name
        if primitive in primitive_options:
            # Reconcile global options with individually-specified options
            options = primitive_options[primitive]
            included_entities = set().union(
                options.get('include_entities') if options.get('include_entities') else set([]),
                options.get('include_variables').keys() if options.get('include_variables') else set([])
            )
            global_ignore_entities = global_ignore_entities.difference(included_entities)
            options['ignore_entities'] = options['ignore_entities'].union(ignore_entities.difference(included_entities))
            for entity, ignore_vars in ignore_variables.items():
                if entity in included_entities:
                    continue
                if entity in options['ignore_variables']:
                    options['ignore_variables'][entity] = options['ignore_variables'][entity].union(ignore_vars)
                else:
                    options['ignore_variables'][entity] = ignore_vars
        else:
            # no user specified options, just use global defaults
            primitive_options[primitive] = {'ignore_entities': ignore_entities,
                                            'ignore_variables': ignore_variables}
    return primitive_options, global_ignore_entities


def _init_primitive_options(primitive_options):
    # Flatten all tuple keys, convert value lists into sets, check for
    # conflicting keys
    flattened_options = {}
    for primitive_key, option_dict in primitive_options.items():
        option_dict = _init_option_dict(primitive_key, option_dict)
        if isinstance(primitive_key, tuple):
            for each_primitive in primitive_key:
                # if primitive is specified more than once, raise error
                if each_primitive in flattened_options:
                    raise KeyError('Multiple options found for primitive %s' %
                                   (each_primitive))
                flattened_options[each_primitive] = option_dict
        else:
            # if primitive is specified more than once, raise error
            if primitive_key in flattened_options:
                raise KeyError('Multiple options found for primitive %s' %
                               (primitive_key))
            flattened_options[primitive_key] = option_dict
    return flattened_options


def _init_option_dict(key, option_dict):
    initialized_option_dict = {}
    primitive_options = _get_primitive_options()
    # verify all keys are valid and match expected type, convert lists to sets
    for option_key, option in option_dict.items():
        if option_key not in primitive_options:
            raise KeyError("Unrecognized primitive option \'%s\' for %s" %
                           (option_key, key))
        if not primitive_options[option_key](option):
            raise TypeError("Incorrect type formatting for \'%s\' for %s" %
                            (option_key, key))
        if isinstance(option, list):
            initialized_option_dict[option_key] = set(option)
        elif isinstance(option, dict):
            initialized_option_dict[option_key] = {key: set(option[key]) for key in option}
    # initialize ignore_entities and ignore_variables to empty sets if not present
    if 'ignore_variables' not in initialized_option_dict:
        initialized_option_dict['ignore_variables'] = {}
    if 'ignore_entities' not in initialized_option_dict:
        initialized_option_dict['ignore_entities'] = set([])
    return initialized_option_dict


def variable_filter_generator(options):
    if 'include_variables' in options:
        def variable_filter(f):
            return (not isinstance(f, IdentityFeature) or
                    (f.entity.id in options['include_variables'] and
                    f.variable.id in options['include_variables'][f.entity.id]))

    elif 'ignore_variables' in options:
        def variable_filter(f):
            return (not isinstance(f, IdentityFeature) or
                    f.entity.id not in options['ignore_variables'] or
                    f.variable.id not in options['ignore_variables'][f.entity.id])

    else:
        def variable_filter(f):
            return True
    return variable_filter


def groupby_filter_generator(options):
    if 'include_groupby_variables' in options:
        def groupby_filter(f):
            return (not isinstance(f, IdentityFeature) or
                    f.entity.id in options['include_groupby_variables'] and
                    f.variable.id in options['include_groupby_variables'][f.entity.id])
    elif 'ignore_groupby_variables' in options:
        def groupby_filter(f):
            return (not isinstance(f, IdentityFeature) or
                    f.entity.id not in options['ignore_groupby_variables'] or
                    f.variable.id not in options['ignore_groupby_variables'][f.entity.id])
    else:
        def groupby_filter(f):
            return True
    return groupby_filter
