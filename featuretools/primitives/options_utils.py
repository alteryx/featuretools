from featuretools import primitives
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
            included_entities = set().union(*[set().union(
                option.get('include_entities') if option.get('include_entities') else set([]),
                option.get('include_variables').keys() if option.get('include_variables') else set([]))
                for option in options])
            global_ignore_entities = global_ignore_entities.difference(included_entities)
            for option in options:
                option['ignore_entities'] = option['ignore_entities'].union(
                    ignore_entities.difference(included_entities)
                )
            for entity, ignore_vars in ignore_variables.items():
                # if already ignoring variables for this entity, add globals
                for option in options:
                    if entity in option['ignore_variables']:
                        option['ignore_variables'][entity] = option['ignore_variables'][entity].union(ignore_vars)
                    # if no ignore_variables and entity is explicitly included, don't ignore the variable
                    elif entity in included_entities:
                        continue
                    # Otherwise, keep the global option
                    else:
                        option['ignore_variables'][entity] = ignore_vars
        else:
            # no user specified options, just use global defaults
            primitive_options[primitive] = [{'ignore_entities': ignore_entities,
                                            'ignore_variables': ignore_variables}]
    return primitive_options, global_ignore_entities


def _init_primitive_options(primitive_options):
    # Flatten all tuple keys, convert value lists into sets, check for
    # conflicting keys
    flattened_options = {}
    for primitive_key, options in primitive_options.items():
        if isinstance(options, list):
            primitive = primitives.get_aggregation_primitives().get(primitive_key) or \
                primitives.get_transform_primitives().get(primitive_key)
            assert len(primitive.input_types[0]) == len(options) if \
                isinstance(primitive.input_types[0], list) else \
                len(primitive.input_types) == len(options), \
                "Number of options does not match number of inputs for primitive %s" \
                % (primitive_key)
            options = [_init_option_dict(primitive_key, option) for option in options]
        else:
            options = [_init_option_dict(primitive_key, options)]
        if not isinstance(primitive_key, tuple):
            primitive_key = (primitive_key,)
        for each_primitive in primitive_key:
            # if primitive is specified more than once, raise error
            if each_primitive in flattened_options:
                raise KeyError('Multiple options found for primitive %s' %
                               (each_primitive))
            flattened_options[each_primitive] = options
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


def _variable_filter_generator(options):
    def filter_ignores(f):
        return (not isinstance(f, IdentityFeature) or
                f.entity.id not in options['ignore_variables'] or
                f.variable.id not in options['ignore_variables'][f.entity.id])

    def filter_includes(f):
        return (not isinstance(f, IdentityFeature) or
                (f.entity.id in options['include_variables'] and
                f.variable.id in options['include_variables'][f.entity.id]))

    if 'include_variables' in options:
        def variable_filter(f):
            return filter_includes(f) or \
                (f.entity.id not in options['include_variables'] and
                    filter_ignores(f))
    # ignore options initialized to set() if not present
    else:
        def variable_filter(f):
            return filter_ignores(f)
    return variable_filter


def _groupby_filter_generator(options):
    def filter_include_groupby(f):
        return (isinstance(f, IdentityFeature) and
                (f.entity.id in options['include_groupby_variables'] and
                f.variable.id in options['include_groupby_variables'][f.entity.id]))

    def filter_ignore_groupby(f):
        return (isinstance(f, IdentityFeature) and (
                f.entity.id not in options['ignore_groupby_variables'] or
                (f.entity.id in options['ignore_groupby_variables'] and
                    f.variable.id not in options['ignore_groupby_variables'][f.entity.id])))

    if 'include_groupby_variables' in options and 'ignore_groupby_variables' in options:
        def groupby_filter(f):
            return filter_include_groupby(f) or \
                (f.entity.id not in options['include_groupby_variables'] and
                    filter_ignore_groupby(f))
    elif 'include_groupby_variables' in options:
        def groupby_filter(f):
            return filter_include_groupby(f) or \
                f.entity.id not in options['include_groupby_variables']
    elif 'ignore_groupby_variables' in options:
        def groupby_filter(f):
            return filter_ignore_groupby(f)
    else:
        def groupby_filter(f):
            return (isinstance(f, IdentityFeature))
    return groupby_filter


def ignore_entity(options, entity, groupby=False):
    # This logic handles whether given options ignore an entity or not
    if len(options) > 1:
        return any([ignore_entity([option], entity, groupby) for option in options])
    else:
        if 'include_entities' in options[0] and \
                entity.id not in options[0]['include_entities'] or \
                groupby and 'include_groupby_entities' in options[0] and \
                entity.id not in options['include_groupby_entities']:
            return True
        elif entity.id in options[0]['ignore_entities'] or \
                groupby and 'ignore_groupby_entities' in options[0] and \
                entity.id in options[0]['ignore_groupby_entities']:
            return True
        else:
            return False


def filter_matches_by_options(matches, options, groupby=False):
    generator = _groupby_filter_generator if groupby else _variable_filter_generator
    if len(options) > 1:
        variable_filter = [generator(option) for option in options]
    else:
        variable_filter = generator(options[0])
    valid_matches = set()
    for match in matches:
        if isinstance(variable_filter, list) and \
                all([variable_filter[i](match[i]) for i in range(len(variable_filter))]):
            valid_matches.add(match)
        elif not isinstance(variable_filter, list) and \
                all([variable_filter(f) for f in match]):
            valid_matches.add(match)
    return valid_matches
