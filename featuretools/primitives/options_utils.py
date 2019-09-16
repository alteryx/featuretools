from featuretools.feature_base import IdentityFeature


def get_primitive_options():
    # all possible option keys:anonymous value type checker
    return {'ignore_entities': (lambda x: isinstance(x, list)),
            'include_entities': (lambda x: isinstance(x, list)),
            'ignore_variables': (lambda x: isinstance(x, dict) and
                                 all([isinstance(y, list) for y in x.values()])),
            'include_variables': (lambda x: isinstance(x, dict) and
                                  all([isinstance(y, list) for y in x.values()])),
            'ignore_groupby_entities': (lambda x: isinstance(x, list)),
            'include_groupby_entities': (lambda x: isinstance(x, list)),
            'ignore_groupby_variables': (lambda x: isinstance(x, dict) and
                                         all([isinstance(y, list) for y in x.values()])),
            'include_groupby_variables': (lambda x: isinstance(x, dict) and
                                          all([isinstance(y, list) for y in x.values()]))}


def generate_all_primitive_options(all_primitives,
                                   primitive_options,
                                   ignore_entities,
                                   ignore_variables):
    primitive_options = init_primitive_options(primitive_options)
    global_ignore_entities = ignore_entities
    for primitive in all_primitives:
        if not isinstance(primitive, str):
            primitive = primitive.name
        if primitive in primitive_options:
            # have to reconcile global ignored with individual options
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


def init_primitive_options(primitive_options):
    # Flatten all tuple keys, convert value lists into sets, check for
    # conflicting keys
    flattened_options = {}
    for primitive_key, option_dict in primitive_options.items():
        option_dict = init_option_dict(primitive_key, option_dict)
        if isinstance(primitive_key, tuple):
            for each_primitive in primitive_key:
                assert each_primitive not in flattened_options,\
                    "Conflicting primitive options found for " + str(each_primitive)
                flattened_options[each_primitive] = option_dict
        else:
            assert primitive_key not in flattened_options,\
                "Conflicting primitive options found for " + str(primitive_key)
            flattened_options[primitive_key] = option_dict
    return flattened_options


def init_option_dict(key, option_dict):
    initialized_option_dict = {}
    primitive_options = get_primitive_options()
    # verify all keys are valid and match expected type, convert lists to sets
    for option_key, option in option_dict.items():
        assert option_key in primitive_options,\
            'Unrecognized primitive option \'' + str(option_key) + '\' for \'' + str(key) + '\''
        assert primitive_options[option_key](option),\
            'Incorrect type formatting for \'' + str(option_key) + '\' for \'' + str(key) + '\''
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
            print(options['include_variables'])
            print(f.entity.id, f.variable.id)
            print((not isinstance(f, IdentityFeature) or
                    (f.entity.id in options['include_variables'] and
                    f.variable.id in options['include_variables'][f.entity.id])))
            print("\n")

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
