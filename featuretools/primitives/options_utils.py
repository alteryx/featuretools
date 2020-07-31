import logging
import warnings
from itertools import permutations

from featuretools import primitives
from featuretools.feature_base import IdentityFeature
from featuretools.variable_types import Discrete

logger = logging.getLogger('featuretools')


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


def dict_to_list_variable_check(option, es):
    if not (isinstance(option, dict) and
            all([isinstance(option_val, list) for option_val in option.values()])):
        return False
    else:
        for entity, variables in option.items():
            if entity not in es:
                warnings.warn("Entity '%s' not in entityset" % (entity))
            else:
                for invalid_var in [variable for variable in variables
                                    if variable not in es[entity]]:
                    warnings.warn("Variable '%s' not in entity '%s'" % (invalid_var, entity))
        return True


def list_entity_check(option, es):
    if not isinstance(option, list):
        return False
    else:
        for invalid_entity in [entity for entity in option if entity not in es]:
            warnings.warn("Entity '%s' not in entityset" % (invalid_entity))
        return True


def generate_all_primitive_options(all_primitives,
                                   primitive_options,
                                   ignore_entities,
                                   ignore_variables,
                                   es):
    entityset_dict = {entity.id: [variable.id for variable in entity.variables]
                      for entity in es.entities}
    primitive_options = _init_primitive_options(primitive_options, entityset_dict)
    global_ignore_entities = ignore_entities
    global_ignore_variables = ignore_variables.copy()
    # for now, only use primitive names as option keys
    for primitive in all_primitives:
        if primitive in primitive_options and primitive.name in primitive_options:
            msg = "Options present for primitive instance and generic " \
                  "primitive class (%s), primitive instance will not use generic " \
                  "options" % (primitive.name)
            warnings.warn(msg)
        if primitive in primitive_options or primitive.name in primitive_options:
            options = primitive_options.get(primitive, primitive_options.get(primitive.name))
            # Reconcile global options with individually-specified options
            included_entities = set().union(*[
                option.get('include_entities', set()).union(
                    option.get('include_variables', {}).keys())
                for option in options])
            global_ignore_entities = global_ignore_entities.difference(included_entities)
            for option in options:
                # don't globally ignore a variable if it's included for a primitive
                if 'include_variables' in option:
                    for entity, include_vars in option['include_variables'].items():
                        global_ignore_variables[entity] = \
                            global_ignore_variables[entity].difference(include_vars)
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
    return primitive_options, global_ignore_entities, global_ignore_variables


def _init_primitive_options(primitive_options, es):
    # Flatten all tuple keys, convert value lists into sets, check for
    # conflicting keys
    flattened_options = {}
    for primitive_keys, options in primitive_options.items():
        if not isinstance(primitive_keys, tuple):
            primitive_keys = (primitive_keys,)
        if isinstance(options, list):
            for primitive_key in primitive_keys:
                if isinstance(primitive_key, str):
                    primitive = primitives.get_aggregation_primitives().get(primitive_key) or \
                        primitives.get_transform_primitives().get(primitive_key)
                    if not primitive:
                        msg = "Unknown primitive with name '{}'".format(primitive_key)
                        raise ValueError(msg)
                else:
                    primitive = primitive_key
                assert len(primitive.input_types[0]) == len(options) if \
                    isinstance(primitive.input_types[0], list) else \
                    len(primitive.input_types) == len(options), \
                    "Number of options does not match number of inputs for primitive %s" \
                    % (primitive_key)
            options = [_init_option_dict(primitive_keys, option, es) for option in options]
        else:
            options = [_init_option_dict(primitive_keys, options, es)]

        for primitive in primitive_keys:
            if isinstance(primitive, type):
                primitive = primitive.name

            # if primitive is specified more than once, raise error
            if primitive in flattened_options:
                raise KeyError('Multiple options found for primitive %s' %
                               (primitive))

            flattened_options[primitive] = options
    return flattened_options


def _init_option_dict(key, option_dict, es):
    initialized_option_dict = {}
    primitive_options = _get_primitive_options()
    # verify all keys are valid and match expected type, convert lists to sets
    for option_key, option in option_dict.items():
        if option_key not in primitive_options:
            raise KeyError("Unrecognized primitive option \'%s\' for %s" %
                           (option_key, ','.join(key)))
        if not primitive_options[option_key](option, es):
            raise TypeError("Incorrect type formatting for \'%s\' for %s" %
                            (option_key, ','.join(key)))
        if isinstance(option, list):
            initialized_option_dict[option_key] = set(option)
        elif isinstance(option, dict):
            initialized_option_dict[option_key] = {key: set(option[key]) for key in option}
    # initialize ignore_entities and ignore_variables to empty sets if not present
    if 'ignore_variables' not in initialized_option_dict:
        initialized_option_dict['ignore_variables'] = dict()
    if 'ignore_entities' not in initialized_option_dict:
        initialized_option_dict['ignore_entities'] = set()
    return initialized_option_dict


def variable_filter(f, options, groupby=False):
    if groupby and not issubclass(f.variable_type, Discrete):
        return False
    include_vars = 'include_groupby_variables' if groupby else 'include_variables'
    ignore_vars = 'ignore_groupby_variables' if groupby else 'ignore_variables'
    include_entities = 'include_groupby_entities' if groupby else 'include_entities'
    ignore_entities = 'ignore_groupby_entities' if groupby else 'ignore_entities'

    dependencies = f.get_dependencies(deep=True) + [f]
    for base_f in dependencies:
        if isinstance(base_f, IdentityFeature):
            if include_vars in options and base_f.entity.id in options[include_vars]:
                if base_f.get_name() in options[include_vars][base_f.entity.id]:
                    continue  # this is a valid feature, go to next
                else:
                    return False  # this is not an included feature
            if ignore_vars in options and base_f.entity.id in options[ignore_vars]:
                if base_f.get_name() in options[ignore_vars][base_f.entity.id]:
                    return False  # ignore this feature
        if include_entities in options and \
                base_f.entity.id not in options[include_entities]:
            return False  # not an included entity
        elif ignore_entities in options and \
                base_f.entity.id in options[ignore_entities]:
            return False  # ignore the entity
    return True


def ignore_entity_for_primitive(options, entity, groupby=False):
    # This logic handles whether given options ignore an entity or not
    def should_ignore_entity(option):
        if groupby:
            if 'include_groupby_variables' not in option or entity.id not in option['include_groupby_variables']:
                if 'include_groupby_entities' in option and entity.id not in option['include_groupby_entities']:
                    return True
                elif 'ignore_groupby_entities' in option and entity.id in option['ignore_groupby_entities']:
                    return True
        if 'include_variables' in option and entity.id in option['include_variables']:
            return False
        elif 'include_entities' in option and entity.id not in option['include_entities']:
            return True
        elif entity.id in option['ignore_entities']:
            return True
        else:
            return False
    return any([should_ignore_entity(option) for option in options])


def filter_groupby_matches_by_options(groupby_matches, options):
    return filter_matches_by_options([(groupby_match, ) for groupby_match in groupby_matches],
                                     options,
                                     groupby=True)


def filter_matches_by_options(matches, options, groupby=False, commutative=False):
    # If more than one option, than need to handle each for each input
    if len(options) > 1:
        def is_valid_match(match):
            if all([variable_filter(m, option, groupby) for m, option in zip(match, options)]):
                return True
            else:
                return False
    else:
        def is_valid_match(match):
            if all([variable_filter(f, options[0], groupby) for f in match]):
                return True
            else:
                return False

    valid_matches = set()
    for match in matches:
        if is_valid_match(match):
            valid_matches.add(match)
        elif commutative:
            for order in permutations(match):
                if is_valid_match(order):
                    valid_matches.add(order)
                    break
    return sorted(list(valid_matches), key=lambda features: ([feature.unique_name() for feature in features]))
