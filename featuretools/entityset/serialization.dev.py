def relationships(es):
    keys = {
        'parent': ['parent_entity', 'parent_variable'],
        'child': ['child_entity', 'child_variable']
    }
    return [{k: [getattr(r, a) for a in attrs]
             for k, attrs in keys.items()} for r in es.relationships]


def entity(es):
    def attr(e, key):
        if key in ['id', 'index', 'time_index']:
            return getattr(e, key)
        elif key == 'variables':
            return [v.create_metadata_dict() for v in getattr(e, key)]
        elif key == 'loading_info':
            # data files
            return {}
        else:
            raise ValueError("unable to serialize '{}'".format(key))

    attrs = ['id', 'index', 'time_index', 'variables', 'loading_info']
    descr = {e.id: {a: attr(e, a) for a in attrs} for e in es.entities}
    # a = 'secondary_time_index'
    # {'properties': {a: getattr(entity, a)}}
    # descr.update(relationships=relationships(es))
    return descr
