from featuretools.entityset.relationship import RelationshipPath


def feature_with_name(features, name):
    for f in features:
        if f.get_name() == name:
            return True

    return False


def backward_path(es, entity_ids):
    """
    Create a backward RelationshipPath through the given entities. Assumes only
    one such path is possible.
    """
    def _get_relationship(child, parent):
        return next(r for r in es.get_forward_relationships(child)
                    if r.parent_entity.id == parent)

    relationships = [_get_relationship(child, parent)
                     for parent, child in zip(entity_ids[:-1], entity_ids[1:])]

    return RelationshipPath([(False, r) for r in relationships])


def forward_path(es, entity_ids):
    """
    Create a forward RelationshipPath through the given entities. Assumes only
    one such path is possible.
    """
    def _get_relationship(child, parent):
        return next(r for r in es.get_forward_relationships(child)
                    if r.parent_entity.id == parent)

    relationships = [_get_relationship(child, parent)
                     for child, parent in zip(entity_ids[:-1], entity_ids[1:])]

    return RelationshipPath([(True, r) for r in relationships])
