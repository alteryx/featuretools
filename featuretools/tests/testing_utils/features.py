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


def check_rename(feat, new_name, new_names):
    copy_feat = feat.rename(new_name)
    assert feat.unique_name() != copy_feat.unique_name()
    assert feat.get_name() != copy_feat.get_name()
    assert feat.base_features[0].generate_name() == copy_feat.base_features[0].generate_name()
    assert feat.entity == copy_feat.entity
    assert feat.get_feature_names() != copy_feat.get_feature_names()
    check_names(copy_feat, new_name, new_names)


def check_names(feat, new_name, new_names):
    assert feat.get_name() == new_name
    assert feat.get_feature_names() == new_names
