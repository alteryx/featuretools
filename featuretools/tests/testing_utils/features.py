import re

from featuretools.entityset.relationship import RelationshipPath


def feature_with_name(features, name):
    for f in features:
        if f.get_name() == name:
            return True
    return False


def number_of_features_with_name_like(features, pattern):
    """Returns number of features with names that match the provided regex pattern"""
    pattern = re.compile(re.escape(pattern))
    names = [f.get_name() for f in features]
    return len([name for name in names if pattern.search(name)])


def backward_path(es, dataframe_ids):
    """
    Create a backward RelationshipPath through the given dataframes. Assumes only
    one such path is possible.
    """

    def _get_relationship(child, parent):
        return next(
            r
            for r in es.get_forward_relationships(child)
            if r._parent_dataframe_name == parent
        )

    relationships = [
        _get_relationship(child, parent)
        for parent, child in zip(dataframe_ids[:-1], dataframe_ids[1:])
    ]

    return RelationshipPath([(False, r) for r in relationships])


def forward_path(es, dataframe_ids):
    """
    Create a forward RelationshipPath through the given dataframes. Assumes only
    one such path is possible.
    """

    def _get_relationship(child, parent):
        return next(
            r
            for r in es.get_forward_relationships(child)
            if r._parent_dataframe_name == parent
        )

    relationships = [
        _get_relationship(child, parent)
        for child, parent in zip(dataframe_ids[:-1], dataframe_ids[1:])
    ]

    return RelationshipPath([(True, r) for r in relationships])


def check_rename(feat, new_name, new_names):
    copy_feat = feat.rename(new_name)
    assert feat.unique_name() != copy_feat.unique_name()
    assert feat.get_name() != copy_feat.get_name()
    assert (
        feat.base_features[0].generate_name()
        == copy_feat.base_features[0].generate_name()
    )
    assert feat.dataframe_name == copy_feat.dataframe_name
    assert feat.get_feature_names() != copy_feat.get_feature_names()
    check_names(copy_feat, new_name, new_names)


def check_names(feat, new_name, new_names):
    assert feat.get_name() == new_name
    assert feat.get_feature_names() == new_names
