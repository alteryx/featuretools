import cloudpickle


def save_features(features, filepath):
    """Saves the features list to a specificed filepath.

    Args:
        features (list[:class:`.FeatureBase`]): List of Feature definitions.

        filepath (str): The location of where to save the pickled features list
             filepath. This must include the name of the file.

    Note:
        Features saved in one version of Featuretools are not guaranteed to work in another.
        After upgrading Featuretools, features may need to be generated again.

    Example:
        .. ipython:: python
            :suppress:

            from featuretools.tests.testing_utils import (
                make_ecommerce_entityset)
            import featuretools as ft
            es = make_ecommerce_entityset()
            import os

        .. code-block:: python

            f1 = ft.Feature(es["log"]["product_id"])
            f2 = ft.Feature(es["log"]["purchased"])
            f3 = ft.Feature(es["log"]["value"])

            features = [f1, f2, f3]

            filepath = os.path.join('/Home/features/', 'list')
            ft.save_features(features, filepath)
    .. seealso::
        :func:`.load_features`
    """
    save_obj_pickle(features, filepath)


def load_features(filepath):
    """Loads the features from a filepath.

    Args:
        filepath (str): The location of where pickled features has been saved.
            This must include the name of the file.

    Returns:
        features (list[:class:`.FeatureBase`]): Feature definitions list.

    Note:
        Features saved in one version of Featuretools are not guaranteed to work in another.
        After upgrading Featuretools, features may need to be generated again.

    Example:
        .. ipython:: python
            :suppress:

            import featuretools as ft
            import os

        .. code-block:: python

            filepath = os.path.join('/Home/features/', 'list')
            ft.load_features(filepath)
    .. seealso::
        :func:`.save_features`
    """
    return load_pickle(filepath)


def save_obj_pickle(obj, filepath):
    with open(filepath, "wb") as out:
        cloudpickle.dump(obj, out)


def load_pickle(filepath):
    filestream = open(filepath, "rb")

    obj = cloudpickle.load(filestream)
    return obj
