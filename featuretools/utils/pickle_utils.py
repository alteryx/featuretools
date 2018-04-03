import cloudpickle

import featuretools as ft


def save_features(features, filepath):
    """Saves the features list to a specificed filepath.

    Args:
        features (list[:class:`.PrimitiveBase`]): List of Feature definitions.

        filepath (str): The location of where to save the pickled features list
             filepath. This must include the name of the file.
    Example:
        .. ipython:: python
            :suppress:

            from featuretools.tests.testing_utils import (
                make_ecommerce_entityset)
            from featuretools.primitives import Feature
            import featuretools as ft
            es = make_ecommerce_entityset()
            import os

        .. code-block:: python

            f1 = Feature(es["log"]["product_id"])
            f2 = Feature(es["log"]["purchased"])
            f3 = Feature(es["log"]["value"])

            features = [f1, f2, f3]

            filepath = os.path.join('/Home/features/', 'list')
            ft.save_features(features, filepath)
    .. seealso::
        :func:`.load_features`
    """
    ft._head_es = features[0].entityset.head(n=10)
    ft._pickling = True
    try:
        save_obj_pickle(features, filepath)
    except Exception:
        ft._pickling = False
        raise
    ft._pickling = False


def load_features(filepath, entityset=None):
    """Loads the features from a filepath.

    Args:
        filepath (str): The location of where pickled features has been saved.
            This must include the name of the file.

        entityset (:class:`.EntitySet`): An already initialized entityset.
            Required.

    Returns:
        features (list[:class:`.PrimitiveBase`]): Feature definitions list.

    Example:
        .. ipython:: python
            :suppress:

            from featuretools.tests.testing_utils import (
                make_ecommerce_entityset)
            from featuretools.primitives import Feature
            import featuretools as ft
            es = make_ecommerce_entityset()
            import os

        .. code-block:: python

            filepath = os.path.join('/Home/features/', 'list')
            ft.load_features(filepath, es)
    .. seealso::
        :func:`.save_features`
    """
    ft._pickling = True
    ft._current_es = entityset

    try:
        features = load_pickle(filepath)
    except Exception:
        ft._current_es = None
        ft._pickling = False
        raise
    ft._current_es = None
    ft._pickling = False
    return features


def save_obj_pickle(obj, filepath):
    with open(filepath, "wb") as out:
        cloudpickle.dump(obj, out)


def load_pickle(filepath):
    filestream = open(filepath, "rb")

    obj = cloudpickle.load(filestream)
    return obj
