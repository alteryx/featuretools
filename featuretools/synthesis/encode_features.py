import logging

import pandas as pd

from featuretools.utils.gen_utils import make_tqdm_iterator
from featuretools.variable_types.variable import Discrete

logger = logging.getLogger('featuretools')

DEFAULT_TOP_N = 10


def encode_features(feature_matrix, features, top_n=DEFAULT_TOP_N, include_unknown=True,
                    to_encode=None, inplace=False, drop_first=False, verbose=False):
    """Encode categorical features

        Args:
            feature_matrix (pd.DataFrame): Dataframe of features.
            features (list[PrimitiveBase]): Feature definitions in feature_matrix.
            top_n (int or dict[string -> int]): Number of top values to include.
                If dict[string -> int] is used, key is feature name and value is
                the number of top values to include for that feature.
                If a feature's name is not in dictionary, a default value of 10 is used.
            include_unknown (pd.DataFrame): Add feature encoding an unknown class.
                defaults to True
            to_encode (list[str]): List of feature names to encode.
                features not in this list are unencoded in the output matrix
                defaults to encode all necessary features.
            inplace (bool): Encode feature_matrix in place. Defaults to False.
            drop_first (bool): Whether to get k-1 dummies out of k categorical
                    levels by removing the first level.
                    defaults to False
            verbose (str): Print progress info.

        Returns:
            (pd.Dataframe, list) : encoded feature_matrix, encoded features

        Example:
            .. ipython:: python
                :suppress:

                from featuretools.tests.testing_utils import make_ecommerce_entityset
                import featuretools as ft
                es = make_ecommerce_entityset()

            .. ipython:: python

                f1 = ft.Feature(es["log"]["product_id"])
                f2 = ft.Feature(es["log"]["purchased"])
                f3 = ft.Feature(es["log"]["value"])

                features = [f1, f2, f3]
                ids = [0, 1, 2, 3, 4, 5]
                feature_matrix = ft.calculate_feature_matrix(features, es,
                                                             instance_ids=ids)

                fm_encoded, f_encoded = ft.encode_features(feature_matrix,
                                                           features)
                f_encoded

                fm_encoded, f_encoded = ft.encode_features(feature_matrix,
                                                           features, top_n=2)
                f_encoded

                fm_encoded, f_encoded = ft.encode_features(feature_matrix, features,
                                                           include_unknown=False)
                f_encoded

                fm_encoded, f_encoded = ft.encode_features(feature_matrix, features,
                                                           to_encode=['purchased'])
                f_encoded

                fm_encoded, f_encoded = ft.encode_features(feature_matrix, features,
                                                           drop_first=True)
                f_encoded
    """
    if not isinstance(feature_matrix, pd.DataFrame):
        msg = "feature_matrix must be a Pandas DataFrame"
        raise TypeError(msg)

    if inplace:
        X = feature_matrix
    else:
        X = feature_matrix.copy()

    old_feature_names = set()
    for feature in features:
        for fname in feature.get_feature_names():
            assert fname in X.columns, ("Feature %s not found in feature matrix" % (fname))
            old_feature_names.add(fname)

    pass_through = [col for col in X.columns if col not in old_feature_names]

    if verbose:
        iterator = make_tqdm_iterator(iterable=features,
                                      total=len(features),
                                      desc="Encoding pass 1",
                                      unit="feature")
    else:
        iterator = features

    new_feature_list = []
    new_columns = []
    encoded_columns = set()

    for f in iterator:
        # TODO: features with multiple columns are not encoded by this method,
        # which can cause an "encoded" matrix with non-numeric vlaues
        is_discrete = issubclass(f.variable_type, Discrete)
        if (f.number_output_features > 1 or not is_discrete):
            if f.number_output_features > 1:
                logger.warning("Feature %s has multiple columns and will not "
                               "be encoded.  This may result in a matrix with"
                               " non-numeric values." % (f))
            new_feature_list.append(f)
            new_columns.extend(f.get_feature_names())
            continue

        if to_encode is not None and f.get_name() not in to_encode:
            new_feature_list.append(f)
            new_columns.extend(f.get_feature_names())
            continue

        val_counts = X[f.get_name()].value_counts().to_frame()
        index_name = val_counts.index.name
        if index_name is None:
            if 'index' in val_counts.columns:
                index_name = 'level_0'
            else:
                index_name = 'index'
        val_counts.reset_index(inplace=True)
        val_counts = val_counts.sort_values([f.get_name(), index_name],
                                            ascending=False)
        val_counts.set_index(index_name, inplace=True)
        select_n = top_n
        if isinstance(top_n, dict):
            select_n = top_n.get(f.get_name(), DEFAULT_TOP_N)
        if drop_first:
            select_n = min(len(val_counts), top_n)
            select_n = max(select_n - 1, 1)
        unique = val_counts.head(select_n).index.tolist()
        for label in unique:
            add = f == label
            add_name = add.get_name()
            new_feature_list.append(add)
            new_columns.append(add_name)
            encoded_columns.add(add_name)
            X[add_name] = (X[f.get_name()] == label)

        if include_unknown:
            unknown = f.isin(unique).NOT().rename(f.get_name() + " is unknown")
            unknown_name = unknown.get_name()
            new_feature_list.append(unknown)
            new_columns.append(unknown_name)
            encoded_columns.add(unknown_name)
            X[unknown_name] = (~X[f.get_name()].isin(unique))

        X.drop(f.get_name(), axis=1, inplace=True)

    new_columns.extend(pass_through)
    new_X = X[new_columns]
    iterator = new_X.columns
    if verbose:
        iterator = make_tqdm_iterator(iterable=new_X.columns,
                                      total=len(new_X.columns),
                                      desc="Encoding pass 2",
                                      unit="feature")
    for c in iterator:
        if c in encoded_columns:
            try:
                new_X[c] = pd.to_numeric(new_X[c], errors='raise')
            except (TypeError, ValueError):
                pass

    return new_X, new_feature_list
