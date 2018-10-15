import pandas as pd

from featuretools.utils.gen_utils import make_tqdm_iterator
from featuretools.variable_types.variable import Discrete


def encode_features(feature_matrix, features, top_n=10, include_unknown=True,
                    to_encode=None, inplace=False, verbose=False):
    """Encode categorical features

        Args:
            feature_matrix (pd.DataFrame): Dataframe of features.
            features (list[PrimitiveBase]): Feature definitions in feature_matrix.
            top_n (pd.DataFrame): Number of top values to include.
            include_unknown (pd.DataFrame): Add feature encoding an unknown class.
                defaults to True
            to_encode (list[str]): List of feature names to encode.
                features not in this list are unencoded in the output matrix
                defaults to encode all necessary features.
            inplace (bool): Encode feature_matrix in place. Defaults to False.
            verbose (str): Print progress info.

        Returns:
            (pd.Dataframe, list) : encoded feature_matrix, encoded features

        Example:
            .. ipython:: python
                :suppress:

                from featuretools.tests.testing_utils import make_ecommerce_entityset
                from featuretools.primitives import Feature
                import featuretools as ft
                es = make_ecommerce_entityset()

            .. ipython:: python

                f1 = Feature(es["log"]["product_id"])
                f2 = Feature(es["log"]["purchased"])
                f3 = Feature(es["log"]["value"])

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
    """
    if inplace:
        X = feature_matrix
    else:
        X = feature_matrix.copy()

    encoded = []
    feature_names = []
    for feature in features:
        fname = feature.get_name()
        assert fname in X.columns, (
            "Feature %s not found in feature matrix" % (fname)
        )
        feature_names.append(fname)

    extra_columns = [col for col in X.columns if col not in feature_names]

    if verbose:
        iterator = make_tqdm_iterator(iterable=features,
                                      total=len(features),
                                      desc="Encoding pass 1",
                                      unit="feature")
    else:
        iterator = features

    for f in iterator:
        if (f.expanding or (not issubclass(f.variable_type, Discrete))):
            encoded.append(f)
            continue

        if to_encode is not None and f.get_name() not in to_encode:
            encoded.append(f)
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
        unique = val_counts.head(top_n).index.tolist()
        for label in unique:
            add = f == label
            encoded.append(add)
            X[add.get_name()] = (X[f.get_name()] == label).astype(int)

        if include_unknown:
            unknown = f.isin(unique).NOT().rename(f.get_name() + " is unknown")
            encoded.append(unknown)
            X[unknown.get_name()] = (~X[f.get_name()].isin(unique)).astype(int)

        X.drop(f.get_name(), axis=1, inplace=True)

    new_X = X[[e.get_name() for e in encoded] + extra_columns]
    iterator = new_X.columns
    if verbose:
        iterator = make_tqdm_iterator(iterable=new_X.columns,
                                      total=len(new_X.columns),
                                      desc="Encoding pass 2",
                                      unit="feature")
    for c in iterator:
        if c in extra_columns:
            continue
        try:
            new_X[c] = pd.to_numeric(new_X[c], errors='raise')
        except (TypeError, ValueError):
            pass

    return new_X, encoded
