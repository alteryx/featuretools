def feature_with_name(features, name):
    for f in features:
        if f.get_name() == name:
            return True

    return False
