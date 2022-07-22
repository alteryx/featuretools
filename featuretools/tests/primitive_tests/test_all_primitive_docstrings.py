from featuretools.primitives import get_aggregation_primitives, get_transform_primitives


def docstring_is_uniform(primitive):
    docstring = primitive.__doc__
    valid_verbs = [
        "Calculates",
        "Determines",
        "Transforms",
        "Computes",
        "Counts",
        "Negates",
        "Adds",
        "Subtracts",
        "Multiplies",
        "Divides",
        "Performs",
        "Returns",
        "Shifts",
        "Extracts",
        "Applies",
    ]
    return any(docstring.startswith(s) for s in valid_verbs)


def test_transform_primitive_docstrings():
    for primitive in get_transform_primitives().values():
        assert docstring_is_uniform(primitive)


def test_aggregation_primitive_docstrings():
    for primitive in get_aggregation_primitives().values():
        assert docstring_is_uniform(primitive)
