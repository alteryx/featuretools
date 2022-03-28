from setuptools import find_packages, setup

setup(
    name="featuretools_primitives",
    packages=find_packages(),
    entry_points={
        "featuretools_primitives": [
            "new = featuretools_primitives.new_primitive",
            "invalid = featuretools_primitives.invalid_primitive",
            "existing = featuretools_primitives.existing_primitive",
        ],
    },
)
