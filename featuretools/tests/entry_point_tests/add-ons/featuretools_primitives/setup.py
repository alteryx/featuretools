from setuptools import setup

setup(
    name='featuretools_primitives',
    packages=['featuretools_primitives'],
    entry_points={
        'featuretools_primitives': [
            'new = featuretools_primitives.new_primitive',
            'invalid = featuretools_primitives.invalid_primitive',
            'existing = featuretools_primitives.existing_primitive',
        ],
    },
)
