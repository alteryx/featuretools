from setuptools import setup

setup(
    name='featuretools_primitives',
    packages=['featuretools_primitives'],
    entry_points={
        'featuretools_primitives': [
            'featuretools_primitives = featuretools_primitives',
        ],
    },
)
