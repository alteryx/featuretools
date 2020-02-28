from setuptools import setup

setup(
    name='featuretools_plugin',
    packages=['featuretools_plugin'],
    install_requires=['pandas<0.24.0,>=0.23.0'],
    entry_points={
        'featuretools_plugin': [
            'module = featuretools_plugin',
        ],
    },
)
