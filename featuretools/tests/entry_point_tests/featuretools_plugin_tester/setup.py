from setuptools import find_packages, setup

setup(
    name='featuretools_plugin_tester',
    packages=find_packages(),
    version='0.1',
    description='Module to test if the featuretools entry point',
    entry_points={
        'featuretools_primitives': [
            'featuretools_plugin_tester = featuretools_plugin_tester'
        ],
        'featuretools_initialize': [
            'initialize = featuretools_plugin_tester.initialize'
        ],
        'featuretools_dfs': [
            'dfs = featuretools_plugin_tester.dfs'
        ]
    },
    include_package_data=True,
    install_requires=[],
)
