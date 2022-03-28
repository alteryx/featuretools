from setuptools import setup

setup(
    name="featuretools_plugin",
    packages=["featuretools_plugin"],
    entry_points={
        "featuretools_plugin": [
            "module = featuretools_plugin",
        ],
    },
)
