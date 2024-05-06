# Install

Featuretools is available for Python 3.9 - 3.12. It can be installed from [pypi](https://pypi.org/project/featuretools/), [conda-forge](https://anaconda.org/conda-forge/featuretools), or from [source](https://github.com/alteryx/featuretools).

To install Featuretools, run the following command:

````{tab} PyPI
```console
$ python -m pip install featuretools
```
````

````{tab} Conda
```console
$ conda install -c conda-forge featuretools
```
````

## Add-ons

Featuretools allows users to install add-ons individually or all at once:

````{tab} PyPI
```{tab} All Add-ons
```console
$ python -m pip install "featuretools[complete]"
```
```{tab} Dask
```console
$ python -m pip install "featuretools[dask]"
```
```{tab} NLP Primitives
```console
$ python -m pip install "featuretools[nlp]"
```
```{tab} Premium Primitives
```console
$ python -m pip install "featuretools[premium]"
```

````
````{tab} Conda
```{tab} All Add-ons
```console
$ conda install -c conda-forge nlp-primitives dask distributed
```
```{tab} NLP Primitives
```console
$ conda install -c conda-forge nlp-primitives
```
```{tab} Dask
```console
$ conda install -c conda-forge dask distributed
```
````

- **NLP Primitives**: Use Natural Language Processing Primitives in Featuretools
- **Premium Primitives**: Use primitives from Premium Primitives in Featuretools
- **Dask**: Use to run `calculate_feature_matrix` in parallel with `n_jobs`

## Installing Graphviz

In order to use `EntitySet.plot` or `featuretools.graph_feature` you will need to install the graphviz library.

````{tab} macOS (Intel, M1)
:new-set:
```{tab} pip
```console
$ brew install graphviz
$ python -m pip install graphviz
```
```{tab} conda
```console
$ brew install graphviz
$ conda install -c conda-forge python-graphviz
```
````

````{tab} Ubuntu
```{tab} pip
```console
$ sudo apt install graphviz
$ python -m pip install graphviz
```
```{tab} conda
```console
$ sudo apt install graphviz
$ conda install -c conda-forge python-graphviz
```
````

````{tab} Windows
```{tab} pip
```console
$ python -m pip install graphviz
```
```{tab} conda
```console
$ conda install -c conda-forge python-graphviz
```
````

If you installed graphviz for **Windows** with `pip`, install graphviz.exe from the [official source](https://graphviz.org/download/#windows).

## Source

To install Featuretools from source, clone the repository from [GitHub](https://github.com/alteryx/featuretools), and install the dependencies.

```bash
git clone https://github.com/alteryx/featuretools.git
cd featuretools
python -m pip install .
```

## Docker

It is also possible to run Featuretools inside a Docker container.
You can do so by installing it as a package inside a container (following the normal install guide) or
creating a new image with Featuretools pre-installed, using the following commands in your `Dockerfile`:

```dockerfile
FROM --platform=linux/x86_64 python:3.9-slim-buster
RUN apt update && apt -y update
RUN apt install -y build-essential
RUN pip3 install --upgrade --quiet pip
RUN pip3 install featuretools
```

# Development

To make contributions to the codebase, please follow the guidelines [here](https://github.com/alteryx/featuretools/blob/main/contributing.md).
