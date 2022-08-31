# Install

Featuretools is available for Python 3.7, 3.8, 3.9, and 3.10. It can be installed from [pypi](https://pypi.org/project/featuretools/), [conda-forge](https://anaconda.org/conda-forge/featuretools), or from [source](https://github.com/alteryx/featuretools).

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

```{hint}
Be sure to install [Scala and Spark](#scala-and-spark) if you want to use Spark
```

````{tab} PyPI
```{tab} All Add-ons
```console
$ python -m pip install "featuretools[complete]"
```
```{tab} NLP Primitives
```console
$ python -m pip install "featuretools[nlp]"
```
```{tab} Spark
```console
$ python -m pip install "featuretools[spark]"
```
```{tab} TSFresh Primitives
```console
$ python -m pip install "featuretools[tsfresh]"
```
```{tab} AutoNormalize
```console
$ python -m pip install "featuretools[autonormalize]"
```
```{tab} Update Checker
```console
$ python -m pip install "featuretools[updater]"
```
```{tab} Featuretools_SQL
```console
$ python -m pip install "featuretools[sql]" 
```
````
````{tab} Conda
```{tab} All Add-ons
```console
$ conda install -c conda-forge nlp-primitives featuretools-tsfresh-primitives pyspark alteryx-open-src-update-checker
```
```{tab} NLP Primitives
```console
$ conda install -c conda-forge nlp-primitives
```
```{tab} TSFresh Primitives
```console
$ conda install -c conda-forge featuretools-tsfresh-primitives
```
```{tab} Spark
```console
$ conda install -c conda-forge pyspark
```
```{tab} Update Checker
```console
$ conda install -c conda-forge alteryx-open-src-update-checker
```
```{tab} Featuretools_SQL
```console
$ conda install -c conda-forge featuretools_sql
```
````

- **NLP Primitives**: Use Natural Language Processing Primitives in Featuretools
- **TSFresh Primitives**: Use 60+ primitives from [tsfresh](https://tsfresh.readthedocs.io/en/latest/) in Featuretools
- **Spark**: Use Woodwork with Spark DataFrames
- **AutoNormalize**: Automated creation of normalized `EntitySet` from denormalized data
- **Update Checker**: Receive automatic notifications of new Featuretools releases
- **Featuretools_SQL**: Automated `EntitySet` creation from relational data stored in a SQL database

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

To install Featuretools from source, clone the repository from [Github](https://github.com/alteryx/featuretools), and install the dependencies.

```{hint}
Be sure to install [Scala and Spark](#scala-and-spark) if you want to run all unit tests
```

```bash
git clone https://github.com/alteryx/featuretools.git
cd featuretools
python -m pip install .
```

## Scala and Spark

````{tab} macOS (Intel)
:new-set:
```console
$ brew tap AdoptOpenJDK/openjdk
$ brew install --cask adoptopenjdk11
$ brew install scala apache-spark
$ echo 'export JAVA_HOME=$(/usr/libexec/java_home)' >> ~/.zshrc
$ echo 'export PATH="/usr/local/opt/openjdk@11/bin:$PATH"' >> ~/.zshrc
```
````

````{tab} macOS (M1)
```console
$ brew install openjdk@11 scala apache-spark graphviz
$ echo 'export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"' >> ~/.zshrc
$ echo 'export CPPFLAGS="-I/opt/homebrew/opt/openjdk@11/include:$CPPFLAGS"' >> ~/.zprofile
$ sudo ln -sfn /opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-11.jdk
```
````

````{tab} Ubuntu
```console
$ sudo apt install openjdk-11-jre openjdk-11-jdk scala -y
$ echo "export SPARK_HOME=/opt/spark" >> ~/.profile
$ echo "export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin" >> ~/.profile
$ echo "export PYSPARK_PYTHON=/usr/bin/python3" >> ~/.profile
```
````

````{tab} Amazon Linux
```console
$ sudo amazon-linux-extras install java-openjdk11 scala -y
$ amazon-linux-extras enable java-openjdk11
```
````

## Docker

It is also possible to run Featuretools inside a Docker container.
You can do so by installing it as a package inside a container (following the normal install guide) or
creating a new image with Featuretools pre-installed, using the following commands in your `Dockerfile`:

```dockerfile
FROM --platform=linux/x86_64 python:3.8-slim-buster
RUN apt update && apt -y update
RUN apt install -y build-essential
RUN pip3 install --upgrade --quiet pip
RUN pip3 install featuretools
```

# Development

To make contributions to the codebase, please follow the guidelines [here](https://github.com/alteryx/featuretools/blob/main/contributing.md).
