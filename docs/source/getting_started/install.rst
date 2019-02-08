Install
*******

Pip and Conda
---------------

Featuretools is available for Python 2.7, 3.5, and 3.6. The recommended way to install Featuretools is using ``pip`` or ``conda``::

    python -m pip install featuretools

or from the Conda-forge channel on `anaconda.org <https://anaconda.org/conda-forge/featuretools>`_::

    conda install -c conda-forge featuretools

.. note ::

    In order to use :meth:`EntitySet.plot <featuretools.entityset.EntitySet.plot>` you will need to install the graphviz library.

    Conda users:
        conda install python-graphviz

    Ubuntu:
        sudo apt-get install graphviz

        pip install graphviz

    Mac OS:
        brew install graphviz

        pip install graphviz

    Windows:
        conda install python-graphviz


Install from Source
-------------------

To install featuretools from source, clone the repository from `github
<https://github.com/featuretools/featuretools>`_::

    git clone https://github.com/featuretools/featuretools.git
    cd featuretools
    python setup.py install

or use ``pip`` locally if you want to install all dependencies as well::

    pip install .

You can view the list of all dependencies within the ``extras_require`` field
of ``setup.py``.



Development
-----------
Before making contributing to the codebase, please follow the guidelines `here <https://github.com/Featuretools/featuretools/blob/master/docs/contributing.md>`_

Virtualenv
~~~~~~~~~~
We recommend developing in a `virtualenv <https://virtualenvwrapper.readthedocs.io/en/latest/>`_::

    mkvirtualenv featuretools

Install development requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run::

    make installdeps

Test
~~~~
.. note::

    In order to the run the featuretools tests you will need to have graphviz installed as described above.

Run featuretools tests::

    make test

Before committing make sure to run linting in order to pass CI::

    make lint

Some linting errors can be automatically fixed by running the command below::

    make lint-fix


Build Documentation
~~~~~~~~~~~~~~~~~~~
Build the docs with the commands below::

    cd docs/

    # small changes
    make html

    # rebuild from scatch
    make clean html

.. note ::

    The Featuretools library must be import-able to build the docs.
