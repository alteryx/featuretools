## Release Process
#### Update repo
1. Bump verison number in `setup.py`, and `__init__.py`
2. Update `changelog.rst`

#### Uploading to PyPI
1. Update circleci's python3 image
    ```bash
    docker pull circleci/python:3
    ```
2. Run upload script
    ```bash
    docker run \
        --rm \
        -it \
        -v /path/to/upload.sh:/home/circleci/upload.sh \
        circleci/python:3
        /bin/bash -c "bash /home/circleci/upload.sh tags/release_tag"
    ```

#### Deploy docs
From root
```
cd docs
python source/upload.py --root
```

## Testing conda-forge release
Conda releases of featuretools rely on pypi's hosted featuretools packages.  Since we cannot reupload featuretools to pypi using the same version number, it is important ensure the version we upload works with conda so that we don't have to immediately release a new version of featuretools or skip supporting conda for one release cycle.  We can test if a featuretools release will run on conda by uploading a release candidate version of featuretools to pypi's test server and building a conda version of featuretools based on the test servers' featuretools package.

#### Set up fork of our conda-forge repo
Branches on the conda-forge featuretools repo are automatically built and the package uploaded to conda-forge, so to test a release without uploading to conda-forge we need to fork into a separate repo and develop there. 
1. Fork conda-forge/featuretools-feedstock: visit https://github.com/conda-forge/featuretools-feedstock and click fork
2. Clone forked repo locally
3. Add conda-forge repo as the 'upstream' repository
    ```bash
    git remote add upstream https://github.com/conda-forge/featuretools-feedstock.git
    ```
4. If your made the fork previously and its master branch is missing commits, update it with any changes from upstream
    ```bash
    git fetch upstream
    git checkout master
    git merge upstream/master
    git push origin master
    ```
5. Make a branch with the version you want to release
    ```bash
    git checkout -b new-featuretools-version
    ```

#### Upload featuretools release candidate to PyPI's test server
Before we can update the conda recipe we need an uploaded package for the recipe to use
1. Make a new release candidate branch on featuretools (in this example we'll use version 0.7.0)
    ```bash
    git checkout -b v0.7.0rc
    ```
2. Update version number in setup.py and featuretools/__init__.py to v0.7.0rc1 and push branch to repo
3. Upload release candidate to test.pypi.org
    ```bash
    docker run \
        --rm \
        -it \
        -v /path/to/upload.sh:/home/circleci/upload.sh \
        circleci/python:3
        /bin/bash -c "bash /home/circleci/upload.sh v0.7.0rc testpypi"
    ```
#### Update conda recipe to use testpypi release of featuretools
In recipe/meta.yaml of feedstock repo:
1. Change {% set version = "X" } to match new release number (v0.7.0rc1)
2. Update source url - visit https://test.pypi.org/project/featuretools/, find correct release, go to download files page, and copy link location of the tar.gz file
3. Update source sha256 - click on SHA256 link next to tar.gz to copy it
4. Update various requirements:
    requirements:host in meta.yaml corresponds to setup-requirements.txt
    requirements:run in meta.yaml corresponds to requiremnts.txt
    test:requires in meta.yaml correpsond to test-requirements.txt

#### Test building the conda package locally
1. Install conda
    1. If using pyenv, pyenv install miniconda3-latest
    2. Otherwise use [conda docs](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Install conda-build package
    ```bash
    conda install conda-build
    ```
3. Build featuretools package
    ```bash
    conda-build /path/to/recipe/dir
    ```
#### Test with conda-forge CI
1. Install conda-smithy (conda-forge tool to update boilerplate in repo)
    ```bash
    conda install -n root -c conda-forge conda-smithy
    ```
2. Run conda-smithy on feedstock
    ```bash
    cd /path/to/feedstock/repo
    conda-smithy rerender --commit auto
    ```
3. Make a PR on conda-forge/featuretools-feedstock from the forked repo and let CI tests run - indicate that this pr should not be merged
4. After the tests pass, close the PR

#### Release on conda-forge
1. Release featuretools on PyPI
1. Wait for bot to create new PR in conda-forge/featuretools-feedstock
2. Update requirements in recipe/meta.yaml (bot should have handled version and source links on its own)
3. After tests pass, merge the PR
