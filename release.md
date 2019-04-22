## Release Process
#### Update repo
1. Bump verison number in `setup.py`, and `__init__.py`
2. Update `changelog.rst`

#### Uploading to pypi
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
Conda releases of featuretools rely on pypi's hosted featuretools packages.  Since we cannot reupload featuretools to pypi using the same version number, it is important ensure the version we upload to pypi will work with conda so that we don't have to immediately release a new version of featuretools or skip supporting conda for one release cycle.  We can test if a featuretools release will run on conda by uploading a release candidate version of featuretools to pypi's test server and building a conda version of featuretools based on the test servers' featuretools package.

#### Set up fork of our conda-forge repo
1. Fork conda-forge/featuretools-feedstock
Visit https://github.com/conda-forge/featuretools-feedstock and click fork
2. Clone forked repo locally
3. Add conda-forge repo as 'upstream' repository
    ```bash
    git remote add upstream https://github.com/conda-forge/featuretools-feedstock.git
    ```
4. If your fork is behind, update it with any changes from upstream
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

#### Upload featuretools release candidate to pypi's test server
1. Make a new release candidate branch on featuretools
    ```bash
    git checkout -b vX.Y.Z.rc
    ```
2. Update version number in setup.py and featuretools/__init__.py to v.X.Y.Z.rc1 and push branch to repo
3. Upload release candidate to test.pypi.org
    ```bash
    docker run \
        --rm \
        -it \
        -v /path/to/upload.sh:/home/circleci/upload.sh \
        circleci/python:3
        /bin/bash -c "bash /home/circleci/upload.sh v.X.Y.Z.rc testpypi"
    ```
#### Update conda recipe to use testpypi release of featuretools
In recipe/meta.yaml of feedstock repo:
1. Change {% set version = "X" } to match new release number
2. Update source url - visit https://test.pypi.org/project/featuretools/, find correct release, go to download files page, and copy link location of the tar.gz file
3. Update source sha256 - click on SHA256 link next to tar.gz to copy it
4. Update various requirements:
    requirements:host in meta.yaml corresponds to setup-requirements.txt
    requirements:run in meta.yaml corresponds to requiremnts.txt
    test:requires in meta.yaml correpsond to test-requirements.txt

#### Test locally
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
1. Install conda-smithy
    ```bash
    conda install -n root -c conda-forge conda-smithy
    ```
2. Run conda-smithy on feedstock
    ```bash
    cd /path/to/feedstock/repo
    conda smithy rerender --commit auto
    ```
3. Make a PR on conda forge from the forked repo and let CI tests run - indicate that this pr should not be merged
4. After the tests pass, close the PR

#### Release on conda-forge
1. Wait for bot to create new PR after featuretools is released on pypi
2. Update requirements in recipe/meta.yaml (bot should have handled version and source links on its own)
3. After tests pass, merge the PR
