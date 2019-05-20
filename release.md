# Release Process
## Test conda-forge release
Conda releases of featuretools rely on PyPI's hosted featuretools packages. Since we cannot reupload featuretools to PyPI using the same version number, it is important ensure the version we upload works with conda so that we don't have to immediately release a new version of featuretools or skip supporting conda for one release cycle.  We can test if a featuretools release will run on conda by uploading a release candidate version of featuretools to PyPI's test server and building a conda version of featuretools based on the test servers' featuretools package.

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
2. Update version number in `setup.py`, `featuretools/version.py`, `featuretools/tests/test_version.py` to v0.7.0rc1 and push branch to repo
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
Fields to update in `recipe/meta.yaml` of feedstock repo:
* Set the new release number (e.g. v0.7.0rc1)
    ```
    {% set version = "0.7.0rc1" %}
    ```
* Source fields
    * url - visit https://test.pypi.org/project/featuretools/, find correct release, go to download files page, and copy link location of the tar.gz file
    * sha256 - click on SHA256 link next to tar.gz to copy it
    ```
    source:
      url: https://test-files.pythonhosted.org/packages/e9/79/4fc79465159f6700c1f7b9cf7403b9e455b40e659f3c979ce282f2eb9bf2/featuretools-0.7.1rc1.tar.gz
      sha256: d9d542172d50b00c6a7154577b8a0ba5f1f500e3f940d83c9e46b4d4a36bf57a
   ```
* setup-requirements.txt dependencies go here
    ```
    requirements:
      host:
        - pip
        - python
    ```
* requirements.txt dependencies go here
    ```
    requirements:
      run:
        - click
        - cloupickle
    ```
* test-requirements.txt dependencies go here
    ```
    test:
      requires:
        - fastparquet
        - mock
    ```

#### Test building the conda package locally
1. Install conda
    1. If using pyenv, `pyenv install miniconda3-latest`
    2. Otherwise follow instructions in [conda docs](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
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
4. After the tests pass, close the PR without merging

## Create featuretools release on github
#### Create release branch
1. Branch off of master and name the branch the release version number (e.g. v0.7.1)
2. Bump verison number in `setup.py`, `featuretools/version.py`, and `featuretools/tests/test_version.py`.

#### Update changelog
1. Grab commit history since last release
    ```bash
    git log --pretty=oneline --abbrev-commit
    ```
    Which displays something like
    ```
    40298ad Automatically generate name for controllable primitives (#481)
    23dc0b1 Check for duplicates Fix (#479)
    cf98910 Update how standard primitives are imported internally (#482)
    c30d842 v0.7.0 (#477)
    ```
2. Copy all the commits since the past release into a new entry in `docs/source/changelog.rst`
    ```
    **v0.7.1** Apr 12, 2019
        * Automatically generate name for controllable primitives (#481)
        * Check for duplicates Fix (#479)
        * Update how standard primitives are imported internally (#482)
    ```
    In order for the documentation to build with proper links, we need to use special syntax to link to github users and pull requests.
    ```rst
        * Automatically generate name for controllable primitives (:pr:`481`)

        Thanks to :user:`kmax12` for contributing to this release
    ```
    Confusing commit descriptions should be updated and commits related to the same feature can be grouped together.  Testing updates and documentation updates can be simplified into one commit each, unless a change deserves its own description.
3. Thank each github user who contributed to the updates in this release
    ```rst
    **v0.7.1** Apr 12, 2019
        * Automatically generate name for controllable primitives (#481)

        Thanks to the following people for contributing to this release: :user:`user1`, :user:`user2`, :user:`user3`
    ```
#### Create Release PR
A [release pr](https://github.com/Featuretools/featuretools/pull/507) should have the version number as the title and the changelog updates as the PR body text. The contributors line is not necessary. The special docs syntax needs to be reverted back to `#547` style for PR links.

#### Create Github Release
After the release pull request has been merged into the master branch, it is time draft the github release. [Example release](https://github.com/Featuretools/featuretools/releases/tag/v0.7.1)
* The target should be the master branch
* The tag should be the version number with a v prefix (e.g. v0.7.1)
* Release title is the same as the tag
* Release description should be the full changelog updates for the release, including the line thanking contributors.

## Release on PyPI
1. Update circleci's python3 image
    ```bash
    docker pull circleci/python:3
    ```
2. Run upload script
    ```bash
    docker run \
        --rm \
        -it \
        -v /absolute/path/to/upload.sh:/home/circleci/upload.sh \
        circleci/python:3 \
        /bin/bash -c "bash /home/circleci/upload.sh tags/release_tag"
    ```

## Deploy docs
From root
```
cd docs
python source/upload.py --root
```

## Release on conda-forge
1. Release featuretools on PyPI
1. Wait for bot to create new PR in conda-forge/featuretools-feedstock
2. Update requirements in `recipe/meta.yaml` (bot should have handled version and source links on its own)
3. After tests pass, merge the PR
