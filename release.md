# Release Process
## Test conda version before releasing on PyPI
Conda releases of featuretools rely on PyPI's hosted featuretools packages. Once a version is uploaded to PyPI we cannot update it, so it is important that the version we upload to PyPI will work for conda.  We can test if a featuretools release will run on conda by uploading a test release to PyPI's test server and building a conda version of featuretools using the test release.

#### Upload featuretools release to PyPI's test server
We need to upload a featuretools package to test with the conda recipe
1. Make a new development release branch on featuretools (in this example we'll be testing the 0.13.3 release)
    ```bash
    git checkout -b v0.13.3.dev
    ```
2. Update version number in `setup.py`, `featuretools/version.py`, `featuretools/tests/test_version.py` to v0.13.3.dev0 and push branch to repo
3. Publish a new release of featuretools on Github.
    1. Go to the [releases page](https://github.com/FeatureLabs/featuretools/releases/) on Github
    2. Click "Draft a new release"
    3. For the target, choose the new branch (v0.13.3.dev)
    4. For the tag, use the new version number (v0.13.3.dev0)
    5. For the release title, use the new version number (v0.13.3.dev0)
    6. For the release description, write "Development release for testing purposes"
    7. Check the "This is a pre-release" box
    8. Publish the release
4. The new release will be uploaded to TestPyPI automatically

#### Set up fork of our conda-forge repo
Branches on the conda-forge featuretools repo are automatically built and the package uploaded to conda-forge, so to test a release without uploading to conda-forge we need to fork the repo and develop on the fork.
1. Fork conda-forge/featuretools-feedstock: visit https://github.com/conda-forge/featuretools-feedstock and click fork
2. Clone forked repo locally, selecting your own github account as the base for the forked repo. 
3. Add conda-forge repo as the 'upstream' repository
    ```bash
    git remote add upstream https://github.com/conda-forge/featuretools-feedstock.git
    ```
4. If you made the fork previously and its master branch is missing commits, update it with any changes from upstream
    ```bash
    git fetch upstream
    git checkout master
    git merge upstream/master
    git push origin master
    ```
5. Make a branch with the version you want to release
    ```bash
    git checkout -b v0.13.0.dev0
    ```

#### Update conda recipe to use TestPyPI release of featuretools
Fields to update in `recipe/meta.yaml` of feedstock repo:
* Always update:
    * Set the new release number (e.g. v0.13.3.dev0)
        ```
        {% set version = "0.13.3.dev0" %}
        ```
    * Source fields
        * url - visit https://test.pypi.org/project/featuretools/, find correct release, go to download files page, and copy link location of the tar.gz file
        * sha256 - from the download files page, click the view hashes button for the tar.gz file and copy the sha256 digest
        ```
        source:
          url: https://test-files.pythonhosted.org/packages/3a/d7/8600d5ffa72d6890344df0811e8431465bcf15f2a9eade143ee34f67c1c4/featuretools-0.13.3.dev0.tar.gz
          sha256: e9c3b4fe4aa40a4606dc3d72c65b4dc3ed03014229d4f225d5e8ce0a727d4462
       ```
* Update if dependencies have changed:
    * setup-requirements.txt dependencies are host requirements
        ```
        requirements:
          host:
            - pip
            - python
        ```
    * requirements.txt dependencies are run requirements
        ```
        requirements:
          run:
            - click
            - cloupickle
        ```
    * test-requirements.txt dependencies are test requirements
        ```
        test:
          requires:
            - fastparquet
            - mock
        ```

#### Test with conda-forge CI
1. Install conda
    1. If using pyenv, `pyenv install miniconda3-latest`
    2. Otherwise follow instructions in [conda docs](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Install conda-smithy (conda-forge tool to update boilerplate in repo)
    ```bash
    conda install -n root -c conda-forge conda-smithy
    ```
3. Run conda-smithy on feedstock
    ```bash
    cd /path/to/feedstock/repo
    conda-smithy rerender --commit auto
    ```
4. Push updated branch to the forked feedstock repo
3. Make a PR on conda-forge/featuretools-feedstock from the forked repo and let CI tests run - add "[DO NOT MERGE]" to the PR name to indicate this is PR should not be merged in
4. After the tests pass, close the PR without merging

## Create featuretools release on github
#### Create release branch
1. Branch off of featuretools main and name the branch the release version number (e.g. v0.13.3)

#### Bump version number
2. Bump version number in `setup.py`, `featuretools/version.py`, and `featuretools/tests/test_version.py`.

#### Update release notes
1. Replace "Future Release" in `docs/source/release_notes.rst` with the current date
    ```
    **v0.13.3 Feb 28, 2020**
    ```
2. Remove any unused sections for this release (e.g. Fixes, Testing Changes)
3. Add yourself to the list of contributors to this release and put the contributors in alphabetical order
4. The release PR does not need to be mentioned in the list of changes
5. Add a commented out "Future Release" section with all of the release notes sections above the current section
    ```
    .. **Future Release**
        * Enhancements
        * Fixes
        * Changes
        * Documentation Changes
        * Testing Changes

    .. Thanks to the following people for contributing to this release:
    ```


#### Create Release PR
A [release pr](https://github.com/FeatureLabs/featuretools/pull/856) should have the version number as the title and the release notes for that release as the PR body text. The contributors list is not necessary. The special sphinx docs syntax (:pr:\`547\`) needs to be changed to github link syntax (#547).

#### Create Github Release
After the release pull request has been merged into the main branch, it is time draft the github release. [Example release](https://github.com/FeatureLabs/featuretools/releases/tag/v0.13.3)
* The target should be the main branch
* The tag should be the version number with a v prefix (e.g. v0.13.3)
* Release title is the same as the tag
* Release description should be the full release notes for the release, including the line thanking contributors.  Contributors should also have their links changed from the docs syntax (:user:\`rwedge\`) to github syntax (@rwedge)
* This is not a pre-release
* Publishing the release will automatically upload the package to PyPI

## Release on conda-forge
1. A bot should automatically create a new PR in conda-forge/featuretools-feedstock - note, the PR may take up to a few hours to be created
2. Update requirements changes in `recipe/meta.yaml` (bot should have handled version and source links on its own)
3. After tests pass, a maintainer will merge the PR in
