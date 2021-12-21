# Release Process

## Create featuretools release on github
#### Create release branch
1. Branch off of featuretools main. For the branch name, please use "release_vX.Y.Z" as the naming scheme (e.g. "release_v0.13.3"). Doing so will bypass our release notes checkin test which requires all other PRs to add a release note entry.

#### Bump version number
2. Bump version number in `setup.py`, `featuretools/version.py`, and `featuretools/tests/test_version.py`.

#### Update release notes
1. Replace "Future Release" in `docs/source/release_notes.rst` with the current date, making sure the heading is fully underlined with `=` characters.
    ```
    v0.13.3 Feb 28, 2020
    ====================
    ```
2. Remove any unused sections for this release (e.g. Fixes, Testing Changes)
3. Add yourself to the list of contributors to this release and put the contributors in alphabetical order
4. The release PR does not need to be mentioned in the list of changes
5. Add a commented out "Future Release" section with all of the release notes sections above the current section
    ```
    .. Future Release
      ==============
        * Enhancements
        * Fixes
        * Changes
        * Documentation Changes
        * Testing Changes

    .. Thanks to the following people for contributing to this release:
    ```


#### Create Release PR
A [release pr](https://github.com/alteryx/featuretools/pull/856) should have the version number as the title and the release notes for that release as the PR body text. The contributors list is not necessary. The special sphinx docs syntax (:pr:\`547\`) needs to be changed to github link syntax (#547).

#### Create Github Release
After the release pull request has been merged into the main branch, it is time draft the github release. [Example release](https://github.com/alteryx/featuretools/releases/tag/v0.13.3)
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
