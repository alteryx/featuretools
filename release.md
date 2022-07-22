# Release Process

## 0. Pre-Release Checklist

Before starting the release process, verify the following:

- All work required for this release has been completed and the team is ready to release.
- [All Github Actions Tests are green on main](https://github.com/alteryx/featuretools/actions?query=branch%3Amain).
- EvalML Tests are green with Featuretools main
  - [![Unit Tests - EvalML with Featuretools main branch](https://github.com/alteryx/evalml/actions/workflows/unit_tests_with_featuretools_main_branch.yaml/badge.svg?branch=main)](https://github.com/alteryx/evalml/actions/workflows/unit_tests_with_featuretools_main_branch.yaml)
- The [ReadtheDocs build](https://readthedocs.com/projects/feature-labs-inc-featuretools/) for "latest" is marked as passed. To avoid mysterious errors, best practice is to empty your browser cache when reading new versions of the docs!
- The [public documentation for the "latest" branch](https://featuretools.alteryx.com/en/latest/) looks correct, and the [release notes](https://featuretools.alteryx.com/en/latest/release_notes.html) includes the last change which was made on `main`.
- Get agreement on the version number to use for the release.

#### Version Numbering

Featuretools uses [semantic versioning](https://semver.org/). Every release has a major, minor and patch version number, and are displayed like so: `<majorVersion>.<minorVersion>.<patchVersion>`.

In certain instances, it may be necessary to create a backport release. This is when commits from a newer version of a library are ported to an older version of the software and then released. This occurs when anything but the latest commit on main is used as the target for release, but can go so far as to add a further patch release, such as 0.11.2, to be released after a 0.12.0 version had already been released. If a backport release is being performed, please see the [Backport Release Guide](docs/backport_release.md) for instructions on how to proceed, as some steps from this guide should be performed differently.

If you'd like to create a development release, which won't be deployed to pypi and conda and marked as a generally-available production release, please add a "dev" prefix to the patch version, i.e. `X.X.devX`. Note this claims the patch number--if the previous release was `0.12.0`, a subsequent dev release would be `0.12.dev1`, and the following release would be `0.12.2`, _not_ `0.12.1`. Development releases deploy to [test.pypi.org](https://test.pypi.org/project/featuretools/) instead of to [pypi.org](https://pypi.org/project/featuretools).

## 1. Create Featuretools release on Github

#### Create Release Branch

1. Branch off of featuretools main. For the branch name, please use "release_vX.Y.Z" as the naming scheme (e.g. "release_v0.13.3"). Doing so will bypass our release notes checkin test which requires all other PRs to add a release note entry.

#### Bump Version Number

1. Bump `__version__` in `featuretools/version.py`, and `featuretools/tests/test_version.py`.

#### Update Release Notes

1. Replace "Future Release" in `docs/source/release_notes.rst` with the current date

   ```
   v0.13.3 Sep 28, 2020
   ====================
   ```

2. Remove any unused Release Notes sections for this release (e.g. Fixes, Testing Changes)
3. Add yourself to the list of contributors to this release and **put the contributors in alphabetical order**
4. The release PR does not need to be mentioned in the list of changes
5. Add a commented out "Future Release" section with all of the Release Notes sections above the current section

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

Checklist before merging:

- All tests are currently green on checkin and on `main`.
- The ReadtheDocs build for the release PR branch has passed, and the resulting docs contain the expected release notes.
- PR has been reviewed and approved.
- Confirm with the team that `main` will be frozen until step 2 (Github Release) is complete.

After merging, verify again that ReadtheDocs "latest" is correct.

## 2. Create Github Release

After the release pull request has been merged into the `main` branch, it is time draft the github release. [Example release](https://github.com/alteryx/featuretools/releases/tag/v0.13.3)

- The target should be the `main` branch
- The tag should be the version number with a v prefix (e.g. v0.13.3)
- Release title is the same as the tag
- Release description should be the full Release Notes updates for the release, including the line thanking contributors. Contributors should also have their links changed from the docs syntax (:user:\`gsheni\`) to github syntax (@gsheni)
- This is not a pre-release
- Publishing the release will automatically upload the package to PyPI

## Release on conda-forge

In order to release on conda-forge, you can either wait for a bot to create a PR, or manually kickoff the creation with GitHub Actions

### Option 1: Manually create the new PR with GitHub Actions
1. Go to this GitHub Action: https://github.com/alteryx/featuretools/actions/workflows/create_feedstock_pr.yaml
2. Input the released version with the v prefix (e.g. v0.13.3)
3. Kickoff the GitHub action, and monitor the Job Summary. At the completion of the job, you should see summary output, with the URL (which you will need to visit and create a PR with).
  a. You can also just go to: https://github.com/machineAYX/featuretools-feedstock/pull/new/conda-autocreate-v0.13.3 (change the last part to the released version)

### Option 2: Waiting for bot to create new PR

1. A bot should automatically create a new PR in [conda-forge/featuretools-feedstock](https://github.com/conda-forge/featuretools-feedstock/pulls) - note, the PR may take up to a few hours to be created
2. Update requirements changes in `recipe/meta.yaml` (bot should have handled version and source links on its own)
3. After tests pass, a maintainer will merge the PR in
