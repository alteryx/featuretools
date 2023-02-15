# Release Process

## 0. Pre-Release Checklist

Before starting the release process, verify the following:

- All work required for this release has been completed and the team is ready to release.
- [All Github Actions Tests are green on main](https://github.com/alteryx/featuretools/actions?query=branch%3Amain).
- EvalML Tests are green with Featuretools main
  - [![Unit Tests - EvalML with Featuretools main branch](https://github.com/alteryx/evalml/actions/workflows/unit_tests_with_featuretools_main_branch.yaml/badge.svg?branch=main)](https://github.com/alteryx/evalml/actions/workflows/unit_tests_with_featuretools_main_branch.yaml)
- Looking Glass performance tests runs should not show any significant performance regressions when comparing the last commit on `main` with the previous release of Featuretools. See Step 1 below for instructions on manually launching the performance tests runs.
- The [ReadtheDocs build](https://readthedocs.com/projects/feature-labs-inc-featuretools/) for "latest" is marked as passed. To avoid mysterious errors, best practice is to empty your browser cache when reading new versions of the docs!
- The [public documentation for the "latest" branch](https://featuretools.alteryx.com/en/latest/) looks correct, and the [release notes](https://featuretools.alteryx.com/en/latest/release_notes.html) includes the last change which was made on `main`.
- Get agreement on the version number to use for the release.

#### Version Numbering

Featuretools uses [semantic versioning](https://semver.org/). Every release has a major, minor and patch version number, and are displayed like so: `<majorVersion>.<minorVersion>.<patchVersion>`.

In certain instances, it may be necessary to create a backport release. This is when commits from a newer version of a library are ported to an older version of the software and then released. This occurs when anything but the latest commit on main is used as the target for release, but can go so far as to add a further patch release, such as 0.11.2, to be released after a 0.12.0 version had already been released. If a backport release is being performed, please see the [Backport Release Guide](docs/backport_release.md) for instructions on how to proceed, as some steps from this guide should be performed differently.

If you'd like to create a development release, which won't be deployed to pypi and conda and marked as a generally-available production release, please add a "dev" prefix to the patch version, i.e. `X.X.devX`. Note this claims the patch number--if the previous release was `0.12.0`, a subsequent dev release would be `0.12.dev1`, and the following release would be `0.12.2`, _not_ `0.12.1`. Development releases deploy to [test.pypi.org](https://test.pypi.org/project/featuretools/) instead of to [pypi.org](https://pypi.org/project/featuretools).

## 1. Evaluate Performance Test Results

Before releasing Featuretools, the person performing the release should launch a performance test run and evaluate the results to make sure no significant performance regressions will be introduced by the release. This can be done by launching a Looking Glass performance test run, which will then post results to Slack. 

To manually launch a Looking Glass performance test run, follow these steps:
1. Navigate to the [Looking Glass performance tests](https://github.com/alteryx/featuretools/actions/workflows/looking_glass_performance_tests.yaml) GitHub action
2. Click on the Run workflow dropdown to set up the run
3. Make sure that the "use workflow from" dropdown is set to `main` to use the workflow version in Featuretools `main`
4. Enter the hash of the most recent commit to `main` in the "new commit to evaluate" field. For example: `cee9607`
5. Enter the version tag of the last release of Featuretools in the "previous commit to evaluate" field. For example, if the last release of Featuretools was version 1.20.0, you would enter `v1.20.0` here.
6. Click the "Run workflow" button to launch the jobs

Once the job has been completed, the results summaries will be posted to Slack automatically. Review the results and make sure the performance has not degraded. If any significant performance issues are noted, discuss with the development team before proceeding.

Note: The procedure above can also be used to launch performance tests runs at any time, even outside of the release process. When launching a test run, the commit fields can take any commit hash, GitHub branch or tag as input to specify the new and previous commits to compare.

## 2. Create Featuretools release on Github

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

A [release pr](https://github.com/alteryx/featuretools/pull/856) should have **the version number as the title** and the release notes for that release as the PR body text. The contributors list is not necessary. The special sphinx docs syntax (:pr:\`547\`) needs to be changed to github link syntax (#547).

Checklist before merging:

- The title of the PR is the version number.
- All tests are currently green on checkin and on `main`.
- The ReadtheDocs build for the release PR branch has passed, and the resulting docs contain the expected release notes.
- PR has been reviewed and approved.
- Confirm with the team that `main` will be frozen until step 3 (Github Release) is complete.

After merging, verify again that ReadtheDocs "latest" is correct.

## 3. Create Github Release

After the release pull request has been merged into the `main` branch, it is time draft the github release. [Example release](https://github.com/alteryx/featuretools/releases/tag/v0.13.3)

- The target should be the `main` branch
- The tag should be the version number with a v prefix (e.g. v0.13.3)
- Release title is the same as the tag
- Release description should be the full Release Notes updates for the release, including the line thanking contributors. Contributors should also have their links changed from the docs syntax (:user:\`gsheni\`) to github syntax (@gsheni)
- This is not a pre-release
- Publishing the release will automatically upload the package to PyPI

## 4. Release on conda-forge

In order to release on conda-forge, you can either wait for a bot to create a pull request, or use a GitHub Actions workflow

### Option a: Use a GitHub Action workflow

1. After the package has been uploaded on PyPI, the **Create Feedstock Pull Request** workflow should automatically kickoff a job. 
    * If it does not, go [here](https://github.com/alteryx/featuretools/actions/workflows/create_feedstock_pr.yaml)
    * Click **Run workflow** and input the letter `v` followed by the release version (e.g. `v0.13.3`)
    * Kickoff the GitHub Action, and monitor the Job Summary.
2. Once the job has been completed, you will see summary output, with a URL. 
    * Visit that URL and create a pull request.
    * Alternatively, create the pull request by clicking the branch name (e.g. - `v0.13.3`): 
      - https://github.com/alteryx/featuretools-feedstock/branches
3. Verify that the PR has the following: 
    * The `build['number']` is 0 (in __recipe/meta.yml__).
    * The `requirements['run']` (in __recipe/meta.yml__) matches the `[project]['dependencies']` in __featuretools/pyproject.toml__.
    * The `test['requires']` (in __recipe/meta.yml__) matches the `[project.optional-dependencies]['test']` in __featuretools/pyproject.toml__
    > There will be 2 entries for graphviz: `graphviz` and `python-graphviz`. 
    > Make sure `python-graphviz` (in __recipe/meta.yml__) matches `graphviz` in `[project.optional-dependencies]['test']` in __featuretools/pyproject.toml__.
4. Satisfy the conditions in pull request description and **merge it if the CI passes**. 

### Option b: Waiting for bot to create new PR

1. A bot should automatically create a new PR in [conda-forge/featuretools-feedstock](https://github.com/conda-forge/featuretools-feedstock/pulls) - note, the PR may take up to a few hours to be created
2. Update requirements changes in `recipe/meta.yaml` (bot should have handled version and source links on its own)
3. After tests pass, a maintainer will merge the PR in

# Miscellaneous
## Add new maintainers to featuretools-feedstock

Per the instructions [here](https://conda-forge.org/docs/maintainer/updating_pkgs.html#updating-the-maintainer-list):
1. Ask an existing maintainer to create an issue on the [repo](https://github.com/conda-forge/featuretools-feedstock).
  a. Select *Bot commands* and put the following title (change `username`):

  ```text
  @conda-forge-admin, please add user @username
  ```

2. A PR will be auto-created on the repo, and will need to be merged by an existing maintainer.
3. The new user will need to **check their email for an invite link to click**, which should be https://github.com/conda-forge
