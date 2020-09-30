# Contributing to Featuretools

:+1::tada: First off, thank you for taking the time to contribute! :tada::+1:

Whether you are a novice or experienced software developer, all contributions and suggestions are welcome!

There are many ways to contribute to Featuretools, with the most common ones being contribution of code or documentation to the project.

**To contribute, you can:**
1. Help users on our [Slack channel](https://join.slack.com/t/featuretools/shared_invite/enQtNTEwODEzOTEwMjg4LTQ1MjZlOWFmZDk2YzAwMjEzNTkwZTZkN2NmOGFjOGI4YzE5OGMyMGM5NGIxNTE4NjkzYWI3OWEwZjkyZGExYmQ). Answer questions under the featuretools tag on [Stack Overflow](https://stackoverflow.com/questions/tagged/featuretools)

2. Submit a pull request for one of [Good First Issues](https://github.com/FeatureLabs/featuretools/issues?q=is%3Aopen+is%3Aissue+label%3A%22Good+First+Issue%22)

3. Make changes to the codebase, see [Contributing to the codebase](#Contributing-to-the-Codebase).

4. Improve our documentation, which can be found under the [docs](docs/) directory or at https://docs.featuretools.com

5. [Report issues](#Report-issues) you're facing, and give a "thumbs up" on issues that others reported and that are relevant to you. Issues should be used for bugs, and feature requests only.

6. Spread the word: reference Featuretools from your blog and articles, link to it from your website, or simply star it in GitHub to say "I use it".
    * If you would like to be featured on [ecosystem page](https://docs.featuretools.com/en/stable/ecosystem.html), you can submit a [pull request](https://github.com/FeatureLabs/featuretools).

## Contributing to the Codebase

Before starting major work, you should touch base with the maintainers of Featuretools by filing an issue on GitHub or posting a message in the [#development channel on Slack](https://featuretools.slack.com/join/shared_invite/enQtNTEwODEzOTEwMjg4LTZiZDdkYjZhZTVkMmVmZDIxNWZiNTVjNDQxYmZkMzI5NGRlOTg5YjcwYmJiNWE2YjIzZmFkMjc1NDZkNjBhZTQ). This will increase the likelihood your pull request will eventually get merged in.

#### 1. Clone repo
* The code is hosted on GitHub, so you will need to use Git to clone the project and make changes to the codebase. Once you have obtained a copy of the code, you should create a development environment that is separate from your existing Python environment so that you can make and test changes without compromising your own work environment.
* You can run the following steps to clone the code, create a separate virtual environment, and install featuretools in editable mode. 
  ```bash
  git clone https://github.com/FeatureLabs/featuretools.git
  cd featuretools
  virtualenv venv
  source venv/bin/activate
  python -m pip install -e .
  python -m pip install -r dev-requirements.txt
  ```
#### 2. Implement your Pull Request

* Implement your pull request. If needed, add new tests or update the documentation.
* Before submitting to GitHub, verify the tests run and the code lints properly
  ```bash
  # runs test
  make test

  # runs linting
  make lint

  # will fix some common linting issues automatically
  make lint-fix
  ```
* If you made changes to the documentation, build the documentation locally.
  ```bash
  # go to docs and build
  cd docs
  make html

  # view docs locally
  open build/html/index.html
  ```

#### 3. Submit your Pull Request

* Once your changes are ready to be submitted, make sure to push your changes to GitHub before creating a pull request. Create a pull request, and our continuous integration will run automatically.
* Update the "Future Release" section of the release notes (`docs/source/release_notes.rst`) to include your pull request and add your github username to the list of contributors.  Add a description of your PR to the subsection that most closely matches your contribution:
    * Enhancements: new features or additions to Featuretools.
    * Fixes: things like bugfixes or adding more descriptive error messages.
    * Changes: modifications to an existing part of Featuretools.
    * Documentation Changes
    * Testing Changes

   Documentation or testing changes rarely warrant an individual release notes entry; the PR number can be added to their respective "Miscellaneous changes" entries.
* We will review your changes, and you will most likely be asked to make additional changes before it is finally ready to merge. However, once it's reviewed by a maintainer of Featuretools, passes continuous integration, we will merge it, and you will have successfully contributed to Featuretools!

## Report issues
When reporting issues please include as much detail as possible about your operating system, featuretools version and python version. Whenever possible, please also include a brief, self-contained code example that demonstrates the problem.
