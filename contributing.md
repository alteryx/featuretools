# Contributing to Featuretools

:+1::tada: First off, thank you for taking the time to contribute! :tada::+1:

Whether you are a novice or experienced software developer, all contributions and suggestions are welcome!

There are many ways to contribute to Featuretools, with the most common ones being contribution of code or documentation to the project.

**To contribute, you can:**
1. Help users on our [Slack channel](https://join.slack.com/t/alteryx-oss/shared_invite/zt-182tyvuxv-NzIn6eiCEf8TBziuKp0bNA). Answer questions under the featuretools tag on [Stack Overflow](https://stackoverflow.com/questions/tagged/featuretools)

2. Submit a pull request for one of [Good First Issues](https://github.com/alteryx/featuretools/issues?q=is%3Aopen+is%3Aissue+label%3A%22Good+First+Issue%22)

3. Make changes to the codebase, see [Contributing to the codebase](#Contributing-to-the-Codebase).

4. Improve our documentation, which can be found under the [docs](docs/) directory or at https://docs.featuretools.com

5. [Report issues](#Report-issues) you're facing, and give a "thumbs up" on issues that others reported and that are relevant to you. Issues should be used for bugs, and feature requests only.

6. Spread the word: reference Featuretools from your blog and articles, link to it from your website, or simply star it in GitHub to say "I use it".
    * If you would like to be featured on [ecosystem page](https://featuretools.alteryx.com/en/stable/resources/ecosystem.html), you can submit a [pull request](https://github.com/alteryx/featuretools).

## Contributing to the Codebase

Before starting major work, you should touch base with the maintainers of Featuretools by filing an issue on GitHub or posting a message in the [#development channel on Slack](https://join.slack.com/t/alteryx-oss/shared_invite/zt-182tyvuxv-NzIn6eiCEf8TBziuKp0bNA). This will increase the likelihood your pull request will eventually get merged in.

#### 1. Fork and clone repo
* The code is hosted on GitHub, so you will need to use Git to fork the project and make changes to the codebase. To start, go to the [Featuretools GitHub page](https://github.com/alteryx/featuretools) and click the `Fork` button.
* After you have created the fork, you will want to clone the fork to your machine and connect your version of the project to the upstream Featuretools repo.
  ```bash
  git clone https://github.com/your-user-name/featuretools.git
  cd featuretools
  git remote add upstream https://github.com/alteryx/featuretools
  ```
* Once you have obtained a copy of the code, you should create a development environment that is separate from your existing Python environment so that you can make and test changes without compromising your own work environment. You can run the following steps to create a separate virtual environment, and install Featuretools in editable mode.
  ```bash
  python -m venv venv
  source venv/bin/activate
  make installdeps
  git checkout -b issue####-branch_name
  ```

* You will need to install GraphViz, and Pandoc to run all unit tests & build docs:

  > Pandoc is only needed to build the documentation locally.

     **macOS (Intel)** (use [Homebrew](https://brew.sh/)):
     ```console
     brew install graphviz pandoc
     ```

     **macOS (M1)** (use [Homebrew](https://brew.sh/)):
     ```console
     brew install graphviz pandoc
     ```

     **Ubuntu**:
     ```console
     sudo apt install graphviz pandoc -y
     ```

#### 2. Implement your Pull Request

* Implement your pull request. If needed, add new tests or update the documentation.
* Before submitting to GitHub, verify the tests run and the code lints properly
  ```bash
  # runs linting
  make lint

  # will fix some common linting issues automatically
  make lint-fix

  # runs test
  make test
  ```
* If you made changes to the documentation, build the documentation locally.
  ```bash
  # go to docs and build
  cd docs
  make html

  # view docs locally
  open build/html/index.html
  ```
* Before you commit, a few lint fixing hooks will run. You can also manually run these.
  ```bash
  # run linting hooks only on changed files
  pre-commit run

  # run linting hooks on all files
  pre-commit run --all-files
  ```

#### 3. Submit your Pull Request

* Once your changes are ready to be submitted, make sure to push your changes to GitHub before creating a pull request.
* If you need to update your code with the latest changes from the main Featuretools repo, you can do that by running the commands below, which will merge the latest changes from the Featuretools `main` branch into your current local branch. You may need to resolve merge conflicts if there are conflicts between your changes and the upstream changes. After the merge, you will need to push the updates to your forked repo after running these commands.
  ```bash
  git fetch upstream
  git merge upstream/main
  ```
* Create a pull request to merge the changes from your forked repo branch into the Featuretools `main` branch. Creating the pull request will automatically run our continuous integration.
* If this is your first contribution, you will need to sign the Contributor License Agreement as directed.
* Update the "Future Release" section of the release notes (`docs/source/release_notes.rst`) to include your pull request and add your github username to the list of contributors.  Add a description of your PR to the subsection that most closely matches your contribution:
    * Enhancements: new features or additions to Featuretools.
    * Fixes: things like bugfixes or adding more descriptive error messages.
    * Changes: modifications to an existing part of Featuretools.
    * Documentation Changes
    * Testing Changes

   Documentation or testing changes rarely warrant an individual release notes entry; the PR number can be added to their respective "Miscellaneous changes" entries.
* We will review your changes, and you will most likely be asked to make additional changes before it is finally ready to merge. However, once it's reviewed by a maintainer of Featuretools, passes continuous integration, we will merge it, and you will have successfully contributed to Featuretools!

## Report issues
When reporting issues please include as much detail as possible about your operating system, Featuretools version and python version. Whenever possible, please also include a brief, self-contained code example that demonstrates the problem.
