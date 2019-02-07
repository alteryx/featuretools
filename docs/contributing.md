# Contributing to Featuretools

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

Whether you are a novice or experienced software developer, all contributions and suggestions are welcome!

There are many ways to contribute to Featuretools, with the most common ones being contribution of code or documentation to the project. Improving the documentation is no less important than improving the library itself. If you find a typo in the documentation, or have made improvements, do not hesitate to submit a GitHub pull request. Documentation can be found under the [docs](docs/) directory. But there are many other ways to help. One way is to report issues you're facing, and give a "thumbs up" on issues that others reported and that are relevant to you. It also helps us if you spread the word: reference the project from your blog and articles, link to it from your website, or simply star it in GitHub to say "I use it".

## Getting Started

* Before starting major work, you should touch base with the maintainers of Featuretools by filing an issue on GitHub or posting a message in the [#development channel on Slack](https://featuretools.slack.com/join/shared_invite/enQtNTEwODEzOTEwMjg4LTZiZDdkYjZhZTVkMmVmZDIxNWZiNTVjNDQxYmZkMzI5NGRlOTg5YjcwYmJiNWE2YjIzZmFkMjc1NDZkNjBhZTQ). This will increase the likelihood your pull request will eventually get merged in.

#### 1. Clone repo
* The code is hosted on GitHub, so you will need to use Git to clone the project and make changes to the codebase. Once you have obtained a copy of the code, you should create a development environment that is separate from your existing Python environment so that you can make and test changes without compromising your own work environment.

#### 2. Implement your PR

* Implement your pull request. If needed, add new tests or update the documentation.

* Before submitting to GitHub, ensure run the tests and linter
  ```
  # runs test
  make tests

  # runs linting
  make lint

  # will fix some common linting issues automatically
  make lint-fix
  ```
* If you made changes to the documentation, build the documentation locally.

#### 3. Submit your PR

* Once your changes are ready to be submitted, make sure to push your changes to GitHub before creating a pull request. Once you create a pull reuest, our continuous integration will run automatically. We will review your changes, and you will most likely be asked to make additional changes before it is finally ready to merge. However, once it's reviewed by a maintainer of Featuretools, passes CI, we will merge it, and you will have successfully contributed to the codebase!

## Reporting issues
* When reporting issues please include as much detail as possible about your operating system, featuretools version and python version. Whenever possible, please also include a brief, self-contained code example that demonstrates the problem.
