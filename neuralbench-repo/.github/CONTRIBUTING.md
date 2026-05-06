# Contributing to _neuralbench_
We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Submitting a fix
- Adding a new task, model, or dataset
- Proposing enhancements
- Improving documentation

## Our Development Process
_neuralbench_ is actively developed and maintained. Bug tracking and development planning are public. We welcome contributions from researchers and developers in the NeuroAI community.


## Pull Requests
We welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. Install the package in editable mode with the `[dev]` extra (this brings in `pytest`, `ruff`, `mypy`, `pre-commit`, and the type stubs that `mypy neuralbench` requires): `pip install -e ".[dev]"`
3. Install precommit hooks: `pre-commit install`
4. If you've added code that should be tested, add tests.
5. If you've changed APIs, update the documentation.
6. Ensure the test suite passes: `pytest neuralbench`
7. Make sure your code lints: `ruff check neuralbench`, `ruff format --check neuralbench`, `mypy neuralbench`
8. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting with a 90 line length.

## License
By contributing to _neuralbench_, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
