# Contributing to NeuralSet

Thanks for your interest in contributing! Before diving in, see the
[design philosophy](../philosophy.md) for the principles behind NeuralSet's
API (pydantic everywhere, lazy evaluation, the event abstraction).

## Ways to Contribute

- **Bug reports** — open an issue with a minimal reproducer.
- **Bug fixes** — include a regression test.
- **New datasets or features** — please open an issue first to discuss.
- **Documentation** — typo fixes, examples, tutorials are always welcome.

----

## Fork & Pull-Request Workflow

1. **Fork** the repository on GitHub and clone your fork locally.
2. **Create a branch** from `main` (or the current dev branch):
   ```bash
   git checkout -b my-feature
   ```
3. **Make your changes** — follow the code style below.
4. **Run the checks** before committing:
   ```bash
   cd neuralset-repo
   ruff check neuralset/
   mypy neuralset/
   pytest neuralset/ -x -q
   ```
5. **Commit** — pre-commit hooks will auto-format with black/isort and lint
   with ruff. If a hook fails, fix the issue and commit again.
6. **Push** your branch and open a Pull Request against the upstream repo.
7. A reviewer will check code quality, test coverage, and API consistency.

```{tip}
Keep PRs focused — one logical change per PR.
PRs are squash-merged; intermediate commit messages don't matter.
```

----

## Reporting & Fixing Bugs

- **Search first** — check existing GitHub issues to avoid duplicates.
- **File a bug** with: a clear title, minimal reproduction steps, expected vs.
  actual behaviour, and your environment (Python version, OS, package version).
- **Fix a bug** — if you want to fix it yourself:
  1. Write a failing test that reproduces the bug.
  2. Implement the fix.
  3. Verify the test now passes.
  4. Open a PR referencing the issue.

----

## Code Quality & Pre-Commit Hooks

Pre-commit hooks run automatically on `git commit`:

| Hook | What it does |
|------|-------------|
| **black** | Auto-formats Python code (line length, quotes, etc.) |
| **isort** | Sorts and groups import statements |
| **ruff** | Fast linter — catches style issues, unused imports, etc. |

You can also run them manually:

```bash
cd neuralset-repo
ruff check neuralset/           # lint
mypy neuralset/                 # type check
pytest neuralset/ -x -q         # tests
```

To run hooks on all files (useful before a PR):

```bash
pre-commit run --all-files
```

All public APIs must have type annotations. Use `tp.ClassVar` for class
constants, `tp.Literal` for string enums, pydantic `BaseModel` fields
for configuration.

----

## Use of Generative AI

Contributors are welcome to use AI tools (Copilot, ChatGPT, etc.) as aids,
subject to these rules:

- **You are responsible** for all code you submit. Review and understand every
  line before opening a PR.
- **Do not submit AI output you don't understand.** If you can't explain a
  change, don't submit it.
- **No AI bots** interacting with the project. Automated agents must not open
  issues, submit PRs, post comments, or review code on behalf of contributors.

----

## Legal

### Contributor License Agreement

To accept your pull request we need a signed CLA. You only need to do this
once for all Meta open-source projects:
<https://code.facebook.com/cla>

### Security

Meta has a bounty program for safe disclosure of security bugs. Report
security issues through that process — do not file a public issue.

### Code of Conduct

See [Code of Conduct](https://github.com/facebookresearch/neuroai/blob/main/.github/CODE_OF_CONDUCT.md).

### License

By contributing, you agree that your contributions will be licensed under
the LICENSE file in the root directory of this source tree.

```{raw} html
<div class="page-nav">
  <a href="../glossary.html" class="page-nav-btn page-nav-btn--outline">&larr; Glossary</a>
</div>
```
