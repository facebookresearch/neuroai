# Contributing

Thanks for your interest in contributing! This is a multi-package
monorepo — see the [documentation](https://facebookresearch.github.io/neuroai/)
for install instructions, tutorials, and API guides.

The repo contains several packages, each in its own directory:

| Directory | Package | Description |
|---|---|---|
| `neuralset-repo/` | **neuralset** | Core pipeline: events, extractors, segmenters, dataloaders |
| `neuralfetch-repo/` | **neuralfetch** | Curated study catalog — download & access public brain datasets |
| `neuraltrain-repo/` | **neuraltrain** | Deep learning tools: models, losses, metrics, optimizers |

## Ways to contribute

- **Bug reports** — open an issue with a minimal reproducer.
- **Bug fixes** — include a regression test.
- **New datasets** — add a Study subclass in `neuralfetch-repo/`; open an issue first to discuss.
- **New features** — please open an issue first to discuss scope and API.
- **Documentation** — typo fixes, examples, tutorials are always welcome.

## Setup

```bash
git clone https://github.com/facebookresearch/neuroai.git
cd neuroai
pip install -e 'neuralset-repo/.[dev,all]'
pip install -e 'neuralfetch-repo/.[dev]'
pip install -e 'neuraltrain-repo/.[dev]'
```

## Pull requests

- Keep PRs focused — one logical change per PR.
- Add or update tests for any code change.
- Run checks in the relevant `*-repo/` directory before submitting:
  ```bash
  cd neuralset-repo        # or neuralfetch-repo, neuraltrain-repo
  ruff check .
  mypy .
  pytest -x -q
  ```
- PRs are squash-merged; intermediate commit messages don't matter.

## Contributor License Agreement

To accept your pull request we need a signed CLA. You only need to do
this once for all Meta open-source projects:
<https://code.facebook.com/cla>

## Use of generative AI

Contributors are welcome to use AI tools (Copilot, ChatGPT, etc.) as
aids, subject to these rules:

- **You are responsible** for all code you submit. Review and understand
  every line before opening a PR.
- **Do not submit AI output you don't understand.** If you can't explain
  a change, don't submit it.
- **No AI bots interacting with the project.** Automated agents must not
  open issues, submit PRs, post comments, or review code on behalf of
  contributors.

## Security

Meta has a [bounty program](https://www.facebook.com/whitehat/) for
safe disclosure of security bugs. Report security issues through that
process — do not file a public issue.

## Code of Conduct

See [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
