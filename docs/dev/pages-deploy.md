# GitHub Pages Deployment

> **Repo policy: no CI/CD workflow files.** Automated `gh-pages` workflows,
> `.github/workflows/*.yml`, `CODEOWNERS`, and CI configs are **intentionally
> forbidden** in `ammar49-cyber/sneppx-algo`. All verification is local.

This page documents how to **manually** publish the static site to GitHub
Pages using `mkdocs gh-deploy`.

## Prerequisites

```powershell
# In the hermes venv (or any venv on the maintainer machine)
python -m pip install "mkdocs-material>=9.5" mkdocs-git-revision-date-localized-plugin
```

The docs build uses **Material for MkDocs** + the `git-revision-date-localized`
plugin (for "last updated" timestamps). No other plugins are required.

## Build locally

```powershell
# From repo root
mkdocs build --strict
# -> site/  (open site/index.html)
```

`--strict` fails the build on broken internal links and missing references.
If it fails:

- Check that every `nav:` entry in `mkdocs.yml` maps to an existing file.
- Check that the Doxygen output directory (`docs/api/doxygen/html/`) either
  exists (run `doxygen Doxyfile`) or that links to it are plain HTML hrefs
  (not Markdown-relative links — those are not link-checked by `--strict`).

## Generate the Doxygen C/C++ reference first

The Doxygen frame (`docs/api/index.md`) references generated HTML. Build it
before `mkdocs build`:

```powershell
doxygen Doxyfile          # writes docs/api/doxygen/html/index.html
mkdocs build --strict      # now the iframe resolves
```

## Deploy to GitHub Pages

```powershell
# One-time: configure Pages on the ammar49-cyber/sneppx-alg repo
# Settings -> Pages -> Source: "Deploy from a branch" -> gh-pages (root)

# Then, to publish a release:
mkdocs gh-deploy --force
```

- `gh-deploy` writes the built site to the `gh-pages` branch and (with
  `--force`) replaces any prior deployment.
- Because there is **no CI**, every publish is a deliberate maintainer action.
  Tag the repo (`git tag v1.2.0 && git push --tags`) **before** deploying so
  the changelog and "last updated" metadata reflect the release.
- `docs/.nojekyll` is present, so GitHub Pages will serve directories like
  `assets/` and `doxygen/html/` without Jekyll filtering.

## Versioned docs (mike)

The `mkdocs.yml` configures `extra.version.provider: mike` for versioned
site URLs. To maintain versioned builds manually:

```powershell
pip install mike
mike deploy --push latest            # tip
mike deploy --push 1.1.x            # backport branch
mike set-default --push 1.1.x       # default version
```

This is **optional** — `mkdocs gh-deploy` alone serves a single latest build.
Use `mike` only if you need per-version docs at `/1.1.x/`.

## Intentional design choices

| Decision | Reason |
|----------|--------|
| No `.github/workflows/*.yml` | Repo policy: all verification is local |
| Manual `gh-deploy` | Every publish is an audited maintainer action |
| `docs/.nojekyll` committed | Lets GitHub serve static files + doxygen dirs |
| `mkdocs build --strict` | Catches broken links and nav drift in PRs |
| `git-revision-date-localized` plugin (not mike `alias-type`) | Lightweight; no version branch required for the default single-build flow |
