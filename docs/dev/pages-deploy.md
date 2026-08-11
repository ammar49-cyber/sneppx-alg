# Documentation Publishing

> **Repo policy: no CI/CD workflow files.** Automated `gh-pages` workflows,
> `.github/workflows/*.yml`, `CODEOWNERS`, and CI configs are **intentionally
> forbidden** in `ammar49-cyber/sneppx-alg`. All verification is local.

The documentation is a static **Material for MkDocs** site. There is a single
**primary** publishing target and one **legacy** target.

## Primary: `sneppxalg.vercel.app/sneppx-alg` (Arix-Site)

The live docs site is served from the `sneppxalg` Vercel project via the
companion `ammar49-cyber/Arix-Site` Next.js repository. MkDocs emits
**depth-relative** asset references and an absolute `site_url` of
`https://sneppxalg.vercel.app/sneppx-alg/`, so the built `site/` is mounted
**unchanged** into `Arix-Site` as static assets:

```text
site/  ──►  Arix-Site/public/sneppx-alg/
```

### Publish a docs release

```powershell
# 1. Build the static site locally (strict = fail on broken links/nav)
mkdocs build --strict          # -> site/

# 2. Mirror the output into the Arix-Site static folder (overwrite)
robocopy site "..\Arix-Site\public\sneppx-alg" /E /NFL /NDL /NJH /NJS /NC /R:1 /W:1

# 3. Commit + push in Arix-Site; Vercel auto-redeploys on main
cd ..\Arix-Site
git add -A
git commit -m "docs: publish SNEPPX-Algo docs from mkdocs build"
git push origin main
```

Vercel rebuilds the Arix-Site Next.js (`output: "export"`) app on every push to
`main`, and `public/sneppx-alg/` is served verbatim at
`/sneppx-alg/`. No `next.config.js` changes are required.

- Point the `ammar49-cyber/sneppx-alg` repo **"Website"** field at
  `https://sneppxalg.vercel.app/sneppx-alg/` (set with
  `gh repo edit ammar49-cyber/sneppx-alg --homepage https://sneppxalg.vercel.app/sneppx-alg/`).
- Tag the repo (`git tag v1.2.1 && git push --tags`) before publishing so the
  changelog and git-revision timestamps reflect the release.

## Legacy: GitHub Pages (`mkdocs gh-deploy`)

GitHub Pages on the `ammar49-cyber/sneppx-alg` `gh-pages` branch is retained as
a backup origin only. Because `mkdocs.yml` -> `site_url` now points at the
**Vercel** primary, `mkdocs gh-deploy` will emit Vercel canonical URLs from a
GitHub-hosted origin — a deliberate mismatch. Do **not** rely on it for SEO;
use the primary target above.

```powershell
mkdocs gh-deploy --force     # legacy / backup only
```

## Prerequisites

```powershell
# In the hermes venv (or any maintainer venv)
python -m pip install "mkdocs-material>=9.5" mkdocs-git-revision-date-localized-plugin
```

The build uses **Material for MkDocs** + the `git-revision-date-localized`
plugin (for "last updated" timestamps). No other plugins are required.

## Build locally

```powershell
mkdocs build --strict          # -> site/  (open site\index.html)
```

`--strict` fails the build on broken internal links and missing references.
If it fails:

- Check every `nav:` entry in `mkdocs.yml` maps to an existing file.
- The Doxygen frame (`docs/api/index.md`) references generated HTML. Build it
  first (`doxygen Doxyfile` writes `docs/api/doxygen/html/index.html`) so the
  iframe resolves; otherwise links to it are plain HTML hrefs (not
  Markdown-relative) and are not link-checked by `--strict`.

## Versioned docs (mike) — optional

`mkdocs.yml` configures `extra.version.provider: mike` for versioned URLs.
Maintain versioned builds manually only if you need per-version docs:

```powershell
pip install mike
mike deploy --push latest            # tip
mike deploy --push 1.1.x            # backport branch
mike set-default --push 1.1.x       # default version
```

This is **optional** — a single latest build (above) is the default flow.

## Intentional design choices

| Decision | Reason |
|----------|--------|
| No `.github/workflows/*.yml` | Repo policy: all verification is local |
| Primary = Vercel/Arix-Site (`public/sneppx-alg/`) | Serves the real `sneppxalg` domain; depth-relative assets need no rewrite |
| GitHub Pages (`gh-deploy`) kept as backup only | `site_url` is Vercel-prefixed, so GH canonicals intentionally diverge |
| `docs/.nojekyll` committed | Lets GitHub serve `assets/` and `doxygen/` dirs without Jekyll filtering |
| `mkdocs build --strict` | Catches broken links and nav drift in PRs |
| `git-revision-date-localized` plugin | Lightweight "last updated" metadata; no version branch required |
