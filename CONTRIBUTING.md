# Contributing to Corpi

Corpi is in development. Read the [development status and setup guide](README.md)
before changing code; `main` and open feature branches do not have the same features.
For substantial changes, discuss the intended behavior and scope in an issue first.
Keep pull requests focused and describe the user-visible result, checks performed,
and any checks that failed or were not run. Do not describe a skipped check as passed.

## Local checks

Follow the README's platform prerequisites and setup instructions. With the pinned
tools and dependencies already installed, run from the repository root:

```powershell
just test
```

This runs the core tests, lint/format/type checks, tooling security-patch checks,
UI unit tests, and API schema drift check. CI also runs the security-patch check
before its UI tests. These checks do not replace desktop builds or native E2E
checks. See the README's native testing instructions for the relevant platform. These require
explicit environment setup and use a dedicated synthetic test library.
Report the exact commands and platform used, and preserve existing resource limits
and failed evidence rather than changing them to make a check pass.

To run the security-patch check on its own, use:

```powershell
pnpm --dir desktop run check:security
```

If this reports stale or missing patched dependencies, run
`pnpm --dir desktop install --frozen-lockfile` from the repository root, then
rerun the check. Do not increase the test heap limit to work around an old install.

Read [patch provenance and limitations](desktop/patches/README.md) before changing
the exact patch selectors or restoring a restricted download path. This bounded
Node check does not start a browser, driver or model, and does not replace native
platform verification.

## Safe test data and review

Before announcing public access or merging search PR #16, follow the
[maintainer checklist](docs/maintainer-checklist.md). It covers private reporting,
security-document updates and the separate Windows CI/native requirements.

- Use synthetic documents or explicitly public fixtures in a dedicated test library.
  Never point a test, benchmark, or unreviewed PR at your personal Corpi library.
- Treat contributed code, build scripts, and documents as untrusted. Review them
  before running commands; pull-request CI also executes contributed code. Use an
  isolated test environment without credentials or private files when needed.
- Do not commit API keys, bearer tokens, `.env` files, personal documents, model
  caches, or local agent notes. `/.agent/` is ignored, but ignored files are not a
  safe place to store secrets.
- Inspect logs, screenshots, filenames, and CI artifacts before sharing them.
  Remove tokens, private paths, and document contents that are not public test data.
  Keep artifact uploads limited to the existing explicit evidence paths, not the
  whole workspace, app-data directory, or installed model directory.
- Keep CI actions pinned to full upstream commit hashes with readable version
  comments. When updating a pin, review its upstream release and action inputs;
  do not replace it with a moving tag or broaden workflow permissions.

Use [SECURITY.md](SECURITY.md) for suspected vulnerabilities. Contributions to
Corpi's own code use its [MIT license](LICENSE); retain third-party licenses and
notices for dependencies, models, fonts, and other assets.
