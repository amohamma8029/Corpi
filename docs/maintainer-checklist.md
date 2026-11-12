# Maintainer checklist

Public source access is not release acceptance. Keep missing hosted, native,
capacity and installer evidence visible rather than treating these checklists as
proof that the app is ready to ship.

## Before announcing public access

- [ ] Merge the reviewed publication safeguards without merging unfinished search
  work. Review retained history and artifacts for private material and unresolved
  third-party rights; replacing an asset does not remove its old history.
- [ ] Obtain the owner's approval before changing repository visibility. Do not
  change billing, buy runners or enable paid services to complete these steps.
- [ ] Once the repository is public, enable **Private vulnerability reporting**
  in its security settings. GitHub provides this feature only for public
  repositories, so it cannot be verified as active while this repository is
  private. See [GitHub's configuration instructions](https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories/configuring-private-vulnerability-reporting-for-a-repository).
- [ ] Confirm the setting, rather than relying on the wording in `SECURITY.md`:

  ```powershell
  gh api repos/amohamma8029/Corpi/private-vulnerability-reporting
  ```

  The response must contain `"enabled": true`. An error, missing value or
  `false` is not a pass.
- [ ] Verify that **Security → Advisories → Report a vulnerability** opens the
  private reporting form. Do not submit a dummy vulnerability to test it.
- [ ] Confirm a repository administrator receives security-report notifications.
  Review the administrator's repository watch/security-alert and personal
  notification preferences; merely enabling reporting does not prove delivery.
  Do not publish a contact address without its owner's permission.
- [ ] Do not announce public access until the private channel is usable. If
  reporting cannot be enabled, obtain a real, owner-approved private contact and
  update `SECURITY.md` before the announcement. Public issues are not a private
  reporting channel.
- [ ] Run the configured Ubuntu, macOS and Windows checks on standard free public
  runners and keep their actual results. A private-repository billing refusal is
  missing verification, not a product-test result. Do not retry on paid runners.
- [ ] Apply the agreed branch protections only after their exact check names are
  present and usable on `main`. Require `Checks (windows-latest)` once this
  Windows workflow is integrated and verified; do not require a native Windows
  check that `main` cannot yet produce. Keep native/platform and release gaps
  separate from source publication.

## Before merging search PR #16

- [ ] Reconcile `README.md` and `SECURITY.md` with the final merged functionality.
  Remove obsolete statements that search is absent. Describe explicit model
  setup, offline queries, file/network access, worker supervision, saved settings
  and their real limits without claiming that resource caps are a PDF sandbox.
- [ ] Recheck `desktop/patches/README.md` against the merged dependency graph and
  callers, including FTP/PAC/ZIP restrictions, serializer configuration and the
  GLib finding. Tooling download restrictions do not secure application model
  downloads. Preserve the distinction between patched versions, source-scoped
  findings and unfixed defects.
- [ ] Retain the MIT fields in all three manifests and the root license. Review
  model and other third-party terms separately; MIT does not replace them.
- [ ] Reconcile both workflows without dropping the publication branch's tooling
  security check or its Windows core/UI/build/launcher checks. The search-memory
  and embedding-supervisor tests belong to PR #16; they must run on Windows
  without skips when that implementation is integrated.
- [ ] Keep `Native checks (windows-latest)` and its driver-provisioning controls
  from the search work. Obtain current-head Windows/WebView2 viewer and search
  evidence, plus Linux/macOS evidence. A Windows unit/build pass does not replace
  native viewer execution.
- [ ] Add the native Windows required check only after its job is on `main` and
  has produced the expected result. Preserve job, download, product-resource and
  evidence-retention limits when resolving workflow conflicts.
- [ ] Record the documentation, dependency and platform rechecks in PR #16 before
  merging it. Do not close its remaining acceptance requirements on the strength
  of this publication PR.
