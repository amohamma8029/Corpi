# Security policy

Corpi is development software, not a security-reviewed release. There is no
supported stable release line or guaranteed response time. Report the affected
commit and platform so maintainers can identify the relevant source.

## Reporting a vulnerability privately

Do not put exploit details, credentials, private documents, or sensitive logs in a
public issue or pull request.

The reporting channel is GitHub private vulnerability reporting. Under
**Security → Advisories**, use **Report a vulnerability** to send a private report:

https://github.com/amohamma8029/Corpi/security/advisories/new

GitHub makes this feature available only for public repositories. Before
announcing public access, maintainers must enable it, verify the private reporting
form and confirm that report notifications reach a maintainer. Follow the
[publication checklist](docs/maintainer-checklist.md#before-announcing-public-access).
Until that is verified, do not post vulnerability details in an issue or pull
request. There is no separate published security email address or response-time
guarantee.

In the private report, include the commit/version, operating system, expected and
observed behavior, and minimal reproduction steps using synthetic or public data.
Redact tokens, personal paths, and private document contents. Do not test against
other people's accounts, documents, or systems.

## Security boundaries and limitations

- Corpi stores notebooks and imported files locally. Local storage is not
  encryption against someone who can read your operating-system account's files.
- The core uses a per-launch bearer token on loopback HTTP. Do not share that token
  in reports, screenshots, or logs; loopback binding alone is not authentication.
- PDF parsing uses a supervised process and resource limits.
  This is not an OS security sandbox or proof that arbitrary hostile files are safe.
- Keep private libraries and credentials out of test environments and unreviewed
  PR runs. See [CONTRIBUTING.md](CONTRIBUTING.md) for safe test-data practices.
- Third-party dependency advisories and release-verification gaps remain under
  review. Public source availability or an MIT license is not a security guarantee.

## Dependency patches and current findings

The desktop test tools use four exact-version local patches and two restricted
download paths. See [patch provenance and limitations](desktop/patches/README.md)
for upstream fixes, exact hashes, regression checks and recheck triggers. FTP PAC
retrieval and Puppeteer's ZIP browser-unpacking path are deliberately unavailable;
ordinary non-FTP proxy support remains. These restrictions cover test-tool
downloads, not application download paths. Adding a feature that downloads
models or changes file, network or worker access requires a fresh security review.
The [search merge checklist](docs/maintainer-checklist.md#before-merging-search-pr-16)
includes that review and the accompanying documentation updates.

Version-only scanners can still report patched glob/minimatch versions. The same
record also explains the current-configuration findings for Mocha's serializer
and build-time rand, and the source-scoped absence of callers for GLib's affected
iterator. **The upstream GLib defect is not fixed here.** Linux-native verification
and Rust maintenance debt remain separate; recheck these findings after relevant
source, feature or dependency changes rather than treating them as permanent
exceptions.
