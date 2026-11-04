# Corpi

Corpi is an in-development, single-user desktop app for working with imported sources.
The current `main` branch includes notebooks, saved PDF imports, a PDF viewer, text
extraction and readability review. Local Words/Meaning search is being developed
separately in [PR #16](https://github.com/amohamma8029/Corpi/pull/16); it is not yet
part of `main`.

Notebooks and imported files stay on the user's device. Provider-generated answers
and external model-provider configuration are planned, not implemented here.

## Development status and limits

Corpi is not a supported stable release. Native compatibility needs evidence for
the particular commit and platform; a unit-test or build pass does not replace it.
Supervised PDF parsing and resource limits are not an OS security sandbox for
hostile files. A user-facing backup/restore feature is deferred to Milestone 2.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for local checks and safe test data, and
[SECURITY.md](SECURITY.md) for private reporting and current dependency limitations.

## Repository layout

- `core/` — Python local application core (FastAPI over loopback HTTP), managed with uv.
- `desktop/` — Tauri 2 desktop shell: React/TypeScript UI (`src/`), Rust shell (`src-tauri/`).
- `docs/` — product requirements and architecture.

## Prerequisites

- [mise](https://mise.jdx.dev/installing-mise.html) — `mise.toml` pins the project tools (just, Node.js LTS, pnpm, uv, Rust) and `mise install` installs them
- Windows builds additionally need:
  - MSVC build tools (Visual Studio Build Tools with the "Desktop development with C++" workload)
  - WebView2 runtime (Windows ships it with Edge)
- macOS builds additionally need Homebrew Python 3.13 with SQLite 3.51.3 or newer; see the macOS setup below. `uv` still manages the core's packages.
- Native E2E tooling depends on the host:
  - Windows: Microsoft Edge WebDriver matching the app's WebView2 runtime (winget: `Microsoft.EdgeDriver`).
    The runner launches the app itself and attaches directly to that driver.
  - Ubuntu: the Tauri build libraries, `webkit2gtk-driver`, `xvfb`, `xauth`, and `dbus-x11`; install `tauri-driver` with `cargo install tauri-driver --version 2.1.0 --locked`.
  - macOS: the test runner uses the embedded WebDriver plugin; it needs no external driver or paid subscription.

On Windows, Edge WebDriver's first three version components must match the **WebView2 runtime**
used by the app, not merely the Edge browser. WebView2 can update independently of the driver.
See [Microsoft's WebView2 testing guidance](https://learn.microsoft.com/en-us/microsoft-edge/webview2/how-to/webdriver).
`CORPI_EDGE_DRIVER` can select an existing compatible `msedgedriver.exe` without changing PATH;
the test runner does not download or update drivers automatically.

If an external Windows run fails with a driver/WebView version mismatch before creating a
session, compare the versions in that error. Select a driver matching WebView2, for example
`$env:CORPI_EDGE_DRIVER = "C:\tools\matching-driver\msedgedriver.exe"`, then rerun. Updating
Edge alone may not fix the mismatch. The embedded runner is an alternative that does not use
Edge WebDriver; its input checks have the synthetic-input limits described below.

`uv` manages the core's Python environment and packages. It downloads Python 3.13 by default; on macOS, select the Homebrew interpreter below so Python loads a supported SQLite version.

## Setup

```powershell
mise trust          # trust this project's mise.toml (once per clone)
mise install        # installs just, Node.js, pnpm, uv, and Rust at the pinned versions
mise activate pwsh | Out-String | Invoke-Expression   # add to your $PROFILE to make it permanent
uv sync --directory core
pnpm --dir desktop install
just dev
```

With mise activated, `node`, `pnpm`, `just`, `uv`, and `cargo` resolve to the pinned versions inside this repository; outside it, your global mise tools are untouched.

## Notebooks and PDF originals

Create or open a notebook, then add one PDF at a time. Corpi copies the bytes into the user's
app-data directory; moving the selected file later doesn't change the saved original. Replacing a
PDF keeps the previous current file available until the new version is ready. The Files section
opens the exact saved import, including older history. Identical current/pending PDFs in one
notebook reuse their source entry, and healthy physical originals can be shared across notebooks.

Unfinished imports reappear when their notebook is reopened. Complete saved input offers Retry;
an interrupted partial upload asks you to select the PDF again for the same attempt. Cancel
discards the unfinished attempt. The PDF viewer keeps keyboard focus inside while open and
returns it to the Open button on close; Escape also closes it. A failed open shows a wrapping error
message while retaining the original file.

The core enforces a configurable **25,000,000-byte** starting upload limit, a **120-second** total
transfer deadline, and a **30-second** no-bytes deadline. Only one incoming PDF transfer runs per
core. PDF.js downloads the complete verified file and renders selected pages under a configurable
**20-megapixel** starting canvas limit; the upload cap doesn't bound decoded memory. PDF.js and its
matching worker/support assets are copied from the pinned `pdfjs-dist` package during `just dev`
and desktop builds. Those generated assets are ignored by Git.

Environment overrides are read by the core at startup:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `CORPI_PDF_MAX_BYTES` | `25000000` | Maximum received PDF bytes. |
| `CORPI_PDF_UPLOAD_TIMEOUT_SECONDS` | `120` | Total time from transfer request acceptance through finalization. |
| `CORPI_PDF_UPLOAD_IDLE_SECONDS` | `30` | Maximum wait for the next non-empty body chunk. |
| `CORPI_PDF_RENDERED_PAGE_PIXELS` | `20000000` | Maximum canvas pixels for a rendered page. |
| `CORPI_PDF_STAGING_AGE_SECONDS` | `86400` | Minimum idle age before unowned staging can be cleaned. |
| `CORPI_MAINTENANCE_INTERVAL_SECONDS` | `3600` | Existing local maintenance interval. |

Original-PDF import, viewing, extraction, and readability review are implemented. Complete
extraction queues local search automatically; partial PDFs require confirmation of the usable
pages. A source becomes ready after both word and meaning-search data pass their publication
checks. The Files section keeps the saved original accessible throughout processing.

### Local search setup

Open **Local search** in the notebook panel and choose **Download English, French and Arabic
search**. New notebooks use the pinned Bekko A8M quint8 AVX2 model (about 140.6 MB).
Existing notebooks keep their saved model choices. Fast English search, all-MiniLM-L6-v2
(about 23.7 MB), and multilingual E5 small (about 135.4 MB) remain available.

Downloads start only through Download or Retry setup. The panel shows received bytes and
offers Cancel; failed or interrupted attempts offer Retry setup after their staging cleanup.
Corpi verifies the pinned file sizes and hashes before installing a model. Notebooks share
one installed copy per model, and installed-model inference runs locally without a download.
A missing model leaves search waiting while other PDF extraction can proceed.

Use **Search this notebook** to find passages in its ready documents. **Meaning** finds related
wording using the notebook's installed model; **Words** finds word matches without loading a
model. Submit the search explicitly. Search in the same language as the document; cross-language
retrieval is not a first-release guarantee. A result's **Open PDF · page …** button opens its
exact saved PDF version and citation page, including when a replacement is still preparing.
Results show the actual saved filename, current/previous saved version, and indexing model.
The replacement's filename never substitutes for the original result's file.

Review, trash and processing-cancel actions clear displayed/in-flight results. A lightweight
background check also clears them when searchable data changes elsewhere, including expiry.
Your query text stays in place; submit it again for current results. These checks do not run
inference, download a model, or automatically repeat a search. A pending replacement or model
choice alone keeps eligible existing search available.

The Files section offers **Cancel search preparation** and, for eligible stopped work,
**Resume search preparation**. Resume keeps already prepared chunks, saved settings, attempt
limits and time charges. It does not grant a fresh allowance. Exhausted work cannot be resumed.
Cancellation remains stopped after restarting the app; non-cancelled interrupted work recovers
from its durable progress.
The piece counter shows embedding progress only. Even when every piece is embedded, validation
and publication must finish before new search data is ready. Cancelled preparation shows why it
stopped, such as Review Again reopening its text review or the selected input/model changing.

**Cancel processing** stops the exact PDF import, including when extraction has finished but
the PDF is waiting for review. The previous ready PDF stays usable. **Review Again** reopens
the current review; **Confirm** schedules preparation for that new review generation. A review
reopened on an already searchable PDF hides that PDF's search until it is confirmed and
published again. Old prepared vectors may be reused without changing saved citations.
Interrupted preparation conservatively keeps any unfinished time reservation. New jobs
reserve across a whole batch attempt, so an interruption can consume allowance even
before inference starts. Existing saved jobs keep their previous accounting rules.

Setup fetches only pinned HTTPS assets from `huggingface.co` and its reviewed
`us.aws.cdn.hf.co` download CDN. Unexpected redirect hosts are rejected; Corpi does not
forward application credentials or cookies. An upstream routing change may require an update.

Changing **Notebook model** rebuilds search for the notebook's ready sources. Existing search
stays usable until every required rebuild completes and the notebook switches models together.
**Cancel model change** keeps the active model. Download cancellation and notebook-choice
cancellation are separate controls. Rebuilding search leaves citation passages and PDF originals intact.
The panel labels both **Active** and **Requested** models during a change. Selecting a model
does not make it active early. A failed/cancelled rebuild reports its stopped counts and keeps
the active model available until you cancel the change or the required work succeeds.

New search preparation allows twenty minutes (1,200 seconds), with two inference threads,
120-second inference calls, 300-second index-build attempts, 10,000 chunks and 20,000,000
eligible UTF-8 bytes per source. Existing jobs retain their recorded policy and charges.
Bekko processes one chunk at a time; MiniLM uses batches of 32 and E5 batches of eight.
New preparation saves `openblas_threads=1` and explicitly sets `OPENBLAS_NUM_THREADS=1`
in its inference workers, without changing ONNX's two-thread setting. Older work without
the saved field keeps its previous worker environment, including on Resume. Meaning
search loads compatible groups serially when a notebook contains both kinds of index;
this can require another model load. Words search still needs no inference worker.
New E5 preparation disables the CPU memory arena to reduce retained worker memory;
Bekko and MiniLM keep it enabled. Existing jobs and indexes retain their recorded
arena choice, including on Resume; this does not migrate or repair old arena-enabled E5 work.
Arabic presentation forms and tatweel are folded only in derived search input. Saved text,
PDFs and citation coordinates remain unchanged.

The app retains its sampled 1-GiB child-memory cutoff. Cached Windows verification uses
separate 1-GiB application-tree and 256-MiB controller-tree jobs. These Job Objects limit
committed memory; strict aggregate resident/cache memory and no-swap acceptance remain
unproved. These limits do not establish a hard production RAM cap or everyday speed.
Setup allows one transfer at a time, up to three transfer attempts within a 600-second budget,
a cumulative 1,000,000,000 received-byte ceiling, and a 30-second no-bytes deadline.

### Local search developer benchmarks

The scripts under `core/scripts/` compare pinned local embedding models using authenticated
core import, extraction, supervised inference, on-disk word/vector indexes, and guarded readiness.
`core/benchmarks/search_questions.json` pins the public PDF hashes and expected evidence.
Run the scripts from `core/` with dedicated absolute directories for fixtures, models, and results:

```powershell
uv run python scripts/fetch_search_pdfs.py <pdf-directory>
uv run python scripts/benchmark_search_runtime.py --directory <model-directory> --profile minilm-en --download --disable-arena
uv run python scripts/benchmark_pdf_search.py --pdf-directory <pdf-directory> --models-directory <model-directory> --output-directory <result-directory> --pdf bert --profile minilm-en --overlap 8 --repetitions 2
uv run python scripts/summarize_search_benchmarks.py <result-directory> --compact
uv run python scripts/render_search_charts.py --results <results-json> --platform-checks <platform-checks-json> --screenshots <native-screenshot-directory> --output <chart-directory>
```

Generated reports, results, and charts are local PR evidence; they are not repository documentation.
Publish the relevant figures and measurements with the implementation PR. Existing paper benchmarks
measure the core API workflow, not desktop-visible waiting time. Managed setup and notebook switching
have scoped integration checks. The current Bekko runtime parses one shared tokenizer
before backend/session allocation. Windows both-mode search/saved-page navigation,
multilingual recovery, repeated original-paper/exact-200-page native imports and long
cancellation/following import pass. The unchanged near-capacity fixture publishes in
1,019.90 seconds in the latest saved-setting run, with every vector and canonical mapping
matching the original references. New arena-off E5 preparation passes switching,
both-mode search and saved-page navigation in two fresh copied native libraries,
with exact original vectors and no job memory-limit events. Its matched worker peak
private memory falls from about 895 to 581 MiB, at a 9.4% document-inference cost in
that single pair. These results are not broad reliability or strict RAM/no-swap proof.
Old arena-enabled E5 work is unchanged. The model panel now reports current failed
or cancelled rebuilds instead of leaving them labelled “Rebuilding”; Cancel model change
keeps the active model. Failed or cancelled model-switch jobs no longer mark the
saved PDF import failed, so its search-result PDF/page shortcuts remain available.
Genuine import failures still report failure; previously failed libraries are not
reset or migrated. The CI matrix separately exercises MiniLM/E5 managed setup, offline
inference and native viewer/search flows on Linux/macOS. Those selected checks do not establish
dedicated default-Bekko coverage or strict aggregate memory/no-swap acceptance. Full M1.6
acceptance remains pending; benchmark results and native runs are evidence for their recorded
source, model and scenario, not any later edited build.
Benchmark-combined quality is not a Meaning-only or Words-only UI success rate.

For desktop-visible timing on Windows, `core/benchmarks/search_readiness.json`
pins two original papers and an unmodified, exactly 200-page World Bank report.
Place those exact files in a dedicated PDF directory and verify their hashes;
the desktop benchmark never downloads fixtures or models. First run the ordinary
embedded native build/smoke command from `desktop/`:

```powershell
pnpm run test:e2e:embedded
```

Then run from `core/`, with the requested model already installed in an explicit
separate test library. The benchmark selects that model even when it is the default;
other installed models do not affect eligibility. Supply the absolute test-library
path and, for a retained binary outside the normal build location, its passing native
receipt and exact binary path. Only a WebDriver-enabled test binary can redirect its
library through `CORPI_E2E_DATA_DIR`; production binaries ignore that variable.

`CORPI_RUN_INSTALLED_MODEL_E2E=1` enables the native default/readiness/switch check
using already verified assets. Keep `CORPI_RUN_MODEL_DOWNLOAD=0` for cached-only
verification; a missing model then fails the check instead of starting a transfer.
This mode is not evidence of a fresh managed download or setup Cancel/Retry.

```powershell
uv run --no-sync python scripts/benchmark_desktop_readiness.py --pdf bert --profile bekko-a8m-quint8-avx2 --pdf-directory <absolute-pdf-directory> --app-data <absolute-separate-test-library> --directory <new-absolute-result-directory> --pnpm <absolute-installed-pnpm.exe> --binary <absolute-test-binary> --binary-evidence <absolute-passing-native-receipt> --repetitions 2
```

Select `attention` or `global-economic-prospects-2015` for the other PDFs. MiniLM
(`minilm-en`) and E5 (`multilingual-e5`) remain explicit alternative profiles. New
measurements record the 1,200-second versioned preparation recipe and verify the
exact saved profile, batch size, budget policy, native readiness and original bytes.
Receipts without a recipe retain their historical 600-second interpretation; they
are not relabelled as new twenty-minute results. Each repetition creates a fresh
notebook/source and performs
normal upload, extraction, model load, search preparation and guarded publication.
The timer starts at the file-input change and ends when both ready labels appear;
fixture injection and initial model setup are excluded and reported separately.
The 200-page report is a partial extraction (187 extracted, 11 blank, 2 unreadable
pages). The benchmark checks that exact page report, explicitly uses the normal
Confirm control, and records review-visible and confirmation times; its elapsed
time is not an unattended complete-import result. No page rules are bypassed.
It records UI frames, existing API polling latency, sampled core/child RSS and host
CPU use, plus readonly verification of saved PDF bytes, extraction and native rows.
Polling reads page/review metadata and counts, not the full passage/geometry graph;
the UI does not accumulate new background polling pairs behind a slow response.
File/select events are synthetic: this is not an OS file-picker/physical-input or
cold-cache measurement. Existing test app data is retained; avoid concurrent native
runs. `--cancel --repetitions 2` cancels the first import after an embedding batch
and checks a following fresh import. Preserve failed results; `--inspect` can
recheck saved source IDs without repeating the actual imports.

The additional static multilingual benchmark uses pinned Potion assets and the existing
NumPy/tokenizers packages. Its 256-dimensional vectors are measured in a separate benchmark
database. `benchmark_shared_search.py` compares identical short chunks and the fixed questions;
`--native-chunks` freezes Potion's own chunk settings instead. Supply `--source-runs` with a JSON
object mapping the five PDF keys in `search_questions.json` to completed relative run folders.
Saved partial extractions must have been confirmed. Existing application model choices remain
as described above.

```powershell
uv run python scripts/benchmark_static_multilingual.py --download --directory <model-directory>
uv run python scripts/benchmark_shared_search.py --freeze --source-runs <source-runs-json> --runs-directory <result-directory> --pdf-directory <pdf-directory> --models-directory <model-directory> --directory <new-comparison-directory>
uv run python scripts/benchmark_shared_search.py --directory <comparison-directory> --models-directory <model-directory> --profile <candidate-id>
uv run python scripts/benchmark_shared_search.py --directory <comparison-directory> --models-directory <model-directory> --summarize
uv run python scripts/benchmark_shared_search.py --directory <comparison-directory> --models-directory <model-directory> --capacity-run <retained-capacity-run-directory>
```

Run every declared candidate before summarizing shared inputs. The capacity command measures
encoding of retained text; `--varied` adds unique observation labels without replacing the dense
paragraphs. Both retain the 600-second and sampled 1-GiB inference limits. These diagnostics
report retrieval evidence coverage and encoding time; full application readiness is measured
through the guarded import/preparation workflow separately.

Two additional contextual candidates are pinned in
`core/benchmarks/contextual_multilingual_models.json`: LEALLA-small and an mmBERT
multilingual sentence-embedding student. Their benchmark worker requests the exported
sentence vector, or restores LEALLA's trained pooler from its pinned reference checkpoint.
It loads compiled tokenizer JSON directly as UTF-8 and retains the existing CPU, memory,
token and deadline controls. These profiles are available only to the benchmark scripts.

```powershell
uv run python scripts/benchmark_contextual_multilingual.py --download --profile lealla-small --directory <model-directory>
uv run python scripts/benchmark_contextual_multilingual.py --download --profile mmbert-student --directory <model-directory>
uv run python scripts/benchmark_shared_search.py --directory <comparison-directory> --models-directory <model-directory> --profile lealla-small
uv run python scripts/benchmark_shared_search.py --freeze --chunk-profile lealla-small --source-runs <source-runs-json> --runs-directory <result-directory> --pdf-directory <pdf-directory> --models-directory <model-directory> --directory <new-native-comparison-directory>
```

The original frozen short corpus can be reused for contextual candidates; the result records
their additional pinned model identities. `--chunk-profile` selects the candidate's intended
chunk settings when freezing a separate corpus. To test the original dense capacity input,
combine `--capacity-run` with `--profile`. Preserve failed attempts alongside successful results.

The additional shortlist compares retrieval-trained `amgix-static-retrieval` and
four-layer `bekko-a8m` against retained Potion results. Amgix uses its pinned
Sentence Transformers static-mean semantics (including UNK) with existing NumPy/tokenizers.
Bekko uses the official vocabulary-only INT8 ONNX artifact, full 384-wide masked mean
and the existing CPU runtime. Neither benchmark adds an application model choice.

```powershell
uv run python scripts/benchmark_static_multilingual.py --retrieval --download --directory <model-directory>
uv run python scripts/benchmark_contextual_multilingual.py --download --profile bekko-a8m --directory <model-directory>
uv run python scripts/inspect_shortlist_models.py --models-directory <model-directory> --output <new-inspection-json>
uv run python scripts/benchmark_shared_search.py --directory <comparison-directory> --models-directory <model-directory> --profile amgix-static-retrieval --retain-rankings
uv run python scripts/benchmark_shared_search.py --directory <comparison-directory> --models-directory <model-directory> --profile bekko-a8m --capacity-run <original-capacity-run> --screen-batches 8 --batch-size 16
uv run python scripts/benchmark_search_capacity.py --directory <result-directory> --models-directory <model-directory> --profile amgix-static-retrieval --input-run <original-capacity-run> --chunks 9950 --batch-size 32
```

`--screen-batches` measures four or eight original-input requests without claiming
capacity or readiness. `--input-run` validates and preserves original input bytes
for the complete four-step preparation diagnostic. The experimental static adapter
is supplied only inside that benchmark process and its separate database. Its
synthetic saved extraction does not establish PDF parsing or desktop timing.

The same benchmark catalog also includes `distiluse-v2`, `granite-107m` and
`granite-97m-r2`. DistilUSE keeps its native 128-token limit and trained masked-mean,
768-to-512 dense/tanh layer; it rejects longer inputs. The Granite profiles use their
trained CLS output, including the model-specific padding token. Both Granite trials
use a recorded 512-token diagnostic input ceiling. `--batch-size 8`, `16` or `32`
records an explicit bounded batch setting for quality/encoding runs. Native chunk
comparisons and failed resource attempts remain separate evidence.

Granite CPU-runtime experiments use the unchanged dense capacity text:

```powershell
uv run python scripts/benchmark_granite_options.py --capacity-run <retained-capacity-run-directory> --models-directory <model-directory> --directory <owned-experiment-directory> --operator-profile --repetitions 2
uv run python scripts/benchmark_granite_options.py --capacity-run <retained-capacity-run-directory> --models-directory <model-directory> --directory <owned-experiment-directory> --repetitions 4
uv run python scripts/benchmark_granite_options.py --capacity-run <retained-capacity-run-directory> --models-directory <model-directory> --directory <owned-experiment-directory> --repetitions 8 --affinity-trial
```

Profiling is separate from option timing. The runner records actual threads/allocator
settings, native operator time, sampled child memory, document/query output agreement,
and stop/reap outcomes. `--derived-graph` compares a separately authorized experimental
graph without changing verified assets. These bounded diagnostics are not full readiness
or retrieval-quality passes; retain rejected outputs and slower candidates as evidence.

On 64-bit Windows, `--affinity-trial` compares ordinary and separately pinned physical-core
placement in ABBA order. It verifies the calling thread and the one ORT compute worker's
actual affinity masks, retains the original dense input, and changes no system power setting.

Saved Potion rankings can be inspected without running inference:

```powershell
uv run python scripts/analyze_potion_misses.py --shared-directory <shared-comparison-directory> --native-directory <potion-native-comparison-directory> --output <new-analysis-json>
```

The analysis verifies frozen input hashes and reconstructs all English evidence scores from
saved citation mappings. It reports the seven shared-input vector misses, word-search recovery
and separate native-chunk results. The output parent must exist. Rankings below the saved top
ten remain unknown unless a fresh frozen-input `benchmark_shared_search.py` run used
`--retain-rankings`. Pass that result with `--rankings-recheck-file`; the analyzer rejects
changed top-ten results before reporting deeper ranks. Evidence labels stay unchanged.

`analyze_amgix_misses.py` compares saved French/cross-language misses against Potion
and Bekko, reconstructing the original citation-span scores. It performs no inference.
Supply the retained Potion top-fifty recheck; it must reproduce the original top ten.
Optional `--models-directory` checks whether complete French statements fit the
original declared token budget. A separately frozen passage-preserving trial uses
the same source text, evidence labels and original counting tokenizers:

```powershell
uv run python scripts/analyze_amgix_misses.py --runs-directory <raw-results> --potion-rankings-recheck <retained-recheck-json> --models-directory <model-directory> --output <new-analysis-json>
uv run python scripts/benchmark_shared_search.py --freeze --passage-preserving-reference <original-shared-corpus-json> --runs-directory <raw-results> --pdf-directory <pdf-directory> --models-directory <model-directory> --directory <new-packing-trial-directory>
uv run python scripts/benchmark_shared_search.py --directory <packing-trial-directory> --models-directory <model-directory> --profile <candidate-id> --retain-rankings
uv run python scripts/benchmark_shared_search.py --directory <packing-trial-directory> --models-directory <model-directory> --summarize --summary-profiles amgix-static-retrieval potion-multilingual-static bekko-a8m --compact
```

Run all three selected profiles before that summary; use batch16 for Bekko.
`--trial-directory` adds the before/after comparison to the analyzer;
`--export-evidence` copies hash-verified source/result receipts beside its output.
The packing policy keeps a fitting saved passage whole rather than splitting it
to fill the preceding chunk. It still splits oversized passages, preserves source
code points/page boundaries and enforces text/chunk limits. It is opt-in for this
diagnostic; application calls keep the original policy. New chunk scores remain
separate from the original frozen benchmark and do not establish full readiness.

`benchmark_openvino_granite.py` is a separately authorized temporary CPU-backend
experiment. It requires an owned `corpi-granite-openvino-experiment` directory with
an ownership marker and an explicit declined telemetry file in temporary LOCALAPPDATA.
`--prepare` records the already-approved OpenVINO/telemetry/NumPy overlay without
importing backend code; it never installs packages. Worker imports must resolve inside
that owned root, and Python network connections are denied. `--batch-size` selects a
bounded smoke comparison; `--profile-operators` records CPU implementations separately
from normal timing. Output checks, memory failures and package/environment cleanup are
experiment evidence, not application adoption. Project dependencies remain as configured
in the component manifests.

`benchmark_bekko_openvino.py` tests Bekko's official pinned default OpenVINO IR.
Its Windows experiment requires the owned `corpi-bekko-openvino-experiment` root,
an ownership marker with `task: m1-6-local-search` and `experiment: bekko-openvino`,
and `localappdata/Intel Corporation/openvino_telemetry` containing `0` before import.
`--inspect-packages` downloads/hash-checks the previously inspected wheels without
executing their code. After separate temporary-runtime approval, use an owned uv
cache/overlay with OpenVINO 2026.4.1, telemetry 2025.2.0 and existing NumPy 2.5.3;
`--prepare` records their versions/origins without importing the backend. It does
not install packages itself. All backend workers use CPU/two threads/one stream/F32.

```powershell
uv run python scripts/benchmark_bekko_openvino.py --download --models-directory <model-directory>
uv run python scripts/benchmark_bekko_openvino.py --directory <owned-runtime-root> --models-directory <model-directory> --capacity-run <original-capacity-run> --batch-size 16
uv run python scripts/benchmark_bekko_openvino.py --directory <owned-runtime-root> --models-directory <model-directory> --export --output-directory <new-evidence-directory>
```

Repeat the paired screen with batches 8 and 32 for the export. Each screen has a
600-second cumulative deadline, supervised per-call deadlines and sampled child
RSS cutoff. Preserve each failure rather than overwrite it. Evidence export verifies
model/package hashes and declined consent. After checking that owned runtime
processes have stopped, `--cleanup --output-directory <exported-evidence-directory>`
reverifies the exports and removes only the exact owned temporary runtime root.
Independently check its absence afterward. Verified benchmark model assets remain
available for reproduction; no permanent application dependency is added.

`benchmark_bekko_gguf.py` is an explicitly approved temporary Windows CPU experiment
for the official Bekko Q8_0 export. It pins llama.cpp b11429 and verifies its CPU archive,
member hashes and version-matched C headers before execution. The owned root must be
`corpi-bekko-gguf-experiment` with the matching Task/experiment/revision marker.
`--inspect`, `--prepare` and model-header inspection do not execute the new runtime.
Native execution uses inspected DLLs through a supervised child, CPU Haswell/AVX2,
two compute threads, non-causal mean pooling and the original HF token IDs. Only
explicitly checked space/tab token-display aliases are allowed; no text is rewritten.

```powershell
uv run python scripts/benchmark_bekko_gguf.py --inspect --directory <owned-runtime-root>
uv run python scripts/benchmark_bekko_gguf.py --download --models-directory <model-directory>
uv run python scripts/benchmark_bekko_gguf.py --inspect-model --directory <owned-runtime-root> --models-directory <model-directory>
uv run python scripts/benchmark_bekko_gguf.py --source-proof --directory <owned-runtime-root>
uv run python scripts/benchmark_bekko_gguf.py --prepare --directory <owned-runtime-root>
uv run python scripts/benchmark_bekko_gguf.py --directory <owned-runtime-root> --models-directory <model-directory> --capacity-run <original-capacity-run> --native-passages 1 --tag <new-diagnostic-tag>
```

Physical forwards of 1,2 or4 passages are separate settings from the logical eight-passage
request. The native token-buffer limit is512 per passage; overflows reject rather than
truncate. All original source/time/RSS/thread limits remain. The optional `--sustained-ort`
diagnostic collects owned-process CPU time, aggregate system load and Windows-reported
MHz for180 seconds, without changing host settings. It is not a capacity pass.
Export with `--export --output-directory <new-evidence-directory> --models-directory
<model-directory> --capacity-file <retained-full-encoding-json>`; after checking owned
processes are stopped, `--cleanup` reverifies exports and removes only the owned runtime.
Independently check absence. Public benchmark assets remain separate from app installation.

### macOS Python and SQLite

The uv-selected Python on macOS CI loaded SQLite 3.50.4, below the version Corpi requires for safe WAL use. Set up the project tools and use Homebrew's Python 3.13 as the interpreter for the `uv` environment. From the repository root in zsh:

```sh
mise trust
mise install
eval "$(mise activate zsh)"
brew install python@3.13
export UV_PYTHON="$(brew --prefix python@3.13)/bin/python3.13"
uv sync --directory core
pnpm --dir desktop install
uv run --directory core python -c 'import sqlite3; print(sqlite3.sqlite_version)'
export CORPI_CORE_DIR="$(pwd)/core"
pnpm --dir desktop tauri dev
```

Confirm that the printed SQLite version is at least 3.51.3. Keep `UV_PYTHON` set when launching the desktop app so its child core uses the same interpreter. The root `just` recipes currently require `pwsh`; the direct macOS launch command above does not. This setup does not change the Python-core packaging decision.

## Commands

Run from the repository root:

| Command | What it does |
| --- | --- |
| `just dev` | Launch the desktop app in dev mode; the shell spawns, monitors, and shuts down the local core. |
| `just test` | Core tests (pytest), lint (Ruff), types (Pyright), tooling security-patch checks, UI unit tests (Vitest), and the OpenAPI schema drift check. |
| `just core` | Core tests + Ruff + Pyright. |
| `just ui` | UI unit tests. |
| `just e2e` | Windows WebdriverIO + Microsoft Edge WebDriver checks against the real window, including PDF import/viewer/worker, then an orphaned-core shutdown check. |
| `pnpm --dir desktop run test:e2e` | Build and test the native app with the external driver on Windows/Linux or the embedded provider on macOS. |
| `pnpm --dir desktop run test:e2e:embedded` | Test the embedded provider explicitly, including on a Windows development host. |
| `pnpm --dir desktop run check:e2e` | Type-check the native E2E tests and WebdriverIO configuration. |
| `pnpm --dir desktop run check:webdriver-isolation` | Check that default app dependencies and capabilities exclude the test automation plugin. |
| `just schema-export` | Re-export `core/openapi.json` after changing core routes. |
| `just schema-check` | Verify the committed OpenAPI schema still matches the live core routes. |

Core setup is `uv sync` inside `core/`; desktop setup is `pnpm install` inside `desktop/`. On macOS, set `UV_PYTHON` as shown above before syncing or launching the core.

### Verify local changes

After setup, run `just test` for the core tests/static checks, UI unit tests and live-schema
comparison. Also run `pnpm --dir desktop build` for generated API types, TypeScript and the
production frontend, plus `pnpm --dir desktop run check:e2e` and
`pnpm --dir desktop run check:webdriver-isolation` for the native test contract. The ordinary
test selection skips opt-in model downloads and installed-model inference; skipped tests are
not model/native acceptance.

For UI or core changes, rerun the appropriate native command from the table without
`--reuse-build`. That flag checks the retained binary's hash but does not embed your new
frontend or rebuild changed application code. Preserve the existing `desktop/e2e-results/`
evidence before rerunning: each provider's output directory is replaced by its next run.
Windows external reuse also requires the original generated test identifier and debugging
port saved in `run.json`. Older binaries without that binding must be rebuilt. A reused
binary keeps its test library/profile; a fresh Windows E2E build gets new isolated data.
Keep test libraries separate from personal app data. Cached-model and download-enabled runs
are distinct checks; enable transfers only when you intend to test setup.

## Native viewer tests in CI

The GitHub Actions matrix includes Ubuntu/WebKitGTK, macOS/WKWebView and Windows/WebView2.
Each matrix job runs core tests/lint/types, the existing pinned-model setup and
offline-inference checks, UI tests/build/contracts, Rust formatting and shell
build/launcher tests. Linux and macOS also run the shared native viewer/search suite
there. Windows runs that full suite in the separate `Native checks (windows-latest)`
job so it no longer shares its budget with core checks and the debug shell build. Both Windows
jobs retain a 45-minute limit; test and application limits are unchanged. The native
job is independent of the matrix, so a missing macOS runner does not block it. This
uses another Windows runner and repeats tool setup, but does not repeat the core
test/model suites. Shared command steps use Bash on all three hosts; Windows
prerequisite steps use PowerShell explicitly.

Windows CI selects uv-managed Python from `core/.python-version`, rather than the runner's
default Python. Every host checks the actual loaded SQLite version before running core
tests. Windows additionally runs the memory-sampler/process-object controls with a
no-skips assertion and retains their JUnit report.

The Windows native job fetches the locked Rust dependencies without a redundant debug
build, then `.github/scripts/Setup-WindowsWebDriver.ps1` queries the installed
WebView2 runtime through the loader from the locked Rust dependency. It downloads that
exact version of EdgeDriver from Microsoft into the runner's temporary directory, checks
the archive layout, Microsoft signature and first-three-component version match, then
sets `CORPI_EDGE_DRIVER` for subsequent CI steps only. It retains a version/hash receipt
as `native-driver-windows-latest-<commit>`, separately from the matrix's platform evidence.
Missing exact-version downloads, invalid signatures or mismatches
fail setup; there is no fallback to the Edge browser's version or an unverified driver.
It does not install a driver, change PATH or update WebView2. The ordinary local E2E
runner still never downloads drivers automatically.

Windows external sessions use classic WebDriver commands rather than WebdriverIO's
automatic BiDi negotiation. On WebView2, the BiDi mapper can replace the loaded app
page with `about:blank`. Keep `wdio:enforceWebDriverClassic` enabled for that path in
`desktop/wdio.conf.ts`; the startup test rejects a negotiated BiDi session and still
requires the real app and core to become ready. It does not navigate a blank page to
make the test pass. Before building, the Windows runner generates an E2E-only configuration
with a loopback debugging port and a unique `app.corpi.desktop.e2e.t...` identifier.
Tauri passes the debugging arguments directly to WebView2 and gives that identifier its own
profile/library in Windows app-data. This works without relying on debugging
environment variables, which [WebView2 ignores in elevated processes](https://github.com/MicrosoftEdge/WebView2Feedback/issues/5640#issuecomment-4923662109).
The pinned Wry browser defaults are preserved; changed Wry versions or existing custom
browser/proxy settings require review rather than silently dropping those settings.
Normal app configuration, Cargo features, CSP and runtime pins are unchanged.

The runner starts one app, waits for its loopback debugging endpoint, then attaches the matching driver.
Session retries reattach to that same app instead of launching duplicate cores. It closes
the native window before stopping the driver; failed graceful cleanup remains a test failure.
The isolated profile/library stays outside the uploaded evidence directory; old test data
is not moved or deleted. Other platforms and the
embedded provider retain their existing settings.

Shared Bash commands fail on the first error on Windows as well as other hosts.
See the [maintainer checklist](docs/maintainer-checklist.md#before-merging-search-pr-16)
before integrating this branch. Configuration alone does not establish a passing
hosted run or native compatibility.

Ubuntu launches the app under Xvfb and D-Bus:

```sh
xvfb-run -a --server-args="-screen 0 1280x1024x24" dbus-run-session -- pnpm --dir desktop run test:e2e
```

Start Xvfb before D-Bus so activated desktop services inherit `DISPLAY`. The CI dependency
step uses bounded APT download timeouts/retries and a ten-minute step timeout, so an unavailable
package mirror cannot consume the whole test job.

The macOS job runs `pnpm --dir desktop run test:e2e` with the Homebrew Python selection above.
It uses WebdriverIO's Tauri launcher and the optional `webdriver` Cargo feature to control the
actual app. Linux uses `tauri-driver`; Windows attaches the verified Microsoft driver
directly to its runner-owned WebView2 window. All three native suites embed the built
frontend, local PDF.js worker and support assets; they use the real core and the app's
CSP. Linux/macOS test builds use the `app.corpi.desktop.e2e` Tauri identifier; fresh
Windows external builds use its unique suffix described above. Their saved notebooks
stay separate from ordinary Corpi data.

All three CI hosts also download the pinned default Bekko, MiniLM and E5 search models
through the authenticated setup API. They reuse those verified files to check inference with socket connections denied and
automatic PDF readiness followed by a notebook model switch. Version, manifest, file-hash,
setup-time and test reports upload as `search-platform-<runner>-<commit>` artifacts for 14 days;
model weights are excluded from the uploads.

CI enables `CORPI_RUN_MODEL_DOWNLOAD=1` for the native suite, adding the desktop download,
PDF-readiness and model-switch flow. Developers can enable the same explicit opt-in locally:

```powershell
$env:CORPI_RUN_MODEL_DOWNLOAD = "1"
pnpm --dir desktop run test:e2e:embedded
Remove-Item Env:CORPI_RUN_MODEL_DOWNLOAD
```

With that opt-in, each test has a 25-minute maximum and the suite launcher allows 30 minutes
for setup plus the existing viewer checks. Ordinary runs retain their three-minute per-test
and five-minute suite limits. The application enforces its own setup/preparation budgets.
Embedded-provider model selection uses a synthetic DOM change event; external providers use
their select control. Passing the embedded flow does not establish trusted physical input.

The original large-PDF search-readiness measurements in `desktop/e2e/specs/readiness.e2e.ts`
do not run in CI. They require an explicit `CORPI_READINESS_CASE`, `CORPI_READINESS_PDFS`
pointing to the original hash-pinned PDFs, and already installed models; the workflow sets
neither readiness variable. The retained measurements described above are not CI replays.
The near-25-MB raster viewer check below measures drawing, not the twenty-minute search
preparation limit. Original-PDF capacity remains an explicit manual release check; a
portable large-fixture replay is still outstanding. Retain the twenty-minute preparation,
10,000-chunk and 20,000,000-byte limits. Hosted functional checks do not prove the separate
1 GiB application-tree RAM/cache/no-swap requirement.

The suite checks painted content on distinct pages, navigation and zoom, a 24,999,190-byte
image-only PDF rendered under the 20-MP canvas cap, worker/canvas release after close and file
switching, and a visible renderer error that leaves the original saved. Blank-canvas and
stale-page controls check that the rendering assertions reject those failures. Controlled response
and animation-frame waits exercise Close during fetch/render and successful reopening. The suite
also checks modal keyboard focus, blocked background focus, and long-filename error wrapping.
External drivers exercise trusted Tab/Escape input. The embedded provider sends synthetic keys,
so its checks cover native modal focus, background inertness, return focus, and the cancel API;
they do not establish physical-key navigation or screen-reader announcements.
The launcher tests stop the real core during a held partial upload, reopen the same isolated data,
and check that recovery reports file reselection rather than claiming a saved complete PDF.

Each run writes screenshots, logs, fixture hashes, OS/webview/PDF.js versions, test results,
the test binary hash, and a shutdown result under `desktop/e2e-results/<provider>/`. CI uploads
these as `native-viewer-<runner>-<commit>` artifacts, including failed runs, for 14 days. A build
or unit-test pass alone doesn't establish native viewer compatibility; use the native run's
results and screenshots for that evidence.

Ordinary builds omit the optional WebDriver dependency and test capabilities. Only the E2E
runner passes `--features webdriver` and `tauri.webdriver.conf.json`; the default capability
file stays unchanged. Production builds must run without that feature or test configuration.

For test-only edits, `pnpm --dir desktop run test:e2e --reuse-build` reruns the external checks
against the previous test binary after verifying its hash. Add `--embedded` for that provider.
This option doesn't rebuild app changes; CI always builds a fresh binary.

## API contract

The core publishes its OpenAPI schema in `core/openapi.json`. The desktop build generates
TypeScript types from it (`pnpm generate:api`), and `just schema-check` fails when the
committed schema no longer matches the live core routes. Re-export and regenerate after
changing core endpoints.

## Communication security

The core binds to 127.0.0.1 on an OS-assigned port and requires a bearer token issued fresh
per launch. Requests must pass `Origin` and `Host` validation. The token authorizes
local API access and must not be shared or persisted. Provider spending is planned
functionality, not an endpoint available on this branch.

## License

Corpi's own source code is available under the [MIT license](LICENSE).
Third-party dependencies, models, fonts and other third-party assets retain their
own licenses and redistribution requirements; Corpi's MIT license does not replace
those terms.
