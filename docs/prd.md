# Corpi v1 Product Requirements

Corpi v1 is a single-user desktop app for asking questions about imported material and checking
answers against cited evidence. This document defines the v1 experience and requirements;
[architecture.md](architecture.md) defines the system responsibilities and technical rules.

| Decision status | Meaning |
| --- | --- |
| **Decided** | An accepted requirement or choice. |
| **Needs Decision** | A product or implementation choice that remains open. |
| **Partially Decided** | Some behavior is settled; the listed details remain open. |
| **Deferred** | Postponed to a named milestone or excluded from initial v1, as stated. |

## Contents

- [Product overview](#product-overview)
  - [V1 scope and main flows](#v1-scope-and-main-flows)
- [Sources and answers](#sources-and-answers)
  - [Sources and imports](#sources-and-imports)
  - [Grounded answers](#grounded-answers)
    - [Answer coverage](#answer-coverage)
    - [Individual claim support](#individual-claim-support)
  - [Citation verification](#citation-verification)
- [Saved work and privacy](#saved-work-and-privacy)
  - [Notes, artifacts, and history](#notes-artifacts-and-history)
  - [Local data and provider use](#local-data-and-provider-use)
- [Failures and recovery](#failures-and-recovery)
  - [Uncertain paid requests](#uncertain-paid-requests)
- [Scope boundaries and acceptance](#scope-boundaries-and-acceptance)
  - [Deferred work and open decisions](#deferred-work-and-open-decisions)
  - [Acceptance expectations](#acceptance-expectations)

## Product overview

Users organize their research in notebooks, saved workspaces that keep sources, conversations,
notes, and generated work together. Sources come from supported files, pasted text, web pages,
or YouTube transcripts that the user imports. The user chooses which of those sources a question
should use.

Corpi answers by linking factual claims to passages saved from the selected sources. These
citations let the user open the supporting text and check the answer. Useful answers can be saved
as notes, while artifacts provide editable, reusable outputs with citations. The exact artifact
types remain deferred.

Notebooks and saved work stay on the user's device between sessions. Producing answers uses an
external model provider configured by the user, so generation and claim checks may send the
question and required selected passages to that service. This is the boundary of Corpi's
local-first experience.

For example, after importing a PDF and waiting for it to become ready, a user selects it and asks
a question about its contents. If Corpi finds supporting evidence, the user can open a citation
to inspect the passage behind the answer, then save that answer as a note. If the PDF does not
contain enough evidence, Corpi must say so.

### V1 scope and main flows

The minimum viable product (MVP) supports the flows below, saved conversations and notes, provider
configuration, and visible processing and recovery status. Its text-based questions and answers
(Q&A) must use evidence from selected sources. Artifacts are part of v1, with settled behavior and
remaining choices listed in the [artifact requirements](#notes-artifacts-and-history).

- **Build a notebook.** Open or create a notebook, import supported material, and see progress,
  usable-source status, warnings, or reasons for failure.
- **Ask and follow up.** Select ready sources in the notebook and receive an answer supported by
  them, or a clear explanation that they only partly answer the question or lack the evidence.
  Follow-ups use only selected sources as evidence.
- **Verify a claim.** Open a citation to inspect the exact supporting page or passage in the saved
  copy used by the answer. Transcript citations identify the corresponding time location.
- **Keep or reuse an output.** Save an answer as a note or create an editable cited artifact under
  the [artifact rules](#notes-artifacts-and-history). Saved outputs retain their history and source references.
- **Maintain sources and results.** Retry, cancel, or replace a source where applicable, and request
  web-source refreshes or artifact regeneration. Earlier work stays tied to its saved versions.

## Sources and answers

A question can use a source only after import processing has made it ready. The requirements below
follow that progression from supported inputs to grounded answers and the citations users inspect.

### Sources and imports

Corpi reads the following formats as text. The import rules cover extraction, progress, and the
limits or readability problems that can prevent a source from becoming usable.

#### Supported formats

| Input | What v1 supports |
| --- | --- |
| PDF | Uploaded PDFs or eligible PDF URLs. Support text stored directly in the file, fully scanned pages of printed text, and mixed documents. |
| DOCX | Read document text and keep headings, lists, and tables. |
| TXT/Markdown | Read text and keep its existing structure. |
| Pasted text | Save the supplied text as a notebook source. |
| Public HTTPS pages and direct PDF URLs | Import user-provided public content on standard HTTPS port 443, subject to access and processing limits. Keep the imported page or PDF as a saved source. |
| YouTube | Read available transcripts only. Keep timestamps, language, and caption details. Do not interpret video frames or transcribe audio. |

Academic papers follow the same format rules; Corpi does not certify that a source is peer reviewed.

Reading printed text is distinct from interpreting visual content. Corpi can read printed text
from PDF page images through optical character recognition (OCR). Embedded images, charts, and
diagrams retain their locations and placeholders, but contribute no interpreted evidence in v1.

#### Import progress and partial sources

Imports must keep the desktop responsive while showing and saving progress, allowing work to
resume after a crash or restart. Offer cancellation, retry, and source replacement where applicable.
A source becomes ready for questions only after all required reading and search preparation has
succeeded, including work on the confirmed usable portion of a partial source. Cancelled, deleted,
replaced, failed, or superseded work must never later make the source available again.

Corpi checks PDFs page by page, using OCR where a page appears to contain printed text but extraction
is missing, garbled, or incomplete. Keep usable text extracted directly from the file and check OCR
results before accepting them. The extraction result determines what the user can use:

| PDF extraction result | User-visible outcome |
| --- | --- |
| Required text is readable | Make the source available after all required processing and search preparation succeeds. |
| Some pages are unreadable, but usable text remains | Identify the unreadable pages and exclude them from answer evidence. Require confirmation before enabling the usable portion. |
| No usable text remains | Show a clear import failure. |

Keep the original PDF available for inspection in every case, including partial and complete
extraction failure, so the user can still check the source.

PDF imports start with configurable limits of 25 MB and 200 pages, plus limits on page processing
and OCR time. Every supported input needs configurable size, length/duration, and processing-time
limits, with a clear explanation when those limits are exceeded. The [PDF/OCR rules](architecture.md#check-pdf-pages-and-handle-partial-imports)
give the detailed defaults.

#### Web imports and refreshes

Corpi saves content from the URL the user supplies rather than searching for alternatives. Private
or internal URLs and nonstandard HTTPS ports are ineligible, and Corpi must never bypass paywalls,
sign-ins, CAPTCHAs (bot checks), robots.txt (site rules for automated fetching), or other access controls.

If a page is inaccessible, explain why when that information is available and ask for content the
user can legally provide. Adding a partial page or teaser, such as an article preview, requires
the user's confirmation.

Refresh web sources only on request, creating a new source version only when the content changes.
Each version is a saved revision, so an earlier answer keeps the evidence it used even after a
refresh saves newer content.

Corpi may use a temporary local copy, or cache, of eligible public content for 15 minutes, subject
to site restrictions. Exclude signed URLs, which carry an access signature, and URLs specific to
a user from that cache.

### Grounded answers

Saved source passages are the evidence for an answer. They must come from the usable source
versions selected in the current notebook, and every factual claim needs checked citations. The
model must not fill gaps with outside knowledge; notes and other generated workspace content
cannot be selected or cited as separate Q&A evidence.

Chat history can help interpret a follow-up question, but summaries of older messages must never
replace source evidence. Each answer must include supporting claims and citations, caveats, and
any source conflicts. It must also report two distinct assessments: how much of the question the
sources answer, and how well they support each claim.

#### Answer coverage

Coverage describes how much of the question can be answered from the selected sources:

| Coverage | Meaning |
| --- | --- |
| Full | The sources provide an answer to the whole question. |
| Partial | They provide an answer to only part of the question. |
| No answer | The sources do not answer the question, even in part. |

Coverage is not a claim-support label. For example, sources may give a date but not explain the
reason for it. An answer to "When and why?" can have partial coverage while its date claim is
fully supported.

#### Individual claim support

For each factual claim, describe how well its cited passages support it:

| Label | Meaning |
| --- | --- |
| Fully supported | The cited passages support the whole claim. |
| Partially supported | They support only part of the claim. |
| Unsupported | They do not provide enough evidence for the claim. |
| Conflicting | Relevant sources give incompatible accounts. |

Corpi controls how answers appear and must show disagreements between sources. It handles failed
checks with separate, bounded allowances:

| Failed check | Allowance | If it still fails |
| --- | --- | --- |
| Citation or claim-support check | One retry | Remove unsupported claims, narrow partial claims to what the evidence supports, or report insufficient evidence. |
| Response format | One repair attempt | Show an error. |

#### Request limits and interruptions

A model can handle only a limited amount of text in one request, its context limit. If a request
still will not fit after Corpi selects supporting passages and reduces chat history, explain the
limit rather than silently cutting more text.

Interactive questions must not wait behind long-running imports or artifact creation. If a question
is interrupted, mark it as interrupted rather than complete.

### Citation verification

Because a valid link alone does not prove support for a claim, citation verification has two parts.
Corpi first checks that the citation points to an existing passage in the exact saved source version
used by the answer, then makes a separate model request to assess whether that passage supports
the factual claim.

Opening a citation must show the supporting page or passage in context, with transcript timestamps
identifying the corresponding moment. Citations target fixed saved passages, not the replaceable
text groups used for search (retrieval chunks), so reorganizing search data must not change what
an old citation opens.

Earlier citations also keep their saved passage when a source changes, subject to the local deletion
policy. The architecture's [passage and chunk rules](architecture.md#citation-passages-and-retrieval-chunks)
define the identifiers and links that preserve this behavior.

Support labels describe the evidence rather than whether a source is correct, current, or peer
reviewed. Since extraction and support checks can also make mistakes, show warnings and let users
inspect the originals to check the evidence themselves.

## Saved work and privacy

Saved work has two related requirements: preserve what was created, and make clear which data
stays on the device or goes to the configured provider. The history and privacy rules below cover both.

### Notes, artifacts, and history

Notes and artifacts give users different ways to keep their work:

| Output | Decided v1 behavior |
| --- | --- |
| Notes | Save answers as workspace content that remains available between sessions, under the [Q&A evidence rules](#grounded-answers). |
| Artifacts | Create editable, reusable outputs with citations. Corpi controls their structure and layout through templates. |

For each artifact, save the finished content with the source versions, template version, and
generation settings used to create it. Edits and regenerations must create new versions while
preserving the originals. Saved answers and artifacts keep their original content and display
format for their saved template version, even when templates change; never silently migrate or
regenerate them.

When a newer version of a referenced source is saved, leave the artifact as it was, mark that it
uses an older source version, and notify the user. Regeneration happens only on request and creates
a new artifact version while keeping the original.

Artifact creation belongs to v1, but its exact types, source coverage, and evidence-selection
budgets await the artifact implementation milestone. Evidence and citation rules for note-based
artifact inputs also remain unresolved. The [decision table](#open-product-decisions)
records these statuses; the details must be chosen before the affected behavior is built.

### Local data and provider use

Store notebooks, source files and saved copies, extracted content, search data, conversations, notes,
artifacts, and settings on the user's device. Corpi also creates embeddings, numerical
representations of text for meaning-based search, locally by default. Users therefore need no
separate provider credential for indexing, and source passages need not leave the device merely
to prepare them for search.

Generation uses a provider and model chosen by the user, with credentials such as an access key
that authorizes requests under their provider account. Producing answers and checking their claims
may send the question and required selected passages to that provider. Store credentials securely
on the device and allow only authorized model operations to use them; imported content must not
use credentials or authorize app actions, and web fetching must not access credentials or browser
login sessions.

The user's provider charges for external generation. When it supplies enough information, track
request usage and token counts locally. Tokens measure pieces of model input and output rather
than a fixed number of words. Invalid or expired credentials, exhausted credit/quota, rate limits,
and other provider restrictions must leave existing local work accessible. Initial v1 has no Corpi
subscription quotas or account upgrade/reset flow.

Ordinary deletions keep Corpi-managed data in local trash for 30 days before permanently deleting
related records, files, search data, caches, and affected derived content. Cancel related work so
deleted content cannot reappear. This policy covers Corpi-managed data only: copies in operating-system
backups, user exports, cloud-sync folders, or other locations remain outside its control.

Corpi operates no cloud backups. Local backup/export/restore behavior remains an
[open product decision](#open-product-decisions).

## Failures and recovery

An unusable source, an incomplete answer, and a rejected provider request need different responses.
The table below distinguishes those outcomes; uncertain paid requests require the additional
handling described afterward.

| Situation | What Corpi must show or do |
| --- | --- |
| Input cannot be read or fetched, or exceeds a limit | Explain the failure and identify affected pages/portions where applicable. Exclude failed content from evidence; keep the original PDF after extraction failure. |
| Only part of a source or web page is usable | Warn and require confirmation before enabling the usable part. Exclude unreadable content from answers. |
| Evidence is missing or sources disagree | Give a supported partial answer, show the disagreement, or report insufficient evidence. |
| Processing or chat is interrupted | Save progress for resumption; offer retry, cancel, or replace where applicable. Mark unfinished chat as interrupted, not complete. |
| A paid request has an unknown outcome | Establish the outcome or flag it for review under the [paid-request rules](#uncertain-paid-requests). |
| A provider restriction or context limit blocks a request | Stop the affected action with a clear error. Keep existing local sources and saved work accessible. |

### Uncertain paid requests

A lost reply does not prove that the provider failed to process or charge for a request. Corpi
must therefore establish the outcome or flag it for review. Automatic retries are allowed only
when the provider supports repeating the same request without duplicate effects, called idempotency.
Retries must never duplicate completed outputs.

## Scope boundaries and acceptance

The scope boundary separates capabilities excluded from initial v1 from choices still needed for
planned v1 work. Acceptance scenarios then provide a way to check the decided behavior.

### Deferred work and open decisions

Artifact details await their implementation milestone, while the capabilities listed below are
outside initial v1. The distinction keeps deferred design details from being mistaken for a
deferral of artifact creation itself.

#### Deferred beyond initial v1

- Finding sources from a topic, automatically crawling links, or searching the live web during
  answers. Automatic academic-source discovery and peer-review verification/filtering are also deferred.
- Audio input/output and interpreting charts, diagrams, photographs, or video frames.
- Importing PPTX, XLSX, standalone images, or handwriting.
- Selecting or citing notes as Q&A evidence.
- Corpi-hosted accounts, collaboration, or a sharing service.
- Arbitrary user-supplied embedding models, additional remote embedding backends, and optional
  user-configured usage/spend safeguards.

#### Open product decisions

| Area | Status | What remains open |
| --- | --- | --- |
| Overall artifact requirements | **Partially Decided** | Saved-output behavior is settled in [notes, artifacts, and history](#notes-artifacts-and-history); the remaining details are listed below. |
| Initial artifact types and evidence selection | **Deferred** | Decide the types, source coverage, and evidence-selection strategy and budgets at the artifact milestone. Rules for evidence/citations from note inputs remain unspecified. |
| Generation provider/model and claim-support checker | **Needs Decision** | No vendor/model is chosen. A supported local-generation alternative remains optional; its implementation and support commitment are unresolved. |
| Local backup/export/restore | **Needs Decision** | Export formats and recovery workflow. |

Other implementation choices are listed in [architecture.md](architecture.md).

### Acceptance expectations

Use these scenarios to check v1 behavior.

| Scenario | Expected result |
| --- | --- |
| Import and select evidence | Sources become usable only after successful processing. Questions use only selected, ready source versions in the current notebook. |
| Import scanned, mixed, or damaged PDFs | Read usable printed text and identify unreadable pages. Require confirmation for a partial source, fail clearly if no usable text remains, and keep the original for inspection. |
| Ask questions with full, missing, or conflicting evidence | Hand-checked questions produce inspectable citations, answer coverage, and individual claim-support labels. Distinguish a partially answered question from a partially supported claim. Show disagreements and follow the retry/fallback rules instead of retaining unsupported factual claims. |
| Revisit work after sources, search data, or templates change | Earlier answers and citations keep their original evidence and presentation. Artifact updates create explicit new versions and show when older sources were used. Notes do not become Q&A evidence. |
| Interrupt, retry, cancel, or delete work | Resume processing without duplicating completed outputs. Establish uncertain paid-call outcomes or flag them for review. Cancelled or deleted work cannot later make a source ready or recreate it. Follow the 30-day local deletion policy. |
| Encounter provider errors or limits | Stop only the affected model action. Keep local research accessible and explain the problem when provider information is available. |
| Check privacy boundaries | Keep saved research on the device and prepare search data locally by default. Only authorized model operations may use generation credentials and send required context. Source content and web fetching cannot obtain those credentials. |
