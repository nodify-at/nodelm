# Decisions

## D-001 — one typed Python application

Use a single `nodelm` package and CLI. Shell remains limited to bootstrap and remote glue. This
keeps schemas, status semantics, and deterministic artifacts consistent across phases.

## D-002 — Python 3.11 exact minor

The lock targets Python 3.11 because the current dataset/training stack publishes compatible
wheels there and the project can reproduce it across local and remote hosts. Upgrades require a
new lock and the full smoke lifecycle.

## D-003 — conservative license gate

Only MIT, Apache-2.0, BSD-2-Clause, and BSD-3-Clause enter V1. Unknown and copyleft rows are
audited, not discarded without trace.

## D-004 — immutable, content-connected evaluation split

Repository identity is the minimum split group. Declared mirrors plus exact or measured-near
task/reference-patch/generated-patch matches join one connected group. Exact or near overlap
with an explicitly selected public benchmark excludes the entire group from training and
private evaluation. The benchmark JSONL and strictly positive threshold are required inputs, so
the currently unfrozen benchmark input and threshold remain honest blockers rather than being
replaced by implicit defaults. Offline mirror detection remains incomplete and still requires
an additional audit before tuning.

## D-005 — trusted local and untrusted execution stay separate

Local subprocess controls protect reproducibility and reduce accidents. Untrusted repository
or model-authored code runs only in a preloaded, digest-pinned rootless Podman image. Its host
workspace bind is read-only, networking and implicit pulls are disabled, resources are bounded,
and a named container is force-removed in a `finally` path. Only bounded text patches reach the
apply step, and protected repository-test paths reject symlink components.

## D-006 — no default model without the source plan (superseded by D-012)

At bootstrap, the NodeLM plan was unavailable. Configurations therefore kept model fields null
and reported `UNVERIFIED`/`BLOCKED` rather than inventing replacements. D-012 records the
activation policy now that the plan is tracked.

## D-007 — disk-backed split and bounded audit summaries

Repository split artifacts are built through SQLite and streamed to immutable canonical JSON,
so registry-scale JSONL does not require several in-memory copies. Audit identity counts are
also disk-backed; percentile, duplicate-ID, and inline rejection examples are capped and
explicitly labeled, while reports point to a complete streamed rejection ledger. Split evidence
records task-table and benchmark digests plus typed exact, near, overlap, connected-group, and
exclusion counts without serializing raw task or patch text.

## D-008 — truthful trusted-local harness capabilities

The local backend consumes a validated, hashed config. It declares network access enabled
because it cannot enforce network isolation, refuses dependency installation, and records a
bounded/redacted command observation. Node.js 20+ is a required local verification tool.

## D-009 — remote verification is executable but remains evidence-gated

The remote doctor records structured GPU, VRAM, driver, CUDA, topology, NCCL, and RDMA/IB
evidence. `remote_verify.sh` binds infrastructure, training-lifecycle, and harness JSON to a
fresh validated run ID and invocation token before copying them locally. Without all five
plan-selected runtime inputs, including a digest-pinned sandbox image, it exits `BLOCKED`. With
them, it installs only the locked training extra before the strict infrastructure gate and
requires rootless Podman with the exact image already loaded; implicit pulls are forbidden. A
real tokenization/forward/backward/optimizer/checkpoint lifecycle, including restricted
optimizer-state reload and a resumed optimizer step, plus inference, the protected
model-authored patch fixture, and general harness fixture must all pass before success.

## D-010 — pilot artifacts retain training-visible behavior

Normalized and pilot samples retain the generated trajectory and model patch, never the task
gold patch. The task-metadata join stores only the four provenance fields needed to complete a
trace. Pilot selection requires a non-empty trajectory and writes a compact manifest alongside
a directly consumable immutable JSONL sample artifact.

## D-011 — only integrity-attested oracles may report resolution

The fixed smoke harness is bound to one exact `SolveContext` and an explicit set of approved
before/after source hashes. It can pass only after the expected failing baseline, exact source
transition, protected-tree integrity, and final tests are observed. A generic repository test
command has no equivalent integrity-attested oracle, so a zero exit remains `UNVERIFIED` rather
than proving task resolution or regression safety.

## D-012 — plan activation separates metadata from execution

The tracked plan settles the teacher and three candidate identities, but explicitly requires an
empirical student selection. Immutable public metadata may report `PASS`; model load, execution,
bake-off, student selection, and training retain separate statuses and remain `NOT RUN` until
their evidence exists. Training configurations stay non-runnable rather than preselecting the
plan's current lead candidate.

## D-013 — full dataset transfer waits for large storage

The dataset registry, validation, download guard, streaming materializer, offline snapshot-audit
and lineage contract, and runbook are prepared locally. Full snapshots are not downloaded in the
project workspace. Transfer and bulk materialization require explicit current-session
confirmation after the user provisions a GPU instance with larger storage and records a capacity
budget using `docs/DATA_DOWNLOAD_RUNBOOK.md`.

## D-014 — snapshot lineage content-binds local audit evidence

The guarded pinned download publishes an immutable transfer receipt that joins the exact source
and raw registry identity to the complete supported snapshot inventory. Filtered receipts are
recorded truthfully but cannot authorize `datasets audit-snapshot`. The aggregate identity covers
normalized sorted relative paths, each file's bytes and digest, and total bytes. Offline audit
copies the receipt-bound files into a private staged view, verifies the staged aggregate, and
parses only those private bytes so a source-path replace/read/restore race cannot split content
identity from logical rows. Legacy raw-file reports preserve the `nodelm.dataset-audit/v1` shape;
complete aggregate-snapshot reports use `nodelm.dataset-audit/v2` and declare the aggregate
identity scheme. Lineage binds the receipt, registry, snapshot, logical rows, canonical report,
and complete rejection ledger. Each immutable publication has one dependency recheck, and
lineage is published last as the completion marker. Private staging is a trusted-local boundary,
not protection from a hostile same-user process.

The lineage status mirrors the core snapshot audit and may retain truthful `FAIL` evidence. A
synthetic-fixture `PASS` proves only this contract. It does not complete Phase 0, attest an
untransferred source, measure tokenizer-based trajectory length, perform patch decontamination,
or establish public-evaluation overlap.

Execution note (2026-08-24): after explicit authorization and provisioning of persistent
large-volume storage, all three pinned snapshots were transferred and their receipt-bound core
audits and lineage manifests passed. This fulfills the deferred transfer/core-audit portion of
D-013; bulk materialization and the remaining Phase 0/decontamination gates remain `NOT RUN`.
The snapshots and complete raw evidence stay on external persistent storage, while Git tracks
their compact digest index in `artifacts/reports/FULL_DATASET_AUDIT.md`.

## D-015 — partition and task provenance are separate receipt-bound contracts

The sealed dataset registry remains unchanged because all three transfer receipts bind its exact
SHA-256. Open-SWE's actual snapshot has 11 leaf partitions and one hyphenated path that cannot be
represented truthfully by the registry's eight historical descriptive split labels. A separate
strict partition contract therefore binds the Open-SWE source/revision, sealed registry,
complete transfer receipt, complete snapshot, and every leaf's exact path, harness,
source-native generating-model label, upstream task family, and normalization status.

Partition materialization v2 selects exactly one contract leaf and verifies its complete file
identity set against the transfer receipt. Normalization consumes and revalidates that manifest,
contract, and receipt; command-line harness/model values are optional equality assertions only.
The materialization digest, partition, and upstream task family enter every sample's lineage.

Task sources cross the trace join only through a second immutable projection. It admits
allowlisted repository licenses and emits only instance ID, repository, base commit, normalized
license, canonical language, and pinned task-source identity. It physically excludes task text,
gold/reference/test patches, installation data, and open metadata. Duplicate IDs with conflicting
safe provenance, or with any rejectable provenance row, are wholly excluded while the original
rejection cause remains in the ledger. The trace join is required and compares canonical
repository, SPDX, language aliases, and case-insensitive commit identity.

Normalized-sample v1 retains a boolean `resolved` field. Source values `-1` and null are recorded
as `unknown_resolution` rejections rather than coerced to unresolved; a future tri-state schema
requires an explicit version and sample-identity migration. Only the seven Open-SWE leaves backed
by pinned SWE-rebench V2 task provenance can proceed. Four Scale-SWE leaves and all V2-PR joins
remain `BLOCKED`. Even eligible normalized artifacts remain non-training-ready until the separate
gold-exposure and decontamination gates pass.

## D-016 — derived provenance is accepted only after deterministic replay

Content-addressed manifests are completion markers, not independent trust roots. The code trust
root pins the exact partition contract and all three real snapshot-transfer receipts by source
revision. Materialization and task projection consume private identity-verified staged files;
normalization additionally requires both raw snapshot roots and deterministically replays both
derived inputs before joining them. Each terminal manifest is published last and revalidates its
inputs and already-published sibling artifacts at the publication boundary.

## D-017 — rollout conflicts and gold exposure fail closed before pilot construction

Trace uniqueness is keyed by source revision, exact partition leaf, task family, canonical
repository, instance ID, and rollout ID. A disk-backed projection admits one exact copy, preserves
distinct rollout IDs, and rejects every occurrence when one key maps to conflicting raw rows.
The raw-row digest enters normalized lineage so dropped source fields still affect sample identity.

Normalization deliberately retains `gold_exposure_audit: NOT RUN`. `build-pilot` requires a
separate reviewed and code-authorized `nodelm.gold-exposure-audit/v1` PASS artifact, complete
oracle-isolation attestation, zero structural findings, a frozen split bound to the same input,
and exact population counts. The training lifecycle rejects pilot manifests that omit this gate.

## D-018 — split and pilot manifests enter training only through reviewed digests

Repository-split construction reads normalized samples, task metadata, benchmark entries, and
optional aliases only from private identity-verified copies, then rechecks the original inputs at
immutable publication. Pilot construction accepts a split only when its exact digest is present
in the code trust root for the normalized artifact digest. A structurally valid or self-authored
split cannot relabel evaluation repositories as training repositories.

The one-step lifecycle likewise accepts only the exact reviewed pilot-manifest digest authorized
for the exact samples digest. It stages both files before parsing and rechecks config, samples,
pilot, and the complete protected evaluation-fixture identity at terminal report publication.
The fixture itself is a code-pinned regular-file tree: extra, missing, altered, symlink, and
special-file entries fail before model execution, and patch evaluation consumes only a private
identity-verified copy. The terminal report records the tree schema, digest, file count, and bytes.
The reviewed pilot may retain `UNVERIFIED` while the checked-in pilot policy awaits empirical
thresholds because this command verifies only one optimizer/checkpoint/resume lifecycle; scaled
training remains blocked until that policy is deliberately frozen and promoted. Empty
authorization maps mean no real split or pilot is approved by default.

## D-019 — unknown resolution recovery remains evidence-bound and non-admitting

Resolution transfer is keyed by the pinned trace-source revision, pinned task-source revision,
instance ID, and exact UTF-8 model-patch digest. A candidate is emitted only when the target is
still unknown and every known label for that target task-and-patch key agrees; an already-known
target is never overwritten. Label conflicts fail closed when their key occurs in the eligible
unknown target population. Conflicts unrelated to that target population cannot block the run.

Targets without unanimous exact-match evidence are grouped globally into one deterministic
evaluator request per task-and-patch key with sorted target references. The disk-backed
projection retains only the instance ID, exact model patch, partition/rollout references, and
projected-row digests needed for evidence binding. Trajectories and gold, reference, or test
content cannot cross into either the candidate sidecar or evaluator queue.

Recovery publication is immutable and records derivation separately from admission. A
successfully derived `nodelm.resolution-recovery/v1` manifest remains admission `BLOCKED`; it is
not permission to normalize recovered rows or train. The real derivation completed on 2026-08-25
and remains blocked; no recovered label may enter downstream data until a sandboxed harness
canary validates the transfer and evaluation path.

## D-020 — resolution canary separates private oracle material from admissible evidence

The recovery canary deterministically covers both transferred controls and deduplicated unknown
requests across observed languages, target partitions, and known labels. Its workset joins only
the selected model patch to the pinned task's base commit, image, test patch, expected tests, test
commands, and named parser; the gold solution patch is structurally absent. The workset stays
private outside Git, while its public manifest contains only counts, strata, revisions, and
content identities.

Evaluator code is fixed to `SWE-rebench/SWE-rebench-V2` revision
`c71902a8cf8d2b725f63d51f199f4d3e56f68d2d`, with separate SHA-256 checks for the parser and
evaluation script plus the parser's status constants. Preparation may clone that source and pull
only selected source image tags.
Before execution, each tag is converted to a matching immutable repository digest. Attempts use
only preloaded digests. The primary rootless-Podman backend disables network, drops capabilities,
sets no-new-privileges, mounts patch input read-only, applies explicit CPU/memory/PID/file/output/
time bounds, and verifies forced cleanup. Restricted hosts that prohibit nested namespaces use
converted OCI layouts bound by local manifest hashes, a fresh reflinked rootfs per attempt, chroot
plus a dedicated numeric UID, a network-denying seccomp filter, equivalent resource bounds, and
verified rootfs cleanup.

Every case first applies only the private test patch to reproduce the base failure, then applies
the exact model patch plus test patch in a fresh container. A canary case fails closed on missing
expected tests, output truncation, an invalid baseline, contradictory exit evidence, sandbox or
cleanup failure, or transferred-label disagreement. Raw logs are retained only in private atomic
per-case evidence; the public result retains hashes and aggregate test counts. Per-case evidence
is restartable, but a terminal execution manifest is admission `PASS` only when every selected
case passes and every control label agrees. That verdict validates the recovery mechanism; policy
for disposing the full 49,572-request queue remains a separate decision.

The pinned `parse_log_js_4` does not recognize Mocha's numbered failure lines. The canary may
conservatively add `FAILED` only when such a line exactly names an expected test missing from the
pinned parser result; existing parser evidence always wins. This preserves ambiguous duplicate
names while retaining the pinned parser as the primary oracle.

## D-021 — V1 quarantines unverified Qwen3.6 recovery and advances the labeled cohort

The terminal 12-case real-repository canary at commit
`3c9690abc7f676762f613b84897ca5cf0156cc4a` completed but failed its all-cases admission rule:
3 cases passed and 9 failed, so execution is `FAIL` and recovery admission remains `BLOCKED`.
Missing or ambiguous expected-test evidence and contradictory candidate exit evidence are not
reclassified after seeing results. No failed case may be replaced post hoc, and no recovered
Qwen3.6 label may enter normalization or training.

For V1, the three all-unknown Qwen3.6 leaves remain preserved as quarantined Tier D evidence.
Phase 0 advances only with these four existing complete-partition `PASS` normalization manifests:

- `openhands/minimax_m25/swe-rebench-v2`:
  `68f21f192c1af397837610ef6e4033fb04e9e2a38043f461a42d70ab6b47082e`;
- `openhands/qwen35_122b/swe-rebench-v2`:
  `339f4a2923eaa92b07155fb35604ce61c2358f554de09146226124ab01ca0996`;
- `sweagent/minimax_m25/swe-rebench-v2`:
  `2ed86030086ededc8c69c8aa8801f49c4034a7cbee2878f7311cf60512cc1c2e`;
- `sweagent/qwen35_122b/swe-rebench-v2`:
  `725ab61444546026c10d4e4d6745c324f0430063151650c905cfbc6b7d80b372`.

Together they contain 160,731 accepted rows. A cohort is a manifest of selected complete members,
not a claim that every source leaf passed. It binds each leaf manifest and artifact, a fixed
partition-name order, global sample-ID uniqueness, and the SHA-256 of their exact ordered byte
concatenation without publishing a second copy. The real cohort built at commit
`24f4eb75f16f6782fdfa85762d3a27cd7fdbef10` contains 4 members, 160,731 samples, and
51,063,015,261 bytes. Its manifest SHA-256 is
`10734b8e20d127bfe69df8c5ffd3c8540038cfa95ad2d2c230ea92fa7e8d2621`; its population SHA-256
is `e91207cdca52c6fd08d0fd672c482fb7072117856f380b0b951bbe403fa85269`. Gold-exposure,
decontamination, split, pilot-quality, and training authorization remain independent fail-closed
gates. The first independent pass through that boundary found zero structural gold-exposure
findings in all four leaves, but correctly left every overall audit `BLOCKED` because oracle
isolation is not attested. After review hardening, commit
`af476d1e85da133b623456d2a34f0ef12a25b857` replayed the real cohort and reproduced the exact
manifest and ordered-population identities, so the implementation changes did not alter the
selected population.

Recovery may be reconsidered only as a separately versioned, pre-registered compatibility-aware
experiment whose selection is label-blind and whose admitted population is restricted to the
same compatibility domain. The immutable blocked V1 artifacts remain evidence; they are not
rewritten or discarded.

## D-022 — oracle isolation requires raw-context review and two explicit trust promotions

The four selected Open-SWE v1.0 leaves may produce
`nodelm.oracle-isolation-attestation/v2` evidence only from their exact complete-partition raw
materialization and replay-verified normalized artifact. The review indexes every normalized
sample by its raw-row lineage, requires complete identity coverage, binds the complete task
provenance projection used to derive repository, license, base commit, and language, and
reproduces every training-visible trace value including resolution and derived patch metadata.
It permits the source reference patch only at the reviewed `metadata.reference_patch` boundary,
requires it for every reviewed row, and rejects it from every recorded model input (including tool
results, tool definitions, and unknown roles) or any other raw location. Findings are fixed-code,
content-free records.

The upstream portion is deliberately narrow: only the exact pinned v1.0 OpenHands/SWE-agent and
MiniMax/Qwen3.5 partitions are covered by the Open-SWE-Traces paper's post-hoc git-hacking review.
This is evidence about the complete recorded context plus reviewed upstream curation, not a claim
that NodeLM cryptographically reconstructed the producer's external model service.

A structural `PASS` attestation is still a candidate. `audit-gold-exposure` reports oracle
isolation `PASS` only when the attestation's exact SHA-256 is code-authorized for the exact
normalized artifact. Pilot construction then independently requires the resulting complete
gold-exposure audit digest to be code-authorized. Both authorization maps are empty until real
artifacts are run, inspected, and deliberately promoted; a self-authored schema-valid JSON cannot
cross either boundary.
