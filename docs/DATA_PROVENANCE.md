# Data provenance

Every normalized sample preserves source dataset and revision, repository and repository
license, base commit, issue/PR identifier, language, harness, generating model, rollout ID,
resolved status, the training-visible trajectory, the model-generated patch, patch metadata,
and ordered lineage. Each sample also has a deterministic provenance-derived `sample_id` that
feeds the repository split without a parallel hand-built identity file.

The normalization flow is:

1. Verify the registry record and immutable source revision. The guarded download publishes an
   immutable transfer receipt that joins that source and raw registry identity to the complete
   supported local snapshot inventory.
2. Audit the raw snapshot before filtering; retain schema, counts, bounded distribution
   samples, duplicate IDs, and a complete streamed license-rejection ledger. For a local
   complete snapshot, require its complete transfer receipt, copy the receipt-bound files into a
   private staged view, publish a `nodelm.dataset-audit/v2` report that declares the aggregate
   identity algorithm, and publish a source-level lineage manifest that binds the receipt,
   registry, snapshot, logical rows, report, and ledger. Legacy raw-file audits remain v1.
3. Bind each Open-SWE leaf to the checked-in 11-partition contract. That contract preserves the
   receipt-bound registry bytes while separately binding the exact snapshot path, harness,
   source-native generating-model label, upstream task family, and normalization eligibility.
   Partition materialization emits a v2 manifest whose selected files must match the complete
   sealed receipt inventory for that leaf; caller-supplied labels are assertions, not provenance.
   The source/revision trust root authorizes the exact partition-contract and snapshot-receipt
   digests. Transformations read only private copies that match those identities.
4. Project a pinned task snapshot into an immutable, license-safe artifact containing only
   instance ID, repository, base commit, normalized repository license, canonical language, and
   task-source identity. Task statements, reference patches, tests, installation metadata, and
   open metadata cannot be represented in this projection. Missing, unsafe, or conflicting rows
   enter a separate immutable rejection ledger. If any row with a usable instance ID is unsafe or
   conflicts, every row for that instance is excluded and the underlying cause remains recorded.
5. Normalize only against the matching partition materialization, partition contract, complete
   task projection, both transfer receipts, and both raw snapshot roots. Replay materialization
   and task projection from receipt-bound private staging and require exact derived identities;
   a self-authored manifest is not a trust root. The disk-backed required join fails closed when
   task provenance is absent and copies only the safe fields above. Each sample lineage includes
   the raw-row digest, materialization digest, partition, upstream source, task source/revision,
   and safe task artifact digest. A trace `resolved` value of `-1` or null is recorded as
   `unknown_resolution`; it is never coerced to unresolved in boolean normalized-sample v1.
6. Classify rollout identity in a temporary SQLite index scoped to source revision and exact
   partition leaf. Preserve distinct rollout IDs, admit only the first exact duplicate, reject
   all occurrences of a conflicting rollout key, and enforce accepted + rejected = input rows.
7. Recover unknown Qwen3.6 resolution labels only through an immutable sidecar. Exact transfer
   requires unanimous known evidence for the same pinned task source, trace source, instance, and
   UTF-8 model-patch digest. Everything else enters one deduplicated evaluator request per exact
   key. Neither sidecar may contain trajectories or gold/reference/test content, and its recovery
   manifest remains non-admitting. Before downstream use, a separate private canary workset joins
   selected cases to test patches and expected tests, explicitly drops the gold solution patch,
   pins evaluator source and container digests, reproduces failing baselines, and validates known
   transfer labels in offline rootless containers or fresh seccomp/chroot OCI rootfs clones.
8. Index task statements plus reference and generated patches only inside the disk-backed split
   gate. Exact and measured near matches form connected repository groups alongside declared
   mirrors/forks.
9. Compare both task and patch fingerprints with an explicitly selected public benchmark.
   Exclude every connected group containing a benchmark-overlap sample, then deterministically
   assign the remaining groups to training or private evaluation. Build the split only from
   private identity-verified copies of every input and require its reviewed digest to be
   authorized for the exact normalized artifact before pilot construction.
10. Derive a pilot manifest and companion training JSONL only when the frozen split, normalization
   manifest, and reviewed/code-authorized `PASS` gold-exposure audit bind the same normalized bytes
   and complete row count. Re-scan every training-visible trajectory, retain oracle-isolation
   attestation identity, and never move repositories across the split. The one-step lifecycle
   also requires a reviewed pilot-manifest digest authorized for the exact samples digest and
   consumes private identity-verified copies of both artifacts.

Audit percentile, duplicate-ID, and rejection examples are capped and explicitly labeled when
truncated. Complete counts and the rejection ledger remain available while dataset-cardinality
identity state lives in a temporary SQLite index. Repository-split CLI construction likewise
uses a disk-backed index and streams canonical JSON; its repository-list reader does not load
per-sample assignments. The split manifest contains source digests and typed exact/near
comparison, match, overlap, group, and exclusion counts, but no raw task or patch text. The
public benchmark and strictly positive near-duplicate threshold are mandatory CLI inputs because
the plan leaves both the benchmark inputs and measured threshold unfrozen, so they cannot be
guessed. The convenience in-memory Python API has an explicit row limit.

The receipt-bound core snapshot audits and source-level lineage manifests now report `PASS` for
all three pinned real sources; see `artifacts/reports/FULL_DATASET_AUDIT.md`. That status covers
only complete-snapshot identity, parsing, counts, license-gate ledgers, and bound lineage. The
current `trajectory_lengths` report measures trajectory steps, not tokenizer tokens. Unique
issue/PR counts, harness and generating-model distributions, tokenizer-based lengths,
exact/near patch duplication, and public-evaluation overlap remain explicitly `NOT RUN`.
Per-example lineage is added later by strict normalized samples and remains distinct from this
source-level artifact binding.

The sealed registry remains byte-for-byte unchanged at SHA-256
`f92315a70a0c75ec909d83f4cb639b3a320f62526069f11ca87f0fe1d891637f` because all three
transfer receipts bind it. The separate Open-SWE partition contract corrects the observed
hyphenated `deepseek-v4-flash` snapshot path without rewriting that historical evidence. Seven
SWE-rebench-V2-backed leaves are eligible for safe projection and normalization. Four Scale-SWE
leaves remain `BLOCKED` until a pinned task source and verified join are available. V2-PRs also
remains blocked: it has no verified Open-SWE leaf relationship and no top-level language.

A normalized manifest deliberately records gold-exposure auditing as `NOT RUN` until a separate
oracle-isolated comparison is executed. Partition-safe normalization alone is not permission to
train. D-021's four selected complete-partition manifests are now bound by a real copy-free cohort
manifest. It contains 4 members, 160,731 globally unique samples, and 51,063,015,261 ordered
population bytes. The cohort-manifest SHA-256 is
`10734b8e20d127bfe69df8c5ffd3c8540038cfa95ad2d2c230ea92fa7e8d2621`; the exact ordered
population SHA-256 is `e91207cdca52c6fd08d0fd672c482fb7072117856f380b0b951bbe403fa85269`.
The cohort manifest itself is not a gold, split, pilot, or training authorization.

Execution note (2026-08-25): the complete pinned V2 task projection and a 1,000-row
OpenHands/MiniMax normalization canary passed both deterministic raw replays at commit
`7366ec06e8c2bb098afc02382e38b5b57f6e9b5d`. The canary admitted 783 rows and retained 217
unknown-resolution rejections. This is canary evidence only; full-partition normalization and
the separate gold-exposure/decontamination gates were still `NOT RUN` at that canary boundary.
Later on 2026-08-25, all seven eligible full-partition leaves reached terminal evidence: four
MiniMax/Qwen3.5 leaves are `PASS`, while three all-unknown Qwen3.6 leaves are truthful `FAIL`.
The subsequent real recovery derivation at commit
`74c9b505eb1a608431ae3a18a3fca5d084f2ae3b` completed with zero conflicts and published
1,804 unique exact-transfer keys plus 49,572 unique evaluator requests. Its admission remains
`BLOCKED`: the terminal real-repository canary completed 3 `PASS` / 9 `FAIL`. D-021 therefore
quarantines all recovered Qwen3.6 labels and advances V1 only with the four labeled
complete-partition `PASS` leaves. Those leaves were bound at commit
`24f4eb75f16f6782fdfa85762d3a27cd7fdbef10` by the 4-member, 160,731-sample cohort described
above. The hardened builder at commit `af476d1e85da133b623456d2a34f0ef12a25b857` reproduced its
manifest and ordered-population identities exactly against the real 51,063,015,261-byte
population. All four independent structural gold scans are `PASS` with zero findings. Oracle
isolation and each overall gold-exposure audit remain `BLOCKED`; decontamination remains
`NOT RUN`. Exact hashes are in `docs/EXPERIMENTS.md`.

`SolveContext` is a separate type from evaluation material. It intentionally rejects unknown
fields so a gold/reference patch cannot be serialized into teacher or student solving input.
