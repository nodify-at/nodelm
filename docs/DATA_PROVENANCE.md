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
3. Materialize pinned Parquet/JSONL files without loading the snapshot into memory, then
   normalize only records that satisfy the required provenance contract. When Open-SWE traces
   lack base metadata, the disk-backed join copies only repository, base commit, license, and
   language from the task source; it never copies task text or a gold patch.
4. Apply the repository-license gate, count every rejection, retain bounded examples in the
   summary, and write every rejected row to the sibling immutable JSONL ledger.
5. Index task statements plus reference and generated patches only inside the disk-backed split
   gate. Exact and measured near matches form connected repository groups alongside declared
   mirrors/forks.
6. Compare both task and patch fingerprints with an explicitly selected public benchmark.
   Exclude every connected group containing a benchmark-overlap sample, then deterministically
   assign the remaining groups to training or private evaluation.
7. Derive a pilot manifest and companion training JSONL from the frozen split; never move
   repositories across it. Pilot rows must carry a non-empty trajectory.

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

`SolveContext` is a separate type from evaluation material. It intentionally rejects unknown
fields so a gold/reference patch cannot be serialized into teacher or student solving input.
