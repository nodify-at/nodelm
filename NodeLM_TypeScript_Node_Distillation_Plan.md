# NodeLM — TypeScript / Node.js Specialist SWE Model Plan

**Status:** Research-backed working plan  
**Prepared:** 2026-08-23  
**Primary goal:** Distill strong repository-level software-engineering behavior into a smaller open model specialized for TypeScript, JavaScript, Node.js, and the surrounding backend/tooling ecosystem.

> **Grounding rule for this project:** every external model/dataset claim below is either linked to a primary source or marked as an assumption/experiment. We do not treat vendor benchmark scores as directly comparable unless the harness and evaluation protocol match.

---

## 0. Executive decision

We should **not** attempt a naive “large model → small model weight compression” project.

We should build a **software-engineering behavior distillation pipeline**:

1. Start from a strong open student model.
2. Train it on high-quality repository-level agent trajectories.
3. Generate additional TypeScript/Node-specific trajectories using a stronger teacher.
4. Verify every generated task with executable tests/tooling.
5. Fine-tune in stages:
   - SFT / behavior cloning
   - preference optimization
   - execution-based RL only after SFT is proven
6. Evaluate on:
   - public multilingual SWE benchmarks for sanity
   - a **private, repository-disjoint NodeLM-Bench** as the real product metric
7. Optimize final inference across multiple validated deployment targets, including 1× DGX Spark, 2× DGX Spark, RTX PRO 5000 72 GB, RTX PRO 6000 96 GB, and datacenter GPUs when useful.

### Recommended V1 path

**Teacher**
- `deepseek-ai/DeepSeek-V4-Flash-0731`

**Student candidates — do not choose by benchmark headline alone**
1. `Qwen/Qwen3.6-27B` — current lead candidate for the deployment target
2. `Qwen/Qwen3.5-35B-A3B` — sparse/MoE candidate
3. `Qwen/Qwen3-Coder-Next` — code-specialized but much larger stored model

**Existing training data backbone**
- `nvidia/Open-SWE-Traces`
- `nebius/SWE-rebench-V2`
- `nebius/SWE-rebench-V2-PRs`

**Project-specific data**
- DeepSeek-generated, execution-verified TS/JS/Node trajectories
- curated package/runtime/framework migration tasks
- failure/recovery trajectories and hard negatives

---

# 1. What is verified today

## 1.1 DeepSeek V4 Flash 0731 as teacher

Verified from the official Hugging Face model card:

- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- License: MIT
- Total model size: 304B parameters
- Repository artifact is approximately 167 GB
- Official model-card agent results include:
  - Terminal Bench 2.1: **82.7**
  - NL2Repo: **54.2**
  - DeepSWE: **54.4**
  - DSBench-FullStack: **68.7**
- DeepSeek states its code-agent evaluations use its own minimal DeepSeek Harness, `max` reasoning effort, `temperature=1.0`, `top_p=0.95`.

**Important limitation:** those scores are not automatically reproducible in our harness. We use DeepSeek V4 as a **teacher candidate**, not as an unquestioned ground-truth oracle.

Source:
- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731

---

## 1.2 Open-SWE-Traces is the strongest existing bootstrap dataset found

Verified from NVIDIA's dataset card and paper:

- Dataset: `nvidia/Open-SWE-Traces`
- License: CC BY 4.0
- Published paper reports **207,489 agent trajectories**
- Trajectories are generated from real SWE tasks using OpenHands and SWE-agent
- Source tasks are drawn from permissively licensed repositories
- Original published language distribution includes:
  - TypeScript: **36,897 trajectories**
  - JavaScript: **29,360 trajectories**
  - Combined TS + JS: **66,257 trajectories**
- The dataset includes:
  - issue/task
  - full agent trajectory
  - tool interactions
  - resolution status
  - generated patch
  - reference patch metadata
  - repository/license metadata
- The Open-SWE-Traces paper reports that fine-tuning Qwen3-30B-A3B-series students produced a best score of:
  - SWE-bench Verified: **61.7%**
  - SWE-bench Multilingual: **57.1%**
  - SWE-bench Pro: **36.8%**

**Important current-state note:** the Hugging Face repository now exposes additional Qwen3.6-generated splits in its config, while the card still reports the original 207,489-row figure. Therefore Phase 0 must **pin a dataset commit and recount the current snapshot**. We will not hard-code row counts from the paper into the training pipeline.

Sources:
- https://huggingface.co/datasets/nvidia/Open-SWE-Traces
- https://arxiv.org/abs/2606.16038

---

## 1.3 SWE-rebench V2 gives us executable real-world tasks

Verified from the official Nebius dataset card:

- Dataset: `nebius/SWE-rebench-V2`
- License: CC BY 4.0 for the dataset
- **32,079 tasks**
- Languages include TypeScript and JavaScript
- Per-row fields include:
  - `repo`
  - `base_commit`
  - `problem_statement`
  - gold `patch`
  - `test_patch`
  - `FAIL_TO_PASS`
  - `PASS_TO_PASS`
  - Docker image/environment information
  - repository `license`
- Individual repository licenses still apply and must be respected.

This dataset is especially valuable for:
- evaluation environments
- teacher trajectory generation
- execution-based reward
- held-out validation tasks

Source:
- https://huggingface.co/datasets/nebius/SWE-rebench-V2

---

## 1.4 SWE-rebench V2 PRs is a larger raw task pool

Verified on Hugging Face:

- Dataset: `nebius/SWE-rebench-V2-PRs`
- Current repository reports approximately **126k PR examples**
- License: CC BY 4.0 at dataset level
- We must still filter by source repository license.

This is a raw teacher-task source, **not automatically training-ready data**.

Source:
- https://huggingface.co/datasets/nebius/SWE-rebench-V2-PRs

---

# 2. Correction to an earlier idea

`Qwen3.8-27B` was mentioned earlier, but I could not verify that model from an official Qwen/Hugging Face source.

**It is removed from this plan.**

The verified current candidates are below.

---

# 3. Student model selection

We should not choose the student until we run the same local harness on the same TS/JS validation tasks.

## Candidate A — Qwen3.6-27B

Verified official properties:

- `Qwen/Qwen3.6-27B`
- Apache 2.0
- 27B dense language model
- native context: **262,144**
- model card says extensible to ~1,010,000 tokens
- Hugging Face repository size: approximately **55.6 GB**
- official Qwen model-card results:
  - SWE-bench Verified: **77.2**
  - SWE-bench Pro: **53.5**
  - SWE-bench Multilingual: **71.3**
  - Terminal-Bench 2.0: **59.3**
  - NL2Repo: **36.2**

### Why it is currently the lead candidate

- Smallest strong dense candidate.
- BF16 repository size is already within the rough capacity of a 72 GB GPU, though **usable KV-cache/context headroom must be measured**, not assumed.
- It should also be straightforward to serve on DGX Spark-class 128 GB unified-memory systems, with more context/KV headroom than a 72 GB card.
- Dense models are straightforward to fine-tune and debug relative to large MoE students.
- Strong current agentic-code baseline.

### Caveat

The model is multimodal and the official benchmark uses Qwen's internal agent scaffold. Scores should not be compared directly to DeepSeek/Claude scores from different harnesses.

Source:
- https://huggingface.co/Qwen/Qwen3.6-27B

---

## Candidate B — Qwen3.5-35B-A3B

Verified official properties:

- `Qwen/Qwen3.5-35B-A3B`
- Apache 2.0
- 35B total parameters
- ~3B activated parameters per token
- native context: **262,144**
- officially described as extensible up to ~1,010,000 tokens
- Hugging Face repository size: approximately **71.9 GB**
- official SWE-bench Verified result: **69.2**

### Why test it

- Sparse activation can give excellent inference efficiency.
- Strong long-context model.
- A model family very close to what Open-SWE-Traces has already been used to fine-tune.

### Caveat

- The full BF16 artifact essentially consumes the entire nominal 72 GB GPU capacity; production would likely require FP8/quantization or a larger GPU.
- MoE fine-tuning is operationally more complex than a dense 27B student.

Source:
- https://huggingface.co/Qwen/Qwen3.5-35B-A3B

---

## Candidate C — Qwen3-Coder-Next

Verified official properties:

- `Qwen/Qwen3-Coder-Next`
- Apache 2.0
- **80B total / 3B activated**
- native context: **262,144**
- specialized for coding agents
- official HF evaluation metadata reports **70.6 SWE-bench Verified**
- BF16 repository is approximately **159 GB**

### Why test it

- Explicitly built for coding agents.
- Tool use / recovery / long-horizon coding is part of the design target.
- Sparse active compute.

### Caveat

- 159 GB BF16 stored model makes single-72GB deployment impossible without aggressive lower precision.
- Fine-tuning has greater storage/memory complexity than Qwen3.6-27B.
- The model is non-thinking by design; that is not necessarily bad, but it changes how we should format distilled trajectories.

Source:
- https://huggingface.co/Qwen/Qwen3-Coder-Next

---

## Student selection gate

Before serious training, run all three with the **same**:

- harness
- tool definitions
- token budget
- TS/JS task subset
- retry budget
- temperature/top-p appropriate to each model
- execution environment

### Pilot scorecard

| Metric | Weight |
|---|---:|
| TS/JS task resolution | 35% |
| hidden/regression test success | 20% |
| tool-call correctness | 10% |
| unnecessary diff rate | 10% |
| recovery after failed test | 10% |
| latency / tokens per solved task | 10% |
| final deployment feasibility | 5% |

**Decision rule:** choose the student that maximizes end-to-end engineering success, not the model with the highest public benchmark headline.

---

# 4. Benchmark reality: SWE-bench Verified is not our target

SWE-bench Verified contains **500 tasks from Python projects**.

Therefore:
- it is useful as a general sanity check
- it is **not** a meaningful product KPI for a TypeScript/Node specialist

SWE-bench Multilingual contains:
- 300 tasks
- 42 repositories
- 9 languages
- **43 JavaScript/TypeScript tasks**

So our public benchmark stack should be:

1. SWE-bench Multilingual — JS/TS subset
2. SWE-rebench-V2 — held-out TS/JS repository subset
3. private NodeLM-Bench — primary metric

Sources:
- https://www.swebench.com/multilingual.html
- https://www.swebench.com/SWE-bench/faq/

---

# 5. The actual product target

The goal is **not** “best at writing TypeScript snippets.”

The model should behave like a senior backend/full-stack engineer in an unfamiliar repository.

Target capabilities:

### Repository understanding
- identify package/workspace boundaries
- discover build/test/lint commands
- follow imports and symbol references
- understand dependency direction
- identify public APIs
- inspect git history when useful
- reason about cross-package impact

### TypeScript
- strict mode
- inference
- generics
- conditional/mapped types
- overloads
- declaration files
- type narrowing
- ESM/CJS boundaries
- module resolution
- project references
- monorepos

### Node.js runtime
- event loop
- async concurrency
- promises
- streams/backpressure
- buffers
- workers
- AbortController
- AsyncLocalStorage / async context
- resource cleanup
- process lifecycle/signals
- HTTP
- Undici
- filesystem
- memory/performance issues

### Backend engineering
- API compatibility
- retries
- idempotency
- transactions
- queues
- Kafka
- MongoDB
- PostgreSQL
- Redis
- distributed failure modes
- caching
- observability
- tracing
- security boundaries

### Ecosystem
- npm
- pnpm
- yarn where necessary
- Vitest
- Jest
- Node test runner
- ESLint
- TypeScript compiler
- tsup/esbuild/Vite
- Nx/Turborepo
- NestJS
- Fastify
- Express
- Prisma
- Drizzle
- Mongoose

### Senior behavior
- minimum-risk patching
- preserve compatibility
- add regression tests
- avoid speculative refactors
- explain invariants
- recover from failed tests
- distinguish symptom from root cause
- validate assumptions before editing
- review own final diff

---

# 6. Data architecture

We should build a **versioned data lake**, not one giant JSONL.

Proposed logical layers:

```text
data/
  raw/
    open_swe_traces/
    swe_rebench_v2/
    swe_rebench_v2_prs/
    generated_teacher/
  normalized/
    tasks/
    trajectories/
    outcomes/
    repository_metadata/
  filtered/
    license_safe/
    ts_js/
    node_relevant/
    verified_success/
    verified_failure/
  splits/
    train/
    validation/
    private_eval/
  manifests/
    dataset_commit.json
    license_manifest.parquet
    decontamination_manifest.parquet
    lineage_manifest.parquet
```

Every training example must retain:

- source dataset
- source dataset commit/hash
- repository
- repository license
- base commit
- issue/PR ID
- language
- task category
- harness
- teacher model
- teacher version
- decoding config
- rollout ID
- resolved/unresolved
- verifier results
- patch
- tool trajectory
- data-generation timestamp

**No anonymous “training sample” should exist without provenance.**

---

# 7. Phase 0 — dataset audit before any training

## Deliverable

`dataset-audit.parquet` + report.

For every existing source dataset:

1. Pin HF revision/commit.
2. Count exact current rows.
3. Count unique repositories.
4. Count unique PR/issues.
5. Count TS vs JS.
6. Count resolved vs unresolved.
7. Count by harness.
8. Count by generating model.
9. Count by source license.
10. Measure trajectory token length distribution.
11. Measure patch size distribution.
12. Identify duplicate `instance_id` rollouts.
13. Identify exact/near duplicate patches.
14. Identify tasks overlapping public eval sets.

### Why this is mandatory

Open-SWE-Traces is actively changing. The current HF config already includes newer Qwen3.6 splits beyond what the original paper describes.

We train from a **pinned snapshot**, not “latest”.

---

# 8. Licensing and provenance gate

## Initial repository-license allowlist

Start conservative:

- MIT
- Apache-2.0
- BSD-2-Clause
- BSD-3-Clause

Potential additions such as ISC should be reviewed explicitly before inclusion.

## Dataset licenses

- Open-SWE-Traces: CC BY 4.0
- SWE-rebench-V2: CC BY 4.0
- SWE-rebench-V2-PRs: CC BY 4.0
- Qwen candidates: Apache 2.0
- DeepSeek V4 Flash 0731: MIT

## Rules

- Preserve attribution/provenance manifest.
- Do not silently mix GPL/AGPL repositories into V1.
- Do not assume dataset-level CC BY overrides source repository licensing.
- Do not assume generated teacher output has no legal/provenance considerations.
- Keep gold human patch separate from teacher trajectory so we know what was generated versus copied.

A legal review is still recommended before commercial distribution of resulting weights/dataset.

---

# 9. Decontamination strategy

This is critical because a specialist model can appear “excellent” simply by seeing the test repositories/tasks during training.

## Split unit

**Repository**, not task.

Never:
- train on PR A from repo X
- evaluate on PR B from repo X

for our private benchmark.

## Stronger decontamination

For eval repositories:

- remove all tasks from same repo from training
- remove forks/mirrors
- remove task descriptions with high text similarity
- remove patches with high MinHash/token similarity
- remove identical stack traces/error messages if traceable to the held-out issue
- remove teacher trajectories generated with access to gold patches

## Public benchmark isolation

Keep independent contamination lists for:

- SWE-bench Verified
- SWE-bench Multilingual
- SWE-bench Pro
- our NodeLM-Bench

The private benchmark must be constructed **before** final training and then frozen.

---

# 10. Open-SWE-Traces usage plan

Do **not** simply fine-tune on all 66k TS/JS trajectories.

## Tiering

### Tier A — highest value
- TS/JS
- `resolved == 1`
- clean executable outcome
- reasonable patch size
- non-trivial trajectory
- no obvious runaway loops
- no corrupted tool output

Use for primary SFT.

### Tier B — useful successful trajectories
- resolved
- larger/noisier trajectory
- acceptable but inefficient path

Use with lower sampling weight.

### Tier C — failures
- `resolved == 0`
- model created patch
- meaningful test/tool feedback

Use for:
- recovery training
- preference pairs
- failure classifier
- critic training

Do **not** mix blindly into successful SFT.

### Tier D — unknown/invalid
- unknown outcome
- broken environment
- incomplete trajectory

Exclude until manually/automatically validated.

---

# 11. Preference pairs from existing trajectories

Open-SWE-Traces has multiple rollouts of overlapping tasks from different model/harness combinations.

Potential pair:

```text
same instance
  successful trajectory -> chosen
  failed trajectory     -> rejected
```

But only create a preference pair if:

- base repository state matches
- task statement matches
- environment is equivalent
- both rollouts have usable tool logs
- failure is model behavior, not infra failure

Otherwise the pair is invalid.

Preference optimization must teach **engineering decisions**, not “which harness happened to work.”

---

# 12. Teacher-generated Node/TypeScript data

Existing data gives breadth.

Our teacher-generated data should provide **specialization**.

## Teacher

Primary:
- DeepSeek V4 Flash 0731

Optional secondary teachers:
- strongest Qwen candidate
- another independent frontier/open teacher for disagreement checks

## Teacher should not see the gold patch

For real PR tasks:

Teacher gets:
- issue
- base commit
- repository
- normal tools

Teacher does **not** get:
- reference patch
- hidden test patch
- future commit

After completion:
1. execute hidden tests
2. compare behavioral result
3. optionally compare diff characteristics with human patch
4. save trajectory only with outcome metadata

This avoids simply teaching the student to mimic human diffs from leaked answers.

---

# 13. Teacher rollout policy

For hard tasks, generate **multiple independent rollouts**.

Initial experiment, not a fixed truth:

- easy task: 1–2 rollouts
- medium: 2–4
- hard: 4–8

Keep:
- shortest successful trajectory
- one alternative successful trajectory if materially different
- informative failures that demonstrate useful recovery signals

Do not keep 8 near-identical successful answers just because compute is available.

### Teacher trajectory format

Prefer training-visible structure like:

```text
task
tool call
observation
concise rationale / state update
tool call
observation
...
patch
test output
final verification
```

Avoid depending on huge hidden-chain-of-thought dumps.

The student needs to learn:
- what to inspect
- which tool to call
- how to react to observations
- when to patch
- how to recover

not to reproduce a teacher's private internal monologue word-for-word.

---

# 14. Node-specific task generation

We need tasks that public SWE datasets underrepresent.

## A. Runtime/concurrency mutations

Generate controlled bugs around:

- missing `await`
- Promise rejection handling
- duplicate retries
- timeout races
- AbortSignal misuse
- stream backpressure
- leaked listeners
- leaked handles
- worker termination
- AsyncLocalStorage context loss
- double callbacks
- partial failure
- shutdown ordering

Each synthetic task must have:
- reproducible failing test
- single known mutation
- hidden test
- clean base/fixed state

---

## B. TypeScript type-system failures

Examples:

- invalid generic constraint
- bad distributive conditional type
- incorrect overload order
- variance break
- unsound `any`
- accidental `unknown` widening
- declaration-file regression
- ESM import typing mismatch
- project-reference breakage
- `moduleResolution` migration failure

Primary verifier:
- `tsc --noEmit`
- package tests
- type tests where available

---

## C. npm/package ecosystem

Create tasks around:

- peer dependency changes
- ESM/CJS migration
- Node version upgrades
- package export maps
- workspace resolution
- pnpm hoisting assumptions
- lockfile behavior
- package deprecations
- test runner migrations
- TypeScript version changes

Prefer tasks reconstructed from real historical upgrade PRs.

---

## D. Backend correctness

- idempotency
- retries
- transaction boundaries
- queue redelivery
- Kafka offset semantics
- Mongo change-stream resume handling
- PostgreSQL isolation issues
- Redis lock lifetime
- API backward compatibility
- schema migrations
- caching invalidation
- observability regressions

These are especially important for “senior engineer” behavior.

---

# 15. Harness specification

The model should train and evaluate through **the same conceptual tool protocol**.

## Read-only tools

- `repo_tree`
- `search`
- `read_file`
- `find_symbol`
- `references`
- `callers`
- `package_info`
- `git_log`
- `git_show`
- `git_diff`
- `git_status`

## Mutation tools

- `apply_patch`
- `create_file`
- `delete_file` with stricter permission

## Execution tools

- `run_test`
- `run_test_file`
- `run_typecheck`
- `run_lint`
- `run_build`
- `run_package_script`
- `run_command` under sandbox policy

## Node-aware discovery

Harness automatically identifies:

- package manager
- workspaces
- package scripts
- tsconfig hierarchy
- eslint config
- test runner
- build system
- monorepo system

The model should not waste 20 turns rediscovering deterministic metadata.

---

# 16. Sandbox design

Every training/eval task should run in an isolated environment.

Requirements:

- pinned base commit
- reproducible dependency install
- resource limits
- command timeout
- filesystem snapshot/reset
- no access to gold patch
- no access to future git history during solve phase
- network disabled after dependencies are installed unless task explicitly requires it
- stdout/stderr captured
- test result parser
- process-tree cleanup

We should distinguish:

- model failure
- test failure
- environment failure
- timeout
- dependency/network failure

Do not label infrastructure failures as rejected model behavior.

---

# 17. Training curriculum

## Stage A — harness alignment / small SFT

Goal:
teach exact tool syntax and expected agent behavior.

Data:
- 2k–5k clean trajectories

Success criteria:
- >99% syntactically valid tool calls
- can complete simple repo tasks
- no major base-model regression

This stage is cheap and catches formatting mistakes before serious training.

---

## Stage B — TS/JS trajectory SFT pilot

Data:
- ~10k highest-quality TS/JS trajectories
- mostly successful Open-SWE-Traces
- small amount of teacher-generated Node-specific data

Method:
- parameter-efficient tuning first (LoRA or equivalent)

Why LoRA first:
- validate data/harness hypothesis
- easy A/B rollback
- avoids spending compute before proving benefit

### Gate

Continue only if the tuned model improves meaningfully on **repository-disjoint validation tasks**.

Example decision threshold — project choice, not a research fact:

- +5 percentage points or more absolute task resolution
- no >2 point regression on general coding sanity checks
- tool-call failure does not increase
- regression-test failure decreases

---

## Stage C — larger SFT

If Stage B passes:

Training mix, initial hypothesis:

- 45% high-quality existing TS trajectories
- 20% high-quality existing JS trajectories
- 20% DeepSeek-generated Node/TS trajectories
- 10% difficult SWE-rebench teacher rollouts
- 5% cross-language SWE trajectories for general engineering transfer

**These percentages are starting hyperparameters, not claimed optimums.**

Tune them using validation.

---

## Stage D — preference optimization

Build pairs from:

1. successful vs failed rollout of same task
2. minimal correct patch vs over-engineered correct patch
3. no-regression patch vs patch that breaks compatibility
4. recovered-after-test-failure trajectory vs repeated-bad-action trajectory

Candidate methods:
- DPO
- ORPO
- another supported offline preference method

Method is chosen based on stable support for the selected base architecture at implementation time.

---

## Stage E — execution-based RL

Only after SFT/preference stages are clearly beneficial.

Primary reward should be deterministic.

Example reward components:

```text
FAIL_TO_PASS hidden tests pass       strong positive
PASS_TO_PASS regression tests pass   strong positive
tsc passes                           positive
lint passes                          small positive
build passes                         positive
public API compatibility             positive if deterministically testable
forbidden architecture edge          negative
deleted/disabled tests               strong negative
new ts-ignore / broad any            configurable negative
timeout                              negative
environment failure                  no model penalty
```

Do not make an LLM judge the primary reward.

LLM critics may be auxiliary, never the sole correctness oracle.

---

# 18. Context-length strategy

Do not train everything at 256K/1M just because the model supports it.

That is expensive and may teach sloppy retrieval.

Initial curriculum:

- majority: 8K–32K
- selected complex tasks: 32K–64K
- smaller long-context subset: 64K+
- only add 128K+ training if evaluation proves it is needed

The harness should retrieve high-value context rather than dump entire repositories.

Product goal:
**better context selection**, not maximum token consumption.

---

# 19. Repository understanding layer

A senior agent should not rely only on embeddings.

Use hybrid retrieval:

### Lexical
- ripgrep
- BM25 / FTS

### Structural
- Tree-sitter
- package graph
- import graph

### Language-aware
- TypeScript language service / tsserver
- definitions
- references
- implementations
- type information

### Git
- relevant commit history
- blame only when useful
- changed-file relationships

### Semantic
- embeddings as an additional retriever, not the only retriever

The harness should build a compact repo map such as:

```text
workspace
  package
    public exports
    entrypoints
    internal modules
    tests
    dependencies
    scripts
```

---

# 20. Private NodeLM-Bench

This is the most important deliverable after the model itself.

## V1 size

Start with ~200 high-quality tasks.

Scale to 500+ only after the harness is stable.

## Categories

Suggested V1 distribution:

| Category | Target |
|---|---:|
| TS typing / build | 25 |
| Node async/concurrency | 25 |
| API/backend bug fix | 25 |
| persistence/transactions | 20 |
| queue/event processing | 15 |
| npm/ESM/build tooling | 20 |
| tests/reliability | 20 |
| performance/resource leaks | 15 |
| architecture/refactor | 20 |
| security/error handling | 15 |

Total: 200

## Rules

- repository-disjoint from training
- frozen before final model selection
- gold patch withheld
- executable hidden tests
- deterministic environment
- manually inspect at least a representative subset for task validity

## Metrics

Primary:
- Resolve@1
- Resolve@3

Secondary:
- FAIL_TO_PASS
- PASS_TO_PASS
- typecheck success
- regression rate
- tool-call error rate
- number of turns
- tokens per solved task
- wall-clock time per solved task
- diff size
- changed files
- human-review corrections required

---

# 21. External benchmark suite

Use for sanity/comparability, not as the optimization target.

### Required

- SWE-bench Multilingual — full and JS/TS slice
- a contamination-clean SWE-rebench-V2 TS/JS holdout
- Terminal-Bench if harness cost is acceptable

### Optional

- SWE-bench Verified for general software-engineering regression only
- SWE-bench Pro if evaluation environment is available and contamination is controlled

**Never tune directly against test-set failures from the final benchmark.**

---

# 22. Hardware plan

Hardware is not the primary constraint, so we optimize for **quality, iteration speed, and deployment flexibility**.

The project must stay **hardware-agnostic**. We should not optimize the model only for RTX cards or only for DGX Spark.

## Target hardware classes

### A. DGX Spark / GB10 — first-class deployment target

We can use:

- **1× DGX Spark**
  - 128 GB unified memory
  - useful for larger BF16/FP8 models and large-context single-agent inference
  - attractive for compact, low-power local deployment
- **2× DGX Spark**
  - 256 GB aggregate model capacity
  - suitable for larger models / higher-precision variants
  - useful as a local teacher, reviewer, or heavyweight coding-agent backend
  - inter-node topology and serving efficiency must be benchmarked on the actual runtime

DGX Spark is therefore a development target, inference target, teacher/reviewer target, and regression/performance target.

### B. RTX PRO workstation GPUs — first-class deployment target

Potential targets:

- RTX PRO 5000 72 GB
- RTX PRO 6000 96 GB

These are attractive for distilled students that fit fully in VRAM and benefit from much higher dedicated memory bandwidth.

### C. H100 / H200 / B200 — training and high-throughput infrastructure

Use datacenter GPUs when they materially improve:

- teacher rollout throughput
- large SFT
- preference optimization
- RL
- high-throughput evaluation
- checkpoint conversion / quantization experiments

They are **not** the only intended deployment environment.

## Teacher rollout generation

Use:
- current **2× DGX Spark** for pipeline bring-up, harness validation, and local teacher/reviewer experiments
- H200/B200 rental for bulk throughput if needed

Measure:
- requests/hour
- successful verified trajectories/hour
- cost per verified trajectory
- tool-call correctness
- successful task latency

The key metric is **verified useful trajectories per euro/hour**, not raw tokens/sec.

## LoRA pilot

Use whichever platform comfortably supports the selected student and target sequence length:
- DGX Spark where the model/training stack supports the experiment cleanly
- H200/B200 when faster iteration or larger training memory is required

Start with a one-step memory/profile run before committing rental size.

## Full SFT

Do **not** hard-code a GPU topology such as “4× B200 is enough.”

Actual memory depends on:
- student architecture
- optimizer
- ZeRO/FSDP strategy
- sequence length
- batch size
- activation checkpointing
- precision

Procedure:
1. run 1-step memory benchmark
2. record peak allocated/reserved memory
3. scale sharding/GPU count
4. add 15–20% operational headroom
5. only then rent the full training cluster

Likely tools:
- PyTorch FSDP or DeepSpeed ZeRO-3
- a training framework with proven support for the selected architecture

## Deployment validation matrix

Every serious checkpoint should be tested against at least two deployment classes where practical:

| Target | Purpose |
|---|---|
| 1× DGX Spark | capacity / large-context local inference |
| 2× DGX Spark | larger model / higher-quality distributed local inference |
| RTX PRO 5000 72 GB | high-bandwidth single-GPU workstation inference |
| RTX PRO 6000 96 GB | premium single-GPU workstation inference |
| H200/B200 | reference throughput / training / evaluation |

For every target measure:
- cold-load time
- prompt-processing tok/s
- decode tok/s
- max stable context
- peak memory
- tool-call correctness
- end-to-end task success
- wall-clock time per solved task
- power / cost where available

**Model selection must not be biased toward one hardware target unless product requirements explicitly choose that target later.**

---

# 23. Training framework gate

Before committing to a stack:

Run a tiny smoke fine-tune for each candidate:

- load model
- forward/backward
- save checkpoint
- reload checkpoint
- run inference
- validate tool-call output
- validate distributed checkpoint resume

Only then select among:
- TRL
- Axolotl
- LLaMA-Factory
- NeMo/Megatron
- custom FSDP/DeepSpeed

**Do not choose a framework from popularity alone.**

The selected stack must support:
- exact architecture
- long sequences
- packing
- gradient checkpointing
- distributed checkpointing
- resume after preemption
- LoRA/PEFT
- full SFT if we reach that stage

---

# 24. Experiment tracking

Every run must log:

- base model + exact revision
- tokenizer revision
- dataset snapshot hashes
- filtering code revision
- train/val split manifest
- hyperparameters
- optimizer
- precision
- sequence length distribution
- GPU type/count
- framework versions
- wall time
- GPU-hours
- loss curves
- benchmark results
- checkpoint hash

Use W&B, MLflow, or an equivalent system; selection can be made during implementation.

No unnamed checkpoint folders like `final2-real-last`.

---

# 25. Data quality scoring

Assign every trajectory a score from measurable features.

Potential features:

- resolved status
- tests actually executed
- regression tests passed
- patch applies cleanly
- patch size
- number of unrelated changed files
- number of repeated tool calls
- loop/repetition score
- command failures
- environment failure flag
- tool-call parse errors
- task category
- teacher/model
- harness
- token length
- recovery after initial failed attempt

Manual review:
- sample top 100
- sample middle 100
- sample bottom 100
- sample each language/framework/category

Do not trust an automatic score before inspecting examples.

---

# 26. What we should deliberately avoid

## Do not

- train on random code-completion dumps as the primary data
- copy gold patches into teacher prompts
- use task-level random split
- optimize on public benchmark test failures
- treat all `resolved=0` as “bad reasoning”
- treat infra failure as model failure
- let a judge LLM determine correctness without execution
- train everything at 256K context
- mix incompatible tool schemas without normalization
- assume a high Python SWE-bench score implies high TypeScript skill
- assume benchmark numbers from different harnesses are directly comparable
- over-specialize so hard that the model loses basic general engineering ability

---

# 27. Suggested repository layout

```text
nodelm/
  README.md

  configs/
    datasets/
    training/
    eval/
    harness/

  data_pipeline/
    ingest/
    normalize/
    filter/
    dedupe/
    license/
    decontam/
    split/
    export/

  harness/
    core/
    tools/
    typescript/
    node/
    docker/
    rewards/

  teacher/
    prompts/
    rollout/
    verify/
    critic/

  training/
    sft/
    preference/
    rl/
    distributed/

  evaluation/
    public/
    nodelm_bench/
    reports/

  benchmarks/
    manifests/
    frozen/

  scripts/
    audit_dataset.py
    generate_rollouts.py
    verify_rollouts.py
    build_sft.py
    build_preferences.py
    train.py
    eval.py

  docs/
    DATA_CARD.md
    MODEL_CARD.md
    LICENSE_NOTES.md
    EXPERIMENTS.md
    ARCHITECTURE.md
```

---

# 28. Milestones and stop/go gates

## M0 — Evidence + reproducibility

Deliver:
- pinned source manifest
- exact dataset counts
- license audit
- candidate base model benchmark
- harness skeleton

**Go if:** all source/license/eval assumptions are reproducible.

---

## M1 — Baseline harness

Deliver:
- Docker task runner
- Node/TS toolset
- public benchmark adapter
- private eval prototype
- baseline scores for all student candidates

**Go if:** same task can be replayed deterministically.

---

## M2 — 10k SFT pilot

Deliver:
- first NodeLM LoRA
- clean A/B against base
- per-category results

**Go if:** meaningful repository-disjoint improvement without general regression.

---

## M3 — Premium teacher data

Deliver:
- 5k–20k verified DeepSeek V4 trajectories
- success/failure pair set
- quality audit

**Go if:** teacher-generated data improves beyond Open-SWE-only SFT.

---

## M4 — Full specialist SFT

Deliver:
- full/large adapter-trained specialist
- public + private benchmark report
- inference profiles for DGX Spark and RTX PRO targets

**Go if:** private NodeLM-Bench shows clear senior-task improvement.

---

## M5 — Preference training

Deliver:
- preference dataset
- DPO/ORPO experiment
- diff-quality and recovery improvements

**Go if:** task resolution or regression rate improves.

---

## M6 — Execution RL

Deliver:
- RL environment
- deterministic reward
- controlled GRPO/RL experiment

**Go only if:** SFT/preference model is strong enough that RL rollouts produce useful signal.

---

## M7 — Production model

Deliver:
- quantized/optimized inference build
- model card
- data card
- benchmark report
- harness package
- reproducible deployment

---

# 29. Initial success criteria

The project should not be called successful because “the model feels smart.”

For V1, define success as all of:

1. statistically meaningful improvement over the selected base model on private NodeLM-Bench
2. lower regression rate
3. better tool-call reliability
4. no major general SWE regression
5. acceptable latency/cost for daily use
6. reproducible evaluation
7. traceable training-data provenance

Stretch goal:

- narrow the gap to frontier models specifically on TypeScript/Node repository tasks, even if the model remains weaker in general knowledge/reasoning.

---

# 30. First implementation sprint

When work starts, do these in order.

## Task 1 — Pin and audit datasets

- snapshot Open-SWE-Traces
- snapshot SWE-rebench-V2
- snapshot SWE-rebench-V2-PRs
- compute TS/JS counts and resolved ratios
- audit licenses
- build lineage manifest

## Task 2 — Build contamination map

- SWE-bench Multilingual repos/tasks
- NodeLM private candidate repos
- repo forks/mirrors
- exact/near duplicate issues/patches

## Task 3 — Build the harness

First support:
- npm/pnpm
- TypeScript
- Vitest/Jest/Node test
- ESLint
- generic shell command
- patching
- git diff

## Task 4 — Select student empirically

Benchmark:
- Qwen3.6-27B
- Qwen3.5-35B-A3B
- Qwen3-Coder-Next

on the same 50–100 TS/JS development tasks.

## Task 5 — Build 10k SFT pilot dataset

Prioritize:
- resolved TS Open-SWE trajectories
- resolved JS Open-SWE trajectories
- clean, bounded trajectory lengths
- no eval overlap

## Task 6 — LoRA pilot

Train selected student.

## Task 7 — Evaluate

Compare:
- base
- LoRA
- DeepSeek teacher

using identical harness and task budget.

Only after this result do we scale teacher generation/full fine-tuning.

---

# 31. Questions intentionally left open

These should be answered by experiments, not guesses.

1. Is Qwen3.6-27B or a sparse Qwen coder the best student after specialization?
2. How much Open-SWE data should be used versus our own teacher data?
3. Does explicit reasoning trajectory improve the student, or are action/observation summaries better?
4. Does full SFT materially outperform LoRA for our narrow domain?
5. What long-context training length gives the best cost/quality tradeoff?
6. Does preference optimization improve actual resolution or just style?
7. Does execution RL add enough value after strong SFT?
8. How much Node-specific synthetic mutation data transfers to real PR tasks?
9. What final precision gives the best production quality across 1× DGX Spark, 2× DGX Spark, RTX PRO 5000 72 GB, and RTX PRO 6000 96 GB?
10. Does a single specialist outperform a router of “fast coder + strong reviewer”?

---

# 32. Source registry

Accessed 2026-08-23.

## Models

### DeepSeek V4 Flash 0731
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731

Verified:
- MIT
- 304B
- agent benchmark table
- recommended agent sampling notes

### Qwen3.6-27B
https://huggingface.co/Qwen/Qwen3.6-27B

Verified:
- Apache 2.0
- 27B
- 262K native context
- ~55.6 GB HF repository
- official benchmark table

### Qwen3.5-35B-A3B
https://huggingface.co/Qwen/Qwen3.5-35B-A3B

Verified:
- Apache 2.0
- 35B total / 3B active
- 262K native context
- ~71.9 GB HF repository
- SWE-bench Verified 69.2

### Qwen3-Coder-Next
https://huggingface.co/Qwen/Qwen3-Coder-Next

Verified:
- Apache 2.0
- 80B total / 3B active
- 262K context
- ~159 GB BF16 repository
- HF evaluation metadata: SWE-bench Verified 70.6

---

## Datasets

### NVIDIA Open-SWE-Traces
https://huggingface.co/datasets/nvidia/Open-SWE-Traces

Paper:
https://arxiv.org/abs/2606.16038

Verified:
- CC BY 4.0
- published 207,489 trajectories
- TS 36,897
- JS 29,360
- trajectory/tool/outcome fields
- permissive source repository filtering
- paper's student fine-tuning results

### Nebius SWE-rebench V2
https://huggingface.co/datasets/nebius/SWE-rebench-V2

Verified:
- 32,079 tasks
- TS/JS supported
- executable test fields
- per-row source license
- dataset CC BY 4.0

### Nebius SWE-rebench V2 PRs
https://huggingface.co/datasets/nebius/SWE-rebench-V2-PRs

Verified:
- approximately 126k current PR records
- CC BY 4.0 dataset metadata

---

## Benchmarks

### SWE-bench Multilingual
https://www.swebench.com/multilingual.html

Verified:
- 300 tasks
- 42 repositories
- 9 languages
- 43 JS/TS tasks

### SWE-bench FAQ
https://www.swebench.com/SWE-bench/faq/

Verified:
- SWE-bench Verified = 500 expert-verified tasks
- standard SWE-bench evaluation design

---

# 33. Final recommendation

Do **not** begin with a giant B200 training run.

Begin with:

```text
Pinned datasets
   ↓
license + decontamination audit
   ↓
common Node/TS harness
   ↓
3-student baseline bake-off
   ↓
10k high-quality trajectory LoRA
   ↓
private repo-disjoint evaluation
```

If that clearly works:

```text
DeepSeek V4 teacher
   ↓
5k–20k premium Node/TS trajectories
   ↓
larger SFT
   ↓
preference optimization
   ↓
execution RL
```

The core durable asset is not just the checkpoint.

It is the combination of:

- `NodeLM` dataset
- executable task environments
- contamination-safe private benchmark
- Node/TypeScript-aware harness
- training recipe
- resulting specialist model

That stack can be re-trained on a better 20–40B base model whenever one appears, without tying the project to one GPU family.
