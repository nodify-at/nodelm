# Evaluation

Evaluation uses one backend-neutral candidate contract and the same conceptual repository
harness for every model. Candidate IDs, revisions, licenses, architectures, parameter counts,
and context limits must be pinned before a run.

Each measured run records precision, backend, GPU memory, prompt/decode speed, tool-call
validity, task resolution, regression-test result, token counts, and wall-clock duration.
Nullable metrics mean “not measured”; they must never default to success or zero.

The fixed smoke fixture accepts only bounded, text-only Git patches and executes model-authored
code only in a preloaded, digest-pinned rootless Podman image. The container has a read-only
host-workspace bind, no network, bounded resources, and forced cleanup. Its harness is bound to
one exact `SolveContext`; `PASS` additionally requires the approved pre-patch source identity,
the exact approved post-patch identity, an expected failing baseline, unchanged protected
content, and successful final tests.

Generic repository execution also rejects symlinks in every protected-path component. A
successful generic test-command exit is useful execution evidence, but without an
integrity-attested oracle it remains `UNVERIFIED` and does not establish task resolution or
regression-test integrity.

Private evaluation is frozen at repository level before serious fine-tuning. Declared mirrors,
exact task/patch duplicates, and measured near duplicates join the same group. Public benchmark
overlap detection is required once the ground-truth plan names the benchmark corpus.

The current candidate registry is `UNVERIFIED` because
`NodeLM_TypeScript_Node_Distillation_Plan.md` is not visible. The bake-off report therefore
contains no benchmark claims.
