# Specification status

`NodeLM_TypeScript_Node_Distillation_Plan.md` is present and tracked as the project ground
truth. Plan activation pins the primary teacher, all three student candidates, their immutable
Hugging Face revisions, licenses, architectures, loader classes, parameter counts, and native
context limits where verified. These are metadata claims only.

## Settled by the plan

- primary teacher: `deepseek-ai/DeepSeek-V4-Flash-0731`;
- candidate set: `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.5-35B-A3B`, and
  `Qwen/Qwen3-Coder-Next`;
- source datasets: Open-SWE-Traces, SWE-rebench V2, and SWE-rebench V2 PRs;
- decision sequence: audit and decontaminate data, build one common harness, run the candidate
  bake-off, then select and train a student;
- no scaling of teacher generation or full fine-tuning before the pilot evidence gates pass.

## Still open by design

- the winning student, precision, inference backend, and training framework;
- candidate load/generate compatibility and the same-harness 50–100-task bake-off;
- the frozen public/private evaluation manifests and measured near-duplicate threshold;
- full snapshot audit results, Tier A–D quality policy, and the actual 10k pilot artifact;
- model memory profiles, training topology, and any paid infrastructure request.

Model metadata is `PASS`; execution, bake-off, full snapshot download, and training are
`NOT RUN`. Full dataset transfer is deferred to a user-provisioned large-storage GPU instance
and follows `docs/DATA_DOWNLOAD_RUNBOOK.md`.
