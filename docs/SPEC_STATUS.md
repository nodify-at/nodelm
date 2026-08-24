# Specification status

The supplied bootstrap command designates `NodeLM_TypeScript_Node_Distillation_Plan.md` as the
ground truth. That file was not visible in the project workspace or Downloads at bootstrap
time. The repository proceeds with independently verified dataset facts and general contracts,
but leaves the following plan-dependent choices unresolved:

- student candidates and teacher model;
- exact model revisions, licenses, architecture, and context limits;
- benchmark/public-overlap corpus;
- training framework compatibility for the selected model;
- pilot thresholds and serious-training hyperparameters;
- precision, backend, GPU memory requirement, and infrastructure recommendation.

When the plan becomes available, compare every claim with primary sources, record discrepancies
in `docs/DECISIONS.md`, update configuration, and rerun all affected checks. Do not silently
replace a plan choice.
