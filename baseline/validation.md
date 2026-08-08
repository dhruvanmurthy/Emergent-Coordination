# Validation Tracker

This file tracks validation work for the controlled tool-use migration. It includes validations that can be run now inside the repo and validations deferred until the full experiment environment is available.

## Executable Now

- [ ] Unit tests for environment transition logic, invalid-call handling, baseline-policy execution, and terminal outcome assignment. Not executed in this session because terminal validation was skipped.
- [ ] Batch runner smoke test for baseline sweeps writing episode JSONL and summary artifacts.
- [x] Static diagnostics on the touched Python modules.
- [x] Legacy scalar/persona helper files removed from the active repository surface.

## Deferred Until Full Pilot Environment Is Ready

- [ ] End-to-end pilot run with 200-400 episodes total, with at least 50 episodes per major baseline cell.
- [ ] Deterministic replay check confirming fixed `(seed, template_id, variant_id)` reproduces the same fixture selection and action trajectory for scripted baselines.
- [ ] JSONL schema validation for required fields, event ordering, and config-hash consistency.
- [ ] Parquet export validation for analysis-ready trajectory tables.
- [ ] Abstraction invariants over the full pilot dataset: every tool event maps to exactly one category in `{retrieve, verify, update, finalize, error_or_noop}`.
- [ ] Invalid-call breakdown audit by reason class: `schema_invalid`, `reference_invalid`, `state_invalid`, `semantic_noop`, `budget_invalid`.
- [ ] Human labeling protocol on a 10-15% stratified trajectory sample with Cohen kappa, confusion matrix, and adjudication pass.
- [ ] MI estimator sanity suite: random-policy near-zero MI, shuffled pairings lower MI, duplicated traces higher redundancy, complementary scripted agents positive MI increase, bootstrap CI stability, and permutation-null p-values.
- [ ] Correlation analysis across success/partial/failure rates, invalid-call rate, recovery after invalid calls, calls-to-completion, test-run efficiency, patch acceptance ratio, and reproducibility variance across seeds.

## Current Gaps

- [ ] Batch runner still needs a smoke run after the baseline sweep cutover.
- [ ] Canonical Parquet export still needs execution-time validation in a real run environment.
- [ ] MI estimation and reporting pipeline still needs execution-time validation in a real run environment.
- [x] Documentation and visualization scripts no longer assume the scalar-game workflow.
- [x] Legacy LLM/persona helper modules and ad-hoc compression/launch scripts are removed from the active tree.