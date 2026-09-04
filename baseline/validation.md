# Validation Tracker

This file tracks validation work for the controlled tool-use migration. It includes validations that can be run now inside the repo and validations deferred until the full experiment environment is available.

## Executable Now

- [x] Unit tests for environment transition logic, invalid-call handling, baseline-policy execution, and terminal outcome assignment. `18/18` pass with `python -m unittest discover -s tests -v`.
- [x] Batch runner run completed with `80` episodes across four baselines; all `80` episode directories were retained.
- [x] Trajectory export completed with `80` episodes and `558` events.
- [x] Coordination report and visualization dashboards generated from the corrected sweep artifacts.
- [x] Lower-cardinality MI representation added and validated. The primary report now uses `coarsened_action_profile`, derived from bounded per-agent action counts over `retrieve_count`, `verify_count`, `update_count`, `finalize_count`, and `error_or_noop_count`; full traces and bounded count vectors remain in the report for comparison.
- [x] Static diagnostics on the touched Python modules.
- [x] Legacy scalar/persona helper files removed from the active repository surface.

## Deferred Until Full Pilot Environment Is Ready

- [ ] End-to-end pilot expansion to 200-400 episodes total, with at least 50 episodes per major baseline cell. The current 80 episodes are a pipeline-validation run.
- [ ] Deterministic replay check confirming fixed `(seed, template_id, variant_id)` reproduces the same fixture selection and action trajectory for scripted baselines.
- [ ] JSONL schema validation for required fields, event ordering, and config-hash consistency.
- [ ] Parquet export validation for analysis-ready trajectory tables.
- [ ] Abstraction invariants over the full pilot dataset: every tool event maps to exactly one category in `{retrieve, verify, update, finalize, error_or_noop}`.
- [ ] Invalid-call breakdown audit by reason class: `schema_invalid`, `reference_invalid`, `state_invalid`, `semantic_noop`, `budget_invalid`.
- [ ] Human labeling protocol on a 10-15% stratified trajectory sample with Cohen kappa, confusion matrix, and adjudication pass.
- [x] MI estimator sanity suite: independent traces low MI, shuffled outcomes lower MI, duplicated agents do not inflate MI, complementary scripted agents create joint MI, bootstrap CI stability, and permutation-null p-values.
- [x] Replace full trajectory strings with a lower-cardinality representation before interpreting MI. In the corrected 80-episode pilot, random policy has `17` eligible two-agent episodes; `coarsened_action_profile` reduces unique joint states from `17` full traces to `9` profiles and lowers MI from `1.48` to `1.04` bits.
- [ ] Correlation analysis across success/partial/failure rates, invalid-call rate, recovery after invalid calls, calls-to-completion, test-run efficiency, patch acceptance ratio, and reproducibility variance across seeds.

## Current Gaps

- [x] Batch runner smoke run after the baseline sweep cutover.
- [ ] Canonical Parquet export still needs execution-time validation of the manifest and files in the active environment.
- [x] MI estimation and reporting pipeline executes on the corrected 80-episode run.
- [ ] MI interpretation still needs methodological validation because three baselines have constant outcomes and the random-policy profile MI remains high in a small sample.
- [x] Single-agent runs are excluded from two-agent MI and reported with a clearly named one-agent reference block.
- [x] Documentation and visualization scripts no longer assume the scalar-game workflow.
- [x] Legacy LLM/persona helper modules and ad-hoc compression/launch scripts are removed from the active tree.