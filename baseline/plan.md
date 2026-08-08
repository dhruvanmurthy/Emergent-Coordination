## Plan: Tool-Use Coordination Research Migration

Migrate the repo from scalar guessing to a controlled, automatically-checkable tool-use environment centered on a synthetic bugfix workflow, then measure coordination using categorical mutual information with robust sanity checks. Treat this as a clean break: retire the scalar-game architecture early and make the new tool-use pipeline the primary system rather than a compatibility layer.

**Steps**
1. Phase 0: Scope lock and hard cutover contract.
2. Finalize the task contract for the synthetic bugfix environment: episode inputs, allowed tools, action argument schemas, progress-state transitions, and terminal outcomes (success, partial, failure).
3. Declare the scalar-game scripts deprecated and stop designing for backward compatibility.
4. Define the pilot contract: 100-300 episodes, 2-agent primary setting, OpenAI model baseline, and random-seed strategy.
5. Phase 1: Controlled Tool-Use Environment.
6. Implement a Tool-GBS-style environment object with deterministic state transitions and automatic progress checks.
7. Implement tool schemas for retrieve_file, search_symbol, run_tests, apply_patch, finalize_ticket with strict argument validation and explicit invalid-call reasons.
8. Add state machine support for not_started, in_progress, completed, while preserving ternary final outcome Y (success, partial, failure).
9. Add deterministic replay mode for debugging and trajectory reproducibility.
10. Remove scalar-only branching from the main run path once the new environment is available.
11. Phase 2: Baseline agents and policies.
12. Implement baseline policy adapters: single-agent, random-policy (uniform over valid schema-constrained actions), independent multi-agent, prompted-coordination multi-agent.
13. Record per-episode metrics: success rate, partial rate, invalid call rate, action distributions, and baseline MI summaries.
14. Keep persona experiments out of the first pilot; revisit only after the controlled environment and estimator are stable.
15. Phase 3: Trajectory logging and abstraction.
16. Create unified event logging schema capturing agent messages, tool calls, tool outputs, environment state snapshots, progress labels, invalid-call diagnostics, and episode outcomes.
17. Write canonical outputs: JSONL event stream plus columnar dataset export (Parquet) for analytics.
18. Implement tool-event abstraction mapping raw events into categories: retrieve, verify, update, finalize, error_or_noop.
19. Add abstraction quality checks: mutual exclusivity and exhaustiveness assertions in code.
20. Phase 4: Abstraction validation protocol.
21. Sample a stratified subset of trajectories (across success/partial/failure and baseline types).
22. Run human labeling protocol with two raters and an adjudication pass on disagreements.
23. Compute inter-rater reliability (Cohen kappa for pairwise labels; agreement tables by category).
24. Document task relevance criteria and failure modes for category mapping.
25. Phase 5: MI estimator and sanity checks.
26. Implement count-based categorical MI estimators for I(A1,A2;Y) with optional small-sample smoothing controls.
27. Add bootstrap confidence intervals for MI and baseline deltas.
28. Implement sanity suites: random agents (near-zero MI), shuffled pairings (drop MI), duplicated agents (inflated redundancy), complementary scripted agents (expected positive coordination).
29. Track MI stability against sample size and category cardinality.
30. Phase 6: Coordination and correlation analysis.
31. Define primary coordination metric as I(A1,A2;Y); keep PID/synergy as secondary and gated by stability thresholds.
32. Compute correlations between coordination score and outcomes: success, reliability, invalid calls, error recovery.
33. Add regression/association summaries with confidence intervals and robustness checks against baseline type.
34. Phase 7: Reporting and cleanup.
35. Rewrite visualization and extraction scripts to consume the trajectory schema instead of scalar-game log parsing.
36. Remove deprecated scalar-only code paths after the new pipeline passes pilot validation and reproducibility checks.
37. Produce a reproducible experiment card: config snapshot, dataset fingerprint, estimator settings, and sanity-check results.

**Relevant files**
- c:/Code/dhruvanmurthy/Emergent-Coordination/baseline/experiment.py — replace scalar game loop with the new environment entrypoint.
- c:/Code/dhruvanmurthy/Emergent-Coordination/baseline/run_baseline_sweep.py — current batch orchestrator for baseline/policy sweeps.
- c:/Code/dhruvanmurthy/Emergent-Coordination/baseline/llm_run.py — removed from the active repository surface during controlled-pilot cleanup.
- c:/Code/dhruvanmurthy/Emergent-Coordination/baseline/settings.py — central place for new experiment contracts, seeds, dataset size, and estimator toggles.
- c:/Code/dhruvanmurthy/Emergent-Coordination/baseline/results_visualization.py — replace scalar convergence plots with MI, reliability, and correlation dashboards.
- c:/Code/dhruvanmurthy/Emergent-Coordination/baseline/export_trajectory_data.py — schema-native trajectory extraction and table export.
- c:/Code/dhruvanmurthy/Emergent-Coordination/baseline/context.md — update repository narrative to reflect tool-use environment and coordination metrics.
- c:/Code/dhruvanmurthy/Emergent-Coordination/baseline/context_methods.md — update method documentation for new environment, abstraction, and estimators.

**Verification**
1. Unit checks for environment transition logic, invalid-action handling, and terminal outcome assignment.
2. Schema checks on JSONL and Parquet outputs: required fields present, type constraints valid, event ordering consistent.
3. Abstraction invariants: every tool event maps to exactly one category; category coverage reports for full pilot dataset.
4. Rater reliability: compute and review kappa plus disagreement matrix on stratified sample.
5. MI sanity suite passes expected directional behaviors under random, shuffled, duplicated, and complementary conditions.
6. End-to-end pilot run (100-300 episodes) reproduces from fixed seeds with stable MI confidence intervals.

**Decisions**
- Included scope: synthetic bugfix workflow first, 2-agent primary analysis, all four baselines, ternary outcome Y, JSONL+Parquet storage, bootstrap CIs.
- Deferred scope: PID/synergy as secondary only after stable controlled-setting MI.
- Deferred scope: persona-conditioned coordination experiments until core environment and estimators are validated.

**Further Considerations**
1. Inter-rater protocol recommendation: sample 10-15 percent of pilot episodes, dual-label independently, adjudicate disagreements, then freeze rubric before full run.
2. Partial outcome policy should be made explicit in the environment contract, since you selected ternary Y and non-terminal progress as partial.
3. The cutover should be explicit in the repository history so the new environment becomes the only supported path after the pilot validates.