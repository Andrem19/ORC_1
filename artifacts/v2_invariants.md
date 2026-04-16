# V2 Cycle Invariants & Naming Convention

## Source
- compiled_plan: compiled_plan_v2_batch_1
- slice: compiled_plan_v2_stage_1
- baseline: active-signal-v1@1
- symbol: BTCUSDT, anchor=1h, execution=5m

## Hard Invariants (non-negotiable)

1. **BASE_IMMUTABLE**: active-signal-v1@1 never changes — no retuning, no threshold adjustment, no hour/day tweaks, no cosmetic variations
2. **NO_RAW_H1_H2**: Raw event-entry constructions ("event + single filter") are a dead end. Do not reopen H1/H2 in raw form.
3. **COMPLEMENT_FIRST**: Every new candidate must first be tested on base-absent surface. Research begins where the base is silent.
4. **MIN_THREE_LAYERS**: Single-filter entries are forbidden. Every candidate needs at least 3 logical layers: (a) new information layer, (b) regime confirmation, (c) density limiter.
5. **NO_MODEL_FIRST**: Model-first approach forbidden until low-friction Wave A completes. Structured hypotheses exhaust simpler options first.
6. **STRICT_ADDITIVE_REQUIRED**: A candidate is rejected if its value disappears in strict-additive mode (fires only when base is absent).
7. **CANNIBALISM_VERDICT_MANDATORY**: Every integration candidate must have formal overlap/new-trade/ownership metrics. High overlap + weak strict-additive = auto-reject.

## Permitted Branches

### Wave A (low-friction, 1h data only)
- **A1**: IV dynamics from iv_est_1 (delta, acceleration, reversal, stretch relative to history)
- **A2**: Event × state interactions (structured, not raw event timing)
- **A3**: Classifier-transition specialists (regime transitions between windows, not static cl_*)
- **A4**: Compression-to-release without model (past-only features)

### Wave B (medium-friction, conditional)
- **B1**: Cross-symbol leadership (requires ETHUSDT 1h data readiness)
- **B2**: Model-backed compression/routing (requires models_train pipeline)
- **B3**: Event-state-model hybrids (only after B1/B2 progress)
- **Gate**: Wave B only starts if Wave A is inconclusive (no candidate passes strict additive integration)

## Naming Convention

### Candidates
- Format: `v2_<wave>_<family>_<index>`
- Examples: `v2_a_ivd_01`, `v2_a_evt_01`, `v2_b_csl_01`

### Runs
- Format: `run_<candidate_id>_<mode>_<timestamp_short>`
- Modes: `sa` (standalone), `co` (complement-only), `si` (strict-integrated), `pi` (permissive-integrated)
- Example: `run_v2_a_ivd_01_co_0414`

### Hypothesis Atlas Dimensions
- hypothesis_family: [iv_dynamics, event_state, classifier_transition, compression_release, cross_symbol, model_routing]
- data_layer: [1h_ohlcv, 1h_custom, cross_symbol, model_output]
- direction: [long_specialist, short_specialist, conditioning_layer]
- expected_missing_regime: [low_vol_base_absent, event_window_base_absent, transition_base_absent, compressed_base_absent]
- wave: [A1, A2, A3, A4, B1, B2, B3]
- status: [hypothesized, data_ready, plausibility_approved, prototype_built, complement_tested, stability_tested, integrated_strict, integrated_permissive, accepted, rejected, deferred]

## Key Paradigm Shift (v1 → v2)

| Dimension | v1 Approach | v2 Approach |
|-----------|-------------|-------------|
| Signal type | Mass low-conviction | Strict specialist |
| Search space | Full market | Base-absent surface only |
| Architecture | Single filter | 3+ logical layers |
| Integration | Standalone-first | Additive-by-construction |
| Model usage | Any time | Late wave only |
| Event usage | Raw timing | Structured interaction with state |

---
*Generated from analysis of compiled_plan_v2/semantic.json, raw_plans/plan_v2.md, and artifacts/v1_postmortem.md*
