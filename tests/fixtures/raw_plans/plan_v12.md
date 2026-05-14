# Plan v12

## ETAP 1: Cross-Market Confirmation

### Goal
Test whether an external confirmation layer adds non-cannibalizing trades.

### Actions
1. Inspect available datasets.
2. Preview feature columns.
3. Run a lightweight candidate analysis.

### Completion Criteria
- Parser extracts stage title and actions.

| metric | value |
| --- | --- |
| parsed | yes |

## ETAP 2: Drawdown Guard

### Goal
Evaluate a guardrail that reduces drawdown while preserving trade count.

### Actions
1. Inspect baseline run diagnostics.
2. Preview guard conditions.
3. Record trade-count impact.

### Completion Criteria
- Parser confidence stays above threshold.

| metric | value |
| --- | --- |
| parsed | yes |
