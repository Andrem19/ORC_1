## Research Plan: Orthogonal Feature Exploration Cycle

| Stage | Name | Depends | Worker | Goal |
|-------|------|---------|--------|------|
| 0 | Baseline + Diagnostics | — | any | Revalidate baseline, run diagnostics to identify weak quarters and load-bearing feature families |
| 1 | Regime-Aware Volatility Gating | [0] | qwen-1 | Build cf_* feature that gates signals by volatility regime classification |
| 2 | Momentum Divergence Signal | [0] | qwen-2 | Build cf_* feature detecting price-vs-indicator divergence for early reversal capture |
| 3 | Adaptive Trailing Exit | [0] | qwen-1 | Build cf_* exit feature that adapts stop distance to recent ATR regime |
| 4 | Integration + Robustness | [1,2,3] | any | Merge promoted features, run walk-forward + condition stability, compare to baseline |

**Verdict criteria**: PROMOTE if integrated backtest improves on ≥2 of (PnL, trades, max DD) vs baseline simultaneously with Sharpe ≥1.06. WATCHLIST if improves 1 metric without degrading others. REJECT if any key metric worsens.