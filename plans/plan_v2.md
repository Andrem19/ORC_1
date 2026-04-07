## Research Plan v2: Integration of Winner + New Orthogonal Exploration

| Stage | Name | Depends | Worker | Goal |
|-------|------|---------|--------|------|
| 0 | Re-integrate cf_regime_vol_gate | — | any | Clone base strategy, add promoted cf_regime_vol_gate, run full backtest + walk-forward + condition stability |
| 1 | Session Weakness Filter | — | qwen-1 | Build cf_* feature that suppresses signals during historically weak sessions (Sunday early UTC) |
| 2 | Volume Microstructure Signal | — | qwen-2 | Build cf_* feature using volume-imbalance or tick-volume profile to capture order-flow shifts |
| 3 | Best-Shell Integration | [0,1,2] | any | Merge all promoted/watchlisted features into single shell, run full diagnostics suite |

**Verdict criteria**: PROMOTE if integrated backtest improves on ≥2 of (PnL, trades, max DD) vs baseline simultaneously with Sharpe ≥1.06. WATCHLIST if improves 1 metric without degrading others. REJECT if any key metric worsens.