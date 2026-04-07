## Research Plan v3: New Orthogonal Feature Exploration

**Comparison baseline**: active-signal-v1-regime-gate@2 (Sharpe 1.385, PnL $1520.78, 388 trades, max DD 3.08%)

| Stage | Name | Depends | Worker | Goal |
|-------|------|---------|--------|------|
| 0 | Verify Best Shell | — | any | Confirm regime-gate@2 backtest metrics as comparison baseline |
| 1 | Funding Rate Regime Gate | [0] | qwen-1 | Build cf_funding_regime using funding event data to suppress signals during adverse funding extremes |
| 2 | Momentum Divergence Filter | [0] | qwen-2 | Build cf_rsi_divergence detecting price-RSI divergence to filter late-trend entries |
| 3 | Best-Shell Integration | [1,2] | any | Merge all promoted/watchlisted features from stages 1-2 into regime-gate shell, run full diagnostics |

**Verdict criteria**: PROMOTE if integrated backtest improves on ≥2 of (PnL, trades, max DD) vs regime-gate@2 baseline simultaneously with Sharpe ≥1.385. WATCHLIST if improves 1 metric without degrading others. REJECT if any key metric worsens.