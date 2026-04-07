## Research Plan v4: Orthogonal Feature Exploration — ML, Exhaustion, Settlement Cycle

**Comparison baseline**: active-signal-v1-regime-gate@2 (Sharpe 1.385, PnL $1520.78, 388 trades, max DD 3.08%)

| Stage | Name | Depends | Worker | Goal |
|-------|------|---------|--------|------|
| 0 | Context & Readiness | — | any | Inspect feature catalog, model registry, research memory for gaps |
| 1 | ML Model Feature (cf_model_predict) | [0] | qwen-1 | Train CatBoost on existing features, project as cf_* feature, standalone backtest |
| 2 | Trend Exhaustion (cf_trend_exhaustion) | [0] | qwen-2 | Build ADX+RSI exhaustion composite, standalone backtest |
| 3 | Settlement Cycle (cf_settlement_cycle) | [0] | qwen-1 or qwen-2 | Use bars_since/until_funding_settlement timing as cyclical signal, standalone backtest |
| 4 | Best-Shell Integration | [1,2,3] | any | Merge all promoted/watchlisted features into regime-gate shell, full diagnostics |

**Verdict criteria**: PROMOTE if standalone backtest improves ≥2 of (PnL, trades, max DD) vs regime-gate@2 baseline with Sharpe ≥1.385. WATCHLIST if improves 1 metric without degrading others. REJECT if any key metric worsens. Integration must improve net over base.