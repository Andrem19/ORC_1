## Plan v2: Second Wave — Three New Orthogonal Dimensions

### v1 Results Carried Forward
- ✅ **cf_vol_term_spread** → PROMOTE: +14.3% PnL ($1496 vs $1309), Sharpe 1.456, 330 trades, DD ~1.68%, PF 1.571
- ❌ **cf_funding_extreme** → REJECT: funding rate extremes lack edge on 5m (Sharpe 0.952, +634 trades with lower quality)
- ❌ **cf_return_skew** → REJECT: raw skewness adds noise (-26.2% PnL, 694 trades)

### v2 Hypothesis
Baseline + vol_term_spread capture: trend direction, momentum, vol levels, vol term structure transitions.
Three untouched information dimensions:
1. **cf_cumulative_delta** — Volume × directional body proxy → order flow imbalance / buying-selling pressure (microstructure)
2. **cf_choppiness_index** — ATR-sum vs price range ratio → trending vs choppy regime TYPE (market structure, not vol level)
3. **cf_session_momentum** — US vs Asian session return differential → cross-session money flow (temporal seasonality)

### DAG
```
Stage 0 (Bootstrap + Gold) 
  ├── Stage 1 (cf_cumulative_delta) ──┐
  ├── Stage 2 (cf_choppiness_index) ──┼── Stage 4 (Integration + Walk-forward)
  └── Stage 3 (cf_session_momentum) ──┘
```

### Verdict Gates
- PROMOTE: ≥5% improvement in ≥2 of {PnL, trades, DD} vs baseline
- WATCHLIST: ≥3% in 1 metric, no degradation in others
- REJECT: no improvement or cannibalization

### Anti-patterns from v1
- Raw higher-order return statistics (skewness) generate noise, not signal
- Funding rate events lack predictive power for 5m execution timeframe
- Features that dramatically inflate trade count without proportional PnL are noise generators