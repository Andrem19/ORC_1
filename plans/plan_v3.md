## Plan v3: Volume × Momentum Acceleration × Order Flow

### Carried Forward (PROMOTED)
- ✅ cf_vol_term_spread → $1496 PnL, Sharpe 1.456, 330 trades, DD 1.68%

### v2 Rejections (anti-patterns)
- ❌ cf_choppiness_index — ATR/range regime = correlated with existing vol features
- ❌ cf_session_momentum — cross-session returns produce no trades

### v3 Hypothesis
Baseline captures: trend direction, momentum level (RSI), vol level (IV), vol term structure, time seasonality, classifier regimes.
Three genuinely new dimensions never explored:
1. **cf_cumulative_delta (RETRY)** — Volume × body direction → order flow buying/selling pressure. Published in v2, never backtested.
2. **cf_trend_acceleration** — ROC(12) - ROC(48) rolling mean → momentum 2nd derivative. Captures whether momentum is accelerating or decelerating. Fundamentally different from RSI level.
3. **cf_relative_volume** — volume / SMA(volume, 96) → volume intensity anomaly. No volume-based signal exists in the system. Completely untapped information source.

### DAG
```
Stage 0 (Bootstrap + Verify)
  ├── Stage 1 (cf_cumulative_delta standalone) ──┐
  ├── Stage 2 (cf_trend_acceleration)           ──┼── Stage 4 (Integration + Walk-forward)
  └── Stage 3 (cf_relative_volume)              ──┘
```

### Verdict Gates
- PROMOTE: ≥5% improvement in ≥2 of {PnL, trades, DD} vs baseline
- WATCHLIST: ≥3% in 1 metric, no degradation in others
- REJECT: no improvement or cannibalization