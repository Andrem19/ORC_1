## Orthogonal Signal Discovery — 3 Parallel Branches

### Hypothesis
Baseline captures trend, momentum, and volatility LEVELS from price data. Three untouched information dimensions:
1. **cf_funding_extreme** — external sentiment/crowding from funding rates (event data, not price-derived)
2. **cf_return_skew** — distribution shape via rolling skewness of returns (higher-order stat, not mean/var)
3. **cf_vol_term_spread** — short/long vol ratio capturing regime TRANSITIONS (not vol levels)

### Flow
- **Stage 0**: Bootstrap, catalog inspection, baseline run
- **Stages 1–3** (parallel, 1 worker each): Feature creation → dataset build → clone+modify baseline → backtest → compare
- **Stage 4**: Merge winners, integration test, walk-forward, ownership analysis

### Verdicts
- PROMOTE: ≥5% improvement in ≥2 of {PnL, trades, DD} vs baseline
- WATCHLIST: ≥3% in 1 metric, no degradation
- REJECT: no improvement or cannibalization