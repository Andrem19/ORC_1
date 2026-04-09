# План дальнейших исследований для v9

## Статус перед стартом v9

**База, которую не трогаем:** `active-signal-v1@1`.

Это immutable reference.
Запрещено:
- менять базу;
- подбирать ей часы, дни, пороги, исполнение;
- заново открывать `full-surface directional classification` на старом data universe;
- заново открывать `trade-quality overlays` на том же самом наборе данных без нового информационного слоя;
- возвращаться к regression на future PnL;
- выбирать feature set, label scheme, horizon, threshold или архитектуру по lockbox-окну;
- объявлять улучшением все, что не прошло matched OOS comparison против базы на том же окне.

---

## Что уже доказано и что это значит для v9

### Что окончательно закрыто
1. **Raw event entries** не работают.
2. **Static IV** и **IV dynamics** не дают directional alpha.
3. **Classifier transitions** сами по себе не создают новый информационный слой.
4. **CatBoost regression на future PnL** — тупик.
5. **Full-surface directional classification** на текущем data universe провалилась.
6. **Trade-quality overlays** на текущем data universe тоже не дали устойчивого улучшения:
   - veto-long дал только marginal signal;
   - precision слишком низкая;
   - lockbox-тест даже не был оправдан.

### Что это означает
На текущем data universe уже проверены обе разумные постановки:
- prediction рынка с нуля;
- improvement уже отобранных базой сделок.

Обе дали слабый или неэкономичный результат.

### Главный вывод для v9
**Если продолжать исследование, то только через новый data universe.**

Не новый threshold.
Не новую перестановку старых features.
Не еще одну CatBoost-конфигурацию на тех же колонках.

Нужен **другой тип информации**, который может содержать то, чего у нас до сих пор не было:
- positioning pressure;
- squeeze risk;
- liquidation stress;
- cross-venue pressure;
- microstructure imbalance;
- options term-structure / skew, если доступно.

---

## Главная идея v9

**v9 — это не новая попытка "докрутить" старые модели.
Это цикл поиска и первичной проверки нового data universe.**

Задача v9:
1. честно закрыть текущий universe как исчерпанный;
2. провести аудит новых доступных источников данных;
3. выбрать **одну** главную новую data family;
4. построить минимальный сильный feature layer;
5. проверить, есть ли в нем хоть какой-то реальный информационный сигнал;
6. только потом обучать модели.

---

## Главная цель v9

Ответить на вопрос:

> есть ли в dev_space1 доступный новый data universe, который добавляет реальную информацию сверх текущего OHLCV + IV + classifier + breadth + event space, и может ли он стать основой для нового model route?

---

## Главные принципы v9

### Принцип 1. Новый data universe или ничего
Если источник данных не новый по сути, а лишь другая упаковка уже использованной информации, он не подходит для v9.

### Принцип 2. Одна главная data family
Нельзя параллельно раздувать:
- order flow,
- options,
- basis,
- liquidations,
- cross-exchange,
- on-chain

в одном широком цикле.

Нужно выбрать:
- **Primary family**
- **Fallback family**

И идти по ним строго по очереди.

### Принцип 3. Сначала information-content triage, потом модели
До CatBoost нужно пройти:
- data readiness;
- causal sanity;
- feature materialization;
- bucketed plausibility;
- simple ranking sanity.

Если источник не показывает информационного содержания на простых тестах, модель его не спасет.

### Принцип 4. Overlay-first, direction-second
Так как полный directional route уже провалился, новый data universe сначала проверяется на:
- **base-conditional veto / overlay surface**.

Только если data family показывает сильный signal, разрешается пробный directional scout.

### Принцип 5. Не тратить цикл на недоступные данные
Если family недоступна или полумертва по coverage, ее нужно быстро закрывать и идти к fallback.

---

## Какие новые data families должны быть проверены в v9

Ниже перечислены только те источники, которые действительно отличаются по смыслу от старого universe.

---

## Family A. Derivatives positioning / pressure
**Главный приоритет.**

### Что сюда входит
- open interest level;
- open interest change;
- open interest acceleration;
- price vs OI divergence;
- funding velocity;
- funding surprise / funding shock;
- premium index;
- perp-spot basis;
- basis change;
- liquidation intensity / liquidation imbalance;
- long-short positioning ratios, если доступны.

### Почему это сильный кандидат
IV показывал только режим волатильности.
А positioning-data может показывать:
- переполненность одной стороны;
- squeeze setup;
- forced unwind;
- non-directional calm vs directional stress.

Это уже другая информация.

---

## Family B. Cross-exchange pressure / venue divergence
**Второй приоритет.**

### Что сюда входит
- Binance spot vs perp divergence;
- Coinbase vs Binance relative move;
- exchange spread expansion;
- lead-lag across venues;
- venue volume shifts;
- cross-venue failed breakout confirmation.

### Почему это интересно
Может содержать ранние признаки:
- real spot-led move;
- perp-led squeeze;
- fake breakout without cross-venue confirmation.

---

## Family C. Microstructure proxies from lower timeframe
**Третий приоритет.**

### Что сюда входит
Если нет true order book / trade tape, то хотя бы:
- 1m/5m signed range proxies;
- repeated close-near-high / close-near-low sequences;
- imbalance-style bar sequences;
- burstiness / shock clustering;
- failure-to-follow-through after impulse;
- intrabar continuation vs rejection proxies.

### Почему это полезно
Это уже ближе к order-flow behavior, а не к старым 1h regime-features.

---

## Family D. Options structure
**Четвертый приоритет, только если реально доступно.**

### Что сюда входит
- term structure;
- skew;
- skew change;
- short-dated vs long-dated IV divergence;
- options event context.

### Почему это интересно
Это может дать не просто volatility regime, а positioning asymmetry and protection demand.

---

## Как выбирать primary family

Нужно использовать строгую оценку по пяти критериям:

1. **Availability**
   Есть ли данные в достаточном объеме и без больших дыр.

2. **Causality**
   Можно ли строить признаки строго по прошлому.

3. **Novelty**
   Это реально новый слой, а не новая форма старого OHLCV.

4. **Economic interpretability**
   Можно ли словами объяснить, почему из этого может родиться edge.

5. **Expected fit to base overlays**
   Есть ли шанс, что family поможет:
   - veto bad trades,
   - cut fragile trades,
   - extend rare strong trades.

---

## Приоритетное решение для v9

### Primary family
**Derivatives positioning / pressure**

### Fallback family
**Cross-exchange pressure / venue divergence**

### Reserve family
**Microstructure proxies from lower timeframe**

### Only-if-available family
**Options structure**

---

## Какие feature groups строить в v9

После выбора primary family нельзя сразу строить все подряд.
Нужен **minimal viable feature layer**.

---

## Если выбрана Family A: Derivatives positioning / pressure

### Core Group A1. Positioning stress
- `cf_oi_change_1h`
- `cf_oi_change_4h`
- `cf_oi_acceleration`
- `cf_price_oi_divergence_1h`
- `cf_price_oi_divergence_4h`

### Core Group A2. Funding pressure
- `cf_funding_velocity`
- `cf_funding_shock`
- `cf_funding_vs_oi_conflict`
- `cf_funding_vs_price_conflict`

### Core Group A3. Basis / premium
- `cf_basis_level`
- `cf_basis_change_1h`
- `cf_basis_change_4h`
- `cf_premium_index_gap`

### Core Group A4. Liquidation stress
- `cf_liquidation_long_pressure`
- `cf_liquidation_short_pressure`
- `cf_liquidation_imbalance`
- `cf_liquidation_vs_price_dislocation`

---

## Если выбрана Family B: Cross-exchange pressure

### Core Group B1. Venue divergence
- `cf_binance_coinbase_rel_ret_1h`
- `cf_binance_coinbase_rel_ret_4h`
- `cf_spot_perp_divergence_1h`
- `cf_spot_perp_divergence_4h`

### Core Group B2. Venue confirmation
- `cf_cross_venue_breakout_confirm`
- `cf_cross_venue_breakout_fail`
- `cf_venue_lead_score`

### Core Group B3. Venue participation
- `cf_volume_share_shift`
- `cf_spread_expansion_score`
- `cf_cross_venue_dispersion`

---

## Если выбрана Family C: Microstructure proxies

### Core Group C1. Intrabar imbalance
- `cf_micro_close_pressure`
- `cf_micro_rejection_score`
- `cf_micro_followthrough_score`

### Core Group C2. Shock / burst behavior
- `cf_micro_burstiness`
- `cf_micro_shock_cluster`
- `cf_impulse_fail_score`

### Core Group C3. Post-entry fragility
- `cf_post_entry_micro_failure`
- `cf_post_entry_micro_support`
- `cf_post_entry_micro_reversal_risk`

---

## Какой research route должен быть в v9

### Route 1. Data-universe triage
Сначала понять:
- что реально доступно;
- что реально ново;
- что реально materializable.

### Route 2. Overlay-first signal test
На выбранной family сначала тестировать:
- **veto overlays** на trade surface базы.

### Route 3. Early-cut / extension only if justified
Если новая family показывает weak or strong signal на veto route, только тогда разрешать:
- early-cut;
- extension.

### Route 4. Optional directional scout
Разрешается только если:
- новая family показала сильный and coherent signal;
- и есть reason полагать, что она может давать not only overlay value but also independent direction.

---

## Новый label contract v9

Label families v8 остаются правильной рамкой, но теперь они должны быть использованы с новым data universe.

### Overlay labels
- `bad_trade_balanced`
- `bad_trade_conservative`
- `cut_now_balanced`
- `cut_now_conservative`
- `extend_hold_balanced`
- `extend_hold_conservative`

### Optional directional scout labels
Только если family прошла overlay-first:
- `direction_long_balanced`
- `direction_short_balanced`

### Жесткое правило
Direction labels нельзя открывать раньше overlay evidence.

---

## OOS дисциплина v9

### Для overlay route
Trade-level split по entry time:
- `train_trades`
- `selection_oos_trades`
- `lockbox_oos_trades`

### Для optional directional scout
Anchor-level split:
- `train_surface`
- `selection_oos_surface`
- `lockbox_oos_surface`

### Purge gap
Минимум:
- `24h` для trade / holding tasks;
- больше, если новая family требует большего lookback.

### Freeze rules
- family choice freeze before modeling;
- feature set freeze before training;
- threshold freeze before lockbox.

---

## Какие модели обучаем в v9

### Основные модели
- `CatBoost-veto-long-v9`
- `CatBoost-veto-short-v9`

### Условные модели
- `CatBoost-earlycut-long-v9`
- `CatBoost-earlycut-short-v9`
- `CatBoost-extend-long-v9`
- `CatBoost-extend-short-v9`

### Запрещено
- не строить сразу full stack;
- не обучать direction models без overlay evidence;
- не обучать router.

---

## Какой evaluation contract должен быть в v9

### Для data triage
- coverage completeness;
- freshness / continuity;
- null-rate;
- causal audit;
- feature stability;
- bucket lift;
- monotonicity sanity.

### Для overlay models
Selection OOS:
- ROC AUC
- PR AUC
- worst-bucket lift
- precision in veto bucket
- economic effect on selection if flagged trades removed

### Для lockbox
- immutable base vs base+overlay
- delta PnL
- delta PF
- delta DD
- removed trades
- PnL of removed trades
- weak-window repair

### Для optional directional scout
Только после overlay pass:
- standalone OOS
- strict additive vs base
- permissive
- ownership

---

# ЭТАП 0. Closure-audit v8 и фиксация pivot v9

## Цель
Формально закрыть текущий data universe как исчерпанный для old routes.

## Что сделать
1. Открыть новый `research_project` для v9.
2. В `research_record` зафиксировать:
   - v8 overlay route terminally failed on current universe;
   - full-surface directional route closed;
   - почему теперь нужен новый data universe.
3. Зафиксировать immutable reference:
   - `snapshot_id = active-signal-v1`
   - `version = 1`
   - `symbol = BTCUSDT`
   - `anchor = 1h`
   - `execution = 5m`
4. В `research_map` завести оси:
   - data_family
   - overlay_route
   - side
   - feature_group
   - model_stage
   - final_decision
5. Создать open-items register:
   - `data_universe_audit`
   - `primary_family_choice`
   - `fallback_family_choice`
   - `overlay_veto_v9`
   - `overlay_exit_lock`
   - `optional_direction_lock`
   - `final_v9_verdict`

## Критерий завершения этапа

Есть clean project v9 и corrected terminology.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 1. Audit нового data universe

## Цель
Понять, какие новые источники реально доступны и пригодны.

## Что сделать
Проверить availability и quality для:

### Derivatives positioning / pressure
- OI
- funding history
- premium index
- basis
- liquidations
- positioning ratios

### Cross-exchange pressure
- exchange-level spot data
- exchange-level perp data
- venue spreads
- venue volume splits

### Microstructure proxies
- 1m / 5m OHLCV quality
- lower timeframe continuity
- lower timeframe volume quality

### Options structure
- skew
- term structure
- near-dated vs far-dated IV
- options event context

Для каждого источника зафиксировать:
- available / unavailable;
- common window;
- data gaps;
- causal usability;
- novelty score.

## Таблица результатов этапа

| data_family | available | common_window_ok | gaps_ok | causality_ok | novelty_ok | usable_now | notes |
|-------------|-----------|------------------|---------|--------------|------------|------------|-------|
| | | | | | | | |

---

# ЭТАП 2. Выбор primary family и fallback family

## Цель
Не расползаться. Выбрать одну главную ветку.

## Что сделать
1. Оценить families по:
   - availability
   - novelty
   - causal usability
   - interpretability
   - overlay relevance
2. Выбрать:
   - `primary_family`
   - `fallback_family`
3. Остальные семьи формально locked.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 3. Feature contract и materialization для primary family

## Цель
Построить minimal viable feature layer для новой family.

## Что сделать
Для primary family:
1. записать feature contract;
2. validate;
3. publish;
4. materialize;
5. проверить coverage;
6. проверить causality;
7. проверить simple bucket sanity.

### Жесткое ограничение
Не строить больше 12 features в первой волне.
Нужен компактный сильный набор.

## Таблица результатов этапа

| feature_name | group | validated | published | materialized | warmup_ok | coverage_ok | bucket_sanity_ok | notes |
|--------------|-------|-----------|-----------|--------------|-----------|-------------|------------------|-------|
| | | | | | | | | |

---

# ЭТАП 4. Trade-surface atlas augmentation

## Цель
Привязать новый data universe к trade surface базы.

## Что сделать
1. Обогатить atlas базы новыми features на:
   - entry time;
   - first 1h / 3h if possible;
   - exit time if relevant.
2. Проверить, где новая family strongest:
   - long / short;
   - worst-decile trades;
   - stop-loss cluster;
   - weak-window cluster;
   - time-limit cluster.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 5. Veto-route plausibility on new family

## Цель
Понять, дает ли новая family meaningful veto signal.

## Что сделать
Обучить:
- `v9-veto-long-balanced`
- `v9-veto-short-balanced`

При необходимости:
- conservative variants, но только если balanced уже looks alive.

Смотреть:
- ROC AUC
- PR AUC
- worst-bucket lift
- precision in veto bucket
- simulated selection improvement if flagged trades removed

## Таблица результатов этапа

| model_id | side | val_auc | test_auc | val_precision | test_precision | veto_bucket_ok | selection_economic_effect_ok | promoted_to_lockbox | notes |
|----------|------|---------|----------|---------------|----------------|----------------|------------------------------|---------------------|-------|
| | | | | | | | | | |

---

# ЭТАП 6. Lockbox test for veto route

## Цель
Понять, улучшает ли новая family базу на реальном OOS.

## Что сделать
1. Freeze thresholds on selection OOS.
2. Прогнать:
   - immutable base on lockbox;
   - base + veto overlay on same lockbox.
3. Посчитать:
   - delta PnL;
   - delta PF;
   - delta DD;
   - removed trades;
   - PnL of removed trades;
   - weak-window repair;
   - ownership.

## Таблица результатов этапа

| overlay_id | side | lockbox_delta_pnl | lockbox_delta_pf | lockbox_delta_dd | removed_trades | pnl_of_removed_trades | ownership_positive | final_decision | notes |
|------------|------|-------------------|------------------|------------------|----------------|-----------------------|--------------------|----------------|-------|
| | | | | | | | | | |

---

# ЭТАП 7. Только если veto route проходит — early-cut и extension decision gate

## Цель
Не открывать дополнительные ветки без права на это.

## Что сделать
1. Если veto route passed:
   - решить, есть ли evidence for early-cut;
   - решить, есть ли evidence for extension.
2. Открыть максимум одну дополнительную ветку:
   - либо early-cut,
   - либо extension.

## Таблица результатов этапа

| route | unlock_condition_met | opened | why_opened_or_locked | notes |
|-------|----------------------|--------|----------------------|-------|
| | | | | |

---

# ЭТАП 8. Optional route execution

## Цель
Выполнить только ту дополнительную route, которая реально заслужила открытие.

## Что сделать
Если открыта `early_cut`:
- train selection model;
- test on lockbox;
- compare vs base.

Если открыта `extension`:
- train selection model;
- test on lockbox;
- compare vs base.

## Таблица результатов этапа

| route | selection_quality_ok | lockbox_quality_ok | delta_vs_base_meaningful | final_decision | notes |
|-------|----------------------|--------------------|--------------------------|----------------|-------|
| | | | | | |

---

# ЭТАП 9. Если primary family fail — fallback family triage only

## Цель
Не оставлять цикл без страховки, но и не раздувать его.

## Что сделать
Если primary family провалилась:
1. Не обучать полный стек заново.
2. Для fallback family сделать только:
   - availability check,
   - 4–6 core features,
   - atlas augmentation,
   - one veto plausibility model.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 10. Финальный вердикт по v9

## Цель
Получить честную квалификацию microstructure-route.

## Возможные исходы

### Исход A. `SUCCESS`
Microstructure family дала рабочий overlay route.

### Исход B. `PARTIAL`
Есть watchlist-quality result, но не полноценный add-on.

### Исход C. `FAILED`
Даже microstructure family не пережила честный lockbox overlay test.

## Критерий завершения этапа

Есть final route classification и понятный next step.

## Итоговая таблица

| Поле | Значение |
|------|----------|
| | |
