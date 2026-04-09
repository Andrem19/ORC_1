# План дальнейших исследований для v6

## Статус перед стартом v6

**База, которую не трогаем:** `active-signal-v1@1`.

Это immutable reference.
Запрещено:
- менять базу;
- подбирать ей часы, дни, пороги, исполнение;
- обучать модель на данных, которые потом используются как финальный OOS;
- выбирать feature set, target, threshold или архитектуру по lockbox-окну;
- объявлять улучшением все, что не прошло matched OOS comparison с базой на том же окне.

---

## Что уже доказано предыдущими циклами

### Что больше не надо повторять
1. **Сырые event-entry идеи не работают.**
2. **Static high-IV regime не дает directional alpha.**
3. **IV dynamics в простом rule-based виде тоже не дают directional alpha.**
4. **Classifier transitions сами по себе не являются новым информационным слоем.**
5. **CatBoost regression на future PnL не сработал.**
   - выбор цели как непрерывного PnL оказался методологически слабым;
   - selection-метрики выглядели неплохо, но на реальном lockbox OOS модель не пережила trading reality;
   - модель хорошо видела режим и волатильность, но плохо переводила это в направление и реализуемый trade edge.

### Что это значит стратегически
Проблема уже не в "еще одном пороге" и не в "еще одной настройке CatBoost".
Проблема в том, что текущий слой признаков mostly объясняет:
- напряжение рынка;
- волатильность;
- состояние режима;

но слабо объясняет:
- **куда именно** пойдет рынок после издержек;
- где появятся **новые прибыльные long/short entries**.

---

## Главная идея v6

**v6 не отказывается от пути "custom features + models", но меняет его правильно.**

Главный разворот:

1. не предсказывать raw future PnL;
2. перейти к **directional / barrier-based long и short labels**;
3. добавить **новый информационный слой**, а не только переставлять старые IV / classifier features;
4. обучать отдельные CatBoost-модели:
   - `CatBoost-long-classifier`
   - `CatBoost-short-classifier`
5. сравнивать их с базой только через:
   - matched OOS;
   - strict additive;
   - permissive;
   - ownership;
   - cannibalization.

---

## Главная цель v6

Найти и проверить **новые custom features**, которые реально могут давать **новые long и short entries**, затем обучить на них **CatBoost classification models**, проверить их на **selection OOS**, потом на **lockbox OOS**, и сравнить с базой на том же окне.

Главный вопрос цикла:

> могут ли новые directional custom features и CatBoost-classifiers дать новый additive layer поверх `active-signal-v1@1`, или текущий model route тоже исчерпан?

---

## Главные принципы v6

### Принцип 1. Новый информационный слой обязателен
В v6 недостаточно просто обучить CatBoost на слегка измененной версии старых признаков.

Нужны признаки, которые содержат **новую directional information**.

Главный кандидат v6:
- **cross-symbol leadership / breadth / relative strength**

Вторичные кандидаты:
- **path-asymmetry / failure geometry**
- **event-state interactions**
- **regime disagreement features**

### Принцип 2. Не regression, а classification
В v6 модели обучаются не на future PnL, а на:
- long success label;
- short success label.

Label должен быть ближе к реальному trade outcome:
- через horizon;
- через barrier logic;
- через net-of-cost requirement.

### Принцип 3. Separate long и short first
Никаких общих multiclass моделей на старте.

Сначала:
- `M1_long_cls`
- `M2_short_cls`

### Принцип 4. Base-absent — discovery surface, matched OOS — truth surface
Сначала новые features и модели ищутся на `base_absent` surface, чтобы повышать шанс найти новые входы.
Но финальная правда — только matched OOS comparison с базой.

### Принцип 5. Не делать широкую feature inflation
Нужен не огромный зоопарк фич, а **3–4 сильных feature families** с небольшим, но содержательным набором custom features.

### Принцип 6. Не делать широкую HPO
Нужен компактный disciplined search:
- few feature packs;
- few label definitions;
- few CatBoost configs.

---

## Какие новые custom features должны стать ядром v6

Ниже перечислены feature families, которые имеют шанс дать **реально новые directional entries**.

---

## Блок A. Cross-symbol leadership и breadth

Это главный приоритет v6.
Причина простая: это уже **другой информационный слой**, а не вариация IV-family.

Если BTC сам по себе не дает нам directional edge, нужно смотреть:
- кто ведет рынок;
- кто слабеет раньше;
- где breadth подтверждает движение;
- где BTC расходится с соседними risk assets.

### A1. `cf_eth_btc_rel_ret_3h`
Разница между ETHUSDT и BTCUSDT по доходности за 3 часа.

### A2. `cf_eth_btc_rel_ret_12h`
То же, но на более широком окне.

### A3. `cf_leadership_gap_3h`
Разница между коротким relative move лидера и BTC.

### A4. `cf_alt_breadth_up_6h`
Доля наблюдаемых risk symbols, которые закрылись вверх за последние 6 часов.

### A5. `cf_alt_breadth_down_6h`
Симметрично вниз.

### A6. `cf_breadth_dispersion_6h`
Насколько разнонаправленно двигаются risk assets.

### A7. `cf_btc_lag_vs_breadth`
BTC еще не двинулся, но breadth уже ухудшилась или улучшилась.

### A8. `cf_cross_symbol_failure_confirm`
Был ли failure breakout не только у BTC, но и у лидирующего risk asset.

---

## Блок B. Directional path-asymmetry

В v5 path-shape features уже частично были, но их использовали внутри модели, которая училась на плохой цели.
В v6 эти признаки остаются, но уже как часть **directional classification**.

### B1. `cf_upside_failure_pressure_6h`
Суммарное давление неудачных ап-сайд попыток.

### B2. `cf_downside_failure_pressure_6h`
Симметрично для вниз.

### B3. `cf_reversal_quality_6h`
Насколько последние 6 часов выглядят как качественный reversal, а не шумовая отскочка.

### B4. `cf_trend_efficiency_12h`
Насколько движение последних 12 часов было чистым и направленным.

### B5. `cf_range_escape_quality_12h`
Если цена вышла из диапазона, удерживается ли выход.

### B6. `cf_failed_move_reclaim_6h`
После failed downside breakout рынок смог reclaim важную часть диапазона?

### B7. `cf_failed_move_reject_6h`
После failed upside breakout рынок снова уходит вниз?

---

## Блок C. Regime disagreement и temporal sequencing

Не просто classifier state, а **несогласованность горизонтов** и порядок их ухудшения / улучшения.

### C1. `cf_fast_slow_regime_gap`
Насколько краткий horizon уже улучшился / ухудшился, а старший еще нет.

### C2. `cf_regime_turn_score`
Composite score:
- short-term transition,
- alignment drift,
- persistence.

### C3. `cf_recovery_before_daily`
1h и 4h уже улучшаются, а 1d еще нет.

### C4. `cf_rollover_before_daily`
1h и 4h уже портятся, а 1d еще не fully turned.

### C5. `cf_state_instability_score`
Сколько раз regime менялся за недавнее окно.

---

## Блок D. Event-state interactions нового типа

Сырые event triggers не работают, но interactions могут быть полезны.

### D1. `cf_funding_breadth_conflict`
Funding proximity + breadth deterioration / recovery.

### D2. `cf_expiry_leadership_conflict`
Expiry proximity + cross-symbol leadership divergence.

### D3. `cf_event_failure_combo`
Event proximity + failed breakout geometry.

### D4. `cf_event_regime_asymmetry`
Event proximity + fast/slow regime disagreement.

**Важно:**
Это не самостоятельные event signals.
Только interaction features.

---

## Feature packs v6

### Pack 1. Leadership-Breadth
- `cf_eth_btc_rel_ret_3h`
- `cf_eth_btc_rel_ret_12h`
- `cf_leadership_gap_3h`
- `cf_alt_breadth_up_6h`
- `cf_alt_breadth_down_6h`
- `cf_breadth_dispersion_6h`
- `cf_btc_lag_vs_breadth`
- `cf_cross_symbol_failure_confirm`

### Pack 2. Path-Asymmetry
- `cf_upside_failure_pressure_6h`
- `cf_downside_failure_pressure_6h`
- `cf_reversal_quality_6h`
- `cf_trend_efficiency_12h`
- `cf_range_escape_quality_12h`
- `cf_failed_move_reclaim_6h`
- `cf_failed_move_reject_6h`

### Pack 3. Regime-Disagreement
- `cf_fast_slow_regime_gap`
- `cf_regime_turn_score`
- `cf_recovery_before_daily`
- `cf_rollover_before_daily`
- `cf_state_instability_score`

### Pack 4. Hybrid-Event
- `cf_funding_breadth_conflict`
- `cf_expiry_leadership_conflict`
- `cf_event_failure_combo`
- `cf_event_regime_asymmetry`

---

## Какие новые long и short entries мы хотим найти в v6

## Новые long families

### L1. Leadership-led recovery
- breadth уже восстанавливается;
- ETH / leader идет сильнее BTC;
- BTC еще не fully reacted;
- path-asymmetry показывает reclaim and support.

### L2. Failed downside continuation with cross-confirmation
- BTC показал failed downside breakout;
- breadth и лидер подтверждают recovery;
- fast horizon уже улучшился;
- daily еще не fully turned.

### L3. Range escape with broad confirmation
- рынок выходит вверх из сжатия;
- breadth confirms;
- BTC не один идет вверх;
- failure geometry не мешает.

## Новые short families

### S1. Leadership-led deterioration
- BTC еще держится;
- breadth уже сыпется;
- ETH / leader слабее BTC;
- rollover начинается на fast horizons.

### S2. Failed upside breakout with broad weakness
- был breakout up;
- он не удержался;
- upper-wick pressure высокий;
- breadth не подтверждает;
- leader уже слабее BTC.

### S3. Downside release from unstable regime
- state instability высокая;
- slow regime еще не fully turned;
- fast regime already rolling over;
- range escape идет вниз с breadth confirmation.

---

## Каким должен быть label design в v6

В v5 мы ошиблись, пытаясь предсказывать future PnL напрямую.

В v6 label должен быть ближе к trade outcome.

### Long label
Положительный класс, если:
- в пределах horizon цена достигает upper barrier раньше lower barrier;
- или net-of-cost move вверх достаточно велик;
- adverse path не разрушает идею.

### Short label
Симметрично вниз.

### Горизонты
Разрешены только:
- `12h`
- `18h`

### Barrier schemes
Использовать максимум два варианта:
- `balanced`
- `conservative`

### Что запрещено
- большой grid по labels;
- выбор label по lockbox.

---

## OOS дисциплина v6

### Сначала определить common coverage window
Поскольку у нас теперь cross-symbol features, сначала нужно зафиксировать:
- какие symbols реально доступны на 1h;
- у каких есть достаточная история;
- на каком отрезке есть все нужные features.

### Frozen split
Только после этого freeze:
- `train_window`
- `selection_oos_window`
- `lockbox_oos_window`

### Purge gap
Между train и later windows:
- не меньше `24h`

---

## Какие модели обучаем в v6

### M1. `CatBoost-long-cls`
Цель:
- классифицировать перспективные long entries.

### M2. `CatBoost-short-cls`
Цель:
- классифицировать перспективные short entries.

### M3. `CatBoost-long-cls-weak-window`
Разрешена только если:
- обычная long-classifier показывает слабый результат;
- но есть конкретный weak-window pocket, который стоит чинить.

### M4. `CatBoost-short-cls-weak-window`
Аналогично для short.

### Router-модель
Запрещена на старте v6.
Разрешается только если M1 и M2 обе показали meaningful OOS value.

---

## Какие CatBoost configs разрешены

Никакого широкого HPO.

### Минимальный disciplined grid
- depth: `4`, `6`, `8`
- learning_rate: `0.03`, `0.05`
- moderate regularization
- 2–3 seeds максимум

### Что обязательно
- early stopping;
- class weights;
- calibration sanity;
- feature importance;
- SHAP summary;
- frozen model card.

---

## Как переводить probability в сигналы

### Шаг 1
Получить:
- `p_long_success`
- `p_short_success`

### Шаг 2
На selection OOS выбрать максимум 3 threshold tiers:
- conservative
- balanced
- aggressive

### Шаг 3
Freeze final threshold **до открытия lockbox**.

### Шаг 4
Построить реальные sleeves:
- `v6-long-balanced`
- `v6-short-balanced`

---

# ЭТАП 0. Старт v6 и фиксация новой model discipline

## Цель

Открыть новый цикл и сразу зафиксировать, что:
- v6 продолжает feature + model path,
- но с новой методологией.

## Что сделать

1. Открыть новый `research_project` под v6.
2. В `research_record` записать:
   - почему v5 regression route признан неудачным;
   - почему в v6 используется classification;
   - почему нужен новый information layer.
3. Зафиксировать immutable reference:
   - `snapshot_id = active-signal-v1`
   - `version = 1`
   - `symbol = BTCUSDT`
   - `anchor = 1h`
   - `execution = 5m`
4. В `research_map` завести оси:
   - branch
   - feature_family
   - side
   - label_scheme
   - model_stage
   - final_decision
5. Отдельно записать anti-leakage и anti-regression-repeat rules.

## Критерий завершения этапа

Есть новый проект, immutable reference и зафиксированные правила v6.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| project_id | |
| reference | |
| atlas_defined | |
| v5_regression_closed | |
| anti_leakage | |
| classification_approach | |
| cross_symbol_features | |
| notes | |

---

# ЭТАП 1. Data readiness для нового information layer

## Цель

Убедиться, что cross-symbol and breadth route вообще имеет качественные данные.

## Что сделать

1. Проверить готовность 1h datasets минимум для:
   - BTCUSDT
   - ETHUSDT
   - при возможности: SOLUSDT, BNBUSDT
2. Если часть символов не готова, сократить состав, но не ломать дисциплину.
3. Для events проверить:
   - funding
   - expiry
4. Зафиксировать final symbol universe для v6.

## Критерий завершения этапа

Есть frozen symbol universe и понятно, какие features реально можно строить.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| btc_1h_ready | |
| eth_1h_ready | |
| bnb_sol_xrp_1h | |
| final_symbol_universe | |
| funding_events | |
| expiry_events | |
| cross_symbol_limit | |

---

# ЭТАП 2. Feature contract и materialization для v6 feature packs

## Цель

Построить новый directional feature layer.

## Что сделать

Для каждого feature из Pack 1–4:
1. записать contract;
2. validate;
3. publish;
4. materialize;
5. inspect coverage;
6. проверить causality.

## Жесткое правило

Если feature family не materialized нормально, не использовать ее в modeling.

## Критерий завершения этапа

Есть materialized feature packs и общий `common_model_window`.

## Таблица результатов этапа

| feature_name | family | validated | published | materialized | warmup_ok | coverage_ok | notes |
|--------------|--------|-----------|-----------|--------------|-----------|-------------|-------|
| | | | | | | | |

### Общая таблица окна

| Поле | Значение |
|------|----------|
| common_model_window | |
| total_active_columns | |
| cross_symbol_columns | |
| no_leakage_detected | |
| deployable_to_runtime | |

---

# ЭТАП 3. Dataset contract и label design без утечек

## Цель

Зафиксировать modeling dataset до обучения.

## Что сделать

1. Определить:
   - `train_window`
   - `selection_oos_window`
   - `lockbox_oos_window`
2. Зафиксировать:
   - `balanced label`
   - `conservative label`
   - `12h`
   - `18h`
3. Построить два типа datasets:
   - full surface
   - base-absent surface
4. Зафиксировать, что selection выбирает:
   - feature pack
   - target horizon
   - threshold tier

А lockbox только проверяет frozen choice.

## Критерий завершения этапа

Dataset split и labels полностью frozen.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| train_window | |
| selection_oos_window | |
| lockbox_oos_window | |
| purge_gap | |
| label_scheme | |
| target_horizon | |
| surface | |
| features_used | |
| dataset_id | |
| total_rows | |
| missing_inputs | |

---

# ЭТАП 4. Быстрая plausibility-проверка feature packs

## Цель

Не обучать все подряд. Сначала отсеять пустые packs.

## Что сделать

1. Для каждого pack:
   - обучить минимальную long-classifier и short-classifier;
   - посмотреть ranking value на selection OOS;
   - проверить economic plausibility top-decile / top-bucket.
2. Для long оставить максимум 2 pack setups.
3. Для short оставить максимум 2 pack setups.

## Критерий завершения этапа

Есть shortlist setups для полноценного обучения.

## Таблица результатов этапа

| setup_id | side | feature_pack | label_scheme | target_horizon | selection_quality_ok | economic_plausibility_ok | promoted_to_full_training | notes |
|----------|------|--------------|--------------|----------------|----------------------|--------------------------|---------------------------|-------|
| | | | | | | | | |

---

# ЭТАП 5. Обучение CatBoost long/short classifiers

## Цель

Обучить disciplined candidate models на лучших setups.

## Что сделать

### Для long
- `M1_long_A`
- при необходимости `M1_long_B`

### Для short
- `M2_short_A`
- при необходимости `M2_short_B`

Для каждой модели записать:
- feature pack;
- label scheme;
- target horizon;
- params;
- best iteration;
- selection OOS quality;
- feature importance;
- SHAP summary.

## Критерий завершения этапа

Есть frozen long/short model candidates.

## Таблица результатов этапа

| model_id | side | feature_pack | label_scheme | target_horizon | params_frozen | best_iteration | selection_oos_ok | frozen_for_lockbox | notes |
|----------|------|--------------|--------------|----------------|---------------|----------------|------------------|--------------------|-------|
| | | | | | | | | | |

---

# ЭТАП 6. Превращение моделей в tradeable sleeves

## Цель

Перевести классификационные вероятности в реальные long / short sleeves.

## Что сделать

1. Получить:
   - `p_long_success`
   - `p_short_success`
2. На selection OOS выбрать threshold tiers.
3. Freeze final threshold до lockbox.
4. Построить:
   - `v6-long-balanced`
   - `v6-short-balanced`

## Критерий завершения этапа

Есть tradeable sleeves для OOS backtest.

## Таблица результатов этапа

| sleeve_id | source_model | side | threshold_tier | threshold_frozen_before_lockbox | expected_density | ready_for_oos_backtest | notes |
|-----------|--------------|------|----------------|---------------------------------|------------------|------------------------|-------|
| | | | | | | | |

---

# ЭТАП 7. Matched OOS comparison против базы

## Цель

Проверить, есть ли у моделей реальная ценность на одинаковом окне с базой.

## Что сделать

Для каждого sleeve прогнать:

### 7.1. Standalone on selection OOS
Sanity-check only.

### 7.2. Standalone on lockbox OOS
Главный test.

### 7.3. Strict additive integration vs base on lockbox
Главный критерий.

### 7.4. Permissive integration vs base on lockbox
Только как дополнительная диагностика.

### 7.5. Base runs on same windows
Прогнать immutable base отдельно на:
- selection OOS
- lockbox OOS

## Критерий завершения этапа

Есть matched OOS comparison:
- base,
- model standalone,
- base + model.

## Таблица результатов этапа

| run_type | sleeve_id | window | pnl | trades | pf | max_dd | avg_trade | median_trade | notes |
|----------|-----------|--------|-----|--------|----|--------|-----------|--------------|-------|
| | | | | | | | | | |

---

# ЭТАП 8. Ownership, cannibalization и proof of new entries

## Цель

Доказать, что модель добавляет новые прибыльные long/short сделки.

## Что сделать

Для лучшего long и лучшего short sleeve посчитать:
1. overlap с базой;
2. truly new trades;
3. replaced base trades;
4. PnL новых сделок;
5. ownership прибыли;
6. weak-window repair effect, если он появился.

## Решения

### `PROMOTE`
Только если:
- lockbox standalone не пустой;
- strict additive положительный;
- новые сделки прибыльны;
- ownership положительный;
- matched OOS delta к базе убедительный.

### `WATCHLIST`
Если:
- локальная ценность есть;
- но mainline-quality не доказана.

### `REJECT`
Если:
- OOS слабый;
- strict additive отрицательный;
- ownership отрицательный;
- прироста к базе нет.

## Критерий завершения этапа

Каждый best sleeve имеет terminal decision.

## Таблица результатов этапа

| candidate | overlap_rate | truly_new_trades | replaced_base_trades | pnl_of_new_trades | ownership_positive | weak_window_repair_seen | final_decision | notes |
|-----------|-------------|------------------|----------------------|-------------------|--------------------|-------------------------|----------------|-------|
| | | | | | | | | |

---

# ЭТАП 9. Финальный вердикт по v6

## Цель

Получить единый итог по новой model route.

## Возможные исходы

### Исход A. Success
Есть хотя бы один promoted candidate:
- long,
- short,
- или оба.

### Исход B. Partial
Есть watchlist candidate, но нет полноценного add-on.

### Исход C. Failed
Ни одна модель не пережила clean OOS и strict additive.

## Критерий завершения этапа

Есть единый итог по v6.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| v6_approach | |
| cross_symbol_features | |
| catboost_long_roc_auc | |
| catboost_short_trained | |
| oos_backtest_done | |
| final_verdict | |
| improvement_vs_v5 | |
| deployable_to_runtime | |

---

# Короткий операционный чек-лист для агента


- [ ] база `active-signal-v1@1` ни разу не менялась;
- [ ] v5 regression route formally closed;
- [ ] symbol universe frozen (BTCUSDT + ETHUSDT);
- [ ] v6 cross-symbol features построены causally;
- [ ] materialization и coverage проверены;
- [ ] train / selection / lockbox frozen до training;
- [ ] purge gap соблюден;
- [ ] long классификатор обучен;
- [ ] short классификатор обучен;
- [ ] thresholds frozen до lockbox;
- [ ] база прогнана на том же OOS окне;
- [ ] strict additive comparison выполнена;
- [ ] permissive comparison выполнена;
- [ ] ownership и cannibalization посчитаны;
- [ ] model sleeve имеет terminal decision;
- [ ] финальный отчет по v6 написан.

---

# Финальный отчет v6

### 1. Что было целью v6

### 2. Что сделано

### 3. Главный вывод v6

### 4. Итог по v1-v6
| План | Подход | Результат |
|------|--------|-----------|
| v1 | | |
| v2 | | |
| v3 | | |
| v4 | | |
| v5 | | |
| v6 | | |

### 5. Что делать дальше
