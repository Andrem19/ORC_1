# План дальнейших исследований для v5

## Статус перед стартом v5

**База, которую не трогаем:** `active-signal-v1@1`.

Это immutable reference.

---

## Почему v5 должен идти через custom features и CatBoost

1. **Rule-based поиск уже сильно выжат.**
2. **Новые входы, скорее всего, лежат в нелинейных взаимодействиях нескольких слоев.**
3. **CatBoost здесь логичен, если использовать его дисциплинированно.**

Но в v5 модель не имеет права быть "черным ящиком ради графика".

---

## Главная цель v5

Построить и проверить **новые custom features**, затем обучить **CatBoost long model** и **CatBoost short model**, прогнать на **selection OOS** и затем на **lockbox OOS**.

Итоговый вопрос цикла: дают ли custom-feature + CatBoost модели новый additive layer?

---

## Что v5 обязан закрыть до финала

1. **Feature contract**
2. **Dataset contract**
3. **Model training contract**
4. **Backtest contract**
5. **Matched OOS comparison contract**
6. **Acceptance chain**

---

## Главные принципы v5

### Принцип 1. Сначала feature engineering, потом модель
### Принцип 2. Separate long and short first
### Принцип 3. Train only on past, validate on later, lockbox untouched
### Принцип 4. Модель должна создавать новые входы
### Принцип 5. Не раздувать feature universe
### Принцип 6. Не делать широкий HPO
### Принцип 7. Base-absent — основной режим для discovery

---

## Новые custom features, которые реально могут дать новые входы

---

## Блок A. IV dynamics и IV-price divergence

### A1. `cf_iv_change_1h`
### A2. `cf_iv_change_3h`
### A3. `cf_iv_acceleration`
### A4. `cf_iv_vs_recent_mean`
### A5. `cf_price_iv_divergence_3h`

---

## Блок B. Regime transitions и multi-horizon disagreement

### B1. `cf_cl1h_prev`
### B2. `cf_cl1h_transition_group`
### B3. `cf_cl_alignment_score`
### B4. `cf_transition_persistence`

---

## Блок C. Compression / release и path-shape

### C1. `cf_range_compression_6h_24h`
### C2. `cf_realized_vol_ratio_6h_24h`
### C3. `cf_body_efficiency_6h`
### C4. `cf_upper_wick_pressure_6h`
### C5. `cf_lower_wick_support_6h`
### C6. `cf_close_location_6h`
### C7. `cf_failed_breakout_up_12h`
### C8. `cf_failed_breakout_down_12h`

---

## Блок D. Event x state interactions

### D1. `cf_funding_proximity_signed`
### D2. `cf_expiry_proximity_signed`
### D3. `cf_funding_iv_shock`
### D4. `cf_expiry_compression_release`

---

## Feature packs v5

### Pack 1. IV-core
### Pack 2. Transition-shape
### Pack 3. Failure-event

---

## Как эти признаки должны давать новые long и short entries

### Новые long entries: гипотеза v5

#### Long family L1. Panic-to-stabilization reversal
#### Long family L2. Compression-to-upside release
#### Long family L3. Failed downside continuation

### Новые short entries: гипотеза v5

#### Short family S1. Euphoria-to-failure short
#### Short family S2. Compression-to-downside release
#### Short family S3. Post-event disappointment

---

## Какие модели обучаем

### Модель M1. `CatBoost-long`
### Модель M2. `CatBoost-short`
### Опциональная модель M3. `CatBoost-router`

---

## Целевые переменные и labels

### Базовый подход
### Горизонты
### Выбор финального target

---

## OOS дисциплина и отсутствие утечек данных

### Шаг 1. Сначала определить common coverage window
### Шаг 2. Разделить common window по времени
### Обязательный purge gap
### Что запрещено
### Что разрешено

---

## Как сравнивать модель с базой

### Сравнение только matched-window
### Обязательные режимы сравнения
### Обязательные метрики

---

## Какие CatBoost конфигурации разрешены

### Разрешенный минимальный grid
### Что обязательно

---

## Как превращать модель в реальные входы

### Шаг 1. Получить probability features
### Шаг 2. Калибровать threshold только на validation / selection OOS
### Шаг 3. Построить 2-3 threshold tiers максимум
### Шаг 4. Materialize model as feature

---

# ЭТАП 0. Старт v5 и фиксация model-research discipline

## Цель

Открыть новый цикл и сразу зафиксировать строгую модельную дисциплину.

## Что сделать

1. Открыть новый `research_project` под v5.
2. Записать в `research_record` цели и ограничения v5.
3. Зафиксировать immutable reference.
4. В `research_map` завести оси: branch, feature_family, side, model_stage, final_decision.
5. Отдельно записать anti-leakage rules.

## Критерий завершения этапа

Есть новый проект, immutable reference и зафиксированные model rules.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| project_id | |
| reference | |
| atlas_defined | |
| discipline | |
| anti_leakage | |
| iv_core_published | |
| notes | |

---

# ЭТАП 1. Feature contract для новых custom features

## Цель

Перевести feature ideas в строгий причинный contract.

## Что сделать

Для каждого custom feature записать: описание, raw input, lookback, causality, интерпретацию, side bias.

## Критерий завершения этапа

Есть полный feature contract по всем planned features.

## Таблица результатов этапа

| feature_name | family | side_bias | causality_checked | lookback_fixed | human_interpretation_recorded | notes |
|--------------|--------|-----------|-------------------|----------------|-------------------------------|-------|
| | | | | | | |

---

# ЭТАП 2. Валидация, публикация и materialization custom features

## Цель

Сделать feature set реально доступным для моделей.

## Что сделать

1. Validate, publish, build/refresh dataset, inspect columns.
2. Проверить NaN, coverage, leakage.
3. Зафиксировать `common_model_window`.

## Критерий завершения этапа

Все selected features materialized и готов общий modeling frame.

## Таблица результатов этапа

| feature_name | validated | published | materialized | warmup_ok | coverage_ok | notes |
|--------------|-----------|-----------|--------------|-----------|-------------|-------|
| | | | | | | |

### Общая таблица окна

| Поле | Значение |
|------|----------|
| common_model_window | |
| total_active_columns | |
| iv_core_columns | |
| no_leakage_detected | |

---

# ЭТАП 3. Dataset contract и chronological split без утечек

## Цель

Зафиксировать train / selection OOS / lockbox до обучения моделей.

## Что сделать

1. Выбрать train_window, selection_oos_window, lockbox_oos_window.
2. Убедиться в корректности разделения и purge gap.
3. Построить modeling datasets.
4. Зафиксировать target definitions.

## Критерий завершения этапа

Все modeling windows frozen до начала training.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| train_window | |
| selection_oos_window | |
| lockbox_oos_window | |
| purge_gap | |
| target_horizons | |
| long_label | |
| short_label | |
| surface | |
| feature_count | |

---

# ЭТАП 4. Быстрая plausibility-проверка feature packs на train/selection discipline

## Цель

Не обучать всё подряд. Сначала отсеять feature packs.

## Что сделать

1. Для каждого pack обучить минимальные long/short prototypes.
2. Проверить signal quality на selection OOS.
3. Выбрать максимум 2 лучших setup для long и short.

## Критерий завершения этапа

Есть shortlist модельных setups для full training.

## Таблица результатов этапа

| setup_id | side | feature_pack | target_horizon | selection_signal_quality_ok | economic_plausibility_ok | promoted_to_full_training | notes |
|----------|------|--------------|----------------|-----------------------------|--------------------------|---------------------------|-------|
| | | | | | | | |

---

# ЭТАП 5. Обучение CatBoost-long и CatBoost-short

## Цель

Обучить по одной-две disciplined candidate models на сторону.

## Что сделать

### Для long side
### Для short side

Для каждой модели зафиксировать: feature pack, target, params, best iteration, validation behavior, feature importance, SHAP summary.

## Критерий завершения этапа

Есть frozen model candidates для long и short.

## Таблица результатов этапа

| model_id | side | feature_pack | target_horizon | params_frozen | best_iteration | selection_oos_ok | frozen_for_lockbox | notes |
|----------|------|--------------|----------------|---------------|----------------|------------------|--------------------|-------|
| | | | | | | | | |

---

# ЭТАП 6. Превращение model outputs в реальные сигналы

## Цель

Перевести вероятности моделей в tradeable sleeves.

## Что сделать

1. Получить p_long / p_short.
2. Откалибровать thresholds на selection OOS.
3. Выбрать final threshold tier до открытия lockbox.
4. Построить реальные signal snapshots.

## Критерий завершения этапа

Есть tradeable model sleeves, готовые к OOS backtest.

## Таблица результатов этапа

| sleeve_id | source_model | side | threshold_tier | threshold_frozen_before_lockbox | expected_density | ready_for_oos_backtest | notes |
|-----------|--------------|------|----------------|---------------------------------|------------------|------------------------|-------|
| | | | | | | | |

---

# ЭТАП 7. OOS backtests на matched windows и сравнение с базой

## Цель

Проверить, дает ли модель новый рабочий слой на настоящем OOS.

## Что сделать

### 7.1. Standalone on selection OOS
### 7.2. Standalone on lockbox OOS
### 7.3. Strict additive integration vs base on lockbox
### 7.4. Permissive integration vs base on lockbox
### 7.5. Сравнение с базой на том же окне

## Критерий завершения этапа

Есть matched OOS comparison.

## Таблица результатов этапа

| run_type | sleeve_id | window | pnl | trades | pf | max_dd | avg_trade | median_trade | notes |
|----------|-----------|--------|-----|--------|----|--------|-----------|--------------|-------|
| | | | | | | | | | |

---

# ЭТАП 8. Ownership, cannibalization и proof of new entries

## Цель

Доказать, что модель добавляет новые прибыльные сделки.

## Что сделать

1. Overlap с базой.
2. Truly new trades и replaced base trades.
3. PnL новых сделок и ownership.
4. Weak-window repair effect.

## Решения

### `PROMOTE`
### `WATCHLIST`
### `REJECT`

## Критерий завершения этапа

Каждый model sleeve имеет terminal decision.

## Таблица результатов этапа

| candidate | overlap_rate | truly_new_trades | replaced_base_trades | pnl_of_new_trades | ownership_positive | weak_window_repair_seen | final_decision | notes |
|-----------|-------------|------------------|----------------------|-------------------|--------------------|-------------------------|----------------|-------|
| | | | | | | | | |

---

# ЭТАП 9. Финальный вердикт по model route

## Цель

Получить ясное решение по всему пути "custom features + CatBoost".

## Возможные итоги

### Итог A. Model route success
### Итог B. Model route partial
### Итог C. Model route failed

## Критерий завершения этапа

Есть единый итог по всему v5.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| iv_core_features_published | |
| dataset_built | |
| catboost_long_roc_auc | |
| catboost_short_roc_auc | |
| oos_backtest_done | |
| final_verdict | |

---

# Короткий операционный чек-лист для агента

- [ ] база `active-signal-v1@1` ни разу не менялась;
- [ ] все IV-core custom features построены causally и published;
- [ ] materialization и coverage проверены;
- [ ] train / selection OOS / lockbox frozen до training;
- [ ] purge gap 24h соблюден;
- [ ] long и short модели обучались отдельно;
- [ ] threshold выбран до lockbox;
- [ ] база прогнана на том же окне;
- [ ] strict additive comparison выполнен;
- [ ] permissive comparison выполнен;
- [ ] ownership и cannibalization проверены;
- [ ] каждый model sleeve имеет terminal decision;
- [ ] финальный отчет по v5 написан.

---

# Финальный отчет v5

### 1. Что было целью v5
### 2. Что сделано
### 3. Главный вывод v5
### 4. Что делать в v6
