# План дальнейших исследований для v7

## Статус перед стартом v7

**База, которую не трогаем:** `active-signal-v1@1`.

Это immutable reference.
Запрещено:
- менять базу;
- подбирать ей часы, дни, пороги, исполнение;
- выбирать feature set, label scheme, horizon, threshold или архитектуру по lockbox-окну;
- объявлять улучшением все, что не прошло matched OOS comparison с базой на том же окне;
- возвращаться к regression-предсказанию future PnL;
- возвращаться к IV-family как к самостоятельному directional layer.

---

## Что уже доказано и что это значит для v7

### Жесткие выводы из v4
1. Low-friction пространство в старом информационном слое **исчерпано**.
2. Static IV и IV dynamics дают режим волатильности, но не направление.
3. Classifier transitions сами по себе не являются новым информационным слоем.

### Жесткие выводы из v5
1. Путь `custom features + CatBoost` сам по себе **не отвергнут**, но его первая реализация была ошибочной:
   - модель учили на future PnL;
   - selection выглядел прилично;
   - real lockbox OOS это не подтвердил.
2. CatBoost regression на future PnL — **тупик**.

### Что реально показал v6
1. Поворот к classification был правильным.
2. Data readiness для нового слоя оказалась лучше, чем ожидалось:
   - BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT на 1h доступны;
   - funding и expiry events доступны.
3. Удалось материализовать **10 из 24** новых directional features.
4. Но исполнение цикла осталось **незавершенным**:
   - протестирован только `long 12h`;
   - `short` side не закрыта;
   - `18h` horizon не закрыт;
   - из planned feature families полноценно в бой пошли только части `Leadership-Breadth` и `Path-Asymmetry`;
   - был drift между исходной идеей barrier labels и фактически использованной схемой `net_pnl>0`.

### Главный вывод для v7
**Model route пока не опровергнут.**
Опровергнут только один частный маршрут:

- `long`
- `12h`
- частично материализованные `Leadership-Breadth` и `Path-Asymmetry`
- слабая label discipline

Поэтому v7 не должен начинать новый широкий поиск.
Он должен:
1. **закрыть долги v6**;
2. **починить label contract**;
3. **достроить минимально необходимый directional feature layer**;
4. **проверить short side и 18h horizon**;
5. только потом давать итог по model route.

---

## Главная цель v7

Довести до честного решения вопрос:

> может ли новый directional feature layer + CatBoost classification дать новый additive long и/или short sleeve поверх `active-signal-v1@1`, если:
> - labels заданы правильно,
> - tested sides закрыты полностью,
> - horizons закрыты полностью,
> - feature layer не обрезан случайно?

---

## Главные принципы v7

### Принцип 1. Сначала закрыть незавершенное, потом расширять
До любого нового большого feature brainstorming нужно закрыть:
- short side;
- 18h horizon;
- label contract;
- missing critical features.

### Принцип 2. Label contract нельзя больше менять по ходу
В v7 labels должны быть frozen до training.

Разрешены только две схемы:

#### Scheme A. Balanced barrier
Для long:
- если в пределах горизонта цена сначала достигает достаточно хорошего ап-сайд барьера,
- и не успевает раньше ударить в защитный даун-сайд барьер,
- это positive long label.

Для short — зеркально вниз.

#### Scheme B. Conservative barrier
То же самое, но с более требовательным ап-сайд/даун-сайд требованием и с большим запасом после учета издержек.

**Важно:**
`net_pnl>0` больше не использовать как primary label scheme.

### Принцип 3. Сначала tail-value, потом общий AUC
Для trading важны:
- top-decile lift;
- precision в top bucket;
- realized OOS edge в high-confidence region.

### Принцип 4. Short-first приоритет
Так как:
- long 12h уже провалился;
- breadth deterioration и failure geometry логически сильнее именно на short side;
- база уже достаточно хороша на longs;

в v7 первым приоритетом должен быть **short 18h** и **short 12h**, а уже потом new long tests.

### Принцип 5. Только компактные feature packs
Нужен не весь зоопарк, а **критический directional core**.

---

## Критический directional core для v7

### Core Group 1. Short deterioration / failure
Главный приоритет:
- `cf_eth_btc_rel_ret_3h`
- `cf_eth_btc_rel_ret_12h`
- `cf_alt_breadth_down_6h`
- `cf_breadth_dispersion_6h`
- `cf_btc_lag_vs_breadth`
- `cf_upside_failure_pressure_6h`
- `cf_failed_move_reject_6h`
- `cf_rollover_before_daily`
- `cf_fast_slow_regime_gap`
- `cf_event_failure_combo`

### Core Group 2. Long recovery / reclaim
Второй приоритет:
- `cf_alt_breadth_up_6h`
- `cf_downside_failure_pressure_6h`
- `cf_failed_move_reclaim_6h`
- `cf_reversal_quality_6h`
- `cf_recovery_before_daily`
- `cf_range_escape_quality_12h`

### Core Group 3. Confirmation / context
Только как context:
- `cf_expiry_leadership_conflict`
- `cf_funding_breadth_conflict`
- `cf_event_regime_asymmetry`
- `cf_state_instability_score`

---

## Какие новые sleeves должен пытаться найти v7

## Short sleeves

### S1. Leadership-led deterioration short
- ETH / leaders уже слабее BTC;
- breadth down ухудшается;
- BTC еще не fully broke;
- fast/slow regime gap негативен;
- failure geometry подтверждает.

### S2. Failed breakout short
- был breakout up;
- breakout не удержался;
- upper-wick pressure высокий;
- breadth не подтверждает;
- leadership ухудшается.

### S3. Event-deterioration short
- рядом funding / expiry;
- leadership divergence и failure geometry ухудшаются;
- fast horizon уже rolled over.

## Long sleeves

### L1. Reclaim recovery long
- downside failure уже был;
- breadth recovers;
- reclaim quality высокая;
- short-term regime улучшается до daily.

### L2. Broad confirmation long
- breadth up широкая;
- ETH / leaders сильнее BTC;
- range escape вверх удерживается;
- failure geometry не мешает.

---

## Какой model contract должен быть в v7

### Основные модели
- `M1_short_cls_12h`
- `M2_short_cls_18h`
- `M3_long_cls_18h`

### Допустимые дополнительные модели
- `M4_short_cls_weak_window`
- `M5_long_cls_weak_window`

Но только если базовые directional models показали слабый, но содержательный signal.

### Что не делать
- не обучать multiclass model;
- не строить router до тех пор, пока хотя бы один standalone sleeve не прошел OOS;
- не возвращаться к PnL regression.

---

## Какой evaluation contract должен быть в v7

### Уровень 1. Model-quality on selection OOS
- ROC AUC
- PR AUC
- precision in top decile
- lift vs base rate
- calibration sanity

### Уровень 2. Economic sanity on selection OOS
- mean realized outcome top bucket
- monotonicity across buckets
- нет ли edge только в одном случайном хвосте на микросэмпле

### Уровень 3. Lockbox standalone
Только frozen model и frozen threshold.

### Уровень 4. Strict additive integration vs base
Главный критерий ценности.

### Уровень 5. Permissive integration
Только дополнительная диагностика.

---

# ЭТАП 0. Closure-audit v6 и фиксация инвариантов v7

## Цель
Сначала честно зафиксировать, что именно в v6 осталось недоделанным.

## Что сделать
1. Открыть новый `research_project` для v7.
2. Записать в `research_record`:
   - что в v6 закрыто;
   - что осталось pending;
   - какие выводы считаются окончательными.
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
   - horizon
   - label_scheme
   - model_stage
   - final_decision
5. Создать open-items register минимум для:
   - `v6_short_12h`
   - `v6_short_18h`
   - `v6_long_18h`
   - `barrier_label_contract`
   - `critical_directional_core`
   - `matched_oos_with_base`
   - `final_model_route_verdict`

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| project_id | |
| base_snapshot | |
| symbol | |
| anchor_timeframe | |
| execution_timeframe | |
| v6_closure_audit_node | |
| open_items_registered | |
| atlas_dimensions_defined | |

---

# ЭТАП 1. Freeze barrier-label contract

## Цель
Убрать дрейф между планом и исполнением, который был в v6.

## Что сделать
1. Формально утвердить только два label schemes:
   - `balanced_barrier`
   - `conservative_barrier`
2. Для каждого horizon:
   - `12h`
   - `18h`
3. Для каждой стороны:
   - `long_success`
   - `short_success`
4. Зафиксировать словами:
   - какое движение считается успехом;
   - какой adverse move ломает идею;
   - как учитываются издержки;
   - как учитываются ambiguous cases.
5. После фиксации labels запретить любые изменения до конца selection-stage.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| decision_node | |
| balanced_barrier_long | |
| balanced_barrier_short | |
| conservative_barrier_long | |
| conservative_barrier_short | |
| horizons | |
| ambiguous_cases | |
| net_pnl_scheme | |

---

# ЭТАП 2. Достроить critical directional core

## Цель
Не строить все 24 features, а достроить именно те, без которых v7 будет недосказан.

## Что сделать

### 2.1. Short-first missing features
Приоритетно построить:
- `cf_failed_move_reject_6h`
- `cf_rollover_before_daily`
- `cf_fast_slow_regime_gap`
- `cf_event_failure_combo`

### 2.2. Long support features
Затем:
- `cf_failed_move_reclaim_6h`
- `cf_reversal_quality_6h`
- `cf_range_escape_quality_12h`
- `cf_recovery_before_daily`

### 2.3. Context features
Только после этого:
- `cf_expiry_leadership_conflict`
- `cf_funding_breadth_conflict`
- `cf_event_regime_asymmetry`
- `cf_state_instability_score`

### 2.4. Coverage and causality check
Для каждого feature:
- validate;
- publish;
- materialize;
- inspect coverage;
- проверить warmup;
- проверить causality.

## Таблица результатов этапа

| feature_name | priority_group | validated | published | materialized | warmup_ok | coverage_ok | included_in_v7_core | notes |
|--------------|----------------|-----------|-----------|--------------|-----------|-------------|---------------------|-------|
| | | | | | | | | |

### Общая таблица окна

| Поле | Значение |
|------|----------|
| total_new_features | |
| published_all | |
| materialized_in_dataset | |
| coverage_issues | |
| next_step | |

---

# ЭТАП 3. Dataset split и modeling surfaces

## Цель
После завершения core-features окончательно freeze modeling surfaces.

## Что сделать
1. Подтвердить frozen windows:
   - train
   - selection OOS
   - lockbox OOS
2. Подготовить три поверхности:
   - `full_surface`
   - `base_absent_surface`
   - `weak_window_surface` — только если понадобится позже
3. Зафиксировать, что:
   - выбор pack делается по selection OOS;
   - выбор threshold делается по selection OOS;
   - lockbox untouched.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| frozen_windows | |
| surfaces_planned | |
| dataset_build_op | |
| selection_rule | |
| modeling_surface_status | |

---

# ЭТАП 4. Short-first plausibility stage

## Цель
Проверить short-side models на plausibility.

## Что сделать
Обучить и проверить:
- `M1_short_12h`
- `M2_short_18h`

## Таблица результатов этапа

| setup_id | feature_pack | horizon | label_scheme | auc_ok | pr_ok | top_decile_ok | economic_plausibility_ok | promoted_to_full_training | notes |
|----------|--------------|---------|--------------|--------|-------|---------------|--------------------------|---------------------------|-------|
| | | | | | | | | | |

---

# ЭТАП 5. Long 18h plausibility stage

## Цель
Вернуться к long side только после того, как short side просмотрена честно.

## Что сделать
Проверить только:
- `18h` horizon
- `balanced_barrier`
- `conservative_barrier`

Использовать packs:
- `Leadership-Breadth`
- `Path-Asymmetry`
- `Long-Recovery-Core`

## Таблица результатов этапа

| setup_id | feature_pack | horizon | label_scheme | auc_ok | pr_ok | top_decile_ok | economic_plausibility_ok | promoted_to_full_training | notes |
|----------|--------------|---------|--------------|--------|-------|---------------|--------------------------|---------------------------|-------|
| | | | | | | | | | |

---

# ЭТАП 6. Full training и frozen model candidates

## Цель
Обучить финальные модели для кандидатов, прошедших plausibility.

## Таблица результатов этапа

| model_id | side | feature_pack | label_scheme | target_horizon | params_frozen | best_iteration | selection_oos_ok | frozen_for_lockbox | notes |
|----------|------|--------------|--------------|----------------|---------------|----------------|------------------|--------------------|-------|
| | | | | | | | | | |

---

# ЭТАП 7. Превращение лучших моделей в tradeable sleeves

## Цель
Перевести модели в tradeable sleeves.

## Таблица результатов этапа

| sleeve_id | source_model | side | threshold_tier | threshold_frozen_before_lockbox | expected_density | ready_for_oos_backtest | notes |
|-----------|--------------|------|----------------|---------------------------------|------------------|------------------------|-------|
| | | | | | | | |

---

# ЭТАП 8. Matched OOS comparison против базы

## Цель
Проверить sleeves на lockbox OOS.

## Таблица результатов этапа

| run_type | sleeve_id | window | pnl | trades | pf | max_dd | avg_trade | median_trade | notes |
|----------|-----------|--------|-----|--------|----|--------|-----------|--------------|-------|
| | | | | | | | | | |

---

# ЭТАП 9. Ownership, cannibalization и proof of new entries

## Цель
Доказать, что model добавляет новые прибыльные сделки.

## Таблица результатов этапа

| candidate | overlap_rate | truly_new_trades | replaced_base_trades | pnl_of_new_trades | ownership_positive | weak_window_repair_seen | final_decision | notes |
|-----------|-------------|------------------|----------------------|-------------------|--------------------|-------------------------|----------------|-------|
| | | | | | | | | |

---

# ЭТАП 10. Финальный вердикт по v7

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| project_id | |
| total_features_built | |
| label_contract | |
| models_trained | |
| short_12h_verdict | |
| short_18h_verdict | |
| long_18h_verdict | |
| model_route_verdict | |
| classification_route | |
| base_strategy | |
| next_recommended_direction | |

---

# Короткий операционный чек-лист для агента


- [ ] база `active-signal-v1@1` ни разу не менялась;
- [ ] v6 closure-audit записан;
- [ ] label contract frozen и не дрейфовал;
- [ ] critical directional core достроен;
- [ ] short 12h и short 18h закрыты;
- [ ] long 18h закрыт;
- [ ] selection OOS использован правильно;
- [ ] no models reached lockbox или locktest completed;
- [ ] classification route получил terminal verdict;
- [ ] финальный отчет по v7 написан.
