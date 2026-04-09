# План дальнейших исследований для v8

## Статус перед стартом v8

**База, которую не трогаем:** `active-signal-v1@1`.

Это immutable reference.
Запрещено:
- менять базу;
- подбирать ей часы, дни, пороги, исполнение;
- повторно открывать full-surface directional classification на тех же семействах `leadership / breadth / path-asymmetry / regime features`;
- возвращаться к regression на future PnL;
- выбирать feature set, label scheme, horizon, threshold или архитектуру по lockbox-окну;
- объявлять улучшением все, что не прошло matched OOS comparison с базой на том же окне.

---

## Что уже доказано и что это значит для v8

### Жесткие выводы из v4
1. Старый low-friction слой исчерпан.
2. Static IV и IV dynamics дают режим волатильности, но не направление.
3. Classifier transitions сами по себе не являются новым информационным слоем.

### Жесткие выводы из v5
1. Первая model-route реализация была методологически слабой.
2. CatBoost regression на future PnL — тупик.

### Жесткие выводы из v6
1. Поворот к classification был правильным.
2. Но цикл остался недозакрыт:
   - short side не была закрыта;
   - 18h не был закрыт;
   - label contract дрейфовал;
   - feature layer был обрезан.

### Жесткие выводы из v7
1. После исправления label contract и закрытия short / 18h classification-путь в его full-surface directional виде **провалился**.
2. Все основные модели дали качество около случайности:
   - short side: AUC около `0.50`;
   - long 18h: лишь немного выше случайности и не tradeable.
3. Это означает не "плохо подобрали threshold", а более сильную вещь:
   - текущие feature families **не содержат directional information** для задачи "предсказать новый long/short entry на всем surface".

### Главный стратегический вывод для v8
**Продолжать full-surface search больше нельзя.**

Но это не означает, что весь feature+model route мертв полностью.
Это означает, что направление нужно сменить:

- не искать новый standalone directional sleeve на всем рынке;
- а искать **base-conditional alpha**:
  - где база сама по себе уже отобрала хорошие candidate trades,
  - а модель и новые features могут:
    - убрать худшие сделки,
    - дать более умный early-cut,
    - дать более умный hold-extension,
    - починить слабые участки внутри уже качественной trade surface.

---

## Главная идея v8

**v8 — это pivot от "predict market direction everywhere" к "improve the quality of already good base trades".**

Это принципиально другой тип задачи.

Вместо вопроса:

> "можем ли мы предсказать новый long/short entry на всем anchor-surface?"

мы задаем более узкий и гораздо более правдоподобный вопрос:

> "можем ли мы на surface уже исполненных базой сделок предсказать:
> - какие сделки надо veto,
> - какие нужно cut раньше,
> - какие можно держать дольше,
> чтобы улучшить базу без ретюнинга ее логики?"

---

## Главная цель v8

Построить и проверить **новые trade-context features** и **base-conditional models**, которые работают **на trade surface базы**, а не на всем рынке подряд, и понять, могут ли они дать один из трех эффектов:

1. **Entry veto alpha** — убрать худшие сделки базы;
2. **Early-cut alpha** — быстрее закрывать явно плохие сделки;
3. **Hold-extension alpha** — дольше держать редкие сильные сделки, если это улучшает итог.

---

## Почему это логичнее после v7

Потому что:
- full-surface directional prediction уже провалился;
- база сама по себе уже делает сильную селекцию;
- на более узкой conditional surface signal-to-noise ratio обычно выше;
- предсказывать **качество уже выбранного трейда** проще, чем предсказывать рынок с нуля;
- это не нарушает принцип immutable base, потому что база остается reference, а исследуется только overlay.

---

## Главные принципы v8

### Принцип 1. Base logic immutable, overlay only
Мы не меняем сигналы базы.
Мы исследуем только overlays поверх уже существующих сделок базы.

### Принцип 2. Сначала veto, потом exit overlays
Первый приоритет:
- **entry-veto** на trade surface базы.

Только если veto route дает signal value или хотя бы weak but meaningful signal, открываем:
- early-cut;
- hold-extension.

### Принцип 3. Не искать новые standalone entries
В v8 больше не строим новый independent directional sleeve на всем surface.

### Принцип 4. Trade-level datasets, не anchor-level
Вся работа v8 должна строиться вокруг уже исполненных сделок базы:
- entry time;
- entry context;
- post-entry path;
- exit reason;
- MFE / MAE;
- path shape after entry.

### Принцип 5. Targets должны быть привязаны к улучшению базы
Никаких абстрактных красивых labels.
Каждый label должен отвечать на практический вопрос:
- стоит ли этот trade пропустить;
- стоит ли его порезать раньше;
- стоит ли его держать дольше.

### Принцип 6. Одна основная ветка одновременно
Порядок:
1. closure-audit classification route;
2. base-trade surface atlas;
3. veto models;
4. только потом exit models.

---

## Что теперь считается закрытым окончательно

Ниже это запрещено переоткрывать в прежнем виде:

- raw event entries;
- static high-IV directional ideas;
- IV dynamics standalone directional ideas;
- classifier-transition route как самостоятельный слой;
- CatBoost regression на future PnL;
- full-surface directional classification на:
  - leadership/breadth,
  - path-asymmetry,
  - regime-disagreement,
  - их простых комбинациях.

Это закрыто не потому, что "мало попробовали", а потому что цикл v4–v7 уже дал системные отрицательные результаты.

---

## Новый объект исследования v8: base-trade surface

v8 должен работать не на всех anchor-bar, а на surface уже исполненных сделок базы.

Для каждой сделки базы нужно собрать:

- side (`long` / `short`);
- entry timestamp;
- entry price;
- exit timestamp;
- exit reason;
- realized PnL;
- MFE;
- MAE;
- time-in-trade;
- path after entry по первым:
  - 1h
  - 3h
  - 6h
  - 12h
- event context at entry;
- cross-symbol context at entry;
- path-shape context before entry;
- regime alignment context at entry.

---

## Новые feature families v8

Теперь feature engineering должен отвечать не на вопрос "куда пойдет рынок", а на вопрос "насколько качественен уже выбранный базой trade".

### Блок A. Entry quality / fragility features

#### A1. `cf_entry_stretch_6h`
Насколько вход базы происходит уже после растянутого движения.

#### A2. `cf_entry_range_position_12h`
Где вход находится внутри недавнего диапазона.

#### A3. `cf_entry_exhaustion_score`
Комбинация:
- wick pressure,
- close off extremes,
- recent stretch,
- failed continuation signs.

#### A4. `cf_signal_freshness`
Насколько рано база вошла относительно своего собственного setup-window.

#### A5. `cf_confluence_density`
Сколько supporting contexts было одновременно:
- regime alignment,
- cross-symbol confirmation,
- event-neutrality,
- low fragility.

### Блок B. Post-entry risk features

#### B1. `cf_first_1h_followthrough`
Что произошло в первый час после entry:
- продолжение;
- stall;
- immediate adverse move.

#### B2. `cf_first_3h_adverse_excursion`
Размер раннего adverse move.

#### B3. `cf_first_3h_support_reclaim`
Если это long — был ли reclaim после слабости.
Для short — был ли reject after bounce.

#### B4. `cf_post_entry_instability`
Насколько path после входа рваный и нестабильный.

#### B5. `cf_early_failure_signature`
Есть ли ранний паттерн, статистически похожий на будущий stop-loss / bad time-exit.

### Блок C. Trade extension / premature exit features

#### C1. `cf_mfe_before_time_limit`
Какой MFE trade успел показать до time_limit.

#### C2. `cf_trend_persistence_at_time_limit`
Сохраняется ли directional pressure в момент базового time-limit.

#### C3. `cf_exit_context_strength`
Насколько в момент базового выхода контекст все еще силен.

#### C4. `cf_reversal_risk_at_time_limit`
Есть ли признаки, что надо закрыться сразу, а не держать дольше.

#### C5. `cf_extension_quality_score`
Composite:
- trend persistence,
- cross-symbol confirmation,
- no reversal pressure,
- stable path.

### Блок D. Cross-symbol and event confirmation on trade surface

#### D1. `cf_trade_eth_btc_confirmation`
Подтверждает ли ETH направление сделки базы на entry и в первые часы после entry.

#### D2. `cf_trade_breadth_confirmation`
Подтверждает ли breadth сделку базы.

#### D3. `cf_trade_event_headwind`
Есть ли рядом funding / expiry context, который мешает сделке.

#### D4. `cf_trade_event_tailwind`
Есть ли event context, который помогает сделке.

---

## Какие overlay routes проверяем в v8

### Route 1. Entry-veto overlay
Не искать новые сделки, а убрать **худшие сделки базы**.

#### Veto Label A. Bad trade label
Положительный класс:
- trade входит в худший хвост по realized outcome;
- или дает отрицательный результат ниже разумного порога;
- или попадает в "bad outcome class" по MAE / PnL.

#### Veto Label B. Fragile trade label
Положительный класс:
- trade выглядит как статистически плохой по сочетанию:
  - low MFE,
  - high MAE,
  - weak followthrough,
  - bad exit path.

### Route 2. Early-cut overlay
Если после входа быстро формируется failure signature, trade надо закрыть раньше базового exit.

### Route 3. Hold-extension overlay
Некоторые сделки базы обрываются слишком рано time-limit'ом.
Нужно понять, есть ли редкий класс trades, которые стоит держать дольше.

---

## Новый label contract v8

### Label Family 1. Veto labels
- `bad_trade_balanced`
- `bad_trade_conservative`

### Label Family 2. Early-cut labels
- `cut_now_balanced`
- `cut_now_conservative`

### Label Family 3. Hold-extension labels
- `extend_hold_balanced`
- `extend_hold_conservative`

### Жесткие правила
- labels frozen до training;
- thresholds не подбирать по lockbox;
- использовать только небольшое число схем.

---

## OOS дисциплина v8

### Новый тип split
Так как это trade-level research, split делается по времени **entry timestamp** сделки.

### Frozen windows
Использовать:
- `train_trades`
- `selection_oos_trades`
- `lockbox_oos_trades`

### Purge gap
Между окнами:
- не меньше максимального relevant holding horizon.

Если базовый hold около 18h, purge делать минимум 24h.

### Matched comparison
База и overlay сравниваются только на тех же самых trade windows.

---

## Какие модели обучаем в v8

### M1. `CatBoost-veto-long`
На surface long trades базы.

### M2. `CatBoost-veto-short`
На surface short trades базы.

### M3. `CatBoost-earlycut-long`
Разрешена позже.

### M4. `CatBoost-earlycut-short`
Разрешена позже.

### M5. `CatBoost-extend-long`
Разрешена позже.

### M6. `CatBoost-extend-short`
Разрешена позже.

### Что запрещено
- не обучать overlay модели, пока trade-level datasets не собраны корректно;
- не смешивать entry-veto и exit-extension в одном label;
- не строить router раньше времени.

---

## Какой evaluation contract должен быть в v8

### Для veto models
Selection OOS:
- ROC AUC
- PR AUC
- lift in worst-trade bucket
- precision in veto bucket
- economic effect if top-risk trades removed

### Для early-cut models
Selection OOS:
- improvement vs baseline exit on same trades
- net delta after fees
- DD improvement
- does it cut real losers more often than future winners

### Для extension models
Selection OOS:
- added incremental PnL on extension subset
- extension hit rate
- does it worsen DD materially

### Финальный truth test
Только:
- matched lockbox OOS vs immutable base
- same notionals
- same trade window
- strict overlay evaluation

---

# ЭТАП 0. Closure-audit v7 и фиксация инвариантов v8

## Цель
Сначала честно зафиксировать terminal verdict v7 и новый pivot.

## Что сделать
1. Открыть новый `research_project` для v8.
2. В `research_record` зафиксировать:
   - classification route on full surface = terminally failed;
   - почему route закрыт;
   - почему v8 pivot идет на trade-quality overlays.
3. Зафиксировать immutable reference:
   - `snapshot_id = active-signal-v1`
   - `version = 1`
   - `symbol = BTCUSDT`
   - `anchor = 1h`
   - `execution = 5m`
4. В `research_map` завести оси:
   - overlay_route
   - side
   - feature_family
   - label_family
   - model_stage
   - final_decision
5. Создать open-items register:
   - `trade_surface_atlas`
   - `veto_label_contract`
   - `veto_long`
   - `veto_short`
   - `earlycut_route_lock`
   - `extension_route_lock`
   - `final_overlay_verdict`

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| project_id | |
| v7_verdict_node | |
| base_snapshot | |
| symbol | |
| anchor_timeframe | |
| execution_timeframe | |
| atlas_dimensions | |
| open_items | |

---

# ЭТАП 1. Построить trade-surface atlas immutable base

## Цель
Перейти от anchor-level thinking к trade-level thinking.

## Что сделать
1. Собрать все сделки immutable base за полный доступный период.
2. Для каждой сделки записать:
   - side
   - entry time
   - exit time
   - realized PnL
   - MFE
   - MAE
   - time-in-trade
   - exit reason
   - weak-window membership
   - context at entry
   - path in first 1h / 3h / 6h
3. Построить разрезы:
   - long vs short
   - winners vs losers
   - stop-loss vs time-limit vs take-profit
   - weak quarters
   - weakest pockets
4. Отдельно выделить:
   - worst-decile trades
   - fragile trades
   - underheld winners

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| run_id | |
| data_window | |
| total_trades | |
| long_trades | |
| short_trades | |
| net_pnl | |
| win_rate | |
| profit_factor | |
| max_drawdown_pct | |
| exit_tp | |
| exit_sl | |
| exit_time_limit | |
| milestone_node | |

---

# ЭТАП 2. Freeze overlay label contracts

## Цель
Задать правильные labels до обучения.

## Что сделать

### 2.1. Veto labels
Определить:
- `bad_trade_balanced`
- `bad_trade_conservative`

### 2.2. Early-cut labels
Определить:
- `cut_now_balanced`
- `cut_now_conservative`

### 2.3. Extension labels
Определить:
- `extend_hold_balanced`
- `extend_hold_conservative`

### 2.4. Ambiguity policy
Зафиксировать:
- что делать с near-zero trades;
- что делать с mixed trades;
- какой material improvement считается meaningful.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| decision_node | |
| bad_trade_balanced_long | |
| bad_trade_balanced_short | |
| bad_trade_conservative_long | |
| bad_trade_conservative_short | |
| ambiguity_policy | |
| material_improvement | |

---

# ЭТАП 3. Feature contract для trade-quality overlays

## Цель
Построить новый feature layer уже на trade surface базы.

## Что сделать
Для feature families A–D:
1. записать contract;
2. validate;
3. publish;
4. materialize;
5. проверить warmup / coverage / causality.

### Приоритет v8
Сначала строить features для veto route:
- `cf_entry_stretch_6h`
- `cf_entry_range_position_12h`
- `cf_entry_exhaustion_score`
- `cf_signal_freshness`
- `cf_confluence_density`
- `cf_trade_eth_btc_confirmation`
- `cf_trade_breadth_confirmation`
- `cf_trade_event_headwind`

### Второй приоритет
Post-entry features:
- `cf_first_1h_followthrough`
- `cf_first_3h_adverse_excursion`
- `cf_post_entry_instability`
- `cf_early_failure_signature`

### Третий приоритет
Exit-time features:
- `cf_mfe_before_time_limit`
- `cf_trend_persistence_at_time_limit`
- `cf_exit_context_strength`
- `cf_reversal_risk_at_time_limit`
- `cf_extension_quality_score`

## Таблица результатов этапа

| feature_name | family | priority | validated | published | materialized | warmup_ok | coverage_ok | notes |
|--------------|--------|----------|-----------|-----------|--------------|-----------|-------------|-------|
| | | | | | | | | |

---

# ЭТАП 4. Dataset split и modeling surfaces для overlays

## Цель
Зафиксировать trade-level OOS split до обучения.

## Что сделать
1. Разбить сделки по entry timestamp:
   - train_trades
   - selection_oos_trades
   - lockbox_oos_trades
2. Собрать surfaces:
   - `all_base_trades`
   - `long_base_trades`
   - `short_base_trades`
   - `weak_window_trades` if needed
3. Зафиксировать:
   - выбор feature set по selection OOS only;
   - выбор thresholds по selection OOS only.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| dataset_id | |
| total_rows | |
| train_window | |
| validation_window | |
| test_window | |
| target_mean | |
| features_used | |
| label_contract | |
| split_method | |

---

# ЭТАП 5. Veto-route plausibility stage

## Цель
Понять, можно ли надежно распознавать худшие сделки базы.

## Что сделать

### Для short trades
Обучить:
- `CatBoost-veto-short-balanced`
- `CatBoost-veto-short-conservative`

### Для long trades
Обучить:
- `CatBoost-veto-long-balanced`
- `CatBoost-veto-long-conservative`

Смотреть:
- ROC AUC
- PR AUC
- worst-decile lift
- precision in veto bucket
- economic effect if veto top-risk trades removed on selection OOS

## Таблица результатов этапа

| model_id | side | label_family | val_auc | test_auc | val_precision | test_precision | veto_bucket_ok | selection_economic_effect_ok | promoted_to_lockbox | notes |
|----------|------|--------------|---------|----------|---------------|----------------|----------------|------------------------------|---------------------|-------|
| | | | | | | | | | | |

---

# ЭТАП 6. Lockbox test for veto overlays

## Цель
Понять, может ли veto route реально улучшить базу на lockbox.

## Что сделать
1. Для лучших veto models:
   - freeze thresholds;
   - применить veto overlay на lockbox.
2. Прогнать:
   - immutable base on lockbox;
   - base + veto overlay on same lockbox.
3. Посчитать:
   - delta PnL;
   - delta PF;
   - delta DD;
   - trades removed;
   - PnL removed trades;
   - effect by weak windows.

## Таблица результатов этапа

| overlay_id | side | lockbox_delta_pnl | lockbox_delta_pf | lockbox_delta_dd | removed_trades | pnl_of_removed_trades | weak_window_repair | final_decision | notes |
|------------|------|-------------------|------------------|------------------|----------------|-----------------------|-------------------|----------------|-------|
| | | | | | | | | | |

---

# ЭТАП 7. Только если veto route показывает signal — открыть early-cut route

## Цель
Исследовать post-entry rescue overlays только если есть основания.

## Условие открытия
Early-cut route разрешается, если:
- veto models дали хотя бы weak meaningful signal;
- и из анализа видно, что плохие trades хорошо распознаются именно в post-entry phase.

## Таблица результатов этапа

| model_id | side | selection_cut_quality_ok | selection_economic_effect_ok | promoted_to_lockbox | notes |
|----------|------|--------------------------|------------------------------|---------------------|-------|
| | | | | | |

---

# ЭТАП 8. Только если есть evidence of underheld winners — открыть hold-extension route

## Цель
Проверить, есть ли редкие winners, которых база недодерживает.

## Условие открытия
Extension route разрешается только если:
- trade-surface atlas показал underheld winner cluster;
- selection diagnostics показывают, что extension имеет шанс.

## Таблица результатов этапа

| model_id | side | selection_extension_quality_ok | selection_incremental_pnl_ok | promoted_to_lockbox | notes |
|----------|------|--------------------------------|------------------------------|---------------------|-------|
| | | | | | |

---

# ЭТАП 9. Ownership, cannibalization и финальный overlay verdict

## Цель
Получить честное решение по v8.

## Что сделать
Для каждого surviving overlay route посчитать:
- ownership;
- removed bad trades / saved DD;
- overlap with weak-window clusters;
- whether improvement is real or cosmetic.

## Возможные исходы

### Исход A. Overlay success
Есть хотя бы один route:
- veto,
- early-cut,
- extension,

который улучшает базу на lockbox materially.

### Исход B. Partial
Есть локальный watchlist effect, но нет mainline-quality add-on.

### Исход C. Failed
Даже base-conditional overlays не дают устойчивого улучшения.

## Таблица результатов этапа

| route | ownership_positive | meaningful_delta_vs_base | watchlist_or_promote | final_decision | notes |
|-------|--------------------|--------------------------|----------------------|----------------|-------|
| | | | | | |

### Итоговая таблица

| Поле | Значение |
|------|----------|
| project_id | |
| trade_surface | |
| veto_models_trained | |
| veto_long_result | |
| early_cut_opened | |
| extension_opened | |
| overlay_route_verdict | |
| next_direction | |

---

# Короткий операционный чек-лист для агента


- [ ] база `active-signal-v1@1` ни разу не менялась;
- [ ] v7 terminal verdict formally recorded;
- [ ] trade-surface atlas построен;
- [ ] overlay labels frozen;
- [ ] veto-core features available;
- [ ] trade-level OOS split frozen;
- [ ] veto route проверен на selection;
- [ ] early-cut route открыт только при выполнении условия;
- [ ] extension route открыт только при выполнении условия;
- [ ] matched comparison with base выполнена;
- [ ] ownership посчитан;
- [ ] финальный отчет v8 написан.
