# План дальнейших исследований для v11

## Статус перед стартом v11

**База, которую не трогаем:** `active-signal-v1@1`.

Это immutable reference.
Запрещено:
- менять базу;
- подбирать ей часы, дни, пороги, исполнение;
- менять microstructure features до завершения аудита v10;
- менять veto thresholds до завершения воспроизводимости и leakage-аудита;
- объявлять успехом результат v10 до повторной проверки;
- смешивать в одном прогоне:
  - проверку воспроизводимости,
  - проверку утечек,
  - новые идеи,
  - новый directional route.

---

## Почему v11 должен быть посвящен только аудиту v10

По итогам v10 зафиксирован очень сильный результат:

- materialization micro-core завершена и выглядит стабильной;
- `cf_micro_imbalance_1h` подтвержден как strongest feature;
- simple bucket/threshold baselines уже показали value;
- short и long veto models показали очень высокие metrics;
- lockbox table показала улучшение против базы: меньше сделок, выше PnL, выше PF, ниже DD.

Но одновременно в v10 есть несколько вещей, которые **нельзя принимать на веру без отдельного аудита**:

1. В lockbox-сравнении baseline указан как `active-signal-v1@3`, хотя весь цикл формально строился вокруг immutable reference `@1`. Это нужно проверить и устранить.
2. У veto models слишком высокие AUC/PR (`~0.95`), что требует отдельной проверки на leakage или hidden shortcut.
3. `short_veto`, `long_veto` и `combined_veto` в lockbox table имеют одинаковый итоговый delta, что выглядит подозрительно и требует разнесения по отдельным runs.
4. Нужно отдельно доказать, что:
   - база прогнана на **том же самом окне**,
   - overlay прогнан на **том же самом окне**,
   - `base + overlay` действительно дает improvement,
   - improvement не является скрытым дублем или ошибкой сборки snapshot.

### Главный вывод для v11

**v11 — это не план поиска новой альфы.**
Это план **жесткой forensic-проверки v10**.

Если v10 пройдет этот цикл, тогда microstructure route можно считать реально доказанным.

Если не пройдет, это тоже сильный и полезный результат:
мы поймем, что v10 был ложноположительным успехом.

---

## Главная цель v11

Получить честный ответ на четыре вопроса:

### Вопрос 1
Воспроизводится ли результат v10 **с нуля**, без скрытых артефактов окружения?

### Вопрос 2
Нет ли в v10:
- leakage по features,
- leakage по labels,
- leakage по threshold selection,
- leakage по lockbox usage,
- leakage через snapshot assembly?

### Вопрос 3
Действительно ли `base + overlay` лучше базы **на том же окне** и против **того же reference**?

### Вопрос 4
Не является ли overlay скрытой формой каннибализма, дубляжа или искажения учета, а действительно улучшает базу за счет удаления плохих сделок?

---

## Главные принципы v11

### Принцип 1. Никаких новых идей
В v11 запрещено:
- строить новые features;
- открывать new directional scout;
- открывать early-cut;
- менять labels ради улучшения результата.

### Принцип 2. Reproduce first
Сначала результат v10 должен быть воспроизведен в максимально чистой форме.

### Принцип 3. Audit before interpretation
Сначала:
- leakage audit,
- reference audit,
- window audit,
- snapshot audit,
- cannibalization audit.

Только потом интерпретация.

### Принцип 4. Same-window truth only
Все сравнения делать только на **одинаковом окне**:
- immutable base,
- overlay standalone diagnostics,
- `base + overlay`.

### Принцип 5. Separate runs, not aggregated stories
Отдельно проверять:
- short-only veto,
- long-only veto,
- combined veto.

Если все три дают одинаковый delta, это надо объяснить, а не принять.

### Принцип 6. Simple rule first
Так как v10 сам показал, что simple threshold rule может быть достаточен, v11 обязан проверить именно rule-based overlay first, а потом уже model-backed overlay.

### Принцип 7. Promotion only after red-team audit
Статус `PROMOTE` сохраняется только если v10 переживает весь forensic cycle.

---

## Что именно нужно проверить в v11

### Блок A. Reference integrity
Нужно проверить:
- почему в v10 baseline в lockbox table указан как `@3`, а reference по плану `@1`;
- одинаковы ли фактически `@1` и `@3`;
- если нет, весь результат должен быть перепроверен на `@1`.

### Блок B. Window integrity
Нужно проверить:
- одинаковы ли train / selection / lockbox окна во всех runs;
- не было ли повторного использования lockbox для выбора thresholds;
- не был ли overlay threshold скорректирован после просмотра lockbox.

### Блок C. Feature leakage audit
Нужно проверить для каждого micro-feature:
- что оно считается только по прошлому;
- что 5m→1h alignment causal;
- что на decision time нет доступа к future 5m bars;
- что aggregation window обрезана правильно.

### Блок D. Label leakage audit
Нужно проверить:
- не используют ли veto labels фактически ту же информацию, что потом модель видит почти напрямую;
- нет ли shortcut между label construction и `cf_micro_imbalance_1h`;
- не является ли сверхвысокий AUC следствием label definition, почти совпадающей с threshold on same feature.

### Блок E. Snapshot / overlay assembly audit
Нужно проверить:
- как именно собран `v10-micro-veto-v2@1`;
- где применяются thresholds;
- на каком шаге блокируется trade;
- нет ли ошибки, когда overlay работает post-factum или после knowledge of outcome.

### Блок F. Same-window matched comparison
Нужно отдельно прогнать:
- immutable base `@1`;
- short-only veto over base;
- long-only veto over base;
- combined veto over base;
- все на одном и том же lockbox окне.

### Блок G. Cannibalization / ownership audit
Нужно доказать:
- какие именно trades removed;
- были ли removed trades реально плохими;
- сколько хороших trades removed;
- какой вклад удаления bad trades в delta PnL;
- нет ли скрытого ухудшения coverage в сильных режимах.

---

## Главная структура v11

v11 идет строго по цепочке:

1. closure audit and freeze;
2. exact reproduction of v10;
3. leakage audit;
4. reference/window audit;
5. matched reruns on same window;
6. separate short / long / combined tests;
7. cannibalization and ownership;
8. final verdict:
   - `CONFIRMED`
   - `DOWNGRADED TO WATCHLIST`
   - `REJECTED`

---

# ЭТАП 0. Freeze и постановка forensic-cycle

## Цель

Зафиксировать, что v11 — это audit-only цикл.

## Что сделать

1. Открыть новый `research_project` для v11.
2. В `research_record` зафиксировать:
   - v10 claimed success;
   - why that success needs forensic validation;
   - what specific red flags must be checked.
3. Зафиксировать immutable reference:
   - `snapshot_id = active-signal-v1`
   - `version = 1`
   - `symbol = BTCUSDT`
   - `anchor = 1h`
   - `execution = 5m`
4. В `research_map` завести оси:
   - audit_block
   - side
   - run_type
   - leakage_type
   - comparison_window
   - final_verdict
5. Создать open-items register:
   - `reference_integrity`
   - `window_integrity`
   - `feature_leakage_audit`
   - `label_leakage_audit`
   - `snapshot_assembly_audit`
   - `reproduce_v10`
   - `same_window_base_runs`
   - `short_long_combined_separation`
   - `cannibalization_ownership`
   - `final_v11_verdict`

## Критерий завершения этапа

Есть отдельный audit-project v11 и жёсткий freeze режима.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 1. Exact reproduction of v10 from scratch

## Цель

Сначала воспроизвести результат v10 без изменений и без "ручной памяти".

## Что сделать

1. Пересобрать dataset и micro-core features.
2. Пересчитать baselines:
   - bucket veto
   - threshold veto
3. Пересобрать best short/long veto models.
4. Пересобрать overlay snapshot.
5. Повторить reported lockbox result.

## Правило

Нельзя ничего подправлять по ходу.
Сначала только проверить: воспроизводится или нет.

## Критерий завершения этапа

Есть ответ:
- `reproduced exactly`
- `partially reproduced`
- `not reproduced`

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 2. Reference integrity audit

## Цель

Убрать главное противоречие v10: `@1` против `@3`.

## Что сделать

1. Открыть и сравнить:
   - `active-signal-v1@1`
   - `active-signal-v1@3`
2. Проверить:
   - логика;
   - параметры;
   - execution;
   - filters;
   - exits;
   - symbol and window coverage.
3. Если snapshots не идентичны:
   - объявить v10 comparison invalid;
   - rerun all v10 comparisons strictly against `@1`.

## Критерий завершения этапа

Есть ясный verdict:
- `@1 == @3`
- или `@1 != @3`.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 3. Window integrity audit

## Цель

Проверить, что все сравнения действительно делались на одном окне и без подглядывания.

## Что сделать

1. Зафиксировать exact windows для:
   - train
   - selection OOS
   - lockbox
2. Проверить:
   - thresholds selected only on selection OOS;
   - lockbox never used to choose thresholds or labels;
   - same lockbox used for base and overlay.
3. Отдельно проверить:
   - не менялось ли окно между short / long / combined.

## Критерий завершения этапа

Есть ответ:
- window discipline valid / invalid.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 4. Feature leakage audit

## Цель

Проверить, что micro-features действительно причинно чистые.

## Что сделать

Для каждого micro-feature проверить:
- source timeframe;
- exact aggregation cutoff;
- whether any future 5m bars leak into 1h decision;
- warmup and null handling;
- dataset materialization alignment.

Особое внимание:
- `cf_micro_imbalance_1h`
- `cf_micro_close_pressure_1h`
- `cf_micro_shock_cluster_1h`

## Дополнительные red-team tests

1. Shift test:
   - сдвинуть feature на +1 bar назад/вперед и проверить, не сохраняется ли слишком хороший результат странным образом.
2. Null-mask sanity:
   - проверить, не является ли сам факт наличия/отсутствия feature hidden signal.
3. Timestamp audit:
   - руками проверить несколько decision timestamps.

## Критерий завершения этапа

Есть verdict:
- leakage absent
- suspicious
- confirmed leakage

## Таблица результатов этапа

| feature_name | causal_cutoff_ok | shift_test_ok | null_mask_ok | timestamp_audit_ok | leakage_verdict | notes |
|--------------|------------------|---------------|--------------|--------------------|-----------------|-------|
| | | | | | | |

---

# ЭТАП 5. Label leakage and shortcut audit

## Цель

Проверить, не стала ли модель слишком хорошей потому, что label почти совпадает с feature.

## Что сделать

1. Проверить построение:
   - `bad_trade_pnl_balanced`
   - `fragile_trade_mae_balanced`
   - `bad_trade_composite_balanced`
2. Измерить связь между:
   - label membership
   - `cf_micro_imbalance_1h`
3. Проверить:
   - нет ли почти прямого threshold shortcut;
   - не объясняется ли почти весь label одной колонкой.
4. Прогнать ablations:
   - model with only `cf_micro_imbalance_1h`
   - model without `cf_micro_imbalance_1h`
   - model with shuffled labels
   - model with shuffled feature

## Критерий завершения этапа

Есть ответ:
- labels valid
- labels too shortcut-prone
- labels invalid

## Таблица результатов этапа

| test_name | result | verdict | notes |
|-----------|--------|---------|-------|
| | | | |

---

# ЭТАП 6. Snapshot assembly audit

## Цель

Понять, как именно veto overlay был внедрен в snapshot и не нарушает ли это causal order.

## Что сделать

1. Inspect overlay snapshot logic.
2. Проверить:
   - когда именно считается veto decision;
   - какой feature value доступен на decision moment;
   - не происходит ли blocking already opened trade retroactively;
   - нет ли mixed logic between long/short veto.
3. Отдельно проверить:
   - short-only snapshot
   - long-only snapshot
   - combined snapshot

## Критерий завершения этапа

Есть causal verdict по overlay assembly.

## Таблица результатов этапа

| snapshot_variant | decision_timing_ok | causal_blocking_ok | side_isolation_ok | mixed_logic_absent | verdict | notes |
|------------------|--------------------|--------------------|-------------------|-------------------|---------|-------|
| | | | | | | |

---

# ЭТАП 7. Same-window matched reruns

## Цель

Проверить базу и overlay на одном и том же окне без двусмысленности.

## Что сделать

Обязательно прогнать на **том же lockbox окне**:

1. Immutable base `@1`
2. Short-only veto overlay over base
3. Long-only veto overlay over base
4. Combined veto overlay over base

Для каждого run посчитать:
- trades
- net PnL
- PF
- DD
- win rate
- removed trades
- removed winners

## Критерий завершения этапа

Есть matched same-window comparison, которому можно доверять.

## Таблица результатов этапа

| run_id | route | window | trades | pnl | pf | max_dd | win_rate | removed_trades | removed_winners | notes |
|--------|-------|--------|--------|-----|----|--------|----------|----------------|-----------------|-------|
| | | | | | | | | | | |

---

# ЭТАП 8. Cannibalization and ownership audit

## Цель

Доказать, что overlay реально улучшает систему, а не маскирует проблему.

## Что сделать

1. Для short-only, long-only, combined:
   - список removed trades;
   - PnL removed trades;
   - доля removed losers;
   - доля removed winners;
   - какой side выиграл больше.
2. Проверить:
   - weak-window repair;
   - regime-specific loss of good coverage;
   - не убивает ли overlay сильные profitable clusters.
3. Посчитать ownership:
   - improvement driven by removed bad trades,
   - or by hidden interaction with base mechanics.

## Критерий завершения этапа

Есть честный ownership/cannibalization verdict.

## Таблица результатов этапа

| route | removed_loser_share | removed_winner_share | pnl_removed_trades | weak_window_repair | coverage_loss_risk | ownership_positive | cannibalization_verdict | notes |
|-------|---------------------|----------------------|--------------------|-------------------|-------------------|--------------------|------------------------|-------|
| | | | | | | | | |

---

# ЭТАП 9. Negative controls and robustness

## Цель

Проверить, что результат не является случайной артефактной подгонкой.

## Что сделать

1. Placebo thresholds:
   - чуть хуже threshold
   - чуть лучше threshold
   - opposite-side threshold
2. Small perturbation test:
   - threshold +/- small epsilon
3. Time-slice robustness:
   - first half of lockbox
   - second half of lockbox
4. Side-perturbation:
   - apply long threshold to short
   - apply short threshold to long

## Критерий завершения этапа

Понятно, устойчив ли результат или он держится на одной случайной точке.

## Таблица результатов этапа

| test_name | result | robustness_ok | notes |
|-----------|--------|---------------|-------|
| | | | |

---

# ЭТАП 10. Финальный вердикт по v11

## Цель

Получить окончательное решение по результату v10.

## Возможные исходы

### Исход A. CONFIRMED
Результат v10 воспроизводится, leakage нет, same-window comparison корректен, overlay реально улучшает базу, cannibalization acceptable.

### Исход B. DOWNGRADED TO WATCHLIST
Signal живой, но:
- есть методологическая слабость,
- или improvement fragile,
- или same-window reruns слабее claimed result.

### Исход C. REJECTED
Есть leakage, mismatched reference, невоспроизводимость или improvement исчезает на честном rerun.

## Критерий завершения этапа

Есть final forensic verdict по microstructure overlay.

## Итоговая таблица

| Поле | Значение |
|------|----------|
| | |
