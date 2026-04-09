# План дальнейших исследований для v12

## Статус перед стартом v12

**База, которую не трогаем:** `active-signal-v1@1`.

Это immutable reference.
Запрещено:
- менять базу;
- подбирать ей часы, дни, пороги, исполнение;
- возвращаться к старым families (`IV`, `breadth`, `path-asymmetry`, `regime-disagreement`) как к новым основным направлениям;
- открывать новый data universe, пока не закрыт microstructure route;
- выдавать microstructure-result за готовый overlay, пока не доказано:
  - что сравнение идет именно против `@1`,
  - что окно одно и то же,
  - что improvement не является артефактом replacement-логики,
  - что thresholds выбраны не по lockbox.

---

## Что уже доказано и что это значит для v12

### Что уже известно из всей траектории исследований
1. Старый low-friction слой исчерпан.
2. `IV`, raw events, classifier transitions и их производные не дали рабочего directional alpha.
3. Full-surface classification на старом universe провалилась.
4. Trade-quality overlays на старом universe тоже не дали устойчивого результата.
5. Первый реально новый живой слой данных — **5m microstructure**.

### Что реально дал v9
- найден новый информационный след;
- strongest feature: `cf_micro_imbalance_1h`;
- signal был promising, но route оставался недоказанным.

### Что реально дал v10
- появился сильный результат на lockbox;
- simple threshold logic уже выглядела лучше сложной истории;
- но результат был интерпретирован как model-based veto overlay слишком рано.

### Что реально доказал v11
1. **Feature leakage по microstructure не найден.**
   `cf_micro_imbalance_1h` выглядит причинно чистым.
2. **Улучшение на том окне реально воспроизводится.**
3. Но:
   - `@1` и `@3` различаются;
   - итоговый route оказался **не overlay**, а **replacement**;
   - заявленные veto models не были фактически интегрированы;
   - thresholds `-0.155` и `0.164` выглядят как hand-tuned и их выбор не задокументирован;
   - итоговый verdict по v11: **DOWNGRADED TO WATCHLIST**, а не PROMOTE.

### Главный вывод для v12
У нас есть **реальный microstructure lead**, но он пока находится в неправильной категории.

Значит v12 должен решить не вопрос "есть ли сигнал", а вопрос:

> что именно мы нашли на самом деле:
> - настоящий overlay поверх `@1`,
> - честный replacement-layer против `@1`,
> - или красивый, но неустойчивый threshold artifact?

---

## Главная идея v12

**v12 — это цикл окончательной квалификации microstructure-route.**

Не новый brainstorming.
Не новый data-universe audit.
Не новые model families.

Задача v12 — довести microstructure-result до одного из трех честных статусов:

### Статус A. `CONFIRMED TRUE OVERLAY`
Microstructure действительно работает как veto/filter поверх **того же** базового снапшота `@1`.

### Статус B. `CONFIRMED REPLACEMENT LAYER`
Microstructure не overlay, но дает честно лучший **новый signal layer** против `@1`, если сравнивать как отдельный snapshot.

### Статус C. `WATCHLIST / REJECT`
Результат держится на hand-tuned thresholds, неверном reference или не переживает честный rerun против `@1`.

---

## Главная цель v12

Получить окончательный ответ на пять вопросов:

### Вопрос 1
Может ли `cf_micro_imbalance_1h` и micro-core дать **настоящий veto overlay** поверх `active-signal-v1@1`?

### Вопрос 2
Если overlay не получается, может ли microstructure-route быть честно оформлен как **replacement-layer**, который стабильно лучше базы `@1` на том же окне?

### Вопрос 3
Как именно должны выбираться thresholds:
- selection-only,
- quantile-based,
- without lockbox peeking?

### Вопрос 4
Есть ли при этом приемлемая каннибализация:
- удаляются ли в основном плохие trades,
- не убиваются ли сильные profitable clusters,
- не теряется ли полезный coverage?

### Вопрос 5
Нужен ли вообще model layer, или простой threshold-rule уже является лучшей формой найденной альфы?

---

## Главные принципы v12

### Принцип 1. Сначала классификация результата, потом развитие
Нельзя развивать route, пока не ясно, что он такое:
- overlay
- replacement
- artifact

### Принцип 2. Reference только `@1`
Все ключевые reruns в v12 обязаны использовать:
- `active-signal-v1@1`

`@3` можно использовать только как вспомогательный исторический reference для сравнения, но не как базу принятия решения.

### Принцип 3. Overlay и replacement — это разные продукты
Нельзя больше смешивать их в одном рассказе.

**Overlay**:
- те же базовые signal IDs,
- та же базовая логика входа,
- microstructure только veto/filter.

**Replacement**:
- новый signal layer,
- новый snapshot,
- честное сравнение с базой как с отдельной системой.

### Принцип 4. Simple rules first
Сначала:
- bucket rule
- threshold rule
- quantile rule

И только если simple rule проигрывает — разрешается logistic / small CatBoost.

### Принцип 5. Short-first, but not short-only
В v11 strongest economics были на short-side, но v12 должен честно закрыть:
- short-only
- long-only
- combined

### Принцип 6. Threshold provenance обязательна
Каждый threshold должен иметь один из двух источников:
- quantile rule;
- selection-OOS search with frozen lockbox.

Никаких "мы просто поставили -0.155".

### Принцип 7. Same-window truth only
На каждом ключевом шаге сравниваются на одном окне:
- base `@1`
- overlay or replacement
- combined if applicable

---

## Что в v12 считается уже закрытым и не должно переоткрываться

Нельзя снова открывать:
- old directional classification route;
- старый overlay-route на old universe;
- model-based veto story из v10 в ее прежнем виде;
- новые macro/data families до завершения microstructure qualification.

---

## Какую структуру должен иметь v12

v12 идет двумя строго разделенными ветками.

### Ветка 1. `TRUE OVERLAY TRACK`
Проверяем, можно ли сделать **реальный veto/filter** поверх базовых сигналов `@1`.

### Ветка 2. `HONEST REPLACEMENT TRACK`
Если true overlay не работает или оказывается слабым, тогда проверяем, не является ли найденный microstructure-route честно лучшим replacement-layer против `@1`.

**Обе ветки нельзя смешивать.**

---

# ЭТАП 0. Freeze терминологии и correction log

## Цель

Сначала исправить язык и рамку, чтобы дальше не путать overlay с replacement.

## Что сделать

1. Открыть новый `research_project` для v12.
2. В `research_record` записать:
   - что v11 downgraded result to watchlist;
   - что v10 result mischaracterized:
     - not model-based veto overlay,
     - but rule-based microstructure replacement candidate.
3. Зафиксировать immutable reference:
   - `snapshot_id = active-signal-v1`
   - `version = 1`
   - `symbol = BTCUSDT`
   - `anchor = 1h`
   - `execution = 5m`
4. В `research_map` завести оси:
   - track
   - side
   - route_type
   - threshold_origin
   - comparison_window
   - final_decision
5. Создать open-items register:
   - `true_overlay_track`
   - `replacement_track`
   - `threshold_provenance`
   - `same_window_ref1_runs`
   - `short_long_combined_split`
   - `cannibalization_audit`
   - `model_need_assessment`
   - `final_v12_verdict`

## Критерий завершения этапа

Есть clean project v12 и corrected terminology.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 1. Rebuild exact base `@1` comparison frame

## Цель

Перестроить все ключевые сравнения строго против `@1`.

## Что сделать

1. Поднять exact base snapshot `@1`.
2. Зафиксировать exact lockbox window, used in v10/v11.
3. Прогнать immutable base `@1` отдельно на этом окне.
4. Отдельно сохранить:
   - trades list
   - timestamps
   - side split
   - PnL
   - PF
   - DD
   - win rate
5. Сохранить этот run как единственный reference-run для v12.

## Критерий завершения этапа

Есть один canonical `base_ref1_lockbox_run`.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 2. Threshold provenance and selection discipline

## Цель

Перевести текущие hand-tuned thresholds в честную процедуру выбора.

## Что сделать

1. Зафиксировать candidate threshold families для `cf_micro_imbalance_1h`:
   - quantile-based long/short cutoffs;
   - fixed numeric cutoffs;
   - symmetric cutoffs;
   - asymmetric cutoffs.
2. Разделить окна:
   - threshold selection window
   - lockbox verification window
3. На selection window выбрать максимум 3 candidate threshold sets:
   - conservative
   - balanced
   - aggressive
4. Freeze thresholds **до** lockbox rerun.
5. Задокументировать происхождение каждого threshold.

## Жесткое правило

В v12 запрещено использовать threshold, происхождение которого нельзя объяснить.

## Критерий завершения этапа

Есть frozen threshold sets with provenance.

## Таблица результатов этапа

| threshold_set_id | side | origin_type | selection_window_used | frozen_before_lockbox | cutoff_long | cutoff_short | notes |
|------------------|------|-------------|------------------------|-----------------------|-------------|--------------|-------|
| | | | | | | | |

---

# ЭТАП 3. Build true overlay implementation over `@1`

## Цель

Проверить, можно ли реализовать microstructure именно как veto/filter, а не как replacement.

## Что сделать

1. Построить **true overlay** variant:
   - базовые signal IDs и базовая логика entry остаются теми же;
   - microstructure применяется только как allow/block gate.
2. Отдельно собрать:
   - short-only overlay
   - long-only overlay
   - combined overlay
3. Проверить на уровне snapshot assembly:
   - signal IDs base preserved;
   - entry timing preserved;
   - trade blocked only at decision moment;
   - no retroactive replacement.
4. Сохранить три separate overlay snapshots.

## Критерий завершения этапа

Есть честные true-overlay snapshots.

## Таблица результатов этапа

| snapshot_id | side_mode | base_signal_ids_preserved | causal_blocking_ok | replacement_absent | ready_for_backtest | notes |
|-------------|-----------|---------------------------|--------------------|--------------------|--------------------|-------|
| | | | | | | |

---

# ЭТАП 4. Simple-rule overlay tests on selection window

## Цель

Понять, работает ли true overlay вообще до lockbox.

## Что сделать

Для каждого frozen threshold set прогнать на selection window:
- short-only true overlay
- long-only true overlay
- combined true overlay

Считать:
- delta PnL vs base
- delta PF vs base
- delta DD vs base
- removed trades
- removed winners
- removed losers
- veto precision

## Правило продвижения

В lockbox идут только те true-overlay variants, которые на selection window:
- дают meaningful delta,
- не режут слишком много winners,
- не превращаются в pure coverage collapse.

## Критерий завершения этапа

Есть shortlist true-overlay candidates.

## Таблица результатов этапа

| candidate_id | side_mode | threshold_set | selection_delta_pnl | selection_delta_pf | selection_delta_dd | removed_trades | removed_winner_share | promoted_to_lockbox | notes |
|--------------|-----------|---------------|---------------------|--------------------|--------------------|----------------|---------------------|---------------------|-------|
| | | | | | | | | | |

---

# ЭТАП 5. True overlay lockbox test against base `@1`

## Цель

Наконец получить честный ответ, работает ли именно overlay поверх нашего базового снапшота.

## Что сделать

Для каждого surviving true-overlay candidate прогнать на **том же lockbox окне**:
- base `@1`
- base `@1` + short-only overlay
- base `@1` + long-only overlay
- base `@1` + combined overlay

Считать:
- trades
- PnL
- PF
- DD
- win rate
- removed trades
- removed winners
- removed losers
- ownership of improvement

## Критерий завершения этапа

Есть final verdict по true-overlay track.

## Таблица результатов этапа

| run_id | side_mode | threshold_set | trades | net_pnl | pf | dd_pct | wr | long | short | verdict | notes |
|--------|-----------|---------------|--------|---------|-----|--------|-----|------|-------|---------|-------|
| | | | | | | | | | | | |

---

# ЭТАП 6. Honest replacement track vs base `@1`

## Цель

Если overlay слабый или невозможен, честно проверить microstructure как replacement-layer.

## Что сделать

1. Построить отдельные replacement snapshots:
   - short-only replacement
   - long-only replacement
   - combined replacement
2. Для них зафиксировать:
   - exact signal logic;
   - threshold logic;
   - difference from base;
   - no claim of being overlay.
3. Прогнать все replacement variants против base `@1` на том же окне.

## Важно

В этой ветке мы **не говорим** "filter bad trades base".
Мы говорим честно: "new signal layer vs base".

## Критерий завершения этапа

Есть final verdict по replacement track.

## Таблица результатов этапа

| run_id | side_mode | threshold_set | trades | pnl_delta | pf_delta | dd_delta | overlap_with_base | better_than_ref1 | final_decision | notes |
|--------|-----------|---------------|--------|-----------|----------|----------|-------------------|------------------|----------------|-------|
| | | | | | | | | | | |

---

# ЭТАП 7. Cannibalization, ownership and regime analysis

## Цель

Понять, полезен ли route по сути, а не только по одной суммарной цифре.

## Что сделать

Для всех surviving variants из overlay/replacement tracks посчитать:
- removed losers share
- removed winners share
- PnL of removed trades
- strongest improved regime
- weakest affected regime
- weak-window repair
- profitable cluster damage
- side asymmetry

Отдельно ответить:
- improvement идет от удаления плохих trades?
- или от полного смены профиля signal coverage?
- не ухудшается ли поведение в важных режимах?

## Критерий завершения этапа

Есть ownership/cannibalization map по всем жизнеспособным вариантам.

## Таблица результатов этапа

| route_variant | removed_loser_share | removed_winner_share | weak_window_repair | profitable_cluster_damage | strongest_regime | weakest_regime | ownership_verdict | notes |
|---------------|---------------------|----------------------|-------------------|---------------------------|------------------|----------------|------------------|-------|
| | | | | | | | | |

---

# ЭТАП 8. Does model layer add anything?

## Цель

Ответить, нужен ли вообще ML сверх simple micro rule.

## Что сделать

Для лучшего surviving route сравнить:
- simple bucket rule
- threshold rule
- logistic baseline
- small CatBoost

Сравнивать только на fixed windows and fixed labels, без wide HPO.

### Вопрос этапа
Есть ли реальная добавка от модели, или лучший продукт — это просто хорошо подобранное microstructure rule?

## Критерий завершения этапа

Есть ясный verdict:
- `simple_rule_enough`
- `model_adds_value`
- `model_not_worth_complexity`

## Таблица результатов этапа

| method | route_type | selection_quality | lockbox_quality | extra_value_vs_threshold | complexity_worth_it | notes |
|--------|------------|-------------------|-----------------|--------------------------|---------------------|-------|
| | | | | | | |

---

# ЭТАП 9. Optional gate: early-cut or directional scout

## Цель

Открывать дальнейшие ветки только если v12 уже что-то подтвердил.

## Условия открытия

### Early-cut gate
Открывается только если:
- true overlay confirmed or strong watchlist;
- видно, что losers распознаются по раннему post-entry micro behavior.

### Directional scout gate
Открывается только если:
- replacement track confirmed or strong watchlist;
- monotonicity and regime behavior coherent;
- signal живет не только как quality filter, но и как entry timing layer.

Открыть максимум одну ветку.

## Критерий завершения этапа

Есть lock/unlock decision по дальнейшему развитию.

## Таблица результатов этапа

| route | unlock_condition_met | opened | why_opened_or_locked | notes |
|-------|----------------------|--------|----------------------|-------|
| | | | | |

---

# ЭТАП 10. Финальный вердикт по v12

## Цель

Получить честную квалификацию microstructure-route.

## Возможные исходы

### Исход A. `CONFIRMED TRUE OVERLAY`
Microstructure реально работает как veto/filter поверх `@1`.

### Исход B. `CONFIRMED REPLACEMENT`
Microstructure не overlay, но как отдельный signal layer честно лучше `@1`.

### Исход C. `WATCHLIST`
Signal живой, но:
- threshold provenance слабая,
- result fragile,
- short/long behavior недостаточно чистое,
- или effect есть только на одном track.

### Исход D. `REJECTED`
При честном rerun против `@1` результат исчезает или становится экономически неинтересным.

## Критерий завершения этапа

Есть final route classification и понятный next step.

## Итоговая таблица

| Поле | Значение |
|------|----------|
| | |
