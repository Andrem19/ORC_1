# План дальнейших исследований для v10

## Статус перед стартом v10

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

## Что уже доказано и что это значит для v10

### Что окончательно закрыто
1. Raw event entries не работают.
2. Static IV и IV dynamics не дают directional alpha.
3. Classifier transitions сами по себе не создают новый информационный слой.
4. CatBoost regression на future PnL — тупик.
5. Full-surface directional classification на текущем data universe провалилась.
6. Trade-quality overlays на текущем data universe тоже не дали устойчивого улучшения:
   - veto-long дал только marginal signal;
   - precision слишком низкая;
   - lockbox-тест даже не был оправдан.

### Что реально дал v9
1. Старый data universe действительно исчерпан.
2. Среди новых families только **microstructure proxies из 5m** дали новый содержательный сигнал.
3. Построены и опубликованы 7 microstructure features.
4. Самый важный признак — `cf_micro_imbalance_1h`.
5. Veto-модели стали заметно лучше, чем раньше:
   - long around `0.61`;
   - short around `0.67`.
6. Но это еще **не доказанный trading success**:
   - overlay-lockbox не доведен до конца;
   - additive improvement базы не доказан;
   - значит корректный статус `v9` — не `SUCCESS`, а `PROMISING SIGNAL DISCOVERED`.

### Главный вывод для v10
`v10` не должен расползаться в новый широкий поиск.

Его задача:
1. **закрыть методологический долг v9**;
2. **честно доказать или опровергнуть**, что microstructure family реально улучшает базу;
3. только после этого решать, заслуживает ли она:
   - `PROMOTE`,
   - `WATCHLIST`,
   - или `REJECT`.

---

## Главная идея v10

**v10 — это цикл валидации и добивания microstructure-family до terminal verdict.**

Не новый brainstorming.
Не новый большой universe audit.
Не новая волна старых признаков.

Только один вопрос:

> может ли microstructure layer, прежде всего `cf_micro_imbalance_1h`, дать реальный overlay alpha поверх immutable base?

---

## Главная цель v10

Получить честный ответ на два вопроса:

### Вопрос 1
Дает ли microstructure-family рабочий **entry-veto overlay** поверх базы?

### Вопрос 2
Если да, то является ли лучший microstructure signal:
- только veto-overlay,
- или он также тянет на `early-cut`,
- или даже на узкий `directional scout`.

---

## Главные принципы v10

### Принцип 1. Один активный data family
В v10 активна только:
- `microstructure_proxies`

Все остальные families locked.

### Принцип 2. Сначала закрыть v9 debt
До новых моделей надо:
- исправить статус v9;
- довести materialization;
- закрыть augmented atlas;
- закрыть lockbox overlay test.

### Принцип 3. Simpler before smarter
Если один признак уже доминирует, сначала проверить:
- single-feature bucket rule,
- single-threshold veto,
- logistic baseline,

и только потом маленький CatBoost.

Если простой rule работает не хуже модели, сложность не нужна.

### Принцип 4. Short-first priority
Так как `short veto` в v9 был заметно сильнее `long veto`, первым идет short side.

### Принцип 5. Overlay-first remains mandatory
Сначала доказать veto-overlay.
Только потом разрешается:
- early-cut,
- directional scout.

### Принцип 6. Label quality важнее model sophistication
Если label шумный, модель будет слабой даже на хорошем feature.
Значит в v10 ключевой шаг — не "еще один CatBoost", а **правильный veto label**.

### Принцип 7. Lockbox is the truth
Никаких сильных выводов до сравнения:
- immutable base
- против
- base + overlay

на том же lockbox окне.

---

## Что в v10 считается уже закрытым и не должно переоткрываться

Нельзя снова открывать:
- leadership / breadth как самостоятельный directional edge;
- path-asymmetry как самостоятельный directional edge;
- regime-disagreement route;
- event-only overlays;
- trade-quality overlays без нового information layer;
- derivatives positioning family до появления реальных OI / liquidation / basis datasets.

---

## Что именно должно стать ядром v10

### Primary microstructure core
- `cf_micro_imbalance_1h`
- `cf_micro_close_pressure_1h`
- `cf_micro_rejection_score_1h`
- `cf_micro_followthrough_1h`
- `cf_micro_burstiness_1h`
- `cf_micro_shock_cluster_1h`
- `cf_impulse_fail_score_1h`

### Secondary context-only features
Разрешены только как support:
- hour
- dow
- side
- weak-window tag
- bars_since_funding
- bars_since_expiry
- existing trade-context features из базы, если они не превращают route в remix старого universe

### Что запрещено
Не подмешивать назад большие старые feature packs.
Microstructure должен пройти тест почти сам по себе.

---

## Почему нужен отдельный этап простых baselines

По v9 видно, что `cf_micro_imbalance_1h` доминирует в importance.

Это означает две возможности:
1. либо feature действительно сильный;
2. либо модель лишь слегка оборачивает один и тот же признак.

Поэтому до full CatBoost в v10 обязательно проверить:

- **Baseline A. Single-feature bucket veto**
- **Baseline B. Single-feature threshold veto**
- **Baseline C. Logistic baseline on micro-core**
- **Baseline D. Small CatBoost on micro-core**

Если `D` не лучше `A/B/C`, route надо упрощать.

---

## Новый label contract v10

В v9 veto labels были слишком грубыми.
В v10 нужно перейти к более содержательным trade-quality labels.

### Label Family 1. `bad_trade_pnl`
Trade считается плохим, если он входит в худший хвост по realized PnL.

### Label Family 2. `fragile_trade_mae`
Trade считается плохим, если:
- adverse excursion был большим,
- followthrough был слабым,
- и trade статистически похож на stop-loss cluster.

### Label Family 3. `bad_trade_composite`
Комбинация:
- realized PnL,
- MAE,
- weak followthrough,
- exit pathology.

### Разрешенные варианты
Для каждой семьи только:
- `balanced`
- `conservative`

### Жесткое правило
Не использовать один-единственный `net_pnl > 0` как veto label.
Он слишком шумный для этой задачи.

---

## Какой route должен быть в v10

### Route 1. Veto validation route
Обязательный и основной.

### Route 2. Early-cut route
Разрешается только если:
- veto route прошел lockbox хотя бы на уровне `WATCHLIST`,
- и есть evidence, что post-entry microstructure помогает распознавать losers раньше базового выхода.

### Route 3. Directional scout
Разрешается только если:
- veto route показывает coherent signal;
- simple bucket analytics показывают monotonicity;
- есть reason считать, что feature может работать не только как trade-quality overlay, но и как entry timing signal.

### Route 4. Extension
Не приоритет для v10.
Открывать только если в atlas явно найден cluster недодержанных winners.

---

## Каким должен быть evaluation contract v10

### Для simple baselines
Selection OOS:
- bucket monotonicity;
- worst-trade concentration in veto bucket;
- simulated delta if flagged trades removed.

### Для veto models
Selection OOS:
- ROC AUC
- PR AUC
- precision in veto bucket
- recall on bad trade cluster
- economic effect if flagged trades removed

### Для lockbox overlay
Обязательно считать:
- delta PnL
- delta PF
- delta DD
- removed trades
- PnL removed trades
- share of removed winners
- weak-window repair
- ownership of improvement

### Для directional scout
Только после veto pass:
- standalone OOS
- strict additive vs base
- permissive
- ownership

---

# ЭТАП 0. Closure-audit v9 и статус-коррекция

## Цель
Формально закрыть v9 без завышенных формулировок.

## Что сделать
1. Открыть новый `research_project` для v10.
2. В `research_record` записать:
   - что именно дал v9;
   - что именно v9 не доказал;
   - почему статус должен быть `signal discovered, route unproven`.
3. Зафиксировать immutable reference:
   - `snapshot_id = active-signal-v1`
   - `version = 1`
   - `symbol = BTCUSDT`
   - `anchor = 1h`
   - `execution = 5m`
4. В `research_map` завести оси:
   - route
   - side
   - feature_group
   - label_family
   - model_stage
   - final_decision
5. Создать open-items register:
   - `micro_core_materialization`
   - `augmented_trade_atlas`
   - `simple_baselines`
   - `veto_short_v10`
   - `veto_long_v10`
   - `lockbox_overlay_test`
   - `earlycut_gate`
   - `directional_scout_gate`
   - `final_v10_verdict`

## Критерий завершения этапа

Есть clean project v10 и corrected status.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 1. Завершить materialization и dataset hardening для micro-core

## Цель
Убрать технический долг вокруг 5m→1h microstructure features.

## Что сделать
1. Убедиться, что все 7 microstructure features:
   - не только published,
   - но и стабильно materialized.
2. Проверить:
   - full coverage;
   - warmup behavior;
   - null-rates;
   - alignment 5m → 1h;
   - absence of future leakage.
3. Зафиксировать один рабочий dataset version для v10.
4. Если materialization нестабильна:
   - не расширять family,
   - а сначала починить pipeline.

## Таблица результатов этапа

| feature_name | published | materialized | warmup_ok | null_rate_ok | alignment_ok | leakage_ok | notes |
|--------------|-----------|--------------|-----------|--------------|--------------|------------|-------|
| | | | | | | | |

### Dataset table

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 2. Augmented trade atlas with micro-core

## Цель
Понять, где именно microstructure сигнал strongest и как его правильно использовать.

## Что сделать
1. Обогатить trade atlas базы micro-features на:
   - entry time;
   - first 1h after entry, если допустимо и causal для post-entry routes.
2. Построить разрезы:
   - long vs short;
   - winners vs losers;
   - stop-loss cluster;
   - worst-decile trades;
   - weak-window trades;
   - time-limit trades.
3. Отдельно проверить:
   - monotonicity for `cf_micro_imbalance_1h`;
   - side asymmetry;
   - whether signal stronger on short veto than long veto.
4. Сохранить 3–5 strongest empirical patterns.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 3. Freeze improved veto labels

## Цель
Перейти от шумного label к более осмысленным veto targets.

## Что сделать
1. Определить и заморозить:
   - `bad_trade_pnl_balanced`
   - `bad_trade_pnl_conservative`
   - `fragile_trade_mae_balanced`
   - `fragile_trade_mae_conservative`
   - `bad_trade_composite_balanced`
   - `bad_trade_composite_conservative`
2. Для каждого label словами зафиксировать:
   - кого считаем плохим trade;
   - какие cases считаются ambiguous;
   - какой порог material harm.
3. Сразу запретить изменения после начала training.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| | |

---

# ЭТАП 4. Simple baselines before CatBoost

## Цель
Проверить, нужен ли вообще сложный model layer сверх `cf_micro_imbalance_1h`.

## Что сделать
Построить на selection OOS:

### Baseline A. Bucket veto
Удалять сделки из worst micro-imbalance bucket.

### Baseline B. Threshold veto
Удалять сделки при crossing fixed threshold по `cf_micro_imbalance_1h`.

### Baseline C. Logistic micro-core
Легкая модель на 3–5 micro-features.

### Baseline D. Small CatBoost micro-core
Небольшой depth, без большого HPO.

Для каждого baseline сравнить:
- selection delta PnL;
- selection delta PF;
- selection delta DD;
- removed trades;
- false veto rate.

## Таблица результатов этапа

| baseline_id | side | method | label_family | selection_delta_pnl | selection_delta_pf | selection_delta_dd | removed_trades | false_veto_rate | promoted | notes |
|-------------|------|--------|--------------|---------------------|--------------------|--------------------|----------------|-----------------|----------|-------|
| | | | | | | | | | | |

---

# ЭТАП 5. Short-first veto models

## Цель
Добить до решения наиболее перспективную сторону — short veto.

## Что сделать
1. Обучить short veto models на frozen labels:
   - short-pnl
   - short-fragile
   - short-composite
2. Использовать только:
   - micro-core
   - минимальный context-support
3. Не делать широкий HPO.
4. Выбрать лучший short route по selection economics, а не только по AUC.

## Таблица результатов этапа

| model_id | label_family | val_auc | test_auc | val_precision | test_precision | veto_bucket_ok | selection_economic_effect_ok | promoted_to_lockbox | notes |
|----------|--------------|---------|----------|---------------|----------------|----------------|------------------------------|---------------------|-------|
| | | | | | | | | | |

---

# ЭТАП 6. Long veto models

## Цель
Проверить, есть ли у microstructure meaningful value и на long side.

## Что сделать
1. Обучить long veto models на frozen labels:
   - long-pnl
   - long-fragile
   - long-composite
2. Сравнить их с short side.
3. Не продвигать long route дальше, если он снова заметно слабее short без компенсации по economics.

## Таблица результатов этапа

| model_id | label_family | val_auc | test_auc | val_precision | test_precision | veto_bucket_ok | selection_economic_effect_ok | promoted_to_lockbox | notes |
|----------|--------------|---------|----------|---------------|----------------|----------------|------------------------------|---------------------|-------|
| | | | | | | | | | |

---

# ЭТАП 7. Lockbox overlay test against immutable base

## Цель
Наконец-то доказать или опровергнуть microstructure overlay на реальном OOS.

## Что сделать
Для best surviving models:
1. Freeze thresholds on selection OOS.
2. Построить snapshot / overlay workflow.
3. Прогнать:
   - immutable base
   - base + short veto
   - base + long veto
   - base + combined veto, если обе стороны живы
4. Посчитать:
   - delta PnL
   - delta PF
   - delta DD
   - removed trades
   - PnL removed trades
   - share of removed winners
   - weak-window repair
   - ownership

## Таблица результатов этапа

| overlay_id | side | lockbox_delta_pnl | lockbox_delta_pf | lockbox_delta_dd | removed_trades | pnl_of_removed_trades | removed_winner_share | weak_window_repair | ownership_positive | final_decision | notes |
|------------|------|-------------------|------------------|------------------|----------------|-----------------------|---------------------|-------------------|--------------------|----------------|-------|
| | | | | | | | | | | | |

---

# ЭТАП 8. Gate for early-cut and directional scout

## Цель
Не открывать новые ветки без права на это.

## Что сделать

### Early-cut gate
Открывается только если:
- veto route не просто живой, а видно, что losers распознаются именно по раннему micro-behavior.

### Directional scout gate
Открывается только если:
- `cf_micro_imbalance_1h` и/или micro-core показывают устойчивую monotonicity;
- veto overlay уже дал хотя бы watchlist-quality result;
- есть reason считать, что feature может быть не только quality filter, но и entry timing signal.

Открыть максимум **одну** из двух веток.

## Таблица результатов этапа

| route | unlock_condition_met | opened | why_opened_or_locked | notes |
|-------|----------------------|--------|----------------------|-------|
| | | | | |

---

# ЭТАП 9. Optional route execution

## Цель
Провести только ту ветку, которая действительно заслужила открытие.

## Что сделать
Если открыт `early_cut`:
- build post-entry micro-features;
- train minimal model;
- test against base on lockbox.

Если открыт `directional_scout`:
- test simple microstructure direction rule first;
- only then small model;
- compare standalone and strict additive.

## Таблица результатов этапа

| route | selection_quality_ok | lockbox_quality_ok | delta_vs_base_meaningful | final_decision | notes |
|-------|----------------------|--------------------|--------------------------|----------------|-------|
| | | | | | |

---

# ЭТАП 10. Финальный вердикт по v10

## Цель
Получить честное решение по microstructure family.

## Возможные исходы

### Исход A. Success
Microstructure family дала хотя бы один working overlay route.

### Исход B. Partial
Есть watchlist-quality result, но не полноценный add-on.

### Исход C. Failed
Даже microstructure family не пережила честный lockbox overlay test.

## Критерий завершения этапа

Есть final route classification и понятный next step.

## Итоговая таблица

| Поле | Значение |
|------|----------|
| | |
