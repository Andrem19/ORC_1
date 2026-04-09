# План дальнейших исследований для v4

## Статус после v3

**База, которую не трогаем:** `active-signal-v1@1`.

Это остается immutable reference.

---

## Что обязательно считать уже закрытым

### Уже отвергнутые ветки
- `H1_post_funding_long`
- `H2_expiry_lead_lag`
- `A1_high_iv_long_v1`
- `A1b_high_iv_momentum`
- `A4_wed_afternoon`

### Что показал v3
- `A1_high_iv_long_v1` выглядел приемлемо standalone, но развалился на stability и integration.
- Главный вывод: **static high-IV regime — это режим волатильности, а не directional signal**.

---

## Что в прошлых планах осталось недозакрытым

### Неполностью закрытые части
1. У v3 не завершен общий финальный цикл.
2. Не везде закрыта исследовательская дисциплина.
3. Нет общего закрывающего решения.

---

## Главная цель v4

### Часть 1. Закрыть весь незавершенный долг прошлых планов
### Часть 2. После закрытия долгов сделать еще один содержательный шаг вперед

## Новая программа v4: weak-window repair specialists

- найти самые слабые окна и карманы у immutable base;
- искать specialists именно для ремонта этих слабых карманов.

---

## Главные принципы v4

### Принцип 1. Никаких незавершенных promising branches
### Принцип 2. Только одна активная основная ветка одновременно
### Принцип 3. Сначала additive truth, потом story
### Принцип 4. Отрицательный результат — это тоже результат
### Принцип 5. Не путать data task и research result

---

## Использование dev_space1 в v4

### Исследовательская дисциплина
### Features и datasets
### Backtests и диагностика
### Experiments

---

# ЭТАП 0. Полный closure-audit по прошлым планам

## Цель

Сначала собрать **единый список всего, что осталось незакрытым**.

## Что сделать

1. Открыть новый `research_project` для v4.
2. Зафиксировать closure-audit.
3. Зафиксировать immutable reference.
4. В `research_map` задать оси.
5. Составить единый open-items register.

## Критерий завершения этапа

Есть единый реестр долгов и единый immutable reference.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| project_id | |
| reference_snapshot | |
| atlas_defined | |
| closure_audit | |
| open_items_register | |
| low_friction_verdict | |
| notes | |

---

# ЭТАП 1. Формально закрыть все незавершенные документы и memory-nodes прошлых планов

## Цель

Убрать организационный долг до новых прогонов.

## Что сделать

1. Для v3 создать terminal result node по A1.
2. Проверить rejected ветки.
3. Создать отдельный research_record: v3_final_report.
4. Обновить registry statuses.
5. Убедиться, что нет веток без terminal node.

## Критерий завершения этапа

В памяти исследований больше нет "висящих" веток без статуса.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| a1_terminal_node | |
| a1b_rejected | |
| a4_rejected | |
| h1_h2_rejected | |
| b1_b2_rejected | |
| v3_final_report | |
| registry_statuses | |

---

# ЭТАП 2. Полностью закрыть ветку A2_iv_acceleration

## Цель

Довести до terminal decision главную незавершенную ветку из v3.

## Что сделать

### 2.1. Feature contract
### 2.2. Materialization
### 2.3. Plausibility on base-absent surface
### 2.4. Build strict prototypes
### 2.5. Acceptance chain

## Жесткие правила

## Критерий завершения этапа

A2 имеет terminal decision.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| feature_cf_iv_delta_1h | |
| a2_v2_standalone | |
| a2_v3_standalone | |
| stability | |
| integration | |
| terminal_decision | |

---

# ЭТАП 3. Только если A2 не promoted — полностью закрыть A3_broader_cl_transition

## Цель

Не оставлять резервную ветку в полуживом состоянии.

## Что сделать

### 3.1. Transition feature contract
### 3.2. Group transitions by meaning
### 3.3. Plausibility on base-absent surface
### 3.4. Build strict prototypes
### 3.5. Full acceptance chain

## Критерий завершения этапа

A3 имеет terminal decision.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| a3_status | |
| reason | |
| terminal_decision | |

---

# ЭТАП 4. Общий вердикт по low-friction wave

## Цель

Получить общий ответ: исчерпано ли low-friction пространство.

## Что сделать

1. Сравнить кандидатов vs base.
2. Определить mainline quality / specialist watchlist / reject.
3. Принять один из трех итогов.

## Критерий завершения этапа

Есть единый и жесткий final verdict.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| a2_verdict | |
| a3_verdict | |
| weak_window_repair | |
| low_friction_final | |

---

# ЭТАП 5. Новая программа v4 — weak-window repair specialists

## Цель

Сделать **еще один содержательный шаг** через targeted repair.

## Что сделать

### 5.1. Построить weakness atlas immutable base
### 5.2. Выбрать максимум 2 weak pockets
### 5.3. Для каждого weak pocket построить one specialist hypothesis
### 5.4. Для каждого specialist hypothesis пройти shortened but honest chain

## Жесткие ограничения

## Критерий завершения этапа

Есть ответ: может ли targeted repair дать что-то более полезное.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| weak_pockets_found | |
| top_weak_pocket | |
| repair_strategy | |
| repair_standalone | |
| repair_verdict | |

---

# ЭТАП 6. Только после этого — решение по medium-friction wave

## Цель

Не открывать medium-friction ветки раньше времени.

## Разрешение на unlock

Medium-friction wave разрешается **только если одновременно**:
1. A2 закрыт terminal decision;
2. A3 закрыт или formally skipped;
3. low-friction final verdict записан;
4. weak-window repair program завершена;
5. нет promoted candidate.

## Если unlock разрешен

### Ветка B1. Cross-symbol leadership
### Ветка B2. Model-backed IV-routing / compression

## Критерий завершения этапа

Есть четкое lock/unlock решение.

## Таблица результатов этапа

| Поле | Значение |
|------|----------|
| unlock_conditions_met | |
| medium_friction_decision | |
| reason | |
| next_recommended | |

---

# ЭТАП 7. Финальный отчет по v4

После завершения цикла агент обязан написать отдельный финальный отчет.
