# Postmortem: Plan V1 Research Cycle

## Executive Summary

Plan V1 ("Найти новые ортогональные сигналы поверх active-signal-v1@1") провалился целиком. Ни одна гипотеза не была проверена через полную воронку (standalone → stability → integration → cannibalization → walk-forward). Цикл остановился на stage_2 из-за инфраструктурных сбоев в MCP-инструментах.

## Хронология событий

### Что сработало
- Stage 1 (setup) завершился успешно один раз: проект `v5-model-research-c890eeaa` создан, atlas с 4 измерениями (240 ячеек), первая гипотеза зарегистрирована
- Source: `state/archive/2026-04-09T21-22-47/plans/runs/20260409T192857Z-c56c4494/reports/compiled_plan_v1_batch_1/turn_f2ce0102d82c.json`

### Где сломалось
- **Stage 2**: `abort` — `worker_parse_failure: contradictory_tool_registry_claim:research_map,research_record,research_search`. Агент не смог согласовать доступные инструменты
- **Stage 3+**: `abort` — `dependency_failed` (stage_2 не прошёл)
- **Stages 4-10**: никогда не запускались

### Повторные попытки
- 2026-04-09, 16:12 — 19:15: несколько прогонов, stage_1 иногда завершался, последующие stages — abort
- 2026-04-13: `direct_attempts` через LM Studio — все preflight failures
- Все попытки показывают один паттерн: setup stage проходит, рабочий stages ломаются

## Корневые причины провала

### 1. Инфраструктурная неготовность (PRIMARY)
MCP-инструменты dev_space1 не были стабильны во время исполнения. Tool registry contradictions — агенты получали противоречивую информацию о доступных инструментах.

### 2. Слабая архитектура кандидатов (из анализа plan_v2 raw)
Даже если бы stage 2 прошёл, план v1 закладывал:
- H1/H2: сырые event-entry конструкции ("событие плюс один фильтр") — тупик
- H3-H5: требовали инфраструктурно не готовые data contracts
- Главный недостаток: слишком слабая селективность, сотни сделок без edge до комиссий

### 3. Методологические ошибки v1
- Гипотезы формулировались как single-filter entries
- Не было complement-first подхода (исследование всего рынка, а не только base-absent surface)
- Не было требования минимум трёх логических слоёв в кандидате
- Model-first подход без предварительного structured hypothesis testing

## Что должно измениться в V2

### Жёсткие инварианты
1. База `active-signal-v1@1` — immutable
2. H1/H2 в сырой форме не переоткрывать
3. Complement-first обязательный: сначала base-absent surface
4. Single-filter entry запрещён — минимум 3 логических слоя
5. Model-first запрещён до завершения low-friction wave A
6. Кандидат отклоняется, если ценность исчезает в strict-additive режиме

### Допустимые ветки v2
- **A1**: IV dynamics from existing `iv_est_1`
- **A2**: Event × state interactions
- **A3**: Classifier-transition specialists
- **A4**: Compression-to-release without model
- **B1**: Cross-symbol leadership (после data readiness)
- **B2**: Model-backed routing (late wave только)

### Ключевой разворот
V1 спрашивал: "Есть ли красивый новый standalone-сигнал?"
V2 спрашивает: "Есть ли новый слой, который на участках, где база молчит, даёт качественно новые входы?"

## Вывод

V2 — это не повторная попытка тех же идей. Это другой класс исследования:
- от mass signals → к specialist signals
- от full-market search → к complement-first
- от single-filter → к structured multi-layer candidates
- от standalone-first → к additive-by-construction

---

*Документ создан на основе анализа архивных данных выполнения plan_v1:*
- `state/archive/2026-04-09T21-22-47/plans/runs/20260409T192857Z-c56c4494/`
- `compiled_plans/plan_v1/semantic.json`
- `compiled_plans/plan_v2/semantic.json`
- `raw_plans/plan_v1.md`, `raw_plans/plan_v2.md`
