---
status: accepted
date: 2026-09-23
tags:
  - adr
  - pulse
  - multimodal
---

# ADR-003: PULSE — архитектурная референция, не готовая модель

## Решение

Использовать PULSE как основное вдохновение для будущего self-supervised продольного мультимодального слоя: Past-State, Visit-State, отдельные encoder/decoder по модальностям, soft alignment и active masking.

Не переносить опубликованные взрослые модели, веса или клинические claims на беременность. Не начинать с PULSE-подобной сложности до получения нескольких визитов и устойчивых baseline.

## Проверяемая гипотеза на будущее

После появления продольной акушерской когорты alignment-pretraining должен сравниваться с late fusion и обычной temporal-моделью по downstream risk, calibration, missing-modality robustness и external validation.
