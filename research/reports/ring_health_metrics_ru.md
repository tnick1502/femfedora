# Итоговые метрики ring health v1

Исторические метрики. Проверка фактического API и покрытия выполнена заново в [v2](ring_health_v2_review_ru.md); эти числа не заменяют новую оценку.

Числа сформированы из сохранённых прогнозов и метрик. Для multiclass Brier — сумма по трём классам; это другой масштаб, чем бинарный Brier. NaN означает, что показатель не применим.

| Эксперимент | Выбранная модель | Macro F1 | F1 положительного класса | ROC AUC | Brier |
|---|---|---:|---:|---:|---:|
| nhanes_age_only | logistic | 0.880 | 0.850 | 0.956 | 0.079 |
| nhanes_context | logistic | 0.872 | 0.842 | 0.954 | 0.081 |
| synthetic_stage_ring_only | random_forest | 0.852 | nan | nan | 0.199 |
| synthetic_stage_hybrid | random_forest | 0.888 | nan | nan | 0.182 |
| synthetic_fertility_calendar | hist_gradient | 0.662 | 0.506 | 0.799 | 0.128 |
| synthetic_fertility_ring_only | random_forest | 0.697 | 0.563 | 0.831 | 0.121 |
| synthetic_fertility_hybrid | random_forest | 0.769 | 0.638 | 0.896 | 0.100 |
| synthetic_fertility_confirmation | hist_gradient | 0.975 | 0.969 | 0.998 | 0.017 |

Выбор семейства — по validation log loss. Порог бинарной классификации — по validation F1; классы этапов — argmax. Подробные результаты каждого кандидата находятся в experiment_summary.csv.
