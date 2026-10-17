"""Create extension 06 without rewriting any earlier CTG notebook or result."""
from pathlib import Path
import argparse
import json
import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "research/notebooks/ctg_uci/06_clustering_embeddings_grid_search.ipynb"
RESULTS = ROOT / "research/experiments/ctg_uci/embeddings"


def summaries():
    receipt = json.loads((RESULTS / "validation_receipt.json").read_text())
    association = pd.read_csv(RESULTS / "association.csv")
    selected = pd.read_csv(RESULTS / "chosen_configurations.csv")
    stability = pd.read_csv(RESULTS / "stability.csv")
    risk = pd.read_csv(RESULTS / "test_risk_summary.csv")
    assignments = pd.read_parquet(RESULTS / "assignments.parquet")
    ae_gmm = association[(association.method == 'AE') & (association.algorithm == 'GMM') & (association.split == 'test')].iloc[0]
    ae_rich = risk[(risk.method == 'AE') & (risk.algorithm == 'GMM')].sort_values('path_fraction').iloc[-1]
    ae_gmm_seeds = stability[(stability.method == 'AE') & (stability.algorithm == 'GMM') & (stability['mode'] == 'seed')]
    path_file_count = assignments[(assignments.key == ae_rich.key) & (assignments.split == 'test') &
        (assignments.cluster == ae_rich.cluster) & (assignments.NSP == 3)].source_file.nunique()
    fcm = association[(association.algorithm == "FCM") & (association.split == "test")].set_index("method")
    old = pd.read_csv(ROOT / "research/experiments/ctg_uci/cluster_association.csv")
    old_fcm = old[(old.algorithm == "FCM") & (old.split == "test")].iloc[0]
    lines = ["# 06 · Сжатие признаков и кластеризация КТГ", "", "## tl;dr", "",
        f"Выполнены **{receipt['representations']} представлений × 30 настроек = {receipt['joint_configurations']} сочетаний**. "
        "UMAP: 27 вариантов; компактный denoising-автоэнкодер: 24; PCA: 3; без сжатия: 1. "
        f"Все {receipt['legacy_files']} прежних файлов кластеризации сохранены побайтно.", "",
        "Выбор проводился без `NSP`/`CLASS`: средний по исходным файлам silhouette на validation "
        "в общем пространстве из 21 подготовленного столбца; проверки размера кластеров, сходимости и trustworthiness. "
        "Затем выполнены диагностика устойчивости и сопоставление с экспертной разметкой.", "",
        "| Представление FCM | k | Test ARI с NSP | Test NMI |", "|---|---:|---:|---:|",
        f"| Прежний PCA 90%, ноутбук 04 | {int(old_fcm.k)} | {old_fcm.ari:.4f} | {old_fcm.nmi:.4f} |"]
    for method in ["RAW", "PCA", "UMAP", "AE"]:
        r = fcm.loc[method]
        lines.append(f"| {method} · новый протокол | {int(r.k)} | {r.ari:.4f} | {r.nmi:.4f} |")
    lines += ["", f"AE + GMM: test ARI **{ae_gmm.ari:.4f}**; в наиболее обогащённом кластере "
        f"патология у **{ae_rich.path_fraction:.1%}** сегментов "
        f"({int(ae_rich.pathological)}/{int(ae_rich.segments)}), но патологические случаи — "
        f"всего из **{path_file_count} исходных файлов**. Обогащение {ae_rich.path_enrichment:.2f}× "
        f"с широким точечным интервалом {ae_rich.path_enrichment_low:.2f}–{ae_rich.path_enrichment_high:.2f}. "
        f"При двух дополнительных seed ARI назначений относительно основного решения всего "
        f"{ae_gmm_seeds.validation_ari_vs_reference.min():.3f}–{ae_gmm_seeds.validation_ari_vs_reference.max():.3f}: "
        "решение неустойчиво к инициализации. Этого недостаточно для подтверждения самостоятельной группы риска.", "",
        "**Test уже просматривался в предыдущем исследовании.** Это исследовательское дополнение, "
        "а не новая слепая проверка и не доказательство клинических групп риска. Рост ARI на этом наборе "
        "не использован для изменения параметров или модели классификации."]
    takeaways = ["## Takeaways", "",
        "1. Сохранены исходные результаты 04; новые варианты находятся в отдельной папке `embeddings/`.",
        "2. Подбор оценивает общую геометрию, а не только расстояния, искажённые сжатием. "
        "Сильный silhouette в UMAP сам по себе не доказывает наличие естественных групп.",
        "3. ARI/NMI и состав кластеров — последующее сопоставление с экспертным классом. "
        "Ни число кластеров, ни сеть, ни UMAP по этим значениям не подбирались.",
        "4. Интервалы обогащения точечные, по исходным файлам, без поправки на множественные сравнения; "
        "их следует рассматривать как гипотезы для независимого набора. NSP — экспертная интерпретация КТГ, не исход ребёнка.",
        "5. Сохранены входной преобразователь со скейлерами, редукторы, веса автоэнкодеров, модели кластеров, "
        "все параметры сетки, кривые обучения, роли записей при early stopping, назначения и принадлежности.", ""]
    if not (risk.path_enrichment_low > 1).any():
        takeaways.append("**Ни у одного выбранного кластера точечный интервал обогащения патологией не лежит целиком выше 1.** "
            f"В наиболее обогащённой группе AE+GMM файлов с патологией: {path_file_count}; высокая доля сегментов "
            "не превращает их в множество независимых клинических наблюдений.")
    takeaways.append("")
    takeaways.append(f"AE+GMM чувствителен к инициализации: ARI совпадения validation-назначений "
        f"при двух дополнительных seed {ae_gmm_seeds.validation_ari_vs_reference.min():.3f}–"
        f"{ae_gmm_seeds.validation_ari_vs_reference.max():.3f}. Более высокий описательный ARI с NSP "
        "на одном test не делает этот вариант устойчивым победителем.")
    takeaways.append("")
    for method in ["UMAP", "AE"]:
        r = selected[(selected.method == method) & (selected.algorithm == "FCM")].iloc[0]
        s = stability[(stability.key == r.key) & (stability['mode'] == 'file_subsample')]
        takeaways.append(f"- {method}+FCM: общий validation silhouette по файлам **{r.validation_group_silhouette_common:.3f}**; "
            f"trustworthiness **{r.trustworthiness:.3f}**; ARI устойчивости при 80% файлов "
            f"**{s.validation_ari_vs_reference.min():.3f}–{s.validation_ari_vs_reference.max():.3f}**. "
            "Это чувствительность фиксированных параметров, не доверительный интервал.")
    return "\n".join(lines), "\n".join(takeaways)


def build():
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
    def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))
    md("# 06 · Сжатие признаков и кластеризация КТГ\n\n## tl;dr\n\nРезультаты появятся после полного выполнения сетки.")
    md("""
## Context & Methods

Продолжение 04 с сохранением его результатов. Источник: UCI Cardiotocography, 2126 сегментов; разбиение по исходным файлам из 01 остаётся прежним. Цель — проверить, выявляет ли нелинейное сжатие воспроизводимые группы измерений и как они соотносятся с экспертной разметкой.

### Key Assumptions

- Обучение всех преобразований и кластеров — только на train. Validation используется для выбора без меток; test только для последующего описания зафиксированных решений. Повторное использование уже просмотренного test делает анализ исследовательским.
- Идентификатора пациентки нет. Разделение, early stopping и bootstrap используют исходный файл как доступную единицу записи; это не доказывает независимость разных файлов одной женщины.
- Вход: ровно 21 исходный признак. Преобразователь 01 клонируется и обучается заново; `DS` и `Width` исключаются по его контракту. Удаляются дублирующие индикаторы наличия редких значений. 18 числовых столбцов стандартизуются; 3 one-hot `Tendency` делятся на √2. Это 21 столбец расстояний, а не 21 независимая исходная переменная.
- Нули сохраняются. В исходном наборе нет пропусков; новый пропуск отклоняется входным контрактом. Метки, строки, имена файлов и даты не входят в матрицу. Идентификатор файла используется только для разбиения, усреднения и ресэмплинга.
- UMAP: `n_components={2,5,10}`, `n_neighbors={10,30,60}`, `min_dist={0,.1,.5}`, евклидова метрика, 300 эпох, фиксированный seed. Число компонент не ограничивается красивой двумерной картинкой. Validation/test преобразуются через `transform`, без повторного fit.
- Автоэнкодер: `21 → {16,32} → {2,5,10} → {16,32} → 21`, активация `tanh/relu`, линейный выход, `alpha={.001,.1}`. Adam, шаг .001, batch 128, гауссов шум .05 на входе, чистая цель реконструкции. Это компактный undercomplete denoising AE на sklearn MLPRegressor, без разметки и без огромной сети для малого набора.
- Early stopping: отдельные 20% исходных файлов внутри train; внутренние скейлеры обучены только на оставшихся файлах. Критерий — чистая реконструкция holdout, patience 60, максимум 1200 эпох, улучшение >1e−5. Затем AE заново обучается на полном исходном train ровно выбранное число эпох. Остановка на внешней validation/test запрещена. Пилот 300 эпох часто упирался в лимит: на одном варианте продолжение уменьшило holdout MSE с .2256 до .1927 (лучшая эпоха 750), поэтому бюджет расширен по реконструкции, без NSP. Достижение лимита эпох явно показано и не равнозначно доказанной сходимости.
- Контроль: PCA с {2,5,10} компонентами и без сжатия. Для всех представлений одна сетка кластеров: k=2..6; FCM m={1.5,2,2.5}; KMeans; GMM diag/full. Несколько инициализаций. Всего 55×30=1650 сочетаний; это полный перебор указанной сетки, не гарантия глобального оптимума.

### Выбор без экспертных классов

Основной критерий — silhouette **в общем пространстве подготовленных признаков**, сначала средний по сегментам каждого validation-файла, затем средний по файлам. Это снижает влияние записей с большим числом сегментов и позволяет сравнить разные сжатия в одной геометрии. Вложенный silhouette в латентном пространстве показан отдельно. Такая цель предпочитает глобально разделимые группы и может недооценивать сложные нелинейные формы; она не является универсальным определением «лучших кластеров».

Исключаются несошедшиеся/вырожденные решения, кластеры <3% train или validation, UMAP/AE/PCA с trustworthiness <.80 (10 соседей). Сохранение соседей показано дополнительно. При равенстве выбираются меньше компонент, меньше k, затем фиксированный ID. Внутри каждой пары «представление × алгоритм» остаётся один вариант. BIC сравним только для GMM в одном представлении, поэтому между сжатиями не оптимизируется.

После сохранения `locked_config.json`: ещё два seed и пять обучений на случайных 80% train-файлов с повторным fit всей подготовки. ARI назначений на одной и той же validation показывает чувствительность. По этим диагностическим результатам и по NSP повторного отбора нет.

UMAP способен искажать плотность и создавать ложные разрывы: [официальное руководство по кластеризации](https://umap-learn.readthedocs.io/en/latest/clustering.html). Индуктивное применение: [transform](https://umap-learn.readthedocs.io/en/latest/transform.html). Сеть и регуляризация: [MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html). Качество локальных соседств: [trustworthiness](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html).
""")
    md("## Data\n\n### 1. Окружение и источники")
    code('''from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.training.ctg_embeddings import (RESULTS, ARTIFACT, run_embeddings, legacy_snapshot,
    versions, project, CTG_FEATURES)
from src.training.ctg_experiments import load_data
sns.set_theme(style='whitegrid', context='notebook', font_scale=1.02)
plt.rcParams.update({'figure.figsize': (10, 5), 'axes.spines.top': False, 'axes.spines.right': False})
COLORS = {'RAW':'#6b7280', 'PCA':'#d97706', 'UMAP':'#0072b2', 'AE':'#009e73'}
def show_save(name):
    plt.gcf().savefig(RESULTS / name, dpi=150, bbox_inches='tight')
    plt.show()
data, source_fingerprint = load_data()
display(pd.Series(versions(), name='version').to_frame())
display(data.groupby('split').agg(segments=('source_file','size'), source_files=('source_file','nunique')))
print('Source fingerprint:', source_fingerprint)
''')
    md("### 2. Выполнить полную сетку\n\nПервый запуск обучает все варианты и диагностику устойчивости. Повторный использует checkpoint только при совпадении данных, кода и версий. Все старые файлы кластеризации проверяются по SHA-256.")
    code('''receipt = run_embeddings()
display(pd.Series({k:v for k,v in receipt.items() if k in [
    'representations','joint_configurations','selected_models','legacy_unchanged',
    'early_stop_fit_files','early_stop_validation_files','elapsed_seconds']}).to_frame('value'))
''')
    md("## Results\n\n### 3. Покрытие сетки и зафиксированные параметры")
    code('''grid = pd.read_csv(RESULTS / 'joint_grid.csv')
embeddings = pd.read_csv(RESULTS / 'embedding_grid.csv')
chosen = pd.read_csv(RESULTS / 'chosen_configurations.csv')
association = pd.read_csv(RESULTS / 'association.csv')
risk = pd.read_csv(RESULTS / 'test_risk_summary.csv')
stability = pd.read_csv(RESULTS / 'stability.csv')
display(grid.groupby('method').agg(representations=('embedding_id','nunique'),
    combinations=('key','size'), eligible=('eligible','sum')))
selected = chosen.merge(embeddings[['embedding_id','params','best_epoch','hit_epoch_cap']], on='embedding_id')
selected['parameters'] = selected.params.map(lambda s: '; '.join(f'{k}={v}' for k,v in json.loads(s).items() if k!='n_components'))
display(selected[['method','algorithm','dimensions','k','variant','parameters']])
display(selected[['method','algorithm','validation_group_silhouette_common',
    'validation_silhouette_embedding','trustworthiness','neighbor_overlap',
    'best_epoch','hit_epoch_cap']].round(4))
print('Полная сетка:', RESULTS / 'joint_grid.csv')
''')
    md("### 4. Разделимость в латентном и общем пространствах\n\nКаждая точка — лучший допустимый вариант кластеризации для одного представления. Диагональ — равенство двух silhouette; завышенная разделимость только после сжатия требует осторожности. Групповой критерий выбора отличается от обычного среднего по строкам, показанного по оси X.")
    code('''best_per_embedding = grid[grid.eligible].sort_values('validation_group_silhouette_common').groupby('embedding_id').tail(1)
fig, ax = plt.subplots(figsize=(9, 6))
for method in ['RAW','PCA','UMAP','AE']:
    part = best_per_embedding[best_per_embedding.method == method]
    ax.scatter(part.validation_silhouette_common, part.validation_silhouette_embedding,
        s=65, alpha=.8, color=COLORS[method], label=f'{method} (n={len(part)})', edgecolor='white')
limits = [-.05, max(.65, best_per_embedding.validation_silhouette_embedding.max()+.03)]
ax.plot(limits, limits, '--', color='.6', lw=1)
ax.set(xlabel='Validation silhouette · common feature space',
    ylabel='Validation silhouette · embedding space', title='Apparent separation after compression vs original geometry')
ax.legend(title='Representation'); fig.tight_layout()
show_save('01_geometry.png')
''')
    md("### 5. Реконструкция и остановка автоэнкодера\n\nОшибка — средний квадрат в общей шкале входа, включая кодированную Tendency; это не физические единицы КТГ. PCA служит линейным ориентиром. Внешняя validation-реконструкция приведена для диагностики и не заменяет основной критерий. Флаг лимита эпох означает, что больший бюджет обучения ещё мог бы изменить результат.")
    code('''ae = embeddings[embeddings.method == 'AE']
display(ae.groupby(['dimensions','activation'] if 'activation' in ae else ['dimensions']).agg(
    variants=('embedding_id','size'), minimum_stop_mse=('stop_mse','min'),
    median_epochs=('best_epoch','median'), reached_cap=('hit_epoch_cap','sum'),
    minimum_validation_mse=('validation_reconstruction_mse','min')).round(4))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for method in ['PCA','AE']:
    p = embeddings[embeddings.method == method]
    axes[0].scatter(p.dimensions, p.validation_reconstruction_mse, color=COLORS[method],
        s=65, alpha=.65, label=method)
axes[0].set(xlabel='Latent dimensions', ylabel='Validation reconstruction MSE', title='Linear vs nonlinear reconstruction')
axes[0].set_xticks([2,5,10]); axes[0].legend()
for eid in chosen.loc[chosen.method == 'AE','embedding_id'].unique():
    curve = pd.read_csv(RESULTS / f'{eid}_learning_curve.csv')
    line, = axes[1].plot(curve.epoch, curve.stop_mse, label=f'{eid} · group holdout')
    axes[1].plot(curve.epoch, curve.train_mse, '--', color=line.get_color(), alpha=.65)
axes[1].set(xlabel='Epoch', ylabel='Reconstruction MSE', title='Inner-train early stopping; dashed = fit')
axes[1].legend(fontsize=9); fig.tight_layout()
show_save('02_autoencoder.png')
''')
    md("### 6. Устойчивость зафиксированных решений\n\nСравниваются назначения на тех же validation-сегментах, ARI инвариантен к перестановке номеров кластеров. Все преобразования переобучаются в подвыборках файлов. Диапазоны по пяти подвыборкам — анализ чувствительности, не 95% интервалы. При смене seed меняются редуктор и инициализация кластеров; при подвыборке seed фиксирован.")
    code('''stability_summary = stability.groupby(['method','algorithm','mode']).validation_ari_vs_reference.agg(['mean','min','max'])
display(stability_summary.round(3))
fig, ax = plt.subplots(figsize=(11, 6))
labels = [f'{r.method} + {r.algorithm}' for r in chosen.itertuples()]
for shift, mode, marker, color in [(-.12,'seed','o','#0072b2'),(.12,'file_subsample','s','#d55e00')]:
    p = stability[stability['mode'] == mode].groupby('key').validation_ari_vs_reference.agg(['mean','min','max']).reindex(chosen.key)
    y = np.arange(len(p)) + shift
    ax.errorbar(p['mean'], y, xerr=np.vstack([p['mean']-p['min'],p['max']-p['mean']]),
        fmt=marker, color=color, capsize=3, label=mode)
ax.set_yticks(np.arange(len(labels)), labels); ax.invert_yaxis()
ax.set(xlabel='Validation ARI vs fixed full-train assignments (range across repeats)',
    title='Sensitivity to random initialization and 80% training-file subsamples', xlim=(-.05,1.03))
ax.legend(loc='lower left'); fig.tight_layout()
show_save('03_stability.png')
''')
    md("### 7. Сопоставление с NSP после фиксации\n\nСначала показан прежний результат 04, затем все 12 решений нового протокола. ARI/NMI не были критериями подбора. Новый PCA отличается от прежнего сеткой и способом выбора; сопоставление не изолирует только влияние метода сжатия.")
    code('''old_association = pd.read_csv(ROOT / 'research/experiments/ctg_uci/cluster_association.csv')
old_test = old_association[old_association.split == 'test'].copy()
old_test['method'] = '04 · PCA90'
new_test = association[association.split == 'test'].copy()
comparison = pd.concat([old_test[['method','algorithm','k','ari','nmi']],
    new_test[['method','algorithm','k','ari','nmi']]], ignore_index=True)
display(comparison.round(4))
fig, ax = plt.subplots(figsize=(10,5))
table = comparison.pivot(index='method', columns='algorithm', values='ari').reindex(['04 · PCA90','RAW','PCA','UMAP','AE'])
table.plot.bar(ax=ax, color=['#0072b2','#009e73','#d55e00'], width=.75)
ax.axhline(0,color='.3',lw=.8); ax.set(title='Descriptive agreement with expert NSP · same 425 test segments',
    xlabel='Representation', ylabel='Adjusted Rand Index (ARI)')
ax.tick_params(axis='x', rotation=0); ax.legend(title='Clustering'); fig.tight_layout()
show_save('04_nsp_association.png')
''')
    md("### 8. Состав кластеров и интервалы обогащения\n\nОсновной фокус — патологический класс. Suspect и abnormal=NSP>1 приведены как описательное уточнение. Bootstrap: 1000 повторов по 68 исходным test-файлам, все сегменты выбранного файла остаются вместе. Интервалы точечные и не корректируются за 12 моделей и несколько кластеров/исходов. Малые числа патологических файлов ограничивают надёжность; отдельно показано число файлов с патологией внутри кластера. Принадлежности FCM и posterior GMM относятся к кластерам и не являются вероятностью клинического риска или класса NSP.")
    code('''all_assignments = pd.read_parquet(RESULTS / 'assignments.parquet')
path_files = all_assignments[(all_assignments.split=='test') & (all_assignments.NSP==3)].groupby(['key','cluster']).source_file.nunique().rename('path_source_files')
risk = risk.merge(path_files,on=['key','cluster'],how='left')
risk['path_source_files'] = risk.path_source_files.fillna(0).astype(int)
risk['limited_path_support'] = risk.small_cluster | (risk.path_source_files < 5)
risk.to_csv(RESULTS / 'test_risk_summary_with_support.csv',index=False)
display(risk[['method','algorithm','cluster','segments','source_files','pathological','path_source_files',
    'limited_path_support','path_fraction']].round(4))
display(risk[['method','algorithm','cluster','path_enrichment','path_enrichment_low',
    'path_enrichment_high','suspect_fraction','abnormal_fraction']].round(4))
fcm_risk = risk[risk.algorithm == 'FCM'].copy()
fig, ax = plt.subplots(figsize=(11,5.5))
for i, row in enumerate(fcm_risk.itertuples()):
    ax.plot([row.path_enrichment_low,row.path_enrichment_high],[i,i],color=COLORS[row.method],lw=2)
    ax.scatter(row.path_enrichment,i,color=COLORS[row.method],s=50,marker='x' if row.limited_path_support else 'o')
ax.set_yticks(np.arange(len(fcm_risk)), [f'{r.method} · cluster {r.cluster} (n={r.segments}, P={r.pathological})' for r in fcm_risk.itertuples()])
ax.axvline(1,color='.3',ls='--'); ax.invert_yaxis()
ax.set(xlabel='Pathologic fraction / overall test pathologic fraction · pointwise 95% file bootstrap',
    title='FCM: pathologic enrichment; x = limited source-file support')
fig.tight_layout(); show_save('05_pathologic_enrichment.png')
''')
    md("### 9. Двумерные проекции FCM\n\nСлева номера кластеров, справа экспертная разметка тех же test-сегментов. Для представления размерности >2 это дополнительная PCA-проекция, обученная на его train; для двумерного — сами координаты. Оси и расстояния между разными строками несопоставимы. Цвета NSP используются только для просмотра после фиксации; плотные островки UMAP не следует автоматически считать рисками.")
    code('''projections = pd.read_parquet(RESULTS / 'test_projections.parquet')
fcm_models = chosen[chosen.algorithm == 'FCM']
fig, axes = plt.subplots(4,2,figsize=(12,16))
for i, row in enumerate(fcm_models.itertuples()):
    p = projections[projections.key == row.key]
    for c in sorted(p.cluster.unique()):
        q=p[p.cluster == c]
        axes[i,0].scatter(q.axis1,q.axis2,s=15,alpha=.65,label=f'Cluster {c}')
    for label, color, marker, name in [(1,'#6b7280','o','Normal'),(2,'#e69f00','^','Suspect'),(3,'#cc3311','x','Pathologic')]:
        q=p[p.NSP == label]
        axes[i,1].scatter(q.axis1,q.axis2,s=22 if label>1 else 12,alpha=.8 if label>1 else .4,
            color=color,marker=marker,label=name)
    for j in range(2):
        axes[i,j].set(title=f'{row.method} + FCM · '+('clusters' if j==0 else 'expert NSP'),xlabel='Projection axis 1',ylabel='Projection axis 2')
        axes[i,j].legend(fontsize=8)
fig.tight_layout(); show_save('06_fcm_projections.png')
''')
    md("### 10. Независимые проверки сохранённых артефактов\n\nПроверки не используют обучающие функции повторно: сопоставляют назначения со входными данными, пересчитывают ARI, состав кластеров, сумму принадлежностей и прогноз загруженного bundle. Отдельно контролируются группы early stopping и контрольные суммы 04.")
    code('''saved = joblib.load(ARTIFACT)
assignments = pd.read_parquet(RESULTS / 'assignments.parquet')
preserved = json.loads((RESULTS / 'preserved_legacy.json').read_text())['sha256']
assert legacy_snapshot() == preserved
assert len(grid) == 1650 and len(embeddings) == 55 and len(chosen) == 12
assert len(assignments) == len(data) * 12
roles = pd.read_csv(RESULTS / 'ae_early_stopping_groups.csv')
assert roles.groupby('source_file').role.nunique().max() == 1
assert set(roles.source_excel_row) == set(data.loc[data.split=='train','source_excel_row'])
from sklearn.metrics import adjusted_rand_score
checks = 0
test = data[data.split=='test'].reset_index(drop=True)
common_test = saved['input_model'].transform(test[list(CTG_FEATURES)])
for row in chosen.itertuples():
    bundle = saved['bundles'][row.key]
    p = assignments[(assignments.key == row.key) & (assignments.split == 'test')].set_index('source_excel_row').loc[test.source_excel_row]
    assert np.array_equal(p.NSP.to_numpy(),test.NSP.to_numpy())
    memberships = p[[c for c in p if c.startswith('membership_') and c != 'membership_entropy']].fillna(0).to_numpy()
    np.testing.assert_allclose(memberships.sum(axis=1),1,atol=1e-7)
    assert np.array_equal(memberships.argmax(axis=1),p.cluster)
    predicted = bundle['clusterer'].predict(project(bundle['reducer'],common_test))
    assert np.array_equal(predicted,p.cluster)
    expected = association[(association.key == row.key) & (association.split == 'test')].iloc[0]
    assert abs(adjusted_rand_score(test.NSP,p.cluster)-expected.ari)<1e-12
    for cluster, q in p.groupby('cluster'):
        r = risk[(risk.key==row.key) & (risk.cluster==cluster)].iloc[0]
        assert len(q)==r.segments and int((q.NSP==3).sum())==r.pathological
        assert abs((q.NSP==3).mean()-r.path_fraction)<1e-12
    checks += 1
print(f'Проверены {checks} bundle, принадлежности и ARI; 55 представлений; 1650 комбинаций; старые файлы не изменились.')
print('Полные модели со скейлерами:', ARTIFACT)
print('Таблицы, параметры и рисунки:', RESULTS)
''')
    md("## Takeaways\n\nВыводы будут заполнены после выполнения и проверки.")
    book = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec":{"display_name":"Python 3 (femfedora)","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.11"}})
    nbf.validate(book)
    nbf.write(book, PATH)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--summaries-only',action='store_true')
    args=parser.parse_args()
    from build_ctg_review_notebooks import build_selected,summaries as current_summaries
    if not args.summaries_only:build_selected('ctg_uci',[6])
    current_summaries()
    print(PATH)
