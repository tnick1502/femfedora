"""Reproducible report-field classifier and independent assessment notebooks."""
from pathlib import Path
import json
import nbformat as nb
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/experiments/jnu_ctg/classifiers_v1'
BOOKS=ROOT/'research/notebooks/jnu_ctg'


def intro03():
    text='''# 03 · Классификаторы JNU по числам бланка

## tl;dr

Обучены **CatBoost с полным Grid Search**, **регуляризованная сплайновая модель с ограниченными взаимодействиями** и контрольная линейная логистика. Сравнение проводится отдельно для пяти наборов полей бланка. Цель — **экспертная normal/abnormal FIGO**, не диагноз гипоксии и не вероятность асфиксии.

Подбор: 5 внешних ×3 внутренних групповых фолда train, веса записей дают каждой пациентке суммарный вес 1. Агрегация single/ensemble5 выбирается внутри обучающей части. Калибровка и исследовательский порог используют две разные группы validation; test не выбирает решение. Все 20769 строк сохранены, фильтрации по флагу качества из 02 нет.
'''
    if (OUT/'test_metrics.csv').exists():
        d=pd.read_csv(OUT/'test_metrics.csv')
        d=d[(d.policy=='primary_0.5')&d.champion.astype(bool)]
        text+='\n### Зафиксированные победители по каждому входному профилю\n\n| Профиль | Модель, выбранная на train | Macro-F1 | Recall abnormal | Specificity | AP abnormal |\n|---|---|---:|---:|---:|---:|\n'
        for _,r in d.iterrows():
            text+=f'| {r.profile} | {r.family} | {r.patient_macro_f1:.3f} | {r.patient_recall:.1%} | {r.patient_specificity:.1%} | {r.patient_ap:.3f} |\n'
        text+='\nОценка по **4140 test-записям 2520 пациенток**. Метрики по записям с суммарным весом 1 на пациентку — это не новая пациентская метка. Профиль выбирается по доступным проверенным полям; таблица не разрешает выбирать наиболее приятный риск.\n'
    text+='''
«Идеальная» модель заранее неизвестна. Сплайны проверяют, достаточно ли плавных зависимостей и немногих взаимодействий; линейная модель может оказаться конкурентоспособной. Результат ограничен собственным экстрактором, одним центром и 30-минутной записью. Перенос на реальные российские бланки не подтверждён. [Подробная проверка, интервалы и устойчивость](04_classifier_assessment.ipynb).
'''
    return text


def intro04():
    text='''# 04 · Проверка классификаторов JNU: неопределённость, ошибки и устойчивость

## tl;dr

Этот ноутбук проверяет **замороженные** модели из 03. Здесь не выбираются новые признаки, параметры, пороги или победители. Оценка: bootstrap по независимым patient-components, калибровка, ошибки, доступность входов, качество, округление и выпадение измерений. Отрицательный контроль использует только train. Префиксы 10/17/20 min не получают метку всей записи.
'''
    if (OUT/'validation_receipt.json').exists():
        r=json.loads((OUT/'validation_receipt.json').read_text())
        text+=f"\nНезависимый валидатор проверил **{r['model_bundles']} экспортированных моделей**: границы групп, все train-строки, медианы/IQR каждого fit, native CatBoost roundtrip, калибровочную группу, исследовательские пороги и ручную арифметику test. Предыдущие данные и результаты сохранены.\n"
    text+='''
**Основной численный результат:** на расширенном 13-полевом профиле CatBoost: Macro-F1 **0.751** (95% CI **0.736–0.766**), recall abnormal **76.2%**, AP **0.856**. Сплайновая модель: F1 **0.751**; выбранная на train линейная: **0.749**. Парная разница CatBoost–сплайны не установила преимущество по F1. У CatBoost остаётся **527 пропущенных abnormal из 2219 записей** и 491 ложная тревога. Это числа записей, поэтому доли из них слегка отличаются от основных метрик с весами пациенток.

Все 15 финальных конфигураций выбрали single: усреднение пяти моделей не выполнило заранее заданный критерий. Округление изменило 0.27–1.21% ответов победителей; принудительное удаление всех представлений STV — 33.9–48.1% у использующих его профилей. NaN-поддержка не означает пригодности неполного бланка для клинического решения.

1000 bootstrap-выборок оценивают неопределённость при повторном наборе пациенток из этой когорты **для уже зафиксированных моделей**. Они не учитывают весь процесс переобучения/подбора и не доказывают перенос между странами/приборами. Изученные здесь численные возмущения не являются клинически валидированной моделью ошибок измерения.
'''
    return text


SETUP='''from pathlib import Path
import sys,json,io,contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_curve
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'pyproject.toml').exists())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.training.jnu_report_experiment import OUT,EXPORT,metadata,inputs
from src.training.jnu_report_models import patient_weights
sns.set_theme(style='whitegrid',font_scale=1.02)
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
pd.set_option('display.max_colwidth',100)
COLORS={'catboost':'#0072b2','spline_logit':'#d55e00','linear':'#009e73'}
def save_show(name):
    plt.gcf().savefig(OUT/name,dpi=140,bbox_inches='tight')
    plt.show()
'''


def build03():
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    cells=[md(intro03()),md('''## Context & Methods

### Key Assumptions

- Только поля отчёта. Нет FHR/UC, NetFea, клинических исходов/Апгар, ID, заключений прибора и внутренних quality-флагов в X. Независимость задаёт patient_component из 01.
- Внешний assessment не используется ни для fit преобразований, ни для выбора параметров/усреднения. Inner folds одинаковы для моделей. Пациентские веса пересчитываются в текущей fit/assessment-когорте.
- CatBoost: **72** комбинации depth/learning_rate/l2/iterations/class weights. Сплайновая логистика: **48** комбинаций числа узлов/регуляризации/взаимодействий/class weights. Линейный контроль: **10** комбинаций C/весов. Это исчерпывающий поиск по заданной конечной сетке, не Random Search и не гарантия глобального оптимума.
- Single/ensemble5: усреднение принимается при улучшении inner Macro-F1 минимум на 0.005 и ухудшении log loss не более 0.005. Проверка ансамбля внутри каждого inner fit использует собственные групповые подфолды, не модели, видевшие оцениваемую строку.
- Политика выбора зафиксирована до test: максимум среднего inner patient-weighted Macro-F1, точные равенства разрешает log loss/сложность. В отличие от предварительного плана эвристика одной стандартной ошибки не применяется: с тремя зависимыми фолдами это не формальный доверительный интервал. Изменение зафиксировано в protocol.json до обучения.
- CatBoost использует NaN и отдельные маски всех колонок; scaler не требуется. Сплайны/логистика fit собственные median/IQR и маски внутри fit. Параметры из 02 не используются.
- Сплайны строят квадратичные B-splines с удалением повторных квантильных узлов, явным поведением констант/бинарных полей и ограниченным набором пар. Для редких/нулевых счётчиков это устойчивее предположения уникальных квантилей.
- Калибровка actual aggregate: sigmoid(logit(p)) с положительным наклоном, fit только на validation-calibration. Порог 0.5 — primary; альтернативный F1-порог на validation-operating — исследовательский, не клинически «безопасный».

Первичные технические источники: [CatBoost parameters](https://catboost.ai/docs/en/references/training-parameters/common), [SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html), [калибровка вероятностей](https://scikit-learn.org/stable/modules/calibration.html). Определения КТГ и границы переноса — [обзор 02](REPORT_FEATURE_METHODS.md).
'''),code(SETUP),md('''## Data · фиксированное разбиение и запуск

Первый расчёт выполняет полный поиск. Повторный использует контрольные точки, привязанные к хешам кода/входов и версиям библиотек. После test изменение кода не может незаметно породить новый «слепой» результат в той же версии. Ход расчёта сохраняется в `progress.json`; полный вывод — в отдельный журнал.
'''),code('''from src.training.jnu_report_experiment import run_all
with (OUT/'notebook_training_replay.log').open('w') as log,contextlib.redirect_stdout(log):
    metrics=run_all()
protocol=json.loads((OUT/'protocol.json').read_text())
frozen=json.loads((OUT/'selection_frozen.json').read_text())
display(pd.DataFrame({'version':protocol['environment']}))
train,valid,test=[metadata(s) for s in ['train','validation','test']]
display(pd.DataFrame([{'split':name,'records':len(d),'patients':d.patient_id.nunique(),
    'normal':int((d.figo_label==0).sum()),'abnormal':int(d.figo_label.sum())}
    for name,d in zip(['train','validation','test'],[train,valid,test])]))
display(pd.read_csv(OUT/'outer_support.csv'))
print('Поиск и замороженная оценка завершены; все исходные строки сохранены.')
'''),md('''## Results · честные внешние фолды train

Точки — внешние фолды после inner подбора параметров и режима агрегации. Линии погрешности — **SD пяти фолдов**, не 95% CI и не независимые повторные эксперименты. Итоговая семья внутри профиля выбирается по этому train-сравнению, а не по test.
'''),code('''nested=pd.read_csv(OUT/'nested_summary.csv')
folds=pd.read_csv(OUT/'nested_fold_metrics.csv')
display(nested.round(4))
profiles=list(protocol['profiles'])
fig,ax=plt.subplots(figsize=(12,5))
for j,family in enumerate(['catboost','spline_logit','linear']):
    for i,profile in enumerate(profiles):
        values=folds[(folds.profile==profile)&(folds.family==family)].patient_macro_f1
        pos=i+(j-1)*.2
        ax.scatter(np.full(len(values),pos),values,s=24,color=COLORS[family],alpha=.4)
        ax.errorbar(pos,values.mean(),yerr=values.std(),fmt='o',capsize=4,color=COLORS[family],label=family if i==0 else None)
ax.set(xticks=range(len(profiles)),xticklabels=profiles,ylabel='Patient-weighted record Macro-F1',title='Nested grouped CV · outer folds, mean and SD')
ax.legend();fig.tight_layout();save_show('01_nested_folds.png')
'''),md('''### Сетки и зафиксированные конфигурации

Карты показывают лучший mean inner score по остальным параметрам при заданной паре осей. Это диагностический срез **train-CV**, не test и не новое основание менять конфигурацию. Для iterations=300 использован точный префикс fit на 700 деревьев при одинаковых остальных параметрах/seed; все 36 соответствий проверяются отдельно на реальных train-данных.
'''),code('''grid=pd.read_json(OUT/'final_grid_results.json')
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,family,row,col in [(axes[0],'catboost','depth','iterations'),(axes[1],'spline_logit','n_knots','C')]:
    selected=grid[(grid.profile=='sonomed_fields')&(grid.family==family)].copy()
    selected=pd.concat([selected.drop(columns='params').reset_index(drop=True),pd.json_normalize(selected.params).reset_index(drop=True)],axis=1)
    heat=selected.groupby([row,col]).mean_f1.max().unstack(col)
    heat.columns=[f'{v:g}' for v in heat.columns]
    sns.heatmap(heat,annot=True,fmt='.3f',cmap='Blues',ax=ax,vmin=.5,vmax=.85)
    ax.tick_params(axis='x',labelrotation=0)
    ax.set_title(f'{family} · max inner Macro-F1')
fig.tight_layout();save_show('02_parameter_grids.png')
choices=pd.DataFrame(json.loads((OUT/'final_choices.json').read_text()))
display(choices[['profile','family','mode','champion','params','research_threshold']])
'''),md('''### Однократная оценка test после фиксации

Основная таблица использует калиброванные вероятности и порог 0.5. Представлены все заранее объявленные модели, победитель помечен **по train**. Второй порог выбран на отдельной validation-operating части и не означает клинический порог. Метрики record_* и patient_* рассчитываются по тем же записям, но с разным весом повторных исследований.
'''),code('''primary=metrics[metrics.policy=='primary_0.5'].copy()
display(primary[['profile','family','champion','patient_macro_f1','patient_recall','patient_specificity','patient_precision','patient_ap','patient_brier','record_macro_f1']].round(4))
alternate=metrics[(metrics.policy=='research_macro_threshold')&metrics.champion.astype(bool)]
display(alternate[['profile','family','threshold','patient_macro_f1','patient_recall','patient_specificity']].round(4))
fig,ax=plt.subplots(figsize=(12,5))
for j,family in enumerate(['catboost','spline_logit','linear']):
    rows=primary[primary.family==family].set_index('profile').loc[profiles]
    ax.plot(np.arange(len(profiles))+(j-1)*.07,rows.patient_macro_f1,'o-',color=COLORS[family],label=family)
ax.set(xticks=range(len(profiles)),xticklabels=profiles,ylabel='Patient-weighted record Macro-F1',title='Frozen test · primary threshold 0.5 · point estimates')
ax.legend();fig.tight_layout();save_show('03_frozen_test_comparison.png')
'''),md('''### Какие ошибки остаются · расширенный профиль

Все 4140 записей test, включая неполные входы, входят в знаменатель. В 04 отдельно показана доля отсутствующего core и технический отказ для таких бланков. На матрице число записей и доля внутри истинного класса; это экспертная FIGO-разметка, не клинические диагнозы.
'''),code('''pred=pd.read_parquet(OUT/'test_predictions.parquet')
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,family in zip(axes,['catboost','spline_logit','linear']):
    d=pred[(pred.profile=='sonomed_fields')&(pred.family==family)]
    cm=confusion_matrix(d.figo_label,d.p>=.5,labels=[0,1]);percent=cm/cm.sum(axis=1,keepdims=True)
    annotations=np.array([[f'{cm[i,j]}\\n{percent[i,j]:.1%}' for j in range(2)] for i in range(2)])
    sns.heatmap(percent,annot=annotations,fmt='',cmap='Blues',vmin=0,vmax=1,cbar=False,ax=ax,
        xticklabels=['normal','abnormal'],yticklabels=['normal','abnormal'])
    ax.set(title=family,xlabel='Predicted expert class',ylabel='Actual expert class')
fig.tight_layout();save_show('04_test_confusion.png')
'''),md('''### Что выучили модели · наблюдаемые зависимости

Показаны средние вероятности на **реальных train-строках**, сгруппированных по квантилям одного измерения, и доля экспертных abnormal в тех же группах; веса дают пациентке суммарно 1. Остальные поля сохраняются как наблюдались: искусственное изменение одного STV(ms) при фиксированном STV(BPM) нарушило бы связь единиц. Это описательные зависимости с коррелирующими признаками, **не причинный эффект и не оценка качества на новых данных**. Здесь заранее показан расширенный профиль. Train уже участвовал в fit; этот рисунок не выбирает новые признаки или модель.
'''),code('''profile='sonomed_fields';xt=inputs(train,profile)
models={f:joblib.load(EXPORT/f'{profile}_{f}.joblib') for f in ['catboost','spline_logit']}
fitted={f:bundle.predict_proba(xt)[:,1] for f,bundle in models.items()}
w=patient_weights(train.patient_id);y=train.figo_label.to_numpy()
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,col in zip(axes,['baseline_bpm','stv_ms','acc15_per_10min']):
    bins=pd.qcut(xt[col],10,duplicates='drop');summary=[]
    for group in bins.cat.categories:
        mask=(bins==group).to_numpy()
        summary.append(dict(x=np.average(xt.loc[mask,col],weights=w[mask]),
            observed=np.average(y[mask],weights=w[mask]),
            **{f:np.average(p[mask],weights=w[mask]) for f,p in fitted.items()}))
    means=pd.DataFrame(summary)
    for family in models:ax.plot(means.x,means[family],'o-',color=COLORS[family],label=family)
    ax.plot(means.x,means.observed,'s--',color='#555555',label='Observed expert abnormal')
    ax.set(xlabel=col,ylabel='Patient-weighted mean probability / fraction',ylim=(0,1))
axes[0].legend();fig.tight_layout();save_show('05_conditional_response.png')
'''),md('''### Экспорт и границы применения

Сохранены полные bundles с калибратором, native CatBoost и **отдельные fit-преобразования** каждой сплайновой/линейной модели. В ансамбле у каждого участника собственные fitted параметры. Model cards фиксируют цель, колонки, параметры, длительность и ограничения. [04](04_classifier_assessment.ipynb) независимо сверяет эти файлы.
'''),code('''display(pd.DataFrame({'profile':list(frozen['champions']),'train_selected_family':list(frozen['champions'].values())}))
print('Экспорты:',EXPORT.relative_to(ROOT))
print('Исходы неонатальной асфиксии не являются целью этого запуска.')
print('Неизвестный прибор, 17-min бланк и клинический риск не получили валидацию от этих внутренних оценок.')
'''),md('''## Takeaways

- Получены проверяемые кандидаты для каждого доступного набора полей. Полная конечная сетка и все результаты сохранены; результат зависит от информации на бланке.
- Второй подход проверяет плавные зависимости с регуляризацией. Его преимущество над CatBoost не предполагается заранее; линейный контроль может выигрывать на части профилей.
- В расширенном профиле CatBoost и сплайны дали F1 около 0.751; на train выбран линейный контроль (test 0.749), и этот выбор не подменяется test-максимумом. Для трёх полей F1 около 0.672, для пяти — 0.721. Добавление информации существенно полезнее наблюдаемой разницы семейств.
- Все финальные решения сохранили single: пять усредняемых моделей не дали достаточной пользы по предварительному критерию. Калибровка не обязана улучшать каждую test-метрику: raw/sigmoid сравнение полностью показано в 04, решения по test не отменяются.
- Отбор по качеству не уменьшал ни train, ни test. Неизвестные значения сопровождались масками, а все fit-преобразования обучались внутри групп.
- Это внутренние оценки expert normal/abnormal на JNU, а не решение о здоровье плода. До работы с реальными больничными цифрами нужны подтверждение измерений и внешняя дородовая валидация. Честные интервалы/ошибки/стресс-проверки — в следующем выполненном ноутбуке.
''')]
    save(BOOKS/'03_report_classifiers.ipynb',cells)


def build04():
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    cells=[md(intro04()),md('''## Context & Methods

### Key Assumptions

Независимая единица bootstrap — patient_component, а не запись. При повторном выборе компоненты её исходные patient-weights умножаются на кратность; веса нельзя пересчитать так, чтобы повторный выбор потерял смысл. AP и AUC учитывают совпадающие вероятности целыми группами порогов. Перекрытие CI не является полноценным тестом разницы; для CatBoost и spline приведены **парные** bootstrap-разницы на одних и тех же выборках.

Округление/отсутствие поля — заранее заданная численная чувствительность. Из двух единиц STV одновременно удаляются обе; иначе одна могла бы скрыто восстановить другую. При отсутствующем обязательном поле техническая политика должна отказывать или выбирать подтверждённый другой профиль; forced predictions в таблице не являются разрешением такого ввода в приложение.
'''),code(SETUP),md('''## Data · замороженные артефакты и запуск оценки

Проверяется хеш `selection_frozen.json`. Все дополнительные результаты описательные; они не изменяют модель. Отрицательные контроли используют фиксированные конфигурации только на train: перестановку между равноразмерными группами и дополнительно одну случайную запись на компоненту с полной перестановкой fit-меток. Второй добавлен после обнаружения ограничения первого; ни один не использует test для обучения/выбора.
'''),code('''from src.training.jnu_report_assessment import run_assessment,check_tree_prefixes
intervals,paired=run_assessment()
frozen=json.loads((OUT/'selection_frozen.json').read_text())
if not (OUT/'tree_prefix_reuse_checks.csv').exists():check_tree_prefixes()
prefix=pd.read_csv(OUT/'tree_prefix_reuse_checks.csv')
assert len(prefix)==36 and prefix.max_probability_difference.max()<=1e-13
display(prefix.groupby('depth').agg(combinations=('train_rows','size'),max_difference=('max_probability_difference','max')))
from tools.validate_jnu_classifiers import validate
with (OUT/'notebook_validation.log').open('w') as log,contextlib.redirect_stdout(log):
    receipt=validate()
display(pd.DataFrame(receipt['checks']))
'''),md('''## Results · интервалы качества и парные различия

95% percentile CI по 1000 кластерным повторным выборкам test. Выделены победители по train. Интервалы условны на зафиксированных моделях и данной популяции, не дают гарантии для другой клиники.
'''),code('''display(intervals[intervals.champion.astype(bool)&intervals.metric.isin(['macro_f1','recall','specificity','precision','ap'])].round(4))
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,metric in zip(axes,['macro_f1','recall']):
    for j,family in enumerate(['catboost','spline_logit','linear']):
        d=intervals[(intervals.metric==metric)&(intervals.family==family)].set_index('profile').loc[list(frozen['champions'])]
        pos=np.arange(len(d))+(j-1)*.2
        ax.errorbar(d.estimate,pos,xerr=np.array([d.estimate-d.ci_low,d.ci_high-d.estimate]),fmt='o',capsize=3,color=COLORS[family],label=family)
    ax.set(yticks=range(len(frozen['champions'])),yticklabels=list(frozen['champions']),xlabel=f'Patient-weighted {metric}',title='Frozen test · 95% component bootstrap CI')
axes[0].legend();fig.tight_layout();save_show('06_bootstrap_intervals.png')
display(paired.round(4))
'''),md('''### Precision–Recall и калибровка

Кривая PR показывает цену повышения recall в ложных тревогах. AP рассчитывается как average precision. Горизонтальная линия — patient-weighted распространённость abnormal в test. Калибровка выполнена на другой группе пациенток; показанные test-точки не использованы для её изменения. Reliability-диаграмма не является индивидуальной гарантией.

Эффект sigmoid небольшой и не во всех test-метриках положительный; например, у расширенного CatBoost log loss 0.5077 → 0.5083. Отказ от калибровки после просмотра test был бы новым выбором по test, поэтому зафиксированный экспорт не изменён. Положительная монотонная sigmoid сохраняет ранжирование и AP.
'''),code('''pred=pd.read_parquet(OUT/'test_predictions.parquet')
bins=pd.read_csv(OUT/'test_reliability_bins.csv')
fig,axes=plt.subplots(1,2,figsize=(12,5))
for family in ['catboost','spline_logit','linear']:
    d=pred[(pred.profile=='sonomed_fields')&(pred.family==family)]
    w=patient_weights(d.patient_id);precision,recall,_=precision_recall_curve(d.figo_label,d.p,sample_weight=w)
    axes[0].plot(recall,precision,color=COLORS[family],label=family)
    z=bins[(bins.profile=='sonomed_fields')&(bins.family==family)]
    axes[1].plot(z.mean_probability,z.observed_fraction,'o-',color=COLORS[family],label=family)
axes[0].axhline(np.average(d.figo_label,weights=w),ls='--',color='#555555',label='Abnormal prevalence')
axes[0].set(xlabel='Recall abnormal',ylabel='Precision abnormal',xlim=(0,1),ylim=(0,1),title='Extended report profile · frozen test')
axes[1].plot([0,1],[0,1],'--',color='#555555');axes[1].set(xlabel='Mean predicted probability',ylabel='Observed abnormal fraction',xlim=(0,1),ylim=(0,1),title='Test calibration · fixed probability bins')
axes[0].legend();fig.tight_layout();save_show('07_pr_and_calibration.png')
display(pd.read_csv(OUT/'calibration_changes.csv').round(4))
'''),md('''### Округление и структурные пропуски

Baseline округляется до 1 BPM, вариабельность до 0.1 единицы; частоты событий не округляются как будто это счётчики. Удаление STV убирает все его представления. Метрики — **принудительные исследовательские предсказания** для анализа чувствительности; отдельно показан охват разрешённых обязательных полей.
'''),code('''stress=pd.read_csv(OUT/'stress_tests.csv')
display(stress[['profile','family','scenario','core_coverage','complete_profile_coverage','patient_macro_f1','patient_recall','p95_probability_change','classification_flip_fraction']].round(4))
fig,ax=plt.subplots(figsize=(11,5))
for name,style in [('rounding','o-'),('missing_STV','s--'),('missing_baseline','^:')]:
    d=stress[stress.scenario==name].copy()
    ax.plot(d.profile,d.classification_flip_fraction,style,label=name)
ax.set(ylabel='Fraction of predictions changing class',title='Frozen champions · numerical stress, not clinical validation')
ax.legend();fig.tight_layout();save_show('08_measurement_stress.png')
'''),md('''### Что меняется на трудных записях

Quality-флаг из 02 не входит в X и не фильтровал обучение. Это описательная стратификация после заморозки. Для маленьких подгрупп метрики не публикуются; численность сохраняется. Patient weights пересчитываются внутри фактически оцениваемой подгруппы.

Для выбранной линейной модели расширенного профиля F1 на quality-pass 0.758, на quality-fail 0.627. Более высокий recall в трудной группе сопровождается другими распределениями классов и ошибками; он не означает лучшего клинического качества. В test 13 записей 12 пациенток не имеют всех core-измерений; они сохранены в общей оценке, отдельное качество такой малой группы не заявляется.
'''),code('''quality=pd.read_csv(OUT/'quality_subgroups.csv')
display(quality[['profile','family','cohort','records','patients','abnormal_records','metrics_supported','patient_macro_f1','patient_recall','patient_specificity','patient_precision']].round(4))
errors=pd.read_csv(OUT/'test_error_cases.csv')
hard=errors[(errors.profile=='sonomed_fields')&(errors.error_type=='false_negative')].sort_values('wrong_class_confidence',ascending=False).head(8)
display(hard[['record_id','figo_label','p','core_available','baseline_bpm','acc15_per_10min','dec15_per_10min','stv_ms','amplitude_bpm']].round(4))
print('Самые уверенные ошибочные normal требуют экспертного разбора; метки не исправляются догадкой и модель не переобучается по test.')
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,metric in zip(axes,['patient_macro_f1','patient_recall']):
    for name,marker in [('reference_quality_pass','o'),('reference_quality_fail','s')]:
        d=quality[quality.cohort==name]
        ax.plot(d.profile,d[metric],marker+'-',label=name)
    ax.set(ylabel=metric,ylim=(0,1),title='Frozen champions · descriptive quality groups')
    ax.tick_params(axis='x',labelrotation=25)
axes[0].legend();fig.tight_layout();save_show('09_quality_subgroups.png')
'''),md('''### Отрицательный контроль

Пять повторов каждого из двух контролей, фиксированные модели, только train. Перестановка между равноразмерными группами сохраняет часть связи «размер группы — метка»; её ROC-AUC может систематически отличаться от 0.5. После обнаружения такого эффекта добавлен второй диагностический контроль: ровно одна случайная запись на компоненту и полная перестановка её fit-меток. Выбор записи не использует target/сигнал и не считается первым по времени. Он не изменяет замороженные модели/test. Это **не формальные permutation p-values** и не доказательство отсутствия всех возможных утечек.

Средняя ROC-AUC первого контроля 0.530/0.534 (CatBoost/сплайны), второго 0.497/0.482. Пять повторов дают заметный разброс; это диагностическая проверка разрушения сигнала, а не оценка значимости результата. Близкие повторы сигналов за пределами найденных точных совпадений остаются отдельным риском источника.
'''),code('''null=pd.read_csv(OUT/'train_only_negative_controls.csv')
display(null[['control','family','repetition','actual_changed_label_fraction','patient_roc_auc','patient_macro_f1']].round(4))
fig,axes=plt.subplots(1,2,figsize=(13,4),sharey=True)
for ax,control in zip(axes,['same_size_blocks','one_record_per_component']):
    for family,offset in [('catboost',-.07),('spline_logit',.07)]:
        d=null[(null.family==family)&(null.control==control)]
        ax.scatter(d.repetition+offset,d.patient_roc_auc,color=COLORS[family],label=family,s=55)
    ax.axhline(.5,color='#555555',ls='--');ax.set(xticks=range(5),xlabel='Permutation repetition',ylabel='ROC-AUC on held-out train patients',ylim=(.35,.65),title=control)
axes[0].legend();fig.tight_layout();save_show('10_negative_control.png')
'''),md('''### Короткая запись: только изменение выхода модели

Те же 128 train-записей из 02, часть уже встречалась fit моделей. Здесь **нет** оценки качества, новых независимых пациенток или размеченной FIGO короткого окна. Сравнение показывает, насколько меняется ответ модели при другой информации во времени. Оно не разрешает использовать модель на 17-min бланке.
'''),code('''drift=pd.read_parquet(OUT/'duration_probability_drift.parquet')
display(drift.groupby('prefix_min').absolute_change.agg(['count','median','mean','max']).round(4))
fig,ax=plt.subplots(figsize=(9,4))
sns.boxplot(data=drift,x='prefix_min',y='absolute_change',color='#a6cbe3',ax=ax)
ax.set(xlabel='Prefix duration, min',ylabel='Absolute change in estimated probability',title='Train sensitivity only · no short-window accuracy claim')
fig.tight_layout();save_show('11_duration_probability_drift.png')
'''),md('''## Takeaways

- Проверено сохранение моделей/преобразований и границ независимости. 300-деревные точки сетки действительно совпадают с префиксами 700-деревных моделей при проверенных настройках.
- Все модели оценены на одной закрытой test-части после выбора решений. Модели/профили не перестраиваются по ошибкам этого test.
- Устойчивость к отсутствующим измерениям и качество на трудных записях видны отдельно от среднего score. Отсутствие обязательного поля требует явного маршрута/отказа, а не заполнения «нормой».
- У расширенного CatBoost пропущены 527 abnormal-записей из 2219. Даже хорошее среднее ранжирование AP 0.856 не устраняет эту ошибку при пороге 0.5. Отдельного доказанно безопасного клинического порога в работе нет.
- Прототип воспроизводит экспертный класс JNU. Нет независимой российской дородовой валидации, отдельного suspect/pathologic target, подтверждённого vendor bridge и валидированного риска асфиксии. Эти границы не устраняются большим Grid Search или узким внутренним CI.
''')]
    save(BOOKS/'04_classifier_assessment.ipynb',cells)


def save(path,cells):
    b=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (.venv)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
    nb.validate(b);nb.write(b,path);print(path)


def refresh_intros():
    for filename,fn in [('03_report_classifiers.ipynb',intro03),('04_classifier_assessment.ipynb',intro04)]:
        p=BOOKS/filename;b=nb.read(p,4);b.cells[0].source=fn();nb.validate(b);nb.write(b,p)


if __name__=='__main__':
    import sys
    if '--legacy' in sys.argv:
        BOOKS=ROOT/'research/notebooks/jnu_ctg/history/generated_classifiers_v1';BOOKS.mkdir(parents=True,exist_ok=True)
        if '--summaries-only' in sys.argv:refresh_intros()
        else:build03();build04()
    else:
        from build_ctg_review_notebooks import build_selected,summaries
        if '--summaries-only' in sys.argv:summaries()
        else:build_selected('jnu_ctg',[3,4])
