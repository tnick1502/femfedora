"""Current CTG notebooks; preserve v1 notebooks and existing clustering evidence."""
from pathlib import Path
import json
import shutil
import sys
import nbformat as nb
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'research/notebooks/ctg_uci'
OUT=ROOT/'research/experiments/ctg_uci/audited_v2'
NAMES=['02_nested_grid_search.ipynb','03_calibration_ensemble_and_test.ipynb',
       '05_results_and_error_analysis.ipynb','07_deep_audit_and_recommendations.ipynb']
SETUP='''from pathlib import Path
import sys,json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'pyproject.toml').exists())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.training.ctg_audit import (AUDIT,EXPORT,context,CTG_FEATURES,score,soft_vote)
data,fingerprint,cache=context()
sns.set_theme(style='whitegrid',font_scale=1.05)
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'figure.figsize':(10,5)})
def show_save(name):
    plt.gcf().savefig(AUDIT/name,dpi=140,bbox_inches='tight')
    plt.show()
'''


def save(name,parts):
    cells=[nb.v4.new_markdown_cell(s.strip()) if t=='md' else nb.v4.new_code_cell(s.strip()) for t,s in parts]
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (femfedora)','name':'python3','language':'python'},
        'language_info':{'name':'python','version':'3.11'}})
    nb.validate(book);nb.write(book,DEST/name)


def archive():
    history=DEST/'history/v1';history.mkdir(parents=True,exist_ok=True)
    for name in NAMES[:3]+['EXPERIMENT_PLAN.md']:
        source=DEST/name;target=history/name
        if source.exists() and not target.exists():shutil.copy2(source,target)
    for name in ['ctg_supervised_audit.json','ctg_supervised_recompute.json',
                 'ctg_data_cluster_audit.json','ctg_data_cluster_independent_checks.json']:
        source=ROOT/'tmp'/name
        if source.exists():shutil.copy2(source,OUT/name)


def build():
    OUT.mkdir(parents=True,exist_ok=True);archive()
    save(NAMES[0],[('md','''# 02 · Исправленная вложенная проверка и Grid Search · v2
## tl;dr
Полная повторная сетка шести семейств на новых общих фолдах. Число независимых записей каждого класса балансируется вместе с числом сегментов. Результаты появятся после запуска. Историческая версия сохранена в `history/v1/`.
'''),('md','''## Context & Methods

Цель — предсказание экспертного NSP, а не исхода ребёнка. Все 1360 исходных train-сегментов сохранены; validation/test не участвуют в этом ноутбуке. Нет идентификатора пациентки: исходный файл — доступная единица группировки.

### Key Assumptions

- Общие 5 внешних × 3 внутренних фолда. Детерминированное распределение файлов балансирует для каждого NSP и число сегментов, и присутствие класса в исходных файлах. Размеры крупных файлов могут мешать одинаковым долям строк. Фолды не подбираются по признакам или качеству модели.
- A: MultiClass, B: SqrtBalanced и Balanced, C: три OvR, D: abnormal и conditional pathologic, плюс логистика. Полный grid CatBoost: depth=3/5/7, iterations=300/600, learning_rate=.03/.1, L2=3/10; логистика C=.1/1/10 × None/Balanced. Итого 630 сочетаний «семейство × внешний фолд × параметры», по три внутренних оценки.
- Поиск деревьев использует средний внутренний macro-F1. Затем **внутри той же внешней обучающей части** выбираются калибровка и политика порога. Внешний holdout оценивает уже всю процедуру. Выбор конкретных параметров по raw score — явно заданная двухэтапная процедура, не глобальная совместная оптимизация всех возможных настроек.
- Калибровка: без изменения / scalar temperature / отдельные sigmoid по трём вероятностям с нормировкой. Метод выбирается по cross-fitted log loss, proper probability score. Для OvR калибровка не навязывается, если ухудшает log loss. Порог патологии оценивается отдельно по cross-fit macro-F1 и принимается при выигрыше ≥.005. Финальный порог на fit-стороне обучается по cross-fitted вероятностям.
- Внешний test-фолд не используется для scaler fit, калибровки, порогов или early stopping. Логистика клонирует и заново обучает подготовку внутри каждого fit. CatBoost использует исходные измерения без скейлера; DS/Width исключены, Tendency категориальная. Веса классов не дублируются.
- Основная метрика остаётся segment macro-F1. Дополнительно: recall/AP патологии, log loss и file-balanced метрики (сначала равновероятный файл, затем сегмент внутри него). Они отвечают разным вопросам; одна не подменяет другую.
- Выбор: лучший средний внешний macro-F1; максимум два семейства в пределах .01 от лучшего входят в shortlist. Это правило задано до нового test. Оценка выбранного победителя всё ещё имеет оптимизм из-за выбора семейства; существующий test уже известен исследованию.
'''),('code',SETUP),('md','## Data'),('code', '''display(data[data.split=='train'].groupby('NSP').agg(segments=('NSP','size'),source_files=('source_file','nunique')))
from src.training.ctg_audit import CAT_GRID,LOG_GRID
display(pd.DataFrame(CAT_GRID));print('Logistic grid:',LOG_GRID)
'''),('md','## Results · выполнить полную вложенную процедуру'),('code','''from src.training.ctg_audit import run_nested
summary,folds=run_nested()
display(summary.round(4))
'''),('md','### Независимая поддержка каждого фолда'),('code','''support=pd.read_csv(AUDIT/'outer_fold_support.csv')
display(support)
old=pd.read_csv(ROOT/'research/experiments/ctg_uci/outer_folds.csv').merge(data[['source_excel_row','source_file','NSP']],on='source_excel_row',validate='one_to_one')
old_counts=old[old.NSP==3].groupby('outer_fold').source_file.nunique()
fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(5)
ax.bar(x-.18,old_counts.reindex(range(5)),.36,label='Historical v1')
ax.bar(x+.18,support.path_files,.36,label='Audited v2')
ax.set(xlabel='Outer fold',ylabel='Recordings containing pathologic segments',title='Balance independent positive recordings, not only segments')
ax.set_xticks(x);ax.legend();fig.tight_layout();show_save('01_fold_support.png')
'''),('md','### Качество всей процедуры на внешних фолдах'),('code','''fig,axes=plt.subplots(1,2,figsize=(13,4.8))
for ax,metric in zip(axes,['macro_f1','recall_path']):
    sns.stripplot(data=folds,x=metric,y='family',ax=ax,jitter=.08,color='#0072b2')
    ax.set(xlim=(0,1),title=f'{metric} · complete nested policy',ylabel='')
fig.tight_layout();show_save('02_nested_scores.png')
display(folds[['family','outer_fold','macro_f1','recall_path','file_macro_f1','file_recall_path']].round(4))
print('Фолды имеют пересекающиеся train: SD не является доверительным интервалом.')
'''),('md','## Takeaways\n\nСравнивается весь алгоритм обучения, а не лучший postprocessing, выбранный после просмотра внешних результатов. Shortlist и параметры сохраняются до test; утверждение о единственном универсальном победителе не делается.')])
    save(NAMES[1],[('md','''# 03 · Исправленная калибровка, soft voting и test · v2
## tl;dr
Финальная сетка на исходном train, выбор калибровки/порога и single/ensemble на validation, затем описательный повторный test. Семейства shortlist уже выбраны в 02; новая независимая тестовая когорта отсутствует.
'''),('md','''## Context & Methods

### Key Assumptions

- Деревья fit только на train. Для каждого семейства новая полная 3-fold сетка из 24 комбинаций (логистика 6). Финальная калибровка использует отдельную validation и group cross-fitting для выбора метода. Деревья после калибровки на validation не переобучаются.
- Для shortlist дополнительно обучаются пять моделей на 4/5 train. Усредняются **три вероятности классов каждой модели**. Для иерархии сначала строятся [1−a,a(1−b),ab], затем среднее: E[ab] отличается от E[a]E[b]. Калибровка отдельно обучается на выходе именно этого среднего.
- Калибровка выбирается по log loss, решение порога — по cross-fitted macro-F1 с минимальным выигрышем .005. Ансамбль принимается только при приросте validation macro-F1 ≥.01. Обе границы зафиксированы до нового test.
- В validation лишь семь исходных файлов с патологией. Даже баланс независимых файлов не позволяет разделить большой файл на независимые наблюдения. Выбор postprocessing может быть шумным; улучшение на validation не гарантирует улучшение test.
- Внутренние cross-fit числа служат подбору: базовые OOF-прогнозы и новые calibration-фолды не дают полностью независимой meta-CV внутри fit. Кроме того, holdout-порог fit по вероятностям своего обученного calibrator, а итоговый cutoff — по cross-fitted probabilities. Поэтому validation-оценка архитектуры приблизительна; внешняя CV02 независимо оценивает фактически возвращаемую процедуру. Эти числа не называются selection-adjusted оценкой уже выбранного среди шести семейств победителя.
- Файл решения сохраняется до test predict. Ранее известный test используется для исследовательской перепроверки, не для выбора семейства, параметров или архитектуры. Интервалы на test — paired bootstrap исходных файлов, условные на фиксированных моделях, без разброса переобучения.
'''),('code',SETUP),('md','## Data'),('code','''selection=json.loads((AUDIT/'selection.json').read_text())
print('Primary family:',selection['primary']);print('Shortlist:',selection['shortlist'])
display(data.groupby('split').agg(segments=('source_file','size'),recordings=('source_file','nunique')))
'''),('md','## Results · финальный поиск и фиксация'),('code','''from src.training.ctg_audit import run_final
results,locked=run_final()
print('Primary:',locked['primary']);print('Exported shortlist:',locked['shortlist'])
'''),('md','### Validation: поддержка и правила'),('code','''display(pd.read_csv(AUDIT/'validation_fold_support.csv'))
v=pd.read_csv(AUDIT/'validation_policy_metrics.csv')
display(v[['candidate','calibration','decision','macro_f1','recall_path','log_loss']].round(4))
display(pd.DataFrame(locked['policies']).T)
'''),('md','### Исследовательский test после фиксации'),('code','''display(results[['candidate','primary','shortlisted','macro_f1','recall_path','precision_path','ap_path','log_loss','brier','file_macro_f1','file_recall_path']].round(4))
display(pd.read_csv(AUDIT/'test_paired_intervals.csv').round(4))
'''),('md','### Сохранённые модели и входной контракт'),('code','''bundle=joblib.load(EXPORT/'champion.joblib')
example=data.loc[data.split=='test',list(CTG_FEATURES)].head(3)
display(pd.DataFrame(bundle.predict_proba(example),columns=['P(normal)','P(suspect)','P(pathologic)']))
print('NSP:',bundle.predict(example));print('Model:',EXPORT/'champion.joblib')
print('Карточка модели:',EXPORT/'model_card.json')
from src.training.ctg_experiments import digest,write_json
write_json(EXPORT/'best_models.json',{'primary':locked['primary'],'selection':'development only; see model_card.json',
    'models':[{'candidate':c,'path':str(EXPORT/f'{c.replace("/","_")}.joblib'),
    'sha256':digest(EXPORT/f'{c.replace("/","_")}.joblib')} for c in locked['shortlist']]})
'''),('md','## Takeaways\n\nОсновной и альтернативный варианты выбираются по развитию модели, а не по рангу на test. Грид даёт лучшую проверенную конфигурацию этой процедуры; ограниченность данных и известный test исключают гарантию глобального или клинического превосходства.')])
    save(NAMES[2],[('md','''# 05 · Лучшие модели, ошибки и неопределённость · v2
## tl;dr
Результаты audited v2 и независимые проверки сохранённых предсказаний. Основная единица метрики — сегмент; дополнительно учитывается равный вклад исходных файлов.
'''),('md','''## Context & Methods

Основной результат выбран без нового test. Macro-F1, recall/AP патологического класса и proper probability scores нужны вместе. AP — Average Precision, не трапецеидальная площадь под PR-кривой. Bootstrap по исходным файлам отражает неопределённость состава данного test для фиксированной модели; он не оценивает весь разброс обучения или переноса на другую клинику. NSP — экспертная разметка КТГ, а не наблюдаемый исход.
'''),('code',SETUP),('md','## Data & Results'),('code','''results=pd.read_csv(AUDIT/'test_metrics.csv');locked=json.loads((AUDIT/'locked_decision.json').read_text())
predictions=pd.read_parquet(AUDIT/'test_predictions.parquet')
primary=locked['primary'];shortlist=locked['shortlist']
display(results.round(4))
test=data[data.split=='test'].reset_index(drop=True)
prior=data[data.split=='train'].NSP.value_counts(normalize=True).reindex([1,2,3]).to_numpy()
baseline=score(test,np.tile(prior,(len(test),1)))
display(pd.Series(baseline,name='always_normal').to_frame().round(4))
'''),('md','### Матрицы ошибок shortlist'),('code','''from sklearn.metrics import confusion_matrix,average_precision_score,precision_recall_curve,roc_curve
fig,axes=plt.subplots(1,len(shortlist),figsize=(6*len(shortlist),4.8),squeeze=False)
for ax,candidate in zip(axes[0],shortlist):
    p=predictions[predictions.candidate==candidate]
    sns.heatmap(confusion_matrix(p.NSP,p.pred,labels=[1,2,3]),annot=True,fmt='d',cmap='Blues',
        xticklabels=['Normal','Suspect','Pathologic'],yticklabels=['Normal','Suspect','Pathologic'],ax=ax,cbar=False)
    ax.set(title=candidate,xlabel='Predicted expert class',ylabel='Expert reference class')
fig.tight_layout();show_save('03_confusion.png')
'''),('md','### Распознавание патологического класса'),('code','''fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for candidate in shortlist:
    p=predictions[predictions.candidate==candidate];y=p.NSP.to_numpy()==3
    precision,recall,_=precision_recall_curve(y,p.p2)
    axes[0].step(recall,precision,where='post',label=f'{candidate} · AP={average_precision_score(y,p.p2):.3f}')
    fpr,tpr,_=roc_curve(y,p.p2);axes[1].plot(fpr,tpr,label=candidate)
axes[0].axhline((test.NSP==3).mean(),ls='--',color='.4',label='Prevalence baseline')
axes[1].plot([0,1],[0,1],'--',color='.5')
axes[0].set(xlabel='Recall pathologic',ylabel='Precision pathologic',title='Precision–recall · 35 pathologic test segments')
axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC · same test segments')
for ax in axes:ax.legend(fontsize=8);ax.set_xlim(0,1);ax.set_ylim(0,1.03)
fig.tight_layout();show_save('04_path_curves.png')
'''),('md','### Описательная проверка вероятностей\n\nПять квантильных групп вероятностей на известном test. Точки показывают средний прогноз и наблюдаемую долю экспертного класса; это не доказательство идеальной калибровки. Log loss/Brier учитывают также различимость и неопределённость данных, поэтому не являются чистой мерой калибровки. Сегменты одной записи зависимы; полосы независимых биномиальных ошибок здесь не используются.'),('code','''p=predictions[predictions.candidate==primary].copy()
fig,axes=plt.subplots(1,3,figsize=(13,4.2))
for j,(ax,name) in enumerate(zip(axes,['Normal','Suspect','Pathologic'])):
    bins=pd.qcut(p[f'p{j}'],q=5,duplicates='drop')
    tab=pd.DataFrame({'bin':bins,'probability':p[f'p{j}'],'observed':p.NSP==j+1}).groupby('bin',observed=True).agg(mean_probability=('probability','mean'),fraction=('observed','mean'),segments=('observed','size'))
    ax.plot([0,1],[0,1],'--',color='.6');ax.plot(tab.mean_probability,tab.fraction,'o-',color='#0072b2')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed fraction',title=f'{name} · descriptive reliability')
fig.tight_layout();show_save('05_probability_reliability.png')
'''),('md','### Неопределённость и вклад исходных записей'),('code','''display(pd.read_csv(AUDIT/'test_paired_intervals.csv').round(4))
p=predictions[predictions.candidate==primary].copy()
p['path_true']=p.NSP==3;p['path_found']=(p.NSP==3)&(p.pred==3)
support=p[p.NSP==3].groupby('source_file').agg(path_segments=('path_true','sum'),found=('path_found','sum'))
support['recall']=support.found/support.path_segments
display(support)
support.to_csv(AUDIT/'pathologic_recording_errors.csv')
comparison=results[results.shortlisted].set_index('candidate')[['macro_f1','file_macro_f1','recall_path','file_recall_path']]
display(comparison.round(4))
'''),('md','### Независимый пересчёт и roundtrip всех текущих моделей'),('code','''from sklearn.metrics import f1_score,log_loss,recall_score
checked=[]
for candidate,part in predictions.groupby('candidate'):
    row=results[results.candidate==candidate].iloc[0]
    p=part[['p0','p1','p2']].to_numpy();y=part.NSP.to_numpy()-1
    np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
    assert abs(f1_score(y,part.pred.to_numpy()-1,average='macro')-row.macro_f1)<1e-12
    assert abs(recall_score(y==2,part.pred.to_numpy()==3)-row.recall_path)<1e-12
    assert abs(average_precision_score(y==2,p[:,2])-row.ap_path)<1e-12
    assert abs(log_loss(y,p,labels=[0,1,2])-row.log_loss)<1e-12
    model=joblib.load(EXPORT/f'{candidate.replace("/","_")}.joblib')
    frame=data.set_index('source_excel_row').loc[part.source_excel_row]
    np.testing.assert_allclose(model.predict_proba(frame[list(CTG_FEATURES)]),p,atol=1e-12)
    np.testing.assert_array_equal(model.predict(frame[list(CTG_FEATURES)]),part.pred)
    checked.append(candidate)
print('Independently checked:',checked)
'''),('md','## Takeaways\n\nРезультаты характеризуют данную выборку и процедуры. Длинные записи не являются множеством независимых пациенток; поэтому рядом показаны поддержка исходными файлами и file-balanced метрики. Клиническая применимость и превосходство на другом источнике здесь не установлены.')])
    save(NAMES[3],[('md','''# 07 · Глубокий аудит данных, методов, кода и выводов
## tl;dr
Исправления реализованы в audited v2; прежние результаты сохранены для сравнения. Подготовка исходных данных и scaler fit прошли независимую проверку. Главные изменения относятся к оценке моделей и интерпретации результатов.
'''),('md','''## Context & Methods

Область аудита: исходная XLS-книга и primary data, ноутбуки 01–06, подготовка, nested CV, калибровка/пороги, soft voting, классификационные и кластеризационные метрики, сохранение моделей/кэша, существенные выводы. Проверки выполнены из исходных строк и сохранённых прогнозов, не только повторным вызовом автора метрик. Проверка повторяющихся графиков исторических ноутбуков выборочная; все новые существенные графики проверяются после исполнения. Взаимодействующих dashboard-контролов нет.

### Критические выводы и исправления

| Приоритет | Наблюдение и значение | Исправление / предел |
|---|---|---|
| P1 | Для hierarchy усреднялись a,b до их произведения; это другой ансамбль | В v2 soft voting трёх вероятностей каждого участника, отдельная калибровка среднего; добавлен независимый арифметический тест |
| P1 | При ровных числах патологических сегментов старый outer-фолд имел лишь 2 патологических файла; calibration holdout — 1 | Фолды балансируют и class segments, и class-presence files; таблицы поддержки сохраняются |
| P1 | В v1 режим postprocessing выбирался после просмотра всех внешних результатов и затем менялся на validation | В v2 выбор калибровки/порога входит внутрь каждой внешней обучающей части; внешний holdout оценивает всю процедуру |
| P2 | Калибровку оценивали преимущественно по macro-F1 | Теперь метод калибровки выбирается по cross-fitted log loss, порог решения отдельно по macro-F1 |
| P2 | Число сегментов переоценивает независимую опору, различия A/D по test не доказывают уникального победителя | Сохранена основная segment-метрика; добавлены file-balanced показатели и paired file bootstrap, shortlist выбирается по развитию модели |
| P2 | Старый cache fingerprint не учитывал live guard source и версии | Новая версия включает исходники guard/алгоритма/фолдов, данные/шаблон подготовки и runtime; результат имеет проверяемые SHA-256 |
| P2 | UMAP transform меняет часть назначений при другом batch | Оставлен исследовательским методом; проверенные 1/20 изменений GMM не скрыты, не экспортируется как supervised champion |
| P2 | Высокий ARI/44.4% патологии AE+GMM опирается на 2 патологических файла и неустойчив к seed | Не используется для выбора классификатора; показатели и ограничения 06 сохранены, добавлена общая интерпретация |

### Что подтверждено

- Все 2126×21 значений независимо восстановлены из `Raw Data`: события AC/FM/UC/DL/DS/DP поделены на e−b, остальные признаки сопоставлены напрямую. Расхождений нет.
- Нет пропусков, пересечений исходных файлов и feature duplicates между train/validation/test. Нули имеют содержательный смысл. Скейлеры совпадают с новым fit только на train; общий fit дал бы отличия до .3214 в проверенной матрице.
- Контракт входа проверяет ровно 21 признак и запрещает неподдерживаемые пропуски/нечисловые значения/неверные категории. Импутация по отсутствующему в train сценарию не добавлена.
- Метки NSP/CLASS не входят в признаки; scaler и UMAP/AE не fit на test. Group early stopping AE корректен. Не обнаружена прямая утечка test в обучение исторических деревьев.
- Старые test macro-F1/recall/AP пересчитаны независимо и совпали. Исправление ансамбля не меняет старую одиночную модель; старая оценка ансамбля признана исторической.

Старая версия 02/03/05 и плана — `history/v1/`. 04 и 06 сохранены побайтно, включая отрицательные результаты; их методы не стали клиническими моделями. Текущие рабочие точки и экспорт — только audited v2.
'''),('code',SETUP),('md','## Data · доказательства аудита'),('code','''for name in ['ctg_supervised_audit.json','ctg_data_cluster_audit.json']:
    evidence=json.loads((AUDIT/name).read_text())
    print(name)
    if 'findings' in evidence:
        display(pd.DataFrame([{'severity':r['severity'],'issue':r.get('issue',r.get('id'))} for r in evidence['findings']]))
display(pd.read_csv(AUDIT/'outer_fold_support.csv'))
display(pd.read_csv(AUDIT/'validation_fold_support.csv'))
'''),('md','## Results · текущие лучшие процедуры'),('code','''nested=pd.read_csv(AUDIT/'nested_summary.csv');test=pd.read_csv(AUDIT/'test_metrics.csv')
lock=json.loads((AUDIT/'locked_decision.json').read_text())
display(nested.round(4));display(test[test.shortlisted].round(4))
display(pd.read_csv(AUDIT/'test_paired_intervals.csv').round(4))
print('Рекомендуемый основной bundle:',EXPORT/'champion.joblib')
print('Shortlist:',lock['shortlist'])
'''),('md','### Кластеризация: поддержка исходными файлами'),('code','''risk=pd.read_csv(ROOT/'research/experiments/ctg_uci/embeddings/test_risk_summary_with_support.csv')
display(risk[(risk.method=='AE')&(risk.algorithm=='GMM')][['cluster','segments','pathological','path_source_files',
    'path_fraction','path_enrichment_low','path_enrichment_high','limited_path_support']].round(4))
stability=pd.read_csv(ROOT/'research/experiments/ctg_uci/embeddings/stability.csv')
display(stability[(stability.method=='AE')&(stability.algorithm=='GMM')].round(4))
print('Высокая доля патологических сегментов и высокий ARI на одном test не подтверждают устойчивые группы риска.')
'''),('md','''## Assessment · качество и охват

**Готово для воспроизводимого исследовательского сравнения; использовать с оговорками.** Исправления выполнены. Полнота: проверены все основные этапы, а не все возможные датасеты, seed, гиперпараметры или исторические графики. Счётчики ниже относятся к указанной области, не являются процентом научной истинности. Нулевой остаток наблюдённых дефектов не доказывает отсутствие всех ошибок.

| Категория качества ноутбуков | Наблюдённые остаточные дефекты | Оценка |
|---|---|---|
| Полезность и полнота постановки | 0 / 7 ноутбуков | Данные, поиск, калибровка, классификация, кластеры и аудит доступны; независимая внешняя когорта отсутствует |
| Аналитическая ясность | 0 / 4 обновлённых ноутбуков | Основной estimand сохранён, альтернативный определён; история отделена от текущей процедуры |
| Визуальная согласованность | 0 / 5 новых рисунков | Все новые рисунки проверены визуально; исторические повторяющиеся представления проверялись выборочно |
| Интерактивные dashboard-контролы | N/A | Статические исследовательские ноутбуки |

| Аналитическая корректность | Наблюдённые остаточные дефекты | Оценка |
|---|---|---|
| Источник и уверенность | 0 / 3 опорных контракта | Сырые признаки, экспертные метки, исходный файл как proxy; пациентская независимость не установлена |
| Значения и вычисления | 0 / 8 строк моделей | Все строки test-метрик текущих экспортированных моделей пересчитаны независимо в 05 и по ручным формулам в validation receipt |
| Полнота происхождения | 0 / 3 этапа | Подготовка, настройка и экспорт имеют хэши/версии и сохраняемые таблицы |
| Согласованность артефактов | 0 / 4 обновлённых ноутбуков | Исторические результаты сохраняются и явно помечаются, текущий путь модели указан |
| Качество данных | 0 / 4 контракта | Source alignment, grouped boundaries, missing/domain rules, train-only scaler |
| Поддержка выводов | 0 / 3 главных вывода | Классификация экспертного NSP, отсутствие доказанной уникальной лучшей модели, отсутствие устойчиво подтверждённых риск-кластеров |

## Takeaways

1. «Лучшая модель» здесь означает выбранную без нового test процедуру среди проверенных конфигураций. Не заявляются идеальность, глобальный оптимум или готовность к клиническим решениям.
2. Новый основной bundle и возможная альтернатива содержат проверку входа, нужную подготовку, деревья, правильное усреднение, калибровку и порог. Отдельные `.cbm` не заменяют bundle.
3. Известный test нельзя снова сделать слепым. Для следующего утверждения о переносе нужна независимая когорта с идентификаторами пациенток и клинически определённой целью.

Источники методики: [UCI](https://archive.ics.uci.edu/dataset/193/cardiotocography), [nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [calibration](https://scikit-learn.org/stable/modules/calibration.html), [soft voting](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier), [UMAP clustering caveats](https://umap-learn.readthedocs.io/en/latest/clustering.html).
''')])


def summaries():
    if not (OUT/'test_metrics.csv').exists():return
    s=pd.read_csv(OUT/'nested_summary.csv');r=pd.read_csv(OUT/'test_metrics.csv')
    lock=json.loads((OUT/'locked_decision.json').read_text());champ=r[r.primary].iloc[0]
    text={NAMES[0]:f'''# 02 · Исправленная вложенная проверка и Grid Search · v2
## tl;dr
Выполнен полный поиск **630 сочетаний** с тремя внутренними оценками и пятью внешними фолдами. Лидер по среднему внешнему macro-F1: **{s.iloc[0].family}**, **{s.iloc[0].mean_macro_f1:.4f}**. Shortlist: **{', '.join(lock['shortlist'])}**. Калибровка/порог выбирались внутри внешней обучающей части; баланс учитывает независимые записи каждого класса. История сохранена в `history/v1/`.
''',NAMES[1]:f'''# 03 · Исправленная калибровка, soft voting и test · v2
## tl;dr
По развитию модели выбран **{lock['primary']}**. После фиксации правил исследовательский повторный test: macro-F1 **{champ.macro_f1:.4f}**, recall патологии **{champ.recall_path:.1%}**, AP патологии **{champ.ap_path:.4f}**. Архитектура/калибровка выбирались по групповой validation, не по test. Ансамбль усредняет вероятности классов; старая ошибка перемножения средних исправлена в новой реализации.
''',NAMES[2]:f'''# 05 · Лучшие модели, ошибки и неопределённость · v2
## tl;dr
Основная процедура **{lock['primary']}**: test macro-F1 **{champ.macro_f1:.4f}**, recall патологического класса **{champ.recall_path:.1%}**, AP **{champ.ap_path:.4f}**. При равном суммарном весе каждого исходного файла macro-F1 **{champ.file_macro_f1:.4f}**, recall патологии **{champ.file_recall_path:.1%}**. Эти метрики описывают разные единицы усреднения. Test был известен до аудита; внешняя клиническая проверка не заменена.
''',NAMES[3]:f'''# 07 · Глубокий аудит данных, методов, кода и выводов
## tl;dr
Исправлены агрегация hierarchy-ансамбля, баланс независимых патологических записей, вложенный выбор postprocessing и критерий калибровки. Все шесть семейств повторно прошли полную сетку. Основной экспорт: **{lock['primary']}**; shortlist **{', '.join(lock['shortlist'])}**. Test macro-F1 **{champ.macro_f1:.4f}**, recall патологии **{champ.recall_path:.1%}**, AP **{champ.ap_path:.4f}**. Прежние результаты сохранены. Источник и train-only scalers прошли независимую проверку. Уникальное универсальное превосходство и клиническая готовность не установлены.
'''}
    for name,summary in text.items():
        p=DEST/name;b=nb.read(p,as_version=4);b.cells[0].source=summary.strip();nb.validate(b);nb.write(b,p)


if __name__=='__main__':
    if '--legacy' in sys.argv:
        DEST=ROOT/'research/notebooks/ctg_uci/history/generated_v2';DEST.mkdir(parents=True,exist_ok=True)
        if '--summaries-only' not in sys.argv:build()
        summaries()
    else:
        from build_ctg_review_notebooks import build_selected,summaries as current_summaries
        if '--summaries-only' not in sys.argv:build_selected('ctg_uci',[2,3,5,7])
        current_summaries()
    print(DEST)
