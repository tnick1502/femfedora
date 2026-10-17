"""Rewrite all 12 active CTG notebooks; preserve pre-review books in the archive."""
from pathlib import Path
import json
import nbformat as nb
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'research/reviews/ctg_20261017'
UCI=ROOT/'research/experiments/ctg_uci/reviewed_v3'
JNU=ROOT/'research/experiments/jnu_ctg/reviewed_v2'
MD=nb.v4.new_markdown_cell;CODE=nb.v4.new_code_cell

SETUP='''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'pyproject.toml').exists())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
REVIEW=ROOT/'research/reviews/ctg_20261017'
UCI=ROOT/'research/experiments/ctg_uci/reviewed_v3'
JNU=ROOT/'research/experiments/jnu_ctg/reviewed_v2'
FIGURES=REVIEW/'figures';FIGURES.mkdir(exist_ok=True)
pd.set_option('display.max_colwidth',170)
sns.set_theme(style='whitegrid',font_scale=1.0)
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
COLORS={'linear':'#009e73','spline':'#d55e00','catboost':'#0072b2'}
def save_show(name):
    plt.gcf().savefig(FIGURES/name,dpi=140,bbox_inches='tight')
    plt.show()
'''

LIMIT_UCI='''## Takeaways

UCI используется как benchmark экспертных классов. Префикс — прокси независимого источника, не подтверждённый ID пациентки. Все данные ранее просмотрены: эта ревизия оценивает исправленную процедуру на повторной nested CV, без нового blind-test заявления. Полный разбор 23 замечаний — в [07](07_deep_audit_and_recommendations.ipynb).
'''
LIMIT_JNU='''## Takeaways

Цель — морфологическая метка normal/abnormal JNU. Она не заменяет клинический исход или полные критерии Доуза–Редмана. Совместимость с российским прибором не подтверждена; общий вход приложения возвращает отказ до модели. Сырые сигналы нужны только для исследования измерений, не обязательны для будущего приложения.
'''


def intro(series,number):
    if series=='ctg_uci':
        titles={1:'Подготовка UCI и более строгие группы',2:'Повторная вложенная CV и Grid Search',3:'Калибровка внутри CV и экспорт benchmark-моделей',4:'Кластеризация: сохранённый отрицательный результат',5:'Неопределённость и ошибки всех трёх классов',6:'UMAP и автоэнкодер: проверка прежней кластерной ветки',7:'Критический ответ на 23 замечания эксперта',8:'Единый вход больничного бланка и границы UCI'}
        s=f'# {number:02d} · {titles[number]}\n\n## tl;dr\n\n'
        summaries={1:'Старые части UCI разделяли файлы, но 38 групп префиксов пересекали train/validation/test. Новая нормализация даёт 177 консервативных групп. Файл и префикс не являются подтверждённым ID пациентки. Первичные данные сохраняются; основная оценка теперь по повторным групповым фолдам.',
        2:'Подбор шести семейств выполнен по group-balanced log loss с эвристикой One-SE. Внешняя оценка — 3 повтора ×5 фолдов, внутри — 3 фолда. Параметры, семья и калибровка выбираются без внешних меток. Все 2126 сегментов уже известны исследователю; это исправленный exploratory benchmark.',
        3:'Калибровка фиксированной температурой обучается на OOF текущей fit-части. Внешние группы в неё не входят. Порог argmax фиксирован; маленькая прежняя validation больше не выбирает несколько политик. Финальные модели fit на всей UCI и не имеют отдельного независимого test-score.',
        4:'Прежние FCM/KMeans/GMM не дали устойчивого клинического разделения. Все расчёты сохранены как исторические; прежние разбиения не доказывают независимость пациенток. Ветка закрыта для дальнейшего поиска диагнозов по кластерам.',
        5:'Показаны ошибки normal, suspect и pathologic, интервалы по группам префиксов и концентрация патологических сегментов. CI условны на выполненных CV fits; разброс повторов не считается набором независимых исследований.',
        6:'Сохранён поиск 1650 сочетаний сжатия/кластеризации. Исключение из низких ARI — AE+GMM с test ARI 0.216, но median seed-stability ARI около 0.032. Это не подтверждённый кластер риска; нового перебора по меткам нет.',
        7:'Каждое замечание проверено отдельно: факт, интерпретация, исправление и источник. Приняты более строгая UCI-группировка, отказ до модели при неподтверждённом бланке, удаление дублирующих единиц и честное сравнение усреднения. Гарантии «0 остаточных дефектов» удалены.',
        8:'У приложения один публичный контракт — src.features.ctg_report_contract. Отсутствующее поле и неизвестное определение останавливают оценку до predict_proba. SisPorto-поля UCI остаются отдельным native benchmark import; STV(ms) не приравнивается к MSTV/ASTV.'}
        s+=summaries[number]+'\n'
        if number in [2,3,5] and (UCI/'repeat_metrics.csv').exists():
            d=pd.read_csv(UCI/'repeat_metrics.csv');z=d[d.family=='selected_procedure']
            s+=f"\nИсправленная процедура: mean group-balanced Macro-F1 **{z.group_macro_f1.mean():.3f}**, recall suspect **{z.group_recall_suspect.mean():.1%}**, recall pathologic **{z.group_recall_path.mean():.1%}**. Это средние метрики трёх повторов, не точность финального full-data fit.\n"
        return s
    titles={1:'JNU: подготовка и повторная проверка источника',2:'Поля бланка: определения, единицы и отказ до модели',3:'Исправленные классификаторы и сигнальные ориентиры',4:'Проверка цели, ошибок, охвата и переносимости'}
    s=f'# {number:02d} · {titles[number]}\n\n## tl;dr\n\n'
    s+={1:'Сохранена глубокая подготовка 20769 записей / 12606 пациенток. Дополнительно проверены связь опубликованных U_* с каналами, UC-максимумы 100/127 и нестабильность морфологической метки между разными КТГ одной беременности. Это не доказательство перестановки сырых каналов или ровно двух типов аппаратов.',
    2:'Измерения v1 сохранены. Minute-range включает акцелерации и не является автоматически амплитудой конкретного прибора. Добавлен train-only probe с исключением событий; новый descriptor не объявляется эталоном. Для новых моделей — одна единица на величину и строгий отказ при неполном/неподтверждённом входе.',
    3:'Новые CatBoost, сплайны и линейные модели работают без quality/duration, дублирующих единиц и missing indicators. Четыре профиля, полные обязательные поля, 5×3 grouped nested CV. Для single/ensemble выделена отдельная mode-группа, для калибровки — другая. Сигнальные ориентиры используют те же внешние группы.',
    4:'Проверены условные интервалы, отдельный столбец отказов, трудные записи, synthetic probes, связь с исходом и изменение PPV при гипотетической распространённости. Все данные уже изучались; новая ревизия не создаёт независимого клинического test.'}[number]+'\n'
    if number in [3,4] and (JNU/'profile_metrics.csv').exists():
        d=pd.read_csv(JNU/'profile_metrics.csv');z=d[d.family=='selected_procedure']
        s+='\n| Профиль | Macro-F1 на принятых | Recall abnormal | Охват JNU |\n|---|---:|---:|---:|\n'
        for _,r in z.iterrows():s+=f'| {r.profile} | {r.group_macro_f1:.3f} | {r.group_recall:.1%} | {r.coverage:.2%} |\n'
        s+='\nОхват JNU относится к полноте собственных измерений; он не означает такой же охват больничных отчётов с неизвестными определениями.\n'
    return s


def save(series,filename,cells):
    path=ROOT/'research/notebooks'/series/filename
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (.venv)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
    nb.validate(book);nb.write(book,path);print(path)


def uci01():
    save('ctg_uci','01_prepare_data.ipynb',[MD(intro('ctg_uci',1)),MD('''## Context & Methods
### Key Assumptions
Нормализация имён консервативна: буквенный initials-like префикс объединяется, S+цифры остаются самостоятельными кодами; регистр игнорируется. Дополнительно объединяются feature-identical группы. Это снижает риск зависимых источников, не восстанавливает истинные patient IDs. Все распределения ниже описательны на уже известном источнике.
'''),CODE(SETUP),MD('## Data · первичная таблица и роли полей'),CODE('''from src.training.ctg_review_evidence import uci_groups
from src.features.ctg_preprocessing import CTG_FEATURES
d=uci_groups();x=d[list(CTG_FEATURES)]
assert len(d)==2126 and x.notna().all().all()
print('Сегментов:',len(d),'файлов:',d.source_file.nunique(),'групп-прокси:',d.review_group.nunique())
display(pd.DataFrame({'dtype':x.dtypes.astype(str),'missing':x.isna().sum(),'zeros':(x==0).sum(),'unique':x.nunique()}))
display(d.groupby('NSP').agg(segments=('NSP','size'),files=('source_file','nunique'),proxy_groups=('review_group','nunique')))
destination=ROOT/'data/processed/ctg_uci/reviewed_v3';destination.mkdir(exist_ok=True)
d.to_parquet(destination/'ctg_primary_grouped.parquet',index=False)
'''),MD('## Results · распределения и закодированная категория'),CODE('''numeric=[c for c in CTG_FEATURES if c!='Tendency']
fig,axes=plt.subplots(5,4,figsize=(15,15))
for ax,col in zip(axes.flat,numeric):
    ax.hist(x[col],bins=35,color='#0072b2');ax.set(title=col,ylabel='Segments')
fig.suptitle('UCI numeric features · all previously inspected observations',y=1.005)
fig.tight_layout();save_show('uci01_feature_distributions.png')
display(x.Tendency.value_counts().sort_index().rename('segments').to_frame())
print('Tendency — категория -1/0/1. DS константа; Width=Max-Min, исключаются моделью как избыточные.')
'''),MD('### Зависимость источников и прежний test'),CODE('''claims=json.loads((REVIEW/'uci_claims.json').read_text());display(pd.Series(claims).to_frame('value'))
support=pd.read_csv(REVIEW/'uci_proxy_group_support.csv')
overlap=pd.read_csv(REVIEW/'uci_adjacent_interval_overlap.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].hist(support.segments,bins=25,color='#0072b2');axes[0].set(xlabel='Segments per prefix group',ylabel='Groups',title='Unequal information per source')
axes[1].bar(['Overlap','No overlap'],[overlap.overlaps.sum(),(~overlap.overlaps).sum()],color=['#0072b2','#999999'])
axes[1].set(ylabel='Adjacent interval pairs',title='Within file, ordered by start b');axes[1].set_ylim(bottom=0)
fig.tight_layout();save_show('uci01_group_dependence.png')
'''),MD('''### Подготовка внутри CV
Скейлеры fit только на текущей fit-части в 02. Пропусков в исходной UCI нет: новый неполный native-вход отклоняется. Готового X_scaled по всей базе для CV нет. Порядок и схема входов проверяются CTGInputGuard; Tendency кодируется как категория. Отдельный fitted logistic preprocessor сохраняется после итогового обучения.
'''),MD(LIMIT_UCI)])


def uci02():
    save('ctg_uci','02_nested_grid_search.ipynb',[MD(intro('ctg_uci',2)),MD('''## Context & Methods
### Key Assumptions
Первичная метрика подбора — log loss с суммарным весом 1 на префикс. One-SE используется как эвристика простоты, не CI зависимых фолдов. Все 6 семейств оценены на одних внешних группах. Отдельная selected_procedure выбирает семью только внутри fit, поэтому её outer score оценивает выбор, а не объявленный задним числом максимум.
'''),CODE(SETUP),MD('## Data · запуск исправленной процедуры'),CODE('''from src.training.ctg_prefix_benchmark import run
metrics=run()
protocol=json.loads((UCI/'protocol.json').read_text())
display(pd.Series({k:v for k,v in protocol.items() if k not in ['input_sha256','cat_grid','logistic_grid']}).to_frame('protocol'))
display(metrics.groupby('family')[['group_macro_f1','group_log_loss','group_recall_suspect','group_recall_path','segment_macro_f1']].agg(['mean','std']).round(4))
display(pd.read_csv(UCI/'outer_class_support.csv'))
print('Группа может содержать несколько классов: group counts разных классов не складываются.')
'''),MD('## Results · метрики трёх повторов'),CODE('''families=list(metrics.family.unique());fig,axes=plt.subplots(1,2,figsize=(13,4))
for ax,col in zip(axes,['group_macro_f1','group_log_loss']):
    for i,family in enumerate(families):
        z=metrics[metrics.family==family];ax.scatter(z[col],np.full(len(z),i),color='#0072b2',s=40)
    ax.set(yticks=range(len(families)),yticklabels=families,xlabel=col,title='Three repeated nested CV estimates')
fig.tight_layout();save_show('uci02_repeated_nested_cv.png')
choices=pd.read_json(UCI/'nested_choices.json')
display(choices[['repetition','outer_fold','family','params','mean_loss','one_se_cutoff','selected_family']].head(18))
print('Полные параметры всех фолдов и кандидатов сохранены в JSON, таблица выше ограничена первыми 18 строками.')
'''),MD('### Конечная сетка и выбор семейства'),CODE('''grid=pd.read_json(UCI/'final_grid_A_multiclass.json')
flat=pd.concat([grid.drop(columns='params'),pd.json_normalize(grid.params)],axis=1)
heat=flat.groupby(['depth','iterations']).mean_loss.min().unstack()
fig,ax=plt.subplots(figsize=(7,4));sns.heatmap(heat,annot=True,fmt='.3f',cmap='Blues_r',ax=ax)
ax.set_title('All-data inner CV · minimum log loss over other parameters')
fig.tight_layout();save_show('uci02_grid_logloss.png')
display(choices[['repetition','outer_fold','selected_family']].drop_duplicates().selected_family.value_counts().rename('outer fits').to_frame())
'''),MD(LIMIT_UCI)])


def uci03():
    save('ctg_uci','03_calibration_ensemble_and_test.ipynb',[MD(intro('ctg_uci',3)),MD('''## Context & Methods
### Key Assumptions
Фиксированная temperature-калибровка fit на OOF текущей внешней fit-части. Выбор гиперпараметров внутри неё уже использовал эти метки; внешний фолд оценивает всю процедуру. Нет выбора калибратора и threshold по прежней маленькой validation. Усреднение в UCI v3 не вводилось: вывод о бесполезности любого ансамбля из старого сравнения не делается.
'''),CODE(SETUP),MD('## Data · границы калибровки и финальный экспорт'),CODE('''from src.training.ctg_prefix_benchmark import run
run()
choices=pd.read_json(UCI/'nested_choices.json')
display(choices[['repetition','outer_fold','family','temperature','fit_groups','hold_groups']].head(18))
export=ROOT/'models/exported/ctg_uci/reviewed_v3'
card=json.loads((export/'model_card.json').read_text())
display(pd.Series({k:v for k,v in card.items() if k!='family_choices'}).to_frame('model card'))
display(pd.DataFrame(card['family_choices']).drop(columns=['fold_losses']))
'''),MD('## Results · OOF calibration и состав сохранённых моделей'),CODE('''from src.training.ctg_review_models import weights
pred=pd.read_parquet(UCI/'nested_predictions.parquet')
g=pred[(pred.repetition==0)&(pred.family=='selected_procedure')].copy()
g['_weight']=weights(g.review_group);g['bin']=pd.qcut(g.p2,8,duplicates='drop');rows=[]
for label,z in g.groupby('bin',observed=True):
    w=z['_weight'].to_numpy();rows.append(dict(p=np.average(z.p2,weights=w),observed=np.average(z.NSP==3,weights=w),segments=len(z),groups=z.review_group.nunique()))
bins=pd.DataFrame(rows);display(bins.round(4))
fig,ax=plt.subplots(figsize=(6,5));ax.plot(bins.p,bins.observed,'o-',color='#0072b2');ax.plot([0,1],[0,1],'--',color='#555555')
ax.set(xlabel='OOF probability of pathologic',ylabel='Observed pathologic fraction',xlim=(0,1),ylim=(0,1),title='Repetition 0 · fixed calibration procedure')
fig.tight_layout();save_show('uci03_oof_calibration.png')
print('Bundles:',len(list(export.glob('*.joblib'))),'native CatBoost:',len(list(export.glob('*.cbm'))))
print('Полный logistic preprocessor сохранён отдельно; после fit на всей UCI независимого test для этого файла нет.')
'''),MD(LIMIT_UCI)])


def uci04():
    save('ctg_uci','04_clustering_without_outcomes.ipynb',[MD(intro('ctg_uci',4)),MD('''## Context & Methods
### Key Assumptions
Здесь воспроизводится чтение прежнего label-free поиска; результаты и модели не перезаписываются. Его группировка по файлам не равна новой группировке префиксов. Слабая ассоциация не доказывает отсутствие любой структуры, но не даёт оснований для клинического назначения кластеров.
'''),CODE(SETUP),MD('## Data · сохранённые таблицы'),CODE('''legacy=ROOT/'research/experiments/ctg_uci'
association=pd.read_csv(legacy/'cluster_association.csv')
summary=pd.read_csv(legacy/'cluster_test_summary.csv')
display(association.round(4));display(summary[['algorithm','cluster','segments','source_files','path_fraction','enrichment','enrichment_low','enrichment_high']].round(3))
'''),MD('## Results · историческое согласие с экспертной меткой'),CODE('''test=association[association.split=='test']
fig,ax=plt.subplots(figsize=(7,4));ax.bar(test.algorithm,test.ari,color='#0072b2')
ax.set(ylabel='Adjusted Rand Index',title='Historical split · exploratory label association',ylim=(-.03,.3));ax.axhline(0,color='#555555')
fig.tight_layout();save_show('uci04_closed_clustering.png')
(UCI/'clustering_branch_status.json').write_text(json.dumps({'status':'closed_for_clinical_risk_use','historical_results_preserved':True,'new_label_guided_search':False},indent=2))
'''),MD('''## Takeaways
FCM/KMeans/GMM не приняты как модели клинического риска. Старые исходные таблицы и модели сохранены; повторять поиск по удобной метке не требуется. Сжатие и устойчивость разобраны в [06](06_clustering_embeddings_grid_search.ipynb).
''')])


def uci05():
    save('ctg_uci','05_results_and_error_analysis.ipynb',[MD(intro('ctg_uci',5)),MD('''## Context & Methods
### Key Assumptions
Bootstrap ресэмплирует префиксы с кратностью, сохраняя исходные веса сегментов внутри выбранной группы. На каждом bootstrap считается средняя метрика трёх повторов. Это условная неопределённость на выполненных fits; не retraining bootstrap и не исправление человеческого post-review выбора постановки.
'''),CODE(SETUP),MD('## Data · интервалы и ошибки'),CODE('''from src.training.ctg_review_assessment import uci_assessment
if not (UCI/'conditional_intervals.csv').exists():uci_assessment()
intervals=pd.read_csv(UCI/'conditional_intervals.csv')
display(intervals[intervals.family=='selected_procedure'][['metric','estimate','ci_low','ci_high']].round(4))
'''),MD('## Results · качество и неопределённость'),CODE('''fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,metric in zip(axes,['macro_f1','recall_path']):
    z=intervals[intervals.metric==metric].reset_index(drop=True)
    ax.errorbar(z.estimate,np.arange(len(z)),xerr=np.array([z.estimate-z.ci_low,z.ci_high-z.estimate]),fmt='o',capsize=3,color='#0072b2')
    ax.set(yticks=range(len(z)),yticklabels=z.family,xlabel=metric,title='95% conditional prefix-bootstrap CI')
fig.tight_layout();save_show('uci05_conditional_intervals.png')
'''),MD('### Подозрительные случаи и альтернативные CLASS-паттерны'),CODE('''cm=pd.read_csv(UCI/'repeat_confusions.csv');z=cm[(cm.family=='selected_procedure')&(cm.repetition==0)]
matrix=z.pivot(index='actual',columns='predicted',values='segments').reindex(index=[0,1,2],columns=[0,1,2])
fig,axes=plt.subplots(1,2,figsize=(13,5))
sns.heatmap(matrix,annot=True,fmt='d',cmap='Blues',cbar=False,ax=axes[0],xticklabels=['normal','suspect','pathologic'],yticklabels=['normal','suspect','pathologic'])
axes[0].set(title='Selected procedure · repetition 0',xlabel='Predicted',ylabel='Expert class')
patterns=pd.read_csv(UCI/'CLASS_pattern_errors.csv');p=patterns[patterns.repetition==0].set_index('CLASS')
values=p[['normal_predictions','suspect_predictions','path_predictions']].div(p.segments,axis=0)
sns.heatmap(values,annot=True,fmt='.1%',cmap='Blues',cbar=False,ax=axes[1]);axes[1].set(title='Alternative CLASS annotations · descriptive')
fig.tight_layout();save_show('uci05_error_patterns.png')
display(patterns.round(4))
details=pd.read_csv(UCI/'per_class_metrics.csv')
display(details.groupby(['family','label'])[['recall','precision','ap','roc_auc']].mean().round(4))
'''),MD(LIMIT_UCI)])


def uci06():
    save('ctg_uci','06_clustering_embeddings_grid_search.ipynb',[MD(intro('ctg_uci',6)),MD('''## Context & Methods
### Key Assumptions
Повторно читаются все прежние кандидаты и seed/subsample-проверки, без изменения выбранной конфигурации. Исторические file-level интервалы и test не превращаются в новые независимые prefix-level доказательства. ARI с метками и ARI устойчивости — разные величины.
'''),CODE(SETUP),MD('## Data · полный сохранённый перебор'),CODE('''folder=ROOT/'research/experiments/ctg_uci/embeddings'
grid=pd.read_csv(folder/'joint_grid.csv');chosen=pd.read_csv(folder/'chosen_configurations.csv')
association=pd.read_csv(folder/'association.csv');stability=pd.read_csv(folder/'stability.csv')
print('Сочетаний:',len(grid));display(chosen[['method','algorithm','dimensions','k','validation_group_silhouette_common']].round(4))
display(association[association.split=='test'][['method','algorithm','ari','nmi']].round(4))
'''),MD('## Results · ассоциация и устойчивость'),CODE('''fig,axes=plt.subplots(1,2,figsize=(13,5))
z=association[association.split=='test'].copy();z['label']=z.method+' / '+z.algorithm
axes[0].barh(z.label,z.ari,color='#0072b2');axes[0].axvline(0,color='#555555');axes[0].set(xlabel='ARI with expert label',title='Historical label association')
s=stability[(stability.method=='AE')&(stability['mode']=='seed')]
for i,algorithm in enumerate(['FCM','GMM','KMeans']):
    a=s[s.algorithm==algorithm];axes[1].scatter(np.full(len(a),i),a.validation_ari_vs_reference,color='#d55e00',s=40)
axes[1].set(xticks=range(3),xticklabels=['FCM','GMM','KMeans'],ylabel='ARI vs reference fit',ylim=(-.1,1),title='AE branch · seed stability')
fig.tight_layout();save_show('uci06_embeddings_closed.png')
display(stability.groupby(['method','algorithm','mode']).validation_ari_vs_reference.agg(['median','min','max']).round(3))
'''),MD('''## Takeaways
AE+GMM выделяется численной ассоциацией, но её seed-устойчивость недостаточна. Сохранить отрицательный/неустойчивый результат полезнее нового перебора до красивого разделения. Ни UMAP, ни AE не включены в классификатор приложения.
''')])


def uci07():
    rows=pd.read_csv(REVIEW/'expert_dispositions.csv')
    cells=[MD(intro('ctg_uci',7)),MD('''## Context & Methods
### Key Assumptions
Вход — приложенное пользователем ревью. Его утверждения считаются проверяемыми гипотезами, не инструкцией переписать цифры под ожидаемый вывод. Исходный текст и все 12 прежних ноутбуков сохранены с хешами. Некоторые клинические предложения приняты только в ограниченной форме.
'''),CODE(SETUP),MD('## Data · источник замечаний и пересчитанные факты'),CODE('''response=pd.read_csv(REVIEW/'expert_dispositions.csv')
assert response.point.tolist()==list(range(1,24))
display(pd.Series(json.loads((REVIEW/'uci_claims.json').read_text())).to_frame('recomputed'))
'''),MD('## Results · критические замечания 1–3')]
    def points(first,last):
        for r in rows[rows.point.between(first,last)].itertuples():
            evidence=r.evidence;name=evidence.split(';')[0].strip()
            candidates=[REVIEW/name,UCI/name,JNU/name]
            match=next((p for p in candidates if p.is_file()),None)
            if match is not None:evidence=f'[{name}](../../{match.relative_to(ROOT/"research")}) · '+evidence
            cells.append(MD(f'### Пункт {r.point}. {r.decision}\n\n**Проверка.** {r.critical_assessment}\n\n**Действие.** {r.action}\n\nПодтверждение: {evidence}\n'))
    points(1,3);cells.append(MD('## UCI · пункты 4–9'));points(4,9)
    cells.append(CODE('''display(pd.read_csv(UCI/'repeat_metrics.csv').groupby('family')[['group_macro_f1','group_recall_suspect','group_recall_path','group_log_loss']].mean().round(4))
'''))
    cells.append(MD('## JNU · пункты 10–16'));points(10,16)
    cells.append(CODE('''display(pd.read_csv(REVIEW/'jnu_train_univariate_auc.csv').round(4))
display(pd.read_csv(REVIEW/'jnu_outcome_record_association.csv').round(5))
'''))
    cells.append(MD('## Методы, перенос и организация · пункты 17–23'));points(17,23)
    cells.append(CODE('''r=json.loads((REVIEW/'validation_receipt.json').read_text())
display(pd.Series({k:r[k] for k in ['passed','review_points','archived_notebooks','historical_model_binaries_preserved','clinical_validation']}).to_frame('validation'))
'''))
    cells.append(MD('''### Источники клинических уточнений
В [FIGO 2015](https://obgyn.onlinelibrary.wiley.com/doi/10.1016/j.ijgo.2015.06.020) уменьшенная и увеличенная вариабельность рассматриваются с длительностью и контекстом; это интранатальные рекомендации. [Philips D/R-2002 application note](https://www.philips.com/c-dam/b2bhc/master/healthcare/room-layout/mri/CTG_analysis_based_on_Dawes_Redman_criteria.pdf) требует сочетания признаков, включая гестационные центили и завершающий участок. Полноценную оценку по нескольким агрегатам не заявляем. Информация о типичных 1–2 неделях до родов — в README JNU, индивидуальная хронология не восстановлена.

## Takeaways
Исправлены воспроизводимые дефекты и ограничены чрезмерные выводы. Открыты: истинные UCI patient IDs, независимые аннотации baseline/events, определения published features, внешний device bridge, срок на момент КТГ и клиническая цель приложения. Конечные проверки не означают «0 остаточных дефектов».
'''))
    save('ctg_uci','07_deep_audit_and_recommendations.ipynb',cells)


def uci08():
    save('ctg_uci','08_hospital_reports_feature_engineering.ipynb',[MD(intro('ctg_uci',8)),MD('''## Context & Methods
### Key Assumptions
Числа примера переписаны с изображения; OCR не выполняется. Строка «8 из 8 в норме» сохраняется как заключение, не становится target или признаком. Лабораторные единицы и происхождение измерений проверяются отдельно. Полный аудит JNU — в [отдельной серии](../jnu_ctg/01_prepare_data.ipynb).
'''),CODE(SETUP),MD('## Data · единый публичный вход'),CODE('''from src.features.ctg_russian_reports import EXAMPLE_TEXT
from src.features.ctg_report_contract import parse_ctg_report,prepare_report,evaluate_report,partial_dr_checklist,PROFILES
parsed=parse_ctg_report(EXAMPLE_TEXT);report=parsed['measurements']
display(pd.DataFrame(parsed['evidence']))
x,audit=prepare_report(report,'report5')
display(x);display(pd.DataFrame({'blocker':audit['blockers']}))
class MustNotBeCalled:
    def predict_proba(self,X):raise AssertionError('Blocked report reached model')
result=evaluate_report(report,MustNotBeCalled(),'report5')
assert result['status']=='abstained' and result['probability_expert_abnormal'] is None
display(pd.Series(result).to_frame('application response'))
'''),MD('## Results · что известно и чего не хватает'),CODE('''display(pd.DataFrame({'profile':list(PROFILES),'fields':[', '.join(v) for v in PROFILES.values()]}))
check=partial_dr_checklist(report);display(pd.DataFrame(check['requirements']))
assert check['criteria_met'] is None
mapping=pd.read_csv(ROOT/'research/experiments/ctg_uci/report_features_v1/hospital_feature_mapping.csv')
display(mapping)
'''),MD('''## Takeaways
Для приложения используется один parser/контракт; UCI-поля не подставляются по сходству названий. Полный D/R или Фишер по неполному бланку не вычисляется. До подтверждения формул целевого устройства и независимой разметки модель не выдаёт медицинскую вероятность.
''')])


def jnu_preserve_preparation(number):
    filename='01_prepare_data.ipynb' if number==1 else '02_report_features.ipynb'
    book=nb.read(REVIEW/'before_revision/jnu_ctg'/filename,4)
    book.cells[0].source=intro('jnu_ctg',number)+'\nПрежние вычисления подготовки сохранены ниже. Fitted reference scalers v1 служат исследовательскому сравнению; новые классификаторы v2 их не используют.\n'
    for c in book.cells:
        if c.cell_type=='markdown':
            c.source=c.source.replace('всех6семейств','всех 6 семейств').replace('5×3nestedCV','5×3 nested CV')
            lines=[]
            for line in c.source.splitlines():
                if line.startswith('Следующий этап — ноутбук 02') or line.startswith('5. Следующий этап описан'):
                    line='Исправленные классификаторы выполнены в 03, проверка — в 04. Дальнейший этап — независимые размеченные бланки и подтверждение измерений; клинический исход остаётся отдельной целью.'
                if line.startswith('5. Скейлеры и нормализатор обучены на train.'):
                    line='5. Скейлеры и нормализатор v1 — reference-подготовка. Новые strict-классификаторы v2 fit собственные преобразования внутри фолдов без imputation/missing indicators; готовые масштабированные матрицы reference для их CV не используются.'
                lines.append(line)
            c.source='\n'.join(lines)
        if c.cell_type=='code' and 'from src.features.ctg_russian_reports import report_profile_input' in c.source:
            start=c.source.index('from src.features.ctg_russian_reports import report_profile_input')
            c.source=c.source[:start]+'''from src.features.ctg_report_contract import prepare_report,evaluate_report
unverified_input,input_audit=prepare_report(parsed['measurements'],'report5')
display(unverified_input)
display(pd.DataFrame({'Отказ до модели':input_audit['blockers']}))
print('Неизвестные определения не превращаются в NaN, который затем интерпретируется моделью.')
'''
    extra=[MD('## Дополнение после экспертного ревью'),CODE(SETUP)]
    if number==1:
        extra.extend([MD('### Сигнатуры UC и подозрение на published-канал'),CODE('''signatures=pd.read_csv(REVIEW/'jnu_raw_uc_max_signatures.csv')
display(signatures.head(12).round(4))
corr=pd.read_csv(REVIEW/'jnu_published_U_channel_correlations.csv')
z=corr[(corr.method=='pearson')&corr.reference.isin(['fhr_bpm_std','uc_device_std'])]
fig,axes=plt.subplots(1,2,figsize=(13,5))
top=signatures.head(8);axes[0].bar(top.raw_uc_max.astype(str),top.records,color='#0072b2');axes[0].set(xlabel='Released UC maximum',ylabel='Recordings',title='Scale signatures, not identified devices',ylim=(0,14000))
sns.heatmap(z.pivot(index='published_feature',columns='reference',values='correlation'),vmin=-1,vmax=1,center=0,cmap='vlag',annot=True,fmt='.2f',ax=axes[1])
axes[1].set_title('Train · suspicious published U_* association')
fig.tight_layout();save_show('jnu01_channel_signatures.png')
display(pd.Series(json.loads((REVIEW/'jnu_label_agreement.json').read_text())).to_frame('within-patient trace pairs'))
'''),MD('''Сырые FHR/UC не переставлены по корреляции; опубликованные U_* остаются карантином. Разные записи одной пациентки могут отражать разные состояния, поэтому каппа не является измерением межэкспертной надёжности одной записи. Типичный интервал до родов указан в README источника; индивидуального срока на КТГ нет.
''')])
    else:
        extra.extend([MD('### Амплитуда: исключение акцелераций на тех же минутах'),CODE('''from src.features.jnu_review_detector_probe import controlled_amplitude_probe
probe=controlled_amplitude_probe();display(probe.head(12).round(3))
fig,ax=plt.subplots(figsize=(7,5));ax.scatter(probe.range_before_same_minutes_bpm,probe.range_after_acceleration_exclusion_bpm,s=25,color='#0072b2',alpha=.6)
limit=np.nanmax(probe[['range_before_same_minutes_bpm','range_after_acceleration_exclusion_bpm']].to_numpy());ax.plot([0,limit],[0,limit],'--',color='#555555')
ax.set(xlabel='Range before exclusion, BPM',ylabel='Range after acceleration exclusion, BPM',title='128 train traces · same eligible minutes, no vendor gold standard')
fig.tight_layout();save_show('jnu02_amplitude_definition_probe.png')
'''),MD('### Одна единица и охват строгих профилей'),CODE('''from src.features.ctg_report_contract import PROFILES
from src.training.jnu_review_experiment import load_data,eligible
cohort=load_data()
display(pd.DataFrame([{'profile':p,'fields':len(cols),'complete':int(eligible(cohort,p).sum()),'source_records':len(cohort),'columns':', '.join(cols)} for p,cols in PROFILES.items()]))
display(pd.read_csv(REVIEW/'jnu_missingness_by_class.csv'))
display(pd.read_csv(REVIEW/'jnu_low_high_variation_support.csv').round(4))
'''),MD('''Minute-range включает акцелерации по операционному определению v1; он не переименовывается в доказанную амплитуду прибора. Baseline-only кандидат также не имеет экспертного эталона и не заменяет основной признак по FIGO-accuracy. Частоты событий сохраняют знаменатель полной записи: поправка на nonlost time не восстанавливает цензурированные события. Длительность 30 min проверяется как область применения, а не бесполезная константа X.
''')])
    # A final updated takeaway follows the retained original preparation sections.
    book.cells.extend(extra+[MD(LIMIT_JNU)])
    nb.validate(book);nb.write(book,ROOT/'research/notebooks/jnu_ctg'/filename)


def jnu03():
    save('jnu_ctg','03_report_classifiers.ipynb',[MD(intro('jnu_ctg',3)),MD('''## Context & Methods
### Key Assumptions
Новая версия использует все уже известные данные JNU для exploratory nested CV. Каждая внешняя fit-часть делится по patient_component: 80% proper-fit /20% calibration; proper-fit отдельно делится на tune/mode-holdout. Сетка видит только tune; single/ensemble оцениваются на mode-holdout; реальный агрегатор fit на proper-fit и sigmoid-калибруется на calibration. Внешние метки ни на один выбор не влияют. Одна единица на величину, без missing indicators/imputation в report-моделях.
'''),CODE(SETUP),MD('## Data · запуск и полный протокол'),CODE('''from src.training.jnu_review_experiment import run
metrics=run()
protocol=json.loads((JNU/'protocol.json').read_text())
display(pd.Series({k:v for k,v in protocol.items() if k not in ['input_sha256','grids']}).to_frame('protocol'))
display(metrics.round(4))
'''),MD('## Results · компактные профили и семейства'),CODE('''fig,ax=plt.subplots(figsize=(10,5))
profiles=list(metrics.profile.unique())
for family,offset in [('linear',-.2),('spline',0),('catboost',.2)]:
    z=metrics[metrics.family==family].set_index('profile').loc[profiles]
    ax.scatter(z.group_macro_f1,np.arange(len(z))+offset,label=family,color=COLORS[family],s=55)
ax.set(yticks=range(len(profiles)),yticklabels=profiles,xlabel='Patient-balanced Macro-F1 on accepted inputs',title='Post-review nested CV · no independent clinical test')
ax.legend();fig.tight_layout();save_show('jnu03_profile_comparison.png')
choices=pd.read_json(JNU/'choices.json')
display(choices.groupby(['profile','family','mode']).size().rename('outer_fits').to_frame())
display(choices[['label','profile','family','mode','inner_loss','selected_family']].head(18))
'''),MD('### Сигнальные ориентиры на одинаковых фолдах'),CODE('''reference=pd.read_csv(JNU/'representation_metrics.csv')
display(reference[['representation','records','groups','group_macro_f1','group_auc','group_ap']].round(4))
fig,ax=plt.subplots(figsize=(10,5));z=reference.sort_values('group_auc')
ax.scatter(z.group_auc,np.arange(len(z)),s=60,color='#0072b2');ax.set(yticks=range(len(z)),yticklabels=z.representation,xlabel='Patient-balanced ROC-AUC',xlim=(.5,1),title='Same HGB grid and same report5-complete cohort')
fig.tight_layout();save_show('jnu03_representation_benchmark.png')
print('raw_minute210 — 210 временных дескрипторов двух сырых каналов, не end-to-end нейросеть. F_* остаются quarantined reference.')
'''),MD('### Экспорт и честное сравнение усреднения'),CODE('''final=pd.read_json(JNU/'final_choices.json')
display(final[['profile','family','params','mode','selected_family','proper_fit_records','calibration_records']])
print('Финальные mode counts:',final['mode'].value_counts().to_dict())
print('Все bundles, native CatBoost и отдельные scalers/bases сохранены в models/exported/jnu_ctg/reviewed_v2.')
print('Выбор modes относится к данным параметрам и критерию, не ко всем возможным ансамблям.')
'''),MD(LIMIT_JNU)])


def jnu04():
    save('jnu_ctg','04_classifier_assessment.ipynb',[MD(intro('jnu_ctg',4)),MD('''## Context & Methods
### Key Assumptions
1000 patient-component bootstrap-выборок условны на выполненных CV fits и принятых входах. Неопределённость всего переобучения и переноса не охвачена. Synthetic probes проверяют поведение и API, но не создают клинически размеченные случаи. Охват JNU и возможность принять реальный бланк — разные величины.
'''),CODE(SETUP),MD('## Data · интервалы и отдельный учёт отказов'),CODE('''from src.training.ctg_review_assessment import jnu_assessment
if not (JNU/'conditional_intervals.csv').exists():jnu_assessment()
intervals=pd.read_csv(JNU/'conditional_intervals.csv');coverage=pd.read_csv(JNU/'coverage_summary.csv')
display(intervals[(intervals.family=='selected_procedure')&intervals.metric.isin(['macro_f1','recall','specificity','ap'])].round(4))
display(coverage.round(5))
'''),MD('## Results · качество, интервалы и поддержка'),CODE('''fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,metric in zip(axes,['macro_f1','recall']):
    z=intervals[(intervals.family=='selected_procedure')&(intervals.metric==metric)].reset_index(drop=True)
    ax.errorbar(z.estimate,np.arange(len(z)),xerr=np.array([z.estimate-z.ci_low,z.ci_high-z.estimate]),fmt='o',capsize=3,color='#0072b2')
    ax.set(yticks=range(len(z)),yticklabels=z.profile,xlabel=metric,title='95% conditional component-bootstrap CI')
fig.tight_layout();save_show('jnu04_intervals.png')
display(pd.read_csv(JNU/'representation_intervals.csv').round(4))
display(pd.read_csv(JNU/'representation_paired_differences.csv').round(4))
'''),MD('''### PR и калибровка на внешних OOF

Для редких probability bins показана поддержка независимых компонент. Бины с <30 компонентами остаются отдельными открытыми маркерами и не соединяются линией: их доля ошибок нестабильна. Это диагностический срез, модель по нему не меняется.
'''),CODE('''from sklearn.metrics import precision_recall_curve
from src.training.ctg_review_models import weights
from src.training.ctg_review_assessment import jnu_calibration_support
if not (JNU/'calibration_bin_support.csv').exists():jnu_calibration_support()
bins=pd.read_csv(JNU/'calibration_bin_support.csv')
display(bins[bins.patient_components<30].round(4))
predictions=pd.read_parquet(JNU/'nested_predictions.parquet')
fig,axes=plt.subplots(1,2,figsize=(12,5))
for family in ['linear','spline','catboost']:
    z=predictions[(predictions.profile=='report5')&(predictions.family==family)&predictions.accepted].copy()
    z['_weight']=weights(z.patient_id);precision,recall,_=precision_recall_curve(z.figo_label,z.p,sample_weight=z['_weight'])
    axes[0].plot(recall,precision,color=COLORS[family],label=family)
    a=bins[(bins.family==family)&(bins.patient_components>=30)]
    axes[1].plot(a.mean_probability,a.observed_fraction,'o-',color=COLORS[family],label=family)
    rare=bins[(bins.family==family)&(bins.patient_components<30)]
    axes[1].scatter(rare.mean_probability,rare.observed_fraction,marker='D',facecolors='none',edgecolors=COLORS[family],s=60)
axes[0].axhline(np.average(z.figo_label,weights=z['_weight']),ls='--',color='#555555')
axes[0].set(xlabel='Recall expert abnormal',ylabel='Precision',xlim=(0,1),ylim=(0,1),title='Report5 · outer OOF predictions')
axes[1].plot([0,1],[0,1],'--',color='#555555');axes[1].set(xlabel='Mean estimated probability',ylabel='Observed expert abnormal',xlim=(0,1),ylim=(0,1),title='Open diamonds: fewer than 30 patient components')
axes[0].legend();fig.tight_layout();save_show('jnu04_pr_calibration.png')
'''),MD('### Матрица с отказом и качество трудных записей'),CODE('''conf=pd.read_csv(JNU/'confusions_with_abstention.csv')
q=pd.read_csv(JNU/'quality_subgroups.csv')
fig,axes=plt.subplots(1,2,figsize=(12,5))
z=conf[(conf.profile=='report5')&(conf.family=='selected_procedure')].pivot(index='actual',columns='predicted',values='records').reindex(index=[0,1],columns=[0,1,2])
sns.heatmap(z,annot=True,fmt='d',cmap='Blues',cbar=False,ax=axes[0],xticklabels=['normal','abnormal','abstain'],yticklabels=['normal','abnormal'])
axes[0].set(title='Report5 · all source rows',xlabel='Research decision',ylabel='Expert class')
for flag,marker in [(True,'o'),(False,'s')]:
    z=q[q.quality_flag==flag];axes[1].plot(z.profile,z.group_specificity,marker+'-',label=f'quality flag={flag}')
axes[1].set(ylabel='Specificity on accepted inputs',ylim=(0,1),title='Quality strata · diagnostic, no filtering');axes[1].legend()
fig.tight_layout();save_show('jnu04_abstention_and_quality.png')
display(q[['profile','quality_flag','source_records','accepted_records','group_macro_f1','group_recall','group_specificity']].round(4))
display(pd.read_csv(JNU/'input_subgroup_metrics.csv').round(4))
errors=pd.read_csv(JNU/'selected_procedure_error_cases.csv')
display(errors[(errors.profile=='report5')&(errors.error_type=='false_negative')].sort_values('wrong_class_confidence',ascending=False).head(10)[['record_id','figo_label','p','baseline_bpm','stv_ms','amplitude_bpm','acc15_per_10min','dec15_per_10min']].round(4))
'''),MD('### Synthetic probes и единый клинический отказ'),CODE('''probes=pd.read_json(JNU/'synthetic_pattern_probes.json')
display(probes[probes.profile=='report5'][['family','pattern','status','probability_expert_abnormal','clinical_risk']])
display(pd.read_csv(JNU/'synthetic_joint_support.csv'))
from src.features.ctg_report_contract import research_patterns,evaluate_report,partial_dr_checklist
export=ROOT/'models/exported/jnu_ctg/reviewed_v2'
bundle=joblib.load(export/'report5_catboost.joblib')
for name,report in research_patterns().items():
    result=bundle.assess_report(report)
    assert result['clinical_risk'] is None and result['probability_expert_abnormal'] is None
display(pd.DataFrame(partial_dr_checklist(research_patterns()['quiet'])['requirements']))
print('Порядок вероятностей этих искусственных сочетаний — свойство expert-label модели, не валидированная шкала благополучия.')
'''),MD('''### Цель, исход и гипотетическая распространённость

Patient-level any_abnormal использует все опубликованные записи пациентки; без хронологии это описательная связь, а не прогноз на дату одной КТГ. Изменение метки между разными записями не измеряет согласие экспертов на одном сигнале. Сценарий 15% ниже не является измеренной клинической распространённостью.
'''),CODE('''display(pd.Series(json.loads((REVIEW/'jnu_label_agreement.json').read_text())).to_frame('different trace pairs'))
display(pd.read_csv(REVIEW/'jnu_outcome_record_association.csv').round(5))
display(pd.read_csv(REVIEW/'jnu_outcome_patient_association.csv').round(5))
shift=pd.read_csv(JNU/'hypothetical_label_shift_ppv.csv');z=shift[(shift.profile=='report5')&(shift.family=='selected_procedure')]
fig,ax=plt.subplots(figsize=(7,4));ax.plot(z.assumed_FIGO_prevalence,z.ppv_under_constant_sensitivity_specificity,'o-',color='#0072b2')
ax.set(xlabel='Hypothetical FIGO-abnormal prevalence',ylabel='PPV under fixed sensitivity/specificity',ylim=(0,1),title='Label-shift illustration, not clinical prevalence')
fig.tight_layout();save_show('jnu04_hypothetical_ppv.png')
'''),MD(LIMIT_JNU)])


def build():
    for fn in [uci01,uci02,uci03,uci04,uci05,uci06,uci07,uci08]:fn()
    jnu_preserve_preparation(1);jnu_preserve_preparation(2);jnu03();jnu04()


def build_selected(series,numbers):
    if series=='ctg_uci':
        dispatch={1:uci01,2:uci02,3:uci03,4:uci04,5:uci05,6:uci06,7:uci07,8:uci08}
        for number in numbers:dispatch[number]()
    else:
        for number in numbers:
            if number<=2:jnu_preserve_preparation(number)
            elif number==3:jnu03()
            elif number==4:jnu04()


def summaries():
    for series in ['ctg_uci','jnu_ctg']:
        for p in (ROOT/'research/notebooks'/series).glob('*.ipynb'):
            number=int(p.name[:2]);b=nb.read(p,4);b.cells[0].source=intro(series,number)
            if series=='jnu_ctg' and number<=2:b.cells[0].source+='\nГлубокая подготовка v1 сохранена ниже; reference scalers не используются новыми строгими классификаторами.\n'
            nb.validate(b);nb.write(b,p)


if __name__=='__main__':
    import sys
    if '--summaries-only' in sys.argv:summaries()
    else:build()
