"""Generate the reproducible CTG training notebook series (02–05)."""
from pathlib import Path
import sys
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'research/notebooks/ctg_uci'
SETUP = """from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.training.ctg_experiments import OUT, MODELS, load_data, FAMILIES, CTG_FEATURES
sns.set_theme(style='whitegrid', palette=['#245b87', '#c47c2d', '#80659e'])
plt.rcParams.update({'figure.dpi':110, 'axes.spines.top':False, 'axes.spines.right':False})
data, fingerprint = load_data()
"""


def save(name, cells):
    book = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(text.strip()) if kind == 'md'
                                    else nb.v4.new_code_cell(text.strip()) for kind, text in cells])
    book.metadata = {'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
                     'language_info':{'name':'python','version':'3.11'}}
    nb.validate(book)
    nb.write(book, DEST / name)


def build_nested():
    save('02_nested_grid_search.ipynb', [
        ('md', '''# 02 · Обучение: общий nested CV и Grid Search
## Кратко
Сравниваются пять CatBoost-процедур и регуляризованная логистика. Все используют одинаковые 5 внешних и 3 внутренних фолда **только исходного train**. Внешняя оценка включает подбор параметров, калибровку и пороги. Результат поиска и выбор семейства показаны после исполнения; test в этом ноутбуке не используется.'''),
        ('md', '''## Контекст и методы
Полный [план эксперимента](EXPERIMENT_PLAN.md) фиксирует сетки, правила выбора, калибровки и проверки ансамбля. A — MultiClass; B — SqrtBalanced и Balanced; C — три OvR; D — normal/abnormal и условный suspect/pathologic. Калибровка C — отдельный sigmoid каждого выхода с последующей нормировкой; D — два условных sigmoid. Для всех остальных — три sigmoid и нормировка. Выбор C допускает только calibrated-режимы.

### Ключевые допущения
`source_file` удерживает вместе сегменты записи; идентификатора женщины нет. `CLASS`, `NSP` и метаданные исключены из X. `DS` и `Width` исключены по контракту 01. CatBoost работает с исходными измерениями; логистика заново обучает clone преобразователя в каждом fit. Веса классов не умножаются дополнительно на sample weights из 01. Внешние фолды не используются для early stopping, калибровки и порога. Внутренняя OOF-разметка используется и для поиска, и для postprocessing; внешняя часть проверяет всю эту процедуру.'''),
        ('code', SETUP),
        ('md', '## Данные и сетка'),
        ('code', '''from src.training.ctg_experiments import CAT_GRID, LOG_GRID, run_nested
display(pd.crosstab(data.split, data.NSP))
print('CatBoost: 24 комбинации × 3 внутренних × 5 внешних на процедуру; у OvR 3 головы, у иерархии 2.')
display(pd.DataFrame(CAT_GRID))
print('Logistic:', len(LOG_GRID), 'комбинаций. Cache привязан к исходным данным и исходному коду.')'''),
        ('md', '''## Вложенный поиск
Полная сетка выполняется при первом запуске. Завершённые семейства/фолды сохраняются отдельными checkpoint; повторный запуск проверяет fingerprint. В каждом внутреннем фолде модель, а для логистики и scaler, учатся заново. Порог патологического класса выбирается по macro-F1 только внутри внешнего train.'''),
        ('code', 'pooled, folds = run_nested()'),
        ('md', '## Результаты · общие фолды'),
        ('code', '''layout = pd.read_csv(OUT / 'outer_folds.csv')
audit = layout.merge(data[['source_excel_row','source_file','NSP']], on='source_excel_row', validate='one_to_one')
display(pd.crosstab(audit.outer_fold, audit.NSP))
print('Уникальных исходных файлов:', audit.source_file.nunique())
assert audit.groupby('source_file').outer_fold.nunique().max() == 1'''),
        ('md', '### Все процедуры и способы получения решения'),
        ('code', '''columns = ['family','mode','macro_f1','balanced_accuracy','recall_path','precision_path','ap_path','log_loss','brier']
display(pooled[columns].sort_values('macro_f1', ascending=False).round(4))
eligible = pooled[~((pooled.family == 'C_ovr') & (pooled['mode'] == 'raw'))]
best = eligible.sort_values(['macro_f1','log_loss'], ascending=[False,True]).groupby('family', sort=False).head(1)
display(best[columns].round(4))'''),
        ('md', '### Разброс по внешним фолдам'),
        ('code', '''selected = folds.merge(best[['family','mode']], on=['family','mode'])
fig, axes = plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
for ax,metric in zip(axes,['macro_f1','recall_path']):
    sns.stripplot(data=selected,x=metric,y='family',color='#245b87',jitter=.10,ax=ax)
    ax.set(title=f'{metric} · 5 внешних фолдов',xlim=(0,1),ylabel='')
plt.show()
print('Фолды имеют пересекающиеся train, поэтому точки не являются 5 независимыми испытаниями.')'''),
        ('md', '### Найденные параметры и выбранное семейство'),
        ('code', '''parameters = pd.DataFrame(json.loads((OUT / 'nested_parameters.json').read_text()))
display(parameters)
selection = json.loads((OUT / 'nested_selection.json').read_text())
print('Зафиксировано семейство:', selection['family'], '· pooled macro-F1:', round(selection['pooled_macro_f1'],4))'''),
        ('md', '''## Выводы
Семейство выбрано по внешним OOF-предсказаниям исходного train. Сетка обеспечивает сравнение 24 заданных комбинаций, но не доказывает глобальный оптимум. Выбор среди процедур делает результат победителя несколько оптимистичным; 03 проверяет его на test после калибровки и проверки fold averaging. Большой macro-F1 не заменяет анализа патологических ошибок и вероятностей.'''),
    ])


def build_final():
    save('03_calibration_ensemble_and_test.ipynb', [
        ('md', '''# 03 · Калибровка, усреднение фолдов и независимый test
## Кратко
Для всех семейств повторяется Grid Search на исходном train. Семейство-победитель уже выбрано в 02. Только для него проверяется усреднение 5 моделей. Калибровка, пороги и выбор single/ensemble используют validation; test-результаты рассчитываются после сохранения решения.'''),
        ('md', '''## Контекст и методы
Каждый финальный CatBoost обучается на train, не на validation. Для проверки postprocessing на validation используются общие 3 групповых cross-fitting-фолда: calibrator и threshold fit на двух частях, оценка на третьей. Калибратор ансамбля обучается на усреднённых выходах ансамбля, отдельно от single. Ensemble выбирается только при приросте macro-F1 не менее 0.01 на validation cross-fit; это заранее заданное практическое правило.

### Ключевые допущения
В validation лишь 29 патологических сегментов; разница небольших метрик шумна. После выбора калибратор и порог fit на всей validation, деревья не переобучаются на ней. Иначе уже оценённая калибровка относилась бы к другой модели. `locked_decision.json` сохраняется до первого test predict. Интервалы на test получаются 1000-повторным bootstrap исходных файлов и условны на уже обученной модели.'''),
        ('code', SETUP),
        ('md', '## Данные и выбранное в 02 семейство'),
        ('code', '''selection = json.loads((OUT / 'nested_selection.json').read_text())
print('Семейство по nested CV:', selection['family'])
display(data.groupby('split').agg(segments=('NSP','size'), recordings=('source_file','nunique')))
print('Test не участвует ни в одном fit; это аудитируемый локальный benchmark, не новая клиническая когорта.')'''),
        ('md', '## Финальный поиск, validation cross-fitting и заблокированный test'),
        ('code', '''from src.training.ctg_experiments import run_final
test_results, validation_results, intervals = run_final()'''),
        ('md', '### Что показала validation'),
        ('code', '''display(validation_results[['family','architecture','mode','macro_f1','recall_path','ap_path','log_loss','brier']].round(4))
locked = json.loads((OUT / 'locked_decision.json').read_text())
print('Выбран:', locked['family'], '/', locked['architecture'])
print('Критерий ensemble: прирост macro-F1 ≥',locked['ensemble_minimum_gain'])'''),
        ('md', '### Финальная оценка test · чемпион выбран заранее'),
        ('code', '''display(test_results[['family','architecture','mode','champion','accuracy','macro_f1','balanced_accuracy','recall_path','precision_path','ap_path','log_loss','brier','path_as_normal']].round(4))
champion = test_results[test_results.champion].iloc[0]
shown = intervals.loc[['macro_f1','recall_path','precision_path','ap_path','log_loss','brier']].copy()
shown.insert(0,'estimate',[champion[name] for name in shown.index])
display(shown.round(4))
print('Случаев pathologic в test:', (data.loc[data.split=='test','NSP']==3).sum())'''),
        ('md', '### Экспорт и проверка сохранённого артефакта'),
        ('code', '''import joblib
bundle = joblib.load(MODELS / 'champion.joblib')
example = data.loc[data.split=='test',list(CTG_FEATURES)].head(3)
probability = bundle.predict_proba(example)
assert np.allclose(probability.sum(axis=1),1)
display(pd.DataFrame(probability,columns=['P(normal)','P(suspect)','P(pathologic)']))
print('Прогнозы NSP:',bundle.predict(example).tolist())
print('Модель:',MODELS/'champion.joblib')
print('Происхождение, версии и правила:',MODELS/'model_card.json')'''),
        ('md', '''## Выводы
Таблица test служит финальной оценкой зафиксированных процедур; она не меняет выбор чемпиона. AP — Average Precision патологического класса; Brier — средняя сумма квадратов ошибки трёх вероятностей (диапазон 0–2). Accuracy следует читать рядом с recall/AP патологии и normal-baseline. Усреднение фолдов принимается только при подтверждённом на validation выигрыше; наличие пяти моделей само по себе не доказывает улучшения. Для эксплуатационного применения требуется проверка на другом источнике и данных с реальными исходами.'''),
    ])


def build_clustering():
    save('04_clustering_without_outcomes.ipynb', [
        ('md', '''# 04 · Fuzzy C-Means, KMeans и GMM без меток исхода
## Кратко
Алгоритмы получают только признаки КТГ. Число кластеров и параметры выбираются по разделимости validation без NSP/CLASS. После фиксации решения проверяется связь кластеров с экспертными классами на test. Кластеры не объявляются клиническими группами риска.'''),
        ('md', '''## Контекст и методы
Нелинейный преобразователь заново fit на train; индикаторы наличия редких событий удалены из расстояния, чтобы не считать их повторно. Числовые признаки стандартизованы; one-hot `Tendency` имеет единичное расстояние при смене категории. PCA сохраняет 90% обучающей дисперсии. Сетки: k=2..6; FCM m=1.5/2/2.5, 5 начальных состояний; KMeans n_init=20; GMM diag/full, n_init=5. Разделимость выбирается по silhouette validation, а не по близости к меткам.

### Ключевые допущения
Расстояния и PCA — модельные предположения; геометрический кластер может отражать тип сигнала, а не экспертный класс. В FCM проверяются сходимость, FPC, энтропия и устойчивость инициализаций; центры могут сближаться в многомерных данных. ARI/NMI и состав кластеров — описательные внешние проверки. Bootstrap по файлам учитывает зависимость сегментов; интервалы малых кластеров особенно широки.'''),
        ('code', SETUP),
        ('md', '## Обучение и выбор без меток'),
        ('code', '''from src.training.ctg_clustering import run_clustering
cluster_grid, association, cluster_summary = run_clustering()
display(cluster_grid.sort_values(['algorithm','validation_silhouette'],ascending=[True,False]).round(4))'''),
        ('md', '''## Результаты · связь с экспертной разметкой после фиксации
Интервалы для suspect и abnormal=NSP>1 добавлены после просмотра таблицы состава как исследовательское уточнение. Число кластеров, параметры и обучение остались зафиксированными; эти интервалы не являются отдельным подтверждающим исследованием.'''),
        ('code', '''display(association.round(4))
display(cluster_summary[['algorithm','cluster','segments','source_files','path_fraction','enrichment','enrichment_low','enrichment_high']].round(4))
display(cluster_summary[['algorithm','cluster','suspect_fraction','suspect_enrichment','suspect_enrichment_low','suspect_enrichment_high','abnormal_fraction','abnormal_enrichment','abnormal_enrichment_low','abnormal_enrichment_high']].round(4))
print('Интервалы обогащения получены bootstrap по исходным файлам. Деление на малую исходную частоту повышает шум.')'''),
        ('md', '### Состав кластеров на test'),
        ('code', '''assignment = pd.read_parquet(OUT/'cluster_assignments.parquet')
fig,axes=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
for ax,algorithm in zip(axes,['FCM','KMeans','GMM']):
    part=assignment[(assignment.algorithm==algorithm)&(assignment.split=='test')]
    composition=pd.crosstab(part.cluster,part.NSP,normalize='index').reindex(columns=[1,2,3],fill_value=0)
    sns.heatmap(composition,annot=True,fmt='.1%',cmap='Blues',vmin=0,vmax=1,ax=ax,cbar=False)
    ax.set(title=f'{algorithm} · доли экспертных классов',xlabel='NSP',ylabel='Кластер')
plt.show()'''),
        ('md', '### Геометрия и неоднозначность FCM'),
        ('code', '''coordinates=pd.read_parquet(OUT/'cluster_projection.parquet')
fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for ax,column,title in [(axes[0],'cluster','FCM: фиксированные кластеры'),(axes[1],'NSP','Экспертная разметка после fit')]:
    sns.scatterplot(data=coordinates,x='PC1',y='PC2',hue=column,palette='deep',s=25,alpha=.7,ax=ax)
    ax.set_title(title+' · test')
plt.show()
fcm=assignment[(assignment.algorithm=='FCM')&(assignment.split=='test')]
fig,ax=plt.subplots(figsize=(8,3))
ax.hist(fcm.membership_entropy,bins=20,color='#245b87')
ax.set(title='FCM · неопределённость принадлежности на test',xlabel='Нормированная энтропия (0=однозначно, 1=равномерно)',ylabel='Сегменты')
plt.show()'''),
        ('md', '''## Выводы
Сначала оцените silhouette и устойчивость, затем ARI/NMI и обогащение. Даже высокая доля pathologic в одном кластере не делает его самостоятельным диагнозом; связь может не переноситься на другой источник. FCM-membership описывает близость к центрам и не является вероятностью патологии. Сравнение трёх алгоритмов и нескольких кластеров исследовательское, без подтверждающих p-value и без выбора конфигурации по test-меткам.'''),
    ])


def build_analysis():
    save('05_results_and_error_analysis.ipynb', [
        ('md', '''# 05 · Итоги, ошибки и неопределённость
## Кратко
Сводка зафиксированных процедур: nested CV train → validation для калибровки и averaging → test. Чемпион сохранён до получения test-метрик. Здесь только анализ результатов, без нового поиска или изменения порогов.'''),
        ('code', SETUP),
        ('md', '## Контекст и ограничения'),
        ('code', '''test=pd.read_csv(OUT/'test_metrics.csv')
pooled=pd.read_csv(OUT/'nested_pooled_metrics.csv')
validation=pd.read_csv(OUT/'validation_crossfit_metrics.csv')
predictions=pd.read_parquet(OUT/'test_predictions.parquet')
locked=json.loads((OUT/'locked_decision.json').read_text())
print('Чемпион:',locked['family'],'/',locked['architecture'])
print('35 патологических сегментов test; интервалы учитывают группировку по файлам, не всю неопределённость переобучения.')
display(test[['family','architecture','champion','macro_f1','recall_path','precision_path','ap_path','log_loss','brier','path_as_normal']].round(4))'''),
        ('md', '## Матрицы ошибок · финальные процедуры'),
        ('code', '''from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
for ax,family in zip(axes.flat,FAMILIES):
    architecture=locked['architecture'] if family==locked['family'] else 'single'
    part=predictions[(predictions.family==family)&(predictions.architecture==architecture)]
    cm=confusion_matrix(part.NSP,part.pred,labels=[1,2,3])
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False,vmin=0,vmax=332,ax=ax,xticklabels=['N','S','P'],yticklabels=['N','S','P'])
    ax.set(title=family+(' · champion' if family==locked['family'] else ''),xlabel='Предсказано',ylabel='Экспертный класс')
plt.show()'''),
        ('md', '### Два определения площади под PR-кривой'),
        ('code', '''areas=[]
for (family,architecture),part in predictions.groupby(['family','architecture']):
    precision,recall,_=precision_recall_curve(part.NSP==3,part.p2)
    areas.append({'family':family,'architecture':architecture,'trapezoidal_pr_auc_path':auc(recall,precision)})
areas=pd.DataFrame(areas).merge(test[['family','architecture','ap_path']],on=['family','architecture'])
areas.to_csv(OUT/'test_precision_recall_areas.csv',index=False)
display(areas.round(4))
print('Average Precision и трапецеидальная PR-AUC — разные интегральные оценки; в отборе использовался macro-F1.')'''),
        ('md', '## Распознавание pathologic · PR и ROC'),
        ('code', '''fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
for index,family in enumerate(FAMILIES):
    architecture=locked['architecture'] if family==locked['family'] else 'single'
    part=predictions[(predictions.family==family)&(predictions.architecture==architecture)]
    precision,recall,_=precision_recall_curve(part.NSP==3,part.p2)
    fpr,tpr,_=roc_curve(part.NSP==3,part.p2)
    style='-' if index<3 else '--'
    width=2.2 if family==locked['family'] else 1.5
    axes[0].plot(recall,precision,label=family,linestyle=style,linewidth=width)
    axes[1].plot(fpr,tpr,label=family,linestyle=style,linewidth=width)
prevalence=(part.NSP==3).mean()
axes[0].axhline(prevalence,color='#555555',linestyle='--',label=f'Доля P: {prevalence:.3f}')
axes[1].plot([0,1],[0,1],color='#555555',linestyle='--')
axes[0].set(title='Pathologic · Precision–Recall на test',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02))
axes[1].set(title='Pathologic · ROC на test',xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1.02))
axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
plt.show()'''),
        ('md', '## Калибровка патологического класса'),
        ('code', '''champ=predictions[(predictions.family==locked['family'])&(predictions.architecture==locked['architecture'])].copy()
champ['pathologic']=(champ.NSP==3).astype(int)
champ['probability_bin']=pd.cut(champ.p2,bins=np.linspace(0,1,6),include_lowest=True)
reliability=champ.groupby('probability_bin',observed=False).agg(segments=('NSP','size'),mean_predicted=('p2','mean'),observed_fraction=('pathologic','mean'))
display(reliability.round(4))
nonempty=reliability[reliability.segments>0]
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
axes[0].plot(nonempty.mean_predicted,nonempty.observed_fraction,'o-',color='#245b87')
for row in nonempty.itertuples():
    axes[0].annotate(f'n={row.segments}',(row.mean_predicted,row.observed_fraction),xytext=(4,6),textcoords='offset points',fontsize=8)
axes[0].plot([0,1],[0,1],'--',color='#555555')
axes[0].set(title='Чемпион · calibration по интервалам 0.2',xlabel='Средняя P(pathologic)',ylabel='Наблюдаемая доля',xlim=(-.02,1.02),ylim=(-.02,1.05))
axes[1].hist([champ.loc[champ.NSP==label,'p2'] for label in [1,2,3]],bins=np.linspace(0,1,21),stacked=True,label=['N','S','P'],color=['#245b87','#c47c2d','#80659e'])
axes[1].legend(title='Экспертный класс')
axes[1].set(title='Распределение вероятностей · test',xlabel='P(pathologic)',ylabel='Сегменты')
plt.show()
print('В средних probability bins мало сегментов; это описательная кривая, точки не независимы по пациенткам.')'''),
        ('md', '## Групповые интервалы и тяжёлые ошибки'),
        ('code', '''ci=pd.read_csv(OUT/'champion_test_intervals.csv',index_col=0)
display(ci.loc[['macro_f1','recall_path','precision_path','ap_path','log_loss','brier']].round(4))
errors=champ[champ.NSP!=champ.pred].copy()
errors.to_csv(OUT/'champion_errors.csv',index=False)
display(errors.groupby(['NSP','pred']).agg(segments=('NSP','size'),source_files=('source_file','nunique')))
by_file=champ.assign(error=champ.NSP!=champ.pred).groupby('source_file').agg(segments=('NSP','size'),errors=('error','sum'))
display(by_file.sort_values('errors',ascending=False).head(12))
print('Pathologic → normal:',int(((champ.NSP==3)&(champ.pred==1)).sum()),'из',int((champ.NSP==3).sum()))'''),
        ('md', '''## Дополнительная точка чувствительности
Порог для recall ≥0.90 был выбран на validation и записан до test. Здесь показан перенос этого фиксированного порога. Нужный уровень recall на новых данных не гарантирован, особенно при 10 исходных файлах с pathologic в test.'''),
        ('code', '''import joblib
from src.training.ctg_experiments import decisions, metrics
bundle=joblib.load(MODELS/'champion.joblib')
test_frame=data[data.split=='test']
heads=np.mean([member.predict_heads(test_frame[list(CTG_FEATURES)]) for member in bundle.models],axis=0)
calibrated=bundle.calibrator.predict_proba(heads)
policy=locked['policies'][locked['family']+'/'+locked['architecture']]
cutoff=policy['sensitivity_threshold']
sens_pred=decisions(calibrated,cutoff)
operating_point=metrics(test_frame.NSP.to_numpy()-1,calibrated,sens_pred)
display(pd.DataFrame([{'validation_threshold':cutoff,**operating_point}]).round(4))
pd.DataFrame([{'validation_threshold':cutoff,**operating_point}]).to_csv(OUT/'sensitivity_operating_point.csv',index=False)'''),
        ('md', '''## Внутренняя важность признаков
Для CatBoost показана PredictionValuesChange по каждой голове, усреднённая только между фолдами одного и того же типа головы. Это описание обученной модели, не причинный эффект и не оценка улучшения от удаления признака. Для логистики приводятся коэффициенты в преобразованном пространстве.'''),
        ('code', '''import joblib
bundle=joblib.load(MODELS/'champion.joblib')
importance=[]
if bundle.family!='logistic':
    for fold,member in enumerate(bundle.models):
        for head,model in enumerate(member.models_):
            importance.extend([{'fold':fold,'head':head,'feature':feature,'importance':float(value)}
                               for feature,value in zip(model.feature_names_,model.get_feature_importance())])
    importance=pd.DataFrame(importance)
    importance.to_csv(OUT/'champion_feature_importance.csv',index=False)
    heads=sorted(importance['head'].unique())
    fig,axes=plt.subplots(1,len(heads),figsize=(max(8,5*len(heads)),5),squeeze=False,constrained_layout=True)
    for ax,head in zip(axes.flat,heads):
        ranked=importance[importance['head']==head].groupby('feature').importance.mean().nlargest(10).sort_values()
        ax.barh(ranked.index,ranked.values,color='#245b87')
        title=({'D_hierarchy':['P(abnormal)','P(pathologic | abnormal)'],'C_ovr':['OvR normal','OvR suspect','OvR pathologic']}.get(bundle.family,['MultiClass'])[head])
        ax.set(title=title,xlabel='PredictionValuesChange')
    plt.show()
else:
    model=bundle.models[0].models_[0]
    coef=pd.DataFrame(model.named_steps['model'].coef_.T,index=model.named_steps['features'].get_feature_names_out(),columns=['N','S','P'])
    display(coef.loc[coef.abs().max(axis=1).nlargest(12).index].round(3))'''),
        ('md', '## Нужен ли ансамбль фолдов'),
        ('code', '''comparison=validation[validation.family==locked['family']]
display(comparison[['architecture','mode','macro_f1','recall_path','log_loss','brier']].round(4))
print('По заранее заданному правилу выбран:',locked['architecture'])
print('Проверенные интервалы относятся к фиксированной выбранной модели; результаты одного разделения не доказывают универсальное превосходство.')'''),
        ('md', '''## Выводы
Ключевое сравнение — macro-F1 вместе с recall/precision/AP патологии, а для вероятностей — log loss и Brier. Значение около 78% accuracy возможно без единого найденного патологического сегмента. Небольшое различие сложных процедур требует осторожности: патологических записей мало, сегменты зависимы, калибровка и подбор порога также имеют шум. Для следующего исследования нужны новые источники, идентификаторы пациенток и реальные исходы; текущий test не следует превращать в следующую validation.'''),
    ])


def refresh_summaries():
    """Write observed findings after execution without clearing cell outputs."""
    import json
    import pandas as pd
    out=ROOT/'research/experiments/ctg_uci'
    nested=json.loads((out/'nested_selection.json').read_text())
    locked=json.loads((out/'locked_decision.json').read_text())
    test=pd.read_csv(out/'test_metrics.csv')
    winner=test[test.champion].iloc[0]
    intervals=pd.read_csv(out/'champion_test_intervals.csv',index_col=0)
    validation=pd.read_csv(out/'validation_crossfit_metrics.csv')
    comparison=validation[validation.family==locked['family']]
    if locked['family']=='C_ovr':
        comparison=comparison[comparison['mode']!='raw']
    single=comparison[comparison.architecture=='single'].macro_f1.max()
    ensemble=comparison[comparison.architecture=='fold_ensemble'].macro_f1.max()
    ordinary=test[test.family=='A_multiclass'].iloc[0]
    cluster=pd.read_csv(out/'cluster_association.csv')
    fcm=cluster[(cluster.algorithm=='FCM')&(cluster.split=='test')].iloc[0]
    cluster_summary=pd.read_csv(out/'cluster_test_summary.csv')
    fcm_rates=cluster_summary[cluster_summary.algorithm=='FCM'].path_fraction.tolist()
    rates_text=' / '.join(f'{value:.1%}' for value in fcm_rates)
    cluster_counts=', '.join(f'{row.algorithm}: {int(row.k)}' for row in cluster[cluster.split=='test'].itertuples())
    all_intervals_include_one=((cluster_summary.enrichment_low<=1)&(cluster_summary.enrichment_high>=1)).all()
    association_note=('Bootstrap-интервалы обогащения **патологией** всех кластеров включают 1; отдельные патологические группы не подтверждены.'
                      if all_intervals_include_one else 'Часть кластеров показывает исследовательское обогащение; необходима независимая проверка.')
    prediction=pd.read_parquet(out/'test_predictions.parquet')
    chosen=prediction[(prediction.family==locked['family'])&(prediction.architecture==locked['architecture'])]
    path_count=int((chosen.NSP==3).sum())
    found=int(((chosen.NSP==3)&(chosen.pred==3)).sum())
    missed_normal=int(((chosen.NSP==3)&(chosen.pred==1)).sum())
    missed_suspect=int(((chosen.NSP==3)&(chosen.pred==2)).sum())
    sensitivity=pd.read_csv(out/'sensitivity_operating_point.csv').iloc[0]
    comparative=('На test обычный MultiClass имеет более высокий macro-F1' if ordinary.macro_f1>winner.macro_f1
                 else 'На test обычный MultiClass имеет macro-F1')
    summaries={
        '02_nested_grid_search.ipynb':f'''# 02 · Обучение: общий nested CV и Grid Search
## Кратко
Выполнены общие 5×3 фолда по исходным файлам только на train и полные сетки 24 комбинаций CatBoost. Лидирует **{nested['family']}**: pooled macro-F1 **{nested['pooled_macro_f1']:.4f}**. Это сравнение процедуры поиска и postprocessing, без участия validation/test. Все параметры, результаты сетки и внешние OOF-предсказания сохранены. Семейство зафиксировано до финального test; небольшое различие CV не является доказательством универсального превосходства.''',
        '03_calibration_ensemble_and_test.ipynb':f'''# 03 · Калибровка, усреднение фолдов и независимый test
## Кратко
По train-CV выбрана **{locked['family']}**, по validation — **{locked['architecture']} / {winner['mode']}**. Её test macro-F1 **{winner.macro_f1:.4f}**, recall pathologic **{winner.recall_path:.1%}**, AP **{winner.ap_path:.4f}**. Групповой 95% интервал macro-F1: **{intervals.loc['macro_f1','lower']:.3f}–{intervals.loc['macro_f1','upper']:.3f}**. В test только 35 патологических сегментов из 10 файлов.

Validation-разница ensemble − single по macro-F1: {ensemble-single:.4f}; по правилу ≥0.01 выбран {locked['architecture']}. {comparative} ({ordinary.macro_f1:.4f}); чемпион по этому результату не переименовывается. Это наблюдение показывает неопределённость ранжирования, а не повод снова подбирать модель по test.''',
        '04_clustering_without_outcomes.ipynb':f'''# 04 · Fuzzy C-Means, KMeans и GMM без меток исхода
## Кратко
По validation silhouette без меток выбрано кластеров: **{cluster_counts}**. FCM на test: ARI **{fcm.ari:.4f}**, NMI **{fcm.nmi:.4f}**; доли pathologic **{rates_text}** при общем уровне {(chosen.NSP==3).mean():.1%}. {association_note}

При этом доля **suspect** у FCM различается: {cluster_summary.loc[cluster_summary.algorithm=='FCM','suspect_fraction'].min():.1%}–{cluster_summary.loc[cluster_summary.algorithm=='FCM','suspect_fraction'].max():.1%}. Это исследовательская связь с подозрительными паттернами; соответствующие групповые интервалы показаны отдельно. Геометрическая разделимость не равна выделению всех трёх уровней риска. Метки не использованы в fit или выборе конфигурации.''',
        '05_results_and_error_analysis.ipynb':f'''# 05 · Итоги, ошибки и неопределённость
## Кратко
Предварительно выбранная **{locked['family']} / {locked['architecture']}** имеет test macro-F1 **{winner.macro_f1:.4f}**, распознаёт **{found}/{path_count}** патологических сегментов; к normal отнесено {missed_normal}, к suspect — {missed_suspect}. Accuracy **{winner.accuracy:.1%}** сравнивается с {test.loc[test.family=='always_normal','accuracy'].iloc[0]:.1%} у постоянного normal-baseline, у которого recall патологии равен 0.

Обычный MultiClass на этом test имеет macro-F1 {ordinary.macro_f1:.4f}; уверенного универсального победителя по одному разделению нет. По validation выбран {locked['architecture']}. Дополнительный порог чувствительности дал test recall {sensitivity.recall_path:.1%}, precision {sensitivity.precision_path:.1%}. Ни одна из этих test-проверок не использовалась для повторного выбора модели.'''
    }
    for filename,text in summaries.items():
        path=DEST/filename
        book=nb.read(path,as_version=4)
        book.cells[0].source=text
        for cell in book.cells:
            if cell.cell_type=='markdown':
                cell.source=cell.source.replace('атомарными отдельными checkpoint','отдельными checkpoint')
        nb.validate(book);nb.write(book,path)


if __name__ == '__main__':
    if '--legacy' not in sys.argv:
        from build_ctg_review_notebooks import build_selected, summaries
        if '--summaries-only' not in sys.argv:
            build_selected('ctg_uci',[2,3,4,5,6,7])
        summaries()
    elif '--summaries-only' in sys.argv:
        DEST = ROOT / 'research/notebooks/ctg_uci/history/v1'
        refresh_summaries()
    else:
        DEST = ROOT / 'research/notebooks/ctg_uci/history/v1'
        DEST.mkdir(parents=True, exist_ok=True)
        build_nested(); build_final(); build_clustering(); build_analysis()
    print(DEST)
