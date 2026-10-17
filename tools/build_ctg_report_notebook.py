"""Build the isolated hospital-summary CTG feature study (08)."""
from pathlib import Path
import json
import nbformat as nb
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'research/notebooks/ctg_uci/08_hospital_reports_feature_engineering.ipynb'
OUT=ROOT/'research/experiments/ctg_uci/report_features_v1'


def md(text):return nb.v4.new_markdown_cell(text.strip())
def code(text):return nb.v4.new_code_cell(text.strip())


def build():
    result_intro=''
    if (OUT/'final_metrics.csv').exists():
        test=pd.read_csv(OUT/'final_metrics.csv');test=test[test.split=='test'].set_index('profile')
        cv=pd.read_csv(OUT/'nested_summary.csv');cv=cv[cv['mode']=='selected'].set_index('profile')
        result_intro='\n\n### Выполненный результат\n\n| Вход | Outer macro-F1 | Test macro-F1 | Recall патологии на test |\n|---|---:|---:|---:|\n'
        for profile in ['core2','common3','sisporto9']:
            r=test.loc[profile];c=cv.loc[profile]
            result_intro+=f'| {profile} | {c.mean_macro_f1:.3f} | {r.macro_f1:.3f} | {r.recall_path:.1%} |\n'
        result_intro+='\nТри условно общих показателя не дают приемлемой полноты выявления патологии. Девять нативных измерений сохраняют значительно больше качества, но недоступны из приложенного бланка. Engineering не устраняет отсутствие измерений. Test известен; это исследование переносимости признаков, не клиническая валидация.'
    cells=[md('''# 08 · Реальные бланки КТГ: семантика, feature engineering и ограниченные модели

## tl;dr

Этот ноутбук проверяет, сколько качества UCI сохраняется при ограничении входов до показателей, похожих на доступные в больничных отчётах. **Совпадение названия и единиц не доказывает совпадение измерения.** Модель для полного набора UCI нельзя применять к бланку, заполнив отсутствующие показатели нулями.

Ниже — выполненный поиск, три экспортированных варианта для разных входов, проверка пропусков/единиц и разбор прикреплённого примера. Существующие модели и кластеризация сохранены. Результаты этого эксперимента — исследовательские: перенос на больничное устройство ещё не проверен.
'''),md('''## Context & Methods · постановка до просмотра новых результатов

**Целевой продукт:** приложение для ведения беременности, дородовые КТГ, аппарат заранее неизвестен (уточнение пользователя). Общий формат принимает значения разных бланков и сохраняет неизвестное явно. Универсальная совместимость модели не следует из универсальности формы ввода.

**Учебная цель:** экспертный класс NSP (1 normal / 2 suspect / 3 pathologic). Он не является диагнозом гипоксии, вероятностью плохого исхода или меткой «8 из 8» конкретного прибора. Единица доступной группировки — исходная запись `source_file`; независимость пациенток не доказана.

### План

1. Сверить все 21 исходных признаков с XLS, восстановить число событий и экспозицию, проверить пропуски/нули/единицы и распределения.
2. Сопоставить больничные показатели по определениям, а не только по названиям. Отдельно сохранить значения, которых нет в обучающем UCI.
3. Обучить модели заново на трёх контрактах входа. `core2`: LB+AC; `common3`: LB+AC+сумма DL/DS/DP; `sisporto9`: LB, AC, UC, DL, DP, ASTV, MSTV, ALTV, MLTV. Последний — нативный ориентир, на фотографии этих 9 измерений нет.
4. Для каждого профиля совместно подобрать семейство, параметры и режим признаков внутри 5×3 grouped nested CV. Режимы: исходные / исходные+производные. A/B(SqrtBalanced/Balanced)/C(OvR)/D(иерархия)/логистика. Общие внешние и внутренние фолды из audited v2. Категории исходного файла, CLASS и выводы прибора в X не входят.
5. Grid CatBoost: depth=3/5, iterations=300/600, L2=3/10, learning_rate=.06. Логистика: C=.1/1/10 × class_weight=None/Balanced. На профиль 92 кандидата; всего 4140 внутренних оценок + 828 финальных. Это конечный проверенный grid, не гарантия глобального оптимума. Веса классов не комбинируются с oversampling.
6. Выбирать по mean inner segment macro-F1; tie — log loss. Внешний фолд оценивает выбор семейства и engineering, с фиксированным argmax. Отдельные raw/engineered процедуры позволяют увидеть пользу engineering без выбора по test.
7. Финальный grid использует train; деревья и скейлеры обучаются только на train. Пять моделей по фолдам усредняют готовые вероятности; ансамбль принимается при выигрыше ≥.01 macro-F1 на validation. Затем scalar temperature обучается на validation, сохраняет argmax. Порог на патологию в этой ветке не подбирается. Нет рефита на train+validation после калибровки.
8. Известный исследованию test используется для описательной проверки после фиксации решений. Это **не новая слепая внешняя проверка**. Доверительные интервалы resample исходные файлы, а не отдельные сегменты; они не включают неопределённость повторного обучения и выбора модели. Nested score не оценивает отдельный финальный выбор ансамбля/калибровку на validation.

### Key Assumptions

`common3` — условный proxy: сумма категорий децелераций UCI предполагает непересекающиеся категории событий. Совпадение с общим числом децелераций другого производителя ещё требуется доказать на одних и тех же сигналах. Поэтому ограничение признаков проверяет потерю информации **внутри UCI**, но не доказывает переносимость на бланк.
'''),code('''from pathlib import Path
import sys,json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'pyproject.toml').exists())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.features.ctg_report_features import PROFILES,ReportFeatures,uci_profile,audit_hospital_report,report_to_common3
from src.features.ctg_report_evidence import prepare_evidence,temporal_counterexample,example_report,SOURCES,INPUTS
from src.training.ctg_report_models import OUT,EXPORT,context,run_nested,run_final,class_details,bootstrap_results
from src.training.ctg_experiments import CTG_FEATURES
data,exposure,dictionary,mapping,example_audit=prepare_evidence()
_,fingerprint,cache,versions=context()
sns.set_theme(style='whitegrid',font_scale=1.0)
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
def show_save(name):
    plt.gcf().savefig(OUT/name,dpi=140,bbox_inches='tight')
    plt.show()
print('Fingerprint:',fingerprint)
display(pd.Series(versions,name='version').to_frame())
'''),md('''## Data · первичные значения и экспозиция

Источник — официальный UCI Cardiotocography и листы Description, Raw Data, Data исходного CTG.xls. Официальная публикация содержит 2126 измерений и 21 признак. [UCI](https://archive.ics.uci.edu/dataset/193/cardiotocography), DOI 10.24432/C51S4N, CC BY 4.0.

Проверка ниже вычисляет, а не предполагает: опубликованные AC/FM/UC/DL/DS/DP равны сырым счётчикам, делённым на `e−b` секунд. Не перепутать events/second с числом событий, events/minute или частотой за час. Исходные длительности сохраняются для аудита; в этой модели длительность не предиктор. Результат проверки всех строк — в отдельном CSV.
'''),code('''display(pd.read_csv(OUT/'event_rate_reconstruction.csv'))
display(dictionary[['feature','definition','missing','in_core2','in_sisporto9']])
display(dictionary[['feature','train_min','train_median','train_max','train_zero_pct','train_nonzero_files']].round(4))
display(exposure.groupby('split').agg(n=('NSP','size'),files=('source_file','nunique'),
    duration_min=('duration_s',lambda x:x.min()/60),duration_median=('duration_s',lambda x:x.median()/60),
    duration_max=('duration_s',lambda x:x.max()/60)))
display(data.groupby(['split','NSP']).agg(segments=('NSP','size'),files=('source_file','nunique')))
assert dictionary.missing.sum()==0
assert data.groupby('source_file').split.nunique().max()==1
assert data.groupby('feature_duplicate_group').split.nunique().max()==1
assert np.allclose(data.Width,data.Max-data.Min)
'''),md('''### Распределения всех исходных признаков · train

Нули событий сохранены: «событий не было» отличается от «не измеряли». Из-за малого числа записей DS с ненулевым значением тяжёлую категорию не используем как отдельный устойчивый предиктор. В common3 она остаётся в явно описанной сумме событий. Tendency — категория −1/0/1, гистограммные поля не являются показателями осцилляций из бланка. Эти поля не входят ни в один из трёх профилей.
'''),code('''train=data[data.split=='train'].copy()
fig,axes=plt.subplots(6,4,figsize=(16,20));axes=axes.ravel()
for ax,name in zip(axes,CTG_FEATURES):
    # Shared bin edges are essential when comparing class densities, especially
    # when one class is all zero and another contains rare nonzero events.
    edges=(np.array([-1.5,-.5,.5,1.5]) if name=='Tendency'
           else np.histogram_bin_edges(train[name],bins=22))
    for label,color in [(1,'#0072b2'),(2,'#e69f00'),(3,'#d55e00')]:
        sns.histplot(train.loc[train.NSP==label,name],bins=edges,stat='density',element='step',
                     fill=False,color=color,ax=ax,label=str(label))
    ax.set(title=name,ylabel='Density');ax.set_xlabel('Original UCI value')
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4),useMathText=True)
    if name=='Tendency':ax.set_xticks([-1,0,1]);ax.set_xlabel('Category')
for ax in axes[len(CTG_FEATURES):]:ax.axis('off')
axes[0].legend(title='NSP');fig.suptitle('Train distributions · original units · no clipping',y=1.0)
fig.tight_layout();show_save('01_original_feature_distributions.png')
'''),md('''## Что можно перенести из реального отчёта

Таблица ниже фиксирует пригодность каждого поля. `conditional` и `conditional_proxy` не означают подтверждённую эквивалентность.

В первичной работе по SisPorto 2 описаны собственные правила базальной линии, акцелераций, вариабельности и разбиение децелераций по длительности. [Авторы, 2004](https://www.sciencedirect.com/science/article/abs/pii/S030121150400212X). Правила менялись между версиями; определения позднего SisPorto нельзя автоматически приписывать каждому полю исторического UCI. [Первичное исследование с таблицей определений](https://pmc.ncbi.nlm.nih.gov/articles/PMC7516409/).

В руководстве Сономед отдельно определены ВКИ/vSTV, STV по 3.75-секундным эпохам и минутная LTV; значимость децелерации связана с площадью потерянных ударов. Расчёт вариабельности исключает некоторые минуты с потерями/децелерациями. [Руководство производителя, разделы 6.3–6.8](https://cordismed.ru/image/pdf/spektromed-sonomed-200-instrukcija.pdf). Это объясняет похожие строки, но **не устанавливает модель устройства на фото**. Похожая форма встречается и в [руководстве MFMon](https://usontec.com/uploads/s/9/2/c/92ckun9gej9h/file/raxxmfuv.pdf).

У Dawes–Redman STV измеряется через изменения средних пульсовых интервалов в 3.75-секундных эпохах; простой перевод ms→bpm не превращает её в SisPorto MSTV. Порог и контекст анализа зависят от метода. [Официальное руководство Huntleigh](https://www.huntleigh-diagnostics.com/wp-content/uploads/2025/07/749339EN-1.1-English-DAWES-REDMAN-CTG-ANALYSIS-Clinical-Application-Guide.pdf).
'''),code('''pd.set_option('display.max_colwidth',160)
display(mapping)
'''),md('''### Почему нельзя восстановить STV из агрегатов

Ниже математический пример, не клинические сигналы и не обучающие наблюдения. Две последовательности имеют одинаковую гистограмму (все её статистики одинаковы), но разный порядок точек и разную среднюю соседнюю разность. Следовательно, гистограмма не определяет временную вариабельность. Эта игрушечная соседняя разность не заявляется точной реализацией STV какого-либо прибора.
'''),code('''toy,slow,fast=temporal_counterexample();display(toy)
fig,axes=plt.subplots(1,2,figsize=(11,3.5),sharey=True)
for ax,x,title in zip(axes,[slow,fast],['Same histogram: two blocks','Same histogram: alternating']):
    ax.plot(x,marker='.',color='#0072b2');ax.set(xlabel='Ordered sample index',ylabel='Toy FHR (bpm)',title=title)
fig.tight_layout();show_save('02_temporal_counterexample.png')
'''),md('''## Разбор прикреплённого примера

В JSON сохранена ручная транскрипция чисел; устройство, версия ПО, срок и антенатальный/интранатальный контекст не выдуманы. Число 8/8 и галочки исключены: это выводы настроенной пользовательской системы. У этого примера нет независимой метки NSP.

- При 17 min частота акцелераций >15 BPM: 1/1020 событий в секунду. Строка >10 BPM содержит 5 событий и не подменяет первую.
- 13/17 — доля времени эпизодов высокой вариабельности в данном отчёте. Она не восстанавливает ASTV/ALTV.
- 17×(1−.085) — номинальное время без потерь. Оно не сообщает распределение артефактов по минутам и не определяет точную экспозицию детектора событий.
- Согласованность ms и BPM проверяем локальным приближением ΔBPM≈STV_ms×LB²/60000. Это проверка единиц, не эквивалентности алгоритмов.

Адаптер показывает доступные преобразования, маску пропусков и причины несовместимости. Для этого изображения выдача модельного класса заблокирована до проверки контракта измерений. Это не оценка состояния плода.
'''),code('''report=example_report()
display(pd.Series({k:v for k,v in report.items() if isinstance(v,(float,int))},name='transcribed_value').to_frame())
display(pd.Series(example_audit['derived'],name='unit_checks_and_descriptors').to_frame())
print('Adapter status:',example_audit['status'])
display(pd.DataFrame({'blocking_reason':example_audit['issues']}))
try:
    report_to_common3(report)
    raise AssertionError('Unverified hospital report must not reach inference')
except ValueError as exc:
    print('Correctly refused incompatible inference:',exc)
missing_example=dict(report,decelerations_15_15_count=None)
assert audit_hospital_report(missing_example)['missing_mask']['decelerations_15_15_count']
assert not example_audit['missing_mask']['decelerations_15_15_count']
print('Missing != zero: checked')
from src.features.ctg_pregnancy_app import prepare_pregnancy_app_record
app_record=prepare_pregnancy_app_record(report)
(INPUTS/'example_app_record.json').write_text(json.dumps(app_record,ensure_ascii=False,indent=2))
print('Unknown-device storage:',app_record['storage_status'])
print('Clinical model eligibility:',app_record['model_prediction_eligible'])
display(pd.DataFrame({'app_notice':app_record['notices']}))
'''),md('''## Feature engineering · формулы и защита от утечки

Обучаемые профили содержат только признаки, реально вычисляемые из соответствующего поднабора UCI. Производные детерминированы: RR=60000/LB; log1p(rate×1200); наличие событий; для common3 — log((D×1200+.5)/(A×1200+.5)) и баланс (A−D)/(A+D+1) в 20-минутном масштабе. .5 — фиксированная стабилизирующая константа, не наблюдавшееся событие. Для нативного профиля добавлены доли, их произведение и вариабельности относительно LB. Производные не создают новую информацию; пользу проверяет CV.

Вариабельность из фотографии, проценты потерь, срок беременности, частоты осцилляций и синусоидальные минуты сохраняются для будущей размеченной больничной базы. Из UCI для них нет обоснованных обучающих значений. **Обучать на синтетически подставленных «аналогах» было бы ошибкой.**

При структурно отсутствующем поле выбирается отдельный профиль модели. При пропуске обязательного значения предсказание прекращается. Маски качества сохраняются в аудите; обучать влияние таких масок пока не на чем, потому что UCI не содержит реального распределения этих пропусков. CatBoost работает без скейлера; у логистики RobustScaler заново fit внутри каждого fit-фолда, затем сохраняется внутри bundle и отдельно.
'''),code('''for profile in PROFILES:
    x=uci_profile(train,profile)
    engineered=ReportFeatures(profile,True).fit_transform(x)
    print(profile,':',len(x.columns),'raw →',len(engineered.columns),'engineered')
    display(pd.DataFrame({'feature':engineered.columns,'missing':engineered.isna().sum().values}))
'''),md('''## Results · выполнить вложенную проверку

Первый запуск выполняет весь grid. Повторный запуск использует кеш с SHA-256 исходных данных, кода методов и версий библиотек. CSV сохраняет все кандидаты и все внешние предсказания; файл фолдов позволяет проверить независимость источников. Семейство и engineering выбираются внутри каждого внешнего fit, а не по таблице test.
'''),code('''summary,fold_metrics=run_nested()
display(summary.round(4))
display(pd.read_csv(OUT/'outer_fold_support.csv'))
'''),code('''fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for ax,metric in zip(axes,['mean_macro_f1','mean_recall_path']):
    sns.barplot(data=summary,x='profile',y=metric,hue='mode',ax=ax,errorbar=None,
                palette={'raw':'#999999','engineered':'#56b4e9','selected':'#0072b2'})
    ax.set(ylim=(0,1),title=metric+' · outer CV',ylabel=metric,xlabel='Input contract')
fig.tight_layout();show_save('03_nested_comparison.png')
display(fold_metrics.pivot_table(index=['profile','outer_fold'],columns='mode',values='macro_f1').round(4))
'''),md('''### Финальные модели, ансамбли и известный test

Для каждого контракта выбирается свой лучший кандидат. Разные профили отвечают разным условиям доступности данных; 9-показательная модель не считается доступной на фотографии. Температура калибровки обучается на 341 validation-сегменте и не меняет класс argmax. Калибровка на другом устройстве не подтверждена. Validation-метрики выбранного ансамбля оптимистичны вследствие выбора на validation; смотрим также отдельный известный test и ограничения.
'''),code('''final_metrics,cards=run_final()
details=class_details()
intervals=bootstrap_results()
display(final_metrics[['profile','split','macro_f1','recall_suspect','recall_path','precision_path','ap_path','log_loss']].round(4))
display(final_metrics[['profile','split','file_macro_f1','file_recall_path','file_ap_path']].round(4))
display(cards[['profile','candidate','ensemble_accepted','members','temperature','scalers']])
display(details.round(4))
display(intervals.round(4))
'''),code('''fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,profile in zip(axes,PROFILES):
    cm=np.load(OUT/f'{profile}_test_confusion.npy')
    ConfusionMatrixDisplay(cm,display_labels=['Normal','Suspect','Pathologic']).plot(ax=ax,colorbar=False,cmap='Blues',values_format='d')
    ax.set_title(profile+' · known test')
    ax.grid(False)
fig.tight_layout();show_save('04_test_confusion_matrices.png')
predictions=pd.read_parquet(OUT/'final_predictions.parquet')
fig,axes=plt.subplots(1,2,figsize=(12,4.4))
for label,ax in zip([2,3],axes):
    for profile in PROFILES:
        t=predictions[(predictions.profile==profile)&(predictions.split=='test')]
        precision,recall,_=precision_recall_curve(t.NSP==label,t[f'p{label-1}'])
        ap=average_precision_score(t.NSP==label,t[f'p{label-1}'])
        ax.step(recall,precision,where='post',label=f'{profile}: AP={ap:.3f}')
    ax.axhline((t.NSP==label).mean(),ls='--',color='grey',label='Prevalence')
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision',title=f'One-vs-rest NSP {label} · known test');ax.legend()
fig.tight_layout();show_save('05_precision_recall.png')
'''),md('''### Сопоставление с полной моделью audited v2

Показываем потерю качества при ограничении измерений. У audited v2 отличается более широкий grid и процедура выбора калибровки/порогов; это описательный ориентир, а не чистая рандомизированная абляция. Чистое сравнение raw и engineering внутри нового эксперимента выше использует одинаковый grid и фолды. Test был известен ещё до этой ветки, поэтому значения не доказывают превосходство на будущих пациентках.
'''),code('''full=pd.read_csv(ROOT/'research/experiments/ctg_uci/audited_v2/test_metrics.csv')
display(full[['candidate','primary','macro_f1','recall_path','ap_path','file_macro_f1']].round(4))
display(final_metrics[final_metrics.split=='test'][['profile','macro_f1','recall_path','precision_path','ap_path','file_macro_f1']].round(4))
'''),md('''## Проверки кода, экспорта и научные границы

Проверка ниже независимо пересчитывает метрики из сохранённых предсказаний, сверяет номера строк/фолды/группы, проверяет SHA-256 bundle, inference после загрузки и единичный/batch режим. Скейлеры не дообучаются при predict. Требуемые пропуски и лишние поля с оценками отвергаются. Проводится отдельный набор тестов для единиц, временной информации и скейлера. Все результаты сохранены в ipynb.

Нативные `.cbm` — только деревья. Для корректного inference нужен bundle: контракт полей, engineering, подготовка, объединение голов и temperature. В файле schema указаны единицы и причина отсутствия скейлера у деревьев. Не заменять девять нативных измерений внешними похожими показателями.

Дополнительное ограничение исходной базы: LB в train находится только в диапазоне 106–160 BPM. Модель не проверена на всём диапазоне больничных КТГ. Метод bundle.support_report отмечает выходы за диапазон обучения; это проверка применимости, не медицинский порог. Даже отсутствие такого флага не доказывает совместимость другого прибора.
'''),code('''from tools.validate_ctg_report_models import validate
receipt=validate()
display(pd.Series(receipt,name='validation_receipt').to_frame())
'''),md('''## Takeaways · приложение для ведения беременности

1. Зафиксировать конкретную клиническую задачу, срок/контекст, аппарат и версию ПО; получить его определения и способ выгрузки исходного цифрового сигнала/сводки. Одна фотография не определяет универсальный формат.
2. На одних и тех же записях сопоставить измерения устройства и учебного экстрактора: окна, потери, детекторы, единицы, длительность. Проверить смещения/согласие; это отдельная проверка измерений, а не CV классификатора.
3. Собрать размеченные обезличенные больничные записи с ID пациентки, независимой экспертной разметкой и заранее выбранным клиническим исходом. Достаточное число патологических пациенток важнее числа похожих сегментов; размер рассчитывается под требуемую точность recall и внешнюю проверку.
4. Учить/калибровать на реальных STV/LTV/ВКИ и масках качества в этом датасете. Разделять по пациенткам, затем по времени/центру/прибору; внешняя проверка должна быть закрыта до фиксации pipeline и порогов. Оценить и подозрительные, и патологические случаи, пропуски тяжёлых случаев, PPV, калибровку, подгруппы и долю отказов.
5. Для цифровых КТГ можно сделать единый версионированный экстрактор и применять его и к обучению, и к новым сигналам. Доступная в проекте CTU-UHB относится к родам и другим исходам; нельзя смешивать её pH с UCI NSP и выдавать результат за проверенную дородовую модель.

### Решение по применению

Готовые экспортированные модели пригодны для воспроизводимого исследования UCI с соответствующими измерениями. **Для автоматической медицинской оценки бланка на фотографии подтверждения пока нет.** Удобный ввод подготовлен; формальная возможность посчитать число не является доказательством, что оно корректно на другом устройстве.
''')]
    cells[0].source+=result_intro
    takeaway=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('## Takeaways'))
    cells[takeaway:takeaway]=[md('''## Дополнительный источник именно для дородового сценария

После уточнения назначения приложения найден [JNU-CTG, Zenodo, версия 1.0.0](https://zenodo.org/records/21800730). Ниже — отдельный аудит скачанного первоисточника. Его данные **не объединяются** с UCI и не меняют уже выбранные модели.

Документация описывает дородовые записи и идентификаторы пациенток. В доступной разметке FIGO только normal/abnormal; это не трёхклассовый NSP. Исход neonatal_asphyxia — другой endpoint. Числа поддержки и пропусков проверены по CSV, содержимое и MD5 сверены с опубликованным архивом.

Полей F_STV/F_LTV по названию недостаточно для сопоставления с фотографией: нужны единицы и реализация экстрактора. Также нельзя добавлять в дородовую модель Apgar, родоразрешение, послеродовые признаки или неизвестно как обученные NetFea. В документации gestational_age описан при родах; это не подтверждённый срок на момент каждого КТГ. Источник перспективнее для следующего дородового исследования, но не устраняет внешнюю проверку устройств.
'''),code('''from src.features.ctg_jnu_candidate import audit_jnu
support_jnu,ranges_jnu,jnu_receipt,wave_jnu=audit_jnu()
display(support_jnu)
display(ranges_jnu.round(4))
display(pd.Series({k:v for k,v in jnu_receipt.items() if k not in ['blockers']},name='source_audit').to_frame())
display(pd.DataFrame({'measurement_or_design_gap':jnu_receipt['blockers']}))
'''),md('''### Маски на исходном сигнале

В этом источнике нулевой FHR — код пропуска, в отличие от нуля числа акцелераций в UCI. На одном реальном цифровом примере показываем разрывы после маски FHR=0. Нули UC автоматически не маскируем: значение может быть настоящим. Не интерполируем длинные разрывы и не называем производные собственным STV производителя.
'''),code('''fig,axes=plt.subplots(2,1,figsize=(12,5),sharex=True)
time_min=np.arange(len(wave_jnu))/4/60
fhr=wave_jnu.FHR.mask(wave_jnu.FHR==0)
axes[0].plot(time_min,fhr,lw=.7,color='#0072b2');axes[0].set(ylabel='FHR (bpm)',title='JNU-CTG · one 30-minute raw recording · zeros masked')
axes[1].plot(time_min,wave_jnu.UC,lw=.7,color='#009e73');axes[1].set(ylabel='UC (device units)',xlabel='Minutes from recording start')
fig.tight_layout();show_save('06_antenatal_raw_signal_mask.png')
print('FHR zero/missing fraction in this recording:',float((wave_jnu.FHR==0).mean()))
''')]
    location=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and c.source.startswith('## Results'))
    cells[location:location]=[md('''### Одинаковые входы могут иметь разные экспертные классы

Проверка только train. Считаем точные совпадения входов с разными NSP; не округляем для создания совпадений. Это не оценка будущего Bayes error: возможны шум разметки и особенности сегментации. Но ни один детерминированный классификатор не может различить эти конкретные строки по одинаковым входам. Дополнительные функции тех же входов не снимают неоднозначность.
'''),code('''collision_rows=[]
for profile in PROFILES:
    x=uci_profile(train.reset_index(drop=True),profile)
    x['NSP']=train.NSP.to_numpy()
    grouped=x.groupby(list(PROFILES[profile])).NSP.agg(['size','nunique'])
    collision_rows.append(dict(profile=profile,conflicting_input_groups=int((grouped['nunique']>1).sum()),
        segments_in_conflicting_groups=int(grouped.loc[grouped['nunique']>1,'size'].sum())))
collision_table=pd.DataFrame(collision_rows)
collision_table.to_csv(OUT/'train_input_collisions.csv',index=False);display(collision_table)
x=uci_profile(train,'common3');x['NSP']=train.NSP
groups=x.groupby(list(PROFILES['common3'])).NSP.agg(['size','nunique'])
shared=groups[groups['nunique']>1].sort_values('size',ascending=False).index[0]
mask=(x[list(PROFILES['common3'])]==shared).all(axis=1)
print('Identical common3 inputs:',shared)
display(train.loc[mask,['source_excel_row','LB','AC','NSP','ASTV','MSTV','ALTV','MLTV']].head(20))
''')]
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (femfedora)','name':'python3','language':'python'},
        'language_info':{'name':'python','version':'3.11'}})
    nb.validate(book);nb.write(book,DEST)
    print(DEST)


if __name__=='__main__':
    from build_ctg_review_notebooks import build_selected
    build_selected('ctg_uci',[8])
