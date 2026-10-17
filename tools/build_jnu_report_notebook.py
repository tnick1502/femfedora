"""Build the executed research companion for report-compatible CTG inputs."""
from pathlib import Path
import json
import nbformat as nb
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/experiments/jnu_ctg/report_features_v1'
BOOK=ROOT/'research/notebooks/jnu_ctg/02_report_features.ipynb'


def summary():
    text='''# 02 · JNU-CTG · признаки для обычных бланков КТГ

## tl;dr

**Вход приложения — числа больничного отчёта.** Сигналы JNU используются только для построения исследовательских аналогов этих измерений. Реализованы обработка цифровой КТГ, разбор русского текста бланка и алгебра признаков без сигнала. Приборы с одинаковыми подписями могут использовать разные формулы; соответствие производителю не заявляется.

Обзор: 18 записей первоисточников, включая 5 руководств/технических документов производителей. Разделены STV 3.75/4 s, несколько правил эпизодов вариабельности, глубина и площадь децелерации, взаимоисключающие/вложенные категории акцелераций. Подробный [обзор и формулы](REPORT_FEATURE_METHODS.md); [новый план классификаторов](ANALYSIS_PLAN.md).
'''
    if (OUT/'validation_receipt.json').exists():
        r=json.loads((OUT/'validation_receipt.json').read_text())
        p=pd.read_csv(OUT/'profiles.csv')
        text+=f"\n**Выполнено:** обработаны {r['records']:,} записей; {len(p)} профилей по 3–13 полям, сохранены train-only преобразования и маски. Скейлеры обучены на {r['fit_records']:,} train-записях с инженерным правилом покрытия/поддержки. Остальные строки сохранены. На вашем переписанном примере разобраны 18 измерений; итог «8 из 8» не попал в признаки.\n"
    if (OUT/'duration_prefix_train.parquet').exists():
        d=pd.read_parquet(OUT/'duration_prefix_train.parquet')
        a=d.query('prefix_min==17').set_index('record_id');b=d.query('prefix_min==30').set_index('record_id')
        delta=(a.stv_ms-b.stv_ms).abs().dropna()
        text+=f'\nНа тех же 128 train-записях сокращение окна с 30 до 17 min изменило STV по модулю в медиане на **{delta.median():.2f} ms**, 95-й перцентиль — **{delta.quantile(.95):.2f} ms**. Нормирование счётчиков не устраняет эту потерю временной информации.\n'
    if (OUT/'train_quality_by_label.csv').exists():
        q=pd.read_csv(OUT/'train_quality_by_label.csv').set_index('figo_label')
        text+=f"\n**Обнаружено смещение отбора:** reference quality-флаг пропускает {q.loc[0,'reference_fit_fraction']:.1%} normal и {q.loc[1,'reference_fit_fraction']:.1%} abnormal train-записей. Его нельзя автоматически использовать для удаления данных классификатора. Полная когорта и отдельные NaN/маски сохранены.\n"
    text+='''
Шевеления, истинная ВКИ и длительность медицинского синусоидального ритма из FHR/UC достоверно не восстанавливаются. На бланке они могут присутствовать и сохраняются как собственные измерения, однако в JNU для них нет подтверждённого обучения. **Новый классификатор в этом ноутбуке не обучается**; точность детекторов событий и перенос на реальные российские бланки ещё не валидированы.
'''
    return text


def build():
    md=nb.v4.new_markdown_cell; code=nb.v4.new_code_cell
    cells=[md(summary()),md('''## Context & Methods

### Key Assumptions

- Приложение дородовое; анализ в родах и клинические пороги из таких работ нельзя автоматически переносить. У JNU бинарная **экспертная FIGO**, не три класса и не диагноз гипоксии.
- 30-минутные записи 4 Hz и фиксированные patient-component splits из 01 сохраняются. Формулы заданы до анализа; графики, чувствительность и опубликованные сравнения — **только train**. Test лишь преобразуется фиксированным кодом и проверяется на целостность.
- Собственная базальная линия — медиана/итеративное отсечение, **не клон Mantel, WMFB, SisPorto или Dawes–Redman**. JNU не содержит экспертной разметки baseline/events; синтетические проверки оценивают код, а не медицинскую точность.
- Заполняются только внутренние разрывы ≤2 s, временная ось сохраняется. Неизвестное ≠ нормальное ≠ ноль. Полные события и цензурированные кандидаты различаются.
- LTV-минуты: >75% исходных наблюдений, без части децелерации. STV дополнительно требует полных эпох после краткого заполнения и ≥80% наблюдавшихся точек в каждой эпохе. Эти дополнительные ограничения — инженерная версия v1.
- Набор полей отчёта определяет отдельную модель. Энтропия, спектр, площади/длительности событий и скрытые временные параметры недоступны, если их нет на бланке. Они не входят в основные report profiles.
- median imputer + RobustScaler fit только по train с заранее заданной минимальной поддержкой; маски всех колонок сохраняются. В CV обязательно fit необученный шаблон заново внутри каждого fit-fold. CatBoost использует значения без масштабирования.
- 10/17/20-минутные префиксы — анализ устойчивости. Полная FIGO-метка **не приписывается** короткому окну: оно может содержать другую картину КТГ.
'''),code('''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'pyproject.toml').exists())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.features.jnu_ctg_report_extraction import *
from src.features.jnu_ctg_report_analysis import (export_preprocessing,train_sensitivity,
    compare_published_train,example_report_outputs,validate_exports,write_contracts,MODELS)
from src.features.ctg_russian_reports import EXAMPLE_TEXT,FIELDS
sns.set_theme(style='whitegrid',font_scale=1.)
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
pd.set_option('display.max_colwidth',85)
OUT.mkdir(parents=True,exist_ok=True)
def show_save(name):
    plt.gcf().savefig(OUT/name,dpi=140,bbox_inches='tight')
    plt.show()
'''),md('''## Data · воспроизводимое извлечение всех записей

Источник: [JNU-CTG 1.0.0](https://zenodo.org/records/21800730). Исходные CRC/SHA и связи с patient_id проверены в 01. Код ниже читает memory-map исходных FHR/UC и сохраняет собственные признаки отдельно от подготовки 01. Кеш принимается только при совпадении кода, конфигурации, source manifest и хешей созданных файлов. Никакие опубликованные NetFea/F_STV не используются для настройки экстрактора.

Инженерное `report_quality_eligible`: ≥80% наблюдаемого FHR, ≥80% покрытия рассчитанной базальной линией и ≥5 пригодных STV-минут. Это **не клиническая пригодность** и не доказанное достаточное время исследования. В 02 он задаёт только fit-когорту reference scaler. Это не автоматическая рекомендация удалить остальные записи из обучения/оценки классификатора. Все строки остаются в X; причины отсутствия STV могут зависеть от децелераций, поэтому сам отбор способен смещать классы. В будущем отказ модели включается в знаменатель охвата.
'''),code('''measurements=extract_dataset()
splits=pd.read_parquet(SOURCE/'splits.parquet')
index=pd.read_parquet(SOURCE/'signal_index.parquet')
signals=np.load(SOURCE/'signals_original_float32.npy',mmap_mode='r')
train=measurements.merge(splits.loc[splits.split=='train',['record_id','patient_id','patient_component']],on='record_id',validate='one_to_one')
profiles=export_preprocessing(measurements)
display(profiles)
support=pd.DataFrame({'records':[len(train)],'patients':[train.patient_id.nunique()],
    'quality_rule_pass':[train.report_quality_eligible.sum()],
    'no_usable_STV_minute':[(train.eligible_stv_min==0).sum()],
    'censored_event_candidates':[(train.event_censored_candidates>0).sum()]})
support.to_csv(OUT/'train_support.csv',index=False)
display(support)
q=splits.loc[splits.split=='train',['record_id','figo_label']].merge(measurements[['record_id','report_quality_eligible']],on='record_id',validate='one_to_one')
quality_by_label=q.groupby('figo_label').agg(records=('record_id','size'),reference_fit_records=('report_quality_eligible','sum'),reference_fit_fraction=('report_quality_eligible','mean'))
quality_by_label.to_csv(OUT/'train_quality_by_label.csv')
display(quality_by_label.round(3))
print('All rows retained:',measurements.shape,'; target columns in X: none')
'''),md('''### Что доступно в вашем примере

18 полей в переписанном примере — это значения в двух единицах вместе с длительностью и качеством; они не являются 18 независимыми физиологическими характеристиками. Таблица различает воспроизводимую формулу, неизвестную формулу прибора и отсутствие нужного канала. Неизвестные поля JNU экспортируются как NaN.
'''),code('''mapping=pd.DataFrame([
    ['Время / потери сигнала','duration_min / signal_loss_pct','вычисляется','маска прибора может отличаться'],
    ['Базальная ЧСС','baseline_bpm','приближение','собственная базальная линия'],
    ['AC10, AC15, DC15','acc10 / acc15 / dec15','приближение','сегментация и категория AC10'],
    ['Децелерации ≥20 ударов','dec20lost_count','приближение','площадь относительно базальной линии'],
    ['STV мс / BPM','stv_ms / stv_bpm','формула + маски v1','3.75 s; отдельный вариант 4 s'],
    ['Амплитуда мс / BPM','amplitude_ms / amplitude_bpm','формула + маски v1','минутный размах; не размах всей КТГ'],
    ['Высокая / низкая вариабельность','sonomed / mfmon / dr candidates','правило + оценки v1','разные пороги и длительности'],
    ['Частота осцилляций','osc_* diagnostics','исследовательский proxy','плавающая линия/фильтры неизвестны'],
    ['Синусоидальный ритм, min','sinusoidal_min','недоступно','спектральная полоса не диагноз'],
    ['ВКИ, ms / BPM','vstv_ms','недоступно','нужны истинные интервалы и формула'],
    ['Частота шевелений','movements_per_hour','недоступно','нет актограммы/маркера'],
    ['ПСП, баллы, итоговые критерии','excluded_conclusions','не predictor','выход другой системы/заключение'],
],columns=['Поле бланка','Столбец / семейство','Статус','Ограничение'])
mapping.to_csv(OUT/'report_field_mapping.csv',index=False)
display(mapping)
'''),md('''## Results · базальная линия, события и маски на train

Примеры выбраны из train по рассчитанным дескрипторам для проверки трёх разных ситуаций, без клинических меток. Цветные участки — найденные полные события; штриховка — пропуски. Это техническая визуальная проверка правдоподобия, не экспертная разметка точности. Кандидат у пропуска/края не считается завершённым событием.
'''),code('''eligible=train[train.report_quality_eligible].copy()
examples=[eligible.sort_values('record_id').iloc[0].record_id,
    eligible.sort_values('dec15_count',ascending=False).iloc[0].record_id,
    train.loc[train.signal_loss_pct.between(15,30)].sort_values('record_id').iloc[0].record_id]
fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True)
for ax,record in zip(axes,examples):
    pos=int(index.set_index('record_id').loc[record,'signal_row'])
    f,ev,m,d=extract_report_features(signals[pos],diagnostics=True)
    time=np.arange(7200)/240
    ax.plot(time,np.where(d['observed'],signals[pos,:,0],np.nan),color='#666666',lw=.65,label='FHR observed')
    ax.plot(time,d['baseline'],color='#9e2a2b',lw=1.5,label='Operational baseline')
    for a,b in runs(~d['observed']):ax.axvspan(a/240,b/240,color='#9e9e9e',alpha=.25)
    for event in ev.itertuples():
        if not event.censored and event.peak_bpm>=15:
            ax.axvspan(event.start_s/60,event.end_s/60,color='#0072b2' if event.kind=='acceleration' else '#e69f00',alpha=.2)
    ax.set(title=f'{record} · loss {f["signal_loss_pct"]:.1f}% · AC15 {f["acc15_count"]:.0f} · DC15 {f["dec15_count"]:.0f}',ylabel='FHR, BPM')
axes[0].legend(handles=[Line2D([],[],color='#666666',label='Observed FHR'),
    Line2D([],[],color='#9e2a2b',label='Operational baseline'),
    Patch(color='#0072b2',alpha=.3,label='Complete acceleration >=15'),
    Patch(color='#e69f00',alpha=.3,label='Complete deceleration >=15'),
    Patch(color='#9e9e9e',alpha=.3,label='Missing FHR')],loc='upper right',fontsize=8,ncol=2)
axes[-1].set_xlabel('Time from start, min')
fig.suptitle('Train only · three signal and mask checks',y=1.01)
fig.tight_layout();show_save('01_signal_event_checks.png')
'''),md('''### Распределения и поддержка измерений

Распределения строятся по исходным значениям без winsorization и без заполнения NaN. Большие значения не удаляются только потому, что необычны. Каждый график показывает число реально рассчитанных значений. Недостаток пригодных минут должен быть виден рядом со STV.
'''),code('''cols=['baseline_bpm','acc15_per_10min','dec15_per_10min','stv_ms','stv_4s_ms','amplitude_bpm','signal_loss_pct','eligible_stv_min']
fig,axes=plt.subplots(2,4,figsize=(16,7))
for ax,col in zip(axes.flat,cols):
    values=train[col].dropna()
    ax.hist(values,bins=40,color='#0072b2',alpha=.85)
    ax.set(title=f'{col} · n={len(values):,}',xlabel=col,ylabel='Train records')
fig.tight_layout();show_save('02_train_distributions.png')
needed=list(dict.fromkeys(c for p in PROFILE_FIELDS.values() for c in p))
available=pd.concat([train[['record_id']],profile_frame(train,'sonomed_fields')],axis=1)
missing=available.drop(columns='record_id').isna().mean().sort_values(ascending=False)
missing.to_csv(OUT/'train_report_missingness.csv',header=['missing_fraction'])
fig,ax=plt.subplots(figsize=(10,5))
ax.barh(missing.index,missing.values*100,color='#d55e00')
ax.set(xlabel='Missing values, % of train records',title='Missing measurements stay unknown before preprocessing')
fig.tight_layout();show_save('03_missing_measurements.png')
display(train[['eligible_stv_min','event_censored_candidates','signal_loss_pct','adjacent_jump_gt25_fraction']].describe().round(3))
'''),md('''### Чувствительность к способу расчёта · 128 независимых train-компонент

Пример из каждой компоненты выбирается по стабильному порядку ID, затем случайная выборка с фиксированным seed. Это не «первая по времени» запись. Сравниваются 11 настроек: семейство/окно базальной линии, заполнение 0/1/2 s, полоса возврата и сглаживание событий. Основная настройка задана заранее и **не выбирается по FIGO**. Таблица показывает изменение измерений на одних записях; это не MAE относительно медицинского эталона.
'''),code('''sensitivity,duration,delta_summary=train_sensitivity(measurements,sample_size=128)
chosen=['baseline_bpm','acc15_count','stv_ms']
fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
for ax,col in zip(axes,chosen):
    part=delta_summary[delta_summary.feature==col].set_index('recipe')
    ax.barh(part.index,part.median_absolute_change,color='#0072b2',label='Median absolute change')
    ax.plot(part.p95_absolute_change,part.index,'o',color='#d55e00',label='95th percentile')
    ax.set(xlabel=f'Change in {col}',title=col)
axes[0].legend(fontsize=9);fig.suptitle('Train only · sensitivity relative to median_trim, 300 s',y=1.02)
fig.tight_layout();show_save('04_recipe_sensitivity.png')
display(delta_summary[delta_summary.feature.isin(chosen)].pivot(index='recipe',columns='feature',values='p95_absolute_change').round(3))
'''),md('''### Длительность исследования и разные эпизоды вариабельности

17 min взяты из примера пользователя. Нормирование числа событий по длительности устраняет только арифметическую разницу знаменателя; оно не возвращает пропущенный эпизод активности. Полная 30-минутная запись JNU не доказывает качество модели на 17-минутном бланке. Префиксы сохраняются без новых target-меток.

Правила эпизодов на графике используют одну и ту же собственную минутную LTV. Поэтому различия вызваны определениями даже до дополнительной разницы приборов. D/R-кандидат не включает гестационное подтверждение.
'''),code('''reference=duration.query('prefix_min==30').set_index('record_id')
short=duration.query('prefix_min==17').set_index('record_id')
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,col in zip(axes,['acc15_per_10min','stv_ms','amplitude_bpm']):
    ax.scatter(reference[col],short[col],alpha=.55,s=18,color='#0072b2')
    vals=pd.concat([reference[col],short[col]]).dropna()
    if len(vals):ax.plot([vals.min(),vals.max()],[vals.min(),vals.max()],'--',color='#555555',lw=1)
    ax.set(xlabel=f'30 min · {col}',ylabel=f'17 min · {col}')
fig.suptitle('Same 128 train records · shorter windows change the measurements',y=1.03)
fig.tight_layout();show_save('05_duration_sensitivity.png')
episode_cols=['sonomed_rule_high_min','mfmon_rule_high_min','dr_5of6_32ms_candidate_min','mak_rule_candidate_high_min']
fig,axes=plt.subplots(1,2,figsize=(13,4))
for col,style in zip(episode_cols,['-','--',':','-.']):
    sns.ecdfplot(data=train,x=col,ax=axes[0],label=col,linestyle=style)
axes[0].legend(fontsize=8);axes[0].set(xlabel='High variation / candidate, min',ylabel='Cumulative fraction',title='Train only · definitions differ')
valid=train[['stv_ms','stv_4s_ms']].dropna()
hexplot=axes[1].hexbin(valid.stv_ms,valid.stv_4s_ms,gridsize=50,mincnt=1,cmap='Blues',norm=LogNorm())
fig.colorbar(hexplot,ax=axes[1],label='Records per hexagon (log scale)')
lims=[0,max(valid.max())];axes[1].plot(lims,lims,'--',color='#555555')
axes[1].set(xlabel='3.75-second epochs, ms',ylabel='4-second epochs, ms',title='STV recipes are separate measurements')
fig.tight_layout();show_save('06_variability_definitions.png')
display(train[episode_cols].agg(['count','median','mean']).round(2))
'''),md('''### Опубликованные признаки — сравнение, не эталон прибора

Официальные F_BL/F_A_N/F_D_N/F_STV/F_LTV сравниваются только на train. В выпущенном архиве нет полного подтверждённого контракта их единиц, вычисления и предобработки. Поэтому показаны отдельные медианы и ранговая связь; **разность неизвестных единиц не объявляется MAE измерения**. Высокая корреляция не доказывает совпадение измерений; низкая не доказывает ошибку авторов. Эти признаки не использованы для настройки и не добавлены в основные X.
'''),code('''comparison=compare_published_train(measurements)
display(comparison.round(3))
published=pd.read_parquet(SOURCE/'published_features_quarantined.parquet')
paired=train.merge(published[['record_id','F_BL','F_STV']],on='record_id',validate='one_to_one')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,own,pub in zip(axes,['baseline_bpm','stv_ms'],['F_BL','F_STV']):
    z=paired[[own,pub]].dropna()
    hexplot=ax.hexbin(z[pub],z[own],gridsize=60,mincnt=1,cmap='Blues',norm=LogNorm())
    fig.colorbar(hexplot,ax=ax,label='Records (log scale)')
    ax.set(xlabel=f'Published {pub} · definition unverified',ylabel=f'Operational {own}',title='Train only · no measurement equivalence claimed')
fig.tight_layout();show_save('07_published_comparison.png')
'''),md('''### Ваш бланк → числа → производные признаки без сигнала

Ниже используется **ручная транскрипция** прикреплённого примера. Это проверяет обработчик текста, не точность OCR фотографии. Парсер сохраняет строку-источник, не захватывает нормативные числа в скобках, отделяет заключения и требует подтверждения. Неизвестный прибор остаётся неизвестным. «Потеря 8.5%» не позволяет восстановить интервалы потерь; ВКИ не заменяется соседней разностью JNU.
'''),code('''parsed,example,audit=example_report_outputs()
contract=write_contracts()
display(pd.DataFrame(parsed['evidence']))
display(example[['acc15_per_10min','dec15_per_10min','high_variation_fraction','baseline_rr_ms','nominal_nonlost_min','small_acc10_15_count']].round(4))
display(pd.DataFrame({'Требует проверки':audit['issues']}))
print('Не вошло в признаки:',parsed['unparsed_lines'])
print('Клинический риск:',audit['clinical_risk'],'; подтверждённый перенос прибора:',audit['vendor_bridge_validated'])
from src.features.ctg_russian_reports import report_profile_input
unverified_input,input_audit=report_profile_input(parsed['measurements'],'sonomed_fields')
display(unverified_input[['stv_ms','sonomed_rule_high_fraction','acc10_per_10min']])
print('Для этих полей нет подтверждённых определений; исходные числа сохранены выше.')
'''),md('''### Независимые проверки и экспорт

Проверяются ID/полнота, fitted median/IQR против независимого расчёта, отсутствие fit на validation/test, маски всех полей и сохранённые необученные шаблоны. Исходные X с NaN нужны для CatBoost и fold-local обработки. Готовые масштабированные матрицы — только reference export; **не вход в nested CV**.

Числа редких исходов и групповые границы взяты из неизменённой подготовки 01. Метки отсутствуют во всех новых матрицах X. Отдельная модель по клиническому исходу требует разрешения противоречий цели и проверки времени КТГ относительно родов.
'''),code('''receipt=validate_exports(measurements)
display(pd.DataFrame(receipt['profiles']))
display(pd.DataFrame({'version':receipt['environment']}))
print('Сохранено:',DEST.relative_to(ROOT))
print('Шаблоны и обученные преобразования:',MODELS.relative_to(ROOT))
print('Финальная научная/клиническая валидация:',receipt['clinical_validation'])
'''),md('''## Takeaways

1. Для приложения основной путь — **несколько моделей по доступным полям отчёта**: минимальный набор, вариабельность и расширенные профили. Четыре семейства определений приборов нельзя незаметно смешать в один STV/LTV.
2. Код позволяет извлечь много исследовательских характеристик, но в deployable X входят только реально доступные поля. Шевеления/ВКИ/синусоидальная длительность не создаются из отсутствующего канала. Счётчик ноль и неизвестный счётчик различаются.
3. Маски, базальная линия и длительность изменяют измерения; точные изменения видны выше и сохранены в таблицах. Выбор версии нельзя оправдывать лучшей test-accuracy. Нужны экспертные аннотации событий или парные измерения.
4. 30-минутную FIGO-метку нельзя автоматически переносить на 10–17-минутное окно. Нельзя по бинарной разметке JNU научить различать suspect и pathologic без новых меток.
5. Следующий этап описан в [плане v2](ANALYSIS_PLAN.md): patient-grouped nested Grid Search, отдельные задачи FIGO/исхода, калибровка фактического ансамбля и пороги без test; внешняя проверка на **реальных бланках** до обещаний приложению.

**Границы обзора:** первоисточники и уровень доступа перечислены в [методах](REPORT_FEATURE_METHODS.md). Полные закрытые формулы устройств, точность OCR и клиническая валидация не получены. Это рабочая исследовательская подготовка и контракт входа, а не медицинский анализатор.
''')]
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (.venv)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
    nb.validate(book);nb.write(book,BOOK)
    print(BOOK)


if __name__=='__main__':
    from build_ctg_review_notebooks import build_selected
    build_selected('jnu_ctg',[2])
