'use strict';
const screenings = (() => {
  let attempt = null, generation = 0, queue = Promise.resolve(), blocked = false;
  const pendingStarts = new Map();
  function reset() {
    generation++; attempt = null; blocked = false; queue = Promise.resolve(); pendingStarts.clear();
    for (const id of ['questionnaire-form','questionnaire-result','questionnaire-drafts','questionnaire-list','next-actions','attempts']) {
      const box = document.getElementById(id); if (box) box.replaceChildren();
    }
    document.getElementById('questionnaire-form').hidden = true;
    document.getElementById('questionnaire-list').hidden = false;document.getElementById('questionnaire-drafts').hidden = false;
  }
  function showSection() { show('questionnaires'); }
  function plural(n, forms) { const a=n%10, b=n%100; return forms[a===1&&b!==11?0:a>=2&&a<=4&&(b<12||b>14)?1:2]; }
  function sourceDetails(q) {
    const details = node('details',undefined,'questionnaire-sources');
    details.append(node('summary','Источники, редакция и метод оценки'));
    details.append(node('p',q.version),node('p',q.definition.validation_note || 'Редакция задана оператором.'));
    const links=node('ul');
    for (const ref of q.definition.references || []) {
      const li=node('li'), a=node('a',ref.title); a.href=ref.url; a.target='_blank'; a.rel='noopener noreferrer';
      li.append(a); if (ref.pages) li.append(document.createTextNode(', стр. '+ref.pages)); links.append(li);
    }
    details.append(links,node('p',q.definition.attribution || '', 'hint'));
    details.append(node('p',`Баллы: 0–${q.scoring_rules.maximum}. Сумма по индивидуальным ключам пунктов; ответы кодируются индексами 0–3. Дополнительный вопрос PHQ-9 в сумму не входит.`,'hint'));
    if (q.scoring_rules.thresholds?.length) details.append(table(['Баллы','Описание'],q.scoring_rules.thresholds.map(b=>[`${b.minimum}–${b.maximum}`,b.label])));
    details.append(node('p',q.definition.limitations || 'Самооценка симптомов, не диагноз.','hint'));
    return details;
  }
  function card(q) {
    const c=node('article',undefined,'card questionnaire-card');
    const count=q.definition.items.length;
    c.append(node('h3',q.definition.title || q.code),node('span',`${count} ${plural(count,['вопрос','вопроса','вопросов'])}`,'tag'));
    if(q.definition.timeframe_days)c.append(node('span',`о последних ${q.definition.timeframe_days} днях`,'tag tag--quiet'));
    c.append(node('p',q.definition.usage_note || 'Добровольная самооценка.'));
    const b=node('button','Начать '+q.code); b.type='button';
    b.onclick=()=>safe(async()=>{b.disabled=true;try{await start(q.id);}finally{b.disabled=false;}});
    c.append(b,sourceDetails(q)); return c;
  }
  async function load() {
    const g=generation;
    const [catalog,drafts,actions]=await Promise.all([
      api('/questionnaires'),api('/questionnaire-attempts?status=in_progress&limit=100'),api('/next-actions')
    ]);
    if(g!==generation)return;
    const box=$('#questionnaire-list');box.replaceChildren();
    for(const q of catalog.items.filter(q=>q.definition.usage!=='alternative_to_epds'))box.append(card(q));
    const alternatives=catalog.items.filter(q=>q.definition.usage==='alternative_to_epds');
    if(alternatives.length){const d=node('details');d.append(node('summary','Альтернатива EPDS: опросник PHQ-9'));d.append(node('p','PHQ-9 доступен для отдельного прохождения вместо EPDS по согласованию со специалистом. Система не предлагает обе шкалы автоматически.'));alternatives.forEach(q=>d.append(card(q)));box.append(d);}
    for(const q of catalog.unavailable){const c=node('div',undefined,'card');c.append(node('h3',q.code+' сейчас недоступен'),node('p',q.reason));box.append(c);}
    const d=$('#questionnaire-drafts');d.replaceChildren();
    if(drafts.items.length)d.append(node('h3','Можно продолжить'));
    for(const a of drafts.items){const row=node('div',undefined,'draft-row');row.append(node('span',`${a.questionnaire_code}: отвечено ${a.progress.answered} из ${a.progress.required}, начат ${localDate(a.started_at)}`));const b=node('button','Продолжить '+a.questionnaire_code);b.type='button';b.className='secondary';b.onclick=()=>safe(()=>open(a.id));row.append(b);d.append(row);}
    if(drafts.items.length===100)d.append(node('p','Остальные черновики доступны в истории.','hint'));
    $('#next-actions').replaceChildren();
    if(!actions.items.length)$('#next-actions').append(node('p','Сейчас ничего не запланировано.','hint'));
    for(const a of actions.items){const row=node('div',undefined,'list-item');row.append(node('span',a.type==='checkin'?'Добровольная самооценка самочувствия':a.type==='professional_support'?'Обратитесь за профессиональной поддержкой':`Можно пройти ${a.code}`));if(a.code){const b=node('button','Отложить');b.className='secondary';b.onclick=()=>safe(async()=>{await post('/assessment-schedule/'+a.code,{action:'declined'});await load();});row.append(b);}$('#next-actions').append(row);}
  }
  async function start(id) {
    if(attempt?.status==='in_progress'){notice('Сначала сохраните и закройте текущий черновик или завершите его.',true);return;}
    const g=generation;
    if(!pendingStarts.has(id))pendingStarts.set(id,crypto.randomUUID());
    const data=await post('/questionnaire-attempts',{questionnaire_id:id,client_request_id:pendingStarts.get(id)});
    if(g!==generation)return;
    pendingStarts.delete(id);generation++;attempt=data;blocked=false;render();showSection();
    await post('/assessment-schedule/'+data.questionnaire_code,{action:'offered'});
    await load();
  }
  async function open(id) {
    const g=generation;await queue.catch(()=>{});if(g!==generation)return;
    const data=await api('/questionnaire-attempts/'+id);if(g!==generation)return;
    generation++;attempt=data;blocked=false;render();showSection();
  }
  function collected() {
    const values=new FormData($('#questionnaire-form')),answers={},auxiliary_answers={};
    for(const i of attempt.questionnaire.definition.items)if(values.has('q-'+i.id))answers[i.id]=Number(values.get('q-'+i.id));
    for(const i of attempt.questionnaire.definition.auxiliary_items || [])if(values.has('aux-'+i.id))auxiliary_answers[i.id]=Number(values.get('aux-'+i.id));
    return {answers,auxiliary_answers};
  }
  function showSafety() {
    const box=$('#questionnaire-safety');if(!box)return;
    const wasHidden=box.hidden;box.hidden=!attempt.safety_flags.length;box.textContent=attempt.safety_guidance || '';if(wasHidden&&!box.hidden)box.scrollIntoView({behavior:'smooth',block:'center'});
  }
  function progress(message) {
    const p=$('#questionnaire-progress');if(!p || !attempt)return;
    const answered=Object.keys(collected().answers).length;
    p.textContent=`Ответов: ${answered} из ${attempt.progress.required}. ${message || ''}`;
    const bar=$('#questionnaire-progress-bar');if(bar)bar.value=answered;
  }
  function enqueueSave(complete=false) {
    const g=generation, payload=collected();
    queue=queue.catch(()=>{}).then(async()=>{
      if(g!==generation || !attempt || attempt.status==='completed')return;
      if(blocked)throw Error('Загрузите актуальный черновик перед сохранением.');
      progress('Сохраняю…');
      try {
        const data=await api(`/questionnaire-attempts/${attempt.id}${complete?'/complete':''}`,{
          method:complete?'POST':'PATCH',body:JSON.stringify({...payload,expected_revision:attempt.revision})
        });
        if(g!==generation)return;
        attempt=data;showSafety();
        if(complete){render();await load();notice('Результат сохранён в истории.');}
        else progress('Сохранено на сервере.');
      }catch(error){
        if(g!==generation)return;
        progress('Не сохранено. '+error.message);
        if(error.status===409){blocked=true;$('#questionnaire-reload').hidden=false;}
        throw error;
      }
    });
    return queue;
  }
  async function close() {
    const g=generation;
    if(attempt?.status==='in_progress')await enqueueSave();
    if(g!==generation)return;
    generation++;attempt=null;$('#questionnaire-list').hidden=false;$('#questionnaire-drafts').hidden=false;$('#questionnaire-form').hidden=true;$('#questionnaire-result').replaceChildren();await load();
    notice('Черновик сохранён. Его можно продолжить позже.');
  }
  function render() {
    const form=$('#questionnaire-form'),result=$('#questionnaire-result');result.replaceChildren();form.replaceChildren();$('#questionnaire-list').hidden=true;$('#questionnaire-drafts').hidden=true;
    if(attempt.status==='completed'){form.hidden=true;renderResult(result);return;}
    form.hidden=false;const q=attempt.questionnaire;
    form.append(node('h3',q.definition.title || q.code),node('p',q.definition.instructions));
    form.append(node('p',`Период оценки: последние ${q.definition.timeframe_days || '—'} дней. Можно сохранить черновик и вернуться позже.`,'hint'));
    const status=node('div',undefined,'questionnaire-status');
    const p=node('p',undefined,'questionnaire-progress');p.id='questionnaire-progress';p.setAttribute('role','status');
    const bar=node('progress');bar.id='questionnaire-progress-bar';bar.max=attempt.progress.required;bar.setAttribute('aria-label','Ответы на обязательные вопросы');
    status.append(p,bar);form.append(status);
    const safety=node('div',undefined,'safety-notice');safety.id='questionnaire-safety';safety.hidden=true;safety.setAttribute('role','alert');form.append(safety);
    for(const [aux,items] of [[false,q.definition.items],[true,q.definition.auxiliary_items || []]])for(const item of items){
      const fs=node('fieldset',undefined,'questionnaire-item');fs.dataset.itemId=(aux?'aux-':'q-')+item.id;
      fs.append(node('legend',(aux?'Дополнительно (не входит в сумму): ':item.id+'. ')+item.text));
      item.options.forEach((text,i)=>{const label=node('label',undefined,'answer-option'),input=node('input');input.type='radio';input.name=(aux?'aux-':'q-')+item.id;input.value=i;input.checked=(aux?attempt.auxiliary_answers:attempt.answers)[item.id]===i;input.onchange=()=>{fs.classList.remove('unanswered');progress('Есть изменения…');notice('');safe(()=>enqueueSave());};label.append(input,document.createTextNode(text));fs.append(label);});form.append(fs);
    }
    const buttons=node('div',undefined,'model-controls');
    const save=node('button','Сохранить и закрыть');save.type='button';save.className='secondary';save.onclick=()=>safe(close);
    const complete=node('button','Завершить и посмотреть результат');complete.type='submit';
    const reload=node('button','Загрузить актуальный черновик');reload.id='questionnaire-reload';reload.type='button';reload.className='secondary';reload.hidden=true;reload.onclick=()=>safe(()=>open(attempt.id));
    buttons.append(save,complete,reload);form.append(buttons,sourceDetails(q));
    form.onsubmit=e=>{e.preventDefault();safe(async()=>{
      const answers=collected().answers,missing=q.definition.items.filter(i=>!(i.id in answers));
      for(const i of missing)form.querySelector(`[data-item-id="q-${CSS.escape(i.id)}"]`)?.classList.add('unanswered');
      if(missing.length){notice(`Неотвеченных вопросов: ${missing.length}.`,true);form.querySelector('.unanswered input')?.focus();return;}
      const g=generation;notice('Сохраняю результат…');
      const done=enqueueSave(true);form.querySelectorAll('button,input').forEach(b=>b.disabled=true);
      try{await done;}finally{if(g===generation)form.querySelectorAll('button,input').forEach(b=>b.disabled=false);}
    });};
    progress('Черновик на сервере.');showSafety();form.scrollIntoView({behavior:'smooth',block:'start'});
  }
  function renderResult(box) {
    const q=attempt.questionnaire,r=attempt.result;
    box.append(node('h2',`${q.definition.title || q.code}: результат`));
    box.append(node('p',`${attempt.total_score} из ${q.scoring_rules.maximum} баллов`,'score-total'));
    if(attempt.safety_flags.length)box.append(node('p',attempt.safety_guidance,'safety-notice'));
    if(r){if(r.interpretation)box.append(node('p',r.interpretation.label));box.append(node('p',r.next_step),node('p',r.interpretation_note,'hint'));
      const details=node('details');details.append(node('summary','Как рассчитан результат'));
      details.append(node('p',`Расчёт Python по ключу ${r.scoring_version}. AI-модели при подсчёте не используются. Редакция: ${attempt.questionnaire_version}. Дата: ${localDate(attempt.completed_at)}.`,'hint'));
      details.append(table(['Пункт','Выбранный ответ','Баллы'],q.definition.items.map(i=>[i.id,i.options[attempt.answers[i.id]],r.item_scores[i.id]])));
      for(const i of q.definition.auxiliary_items || [])details.append(node('p',`${i.text} ${i.options[attempt.auxiliary_answers[i.id]] || 'Не отвечено'}. Не входит в сумму.`,'hint'));
      const raw=node('details');raw.append(node('summary','Ответ API / отчёт подсчёта'));raw.append(node('pre',JSON.stringify(r,null,2)));details.append(raw);box.append(details);
      if(r.recommended_action==='gad7_or_professional_assessment'){const b=node('button','Перейти к GAD-7');b.type='button';b.onclick=()=>safe(async()=>{const catalog=await api('/questionnaires'),q=catalog.items.find(q=>q.code==='GAD7');if(q)await start(q.id);else notice('GAD-7 сейчас недоступен.');});box.append(b);}
    }else box.append(node('p','Это историческое прохождение. Детальный отчёт для старой версии не пересчитывается.'));
    box.append(node('p','Результат самооценки не является диагнозом.','hint'),sourceDetails(q));
    const back=node('button','К списку опросников');back.type='button';back.className='secondary';back.onclick=()=>safe(async()=>{generation++;attempt=null;box.replaceChildren();$('#questionnaire-list').hidden=false;$('#questionnaire-drafts').hidden=false;await load();$('#questionnaires .page-head').scrollIntoView({behavior:'smooth',block:'start'});});box.append(back);box.scrollIntoView({behavior:'smooth',block:'start'});
  }
  function historyLinks(items) {
    const box=node('div',undefined,'attempt-links');
    for(const a of items){const b=node('button',`${a.status==='completed'?'Результат':'Продолжить'} ${a.questionnaire_code}, ${localDate(a.started_at)}`);b.type='button';b.className='secondary';b.onclick=()=>safe(()=>open(a.id));box.append(b);}return box;
  }
  return {load,open,reset,historyLinks};
})();
