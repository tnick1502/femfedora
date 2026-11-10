'use strict';
const diagnostics = (() => {
  let adminToken = '', registry = null, editing = null;
  const stageNames = {
    authorization:'Авторизация', persistence:'Сохранение сообщения', input_rules:'Правила на входе',
    jev_analysis:'Анализ Jev', safety_route:'Маршрут безопасности', complaints:'Наблюдения и жалобы',
    context:'Сборка контекста', state_evaluation:'Оценка потребности', protocol_selection:'Выбор протокола',
    llm_generation:'Генерация ответа', output_safety:'Проверка ответа', memory:'Обновление памяти'
  };
  const reasons = {
    jwt_verified:'JWT проверен', conversation_ownership_verified:'Проверен владелец беседы',
    client_message_id_new:'Новое сообщение; повторный вызов не нужен', user_message_saved:'Входящая реплика сохранена',
    urgent_bypass:'Срочная реакция до вызова моделей', external_ai_consent_missing:'Нет согласия на внешний AI',
    ai_disabled:'Внешняя обработка отключена', classifier_completed:'Классификация завершена',
    classifier_unavailable:'Классификатор недоступен; оценок нет', analysis_skipped_for_urgent:'Классификация пропущена для немедленного ответа',
    no_risk_rule_triggered:'Правила риска не сработали', deterministic_urgent_rule:'Сработало детерминированное срочное правило',
    deterministic_referral_rule:'Сработало правило рекомендации специалиста',
    highest_marker_above_action_threshold:'Основная потребность выбрана по максимальному маркеру выше порога',
    equal_markers_fixed_category_priority:'При равных маркерах использован фиксированный порядок категорий',
    no_support_marker_above_threshold:'Нет эмоционального маркера выше порога; выбрана общая поддержка',
    recurring_complaint_observations:'Повторные наблюдения жалобы; это не оценка тяжести',
    functional_impact_above_threshold:'Маркер влияния на повседневную жизнь выше порога',
    postpartum_context_present:'Учтён послеродовой контекст', approved_protocol_matched:'Подходящий протокол утверждён и применён',
    requested_protocol_not_approved_or_missing:'Протокол ещё не утверждён или отсутствует; применён safe_baseline',
    urgent_protocol_bypass:'Обычный протокол пропущен в пользу срочного шаблона', provider_called:'Провайдер вызван',
    urgent_template:'Использован срочный шаблон', limited_mode:'Использован шаблон ограниченной поддержки',
    llm_unavailable:'LLM недоступна; использован технический шаблон', output_safety_fallback:'Ответ заблокирован после повторной генерации',
    deterministic_output_rule_blocked:'Программная проверка заблокировала ответ', required_referral_missing:'В ответе нет обязательной рекомендации специалиста',
    output_classifier_unavailable:'Проверка ответа Jev недоступна; ответ не считается безопасным',
    all_output_checks_passed:'Все проверки ответа пройдены', configured_template_used:'Использован заранее заданный шаблон',
    bounded_history_and_personal_context:'Загружены ограниченная история и персональный контекст',
    user_content_is_untrusted_data:'История и память переданы как данные',
    observations_stored_without_inferred_resolution:'Наблюдения сохранены; молчание не трактуется как выздоровление',
    no_fabricated_observations:'Новые наблюдения не создавались без классификации',
    summary_refreshed_from_user_sources:'Резюме обновлено из сообщений пользователя', summary_refresh_not_due:'Обновление резюме пока не требуется',
    medical_urgent_template:'Использован шаблон срочной медицинской помощи', mental_state_urgent_template:'Использован шаблон срочной оценки психического состояния',
    outbound_text_pseudonymized:'Текст для внешних моделей обезличен; значения в отчёт не попадают', anonymization_disabled:'Обезличивание выключено',
    context_over_budget:'Контекст не помещается в окно модели; запрос не отправлен', unverified_contact_blocked:'В ответе модели был телефон или сайт не из утверждённого списка',
    standing_referral_due:'Постоянное основание для обращения: пора напомнить о специалисте',
    standing_referral_recently_made:'Специалист уже рекомендован недавно; рекомендация не повторяется в каждом ответе',
    'specialist:safety_route':'Основание: маршрут безопасности', 'specialist:persistent_complaint':'Основание: жалоба повторяется в разные дни не менее двух недель',
    'specialist:functional_impact':'Основание: состояние мешает повседневной жизни',
    'specialist:persistent_maternal_estrangement':'Основание: устойчивое ощущение несостоятельности как матери или отчуждения от ребёнка'
  };
  const routeNames = {normal:'обычный маршрут', recommend_professional:'рекомендован специалист', urgent:'срочный маршрут'};
  const statuses = {passed:'Пройдено', completed:'Выполнено', skipped:'Пропущено', unavailable:'Недоступно', ok:'В пределах окна', compressed:'Сжато', over_budget:'Превышение окна', applied:'Применено', disabled:'Выключено',
    failed:'Ошибка', blocked:'Заблокировано', updated:'Обновлено', unchanged:'Без изменений'};
  function reasonText(value) {
    if (reasons[value]) return reasons[value];
    const screening = /^specialist:screening_(followup|safety_item):(\w+)$/.exec(value);
    if (screening) return screening[1] === 'followup'
      ? `Основание: результат ${screening[2]} на уровне порога направления (самооценка, не диагноз)`
      : `Основание: отмечен пункт безопасности в ${screening[2]}`;
    const marker = /^(\w+):above_(action|review)_threshold$/.exec(value);
    if (marker) return `Маркер ${marker[1]} выше порога ${marker[2] === 'action' ? 'действия' : 'проверки'}`;
    return value;
  }
  function renderMessage(message, host) {
    if (message.role !== 'assistant') return;
    const report = message.metadata?.decision_report;
    if (!report) return;
    const root = node('details', undefined, 'decision-report dev-only');
    root.dataset.traceId=report.trace_id;
    const summary = node('summary'), route = report.result.route;
    summary.append(node('span', 'Как принят ответ'), node('span', routeNames[route] || route, 'route route--' + route), node('span', `${(report.total_ms / 1000).toFixed(1)} с`));
    root.append(summary);
    const models = node('div', undefined, 'report-models');
    for (const kind of ['llm','jev']) {
      const m = report.models[kind];
      const calls = kind === 'llm' ? report.result.llm_calls : Number(report.result.jev_input_called) + report.result.jev_output_calls;
      models.append(node('span', `${kind.toUpperCase()}: ${m.model || 'отключена'}, ${m.provider}, вызовов: ${calls}`, 'tag'));
    }
    root.append(models, node('p', `Настройки моделей: ревизия ${report.models.revision}. Trace: ${report.trace_id}`, 'hint'));
    const protocol = report.stages.find(s=>s.stage==='protocol_selection');
    if (protocol) root.append(node('p', `Паттерн: ${protocol.facts.requested}. Применён: ${protocol.facts.applied || 'срочный шаблон'}. ${protocol.reasons.map(reasonText).join('. ')}`));
    const jev = report.stages.find(s=>s.stage==='jev_analysis');
    if (jev) {
      root.append(node('h4','Результат анализа Jev'), node('p', `${statuses[jev.status] || jev.status}. ${jev.reasons.map(reasonText).join('. ')}`));
      if (Object.keys(jev.facts.markers).length) {
        root.append(node('p', `Намерение: ${jev.facts.intent}. Порог: ${jev.facts.action_threshold}; порог проверки: ${jev.facts.review_threshold}. Значения — оценки маркеров текста, не вероятность заболевания.`, 'hint'));
        root.append(table(['Маркер','Значение Noul','Выше порога'], Object.entries(jev.facts.markers)
          .sort((a,b)=>b[1]-a[1]).map(([k,v])=>[names[k] || k, Number(v).toFixed(3), v>=jev.facts.action_threshold?'Да':'Нет'])));
      }
    }
    root.append(node('h4','Этапы обработки'));
    const timeline = node('ol', undefined, 'decision-timeline');
    for (const stage of report.stages) {
      const item = node('li');
      item.append(node('strong', stageNames[stage.stage] || stage.stage), node('span', ` — ${(statuses[stage.status] || stage.status).toLowerCase()}`, 'hint'));
      item.append(node('p', stage.reasons.map(reasonText).join('. ') || 'Проверка завершена.'));
      const facts = node('details'); facts.append(node('summary','Данные этапа'),node('pre',JSON.stringify(stage.facts,null,2)));
      item.append(facts); timeline.append(item);
    }
    const raw = node('details'); raw.append(node('summary','Полный JSON-отчёт'),node('pre',JSON.stringify(report,null,2)));
    root.append(timeline,raw); host.append(root);
  }
  async function showModels() {
    if (!token) return;
    const data = await api('/models');
    const box = $('#active-models'); box.replaceChildren();
    box.append(node('span', `Режим ${data.ai_mode}, ревизия ${data.revision}`, 'models-strip__label'));
    for (const kind of ['llm','jev']) { const tag = node('span', `${kind.toUpperCase()}: ${data[kind].model || 'отключена'}`, 'tag'); tag.title = `Провайдер: ${data[kind].provider}. Модели общие для всех профилей.`; box.append(tag); }
    $('#mode').textContent = data.ai_mode === 'mock' ? 'Режим: шаблонная модель' : data.ai_mode === 'disabled' ? 'Режим: AI отключён' : 'Режим: внешние модели';
    $('#model-hint').textContent = data.ai_mode === 'mock' ? 'Включены детерминированные mock-модели. Реальные вызовы Jev/LLM не выполняются. Отчёт обработки доступен под ответом.' : data.ai_mode === 'disabled' ? 'AI отключён. Используются ограниченные шаблоны; классификация не выполняется.' : 'Модели общие для всех профилей. Фактические вызовы и проверки указаны в отчёте под каждым новым ответом.';
  }
  async function adminApi(path, options={}) {
    const requestToken=adminToken;
    const response = await fetch('/api/v1/emo'+path,{...options,headers:{'X-Admin-Token':adminToken,...(options.body?{'Content-Type':'application/json'}:{})}});
    const data = await response.json();
    if(requestToken!==adminToken)throw Error('Админская сессия изменилась.');
    if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
    return data;
  }
  function clearEditor() {
    editing = null; const form = $('#model-form'); form.reset();
    form.elements.kind.disabled=false; $('#model-form-title').textContent='Добавить модель';
    $('#model-keep-option').hidden=true; $('#model-secret-source').value='env'; updateCredentialFields();
  }
  function updateCredentialFields() {
    const source=$('#model-secret-source').value;
    $('#credential-ref-label').hidden=source!=='env'; $('#api-key-label').hidden=source!=='inline';
    $('#model-api-key').required=source==='inline'; $('#model-credential-ref').required=source==='env';
    if(source!=='inline')$('#model-api-key').value='';
  }
  function editProfile(profile) {
    editing=profile.id; const form=$('#model-form');
    for (const field of ['kind','name','model','base_url','timeout_seconds'])form.elements[field].value=profile[field];
    form.elements.kind.disabled=true; $('#model-api-key').value=''; $('#model-form-title').textContent='Изменить модель';
    $('#model-keep-option').hidden=false; $('#model-secret-source').value='keep'; updateCredentialFields();
    $('#model-form').scrollIntoView({behavior:'smooth',block:'start'});
  }
  async function setRuntime(changes) {
    if(!registry)throw Error('Обновите список моделей.');
    await adminApi('/admin/model-runtime',{method:'PATCH',body:JSON.stringify({...changes,expected_revision:registry.runtime.revision})});
    await refreshAdmin(); if(token)await showModels();
    notice('Общие настройки применены. Новые сообщения всех пользователей используют эту конфигурацию.');
  }
  async function refreshAdmin() {
    if(!adminToken)return;
    registry=await adminApi('/admin/models');
    $('#admin-panel').hidden=false; $('#admin-login-state').textContent='Администратор авторизован';
    $('#runtime-mode').value=registry.runtime.ai_mode;
    $('#runtime-description').textContent=registry.runtime.configuration_error || `Ревизия ${registry.runtime.revision}. Общая LLM: ${registry.runtime.llm.model||'отключена'}; общий Jev: ${registry.runtime.jev.model||'отключён'}.`;
    $('#model-hosts').textContent=`Разрешённые hosts из .env: LLM — ${registry.allowed_hosts.llm||'не заданы'}; Jev — ${registry.allowed_hosts.jev||'не заданы'}.`;
    const refs=$('#model-credential-ref'); refs.replaceChildren(node('option','Выберите ключ из .env')); refs.firstElementChild.value='';
    for(const ref of registry.credential_refs){const option=node('option',`${ref.name}${ref.configured?'':' · не заполнен'}`);option.value=ref.name;option.disabled=!ref.configured;refs.append(option);}
    $('#key-encryption-note').textContent=registry.key_encryption_configured?'Ключи из панели шифруются на сервере. При редактировании прежний ключ не показывается.':'Для ввода ключа в панели сначала задайте MODEL_SECRETS_KEY в .env. Можно использовать ссылку на env-ключ.';
    const list=$('#model-profiles'); list.replaceChildren();
    if(!registry.items.length)list.append(node('p','Профилей пока нет. Сейчас используются настройки .env.'));
    for(const profile of registry.items){
      const card=node('div',undefined,'model-profile card'); card.append(node('h4',`${profile.name} (${profile.kind.toUpperCase()})`),node('code',profile.model),node('p',profile.base_url),node('span',profile.active?'Активный профиль':'Сохранённый профиль','tag'),node('p',`Ключ: ${profile.key_source==='env'?'ссылка '+profile.credential_ref:'сохранён зашифрованно'}. Таймаут: ${profile.timeout_seconds} с.`,'hint'));
      const edit=node('button','Редактировать','secondary');edit.type='button';edit.onclick=()=>editProfile(profile);
      const activate=node('button',`Выбрать для всех как ${profile.kind.toUpperCase()}`);activate.type='button';activate.disabled=profile.active;activate.onclick=()=>safe(()=>setRuntime({[profile.kind+'_profile_id']:profile.id}));
      const controls=node('div',undefined,'model-controls');controls.append(edit,activate);card.append(controls);list.append(card);
    }
    const events=await adminApi('/admin/model-events?limit=20');
    $('#model-events').replaceChildren(table(['Ревизия','Когда','Изменение','Поля'],events.items.map(e=>[e.revision,localDate(e.created_at),e.action,e.metadata.fields.join(', ')])));
  }
  function bind() {
    $('#admin-login-form').onsubmit=e=>{e.preventDefault();safe(async()=>{adminToken=$('#admin-token').value;$('#admin-token').value='';try{await refreshAdmin();notice('Доступ администратора подтверждён.');}catch(error){adminToken='';$('#admin-panel').hidden=true;$('#admin-login-state').textContent='Доступ не подтверждён';throw error;}});};
    $('#admin-logout').onclick=()=>{adminToken='';registry=null;clearEditor();$('#admin-panel').hidden=true;$('#admin-login-state').textContent='Вход не выполнен';notice('Админский токен удалён из памяти страницы.');};
    $('#model-secret-source').onchange=updateCredentialFields;
    $('#cancel-model-edit').onclick=clearEditor;
    $('#model-form').onsubmit=e=>{e.preventDefault();safe(async()=>{
      const wasActive=editing && registry.items.some(p=>p.id===editing && p.active);
      const f=e.target;const body={name:f.elements.name.value.trim(),model:f.elements.model.value.trim(),base_url:f.elements.base_url.value.trim(),timeout_seconds:+f.elements.timeout_seconds.value};
      if(!editing)body.kind=f.elements.kind.value;
      const source=$('#model-secret-source').value;
      if(source==='env')body.credential_ref=$('#model-credential-ref').value;
      if(source==='inline')body.api_key=$('#model-api-key').value.trim();
      $('#model-api-key').value='';
      await adminApi('/admin/models'+(editing?'/'+editing:''),{method:editing?'PATCH':'POST',body:JSON.stringify(body)});
      clearEditor();await refreshAdmin();if(token)await showModels();notice(wasActive?'Активный профиль изменён для всех пользователей; новые сообщения используют обновление.':'Профиль сохранён. Выберите его для всех, чтобы применить.');
    });};
    $('#runtime-form').onsubmit=e=>{e.preventDefault();safe(()=>setRuntime({ai_mode:$('#runtime-mode').value}));};
    $('#reset-model-profiles').onclick=()=>safe(()=>setRuntime({llm_profile_id:null,jev_profile_id:null}));
    $('#refresh-models').onclick=()=>safe(refreshAdmin);
    clearEditor();
  }
  window.addEventListener('DOMContentLoaded',bind);
  async function useConsoleSession(value){adminToken=value;$('#admin-login-form').hidden=!!value;if(!value){registry=null;$('#admin-panel').hidden=true;}}
  return {renderMessage,showModels,refreshAdmin,useConsoleSession};
})();
