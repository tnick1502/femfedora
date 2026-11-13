'use strict';
const $ = s => document.querySelector(s);
const P = '/api/v1/emo';
const SVG_NS = 'http://www.w3.org/2000/svg';
let profileEpoch = 0;
let gateToken = '', loginRequired = false;
let token = '', conversationId = '', lastAnswer = '', attemptOffset = 0, memoryPreferences = {};
let contacts = null, sending = false, pendingMessage = null, noticeTimer = 0;
let messageLoad = 0;
const names = {anxiety:'Тревога',fear_for_baby:'Переживания за ребёнка',low_mood:'Сниженное настроение',loss_of_interest:'Потеря интереса',sleep_problem:'Трудности со сном',fatigue:'Усталость',irritability:'Раздражительность',loneliness:'Одиночество',relationship_stress:'Напряжение в отношениях',overwhelm:'Перегрузка',physical_discomfort:'Физический дискомфорт'};
const complaintStatus = {new:'Замечено впервые',active:'Повторяется',improving:'Становится легче',resolved:'Больше не беспокоит',unknown:'Не уточнялось'};
const signalNames = {...names, request_for_support:'Просьба о поддержке', self_reported_worsening:'Стало хуже', need_for_clarification:'Нужно уточнить', functional_impact:'Трудно справляться с делами', possible_self_harm:'Тяжёлые мысли', self_harm_intent:'Тяжёлые мысли', possible_violence_exposure:'Небезопасно дома', maternal_estrangement:'Трудно чувствовать связь с ребёнком', possible_urgent_medical_issue:'Тревожные симптомы', request_for_medical_advice:'Медицинский вопрос', possible_psychosis:'Необычные ощущения'};
const journalKinds = {all:'Всё', checkin:'Мои отметки', signals:'Наблюдения EMO', questionnaire:'Опросники', safety:'Безопасность'};
let journalFilter = 'all', journalData = null;
const entityNames = {name:'имена',phone:'телефоны',email:'почта',link:'ссылки',address:'адреса',facility:'учреждения',number:'номера',date:'даты',protected_term:'твои слова'};

/* ---------- Мелкие помощники ---------- */
function node(tag, text, cls) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; }
function icon(name) {
  const svg = document.createElementNS(SVG_NS, 'svg'), use = document.createElementNS(SVG_NS, 'use');
  svg.setAttribute('class', 'icon'); svg.setAttribute('aria-hidden', 'true'); use.setAttribute('href', '#i-' + name); svg.append(use); return svg;
}
function stored(key, fallback) { try { return localStorage.getItem(key) || fallback; } catch (e) { return fallback; } }
function store(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* хранилище недоступно — настройка живёт до перезагрузки */ } }
function notice(text, error = false) {
  const box = $('#notice'); clearTimeout(noticeTimer);
  if (!text) { box.classList.remove('show'); return; }
  box.textContent = text; box.className = (error ? 'error ' : '') + 'show';
  noticeTimer = setTimeout(() => box.classList.remove('show'), error ? 9000 : 4500);
}
async function api(path, options = {}) {
  const requestToken = token;
  const headers = {...(options.body ? {'Content-Type': 'application/json'} : {}), ...(token ? {Authorization: `Bearer ${token}`} : {})};
  const response = await fetch(path.startsWith('/dev/') ? path : P + path, {...options, headers: {...headers, ...options.headers}});
  const data = await response.json();
  if (requestToken !== token) throw Error('Профиль изменён. Повтори действие.');
  if (!response.ok) {
    const error = Error(typeof data.detail === 'string' ? data.detail : data.detail?.message || JSON.stringify(data.detail));
    error.status = response.status; error.detail = data.detail; throw error;
  }
  return data;
}
function post(path, body) { return api(path, {method: 'POST', body: JSON.stringify(body)}); }
async function safe(fn) { try { await fn(); } catch (e) { notice(e.message, true); } }
function localDate(value) { return new Date(value).toLocaleString('ru-RU', {day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit'}); }
function clock(value) { return new Date(value).toLocaleTimeString('ru-RU', {hour: '2-digit', minute: '2-digit'}); }
function dayLabel(value) {
  const d = new Date(value), today = new Date(), start = x => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((start(today) - start(d)) / 864e5);
  if (diff === 0) return 'Сегодня';
  if (diff === 1) return 'Вчера';
  return d.toLocaleDateString('ru-RU', {day: 'numeric', month: 'long', ...(d.getFullYear() !== today.getFullYear() ? {year: 'numeric'} : {})});
}
function table(headers, rows) {
  const wrap = node('div', undefined, 'table-wrap'), t = node('table'), tr = node('tr');
  headers.forEach(v => tr.append(node('th', v)));
  const head = node('thead'); head.append(tr); t.append(head);
  const body = node('tbody');
  rows.forEach(row => { const r = node('tr'); row.forEach(v => r.append(node('td', String(v ?? '—')))); body.append(r); });
  t.append(body); wrap.append(t); return wrap;
}
function emptyState(text, actionLabel, action) {
  const box = node('div', undefined, 'empty-state'); box.append(node('p', text));
  if (actionLabel) { const b = node('button', actionLabel); b.type = 'button'; b.className = 'secondary'; b.onclick = action; box.append(b); }
  return box;
}
function confirmDialog(title, text, actionLabel) {
  const dialog = $('#confirm');
  $('#confirm-title').textContent = title; $('#confirm-text').textContent = text; $('#confirm-yes').textContent = actionLabel;
  return new Promise(resolve => {
    const done = value => { dialog.onclose = null; if (dialog.open) dialog.close(); resolve(value); };
    $('#confirm-yes').onclick = () => done(true); $('#confirm-no').onclick = () => done(false); dialog.onclose = () => resolve(false);
    dialog.showModal(); $('#confirm-no').focus();
  });
}

/* ---------- Вид: раздел, тема, служебный режим ---------- */
function show(view) {
  if (view !== 'chat' && voice.active) voice.reset();
  document.querySelectorAll('.view').forEach(v => v.hidden = v.id !== view);
  document.querySelectorAll('#nav button[data-view]').forEach(b => {
    const on = b.dataset.view === view; b.classList.toggle('active', on);
    if (on) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
  });
  $('#nav-more').classList.toggle('active', !!$(`#nav .nav-sheet button[data-view="${view}"]`));
  closeMore(); document.body.dataset.view = view; window.scrollTo(0, 0);
}
function closeMore() { $('#nav').classList.remove('open'); $('#nav-more').setAttribute('aria-expanded', 'false'); }
$('#nav-more').onclick = () => { const open = $('#nav').classList.toggle('open'); $('#nav-more').setAttribute('aria-expanded', String(open)); };
document.addEventListener('click', e => { if (!e.target.closest('#nav')) closeMore(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMore(); });
for (const button of document.querySelectorAll('#nav button[data-view]')) button.onclick = () => safe(async () => { show(button.dataset.view); notice(''); if (token) await refresh(); });

const themes = {auto: ['auto', 'как в системе'], light: ['sun', 'светлая'], dark: ['moon', 'тёмная']};
function applyTheme(value) {
  const theme = themes[value] ? value : 'auto', [glyph, label] = themes[theme], button = $('#theme-toggle');
  document.documentElement.dataset.theme = theme;
  button.replaceChildren(icon(glyph)); button.title = `Тема: ${label}`; button.setAttribute('aria-label', `Тема: ${label}. Нажми, чтобы сменить`);
}
$('#theme-toggle').onclick = () => { const order = Object.keys(themes), next = order[(order.indexOf(document.documentElement.dataset.theme) + 1) % order.length]; store('emo.theme', next); applyTheme(next); };
applyTheme(stored('emo.theme', 'auto'));

function applyDev(on) {
  document.documentElement.dataset.dev = on ? 'on' : 'off'; $('#dev-toggle').setAttribute('aria-pressed', String(on));
  if (!on && !$('#admin').hidden) show('chat');
}
$('#dev-toggle').onclick = () => { const on = document.documentElement.dataset.dev !== 'on'; store('emo.dev', on ? 'on' : 'off'); applyDev(on); notice(on ? 'Служебные данные показаны: отчёты под ответами, модели, API-ключ.' : 'Служебные данные скрыты. Так интерфейс видит пользователь.'); };
applyDev(stored('emo.dev', 'on') === 'on');

/* ---------- Срочная помощь ---------- */
function phoneBook() {
  if (!contacts) return [];
  const book = (contacts.emergency.phones || []).map(p => ({digits: String(p.number), show: String(p.number), label: p.label, emergency: true}));
  const s = contacts.support;
  if (s?.phone) book.push({digits: String(s.phone), show: s.phone_display || s.phone, label: s.label || 'Линия психологической помощи', hours: s.hours});
  return book;
}
function phonePattern(book) {
  if (!book.length) return null;
  return new RegExp('(?<!\\d)(' + book.map(p => p.digits.split('').join('[\\s\\-()]*')).join('|') + ')(?!\\d)', 'g');
}
function renderSos() {
  const list = $('#sos-list'), book = phoneBook(); list.replaceChildren();
  for (const p of book) {
    const a = node('a', undefined, 'sos-row'), text = node('span'); a.href = 'tel:' + p.digits;
    text.append(node('strong', p.show), node('span', p.label + (p.hours ? ' — ' + p.hours : '')));
    a.append(icon('phone'), text); list.append(a);
  }
  if (!book.length) list.append(node('p', contacts?.emergency.text || 'Номер экстренных служб зависит от страны. Войди в профиль, чтобы увидеть номера.'));
  $('#sos-note').textContent = contacts?.support?.operator ? `Линию поддержки ведёт ${contacts.support.operator}. EMO с ней не связана.` : '';
}
function openSos() { renderSos(); $('#sos').showModal(); }
$('#sos-open').onclick = openSos;
document.addEventListener('click', e => { if (e.target.closest('[data-sos]')) openSos(); });
$('#sos').addEventListener('click', e => { if (e.target === $('#sos')) $('#sos').close(); });

/* ---------- Вход ---------- */
function gateHeaders() { return {'Content-Type': 'application/json', ...(gateToken ? {Authorization: `Bearer ${gateToken}`} : {})}; }
async function login() {
  const epoch = ++profileEpoch;
  screenings.reset(); pendingMessage = null; sending = false; token = ''; conversationId = ''; lastAnswer = ''; contacts = null;
  voice.reset(); voice.setBusy(false);
  for (const control of [$('#text-send'), $('#message'), $('#conversations'), $('#new-chat')]) control.disabled = false;
  $('#messages').replaceChildren(); $('#messages').dataset.signature = ''; $('#feedback').hidden = true;
  $('#complaint-list').replaceChildren(); $('#trends').replaceChildren(); $('#journal-list').replaceChildren(); journalData = null; $('#memory-summary').value = '';
  const response = await fetch('/dev/token', {method: 'POST', headers: gateHeaders(), body: JSON.stringify({user: $('#user').value})});
  const data = await response.json();
  if (!response.ok) throw Error(data.detail);
  if (epoch !== profileEpoch) return;
  token = data.access_token;
  contacts = await api('/support-contacts').catch(() => null);
  await loadConversations(); await refresh(); await voice.refresh();
}
async function copyApiKey() {
  // Отдельный долгоживущий токен выбранного профиля; сессионный токен интерфейса не раскрывается.
  const response = await fetch('/dev/token', {method: 'POST', headers: gateHeaders(), body: JSON.stringify({user: $('#user').value, purpose: 'api'})});
  const data = await response.json(); if (!response.ok) throw Error(data.detail);
  try { await navigator.clipboard.writeText(data.access_token); }
  catch (e) { const area = document.createElement('textarea'); area.value = data.access_token; document.body.append(area); area.select(); document.execCommand('copy'); area.remove(); }
  notice(`API-ключ профиля «${$('#user').selectedOptions[0].textContent}» скопирован. Передавай его в заголовке Authorization: Bearer …; действует до ${localDate(data.expires_at)}.`);
}
$('#copy-key').onclick = () => safe(copyApiKey);
$('#gate-form').onsubmit = e => { e.preventDefault(); (async () => {
  $('#gate-error').textContent = '';
  const response = await fetch('/auth/login', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({login: $('#gate-login').value, password: $('#gate-password').value})});
  const data = await response.json(); $('#gate-password').value = '';
  if (!response.ok) { $('#gate-error').textContent = typeof data.detail === 'string' ? data.detail : 'Не удалось войти. Проверь логин и пароль.'; return; }
  gateToken = data.access_token; $('#gate').hidden = true; $('#gate-logout').hidden = false;
  await diagnostics.useConsoleSession(gateToken); await safe(login);
})().catch(() => { $('#gate-error').textContent = 'Сервис не отвечает. Проверь, что он запущен, и попробуй снова.'; }); };
$('#gate-logout').onclick = () => { gateToken = ''; token = ''; profileEpoch++; voice.reset(); voice.setBusy(false); diagnostics.useConsoleSession(''); $('#messages').replaceChildren(); $('#gate-logout').hidden = true; $('#gate').hidden = false; $('#gate-login').focus(); };
$('#login').onclick = () => safe(async () => { await login(); notice('Сессия профиля обновлена.'); });
$('#user').onchange = () => safe(login);

/* ---------- Разговор ---------- */
function greeting() {
  const hour = new Date().getHours();
  const line = hour >= 5 && hour < 11 ? 'Доброе утро.' : hour >= 11 && hour < 17 ? 'Добрый день.' : hour >= 17 && hour < 23 ? 'Добрый вечер.' : 'Ночью всё кажется тяжелее.';
  const box = node('div', undefined, 'hello'), row = node('div', undefined, 'starters');
  const title = node('h1', undefined, 'voice'); title.append(node('span', line), node('span', 'Я здесь, чтобы выслушать.'));
  box.append(title, node('p', 'Напиши, что на душе, — можно коротко и сбивчиво. Или начни с одной из тем.'));
  for (const text of ['Мне тревожно', 'Не могу уснуть', 'Устала, нет сил', 'Хочу просто выговориться']) {
    const b = node('button', text, 'chip'); b.type = 'button';
    b.onclick = () => { const field = $('#message'); field.value = text + '. '; autosize(); field.focus(); };
    row.append(b);
  }
  box.append(row); return box;
}
function richText(text, host, book) {
  // Абзацы и телефоны из утверждённого списка; содержимое вставляется только как текст.
  const pattern = phonePattern(book), found = new Set();
  for (const part of text.split(/\n{2,}/)) {
    const p = node('p'); let last = 0;
    if (pattern) for (const match of part.matchAll(pattern)) {
      const digits = match[0].replace(/\D/g, ''), a = node('a', match[0], 'tel');
      a.href = 'tel:' + digits; found.add(digits);
      p.append(part.slice(last, match.index), a); last = match.index + match[0].length;
    }
    p.append(part.slice(last)); host.append(p);
  }
  return found;
}
function messageNode(m, book) {
  const mine = m.role === 'user', e = node('article', undefined, 'msg ' + (mine ? 'msg--user' : 'msg--emo'));
  if (mine) e.append(node('p', m.content, 'msg__text'));
  else {
    const body = node('div', undefined, 'msg__text'), found = richText(m.content, body, book); e.append(body);
    if (m.origin === 'safety_template') {
      const numbers = book.filter(p => found.has(p.digits));
      e.classList.add(numbers.length ? 'msg--safety' : 'msg--template');
      if (numbers.length) {
        const row = node('div', undefined, 'calls');
        for (const p of numbers) { const a = node('a', undefined, 'call'); a.href = 'tel:' + p.digits; a.title = p.label; a.append(icon('phone'), node('span', 'Позвонить ' + p.show)); row.append(a); }
        e.append(row);
      }
    }
  }
  const meta = node('footer', undefined, 'msg__meta'), time = node('time', clock(m.created_at));
  time.dateTime = m.created_at; time.title = localDate(m.created_at);
  if (m.origin === 'proactive') meta.append(node('span', 'EMO написала первой'));
  if (mine && m.metadata?.voice) meta.append(node('span', 'Голосом · ' + (m.metadata.voice.backend === 'gigaam' ? 'GigaAM' : 'Deepgram')));
  meta.append(time); e.append(meta);
  diagnostics.renderMessage(m, e);
  return e;
}
async function loadConversations() {
  const data = await api('/conversations?limit=100'), select = $('#conversations'); select.replaceChildren();
  for (const c of data.items) { const option = node('option', c.title || localDate(c.created_at)); option.value = c.id; select.append(option); }
  if (!data.items.length) { await newConversation(); return; }
  conversationId = data.items[0].id; await loadMessages();
}
async function newConversation() {
  voice.reset();
  pendingMessage = null;
  const c = await post('/conversations', {title: 'Разговор ' + new Date().toLocaleString('ru-RU', {day: 'numeric', month: 'long', hour: '2-digit', minute: '2-digit'})});
  conversationId = c.id; const option = node('option', c.title); option.value = c.id;
  $('#conversations').prepend(option); $('#conversations').value = c.id; $('#feedback').hidden = true;
  await loadMessages(); $('#message').focus();
}
async function loadMessages() {
  if (!conversationId) return;
  const cid = conversationId, epoch = profileEpoch, request = ++messageLoad, wasSending = sending, box = $('#messages');
  const known = box.dataset.conversation === cid && box.dataset.signature ? box.dataset.version : '';
  const data = await api(`/conversations/${cid}/messages?limit=100${known ? '&known_version=' + known : ''}`);
  if (epoch !== profileEpoch || cid !== conversationId || request !== messageLoad || (sending && !wasSending)) return;
  if (data.unchanged) return;
  const signature = cid + ':' + data.items.map(m => m.id).join(',');
  if (box.dataset.signature === signature && box.childNodes.length) return;
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80, scrollTop = box.scrollTop;
  const expanded = new Set([...box.querySelectorAll('.decision-report[open]')].map(e => e.dataset.traceId));
  const book = phoneBook(); let day = '';
  box.replaceChildren(); box.dataset.signature = signature;
  box.dataset.conversation = cid; box.dataset.version = data.version;
  for (const m of data.items.reverse()) {
    const label = dayLabel(m.created_at);
    if (label !== day) { box.append(node('div', label, 'day')); day = label; }
    const e = messageNode(m, book), report = e.querySelector('.decision-report');
    if (report && expanded.has(report.dataset.traceId)) report.open = true;
    box.append(e); if (m.role === 'assistant') lastAnswer = m.id;
  }
  if (!data.items.length) box.append(greeting());
  box.scrollTop = atBottom ? box.scrollHeight : scrollTop;
  await post(`/conversations/${cid}/read`, {});
}
function autosize() { const field = $('#message'); field.style.height = 'auto'; field.style.height = Math.min(field.scrollHeight, innerHeight * .38) + 'px'; }
$('#message').oninput = autosize;
$('#message').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('#chat-form').requestSubmit(); } };
// Пустой разговор уже открыт — второй такой же не нужен.
$('#new-chat').onclick = () => safe(async () => { if ($('#messages .hello')) { $('#message').focus(); return; } await newConversation(); });
$('#conversations').onchange = () => safe(async () => { voice.reset(); pendingMessage = null; conversationId = $('#conversations').value; $('#feedback').hidden = true; await loadMessages(); $('#messages').scrollTop = $('#messages').scrollHeight; });
async function deliverMessage(label, request) {
  if (sending) throw Error('Предыдущее сообщение ещё отправляется.');
  const epoch = profileEpoch, cid = conversationId, box = $('#messages');
  sending = true; voice.setBusy(true);
  for (const control of [$('#text-send'), $('#message'), $('#conversations'), $('#new-chat')]) control.disabled = true;
  $('#feedback').hidden = true; notice('');
  const mine = node('article', undefined, 'msg msg--user msg--pending'), typing = node('div', 'EMO готовит ответ…', 'msg msg--emo msg--typing');
  mine.append(node('p', label, 'msg__text')); typing.setAttribute('role', 'status');
  box.querySelector('.hello')?.remove(); box.append(mine, typing); box.scrollTop = box.scrollHeight;
  try {
    const result = await request();
    if (epoch !== profileEpoch || cid !== conversationId) return result;
    lastAnswer = result.assistant_message.id; box.dataset.signature = '';
    await loadMessages(); box.scrollTop = box.scrollHeight;
    $('#feedback').hidden = result.assistant_message.origin === 'safety_template';
    return result;
  } catch (error) {
    if (epoch === profileEpoch && cid === conversationId) { mine.remove(); typing.remove(); if (!box.children.length) box.append(greeting()); }
    throw error;
  } finally {
    if (epoch === profileEpoch) {
      sending = false; voice.setBusy(false);
      for (const control of [$('#text-send'), $('#message'), $('#conversations'), $('#new-chat')]) control.disabled = false;
      $('#message').focus();
    }
  }
}
$('#chat-form').onsubmit = e => { e.preventDefault(); safe(async () => {
  if (!token) throw Error('Сначала войди в профиль.');
  const field = $('#message'), text = field.value.trim(), epoch = profileEpoch, cid = conversationId;
  if (!text || sending) return;
  if (voice.active) throw Error('Сначала останови или удали голосовую запись.');
  if (voice.hasDraft) throw Error('Сначала отправь или удали голосовую запись.');
  // Повтор того же текста после сбоя уходит с прежним идентификатором: сервер не создаст дубль.
  if (!pendingMessage || pendingMessage.content !== text) pendingMessage = {content: text, client_message_id: crypto.randomUUID()};
  const payload = pendingMessage; field.value = ''; autosize();
  try {
    await deliverMessage(text, () => post(`/conversations/${cid}/messages`, payload));
    if (epoch === profileEpoch) pendingMessage = null;
  } catch (error) {
    if (epoch === profileEpoch) { field.value = text; autosize(); }
    throw error;
  }
}); };
$('#feedback').onclick = e => { if (e.target.dataset.rating) safe(async () => { await post('/feedback', {message_id: lastAnswer, rating: e.target.dataset.rating}); $('#feedback').hidden = true; notice(e.target.dataset.rating === 'skip' ? 'Хорошо, без оценки.' : 'Спасибо, отметила.'); }); };

/* ---------- Самочувствие ---------- */
for (const input of document.querySelectorAll('input[type=range]')) {
  const sync = () => { const out = $(`output[for="${input.id}"]`); if (out) out.value = input.value; input.style.setProperty('--fill', (input.value - input.min) / (input.max - input.min) * 100 + '%'); };
  input.oninput = sync; sync();
}
$('#skip-sleep').onchange = e => { $('#ci-sleep').disabled = e.target.checked; $('#ci-sleep').closest('.scale').classList.toggle('scale--off', e.target.checked); };
async function checkin() {
  const last = (await api('/checkins?limit=1')).items[0];
  $('#last-checkin').textContent = last
    ? `Прошлая отметка — ${dayLabel(last.created_at).toLowerCase()}, ${clock(last.created_at)}: настроение ${last.mood_score}, тревога ${last.anxiety_score}${last.sleep_quality != null ? ', сон ' + last.sleep_quality : ''}.`
    : 'Это будет твоя первая отметка.';
}
$('#checkin-form').onsubmit = e => { e.preventDefault(); safe(async () => {
  const d = new FormData(e.target);
  await post('/checkins', {mood_score: +d.get('mood_score'), anxiety_score: +d.get('anxiety_score'), sleep_quality: $('#skip-sleep').checked ? null : +$('#ci-sleep').value});
  await checkin(); notice('Самочувствие сохранено. Перемены видны в разделе «Динамика».');
}); };

/* ---------- Динамика ---------- */
async function questionnaires() { await screenings.load(); }
function svgNode(tag, attrs = {}, cls) { const e = document.createElementNS(SVG_NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (cls) e.setAttribute('class', cls); return e; }
function trendChart(points, key, title, delta) {
  const values = points.filter(p => p[key] != null), figure = node('figure', undefined, 'trend'), head = node('div', undefined, 'trend__head');
  head.append(node('h3', title)); figure.append(head);
  if (!values.length) { figure.append(node('p', 'Пока нет отметок.', 'trend__note')); return figure; }
  const last = values.at(-1), now = node('div', undefined, 'trend__now');
  now.append(node('strong', String(last[key])), document.createTextNode(
    delta == null ? 'последняя отметка' : delta === 0 ? 'как в прошлый раз' : `на ${Math.abs(delta)} ${delta > 0 ? 'выше' : 'ниже'}, чем в прошлый раз`));
  head.append(now);
  // Одна шкала 0–10 на всех трёх графиках: их можно сравнивать между собой.
  const W = 640, H = 150, L = 30, R = 626, T = 12, B = 116, start = +new Date(values[0].at), end = +new Date(last.at);
  const x = p => end === start ? (L + R) / 2 : L + (+new Date(p.at) - start) / (end - start) * (R - L), y = v => B - v / 10 * (B - T);
  const coords = values.map(p => [x(p), y(p[key])]);
  const svg = svgNode('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${title}: ${values.length} отметок с ${localDate(values[0].at)} по ${localDate(last.at)}, последняя ${last[key]} из 10`});
  for (const tick of [0, 5, 10]) {
    svg.append(svgNode('line', {x1: L, x2: R, y1: y(tick), y2: y(tick)}, 'trend__grid'));
    const label = svgNode('text', {x: L - 8, y: y(tick) + 4, 'text-anchor': 'end'}, 'trend__axis'); label.textContent = tick; svg.append(label);
  }
  const from = svgNode('text', {x: L, y: H - 8}, 'trend__axis'); from.textContent = dayLabel(values[0].at); svg.append(from);
  if (values.length > 1) {
    const to = svgNode('text', {x: R, y: H - 8, 'text-anchor': 'end'}, 'trend__axis'); to.textContent = dayLabel(last.at); svg.append(to);
    svg.append(svgNode('polygon', {points: `${coords[0][0]},${B} ${coords.map(c => c.join(',')).join(' ')} ${coords.at(-1)[0]},${B}`}, 'trend__area'));
    svg.append(svgNode('polyline', {points: coords.map(c => c.join(',')).join(' ')}, 'trend__line'));
  }
  if (values.length <= 14) coords.slice(0, -1).forEach(([cx, cy]) => svg.append(svgNode('circle', {cx, cy, r: 4}, 'trend__dot')));
  svg.append(svgNode('circle', {cx: coords.at(-1)[0], cy: coords.at(-1)[1], r: 5}, 'trend__dot'));
  const cross = svgNode('line', {y1: T, y2: B}, 'trend__cross'), focus = svgNode('circle', {r: 6}, 'trend__dot');
  cross.setAttribute('visibility', 'hidden'); focus.setAttribute('visibility', 'hidden'); svg.append(cross, focus);
  const plot = node('div', undefined, 'trend__plot'), tip = node('div', undefined, 'trend__tip'); tip.hidden = true; plot.tabIndex = 0;
  plot.setAttribute('aria-label', `${title}. Стрелки влево и вправо показывают отметки по очереди.`);
  let index = values.length - 1;
  const pick = i => {
    index = Math.max(0, Math.min(values.length - 1, i)); const [cx, cy] = coords[index], scale = plot.clientWidth / W;
    cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); focus.setAttribute('cx', cx); focus.setAttribute('cy', cy);
    cross.setAttribute('visibility', 'visible'); focus.setAttribute('visibility', 'visible');
    tip.replaceChildren(node('strong', String(values[index][key])), document.createTextNode(localDate(values[index].at)));
    tip.hidden = false; tip.style.left = Math.max(70, Math.min(plot.clientWidth - 70, cx * scale)) + 'px'; tip.style.top = cy * scale - 12 + 'px';
  };
  const hide = () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); focus.setAttribute('visibility', 'hidden'); };
  plot.onpointermove = e => { const px = (e.clientX - plot.getBoundingClientRect().left) / plot.clientWidth * W; pick(coords.reduce((best, c, i) => Math.abs(c[0] - px) < Math.abs(coords[best][0] - px) ? i : best, 0)); };
  plot.onpointerleave = hide; plot.onblur = hide; plot.onfocus = () => pick(index);
  plot.onkeydown = e => { if (e.key === 'ArrowLeft') { e.preventDefault(); pick(index - 1); } if (e.key === 'ArrowRight') { e.preventDefault(); pick(index + 1); } };
  plot.append(svg, tip); figure.append(plot);
  if (values.length === 1) figure.append(node('p', 'Линия появится после второй отметки.', 'trend__note'));
  return figure;
}
async function history(append = false) {
  const t = await api('/trends'), box = $('#trends');
  if (!append) {
    const points = t.checkins.points, deltas = t.checkins.sufficient_data ? t.checkins.deltas : {};
    box.replaceChildren();
    const toCheckin = () => safe(async () => { show('checkin'); await refresh(); });
    if (!points.length) box.append(emptyState('Здесь появятся линии настроения, тревоги и сна — после первой отметки самочувствия.', 'Отметить самочувствие', toCheckin));
    else if (points.length === 1) {
      const p = points[0];
      box.append(emptyState(`Пока одна отметка — ${dayLabel(p.at).toLowerCase()}: настроение ${p.mood}, тревога ${p.anxiety}${p.sleep != null ? ', сон ' + p.sleep : ''}. Линии появятся после второй.`, 'Отметить самочувствие', toCheckin));
    } else {
      box.append(trendChart(points, 'mood', 'Настроение', deltas.mood_score), trendChart(points, 'anxiety', 'Тревога', deltas.anxiety_score), trendChart(points, 'sleep', 'Сон', deltas.sleep_quality));
      const details = node('details'); details.append(node('summary', 'Показать отметки таблицей'), table(['Когда', 'Настроение', 'Тревога', 'Сон'], points.map(p => [localDate(p.at), p.mood, p.anxiety, p.sleep])));
      box.append(details);
    }
    // Все шкалы в одной таблице: у каждой своя сумма баллов, сравнивать их между собой нельзя.
    const scores = Object.entries(t.questionnaires).flatMap(([code, list]) => list.map(p => ({code, ...p}))).sort((a, b) => new Date(b.at) - new Date(a.at));
    if (scores.length) box.append(node('h2', 'Результаты опросников'), table(['Опросник', 'Когда', 'Баллы', 'К прошлому разу'], scores.map(p => [p.code, localDate(p.at), p.score, p.delta == null ? '—' : p.delta === 0 ? 'без изменений' : (p.delta > 0 ? '+' : '−') + Math.abs(p.delta)])), node('p', 'У каждой шкалы своя сумма баллов: сравнивать их между собой не нужно. Это самооценка, а не диагноз.', 'hint'));
    attemptOffset = 0; $('#attempts').replaceChildren();
  }
  const a = await api('/questionnaire-attempts?limit=30&offset=' + attemptOffset);
  if (!a.items.length && !attemptOffset) $('#attempts').append(node('p', 'Ты ещё не проходила опросники.', 'hint'));
  else $('#attempts').append(screenings.historyLinks(a.items));
  attemptOffset += a.items.length; $('#more-attempts').hidden = a.items.length < 30;
}
$('#more-attempts').onclick = () => safe(() => history(true));

/* ---------- Журнал ---------- */
function meter(label, value) {
  const row = node('div', undefined, 'meter'), bar = node('span', undefined, 'meter__bar'), fill = node('span', undefined, 'meter__fill');
  fill.style.width = value * 10 + '%'; bar.append(fill);
  row.append(node('span', label, 'meter__label'), bar, node('span', value == null ? '—' : String(value), 'meter__value'));
  if (value == null) row.classList.add('meter--empty');
  return row;
}
function journalEntry(item) {
  const e = node('article', undefined, 'entry entry--' + item.kind), body = node('div', undefined, 'entry__body'), head = node('div', undefined, 'entry__head');
  const time = node('time', clock(item.at)); time.dateTime = item.at; time.title = localDate(item.at);
  let glyph = 'heart', title = '';
  if (item.kind === 'checkin') {
    title = 'Отметка самочувствия';
    const meters = node('div', undefined, 'meters'); meters.append(meter('Настроение', item.mood), meter('Тревога', item.anxiety), meter('Сон', item.sleep)); body.append(meters);
  } else if (item.kind === 'signals') {
    glyph = 'eye'; title = item.source === 'user_report' ? 'Ты уточнила сама' : 'EMO заметила в сообщении';
    const chips = node('div', undefined, 'entry__chips');
    for (const s of item.signals) {
      const chip = node('span', signalNames[s.code] || s.code, 'tag');
      if (s.confidence != null) chip.append(node('span', ' ' + Number(s.confidence).toFixed(2), 'dev-only tag__num'));
      chips.append(chip);
    }
    body.append(chips, node('p', `Код: ${item.signals.map(s => s.code).join(', ')}. Источник: ${item.source}${item.model_version ? ', ' + item.model_version : ''}.`, 'hint dev-only'));
  } else if (item.kind === 'questionnaire') {
    glyph = 'forms'; title = `Опросник ${item.code}: ${item.score} из ${item.maximum ?? '—'}`;
    const open = node('button', 'Открыть результат', 'link'); open.type = 'button'; open.onclick = () => safe(() => screenings.open(item.attempt_id)); body.append(open);
    if (item.safety_flag) body.append(node('p', 'В ответах был пункт, о котором стоит поговорить со специалистом.', 'hint'));
  } else {
    glyph = 'shield';
    title = item.route === 'urgent' ? 'Срочный ответ: показаны номера помощи' : item.route === 'recommend_professional' ? 'Совет обратиться к специалисту' : 'Служебная проверка ответа';
    if (item.route === 'normal') e.classList.add('dev-only');
    body.append(node('p', `Маршрут: ${item.route}. Риск: ${item.risk}. Действие: ${item.action}.`, 'hint dev-only'));
  }
  const mark = node('span', undefined, 'entry__mark'); mark.append(icon(glyph));
  head.append(node('strong', title), time); body.prepend(head); e.append(mark, body); return e;
}
function renderJournal() {
  const filters = $('#journal-filters'), list = $('#journal-list'), d = journalData; filters.replaceChildren(); list.replaceChildren();
  if (!d) return;
  for (const [kind, label] of Object.entries(journalKinds)) {
    const count = kind === 'all' ? d.items.length : d.counts[kind] || 0, b = node('button', label, 'chip'); b.type = 'button';
    b.append(node('span', String(count), 'chip__count')); b.setAttribute('aria-pressed', String(journalFilter === kind));
    b.onclick = () => { journalFilter = kind; renderJournal(); }; filters.append(b);
  }
  const items = d.items.filter(i => journalFilter === 'all' || i.kind === journalFilter);
  if (!items.length) {
    list.append(d.items.length ? emptyState('За этот период таких записей нет. Попробуй другой фильтр или период.')
      : emptyState('Журнал пока пуст. Записи появятся после отметки самочувствия, опросника или разговора.', 'Отметить самочувствие', () => safe(async () => { show('checkin'); await refresh(); })));
    return;
  }
  let day = '', group = null;
  for (const item of items) {
    const label = dayLabel(item.at);
    if (label !== day) { day = label; group = node('section', undefined, 'day-group'); group.append(node('h2', label)); list.append(group); }
    group.append(journalEntry(item));
  }
  if (d.truncated) list.append(node('p', 'Показаны последние записи. Чтобы увидеть меньше, сузь период.', 'hint'));
}
async function journal() { journalData = await api('/journal?days=' + $('#journal-days').value); renderJournal(); }
$('#journal-days').onchange = () => safe(journal);

/* ---------- Что беспокоит ---------- */
async function complaints() {
  const d = await api('/complaints'), box = $('#complaint-list'); box.replaceChildren();
  if (!d.items.length) { box.append(emptyState('Пока пусто. Темы появятся здесь, когда EMO заметит их в разговоре.', 'Перейти к разговору', () => show('chat'))); return; }
  for (const c of d.items) {
    const row = node('div', undefined, 'row'), main = node('div'), tags = node('div', undefined, 'row__tags');
    main.append(node('strong', names[c.category] || c.category), node('p', `Впервые — ${dayLabel(c.first_seen_at).toLowerCase()}, в последний раз — ${dayLabel(c.last_seen_at).toLowerCase()}.`, 'hint'));
    tags.append(node('span', complaintStatus[c.status] || c.status, 'tag'), node('span', c.user_confirmed ? 'Ты подтвердила' : 'Предположение EMO', 'tag tag--quiet'));
    row.append(main, tags); box.append(row);
  }
}

/* ---------- Настройки ---------- */
function isoDay(date) { return new Date(date.getTime() - date.getTimezoneOffset() * 6e4).toISOString().slice(0, 10); }
function syncStage() {
  const f = $('#perinatal-form').elements, stage = f.stage.value, due = f.estimated_due_date.value;
  $('#due-label').hidden = stage !== 'pregnant'; $('#birth-label').hidden = stage !== 'postpartum';
  let note = '';
  if (stage === 'pregnant' && due) { const week = Math.floor((280 - Math.round((new Date(due + 'T12:00:00') - Date.now()) / 864e5)) / 7); if (week >= 0 && week <= 43) note = `Сейчас примерно ${week}-я неделя. EMO сама пересчитывает срок с течением времени.`; }
  if (stage === 'loss') note = 'Мне очень жаль. EMO учтёт это: не будет спрашивать о сроке и говорить о родах как о предстоящих.';
  if (stage === 'none') note = 'EMO не будет опираться на срок беременности или возраст ребёнка.';
  if (stage === 'preconception') note = 'Для подготовки доступны PHQ-9 и GAD-7. Дополнительные шкалы появятся на соответствующем этапе.';
  $('#stage-note').textContent = note;
}
function fillPerinatal(d) {
  const f = $('#perinatal-form').elements, today = new Date();
  f.stage.value = d.reproductive_stage === 'preconception' ? 'preconception' : d.reproductive_stage === 'unknown' ? 'none' : d.pregnancy_outcome === 'loss' ? 'loss' : d.birth_date ? 'postpartum' : d.estimated_due_date || d.gestational_week != null ? 'pregnant' : 'none';
  f.estimated_due_date.value = d.estimated_due_date || ''; f.birth_date.value = d.birth_date ? isoDay(new Date(d.birth_date)) : '';
  f.estimated_due_date.min = isoDay(new Date(+today - 30 * 864e5)); f.estimated_due_date.max = isoDay(new Date(+today + 310 * 864e5)); f.birth_date.max = isoDay(today);
  syncStage();
  if (f.stage.value === 'pregnant' && !d.estimated_due_date) $('#stage-note').textContent = `Срок передан основным приложением: ${d.gestational_week}-я неделя на момент записи. Дата родов позволит считать его точнее.`;
}
$('#perinatal-form').onchange = syncStage;
$('#perinatal-form').onsubmit = e => { e.preventDefault(); safe(async () => {
  const f = e.target.elements, stage = f.stage.value, body = {reproductive_stage: stage === 'none' ? 'unknown' : stage === 'loss' ? 'after_loss' : stage, estimated_due_date: null, birth_date: null, pregnancy_outcome: null};
  if (stage === 'pregnant') { if (!f.estimated_due_date.value) throw Error('Укажи предполагаемую дату родов.'); body.estimated_due_date = f.estimated_due_date.value; body.pregnancy_outcome = 'ongoing'; }
  else body.gestational_week = null;
  if (stage === 'postpartum') { if (!f.birth_date.value) throw Error('Укажи дату родов.'); body.birth_date = new Date(f.birth_date.value + 'T00:00:00').toISOString(); body.pregnancy_outcome = 'live_birth'; }
  if (stage === 'loss') body.pregnancy_outcome = 'loss';
  fillPerinatal(await api('/settings', {method: 'PATCH', body: JSON.stringify(body)})); notice('Сохранено.');
}); };
async function settings() {
  const d = await api('/settings'), f = $('#settings-form').elements;
  for (const key of ['proactive_enabled', 'push_enabled']) f[key].checked = d[key];
  for (const key of ['timezone', 'quiet_hours_start', 'quiet_hours_end']) f[key].value = d[key];
  const m = await api('/state'); $('#memory-summary').value = m.summary; memoryPreferences = m.preferences;
  fillPerinatal(d); fillAi(d);
}
function fillAi(d) {
  const e = $('#ai-form').elements, l = d.limits, k = v => Math.round(v / 1000);
  e.external_ai_consent.checked = d.external_ai_consent;
  e.voice_enabled.checked = d.voice.enabled;
  e.audio_consent.checked = d.voice.external_audio_consent;
  e.anonymize.checked = l.anonymize_policy === 'always' ? true : l.anonymize_policy === 'never' ? false : d.privacy.anonymize;
  e.anonymize.disabled = l.anonymize_policy !== 'user';
  $('#anonymize-policy').textContent = l.anonymize_policy === 'user' ? '' : 'Этот режим задан оператором сервера и здесь не меняется.';
  e.level.value = d.privacy.level; e.protected_terms.value = d.privacy.protected_terms.join('\n');
  $('#learned-names').textContent = d.learned_names.length ? `EMO запомнила имён из разговоров: ${d.learned_names.length}. Они скрываются всегда.` : 'Имён из разговоров пока не запомнено.';
  $('#forget-names').hidden = !d.learned_names.length;
  const c = e.compression; c.replaceChildren();
  for (const [value, label] of Object.entries(l.compression_modes)) { const o = node('option', label); o.value = value; c.append(o); }
  c.value = d.context.compression; e.input_target.value = k(d.context.input_target_tokens); e.input_target.max = Math.floor(l.hard_input_budget_tokens / 1000);
  e.keep_recent.value = d.context.keep_recent_messages;
  $('#context-limits').textContent = `Окно модели — ${k(l.context_window_tokens)} тыс. токенов: до ${k(l.hard_input_budget_tokens)} тыс. на разговор, ${k(l.safety_margin_tokens)} тыс. в резерве, ответ до ${l.max_output_tokens} токенов. Значение 0 в поле «Сжимать после» — использовать всё окно.`;
}
$('#ai-form').onsubmit = e => { e.preventDefault(); safe(async () => {
  const f = e.target.elements, terms = f.protected_terms.value.split(/[\n,;]/).map(v => v.trim()).filter(Boolean);
  const d = await api('/settings', {method: 'PATCH', body: JSON.stringify({
    external_ai_consent: f.external_ai_consent.checked,
    voice: {enabled: f.voice_enabled.checked, external_audio_consent: f.audio_consent.checked},
    privacy: {anonymize: f.anonymize.checked, level: f.level.value, protected_terms: terms},
    context: {compression: f.compression.value, input_target_tokens: Math.max(0, Math.round(+f.input_target.value || 0)) * 1000, keep_recent_messages: Math.max(2, Math.round(+f.keep_recent.value || 12))},
  })});
  fillAi(d); await voice.refresh(); notice('Сохранено.');
}); };
$('#privacy-preview').onclick = () => safe(async () => {
  const text = $('#privacy-sample').value.trim(), out = $('#privacy-result'); if (!text) { $('#privacy-sample').focus(); return; }
  const r = await post('/privacy/preview', {text}); out.replaceChildren();
  if (!r.enabled) { out.append(node('p', 'Скрытие выключено: текст уйдёт модели как есть.')); return; }
  const shown = node('div', undefined, 'preview__text'); let last = 0;
  for (const match of r.text.matchAll(/\[[А-ЯЁ]+_\d+(?:\.\d+)?\]/g)) { shown.append(r.text.slice(last, match.index), node('span', match[0], 'token')); last = match.index + match[0].length; }
  shown.append(r.text.slice(last));
  const replaced = Object.entries(r.entities).map(([kind, count]) => `${entityNames[kind] || kind} — ${count}`).join(', ');
  out.append(shown, node('p', replaced ? `Заменено: ${replaced}.` : 'Личных данных не найдено: текст уйдёт без изменений.', 'hint'));
});
$('#forget-names').onclick = () => safe(async () => { await api('/privacy/learned-names', {method: 'DELETE'}); await settings(); notice('Запомненные имена удалены.'); });
$('#settings-form').onsubmit = e => { e.preventDefault(); safe(async () => {
  const f = e.target;
  await api('/settings', {method: 'PATCH', body: JSON.stringify({proactive_enabled: f.proactive_enabled.checked, push_enabled: f.push_enabled.checked, timezone: f.timezone.value, quiet_hours_start: f.quiet_hours_start.value, quiet_hours_end: f.quiet_hours_end.value})});
  notice('Сохранено.');
}); };
$('#memory-form').onsubmit = e => { e.preventDefault(); safe(async () => { await api('/memory', {method: 'PATCH', body: JSON.stringify({summary: $('#memory-summary').value, preferences: memoryPreferences})}); notice('Исправления сохранены.'); }); };
$('#clear-memory').onclick = () => safe(async () => {
  if (!await confirmDialog('Стереть память?', 'EMO забудет заметку о тебе. История разговоров хранится отдельно и останется.', 'Стереть память')) return;
  await api('/memory', {method: 'DELETE'}); await settings(); notice('Память стёрта. История разговоров осталась.');
});
$('#followup-form').onsubmit = e => { e.preventDefault(); safe(async () => {
  if (!conversationId) throw Error('Сначала начни разговор.');
  await post('/proactive/followup', {conversation_id: conversationId, scheduled_for: new Date($('#followup-at').value).toISOString(), consent: $('#followup-consent').checked, client_request_id: crypto.randomUUID()});
  e.target.reset(); notice('Запланировано. EMO напишет, если это не попадёт на тихие часы и не превысит лимит.');
}); };
$('#run-worker').onclick = () => safe(async () => { const value = $('#worker-now').value; const d = await post('/dev/worker/run', {now: value ? new Date(value).toISOString() : null}); notice(`Планировщик выполнен. Отправлено сообщений: ${d.sent}.`); await loadMessages(); });
$('#delete-data').onclick = () => safe(async () => {
  if (!await confirmDialog('Удалить все данные?', 'Разговоры, память и результаты этого профиля будут удалены, профиль — отключён. Отменить это нельзя.', 'Удалить мои данные')) return;
  await api('/data', {method: 'DELETE'}); token = ''; profileEpoch++; voice.reset(); voice.setBusy(false); $('#messages').replaceChildren(); $('#memory-summary').value = ''; notice('Данные профиля удалены.');
});

/* ---------- Обновление и запуск ---------- */
async function refresh() {
  const view = [...document.querySelectorAll('.view')].find(v => !v.hidden)?.id;
  if (view === 'checkin') await checkin();
  if (view === 'questionnaires') await questionnaires();
  if (view === 'history') await history();
  if (view === 'journal') await journal();
  if (view === 'complaints') await complaints();
  if (view === 'settings') await settings();
  if (view === 'admin') await diagnostics.refreshAdmin();
  if (token) await diagnostics.showModels();
}
voice.init({context: () => ({ready: !!token && !!conversationId, busy: sending, epoch: profileEpoch, cid: conversationId}), api, deliver: deliverMessage, notice, safe,
  openSettings: () => { show('settings'); safe(settings); $('#set-voice').scrollIntoView({block: 'center'}); }});
setInterval(() => { if (token && !sending && !$('#chat').hidden) safe(async () => { await loadMessages(); await diagnostics.showModels(); await voice.refresh(); }); }, 15000);
safe(async () => {
  const c = await (await fetch('/demo/config')).json();
  $('#mode').textContent = c.ai_mode === 'mock' ? 'Режим: шаблонная модель' : c.ai_mode === 'disabled' ? 'Режим: AI отключён' : 'Режим: внешние модели';
  $('#dev-tools').hidden = !c.demo_auth; loginRequired = !!c.login_required;
  if (!c.demo_auth) {
    $('#login').hidden = true; $('.profile').hidden = true; $('#copy-key').hidden = true;
    notice('Демо-вход отключён. В production используется JWT основного приложения.');
  }
  if (loginRequired) { $('#gate').hidden = false; $('#gate-login').focus(); }
  else if (c.demo_auth) await login();
});
