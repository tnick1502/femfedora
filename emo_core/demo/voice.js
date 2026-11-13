'use strict';
const voice = (() => {
  const el = id => document.getElementById('voice-' + id);
  let hooks, config = null, recorder = null, stream = null, timer = null, chunks = [], bytes = 0;
  let draft = null, preview = '', serial = 0, started = 0, busy = false, opening = false;
  let press = null, stopAfterOpen = false, seconds = 0, recordingBackend = '';
  let configEpoch = null;
  let configRequest = 0;
  const backend = () => recordingBackend || config?.backend || 'deepgram';
  const limit = () => config?.backend_max_seconds?.[backend()] || config?.max_seconds || 90;
  const clock = value => `${Math.floor(Math.max(0, value) / 60)}:${String(Math.floor(Math.max(0, value) % 60)).padStart(2, '0')}`;
  function stopTracks() { stream?.getTracks().forEach(track => track.stop()); stream = null; clearInterval(timer); timer = null; }
  function releasePreview() {
    el('preview').pause(); el('preview').removeAttribute('src');
    if (preview) URL.revokeObjectURL(preview);
    preview = ''; el('play-icon').setAttribute('href', '#i-play');
  }
  function update() {
    if (!hooks) return;
    const context = hooks.context(), active = !!recorder || opening;
    const supported = !!navigator.mediaDevices?.getUserMedia && !!window.MediaRecorder && window.isSecureContext;
    const shown = !!config?.enabled && configEpoch === context.epoch && context.ready && supported;
    el('record').hidden = !shown || !!draft;
    el('record').disabled = busy || !config?.backends?.[backend()];
    el('record').setAttribute('aria-pressed', String(active));
    el('record').setAttribute('aria-label', active ? 'Остановить запись' : 'Записать голосовое сообщение');
    el('record').title = active ? 'Остановить запись' : `Удерживай для записи · до ${limit()} сек. Клик — запись без удержания`;
    el('record-icon').setAttribute('href', active ? '#i-stop' : '#i-mic');
    el('draft').hidden = !shown || !(active || draft);
    el('cancel').disabled = busy; el('send').hidden = !draft; el('send').disabled = busy;
    el('play').hidden = !draft; el('play').disabled = busy;
    el('wave').hidden = !active; el('progress').hidden = !draft;
    const form = document.getElementById('chat-form');
    if (shown && (active || draft)) form.dataset.voice = draft ? 'ready' : 'recording';
    else delete form.dataset.voice;
    document.getElementById('message').required = !(shown && (active || draft));
    if (!active) el('time').textContent = clock(draft?.seconds || 0);
  }
  function reset() {
    serial++; opening = false; stopAfterOpen = false; press = null;
    if (recorder && recorder.state !== 'inactive') { recorder.onstop = null; recorder.stop(); }
    recorder = null; stopTracks(); chunks = []; bytes = 0; draft = null; recordingBackend = '';
    releasePreview(); el('progress').value = 0; el('status').textContent = ''; update();
  }
  function stop() {
    if (opening && !recorder) { stopAfterOpen = true; return; }
    if (recorder?.state === 'recording') recorder.stop();
    stopTracks();
  }
  async function start() {
    const context = hooks.context();
    if (busy || context.busy || opening || recorder || draft) return;
    if (!context.ready || !config?.enabled || configEpoch !== context.epoch) return;
    if (!config.backends[config.backend]) throw Error('Выбранный распознаватель недоступен. Проверь раздел «Модели».');
    if (config.backend === 'deepgram' && !(config.external_audio_consent && config.user_ai_consent)) {
      hooks.openSettings?.(); throw Error('В настройках AI разреши внешнюю обработку сообщений и аудио.');
    }
    const mine = ++serial; opening = true; stopAfterOpen = false; recordingBackend = config.backend; update();
    el('status').textContent = 'Подключаем микрофон…';
    try {
      const acquired = await navigator.mediaDevices.getUserMedia({audio: true});
      if (mine !== serial || hooks.context().epoch !== context.epoch || !config?.enabled) { acquired.getTracks().forEach(t => t.stop()); return; }
      stream = acquired;
      if (stopAfterOpen) { reset(); return; }
      const mime = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg;codecs=opus'].find(type => MediaRecorder.isTypeSupported(type));
      recorder = new MediaRecorder(stream, mime ? {mimeType: mime} : undefined);
      const active = recorder; chunks = []; bytes = 0; seconds = 0; started = Date.now();
      active.ondataavailable = event => {
        if (mine !== serial || !event.data.size) return;
        bytes += event.data.size;
        if (bytes > config.max_bytes) { reset(); hooks.notice('Запись слишком большая. Запиши сообщение короче.', true); return; }
        chunks.push(event.data);
      };
      active.onerror = () => { if (mine === serial) { reset(); hooks.notice('Не удалось записать звук. Проверь микрофон и попробуй ещё раз.', true); } };
      active.onstop = () => {
        stopTracks(); if (mine !== serial) return;
        const blob = new Blob(chunks, {type: active.mimeType || mime || 'audio/webm'});
        recorder = null; chunks = []; press = null;
        if (!blob.size) { reset(); hooks.notice('Запись пустая. Попробуй ещё раз.', true); return; }
        draft = {blob, backend: recordingBackend, id: crypto.randomUUID(), epoch: context.epoch, cid: context.cid, seconds: (Date.now() - started) / 1000};
        preview = URL.createObjectURL(blob); el('preview').src = preview;
        el('status').textContent = 'Запись готова. Прослушай её перед отправкой.'; update();
      };
      active.start(1000); el('status').textContent = 'Идёт запись. Отпусти кнопку или нажми ещё раз, чтобы остановить.';
      const tick = () => {
        seconds = (Date.now() - started) / 1000; el('time').textContent = clock(seconds);
        if (seconds >= limit() - 1) stop();
      };
      timer = setInterval(tick, 200); tick();
    } catch (error) {
      if (mine !== serial) return;
      reset();
      throw Error(error.name === 'NotAllowedError' ? 'Разреши доступ к микрофону в настройках браузера.' : 'Микрофон недоступен. Проверь подключение и разрешения браузера.');
    } finally { if (mine === serial) { opening = false; update(); } }
  }
  async function send() {
    if (!draft || busy || hooks.context().busy) return;
    const item = draft, context = hooks.context();
    if (item.epoch !== context.epoch || item.cid !== context.cid) { reset(); return; }
    el('preview').pause(); el('status').textContent = 'Распознаём речь и готовим ответ…';
    const params = new URLSearchParams({client_message_id: item.id, backend: item.backend, external_audio_consent: String(!!config.external_audio_consent)});
    try {
      const result = await hooks.deliver('Голосовое сообщение · распознаём речь…', () => hooks.api(`/conversations/${item.cid}/voice-messages?${params}`, {method: 'POST', body: item.blob, headers: {'Content-Type': item.blob.type}}));
      if (item.epoch !== hooks.context().epoch) return;
      reset();
      if (result?.user_message?.metadata?.voice?.emotion?.status === 'unavailable') hooks.notice('Сообщение отправлено. Анализ интонации недоступен — ответ подготовлен по тексту.');
    } catch (error) {
      if (item.epoch === hooks.context().epoch) el('status').textContent = 'Отправка не завершена. Запись сохранена здесь; можно повторить.';
      throw error;
    }
  }
  return {
    init(options) {
      hooks = options;
      el('record').onclick = event => {
        if (event?.detail > 0) return;
        return hooks.safe(async () => recorder || opening ? stop() : start());
      };
      el('record').onpointerdown = event => {
        if (event.button !== 0 || !event.isPrimary || busy) return;
        event.preventDefault(); press = {id: event.pointerId, at: Date.now(), alreadyRecording: !!recorder || opening};
        el('record').setPointerCapture?.(event.pointerId);
        if (!recorder && !opening) return hooks.safe(start);
      };
      el('record').onpointerup = event => {
        if (!press || press.id !== event.pointerId) return;
        const held = Date.now() - press.at >= 300, wasRecording = press.alreadyRecording; press = null;
        if (held || wasRecording) stop();
      };
      el('record').onpointercancel = () => { if (press) reset(); };
      el('record').onlostpointercapture = () => { if (press) reset(); };
      el('cancel').onclick = reset; el('send').onclick = () => hooks.safe(send);
      el('play').onclick = () => hooks.safe(async () => {
        if (el('preview').paused) await el('preview').play(); else el('preview').pause();
      });
      el('preview').onplay = () => { el('play-icon').setAttribute('href', '#i-pause'); el('play').setAttribute('aria-label', 'Приостановить прослушивание'); };
      el('preview').onpause = el('preview').onended = () => { el('play-icon').setAttribute('href', '#i-play'); el('play').setAttribute('aria-label', 'Прослушать запись'); };
      el('preview').ontimeupdate = () => {
        const audio = el('preview'), duration = Number.isFinite(audio.duration) ? audio.duration : draft?.seconds || 1;
        el('progress').value = Math.min(1, audio.currentTime / Math.max(duration, .1)); el('time').textContent = clock(audio.currentTime);
      };
      document.addEventListener('keydown', event => { if (event.key === 'Escape' && (draft || recorder || opening) && !busy) reset(); });
      document.addEventListener('visibilitychange', () => { if (document.hidden && (recorder || opening)) stop(); });
      window.addEventListener('pagehide', reset); update();
    },
    async refresh() {
      if (!hooks.context().ready) return;
      const epoch = hooks.context().epoch, request = ++configRequest;
      try {
        const result = await hooks.api('/voice/config');
        if (epoch !== hooks.context().epoch || request !== configRequest) return;
        config = result; configEpoch = epoch; if (!config.enabled) reset();
      } catch (error) {
        if (epoch !== hooks.context().epoch || request !== configRequest) return;
        // A transient polling failure must not destroy a recorded, unsent message.
        // Actual sends are still authorized by the server against current preferences.
        if ([401,403,410].includes(error.status)) { config = null; reset(); }
      }
      update();
    },
    reset,
    setBusy(value) { busy = value; update(); },
    get active() { return !!recorder || opening; },
    get hasDraft() { return !!draft; },
  };
})();
