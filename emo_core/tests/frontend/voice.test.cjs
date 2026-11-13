// Browser recording state-machine tests; no microphone, network or patient recordings.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../../demo/voice.js'), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));

function setup({backend = 'deepgram', consent = true, failOnce = false, getUserMedia} = {}) {
  const elements = new Map(), errors = [], requests = [], stopped = [], urls = [];
  const state = {ready: true, busy: false, epoch: 1, cid: 'conversation-a'};
  const caps = {enabled: true, backend, external_audio_consent: consent, user_ai_consent: consent, backends: {deepgram: true, gigaam: true}, behavioral_signals: true, max_seconds: 90, max_bytes: 8388608};
  let now = 1000;
  class FakeDate extends Date { static now() { return now; } }
  const el = name => {
    if (!elements.has(name)) elements.set(name, {value: '', checked: false, hidden: false, disabled: false,
      textContent: '', dataset: {}, paused: true, currentTime: 0, duration: Infinity,
      pause() { this.paused = true; this.onpause?.(); }, async play() { this.paused = false; this.onplay?.(); },
      setAttribute() {}, removeAttribute() {}});
    return elements.get(name);
  };
  const track = () => ({getTracks: () => [{stop: () => stopped.push(true)}]});
  class Recorder {
    static isTypeSupported() { return true; }
    constructor(stream, options) { this.mimeType = options.mimeType; this.state = 'inactive'; }
    start() { this.state = 'recording'; }
    stop() {
      this.state = 'inactive';
      queueMicrotask(() => {
        this.ondataavailable?.({data: new Blob(['audio'], {type: this.mimeType})});
        this.onstop?.();
      });
    }
  }
  const context = vm.createContext({
    document: {getElementById: id => el(id.replace('voice-', '')), addEventListener() {}},
    window: {isSecureContext: true, MediaRecorder: Recorder, addEventListener() {}},
    navigator: {mediaDevices: {getUserMedia: getUserMedia || (async () => track())}},
    MediaRecorder: Recorder, Blob, URLSearchParams, Date: FakeDate, crypto: require('node:crypto').webcrypto,
    URL: {createObjectURL: () => 'blob:test', revokeObjectURL: value => urls.push(value)},
    localStorage: {getItem() { return null; }, setItem() {}},
    setInterval: () => 1, clearInterval() {},
  });
  vm.runInContext(source + '\nglobalThis.client = voice;', context);
  const client = context.client;
  const hooks = {
    context: () => state,
    safe: async fn => { try { return await fn(); } catch (error) { errors.push(error.message); } },
    notice: text => errors.push(text),
    api: async (path, options) => {
      if (path === '/voice/config') return {...caps, backends: {...caps.backends}};
      requests.push({path, options});
      if (failOnce) { failOnce = false; throw Error('Network interrupted'); }
      return {user_message: {metadata: {voice: {emotion: {status: 'available'}}}}};
    },
    deliver: async (label, request) => {
      client.setBusy(true);
      try { return await request(); } finally { client.setBusy(false); }
    },
  };
  client.init(hooks);
  return {client, el, state, errors, requests, stopped, urls, track, caps, hooks, advance: value => {now += value;}};
}

async function record(fixture) {
  await fixture.client.refresh();
  await fixture.el('record').onclick();
  assert.equal(fixture.client.active, true);
  await fixture.el('record').onclick();
  await flush();
  assert.equal(fixture.client.active, false);
  assert.equal(fixture.el('draft').hidden, false);
}

test('a failed send retains the same blob and idempotency key for retry', async () => {
  const f = setup({failOnce: true});
  await record(f);
  assert.equal(f.stopped.length, 1);
  await f.el('send').onclick();
  assert.equal(f.el('draft').hidden, false);
  await f.el('send').onclick();
  assert.equal(f.requests.length, 2);
  assert.equal(f.requests[0].path, f.requests[1].path);
  assert.equal(f.requests[0].options.body, f.requests[1].options.body);
  assert.equal(f.requests[0].options.headers['Content-Type'], 'audio/webm;codecs=opus');
  assert.equal(f.el('draft').hidden, true);
  assert.equal(f.urls.length, 1);
});

test('reset after a profile change releases audio and cannot send the old draft', async () => {
  const f = setup();
  await record(f);
  f.state.epoch++; f.state.cid = 'conversation-b';
  f.client.reset();
  await f.el('send').onclick();
  assert.equal(f.requests.length, 0);
  assert.equal(f.el('draft').hidden, true);
  assert.equal(f.el('record').hidden, true); // previous profile's consent is not reused
});

test('a late microphone permission grant cannot attach to another profile', async () => {
  const pending = [];
  const f = setup({backend: 'gigaam', consent: false, getUserMedia: () => new Promise(resolve => pending.push(resolve))});
  await f.client.refresh();
  const oldStart = f.el('record').onclick();
  f.state.epoch++; f.client.reset(); await f.client.refresh();
  const newStart = f.el('record').onclick();
  pending[0](f.track()); await oldStart;
  assert.equal(f.client.active, true); // second permission request is still pending
  assert.equal(f.stopped.length, 1);
  pending[1](f.track()); await newStart;
  assert.equal(f.client.active, true);
  f.client.reset();
  assert.equal(f.stopped.length, 2);
});

test('Deepgram requires explicit audio consent before opening the microphone', async () => {
  const f = setup({consent: false});
  await f.client.refresh(); await f.el('record').onclick();
  assert.equal(f.client.active, false);
  assert.match(f.errors[0], /разреши внешнюю обработку/);
  assert.equal(f.requests.length, 0);
});

test('GigaAM can send a recording without external audio consent', async () => {
  const f = setup({backend: 'gigaam', consent: false});
  await record(f); await f.el('send').onclick();
  assert.equal(f.requests.length, 1);
  assert.match(f.requests[0].path, /backend=gigaam/);
  assert.match(f.requests[0].path, /external_audio_consent=false/);
});

const pointer = {button: 0, isPrimary: true, pointerId: 7, preventDefault() {}};
test('hold and release stops recording and makes a preview without sending', async () => {
  const f = setup(); await f.client.refresh();
  await f.el('record').onpointerdown(pointer);
  assert.equal(f.client.active, true);
  f.advance(1200); f.el('record').onpointerup(pointer); await flush();
  assert.equal(f.client.active, false); assert.equal(f.el('draft').hidden, false);
  assert.equal(f.requests.length, 0); assert.equal(f.stopped.length, 1);
  await f.el('play').onclick(); assert.equal(f.el('preview').paused, false);
  await f.el('play').onclick(); assert.equal(f.el('preview').paused, true);
});

test('release while microphone permission is pending never starts unattended recording', async () => {
  let resolve;
  const f = setup({getUserMedia: () => new Promise(done => {resolve = done;})});
  await f.client.refresh(); const opening = f.el('record').onpointerdown(pointer);
  f.advance(1200); f.el('record').onpointerup(pointer);
  resolve(f.track()); await opening;
  assert.equal(f.client.active, false); assert.equal(f.stopped.length, 1);
  assert.equal(f.el('draft').hidden, true);
});

test('short tap records hands-free and a second tap stops without a duplicate click', async () => {
  const f = setup(); await f.client.refresh(); await f.el('record').onpointerdown(pointer);
  f.advance(100); f.el('record').onpointerup(pointer);
  await f.el('record').onclick({detail: 1}); assert.equal(f.client.active, true);
  await f.el('record').onpointerdown(pointer); f.advance(100); f.el('record').onpointerup(pointer); await flush();
  assert.equal(f.client.active, false); assert.equal(f.el('draft').hidden, false);
});

test('disabling voice removes all controls and releases an active microphone', async () => {
  const f = setup(); await f.client.refresh(); await f.el('record').onclick();
  f.caps.enabled = false; await f.client.refresh();
  assert.equal(f.client.active, false); assert.equal(f.el('record').hidden, true);
  assert.equal(f.el('draft').hidden, true); assert.equal(f.stopped.length, 1);
  assert.equal(f.el('chat-form').dataset.voice, undefined);
});

test('pointer cancellation drops recording without an upload', async () => {
  const f = setup(); await f.client.refresh(); await f.el('record').onpointerdown(pointer);
  f.el('record').onpointercancel(); await flush();
  assert.equal(f.client.active, false); assert.equal(f.el('draft').hidden, true);
  assert.equal(f.requests.length, 0);
});

test('a transient configuration polling failure preserves the unsent recording', async () => {
  const f = setup(); await record(f);
  const original = f.hooks.api;
  f.hooks.api = async () => {throw Error('temporary connection failure');};
  await f.client.refresh();
  assert.equal(f.client.hasDraft, true); assert.equal(f.urls.length, 0);
  f.hooks.api = original; await f.el('send').onclick();
  assert.equal(f.requests.length, 1);
});

test('an access revocation clears the private draft', async () => {
  const f = setup(); await record(f);
  f.hooks.api = async () => {throw Object.assign(Error('profile deleted'), {status: 410});};
  await f.client.refresh();
  assert.equal(f.client.hasDraft, false); assert.equal(f.urls.length, 1);
});

test('an older configuration response cannot reenable a disabled widget', async () => {
  const f = setup(); let resolve, calls = 0;
  f.hooks.api = () => ++calls === 1 ? new Promise(done=>{resolve=done;}) : Promise.resolve({...f.caps, enabled:false});
  const old = f.client.refresh(); await f.client.refresh();
  resolve({...f.caps}); await old;
  assert.equal(f.el('record').hidden, true);
});
