// Voice shopping agent — client.
// Tap-to-talk: tap the button (or press Space) to start a turn, speak, tap again
// to send. Plays back Gemini's 24 kHz PCM reply, shows running transcripts and
// ranked product cards, and reflects the agent's state in the status line.

const INPUT_RATE = 16000;
const OUTPUT_RATE = 24000;

const els = {
  mic: document.getElementById('mic'),
  status: document.getElementById('status'),
  transcript: document.getElementById('transcript'),
  results: document.getElementById('results'),
  resultsTitle: document.getElementById('results-title'),
  cards: document.getElementById('cards'),
};

let ws = null;
let micCtx = null;       // 16 kHz capture context
let workletNode = null;
let micStream = null;
let recording = false;
let speaking = false;    // true once the agent starts replying this turn
let reconnectAttempts = 0;

// Persisted across reconnects so conversation context survives an idle drop:
// `resumeHandle` resumes the Gemini Live session; `lastResults` is replayed so the
// server can rebuild its rank->product map.
let resumeHandle = sessionStorage.getItem('vs_resume_handle') || null;
let lastResults = JSON.parse(sessionStorage.getItem('vs_last_results') || 'null');

function setResumeHandle(h) {
  resumeHandle = h;
  try { sessionStorage.setItem('vs_resume_handle', h); } catch (e) {}
}
function persistResults() {
  try { sessionStorage.setItem('vs_last_results', JSON.stringify(lastResults)); } catch (e) {}
}

// --- Playback (24 kHz) ---------------------------------------------------
let playCtx = null;
let nextPlayTime = 0;

function ensurePlayCtx() {
  // Don't force a sample rate — iOS Safari ignores/mishandles a requested rate and
  // can produce a silent context. We declare 24 kHz on each buffer instead and let
  // the graph resample to whatever the hardware context runs at.
  if (!playCtx) playCtx = new (window.AudioContext || window.webkitAudioContext)();
  return playCtx;
}

// iOS unlock: a context starts "suspended" and only produces sound if resumed and
// primed with a buffer during a user gesture. Safe no-op elsewhere.
function unlockAudio() {
  const ctx = ensurePlayCtx();
  if (ctx.state === 'suspended') ctx.resume();
  const buf = ctx.createBuffer(1, 1, 22050);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start(0);
}

function playPCM(arrayBuffer) {
  const ctx = ensurePlayCtx();
  if (ctx.state === 'suspended') ctx.resume(); // iOS re-suspends aggressively
  const pcm = new Int16Array(arrayBuffer);
  const f32 = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 0x8000;

  const buffer = ctx.createBuffer(1, f32.length, OUTPUT_RATE);
  buffer.getChannelData(0).set(f32);
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);

  markSpeaking();
  const now = ctx.currentTime;
  if (nextPlayTime < now) nextPlayTime = now;
  src.start(nextPlayTime);
  nextPlayTime += buffer.duration;
}

function stopPlayback() {
  // On interruption (barge-in) reset the schedule so queued audio is dropped.
  if (playCtx) nextPlayTime = playCtx.currentTime;
}

// --- Transcript UI -------------------------------------------------------
let curUser = null;
let curAgent = null;

function appendTranscript(role, text) {
  let node = role === 'user' ? curUser : curAgent;
  if (!node) {
    node = document.createElement('div');
    node.className = `bubble ${role}`;
    node.innerHTML = `<span class="who">${role === 'user' ? 'You' : 'Agent'}</span><span class="msg"></span>`;
    els.transcript.appendChild(node);
    if (role === 'user') curUser = node; else curAgent = node;
  }
  node.querySelector('.msg').textContent += text;
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function endTurnUI() {
  curUser = null;
  curAgent = null;
}

// --- Product cards -------------------------------------------------------
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function showSearching(query, stores) {
  const where = stores && stores.length ? stores.join(' + ') : 'Amazon';
  els.results.hidden = false;
  els.resultsTitle.textContent = `Searching ${where} for “${query}”…`;
  els.cards.innerHTML = '<div class="searching">🔎 Finding and ranking products…</div>';
  els.results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderCards(query, cards) {
  lastResults = { query, cards };
  persistResults();
  els.results.hidden = false;
  els.resultsTitle.textContent = `Top picks for “${query}”`;
  els.cards.innerHTML = '';
  for (const c of cards) {
    const price = c.price && c.price !== 'N/A' ? `$${escapeHtml(c.price)}` : 'Price n/a';
    const rating = c.rating && c.rating !== 'N/A'
      ? `★ ${escapeHtml(c.rating)} <span class="muted">(${escapeHtml(c.reviews || '0')})</span>` : '';
    const card = document.createElement('article');
    card.className = 'card';
    card.dataset.rank = c.rank;
    card.dataset.store = c.store || 'product';
    const store = c.store ? `<span class="store store-${escapeHtml(c.source || '')}">${escapeHtml(c.store)}</span>` : '';
    card.innerHTML = `
      <div class="rank">#${escapeHtml(c.rank)}</div>
      ${c.image ? `<img class="thumb" src="${escapeHtml(c.image)}" alt="" loading="lazy">` : '<div class="thumb"></div>'}
      <div class="info">
        <a class="title" href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>
        <div class="meta"><span class="price">${price}</span>${rating ? `<span class="rating">${rating}</span>` : ''}${store}</div>
        ${c.why ? `<div class="why">${escapeHtml(c.why)}</div>` : ''}
      </div>`;
    els.cards.appendChild(card);
  }
}

function cardByRank(rank) {
  return els.cards.querySelector(`.card[data-rank="${rank}"]`);
}

function ensureDetailEl(card) {
  let det = card.querySelector('.detail');
  if (!det) {
    det = document.createElement('div');
    det.className = 'detail';
    card.querySelector('.info').appendChild(det);
  }
  return det;
}

function showDetailLoading(rank) {
  const card = cardByRank(rank);
  if (!card) return;
  const store = card.dataset.store || 'product';
  ensureDetailEl(card).innerHTML =
    `<span class="detail-src">📄 Reading the ${escapeHtml(store)} product page…</span>`;
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showDetail(rank, summary) {
  const card = cardByRank(rank);
  if (!card) return;
  const store = card.dataset.store || 'product';
  // Attribute the summary to its source so it doesn't read as made-up.
  ensureDetailEl(card).innerHTML =
    `<span class="detail-src">📄 From the ${escapeHtml(store)} product page &amp; reviews</span>${escapeHtml(summary)}`;
}

function highlightCard(rank) {
  const card = cardByRank(rank);
  if (!card) return;
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  card.classList.remove('flash');
  void card.offsetWidth; // restart the animation
  card.classList.add('flash');
}

function showNoProducts(query) {
  els.results.hidden = false;
  els.resultsTitle.textContent = `No results for “${query}”`;
  els.cards.innerHTML = '<div class="searching">Nothing came back — the store may have blocked the request. Ask me to try again or rephrase.</div>';
}

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = `status ${cls || ''}`;
}

const READY_TEXT = 'Ready — tap to talk';
function setReady() {
  speaking = false;
  els.mic.disabled = false;
  els.mic.classList.remove('speaking');
  setStatus(READY_TEXT, 'ok');
}

// First sign of the agent's reply this turn (audio or text) -> "Speaking".
function markSpeaking() {
  if (speaking || recording) return;
  speaking = true;
  els.mic.classList.add('speaking');
  setStatus('Speaking…', 'ok');
}

// --- WebSocket -----------------------------------------------------------
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    reconnectAttempts = 0;
    setStatus('Connecting to agent…', 'wait');
    // First message: hand back our resume handle + last results so the session and
    // server state can be restored.
    ws.send(JSON.stringify({ type: 'init', resume_handle: resumeHandle, last_results: lastResults }));
  };
  ws.onclose = () => {
    els.mic.disabled = true;
    els.mic.classList.remove('live', 'speaking');
    const delay = Math.min(1000 * 2 ** reconnectAttempts, 10000); // backoff, cap 10s
    reconnectAttempts++;
    setStatus(`Disconnected — reconnecting in ${Math.round(delay / 1000)}s…`, 'err');
    setTimeout(connect, delay);
  };
  ws.onerror = () => {};

  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) { playPCM(ev.data); return; }
    const msg = JSON.parse(ev.data);
    switch (msg.type) {
      case 'ready': setReady(); break;
      case 'resume_handle': setResumeHandle(msg.handle); break;
      case 'user_transcript': appendTranscript('user', msg.text); break;
      case 'agent_transcript': markSpeaking(); appendTranscript('agent', msg.text); break;
      case 'interrupted': stopPlayback(); break;
      case 'turn_complete': endTurnUI(); setReady(); break;
      case 'search_running': {
        const where = msg.stores && msg.stores.length ? msg.stores.join(' + ') : 'Amazon';
        setStatus(`Searching ${where}…`, 'wait');
        showSearching(msg.query, msg.stores);
        break;
      }
      case 'products': renderCards(msg.query, msg.cards); break;
      case 'no_products': showNoProducts(msg.query); break;
      case 'detail_running': showDetailLoading(msg.rank); break;
      case 'product_detail': showDetail(msg.rank, msg.summary); break;
      case 'highlight_product': highlightCard(msg.rank); break;
      case 'error': setStatus(`Error: ${msg.message}`, 'err'); setTimeout(setReady, 2500); break;
    }
  };
}

// --- Mic capture ---------------------------------------------------------
async function startRecording() {
  if (recording || !ws || ws.readyState !== WebSocket.OPEN) return;
  recording = true;
  speaking = false;
  els.mic.classList.remove('speaking');
  unlockAudio();    // unlock/resume playback within this user gesture (iOS)
  stopPlayback();   // barge-in: cut any current reply

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
  } catch (e) {
    setStatus('Mic permission denied', 'err');
    recording = false;
    return;
  }

  micCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: INPUT_RATE });
  await micCtx.audioWorklet.addModule('/static/pcm-processor.js');
  const source = micCtx.createMediaStreamSource(micStream);
  workletNode = new AudioWorkletNode(micCtx, 'pcm-processor');
  workletNode.port.onmessage = (e) => {
    if (recording && ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
  };
  source.connect(workletNode);
  // Worklet needs a sink in some browsers; route to a muted gain.
  const sink = micCtx.createGain();
  sink.gain.value = 0;
  workletNode.connect(sink).connect(micCtx.destination);

  ws.send(JSON.stringify({ type: 'start_turn' }));
  els.mic.classList.add('live');
  setStatus('Listening… tap again to send', 'live');
}

async function stopRecording() {
  if (!recording) return;
  recording = false;
  els.mic.classList.remove('live');
  setStatus('Thinking…', 'wait');

  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'end_turn' }));
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (micCtx) { await micCtx.close(); micCtx = null; }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
}

// Tap to start a turn, tap again to send. (Avoids hold-drift cutting speech off.)
function toggleRecording() {
  if (els.mic.disabled) return;
  if (recording) stopRecording(); else startRecording();
}
els.mic.addEventListener('click', (e) => { e.preventDefault(); toggleRecording(); });

// Space bar also toggles
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.repeat && !els.mic.disabled) { e.preventDefault(); toggleRecording(); }
});

// Prime/unlock audio on the very first user interaction (iOS needs a gesture).
function primeAudioOnce() { unlockAudio(); }
document.addEventListener('touchend', primeAudioOnce, { once: true });
document.addEventListener('click', primeAudioOnce, { once: true });

// Restore the last results visually on reload (the socket also re-seeds the server).
if (lastResults && lastResults.cards && lastResults.cards.length) {
  renderCards(lastResults.query, lastResults.cards);
}

connect();
