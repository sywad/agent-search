// Voice shopping agent — client.
// Hands-free: tap the mic (or press Space) once to start a conversation; the mic
// streams continuously and Gemini's automatic VAD detects turns (you can talk over
// it). Tap again to mute/end. Plays back Gemini's 24 kHz PCM reply and shows
// transcripts, ranked product cards, and a stage rail of what the agent is doing.

const INPUT_RATE = 16000;
const OUTPUT_RATE = 24000;

const els = {
  mic: document.getElementById('mic'),
  status: document.getElementById('status'),
  transcript: document.getElementById('transcript'),
  transcriptWrap: document.getElementById('transcript-wrap'),
  transcriptHead: document.getElementById('transcript-head'),
  transcriptLatest: document.getElementById('transcript-latest'),
  results: document.getElementById('results'),
  resultsTitle: document.getElementById('results-title'),
  cards: document.getElementById('cards'),
  stages: document.querySelectorAll('.stage'),
  stageDetail: document.getElementById('stage-detail'),
  empty: document.getElementById('empty'),
  intro: document.getElementById('intro'),
  helpToggle: document.getElementById('help-toggle'),
  newSession: document.getElementById('new-session'),
};

// --- Agent activity (stage rail) ----------------------------------------
const STAGE_ORDER = ['listen', 'think', 'search', 'rank', 'speak'];

function setStage(name) {
  const active = STAGE_ORDER.indexOf(name);
  els.stages.forEach((el) => {
    const i = STAGE_ORDER.indexOf(el.dataset.stage);
    el.classList.toggle('active', i === active);
    el.classList.toggle('done', active >= 0 && i < active);
  });
}

// One call to update the whole activity display: stage chip, big detail line,
// and the small status under the mic.
function setActivity(stage, text, cls) {
  if (stage !== undefined) setStage(stage);
  if (text !== undefined) {
    els.stageDetail.textContent = text;
    els.stageDetail.className = `stage-detail ${cls || ''}`;
    setStatus(text, cls);
  }
}

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

// True while the agent's audio is still scheduled/playing. We gate the mic on this
// so it doesn't capture the agent's own voice (echo) and self-interrupt in
// hands-free mode on a speaker.
function isAgentPlaying() {
  return !!playCtx && nextPlayTime > playCtx.currentTime + 0.05;
}

// --- Transcript UI -------------------------------------------------------
let curUser = null;
let curAgent = null;

function appendTranscript(role, text) {
  let node = role === 'user' ? curUser : curAgent;
  if (!node) {
    node = document.createElement('div');
    node.className = `line ${role}`;
    node.innerHTML = `<span class="who">${role === 'user' ? 'You' : 'Agent'}:</span> <span class="msg"></span>`;
    els.transcript.appendChild(node);
    if (role === 'user') curUser = node; else curAgent = node;
  }
  node.querySelector('.msg').textContent += text;
  els.transcript.scrollTop = els.transcript.scrollHeight;
  // Mirror the latest line into the collapsed dock header.
  const who = role === 'user' ? 'You' : 'Agent';
  els.transcriptLatest.textContent = `${who}: ${node.querySelector('.msg').textContent}`;
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

function hideEmpty() { if (els.empty) els.empty.hidden = true; }

function showSearching(query, stores) {
  const where = stores && stores.length ? stores.join(' + ') : 'Amazon';
  hideEmpty();
  els.results.hidden = false;
  els.resultsTitle.textContent = `Searching ${where} for “${query}”…`;
  els.cards.innerHTML = '<div class="searching">🔎 Finding and ranking products…</div>';
  els.results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function cardEl(c) {
  const price = c.price && c.price !== 'N/A' ? `$${escapeHtml(c.price)}` : 'Price n/a';
  const rating = c.rating && c.rating !== 'N/A'
    ? `<span class="rating">★ ${escapeHtml(c.rating)} <span class="muted">(${escapeHtml(c.reviews || '0')})</span></span>` : '';
  const store = c.store ? `<span class="store store-${escapeHtml(c.source || '')}">${escapeHtml(c.store)}</span>` : '';
  const card = document.createElement('article');
  card.className = 'card';
  card.dataset.rank = c.rank;
  card.dataset.store = c.store || 'product';
  card.innerHTML = `
    <div class="thumb-wrap">
      ${c.image ? `<img class="thumb" src="${escapeHtml(c.image)}" alt="" loading="lazy">` : '<div class="thumb empty-thumb"></div>'}
      <span class="rank">#${escapeHtml(c.rank)}</span>
      ${store}
    </div>
    <div class="info">
      <a class="title" href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>
      <div class="meta"><span class="price">${price}</span>${rating}</div>
      ${c.why ? `<div class="why">${escapeHtml(c.why)}</div>` : ''}
    </div>`;
  return card;
}

function renderCards(query, cards) {
  lastResults = { query, cards };
  persistResults();
  hideEmpty();
  els.results.hidden = false;
  els.resultsTitle.textContent = `${cards.length} picks for “${query}”`;
  els.cards.innerHTML = '';
  for (const c of cards) els.cards.appendChild(cardEl(c));
}

// Re-render the current results in a new order/subset (sort/filter/limit) without
// re-fetching. Cards are pulled by rank from the results we already hold.
function arrangeCards(order, title) {
  if (!lastResults || !lastResults.cards) return;
  const byRank = new Map(lastResults.cards.map((c) => [c.rank, c]));
  const arranged = order.map((r) => byRank.get(r)).filter(Boolean);
  if (!arranged.length) return;
  hideEmpty();
  els.results.hidden = false;
  if (title) els.resultsTitle.textContent = title;
  els.cards.innerHTML = '';
  for (const c of arranged) els.cards.appendChild(cardEl(c));
  els.results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
  hideEmpty();
  els.results.hidden = false;
  els.resultsTitle.textContent = `No results for “${query}”`;
  els.cards.innerHTML = '<div class="searching">Nothing came back — the store may have blocked the request. Ask me to try again or rephrase.</div>';
}

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = `status ${cls || ''}`;
}

const READY_TEXT = 'Tap the mic to start a conversation';
function setReady() {
  speaking = false;
  els.mic.disabled = false;
  els.mic.classList.remove('speaking', 'live');
  setActivity('', READY_TEXT, 'ok'); // '' clears the stage rail
}

// Back to listening between turns while the conversation stays active.
function setListening() {
  speaking = false;
  setActivity('listen', 'Listening… tap to mute', 'live');
}

// First sign of the agent's reply this turn (audio or text) -> "Speaking".
// In hands-free the mic stays open (for barge-in); only the stage rail changes.
function markSpeaking() {
  if (speaking) return;
  speaking = true;
  setActivity('speak', 'Speaking…', 'ok');
}

// --- WebSocket -----------------------------------------------------------
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    reconnectAttempts = 0;
    setActivity('', 'Connecting…');
    // First message: hand back our resume handle + last results so the session and
    // server state can be restored.
    ws.send(JSON.stringify({ type: 'init', resume_handle: resumeHandle, last_results: lastResults }));
  };
  ws.onclose = () => {
    els.mic.disabled = true;
    els.mic.classList.remove('live', 'speaking');
    const delay = Math.min(1000 * 2 ** reconnectAttempts, 10000); // backoff, cap 10s
    reconnectAttempts++;
    setActivity('', `Disconnected — reconnecting in ${Math.round(delay / 1000)}s…`, 'err');
    setTimeout(connect, delay);
  };
  ws.onerror = () => {};

  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) { playPCM(ev.data); return; }
    const msg = JSON.parse(ev.data);
    switch (msg.type) {
      case 'ready': recording ? setListening() : setReady(); break;
      case 'resume_handle': setResumeHandle(msg.handle); break;
      case 'user_transcript':
        if (recording && !msg.text.trim()) break;
        speaking = false;
        if (recording) setListening();
        appendTranscript('user', msg.text);
        break;
      case 'agent_transcript': markSpeaking(); appendTranscript('agent', msg.text); break;
      case 'interrupted': stopPlayback(); speaking = false; break;
      case 'turn_complete':
        endTurnUI();
        speaking = false;
        if (recording) setListening(); else setReady();
        break;
      case 'search_running': {
        const where = msg.stores && msg.stores.length ? msg.stores.join(' + ') : 'Amazon';
        setActivity('search', `Searching ${where}…`);
        showSearching(msg.query, msg.stores);
        break;
      }
      case 'products':
        setActivity('rank', `Ranked ${msg.cards.length} picks`, 'ok');
        renderCards(msg.query, msg.cards);
        break;
      case 'no_products': setActivity('rank', 'No matches found'); showNoProducts(msg.query); break;
      case 'detail_running': setActivity(undefined, 'Reading product details…'); showDetailLoading(msg.rank); break;
      case 'product_detail': showDetail(msg.rank, msg.summary); break;
      case 'highlight_product': highlightCard(msg.rank); break;
      case 'arrange': arrangeCards(msg.order, msg.title); break;
      case 'visual_running': setActivity('search', `Looking at the photos for “${msg.criteria}”…`); break;
      case 'error': setActivity(undefined, `Error: ${msg.message}`, 'err'); setTimeout(setReady, 2500); break;
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
    setActivity(undefined, 'Mic permission denied', 'err');
    recording = false;
    return;
  }

  micCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: INPUT_RATE });
  await micCtx.audioWorklet.addModule('/static/pcm-processor.js');
  const source = micCtx.createMediaStreamSource(micStream);
  workletNode = new AudioWorkletNode(micCtx, 'pcm-processor');
  workletNode.port.onmessage = (e) => {
    // Half-duplex: don't stream mic audio while the agent is talking, or it hears
    // itself through the speaker and self-interrupts.
    if (recording && !isAgentPlaying() && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(e.data);
    }
  };
  source.connect(workletNode);
  // Worklet needs a sink in some browsers; route to a muted gain.
  const sink = micCtx.createGain();
  sink.gain.value = 0;
  workletNode.connect(sink).connect(micCtx.destination);

  // Hands-free: the mic now streams continuously; Gemini's VAD detects turns.
  els.mic.classList.add('live');
  setListening();
}

async function stopRecording() {
  if (!recording) return;
  recording = false;
  els.mic.classList.remove('live');
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (micCtx) { await micCtx.close(); micCtx = null; }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  setReady();
}

// Tap once to start the conversation, tap again to mute/end it.
function toggleRecording() {
  if (els.mic.disabled) return;
  if (recording) stopRecording(); else startRecording();
}
els.mic.addEventListener('click', (e) => { e.preventDefault(); toggleRecording(); });

// Space bar also toggles
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.repeat && !els.mic.disabled) { e.preventDefault(); toggleRecording(); }
});

// Expand/collapse the small transcript window.
els.transcriptHead.addEventListener('click', () => {
  const open = els.transcriptWrap.classList.toggle('open');
  els.transcriptHead.setAttribute('aria-expanded', String(open));
  if (open) els.transcript.scrollTop = els.transcript.scrollHeight;
});

// Toggle the how-to intro.
els.helpToggle.addEventListener('click', () => { els.intro.hidden = !els.intro.hidden; });

// Start a brand-new session: clear persisted context + UI and reconnect with no
// resume handle, so Gemini starts a fresh conversation.
function startNewSession() {
  if (recording) stopRecording();
  stopPlayback();
  resumeHandle = null;
  lastResults = null;
  try {
    sessionStorage.removeItem('vs_resume_handle');
    sessionStorage.removeItem('vs_last_results');
  } catch (e) {}
  // Reset UI
  els.cards.innerHTML = '';
  els.results.hidden = true;
  els.transcript.innerHTML = '';
  els.transcriptLatest.textContent = 'Conversation';
  els.transcriptWrap.classList.remove('open');
  curUser = null; curAgent = null;
  if (els.empty) els.empty.hidden = false;
  // Reconnect cleanly (drop the old socket without triggering its auto-reconnect).
  reconnectAttempts = 0;
  if (ws) { ws.onclose = null; try { ws.close(); } catch (e) {} }
  connect();
}
els.newSession.addEventListener('click', startNewSession);

// Prime/unlock audio on the very first user interaction (iOS needs a gesture).
function primeAudioOnce() { unlockAudio(); }
document.addEventListener('touchend', primeAudioOnce, { once: true });
document.addEventListener('click', primeAudioOnce, { once: true });

// Restore the last results visually on reload (the socket also re-seeds the server).
if (lastResults && lastResults.cards && lastResults.cards.length) {
  renderCards(lastResults.query, lastResults.cards);
}

connect();
