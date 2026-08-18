(function () {
  'use strict';

  var ADDON_VERSION = '?';
  var TARGET_SAMPLE_RATE = 16000;
  var ENROLL_RECORD_SECONDS = 5.0;
  var VERIFY_RECORD_SECONDS = 3.0;
  var MIN_SAMPLES_FOR_ENROLL = 3;
  var SPEAKER_NAME_REGEX = /^[A-Za-z0-9_\-]{1,32}$/;
  var WS_CHUNK_INT16 = 1280;
  var WS_CHUNK_BYTES = WS_CHUNK_INT16 * 2;
  var CHART_WINDOW_SECONDS = 30;
  var MAX_TRIGGER_LOG = 20;
  var SPEAKER_COLORS = ['#ba68c8', '#4dd0e1', '#ffb74d', '#ff8a65', '#aed581'];
  var KEYWORD_COLORS = [
    '#03a9f4',
    '#ff9800',
    '#9c27b0',
    '#4caf50',
    '#e91e63',
    '#00bcd4',
    '#cddc39',
    '#795548',
  ];

  var state = {
    activeTab: 'enroll',
    theme: localStorage.getItem('wls.theme') || 'dark',

    speakers: [],
    speakerThreshold: 0.25,
    verifierActive: false,
    speakersLoaded: false,

    speakerName: localStorage.getItem('wls.lastSpeaker') || '',
    samples: [],
    recording: false,
    verifyRecording: false,

    streaming: false,
    debugWS: null,
    scoreWS: null,
    debugAudio: null,
    scoreHistory: [],
    wakeWords: [],
    sensitivities: [],
    predictIntervalSeconds: 0.5,
    debugSpeakerThreshold: 0.25,
    debugVerifierActive: false,
    lastSpeakerScores: null,
    lastBestMatch: null,
    triggerLog: [],
    midSentenceLast: null,
    lastAcceptedKeyword: null,
    chartFlashUntil: 0,
    debugVuLevel: 0,
  };

  function $(id) { return document.getElementById(id); }

  function setText(id, text) {
    var n = $(id);
    if (n) n.textContent = text;
  }

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        var v = attrs[k];
        if (v == null || v === false) return;
        if (k === 'class') node.className = v;
        else if (k === 'text') node.textContent = v;
        else if (k === 'html') node.innerHTML = v;
        else if (k.indexOf('on') === 0 && typeof v === 'function') node.addEventListener(k.slice(2), v);
        else if (k === 'style' && typeof v === 'object') {
          Object.keys(v).forEach(function (sk) { node.style[sk] = v[sk]; });
        } else if (v === true) node.setAttribute(k, '');
        else node.setAttribute(k, v);
      });
    }
    if (children) {
      children.forEach(function (c) {
        if (c == null || c === false) return;
        if (typeof c === 'string' || typeof c === 'number') node.appendChild(document.createTextNode(String(c)));
        else node.appendChild(c);
      });
    }
    return node;
  }

  function clearChildren(node) {
    while (node && node.firstChild) node.removeChild(node.firstChild);
  }

  function formatBytes(bytes) {
    if (!bytes && bytes !== 0) return '—';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }

  function formatTime(ts) {
    var d = new Date(ts * 1000);
    var h = String(d.getHours()).padStart(2, '0');
    var m = String(d.getMinutes()).padStart(2, '0');
    return h + ':' + m;
  }

  function nowSec() { return Date.now() / 1000; }

  function announce(msg) {
    var r = $('status-region');
    if (r) r.textContent = msg;
  }

  function toast(message, type, duration) {
    type = type || 'info';
    duration = duration || 4000;
    var container = $('toast-container');
    if (!container) return;
    var node = el('div', { class: 'toast toast-' + type, role: 'status' }, [message]);
    container.appendChild(node);
    requestAnimationFrame(function () { node.classList.add('show'); });
    setTimeout(function () {
      node.classList.remove('show');
      setTimeout(function () { if (node.parentNode) node.parentNode.removeChild(node); }, 250);
    }, duration);
    announce(message);
  }

  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    state.theme = t;
    localStorage.setItem('wls.theme', t);
  }

  function initTheme() {
    applyTheme(state.theme === 'light' ? 'light' : 'dark');
    var btn = $('theme-toggle');
    if (btn) btn.addEventListener('click', function () {
      applyTheme(state.theme === 'dark' ? 'light' : 'dark');
    });
  }

  function setActiveTab(tab) {
    state.activeTab = tab;
    var buttons = document.querySelectorAll('.tab-button');
    buttons.forEach(function (b) {
      var active = b.getAttribute('data-tab') === tab;
      b.classList.toggle('active', active);
      b.setAttribute('aria-selected', active ? 'true' : 'false');
      b.setAttribute('tabindex', active ? '0' : '-1');
    });
    var panels = document.querySelectorAll('.tab-panel');
    panels.forEach(function (p) {
      var active = p.id === 'tab-' + tab;
      p.classList.toggle('active', active);
      if (active) p.removeAttribute('hidden');
      else p.setAttribute('hidden', '');
    });
  }

  function initTabs() {
    var buttons = document.querySelectorAll('.tab-button');
    buttons.forEach(function (b) {
      b.addEventListener('click', function () { setActiveTab(b.getAttribute('data-tab')); });
      b.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
          e.preventDefault();
          var arr = Array.prototype.slice.call(buttons);
          var idx = arr.indexOf(b);
          var next = e.key === 'ArrowRight' ? (idx + 1) % arr.length : (idx - 1 + arr.length) % arr.length;
          arr[next].focus();
          setActiveTab(arr[next].getAttribute('data-tab'));
        }
      });
    });
  }

  async function apiRequest(path, options) {
    options = options || {};
    var resp;
    try {
      resp = await fetch(path, options);
    } catch (e) {
      throw new Error('Network error: ' + (e && e.message ? e.message : e));
    }
    if (!resp.ok) {
      var msg = '';
      try {
        var ct = (resp.headers.get('Content-Type') || '').toLowerCase();
        if (ct.indexOf('application/json') !== -1) {
          var data = await resp.json();
          msg = data.reason || data.error || data.message || '';
        } else {
          msg = (await resp.text()).trim();
          if (msg.length > 200) msg = msg.slice(0, 200) + '…';
        }
      } catch (_) {}
      if (!msg) msg = resp.statusText || ('HTTP ' + resp.status);
      var err = new Error(msg);
      err.status = resp.status;
      throw err;
    }
    var ct2 = (resp.headers.get('Content-Type') || '').toLowerCase();
    if (ct2.indexOf('application/json') !== -1) return resp.json();
    return resp.text();
  }

  function downsampleLinear(samples, fromRate, toRate) {
    if (fromRate === toRate) return new Float32Array(samples);
    var ratio = fromRate / toRate;
    var outLen = Math.max(0, Math.floor(samples.length / ratio));
    var out = new Float32Array(outLen);
    for (var i = 0; i < outLen; i++) {
      var srcIdx = i * ratio;
      var i0 = Math.floor(srcIdx);
      var i1 = Math.min(i0 + 1, samples.length - 1);
      var frac = srcIdx - i0;
      out[i] = samples[i0] * (1 - frac) + samples[i1] * frac;
    }
    return out;
  }

  function floatToInt16(samples) {
    var out = new Int16Array(samples.length);
    for (var i = 0; i < samples.length; i++) {
      var s = samples[i];
      if (s > 1) s = 1;
      else if (s < -1) s = -1;
      out[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
    }
    return out;
  }

  function concatFloat32(chunks) {
    var total = 0;
    for (var i = 0; i < chunks.length; i++) total += chunks[i].length;
    var out = new Float32Array(total);
    var off = 0;
    for (var j = 0; j < chunks.length; j++) {
      out.set(chunks[j], off);
      off += chunks[j].length;
    }
    return out;
  }

  function rms(samples) {
    if (!samples.length) return 0;
    var sum = 0;
    for (var i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
    return Math.sqrt(sum / samples.length);
  }

  function peak(samples) {
    var p = 0;
    for (var i = 0; i < samples.length; i++) {
      var v = Math.abs(samples[i]);
      if (v > p) p = v;
    }
    return p;
  }

  async function createMicCapture(opts) {
    opts = opts || {};
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Microphone API unavailable. Use HTTPS or localhost.');
    }

    var stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: TARGET_SAMPLE_RATE,
        },
      });
    } catch (e) {
      var name = e && e.name ? e.name : '';
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        throw new Error('Microphone permission denied. Grant access in your browser settings and reload.');
      }
      if (name === 'NotFoundError' || name === 'OverconstrainedError') {
        throw new Error('No microphone available on this device.');
      }
      throw new Error('Could not start microphone: ' + (e && e.message ? e.message : e));
    }

    var Ctor = window.AudioContext || window.webkitAudioContext;
    if (!Ctor) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      throw new Error('AudioContext is not supported by this browser.');
    }

    var ctx;
    try {
      ctx = new Ctor({ sampleRate: TARGET_SAMPLE_RATE });
    } catch (_) {
      ctx = new Ctor();
    }

    var source = ctx.createMediaStreamSource(stream);
    var analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.4;
    source.connect(analyser);

    var processor = ctx.createScriptProcessor(4096, 1, 1);
    var muteGain = ctx.createGain();
    muteGain.gain.value = 0;
    source.connect(processor);
    processor.connect(muteGain);
    muteGain.connect(ctx.destination);

    var sourceRate = ctx.sampleRate;
    var stopped = false;
    var pending16k = new Float32Array(0);

    processor.onaudioprocess = function (event) {
      if (stopped) return;
      var input = event.inputBuffer.getChannelData(0);
      var copy = new Float32Array(input.length);
      copy.set(input);
      var down = sourceRate === TARGET_SAMPLE_RATE ? copy : downsampleLinear(copy, sourceRate, TARGET_SAMPLE_RATE);
      if (opts.onSamples) opts.onSamples(down, copy, sourceRate);
      if (opts.onChunk16k) {
        var combined = new Float32Array(pending16k.length + down.length);
        combined.set(pending16k);
        combined.set(down, pending16k.length);
        var offset = 0;
        while (combined.length - offset >= WS_CHUNK_INT16) {
          var slice = combined.subarray(offset, offset + WS_CHUNK_INT16);
          opts.onChunk16k(floatToInt16(slice));
          offset += WS_CHUNK_INT16;
        }
        pending16k = combined.slice(offset);
      }
    };

    return {
      ctx: ctx,
      stream: stream,
      analyser: analyser,
      sourceRate: sourceRate,
      stop: function () {
        if (stopped) return;
        stopped = true;
        try { processor.disconnect(); } catch (_) {}
        try { analyser.disconnect(); } catch (_) {}
        try { source.disconnect(); } catch (_) {}
        try { muteGain.disconnect(); } catch (_) {}
        stream.getTracks().forEach(function (t) { try { t.stop(); } catch (_) {} });
        try { ctx.close(); } catch (_) {}
      },
    };
  }

  function drawAnalyserWaveform(canvas, analyser) {
    if (!canvas || !analyser) return;
    var ctx2d = canvas.getContext('2d');
    var buf = new Uint8Array(analyser.fftSize);
    var dpr = window.devicePixelRatio || 1;

    function resize() {
      var rect = canvas.getBoundingClientRect();
      var w = Math.max(1, Math.round(rect.width * dpr));
      var h = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;
    }

    var primary = getComputedStyle(document.documentElement).getPropertyValue('--color-primary').trim() || '#03a9f4';
    var stopped = false;

    function renderFrame() {
      if (stopped) return;
      resize();
      analyser.getByteTimeDomainData(buf);
      var w = canvas.width;
      var h = canvas.height;
      ctx2d.clearRect(0, 0, w, h);

      ctx2d.strokeStyle = 'rgba(255,255,255,0.06)';
      if (document.documentElement.getAttribute('data-theme') === 'light') {
        ctx2d.strokeStyle = 'rgba(0,0,0,0.08)';
      }
      ctx2d.lineWidth = 1 * dpr;
      ctx2d.beginPath();
      ctx2d.moveTo(0, h / 2);
      ctx2d.lineTo(w, h / 2);
      ctx2d.stroke();

      ctx2d.lineWidth = 2 * dpr;
      ctx2d.strokeStyle = primary;
      ctx2d.beginPath();
      var binWidth = w / buf.length;
      for (var i = 0; i < buf.length; i++) {
        var v = (buf[i] - 128) / 128;
        var y = h / 2 + v * (h / 2 - 2 * dpr);
        if (i === 0) ctx2d.moveTo(0, y);
        else ctx2d.lineTo(i * binWidth, y);
      }
      ctx2d.stroke();

      requestAnimationFrame(renderFrame);
    }

    renderFrame();
    return function () { stopped = true; };
  }

  function drawSampleThumb(canvas, int16) {
    if (!canvas) return;
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    var w = Math.max(1, Math.round((rect.width || 120) * dpr));
    var h = Math.max(1, Math.round((rect.height || 32) * dpr));
    canvas.width = w;
    canvas.height = h;
    var ctx2d = canvas.getContext('2d');
    ctx2d.clearRect(0, 0, w, h);
    var primary = getComputedStyle(document.documentElement).getPropertyValue('--color-primary').trim() || '#03a9f4';
    var bins = Math.min(80, w);
    var step = Math.floor(int16.length / bins) || 1;
    ctx2d.fillStyle = primary;
    for (var i = 0; i < bins; i++) {
      var startIdx = i * step;
      var endIdx = Math.min(int16.length, startIdx + step);
      var maxV = 0;
      for (var j = startIdx; j < endIdx; j++) {
        var v = Math.abs(int16[j]);
        if (v > maxV) maxV = v;
      }
      var norm = maxV / 32768;
      var barH = Math.max(1, norm * (h - 2));
      var x = (i / bins) * w;
      var barW = Math.max(1, (w / bins) - dpr);
      ctx2d.fillRect(x, (h - barH) / 2, barW, barH);
    }
  }

  function pathRelative(p) {
    if (!p) return p;
    return p.charAt(0) === '/' ? p.slice(1) : p;
  }

  function wsUrlRelative(p) {
    var rel = pathRelative(p);
    var base = new URL('.', window.location.href);
    var u = new URL(rel, base);
    u.protocol = (u.protocol === 'https:') ? 'wss:' : 'ws:';
    return u.toString();
  }

  function createReconnectingWS(path, handlers) {
    var stopped = false;
    var ws = null;
    var backoffMs = 1000;
    var reconnectTimer = null;

    function connect() {
      if (stopped) return;
      try {
        if (handlers.onConnecting) handlers.onConnecting();
        ws = new WebSocket(wsUrlRelative(path));
      } catch (e) {
        scheduleReconnect();
        return;
      }
      ws.binaryType = 'arraybuffer';
      ws.onopen = function () {
        backoffMs = 1000;
        if (handlers.onOpen) handlers.onOpen();
      };
      ws.onmessage = function (ev) {
        if (typeof ev.data === 'string') {
          var msg;
          try { msg = JSON.parse(ev.data); } catch (_) { return; }
          if (handlers.onMessage) handlers.onMessage(msg);
        }
      };
      ws.onclose = function () {
        if (handlers.onClose) handlers.onClose();
        if (!stopped) scheduleReconnect();
      };
      ws.onerror = function () {
        try { ws.close(); } catch (_) {}
      };
    }

    function scheduleReconnect() {
      if (stopped) return;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connect, backoffMs);
      backoffMs = Math.min(backoffMs * 2, 30000);
    }

    connect();

    return {
      send: function (data) {
        if (ws && ws.readyState === 1) {
          try { ws.send(data); } catch (_) {}
        }
      },
      close: function () {
        stopped = true;
        if (reconnectTimer) clearTimeout(reconnectTimer);
        if (ws) try { ws.close(); } catch (_) {}
      },
      reconnectNow: function () {
        if (reconnectTimer) clearTimeout(reconnectTimer);
        if (ws) try { ws.close(); } catch (_) {}
        backoffMs = 1000;
        connect();
      },
      isOpen: function () { return !!ws && ws.readyState === 1; },
    };
  }

  function recordFixed(seconds, onProgress, onLevel) {
    return new Promise(function (resolve, reject) {
      var samples = [];
      var startTime = 0;
      var capture = null;
      var rafId = null;
      var stopWave = null;
      var stopTimer = null;
      var settled = false;

      function settle(fn, val) {
        if (settled) return;
        settled = true;
        if (rafId) cancelAnimationFrame(rafId);
        if (stopTimer) clearTimeout(stopTimer);
        if (stopWave) stopWave();
        if (capture) capture.stop();
        fn(val);
      }

      createMicCapture({
        onSamples: function (down) {
          samples.push(down);
          if (onLevel) onLevel(rms(down));
        },
      }).then(function (cap) {
        capture = cap;
        startTime = performance.now();
        stopWave = drawAnalyserWaveform($('live-waveform'), cap.analyser);

        function tick() {
          if (settled) return;
          var elapsed = (performance.now() - startTime) / 1000;
          var remaining = Math.max(0, seconds - elapsed);
          if (onProgress) onProgress(remaining, elapsed);
          if (elapsed >= seconds) {
            var merged = concatFloat32(samples);
            settle(resolve, { float32: merged, durationSeconds: merged.length / TARGET_SAMPLE_RATE });
            return;
          }
          rafId = requestAnimationFrame(tick);
        }
        rafId = requestAnimationFrame(tick);

        stopTimer = setTimeout(function () {
          if (settled) return;
          var merged = concatFloat32(samples);
          settle(resolve, { float32: merged, durationSeconds: merged.length / TARGET_SAMPLE_RATE });
        }, (seconds + 0.4) * 1000);
      }).catch(function (e) {
        settle(reject, e);
      });
    });
  }

  function validateSpeakerName(name) {
    if (!name) return 'Name is required.';
    if (!SPEAKER_NAME_REGEX.test(name)) return 'Use 1–32 letters, digits, underscores, or dashes.';
    return null;
  }

  async function fetchSpeakers() {
    try {
      var data = await apiRequest('api/speakers');
      state.speakers = data.speakers || [];
      state.speakerThreshold = typeof data.threshold === 'number' ? data.threshold : 0.25;
      state.verifierActive = !!data.verifier_active;
      state.speakersLoaded = true;
      renderSpeakersList();
      renderEnrollBanner();
      setText('speaker-threshold-display', state.speakerThreshold.toFixed(2));
    } catch (e) {
      toast('Failed to load speakers: ' + e.message, 'error');
    }
  }

  function renderEnrollBanner() {
    var banner = $('enroll-status-banner');
    if (!banner) return;
    if (!state.speakersLoaded) {
      banner.setAttribute('data-tone', '');
      setText('banner-title', 'Loading speaker status…');
      setText('banner-detail', 'Connecting to the wake word service.');
      return;
    }
    if (state.speakers.length === 0) {
      banner.setAttribute('data-tone', 'warning');
      setText('banner-title', 'No speakers enrolled yet');
      setText('banner-detail', 'Record samples below to start verifying who triggers the wake word.');
    } else if (!state.verifierActive) {
      banner.setAttribute('data-tone', 'warning');
      setText('banner-title', state.speakers.length + ' profile(s) on disk, but speaker verification is inactive');
      setText('banner-detail', 'Restart the add-on so profiles can load, or check the logs for errors.');
    } else {
      banner.setAttribute('data-tone', 'success');
      setText('banner-title', state.speakers.length + ' speaker(s) enrolled');
      setText('banner-detail', 'Speaker verification is active. Threshold ' + state.speakerThreshold.toFixed(2) + '.');
    }
  }

  function renderSpeakersList() {
    var container = $('speakers-list');
    if (!container) return;
    clearChildren(container);
    if (!state.speakers.length) {
      container.appendChild(el('div', { class: 'empty-state' }, ['No speakers enrolled yet.']));
      return;
    }
    state.speakers.forEach(function (sp) {
      var meta = formatBytes(sp.size_bytes) + ' · modified ' + new Date((sp.modified || 0) * 1000).toLocaleString();
      var row = el('div', { class: 'speaker-row' }, [
        el('div', { class: 'speaker-row-main' }, [
          el('span', { class: 'speaker-name' }, [sp.name]),
          el('span', { class: 'speaker-meta' }, [meta]),
        ]),
        el('div', { class: 'speaker-actions' }, [
          el('button', { class: 'text-button', type: 'button', onclick: function () { handleReenroll(sp.name); } }, ['Re-enroll']),
          el('button', { class: 'text-button', type: 'button', onclick: function () { handleRename(sp.name); } }, ['Rename']),
          el('button', { class: 'danger-button', type: 'button', onclick: function () { handleDelete(sp.name); } }, ['Delete']),
        ]),
      ]);
      container.appendChild(row);
    });
  }

  function handleReenroll(name) {
    var input = $('speaker-name');
    if (input) {
      input.value = name;
      state.speakerName = name;
      state.samples = [];
      renderSamples();
      updateEnrollButton();
      validateNameField();
    }
    setActiveTab('enroll');
    var card = input ? input.closest('.card') : null;
    if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    toast('Recording new samples will replace "' + name + '"', 'info');
  }

  async function handleRename(oldName) {
    var newName = window.prompt('New name for "' + oldName + '"', oldName);
    if (newName == null) return;
    newName = newName.trim();
    var err = validateSpeakerName(newName);
    if (err) { toast(err, 'error'); return; }
    if (newName === oldName) return;
    try {
      await apiRequest('api/speakers/' + encodeURIComponent(oldName) + '/rename', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_name: newName }),
      });
      toast('Renamed "' + oldName + '" → "' + newName + '"', 'success');
      await fetchSpeakers();
    } catch (e) {
      toast('Rename failed: ' + e.message, 'error');
    }
  }

  async function handleDelete(name) {
    if (!window.confirm('Delete speaker profile "' + name + '"? This cannot be undone.')) return;
    try {
      await apiRequest('api/speakers/' + encodeURIComponent(name), { method: 'DELETE' });
      toast('Deleted "' + name + '"', 'success');
      await fetchSpeakers();
    } catch (e) {
      toast('Delete failed: ' + e.message, 'error');
    }
  }

  function validateNameField() {
    var input = $('speaker-name');
    var errEl = $('speaker-name-error');
    if (!input || !errEl) return true;
    var val = input.value.trim();
    var msg = validateSpeakerName(val);
    if (val.length === 0) {
      errEl.hidden = true;
      input.removeAttribute('aria-invalid');
      return false;
    }
    if (msg) {
      errEl.textContent = msg;
      errEl.hidden = false;
      input.setAttribute('aria-invalid', 'true');
      return false;
    }
    errEl.hidden = true;
    input.removeAttribute('aria-invalid');
    return true;
  }

  function updateEnrollButton() {
    var btn = $('enroll-btn');
    var clear = $('clear-samples-btn');
    var lbl = $('enroll-btn-label');
    var nameOk = validateSpeakerName((state.speakerName || '').trim()) == null;
    var enough = state.samples.length >= MIN_SAMPLES_FOR_ENROLL;
    if (btn) btn.disabled = !(nameOk && enough);
    if (clear) clear.disabled = state.samples.length === 0;
    if (lbl) {
      if (state.samples.length < MIN_SAMPLES_FOR_ENROLL) {
        lbl.textContent = 'Enroll Speaker (need ' + (MIN_SAMPLES_FOR_ENROLL - state.samples.length) + ' more)';
      } else {
        lbl.textContent = 'Enroll Speaker (' + state.samples.length + ' sample' + (state.samples.length === 1 ? '' : 's') + ')';
      }
    }
  }

  function computeConsistency() {
    if (state.samples.length < 2) return null;
    var rmsValues = state.samples.map(function (s) { return s.rms; });
    var mean = rmsValues.reduce(function (a, b) { return a + b; }, 0) / rmsValues.length;
    if (mean === 0) return { score: 0, label: 'Samples are silent — check microphone.', state: 'bad' };
    var variance = rmsValues.reduce(function (acc, v) { return acc + (v - mean) * (v - mean); }, 0) / rmsValues.length;
    var stdDev = Math.sqrt(variance);
    var cv = stdDev / mean;
    var consistency = Math.max(0, Math.min(1, 1 - cv));
    var label;
    var st;
    if (consistency >= 0.85) { label = 'Levels consistent.'; st = 'good'; }
    else if (consistency >= 0.6) { label = 'Levels vary (auto-normalized at enrollment).'; st = 'ok'; }
    else { label = 'Levels vary widely — check for clipping or near-silent samples.'; st = 'bad'; }
    return { score: consistency, label: label, state: st };
  }

  function renderQuality() {
    var row = $('quality-row');
    if (!row) return;
    var info = computeConsistency();
    if (!info) {
      row.hidden = true;
      return;
    }
    row.hidden = false;
    setText('quality-value', Math.round(info.score * 100) + '%');
    setText('quality-hint', info.label);
    var dot = $('quality-dot');
    if (dot) dot.setAttribute('data-state', info.state);
  }

  function renderSamples() {
    var list = $('samples-list');
    if (!list) return;
    clearChildren(list);
    if (!state.samples.length) {
      list.appendChild(el('div', { class: 'empty-state' }, [
        'No samples yet. Press ',
        el('strong', {}, ['Start Recording Sample']),
        ' to capture your voice.',
      ]));
      renderQuality();
      return;
    }
    state.samples.forEach(function (s, idx) {
      var canvas = el('canvas', { class: 'sample-waveform', width: 120, height: 32 });
      var row = el('div', { class: 'sample-row' }, [
        el('span', { class: 'sample-num' }, ['#' + (idx + 1)]),
        canvas,
        el('div', { class: 'sample-meta' }, [
          el('strong', {}, [s.durationSeconds.toFixed(2) + 's']),
          el('span', {}, ['peak ' + Math.round(s.peak * 100) + '% · rms ' + s.rms.toFixed(3)]),
        ]),
        el('button', {
          class: 'sample-discard',
          type: 'button',
          'aria-label': 'Discard sample ' + (idx + 1),
          title: 'Discard sample',
          onclick: function () { discardSample(idx); },
        }, ['✕']),
      ]);
      list.appendChild(row);
      requestAnimationFrame(function () { drawSampleThumb(canvas, s.int16); });
    });
    renderQuality();
  }

  function discardSample(idx) {
    state.samples.splice(idx, 1);
    renderSamples();
    updateEnrollButton();
  }

  function setRecordButtonRecording(recording) {
    var btn = $('record-btn');
    var lbl = $('record-btn-label');
    if (!btn) return;
    btn.setAttribute('aria-pressed', recording ? 'true' : 'false');
    btn.disabled = false;
    if (lbl) lbl.textContent = recording ? 'Stop Recording' : 'Start Recording Sample';
  }

  function setVuLevel(barId, level) {
    var bar = $(barId);
    if (!bar) return;
    var pct = Math.min(100, Math.max(0, Math.round(level * 400)));
    bar.style.width = pct + '%';
  }

  async function startSampleRecording() {
    if (state.recording) return;
    var nameInput = $('speaker-name');
    if (nameInput) {
      state.speakerName = nameInput.value.trim();
      localStorage.setItem('wls.lastSpeaker', state.speakerName);
    }
    state.recording = true;
    setRecordButtonRecording(true);
    announce('Recording sample…');
    try {
      var result = await recordFixed(ENROLL_RECORD_SECONDS, function (remaining) {
        setText('record-countdown', remaining.toFixed(1) + 's');
      }, function (level) {
        setVuLevel('vu-bar', level);
      });
      var int16 = floatToInt16(result.float32);
      state.samples.push({
        int16: int16,
        float32: result.float32,
        durationSeconds: result.durationSeconds,
        rms: rms(result.float32),
        peak: peak(result.float32),
      });
      toast('Captured sample ' + state.samples.length + ' (' + result.durationSeconds.toFixed(2) + 's)', 'success', 2200);
      renderSamples();
      updateEnrollButton();
    } catch (e) {
      toast(e.message, 'error');
    } finally {
      state.recording = false;
      setRecordButtonRecording(false);
      setText('record-countdown', ENROLL_RECORD_SECONDS.toFixed(1) + 's');
      setVuLevel('vu-bar', 0);
    }
  }

  async function submitEnrollment() {
    var name = (state.speakerName || '').trim();
    var err = validateSpeakerName(name);
    if (err) { toast(err, 'error'); return; }
    if (state.samples.length < MIN_SAMPLES_FOR_ENROLL) {
      toast('Record at least ' + MIN_SAMPLES_FOR_ENROLL + ' samples', 'warning');
      return;
    }
    var btn = $('enroll-btn');
    var prevLabel = $('enroll-btn-label') ? $('enroll-btn-label').textContent : '';
    if (btn) btn.disabled = true;
    setText('enroll-btn-label', 'Enrolling…');
    try {
      var fd = new FormData();
      state.samples.forEach(function (s, i) {
        var blob = new Blob([s.int16.buffer], { type: 'application/octet-stream' });
        fd.append('sample', blob, 'sample_' + i + '.pcm');
      });
      var resp = await apiRequest('api/speakers/' + encodeURIComponent(name) + '/enroll', {
        method: 'POST',
        body: fd,
      });
      toast('Enrolled "' + (resp.name || name) + '" with ' + (resp.samples_used || state.samples.length) + ' samples', 'success');
      state.samples = [];
      renderSamples();
      updateEnrollButton();
      await fetchSpeakers();
    } catch (e) {
      toast('Enrollment failed: ' + e.message, 'error');
    } finally {
      if (btn) btn.disabled = false;
      setText('enroll-btn-label', prevLabel || 'Enroll Speaker');
      updateEnrollButton();
    }
  }

  function clearSamples() {
    if (!state.samples.length) return;
    if (!window.confirm('Discard all ' + state.samples.length + ' samples?')) return;
    state.samples = [];
    renderSamples();
    updateEnrollButton();
  }

  function renderVerifyResults(result) {
    var container = $('verify-results');
    if (!container) return;
    clearChildren(container);
    if (!result) {
      container.appendChild(el('div', { class: 'empty-state' }, ['Run a test to see verification scores.']));
      return;
    }
    if (!result.scores || !Object.keys(result.scores).length) {
      container.appendChild(el('div', { class: 'empty-state' }, ['No enrolled speakers to compare against.']));
      return;
    }
    var threshold = typeof result.threshold === 'number' ? result.threshold : state.speakerThreshold;
    var names = Object.keys(result.scores).sort(function (a, b) { return result.scores[b] - result.scores[a]; });
    names.forEach(function (name) {
      container.appendChild(makeBarRow(name, result.scores[name], threshold, name === result.best_speaker));
    });
  }

  function makeBarRow(name, score, threshold, isBest) {
    var displayed = Math.max(0, Math.min(1, score));
    var aboveThreshold = score >= threshold;
    var fillClass = 'bar-fill ' + (aboveThreshold ? 'above' : 'below');
    return el('div', { class: 'speaker-bar' }, [
      el('span', { class: 'bar-name' + (isBest ? ' best-match' : '') }, [name]),
      el('div', { class: 'bar-track' }, [
        el('div', { class: fillClass, style: { width: (displayed * 100).toFixed(1) + '%', left: '0' } }),
        el('div', { class: 'bar-threshold', style: { left: (Math.max(0, Math.min(1, threshold)) * 100).toFixed(1) + '%' } }),
      ]),
      el('span', { class: 'bar-value' + (aboveThreshold ? ' above' : '') }, [score.toFixed(3)]),
    ]);
  }

  async function runVerifyRecording() {
    if (state.verifyRecording) return;
    state.verifyRecording = true;
    var btn = $('verify-record-btn');
    if (btn) {
      btn.disabled = true;
      btn.setAttribute('aria-pressed', 'true');
    }
    setText('verify-btn-label', 'Recording…');
    try {
      var result = await recordFixed(VERIFY_RECORD_SECONDS, function (remaining) {
        setText('verify-countdown', remaining.toFixed(1) + 's');
      }, function (level) {
        setVuLevel('vu-bar', level);
      });
      setText('verify-btn-label', 'Verifying…');
      var int16 = floatToInt16(result.float32);
      var resp = await apiRequest('api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: int16.buffer,
      });
      renderVerifyResults(resp);
      if (resp.accepted) {
        toast('Match: ' + resp.best_speaker + ' (' + (resp.best_score || 0).toFixed(3) + ')', 'success');
      } else if (resp.best_speaker) {
        toast('Best guess: ' + resp.best_speaker + ' (' + (resp.best_score || 0).toFixed(3) + ') — below threshold', 'warning');
      } else {
        toast('No match', 'warning');
      }
    } catch (e) {
      toast('Verify failed: ' + e.message, 'error');
    } finally {
      state.verifyRecording = false;
      if (btn) {
        btn.disabled = false;
        btn.setAttribute('aria-pressed', 'false');
      }
      setText('verify-btn-label', 'Record & Verify');
      setText('verify-countdown', VERIFY_RECORD_SECONDS.toFixed(1) + 's');
      setVuLevel('vu-bar', 0);
    }
  }

  function initEnrollTab() {
    var input = $('speaker-name');
    if (input) {
      input.value = state.speakerName;
      input.addEventListener('input', function () {
        state.speakerName = input.value.trim();
        localStorage.setItem('wls.lastSpeaker', state.speakerName);
        validateNameField();
        updateEnrollButton();
      });
      input.addEventListener('blur', validateNameField);
    }
    var recBtn = $('record-btn');
    if (recBtn) recBtn.addEventListener('click', startSampleRecording);

    var enrollBtn = $('enroll-btn');
    if (enrollBtn) enrollBtn.addEventListener('click', submitEnrollment);

    var clearBtn = $('clear-samples-btn');
    if (clearBtn) clearBtn.addEventListener('click', clearSamples);

    var verifyBtn = $('verify-record-btn');
    if (verifyBtn) verifyBtn.addEventListener('click', runVerifyRecording);

    var refresh = $('refresh-speakers');
    if (refresh) refresh.addEventListener('click', fetchSpeakers);

    setText('record-countdown', ENROLL_RECORD_SECONDS.toFixed(1) + 's');
    setText('verify-countdown', VERIFY_RECORD_SECONDS.toFixed(1) + 's');
    updateEnrollButton();
    renderSamples();
    initMicSource();
    var vpeSel = $('enroll-vpe-select');
    if (vpeSel) vpeSel.addEventListener('change', function () { state.lastEnrollDeviceId = vpeSel.value; });
  }

  function setWsStatus(status, label) {
    var dot = $('ws-status-dot');
    var lab = $('ws-status-label');
    if (dot) dot.setAttribute('data-status', status);
    if (lab) lab.textContent = label;
  }

  function showReconnectBanner(show) {
    var banner = $('reconnect-banner');
    if (banner) banner.hidden = !show;
  }

  function renderChartLegend() {
    var legend = $('chart-legend');
    if (!legend) return;
    clearChildren(legend);
    state.wakeWords.forEach(function (kw, i) {
      var color = KEYWORD_COLORS[i % KEYWORD_COLORS.length];
      legend.appendChild(el('span', { class: 'legend-entry' }, [
        el('span', { class: 'legend-stripe', style: { background: color } }),
        el('span', {}, [kw]),
      ]));
    });
    legend.appendChild(el('span', { class: 'legend-entry' }, [
      el('span', { class: 'legend-stripe', style: { background: 'var(--color-warning)' } }),
      el('span', {}, ['threshold']),
    ]));
  }

  function drawScoreChart() {
    var canvas = $('score-chart');
    if (!canvas) return;
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    var w = Math.max(1, Math.round(rect.width * dpr));
    var h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;

    var ctx2d = canvas.getContext('2d');
    ctx2d.clearRect(0, 0, w, h);

    var pad = { left: 32 * dpr, right: 10 * dpr, top: 10 * dpr, bottom: 18 * dpr };
    var plotW = w - pad.left - pad.right;
    var plotH = h - pad.top - pad.bottom;

    var theme = document.documentElement.getAttribute('data-theme');
    var gridColor = theme === 'light' ? 'rgba(0,0,0,0.07)' : 'rgba(255,255,255,0.07)';
    var axisColor = theme === 'light' ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.4)';

    ctx2d.strokeStyle = gridColor;
    ctx2d.lineWidth = 1 * dpr;
    ctx2d.font = (11 * dpr) + 'px ' + 'ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx2d.fillStyle = axisColor;
    ctx2d.textBaseline = 'middle';
    for (var yv = 0; yv <= 1; yv += 0.25) {
      var y = pad.top + plotH - yv * plotH;
      ctx2d.beginPath();
      ctx2d.moveTo(pad.left, y);
      ctx2d.lineTo(pad.left + plotW, y);
      ctx2d.stroke();
      ctx2d.fillText(yv.toFixed(2), 4 * dpr, y);
    }

    var thresholds = state.sensitivities && state.sensitivities.length ? state.sensitivities : [0.5];
    var primaryThresh = thresholds[0];
    var warningColor = getComputedStyle(document.documentElement).getPropertyValue('--color-warning').trim() || '#ff9800';
    ctx2d.save();
    ctx2d.setLineDash([5 * dpr, 4 * dpr]);
    ctx2d.strokeStyle = warningColor;
    ctx2d.lineWidth = 1.5 * dpr;
    var yt = pad.top + plotH - primaryThresh * plotH;
    ctx2d.beginPath();
    ctx2d.moveTo(pad.left, yt);
    ctx2d.lineTo(pad.left + plotW, yt);
    ctx2d.stroke();
    ctx2d.restore();

    ctx2d.strokeStyle = axisColor;
    ctx2d.lineWidth = 1 * dpr;
    ctx2d.beginPath();
    ctx2d.moveTo(pad.left, pad.top);
    ctx2d.lineTo(pad.left, pad.top + plotH);
    ctx2d.lineTo(pad.left + plotW, pad.top + plotH);
    ctx2d.stroke();

    var now = nowSec();
    var oldest = now - CHART_WINDOW_SECONDS;
    var hist = state.scoreHistory.filter(function (p) { return p.t >= oldest; });
    state.scoreHistory = hist;

    if (!hist.length) {
      ctx2d.fillStyle = axisColor;
      ctx2d.textAlign = 'center';
      ctx2d.textBaseline = 'middle';
      ctx2d.fillText('Streaming idle. Press Start to capture scores.', pad.left + plotW / 2, pad.top + plotH / 2);
      ctx2d.textAlign = 'start';
      return;
    }

    state.wakeWords.forEach(function (kw, kwIdx) {
      var color = KEYWORD_COLORS[kwIdx % KEYWORD_COLORS.length];
      ctx2d.strokeStyle = color;
      ctx2d.lineWidth = 2 * dpr;
      ctx2d.beginPath();
      var started = false;
      hist.forEach(function (point) {
        var s = point.scores && typeof point.scores[kw] === 'number' ? point.scores[kw] : null;
        if (s == null) return;
        var x = pad.left + ((point.t - oldest) / CHART_WINDOW_SECONDS) * plotW;
        var y = pad.top + plotH - Math.max(0, Math.min(1, s)) * plotH;
        if (!started) { ctx2d.moveTo(x, y); started = true; }
        else { ctx2d.lineTo(x, y); }
      });
      if (started) ctx2d.stroke();

      var lastPoint = hist[hist.length - 1];
      var lastScore = lastPoint && lastPoint.scores ? lastPoint.scores[kw] : null;
      if (typeof lastScore === 'number') {
        var lx = pad.left + ((lastPoint.t - oldest) / CHART_WINDOW_SECONDS) * plotW;
        var ly = pad.top + plotH - Math.max(0, Math.min(1, lastScore)) * plotH;
        ctx2d.fillStyle = color;
        ctx2d.beginPath();
        ctx2d.arc(lx, ly, 3 * dpr, 0, Math.PI * 2);
        ctx2d.fill();
      }
    });

    var speakerNames = {};
    hist.forEach(function (p) {
      if (!p.speakerScores) return;
      Object.keys(p.speakerScores).forEach(function (n) { speakerNames[n] = true; });
    });
    var speakerList = Object.keys(speakerNames);
    speakerList.forEach(function (name, sIdx) {
      var color = SPEAKER_COLORS[sIdx % SPEAKER_COLORS.length];
      ctx2d.strokeStyle = color;
      ctx2d.lineWidth = 1.5 * dpr;
      ctx2d.setLineDash([3 * dpr, 3 * dpr]);
      ctx2d.beginPath();
      var startedSp = false;
      hist.forEach(function (point) {
        var s = point.speakerScores && typeof point.speakerScores[name] === 'number' ? point.speakerScores[name] : null;
        if (s == null) return;
        var x = pad.left + ((point.t - oldest) / CHART_WINDOW_SECONDS) * plotW;
        var y = pad.top + plotH - Math.max(0, Math.min(1, s)) * plotH;
        if (!startedSp) { ctx2d.moveTo(x, y); startedSp = true; }
        else { ctx2d.lineTo(x, y); }
      });
      if (startedSp) ctx2d.stroke();
      ctx2d.setLineDash([]);
    });

    var vadBarTop = pad.top + plotH + 2 * dpr;
    var vadBarH = 6 * dpr;
    if (vadBarTop + vadBarH < h - 12 * dpr) {
      var vadColorOn = getComputedStyle(document.documentElement).getPropertyValue('--color-success').trim() || '#4caf50';
      ctx2d.fillStyle = gridColor;
      ctx2d.fillRect(pad.left, vadBarTop, plotW, vadBarH);
      ctx2d.fillStyle = vadColorOn;
      hist.forEach(function (point, idx) {
        if (!point.vad) return;
        var x = pad.left + ((point.t - oldest) / CHART_WINDOW_SECONDS) * plotW;
        var nextX = idx + 1 < hist.length
          ? pad.left + ((hist[idx + 1].t - oldest) / CHART_WINDOW_SECONDS) * plotW
          : pad.left + plotW;
        ctx2d.fillRect(x, vadBarTop, Math.max(2 * dpr, nextX - x), vadBarH);
      });
    }

    ctx2d.textAlign = 'start';
    ctx2d.fillStyle = axisColor;
    ctx2d.fillText('-' + CHART_WINDOW_SECONDS + 's', pad.left, h - 4 * dpr);
    ctx2d.textAlign = 'end';
    ctx2d.fillText('now', pad.left + plotW, h - 4 * dpr);
    ctx2d.textAlign = 'start';
  }

  function flashChart() {
    var card = $('score-chart-card');
    if (!card) return;
    card.classList.add('flash');
    setTimeout(function () { card.classList.remove('flash'); }, 1000);
  }

  function renderSpeakerBars(scores, best, accepted, threshold) {
    var container = $('speaker-bars');
    if (!container) return;
    clearChildren(container);
    if (!scores || !Object.keys(scores).length) {
      container.appendChild(el('div', { class: 'empty-state' }, ['No verification results yet.']));
      return;
    }
    var thr = typeof threshold === 'number' ? threshold : state.debugSpeakerThreshold;
    var names = Object.keys(scores).sort(function (a, b) { return scores[b] - scores[a]; });
    names.forEach(function (name) {
      container.appendChild(makeBarRow(name, scores[name], thr, name === best));
    });
    setText('best-match-display', best ? (best + ' · ' + (accepted ? 'accepted' : 'below threshold')) : '—');
  }

  function pushTriggerLog(entry) {
    state.triggerLog.unshift(entry);
    if (state.triggerLog.length > MAX_TRIGGER_LOG) state.triggerLog.length = MAX_TRIGGER_LOG;
    renderTriggerLog();
  }

  function renderTriggerLog() {
    var container = $('event-log');
    if (!container) return;
    clearChildren(container);
    if (!state.triggerLog.length) {
      container.appendChild(el('div', { class: 'empty-state' }, ['Triggers will appear here when the wake word fires.']));
      return;
    }
    state.triggerLog.forEach(function (ev) {
      var d = new Date(ev.t * 1000);
      var hh = String(d.getHours()).padStart(2, '0');
      var mm = String(d.getMinutes()).padStart(2, '0');
      var ss = String(d.getSeconds()).padStart(2, '0');
      var time = hh + ':' + mm + ':' + ss;
      var status;
      if (ev.accepted === true) status = 'accepted';
      else if (ev.accepted === false) status = 'rejected';
      else status = 'fired';
      var statusText = status === 'fired' ? 'fired' : (status === 'accepted' ? '✓ accepted' : '✗ rejected');
      container.appendChild(el('div', { class: 'event-row' }, [
        el('span', { class: 'ev-time' }, [time]),
        el('span', { class: 'ev-keyword' }, [ev.keyword || '—']),
        el('span', { class: 'ev-speaker' }, [ev.speaker ? '@' + ev.speaker + (ev.score != null ? ' ' + ev.score.toFixed(3) : '') : '']),
        el('span', { class: 'ev-status', 'data-state': status }, [statusText]),
      ]));
    });
  }

  function setMidSentence(rejected) {
    state.midSentenceLast = rejected;
    var node = $('mid-sentence-value');
    if (!node) return;
    if (rejected == null) {
      node.setAttribute('data-state', 'unknown');
      node.textContent = '— never seen —';
    } else if (rejected) {
      node.setAttribute('data-state', 'rejected');
      node.textContent = 'true · rejected';
    } else {
      node.setAttribute('data-state', 'accepted');
      node.textContent = 'false · passed';
    }
  }

  function handleScoreStreamMessage(msg) {
    if (msg.event === 'hello') {
      state.wakeWords = msg.wake_words || [];
      state.sensitivities = msg.sensitivities && msg.sensitivities.length ? msg.sensitivities : state.wakeWords.map(function () { return 0.5; });
      state.predictIntervalSeconds = typeof msg.predict_interval_seconds === 'number' ? msg.predict_interval_seconds : 0.5;
      state.debugSpeakerThreshold = typeof msg.speaker_threshold === 'number' ? msg.speaker_threshold : state.debugSpeakerThreshold;
      state.debugVerifierActive = !!msg.speaker_verifier_active;
      setText('threshold-display', (state.sensitivities[0] || 0.5).toFixed(2));
      setText('interval-display', state.predictIntervalSeconds.toFixed(2) + 's');
      setText('speaker-threshold-display', state.debugSpeakerThreshold.toFixed(2));
      renderChartLegend();
    } else if (msg.event === 'source') {
      state.debugSource = msg.source;
      state.debugSourceDeviceId = msg.device_id || null;
      var note = $('source-claim-note');
      if (note) note.hidden = msg.source !== 'voice-pe';
    } else if (msg.event === 'ww_score') {
      var point = {
        t: nowSec(),
        scores: msg.scores || {},
        peak: msg.peak || 0,
        rms: typeof msg.rms === 'number' ? msg.rms : 0,
        vad: !!msg.vad_active,
        speakerScores: msg.speaker_scores || {},
        fired: msg.fired_keyword || null,
      };
      state.scoreHistory.push(point);
      var topScore = 0;
      var topKw = null;
      Object.keys(point.scores).forEach(function (k) {
        if (point.scores[k] > topScore) {
          topScore = point.scores[k];
          topKw = k;
        }
      });
      setText('latest-score', topScore.toFixed(3) + (topKw ? ' (' + topKw + ')' : ''));
      if (msg.fired_keyword) {
        setText('latest-keyword', msg.fired_keyword);
        flashChart();
      }
    }
  }

  function handleDebugStreamMessage(msg) {
    if (msg.event === 'hello') {
      state.debugSpeakerThreshold = typeof msg.speaker_threshold === 'number' ? msg.speaker_threshold : state.debugSpeakerThreshold;
      state.debugVerifierActive = !!msg.speaker_verifier_active;
      setText('speaker-threshold-display', state.debugSpeakerThreshold.toFixed(2));
    } else if (msg.event === 'ww_score' && msg.fired) {
      state.lastFiredKeyword = msg.keyword;
      pushTriggerLog({
        t: nowSec(),
        keyword: msg.keyword,
        accepted: state.debugVerifierActive ? null : true,
        speaker: null,
        score: null,
      });
      setText('latest-keyword', msg.keyword);
    } else if (msg.event === 'speaker') {
      state.lastSpeakerScores = msg.scores || {};
      state.lastBestMatch = msg.best || null;
      renderSpeakerBars(msg.scores, msg.best, !!msg.accepted, state.debugSpeakerThreshold);
      var topEntry = state.triggerLog[0];
      var bestScore = msg.best && msg.scores ? msg.scores[msg.best] : null;
      if (topEntry && !msg.triggered_manually && (nowSec() - topEntry.t) < 4) {
        topEntry.accepted = !!msg.accepted;
        topEntry.speaker = msg.best || null;
        topEntry.score = bestScore;
      } else {
        pushTriggerLog({
          t: nowSec(),
          keyword: msg.triggered_manually ? 'manual verify' : (state.lastFiredKeyword || '—'),
          accepted: !!msg.accepted,
          speaker: msg.best || null,
          score: bestScore,
        });
      }
      if (msg.accepted) {
        state.lastAcceptedKeyword = state.lastFiredKeyword || '(manual)';
        setText('last-accepted-value', state.lastAcceptedKeyword + ' · ' + (msg.best || '?'));
      }
      renderTriggerLog();
    } else if (msg.event === 'mid_sentence') {
      setMidSentence(!!msg.rejected);
    } else if (msg.event === 'error') {
      toast('Stream error: ' + (msg.message || 'unknown'), 'error');
    }
  }

  async function startDebugStream() {
    if (state.streaming) return;
    state.streaming = true;
    setText('debug-toggle-label', 'Stop Streaming');
    setWsStatus('connecting', 'Opening microphone…');

    var debugChunkSender;
    var scoreChunkSender;

    if (state.debugSource === 'browser') {
      try {
        state.debugAudio = await createMicCapture({
          onSamples: function (down, raw) {
            var lvl = rms(down);
            state.debugVuLevel = lvl;
            setVuLevel('debug-vu-bar', lvl);
          },
          onChunk16k: function (int16) {
            if (debugChunkSender) debugChunkSender(int16.buffer);
            if (scoreChunkSender) scoreChunkSender(int16.buffer);
          },
        });
        setText('debug-audio-info', 'Microphone ' + state.debugAudio.sourceRate + ' Hz → 16 kHz int16');
      } catch (e) {
        toast(e.message, 'error');
        state.streaming = false;
        setText('debug-toggle-label', 'Start Streaming');
        setWsStatus('idle', 'Idle.');
        return;
      }
    } else {
      setText('debug-audio-info', 'Streaming from Voice PE device');
    }

    setWsStatus('connecting', 'Connecting to score stream…');

    function subscribeIfVoicePE(ws) {
      if (state.debugSource === 'voice-pe' && state.debugSourceDeviceId) {
        ws.send(JSON.stringify({ cmd: 'subscribe_device', device_id: state.debugSourceDeviceId }));
      }
    }
    state.debugWS = createReconnectingWS('api/debug-stream', {
      onConnecting: function () { showReconnectBanner(false); },
      onOpen: function () {
        setWsStatus('connected', state.debugSource === 'voice-pe' ? 'Streaming from Voice PE.' : 'Streaming live audio.');
        showReconnectBanner(false);
        subscribeIfVoicePE(state.debugWS);
      },
      onClose: function () {
        if (state.streaming) {
          setWsStatus('connecting', 'Reconnecting debug stream…');
          showReconnectBanner(true);
        }
      },
      onMessage: handleDebugStreamMessage,
    });

    state.scoreWS = createReconnectingWS('api/score-stream', {
      onOpen: function () { subscribeIfVoicePE(state.scoreWS); },
      onMessage: handleScoreStreamMessage,
    });

    debugChunkSender = function (buf) { state.debugWS && state.debugWS.send(buf); };
    scoreChunkSender = function (buf) { state.scoreWS && state.scoreWS.send(buf); };
  }

  function stopDebugStream() {
    if (!state.streaming) return;
    state.streaming = false;
    setText('debug-toggle-label', 'Start Streaming');
    setWsStatus('idle', 'Idle.');
    showReconnectBanner(false);
    if (state.debugClaimedDeviceId) {
      apiRequest('api/devices/' + encodeURIComponent(state.debugClaimedDeviceId) + '/debug-release', { method: 'POST' }).catch(function () {});
      state.debugClaimedDeviceId = null;
    }
    if (state.debugWS) { try { state.debugWS.close(); } catch (_) {} state.debugWS = null; }
    if (state.scoreWS) { try { state.scoreWS.close(); } catch (_) {} state.scoreWS = null; }
    if (state.debugAudio) { try { state.debugAudio.stop(); } catch (_) {} state.debugAudio = null; }
    setVuLevel('debug-vu-bar', 0);
    setText('debug-audio-info', 'Microphone idle');
  }

  function toggleDebugStream() {
    if (state.streaming) stopDebugStream();
    else startDebugStream();
  }

  function initDebugTab() {
    var toggle = $('debug-toggle');
    if (toggle) toggle.addEventListener('click', toggleDebugStream);

    var verifyNow = $('verify-now-btn');
    if (verifyNow) verifyNow.addEventListener('click', function () {
      if (state.debugWS && state.debugWS.isOpen()) {
        state.debugWS.send(JSON.stringify({ cmd: 'verify_now' }));
        toast('Verify request sent', 'info', 1500);
      } else {
        toast('Start streaming first to verify audio', 'warning');
      }
    });

    var clearLog = $('clear-log-btn');
    if (clearLog) clearLog.addEventListener('click', function () {
      state.triggerLog = [];
      renderTriggerLog();
    });

    var manualRetry = $('manual-retry');
    if (manualRetry) manualRetry.addEventListener('click', function () {
      if (state.debugWS) state.debugWS.reconnectNow();
      if (state.scoreWS) state.scoreWS.reconnectNow();
    });

    setMidSentence(null);

    function chartLoop() {
      drawScoreChart();
      requestAnimationFrame(chartLoop);
    }
    requestAnimationFrame(chartLoop);

    window.addEventListener('beforeunload', stopDebugStream);
  }

  state.serverSamples = [];
  state.serverSamplesFilter = 'all';
  state.serverSamplesLimit = 50;
  state.serverSamplesTotal = 0;
  state.selectedSampleIds = {};
  state.lastClickedSampleId = null;
  state.devices = [];
  state.devicePollTimer = null;
  state.samplesPollTimer = null;
  state.debugSource = 'browser';
  state.debugSourceDeviceId = null;
  state.debugClaimedDeviceId = null;
  state.micSource = 'browser';
  state.selectedDeviceId = null;
  state.deviceCaptureDuration = {};
  state.deviceCaptureLabel = {};

  function timeAgo(unixSec) {
    if (!unixSec) return '—';
    var d = Math.max(0, Math.floor(Date.now() / 1000 - unixSec));
    if (d < 5) return 'just now';
    if (d < 60) return d + 's ago';
    if (d < 3600) return Math.floor(d / 60) + 'm ago';
    if (d < 86400) return Math.floor(d / 3600) + 'h ago';
    return Math.floor(d / 86400) + 'd ago';
  }

  function formatBytes(n) {
    if (!n) return '0 B';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / 1024 / 1024).toFixed(1) + ' MB';
  }

  function deviceLabel(dev) {
    return dev.remote_addr || ('VoicePE-' + (dev.id || '').slice(0, 6));
  }

  function sourceBadgeClass(source) {
    if (source === 'manual') return 'source-badge source-manual';
    if (source === 'auto-below-threshold') return 'source-badge source-near-miss';
    if (source && source.indexOf('rejected') !== -1) return 'source-badge source-rejected';
    return 'source-badge source-auto';
  }

  function sourceBadgeText(source) {
    if (source === 'manual') return 'Manual';
    if (source === 'auto-rejected-speaker') return 'Speaker-rej';
    if (source === 'auto-below-threshold') return 'Near-miss';
    return 'Triggered';
  }

  function labelBadgeClass(label) {
    if (label === 'fp') return 'label-badge label-fp';
    if (label === 'tp') return 'label-badge label-tp';
    return 'label-badge label-none';
  }

  function labelBadgeText(label) {
    if (label === 'fp') return 'FP';
    if (label === 'tp') return 'TP';
    return '—';
  }

  async function pollDevices() {
    try {
      var data = await apiRequest('api/devices');
      state.devices = data.devices || [];
    } catch (e) {
      state.devices = [];
    }
    renderDeviceList();
    renderEnrollDeviceSelect();
  }

  function renderDeviceList() {
    var box = $('device-list');
    if (!box) return;
    setText('vpe-session-count', state.devices.length + ' active session(s)');
    box.innerHTML = '';
    if (state.devices.length === 0) {
      box.appendChild(el('div', { class: 'empty-state' }, [
        'No active Voice PE sessions. The device must be actively listening (have an open Wyoming session) to capture from it. Wake it first by saying the wake word, then retry within ~30 seconds.'
      ]));
      return;
    }
    state.devices.forEach(function (dev) {
      var devId = dev.id;
      var dur = state.deviceCaptureDuration[devId] || 5;
      var preLabel = state.deviceCaptureLabel[devId] || '';
      var detecting = !!dev.is_detecting;
      var row = el('div', { class: 'device-row' });
      row.appendChild(el('div', { class: 'device-name' }, [deviceLabel(dev)]));
      row.appendChild(el('span', {
        class: 'status-badge ' + (detecting ? 'status-on' : 'status-off')
      }, [detecting ? 'Detecting' : 'Idle']));

      var durInput = el('input', {
        type: 'number', class: 'dur-input', min: '0.5', max: '30', step: '0.5',
        value: String(dur),
        oninput: function (e) {
          var v = parseFloat(e.target.value);
          if (!isNaN(v)) state.deviceCaptureDuration[devId] = v;
        },
      });
      row.appendChild(durInput);

      var labelSel = el('div', { class: 'mini-segmented', title: 'Pre-label captured sample' });
      [['', '—'], ['tp', 'TP'], ['fp', 'FP']].forEach(function (pair) {
        var btn = el('button', {
          type: 'button',
          class: 'mini-seg-button' + (preLabel === pair[0] ? ' active' : ''),
          onclick: function () {
            state.deviceCaptureLabel[devId] = pair[0];
            renderDeviceList();
          },
        }, [pair[1]]);
        labelSel.appendChild(btn);
      });
      row.appendChild(labelSel);

      var recBtn = el('button', {
        class: 'primary-button device-record-btn',
        type: 'button',
        disabled: !detecting || dev.is_capturing,
        onclick: function () { captureFromDevice(devId); },
      }, [dev.is_capturing ? 'Capturing…' : ('Record ' + dur + 's')]);
      row.appendChild(recBtn);
      box.appendChild(row);
    });
  }

  async function captureFromDevice(devId) {
    var dur = state.deviceCaptureDuration[devId] || 5;
    var label = state.deviceCaptureLabel[devId] || '';
    var q = 'duration=' + dur + (label ? '&label=' + label : '');
    state.devices.forEach(function (d) { if (d.id === devId) d.is_capturing = true; });
    renderDeviceList();
    try {
      var res = await apiRequest('api/devices/' + encodeURIComponent(devId) + '/capture?' + q, { method: 'POST' });
      toast('Sample saved: ' + (res.sample && res.sample.id || '?'), 'success', 2000);
      loadServerSamples();
    } catch (e) {
      toast('Capture failed: ' + e.message, 'error', 4000);
    } finally {
      state.devices.forEach(function (d) { if (d.id === devId) d.is_capturing = false; });
      renderDeviceList();
    }
  }

  async function loadServerSamples(append) {
    var box = $('server-samples-list');
    if (!box) return;
    if (!append) {
      state.serverSamplesLimit = 50;
    } else {
      state.serverSamplesLimit += 50;
    }
    try {
      var data = await apiRequest('api/samples?label=' + state.serverSamplesFilter + '&limit=' + state.serverSamplesLimit);
      var nextSamples = data.samples || [];
      var nextTotal = data.total_size_bytes || 0;
      var prevKey = JSON.stringify((state.serverSamples || []).map(function (s) { return [s.id, s.label]; }));
      var nextKey = JSON.stringify(nextSamples.map(function (s) { return [s.id, s.label]; }));
      state.serverSamples = nextSamples;
      state.serverSamplesTotal = nextTotal;
      var maxN = data.max_samples || 0;
      setText('samples-size-info',
        formatBytes(state.serverSamplesTotal) + ' · ' + nextSamples.length + (maxN ? ' / ' + maxN + ' max' : ''));
      if (prevKey !== nextKey || box.children.length === 0 || box.firstChild.classList.contains('empty-state')) {
        renderServerSamplesList();
      }
    } catch (e) {
      box.innerHTML = '';
      box.appendChild(el('div', { class: 'empty-state' }, ['Failed to load samples: ' + e.message]));
    }
  }

  function renderServerSamplesList() {
    var box = $('server-samples-list');
    if (!box) return;
    box.innerHTML = '';
    var samples = state.serverSamples;
    if (state.serverSamplesFilter === 'unlabeled') {
      samples = samples.filter(function (s) { return s.label == null; });
    }
    if (samples.length === 0) {
      box.appendChild(el('div', { class: 'empty-state' }, [
        'No samples match this filter. Triggers are auto-captured as you use the wake word.'
      ]));
      return;
    }
    samples.forEach(function (s) { box.appendChild(renderServerSampleRow(s)); });
    var moreRow = $('load-more-row');
    if (moreRow) moreRow.hidden = state.serverSamples.length < state.serverSamplesLimit;
  }

  function scoreBar(score, threshold) {
    var pct = Math.max(0, Math.min(1, score || 0)) * 100;
    var bar = el('div', { class: 'score-bar-wrap' });
    bar.appendChild(el('div', { class: 'score-bar-track' }, [
      el('div', { class: 'score-bar-fill', style: { width: pct.toFixed(1) + '%' } }),
      (threshold != null) ? el('div', { class: 'score-bar-threshold', style: { left: (threshold * 100).toFixed(1) + '%' } }) : null,
    ]));
    bar.appendChild(el('span', { class: 'score-bar-value' }, [score != null ? score.toFixed(3) : '—']));
    return bar;
  }

  function renderServerSampleRow(s) {
    var row = el('div', {
      class: 'sample-row' + (state.selectedSampleIds[s.id] ? ' selected' : ''),
      'data-sample-id': s.id,
    });
    var checkbox = el('input', {
      type: 'checkbox',
      class: 'sample-checkbox',
      checked: !!state.selectedSampleIds[s.id],
      onclick: function (e) { toggleSampleSelection(s.id, e); },
    });
    row.appendChild(checkbox);
    row.appendChild(el('div', { class: 'col-time' }, [timeAgo(s.timestamp)]));
    row.appendChild(el('span', { class: sourceBadgeClass(s.source) }, [sourceBadgeText(s.source)]));
    row.appendChild(el('div', { class: 'col-keyword' }, [s.keyword || '—']));
    row.appendChild(scoreBar(s.score, null));
    row.appendChild(el('div', { class: 'col-speaker' }, [s.speaker || '—']));

    var audio = document.createElement('audio');
    audio.controls = true;
    audio.preload = 'metadata';
    audio.src = 'api/samples/' + encodeURIComponent(s.id) + '/audio.wav';
    audio.className = 'sample-audio';
    row.appendChild(audio);

    var actions = el('div', { class: 'sample-actions' });
    var labelGroup = el('div', { class: 'label-segmented' });
    [['unassigned', null, '—'], ['tp', 'tp', 'TP'], ['fp', 'fp', 'FP']].forEach(function (opt) {
      var active = (opt[1] === null && s.label == null) || s.label === opt[1];
      labelGroup.appendChild(el('button', {
        class: 'seg-mini' + (active ? ' active label-' + (opt[1] || 'none') : ''),
        type: 'button',
        onclick: function () { setSampleLabel(s.id, opt[1]); },
      }, [opt[2]]));
    });
    actions.appendChild(labelGroup);
    actions.appendChild(el('button', {
      class: 'text-button',
      type: 'button',
      onclick: function () { retestSample(s.id); },
    }, ['Retest']));
    actions.appendChild(el('button', {
      class: 'text-button danger',
      type: 'button',
      onclick: function () { deleteSample(s.id); },
    }, ['Delete']));
    row.appendChild(actions);
    return row;
  }

  async function setSampleLabel(id, label) {
    try {
      await apiRequest('api/samples/' + encodeURIComponent(id) + '/label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: label }),
      });
      state.serverSamples.forEach(function (s) { if (s.id === id) s.label = label; });
      renderServerSamplesList();
    } catch (e) {
      toast('Label update failed: ' + e.message, 'error', 3000);
    }
  }

  async function deleteSample(id) {
    if (!confirm('Delete sample ' + id + '?')) return;
    try {
      await apiRequest('api/samples/' + encodeURIComponent(id), { method: 'DELETE' });
      state.serverSamples = state.serverSamples.filter(function (s) { return s.id !== id; });
      renderServerSamplesList();
    } catch (e) {
      toast('Delete failed: ' + e.message, 'error', 3000);
    }
  }

  async function retestSample(id) {
    try {
      var data = await apiRequest('api/samples/' + encodeURIComponent(id) + '/retest');
      var existing = document.querySelector('.retest-panel[data-for="' + id + '"]');
      if (existing) { existing.remove(); return; }
      var row = document.querySelector('.sample-row[data-sample-id="' + id + '"]');
      if (!row) return;
      var panel = el('div', { class: 'retest-panel', 'data-for': id });
      panel.appendChild(el('div', { class: 'retest-section-title' }, ['Current wake-word scores']));
      Object.keys(data.ww_scores || {}).forEach(function (kw) {
        var line = el('div', { class: 'retest-line' });
        line.appendChild(el('span', { class: 'retest-name' }, [kw]));
        line.appendChild(scoreBar(data.ww_scores[kw], null));
        panel.appendChild(line);
      });
      if (data.speaker_scores && Object.keys(data.speaker_scores).length) {
        panel.appendChild(el('div', { class: 'retest-section-title' }, ['Speaker scores (threshold ' + (data.speaker_threshold || 0.25).toFixed(2) + ')']));
        Object.keys(data.speaker_scores).forEach(function (name) {
          var line = el('div', { class: 'retest-line' });
          line.appendChild(el('span', { class: 'retest-name' }, [name]));
          line.appendChild(scoreBar(data.speaker_scores[name], data.speaker_threshold));
          panel.appendChild(line);
        });
      }
      row.parentNode.insertBefore(panel, row.nextSibling);
    } catch (e) {
      toast('Retest failed: ' + e.message, 'error', 3000);
    }
  }

  function toggleSampleSelection(id, event) {
    var isShift = !!(event && event.shiftKey);
    if (isShift && state.lastClickedSampleId && state.lastClickedSampleId !== id) {
      var visible = state.serverSamples.map(function (s) { return s.id; });
      var i1 = visible.indexOf(state.lastClickedSampleId);
      var i2 = visible.indexOf(id);
      if (i1 >= 0 && i2 >= 0) {
        var lo = Math.min(i1, i2), hi = Math.max(i1, i2);
        for (var i = lo; i <= hi; i++) {
          state.selectedSampleIds[visible[i]] = true;
        }
      }
    } else {
      if (state.selectedSampleIds[id]) delete state.selectedSampleIds[id];
      else state.selectedSampleIds[id] = true;
    }
    state.lastClickedSampleId = id;
    renderServerSamplesList();
    renderBulkBar();
  }

  function clearSelection() {
    state.selectedSampleIds = {};
    state.lastClickedSampleId = null;
    renderServerSamplesList();
    renderBulkBar();
  }

  function selectAllVisible() {
    state.serverSamples.forEach(function (s) { state.selectedSampleIds[s.id] = true; });
    renderServerSamplesList();
    renderBulkBar();
  }

  function selectedIds() {
    return Object.keys(state.selectedSampleIds);
  }

  async function bulkLabel(label) {
    var ids = selectedIds();
    if (!ids.length) return;
    try {
      var res = await apiRequest('api/samples/bulk-label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids: ids, label: label }),
      });
      state.serverSamples.forEach(function (s) {
        if (state.selectedSampleIds[s.id]) s.label = label;
      });
      toast('Updated ' + (res.updated || 0) + ' samples', 'success', 2000);
      renderServerSamplesList();
    } catch (e) {
      toast('Bulk label failed: ' + e.message, 'error', 3000);
    }
  }

  async function bulkDelete() {
    var ids = selectedIds();
    if (!ids.length) return;
    if (!confirm('Delete ' + ids.length + ' samples?')) return;
    try {
      var res = await apiRequest('api/samples/bulk-delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids: ids }),
      });
      state.serverSamples = state.serverSamples.filter(function (s) { return !state.selectedSampleIds[s.id]; });
      state.selectedSampleIds = {};
      toast('Deleted ' + (res.deleted || 0) + ' samples', 'success', 2000);
      renderServerSamplesList();
      renderBulkBar();
    } catch (e) {
      toast('Bulk delete failed: ' + e.message, 'error', 3000);
    }
  }

  function renderBulkBar() {
    var bar = $('bulk-action-bar');
    if (!bar) return;
    var count = selectedIds().length;
    bar.hidden = count === 0;
    setText('bulk-action-count', count + ' selected');
  }

  function initSamplesTab() {
    var pills = document.querySelectorAll('#samples-filter-pills .pill');
    pills.forEach(function (p) {
      p.addEventListener('click', function () {
        pills.forEach(function (pp) {
          pp.classList.remove('active');
          pp.setAttribute('aria-selected', 'false');
        });
        p.classList.add('active');
        p.setAttribute('aria-selected', 'true');
        state.serverSamplesFilter = p.dataset.filter;
        loadServerSamples();
      });
    });
    var refresh = $('refresh-samples-btn');
    if (refresh) refresh.addEventListener('click', function () { loadServerSamples(); pollDevices(); });
    var loadMore = $('load-more-samples-btn');
    if (loadMore) loadMore.addEventListener('click', function () { loadServerSamples(true); });
    var selAll = $('select-all-btn');
    if (selAll) selAll.addEventListener('click', selectAllVisible);
    var clearSel = $('clear-selection-btn');
    if (clearSel) clearSel.addEventListener('click', clearSelection);
    var bulkFp = $('bulk-mark-fp');
    if (bulkFp) bulkFp.addEventListener('click', function () { bulkLabel('fp'); });
    var bulkTp = $('bulk-mark-tp');
    if (bulkTp) bulkTp.addEventListener('click', function () { bulkLabel('tp'); });
    var bulkUn = $('bulk-mark-unassigned');
    if (bulkUn) bulkUn.addEventListener('click', function () { bulkLabel(null); });
    var bulkDel = $('bulk-delete');
    if (bulkDel) bulkDel.addEventListener('click', bulkDelete);
    renderBulkBar();
  }

  function renderEnrollDeviceSelect() {
    var sel = $('enroll-vpe-select');
    if (!sel) return;
    var prev = sel.value;
    sel.innerHTML = '';
    if (state.devices.length === 0) {
      sel.appendChild(el('option', { value: '' }, ['(no active devices)']));
      sel.disabled = true;
      var hint = $('mic-source-hint');
      if (hint && state.micSource === 'voice-pe') hint.hidden = false;
    } else {
      state.devices.forEach(function (d) {
        sel.appendChild(el('option', { value: d.id }, [deviceLabel(d) + (d.is_detecting ? ' (live)' : ' (idle)')]));
      });
      sel.disabled = false;
      var hint2 = $('mic-source-hint');
      if (hint2) hint2.hidden = true;
    }
    if (prev && state.devices.some(function (d) { return d.id === prev; })) sel.value = prev;
    var vpeBtn = document.querySelector('#mic-source-control .seg-button[data-source="voice-pe"]');
    if (vpeBtn) vpeBtn.disabled = state.devices.length === 0;
  }

  function setMicSource(src) {
    state.micSource = src;
    document.querySelectorAll('#mic-source-control .seg-button').forEach(function (b) {
      var on = b.dataset.source === src;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    var sel = $('enroll-vpe-select');
    if (sel) sel.hidden = src !== 'voice-pe';
    var hint = $('mic-source-hint');
    if (hint) hint.hidden = src !== 'voice-pe' || state.devices.length > 0;
    var lbl = $('record-btn-label');
    if (lbl) lbl.textContent = src === 'voice-pe' ? 'Record from Voice PE' : 'Start Recording Sample';
  }

  function initMicSource() {
    document.querySelectorAll('#mic-source-control .seg-button').forEach(function (b) {
      b.addEventListener('click', function () {
        if (b.disabled) return;
        setMicSource(b.dataset.source);
      });
    });
    setMicSource('browser');

    var recBtn = $('record-btn');
    if (recBtn) {
      recBtn.addEventListener('click', function (e) {
        if (state.micSource !== 'voice-pe') return;
        e.stopImmediatePropagation();
        var sel = $('enroll-vpe-select');
        var devId = sel ? sel.value : '';
        if (!devId) {
          toast('Pick a Voice PE device first', 'warning', 3000);
          return;
        }
        recordFromVoicePE(devId);
      }, true);
    }
  }

  async function recordFromVoicePE(devId) {
    var name = (state.speakerName || '').trim();
    if (!name) { toast('Enter a speaker name first', 'warning', 3000); return; }
    var recBtn = $('record-btn');
    if (recBtn) recBtn.disabled = true;
    setText('record-btn-label', 'Recording from Voice PE…');
    try {
      var res = await apiRequest('api/devices/' + encodeURIComponent(devId) + '/capture?duration=5', { method: 'POST' });
      var sampleId = res.sample && res.sample.id;
      if (!sampleId) throw new Error('No sample id returned');
      var resp = await fetch('api/samples/' + encodeURIComponent(sampleId) + '/audio.wav');
      var buf = await resp.arrayBuffer();
      var ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      var decoded = await ctx.decodeAudioData(buf);
      var f32orig = decoded.getChannelData(0);
      var f32 = new Float32Array(f32orig);
      var int16 = new Int16Array(f32.length);
      var rmsAcc = 0;
      var peakVal = 0;
      for (var i = 0; i < f32.length; i++) {
        var s = f32[i];
        var a = s < 0 ? -s : s;
        rmsAcc += s * s;
        if (a > peakVal) peakVal = a;
        var v = s * 32768;
        int16[i] = v > 32767 ? 32767 : (v < -32768 ? -32768 : (v | 0));
      }
      ctx.close();
      state.samples.push({
        int16: int16,
        float32: f32,
        durationSeconds: f32.length / 16000,
        rms: Math.sqrt(rmsAcc / Math.max(1, f32.length)),
        peak: peakVal,
        source: 'voice-pe',
        sampleId: sampleId,
      });
      renderSamples();
      updateEnrollButton();
      toast('Captured from Voice PE: ' + sampleId, 'success', 2000);
    } catch (e) {
      toast('Voice PE capture failed: ' + e.message, 'error', 4000);
    } finally {
      if (recBtn) recBtn.disabled = false;
      setText('record-btn-label', 'Record from Voice PE');
    }
  }

  function handleTabActivated(tab) {
    if (state.devicePollTimer) { clearInterval(state.devicePollTimer); state.devicePollTimer = null; }
    if (state.samplesPollTimer) { clearInterval(state.samplesPollTimer); state.samplesPollTimer = null; }
    if (tab === 'samples') {
      pollDevices();
      loadServerSamples();
      state.devicePollTimer = setInterval(pollDevices, 10000);
    } else if (tab === 'debug') {
      pollDevices();
      state.devicePollTimer = setInterval(function () { pollDevices(); renderDebugDeviceSelect(); }, 10000);
      renderDebugDeviceSelect();
    } else if (tab === 'enroll') {
      pollDevices();
      state.devicePollTimer = setInterval(function () { pollDevices(); renderEnrollDeviceSelect(); }, 10000);
      renderEnrollDeviceSelect();
    }
  }

  function renderDebugDeviceSelect() {
    var sel = $('debug-device-select');
    if (!sel) return;
    var prev = sel.value;
    sel.innerHTML = '';
    if (!state.devices.length) {
      sel.appendChild(el('option', { value: '' }, ['(no active devices)']));
      sel.disabled = true;
      var radio = $('debug-source-voicepe');
      if (radio) radio.disabled = true;
      return;
    }
    state.devices.forEach(function (d) {
      sel.appendChild(el('option', { value: d.id }, [deviceLabel(d) + (d.is_detecting ? ' (live)' : ' (idle)')]));
    });
    if (prev && state.devices.some(function (d) { return d.id === prev; })) sel.value = prev;
    sel.disabled = state.debugSource !== 'voice-pe';
    var radio2 = $('debug-source-voicepe');
    if (radio2) radio2.disabled = false;
  }

  async function setDebugSource(newSource, deviceId) {
    if (state.debugClaimedDeviceId && state.debugClaimedDeviceId !== deviceId) {
      try { await apiRequest('api/devices/' + encodeURIComponent(state.debugClaimedDeviceId) + '/debug-release', { method: 'POST' }); } catch (_) {}
      state.debugClaimedDeviceId = null;
    }
    if (newSource === 'voice-pe' && deviceId) {
      try {
        await apiRequest('api/devices/' + encodeURIComponent(deviceId) + '/debug-claim', { method: 'POST' });
        state.debugClaimedDeviceId = deviceId;
        if (state.scoreWS && state.scoreWS.isOpen()) {
          state.scoreWS.send(JSON.stringify({ cmd: 'subscribe_device', device_id: deviceId }));
        }
        if (state.debugWS && state.debugWS.isOpen()) {
          state.debugWS.send(JSON.stringify({ cmd: 'subscribe_device', device_id: deviceId }));
        }
        toast('Claimed ' + deviceId.slice(0, 6) + '. Wyoming pipeline paused for this device.', 'info', 3000);
      } catch (e) {
        toast('Failed to claim device: ' + e.message, 'error', 3000);
        return;
      }
    } else {
      if (state.scoreWS && state.scoreWS.isOpen()) {
        state.scoreWS.send(JSON.stringify({ cmd: 'unsubscribe_device' }));
      }
      if (state.debugWS && state.debugWS.isOpen()) {
        state.debugWS.send(JSON.stringify({ cmd: 'unsubscribe_device' }));
      }
    }
    state.debugSource = newSource;
    state.debugSourceDeviceId = newSource === 'voice-pe' ? deviceId : null;
    var sel = $('debug-device-select');
    if (sel) sel.disabled = newSource !== 'voice-pe';
    var note = $('source-claim-note');
    if (note) note.hidden = newSource !== 'voice-pe';
  }

  function initDebugSourcePicker() {
    var radios = document.querySelectorAll('input[name="debug-source"]');
    radios.forEach(function (r) {
      r.addEventListener('change', function () {
        var sel = $('debug-device-select');
        var devId = (sel && sel.value) || (state.devices[0] && state.devices[0].id);
        if (r.value === 'voice-pe' && !devId) {
          toast('No active Voice PE devices', 'warning');
          var browserRadio = $('debug-source-browser');
          if (browserRadio) browserRadio.checked = true;
          return;
        }
        setDebugSource(r.value, devId);
      });
    });
    var sel = $('debug-device-select');
    if (sel) sel.addEventListener('change', function () {
      if (state.debugSource === 'voice-pe') setDebugSource('voice-pe', sel.value);
    });
  }

  function init() {
    setText('version-display', 'v' + ADDON_VERSION);
    apiRequest('api/version').then(function (info) {
      if (info && info.version) setText('version-display', 'v' + info.version);
    }).catch(function () {});
    initTheme();
    initTabs();
    initEnrollTab();
    initDebugTab();
    initSamplesTab();
    initDebugSourcePicker();
    fetchSpeakers();
    var banner = $('reconnect-banner');
    if (banner) banner.hidden = true;
    document.querySelectorAll('.tab-button').forEach(function (b) {
      b.addEventListener('click', function () { handleTabActivated(b.dataset.tab); });
    });
    handleTabActivated(state.activeTab || 'enroll');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
