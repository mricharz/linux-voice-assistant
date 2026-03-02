#!/usr/bin/env python3
"""Trim WAV files in a visual web UI with waveform display and selection.

Serves a web page where WAV files can be visually inspected, a region selected
on the waveform, and trimmed -- overwriting the original file with only the
selected portion.  Only uses the Python standard library.

Usage:
    python3 tools/wav_trimmer.py \
        --targetDir ./recordings \
        --port 8080
"""

import argparse
import http.server
import json
import os
import tempfile
import threading
import urllib.parse
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# FileManager -- keeps track of WAV files and performs trim operations
# ---------------------------------------------------------------------------


class FileManager:
    def __init__(self, target_dir: Path):
        self.target_dir = target_dir.resolve()
        self._lock = threading.Lock()
        self._files: list[str] = self._scan()
        self._index = 0
        self._trimmed: set[str] = set()

    def _scan(self) -> list[str]:
        base = self.target_dir
        return sorted(
            str(f.relative_to(base))
            for f in base.rglob("*.wav")
            if f.is_file()
        )

    def __len__(self) -> int:
        return len(self._files)

    # -- public api --------------------------------------------------------

    def file_list(self) -> dict:
        with self._lock:
            return self._state()

    def get_info(self, rel_path: str) -> dict:
        full = self._resolve(rel_path)
        if full is None or not full.is_file():
            return {"error": "not found"}
        try:
            with wave.open(str(full), "rb") as wf:
                return {
                    "path": rel_path,
                    "sampleRate": wf.getframerate(),
                    "channels": wf.getnchannels(),
                    "sampleWidth": wf.getsampwidth(),
                    "numFrames": wf.getnframes(),
                    "duration": wf.getnframes() / wf.getframerate(),
                }
        except Exception as exc:
            return {"error": str(exc)}

    def trim(self, rel_path: str, start_frame: int, end_frame: int) -> dict:
        with self._lock:
            full = self._resolve(rel_path)
            if full is None or not full.is_file():
                return {"error": "not found"}
            try:
                self._do_trim(full, start_frame, end_frame)
                self._trimmed.add(rel_path)
                # advance to next
                if rel_path in self._files:
                    idx = self._files.index(rel_path)
                    self._index = min(idx + 1, len(self._files) - 1)
            except Exception as exc:
                return {"error": str(exc)}
            return self._state()

    def skip(self, rel_path: str) -> dict:
        with self._lock:
            if rel_path in self._files:
                idx = self._files.index(rel_path)
                self._index = min(idx + 1, len(self._files) - 1)
            return self._state()

    def select(self, index: int) -> dict:
        with self._lock:
            if 0 <= index < len(self._files):
                self._index = index
            return self._state()

    # -- internal ----------------------------------------------------------

    def _resolve(self, rel_path: str) -> Path | None:
        if not rel_path or "\\" in rel_path or "\0" in rel_path:
            return None
        # reject path traversal
        if ".." in rel_path.split("/"):
            return None
        full = (self.target_dir / rel_path).resolve()
        # prefix check
        if not str(full).startswith(str(self.target_dir)):
            return None
        return full

    def resolve_for_audio(self, rel_path: str) -> Path | None:
        return self._resolve(rel_path)

    @staticmethod
    def _do_trim(filepath: Path, start_frame: int, end_frame: int) -> None:
        with wave.open(str(filepath), "rb") as wf:
            params = wf.getparams()
            n_frames = wf.getnframes()

        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)
        if start_frame >= end_frame:
            raise ValueError("empty selection")

        frame_size = params.nchannels * params.sampwidth

        with wave.open(str(filepath), "rb") as wf:
            wf.setpos(start_frame)
            data = wf.readframes(end_frame - start_frame)

        # write to temp file, then atomic replace
        dir_path = filepath.parent
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=str(dir_path))
        try:
            os.close(fd)
            with wave.open(tmp_path, "wb") as out:
                out.setparams(params)
                out.setnframes(0)  # will be set by writeframes
                out.writeframes(data)
            os.replace(tmp_path, str(filepath))
        except Exception:
            # clean up temp on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _state(self) -> dict:
        if not self._files:
            return {
                "files": [],
                "currentIndex": 0,
                "total": 0,
                "trimmedCount": len(self._trimmed),
                "trimmedFiles": sorted(self._trimmed),
            }
        self._index = min(self._index, len(self._files) - 1)
        return {
            "files": self._files,
            "currentIndex": self._index,
            "total": len(self._files),
            "trimmedCount": len(self._trimmed),
            "trimmedFiles": sorted(self._trimmed),
        }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class TrimHandler(http.server.BaseHTTPRequestHandler):
    file_manager: FileManager  # set before server starts

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_html()
        elif path == "/api/files":
            self._send_json(self.file_manager.file_list())
        elif path.startswith("/api/audio/"):
            rel = urllib.parse.unquote(path[len("/api/audio/") :])
            self._send_audio(rel)
        elif path.startswith("/api/info/"):
            rel = urllib.parse.unquote(path[len("/api/info/") :])
            self._send_json(self.file_manager.get_info(rel))
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        body = self._read_body()

        if path == "/api/trim":
            rel = body.get("path", "")
            start = body.get("startFrame", 0)
            end = body.get("endFrame", 0)
            if not isinstance(start, int) or not isinstance(end, int):
                self._send_json({"error": "invalid frame values"}, 400)
                return
            self._send_json(self.file_manager.trim(rel, start, end))
        elif path == "/api/skip":
            rel = body.get("path", "")
            self._send_json(self.file_manager.skip(rel))
        elif path == "/api/select":
            index = body.get("index", 0)
            if not isinstance(index, int):
                self._send_json({"error": "invalid index"}, 400)
                return
            self._send_json(self.file_manager.select(index))
        else:
            self.send_error(404)

    # -- helpers -----------------------------------------------------------

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _send_json(self, data: dict, code: int = 200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self):
        body = HTML_PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_audio(self, rel_path: str):
        full = self.file_manager.resolve_for_audio(rel_path)
        if full is None or not full.is_file():
            self.send_error(404)
            return
        data = full.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # noqa: A002
        pass  # silence per-request logging


# ---------------------------------------------------------------------------
# HTML / CSS / JS  (single-page app, no external resources)
# ---------------------------------------------------------------------------

HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WAV Trimmer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,
  "Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:#1a1a2e;color:#eee}

/* sidebar */
#sidebar{position:fixed;top:0;left:0;bottom:0;width:280px;background:#16213e;
  border-right:1px solid #333;display:flex;flex-direction:column;z-index:10;
  transition:transform .3s ease}
#sidebar.collapsed{transform:translateX(-280px)}
#sidebar-header{padding:12px 16px;border-bottom:1px solid #333;display:flex;
  align-items:center;justify-content:space-between}
#sidebar-header h2{font-size:16px;font-weight:600;color:#4ecca3}
#sidebar-toggle{background:none;border:none;color:#aaa;font-size:20px;
  cursor:pointer;padding:4px 8px}
#sidebar-toggle:hover{color:#fff}
#sidebar-stats{padding:8px 16px;font-size:12px;color:#888;
  border-bottom:1px solid #333}
#file-list{flex:1;overflow-y:auto;padding:4px 0}
.file-item{padding:8px 16px;font-size:13px;font-family:monospace;color:#aaa;
  cursor:pointer;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  border-left:3px solid transparent;transition:background .15s,color .15s}
.file-item:hover{background:rgba(78,204,163,.1);color:#ccc}
.file-item.active{background:rgba(78,204,163,.15);color:#4ecca3;
  border-left-color:#4ecca3}
.file-item.trimmed{color:#666}
.file-item.trimmed::after{content:" \\2713";color:#4ecca3}
#sidebar-expand{position:fixed;top:8px;left:8px;z-index:11;background:#16213e;
  border:1px solid #333;border-radius:6px;color:#aaa;font-size:18px;
  cursor:pointer;padding:4px 10px;display:none}
#sidebar-expand:hover{color:#fff;border-color:#4ecca3}

/* main editor */
#editor{position:fixed;top:0;right:0;bottom:0;left:280px;display:flex;
  flex-direction:column;transition:left .3s ease}
#editor.expanded{left:0}

/* top bar */
#topbar{height:48px;display:flex;align-items:center;padding:0 16px;gap:12px;
  background:rgba(26,26,46,.9);border-bottom:1px solid #333;flex-shrink:0}
#current-file{font-size:14px;font-family:monospace;color:#ccc;flex:1;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.topbar-btn{background:#16213e;border:1px solid #444;border-radius:6px;
  color:#ccc;font-size:13px;padding:6px 12px;cursor:pointer;
  transition:border-color .15s,color .15s}
.topbar-btn:hover{border-color:#4ecca3;color:#4ecca3}
.topbar-btn.primary{background:#4ecca3;color:#1a1a2e;border-color:#4ecca3;
  font-weight:600}
.topbar-btn.primary:hover{background:#3dbb94}

/* waveform area */
#wave-container{flex:1;position:relative;padding:16px;display:flex;
  align-items:center;justify-content:center;min-height:0}
#canvas-wrap{position:relative;width:100%;height:100%;max-height:400px}
#waveform-canvas{position:absolute;top:0;left:0;width:100%;height:100%}
#overlay-canvas{position:absolute;top:0;left:0;width:100%;height:100%;
  cursor:crosshair}

/* time display */
#time-bar{height:32px;display:flex;align-items:center;justify-content:center;
  gap:24px;font-size:13px;font-family:monospace;color:#aaa;flex-shrink:0;
  border-top:1px solid #333;background:rgba(22,33,62,.8)}

/* bottom controls */
#controls{height:64px;display:flex;align-items:center;justify-content:center;
  gap:12px;padding:0 16px;background:rgba(26,26,46,.9);
  border-top:1px solid #333;flex-shrink:0}
.ctrl-btn{background:#16213e;border:1px solid #444;border-radius:8px;
  color:#ccc;font-size:13px;padding:8px 16px;cursor:pointer;
  transition:all .15s;display:flex;align-items:center;gap:6px}
.ctrl-btn:hover{border-color:#4ecca3;color:#4ecca3}
.ctrl-btn.save{background:#4ecca3;color:#1a1a2e;border-color:#4ecca3;
  font-weight:600}
.ctrl-btn.save:hover{background:#3dbb94}
.ctrl-btn kbd{font-size:11px;background:rgba(255,255,255,.15);
  padding:1px 5px;border-radius:3px}

/* zoom controls */
#zoom-bar{position:absolute;top:8px;right:8px;display:flex;gap:4px;z-index:5}
.zoom-btn{background:rgba(22,33,62,.9);border:1px solid #444;border-radius:4px;
  color:#ccc;font-size:14px;width:28px;height:28px;cursor:pointer;
  display:flex;align-items:center;justify-content:center}
.zoom-btn:hover{border-color:#4ecca3;color:#4ecca3}

/* empty state */
#empty-msg{display:none;position:fixed;inset:0;
  flex-direction:column;align-items:center;justify-content:center;gap:16px;
  z-index:20;background:#1a1a2e}
#empty-msg h1{font-size:24px;color:#4ecca3}
#empty-msg p{font-size:16px;color:#888}

/* keyboard hints */
#kbd-hints{position:absolute;bottom:4px;left:0;right:0;text-align:center;
  font-size:11px;color:#444;pointer-events:none}
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h2>WAV Trimmer</h2>
    <button id="sidebar-toggle" title="Collapse sidebar">&laquo;</button>
  </div>
  <div id="sidebar-stats">0 files &middot; 0 trimmed</div>
  <div id="file-list"></div>
</div>
<button id="sidebar-expand" title="Show file list">&#9776;</button>

<div id="editor">
  <div id="topbar">
    <span id="current-file">No file selected</span>
    <button class="topbar-btn" id="btn-play-all" title="Play All (Space)">
      &#9654; Play All</button>
    <button class="topbar-btn" id="btn-play-sel" title="Play Selection (Enter)">
      &#9654; Selection</button>
    <button class="topbar-btn" id="btn-stop" title="Stop (Esc)">
      &#9632; Stop</button>
  </div>
  <div id="wave-container">
    <div id="canvas-wrap">
      <canvas id="waveform-canvas"></canvas>
      <canvas id="overlay-canvas"></canvas>
      <div id="zoom-bar">
        <button class="zoom-btn" id="btn-zoom-in" title="Zoom In (+)">+</button>
        <button class="zoom-btn" id="btn-zoom-out" title="Zoom Out (-)">-</button>
        <button class="zoom-btn" id="btn-zoom-fit" title="Zoom to Fit (0)">&#8596;</button>
      </div>
      <div id="kbd-hints">
        Space=Play All &middot; Enter=Play Selection &middot;
        Ctrl+S=Save &middot; N=Skip &middot; A=Select All</div>
    </div>
  </div>
  <div id="time-bar">
    <span>Selection: <span id="sel-start">-</span> &ndash;
      <span id="sel-end">-</span></span>
    <span>Duration: <span id="sel-dur">-</span></span>
    <span>Total: <span id="total-dur">-</span></span>
  </div>
  <div id="controls">
    <button class="ctrl-btn" id="btn-sel-all">Select All <kbd>A</kbd></button>
    <button class="ctrl-btn save" id="btn-save">Save Trim <kbd>Ctrl+S</kbd></button>
    <button class="ctrl-btn" id="btn-skip">Skip <kbd>N</kbd></button>
  </div>
</div>

<div id="empty-msg">
  <h1>No WAV files found</h1>
  <p>The target directory contains no .wav files.</p>
</div>

<script>
(function(){
  // ---- elements ----
  const sidebar     = document.getElementById('sidebar');
  const sidebarExp  = document.getElementById('sidebar-expand');
  const fileListEl  = document.getElementById('file-list');
  const statsEl     = document.getElementById('sidebar-stats');
  const editorEl    = document.getElementById('editor');
  const curFileEl   = document.getElementById('current-file');
  const emptyEl     = document.getElementById('empty-msg');
  const waveCanvas  = document.getElementById('waveform-canvas');
  const overCanvas  = document.getElementById('overlay-canvas');
  const waveCtx     = waveCanvas.getContext('2d');
  const overCtx     = overCanvas.getContext('2d');
  const selStartEl  = document.getElementById('sel-start');
  const selEndEl    = document.getElementById('sel-end');
  const selDurEl    = document.getElementById('sel-dur');
  const totalDurEl  = document.getElementById('total-dur');

  // ---- state ----
  let files = [];
  let currentIndex = 0;
  let trimmedFiles = new Set();
  let audioBuffer = null;
  let audioCtx = null;
  let sourceNode = null;
  let sampleRate = 44100;
  let totalFrames = 0;

  // view (in frames)
  let viewStart = 0;
  let viewEnd = 0;

  // selection (in frames)
  let selStart = 0;
  let selEnd = 0;
  let hasSelection = false;

  // interaction
  let isDragging = false;
  let dragMode = null;  // 'create', 'move', 'left', 'right'
  let dragStartX = 0;
  let dragOrigSelStart = 0;
  let dragOrigSelEnd = 0;

  // playback position animation
  let playbackRAF = null;
  let playStartTime = 0;
  let playStartFrame = 0;
  let playEndFrame = 0;
  let isPlaying = false;

  // ---- utility ----
  function fmt(seconds) {
    if (seconds == null || isNaN(seconds)) return '-';
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(3);
    return m > 0 ? m + ':' + s.padStart(6, '0') : s + 's';
  }

  function frameToTime(frame) {
    return frame / sampleRate;
  }

  function ensureAudioCtx() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    return audioCtx;
  }

  // ---- API ----
  async function api(path, body) {
    const opts = body !== undefined
      ? {method:'POST', headers:{'Content-Type':'application/json'},
         body:JSON.stringify(body)}
      : {};
    const r = await fetch(path, opts);
    return r.json();
  }

  // ---- file list & sidebar ----
  function renderFileList() {
    statsEl.textContent = files.length + ' files \\u00b7 ' +
      trimmedFiles.size + ' trimmed';
    fileListEl.innerHTML = files.map((f, i) => {
      const cls = ['file-item'];
      if (i === currentIndex) cls.push('active');
      if (trimmedFiles.has(f)) cls.push('trimmed');
      // show just filename for display, store full path
      const display = f.includes('/') ? f : f;
      return '<div class="' + cls.join(' ') + '" data-index="' + i +
        '" title="' + f.replace(/"/g,'&quot;') + '">' +
        f.replace(/</g,'&lt;') + '</div>';
    }).join('');
    // scroll active into view
    const active = fileListEl.querySelector('.active');
    if (active) active.scrollIntoView({block:'nearest'});
  }

  fileListEl.addEventListener('click', e => {
    const item = e.target.closest('.file-item');
    if (!item) return;
    const idx = parseInt(item.dataset.index, 10);
    selectFile(idx);
  });

  // sidebar toggle
  document.getElementById('sidebar-toggle').addEventListener('click', () => {
    sidebar.classList.add('collapsed');
    editorEl.classList.add('expanded');
    sidebarExp.style.display = 'block';
    onResize();
  });
  sidebarExp.addEventListener('click', () => {
    sidebar.classList.remove('collapsed');
    editorEl.classList.remove('expanded');
    sidebarExp.style.display = 'none';
    onResize();
  });

  // ---- load & draw ----
  async function loadState() {
    const d = await api('/api/files');
    files = d.files || [];
    currentIndex = d.currentIndex || 0;
    trimmedFiles = new Set(d.trimmedFiles || []);
    if (!files.length) {
      emptyEl.style.display = 'flex';
      return;
    }
    emptyEl.style.display = 'none';
    renderFileList();
    await loadAudio(files[currentIndex]);
  }

  async function selectFile(idx) {
    stopPlayback();
    const d = await api('/api/select', {index: idx});
    files = d.files || [];
    currentIndex = d.currentIndex || 0;
    trimmedFiles = new Set(d.trimmedFiles || []);
    renderFileList();
    await loadAudio(files[currentIndex]);
  }

  async function loadAudio(relPath) {
    curFileEl.textContent = relPath;
    // fetch info
    const info = await api('/api/info/' + encodePathSegments(relPath));
    if (info.error) { curFileEl.textContent = 'Error: ' + info.error; return; }
    sampleRate = info.sampleRate;
    totalFrames = info.numFrames;
    totalDurEl.textContent = fmt(info.duration);

    // fetch audio bytes and decode
    const ctx = ensureAudioCtx();
    const resp = await fetch('/api/audio/' + encodePathSegments(relPath));
    const arrayBuf = await resp.arrayBuffer();
    audioBuffer = await ctx.decodeAudioData(arrayBuf);

    // reset view & selection
    viewStart = 0;
    viewEnd = totalFrames;
    selStart = 0;
    selEnd = totalFrames;
    hasSelection = true;
    updateTimeDisplay();
    drawWaveform();
    drawOverlay();
  }

  function encodePathSegments(p) {
    return p.split('/').map(s => encodeURIComponent(s)).join('/');
  }

  // ---- waveform rendering ----
  function drawWaveform() {
    const dpr = window.devicePixelRatio || 1;
    const rect = waveCanvas.parentElement.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    waveCanvas.width = w * dpr;
    waveCanvas.height = h * dpr;
    waveCanvas.style.width = w + 'px';
    waveCanvas.style.height = h + 'px';
    waveCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // clear
    waveCtx.fillStyle = '#0f0f23';
    waveCtx.fillRect(0, 0, w, h);

    if (!audioBuffer) return;

    const data = audioBuffer.getChannelData(0);
    const viewLen = viewEnd - viewStart;
    if (viewLen <= 0) return;

    const mid = h / 2;
    const amp = (h / 2) - 4;

    waveCtx.strokeStyle = '#4ecca3';
    waveCtx.lineWidth = 1;
    waveCtx.beginPath();

    // min/max per pixel
    for (let px = 0; px < w; px++) {
      const fStart = viewStart + (px / w) * viewLen;
      const fEnd = viewStart + ((px + 1) / w) * viewLen;
      const iStart = Math.floor(fStart);
      const iEnd = Math.min(Math.ceil(fEnd), data.length);
      let mn = 1, mx = -1;
      for (let i = iStart; i < iEnd; i++) {
        const v = data[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      const y1 = mid - mx * amp;
      const y2 = mid - mn * amp;
      waveCtx.moveTo(px + 0.5, y1);
      waveCtx.lineTo(px + 0.5, y2);
    }
    waveCtx.stroke();

    // center line
    waveCtx.strokeStyle = 'rgba(255,255,255,0.1)';
    waveCtx.beginPath();
    waveCtx.moveTo(0, mid);
    waveCtx.lineTo(w, mid);
    waveCtx.stroke();
  }

  // ---- overlay (selection + playback cursor) ----
  function drawOverlay() {
    const dpr = window.devicePixelRatio || 1;
    const rect = overCanvas.parentElement.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    overCanvas.width = w * dpr;
    overCanvas.height = h * dpr;
    overCanvas.style.width = w + 'px';
    overCanvas.style.height = h + 'px';
    overCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    overCtx.clearRect(0, 0, w, h);

    if (!audioBuffer || !hasSelection) return;

    const viewLen = viewEnd - viewStart;
    if (viewLen <= 0) return;

    const sx = ((selStart - viewStart) / viewLen) * w;
    const ex = ((selEnd - viewStart) / viewLen) * w;

    // dim outside selection
    overCtx.fillStyle = 'rgba(0,0,0,0.55)';
    if (sx > 0) overCtx.fillRect(0, 0, sx, h);
    if (ex < w) overCtx.fillRect(ex, 0, w - ex, h);

    // selection edges
    overCtx.strokeStyle = '#4ecca3';
    overCtx.lineWidth = 2;
    overCtx.beginPath();
    overCtx.moveTo(sx, 0); overCtx.lineTo(sx, h);
    overCtx.moveTo(ex, 0); overCtx.lineTo(ex, h);
    overCtx.stroke();

    // handles (triangles)
    const hs = 8;
    overCtx.fillStyle = '#4ecca3';
    // left handle top
    overCtx.beginPath();
    overCtx.moveTo(sx, 0); overCtx.lineTo(sx + hs, 0);
    overCtx.lineTo(sx, hs); overCtx.fill();
    // left handle bottom
    overCtx.beginPath();
    overCtx.moveTo(sx, h); overCtx.lineTo(sx + hs, h);
    overCtx.lineTo(sx, h - hs); overCtx.fill();
    // right handle top
    overCtx.beginPath();
    overCtx.moveTo(ex, 0); overCtx.lineTo(ex - hs, 0);
    overCtx.lineTo(ex, hs); overCtx.fill();
    // right handle bottom
    overCtx.beginPath();
    overCtx.moveTo(ex, h); overCtx.lineTo(ex - hs, h);
    overCtx.lineTo(ex, h - hs); overCtx.fill();

    // playback cursor
    if (isPlaying && audioCtx) {
      const elapsed = audioCtx.currentTime - playStartTime;
      const curFrame = playStartFrame + elapsed * sampleRate;
      if (curFrame >= playStartFrame && curFrame <= playEndFrame) {
        const cx = ((curFrame - viewStart) / viewLen) * w;
        overCtx.strokeStyle = '#fff';
        overCtx.lineWidth = 1;
        overCtx.beginPath();
        overCtx.moveTo(cx, 0);
        overCtx.lineTo(cx, h);
        overCtx.stroke();
      }
    }
  }

  function animatePlayback() {
    if (!isPlaying) return;
    drawOverlay();
    playbackRAF = requestAnimationFrame(animatePlayback);
  }

  // ---- mouse interaction on overlay canvas ----
  function pxToFrame(px) {
    const rect = overCanvas.getBoundingClientRect();
    const x = px - rect.left;
    const w = rect.width;
    const viewLen = viewEnd - viewStart;
    return viewStart + (x / w) * viewLen;
  }

  function frameToPx(frame) {
    const rect = overCanvas.getBoundingClientRect();
    const w = rect.width;
    const viewLen = viewEnd - viewStart;
    return ((frame - viewStart) / viewLen) * w;
  }

  function getHandleZone(clientX) {
    if (!hasSelection) return null;
    const rect = overCanvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const sx = frameToPx(selStart);
    const ex = frameToPx(selEnd);
    const tol = 8;
    if (Math.abs(x - sx) < tol) return 'left';
    if (Math.abs(x - ex) < tol) return 'right';
    if (x > sx + tol && x < ex - tol) return 'move';
    return null;
  }

  overCanvas.addEventListener('mousedown', e => {
    if (!audioBuffer) return;
    e.preventDefault();
    const zone = getHandleZone(e.clientX);
    isDragging = true;
    dragStartX = e.clientX;
    dragOrigSelStart = selStart;
    dragOrigSelEnd = selEnd;

    if (zone === 'left') {
      dragMode = 'left';
    } else if (zone === 'right') {
      dragMode = 'right';
    } else if (zone === 'move') {
      dragMode = 'move';
    } else {
      dragMode = 'create';
      const frame = Math.round(pxToFrame(e.clientX));
      selStart = Math.max(0, Math.min(totalFrames, frame));
      selEnd = selStart;
      hasSelection = true;
    }
  });

  document.addEventListener('mousemove', e => {
    if (!isDragging) {
      // update cursor
      if (audioBuffer) {
        const zone = getHandleZone(e.clientX);
        if (zone === 'left' || zone === 'right') {
          overCanvas.style.cursor = 'col-resize';
        } else if (zone === 'move') {
          overCanvas.style.cursor = 'grab';
        } else {
          overCanvas.style.cursor = 'crosshair';
        }
      }
      return;
    }
    const frame = Math.round(pxToFrame(e.clientX));
    const clamped = Math.max(0, Math.min(totalFrames, frame));
    const dx = e.clientX - dragStartX;
    const viewLen = viewEnd - viewStart;
    const rect = overCanvas.getBoundingClientRect();
    const frameDelta = Math.round((dx / rect.width) * viewLen);

    if (dragMode === 'create') {
      const anchor = Math.round(pxToFrame(dragStartX));
      selStart = Math.max(0, Math.min(anchor, clamped));
      selEnd = Math.min(totalFrames, Math.max(anchor, clamped));
    } else if (dragMode === 'left') {
      selStart = Math.max(0, Math.min(clamped, selEnd - 1));
    } else if (dragMode === 'right') {
      selEnd = Math.min(totalFrames, Math.max(clamped, selStart + 1));
    } else if (dragMode === 'move') {
      let ns = dragOrigSelStart + frameDelta;
      let ne = dragOrigSelEnd + frameDelta;
      const len = dragOrigSelEnd - dragOrigSelStart;
      if (ns < 0) { ns = 0; ne = len; }
      if (ne > totalFrames) { ne = totalFrames; ns = totalFrames - len; }
      selStart = ns;
      selEnd = ne;
    }
    updateTimeDisplay();
    drawOverlay();
  });

  document.addEventListener('mouseup', () => {
    if (!isDragging) return;
    isDragging = false;
    if (selStart === selEnd) hasSelection = false;
    drawOverlay();
  });

  // ---- time display ----
  function updateTimeDisplay() {
    if (!hasSelection || selStart === selEnd) {
      selStartEl.textContent = '-';
      selEndEl.textContent = '-';
      selDurEl.textContent = '-';
    } else {
      selStartEl.textContent = fmt(frameToTime(selStart));
      selEndEl.textContent = fmt(frameToTime(selEnd));
      selDurEl.textContent = fmt(frameToTime(selEnd - selStart));
    }
  }

  // ---- playback ----
  function playRange(startFrame, endFrame) {
    stopPlayback();
    if (!audioBuffer) return;
    const ctx = ensureAudioCtx();
    sourceNode = ctx.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.connect(ctx.destination);
    const offset = startFrame / sampleRate;
    const duration = (endFrame - startFrame) / sampleRate;
    sourceNode.start(0, offset, duration);
    isPlaying = true;
    playStartTime = ctx.currentTime;
    playStartFrame = startFrame;
    playEndFrame = endFrame;
    sourceNode.onended = () => {
      isPlaying = false;
      drawOverlay();
    };
    animatePlayback();
  }

  function stopPlayback() {
    if (sourceNode) {
      try { sourceNode.stop(); } catch(e) {}
      sourceNode.disconnect();
      sourceNode = null;
    }
    isPlaying = false;
    if (playbackRAF) {
      cancelAnimationFrame(playbackRAF);
      playbackRAF = null;
    }
    drawOverlay();
  }

  function playAll() {
    if (!audioBuffer) return;
    playRange(0, totalFrames);
  }

  function playSelection() {
    if (!audioBuffer || !hasSelection || selStart >= selEnd) return;
    playRange(selStart, selEnd);
  }

  // ---- zoom ----
  function zoomIn() {
    const len = viewEnd - viewStart;
    const center = (viewStart + viewEnd) / 2;
    const newLen = Math.max(100, len * 0.5);
    viewStart = Math.max(0, Math.round(center - newLen / 2));
    viewEnd = Math.min(totalFrames, Math.round(center + newLen / 2));
    drawWaveform();
    drawOverlay();
  }

  function zoomOut() {
    const len = viewEnd - viewStart;
    const center = (viewStart + viewEnd) / 2;
    const newLen = Math.min(totalFrames, len * 2);
    viewStart = Math.max(0, Math.round(center - newLen / 2));
    viewEnd = Math.min(totalFrames, Math.round(center + newLen / 2));
    drawWaveform();
    drawOverlay();
  }

  function zoomFit() {
    viewStart = 0;
    viewEnd = totalFrames;
    drawWaveform();
    drawOverlay();
  }

  function selectAll() {
    selStart = 0;
    selEnd = totalFrames;
    hasSelection = true;
    updateTimeDisplay();
    drawOverlay();
  }

  // ---- actions ----
  async function saveTrim() {
    if (!audioBuffer || !hasSelection || selStart >= selEnd) return;
    const rel = files[currentIndex];
    if (!rel) return;
    const d = await api('/api/trim', {
      path: rel,
      startFrame: selStart,
      endFrame: selEnd
    });
    if (d.error) { alert('Trim failed: ' + d.error); return; }
    files = d.files || [];
    currentIndex = d.currentIndex || 0;
    trimmedFiles = new Set(d.trimmedFiles || []);
    renderFileList();
    if (files.length && currentIndex < files.length) {
      await loadAudio(files[currentIndex]);
    }
  }

  async function skipFile() {
    if (!files.length) return;
    const rel = files[currentIndex];
    const d = await api('/api/skip', {path: rel});
    files = d.files || [];
    currentIndex = d.currentIndex || 0;
    trimmedFiles = new Set(d.trimmedFiles || []);
    renderFileList();
    if (files.length && currentIndex < files.length) {
      await loadAudio(files[currentIndex]);
    }
  }

  // ---- resize ----
  function onResize() {
    drawWaveform();
    drawOverlay();
  }
  window.addEventListener('resize', onResize);

  // ---- button bindings ----
  document.getElementById('btn-play-all').addEventListener('click', playAll);
  document.getElementById('btn-play-sel').addEventListener('click', playSelection);
  document.getElementById('btn-stop').addEventListener('click', stopPlayback);
  document.getElementById('btn-zoom-in').addEventListener('click', zoomIn);
  document.getElementById('btn-zoom-out').addEventListener('click', zoomOut);
  document.getElementById('btn-zoom-fit').addEventListener('click', zoomFit);
  document.getElementById('btn-sel-all').addEventListener('click', selectAll);
  document.getElementById('btn-save').addEventListener('click', saveTrim);
  document.getElementById('btn-skip').addEventListener('click', skipFile);

  // ---- keyboard shortcuts ----
  document.addEventListener('keydown', e => {
    // ignore if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.key === ' ') { e.preventDefault(); playAll(); }
    else if (e.key === 'Enter') { e.preventDefault(); playSelection(); }
    else if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault(); saveTrim();
    }
    else if (e.key === 'n' || e.key === 'N') { skipFile(); }
    else if (e.key === 'Escape') { stopPlayback(); }
    else if (e.key === '+' || e.key === '=') { zoomIn(); }
    else if (e.key === '-') { zoomOut(); }
    else if (e.key === '0') { zoomFit(); }
    else if (e.key === 'a' || e.key === 'A') {
      if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); selectAll(); }
    }
  });

  // ---- init ----
  loadState();
})();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Trim WAV files in a visual web UI with waveform display",
    )
    parser.add_argument(
        "--targetDir",
        required=True,
        help="Directory containing WAV files to trim (scanned recursively)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP server port (default: 8080)",
    )
    args = parser.parse_args()

    target_dir = Path(args.targetDir)
    if not target_dir.is_dir():
        parser.error(f"Target directory does not exist: {target_dir}")

    fm = FileManager(target_dir)
    TrimHandler.file_manager = fm

    server = http.server.HTTPServer(("0.0.0.0", args.port), TrimHandler)
    print("WAV Trimmer")
    print(f"  http://localhost:{args.port}")
    print(f"  Target: {target_dir.resolve()} ({len(fm)} files)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
