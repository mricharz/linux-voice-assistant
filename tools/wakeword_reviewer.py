#!/usr/bin/env python3
"""Review wake word audio recordings in a swipeable web UI.

Serves a web page where WAV files can be classified as positive (right swipe)
or negative (left swipe).  Files are moved to the corresponding output
directory.  Only uses the Python standard library -- no extra install needed.

Usage:
    python3 tools/wakeword_reviewer.py \\
        --inputDir ./wake_recordings \\
        --posDir ./positive \\
        --negDir ./negative \\
        --port 8080
"""

import argparse
import http.server
import json
import shutil
import threading
import urllib.parse
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# FileManager -- keeps track of pending / classified files
# ---------------------------------------------------------------------------


class FileManager:
    def __init__(self, input_dir: Path, pos_dir: Path, neg_dir: Path):
        self.input_dir = input_dir
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir
        self._lock = threading.Lock()
        self._files: list[str] = sorted(
            f.name for f in input_dir.iterdir() if f.suffix.lower() == ".wav"
        )
        self._index = 0
        self._positive = 0
        self._negative = 0
        self._history: list[tuple[str, str, Path]] = []  # (name, cls, dest)

    def __len__(self) -> int:
        return len(self._files)

    # -- public api --------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            return self._status_unlocked()

    def classify(self, filename: str, classification: str) -> dict:
        with self._lock:
            src = self.input_dir / filename
            dest_dir = (
                self.pos_dir if classification == "positive" else self.neg_dir
            )
            dest = dest_dir / filename
            if src.is_file():
                self._write_json(src, dest_dir, classification)
                shutil.move(str(src), str(dest))
                self._history.append((filename, classification, dest))
                if classification == "positive":
                    self._positive += 1
                else:
                    self._negative += 1
                self._files = [f for f in self._files if f != filename]
            return self._status_unlocked()

    def undo(self) -> dict:
        with self._lock:
            if not self._history:
                return self._status_unlocked()
            filename, classification, dest = self._history.pop()
            src = self.input_dir / filename
            if dest.is_file():
                shutil.move(str(dest), str(src))
                json_path = dest.with_suffix(".json")
                if json_path.is_file():
                    json_path.unlink()
                if classification == "positive":
                    self._positive -= 1
                else:
                    self._negative -= 1
                self._files.append(filename)
                self._files.sort()
            return self._status_unlocked()

    def delete_current(self, filename: str) -> dict:
        with self._lock:
            path = self.input_dir / filename
            if path.is_file():
                path.unlink()
                self._files = [f for f in self._files if f != filename]
            return self._status_unlocked()

    def history(self) -> dict:
        with self._lock:
            return self._history_unlocked()

    def delete_classified(self, filename: str, classification: str) -> dict:
        with self._lock:
            target_dir = (
                self.pos_dir if classification == "positive" else self.neg_dir
            )
            wav_path = target_dir / filename
            json_path = wav_path.with_suffix(".json")
            if wav_path.is_file():
                wav_path.unlink()
            if json_path.is_file():
                json_path.unlink()
            if classification == "positive":
                self._positive = max(0, self._positive - 1)
            else:
                self._negative = max(0, self._negative - 1)
            self._history = [
                h for h in self._history
                if not (h[0] == filename and h[1] == classification)
            ]
            return self._history_unlocked()

    # -- internal ----------------------------------------------------------

    def _history_unlocked(self) -> dict:
        pos = sorted(
            f.name for f in self.pos_dir.iterdir() if f.suffix.lower() == ".wav"
        )
        neg = sorted(
            f.name for f in self.neg_dir.iterdir() if f.suffix.lower() == ".wav"
        )
        return {"positive": pos, "negative": neg}

    @staticmethod
    def _write_json(wav_src: Path, dest_dir: Path, classification: str) -> None:
        """Write a metadata JSON alongside the classified WAV."""
        kind = "person" if classification == "positive" else "conversation"
        meta: dict = {"kind": kind}
        try:
            with wave.open(str(wav_src), "rb") as wf:
                meta["sample_rate"] = wf.getframerate()
                meta["channels"] = wf.getnchannels()
        except Exception:
            pass
        json_path = dest_dir / wav_src.with_suffix(".json").name
        json_path.write_text(json.dumps(meta, indent=2) + "\n")

    def _status_unlocked(self) -> dict:
        total = len(self._files) + self._positive + self._negative
        if not self._files:
            return {
                "done": True,
                "reviewed": self._positive + self._negative,
                "total": total,
                "stats": {
                    "positive": self._positive,
                    "negative": self._negative,
                },
            }
        return {
            "done": False,
            "filename": self._files[0],
            "reviewed": self._positive + self._negative,
            "total": total,
        }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class ReviewHandler(http.server.BaseHTTPRequestHandler):
    file_manager: FileManager  # set before server starts

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._send_html()
        elif path == "/api/status":
            self._send_json(self.file_manager.status())
        elif path == "/api/history":
            self._send_json(self.file_manager.history())
        elif path.startswith("/api/audio/positive/"):
            name = urllib.parse.unquote(path[len("/api/audio/positive/") :])
            self._send_audio(name, self.file_manager.pos_dir)
        elif path.startswith("/api/audio/negative/"):
            name = urllib.parse.unquote(path[len("/api/audio/negative/") :])
            self._send_audio(name, self.file_manager.neg_dir)
        elif path.startswith("/api/audio/"):
            name = urllib.parse.unquote(path[len("/api/audio/") :])
            self._send_audio(name, self.file_manager.input_dir)
        else:
            self.send_error(404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        body = self._read_body()
        if path == "/api/classify":
            filename = body.get("filename", "")
            classification = body.get("classification", "")
            if classification not in ("positive", "negative"):
                self._send_json({"error": "invalid classification"}, 400)
                return
            if not self._safe_name(filename):
                self._send_json({"error": "invalid filename"}, 400)
                return
            self._send_json(self.file_manager.classify(filename, classification))
        elif path == "/api/undo":
            self._send_json(self.file_manager.undo())
        elif path == "/api/delete":
            filename = body.get("filename", "")
            if not self._safe_name(filename):
                self._send_json({"error": "invalid filename"}, 400)
                return
            self._send_json(self.file_manager.delete_current(filename))
        elif path == "/api/delete-classified":
            filename = body.get("filename", "")
            classification = body.get("classification", "")
            if classification not in ("positive", "negative"):
                self._send_json({"error": "invalid classification"}, 400)
                return
            if not self._safe_name(filename):
                self._send_json({"error": "invalid filename"}, 400)
                return
            self._send_json(self.file_manager.delete_classified(filename, classification))
        else:
            self.send_error(404)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _safe_name(name: str) -> bool:
        return bool(name) and ".." not in name and "/" not in name and "\\" not in name

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

    def _send_audio(self, name: str, base_dir: Path = None):
        if not self._safe_name(name):
            self.send_error(403)
            return
        if base_dir is None:
            base_dir = self.file_manager.input_dir
        filepath = base_dir / name
        if not filepath.is_file():
            self.send_error(404)
            return
        data = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
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
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>Wake Word Reviewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,
  "Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:#1a1a2e;color:#eee}

/* progress */
#progress{position:fixed;top:0;left:0;right:0;height:48px;display:flex;
  align-items:center;justify-content:center;gap:12px;z-index:10;
  background:rgba(26,26,46,.9);backdrop-filter:blur(4px)}
#progress-bar-outer{width:60%;max-width:320px;height:6px;background:#333;
  border-radius:3px;overflow:hidden}
#progress-bar-inner{height:100%;width:0%;background:linear-gradient(90deg,#4ecca3,#36d1dc);
  transition:width .3s}
#progress-text{font-size:14px;color:#aaa;min-width:60px;text-align:center}

/* card area */
#card-area{display:flex;align-items:center;justify-content:center;
  height:100%;padding:60px 16px 100px}
#card{position:relative;width:100%;max-width:380px;min-height:220px;
  background:#16213e;border-radius:20px;box-shadow:0 8px 30px rgba(0,0,0,.4);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  cursor:grab;user-select:none;touch-action:none;will-change:transform;
  transition:transform .3s ease,opacity .3s ease}
#card.dragging{transition:none;cursor:grabbing}

.filename{font-size:15px;color:#ccc;word-break:break-all;text-align:center;
  padding:24px 20px 8px;font-family:monospace}
.audio-hint{font-size:13px;color:#666;padding-bottom:24px}

/* swipe indicators */
.indicator{position:absolute;top:50%;transform:translateY(-50%);
  font-size:28px;font-weight:800;letter-spacing:2px;opacity:0;
  transition:opacity .1s;pointer-events:none;padding:12px 18px;
  border-radius:12px;border:3px solid}
.indicator.accept{right:20px;color:#4ecca3;border-color:#4ecca3}
.indicator.reject{left:20px;color:#e74c3c;border-color:#e74c3c}

/* controls */
#controls{position:fixed;bottom:0;left:0;right:0;display:flex;
  align-items:center;justify-content:center;gap:12px;padding:16px;
  background:rgba(26,26,46,.9);backdrop-filter:blur(4px)}
.btn{border:none;border-radius:50%;width:56px;height:56px;font-size:22px;
  cursor:pointer;display:flex;align-items:center;justify-content:center;
  transition:transform .15s,box-shadow .15s}
.btn:active{transform:scale(.9)}
.btn-reject{background:#e74c3c;color:#fff}
.btn-replay{background:#555;color:#fff;width:44px;height:44px;font-size:18px}
.btn-accept{background:#4ecca3;color:#fff}
.btn-undo{background:transparent;color:#888;border:2px solid #444;
  width:40px;height:40px;font-size:16px}
.btn-delete{background:#ff6b35;color:#fff;width:44px;height:44px;font-size:18px}
.btn-history{background:transparent;color:#aaa;border:2px solid #555;
  border-radius:8px;width:auto;height:32px;font-size:13px;padding:0 12px;
  cursor:pointer;display:flex;align-items:center;gap:4px}
.btn-history:hover{border-color:#888;color:#eee}

/* history overlay */
#history{display:none;position:fixed;inset:0;background:#1a1a2e;z-index:30;
  flex-direction:column;overflow:hidden}
#history.open{display:flex}
#hist-header{display:flex;align-items:center;justify-content:space-between;
  padding:12px 16px;border-bottom:1px solid #333}
#hist-header h2{font-size:18px;font-weight:600}
#hist-close{background:none;border:none;color:#aaa;font-size:24px;
  cursor:pointer;padding:4px 8px}
#hist-close:hover{color:#fff}
#hist-tabs{display:flex;gap:0;border-bottom:1px solid #333}
.hist-tab{flex:1;padding:10px;text-align:center;cursor:pointer;
  font-size:14px;font-weight:600;border-bottom:3px solid transparent;
  transition:border-color .2s,color .2s;color:#888}
.hist-tab.active{color:#eee}
.hist-tab.pos.active{border-color:#4ecca3;color:#4ecca3}
.hist-tab.neg.active{border-color:#e74c3c;color:#e74c3c}
#hist-list{flex:1;overflow-y:auto;padding:8px 0}
.hist-item{display:flex;align-items:center;padding:8px 16px;gap:10px;
  border-bottom:1px solid #222}
.hist-item .fname{flex:1;font-size:13px;font-family:monospace;color:#ccc;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hist-btn{background:none;border:1px solid #555;border-radius:6px;
  color:#ccc;width:36px;height:36px;font-size:16px;cursor:pointer;
  display:flex;align-items:center;justify-content:center}
.hist-btn:hover{border-color:#aaa;color:#fff}
.hist-btn.del:hover{border-color:#e74c3c;color:#e74c3c}
.hist-empty{text-align:center;color:#555;padding:40px;font-size:14px}

/* done screen */
#done{display:none;position:fixed;inset:0;background:#1a1a2e;
  flex-direction:column;align-items:center;justify-content:center;gap:16px;
  z-index:20}
#done h1{font-size:28px;color:#4ecca3}
#done .stat{font-size:18px;color:#aaa}
#done .stat b{color:#eee}

/* keyboard hints */
#kbd-hints{position:fixed;bottom:90px;left:0;right:0;text-align:center;
  font-size:11px;color:#444;pointer-events:none}
</style>
</head>
<body>

<div id="progress">
  <div id="progress-bar-outer"><div id="progress-bar-inner"></div></div>
  <div id="progress-text">-</div>
  <button class="btn-history" id="btn-history" title="History">&#9776; History</button>
</div>

<div id="card-area">
  <div id="card">
    <div class="indicator reject">REJECT</div>
    <div class="indicator accept">ACCEPT</div>
    <div class="filename" id="filename">Loading...</div>
    <div class="audio-hint" id="audio-hint">tap to play</div>
  </div>
</div>

<div id="controls">
  <button class="btn btn-reject" id="btn-reject" title="Reject (Arrow Left)">&#10007;</button>
  <button class="btn btn-undo" id="btn-undo" title="Undo (Ctrl+Z)">&#8630;</button>
  <button class="btn btn-delete" id="btn-discard" title="Delete (D)">&#128465;</button>
  <button class="btn btn-replay" id="btn-replay" title="Replay (Space)">&#9654;</button>
  <button class="btn btn-accept" id="btn-accept" title="Accept (Arrow Right)">&#10003;</button>
</div>

<div id="kbd-hints">Arrow keys / swipe &middot; Space = replay &middot; Ctrl+Z = undo</div>

<div id="done">
  <h1>All done!</h1>
  <div class="stat">Reviewed: <b id="done-total">0</b></div>
  <div class="stat" style="color:#4ecca3">Positive: <b id="done-pos">0</b></div>
  <div class="stat" style="color:#e74c3c">Negative: <b id="done-neg">0</b></div>
</div>

<div id="history">
  <div id="hist-header">
    <h2>History</h2>
    <button id="hist-close">&times;</button>
  </div>
  <div id="hist-tabs">
    <div class="hist-tab pos active" data-cls="positive">Positive (0)</div>
    <div class="hist-tab neg" data-cls="negative">Negative (0)</div>
  </div>
  <div id="hist-list"></div>
</div>

<script>
(function(){
  const card      = document.getElementById('card');
  const elName    = document.getElementById('filename');
  const elHint    = document.getElementById('audio-hint');
  const indAccept = card.querySelector('.accept');
  const indReject = card.querySelector('.reject');
  const progBar   = document.getElementById('progress-bar-inner');
  const progText  = document.getElementById('progress-text');
  const doneEl    = document.getElementById('done');

  let state = {filename:null, done:false, reviewed:0, total:0};
  let audio = new Audio();
  let busy  = false;          // prevents double-swipe during animation
  let dragging = false;
  let startX = 0, dx = 0;

  // ---- API helpers ----
  async function api(path, body){
    const opts = body
      ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}
      : {};
    const r = await fetch(path, opts);
    return r.json();
  }

  // ---- UI updates ----
  function updateProgress(d){
    const pct = d.total ? Math.round(d.reviewed/d.total*100) : 0;
    progBar.style.width = pct+'%';
    progText.textContent = d.reviewed+' / '+d.total;
  }

  function showDone(d){
    doneEl.style.display='flex';
    document.getElementById('done-total').textContent = d.stats.positive+d.stats.negative;
    document.getElementById('done-pos').textContent   = d.stats.positive;
    document.getElementById('done-neg').textContent   = d.stats.negative;
  }

  function loadCard(d){
    state = d;
    updateProgress(d);
    if(d.done){ showDone(d); return; }
    doneEl.style.display='none';
    elName.textContent = d.filename;
    elHint.textContent = 'tap to play';
    resetCardPos();
    playFile(d.filename);
  }

  function resetCardPos(){
    card.classList.remove('dragging');
    card.style.transform = '';
    card.style.opacity = '';
    indAccept.style.opacity = 0;
    indReject.style.opacity = 0;
  }

  // ---- audio ----
  function playFile(name){
    audio.pause();
    audio.src = '/api/audio/'+encodeURIComponent(name);
    const p = audio.play();
    if(p) p.catch(()=>{ elHint.textContent='tap to play'; });
  }
  function replay(){
    if(!state.filename) return;
    audio.currentTime=0;
    const p = audio.play();
    if(p) p.catch(()=>{});
  }

  // ---- classify ----
  async function classify(cls){
    if(busy || state.done || !state.filename) return;
    busy = true;
    audio.pause();

    // animate out
    const dir = cls==='positive' ? 1 : -1;
    card.style.transition = 'transform .35s ease, opacity .35s ease';
    card.style.transform  = 'translateX('+dir*120+'%) rotate('+dir*20+'deg)';
    card.style.opacity    = '0';

    const d = await api('/api/classify',{filename:state.filename, classification:cls});

    await new Promise(r=>setTimeout(r,300));

    // reset & load next
    card.style.transition = 'none';
    card.style.transform  = 'translateX(0) scale(.95)';
    card.style.opacity    = '0';
    indAccept.style.opacity = 0;
    indReject.style.opacity = 0;

    // tiny delay then animate in
    requestAnimationFrame(()=>{
      requestAnimationFrame(()=>{
        card.style.transition = 'transform .25s ease, opacity .25s ease';
        card.style.transform  = '';
        card.style.opacity    = '1';
        loadCard(d);
        busy = false;
      });
    });
  }

  async function undo(){
    if(busy) return;
    busy = true;
    audio.pause();
    const d = await api('/api/undo', {});
    loadCard(d);
    busy = false;
  }

  async function discard(){
    if(busy || state.done || !state.filename) return;
    busy = true;
    audio.pause();
    // animate down
    card.style.transition = 'transform .35s ease, opacity .35s ease';
    card.style.transform  = 'translateY(120%) scale(.8)';
    card.style.opacity    = '0';
    const d = await api('/api/delete',{filename:state.filename});
    await new Promise(r=>setTimeout(r,300));
    card.style.transition = 'none';
    card.style.transform  = 'translateX(0) scale(.95)';
    card.style.opacity    = '0';
    indAccept.style.opacity = 0;
    indReject.style.opacity = 0;
    requestAnimationFrame(()=>{
      requestAnimationFrame(()=>{
        card.style.transition = 'transform .25s ease, opacity .25s ease';
        card.style.transform  = '';
        card.style.opacity    = '1';
        loadCard(d);
        busy = false;
      });
    });
  }

  // ---- swipe gestures ----
  function onStart(x){
    if(busy || state.done) return;
    dragging = true; dx = 0; startX = x;
    card.classList.add('dragging');
  }
  function onMove(x){
    if(!dragging) return;
    dx = x - startX;
    const rot = dx * 0.08;
    card.style.transform = 'translateX('+dx+'px) rotate('+rot+'deg)';
    const t = 50;
    indAccept.style.opacity = dx >  t ? Math.min(1,(dx-t)/100) : 0;
    indReject.style.opacity = dx < -t ? Math.min(1,(-dx-t)/100) : 0;
  }
  function onEnd(){
    if(!dragging) return;
    dragging = false;
    card.classList.remove('dragging');
    if(Math.abs(dx) < 10){
      // tap
      resetCardPos();
      replay();
    } else if(dx > 100){
      classify('positive');
    } else if(dx < -100){
      classify('negative');
    } else {
      // snap back
      card.style.transition = 'transform .25s ease';
      card.style.transform = '';
      indAccept.style.opacity = 0;
      indReject.style.opacity = 0;
    }
  }

  // touch
  card.addEventListener('touchstart', e=>{e.preventDefault(); onStart(e.touches[0].clientX);});
  card.addEventListener('touchmove',  e=>{e.preventDefault(); onMove(e.touches[0].clientX);});
  card.addEventListener('touchend',   e=>{e.preventDefault(); onEnd();});
  // mouse
  card.addEventListener('mousedown', e=>{onStart(e.clientX);});
  document.addEventListener('mousemove', e=>{onMove(e.clientX);});
  document.addEventListener('mouseup', ()=>{onEnd();});

  // ---- history dialog ----
  const historyEl  = document.getElementById('history');
  const histList   = document.getElementById('hist-list');
  const histTabs   = document.querySelectorAll('.hist-tab');
  let histAudio    = new Audio();
  let histTab      = 'positive';
  let histData     = {positive:[], negative:[]};

  async function openHistory(){
    audio.pause();
    histData = await api('/api/history');
    renderHistory();
    historyEl.classList.add('open');
  }
  function closeHistory(){
    histAudio.pause();
    historyEl.classList.remove('open');
    // refresh main status in case deletions happened
    api('/api/status').then(d=>loadCard(d));
  }
  function renderHistory(){
    histTabs.forEach(t=>{
      const cls = t.dataset.cls;
      t.classList.toggle('active', cls===histTab);
      const n = histData[cls] ? histData[cls].length : 0;
      t.textContent = (cls==='positive'?'Positive':'Negative')+' ('+n+')';
    });
    const files = histData[histTab] || [];
    if(!files.length){
      histList.innerHTML = '<div class="hist-empty">No files</div>';
      return;
    }
    histList.innerHTML = files.map(f=>
      '<div class="hist-item" data-name="'+f+'">'
      +'<span class="fname">'+f+'</span>'
      +'<button class="hist-btn play" title="Play">&#9654;</button>'
      +'<button class="hist-btn del" title="Delete">&#128465;</button>'
      +'</div>'
    ).join('');
  }
  histList.addEventListener('click', async e=>{
    const btn = e.target.closest('.hist-btn');
    if(!btn) return;
    const item = btn.closest('.hist-item');
    const name = item.dataset.name;
    if(btn.classList.contains('play')){
      histAudio.pause();
      histAudio.src = '/api/audio/'+histTab+'/'+encodeURIComponent(name);
      histAudio.play().catch(()=>{});
    } else if(btn.classList.contains('del')){
      await api('/api/delete-classified',{filename:name, classification:histTab});
      histData[histTab] = histData[histTab].filter(f=>f!==name);
      renderHistory();
    }
  });
  histTabs.forEach(t=>t.addEventListener('click',()=>{
    histTab = t.dataset.cls;
    histAudio.pause();
    renderHistory();
  }));
  document.getElementById('hist-close').addEventListener('click', closeHistory);

  // buttons
  document.getElementById('btn-reject').addEventListener('click', ()=>classify('negative'));
  document.getElementById('btn-accept').addEventListener('click', ()=>classify('positive'));
  document.getElementById('btn-replay').addEventListener('click', ()=>replay());
  document.getElementById('btn-undo').addEventListener('click',   ()=>undo());
  document.getElementById('btn-discard').addEventListener('click', ()=>discard());
  document.getElementById('btn-history').addEventListener('click', ()=>openHistory());

  // keyboard
  document.addEventListener('keydown', e=>{
    if(historyEl.classList.contains('open')){
      if(e.key==='Escape') closeHistory();
      return;
    }
    if(e.key==='ArrowRight') classify('positive');
    else if(e.key==='ArrowLeft') classify('negative');
    else if(e.key===' '){e.preventDefault(); replay();}
    else if(e.key==='z' && (e.ctrlKey||e.metaKey)){e.preventDefault(); undo();}
    else if(e.key==='d' || e.key==='Delete') discard();
  });

  // init
  api('/api/status').then(d=>loadCard(d));
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
        description="Review wake word audio recordings in a swipeable web UI",
    )
    parser.add_argument(
        "--inputDir",
        required=True,
        help="Directory containing WAV files to review",
    )
    parser.add_argument(
        "--posDir",
        required=True,
        help="Directory for positively classified files (right swipe)",
    )
    parser.add_argument(
        "--negDir",
        required=True,
        help="Directory for negatively classified files (left swipe)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP server port (default: 8080)",
    )
    args = parser.parse_args()

    input_dir = Path(args.inputDir)
    pos_dir = Path(args.posDir)
    neg_dir = Path(args.negDir)

    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    fm = FileManager(input_dir, pos_dir, neg_dir)
    ReviewHandler.file_manager = fm

    server = http.server.HTTPServer(("0.0.0.0", args.port), ReviewHandler)
    print(f"Wake Word Reviewer")
    print(f"  http://localhost:{args.port}")
    print(f"  Input:    {input_dir.resolve()} ({len(fm)} files)")
    print(f"  Positive: {pos_dir.resolve()}")
    print(f"  Negative: {neg_dir.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
