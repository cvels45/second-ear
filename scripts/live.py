"""Near-live check of an interpreted prep session: findings appear a few seconds after each rendition.

Live mode is only for sessions where every participant agreed to be recorded (client prep sessions)
or a deposition noticed for recording. It refuses to start without --consent. Never use it in
immigration court or at a USCIS interview.

Usage (from second-ear/):
  uv run python scripts/live.py --file ..\\data\\dev\\sessions\\es-S02.wav --lang es --consent --play
  uv run python scripts/live.py --mic --lang es --consent
Then open http://127.0.0.1:8765. At the end, the full report is written next to the findings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from second_ear import audio, report  # noqa: E402
from second_ear.live import EventBus, LiveSession, Segmenter, file_blocks, mic_blocks, run_audio, sse_server  # noqa: E402
from second_ear.llm import LLMClient  # noqa: E402
from second_ear.pairing import lexical_sim  # noqa: E402
from second_ear.pipeline import Calibrator  # noqa: E402
from second_ear.settings import PKG_ROOT, load_settings  # noqa: E402

VT_LANG = {"es": "Spanish", "pt": "Portuguese", "zh": "Chinese", "hi": "Hindi", "ar": "Arabic", "ta": "Tamil"}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Second Ear live check</title>
<style>__CSS__
.live{display:flex;align-items:center;gap:10px;color:var(--soft);font-size:15px;margin:0 0 14px}
.dot{width:10px;height:10px;border-radius:50%;background:var(--del)}
.dot.on{animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{50%{opacity:.25}}
.now{font:17px/1.5 var(--serif);color:var(--soft);min-height:1.5em;margin:0;max-width:72ch}
.row{display:grid;grid-template-columns:64px 1fr 1fr;gap:14px;padding:12px 0;border-top:1px solid var(--rule)}
.row .t{color:var(--soft);font-variant-numeric:tabular-nums;font-size:14px;padding-top:3px}
.row p{margin:0;font:16px/1.55 var(--serif)}
.row .state{grid-column:2/4;color:var(--soft);font-size:13px}
.row.flagged{background:var(--wash);box-shadow:inset 3px 0 var(--del);padding-left:10px}
.fcard{grid-column:2/4;margin-top:4px}
.fcard h3{font-size:17px;margin:0 0 6px}
.fresh{animation:arrive .9s ease-out}
@keyframes arrive{from{background:var(--mark)}}
.cols{display:grid;grid-template-columns:64px 1fr 1fr;gap:14px;color:var(--soft);font-size:13px;padding:0 0 6px}
@media (max-width:640px){.row,.cols{grid-template-columns:1fr}.row .state,.fcard{grid-column:auto}}
@media (prefers-reduced-motion:reduce){.dot.on,.fresh{animation:none}}
</style></head><body>
<header class="top"><p class="product">Second Ear live check</p><h1>__TITLE__</h1>
<p class="facts">English and __LANG__. Findings appear a few seconds after each rendition.</p>
<p class="notice">Recording with every participant's consent. Live findings are leads for the attorney to raise with the interpreter, not conclusions.</p></header>
<main><section>
<p class="live"><span class="dot on" id="dot"></span><span id="status">Listening</span></p>
<p class="now" id="now" aria-live="polite"></p></section>
<section aria-labelledby="h-x"><div class="bar"><h2 id="h-x">Exchanges</h2><span class="conf" id="count">No findings yet</span></div>
<div class="cols"><span>Time</span><span>Speaker</span><span>Interpreter</span></div>
<div id="rows"></div></section></main>
<script>
var TYPES=__TYPES__, rows=document.getElementById('rows'), nf=0;
function esc(s){return String(s||'').replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})}
function mmss(t){t=Math.floor(t);return Math.floor(t/60)+':'+String(t%60).padStart(2,'0')}
function red(text,w,c){var i=w?text.toLowerCase().indexOf(w.toLowerCase()):-1, ins=c?'<ins>'+esc(c)+'</ins>':'';
  if(i<0)return esc(text)+(w?' <del>'+esc(w)+'</del>'+ins:(ins?' '+ins:''));
  return esc(text.slice(0,i))+'<del>'+esc(text.slice(i,i+w.length))+'</del>'+ins+esc(text.slice(i+w.length))}
function row(id,t,o,r){var el=document.getElementById(id);if(el)return el;el=document.createElement('div');el.className='row';el.id=id;
  el.innerHTML='<span class="t">'+mmss(t)+'</span><p>'+esc(o)+'</p><p class="r">'+esc(r)+'</p><span class="state">Checking</span>';
  var kids=[].slice.call(rows.children), after=kids.filter(function(k){return +k.dataset.t>t})[0];el.dataset.t=t;
  rows.insertBefore(el,after||null);return el}
var es=new EventSource('/events');
es.onmessage=function(m){var e=JSON.parse(m.data);
  if(e.kind==='turn'){document.getElementById('now').textContent=e.text}
  else if(e.kind==='pair'){row(e.pair_id,e.start_s,e.orig,e.rend).scrollIntoView({block:'nearest'})}
  else if(e.kind==='judged'){var el=document.getElementById(e.pair_id),st=el&&el.querySelector('.state');if(st)st.remove()}
  else if(e.kind==='finding'){var id=e.pair_id||e.finding_id, el=row(id,e.orig_start_s,e.orig_text,e.rend_text);
    el.classList.add('flagged');var st=el.querySelector('.state');if(st)st.remove();
    if(e.rend_text)el.querySelector('.r').innerHTML=red(e.rend_text,e.rendition_words,e.correction);
    var c=document.createElement('div');c.className='fcard fresh';
    c.innerHTML='<h3>'+esc(TYPES[e.type]||e.type)+' <span class="sev '+e.severity+'">'+(e.severity==='major'?'Major':'Minor')+
      '</span></h3><p class="why">'+esc(e.explanation)+'</p>';
    el.appendChild(c);el.scrollIntoView({block:'nearest'});nf++;document.getElementById('count').textContent=nf+(nf===1?' finding':' findings')}
  else if(e.kind==='status'){document.getElementById('status').textContent=e.text;
    if(e.text==='Session ended')document.getElementById('dot').classList.remove('on')}};
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="play a recording at real-time speed")
    src.add_argument("--mic", action="store_true")
    ap.add_argument("--lang", required=True, choices=sorted(VT_LANG))
    ap.add_argument("--consent", action="store_true", help="every participant agreed to be recorded")
    ap.add_argument("--play", action="store_true", help="play the file through the speakers (needs sounddevice)")
    ap.add_argument("--title", default="Prep session")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--yes", action="store_true", help="start at once and exit at the end (no Enter prompts)")
    args = ap.parse_args()
    if not args.consent:
        print("Live mode needs every participant's consent to record (pass --consent). It is for client prep "
              "sessions and noticed depositions only, never immigration court or USCIS interviews.")
        return 2
    s = load_settings()
    out = args.out or s.data_dir / "live" / (args.file.stem if args.file else "mic")
    out.mkdir(parents=True, exist_ok=True)

    try:
        from second_ear.sim import LabseSim

        sim = LabseSim()
    except ImportError:
        sim = lexical_sim
    bus = EventBus()
    client = LLMClient(settings=s)
    live = LiveSession(out.name, args.lang, client=client, sim=sim, calibrator=Calibrator(PKG_ROOT / "eval" / "calibrator.json"),
                       bus=bus, language_bias=[VT_LANG[args.lang], "English"])
    page = (PAGE.replace("__CSS__", report.CSS).replace("__TITLE__", args.title)
            .replace("__LANG__", report.LANG.get(args.lang, args.lang)).replace("__TYPES__", json.dumps(report.TYPE_LABEL)))
    srv = sse_server(bus, page, args.port)
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    print(f"Live page: {url}")
    if not args.yes:
        webbrowser.open(url)

    sr = 16000
    player = None
    if args.file:
        wav = args.file
        if wav.suffix.lower() != ".wav":
            wav = audio.to_wav(args.file, out / "audio16k.wav")
        blocks = file_blocks(wav, sr)
        if args.play:
            try:
                import sounddevice as sd

                player = sd.OutputStream(samplerate=sr, channels=1, dtype="int16")
                player.start()
            except (ImportError, OSError) as exc:  # no audio device or package: run silently, still in real time
                print(f"Playback unavailable ({exc}); continuing without sound. Install with: uv sync")
    else:
        blocks = mic_blocks(sr)
    if not args.yes:
        input("Press Enter to start. ")
    t_start = time.time()
    bus.emit("status", text="Listening")
    try:
        run_audio(blocks, sr, Segmenter(sr), live, player=player)
    except KeyboardInterrupt:
        pass
    finally:
        if player is not None:
            player.stop()
    bus.emit("status", text="Session ended")
    res = live.result(live.clock)
    (out / "findings.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out / "events.jsonl").open("w", encoding="utf-8") as fh:  # for scripts/replay_live.py
        for ev in bus.events:
            fh.write(json.dumps({**ev, "t_rel": round(ev["t"] - t_start, 2)}, ensure_ascii=False) + "\n")
    path = report.write(out / "report.html", res, title=args.title,
                        models={"judges": "Muse Spark 1.3", "speech": "Voice Transcribe"})
    print(f"{len(res['findings'])} findings. Report: {path}. Cost ${client.ledger.run_spent():.3f}")
    if not args.yes:
        input("Press Enter to close the live page. ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
