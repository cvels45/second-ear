"""Self-contained HTML review report: ranked findings with a redline of the record, both audio clips,
reviewer decisions, consistency findings, the comparison transcript, and a declaration template.

One file, no network: clips are embedded (MP3 when ffmpeg is available, else WAV), so the report
can be emailed to the reviewing interpreter or opened offline.
"""

from __future__ import annotations

import base64
import html
import json
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path

from .judges import LANG

TYPE_LABEL = {
    "omission": "Omission", "addition": "Addition", "distortion": "Changed meaning",
    "count_ordinal": "Changed number or amount", "referent_unit": "Changed unit or referent",
    "date_duration_age": "Changed date, duration, or age", "name": "Changed name",
    "yes_no_reversal": "Reversed yes or no", "hedge": "Dropped or added hedge", "pronoun_shift": "Pronoun shift",
    "softening": "Softened or intensified", "legal_term": "Legal term", "wrong_question": "Different question",
    "self_corrected": "Corrected mid-rendition", "uninterpreted_exchange": "Not interpreted",
}
CHECK_LABEL = {"interpreter_error": "Audio confirms", "uncertain_transcription": "Check the transcript",
               "unchecked": "Not audio-checked"}

e = html.escape


def mmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _embed(path: str | None) -> str:
    if not path or not Path(path).exists():
        return ""
    p = Path(path)
    if shutil.which("ffmpeg"):
        from .audio import to_mp3

        with tempfile.TemporaryDirectory() as d:
            mp3 = to_mp3(p, Path(d) / "c.mp3", bitrate="32k")
            return "data:audio/mpeg;base64," + base64.b64encode(mp3.read_bytes()).decode("ascii")
    return "data:audio/wav;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _mark(text: str, span: str, cls: str = "q") -> str:
    """Escape text and wrap the first case-insensitive occurrence of span."""
    if span:
        m = re.search(re.escape(span.strip()), text, re.I)
        if m:
            return f"{e(text[:m.start()])}<mark class='{cls}'>{e(m.group(0))}</mark>{e(text[m.end():])}"
    return e(text)


def _clean(s: str) -> str:
    """Model text sometimes carries literal \\uXXXX escapes; decode them and tidy whitespace."""
    s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s or "")
    return re.sub(r"\s+", " ", s).strip()


def _first_sentences(s: str, limit: int = 240) -> str:
    if len(s) <= limit:
        return s
    cut = s[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[: end + 1] if end > 60 else cut.rstrip() + "…"


def redline(rend_text: str, rend_words: str, correction: str) -> str:
    """The record sentence with the faulty words struck and the correction inserted after them."""
    ins = f"<ins>{e(correction)}</ins>" if correction else ""
    if rend_words:
        m = re.search(re.escape(rend_words.strip()), rend_text, re.I)
        if m:
            return f"{e(rend_text[:m.start()])}<del>{e(m.group(0))}</del>{ins}{e(rend_text[m.end():])}"
        return f"{e(rend_text)}<span class='alt'><del>{e(rend_words)}</del>{ins}</span>"
    return f"{e(rend_text)} {ins}" if ins else e(rend_text)


def _rank(findings: list[dict], review: dict | None) -> list[dict]:
    order = {fid: k for k, fid in enumerate((review or {}).get("ranking", []))}
    sev = {"major": 0, "minor": 1}
    return sorted(findings, key=lambda f: (order.get(f["finding_id"], 10_000), sev.get(f["severity"], 2),
                                           -f.get("confidence", 0), f["rend_start_s"]))


def render(result: dict, *, title: str, review: dict | None = None, decision: str | None = None,
           models: dict | None = None) -> str:
    lang = LANG.get(result["src_lang"], result["src_lang"])
    findings = _rank(result["findings"], review)
    links = {d["finding_id"]: d for d in (review or {}).get("decision_links", [])}
    by_pair = {}
    for f in findings:
        by_pair.setdefault(f["pair_id"], []).append(f)
    n_major = sum(f["severity"] == "major" for f in findings)
    n_conf = sum(f["check"] == "interpreter_error" for f in findings)
    n_unc = sum(f["check"] == "uncertain_transcription" for f in findings)
    n_unint = sum(f["type"] == "uninterpreted_exchange" for f in findings)
    report_id = re.sub(r"\W+", "-", f"{result['session_id']}-{len(findings)}")

    rows = []
    for k, f in enumerate(findings, 1):
        o_lang, r_lang = (lang, "English") if f["direction"] == "src->en" else ("English", lang)
        clips = f.get("clips") or {}
        oa, ra = _embed(clips.get("orig")), _embed(clips.get("rend"))
        adj = f.get("adjudication") or {}
        link = links.get(f["finding_id"])
        sev = "Major" if f["severity"] == "major" else "Minor"
        conf = ("Found by turn pairing" if f.get("origin") == "structure"
                else f"Model confidence {f.get('confidence', 0):.0%}")
        heard = ""
        if adj.get("verdict"):
            hs, hr = _clean(adj.get("heard_source", "")), _clean(adj.get("heard_rendition", ""))
            spans = (f"Heard: speaker “{e(hs)}”, interpreter “{e(hr)}”. " if hs and hr and len(hs) + len(hr) < 160
                     else "")
            heard = f"<p class='heard'>Audio check: {spans}{e(_first_sentences(_clean(adj.get('reason', ''))))}</p>"
            sec = adj.get("second_asr")
            if sec:
                heard += (f"<p class='heard'>Second transcription (Omnilingual ASR): speaker “{e(sec['orig'])}”; "
                          f"interpreter “{e(sec['rend'])}”.</p>")
        if f["type"] == "uninterpreted_exchange":
            body = (f"<div class='said'><p class='who'>Said in {e(lang)}, never rendered into English</p>"
                    f"<p class='src'>{e(f['orig_text'])}</p></div>")
        else:
            body = (f"<div class='said'><p class='who'>Speaker, in {e(o_lang)}</p>"
                    f"<p class='src' lang='{e(result['src_lang'] if o_lang == lang else 'en')}'>"
                    f"{_mark(f['orig_text'], f['source_words'])}</p></div>"
                    f"<div class='record'><p class='who'>Interpreter, in {e(r_lang)}"
                    f"{', with the proposed correction' if f.get('correction') else ''}</p>"
                    f"<p class='rec'>{redline(f['rend_text'], f['rendition_words'], f.get('correction', ''))}</p></div>")
        play = ""
        if oa or ra:
            play = ("<div class='play'>"
                    + (f"<button type='button' class='pb' data-src='{oa}'>Play speaker</button>" if oa else "")
                    + (f"<button type='button' class='pb' data-src='{ra}'>Play interpreter</button>" if ra else "")
                    + "</div>")
        linked = (f"<blockquote class='link'><p>The decision relies on this testimony: “{e(link['decision_sentence'])}”</p>"
                  f"<p>{e(link['how'])}</p></blockquote>") if link else ""
        rows.append(f"""
<article class="finding" id="{e(f['finding_id'])}" data-check="{e(f['check'])}" data-sev="{e(f['severity'])}">
  <div class="gutter"><span class="rank">{k}</span><a class="ts" href="#{e(f['finding_id'])}">{mmss(f['rend_start_s'])}</a></div>
  <div class="main">
    <h3>{e(TYPE_LABEL.get(f['type'], f['type']))} <span class="sev {e(f['severity'])}">{sev}</span>
      <span class="chk {e(f['check'])}">{e(CHECK_LABEL.get(f['check'], f['check']))}</span></h3>
    {body}
    <p class="why">{e(f['explanation'])}</p>
    {heard}{linked}{play}
    <div class="decide" role="group" aria-label="Reviewer decision for finding {k}">
      <button type="button" data-v="confirmed">Confirm</button><button type="button" data-v="rejected">Reject</button>
      <input type="text" placeholder="Reviewer note" aria-label="Reviewer note for finding {k}">
      <span class="conf">{conf}</span>
    </div>
  </div>
</article>""")

    cons = (review or {}).get("consistency", [])
    cons_html = "".join(
        f"<li><p><strong>{'Record-only inconsistency' if c['kind'] == 'record_only_inconsistency' else 'Inconsistent rendering'}</strong>"
        f" ({e(', '.join(c['pair_ids']))})</p><p>The record says: “{e(c['english_record'])}”. The speaker said: "
        f"“{e(c['source_says'])}”.</p><p class='why'>{e(c['explanation'])}</p></li>" for c in cons)

    trows = []
    for p in result["pairs"]:
        fl = by_pair.get(p["pair_id"], [])
        refs = " ".join(f"<a href='#{e(f['finding_id'])}'>{e(TYPE_LABEL.get(f['type'], f['type']))}</a>" for f in fl)
        trows.append(f"<tr class='{'flag' if fl else ''}'><td class='t'>{mmss(p['orig_start_s'])}</td>"
                     f"<td>{e(p['orig_text'])}</td><td>{e(p['rend_text'])}</td><td>{refs}</td></tr>")

    summary = e((review or {}).get("summary", "")) or (
        f"{len(findings)} possible interpreting errors were found, {n_major} of them major. Each needs review by a "
        "qualified interpreter before it is relied on.")
    meta = models or {}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title>
<style>{CSS}</style></head>
<body data-report="{e(report_id)}">
<header class="top">
  <p class="product">Second Ear interpretation review</p>
  <h1>{e(title)}</h1>
  <p class="facts">English and {e(lang)}, {mmss(result['duration_s'])} of audio, {len(result['pairs'])} interpreted exchanges.
  Generated {date.today().isoformat()}.</p>
  <p class="notice">Draft for review by a qualified interpreter. This is not a certified translation, and every finding must be checked against the audio before it is relied on.</p>
</header>
<main>
<section class="summary">
  <p>{summary}</p>
  <dl class="counts">
    <div><dt>Findings</dt><dd>{len(findings)}</dd></div><div><dt>Major</dt><dd>{n_major}</dd></div>
    <div><dt>Audio confirms</dt><dd>{n_conf}</dd></div><div><dt>Check the transcript</dt><dd>{n_unc}</dd></div>
    <div><dt>Not interpreted</dt><dd>{n_unint}</dd></div><div><dt>Linked to the decision</dt><dd>{len(links)}</dd></div>
  </dl>
</section>
<section aria-labelledby="h-f">
  <div class="bar"><h2 id="h-f">Findings, most consequential first</h2>
  <div class="filters" role="group" aria-label="Filter findings">
    <button type="button" data-f="all" aria-pressed="true">All</button>
    <button type="button" data-f="major" aria-pressed="false">Major</button>
    <button type="button" data-f="interpreter_error" aria-pressed="false">Audio confirms</button>
    <button type="button" data-f="uncertain_transcription" aria-pressed="false">Check the transcript</button>
  </div></div>
  {''.join(rows) or "<p class='empty'>No interpreting errors were found. Spot-check the comparison transcript below.</p>"}
  <p class="export"><button type="button" id="export">Download reviewer decisions (CSV)</button></p>
</section>
{f'<section aria-labelledby="h-c"><h2 id="h-c">Across the hearing</h2><ul class="cons">{cons_html}</ul></section>' if cons else ''}
{f'<section aria-labelledby="h-d"><h2 id="h-d">Decision</h2><p class="decision">{e(decision)}</p></section>' if decision else ''}
<section aria-labelledby="h-t"><h2 id="h-t">Comparison transcript</h2>
  <p class="hint">Machine transcript of both voices, paired by exchange. Flagged rows link to their finding.</p>
  <div class="tw"><table><thead><tr><th>Time</th><th>Speaker</th><th>Interpreter</th><th>Flag</th></tr></thead>
  <tbody>{''.join(trows)}</tbody></table></div>
</section>
<section aria-labelledby="h-s"><h2 id="h-s">Declaration template</h2>
  <p class="hint">A starting point for the reviewing interpreter. Adapt it to the forum's requirements; it is not legal advice.</p>
  <div class="decl">
  <p>I, [full name], declare that I am competent to translate from {e(lang)} into English and from English into {e(lang)}[, and that I am certified by (certifying body)].</p>
  <p>I listened to the recording of [proceeding, date]. I compared what was said in {e(lang)} with the interpreter's English rendition, and found the discrepancies listed in the attached table, each identified by its time in the recording. For each, the table gives the words spoken, the words interpreted, and my translation.</p>
  <p>I declare under penalty of perjury that the foregoing is true and correct.</p>
  <p>[Signature] [Date] [Address]</p>
  </div>
</section>
</main>
<footer class="foot"><p>Models: {e(', '.join(f'{k} {v}' for k, v in meta.items()) or 'see run metadata')}. Findings are machine-generated leads, not conclusions.</p></footer>
<audio id="player" preload="none"></audio>
<script>{JS}</script>
</body></html>"""


CSS = r"""
:root{--paper:#FAFAFB;--ink:#1C2230;--soft:#5B6475;--rule:#D9DCE3;--del:#B3261E;--ins:#1F4FB3;--unc:#8A5A00;
--mark:#FFF1B8;--wash:#F1F3F7;--serif:Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#14171E;--ink:#E6E8EE;--soft:#A3AAB8;--rule:#2E3442;
--del:#FF8A80;--ins:#8AB4FF;--unc:#F2C46D;--mark:#4A3F12;--wash:#1C2029}}
:root[data-theme="dark"]{--paper:#14171E;--ink:#E6E8EE;--soft:#A3AAB8;--rule:#2E3442;--del:#FF8A80;--ins:#8AB4FF;--unc:#F2C46D;--mark:#4A3F12;--wash:#1C2029}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 var(--sans);-webkit-text-size-adjust:100%}
header.top,main,footer.foot{max-width:920px;margin:0 auto;padding:0 20px}
header.top{padding-top:40px;padding-bottom:8px;border-bottom:2px solid var(--ink)}
.product{margin:0;color:var(--soft);font-size:15px}
h1{font:600 34px/1.15 var(--serif);margin:6px 0 10px;letter-spacing:-.01em}
h2{font:600 22px/1.25 var(--serif);margin:0}
h3{font:600 18px/1.3 var(--serif);margin:0 0 10px;display:flex;flex-wrap:wrap;gap:8px;align-items:baseline}
.facts{margin:0 0 6px;color:var(--soft)}
.notice{margin:12px 0 16px;padding-left:12px;border-left:3px solid var(--unc);font-size:15px}
section{padding:28px 0;border-bottom:1px solid var(--rule)}
.summary p{font:18px/1.6 var(--serif);max-width:68ch;margin:0 0 18px}
.counts{display:grid;grid-template-columns:repeat(6,1fr);gap:0;margin:0;border-top:1px solid var(--rule)}
.counts div{padding:10px 12px 0 0}
.counts dt{color:var(--soft);font-size:13px}
.counts dd{margin:0;font:600 26px/1.2 var(--serif);font-variant-numeric:tabular-nums}
.bar{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;margin-bottom:8px}
button{font:inherit;font-size:14px;color:var(--ink);background:transparent;border:1px solid var(--rule);border-radius:6px;padding:5px 11px;cursor:pointer}
button:hover{border-color:var(--soft)}
button:focus-visible,a:focus-visible,input:focus-visible{outline:2px solid var(--ins);outline-offset:2px}
.filters button[aria-pressed="true"]{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.finding{display:grid;grid-template-columns:72px 1fr;gap:16px;padding:22px 0;border-top:1px solid var(--rule)}
.finding[hidden]{display:none}
.gutter{display:flex;flex-direction:column;align-items:flex-start;gap:2px}
.rank{color:var(--soft);font-size:13px}
.ts{font:600 17px/1.2 var(--sans);font-variant-numeric:tabular-nums;color:var(--ink);text-decoration:none;border-bottom:1px solid var(--rule)}
.sev{font:500 13px/1 var(--sans);padding:3px 7px;border-radius:4px;border:1px solid var(--rule)}
.sev.major{border-color:var(--del);color:var(--del)}
.chk{font:500 13px/1 var(--sans);color:var(--soft)}
.chk.interpreter_error{color:var(--ins)}
.chk.uncertain_transcription{color:var(--unc)}
.who{margin:0 0 2px;color:var(--soft);font-size:13px}
.said,.record{margin:0 0 12px}
.src,.rec{margin:0;font:18px/1.6 var(--serif);max-width:72ch}
mark.q{background:var(--mark);color:inherit;padding:0 2px;border-radius:2px}
del{color:var(--del);text-decoration:line-through;text-decoration-thickness:2px}
ins{color:var(--ins);text-decoration:underline;text-decoration-thickness:2px;text-underline-offset:3px;margin-left:4px}
.alt{display:block;margin-top:4px}
.why{margin:0 0 10px;max-width:72ch}
.heard{margin:0 0 10px;color:var(--soft);font-size:15px;max-width:72ch}
.link{margin:0 0 12px;padding:8px 12px;background:var(--wash);border-left:3px solid var(--ins)}
.link p{margin:0 0 4px}
.play{display:flex;gap:8px;margin:0 0 12px;flex-wrap:wrap}
.pb[aria-pressed="true"]{border-color:var(--ins);color:var(--ins)}
.decide{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.decide input{flex:1 1 220px;min-width:0;font:inherit;font-size:14px;padding:5px 9px;border:1px solid var(--rule);border-radius:6px;background:transparent;color:var(--ink)}
.decide button[data-on="confirmed"]{background:var(--ins);border-color:var(--ins);color:var(--paper)}
.decide button[data-on="rejected"]{background:var(--soft);border-color:var(--soft);color:var(--paper)}
.conf{color:var(--soft);font-size:13px;font-variant-numeric:tabular-nums}
.export{margin:18px 0 0}
.cons{padding-left:20px}.cons li{margin-bottom:14px}.cons p{margin:0 0 4px}
.decision{font:17px/1.6 var(--serif);max-width:72ch}
.hint{color:var(--soft);margin:6px 0 12px;max-width:72ch}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:15px}
th{text-align:left;font-weight:600;color:var(--soft);font-size:13px;border-bottom:1px solid var(--ink);padding:6px 8px}
td{vertical-align:top;padding:7px 8px;border-bottom:1px solid var(--rule)}
td.t{font-variant-numeric:tabular-nums;color:var(--soft);white-space:nowrap}
tr.flag td{background:var(--wash)}tr.flag td:first-child{box-shadow:inset 3px 0 var(--del)}
.decl{font:16px/1.65 var(--serif);max-width:72ch;padding:4px 0 0}
.empty{font:18px/1.6 var(--serif)}
footer.foot{padding:18px 20px 40px;color:var(--soft);font-size:13px}
@media (max-width:640px){h1{font-size:27px}.counts{grid-template-columns:repeat(3,1fr)}
.finding{grid-template-columns:1fr;gap:6px}.gutter{flex-direction:row;gap:10px;align-items:baseline}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
@media print{.filters,.play,.decide button,.export,.decide input{display:none}.finding{break-inside:avoid}
body{background:#fff;color:#000}section{border-color:#bbb}}
"""

JS = r"""
(function(){
var id=document.body.dataset.report, key='second-ear:'+id, state={};
try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function save(){try{localStorage.setItem(key,JSON.stringify(state))}catch(e){}}
document.querySelectorAll('.finding').forEach(function(el){
  var s=state[el.id]||{}, inp=el.querySelector('.decide input');
  function paint(){el.querySelectorAll('.decide button').forEach(function(b){
    if(s.v===b.dataset.v){b.dataset.on=s.v;b.setAttribute('aria-pressed','true')}else{delete b.dataset.on;b.setAttribute('aria-pressed','false')}})}
  if(inp){inp.value=s.note||'';inp.addEventListener('input',function(){s.note=inp.value;state[el.id]=s;save()})}
  el.querySelectorAll('.decide button').forEach(function(b){b.addEventListener('click',function(){
    s.v=(s.v===b.dataset.v)?null:b.dataset.v;state[el.id]=s;save();paint()})});
  paint();
});
var player=document.getElementById('player'), current=null;
document.querySelectorAll('.pb').forEach(function(b){b.addEventListener('click',function(){
  if(current===b&&!player.paused){player.pause();return}
  if(current)current.setAttribute('aria-pressed','false');
  current=b;player.src=b.dataset.src;player.play();b.setAttribute('aria-pressed','true')})});
player.addEventListener('ended',function(){if(current)current.setAttribute('aria-pressed','false')});
player.addEventListener('pause',function(){if(current)current.setAttribute('aria-pressed','false')});
document.querySelectorAll('.filters button').forEach(function(b){b.addEventListener('click',function(){
  document.querySelectorAll('.filters button').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false')});
  var f=b.dataset.f;document.querySelectorAll('.finding').forEach(function(el){
    el.hidden=!(f==='all'||(f==='major'&&el.dataset.sev==='major')||el.dataset.check===f)})})});
var ex=document.getElementById('export');
if(ex)ex.addEventListener('click',function(){
  var rows=[['rank','time','finding','type','decision','note']];
  document.querySelectorAll('.finding').forEach(function(el){var s=state[el.id]||{};
    rows.push([el.querySelector('.rank').textContent,el.querySelector('.ts').textContent,el.id,
      el.querySelector('h3').firstChild.textContent.trim(),s.v||'',s.note||''])});
  var csv=rows.map(function(r){return r.map(function(c){return '"'+String(c).replace(/"/g,'""')+'"'}).join(',')}).join('\n');
  var a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download=id+'-decisions.csv';a.click()});
})();
"""


def write(path: Path, result: dict, **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(result, **kw), encoding="utf-8")
    return path


def load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
