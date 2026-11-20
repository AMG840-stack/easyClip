"""VIBRA CLIP ENGINE — upload a video, get finished 9:16 clips (auto transcription, viral moment
detection, face-tracked reframing, word-by-word animated captions)."""
import os, json, uuid, shutil, subprocess, threading, zipfile, statistics, secrets
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

ROOT = Path(os.getenv("DATA_DIR", "/tmp/jobs")); ROOT.mkdir(parents=True, exist_ok=True)
WHISPER = os.getenv("WHISPER_MODEL", "small")
CLAUDE = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
PASSWORD = os.getenv("APP_PASSWORD", "")
app = FastAPI(title="VIBRA CLIP ENGINE")
sec = HTTPBasic(auto_error=False)
JOBS, LOCK, _model = {}, threading.Semaphore(1), None

def auth(c: HTTPBasicCredentials = Depends(sec)):
    if PASSWORD and (not c or not secrets.compare_digest(c.password, PASSWORD)):
        raise HTTPException(401, headers={"WWW-Authenticate": "Basic"})

def run(cmd, **k):
    r = subprocess.run(cmd, capture_output=True, text=True, **k)
    if r.returncode: raise RuntimeError(r.stderr[-600:])
    return r.stdout

def probe(p):
    o = json.loads(run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height:format=duration","-of","json",str(p)]))
    return o["streams"][0]["width"], o["streams"][0]["height"], float(o["format"]["duration"])

def transcribe(wav):
    global _model
    from faster_whisper import WhisperModel
    if _model is None: _model = WhisperModel(WHISPER, device="cpu", compute_type="int8")
    segs, info = _model.transcribe(str(wav), word_timestamps=True, vad_filter=True)
    out = []
    for s in segs:
        out.append({"s": s.start, "e": s.end, "t": s.text.strip(),
                    "w": [{"s": w.start, "e": w.end, "t": w.word.strip()} for w in (s.words or [])]})
    return out, info.language

def detect(segs, n, mn, mx, lang, platform, total):
    import anthropic
    txt = "\n".join(f"[{x['s']:.1f}-{x['e']:.1f}] {x['t']}" for x in segs)[:120000]
    prompt = f"""You are an elite short-form video editor and viral strategist. From this timestamped transcript pick the {n} best moments to cut into vertical clips for {platform}.
Rules: each clip {mn}-{mx} seconds; open on a hook that grabs in the first 3 seconds; end on a complete thought (never mid-sentence); no overlaps; use exact transcript timestamps in seconds. Score 0-100 (hook, payoff, emotion, shareability, standalone clarity). Sort by score descending.
Write title, hook, caption and hashtags in {lang}. Return ONLY a JSON array of {{"start":num,"end":num,"score":num,"title":str,"hook":str,"why":str,"caption":str,"hashtags":[str]}}.
TRANSCRIPT:
{txt}"""
    r = anthropic.Anthropic().messages.create(model=CLAUDE, max_tokens=6000, messages=[{"role":"user","content":prompt}])
    t = r.content[0].text; arr = json.loads(t[t.index("["): t.rindex("]")+1])
    clips = []
    for c in arr:
        s, e = max(0, float(c["start"])), min(total, float(c["end"]))
        ss = [x for x in segs if x["s"] <= s + 0.5 and x["e"] > s] or [x for x in segs if x["s"] >= s]
        es = [x for x in segs if x["e"] >= e - 0.5 and x["s"] < e] or segs[-1:]
        if ss: s = ss[0]["s"]
        if es: e = min(total, es[0]["e"] if es[0]["e"] > e else e)
        if e - s >= 5: clips.append({**c, "start": s, "end": e + 0.3, "score": int(c.get("score", 0))})
    clips.sort(key=lambda c: -c["score"])
    keep = []
    for c in clips:
        if all(c["end"] <= k["start"] or c["start"] >= k["end"] for k in keep): keep.append(c)
    return keep[:n]

def face_center(src, s, e, w, h):
    import cv2
    cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap, xs, t = cv2.VideoCapture(str(src)), [], s
    while t < e:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
        if not ok: break
        sc = 480 / f.shape[0]; g = cv2.cvtColor(cv2.resize(f, None, fx=sc, fy=sc), cv2.COLOR_BGR2GRAY)
        fs = cas.detectMultiScale(g, 1.2, 5, minSize=(40, 40))
        if len(fs): x, y, fw, fh = max(fs, key=lambda b: b[2]*b[3]); xs.append((x + fw/2) / sc)
        t += 0.75
    cap.release()
    return statistics.median(xs) if len(xs) >= 3 else w / 2

def ts(x):
    return f"{int(x//3600)}:{int(x%3600//60):02d}:{x%60:05.2f}"

def make_ass(segs, s, e, path):
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            "Style: C,DejaVu Sans,78,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,6,2,2,60,60,520,1\n\n"
            "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    words = [w for x in segs for w in x["w"] if w["e"] > s and w["s"] < e]
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) == 3 or w["t"].endswith((".", "?", "!", ",")): lines.append(cur); cur = []
    if cur: lines.append(cur)
    ev = ""
    for g in lines:
        for i, w in enumerate(g):
            a = max(0, w["s"] - s); b = (g[i+1]["s"] if i+1 < len(g) else w["e"]) - s
            txt = " ".join((f"{{\\c&H00D7FF&\\fscx115\\fscy115}}{x['t'].upper()}{{\\c&HFFFFFF&\\fscx100\\fscy100}}" if j == i else x["t"].upper()) for j, x in enumerate(g))
            ev += f"Dialogue: 0,{ts(a)},{ts(max(b, a+0.05))},C,,0,0,0,,{txt}\n"
    Path(path).write_text(head + ev, encoding="utf-8")

def srt(segs, s, e):
    f = lambda x: f"{int(x//3600):02d}:{int(x%3600//60):02d}:{int(x%60):02d},{int(x%1*1000):03d}"
    o, k = "", 1
    for g in segs:
        if g["e"] > s and g["s"] < e:
            o += f"{k}\n{f(max(0,g['s']-s))} --> {f(min(e,g['e'])-s)}\n{g['t']}\n\n"; k += 1
    return o

def pipeline(jid, src, opt):
    j = JOBS[jid]; d = ROOT / jid
    def st(p, m): j.update(progress=p, message=m)
    try:
        with LOCK:
            if opt.get("source_url"):
                import urllib.request
                st(3, "Téléchargement de la vidéo…")
                with urllib.request.urlopen(opt["source_url"], timeout=120) as r, open(src, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            st(5, "Analyse de la vidéo…"); w, h, dur = probe(src)
            st(10, "Extraction audio…"); wav = d / "a.wav"
            run(["ffmpeg","-y","-i",str(src),"-vn","-ac","1","-ar","16000",str(wav)])
            st(15, "Transcription IA (Whisper)…"); segs, lg = transcribe(wav)
            if not segs: raise RuntimeError("Aucune parole détectée dans la vidéo.")
            (d / "transcript.json").write_text(json.dumps(segs, ensure_ascii=False))
            st(45, "Détection des meilleurs moments (Claude)…")
            clips = detect(segs, opt["n"], opt["mn"], opt["mx"], opt["lang"], opt["platform"], dur)
            if not clips: raise RuntimeError("Aucun clip pertinent trouvé.")
            for i, c in enumerate(clips, 1):
                st(50 + int(45 * (i-1) / len(clips)), f"Montage du clip {i}/{len(clips)}…")
                a, b = c["start"], c["end"]; ar = w / h
                if ar <= 9/16 + 0.02:
                    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
                else:
                    cw = h * 9 / 16; cx = face_center(src, a, b, w, h)
                    x = int(min(max(cx - cw/2, 0), w - cw)); vf = f"crop={int(cw)//2*2}:{h}:{x}:0,scale=1080:1920"
                make_ass(segs, a, b, d / f"clip_{i}.ass")
                out = f"clip_{i}.mp4"
                run(["ffmpeg","-y","-ss",f"{a:.2f}","-i",str(src),"-t",f"{b-a:.2f}","-vf",f"{vf},ass=clip_{i}.ass",
                     "-af","loudnorm=I=-16:TP=-1.5","-c:v","libx264","-crf","19","-preset","medium","-pix_fmt","yuv420p",
                     "-c:a","aac","-b:a","160k","-movflags","+faststart",out], cwd=d)
                (d / f"clip_{i}.srt").write_text(srt(segs, a, b), encoding="utf-8")
                c["file"] = out; c["duration"] = round(b - a, 1); c["id"] = i
            j["clips"] = clips
            (d / "clips.json").write_text(json.dumps(clips, ensure_ascii=False, indent=2))
            with zipfile.ZipFile(d / "clips.zip", "w") as z:
                for c in clips: z.write(d / c["file"], c["file"]); z.write(d / f"clip_{c['id']}.srt", f"clip_{c['id']}.srt")
                z.write(d / "clips.json", "clips.json")
        j["status"] = "done"; st(100, "Terminé")
    except Exception as ex:
        j["status"] = "error"; j["message"] = str(ex)[:500]
    finally:
        for f in (d / "a.wav",): f.unlink(missing_ok=True)

@app.get("/health")
def health(): return {"ok": True}

@app.post("/api/jobs", dependencies=[Depends(auth)])
async def create(file: UploadFile | None = File(None), source_url: str = Form(""), n: int = Form(6), mn: int = Form(20), mx: int = Form(60),
                 lang: str = Form("English"), platform: str = Form("TikTok / Reels / Shorts")):
    jid = uuid.uuid4().hex[:10]; d = ROOT / jid; d.mkdir(parents=True)
    if source_url:
        src = d / "source.mp4"
    elif file:
        src = d / ("source" + Path(file.filename or "v.mp4").suffix)
        with open(src, "wb") as f:
            while chunk := await file.read(1 << 20): f.write(chunk)
    else:
        raise HTTPException(400, "file or source_url required")
    JOBS[jid] = {"status": "running", "progress": 1, "message": "En file d’attente…", "clips": []}
    threading.Thread(target=pipeline, args=(jid, src, dict(n=max(1,min(n,15)), mn=mn, mx=mx, lang=lang, platform=platform, source_url=source_url)), daemon=True).start()
    return {"id": jid}

@app.get("/api/jobs/{jid}", dependencies=[Depends(auth)])
def status(jid: str):
    if jid not in JOBS: raise HTTPException(404)
    return JOBS[jid]

@app.get("/files/{jid}/{name}", dependencies=[Depends(auth)])
def files(jid: str, name: str):
    p = (ROOT / jid / name).resolve()
    if ROOT.resolve() not in p.parents or not p.is_file() or name.startswith("source"): raise HTTPException(404)
    return FileResponse(p, filename=name)

UI = """<!DOCTYPE html><html lang=fr><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>VIBRA CLIP ENGINE</title><style>
:root{--bg:#0f1112;--c:#181b1c;--t:#f1efe8;--m:#9a9a94;--a:#e0b04a;--g:#4fc08d;--b:#2a2e30}
@media(prefers-color-scheme:light){:root{--bg:#faf8f3;--c:#fff;--t:#1a1a1a;--m:#6b6b6b;--a:#b8860b;--g:#1f6f4a;--b:#e4dfd3}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--t);font:15px/1.5 system-ui,sans-serif}main{max-width:900px;margin:auto;padding:16px}
h1{font-size:20px}h1 b{color:var(--a)}.box{background:var(--c);border:1px solid var(--b);border-radius:12px;padding:14px;margin-bottom:12px}
#drop{border:2px dashed var(--b);border-radius:12px;padding:30px;text-align:center;cursor:pointer;color:var(--m)}#drop.on{border-color:var(--a)}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}.row div{flex:1;min-width:110px}label{font-size:11px;color:var(--m);text-transform:uppercase;display:block}
input,select{width:100%;background:var(--bg);color:var(--t);border:1px solid var(--b);border-radius:8px;padding:8px;font:inherit}
button,a.btn{background:var(--a);color:#111;border:0;border-radius:8px;padding:10px 14px;font:600 14px inherit;cursor:pointer;text-decoration:none;display:inline-block}
.bar{height:8px;background:var(--b);border-radius:8px;overflow:hidden;margin:10px 0}.bar i{display:block;height:100%;background:var(--g);width:0;transition:width .4s}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}video{width:100%;border-radius:8px;background:#000;aspect-ratio:9/16}
.sc{background:var(--g);color:#fff;border-radius:20px;padding:1px 9px;font-weight:700;font-size:12px}.m{color:var(--m);font-size:13px}.err{color:#e0584a}
</style></head><body><main><h1>VIBRA <b>CLIP ENGINE</b></h1>
<div class=box><div id=drop>Glisse ta vidéo ici ou clique pour la choisir</div><input type=file id=f accept="video/*" hidden>
<div class=row><div><label>Clips</label><input id=n type=number value=6></div><div><label>Min (s)</label><input id=mn type=number value=20></div>
<div><label>Max (s)</label><input id=mx type=number value=60></div><div><label>Langue</label><select id=lg><option>English</option><option>French</option></select></div>
<div><label>Plateforme</label><select id=pf><option>TikTok / Reels / Shorts</option><option>LinkedIn</option><option>Facebook</option></select></div></div>
<div class=row><button id=go>Générer mes clips</button></div><div id=s class=m></div><div class=bar><i id=p></i></div></div><div id=o></div></main><script>
const $=i=>document.getElementById(i);let file;$("drop").onclick=()=>$("f").click();
$("f").onchange=e=>pick(e.target.files[0]);const pick=x=>{file=x;$("drop").textContent=x?x.name+" ("+(x.size/1e6).toFixed(0)+" Mo)":"Glisse ta vidéo ici"};
$("drop").ondragover=e=>{e.preventDefault();$("drop").classList.add("on")};$("drop").ondragleave=()=>$("drop").classList.remove("on");
$("drop").ondrop=e=>{e.preventDefault();$("drop").classList.remove("on");pick(e.dataTransfer.files[0])};
$("go").onclick=()=>{if(!file){$("s").textContent="Choisis une vidéo.";return}$("go").disabled=true;$("o").innerHTML="";
const fd=new FormData();fd.append("file",file);fd.append("n",$("n").value);fd.append("mn",$("mn").value);fd.append("mx",$("mx").value);fd.append("lang",$("lg").value);fd.append("platform",$("pf").value);
const x=new XMLHttpRequest();x.open("POST","/api/jobs");x.upload.onprogress=e=>{$("s").textContent="Envoi… "+Math.round(e.loaded/e.total*100)+"%"};
x.onload=()=>{if(x.status!=200){$("s").innerHTML='<span class=err>Erreur '+x.status+'</span>';$("go").disabled=false;return}poll(JSON.parse(x.responseText).id)};x.send(fd)};
async function poll(id){const j=await(await fetch("/api/jobs/"+id)).json();$("p").style.width=j.progress+"%";$("s").textContent=j.message;
if(j.status=="running")return setTimeout(()=>poll(id),2500);$("go").disabled=false;if(j.status=="error"){$("s").innerHTML='<span class=err>'+j.message+'</span>';return}
let h='<div class=box><a class=btn href="/files/'+id+'/clips.zip">Tout télécharger (.zip)</a></div><div class=grid>';
j.clips.forEach(c=>{h+='<div class=box><video controls preload=metadata src="/files/'+id+'/'+c.file+'"></video><div style="display:flex;justify-content:space-between;margin-top:6px"><b>'+esc(c.title)+'</b><span class=sc>'+c.score+'</span></div><div class=m>'+c.duration+' s · '+esc(c.why||"")+'</div><p class=m>'+esc(c.caption||"")+'<br>'+(c.hashtags||[]).map(t=>"#"+t.replace(/^#/,"")).join(" ")+'</p><a class=btn href="/files/'+id+'/'+c.file+'">MP4</a></div>'});
$("o").innerHTML=h+"</div>"}
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
</script></body></html>"""

@app.get("/", response_class=HTMLResponse, dependencies=[Depends(auth)])
def home(): return UI
