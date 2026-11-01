"""Local gallery and seed/interpolation sampler for the trained DCGAN."""
import json, random, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
import torch
from PIL import Image
from model import Generator, LATENT

ROOT = Path(__file__).resolve().parent
PORT = 8000

torch.set_num_threads(2)
DEVICE = torch.device('cpu')
GEN = Generator().to(DEVICE).eval()
CKPT = ROOT / 'checkpoints' / 'latest.pt'
if CKPT.exists():
    state = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    GEN.load_state_dict(state['generator'])
    EPOCH = state.get('epoch', '?')
else:
    EPOCH = 0
AVATAR_GEN = None
AVATAR_EPOCH = 0
AVATAR_CKPT = ROOT / 'checkpoints' / 'avatar_latest.pt'
if AVATAR_CKPT.exists():
    avatar_state = torch.load(AVATAR_CKPT, map_location=DEVICE, weights_only=False)
    AVATAR_GEN = Generator().to(DEVICE).eval()
    AVATAR_GEN.load_state_dict(avatar_state['generator'])
    AVATAR_EPOCH = avatar_state.get('epoch', 0)


def synth(seed_a, seed_b=None, t=0.0, mode='theory'):
    a = torch.Generator().manual_seed(int(seed_a) % (2**31-1))
    za = torch.randn((1, LATENT, 1, 1), generator=a)
    if seed_b is not None:
        b = torch.Generator().manual_seed(int(seed_b) % (2**31-1))
        zb = torch.randn((1, LATENT, 1, 1), generator=b)
        # Spherical interpolation is smoother than blending raw noise vectors.
        alpha = max(0.0, min(1.0, float(t)))
        dot = torch.nn.functional.cosine_similarity(za.flatten(1), zb.flatten(1)).clamp(-0.999, 0.999)
        omega = torch.acos(dot).view(1,1,1,1)
        den = torch.sin(omega).clamp_min(1e-6)
        z = torch.sin((1-alpha)*omega)/den*za + torch.sin(alpha*omega)/den*zb
    else:
        z = za
    active_gen = AVATAR_GEN if mode == 'avatar' and AVATAR_GEN is not None else GEN
    with torch.inference_mode():
        x = active_gen(z.to(DEVICE))[0].cpu().clamp(-1,1).add(1).mul(127.5).byte().permute(1,2,0).numpy()
    im = Image.fromarray(x, 'RGB')
    out = BytesIO(); im.save(out, format='PNG')
    return out.getvalue()


def page():
    return r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Theory Thumbnail Lab</title>
<style>
:root{--bg:#090b13;--panel:#111522;--muted:#9aa3b9;--line:#272c3c;--pink:#ff557d;--violet:#9175ff;--mint:#62e5c0}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 80% -10%,#312253 0,transparent 38%),var(--bg);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;color:#f4f4fa}.wrap{max-width:1180px;margin:auto;padding:30px 24px 70px}.top{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:40px}.brand{display:flex;gap:12px;align-items:center;font-weight:800;letter-spacing:-.03em}.logo{width:40px;height:40px;border-radius:14px;background:linear-gradient(135deg,var(--pink),var(--violet));display:grid;place-items:center;font-size:21px;box-shadow:0 8px 28px #9f4d9d55}.pill{border:1px solid #34394b;border-radius:999px;padding:8px 12px;color:#bdc3d4;font-size:12px;background:#111522}.hero{display:grid;grid-template-columns:1.02fr .98fr;gap:34px;align-items:center}.eyebrow{color:#bd9dff;text-transform:uppercase;letter-spacing:.18em;font-weight:800;font-size:11px}.copy h1{font-size:clamp(38px,6vw,68px);line-height:.98;letter-spacing:-.065em;margin:15px 0 18px}.copy h1 span{background:linear-gradient(90deg,#ff6688,#a78aff);background-clip:text;color:transparent}.copy p{color:#aab1c2;line-height:1.7;max-width:540px;font-size:15px}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.stat{padding:11px 14px;background:#111522;border:1px solid var(--line);border-radius:14px;font-size:12px;color:#aab1c2}.stat strong{color:#f7f4ff;font-size:17px;display:block;margin-bottom:2px}.card{background:linear-gradient(160deg,#171a29,#10131e);border:1px solid #2c3041;border-radius:23px;padding:18px;box-shadow:0 20px 80px #0006}.cardhead{display:flex;justify-content:space-between;align-items:center;margin:0 0 14px}.cardhead h2{font-size:15px;margin:0}.live{font-size:11px;color:var(--mint);display:flex;align-items:center;gap:7px}.dot{width:7px;height:7px;border-radius:50%;background:var(--mint);box-shadow:0 0 12px var(--mint)}.output{aspect-ratio:1;display:grid;place-items:center;background:#080a10;border:1px solid #292d3c;border-radius:16px;overflow:hidden;position:relative}.output img{width:100%;height:100%;object-fit:cover;image-rendering:auto}.overlay{position:absolute;bottom:10px;left:10px;padding:6px 9px;background:#090b13c9;border:1px solid #ffffff22;border-radius:8px;font-size:10px;color:#c4c9d7}.controls{display:grid;grid-template-columns:1fr auto;gap:10px;margin-top:14px}.rangebox{background:#0b0e17;border:1px solid var(--line);border-radius:12px;padding:9px 12px}.rangebox label{display:flex;justify-content:space-between;color:#9da6b9;font-size:10px;margin-bottom:4px}input[type=range]{accent-color:#aa82ff;width:100%;margin:4px 0}button{border:0;border-radius:12px;padding:0 17px;color:white;font-weight:750;background:linear-gradient(100deg,#ff557d,#8a69ff);cursor:pointer;box-shadow:0 7px 22px #9a4eac44}button:hover{filter:brightness(1.12);transform:translateY(-1px)}.seedline{font-size:11px;color:#777f94;margin-top:10px}.section{margin-top:52px}.sectionhead{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:16px}.section h2{font-size:24px;letter-spacing:-.04em;margin:5px 0}.sectionhead p,.note{color:var(--muted);font-size:12px;margin:0;line-height:1.6}.gallery{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}.tile{aspect-ratio:16/9;background:#171a25;border-radius:12px;overflow:hidden;border:1px solid #292d3b}.tile img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .25s}.tile:hover img{transform:scale(1.06)}.notice{border:1px solid #453e56;background:#171326;border-radius:14px;padding:14px 16px;color:#b7afc8;font-size:12px;line-height:1.6;margin-top:20px}.notice b{color:#eee1ff}.footer{margin-top:44px;border-top:1px solid #242838;padding-top:17px;display:flex;justify-content:space-between;gap:15px;color:#747d91;font-size:11px}.loading{opacity:.55}@media(max-width:850px){.hero{grid-template-columns:1fr}.gallery{grid-template-columns:repeat(3,1fr)}.wrap{padding:22px 16px 50px}.top{margin-bottom:28px}.output{max-height:520px}.sectionhead{align-items:start;flex-direction:column}}@media(max-width:470px){.gallery{grid-template-columns:repeat(2,1fr)}.controls{grid-template-columns:1fr}.controls button{padding:13px}}
</style></head><body><main class="wrap"><header class="top"><div class="brand"><div class="logo">✦</div><span>THEORY THUMBNAIL LAB</span></div><div class="pill">TWO MODELS · EXPERIMENTAL</div></header>
<section class="hero"><div class="copy"><div class="eyebrow">A tiny generative experiment</div><div style="display:flex;gap:9px;flex-wrap:wrap;margin:15px 0"><button id="theoryMode" style="padding:10px 13px">Game Theory</button><button id="avatarMode" style="padding:10px 13px;background:#293143;box-shadow:none">My Character</button></div><h1 id="headline">What if the<br>theory was <span>generated?</span></h1><p id="intro">A DCGAN trained on public thumbnails from MatPat’s <b>The Game Theorists</b> channel. Generate a fresh sample or interpolate between latent seeds.</p><div class="stats"><div class="stat"><strong id="count">272</strong><span id="sourceLabel">source thumbnails</span></div><div class="stat"><strong id="epoch">500</strong>training epochs</div><div class="stat"><strong>64 × 64</strong>image resolution</div></div><div class="notice" id="notice"><b>Research/demo note:</b> this experimental, low-resolution CPU-trained GAN can produce abstract or artifacted images.</div></div>
<div class="card"><div class="cardhead"><h2>Latent seed explorer</h2><span class="live"><i class="dot"></i> MODEL LOADED</span></div><div class="output"><img id="generated" alt="GAN generated sample"><div class="overlay" id="overlay">SYNTHETIC · NOT AN ORIGINAL THUMBNAIL</div></div><div class="controls"><div class="rangebox"><label><span>INTERPOLATION · SEED A → B</span><span id="tval">50%</span></label><input id="mix" type="range" min="0" max="100" value="50"></div><button id="generate">Generate ↗</button></div><div class="seedline" id="seeds">Loading model sample…</div></div></section>
<section class="section"><div class="sectionhead"><div><div class="eyebrow">Training material</div><h2 id="galleryTitle">From the real channel</h2></div><p id="galleryInfo">Public thumbnail previews · 272 videos<br>Source: The Game Theorists</p></div><div class="gallery" id="gallery"></div></section>
<section class="section"><div class="sectionhead"><div><div class="eyebrow">Training snapshot</div><h2>Generated batch</h2></div><p>16 outputs from one saved model checkpoint</p></div><div class="tile" style="aspect-ratio:1;max-width:520px"><img id="batchImage" src="/samples/epoch_0500.jpg" alt="16 generated samples from training checkpoint" style="width:100%;height:100%;object-fit:cover"></div></section>
<footer class="footer"><span id="footerModel">Thumbnail DCGAN · 272 images · 500 epochs</span><span>MatPat / The Game Theorists is not affiliated with this demo.</span></footer></main>
<script>
let mode='theory',seedA=Math.floor(Math.random()*10000000),seedB=Math.floor(Math.random()*10000000);const img=document.getElementById('generated'),slider=document.getElementById('mix'),seeds=document.getElementById('seeds');
function gallery(){fetch('/api/gallery?mode='+mode).then(r=>r.json()).then(items=>{document.getElementById('gallery').innerHTML=items.map(x=>{let url=mode==='avatar'?'/avatar/'+x.id+'.png':'/thumbs/'+x.id+'.jpg';let title=(x.title||'Character reference').replaceAll('&','&amp;').replaceAll('"','&quot;');return '<a class="tile" href="'+url+'" target="_blank" title="'+title+'"><img loading="lazy" src="'+url+'" alt="'+title+'"></a>'}).join('')})}
function update(){let t=Number(slider.value)/100;document.getElementById('tval').textContent=slider.value+'%';img.classList.add('loading');img.onload=()=>img.classList.remove('loading');img.src='/api/sample?mode='+mode+'&seedA='+seedA+'&seedB='+seedB+'&t='+t+'&_='+Date.now();seeds.textContent='Seed A '+seedA+'   ·   Seed B '+seedB+'   ·   t = '+t.toFixed(2)}
function setMode(next){mode=next;const avatar=mode==='avatar';document.getElementById('theoryMode').style.background=avatar?'#293143':'linear-gradient(100deg,#ff557d,#8a69ff)';document.getElementById('theoryMode').style.boxShadow=avatar?'none':'0 7px 22px #9a4eac44';document.getElementById('avatarMode').style.background=avatar?'linear-gradient(100deg,#ff557d,#8a69ff)':'#293143';document.getElementById('avatarMode').style.boxShadow=avatar?'0 7px 22px #9a4eac44':'none';document.getElementById('headline').innerHTML=avatar?'Your character,<br><span>remixed.</span>':'What if the<br>theory was <span>generated?</span>';document.getElementById('intro').innerHTML=avatar?'A second DCGAN trained specifically on your six Roblox character screenshots. Generate a low-resolution sample or interpolate between latent seeds.':'A DCGAN trained on public thumbnails from MatPat’s <b>The Game Theorists</b> channel. Explore the learned thumbnail colors and layouts.';document.getElementById('notice').innerHTML=avatar?'<b>Small-data warning:</b> this character model was trained from only six screenshots. It is a tiny 64×64 experiment and may produce abstract, artifacted results; it cannot reliably reproduce the avatar yet.':'<b>Research/demo note:</b> this low-resolution CPU-trained GAN learns broad color and layout cues, not polished thumbnails. Source images remain credited to their creators.';document.getElementById('count').textContent=avatar?'6':'272';document.getElementById('sourceLabel').textContent=avatar?'character screenshots':'source thumbnails';document.getElementById('epoch').textContent=avatar?'400':'500';document.getElementById('galleryTitle').textContent=avatar?'Your character references':'From the real channel';document.getElementById('galleryInfo').innerHTML=avatar?'Six screenshots you provided · used to train the character model':'Public thumbnail previews · 272 unique videos<br>Source: The Game Theorists, MatPat’s gaming theory channel';document.getElementById('batchImage').src=avatar?'/samples/avatar_epoch_0400.jpg':'/samples/epoch_0500.jpg';document.getElementById('overlay').textContent=avatar?'SYNTHETIC · CHARACTER GAN':'SYNTHETIC · NOT AN ORIGINAL THUMBNAIL';document.getElementById('footerModel').textContent=avatar?'Character DCGAN · six screenshots · 400 epochs':'Thumbnail DCGAN · 272 images · 500 epochs';gallery();update()}
document.getElementById('theoryMode').onclick=()=>setMode('theory');document.getElementById('avatarMode').onclick=()=>setMode('avatar');document.getElementById('generate').onclick=()=>{seedA=Math.floor(Math.random()*10000000);seedB=Math.floor(Math.random()*10000000);slider.value=50;update()};slider.oninput=update;gallery();update();
</script></body></html>'''

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if self.path.startswith('/api/'): return
        super().log_message(fmt, *args)
    def send(self, data, typ, status=200):
        self.send_response(status); self.send_header('Content-Type',typ); self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        u=urllib.parse.urlparse(self.path); q=urllib.parse.parse_qs(u.query)
        if u.path=='/':
            html = (ROOT/'index.html').read_bytes() if (ROOT/'index.html').exists() else page().encode()
            return self.send(html, 'text/html; charset=utf-8')
        if u.path=='/api/status':
            m=json.loads((ROOT/'data'/'manifest.json').read_text()) if (ROOT/'data'/'manifest.json').exists() else []
            av=json.loads((ROOT/'data'/'avatar_manifest.json').read_text()) if (ROOT/'data'/'avatar_manifest.json').exists() else []
            return self.send(json.dumps({'epoch':EPOCH,'images':len(m),'avatar_epoch':AVATAR_EPOCH,'avatar_images':len(av)}).encode(),'application/json')
        if u.path=='/api/gallery':
            if q.get('mode',['theory'])[0]=='avatar':
                try: m=json.loads((ROOT/'data'/'avatar_manifest.json').read_text())
                except: m=[]
                return self.send(json.dumps([{'id':Path(x['file']).stem,'title':'Your character screenshot'} for x in m]).encode(),'application/json')
            try: m=json.loads((ROOT/'data'/'manifest.json').read_text())
            except: m=[]
            random.Random(17).shuffle(m)
            return self.send(json.dumps([{'id':x['id'],'title':x.get('title','')} for x in m[:24]]).encode(),'application/json')
        if u.path=='/api/sample':
            try:
                a=int(q.get('seedA',[0])[0]); b=int(q.get('seedB',[a])[0]); t=float(q.get('t',[0.5])[0]); mode=q.get('mode',['theory'])[0]
                return self.send(synth(a,b,t,mode=mode),'image/png')
            except Exception as e: return self.send(json.dumps({'error':str(e)}).encode(),'application/json',400)
        if u.path.startswith('/avatar/') and u.path.endswith('.png'):
            name=Path(u.path).name
            if not name.startswith('avatar_'): return self.send(b'Not found','text/plain',404)
            path=ROOT/'data'/'avatar_source'/name
            if path.exists(): return self.send(path.read_bytes(),'image/png')
        if u.path.startswith('/thumbs/') and u.path.endswith('.jpg'):
            name=Path(u.path).name; path=ROOT/'data'/'raw'/name
            if path.exists(): return self.send(path.read_bytes(),'image/jpeg')
        if u.path.startswith('/samples/'):
            name=Path(u.path).name; path=ROOT/'samples'/name
            if path.exists(): return self.send(path.read_bytes(),'image/jpeg')
        return self.send(b'Not found','text/plain',404)

if __name__=='__main__':
    print(f'Theory Thumbnail Lab running on 0.0.0.0:{PORT}; checkpoint epoch {EPOCH}',flush=True)
    ThreadingHTTPServer(('0.0.0.0',PORT),Handler).serve_forever()
