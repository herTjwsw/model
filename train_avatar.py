"""Train a dedicated small DCGAN from the six user-provided character screenshots.

This is a tiny-data personalization experiment, not a high-fidelity identity model.
"""
import argparse, random, time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps, ImageEnhance
from model import Generator, Discriminator, LATENT
import numpy as np

ROOT=Path(__file__).resolve().parent

class AvatarSet(Dataset):
    def __init__(self): self.files=sorted((ROOT/'data'/'avatar_source').glob('*.png'))
    def __len__(self): return max(len(self.files), 1)*4
    def __getitem__(self, idx):
        p=self.files[idx % len(self.files)]
        im=Image.open(p).convert('RGB')
        # Crop square around the avatar; minor random reframing and light jitter help with six examples.
        w,h=im.size; side=min(w,h); cx=w*0.5; cy=h*0.48
        scale=random.uniform(0.84,1.0); side*=scale
        left=max(0,min(w-side,cx-side/2+random.uniform(-0.035,0.035)*w))
        top=max(0,min(h-side,cy-side/2+random.uniform(-0.035,0.035)*h))
        im=im.crop((int(left),int(top),int(left+side),int(top+side))).resize((64,64),Image.Resampling.LANCZOS)
        if random.random()<.5: im=ImageOps.mirror(im)
        if random.random()<.75:
            im=ImageEnhance.Brightness(im).enhance(random.uniform(.88,1.12))
            im=ImageEnhance.Color(im).enhance(random.uniform(.88,1.12))
        x=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()).permute(2,0,1).div_(127.5).sub_(1)
        return x

def save_grid(x,path):
    x=x.detach().cpu().clamp(-1,1).add(1).mul(127.5).byte().permute(0,2,3,1).numpy()
    im=Image.new('RGB',(4*64,4*64),(20,22,30))
    for i,a in enumerate(x[:16]): im.paste(Image.fromarray(a),((i%4)*64,(i//4)*64))
    im.save(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=600); ap.add_argument('--batch-size',type=int,default=8); ap.add_argument('--seed',type=int,default=52); args=ap.parse_args()
    if not list((ROOT/'data'/'avatar_source').glob('*.png')): raise SystemExit('No avatar source PNGs found.')
    random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(2)
    ds=AvatarSet(); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,drop_last=True)
    g=Generator().cpu(); d=Discriminator().cpu()
    og=torch.optim.Adam(g.parameters(),lr=.0002,betas=(.5,.999)); od=torch.optim.Adam(d.parameters(),lr=.0002,betas=(.5,.999)); bce=nn.BCEWithLogitsLoss()
    fixed=torch.randn(16,LATENT,1,1); outdir=ROOT/'checkpoints'; outdir.mkdir(exist_ok=True); samples=ROOT/'samples'; samples.mkdir(exist_ok=True)
    tic=time.time()
    for ep in range(1,args.epochs+1):
        g.train(); d.train(); sums=[0.,0.]
        for real in loader:
            bs=len(real); fake=g(torch.randn(bs,LATENT,1,1)).detach()
            od.zero_grad(set_to_none=True)
            ld=bce(d(real),torch.full((bs,),.9))+bce(d(fake),torch.zeros(bs)); ld.backward(); od.step()
            og.zero_grad(set_to_none=True)
            lg=bce(d(g(torch.randn(bs,LATENT,1,1))),torch.ones(bs)); lg.backward(); og.step()
            sums[0]+=ld.item(); sums[1]+=lg.item()
        if ep==1 or ep%50==0 or ep==args.epochs:
            g.eval()
            with torch.inference_mode(): preview=g(fixed)
            save_grid(preview,samples/f'avatar_epoch_{ep:04d}.jpg')
            torch.save({'epoch':ep,'generator':g.state_dict(),'discriminator':d.state_dict(),'latent':LATENT,
                        'dataset':'six user-provided Roblox character screenshots','loss_g':sums[1]/len(loader),'loss_d':sums[0]/len(loader)},outdir/'avatar_latest.pt')
            print(f'avatar epoch {ep}/{args.epochs} | D {sums[0]/len(loader):.3f} | G {sums[1]/len(loader):.3f} | {time.time()-tic:.0f}s',flush=True)
    print('Saved checkpoints/avatar_latest.pt')
if __name__=='__main__': main()
