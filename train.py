"""Train a compact DCGAN on the prepared MatPat/Game Theorists thumbnails.

Example: python3 train.py --epochs 40 --batch-size 16
"""
import argparse, copy, random, time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
from model import Generator, Discriminator, LATENT

ROOT = Path(__file__).resolve().parent

class ThumbDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(Path(folder).glob('*.jpg'))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        im = Image.open(self.files[i]).convert('RGB')
        if random.random() < 0.5: im = ImageOps.mirror(im)
        import numpy as np
        x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2,0,1).div_(127.5).sub_(1)
        return x

def save_grid(images, path, nrow=4):
    images = images.detach().cpu().clamp(-1, 1).add(1).mul(127.5).byte().permute(0,2,3,1).numpy()
    w, h = images.shape[2], images.shape[1]
    ncol = min(nrow, len(images)); nrows = (len(images)+ncol-1)//ncol
    canvas = Image.new("RGB", (ncol*w, nrows*h), (24,24,30))
    for i, arr in enumerate(images):
        canvas.paste(Image.fromarray(arr), ((i%ncol)*w, (i//ncol)*h))
    canvas.save(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.0002)
    ap.add_argument('--seed', type=int, default=41)
    ap.add_argument('--resume', action='store_true', help='continue from checkpoints/latest.pt')
    args = ap.parse_args()
    torch.set_num_threads(2); torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device('cpu')
    data = ThumbDataset(ROOT / 'data' / 'processed')
    if len(data) == 0:
        raise SystemExit('No prepared images found; run python3 prepare.py first.')
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=len(data)>args.batch_size)
    g, d = Generator().to(device), Discriminator().to(device)
    start_epoch = 0
    if args.resume and (ROOT/'checkpoints'/'latest.pt').exists():
        state = torch.load(ROOT/'checkpoints'/'latest.pt', map_location=device, weights_only=False)
        g.load_state_dict(state['generator']); d.load_state_dict(state['discriminator'])
        start_epoch = int(state.get('epoch', 0))
        print(f"Resuming from epoch {start_epoch}", flush=True)
    opt_g = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(0.5,0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=args.lr, betas=(0.5,0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed_z = torch.randn(16, LATENT, 1, 1, device=device)
    (ROOT/'checkpoints').mkdir(exist_ok=True); (ROOT/'samples').mkdir(exist_ok=True)
    start=time.time()
    for epoch in range(start_epoch+1,start_epoch+args.epochs+1):
        g.train(); d.train(); sums=[0.0,0.0]
        for real in loader:
            real=real.to(device); bs=real.shape[0]
            # D update, with a small amount of label smoothing.
            z=torch.randn(bs,LATENT,1,1,device=device)
            fake=g(z).detach()
            opt_d.zero_grad(set_to_none=True)
            loss_d=bce(d(real),torch.full((bs,),0.9,device=device))+bce(d(fake),torch.zeros(bs,device=device))
            loss_d.backward(); opt_d.step()
            # G update
            z=torch.randn(bs,LATENT,1,1,device=device)
            opt_g.zero_grad(set_to_none=True)
            loss_g=bce(d(g(z)),torch.ones(bs,device=device))
            loss_g.backward(); opt_g.step()
            sums[0]+=loss_d.item(); sums[1]+=loss_g.item()
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            g.eval()
            with torch.no_grad(): sample=g(fixed_z)
            save_grid(sample,ROOT/'samples'/f'epoch_{epoch:04d}.jpg')
            ckpt={'epoch':epoch,'generator':g.state_dict(),'discriminator':d.state_dict(),'latent':LATENT,
                  'dataset':'The Game Theorists / MatPat public video thumbnails','loss_g':sums[1]/len(loader),'loss_d':sums[0]/len(loader)}
            torch.save(ckpt,ROOT/'checkpoints'/'latest.pt')
            torch.save(ckpt,ROOT/'checkpoints'/f'epoch_{epoch:04d}.pt')
            print(f"epoch {epoch:4d}/{args.epochs} | D {sums[0]/len(loader):.3f} | G {sums[1]/len(loader):.3f} | {time.time()-start:.0f}s",flush=True)
    print('Training complete. Latest checkpoint: checkpoints/latest.pt')
if __name__=='__main__': main()
