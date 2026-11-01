"""Compact 64x64 DCGAN for CPU training."""
import torch
from torch import nn

LATENT = 96
BASE = 16

class Generator(nn.Module):
    def __init__(self, latent=LATENT, base=BASE):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent, base * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base * 8), nn.ReLU(True),
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4), nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2), nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(True),
            nn.ConvTranspose2d(base, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, base=BASE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 4, 1, 0, bias=False),
        )
    def forward(self, x): return self.net(x).view(-1)
