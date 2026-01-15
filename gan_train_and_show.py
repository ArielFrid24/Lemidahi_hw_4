import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms, utils


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    json_path: str = "category_to_images.json"
    image_root: str = "./jpg"
    out_dir: str = "./gan_out"

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    img_size: int = 64
    batch_size: int = 128
    num_workers: int = 2

    z_dim: int = 128
    lrG: float = 2e-4
    lrD: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    epochs: int = 40

    n_sample_grid: int = 20
    grid_cols: int = 5
    real_label: float = 0.9

    show_every_epochs: int = 5   # <<< SHOW IMAGES INLINE EVERY N EPOCHS


# ============================================================
# UTILS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def show_images(images, title):
    grid = utils.make_grid(images, nrow=5, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.title(title)
    plt.axis("off")
    plt.show()


# ============================================================
# DATASET
# ============================================================

class Flowers102FromJSON(Dataset):
    def __init__(self, json_path, image_root, transform):
        with open(json_path) as f:
            mapping = json.load(f)

        self.items = []
        for k, files in mapping.items():
            for fn in files:
                self.items.append(fn)

        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_root, self.items[idx])).convert("RGB")
        return self.transform(img)


# ============================================================
# MODELS
# ============================================================

class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.net(x).view(-1)


# ============================================================
# TRAIN
# ============================================================

def train(cfg):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])

    ds = Flowers102FromJSON(cfg.json_path, cfg.image_root, transform)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    G = Generator(cfg.z_dim).to(cfg.device)
    D = Discriminator().to(cfg.device)

    optG = torch.optim.Adam(G.parameters(), lr=cfg.lrG, betas=cfg.betas)
    optD = torch.optim.Adam(D.parameters(), lr=cfg.lrD, betas=cfg.betas)

    loss_fn = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(cfg.n_sample_grid, cfg.z_dim, device=cfg.device)

    G_losses, D_losses = [], []

    for epoch in range(cfg.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for real in pbar:
            real = real.to(cfg.device)
            B = real.size(0)

            # ---- D ----
            D.zero_grad()
            z = torch.randn(B, cfg.z_dim, device=cfg.device)
            fake = G(z).detach()

            lossD = (
                loss_fn(D(real), torch.full((B,), cfg.real_label, device=cfg.device)) +
                loss_fn(D(fake), torch.zeros(B, device=cfg.device))
            )
            lossD.backward()
            optD.step()

            # ---- G ----
            G.zero_grad()
            z = torch.randn(B, cfg.z_dim, device=cfg.device)
            gen = G(z)
            lossG = loss_fn(D(gen), torch.ones(B, device=cfg.device))
            lossG.backward()
            optG.step()

            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

        # ---- SHOW IMAGES INLINE ----
        if (epoch + 1) % cfg.show_every_epochs == 0:
            with torch.no_grad():
                imgs = (G(fixed_z) + 1) / 2
            show_images(imgs, f"Generated images at epoch {epoch+1}")

    # ---- FINAL RESULTS ----
    torch.save(G.state_dict(), f"{cfg.out_dir}/gan_G.pkl")
    torch.save(D.state_dict(), f"{cfg.out_dir}/gan_D.pkl")

    with torch.no_grad():
        final_imgs = (G(fixed_z) + 1) / 2
    show_images(final_imgs, "FINAL GAN SAMPLES")

    # ---- LOSS CURVES ----
    plt.figure(figsize=(6,4))
    plt.plot(D_losses, label="D loss")
    plt.plot(G_losses, label="G loss")
    plt.legend()
    plt.title("GAN Loss Curves")
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    cfg = Config()
    print("Config:", cfg)
    train(cfg)
