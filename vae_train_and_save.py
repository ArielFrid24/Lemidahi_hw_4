import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from torchvision import transforms, utils


@dataclass
class Config:
    json_path: str = "category_to_images.json"
    image_root: str = "./jpg"
    out_dir: str = "./vae_out"

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    img_size: int = 64
    batch_size: int = 64
    num_workers: int = 2

    z_dim: int = 128
    lr: float = 2e-4
    weight_decay: float = 0.0
    epochs: int = 20

    # If you want sharper: try beta_end=0.2
    beta_start: float = 0.0
    beta_end: float = 1.0
    beta_warmup_epochs: int = 5

    # Outputs
    n_sample_grid: int = 20
    grid_cols: int = 5
    save_every_epochs: int = 5

    # Latent vis requirement
    per_class_samples_for_latent: int = 20


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_grid_png(images01: torch.Tensor, path: str, nrow: int):
    # images01: (N,3,H,W) in [0,1]
    grid = utils.make_grid(images01, nrow=nrow, normalize=False)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_loss_png(recon_hist, kl_hist, total_hist, path: str):
    plt.figure(figsize=(8, 4))
    plt.plot(recon_hist, label="recon")
    plt.plot(kl_hist, label="kl")
    plt.plot(total_hist, label="total")
    plt.legend()
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def pca_project(X: np.ndarray, k: int):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T


class Flowers102FromJSON(Dataset):
    def __init__(self, json_path: str, image_root: str, transform):
        with open(json_path, "r", encoding="utf-8") as f:
            mapping: Dict[str, List[str]] = json.load(f)

        items: List[Tuple[str, int]] = []
        for k, files in mapping.items():
            cls = int(k) - 1
            for fn in files:
                items.append((fn, cls))

        if not items:
            raise ValueError("No items found in JSON mapping.")

        self.items = items
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn, cls = self.items[idx]
        img = Image.open(os.path.join(self.image_root, fn)).convert("RGB")
        x = self.transform(img)
        return x, cls


class Encoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(True),      # 64->32
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),    # 32->16
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),   # 16->8
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(True),   # 8->4
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, z_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.fc = nn.Linear(z_dim, 512 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(True),  # 4->8
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),  # 8->16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),   # 16->32
            nn.ConvTranspose2d(64, 3, 4, 2, 1),                    # 32->64
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 512, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.enc = Encoder(z_dim)
        self.dec = Decoder(z_dim)
        self.z_dim = z_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.z_dim, device=device)
        return self.dec(z)


def vae_loss(x, x_hat, mu, logvar, beta: float):
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar) / x.size(0)
    total = recon + beta * kl
    return total, recon, kl


def beta_for_epoch(cfg: Config, epoch0: int) -> float:
    if cfg.beta_warmup_epochs <= 0:
        return cfg.beta_end
    t = min(1.0, (epoch0 + 1) / cfg.beta_warmup_epochs)
    return cfg.beta_start + t * (cfg.beta_end - cfg.beta_start)


@torch.no_grad()
def save_latent_plots(cfg: Config, model: VAE, dataset: Flowers102FromJSON):
    model.eval()

    per_class = {c: [] for c in range(102)}
    for idx, (_fn, cls) in enumerate(dataset.items):
        if len(per_class[cls]) < cfg.per_class_samples_for_latent:
            per_class[cls].append(idx)

    selected = [(c, idxs) for c, idxs in per_class.items() if len(idxs) == cfg.per_class_samples_for_latent]
    if len(selected) == 0:
        print("No class has enough samples for latent visualization.")
        return

    X_mu = []
    Y = []
    for c, idxs in selected:
        for idx in idxs:
            x, y = dataset[idx]
            x = x.unsqueeze(0).to(cfg.device)
            mu, _ = model.enc(x)
            X_mu.append(mu.squeeze(0).cpu().numpy())
            Y.append(c)

    X_mu = np.stack(X_mu, axis=0)
    Y = np.array(Y)

    Z2 = pca_project(X_mu, k=2)
    plt.figure(figsize=(10, 7))
    plt.scatter(Z2[:, 0], Z2[:, 1], s=10, c=Y, cmap="tab20")
    plt.title("VAE latent space (mu) PCA (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "vae_latent_pca_2d.png"), dpi=150)
    plt.close()

    Z3 = pca_project(X_mu, k=3)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], s=10, c=Y, cmap="tab20")
    ax.set_title("VAE latent space (mu) PCA (3D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "vae_latent_pca_3d.png"), dpi=150)
    plt.close()


def train(cfg: Config):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    # [-1,1] because decoder uses Tanh
    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    ds = Flowers102FromJSON(cfg.json_path, cfg.image_root, transform)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=(cfg.device == "cuda"),
    )

    model = VAE(cfg.z_dim).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_hist, recon_hist, kl_hist = [], [], []
    fixed_z = torch.randn(cfg.n_sample_grid, cfg.z_dim, device=cfg.device)

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        beta = beta_for_epoch(cfg, epoch)

        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.epochs} (beta={beta:.3f})")
        for x, _y in pbar:
            x = x.to(cfg.device, non_blocking=True)

            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_hist.append(float(loss.item()))
            recon_hist.append(float(recon.item()))
            kl_hist.append(float(kl.item()))

            if step % 100 == 0:
                pbar.set_postfix({"total": f"{loss.item():.1f}", "recon": f"{recon.item():.1f}", "kl": f"{kl.item():.2f}"})
            step += 1

        # save samples every N epochs
        if (epoch + 1) % cfg.save_every_epochs == 0:
            model.eval()
            with torch.no_grad():
                samples = model.dec(fixed_z)
                samples01 = (samples + 1) / 2  # [-1,1] -> [0,1]
            save_grid_png(samples01, os.path.join(cfg.out_dir, f"vae_samples_epoch_{epoch+1:03d}.png"), nrow=cfg.grid_cols)
            print(f"Saved: {os.path.join(cfg.out_dir, f'vae_samples_epoch_{epoch+1:03d}.png')}")

    # final samples
    model.eval()
    with torch.no_grad():
        final = model.sample(cfg.n_sample_grid, cfg.device)
        final01 = (final + 1) / 2
    save_grid_png(final01, os.path.join(cfg.out_dir, "vae_samples_final.png"), nrow=cfg.grid_cols)
    print(f"Saved: {os.path.join(cfg.out_dir, 'vae_samples_final.png')}")

    # loss curves
    plot_loss_png(recon_hist, kl_hist, total_hist, os.path.join(cfg.out_dir, "vae_loss_curves.png"))
    print(f"Saved: {os.path.join(cfg.out_dir, 'vae_loss_curves.png')}")

    # latent plots
    save_latent_plots(cfg, model, ds)
    print(f"Saved: {os.path.join(cfg.out_dir, 'vae_latent_pca_2d.png')}")
    print(f"Saved: {os.path.join(cfg.out_dir, 'vae_latent_pca_3d.png')}")

    # model
    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "vae_model.pkl"))
    print(f"Saved: {os.path.join(cfg.out_dir, 'vae_model.pkl')}")


if __name__ == "__main__":
    cfg = Config()
    print("Config:", cfg)
    print("Device:", cfg.device)
    train(cfg)
