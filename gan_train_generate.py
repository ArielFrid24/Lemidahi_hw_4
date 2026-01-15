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


# ============================================================
# Config
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

    # DCGAN defaults (often ok for 64x64)
    lrG: float = 2e-4
    lrD: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    epochs: int = 40

    n_sample_grid: int = 20
    grid_cols: int = 5

    # Stabilizers
    real_label: float = 0.9  # one-sided label smoothing
    use_instance_noise: bool = True
    noise_std_start: float = 0.08
    noise_std_end: float = 0.0

    # Logging
    tqdm_update_every: int = 20


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = torch.from_numpy(np.array(img)).float() / 255.0
    x = x.permute(2, 0, 1).contiguous()
    return x


def to_gan_range(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0


def from_gan_range(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0


def save_image_grid(imgs: torch.Tensor, out_path: str, ncol: int = 5, show: bool = False, title: str = ""):
    imgs = imgs.detach().cpu().clamp(0, 1)
    N, C, H, W = imgs.shape
    nrow = int(np.ceil(N / ncol))

    grid = torch.ones((C, nrow * H, ncol * W))
    for i in range(N):
        r = i // ncol
        c = i % ncol
        grid[:, r*H:(r+1)*H, c*W:(c+1)*W] = imgs[i]

    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(ncol * 2.4, nrow * 2.4))
    plt.imshow(grid_np)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def linear_anneal(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    t = epoch / (epochs - 1)
    return float(start * (1.0 - t) + end * t)


# ============================================================
# Dataset
# ============================================================

class Flowers102FromJSON(Dataset):
    def __init__(self, json_path: str, image_root: str, img_size: int):
        self.image_root = image_root
        self.img_size = img_size

        with open(json_path, "r", encoding="utf-8") as f:
            mapping: Dict[str, List[str]] = json.load(f)

        items: List[Tuple[str, int]] = []
        for k, files in mapping.items():
            cls = int(k) - 1
            for fn in files:
                items.append((fn, cls))

        if not items:
            raise ValueError("No images found in JSON mapping.")
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn, cls = self.items[idx]
        img = Image.open(os.path.join(self.image_root, fn))
        img = resize_center_crop(img, self.img_size)
        x = pil_to_tensor(img)          # [0,1]
        x = to_gan_range(x)             # [-1,1]
        return x, cls


# ============================================================
# Models (DCGAN)
# ============================================================

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.size(0), self.z_dim, 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


# ============================================================
# Train + Generate + Plots
# ============================================================

@torch.no_grad()
def generate_samples(G: nn.Module, n: int, z_dim: int, device: str) -> torch.Tensor:
    G.eval()
    z = torch.randn(n, z_dim, device=device)
    fake = G(z)
    return from_gan_range(fake)


def train_and_generate(cfg: Config):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    ds = Flowers102FromJSON(cfg.json_path, cfg.image_root, cfg.img_size)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=(cfg.device == "cuda"),
    )

    G = Generator(cfg.z_dim).to(cfg.device)
    D = Discriminator().to(cfg.device)
    G.apply(weights_init)
    D.apply(weights_init)

    optG = torch.optim.Adam(G.parameters(), lr=cfg.lrG, betas=cfg.betas)
    optD = torch.optim.Adam(D.parameters(), lr=cfg.lrD, betas=cfg.betas)

    bce = nn.BCELoss()

    fixed_z = torch.randn(cfg.n_sample_grid, cfg.z_dim, device=cfg.device)

    D_losses: List[float] = []
    G_losses: List[float] = []

    step = 0
    for epoch in range(cfg.epochs):
        G.train()
        D.train()

        noise_std = 0.0
        if cfg.use_instance_noise:
            noise_std = linear_anneal(epoch, cfg.epochs, cfg.noise_std_start, cfg.noise_std_end)

        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=True)

        lossD_running = 0.0
        lossG_running = 0.0

        for i, (real, _y) in enumerate(pbar):
            real = real.to(cfg.device, non_blocking=True)
            B = real.size(0)

            # Optional instance noise
            if noise_std > 0:
                real_in = real + noise_std * torch.randn_like(real)
            else:
                real_in = real

            # --------------------
            # Train Discriminator
            # --------------------
            D.zero_grad(set_to_none=True)

            z = torch.randn(B, cfg.z_dim, device=cfg.device)
            fake = G(z).detach()
            if noise_std > 0:
                fake_in = fake + noise_std * torch.randn_like(fake)
            else:
                fake_in = fake

            out_real = D(real_in)
            out_fake = D(fake_in)

            lossD_real = bce(out_real, torch.full((B,), cfg.real_label, device=cfg.device))
            lossD_fake = bce(out_fake, torch.zeros(B, device=cfg.device))
            lossD = lossD_real + lossD_fake

            lossD.backward()
            optD.step()

            # --------------------
            # Train Generator
            # --------------------
            G.zero_grad(set_to_none=True)

            z = torch.randn(B, cfg.z_dim, device=cfg.device)
            gen = G(z)
            out_gen = D(gen)

            lossG = bce(out_gen, torch.ones(B, device=cfg.device))
            lossG.backward()
            optG.step()

            lossD_running += float(lossD.item())
            lossG_running += float(lossG.item())

            if (i % cfg.tqdm_update_every) == 0:
                pbar.set_postfix({
                    "lossD": f"{lossD_running/(i+1):.3f}",
                    "lossG": f"{lossG_running/(i+1):.3f}",
                    "D(real)": f"{out_real.mean().item():.2f}",
                    "D(fake)": f"{out_fake.mean().item():.2f}",
                    "noise": f"{noise_std:.3f}",
                    "step": step
                })

            step += 1

        # epoch-average loss tracking
        D_losses.append(lossD_running / max(1, len(dl)))
        G_losses.append(lossG_running / max(1, len(dl)))

        # Save per-epoch sample grid (also useful to pick the best epoch later)
        G.eval()
        with torch.no_grad():
            imgs = from_gan_range(G(fixed_z))
        save_image_grid(
            imgs,
            os.path.join(cfg.out_dir, f"samples_epoch_{epoch+1:03d}.png"),
            ncol=cfg.grid_cols,
            show=False,
            title=f"GAN samples epoch {epoch+1}"
        )

    # Save weights
    torch.save(G.state_dict(), os.path.join(cfg.out_dir, "gan_G.pkl"))
    torch.save(D.state_dict(), os.path.join(cfg.out_dir, "gan_D.pkl"))

    # Plot loss curves (epochs)
    plt.figure()
    plt.plot(D_losses, label="D loss")
    plt.plot(G_losses, label="G loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "gan_loss_curves.png"), dpi=150)
    plt.close()

    # Final generated grid (SHOW + SAVE)
    final = generate_samples(G, cfg.n_sample_grid, cfg.z_dim, cfg.device)
    save_image_grid(
        final,
        os.path.join(cfg.out_dir, "gan_generated_samples.png"),
        ncol=cfg.grid_cols,
        show=True,
        title="GAN generated samples (final)"
    )

    print(f"Saved: {os.path.join(cfg.out_dir, 'gan_G.pkl')}")
    print(f"Saved: {os.path.join(cfg.out_dir, 'gan_D.pkl')}")
    print(f"Saved: {os.path.join(cfg.out_dir, 'gan_loss_curves.png')}")
    print(f"Saved: {os.path.join(cfg.out_dir, 'gan_generated_samples.png')}")


if __name__ == "__main__":
    cfg = Config()
    print("Config:", cfg)
    print("Device:", cfg.device)
    train_and_generate(cfg)
