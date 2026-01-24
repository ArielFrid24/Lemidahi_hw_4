import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

import torchvision
from torchvision import transforms
from tqdm.auto import tqdm


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    # data
    json_path: str = "category_to_images.json"
    image_root: str = "./jpg"
    img_size: int = 64

    # output
    out_dir: str = "./gan_out"
    grid_cols: int = 5
    n_sample_grid: int = 20

    # training
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 40

    # GAN
    z_dim: int = 128
    lrG: float = 2e-4
    lrD: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    real_label: float = 0.9  # one-sided label smoothing

    # stability knobs
    use_instance_noise: bool = True
    noise_std_start: float = 0.08
    noise_std_end: float = 0.0

    # logging
    tqdm_update_every: int = 20
    sample_every_epochs: int = 1
    show_every_epochs: int = 1  # set to 0 to never plt.show()


# ============================================================
# UTIL
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def linear_schedule(epoch_idx: int, total_epochs: int, start: float, end: float) -> float:
    if total_epochs <= 1:
        return end
    t = epoch_idx / (total_epochs - 1)
    return (1 - t) * start + t * end

def save_and_maybe_show_grid(images_01: torch.Tensor, out_path: str, title: str, grid_cols: int, show: bool):
    """
    images_01: (N,3,H,W) in [0,1]
    """
    images_01 = images_01.detach().cpu().clamp(0, 1)
    grid = torchvision.utils.make_grid(images_01, nrow=grid_cols, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(grid_cols * 2.2, int(np.ceil(images_01.size(0) / grid_cols)) * 2.2))
    plt.imshow(grid_np)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

def plot_loss_curves(D_losses: List[float], G_losses: List[float], out_path: str, show: bool):
    steps = list(range(len(D_losses)))
    plt.figure()
    plt.plot(steps, D_losses, label="D loss")
    plt.plot(steps, G_losses, label="G loss")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


# ============================================================
# DATASET (from JSON mapping)
# ============================================================

class Flowers102FromJSON(Dataset):
    def __init__(self, json_path: str, image_root: str, transform):
        self.image_root = image_root
        self.transform = transform

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
        path = os.path.join(self.image_root, fn)
        img = Image.open(path).convert("RGB")
        x = self.transform(img)  # normalized to [-1,1]
        return x, cls


# ============================================================
# MODEL (DCGAN-ish)
# ============================================================

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
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

            # extra refinement layer idea (helps slightly sometimes)
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),  # output in [-1,1]
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
            # NOTE: no Sigmoid here (we use BCEWithLogitsLoss)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


# ============================================================
# VIS / SAMPLING (from the other notebook ideas)
# ============================================================

@torch.no_grad()
def show_generated_images(epoch: int, G: nn.Module, z_dim: int, device: str, out_dir: str, n_images: int, grid_cols: int, show: bool):
    G.eval()
    z = torch.randn(n_images, z_dim, device=device)
    fake = G(z)  # [-1,1]
    fake_01 = (fake + 1.0) / 2.0
    out_path = os.path.join(out_dir, f"samples_epoch_{epoch:03d}.png")
    save_and_maybe_show_grid(fake_01, out_path, f"Generated samples (epoch {epoch})", grid_cols, show)

@torch.no_grad()
def interpolate_and_generate(G: nn.Module, z_dim: int, device: str, out_path: str, num_steps: int = 10, show: bool = True):
    """
    Interpolate between two random z vectors and visualize the morphing.
    (Idea appears in the provided notebook.) :contentReference[oaicite:3]{index=3}
    """
    G.eval()
    z1 = torch.randn(1, z_dim, device=device)
    z2 = torch.randn(1, z_dim, device=device)
    alphas = np.linspace(0, 1, num_steps)

    imgs = []
    for a in alphas:
        z = a * z1 + (1.0 - a) * z2
        x = G(z).squeeze(0)          # (3,H,W) in [-1,1]
        x = (x + 1.0) / 2.0
        imgs.append(x.detach().cpu())

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2.2, 2.2))
    for i in range(num_steps):
        axes[i].imshow(imgs[i].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


# ============================================================
# TRAIN
# ============================================================

def train(cfg: Config):
    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    # Normalize to [-1,1] exactly like the other notebook suggestion :contentReference[oaicite:4]{index=4}
    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1,1]
    ])

    ds = Flowers102FromJSON(cfg.json_path, cfg.image_root, transform=transform)
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

    # more stable than Sigmoid + BCELoss
    bce_logits = nn.BCEWithLogitsLoss()

    D_losses: List[float] = []
    G_losses: List[float] = []

    fixed_z = torch.randn(cfg.n_sample_grid, cfg.z_dim, device=cfg.device)

    step = 0
    for epoch in range(cfg.epochs):
        G.train()
        D.train()

        # instance noise schedule (high early, goes to 0)
        noise_std = linear_schedule(epoch, cfg.epochs, cfg.noise_std_start, cfg.noise_std_end)

        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=True)
        lossD_running = 0.0
        lossG_running = 0.0

        for i, (real, _y) in enumerate(pbar):
            real = real.to(cfg.device, non_blocking=True)
            B = real.size(0)

            if cfg.use_instance_noise and noise_std > 0:
                real_in = real + noise_std * torch.randn_like(real)
            else:
                real_in = real

            # -----------------------------------
            # Train D
            # -----------------------------------
            D.zero_grad(set_to_none=True)

            real_targets = torch.full((B,), cfg.real_label, device=cfg.device)  # 0.9
            fake_targets = torch.zeros(B, device=cfg.device)

            real_logits = D(real_in)
            lossD_real = bce_logits(real_logits, real_targets)

            z = torch.randn(B, cfg.z_dim, device=cfg.device)
            fake = G(z).detach()
            if cfg.use_instance_noise and noise_std > 0:
                fake_in = fake + noise_std * torch.randn_like(fake)
            else:
                fake_in = fake

            fake_logits = D(fake_in)
            lossD_fake = bce_logits(fake_logits, fake_targets)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optD.step()

            # -----------------------------------
            # Train G
            # -----------------------------------
            G.zero_grad(set_to_none=True)

            z = torch.randn(B, cfg.z_dim, device=cfg.device)
            gen = G(z)
            if cfg.use_instance_noise and noise_std > 0:
                gen_in = gen + noise_std * torch.randn_like(gen)
            else:
                gen_in = gen

            gen_logits = D(gen_in)
            # want D(gen) to be "real"
            lossG = bce_logits(gen_logits, torch.ones(B, device=cfg.device))
            lossG.backward()
            optG.step()

            D_losses.append(float(lossD.item()))
            G_losses.append(float(lossG.item()))

            lossD_running += float(lossD.item())
            lossG_running += float(lossG.item())

            if (i % cfg.tqdm_update_every) == 0:
                with torch.no_grad():
                    D_real_prob = torch.sigmoid(real_logits).mean().item()
                    D_fake_prob = torch.sigmoid(fake_logits).mean().item()
                pbar.set_postfix({
                    "lossD": f"{lossD_running/(i+1):.3f}",
                    "lossG": f"{lossG_running/(i+1):.3f}",
                    "D(real)": f"{D_real_prob:.2f}",
                    "D(fake)": f"{D_fake_prob:.2f}",
                    "noise": f"{noise_std:.3f}",
                })

            step += 1

        # -----------------------------------
        # Samples each epoch (fixed z)
        # -----------------------------------
        if cfg.sample_every_epochs and ((epoch + 1) % cfg.sample_every_epochs == 0):
            G.eval()
            with torch.no_grad():
                fake_fixed = G(fixed_z)            # [-1,1]
                fake_fixed_01 = (fake_fixed + 1.0) / 2.0

            out_path = os.path.join(cfg.out_dir, f"samples_epoch_{epoch+1:03d}.png")
            show_now = (cfg.show_every_epochs and ((epoch + 1) % cfg.show_every_epochs == 0))
            save_and_maybe_show_grid(
                fake_fixed_01,
                out_path,
                title=f"Generated samples (epoch {epoch+1})",
                grid_cols=cfg.grid_cols,
                show=show_now,
            )

    # -----------------------------------
    # Save weights
    # -----------------------------------
    torch.save(G.state_dict(), os.path.join(cfg.out_dir, "gan_G.pkl"))
    torch.save(D.state_dict(), os.path.join(cfg.out_dir, "gan_D.pkl"))
    print(f"Saved: {os.path.join(cfg.out_dir, 'gan_G.pkl')}")
    print(f"Saved: {os.path.join(cfg.out_dir, 'gan_D.pkl')}")

    # -----------------------------------
    # Plot losses (and show)
    # -----------------------------------
    loss_plot_path = os.path.join(cfg.out_dir, "gan_loss_curves.png")
    plot_loss_curves(D_losses, G_losses, loss_plot_path, show=True)
    print(f"Saved: {loss_plot_path}")

    # -----------------------------------
    # Final samples (and show)
    # -----------------------------------
    G.eval()
    with torch.no_grad():
        z = torch.randn(cfg.n_sample_grid, cfg.z_dim, device=cfg.device)
        final = G(z)
        final_01 = (final + 1.0) / 2.0

    final_path = os.path.join(cfg.out_dir, "gan_generated_samples.png")
    save_and_maybe_show_grid(final_01, final_path, "GAN generated samples (final)", cfg.grid_cols, show=True)
    print(f"Saved: {final_path}")

    # -----------------------------------
    # Latent interpolation (and show)
    # -----------------------------------
    interp_path = os.path.join(cfg.out_dir, "gan_latent_interpolation.png")
    interpolate_and_generate(G, cfg.z_dim, cfg.device, interp_path, num_steps=10, show=True)
    print(f"Saved: {interp_path}")


if __name__ == "__main__":
    cfg = Config()
    print("Config:", cfg)
    print("Device:", cfg.device)
    train(cfg)
