import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os


# -------------------------
# Config
# -------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM = 128
N_IMAGES = 20
GRID_COLS = 5
GEN_PATH = "./gan_out/gan_G.pkl"   # or gan_G_best.pkl


# -------------------------
# Generator (must match training)
# -------------------------

class Generator(nn.Module):
    def __init__(self, z_dim):
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

    def forward(self, z):
        z = z.view(z.size(0), self.z_dim, 1, 1)
        return self.net(z)


# -------------------------
# Utils
# -------------------------

def from_gan_range(x):
    return (x + 1.0) / 2.0


def show_grid(imgs, ncol=5, title="GAN samples"):
    imgs = imgs.clamp(0, 1)
    N, C, H, W = imgs.shape
    nrow = int(np.ceil(N / ncol))

    grid = torch.ones((C, nrow * H, ncol * W))
    for i in range(N):
        r = i // ncol
        c = i % ncol
        grid[:, r*H:(r+1)*H, c*W:(c+1)*W] = imgs[i]

    grid = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(ncol * 2.5, nrow * 2.5))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -------------------------
# Main
# -------------------------

def main():
    assert os.path.exists(GEN_PATH), f"Generator not found: {GEN_PATH}"

    G = Generator(Z_DIM).to(DEVICE)
    G.load_state_dict(torch.load(GEN_PATH, map_location=DEVICE))
    G.eval()

    with torch.no_grad():
        z = torch.randn(N_IMAGES, Z_DIM, device=DEVICE)
        fake = G(z)
        fake = from_gan_range(fake)

    show_grid(fake, ncol=GRID_COLS)


if __name__ == "__main__":
    main()
