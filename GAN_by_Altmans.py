
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA


class Generator(nn.Module):
    """
    A smaller DCGAN-style Generator.
    Outputs: Image of shape (3, 64, 64).
    """
    def __init__(self, latent_dim=100, g_channels=512):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, g_channels * 4 * 4),
            nn.BatchNorm1d(g_channels * 4 * 4),
            nn.ReLU(True)
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(g_channels, g_channels // 2, 4, 2, 1),
            nn.BatchNorm2d(g_channels // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_channels // 2, g_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(g_channels // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_channels // 4, g_channels // 8, 4, 2, 1),
            nn.BatchNorm2d(g_channels // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(g_channels // 8, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z).view(-1, int(self.fc[0].out_features / 16), 4, 4)
        img = self.model(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, d_channels=512):
        super(Discriminator, self).__init__()
        def disc_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_feat, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *disc_block(3, d_channels // 8, normalize=False),
            *disc_block(d_channels // 8, d_channels // 4),
            *disc_block(d_channels // 4, d_channels // 2),
            *disc_block(d_channels // 2, d_channels),
            nn.Conv2d(d_channels, 1, 4, 1, 0)  # single scalar output
        )

    def forward(self, img):
        return self.model(img)


def visualize_outputs_and_2d(generator, latent_dim, device="cpu", num_images=16):
    """
    1) Generate `num_images` random latent vectors z.
    2) Use the generator to produce 16 images.
    3) Display them in a 4x4 grid (like a normal visualize).
    4) Apply PCA to the 16 latent vectors to get 2D coords.
    5) Display those 16 images in a 2D scatter plot, positioned by their PCA coords.
    """

    generator.eval()  # disable dropout/batchnorm updates, if any

    # -------------------------------------------------
    # STEP 1: Sample random latent vectors (z) [16 x latent_dim]
    # -------------------------------------------------
    z = torch.randn(num_images, latent_dim, device=device)

    # -------------------------------------------------
    # STEP 2: Generate images from these latent vectors
    # -------------------------------------------------
    with torch.no_grad():
        generated_imgs = generator(z).cpu()  # shape: [16, 3, H, W]

    # Rescale images from [-1, 1] to [0, 1]
    generated_imgs = (generated_imgs * 0.5) + 0.5

    # -------------------------------------------------
    # STEP 3: Display images in a 4x4 grid
    # -------------------------------------------------
    rows = cols = int(num_images**0.5)  # e.g., 4 if num_images=16
    fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axs = axs.flatten()

    for i in range(num_images):
        img_np = generated_imgs[i].permute(1, 2, 0).numpy()  # [H, W, 3]
        axs[i].imshow(img_np)
        axs[i].axis('off')

    plt.suptitle("Generated Images (Random Z)", fontsize=14)
    plt.tight_layout()
    plt.show()

    z_cpu = z.cpu().numpy()  # shape [16, latent_dim]
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_cpu)

    print("z_2d:\n", z_2d)  # Debug: check if it's non-zero

    x_min, x_max = z_2d[:, 0].min(), z_2d[:, 0].max()
    y_min, y_max = z_2d[:, 1].min(), z_2d[:, 1].max()

    # If the entire range is zero, you won't see anything
    if x_min == x_max and y_min == y_max:
        print("All PCA points are identical or extremely close. Nothing to plot in 2D.")
        # Possibly skip the scatter code altogether or add a small offset

    margin = 0.1 * max(x_max - x_min, y_max - y_min)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (x, y) in enumerate(z_2d):
        img = generated_imgs[i].permute(1, 2, 0).numpy()
        im = OffsetImage(img, zoom=0.6)
        ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)
        ax.add_artist(ab)

    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_title("PCA of Latent Vectors (2D) with Images")

    # Save or show
    plt.savefig("latent_space_pca.png")
    print("Saved figure: latent_space_pca.png")
    plt.show()  # If you have an interactive GUI

    generator.train()  # back to training mode


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device="cpu"):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    fake = torch.ones_like(d_interpolates, device=device, requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def visualize_outputs(generator, latent_dim, device="cpu", num_images=16):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        generated_imgs = generator(z).cpu()

    generated_imgs = (generated_imgs * 0.5) + 0.5

    rows = int(num_images ** 0.5)
    cols = (num_images + rows - 1) // rows
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axs = axs.flatten()
    for i, img in enumerate(generated_imgs):
        img = img.permute(1, 2, 0).numpy()
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()
    generator.train()


def plot_losses(times, d_losses, g_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(times, d_losses, label='Discriminator Loss')
    plt.plot(times, g_losses, label='Generator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Absolute Value)')
    plt.title('Training Convergence')
    plt.legend()
    plt.show()


def train_gan(latent_dim, data_loader, epochs=20, lr=0.0002, n_critic=1, lambda_gp=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    generator = Generator(latent_dim=latent_dim, g_channels=512).to(device)
    discriminator = Discriminator(d_channels=512).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))

    # For plotting
    d_losses_plot = []
    g_losses_plot = []
    iteration_list = []

    iteration = 0

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(data_loader):
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z).detach()

            optimizer_D.zero_grad()

            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            gradient_pen = compute_gradient_penalty(discriminator, real_imgs, fake_imgs, device)
            d_loss_gp = d_loss + lambda_gp * gradient_pen
            d_loss_gp.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            gen_validity = discriminator(gen_imgs)
            g_loss = -torch.mean(gen_validity)
            g_loss.backward()
            optimizer_G.step()

            d_losses_plot.append(abs(d_loss.item()))
            g_losses_plot.append(abs(g_loss.item()))
            iteration_list.append(iteration)
            iteration += 1

            if i % 50 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(data_loader)}] "
                      f"D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f} | GP: {gradient_pen.item():.4f}")

        visualize_outputs(generator, latent_dim, device=device, num_images=16)

    plot_losses(iteration_list, d_losses_plot, g_losses_plot)
    visualize_outputs_and_2d(generator, latent_dim=100, device="cuda", num_images=16)

    return generator, discriminator

def reproduce_hw4():
    generator = Generator(100)
    generator = Generator.load_state_dict(torch.load("generator.pkl"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    visualize_outputs(generator,100)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # Dataset & DataLoader
    dataset = datasets.ImageFolder("102flowers", transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Launch training
    generator, discriminator = train_gan(
        latent_dim=100,
        data_loader=data_loader,
        epochs=20,
        lr=0.0002,
        n_critic=1,
        lambda_gp=10
    )
    torch.save(generator.state_dict(), "generator.pkl")
    torch.save(discriminator.state_dict(), "discriminator.pkl")