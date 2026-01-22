"""
Complete GAN Training Pipeline for 102 Category Flower Dataset

This module provides a comprehensive implementation of a DCGAN for generating
flower images. It includes the Generator, Discriminator, data loading, training
loop with progress visualization, and loss plotting.
"""

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchvision import transforms, utils


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    """Configuration parameters for GAN training."""
    
    # Data paths
    json_path: str = "category_to_images.json"
    image_root: str = "./jpg"
    
    # Output settings
    out_dir: str = "./GAN_output"
    
    # Training hyperparameters
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_size: int = 64
    batch_size: int = 32  # Smaller batch for more stable gradients
    num_workers: int = 2
    epochs: int = 40
    
    # Model architecture
    z_dim: int = 100  # Standard latent dimension
    
    # Optimizer settings
    lr_generator: float = 2e-4
    lr_discriminator: float = 2e-4
    betas: Tuple[float, float] = (0.0, 0.9)  # WGAN-GP requires (0.0, 0.9)
    
    # WGAN-GP settings
    lambda_gp: float = 10.0  # Gradient penalty coefficient
    n_critic: int = 1  # Number of discriminator updates per generator update
    
    # Visualization
    n_sample_images: int = 25
    sample_grid_cols: int = 5
    save_samples_every: int = 1  # Save samples every N epochs


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 5) -> None:
    """
    Save a grid of images to disk.
    
    Args:
        images: Tensor of images with shape (N, C, H, W) in range [0, 1]
        path: Output file path
        nrow: Number of images per row in the grid
    """
    grid = utils.make_grid(images, nrow=nrow, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_gradient_penalty(discriminator: nn.Module, real_samples: torch.Tensor, 
                            fake_samples: torch.Tensor, device: str) -> torch.Tensor:
    """
    Computes the gradient penalty term used in Wasserstein GAN with Gradient Penalty (WGAN-GP).
    
    This function enforces the Lipschitz constraint on the discriminator by penalizing the 
    gradients of the discriminator's output with respect to its inputs when evaluated on 
    interpolated samples between real and fake data.
    
    Args:
        discriminator: The discriminator network
        real_samples: Real images tensor of shape (batch_size, 3, H, W)
        fake_samples: Fake images tensor of shape (batch_size, 3, H, W)
        device: Device to run computation on
        
    Returns:
        Gradient penalty scalar tensor
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    
    # Get random interpolation between real and fake samples
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    
    # Calculate discriminator output for interpolated samples
    d_interpolates = discriminator(interpolates)
    
    # Get gradient w.r.t. interpolates
    fake = torch.ones_like(d_interpolates, device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Flatten gradients
    gradients = gradients.view(gradients.size(0), -1)
    
    # Calculate penalty: (||gradient|| - 1)^2
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


def plot_losses(d_losses: List[float], g_losses: List[float], 
                save_path: str) -> None:
    """
    Plot and save discriminator and generator losses.
    
    Args:
        d_losses: List of discriminator losses
        g_losses: List of generator losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('WGAN-GP Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def analyze_latent_space(generator: Generator, z_dim: int, device: str, 
                         out_dir: str, num_images: int = 20) -> None:
    """
    Generate random images and visualize them in PCA latent space.
    
    This function:
    1. Generates random latent vectors and corresponding images
    2. Projects latent vectors to 2D using PCA
    3. Displays images at their PCA coordinates
    
    Args:
        generator: Trained generator model
        z_dim: Dimension of latent space
        device: Device to run on
        out_dir: Output directory for saving results
        num_images: Number of images to generate and visualize (default: 16)
    """
    print(f"\nAnalyzing latent space with {num_images} images...")
    generator.eval()
    
    # Generate random latent vectors
    z_vectors = torch.randn(num_images, z_dim, device=device)
    
    # Generate images
    with torch.no_grad():
        generated_images = generator(z_vectors)
        generated_images = (generated_images + 1) / 2.0  # Convert to [0, 1]
        generated_images = torch.clamp(generated_images, 0, 1)
    
    # Use PCA to project latent vectors to 2D for visualization
    from sklearn.decomposition import PCA
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    
    z_cpu = z_vectors.cpu().numpy()
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_cpu)
    
    # Calculate plot limits with margin
    x_min, x_max = z_2d[:, 0].min(), z_2d[:, 0].max()
    y_min, y_max = z_2d[:, 1].min(), z_2d[:, 1].max()
    margin = 0.15 * max(x_max - x_min, y_max - y_min)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Function to add image to plot
    def add_image_to_plot(img_tensor, x, y, zoom=0.8):
        """Add an image at specified coordinates"""
        img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
        imagebox = OffsetImage(img_np, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x, y), frameon=True, 
                           pad=0.1, bboxprops=dict(edgecolor='black', linewidth=2))
        ax.add_artist(ab)
    
    # Plot each image at its PCA coordinate
    for i in range(num_images):
        add_image_to_plot(generated_images[i], z_2d[i, 0], z_2d[i, 1], zoom=0.8)
    
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)
    ax.set_title('PCA of Latent Vectors (2D) with Images', fontsize=16)
    ax.grid(False)
    
    # Save figure
    analysis_path = os.path.join(out_dir, "latent_space_analysis.png")
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    print(f"Saved latent space analysis to {analysis_path}")
    
    plt.show()
    plt.close('all')
    
    print(f"\nGenerated and plotted {num_images} images using PCA coordinates")


# ============================================================
# DATASET
# ============================================================

class Flowers102Dataset(Dataset):
    """
    Dataset class for 102 Category Flower Dataset.
    
    Loads flower images based on a JSON mapping file that maps
    category IDs to image filenames.
    """
    
    def __init__(self, json_path: str, image_root: str, transform=None):
        """
        Initialize the dataset.
        
        Args:
            json_path: Path to JSON file mapping categories to images
            image_root: Root directory containing image files
            transform: Optional transform to apply to images
        """
        self.image_root = image_root
        self.transform = transform
        
        # Load category to images mapping
        with open(json_path, "r", encoding="utf-8") as f:
            mapping: Dict[str, List[str]] = json.load(f)
        
        # Build list of (filename, class_id) tuples
        self.items: List[Tuple[str, int]] = []
        for category_id, filenames in mapping.items():
            class_idx = int(category_id) - 1  # Convert to 0-indexed
            for filename in filenames:
                self.items.append((filename, class_idx))
        
        if not self.items:
            raise ValueError("No images found in JSON mapping.")
        
        print(f"Loaded {len(self.items)} images from {len(mapping)} categories")
    
    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its class label.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (image_tensor, class_id)
        """
        filename, class_id = self.items[idx]
        image_path = os.path.join(self.image_root, filename)
        
        # Load and convert image
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_id


# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):
    """
    DCGAN Generator network with FC layer.
    
    Takes a random noise vector z and generates a synthetic image.
    Architecture uses a fully connected layer first, then transposed 
    convolutions to progressively upsample from latent to image space.
    This architecture provides more capacity and better results.
    """
    
    def __init__(self, z_dim: int = 100, g_channels: int = 512):
        """
        Initialize the Generator.
        
        Args:
            z_dim: Dimension of the latent noise vector
            g_channels: Base number of channels (will be multiplied)
        """
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.g_channels = g_channels
        
        # Fully connected layer to expand latent vector
        self.fc = nn.Sequential(
            nn.Linear(z_dim, g_channels * 4 * 4),
            nn.BatchNorm1d(g_channels * 4 * 4),
            nn.ReLU(True)
        )
        
        # Convolutional layers for upsampling
        self.model = nn.Sequential(
            # Input: g_channels x 4 x 4
            nn.ConvTranspose2d(g_channels, g_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(g_channels // 2),
            nn.ReLU(True),
            # State: g_channels//2 x 8 x 8
            
            nn.ConvTranspose2d(g_channels // 2, g_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(g_channels // 4),
            nn.ReLU(True),
            # State: g_channels//4 x 16 x 16
            
            nn.ConvTranspose2d(g_channels // 4, g_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(g_channels // 8),
            nn.ReLU(True),
            # State: g_channels//8 x 32 x 32
            
            nn.ConvTranspose2d(g_channels // 8, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # Output: 3 x 64 x 64, values in [-1, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            z: Noise vector of shape (batch_size, z_dim)
            
        Returns:
            Generated images of shape (batch_size, 3, 64, 64)
        """
        # Pass through FC layer and reshape
        out = self.fc(z)
        out = out.view(out.size(0), self.g_channels, 4, 4)
        # Pass through convolutional layers
        img = self.model(out)
        return img


# ============================================================
# DISCRIMINATOR
# ============================================================

class Discriminator(nn.Module):
    """
    WGAN-GP Discriminator (Critic) network.
    
    Takes an image and outputs a scalar score (not probability).
    Uses InstanceNorm2d instead of BatchNorm2d for WGAN-GP stability.
    Architecture uses strided convolutions to progressively downsample.
    """
    
    def __init__(self, d_channels: int = 512):
        """
        Initialize the Discriminator.
        
        Args:
            d_channels: Base number of channels for discriminator
        """
        super(Discriminator, self).__init__()
        
        def disc_block(in_feat: int, out_feat: int, normalize: bool = True):
            """Discriminator block with optional normalization."""
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                # InstanceNorm2d is better for WGAN-GP (no batch statistics coupling)
                layers.append(nn.InstanceNorm2d(out_feat, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            # Input: 3 x 64 x 64
            *disc_block(3, d_channels // 8, normalize=False),
            # State: d_channels//8 x 32 x 32
            
            *disc_block(d_channels // 8, d_channels // 4),
            # State: d_channels//4 x 16 x 16
            
            *disc_block(d_channels // 4, d_channels // 2),
            # State: d_channels//2 x 8 x 8
            
            *disc_block(d_channels // 2, d_channels),
            # State: d_channels x 4 x 4
            
            nn.Conv2d(d_channels, 1, kernel_size=4, stride=1, padding=0)
            # Output: 1 x 1 x 1 (single scalar output)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input images of shape (batch_size, 3, 64, 64)
            
        Returns:
            Scores of shape (batch_size, 1, 1, 1) or (batch_size,) after view
        """
        return self.model(x)


def weights_init(m):
    """
    Initialize network weights according to DCGAN paper.
    
    Args:
        m: Neural network module
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ============================================================
# TRAINING
# ============================================================

def generate_and_save_samples(generator, noise: torch.Tensor, output_path: str, 
                             title: str = "Generated Samples", nrow: int = 8, 
                             figsize: tuple = (10, 10), show: bool = True) -> None:
    """
    Generate images from noise and save them as a grid.
    
    Args:
        generator: Generator model
        noise: Latent noise tensor of shape (N, z_dim)
        output_path: Path to save the image grid
        title: Title for the plot
        nrow: Number of images per row in the grid
        figsize: Figure size for matplotlib
        show: Whether to display the image with plt.show()
    """
    generator.eval()
    with torch.no_grad():
        fake_samples = generator(noise)
        # Convert from [-1, 1] to [0, 1]
        fake_samples = (fake_samples + 1) / 2.0
        fake_samples = torch.clamp(fake_samples, 0, 1)
        
        # Save samples
        save_image_grid(fake_samples, output_path, nrow=nrow)
        print(f"Saved samples to {output_path}")
        
        # Display the samples if requested
        if show:
            img = plt.imread(output_path)
            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.axis('off')
            plt.title(title)
            plt.show()
            plt.close()
    
    generator.train()


def train_gan(generator: Generator, discriminator: Discriminator, 
             dataloader: DataLoader, config: Config) -> Tuple[Generator, Discriminator]:
    """
    Train the GAN model using WGAN-GP (Wasserstein GAN with Gradient Penalty).
    
    This function handles the complete training loop including:
    - Model initialization with proper weights
    - Alternating optimization of Generator and Discriminator
    - Gradient penalty computation for Lipschitz constraint
    - Progress visualization and checkpointing
    - Final loss plotting
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        dataloader: DataLoader providing training images
        config: Configuration object with training parameters
        
    Returns:
        Tuple of (trained_generator, trained_discriminator)
    """
    ensure_dir(config.out_dir)
    print(f"Training on device: {config.device}")
    
    # Models are already initialized and passed as parameters
    
    # Optimizers (WGAN-GP typically uses different betas)
    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=config.lr_generator,
        betas=config.betas
    )
    
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.lr_discriminator,
        betas=config.betas
    )
    
    # Fixed noise for consistent visualization
    fixed_noise = torch.randn(config.n_sample_images, config.z_dim, 
                             device=config.device)
    
    # Training history
    d_losses = []
    g_losses = []
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(config.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0
        
        for real_images, _ in progress_bar:
            batch_size = real_images.size(0)
            real_images = real_images.to(config.device)
            
            # ====================================
            # Train Discriminator (Critic)
            # ====================================
            # Generate fake images ONCE for discriminator training
            z = torch.randn(batch_size, config.z_dim, device=config.device)
            fake_images = generator(z).detach()
            
            optimizer_D.zero_grad()
            
            # Get discriminator outputs
            real_validity = discriminator(real_images)
            fake_validity = discriminator(fake_images)
            
            # Compute Wasserstein loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            
            # Compute gradient penalty
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_images, fake_images, config.device
            )
            
            # Total discriminator loss with gradient penalty
            d_loss_gp = d_loss + config.lambda_gp * gradient_penalty
            d_loss_gp.backward()
            optimizer_D.step()
            
            # ====================================
            # Train Generator
            # ====================================
            optimizer_G.zero_grad()
            
            # Generate NEW fake images for generator training
            z = torch.randn(batch_size, config.z_dim, device=config.device)
            gen_images = generator(z)
            gen_validity = discriminator(gen_images)
            
            # Generator loss: maximize D(fake) = minimize -D(fake)
            g_loss = -torch.mean(gen_validity)
            g_loss.backward()
            optimizer_G.step()
            
            # Record losses (use absolute values for stability monitoring)
            d_losses.append(abs(d_loss.item()))
            g_losses.append(abs(g_loss.item()))
            epoch_d_loss += abs(d_loss.item())
            epoch_g_loss += abs(g_loss.item())
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}',
                'D(real)': f'{real_validity.mean().item():.3f}',
                'D(fake)': f'{fake_validity.mean().item():.3f}',
                'GP': f'{gradient_penalty.item():.3f}'
            })
        
        # Epoch summary
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        print(f"Epoch {epoch+1} - Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")
        
        # Generate and save sample images
        if (epoch + 1) % config.save_samples_every == 0:
            sample_path = os.path.join(config.out_dir, 
                                      f"samples_epoch_{epoch+1:03d}.png")
            generate_and_save_samples(
                generator=generator,
                noise=fixed_noise,
                output_path=sample_path,
                title=f'Generated Samples - Epoch {epoch+1}',
                nrow=config.sample_grid_cols,
                figsize=(10, 10),
                show=True
            )
    
    print(f"\nTraining complete!")
    
    # Plot and display final losses
    loss_plot_path = os.path.join(config.out_dir, "training_losses.png")
    plot_losses(d_losses, g_losses, loss_plot_path)
    print(f"Saved loss plot to {loss_plot_path}")
    
    return generator, discriminator


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Create configuration
    config = Config()
    
    print("=" * 60)
    print("WGAN-GP Training Configuration")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Image size: {config.img_size}x{config.img_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Latent dimension: {config.z_dim}")
    print(f"Learning rate (G): {config.lr_generator}")
    print(f"Learning rate (D): {config.lr_discriminator}")
    print(f"Lambda GP: {config.lambda_gp}")
    print(f"Output directory: {config.out_dir}")
    print("=" * 60)
    
    # Set random seed
    set_seed(config.seed)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Initialize dataset
    print("\nLoading dataset...")
    dataset = Flowers102Dataset(
        json_path=config.json_path,
        image_root=config.image_root,
        transform=transform
    )
    
    # Initialize dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=(config.device == "cuda")
    )
    
    # Initialize models
    print("\nInitializing models...")
    generator = Generator(config.z_dim).to(config.device)
    discriminator = Discriminator().to(config.device)
    
    # Apply weight initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Train the GAN
    print("\nStarting training...")
    generator, discriminator = train_gan(generator, discriminator, dataloader, config)
    
    # Save trained models as pickle files
    print("\nSaving trained models...")
    import pickle
    
    generator_path = os.path.join(config.out_dir, "generator.pkl")
    discriminator_path = os.path.join(config.out_dir, "discriminator.pkl")
    
    with open(generator_path, 'wb') as f:
        pickle.dump(generator.state_dict(), f)
    print(f"Saved generator to {generator_path}")
    
    with open(discriminator_path, 'wb') as f:
        pickle.dump(discriminator.state_dict(), f)
    print(f"Saved discriminator to {discriminator_path}")
    
    # Analyze latent space and generate similar/dissimilar pairs
    analyze_latent_space(generator, config.z_dim, config.device, config.out_dir)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
