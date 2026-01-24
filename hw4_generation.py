"""
HW4 Generation Script - Minimal script to load trained model and generate images
"""

import os
import json
import pickle
import torch
import matplotlib.pyplot as plt
from torchvision import utils
import numpy as np

# Import models and functions from training script
from gan_complete import Generator


def reproduce_hw4(generator_path: str = "./GAN_output/generator.pkl",
                  json_path: str = "./category_to_images.json",
                  output_path: str = "./hw4_generated_samples.png",
                  num_images: int = 10):
    """
    Load trained generator and generate images.
    
    Note: GANs generate random samples from learned distribution, not specific
    categories. The grid shows generated flower images that resemble the 
    training data but are not organized by flower category.
    
    Args:
        generator_path: Path to saved generator .pkl file
        json_path: Path to category_to_images.json
        output_path: Where to save the generated image grid
        num_images: Number of images to generate (default: 10)
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load generator - MUST match the architecture used during training
    print(f"Loading generator from {generator_path}...")
    generator = Generator(z_dim=100, g_channels=1024)  # Match training config
    with open(generator_path, 'rb') as f:
        generator.load_state_dict(pickle.load(f))
    generator = generator.to(device)
    generator.eval()
    
    # Get category count for display purposes
    with open(json_path, 'r') as f:
        categories = json.load(f)
    num_categories = len(categories)
    
    # Generate sample images (note: these are random samples, not per-category)
    # GANs don't generate specific categories unless trained as conditional GANs
    images_per_row = min(10, num_images)  # Max 10 images per row
    total_images = num_images
    print(f"Generating {total_images} random flower samples...")
    
    # Generate random latent vectors
    z = torch.randn(total_images, 100, device=device)
    
    # Generate images
    with torch.no_grad():
        fake_images = generator(z)
        # Convert from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2.0
        fake_images = torch.clamp(fake_images, 0, 1)
    
    # Create grid
    grid = utils.make_grid(fake_images, nrow=images_per_row, padding=2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(grid.permute(1, 2, 0).cpu())
    ax.axis('off')
    
    plt.title(f"Generated Flower Samples (Random from learned distribution)\n"
             f"Trained on {num_categories} flower categories", 
             fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved generated images to {output_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    reproduce_hw4()
