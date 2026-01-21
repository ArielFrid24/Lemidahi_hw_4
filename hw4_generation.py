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
                  output_path: str = "./hw4_generated_samples.png"):
    """
    Load trained generator and generate 10 images per flower category.
    
    Args:
        generator_path: Path to saved generator .pkl file
        json_path: Path to category_to_images.json
        output_path: Where to save the generated image grid
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load generator
    print(f"Loading generator from {generator_path}...")
    generator = Generator(z_dim=100, g_channels=512)
    with open(generator_path, 'rb') as f:
        generator.load_state_dict(pickle.load(f))
    generator = generator.to(device)
    generator.eval()
    
    # Get category IDs in order
    with open(json_path, 'r') as f:
        categories = json.load(f)
    category_ids = sorted(categories.keys(), key=int)
    num_categories = len(category_ids)
    
    # Generate 10 images per category
    images_per_category = 10
    total_images = num_categories * images_per_category
    print(f"Generating {total_images} images ({images_per_category} per category)...")
    
    # Generate random latent vectors
    z = torch.randn(total_images, 100, device=device)
    
    # Generate images
    with torch.no_grad():
        fake_images = generator(z)
        # Convert from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2.0
        fake_images = torch.clamp(fake_images, 0, 1)
    
    # Create grid
    grid = utils.make_grid(fake_images, nrow=images_per_category, padding=2)
    
    # Plot with category labels
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(grid.permute(1, 2, 0).cpu())
    ax.axis('off')
    
    # Add category labels on the left side of each row
    img_height = fake_images.shape[2]
    padding = 2
    row_height = img_height + padding
    
    for i, cat_id in enumerate(category_ids):
        # Calculate y position for each row
        y_pos = (i * row_height) + (row_height / 2)
        # Add text label with white background
        ax.text(-20, y_pos, f"Class {cat_id}", 
               fontsize=8, va='center', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(f"Generated Flowers: {images_per_category} images per category", 
             fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved generated images to {output_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    reproduce_hw4()
