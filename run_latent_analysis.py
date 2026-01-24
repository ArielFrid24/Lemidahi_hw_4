"""Run latent space analysis on trained GAN models"""

import os
import pickle
from gan_complete import Generator, Config, analyze_latent_space
from GAN_by_Altmans import visualize_outputs_and_2d

# Create configuration
config = Config()

# Initialize generator
generator = Generator(config.z_dim).to(config.device)

# Load trained generator
generator_path = os.path.join(config.out_dir, "generator.pkl")
with open(generator_path, 'rb') as f:
    generator.load_state_dict(pickle.load(f))
print(f"Loaded generator from {generator_path}")

# Analyze latent space and generate similar/dissimilar pairs
analyze_latent_space(generator, config.z_dim, config.device, config.out_dir)

# visualize_outputs_and_2d(generator, config.z_dim, config.device)

print("\nLatent space analysis complete!")
