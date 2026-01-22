# WGAN-GP Flower Generation

A high-quality implementation of Wasserstein GAN with Gradient Penalty (WGAN-GP) for generating realistic flower images from the 102 Category Flower Dataset.

## Model Architecture

This project implements a **DCGAN-style WGAN-GP** with the following improvements:

### Key Features
- **WGAN-GP Framework**: Uses Wasserstein distance with gradient penalty for stable training
- **High Capacity Networks**: 1024 channels in both Generator and Discriminator (~12.7M and ~11M parameters respectively)
- **Learning Rate Scheduling**: Cosine annealing for optimal convergence
- **Extended Training**: 60 epochs for superior image quality

### Generator Architecture
- **Input**: 100-dimensional latent vector
- **FC Expansion**: Dense layer to 1024×4×4 feature maps
- **Progressive Upsampling**: 4×4 → 8×8 → 16×16 → 32×32 → 64×64
- **Normalization**: BatchNorm2d for stable training
- **Output**: 64×64 RGB images via Tanh activation

### Discriminator (Critic) Architecture
- **Input**: 64×64 RGB images
- **Progressive Downsampling**: Strided convolutions with InstanceNorm2d
- **Output**: Unbounded scalar score (not probability)
- **Architecture**: 5 convolutional layers with LeakyReLU activations

## Training Configuration

```python
Image Size: 64×64
Batch Size: 32
Epochs: 60
Latent Dimension: 100
Learning Rate: 2e-4 (both G and D)
Adam Betas: (0.0, 0.9)  # WGAN-GP specific
Gradient Penalty Weight: 10.0
```

## Quick Start

### Prerequisites
```bash
conda activate ML2
```

### Training the Model
```bash
python gan_complete.py
```

The training will:
- Load 8,189 flower images from 102 categories
- Train for 60 epochs with progress bars
- Save sample images every epoch to `./GAN_output/`
- Save trained models as `generator.pkl` and `discriminator.pkl`
- Generate training loss plots

### Training Output
- **Sample Images**: `./GAN_output/samples_epoch_XXX.png`
- **Loss Plot**: `./GAN_output/training_losses.png`
- **Trained Models**: `./GAN_output/generator.pkl` and `discriminator.pkl`
- **Latent Space Analysis**: `./GAN_output/latent_space_analysis.png`

## Model Improvements

This implementation includes key "Quick Wins" optimizations:

1. ✅ **Increased Channels to 1024** - More representational capacity
2. ✅ **Learning Rate Schedulers** - Cosine annealing for fine-tuning
3. ✅ **Extended Training** - 60 epochs for better convergence

### Expected Results
- **High-quality flower images** with sharp petals and realistic textures
- **Diverse outputs** covering all 102 flower categories
- **Reduced artifacts** compared to baseline models
- **Stable training** with meaningful loss metrics

## Technical Details

### Why WGAN-GP?
- **Stable Training**: Wasserstein distance provides better gradients than JS divergence
- **Meaningful Loss**: Discriminator loss correlates with generation quality
- **No Mode Collapse**: Gradient penalty prevents discriminator from becoming too strong

### Key Design Choices
- **InstanceNorm2d** in discriminator (required for proper gradient penalty)
- **No Sigmoid Output** (critic outputs unbounded scores)
- **Beta₁=0.0** in Adam (WGAN-GP requirement)
- **Normalized Images** to [-1, 1] range (matches Tanh output)

## Files

- `gan_complete.py` - Complete training pipeline
- `ARCHITECTURE_DECISIONS.md` - Detailed design rationale
- `category_to_images.json` - Dataset mapping file
- `./jpg/` - Flower images directory
- `./GAN_output/` - Training outputs and saved models

## References

- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [DCGAN](https://arxiv.org/abs/1511.06434)
