# HW4 Final Generation Features

## New Features Added

### 1. Latent Space Analysis (in gan_complete.py)

After training completes, the script automatically analyzes the latent space:

**What it does:**
- Generates 100 sample images from random latent vectors
- Computes L2 distances between all pairs of latent vectors
- Identifies 3 most **similar** pairs (smallest L2 distance)
- Identifies 3 most **dissimilar** pairs (largest L2 distance)
- Creates PCA visualization showing these pairs
- Saves L2 norms to file

**Outputs:**
- `GAN_output/latent_space_analysis.png` - Visual plot with pairs highlighted
- `GAN_output/latent_l2_norms.txt` - Text file with exact distances

**Visualization:**
- Green circles + solid lines = Similar pairs
- Red squares + dashed lines = Dissimilar pairs
- Shows actual generated images for each pair

### 2. HW4 Reproduction Script (hw4_generation.py)

New standalone script to load trained model and generate images.

**Usage:**
```bash
python hw4_generation.py
```

**What it does:**
- Loads trained generator from `GAN_output/generator.pkl`
- Reads `category_to_images.json` to get 102 categories
- Generates 10 images per category (1,020 total images)
- Saves multiple output formats

**Outputs:**
```
hw4_reproduction/
├── all_generated_flowers.png          # Grid of all 1,020 images
├── sample_generated_flowers.png       # Grid of first 100 images
├── high_quality_samples.png           # 9 high-quality samples (3×3 grid)
└── examples/
    ├── flower_001.png                 # Individual high-res images
    ├── flower_002.png
    └── ... (20 examples)
```

**Function API:**
```python
from hw4_generation import reproduce_hw4

reproduce_hw4(
    model_path="./GAN_output/generator.pkl",
    json_path="./category_to_images.json", 
    output_dir="./hw4_reproduction",
    z_dim=100
)
```

## Complete Workflow

### Step 1: Train the GAN
```bash
python gan_complete.py
```

This will:
1. Train for 40 epochs
2. Save samples every epoch to `GAN_output/samples_epoch_*.png`
3. Save training losses plot to `GAN_output/training_losses.png`
4. Save models to `GAN_output/generator.pkl` and `discriminator.pkl`
5. **NEW:** Analyze latent space and save analysis
6. **NEW:** Save L2 norms of similar/dissimilar pairs

### Step 2: Generate New Images
```bash
python hw4_generation.py
```

This will:
1. Load the trained generator
2. Generate 1,020 diverse flower images (10 per category)
3. Save multiple visualization formats
4. Create individual high-quality examples

## Output Files Summary

### Training Outputs (GAN_output/)
```
GAN_output/
├── generator.pkl                      # Trained generator weights
├── discriminator.pkl                  # Trained discriminator weights
├── samples_epoch_001.png              # Samples from epoch 1
├── samples_epoch_002.png              # Samples from epoch 2
├── ...
├── samples_epoch_040.png              # Samples from epoch 40
├── training_losses.png                # D and G loss curves
├── latent_space_analysis.png          # ✨ NEW: Similar/dissimilar pairs
└── latent_l2_norms.txt               # ✨ NEW: L2 distances
```

### Reproduction Outputs (hw4_reproduction/)
```
hw4_reproduction/
├── all_generated_flowers.png          # ✨ NEW: All 1,020 images
├── sample_generated_flowers.png       # ✨ NEW: Sample 100 images
├── high_quality_samples.png           # ✨ NEW: 9 HQ samples
└── examples/                          # ✨ NEW: Individual images
    └── flower_*.png
```

## Latent Space Analysis Details

### Similar Pairs
Images whose latent vectors are close in L2 distance should look visually similar (e.g., same color, similar petal structure).

Example output in `latent_l2_norms.txt`:
```
Similar Pairs (Small L2 Distance):
==================================================
Pair 1: Image 23 <-> Image 67, L2 Distance: 4.2341
Pair 2: Image 45 <-> Image 89, L2 Distance: 4.5782
Pair 3: Image 12 <-> Image 56, L2 Distance: 4.8923
```

### Dissimilar Pairs
Images whose latent vectors are far in L2 distance should look visually different (e.g., different colors, different structures).

Example output:
```
Dissimilar Pairs (Large L2 Distance):
==================================================
Pair 1: Image 5 <-> Image 91, L2 Distance: 18.3456
Pair 2: Image 34 <-> Image 78, L2 Distance: 17.9823
Pair 3: Image 19 <-> Image 82, L2 Distance: 17.7234
```

## Quality Assurance

The generated images should have **similar quality** to training outputs because:
1. Uses the same trained generator
2. Samples from the same latent distribution (Normal(0, 1))
3. Generator is in eval mode (consistent BatchNorm behavior)
4. Same image post-processing ([-1,1] → [0,1])

## Troubleshooting

**Issue:** `FileNotFoundError: generator.pkl`
- **Solution:** Run `python gan_complete.py` first to train and save the model

**Issue:** Generated images look noisy
- **Solution:** Model may need more training epochs. Try increasing `epochs` in Config

**Issue:** Out of memory
- **Solution:** In `hw4_generation.py`, reduce `batch_size` in the generation loop

**Issue:** Images don't match training quality
- **Solution:** Ensure `z_dim=100` matches training configuration

## Parameters

### Key Configuration in gan_complete.py
- `z_dim: int = 100` - Latent dimension
- `batch_size: int = 32` - Training batch size
- `epochs: int = 40` - Number of training epochs
- `lambda_gp: float = 10.0` - Gradient penalty weight

### Key Parameters in hw4_generation.py
- `images_per_category: int = 10` - Images generated per category
- `z_dim: int = 100` - Must match training
- `seed: int = 42` - For reproducible generation

## Notes

- The GAN is **unconditional** - it doesn't know flower categories explicitly
- "10 images per category" means generating diverse images that collectively represent the dataset variety
- Total of 1,020 images ensures comprehensive coverage of the learned distribution
- L2 distances are computed in the **latent space** (not pixel space)
