# Mid-Semester Report

## 1. Understanding of the Problem Statement

This project solves image inpainting using diffusion models.

Given:
- an input image with missing or corrupted region,
- a binary mask where white indicates missing region and black indicates known region,

the goal is to generate realistic content inside the masked region while preserving consistency with unmasked context (structure, texture, color, and semantics).

### Project Objective
Build a single, reproducible inpainting baseline using Stable Diffusion Inpainting + Classifier-Free Guidance + DDIM sampling.

Scope:
- Use pretrained model (no training required)
- Small dataset (10–30 images from picsum.photos)
- Two mask types: center (fixed rectangle) and irregular (random strokes)
- Evaluate on PSNR, SSIM, and visual quality

### Constraints
- Limited compute (CPU or single GPU)
- No fine-tuning or multi-level complexity
- Reproducible pipeline with clear outputs

## 2. Literature Survey

### Reference 1: RePaint
- Inference-time resampling
- No retraining required

### Reference 2: Diffusion Customization
- Mask blending during sampling

## 3. Approach

Stable Diffusion Inpainting + CFG + DDIM

## 4. Dataset Selection

Picsum dataset with center + irregular masks

## 5. Evaluation Metrics

PSNR, SSIM, Visual Quality

## 6. Diffusion Inpainting Taxonomy

```
Diffusion Inpainting
│
├── 1. Conditioning Strategy
├── 2. Diffusion Space
├── 3. Mask Representation
├── 4. Constraint Enforcement
├── 5. Region Interaction
├── 6. Guidance Mechanism
├── 7. Sampling Strategy
├── 8. Model Type
└── 9. Training Objective
```

## 7. Paper Comparisons

| Dimension | RePaint | Stable Diffusion | ControlNet |
|----------|--------|-----------------|------------|
| Conditioning | Inference | Training | Training + control |
| Space | Pixel | Latent | Latent |
| Guidance | None | Text | Structural |
| Strength | Flexible | Practical | Precise |

### Key Insight
Sampling vs Conditioning tradeoff defines most methods.
