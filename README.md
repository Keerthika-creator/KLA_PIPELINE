# AI-Based Restoration of Degraded Images for Semiconductor Inspection

### SEMICON India × KLA Hackathon 2026

> **A reproducible GPU-accelerated image restoration pipeline for recovering high-fidelity semiconductor inspection images from noisy and low-resolution observations.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#environment-setup)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](#environment-setup)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-green.svg)](#inference)
[![Task](https://img.shields.io/badge/Task-Image%20Restoration-purple.svg)](#problem-statement)

---

## 1. Overview

Semiconductor inspection systems operate on images where small structural details can determine whether a feature is correctly interpreted by downstream inspection and computer-vision systems.

In realistic acquisition pipelines, image quality can be degraded by:

* Speckle noise
* Additive Gaussian noise
* Spatial downsampling / resolution loss

The objective of this project is to recover the clean, high-resolution image from the degraded observation while **preserving genuine structure and minimizing hallucinated detail**.

This repository provides a complete restoration pipeline covering:

**Degraded NoisyLR → Preprocessing → Restoration Network → Post-processing → Restored GT-resolution Image**

The implementation is designed around the KLA evaluation requirements: restoration quality, generalization to unfamiliar image content, reproducibility and complete end-to-end GPU inference.

---

# 2. Problem Statement

The KLA challenge provides paired training data:

* **GT:** clean ground-truth image
* **NoisyLR:** degraded low-resolution image

The benchmark degradation mechanisms are:

1. Speckle noise
2. Additive Gaussian noise
3. Downsampling

The degradation operations may occur in different orders, and the model is therefore trained to learn a robust inverse transformation rather than relying on a fixed degradation ordering.

### Input

Degraded low-resolution image (`NoisyLR`).

Important: the input may contain values slightly outside `[0,1]`.

### Output

Restored image at the expected ground-truth resolution with values normalized to `[0,1]`.

### Target

Maximize:

* Pixel fidelity
* Structural similarity
* Perceptual quality
* Generalization to unseen image content
* End-to-end inference efficiency

The official evaluation combines PSNR, SSIM and LPIPS and also evaluates complete pipeline throughput on an NVIDIA H100 GPU.

---

# 3. Solution at a Glance

```text
                 ┌───────────────────────┐
                 │   Degraded NoisyLR     │
                 │  Speckle + Gaussian   │
                 │     + Downsampling     │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Input Preprocessing  │
                 │ Range / Tensor Format  │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Restoration Model    │
                 │                         │
                 │ Feature Extraction      │
                 │       ↓                 │
                 │ Multi-scale Restoration│
                 │       ↓                 │
                 │ Detail Reconstruction   │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    Post-processing     │
                 │   Range enforcement   │
                 │      [0,1]             │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Restored GT Resolution │
                 └───────────────────────┘
```

---

# 4. Why This Design?

The restoration problem is not simply denoising.

The model must simultaneously address:

| Challenge            | Required behaviour                                |
| -------------------- | ------------------------------------------------- |
| Speckle noise        | Suppress multiplicative/local intensity variation |
| Gaussian noise       | Recover signal buried under additive noise        |
| Downsampling         | Reconstruct lost spatial information              |
| Combined degradation | Handle multiple degradation mechanisms jointly    |
| Unfamiliar content   | Avoid overfitting to training structures          |
| Fine details         | Preserve genuine inspection-relevant structures   |
| Inference            | Maintain practical end-to-end throughput          |

The design therefore prioritizes **fidelity + structural preservation + perceptual quality**, rather than optimizing a single metric.

---

# 5. Model Architecture

## Architecture Summary

**Model:** `[INSERT EXACT MODEL NAME]`

**Core architecture:** `[CNN / Transformer / Hybrid / Custom]`

**Parameters:** `[X] M`

**Input resolution:** `[H × W]`

**Output resolution:** `[H_GT × W_GT]`

**Trainable parameters:** `[X]`

### High-level architecture

```text
Input
  │
  ▼
Feature Projection
  │
  ▼
Multi-scale Feature Extraction
  │
  ├──────────────┐
  │              │
  ▼              ▼
Local Detail     Contextual
Restoration      Representation
  │              │
  └───────┬──────┘
          ▼
Feature Fusion
          │
          ▼
Resolution Reconstruction
          │
          ▼
Residual Refinement
          │
          ▼
Restored Image
```

### Design rationale

The architecture is designed to separate the restoration problem into:

1. **Feature extraction** – identify useful image structures from degraded observations.
2. **Noise suppression** – remove degradation while retaining signal.
3. **Multi-scale/context modelling** – capture both local features and larger structural relationships.
4. **Resolution reconstruction** – recover the expected GT spatial resolution.
5. **Residual refinement** – refine reconstructed details while minimizing unnecessary modifications.

> The exact architecture and implementation are defined in `src/models.py`.

---

# 6. Loss Function

The training objective combines complementary restoration objectives rather than relying on a single pixel loss.

### Composite objective

```text
L_total =
    λ1 · L_Charbonnier
  + λ2 · L_SSIM
  + λ3 · L_LPIPS
```

### Charbonnier Loss

Encourages robust pixel-level reconstruction while being less sensitive to outliers than standard L2 loss.

### SSIM Loss

Encourages structural consistency between restored and ground-truth images.

### LPIPS Loss

Provides a perceptual constraint that helps prevent visually implausible restoration.

### Why a composite loss?

PSNR-oriented optimization alone can produce overly smooth results.

Perceptual optimization alone can introduce visually plausible but incorrect structures.

The combined objective therefore balances:

**Pixel fidelity + structural fidelity + perceptual similarity**

---

# 7. Data Pipeline

The training pipeline uses paired:

```text
NoisyLR → GT
```

samples.

The augmentation strategy is designed around the official benchmark degradation mechanisms.

### Supported degradation mechanisms

* Speckle noise
* Additive Gaussian noise
* Spatial downsampling

### Important input handling

`NoisyLR` values are not assumed to be strictly inside `[0,1]`.

The preprocessing pipeline explicitly handles the input range before model inference.

Ground-truth targets are represented in `[0,1]`.

---

# 8. Training Pipeline

```bash
python train.py --config configs/train_config.yaml
```

Training configuration controls:

* Dataset paths
* Validation split
* Batch size
* Learning rate
* Number of epochs
* Loss weights
* Augmentation
* Checkpointing
* Random seed

### Reproducibility

Every experiment should record:

* Random seed
* Configuration
* Hyperparameters
* Checkpoint
* Validation metrics
* Training environment

The final submitted checkpoint is:

```text
weights/best_checkpoint.pth
```

---

# 9. Inference

The submission provides a standalone inference script accepting input and output directories.

```bash
python inference.py \
    --input_dir ./data/test \
    --output_dir ./results/predictions
```

The script:

1. Loads every input image.
2. Performs preprocessing.
3. Runs GPU inference.
4. Performs post-processing.
5. Enforces the required output range.
6. Saves restored images using the required output format.

No source-code modification or notebook editing should be required.

---

# 10. Reproducible Environment

Create the environment:

```bash
python -m venv .venv
```

Activate it:

### Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

# 11. Evaluation Protocol

The validation set is separated from training data to prevent model-selection leakage.

The final model is evaluated using:

### Restoration Quality

* **PSNR ↑**
* **SSIM ↑**
* **LPIPS ↓**

### Efficiency

* End-to-end runtime
* Throughput
* Batch size
* GPU memory usage

### Reproducibility

* Fixed configuration
* Recorded random seed
* Pinned dependencies
* Reproducible checkpoint

---

# 12. Results

> Replace the following values with measured results from the final checkpoint. Do not use estimated values.

| Model              |      PSNR ↑ |       SSIM ↑ |      LPIPS ↓ |     Runtime | Parameters |
| ------------------ | ----------: | -----------: | -----------: | ----------: | ---------: |
| Baseline           |   `[XX.XX]` |   `[0.XXXX]` |   `[0.XXXX]` |   `[XX ms]` |    `[X M]` |
| **Proposed Model** | **[XX.XX]** | **[0.XXXX]** | **[0.XXXX]** | **[XX ms]** |  **[X M]** |

### Improvement over baseline

```text
PSNR improvement  : [X.XX] dB
SSIM improvement  : [X.XX]
LPIPS reduction   : [X.XX] %
Runtime change    : [X.XX] %
```

---

# 13. Generalization Evaluation

Because the hidden evaluation includes both familiar and unfamiliar image content, model selection is not based only on training-set performance.

Validation analysis includes:

### In-distribution samples

Images similar to structures observed during training.

### Out-of-distribution samples

Images containing previously unseen structural patterns.

The objective is to verify that the network restores the underlying image rather than memorizing training-specific structures.

---

# 14. Visual Results

Representative results should be organized as:

```text
GT              NoisyLR          Restored
│                  │                │
▼                  ▼                ▼
[image]          [image]          [image]
```

Visual inspection focuses on:

* Fine structural preservation
* Noise removal
* Edge preservation
* Texture reconstruction
* Absence of artificial structures
* Absence of excessive smoothing

---

# 15. Failure Analysis

A restoration model should not be evaluated only on successful examples.

Observed failure cases should be documented.

### Failure Case 1

**Condition:** `[describe]`

**Observed behaviour:** `[describe]`

**Likely cause:** `[describe]`

**Impact:** `[describe]`

### Failure Case 2

**Condition:** `[describe]`

**Observed behaviour:** `[describe]`

**Likely cause:** `[describe]`

**Future improvement:** `[describe]`

Failure analysis is included to distinguish genuine model limitations from dataset or preprocessing issues.

---

# 16. Runtime Benchmark

End-to-end timing includes:

```text
Disk I/O
   ↓
Preprocessing
   ↓
CPU → GPU transfer
   ↓
Model inference
   ↓
GPU → CPU transfer
   ↓
Post-processing
   ↓
Image saving
```

### Benchmark configuration

| Parameter          | Value                        |
| ------------------ | ---------------------------- |
| GPU                | `[NVIDIA H100 / actual GPU]` |
| Batch size         | `[X]`                        |
| Input resolution   | `[H × W]`                    |
| Parameters         | `[X M]`                      |
| End-to-end latency | `[X ms/image]`               |
| Throughput         | `[X images/sec]`             |
| CUDA               | `[version]`                  |
| PyTorch            | `[version]`                  |

This measures the **complete inference pipeline**, rather than reporting neural-network forward-pass latency alone.

---

# 17. Repository Structure

```text
repository/
│
├── README.md
├── requirements.txt
├── train.py
├── inference.py
│
├── configs/
│   └── train_config.yaml
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── models.py
│   └── losses.py
│
├── weights/
│   └── best_checkpoint.pth
│
├── results/
│   ├── metrics/
│   ├── visual_results/
│   └── failure_cases/
│
└── solution_presentation.pptx
```

---

# 18. External Resources

Any external dataset, pretrained model or publicly available implementation used in this project must be disclosed here.

| Resource     | Purpose     | License     | Link    |
| ------------ | ----------- | ----------- | ------- |
| `[Resource]` | `[Purpose]` | `[License]` | `[URL]` |

If no external resource was used:

```text
No external training dataset or pretrained model was used.
```

---

# 19. Reproducibility Checklist

Before submission, the repository is verified for:

* [ ] Clean-environment installation
* [ ] Training command executes without source modification
* [ ] Inference accepts input/output directory arguments
* [ ] Final checkpoint loads successfully
* [ ] Required dependencies are specified
* [ ] PSNR reported
* [ ] SSIM reported
* [ ] LPIPS reported
* [ ] Baseline comparison included
* [ ] Visual examples included
* [ ] Failure cases included
* [ ] Runtime measured end-to-end
* [ ] Hardware and software versions documented
* [ ] Random seed recorded
* [ ] External resources disclosed
* [ ] Repository tested before submission

---

# 20. Limitations

The current system is optimized specifically for the benchmark degradation mechanisms:

* Speckle noise
* Additive Gaussian noise
* Downsampling

Performance outside these degradation families has not been claimed unless explicitly evaluated.

The model may also encounter difficult cases where information destroyed by severe downsampling cannot be uniquely recovered from the observation.

For semiconductor inspection applications, restored imagery should therefore be treated as an image-reconstruction stage and validated against downstream inspection requirements before production deployment.

---

# 21. Future Work

Potential extensions include:

1. Hardware-aware model optimization.
2. Mixed-precision inference.
3. TensorRT deployment.
4. Further end-to-end latency reduction.
5. More robust uncertainty estimation.
6. Joint restoration + downstream defect-detection optimization.
7. Evaluation across additional semiconductor inspection datasets.

---

# 22. Key Takeaway

The goal of this project is not simply to make degraded images look better.

The objective is to build a **reproducible, quantitatively evaluated and computationally efficient restoration pipeline** capable of recovering useful structural information while minimizing artificial detail.

The final system is evaluated across three dimensions:

```text
             ┌───────────────────┐
             │ Restoration       │
             │ Quality            │
             │ PSNR / SSIM / LPIPS│
             └─────────┬─────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌──────────────────┐       ┌──────────────────┐
│ Generalization   │       │ Engineering      │
│ ID + OOD Content │       │ Runtime + Repro. │
└──────────────────┘       └──────────────────┘
```

**Quality. Generalization. Efficiency. Reproducibility.**

---

## Citation

This project was developed for the **SEMICON India / KLA Hackathon 2026 — AI-Based Restoration of Degraded Images for Semiconductor Inspection**.
