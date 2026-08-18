# AI-Based Restoration of Degraded Images for Semiconductor Inspection
**SEMICON India / KLA Hackathon 2026**

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Problem Statement & Constraints](#-problem-statement--constraints)
3. [Repository Structure](#-repository-structure)
4. [Environment Setup & Dependencies](#-environment-setup--dependencies)
5. [Training Pipeline](#-training-pipeline)
6. [Inference Pipeline](#-inference-pipeline)
7. [Model Architecture & Design Rationale](#-model-architecture--design-rationale)
8. [Experimental Results & Validation](#-experimental-results--validation)
9. [Failure Analysis & Limitations](#-failure-analysis--limitations)

---

## 🔍 Overview
Inspection and imaging systems in semiconductor manufacturing rarely capture perfectly clean images. Noise, resolution loss, and distortion hide critical microscopic details, impeding downstream computer vision and inspection algorithms. 

This project implements a high-throughput, robust, end-to-end deep learning restoration pipeline designed to reverse combined degradations (**speckle noise, additive Gaussian noise, and spatial downsampling**), recovering high-fidelity ground-truth (GT) resolution images without hallucinating structural details.

---

## ⚙️ Problem Statement & Constraints
* **Input**: Degraded Low-Resolution images (`NoisyLR`), which may contain pixel values slightly outside the standard $[0, 1]$ range[cite: 15].
* **Output**: Restored high-resolution clean images normalized to $[0, 1]$ matching ground-truth dimensions[cite: 15].
* **Degradations Handled**: Multi-stage/arbitrary-order Speckle Noise, Additive Gaussian Noise, and Downsampling[cite: 15].
* **Hardware Target**: Optimized for NVIDIA GPUs (benchmarked for high throughput on NVIDIA H100 architecture)[cite: 15].

---

## 📁 Repository Structure
```text
repository/
├── README.md               # Complete documentation & execution guide
├── requirements.txt        # Pinned Python dependencies
├── train.py                # Reproducible training script
├── inference.py            # Standalone evaluation & inference script
├── configs/
│   └── train_config.yaml   # Hyperparameters, paths, and training settings
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Custom DataLoader handling out-of-range [0,1] inputs
│   ├── models.py           # Network architecture definitions
│   └── losses.py           # Combined Charbonnier + LPIPS / SSIM loss functions
├── weights/
│   └── best_checkpoint.pth # Final trained model weights
├── results/                # Generated visual samples and evaluation metrics log
└── solution_presentation.pptx # Phase 1 submission slide deck
