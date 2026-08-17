# KLA Image Restoration

## Overview

A lightweight GPU-accelerated image restoration pipeline for converting
degraded 128x128 low-resolution images into 256x256 restored images.

The final system combines a lightweight Multi-Scale Restoration CNN
with four-way test-time augmentation (TTA).

## Final Pipeline

Noisy LR 128x128
        |
        v
Bicubic 2x Upsampling
        |
        v
256x256 Bicubic Image
        |
        |  No clipping before CNN
        v
Multi-Scale Restoration CNN
        |
        +---- Normal
        +---- Horizontal Flip
        +---- Vertical Flip
        +---- Horizontal + Vertical Flip
                    |
                    v
             Inverse Transform
                    |
                    v
             Prediction Average
                    |
                    v
             Final [0,1] Clamp
                    |
              +-----+-----+
              |           |
              v           v
         Float32 NPY     PNG
                        uint8

## Model

Multi-Scale Restoration CNN

Architecture:

- Input channels: 1
- Feature channels: 32
- Multi-scale residual blocks: 5
- Trainable parameters: 251,457
- Checkpoint size: approximately 2.96 MB

The network uses fine-scale convolutional processing together with a
dilated-convolution branch to capture larger receptive fields.
Residual connections are used for stable image restoration.

## Environment & Reproducibility

Tested environment:

- OS: Ubuntu 24.04 LTS
- Python: 3.12
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- Batch size for final evaluation: 4

### Environment Setup

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install the required dependencies:

    pip install -r requirements.txt

### Training

The training script is provided as `train_multiscale.py`.

Training expects the official KLA dataset and split files:

    splits/train.csv
    splits/val.csv

Each split CSV uses the following fields:

    filename,gt_path,lr_path

The training configuration includes:

- Random seed: 42
- Optimizer: Adam
- Loss: L1
- Batch size: 8
- Epochs: 20
- Learning rate: 1e-3

### Final Inference

Run directory-based restoration using:

    python final_inference.py \
        --input_dir <INPUT_DIR> \
        --output_dir <OUTPUT_DIR> \
        --batch_size 4 \
        --save_npy

Input:
- Grayscale degraded `.npy` images
- Expected validation input resolution: 128x128

Output:
- One float32 `.npy` restoration per input image
- Expected validation output resolution: 256x256
- Output values are clipped to [0,1]

### PNG Visualization

PNG generation is intentionally separated from restoration inference.

Run:

    python convert_npy_to_png.py \
        --input_dir <RESTORED_NPY_DIR> \
        --output_dir <PNG_OUTPUT_DIR>

The conversion applies:

    clamp(x, 0, 1) -> round(x * 255) -> uint8

PNG files are intended for visualization and qualitative inspection.
The float32 NPY files preserve the full numerical restoration output.

## Validation Results

Validation set: 320 images.

| Method | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Bicubic | 23.278916 dB | 0.555406 | 0.431230 |
| Multi-Scale CNN | 28.299785 dB | 0.761238 | 0.284471 |
| Multi-Scale CNN + 4-Way TTA | 28.382919 dB | 0.763858 | 0.291823 |

Higher PSNR and SSIM indicate better restoration.
Lower LPIPS indicates better perceptual similarity.

## TTA Improvement

Compared with the normal Multi-Scale CNN:

- PSNR improvement: +0.083134 dB
- SSIM improvement: +0.002620

## Pre-Model Clipping Experiment

An A/B experiment was performed because degraded images contain
intentional signal outside the [0,1] range.

Bicubic pixels outside [0,1]:

2.8675%

| Pipeline | PSNR | SSIM |
|---|---:|---:|
| Bicubic -> Clamp -> CNN + TTA | 28.102981 | 0.762883 |
| Bicubic -> CNN + TTA | 28.381920 | 0.763841 |

Removing the pre-model clamp improves:

- PSNR: +0.278939 dB
- SSIM: +0.000958

Therefore, the final pipeline does not clamp the bicubic image before
CNN inference.

The restored prediction is clipped to [0,1] only after model inference.

## Output Formats

### Float32 NPY

The primary restoration output is saved as:

- 256x256
- float32
- values in [0,1]

The float32 NPY representation preserves the numerical restoration
result without 8-bit quantization.

### PNG

PNG generation is intentionally separated from restoration inference
and is provided for visualization and qualitative inspection.

PNG output is:

- 256x256
- grayscale
- uint8

Conversion:

    clamp(x, 0, 1) -> round(x * 255) -> uint8

A sample quantization test on sample 000000 showed:

Float32 NPY:
- PSNR: 31.635802 dB
- SSIM: 0.894125

PNG converted back to float:
- PSNR: 31.626228 dB
- SSIM: 0.893419

Quantization effect:
- PSNR: -0.009574 dB
- SSIM: -0.000705

## Output Generation Workflow

The final restoration pipeline and PNG visualization pipeline are
separate.

### Step 1: Run restoration inference

    python final_inference.py \
        --input_dir path/to/degraded_npy \
        --output_dir path/to/restored_npy \
        --batch_size 4 \
        --save_npy

This produces one float32 `.npy` restoration for each input image.

The inference script performs:

- Bicubic 2x upsampling
- No pre-model clipping
- Multi-Scale Restoration CNN inference
- 4-way test-time augmentation
- Prediction averaging
- Final [0,1] output clipping

### Step 2: Generate PNG visualization files

    python convert_npy_to_png.py \
        --input_dir path/to/restored_npy \
        --output_dir path/to/png_results

The converter operates on the restored `.npy` files and does not modify
the numerical inference output.

## Inference

Directory-based inference is supported:

    python final_inference.py \
        --input_dir path/to/input_directory \
        --output_dir path/to/output_directory \
        --batch_size 4 \
        --save_npy

The final inference script produces float32 NPY files only.

Each output preserves the input filename stem.

PNG files are generated separately with `convert_npy_to_png.py`.

## GPU Performance

Hardware:

NVIDIA GeForce RTX 4050 Laptop GPU

End-to-end benchmark:

- Validation images: 320
- Batch size: 4
- Total runtime: 23.3124 s
- End-to-end latency: 72.851 ms/image
- End-to-end throughput: 13.73 images/sec

The end-to-end measurement includes file loading, preprocessing,
CPU-to-GPU transfer, model inference, GPU-to-CPU transfer,
post-processing and output saving.

Model-only benchmark:

- Latency: 12.522 ms/image
- Throughput: 79.86 images/sec

Four-way TTA model-only benchmark:

- Latency: 50.225 ms/image
- Throughput: 19.91 images/sec

## Batch Size Selection

End-to-end throughput was empirically evaluated using batch sizes 4, 8,
16, and 32 on the NVIDIA GeForce RTX 4050 Laptop GPU.

Measured throughput:

- Batch 4: 13.73 images/sec
- Batch 8: 13.16 images/sec
- Batch 16: 13.07 images/sec
- Batch 32: 12.94 images/sec

Batch size 4 was therefore retained for the final configuration.
Increasing the batch size did not improve end-to-end throughput, indicating
that factors outside the raw model computation contribute significantly
to total pipeline runtime.

## Failure-Case Analysis

Sample 001289 is a difficult restoration case.

Results:

| Method | PSNR | SSIM |
|---|---:|---:|
| Bicubic | 23.119561 | 0.754980 |
| CNN | 23.679826 | 0.755093 |
| TTA | 23.702488 | 0.756310 |

The improvement is modest because the remaining reconstruction error
is concentrated in image details that are difficult to recover
reliably from the degraded input.

This case demonstrates that the model does not produce uniformly
large improvements on every image.

## Selected Visual Results

The submission contains representative visual comparisons:

- results/BEST_IMPROVEMENT_003116.png
- results/MEDIAN_001111.png
- results/DIFFICULT_001289.png

## Configuration

config.yaml contains the architecture, preprocessing, inference,
post-processing and validation configuration.

## External Resources and License Disclosure

The restoration model was trained using the resources provided for
the KLA image restoration challenge.

No external pretrained image-restoration model weights are used by
the final restoration model.

The inference implementation uses open-source libraries including
NumPy, PyTorch, Pillow, scikit-image and LPIPS. Their respective
licenses remain applicable.

## Submission Contents

    final_inference.py
    train_multiscale.py
    model/
        MODEL_CHAMPION_28_2983.pth
    config.yaml
    metrics.txt
    requirements.txt
    results/
        BEST_IMPROVEMENT_003116.png
        MEDIAN_001111.png
        DIFFICULT_001289.png
    README.md

## Reproducibility

The supplied checkpoint and inference code are sufficient to reproduce
the final restoration pipeline.

Final configuration:

- Multi-Scale Restoration CNN
- 32 feature channels
- 5 residual blocks
- Bicubic 2x upsampling
- No pre-model clipping
- Four-way TTA
- Final [0,1] output clipping
- Batch size 4
