import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3

NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "multiscale_best.pth"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class RestorationDataset(Dataset):

    def __init__(self, csv_file):

        self.samples = []

        with open(csv_file, "r") as f:

            reader = csv.DictReader(f)

            for row in reader:

                self.samples.append({
                    "gt": row["gt_path"],
                    "lr": row["lr_path"],
                    "filename": row["filename"]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        gt = np.load(
            sample["gt"]
        ).astype(np.float32)

        lr = np.load(
            sample["lr"]
        ).astype(np.float32)

        gt = torch.from_numpy(gt).unsqueeze(0)

        lr = torch.from_numpy(lr).unsqueeze(0)

        return lr, gt, sample["filename"]


# ============================================================
# MULTI-SCALE RESIDUAL BLOCK
# ============================================================

class MultiScaleResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        # Fine-scale branch
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )
        )

        # Larger receptive-field branch
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2
            )
        )

        # Fuse both scales
        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        fine = self.branch_3x3(x)

        coarse = self.branch_dilated(x)

        combined = torch.cat(
            [fine, coarse],
            dim=1
        )

        out = self.fusion(combined)

        # Residual connection
        return x + out


# ============================================================
# MULTI-SCALE RESTORATION CNN
# ============================================================

class MultiScaleRestorationCNN(nn.Module):

    def __init__(
        self,
        channels=32,
        num_blocks=5
    ):

        super().__init__()

        # Initial feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(
                1,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        # Multi-scale feature processing
        self.blocks = nn.Sequential(
            *[
                MultiScaleResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                1,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, bicubic):

        features = self.head(bicubic)

        features = self.blocks(features)

        residual = self.refinement(features)

        # Global residual learning
        output = bicubic + residual

        return output


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(model, loader):

    model.eval()

    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():

        for lr, gt, _ in loader:

            lr = lr.to(DEVICE)
            gt = gt.to(DEVICE)

            # Bicubic upsampling
            bicubic = F.interpolate(
                lr,
                scale_factor=2,
                mode="bicubic",
                align_corners=False
            )

            # CNN restoration
            prediction = model(bicubic)

            prediction_np = (
                prediction[:, 0]
                .cpu()
                .numpy()
            )

            gt_np = (
                gt[:, 0]
                .cpu()
                .numpy()
            )

            for pred, target in zip(
                prediction_np,
                gt_np
            ):

                psnr = peak_signal_noise_ratio(
                    target,
                    pred,
                    data_range=1.0
                )

                ssim = structural_similarity(
                    target,
                    pred,
                    data_range=1.0
                )

                psnr_scores.append(psnr)
                ssim_scores.append(ssim)

    return (
        float(np.mean(psnr_scores)),
        float(np.mean(ssim_scores))
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MULTI-SCALE RESIDUAL CNN")
    print("=" * 60)

    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = RestorationDataset(
        "splits/train.csv"
    )

    val_dataset = RestorationDataset(
        "splits/val.csv"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print()
    print(
        "Training samples  :",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MultiScaleRestorationCNN().to(DEVICE)

    params = count_parameters(model)

    print(
        "Trainable parameters:",
        f"{params:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.L1Loss()

    best_psnr = -float("inf")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0

        for lr, gt, _ in train_loader:

            lr = lr.to(
                DEVICE,
                non_blocking=True
            )

            gt = gt.to(
                DEVICE,
                non_blocking=True
            )

            # ----------------------------------------------
            # Bicubic baseline
            # ----------------------------------------------

            bicubic = F.interpolate(
                lr,
                scale_factor=2,
                mode="bicubic",
                align_corners=False
            )

            # ----------------------------------------------
            # Forward
            # ----------------------------------------------

            prediction = model(
                bicubic
            )

            # ----------------------------------------------
            # L1 loss
            # ----------------------------------------------

            loss = criterion(
                prediction,
                gt
            )

            # ----------------------------------------------
            # Backpropagation
            # ----------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * lr.size(0)
            )

        train_loss = (
            running_loss
            / len(train_dataset)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_psnr, val_ssim = validate(
            model,
            val_loader
        )

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Loss: {train_loss:.6f} "
            f"PSNR: {val_psnr:.4f} dB "
            f"SSIM: {val_ssim:.6f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_psnr > best_psnr:

            best_psnr = val_psnr

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim
                },
                CHECKPOINT_PATH
            )

            print(
                f"  ✓ Saved best model "
                f"(PSNR {val_psnr:.4f} dB)"
            )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation PSNR : "
        f"{best_psnr:.4f} dB"
    )

    print(
        "Checkpoint            :",
        CHECKPOINT_PATH
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
