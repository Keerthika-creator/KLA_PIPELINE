import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_multiscale import MultiScaleRestorationCNN


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "MODEL_CHAMPION_28_2983.pth"
)

DEFAULT_BATCH_SIZE = 4


# ============================================================
# MODEL
# ============================================================

def load_model():

    model = MultiScaleRestorationCNN().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ============================================================
# 4-WAY TTA
# ============================================================

def tta_inference(model, bicubic):

    predictions = []

    for mode in range(4):

        if mode == 0:

            augmented = bicubic

        elif mode == 1:

            augmented = torch.flip(
                bicubic,
                dims=[3]
            )

        elif mode == 2:

            augmented = torch.flip(
                bicubic,
                dims=[2]
            )

        else:

            augmented = torch.flip(
                bicubic,
                dims=[2, 3]
            )

        prediction = model(
            augmented
        )

        if mode == 1:

            prediction = torch.flip(
                prediction,
                dims=[3]
            )

        elif mode == 2:

            prediction = torch.flip(
                prediction,
                dims=[2]
            )

        elif mode == 3:

            prediction = torch.flip(
                prediction,
                dims=[2, 3]
            )

        predictions.append(prediction)

    return torch.stack(
        predictions,
        dim=0
    ).mean(dim=0)


# ============================================================
# PROCESS BATCH
# ============================================================

def process_batch(
    model,
    paths,
    output_dir,
    save_npy
):

    arrays = []

    shapes = []

    for path in paths:

        lr = np.load(
            path
        ).astype(
            np.float32
        )

        if lr.ndim != 2:

            raise ValueError(
                f"{path.name}: expected a 2-D grayscale "
                f"array, got shape {lr.shape}"
            )

        arrays.append(lr)
        shapes.append(lr.shape)

    # --------------------------------------------------------
    # Current KLA task uses same-size images.
    # Keep batching safe by requiring equal dimensions.
    # --------------------------------------------------------

    if len(set(shapes)) != 1:

        raise ValueError(
            "Images in the same batch have different "
            f"dimensions: {shapes}"
        )

    lr_np = np.stack(
        arrays,
        axis=0
    )

    lr_tensor = torch.from_numpy(
        lr_np
    ).unsqueeze(1).to(
        DEVICE,
        non_blocking=True
    )

    # --------------------------------------------------------
    # Bicubic upsampling
    #
    # IMPORTANT:
    # NO [0,1] CLAMP HERE.
    #
    # This matches training/validation and preserves the
    # intentional out-of-range signal in NoisyLR.
    # --------------------------------------------------------

    bicubic = F.interpolate(
        lr_tensor,
        scale_factor=2,
        mode="bicubic",
        align_corners=False
    )

    # --------------------------------------------------------
    # 4-way TTA
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = tta_inference(
            model,
            bicubic
        )

    # --------------------------------------------------------
    # Final output clamp ONLY
    # --------------------------------------------------------

    prediction = torch.clamp(
        prediction,
        0.0,
        1.0
    )

    prediction = (
        prediction[:, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    for i, path in enumerate(paths):

        stem = path.stem

        output = prediction[i]

        if save_npy:

            np.save(
                output_dir / f"{stem}.npy",
                output
            )


    return shapes, prediction


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "KLA Multi-Scale Restoration CNN "
            "+ 4-Way TTA directory inference"
        )
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing degraded .npy files"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for restored outputs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size"
    )

    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Save float32 restored .npy files"
    )

    args = parser.parse_args()

    if not args.save_npy:

        # Safe default: save float32 NPY.
        args.save_npy = True

    if args.batch_size < 1:

        raise ValueError(
            "batch_size must be >= 1"
        )

    input_dir = Path(
        args.input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    if not input_dir.is_dir():

        raise ValueError(
            f"Input directory does not exist: "
            f"{input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    input_files = sorted(
        input_dir.glob("*.npy")
    )

    if not input_files:

        raise ValueError(
            f"No .npy files found in {input_dir}"
        )

    print("=" * 70)
    print("KLA FINAL IMAGE RESTORATION")
    print("MULTI-SCALE CNN + 4-WAY TTA")
    print("=" * 70)

    print(
        "Device      :",
        DEVICE
    )

    if DEVICE.type == "cuda":

        print(
            "GPU         :",
            torch.cuda.get_device_name(0)
        )

    model, checkpoint = load_model()

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Parameters  :",
        f"{params:,}"
    )

    print(
        "Checkpoint PSNR :",
        f"{checkpoint['val_psnr']:.4f} dB"
    )

    print(
        "Checkpoint SSIM :",
        f"{checkpoint['val_ssim']:.6f}"
    )

    print(
        "Input files :",
        len(input_files)
    )

    print(
        "Batch size  :",
        args.batch_size
    )

    print(
        "Save NPY    :",
        args.save_npy
    )

    print()

    start_time = time.perf_counter()

    processed = 0
    first_shape = None
    output_shape = None

    for start in range(
        0,
        len(input_files),
        args.batch_size
    ):

        batch_paths = input_files[
            start:start + args.batch_size
        ]

        shapes, predictions = process_batch(
            model=model,
            paths=batch_paths,
            output_dir=output_dir,
            save_npy=args.save_npy
        )

        if first_shape is None:

            first_shape = shapes[0]

            output_shape = predictions[0].shape

        processed += len(
            batch_paths
        )

        print(
            f"Processed {processed}/{len(input_files)}"
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    per_image = (
        elapsed / processed
    )

    throughput = (
        processed / elapsed
    )

    print()
    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)

    print(
        "Input shape example  :",
        first_shape
    )

    print(
        "Output shape example :",
        output_shape
    )

    print(
        "Images processed     :",
        processed
    )

    print(
        "End-to-end runtime   :",
        f"{elapsed:.4f} s"
    )

    print(
        "End-to-end latency   :",
        f"{per_image * 1000:.3f} ms/image"
    )

    print(
        "End-to-end throughput:",
        f"{throughput:.2f} images/sec"
    )

    print(
        "Output directory     :",
        output_dir
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
