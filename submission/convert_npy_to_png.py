import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def convert_file(input_path, output_path):
    arr = np.load(input_path).astype(np.float32)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale array, got shape {arr.shape}"
        )

    if not np.isfinite(arr).all():
        raise ValueError(
            f"Non-finite values found in {input_path}"
        )

    # Explicit visualization mapping:
    # clamp to [0,1] -> multiply by 255 -> round -> uint8
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.rint(arr * 255.0).astype(np.uint8)

    Image.fromarray(arr, mode="L").save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert restored float32 NPY images to PNG."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing restored .npy files"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for PNG visualization outputs"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise ValueError(
            f"Input directory does not exist: {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(input_dir.glob("*.npy"))

    if not files:
        raise ValueError(
            f"No .npy files found in {input_dir}"
        )

    print("=" * 65)
    print("KLA NPY -> PNG CONVERTER")
    print("=" * 65)
    print("Input files :", len(files))
    print("Output dir  :", output_dir)

    for i, npy_path in enumerate(files, 1):

        png_path = output_dir / (
            npy_path.stem + ".png"
        )

        convert_file(
            npy_path,
            png_path
        )

        print(
            f"Converted {i}/{len(files)}: "
            f"{npy_path.name} -> {png_path.name}"
        )

    print()
    print("=" * 65)
    print("CONVERSION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
