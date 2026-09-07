"""Compare all C++ tensor elements against Pillow + the real HF processor.

Usage: python cpp/tests/compare_preprocess.py --executable build/cpp/test_preprocess
Requires locally installed numpy, Pillow and transformers; never downloads models.
The fixed resize is explicit: it is not Qwen's aspect-ratio-preserving smart_resize.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import PIL
from PIL import Image
import transformers
from transformers import Qwen2VLImageProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    processor = Qwen2VLImageProcessor(do_resize=False)
    cases = [
        (84, 56, 84, 56),
        (11, 17, 28, 28),
        (97, 65, 56, 28),
        (56, 19, 56, 28),
        (29, 56, 56, 56),
        (300, 17, 28, 56),
        (1, 33, 28, 28),
        (33, 1, 28, 28),
        (1920, 1080, 896, 896),
        (51, 79, 896, 896),
        (896, 896, 896, 896),
    ]
    print(f"transformers={transformers.__version__}, Pillow={PIL.__version__}, numpy={np.__version__}")
    with tempfile.TemporaryDirectory(prefix="qwen_cpp_preprocess_") as directory:
        tensor_path = Path(directory) / "pixels.bin"
        for width, height, out_width, out_height in cases:
            y, x, channel = np.indices((height, width, 3), dtype=np.int64)
            rgb = ((x * 17 + y * 29 + channel * 71 + (x * y % 31) * 3) % 256).astype(np.uint8)
            resized = Image.fromarray(rgb).resize((out_width, out_height), Image.Resampling.BICUBIC)
            reference = processor(images=resized, do_resize=False, return_tensors="np")
            expected = reference["pixel_values"].astype(np.float32, copy=False)
            np.testing.assert_array_equal(reference["image_grid_thw"], [[1, out_height // 14, out_width // 14]])
            subprocess.run(
                [str(executable), "--dump", str(tensor_path), str(width), str(height),
                 str(out_width), str(out_height)],
                check=True,
            )
            actual = np.fromfile(tensor_path, dtype=np.float32).reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
            error = float(np.max(np.abs(actual - expected)))
            print(f"{width}x{height} -> {out_width}x{out_height}: shape={actual.shape}, max_abs_error={error:.9g}")
    print(f"PASS: all tensor elements matched for {len(cases)} image cases")


if __name__ == "__main__":
    main()
