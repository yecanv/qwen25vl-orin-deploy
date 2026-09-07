"""Exercise the production image loader, shape contract and preprocessing CLI.

PNG/PPM: compare every tensor element with Pillow + HF for the same fixed resize.
JPEG: decoding/shape/finiteness smoke check; different decoders need not be bit exact.
No models or network access are required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
from transformers import Qwen2VLImageProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    y, x, c = np.indices((79, 51, 3), dtype=np.int64)
    pixels = ((x * 17 + y * 29 + c * 71 + (x * y % 31) * 3) % 256).astype(np.uint8)
    image = Image.fromarray(pixels)
    processor = Qwen2VLImageProcessor(do_resize=False)
    expected = processor(images=image.resize((896, 896), Image.Resampling.BICUBIC),
                         do_resize=False, return_tensors="np")["pixel_values"]
    contract = {"static": True, "patch_dim": 1176, "grid_thw": [1, 64, 64],
                "input": {"pixel_values": [4096, 1176]}, "output": {"vision_embeds": [1024, 2048]}}
    with tempfile.TemporaryDirectory(prefix="qwen_cpp_cli_") as directory:
        root = Path(directory)
        shapes = root / "shape.json"
        shapes.write_text(json.dumps(contract), encoding="utf-8")
        output = root / "pixels.f32"
        for suffix in ("png", "ppm", "jpg"):
            source = root / f"image.{suffix}"
            image.save(source)
            subprocess.run([str(executable), "--image", str(source), "--vision-contract", str(shapes),
                            "--output", str(output)], check=True)
            actual = np.fromfile(output, dtype=np.float32).reshape(4096, 1176)
            assert np.isfinite(actual).all()
            if suffix != "jpg":
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
                print(f"PASS {suffix}: max_abs_error={np.max(np.abs(actual - expected)):.9g}")
            else:
                print("PASS jpg: decoded, correct tensor shape, all finite (no decoder parity claim)")


if __name__ == "__main__":
    main()
