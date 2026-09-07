# Third-party notices

The vendored files are unmodified upstream single-header distributions, including
their embedded copyright and license notices. Exact source URLs and SHA-256
checksums are recorded in `../dependencies.lock.json`.

| File | Upstream | Revision | License |
|---|---|---|---|
| `stb_image.h` | https://github.com/nothings/stb | `2c980bb59875b0d32144a71867fbdebb2f77cd20` | MIT or public domain, as specified at the end of the header |
| `json.hpp` | https://github.com/nlohmann/json | `v3.11.3` | MIT, notice at the beginning of the header |

The project preprocessing code follows the published Qwen image layout and Pillow
BICUBIC algorithm. Reference sources:

- https://github.com/python-pillow/Pillow/blob/11.0.0/src/libImaging/Resample.c
- https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py

Pillow and Transformers license copies accompany these references. Their libraries
are used only by optional Python comparison tests; the C++ runtime does not embed
Python, Pillow, Transformers, NumPy or PyTorch.

TensorRT and llama.cpp SDK headers are fetched to an ignored build directory for
compilation and retain their upstream notices. They are not vendored here.
