"""Fetch checksum-pinned official headers for compile-only verification.

Does not install packages, download models, or provide runtime libraries.
Run from any directory; defaults to the repository's ignored .build directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main() -> None:
    cpp = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=cpp.parent / ".build" / "cpp-deps")
    args = parser.parse_args()
    lock = json.loads((cpp / "dependencies.lock.json").read_text(encoding="utf-8"))
    output = args.output.resolve()
    jobs = []
    for dependency in ("llama", "tensorrt"):
        spec = lock[dependency]
        for relative, digest in spec["headers"].items():
            target = (output / dependency / relative).resolve()
            if not target.is_relative_to(output):
                raise ValueError(f"Unsafe header path: {relative}")
            url = f"https://raw.githubusercontent.com/{spec['repository']}/{spec['revision']}/{relative}"
            jobs.append((url, target, digest))

    def fetch(job: tuple[str, Path, str]) -> str:
        url, target, digest = job
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            return f"OK {target.relative_to(output)}"
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "qwen-cpp-header-check"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError(f"Checksum mismatch: {url}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(data)
                temporary.replace(target)
                return f"FETCHED {target.relative_to(output)}"
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(fetch, jobs):
            print(result)


if __name__ == "__main__":
    main()
