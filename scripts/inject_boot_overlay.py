"""Insert the loading overlay into an exported WASM bundle's index.html, in place.

Run after ``marimo export html-wasm`` (the export rewrites the output directory). The step
list and channel name come from notebooks/boot.py, so the overlay and the notebook agree.

Usage: ``uv run python scripts/inject_boot_overlay.py _site/index.html``
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "notebooks"))
from boot import CHANNEL, STEPS

TEMPLATE = Path(__file__).with_name("boot_overlay.html")


def inject(page: str) -> str:
    bodies = list(re.finditer(r"<body[^>]*>", page))
    if len(bodies) != 1 or 'id="vrs-boot"' in page:
        raise ValueError("expected exactly one <body> and no overlay yet")
    overlay = TEMPLATE.read_text(encoding="utf-8")
    overlay = overlay.replace("__BOOT_STEPS__", json.dumps(STEPS)).replace("__BOOT_CHANNEL__", json.dumps(CHANNEL))
    return page[: bodies[0].end()] + "\n" + overlay + page[bodies[0].end() :]


if __name__ == "__main__":
    for path in map(Path, sys.argv[1:]):
        path.write_text(inject(path.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"injected the loading overlay into {path}")
