"""Progress steps for the static site's loading overlay (scripts/boot_overlay.html).

The notebook calls :func:`report` as it moves through a cold start; the overlay, which
paints before Python exists, listens on a BroadcastChannel. Outside the browser this does
nothing. Kept free of heavy imports so it can report before pandas & co. load.
"""

import sys

CHANNEL = "vr-summary-boot"

#: (key, label) in order. The first two are inferred by the overlay from marimo's own
#: loading text; the rest are posted by the notebook. ``"done"`` closes the overlay.
STEPS = [
    ("python", "Starting Python in your browser"),
    ("packages", "Installing packages (pandas, pyarrow, matplotlib)"),
    ("libraries", "Loading the Python libraries"),
    ("session", "Finding the session in the AIND metadata service"),
    ("download", "Downloading the session's data"),
    ("processing", "Processing and plotting"),
]


def report(step: str) -> None:
    """Tell the overlay that ``step`` has started (or ``"done"``)."""
    if sys.platform == "emscripten":
        import js

        js.BroadcastChannel.new(CHANNEL).postMessage(step)
