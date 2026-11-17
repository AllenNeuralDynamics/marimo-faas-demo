"""Messages from Python to the static site's page script (scripts/boot_overlay.html).

* :func:`report`: progress steps for the loading overlay, which paints before Python exists.
* :func:`save_file`: hand the page a file to download (Python runs in a Web Worker, which
  cannot start a download itself).

Both go over a BroadcastChannel and do nothing outside the browser. Kept free of heavy
imports so it can report before pandas & co. load.
"""

import sys

CHANNEL = "vr-summary-boot"
DOWNLOAD_CHANNEL = "vr-summary-download"

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


def save_file(filename: str, data: bytes, mimetype: str) -> bool:
    """Ask the page to download ``data`` as ``filename``; False outside the browser."""
    if sys.platform != "emscripten":
        return False
    import js
    from pyodide.ffi import to_js

    message = to_js({"name": filename, "type": mimetype, "data": data}, dict_converter=js.Object.fromEntries)
    js.BroadcastChannel.new(DOWNLOAD_CHANNEL).postMessage(message)
    return True
