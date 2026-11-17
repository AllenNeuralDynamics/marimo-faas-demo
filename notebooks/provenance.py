"""Where this build of the notebook came from: ``<repo> @ <sha>``.

CI writes REPO and SHA below before exporting the static site (scripts/stamp_provenance.py).
Run locally, they are empty and the values come from git instead.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = ""
SHA = ""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
    ).stdout.strip()


def source() -> tuple[str, str]:
    """(repo, sha). Locally, a sha with uncommitted changes ends in ``-dirty``."""
    if SHA or sys.platform == "emscripten":
        return REPO or "unknown", SHA or "unknown"
    try:
        remote = _git("remote", "get-url", "origin")
        sha = _git("rev-parse", "HEAD") + ("-dirty" if _git("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):  # no git, or not a checkout
        return "unknown", "unknown"
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote)
    return (match[1] if match else remote), sha


def label() -> str:
    repo, sha = source()
    return f"Generated from {repo} @ {sha[:7]}{'-dirty' if sha.endswith('-dirty') else ''}"


def commit_url() -> str | None:
    repo, sha = source()
    return None if "unknown" in (repo, sha) or sha.endswith("-dirty") else f"https://github.com/{repo}/commit/{sha}"
