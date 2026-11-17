"""Write the repo and commit into notebooks/provenance.py before the static export.

Usage (CI): ``uv run python scripts/stamp_provenance.py "$GITHUB_REPOSITORY" "$GITHUB_SHA"``
"""

import re
import sys
from pathlib import Path

PATH = Path(__file__).parents[1] / "notebooks" / "provenance.py"

repo, sha = sys.argv[1], sys.argv[2]
source = PATH.read_text(encoding="utf-8")
stamped = re.sub(r'^REPO = "[^"]*"$', f'REPO = "{repo}"', source, count=1, flags=re.MULTILINE)
stamped = re.sub(r'^SHA = "[^"]*"$', f'SHA = "{sha}"', stamped, count=1, flags=re.MULTILINE)
if stamped.count(repo) < 1 or stamped.count(sha) < 1:
    raise SystemExit("could not find the REPO/SHA lines in provenance.py")
PATH.write_text(stamped, encoding="utf-8")
print(f"stamped {PATH.name}: {repo} @ {sha[:7]}")
