#!/usr/bin/env python3
"""Check relative Markdown links under docs/.

By default archive/ is skipped: it is cold storage and is not maintained.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "#", "mailto:", "file://")
CODE_LIKE = ("::", "*", "<", ">", "[", "]", "(", ")", ",")
LINE_SUFFIX_RE = re.compile(r":\d+$")


def iter_docs(root: Path, include_archive: bool):
    for path in sorted(root.rglob("*.md")):
        if path.is_symlink():
            continue
        if not include_archive and "archive" in path.parts:
            continue
        yield path


def check_file(path: Path) -> list[str]:
    broken: list[str] = []
    for match in LINK_RE.finditer(path.read_text(encoding="utf-8")):
        target = match.group(1).strip()
        if target.startswith(SKIP_PREFIXES) or any(c in target for c in CODE_LIKE):
            continue
        target = LINE_SUFFIX_RE.sub("", target.split("#", 1)[0])
        if not target:
            continue
        if not (path.parent / target).resolve().exists():
            broken.append(f"{path}: broken link -> {target}")
    return broken


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="docs", help="root directory to scan")
    parser.add_argument("--include-archive", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    broken: list[str] = []
    count = 0
    for path in iter_docs(root, args.include_archive):
        count += 1
        broken.extend(check_file(path))

    for item in broken:
        print(item)
    print(f"checked {count} files, {len(broken)} broken links")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
