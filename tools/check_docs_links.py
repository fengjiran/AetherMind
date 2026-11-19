#!/usr/bin/env python3
"""Check relative inline links and Markdown heading anchors.

The default scope is the repository entry pages and docs/, excluding archive/.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from functools import cache
from pathlib import Path
from urllib.parse import unquote

ENTRY_PAGES = ("README.md", "AGENTS.md", "CHANGELOG.md")
LINK_RE = re.compile(r"\[((?:\[[^\[\]\n]*\]|[^\[\]\n])*)\]\(((?:\([^()\n]*\)|[^()\n])*)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "file://")
CODE_LIKE = ("::", "*", "<", ">")
CODE_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
HEADING_RE = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)\s*#*\s*$")
UNDERSCORE_EMPHASIS_RE = re.compile(r"(?<!\w)(_{1,3})(\S(?:.*?\S)?)\1(?!\w)")
HEADING_CODE_OR_TAG_RE = re.compile(r"`([^`\n]*)`|<[^>]*>")
LINE_SUFFIX_RE = re.compile(r":\d+$")


def iter_docs(root: Path, include_archive: bool):
    for path in sorted(root.rglob("*.md")):
        if path.is_symlink():
            continue
        if not include_archive and "archive" in path.parts:
            continue
        yield path


def prose_lines(path: Path):
    fence = ""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        delimiter = CODE_FENCE_RE.match(line)
        if fence:
            if delimiter and delimiter[1].startswith(fence) and not delimiter[2].strip():
                fence = ""
            continue
        if delimiter:
            fence = delimiter[1]
            continue
        yield number, line


@cache
def heading_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    for _, line in prose_lines(path):
        heading = HEADING_RE.match(line)
        if not heading:
            continue
        label = LINK_RE.sub(r"\1", heading[1])
        label = UNDERSCORE_EMPHASIS_RE.sub(r"\2", label)
        label = HEADING_CODE_OR_TAG_RE.sub(lambda match: match[1] or "", label)
        slug = "".join(
            char for char in label.strip().lower().replace(" ", "-")
            if char in "-_" or unicodedata.category(char)[0] in "LMN"
        )
        anchor = slug
        suffix = 1
        while anchor in anchors:
            anchor = f"{slug}-{suffix}"
            suffix += 1
        anchors.add(anchor)
    return anchors


def check_file(path: Path) -> list[str]:
    broken: list[str] = []
    for number, line in prose_lines(path):
        for match in LINK_RE.finditer(INLINE_CODE_RE.sub("", line)):
            target = match[2].strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            if not target or target.startswith(SKIP_PREFIXES):
                continue

            filename, _, fragment = target.partition("#")
            if any(marker in filename for marker in CODE_LIKE):
                continue
            filename = LINE_SUFFIX_RE.sub("", unquote(filename))
            destination = (path.parent / filename if filename else path).resolve()
            if not destination.exists():
                broken.append(f"{path}:{number}: broken link -> {target}")
            elif fragment and destination.is_file() and destination.suffix.lower() == ".md":
                if unquote(fragment) not in heading_anchors(destination):
                    broken.append(f"{path}:{number}: unknown heading anchor -> {target}")
    return broken


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="scan this directory instead of the default scope")
    parser.add_argument("--include-archive", action="store_true")
    args = parser.parse_args()

    root = args.root if args.root is not None else Path("docs")
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    paths = list(iter_docs(root, args.include_archive))
    if args.root is None:
        paths[0:0] = [Path(name) for name in ENTRY_PAGES if Path(name).is_file()]

    broken: list[str] = []
    for path in paths:
        broken.extend(check_file(path))

    for item in broken:
        print(item)
    print(f"checked {len(paths)} files, {len(broken)} broken links")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
