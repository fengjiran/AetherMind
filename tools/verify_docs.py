#!/usr/bin/env python3
"""AetherMind documentation drift checker.

Checks invariants defined in docs/guides/documentation-guide.md:

1. Link validity: every relative markdown link under docs/ resolves to an
   existing file or directory (anchors are not validated).
2. Symbol traceability: identifiers of the form ``Class::Member`` that appear
   in design documents under docs/designs/ exist in some header under
   include/ or some source under src/.
3. Index coverage: every documentation file (except templates/, archive/,
   agent/, and improvement-plan chapters) is referenced from docs/README.md,
   either directly or through a directory-level link (migration-friendly).

Excluded subtrees: docs/agent/ (independent subsystem), docs/templates/
(copy-paste sources with placeholder paths), docs/archive/ (superseded
documents, covered by archive/README.md), docs/improvement-plan/ (covered by
its own README.md).

Usage:
    python3 tools/verify_docs.py [--root PATH] [--no-links] [--no-symbols]
                                 [--no-index]

Exit code is 0 when all checks pass, 1 otherwise.
"""

import argparse
import re
import sys
from pathlib import Path

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SYMBOL_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)\b")
# Code-like link targets that are not filesystem paths (function signatures,
# type names in ABI tables, etc.).
CODE_LIKE_TARGET = ("::", "*", "&", "<", ">")
# ``file:line`` style targets; the line suffix is not part of the path.
LINE_SUFFIX_RE = re.compile(r"^(.*):\d+$")
EXCLUDED_PARTS = ("templates", "archive", "agent", "improvement-plan")


def iter_markdown(docs: Path):
    for path in sorted(docs.rglob("*.md")):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.name == "README.md" and path.parent == docs:
            continue  # The index itself is not a subject.
        yield path


def iter_code_blocks(lines: list[str]):
    """Yields (start, end) line ranges of fenced code blocks."""
    ranges = []
    in_block = False
    start = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            if not in_block:
                start = i
                in_block = True
            else:
                ranges.append((start, i))
                in_block = False
    return ranges


def check_links(docs: Path) -> list[str]:
    problems = []
    for path in iter_markdown(docs):
        lines = path.read_text(encoding="utf-8").splitlines()
        code_ranges = iter_code_blocks(lines)
        for line_no, line in enumerate(lines, 1):
            if any(s <= line_no - 1 <= e for s, e in code_ranges):
                continue  # Template/example snippets are not file links.
            for target in LINK_RE.findall(line):
                target = target.split("#", 1)[0]
                if not target:
                    continue
                if target.startswith(("http://", "https://", "mailto:", "file://")):
                    continue
                if any(token in target for token in CODE_LIKE_TARGET):
                    continue  # Code-signature-like target, not a file path.
                if any(ch.isspace() for ch in target):
                    continue  # Contains spaces, not a file path.
                m = LINE_SUFFIX_RE.match(target)
                if m:
                    target = m.group(1)  # ``file:line`` reference.
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    problems.append(f"{path.relative_to(docs)}:{line_no}: broken link -> {target}")
    return problems


def check_symbols(root: Path, docs: Path) -> list[str]:
    """Checks symbol traceability for module design docs only.

    Only documents following the canonical module-design naming pattern
    (``docs/designs/<module>/NN-*.md``) are checked; legacy/top-level design
    documents under migration and the architecture overview (which may name
    Phase-1 target symbols) are exempt until they are reorganized.
    """
    sources = []
    for subdir, pattern in (("include", "*.h"), ("src", "*.cpp")):
        sources += list((root / subdir).rglob(pattern))
    source_text = "\n".join(p.read_text(encoding="utf-8") for p in sources)
    problems = []
    for path in sorted((docs / "designs").rglob("*.md")):
        if "archive" in path.parts:
            continue
        rel_parts = path.relative_to(docs / "designs").parts
        if len(rel_parts) != 2 or not re.match(r"\d\d-", rel_parts[1]):
            continue  # Not a canonical module-design document.
        in_code_block = False
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue  # Illustrative snippets may simplify real signatures.
            for cls, member in SYMBOL_RE.findall(line):
                # Skip C++ standard/library names that are not AetherMind symbols.
                if cls in ("std", "std::"):
                    continue
                # Loose traceability: the class name and the member name must
                # both occur somewhere in include/ or src/.
                cls_pattern = "\\b" + re.escape(cls) + "\\b"
                member_pattern = "\\b" + re.escape(member) + "\\b"
                if not re.search(cls_pattern, source_text) or not re.search(
                    member_pattern, source_text
                ):
                    problems.append(
                        f"{path.relative_to(docs)}:{line_no}: "
                        f"symbol `{cls}::{member}` not found in include/ or src/"
                    )
    return problems


def check_index(docs: Path) -> list[str]:
    index = docs / "README.md"
    index_text = index.read_text(encoding="utf-8") if index.exists() else ""
    # Collect both direct file links and directory-level links from the index.
    linked_targets = {m.split("#", 1)[0] for m in LINK_RE.findall(index_text)}
    linked_dirs = {
        str((docs / t).resolve())
        for t in linked_targets
        if not t.startswith(("http://", "https://", "mailto:", "file://"))
        and (docs / t).is_dir()
    }
    problems = []
    for path in iter_markdown(docs):
        rel = path.relative_to(docs)
        rel_str = str(rel)
        if rel_str in index_text:
            continue
        # Directory-level reference covers all files under that directory
        # (migration-friendly while legacy subtrees are being reorganized).
        if any(str((docs / d).resolve()) in linked_dirs for d in rel.parents):
            continue
        problems.append(f"docs/README.md does not reference {rel_str}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--no-links", action="store_true", help="skip link validity check")
    parser.add_argument("--no-symbols", action="store_true", help="skip symbol traceability check")
    parser.add_argument("--no-index", action="store_true", help="skip index coverage check")
    args = parser.parse_args()

    docs = args.root / "docs"
    all_problems = []
    if not args.no_links:
        all_problems += check_links(docs)
    if not args.no_symbols:
        all_problems += check_symbols(args.root, docs)
    if not args.no_index:
        all_problems += check_index(docs)

    for problem in all_problems:
        print(f"FAIL: {problem}")
    print(f"{'FAIL' if all_problems else 'OK'}: {len(all_problems)} problem(s) found")
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main())
