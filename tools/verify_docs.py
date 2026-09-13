#!/usr/bin/env python3
"""AetherMind documentation drift checker.

Checks invariants defined in docs/guides/documentation-guide.md:

1. Link validity: every relative markdown link under docs/ resolves to an
   existing file or directory.
2. Anchor validity: every ``#heading`` anchor in relative links resolves to
   an actual heading in the target document.
3. Symbol traceability: identifiers of the form ``Class::Member`` that appear
   in design documents under docs/designs/ exist as a combined pattern in
   some header under include/ or some source under src/.
4. Index coverage: every documentation file (except templates/, archive/,
   and agent/) is referenced from docs/README.md, either directly or through
   a directory-level link.

Report-only diagnostics (do not affect exit code unless --strict-* is given):

5. designs/ draft detection: files in designs/ whose name suggests plan/
   milestone/checklist content.
6. Canonical naming coverage: designs/<module>/ files not following NN-*.md.
7. README consistency: docs/README.md vs sub-directory README index entries.

Excluded subtrees: docs/agent/ (independent subsystem), docs/templates/
(copy-paste sources with placeholder paths), docs/archive/ (superseded
documents, covered by archive/README.md).

Usage:
    python3 tools/verify_docs.py [--root PATH] [--no-links] [--no-anchors]
                                 [--no-symbols] [--no-index]
                                 [--strict-anchors] [--strict-symbols]
                                 [--strict-designs] [--strict-naming]
                                 [--strict-readme]

Exit code is 0 when all gate checks pass, 1 otherwise.
Report-only diagnostics print WARN lines but do not affect exit code
unless the corresponding --strict-* flag is given.
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
EXCLUDED_PARTS = ("templates", "archive", "agent")

# Anchor extraction: GitHub-style slug from heading text.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
ANCHOR_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]*#[^)]+)\)")
# Plan-like filenames that should not live in designs/.
PLAN_NAME_RE = re.compile(
    r"(development[_-]?plan|milestones?|task[_-]?checklist|implementation[_-]?plan)",
    re.IGNORECASE,
)
# Canonical module-design naming: NN-<kebab>.md
CANONICAL_DESIGN_RE = re.compile(r"^\d\d-[a-z0-9][a-z0-9_-]*\.md$")


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
    """Checks symbol traceability for module design docs (precise matching).

    Only documents following the canonical module-design naming pattern
    (``docs/designs/<module>/NN-*.md``) are checked. ``Class::Member`` must
    appear as a combined token in include/ or src/.
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
            continue
        in_code_block = False
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            for cls, member in SYMBOL_RE.findall(line):
                if cls in ("std", "std::"):
                    continue
                combined = re.escape(cls) + "::" + re.escape(member)
                if not re.search(combined, source_text):
                    problems.append(
                        f"{path.relative_to(docs)}:{line_no}: "
                        f"symbol `{cls}::{member}` not found in include/ or src/"
                    )
    return problems


def check_index(docs: Path) -> list[str]:
    index = docs / "README.md"
    index_text = index.read_text(encoding="utf-8") if index.exists() else ""
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
        if any(str((docs / d).resolve()) in linked_dirs for d in rel.parents):
            continue
        problems.append(f"docs/README.md does not reference {rel_str}")
    return problems


def _slugify(heading_text: str) -> str:
    """Convert heading text to GitHub-style anchor slug."""
    slug = heading_text.lower().strip()
    slug = re.sub(r"<[^>]+>", "", slug)
    slug = re.sub(r"[^\w\u4e00-\u9fff\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    return slug.strip("-")


def check_anchors(docs: Path) -> list[str]:
    """Validates that #anchor links resolve to actual headings in target docs."""
    heading_cache: dict[Path, set[str]] = {}
    for path in sorted(docs.rglob("*.md")):
        headings = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            m = HEADING_RE.match(line)
            if m:
                headings.add(_slugify(m.group(2)))
        heading_cache[path.resolve()] = headings

    problems = []
    for path in iter_markdown(docs):
        lines = path.read_text(encoding="utf-8").splitlines()
        code_ranges = iter_code_blocks(lines)
        for line_no, line in enumerate(lines, 1):
            if any(s <= line_no - 1 <= e for s, e in code_ranges):
                continue
            for target in ANCHOR_LINK_RE.findall(line):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                if any(token in target for token in CODE_LIKE_TARGET):
                    continue
                file_part, _, anchor = target.partition("#")
                if not anchor:
                    continue
                if file_part:
                    resolved = (path.parent / file_part).resolve()
                else:
                    resolved = path.resolve()
                if not resolved.exists():
                    continue  # Broken link caught by check_links
                target_headings = heading_cache.get(resolved, set())
                slug = _slugify(anchor)
                if slug not in target_headings and anchor not in target_headings:
                    problems.append(
                        f"{path.relative_to(docs)}:{line_no}: "
                        f"anchor #{anchor} not found in {resolved.name}"
                    )
    return problems


def check_designs_drafts(docs: Path) -> list[str]:
    """Report-only: detects plan-like filenames in designs/."""
    warnings = []
    designs = docs / "designs"
    if not designs.exists():
        return warnings
    for path in sorted(designs.rglob("*.md")):
        if PLAN_NAME_RE.search(path.stem):
            warnings.append(
                f"{path.relative_to(docs)}: filename suggests plan/milestone/checklist in designs/"
            )
    return warnings


def check_canonical_naming(docs: Path) -> list[str]:
    """Report-only: designs/<module>/ files not following NN-*.md naming."""
    warnings = []
    designs = docs / "designs"
    if not designs.exists():
        return warnings
    for subdir in sorted(designs.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name == "architecture":
            continue  # Explicit exception per documentation-guide.md §2
        for path in sorted(subdir.rglob("*.md")):
            if not CANONICAL_DESIGN_RE.match(path.name):
                warnings.append(
                    f"{path.relative_to(docs)}: non-canonical naming (expected NN-<kebab>.md)"
                )
    for path in sorted(designs.glob("*.md")):
        warnings.append(
            f"{path.relative_to(docs)}: top-level design file (expected designs/<module>/NN-*.md)"
        )
    return warnings


def check_readme_consistency(docs: Path) -> list[str]:
    """Report-only: checks docs/README.md references sub-directory READMEs."""
    warnings = []
    main_readme = docs / "README.md"
    if not main_readme.exists():
        return warnings
    main_text = main_readme.read_text(encoding="utf-8")
    for subdir in sorted(docs.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name in EXCLUDED_PARTS:
            continue
        sub_readme = subdir / "README.md"
        if sub_readme.exists():
            rel = f"{subdir.name}/README.md"
            if rel not in main_text and f"{subdir.name}/" not in main_text:
                warnings.append(f"docs/README.md does not reference {rel}")
    return warnings


# Allowed status values for frontmatter validation (03 号方案 §3.2).
ALLOWED_STATUSES = frozenset({
    "Current", "Draft", "Historical Snapshot", "Superseded", "Deprecated",
    "In Progress", "Implemented", "Proposed", "Accepted",
})


def check_frontmatter_status(docs: Path) -> list[str]:
    """Report-only: validates status field in YAML frontmatter."""
    warnings = []
    for path in iter_markdown(docs):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue  # No frontmatter
        end = text.find("\n---", 3)
        if end < 0:
            continue
        fm = text[3:end]
        for line in fm.splitlines():
            # Match "状态: value" or "status: value"
            m = re.match(r"^\s*(?:\u72b6\u6001|status)\s*[:\uff1a]\s*(.+)$", line, re.IGNORECASE)
            if m:
                value = m.group(1).strip().strip("'\"")
                if value and value not in ALLOWED_STATUSES:
                    warnings.append(
                        f"{path.relative_to(docs)}: frontmatter status '{value}' not in allowed enum"
                    )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    # Gate checks (failure = exit 1)
    parser.add_argument("--no-links", action="store_true", help="skip link validity check")
    parser.add_argument("--no-index", action="store_true", help="skip index coverage check")
    # Report-only checks (disable or promote to gate)
    parser.add_argument("--no-anchors", action="store_true", help="skip anchor validity check")
    parser.add_argument("--no-symbols", action="store_true", help="skip symbol traceability check")
    parser.add_argument("--strict-anchors", action="store_true", help="anchor failures exit non-zero")
    parser.add_argument("--strict-symbols", action="store_true", help="symbol failures exit non-zero")
    parser.add_argument("--strict-designs", action="store_true", help="designs draft detection exits non-zero")
    parser.add_argument("--strict-naming", action="store_true", help="canonical naming check exits non-zero")
    parser.add_argument("--strict-readme", action="store_true", help="README consistency check exits non-zero")
    args = parser.parse_args()

    docs = args.root / "docs"
    gate_problems: list[str] = []
    report_warnings: list[str] = []

    # Gate checks (always fail on error unless --no-*)
    if not args.no_links:
        gate_problems += check_links(docs)
    if not args.no_index:
        gate_problems += check_index(docs)

    # Anchor check: report-only by default
    if not args.no_anchors:
        anchor_issues = check_anchors(docs)
        if args.strict_anchors:
            gate_problems += anchor_issues
        else:
            report_warnings += anchor_issues

    # Symbol check (precise): report-only by default
    if not args.no_symbols:
        symbol_issues = check_symbols(args.root, docs)
        if args.strict_symbols:
            gate_problems += symbol_issues
        else:
            report_warnings += symbol_issues

    # Report-only diagnostics
    designs_w = check_designs_drafts(docs)
    gate_problems += designs_w if args.strict_designs else []
    report_warnings += [] if args.strict_designs else designs_w

    naming_w = check_canonical_naming(docs)
    gate_problems += naming_w if args.strict_naming else []
    report_warnings += [] if args.strict_naming else naming_w

    readme_w = check_readme_consistency(docs)
    gate_problems += readme_w if args.strict_readme else []
    report_warnings += [] if args.strict_readme else readme_w

    status_w = check_frontmatter_status(docs)
    gate_problems += status_w if args.strict_readme else []
    report_warnings += [] if args.strict_readme else status_w

    for warning in report_warnings:
        print(f"WARN: {warning}")
    for problem in gate_problems:
        print(f"FAIL: {problem}")

    n_fail = len(gate_problems)
    n_warn = len(report_warnings)
    status = "FAIL" if n_fail else "OK"
    print(f"{status}: {n_fail} problem(s), {n_warn} warning(s)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
