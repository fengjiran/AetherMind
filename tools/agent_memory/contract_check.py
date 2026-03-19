#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / ".sisyphus" / "evidence"
BASELINE_PATH = EVIDENCE_DIR / "agent_memory_default_startup_baseline.json"

AGENTS_PATH = REPO_ROOT / "AGENTS.md"
PROJECT_MEMORY_PATH = REPO_ROOT / "docs/agent/memory/project.md"
MEMORY_README_PATH = REPO_ROOT / "docs/agent/memory/README.md"
MEMORY_QUICKSTART_PATH = REPO_ROOT / "docs/agent/memory/QUICKSTART.md"
MEMORY_SYSTEM_PATH = REPO_ROOT / "docs/agent/memory_system.md"
PROMPTS_README_PATH = REPO_ROOT / "docs/agent/prompts/README.md"
QUICK_RESUME_PATH = REPO_ROOT / "docs/agent/prompts/quick_resume.md"
NEW_SESSION_PATH = REPO_ROOT / "docs/agent/prompts/new_session_template.md"
HANDOFF_TEMPLATE_PATH = REPO_ROOT / "docs/agent/prompts/handoff_template.md"
HANDOFF_README_PATH = REPO_ROOT / "docs/agent/handoff/README.md"
MODULE_MEMORY_ROOT = REPO_ROOT / "docs/agent/memory/modules"
HANDOFF_ROOT = REPO_ROOT / "docs/agent/handoff/workstreams"

STARTUP_FACING_DOCS = [
    AGENTS_PATH,
    MEMORY_README_PATH,
    PROJECT_MEMORY_PATH,
    MEMORY_QUICKSTART_PATH,
    MEMORY_SYSTEM_PATH,
    PROMPTS_README_PATH,
    QUICK_RESUME_PATH,
    NEW_SESSION_PATH,
    HANDOFF_TEMPLATE_PATH,
    HANDOFF_README_PATH,
]

DEFAULT_STARTUP_DOCS = [
    AGENTS_PATH,
    PROJECT_MEMORY_PATH,
    MEMORY_README_PATH,
    QUICK_RESUME_PATH,
    PROMPTS_README_PATH,
]


@dataclass(frozen=True)
class HandoffInfo:
    path: Path
    metadata: dict[str, object]


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _strip_inline_comment(raw: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(raw):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return raw[:idx].rstrip()
    return raw.rstrip()


def _parse_scalar(raw: str) -> object:
    value = _strip_inline_comment(raw.strip())
    if not value:
        return ""
    if value in {"null", "Null", "NULL"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _extract_frontmatter(text: str) -> dict[str, object]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}
    metadata: dict[str, object] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if not stripped or stripped.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        metadata[key.strip()] = _parse_scalar(raw_value)
    return metadata


def _handoff_metadata(path: Path) -> dict[str, object]:
    metadata = _extract_frontmatter(_read_text(path))
    if not metadata:
        return {}
    metadata.setdefault("schema_version", "1.0")
    metadata.setdefault("memory_status", "not_needed")
    metadata.setdefault("status", "active")
    metadata.setdefault("supersedes", None)
    metadata.setdefault("closed_at", None)
    metadata.setdefault("closed_reason", None)
    metadata.setdefault("bootstrap_ready", False)
    return metadata


def _project_slugs() -> set[str]:
    slugs: set[str] = set()
    for line in _read_text(PROJECT_MEMORY_PATH).splitlines():
        match = re.match(r"\|\s*`([^`]+)`\s*\|", line)
        if match:
            slugs.add(match.group(1))
    return slugs


def _existing_modules() -> set[str]:
    if not MODULE_MEMORY_ROOT.exists():
        return set()
    return {path.name for path in MODULE_MEMORY_ROOT.iterdir() if path.is_dir()}


def _module_has_submodule(module: str, submodule: str) -> bool:
    return (MODULE_MEMORY_ROOT / module / "submodules" / f"{submodule}.md").exists()


def _resolve_workstream_key(raw: str | None) -> tuple[str, str, str, str | None]:
    if not raw:
        return ("unknown", "", "", None)

    if raw.startswith("project__"):
        slug = raw.split("__", 1)[1]
        if slug in _project_slugs():
            return ("project", raw, "", slug)
        return ("invalid", raw, "", slug)

    if "__" in raw:
        module, submodule = raw.split("__", 1)
        if module in _existing_modules():
            if submodule == "none" or _module_has_submodule(module, submodule):
                return ("module", raw, module, None)
        return ("invalid", raw, module, None)

    if raw in _existing_modules():
        return ("module", f"{raw}__none", raw, None)

    return ("ambiguous", raw, "", None)


def _handoffs_for_workstream(workstream_key: str) -> list[HandoffInfo]:
    workstream_dir = HANDOFF_ROOT / workstream_key
    if not workstream_dir.exists():
        return []
    handoffs: list[HandoffInfo] = []
    for path in sorted(workstream_dir.glob("*.md")):
        metadata = _handoff_metadata(path)
        if not metadata:
            continue
        if metadata.get("status") != "active":
            continue
        handoffs.append(HandoffInfo(path=path, metadata=metadata))
    handoffs.sort(
        key=lambda item: (
            str(item.metadata.get("created_at", "")),
            item.path.name,
        ),
        reverse=True,
    )
    return handoffs


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _int_value(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _agents_text() -> str:
    return _read_text(AGENTS_PATH)


def _agents_canonical_contract_enabled() -> bool:
    return "唯一规范性启动契约" in _agents_text()


def _confirmation_gate_present() -> bool:
    phrases = [
        '等待用户明确说"继续"',
        '等待用户明确说"继续"、"执行"或"是"',
        "必须等待用户明确确认",
        "用户确认",
    ]
    return _contains_any(_read_text(AGENTS_PATH), phrases) or _contains_any(
        _read_text(QUICK_RESUME_PATH), phrases
    )


def _readme_autoload_enabled() -> bool:
    agents_text = _agents_text()
    if _agents_canonical_contract_enabled():
        default_section = agents_text.split("3. **按需升级读取**", 1)[0]
        return "docs/agent/memory/README.md" in default_section

    phrases = [
        "AGENTS.md → docs/agent/memory/README.md",
        "AGENTS.md -> docs/agent/memory/README.md",
        "docs/agent/memory/README.md（操作规范，也是 workstream 键与 handoff frontmatter 的最终依据）",
        "`AGENTS.md` 和 `docs/agent/memory/README.md` 仍必须读取",
    ]
    return any(
        _contains_any(_read_text(path), phrases)
        for path in [
            AGENTS_PATH,
            QUICK_RESUME_PATH,
            NEW_SESSION_PATH,
            PROMPTS_README_PATH,
        ]
    )


def _memory_system_autoload_enabled() -> bool:
    phrases = [
        "AGENTS.md -> docs/agent/memory_system.md",
        "AGENTS.md → docs/agent/memory_system.md",
        "新会话启动.*memory_system",
    ]
    return any(_contains_any(_read_text(path), phrases) for path in STARTUP_FACING_DOCS)


def _default_startup_files() -> list[Path]:
    if _agents_canonical_contract_enabled():
        return [AGENTS_PATH, PROJECT_MEMORY_PATH]

    files = [AGENTS_PATH, PROJECT_MEMORY_PATH]
    if _readme_autoload_enabled():
        files.append(MEMORY_README_PATH)
    if QUICK_RESUME_PATH.exists():
        files.append(QUICK_RESUME_PATH)
    if PROMPTS_README_PATH.exists():
        files.append(PROMPTS_README_PATH)
    return files


def _file_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size


def _approx_tokens_from_bytes(total_bytes: int) -> int:
    return math.ceil(total_bytes / 4)


def _baseline_payload(current_files: list[Path]) -> dict[str, object]:
    total_bytes = sum(_file_bytes(path) for path in current_files)
    return {
        "default_files": [str(path.relative_to(REPO_ROOT)) for path in current_files],
        "default_bytes": total_bytes,
        "default_tokens_approx": _approx_tokens_from_bytes(total_bytes),
    }


def _ensure_baseline(current_files: list[Path]) -> dict[str, object]:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    payload = _baseline_payload(current_files)
    BASELINE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _load_order_sources() -> list[Path]:
    sources: list[Path] = []
    for path in STARTUP_FACING_DOCS:
        text = _read_text(path)
        if path == AGENTS_PATH and "默认启动路径" in text:
            sources.append(path)
            continue
        if _contains_any(
            text,
            [
                "加载顺序（按此固定顺序）",
                "AGENTS.md → docs/agent/memory/README.md",
                "AGENTS.md -> docs/agent/memory/README.md",
                "按固定顺序加载",
                "Agent 快速启动",
            ],
        ):
            sources.append(path)
    return sources


def _resume_gate_sources() -> list[Path]:
    phrases = ["## Resume Gate", "Resume Gate（", "执行 Resume Gate 检查点"]
    sources: list[Path] = []
    for path in STARTUP_FACING_DOCS:
        if _contains_any(_read_text(path), phrases):
            sources.append(path)
    return sources


def _emit_lines(payload: dict[str, object]) -> int:
    for key, value in payload.items():
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        print(f"{key}={value}")
    return 0


def _scenario_fresh_session_default() -> dict[str, object]:
    current_files = _default_startup_files()
    baseline = _ensure_baseline(current_files)
    return {
        "SCENARIO": "fresh_session_default",
        "README_AUTOLOAD": _bool_str(_readme_autoload_enabled()),
        "MEMORY_SYSTEM_AUTOLOAD": _bool_str(_memory_system_autoload_enabled()),
        "CONFIRMATION_GATE": _bool_str(_confirmation_gate_present()),
        "LOAD_PATH": ",".join(
            path.stem.upper().replace("_", "-") for path in current_files
        ),
        "DEFAULT_STARTUP_FILES": [
            str(path.relative_to(REPO_ROOT)) for path in current_files
        ],
        "DEFAULT_STARTUP_BYTES": sum(_file_bytes(path) for path in current_files),
        "BASELINE_PATH": str(BASELINE_PATH.relative_to(REPO_ROOT)),
        "BASELINE_DEFAULT_BYTES": baseline["default_bytes"],
    }


def _scenario_quick_resume(workstream: str | None) -> dict[str, object]:
    kind, resolved, module, slug = _resolve_workstream_key(workstream)
    if kind == "invalid":
        return {
            "SCENARIO": "quick_resume",
            "WORKSTREAM": workstream or "",
            "RESUME_STATUS": "blocked",
            "ACTION": "invalid_workstream",
            "README_AUTOLOAD": _bool_str(_readme_autoload_enabled()),
        }
    if kind == "ambiguous":
        return {
            "SCENARIO": "quick_resume",
            "WORKSTREAM": workstream or "",
            "RESUME_STATUS": "blocked",
            "ACTION": "request_clarification",
            "README_AUTOLOAD": _bool_str(_readme_autoload_enabled()),
        }

    handoffs = _handoffs_for_workstream(resolved)
    latest = handoffs[0] if handoffs else None
    bootstrap_ready = bool(latest and latest.metadata.get("bootstrap_ready", False))
    bootstrap_fallback = bool(latest and not bootstrap_ready)
    effective_readme_autoload = _readme_autoload_enabled()
    if _agents_canonical_contract_enabled() and bootstrap_fallback:
        effective_readme_autoload = True
        if kind == "project":
            load_path = ["AGENTS", "PROJECT", "README", "HANDOFF"]
        else:
            load_path = ["AGENTS", "PROJECT", "README", "MODULE", "SUBMODULE/HANDOFF"]
    elif _agents_canonical_contract_enabled():
        load_path = ["AGENTS", "PROJECT", "HANDOFF"]
    elif kind == "project":
        load_path = (
            ["AGENTS", "README", "PROJECT"]
            if _readme_autoload_enabled()
            else ["AGENTS", "PROJECT"]
        )
        load_path.append("HANDOFF")
    else:
        load_path = (
            ["AGENTS", "README", "PROJECT"]
            if _readme_autoload_enabled()
            else ["AGENTS", "PROJECT"]
        )
        load_path.extend(["MODULE", "SUBMODULE/HANDOFF"])

    resume_status = "complete" if latest else "partial"
    if kind == "module" and module and resolved.endswith("__none"):
        load_path[-1] = "HANDOFF"

    payload = {
        "SCENARIO": "quick_resume",
        "WORKSTREAM": resolved,
        "WORKSTREAM_KIND": kind,
        "README_AUTOLOAD": _bool_str(effective_readme_autoload),
        "CONFIRMATION_GATE": _bool_str(_confirmation_gate_present()),
        "LOAD_PATH": ",".join(load_path),
        "RESUME_STATUS": resume_status,
        "ACTIVE_HANDOFF_COUNT": len(handoffs),
        "LATEST_HANDOFF": str(latest.path.relative_to(REPO_ROOT)) if latest else "",
        "BOOTSTRAP_READY": _bool_str(bootstrap_ready),
        "BOOTSTRAP_READY_FALLBACK": _bool_str(bootstrap_fallback),
    }
    if slug:
        payload["SLUG"] = slug
    return payload


def _scenario_missing_handoff(workstream: str | None) -> dict[str, object]:
    kind, resolved, _module, slug = _resolve_workstream_key(workstream)
    if kind == "invalid":
        return {
            "SCENARIO": "missing_handoff",
            "WORKSTREAM": workstream or "",
            "RESUME_STATUS": "blocked",
            "ACTION": "invalid_workstream",
            "README_AUTOLOAD": _bool_str(_readme_autoload_enabled()),
        }
    if kind == "ambiguous":
        return {
            "SCENARIO": "missing_handoff",
            "WORKSTREAM": workstream or "",
            "RESUME_STATUS": "blocked",
            "ACTION": "request_clarification",
            "README_AUTOLOAD": _bool_str(_readme_autoload_enabled()),
        }

    handoffs = _handoffs_for_workstream(resolved)
    return {
        "SCENARIO": "missing_handoff",
        "WORKSTREAM": resolved,
        "WORKSTREAM_KIND": kind,
        "SLUG": slug or "",
        "ACTIVE_HANDOFF_COUNT": len(handoffs),
        "RESUME_STATUS": "partial" if not handoffs else "complete",
        "README_AUTOLOAD": _bool_str(_readme_autoload_enabled()),
    }


def _scenario_ambiguous_workstream(raw_input: str | None) -> dict[str, object]:
    if not raw_input or "__" not in raw_input:
        return {
            "SCENARIO": "ambiguous_workstream",
            "INPUT": raw_input or "",
            "ACTION": "request_clarification",
            "README_AUTOLOAD": _bool_str(_readme_autoload_enabled()),
        }
    kind, resolved, _module, _slug = _resolve_workstream_key(raw_input)
    return {
        "SCENARIO": "ambiguous_workstream",
        "INPUT": raw_input,
        "ACTION": "resolved"
        if kind in {"project", "module"}
        else "request_clarification",
        "RESOLVED_WORKSTREAM": resolved,
    }


def _scenario_explicit_escalation(reason: str | None) -> dict[str, object]:
    return {
        "SCENARIO": "explicit_escalation",
        "REASON": reason or "",
        "README_AUTOLOAD": "true",
        "MEMORY_SYSTEM_AUTOLOAD": _bool_str(reason == "need_architecture_reference"),
    }


def _scenario_duplication_scan() -> dict[str, object]:
    load_order_sources = _load_order_sources()
    resume_gate_sources = _resume_gate_sources()
    return {
        "SCENARIO": "duplication_scan",
        "LOAD_ORDER_SOURCES": [
            str(path.relative_to(REPO_ROOT)) for path in load_order_sources
        ],
        "RESUME_GATE_SOURCES": [
            str(path.relative_to(REPO_ROOT)) for path in resume_gate_sources
        ],
        "DUPLICATE_LOAD_ORDER_RULES": max(0, len(load_order_sources) - 1),
        "DUPLICATE_RESUME_GATE_TEMPLATES": max(0, len(resume_gate_sources) - 1),
        "README_IS_LOAD_ORDER_SOURCE": _bool_str(
            MEMORY_README_PATH in load_order_sources
        ),
        "README_IS_RESUME_GATE_SOURCE": _bool_str(
            MEMORY_README_PATH in resume_gate_sources
        ),
    }


def _scenario_token_budget() -> dict[str, object]:
    current_files = _default_startup_files()
    baseline = _ensure_baseline(current_files)
    current_bytes = sum(_file_bytes(path) for path in current_files)
    baseline_bytes = _int_value(baseline.get("default_bytes"))
    baseline_tokens = _int_value(baseline.get("default_tokens_approx"))
    reduction_pct = 0.0
    if baseline_bytes > 0:
        reduction_pct = (baseline_bytes - current_bytes) / baseline_bytes * 100.0
    return {
        "SCENARIO": "token_budget",
        "BASELINE_DEFAULT_BYTES": baseline_bytes,
        "CURRENT_DEFAULT_BYTES": current_bytes,
        "BASELINE_DEFAULT_TOKENS_APPROX": baseline_tokens,
        "CURRENT_DEFAULT_TOKENS_APPROX": _approx_tokens_from_bytes(current_bytes),
        "DEFAULT_CONTEXT_REDUCTION_PCT": f"{reduction_pct:.2f}",
        "DEFAULT_CONTEXT_REDUCTION_PCT>=60": _bool_str(reduction_pct >= 60.0),
    }


def _scenario_full_matrix() -> dict[str, object]:
    fresh_default = _scenario_fresh_session_default()
    legacy_project = _scenario_quick_resume("project__agent-memory-v1.1")
    legacy_module = _scenario_quick_resume("ammalloc__thread_cache")
    escalation = _scenario_explicit_escalation("need_operational_manual")
    missing = _scenario_missing_handoff("project__missing")
    ambiguous = _scenario_ambiguous_workstream("docs-reorg")
    duplication = _scenario_duplication_scan()
    budget = _scenario_token_budget()

    checks = {
        "fresh_session_default": (
            fresh_default["README_AUTOLOAD"] == "false"
            and fresh_default["MEMORY_SYSTEM_AUTOLOAD"] == "false"
            and fresh_default["CONFIRMATION_GATE"] == "true"
        ),
        "quick_resume_project_legacy": (
            legacy_project["BOOTSTRAP_READY_FALLBACK"] == "true"
            and legacy_project["README_AUTOLOAD"] == "true"
        ),
        "quick_resume_module_legacy": (
            legacy_module["BOOTSTRAP_READY_FALLBACK"] == "true"
            and legacy_module["README_AUTOLOAD"] == "true"
        ),
        "explicit_escalation": escalation["README_AUTOLOAD"] == "true",
        "missing_handoff": missing["RESUME_STATUS"] == "blocked",
        "ambiguous_workstream": ambiguous["ACTION"] == "request_clarification",
        "duplication_scan": (
            duplication["DUPLICATE_LOAD_ORDER_RULES"] == 0
            and duplication["DUPLICATE_RESUME_GATE_TEMPLATES"] == 0
        ),
        "token_budget": budget["DEFAULT_CONTEXT_REDUCTION_PCT>=60"] == "true",
    }

    full_matrix_pass = all(checks.values())
    return {
        "SCENARIO": "full_matrix",
        "FULL_MATRIX_PASS": _bool_str(full_matrix_pass),
        "RESULT_COUNT": len(checks),
        "RESULT_KEYS": list(checks.keys()),
        "CHECK_fresh_session_default": _bool_str(checks["fresh_session_default"]),
        "CHECK_quick_resume_project_legacy": _bool_str(
            checks["quick_resume_project_legacy"]
        ),
        "CHECK_quick_resume_module_legacy": _bool_str(
            checks["quick_resume_module_legacy"]
        ),
        "CHECK_explicit_escalation": _bool_str(checks["explicit_escalation"]),
        "CHECK_missing_handoff": _bool_str(checks["missing_handoff"]),
        "CHECK_ambiguous_workstream": _bool_str(checks["ambiguous_workstream"]),
        "CHECK_duplication_scan": _bool_str(checks["duplication_scan"]),
        "CHECK_token_budget": _bool_str(checks["token_budget"]),
        "LEGACY_PROJECT_LOAD_PATH": legacy_project["LOAD_PATH"],
        "LEGACY_MODULE_LOAD_PATH": legacy_module["LOAD_PATH"],
        "README_AUTOLOAD": fresh_default["README_AUTOLOAD"],
        "DEFAULT_CONTEXT_REDUCTION_PCT>=60": budget[
            "DEFAULT_CONTEXT_REDUCTION_PCT>=60"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the agent-memory startup contract against repository docs."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--workstream", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--input", default="")
    args = parser.parse_args()

    scenario = args.scenario
    if scenario == "fresh_session_default":
        payload = _scenario_fresh_session_default()
    elif scenario == "quick_resume_project":
        payload = _scenario_quick_resume(args.workstream)
    elif scenario == "quick_resume_module":
        payload = _scenario_quick_resume(args.workstream)
    elif scenario == "explicit_escalation":
        payload = _scenario_explicit_escalation(args.reason)
    elif scenario == "missing_handoff":
        payload = _scenario_missing_handoff(args.workstream)
    elif scenario == "ambiguous_workstream":
        payload = _scenario_ambiguous_workstream(args.input)
    elif scenario == "duplication_scan":
        payload = _scenario_duplication_scan()
    elif scenario == "token_budget":
        payload = _scenario_token_budget()
    elif scenario == "full_matrix":
        payload = _scenario_full_matrix()
    else:
        print(f"ERROR=unknown_scenario:{scenario}")
        return 1

    return _emit_lines(payload)


if __name__ == "__main__":
    raise SystemExit(main())
