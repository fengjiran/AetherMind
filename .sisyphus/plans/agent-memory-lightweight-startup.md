# Agent Memory Lightweight Startup

## TL;DR
> **Summary**: Redesign the agent memory startup contract so new conversations load only the minimum safe context by default, while detailed operational and architecture docs become explicit escalation-only references.
> **Deliverables**:
> - Canonical minimal startup contract in `AGENTS.md`
> - Reference-only `docs/agent/memory/README.md` and `docs/agent/memory_system.md`
> - Updated startup entrypoints and handoff contract compatible with aggressive low-token resume
> - Machine-checkable startup contract validator and token-budget regression checks
> **Effort**: Medium
> **Parallel**: YES - 3 waves
> **Critical Path**: 1 -> 2 -> 4 -> 7 -> 9 -> F1-F4

## Context
### Original Request
- The current memory system loads too much material when starting a new conversation, producing roughly 100k tokens of initial context.

### Interview Summary
- Priority is aggressive startup load reduction rather than conservative backward compatibility.
- It is acceptable to change the default startup contract so new sessions no longer read `docs/agent/memory/README.md` in full by default.
- Immediate focus is simplifying `AGENTS.md` and `docs/agent/memory/README.md`, while aligning the rest of the startup path around that change.
- Existing Chinese-language files must remain Chinese; do not translate or anglicize files that are already written in Chinese.

### Metis Review (gaps addressed)
- Plan keeps scope limited to startup contract, prompt entrypoints, handoff format, and verification tooling; it explicitly avoids a full memory-system redesign.
- Plan adds a machine-checkable validator so startup behavior is not enforced only by prose.
- Plan includes compatibility handling for existing handoffs and duplicate-rule elimination across prompts and examples.

## Work Objectives
### Core Objective
- Reduce new-session startup context cost by making the default startup path load only always-needed policy and workstream delta, while preserving safety-critical resume behavior.

### Deliverables
- A slimmed `AGENTS.md` with the only normative startup algorithm.
- A refactored `docs/agent/memory/README.md` that serves as escalation-only operational reference.
- A refactored `docs/agent/memory_system.md` that serves as architecture/reference-only documentation.
- Updated `docs/agent/prompts/quick_resume.md`, `docs/agent/prompts/new_session_template.md`, `docs/agent/prompts/README.md`, `docs/agent/memory/QUICKSTART.md`, `docs/agent/prompts/handoff_template.md`, and `docs/agent/handoff/README.md` aligned to the new contract.
- A startup contract validator under `tools/agent_memory/` with scenario checks and token-budget regression output.
- Compatibility guidance for legacy handoffs and an explicit bootstrap-readiness rule for low-context startup.

### Definition of Done (verifiable conditions with commands)
- `python3 tools/agent_memory/contract_check.py --scenario fresh_session_default` exits `0` and reports `README_AUTOLOAD=false`.
- `python3 tools/agent_memory/contract_check.py --scenario quick_resume_project --workstream project__docs-reorg` exits `0` and reports `README_AUTOLOAD=false`.
- `python3 tools/agent_memory/contract_check.py --scenario quick_resume_module --workstream ammalloc__thread_cache` exits `0` and reports `README_AUTOLOAD=false`.
- `python3 tools/agent_memory/contract_check.py --scenario explicit_escalation --reason need_operational_manual` exits `0` and reports `README_AUTOLOAD=true`.
- `python3 tools/agent_memory/contract_check.py --scenario duplication_scan` exits `0` and reports `DUPLICATE_LOAD_ORDER_RULES=0` and `DUPLICATE_RESUME_GATE_TEMPLATES=0`.
- `python3 tools/agent_memory/contract_check.py --scenario token_budget` exits `0` and reports `DEFAULT_CONTEXT_REDUCTION_PCT>=60`.

### Must Have
- One canonical startup contract, owned by `AGENTS.md`.
- Explicit escalation triggers for when deeper docs must be loaded.
- Resume gate and explicit user-confirmation rule preserved in the always-loaded path.
- Legacy handoffs continue to resume safely, even if they cannot use the minimal path.
- Handoffs remain runtime state only and stop duplicating long-term startup policy.
- Existing Chinese-language docs remain in Chinese unless a specific field or code-facing identifier must stay in English.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- No second normative definition of load order or resume-gate semantics outside `AGENTS.md`.
- No default startup dependency on `docs/agent/memory_system.md`.
- No tutorial-scale examples in the always-loaded path.
- No full rewrite of module/submodule memory taxonomy or retrieval architecture.
- No ambiguous “read more if needed” wording without explicit escalation triggers.
- No unnecessary translation of existing Chinese documentation into English.

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: tests-after with a dedicated Python validator plus focused grep/read checks for doc drift.
- QA policy: Every task includes agent-executed happy-path and edge-case scenarios.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: foundation contract and compatibility rules (`1`, `2`, `3`)
Wave 2: entrypoint and reference-doc rewrites (`4`, `5`, `6`)
Wave 3: validator, compatibility cleanup, and final evidence (`7`, `8`, `9`)

### Dependency Matrix (full, all tasks)
- `1` blocks `2`, `3`, `4`, `5`, `6`, `7`, `8`
- `2` blocks `4`, `5`, `6`, `7`, `8`
- `3` blocks `6`, `8`
- `4` blocks `8`
- `5` blocks `8`
- `6` blocks `8`
- `7` blocks `9`
- `8` blocks `9`
- `9` blocks `F1`, `F2`, `F3`, `F4`

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 3 tasks → `unspecified-high`, `writing`
- Wave 2 → 3 tasks → `writing`, `unspecified-high`
- Wave 3 → 3 tasks → `deep`, `writing`, `unspecified-high`
- Final Verification → 4 tasks → `oracle`, `unspecified-high`, `unspecified-high`, `deep`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Capture baseline startup contract and add validator scaffold

  **What to do**: Create `tools/agent_memory/contract_check.py` and any small supporting fixtures needed to encode the current and target startup scenarios as machine-checkable cases. The script must inspect repo docs and handoff metadata, emit deterministic key-value or JSON output, and support at least `fresh_session_default`, `quick_resume_project`, `quick_resume_module`, `explicit_escalation`, `missing_handoff`, `ambiguous_workstream`, `duplication_scan`, and `token_budget` scenarios. Record the pre-rewrite baseline so later tasks can prove reduction instead of claiming it.
  **Must NOT do**: Do not hardcode passing outputs without reading repo files; do not implement a general-purpose token estimator beyond startup docs; do not rewrite documentation in this task except the minimum command references needed by the validator.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: introduces a new repo-side validation utility with deterministic scenario logic.
  - Skills: `[]` — No special skill is needed beyond careful scripting and repo inspection.
  - Omitted: `['playwright']` — No browser workflow exists.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: `2`, `3`, `4`, `5`, `6`, `7`, `8` | Blocked By: none

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `AGENTS.md:137` — Current startup contract is embedded here and must be captured before rewriting.
  - Pattern: `docs/agent/memory/README.md:15` — Current memory hierarchy and load-order reference.
  - Pattern: `docs/agent/prompts/quick_resume.md:121` — Current duplicated load-order contract and resume-gate templates.
  - Pattern: `docs/agent/prompts/new_session_template.md:38` — Explicit startup path that currently requires README in the load chain.
  - Pattern: `docs/agent/prompts/handoff_template.md:53` — Handoff body shape that the validator must understand.
  - Pattern: `docs/agent/handoff/workstreams/project__agent-memory-v1.1/20260312T080000Z--ses_31f8f5b8affe1bnCOcehtKLMvq--sisyphus.md` — Legacy active handoff example for compatibility checks.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `tools/agent_memory/contract_check.py` exists and each required scenario exits `0` with deterministic output keys.
  - [ ] Running the baseline scenarios writes evidence under `.sisyphus/evidence/` and captures pre-rewrite startup expectations.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```text
  Scenario: Baseline contract matrix
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario fresh_session_default > .sisyphus/evidence/task-1-startup-baseline.json` and `python3 tools/agent_memory/contract_check.py --scenario quick_resume_project --workstream project__agent-memory-v1.1 > .sisyphus/evidence/task-1-quick-project.json`
    Expected: Both commands exit 0; evidence files exist; outputs contain current-contract fields including `README_AUTOLOAD=` and `RESUME_STATUS=`
    Evidence: .sisyphus/evidence/task-1-startup-baseline.json

  Scenario: Missing workstream edge case
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario missing_handoff --workstream project__missing > .sisyphus/evidence/task-1-missing-handoff.json`
    Expected: Command exits 0 and output contains `RESUME_STATUS=blocked` or `RESUME_STATUS=partial` exactly as documented by the implemented baseline rule
    Evidence: .sisyphus/evidence/task-1-missing-handoff.json
  ```

  **Commit**: YES | Message: `test(agent-memory): capture startup contract baseline` | Files: `tools/agent_memory/`, `.sisyphus/evidence/`

- [x] 2. Slim `AGENTS.md` into the sole normative startup contract

  **What to do**: Rewrite section 10 of `AGENTS.md` so it contains only the always-loaded startup rules: workstream-type resolution, minimal default load path, explicit user-confirmation gate, active-handoff selection rule, handoff-template vs runtime-state distinction, and explicit escalation triggers for loading deeper references or module docs. Preserve the existing project build/test/coding guidance outside section 10. Make `AGENTS.md` the only normative definition of startup behavior, with direct language that other docs should reference rather than repeat.
  **Must NOT do**: Do not delete non-memory project guidance; do not leave duplicated full load order in other files once this task lands; do not rely on vague “read README if needed” wording without naming exact escalation conditions.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: this is a precision rewrite of the canonical policy document.
  - Skills: `[]` — No special skill is needed.
  - Omitted: `['playwright']` — No UI behavior is involved.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: `4`, `5`, `6`, `7`, `8` | Blocked By: `1`

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `AGENTS.md:137` — Existing section to replace with the minimal contract.
  - Pattern: `docs/agent/memory/README.md:20` — Current project-level vs module-level load rules to preserve semantically.
  - Pattern: `docs/agent/prompts/quick_resume.md:170` — Resume-gate wording currently duplicated here and should collapse into `AGENTS.md` ownership.
  - Pattern: `docs/agent/prompts/handoff_template.md:72` — Runtime-state storage rules that still need a concise pointer from `AGENTS.md`.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `AGENTS.md` defines the only normative startup algorithm and explicitly states that `docs/agent/memory/README.md` is escalation-only for operational detail.
  - [ ] The validator scenario `fresh_session_default` reports `README_AUTOLOAD=false` after this rewrite.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```text
  Scenario: Default startup contract after AGENTS rewrite
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario fresh_session_default > .sisyphus/evidence/task-2-fresh-default.json`
    Expected: Command exits 0 and output contains `README_AUTOLOAD=false`, `MEMORY_SYSTEM_AUTOLOAD=false`, and a load path rooted in `AGENTS.md`
    Evidence: .sisyphus/evidence/task-2-fresh-default.json

  Scenario: Ambiguous workstream remains clarification-first
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario ambiguous_workstream --input docs-reorg > .sisyphus/evidence/task-2-ambiguous.json`
    Expected: Command exits 0 and output contains `ACTION=request_clarification`; no auto-invented slug appears
    Evidence: .sisyphus/evidence/task-2-ambiguous.json
  ```

  **Commit**: YES | Message: `docs(agent-memory): define minimal startup contract in agents` | Files: `AGENTS.md`

- [x] 3. Define bootstrap-ready handoff compatibility and delta-only handoff guidance

  **What to do**: Update `docs/agent/prompts/handoff_template.md` and `docs/agent/handoff/README.md` so handoffs are explicitly delta-only runtime state, not long-term policy carriers. Add a compatibility rule for low-context startup by introducing a `bootstrap_ready` marker in handoff metadata; missing marker must default to `false` for legacy files. Define exactly when a handoff qualifies for minimal startup and when startup must escalate to module docs or README.
  **Must NOT do**: Do not break parsing of existing `schema_version: 1.1` handoffs; do not require manual migration of historical handoffs before the new startup path can function.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: this is a contract and compatibility rewrite across documentation templates.
  - Skills: `[]` — No special skill is needed.
  - Omitted: `['playwright']` — No browser validation exists.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: `6`, `8` | Blocked By: `1`

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `docs/agent/prompts/handoff_template.md:5` — Existing frontmatter contract to extend safely.
  - Pattern: `docs/agent/handoff/README.md` — Storage and selection rules that must reflect bootstrap compatibility.
  - Pattern: `docs/agent/memory/README.md:145` — Current backward-compatibility defaults for handoff fields.
  - Pattern: `docs/agent/handoff/workstreams/project__agent-memory-v1.1/20260312T080000Z--ses_31f8f5b8affe1bnCOcehtKLMvq--sisyphus.md` — Real legacy handoff used to confirm missing `bootstrap_ready` fallback.

  **Acceptance Criteria** (agent-executable only):
  - [ ] Template and storage docs specify `bootstrap_ready` semantics and explicitly define missing-as-false legacy behavior.
  - [ ] Validator can distinguish bootstrap-ready vs legacy handoffs and route startup accordingly.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```text
  Scenario: Legacy handoff forces expanded path
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario quick_resume_project --workstream project__agent-memory-v1.1 > .sisyphus/evidence/task-3-legacy-handoff.json`
    Expected: Command exits 0 and output marks the legacy handoff as not bootstrap-ready; startup result includes the documented compatibility fallback
    Evidence: .sisyphus/evidence/task-3-legacy-handoff.json

  Scenario: Bootstrap-ready happy path
    Tool: Bash
    Steps: Create or use a fixture handoff with `bootstrap_ready: true`, then run `python3 tools/agent_memory/contract_check.py --scenario quick_resume_project --workstream project__docs-reorg > .sisyphus/evidence/task-3-bootstrap-ready.json`
    Expected: Command exits 0 and output shows minimal startup path with `README_AUTOLOAD=false`
    Evidence: .sisyphus/evidence/task-3-bootstrap-ready.json
  ```

  **Commit**: YES | Message: `docs(handoff): add bootstrap-ready startup compatibility` | Files: `docs/agent/prompts/handoff_template.md`, `docs/agent/handoff/README.md`

- [x] 4. Refactor `docs/agent/memory/README.md` into escalation-only operational reference

  **What to do**: Rewrite `docs/agent/memory/README.md` so it stops acting like a default startup document. Keep detailed operational rules that are still needed for maintenance, schema details, naming, storage, conflict handling, and writeback flows, but remove duplicated startup algorithms, duplicated resume-gate templates, and repeated examples that belong in prompts or examples. Add explicit language near the top stating that startup behavior is owned by `AGENTS.md` and README is consulted only on escalation.
  **Must NOT do**: Do not delete required handoff/storage rules that the writer path still needs; do not leave any section implying README must be fully read in every fresh session.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: high-precision document reduction and role clarification.
  - Skills: `[]` — No special skill is required.
  - Omitted: `['playwright']` — No UI interaction exists.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: `8` | Blocked By: `2`

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `docs/agent/memory/README.md:1` — Current heavy operational spec to slim.
  - Pattern: `AGENTS.md:137` — New canonical startup contract that README must defer to.
  - Pattern: `docs/agent/memory_system.md:47` — Architecture-vs-operations split already described and should be tightened.
  - Pattern: `docs/agent/prompts/quick_resume.md:121` — Startup algorithm currently duplicated outside README.

  **Acceptance Criteria** (agent-executable only):
  - [ ] README states that startup behavior is defined by `AGENTS.md` and is not a default full-read requirement.
  - [ ] Duplication scan shows README no longer contains a second normative load-order or resume-gate template.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```text
  Scenario: README is reference-only
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario explicit_escalation --reason need_operational_manual > .sisyphus/evidence/task-4-escalation.json`
    Expected: Command exits 0 and output contains `README_AUTOLOAD=true` only for explicit escalation, not for default startup
    Evidence: .sisyphus/evidence/task-4-escalation.json

  Scenario: Duplicate startup rules removed from README
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario duplication_scan > .sisyphus/evidence/task-4-duplication.json`
    Expected: Command exits 0 and output shows no duplicate normative load-order block remaining in README
    Evidence: .sisyphus/evidence/task-4-duplication.json
  ```

  **Commit**: YES | Message: `docs(memory): make readme escalation-only` | Files: `docs/agent/memory/README.md`

- [x] 5. Move architecture/tutorial content out of the startup path

  **What to do**: Refactor `docs/agent/memory_system.md` into architecture/reference-only material and convert `docs/agent/memory/QUICKSTART.md` from a long tutorial into a terse operator cheat sheet or startup map. Both files must point to `AGENTS.md` for startup behavior and avoid re-embedding the default load path. Preserve useful architecture rationale and examples, but move them out of the always-loaded path and remove tutorial-scale startup examples that train agents into loading everything.
  **Must NOT do**: Do not delete architecture rationale needed for maintainers; do not leave `QUICKSTART.md` as a full walk-through that contradicts the new minimal contract.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: this is document-role refactoring with substantial content compression.
  - Skills: `[]` — No special skill is needed.
  - Omitted: `['playwright']` — No browser work exists.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: `8` | Blocked By: `2`

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `docs/agent/memory_system.md:189` — Current startup workflow examples that must no longer imply default loading.
  - Pattern: `docs/agent/memory/QUICKSTART.md:106` — Long startup tutorial currently encoding the old contract.
  - Pattern: `AGENTS.md:137` — Canonical startup contract to reference instead of restating.
  - Pattern: `docs/agent/prompts/README.md` — Prompt documentation should remain consistent with these new roles.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `docs/agent/memory_system.md` is clearly architecture/reference-only and no validator scenario treats it as default autoload.
  - [ ] `docs/agent/memory/QUICKSTART.md` is concise and no longer embeds the old full-read startup tutorial.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```text
  Scenario: Memory system doc removed from autoload path
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario fresh_session_default > .sisyphus/evidence/task-5-fresh-default.json`
    Expected: Command exits 0 and output contains `MEMORY_SYSTEM_AUTOLOAD=false`
    Evidence: .sisyphus/evidence/task-5-fresh-default.json

  Scenario: Quickstart no longer teaches legacy startup
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario duplication_scan > .sisyphus/evidence/task-5-duplication.json`
    Expected: Command exits 0 and output confirms no duplicate normative load-order tutorial remains in `docs/agent/memory/QUICKSTART.md`
    Evidence: .sisyphus/evidence/task-5-duplication.json
  ```

  **Commit**: YES | Message: `docs(memory): move architecture and tutorial content out of startup path` | Files: `docs/agent/memory_system.md`, `docs/agent/memory/QUICKSTART.md`

- [x] 6. Rewrite prompt entrypoints to reference the canonical contract instead of restating it

  **What to do**: Update `docs/agent/prompts/quick_resume.md`, `docs/agent/prompts/new_session_template.md`, and `docs/agent/prompts/README.md` so they stop duplicating full load-order lists, resume-gate templates, and edge-case policy already owned by `AGENTS.md`. Each prompt should say what it is for, what extra input it requires, when it escalates to README, and how it interacts with handoff/bootstrap-ready behavior. Keep only prompt-specific instructions; move shared startup logic out.
  **Must NOT do**: Do not preserve a second source of truth for startup order inside prompt docs; do not remove the explicit user-confirmation checkpoint.

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: coordinated prompt and doc rewrites with cross-file consistency requirements.
  - Skills: `[]` — No special skill is needed.
  - Omitted: `['playwright']` — No UI interaction exists.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: `8` | Blocked By: `2`, `3`

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `docs/agent/prompts/quick_resume.md:9` — Current hard constraint and duplicated startup flow.
  - Pattern: `docs/agent/prompts/quick_resume.md:121` — Current fixed load-order section to eliminate as normative policy.
  - Pattern: `docs/agent/prompts/new_session_template.md:38` — Current explicit startup instructions that still inline README-first behavior.
  - Pattern: `docs/agent/prompts/README.md` — Entry-point catalog that must describe the new split correctly.
  - Pattern: `AGENTS.md:137` — Canonical startup contract prompts must reference.
  - Pattern: `docs/agent/prompts/handoff_template.md:72` — Handoff semantics prompts must stay compatible with.

  **Acceptance Criteria** (agent-executable only):
  - [ ] Prompt docs no longer contain a second normative load-order algorithm or duplicated resume-gate template.
  - [ ] Prompt docs preserve explicit user confirmation and documented escalation to README only when needed.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```text
  Scenario: Quick resume project path uses canonical contract
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario quick_resume_project --workstream project__docs-reorg > .sisyphus/evidence/task-6-quick-project.json`
    Expected: Command exits 0 and output contains `README_AUTOLOAD=false` and `CONFIRMATION_GATE=true`
    Evidence: .sisyphus/evidence/task-6-quick-project.json

  Scenario: Duplication removed from prompt docs
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario duplication_scan > .sisyphus/evidence/task-6-duplication.json`
    Expected: Command exits 0 and output contains `DUPLICATE_RESUME_GATE_TEMPLATES=0`
    Evidence: .sisyphus/evidence/task-6-duplication.json
  ```

  **Commit**: YES | Message: `docs(prompts): point startup entrypoints at canonical contract` | Files: `docs/agent/prompts/quick_resume.md`, `docs/agent/prompts/new_session_template.md`, `docs/agent/prompts/README.md`

- [x] 7. Enforce duplication and token-budget regressions in the validator

  **What to do**: Extend `tools/agent_memory/contract_check.py` so it not only models startup scenarios but also scans the relevant docs for forbidden duplicate policy blocks and calculates a consistent startup-size budget using file-byte or token-estimation heuristics. Define exactly which files count toward default startup, which count only on escalation, and how `DEFAULT_CONTEXT_REDUCTION_PCT` is measured relative to the captured baseline.
  **Must NOT do**: Do not use unverifiable “looks smaller” judgment; do not let the token-budget result vary nondeterministically across runs.

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: deterministic analysis logic and regression-proof validation are required.
  - Skills: `[]` — No special skill is needed.
  - Omitted: `['playwright']` — No browser workflow exists.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: `8` | Blocked By: `1`, `2`

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `tools/agent_memory/contract_check.py` — Validator created in task 1 and extended here.
  - Pattern: `AGENTS.md:137` — Canonical contract used to determine default startup set.
  - Pattern: `docs/agent/memory/README.md:1` — Reference-only doc that should now count only on escalation.
  - Pattern: `docs/agent/memory_system.md:1` — Architecture-only doc removed from default startup.
  - Pattern: `docs/agent/prompts/quick_resume.md:121` — Known duplication hotspot to ensure it no longer persists.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `duplication_scan` returns zero duplicate normative contract blocks across the targeted docs.
  - [ ] `token_budget` reports at least 60% reduction against the baseline captured in task 1.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```text
  Scenario: Duplicate contract regression scan
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario duplication_scan > .sisyphus/evidence/task-7-duplication.json`
    Expected: Command exits 0 and output contains `DUPLICATE_LOAD_ORDER_RULES=0` and `DUPLICATE_RESUME_GATE_TEMPLATES=0`
    Evidence: .sisyphus/evidence/task-7-duplication.json

  Scenario: Startup budget reduction
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario token_budget > .sisyphus/evidence/task-7-token-budget.json`
    Expected: Command exits 0 and output contains `DEFAULT_CONTEXT_REDUCTION_PCT>=60`
    Evidence: .sisyphus/evidence/task-7-token-budget.json
  ```

  **Commit**: YES | Message: `test(agent-memory): enforce duplication and token budget regressions` | Files: `tools/agent_memory/contract_check.py`

- [x] 8. Finalize compatibility sweep across startup-facing docs

  **What to do**: Do a final cross-file cleanup so every startup-related doc consistently reflects the new contract, including `docs/agent/handoff/README.md`, active project handoff guidance, and any references that still imply README-first startup. Ensure legacy handoffs remain readable through documented fallback behavior, and capture final evidence for the full scenario matrix after all rewrites. This task is the integration pass that closes remaining wording drift after tasks 3-7 land.
  **Must NOT do**: Do not quietly change startup semantics beyond what the plan specifies; do not leave stale examples or references to the retired full-read path.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: multi-file integration and consistency sweep across docs and validation output.
  - Skills: `[]` — No special skill is needed.
  - Omitted: `['playwright']` — No UI interaction exists.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: `9` | Blocked By: `1`, `3`, `4`, `5`, `6`

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `docs/agent/handoff/README.md` — Must match the final handoff and compatibility story.
  - Pattern: `docs/agent/prompts/README.md` — Must describe the new entrypoint semantics correctly.
  - Pattern: `docs/agent/memory/project.md:86` — Slug mapping remains startup-relevant and must not accumulate temporary workflow prose.
  - Pattern: `docs/agent/handoff/workstreams/project__agent-memory-v1.1/20260312T080000Z--ses_31f8f5b8affe1bnCOcehtKLMvq--sisyphus.md` — Legacy workstream used to verify compatibility fallback.
  - Pattern: `tools/agent_memory/contract_check.py` — Final evidence runs depend on the finished validator.

  **Acceptance Criteria** (agent-executable only):
  - [ ] No startup-facing doc still teaches the retired full-read contract.
  - [ ] Compatibility notes for legacy handoffs are consistent across handoff and prompt docs.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```text
  Scenario: Legacy workstream wording remains compatible
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario quick_resume_project --workstream project__agent-memory-v1.1 > .sisyphus/evidence/task-8-legacy-project.json`
    Expected: Command exits 0 and output shows a valid, documented fallback path instead of failure or silent under-loading
    Evidence: .sisyphus/evidence/task-8-legacy-project.json

  Scenario: No stale full-read references remain
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario duplication_scan > .sisyphus/evidence/task-8-duplication.json`
    Expected: Command exits 0 and output confirms no startup-facing doc still encodes the retired full-read contract
    Evidence: .sisyphus/evidence/task-8-duplication.json
  ```

  **Commit**: YES | Message: `docs(agent-memory): finalize low-context startup compatibility sweep` | Files: `docs/agent/`

- [x] 9. Run full startup matrix and publish rollout evidence

  **What to do**: After all rewrites and compatibility cleanup land, run the full validator matrix and publish the final evidence bundle proving the new startup contract, duplication elimination, and token-budget reduction. This task is the release-gate proof step: every required scenario must pass from the final repo state, and the evidence files must be complete enough for later audit.
  **Must NOT do**: Do not modify docs or validator logic in this task except for trivial fixes needed to get the already-planned checks to execute; this is a verification-and-proof pass, not another rewrite pass.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: final integration verification across the completed documentation and validator.
  - Skills: `[]` — No special skill is needed.
  - Omitted: `['playwright']` — No UI interaction exists.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: `F1`, `F2`, `F3`, `F4` | Blocked By: `7`, `8`

  **References** (executor has NO interview context — be exhaustive):
  - Pattern: `tools/agent_memory/contract_check.py` — Final evidence runs depend on the completed validator.
  - Pattern: `AGENTS.md:137` — Final startup contract under test.
  - Pattern: `docs/agent/prompts/quick_resume.md` — Final quick-resume contract under test.
  - Pattern: `docs/agent/memory/README.md` — Final escalation-only reference under test.
  - Pattern: `docs/agent/handoff/workstreams/project__agent-memory-v1.1/20260312T080000Z--ses_31f8f5b8affe1bnCOcehtKLMvq--sisyphus.md` — Legacy handoff used in the final fallback proof.

  **Acceptance Criteria** (agent-executable only):
  - [ ] Full scenario matrix passes after all doc rewrites and compatibility notes are in place.
  - [ ] Final evidence includes duplication and token-budget outputs from the finished repo state.

  **QA Scenarios** (MANDATORY — task incomplete without these):
  ```text
  Scenario: Full matrix pass
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario full_matrix > .sisyphus/evidence/task-9-full-matrix.json`
    Expected: Command exits 0 and output reports all startup scenarios passing, including legacy fallback and explicit escalation
    Evidence: .sisyphus/evidence/task-9-full-matrix.json

  Scenario: Final token-budget proof
    Tool: Bash
    Steps: Run `python3 tools/agent_memory/contract_check.py --scenario token_budget > .sisyphus/evidence/task-9-token-budget.json`
    Expected: Command exits 0 and output contains `DEFAULT_CONTEXT_REDUCTION_PCT>=60` from the finished repo state
    Evidence: .sisyphus/evidence/task-9-token-budget.json
  ```

  **Commit**: YES | Message: `test(agent-memory): publish final startup rollout evidence` | Files: `.sisyphus/evidence/`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit 1: capture baseline startup-contract scenarios and add validator scaffolding.
- Commit 2: slim `AGENTS.md` and refactor `docs/agent/memory/README.md` / `docs/agent/memory_system.md` into summary-vs-reference roles.
- Commit 3: update prompt entrypoints, quickstart material, and handoff docs/schema compatibility.
- Commit 4: add duplication/token-budget enforcement and finalize migration cleanup.

## Success Criteria
- New-session default startup no longer depends on loading `docs/agent/memory/README.md` or `docs/agent/memory_system.md` in full.
- Startup safety rules remain explicit, testable, and visible in the always-loaded path.
- Prompt/example docs no longer restate a conflicting full-read startup contract.
- Legacy workstreams remain resumable through an explicit compatibility path.
- Automated checks prevent load-order drift and token-budget regression.
