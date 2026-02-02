---
name: nmp-agile
description: NMP agile process reference for PICs, labels, milestones, and estimation. Use when creating issues or MRs to look up component owners, available labels, milestone naming, and story point guidelines.
---

# NMP Agile Process Reference

This skill provides reference information for the NMP agile process, including component PICs (Pilots in Command), available labels, milestone conventions, and estimation guidelines.

For the full SDLC documentation, see [Agile Process](resources/agile-process.md).

## Component PICs (Pilots in Command)

Each component has a designated PIC responsible for that area. Use these when assigning issues or requesting MR reviews.

| Component Label | PIC | GitLab Username |
|-----------------|-----|-----------------|
| `component::auditor` | Paul Parkanzky | `pparkanzky` |
| `component::cli` | Piotr Mlocek | `pmlocek` |
| `component::customizer` | Aaron Gabow | `agabow` |
| `component::data-designer` | Mike Knepper | `mknepper` |
| `component::data store` | Matt Grossman | `mgrossman` |
| `component::deployment` | Drew Newberry | `dnewberry` |
| `component::dms` | Benjamin McCown | `bmccown` |
| `component::entity-store` (deprecated) | Max Dubrinsky | `mdubrinsky` |
| `component::entities` | Max Dubrinsky | `mdubrinsky` |
| `component::evaluator` | Sandy Chapman | `schapman` |
| `component::files` | Matt Grossman | `mgrossman` |
| `component::guardrails` | Aaron Gabow | `agabow` |
| `component::helm` | Drew Newberry | `dnewberry` |
| `component::inference-gateway` | Benjamin McCown | `bmccown` |
| `component::jobs` | Taylor Mutch | `tmutch` |
| `component::mcp` | Jeff Farris | `jfarris` |
| `component::models` | Benjamin McCown | `bmccown` |
| `component::python-sdk` | Piotr Mlocek | `pmlocek` |
| `component::safe-synthesizer` | Kendrick Boyd | `kboyd` |
| `component::sdk` | Piotr Mlocek | `pmlocek` |
| `component::secrets` | Taylor Mutch | `tmutch` |
| `component::studio` | Ashley Murray | `amurray` |
| `component::ui-research` | Jessica Buhl | `jbuhl` |
| `component::auth` | Piotr Mlocek | `pmlocek` |

### Parent/Rollup PICs

For escalation or cross-cutting issues spanning multiple components:

| Area | PIC | GitLab Username |
|------|-----|-----------------|
| Functional Microservices (evaluator, guardrails, customizer, etc.) | Aaron Gabow | `agabow` |
| Core Microservices (jobs, files, secrets, models, etc.) | Travis McKee | `tmckee` |
| Authentication & Authorization | Razvan Dinu | `rdinu` |
| DevX / CI/CD | Gustavo Prado Alkmim | `gpradoalkmim` |
| Infrastructure / Build / Systems | Philip Mattingly | `pmattingly` |

### Determining the PIC for an MR

When creating an MR, request review from the PIC(s) of the affected components:

1. **Identify affected components** from the files changed (e.g., changes in `services/evaluator/` → `component::evaluator`)
2. **Look up the PIC** from the table above
3. **For multiple components**, request review from each relevant PIC
4. **For cross-cutting changes** (CI/CD, build, auth), use the parent/rollup PICs

## Labels

### Type Labels (pick one)

- `bug` - Something is broken
- `feature` - New functionality
- `enhancement` - Improvement to existing functionality
- `documentation` - Docs-only changes
- `security` - Security-related

### Priority Labels (optional)

- `priority::showstopper` - Blocks release
- `priority::p0` - Critical
- `priority::p1` - High
- `priority::p2` - Normal

### Component Labels

- `component::auditor`
- `component::cli`
- `component::customizer`
- `component::data-designer`
- `component::datamodel`
- `component::data store`
- `component::deployment`
- `component::dms`
- `component::entities`
- `component::entity-store`
- `component::evaluator`
- `component::files`
- `component::garak`
- `component::guardrails`
- `component::hello-world`
- `component::helm`
- `component::inference-gateway`
- `component::intake`
- `component::jobs`
- `component::mcp`
- `component::models`
- `component::nemo-operator`
- `component::nim`
- `component::nim-proxy`
- `component::openapi`
- `component::platform`
- `component::python-sdk`
- `component::safe-synthesizer`
- `component::sdk`
- `component::secrets`
- `component::studio`
- `component::ui-research`
- `component::use-case`

### Development Phase Labels

- `phase::prd` - Product Requirements (PM-led requirements gathering)
- `phase::rfc` - Architectural Design (RFC writing, API review)
- `phase::dev` - Development (core development after POR)
- `phase::qa` - QA/Validation (testing, VDR, bug fixes)

### Origin Labels

- `origin::customer` - Customer-reported
- `origin::engineering` - Internal engineering
- `origin::qa` - QA/testing found
- `origin::vdr` - VDR review
- `origin::vpr` - VPR review

### Platform Labels

- `platform::aws`
- `platform::azure`
- `platform::dgx cloud`
- `platform::gcp`
- `platform::oci`
- `platform::on-prem`

### Severity Labels

- `severity::corruption`
- `severity::crash`
- `severity::enhancement`
- `severity::functional`
- `severity::performance`
- `severity::syscrash`

### Size Labels

- `size::small`
- `size::medium`
- `size::large`
- `size::xlarge`

### Type Labels (group level)

- `type::bug`
- `type::epic`
- `type::story`
- `type::task`

### UX Area Labels

- `ux::API`
- `ux::deployment`
- `ux::error-handling`
- `ux::onboarding`

### Informational Flags

- `blocked` - Blocked by external dependency
- `needs-refinement` - Needs discussion before work begins
- `needs-review` - Needs review
- `critical-path` - On critical path for a release
- `POR` - Plan of Record

### Other Labels

- `API` - API-related
- `api-improvement`
- `architecture`
- `CI/CD` - Pipeline/CI changes
- `cli`
- `complexity:moderate`
- `developer-experience`
- `devex`
- `feat:hardware platform`
- `feat:model support`
- `flywheel`
- `package::nmp_common`

## Milestones

Milestones follow the format `Platform YY.MM` (e.g., `Platform 26.02` for February 2026).

### Release Schedule

- Releases target the **first week of each month**
- **Code freeze** is **mid-month of the previous month** (~15th)
- Development for a release should be complete before code freeze

### Choosing a Milestone

| Scenario | Target Milestone |
|----------|------------------|
| Bug fix (weight 1-2) created before code freeze | Current release |
| Bug fix (weight 1-2) found during QA (after code freeze) | Current release |
| Bug fix (weight 1-2) created after release | Next release |
| Small feature (weight 3-5) | Next release (if time permits) or release after |
| Large feature (weight 8+) or needs RFC | 2+ releases out |
| Urgent/critical bug | Current release (if before code freeze) |

### Timeline Example (for Platform 26.03 release)

- ~Jan 15: Code freeze for Platform 26.02
- ~Feb 1: Platform 26.02 release
- ~Feb 15: Code freeze for Platform 26.03
- ~Mar 1: Platform 26.03 release
- Work targeting 26.03 must be merged before Feb 15

## Story Points (Estimation)

Estimate based on **complexity**, not just time. Assume coding assistants (Cursor, Claude Code) are used.

| Weight | Complexity | Time Estimate | Examples |
|--------|------------|---------------|----------|
| 1 | Simple | < 1 day | Simple refactoring, trivial bug fix, documentation |
| 2 | Minor | 1-2 days | Small code changes, writing tests |
| 3 | Moderate | 2-4 days | Simple multi-file change with minimal dependencies |
| 5 | Major | < 1 week | Typical code change with dependencies, testing, review |
| 8 | Difficult | 1-2 weeks | Large changes, complex interactions (consider breaking down) |

**Guidelines:**

- If complexity exceeds 8, break into smaller issues
- Consider: code changes, testing, review, documentation, dependencies
- Tasks that might take 1 day manually can often be completed in 2-3 hours with AI assistance

## Tee-Shirt Sizes (Epic Level)

For low-fidelity estimates on Epics:

- **X-Small** - 0.5 Developer Sprint
- **Small** - 1 Developer Sprint
- **Medium** - 2 Developer Sprints
- **Large** - 4 Developer Sprints
- **X-Large** - 6+ Developer Sprints
