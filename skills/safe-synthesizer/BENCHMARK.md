# Skill Benchmark: safe-synthesizer

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `safe-synthesizer`
- Validation status: `passed`
- Evaluation date: 2026-11-09
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 6 evaluation tasks (4 positive, 2 negative)
- Dataset digest: `sha256:df66b3ebf082deded7ee2c4e547d04fa9f1df1eb26006955db8a554ef2265c63` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| safe-synthesizer | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| safe-synthesizer | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| safe-synthesizer | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| safe-synthesizer | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| safe-synthesizer | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `safe-synthesizer`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.0% — baseline ran, but no comparable score was available; uplift unavailable | 90.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 56.7% → 100.0% (+43.3 points) | 80.0% → 100.0% (+20.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 37.1% → 98.8% (+61.7 points) | 51.4% → 89.1% (+37.7 points) |
| Efficiency | 81.2% — baseline ran, but no comparable score was available; uplift unavailable | 69.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.4% → 97.8% (-0.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.7% → 89.7% (-1.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 847,619 | 644,755 | +202,864 | +31.46% | skill 6/6; base 6/6 |
| claude-code | safe-synthesizer-artifact-locations | 259,374 | 179,004 | +80,370 | +44.90% | skill 1/1; base 1/1 |
| claude-code | safe-synthesizer-config-num-records | 108,264 | 143,178 | -34,914 | -24.39% | skill 1/1; base 1/1 |
| claude-code | safe-synthesizer-diagnose-generation-oom | 257,202 | 107,662 | +149,540 | +138.90% | skill 1/1; base 1/1 |
| claude-code | safe-synthesizer-negative-general-dp | 37,675 | 36,514 | +1,161 | +3.18% | skill 1/1; base 1/1 |
| claude-code | safe-synthesizer-negative-react-ui | 35,425 | 35,248 | +177 | +0.50% | skill 1/1; base 1/1 |
| claude-code | safe-synthesizer-run-dp-cli | 149,679 | 143,149 | +6,530 | +4.56% | skill 1/1; base 1/1 |
| codex | All cases | 262,125 | 163,219 | +98,906 | +60.60% | skill 6/6; base 6/6 |
| codex | safe-synthesizer-artifact-locations | 73,080 | 35,047 | +38,033 | +108.52% | skill 1/1; base 1/1 |
| codex | safe-synthesizer-config-num-records | 51,338 | 21,746 | +29,592 | +136.08% | skill 1/1; base 1/1 |
| codex | safe-synthesizer-diagnose-generation-oom | 54,474 | 21,756 | +32,718 | +150.39% | skill 1/1; base 1/1 |
| codex | safe-synthesizer-negative-general-dp | 18,148 | 13,584 | +4,564 | +33.60% | skill 1/1; base 1/1 |
| codex | safe-synthesizer-negative-react-ui | 13,728 | 17,949 | -4,221 | -23.52% | skill 1/1; base 1/1 |
| codex | safe-synthesizer-run-dp-cli | 51,357 | 53,137 | -1,780 | -3.35% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,109,744 | 807,974 | +301,770 | +37.35% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 9 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/safe-synthesizer/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (408 chars, recommend 50-150) (`skills/safe-synthesizer/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Broad description without negative triggers may cause over-triggering (`skills/safe-synthesizer/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/safe-synthesizer/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/safe-synthesizer/SKILL.md`)
- 4 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
