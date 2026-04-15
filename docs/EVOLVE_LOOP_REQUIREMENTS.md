# Evolve Loop Analysis — Requirements

## 1. Purpose

A tool to analyze evolve loop experiment traces and answer: **"What works well and what doesn't for this experiment?"**

It operates on data produced by evolutionary coding frameworks (SkyDiscover, ShinkaEvolve, OpenEvolve) where an LLM iteratively mutates a program, an evaluator scores each mutation, and the best programs are kept across generations.

---

## 2. Core Analytical Dimensions

### 2.1 Evolutionary Effectiveness

- Track `score_delta` per iteration (child_score − parent_score)
- Identify breakthrough iterations (score jump above a configurable threshold)
- Correlate effectiveness with:
  - Mutation type (diff / full rewrite / cross)
  - LLM model used
  - Island / population
  - Parent selection strategy

### 2.2 Instruction Compliance

Did the LLM code-writer respect the instructions it was given?

**Deterministic checks (zero LLM cost):**

| Check | What it detects |
|---|---|
| EVOLVE-BLOCK boundary | Did the mutation touch code outside the marked block? |
| Format validity | Is the diff parseable / is the full rewrite a complete file? |
| API preservation | Were function signatures maintained? |

**LLM-as-judge (semantic compliance):**

- Given the system message constraints and the generated code, were all constraints respected?
- Identify specific violations: which constraint was broken and where in the code
- Rate overall compliance: `fully_compliant` / `partially_compliant` / `non_compliant`

### 2.3 Stagnation Detection

Detect sequences of consecutive iterations with no progress toward the goal.

**Alert threshold:** configurable, default = 10 consecutive non-improving iterations.

**Stagnation subtypes:**

| Type | Signal |
|---|---|
| Score plateau | `child_score ≤ parent_score` repeatedly |
| Identical output | Generated code is identical to parent code |
| Format failure loop | Same format error repeating across iterations |
| Compliance violation loop | Same constraint violated repeatedly |
| Micro-improvement trap | Improvement < ε per iteration (false progress) |

**Per stagnation period, the tool must capture:**

- Start and end iteration
- Length of the streak
- The full sequence of what was attempted (iteration, mutation type, failure type, score delta)
- Dominant failure type within the streak
- Recovery event: which iteration broke the stagnation, what mutation type, what model, what paradigm shift
- Best score before stagnation started and after recovery

**Alert severity levels:**

| Condition | Severity |
|---|---|
| Streak ≥ threshold | Warning |
| Streak ≥ 2× threshold | Critical |
| Dominant failure = `identical_output` + streak ≥ threshold | Critical |
| Dominant failure = `compliance_violation` + streak ≥ threshold | High |
| Multiple stagnation periods in same run | Warning |

Each alert must include: sequence visualization, dominant failure type, LLM root cause analysis, and a concrete recommendation.

### 2.4 Evaluator Output Analysis

The evaluator is a first-class input — not just a source of scores.

**Failure mode classification (deterministic):**

| Mode | Signal |
|---|---|
| `crash` | Evaluator raised an exception |
| `timeout` | Exceeded the time limit |
| `wrong_output` | Correctness metric = 0 |
| `worse` | Ran correctly but combined_score dropped |
| `partial` | Some sub-metrics improved, combined_score did not |
| `success` | combined_score improved |

**Sub-metric divergence:**

- When `combined_score` does not improve, detect whether individual sub-metrics did
- This surfaces hidden partial progress the primary score obscures
- Example output: *"11 iterations improved execution_time without improving combined_score — metric weighting may be suppressing useful exploration"*

**Cascade stage analysis:**

- When cascade evaluation is enabled, identify at which stage (1 / 2 / 3) most failures occur
- High Stage-1 failure rate indicates syntactically broken or non-runnable LLM output — a system message problem, not an evolution strategy problem

**Evaluator artifact analysis (LLM-as-judge):**

- Cluster recurring content from `stderr`, `build_warnings`, and `llm_feedback` across failed iterations
- Identify recurring error patterns (e.g., *"23/38 failures: IndexError on line 42 — boundary condition not preserved by LLM"*)
- Produce actionable system message recommendations derived from error patterns

**Evaluator determinism check:**

- When `num_runs > 1`, flag iterations where score variance across runs is high for the same code
- Stagnation caused by evaluator noise is a different problem than LLM quality and must be distinguished

---

## 3. Alert System

Alerts must be actionable. Each alert contains:

1. Experiment identifier (run name, algorithm, model)
2. Stagnation period (start iteration, end iteration, length)
3. Sequence of what was attempted (compact table)
4. Dominant failure type and secondary types
5. LLM-generated root cause explanation
6. Concrete recommendation (system message change, strategy change, evaluator change)

---

## 4. Output and Reporting

### Per Stagnation Period
- Sequence table (iteration, mutation type, failure type, score delta)
- Root cause category: `LOCAL_OPTIMUM` / `INSTRUCTION_CONFUSION` / `APPROACH_EXHAUSTION` / `FORMAT_ISSUE` / `EVALUATOR_NOISE` / `OTHER`
- LLM explanation and recommendation

### Per Run (Aggregate)
- Total stagnation periods and total stagnated iterations as % of budget
- Alert-level streaks (count and locations)
- Most common stagnation type
- Fastest and slowest recovery
- Runs that never recovered (stagnation hit end of budget)
- Systemic patterns (e.g., *"stagnation always occurs after score > 0.7 — likely local optimum"*)
- Cross-tabulation: compliance level × score improvement rate
- Failure budget breakdown (% of iterations per failure mode)
- Evaluator artifact cluster summary

### Dashboard
- Score progression curve with stagnation periods highlighted and breakthroughs annotated
- Alert panel listing all triggered alerts
- Mutation success rate by type and by LLM model
- "What works" issue clusters
- "What doesn't" failure clusters
- Cascade stage failure distribution

---

## 5. Input Format

### Primary Format: JSONL

One record per iteration. Required and optional fields:

| Field | Type | Required | Source |
|---|---|---|---|
| `iteration` | int | Yes | Framework |
| `parent_id` | uuid | Yes | Framework |
| `child_id` | uuid | Yes | Framework |
| `mutation_type` | string | Yes | Framework |
| `llm_model` | string | Yes | Framework |
| `system_message` | string | Yes | Framework |
| `prompt` | string | Yes | Framework |
| `llm_response` | string | Yes | Framework |
| `parent_code` | string | Yes | Framework |
| `child_code` | string | Yes | Framework |
| `diff` | string | If diff-based | Framework |
| `parent_score` | float | Yes | Framework |
| `child_score` | float | Yes | Evaluator |
| `score_delta` | float | Derived | Computed |
| `evaluation_status` | enum | Derived | Computed |
| `cascade_stage_failed` | int / null | If cascade enabled | Evaluator |
| `evaluator_metrics` | dict | Yes | Evaluator |
| `evaluator_artifacts` | dict | If available | Evaluator |
| `evolved_block_only` | bool | Derived | Compliance check |
| `format_valid` | bool | Derived | Compliance check |
| `signature_preserved` | bool | Derived | Compliance check |
| `stagnation_streak_id` | string | Derived | Stagnation detector |
| `streak_position` | int | Derived | Stagnation detector |
| `island_id` | int | If multi-island | Framework |

### Supported Source Frameworks

| Framework | Adapter reads |
|---|---|
| SkyDiscover | `checkpoints/checkpoint_N/` directories + run logs |
| ShinkaEvolve | SQLite database (`dbase.py` schema) |
| OpenEvolve | Checkpoint directories + `evolution_trace.jsonl` (requires `evolution_trace.enabled: true` in config) |
| Generic | Direct JSONL (no adapter needed) |

---

## 6. Build Strategy

**Extend CLEAR** rather than build a new tool, to reuse:
- LLM-as-judge pipeline with caching and resume
- Three inference backends (LangChain / LiteLLM / Endpoint)
- Issue synthesis and aggregation
- Streamlit dashboard infrastructure

New components added to CLEAR are described in the Design document.
