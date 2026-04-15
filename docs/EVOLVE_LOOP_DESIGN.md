# Evolve Loop Analysis — Design

## 1. Overview

This document describes the design for extending CLEAR with an `evolve_loop` analysis mode. The extension reuses CLEAR's existing LLM-as-judge pipeline, inference backends, caching system, and dashboard infrastructure. New components are confined to input adapters, a stagnation detector, a compliance checker, an evaluator analyzer, a new use case class, and dashboard extensions.

---

## 2. Architecture

### 2.1 High-Level Flow

```
Input (JSONL or framework checkpoint)
        │
        ▼
 [Adapter Layer]
  checkpoint_adapter.py
  → normalizes to standard JSONL
        │
        ▼
 [Step 0] Stagnation Detection          deterministic — zero LLM cost
  stagnation_detector.py
  → tags each record: streak_id, streak_position, is_alert
  → produces List[StagnationPeriod]
        │
        ▼
 [Step 1] Evaluator Failure Classification   deterministic
  evaluator_analyzer.py
  → failure_mode: crash/timeout/wrong_output/worse/partial/success
  → cascade_stage_analysis
  → sub_metric_divergence flags
  → evaluator_variance flags
        │
        ▼
 [Step 2] Compliance Checks             deterministic
  compliance_checker.py
  → evolved_block_only (bool)
  → format_valid (bool)
  → signature_preserved (bool)
        │
        ▼
 [Step 3] LLM Judge — stagnation root cause
  EvolveLoopUseCase (per stagnation period)
  → input: sequence + system_message + parent_code + evaluator artifacts
  → output: root_cause_category + explanation + recommendation
        │
        ▼
 [Step 4] LLM Judge — evaluator artifact clustering
  EvolveLoopUseCase (across all failed iterations)
  → input: aggregated stderr / build_warnings / llm_feedback
  → output: recurring error patterns + system message fixes
        │
        ▼
 [Step 5] LLM Judge — per-mutation quality
  EvolveLoopUseCase (non-stagnation iterations)
  → input: diff + evaluator_metrics + compliance flags
  → output: quality assessment + what made good mutations good
        │
        ▼
 [Step 6] Aggregate Synthesis           existing CLEAR pattern
  → "What works" clusters
  → "What doesn't" clusters
  → % budget wasted per failure category
  → compliance × effectiveness cross-tabulation
  → stagnation summary
        │
        ▼
 Dashboard / ZIP output
```

Steps 0–2 are deterministic and always run first at zero LLM cost. They tag every record before any LLM judge call, so the judge focuses only on the high-signal records.

---

## 3. New Components

### 3.1 Directory Structure

```
src/clear_eval/
├── evolve_loop/
│   ├── __init__.py
│   ├── checkpoint_adapter.py       input normalization
│   ├── stagnation_detector.py      deterministic streak detection
│   ├── evaluator_analyzer.py       failure classification + artifact clustering
│   └── compliance_checker.py       EVOLVE-BLOCK, format, signature checks
│
├── pipeline/
│   └── use_cases/
│       └── EvolveLoopUseCase.py    new use case class
│
└── dashboard/
    └── evolve_dashboard.py         extended Streamlit dashboard
```

---

### 3.2 Checkpoint Adapter (`checkpoint_adapter.py`)

Converts framework-specific output into standard JSONL. One adapter function per framework.

```python
def adapt_skydiscover(checkpoint_dir: str) -> Iterator[dict]: ...
def adapt_shinkaevolve(db_path: str) -> Iterator[dict]: ...
def adapt_openevolve(checkpoint_dir: str, trace_path: str) -> Iterator[dict]: ...
def load_jsonl(path: str) -> Iterator[dict]: ...  # passthrough for direct JSONL

def load_evolve_records(source: str) -> Iterator[dict]:
    """Auto-detects format and returns normalized records."""
```

Each adapter produces records conforming to the standard JSONL schema (see Requirements §5).

**ShinkaEvolve note:** reads from SQLite via the existing `dbase.py` schema. All fields are already stored.

**OpenEvolve note:** requires `evolution_trace.enabled: true` and `evolution_trace.format: jsonl` in the run config to produce `evolution_trace.jsonl`. Without it, only checkpoint-derived fields (code, score, metadata) are available.

---

### 3.3 Stagnation Detector (`stagnation_detector.py`)

Pure function — no side effects, no LLM calls.

```python
@dataclass
class StagnationPeriod:
    streak_id: str
    start_iteration: int
    end_iteration: Optional[int]        # None if still ongoing at run end
    length: int
    failure_sequence: List[IterationSummary]
    dominant_failure_type: str
    secondary_failure_types: List[str]
    is_alert: bool                      # length >= threshold
    severity: str                       # "warning" | "high" | "critical"
    recovery_iteration: Optional[int]
    recovery_mutation_type: Optional[str]
    recovery_model: Optional[str]
    score_at_stagnation_start: float
    score_at_recovery: Optional[float]

@dataclass
class IterationSummary:
    iteration: int
    mutation_type: str
    failure_type: str                   # see §3.4 for enum
    score_delta: float
    compliance_status: Optional[str]
    format_valid: bool

def detect_stagnation(
    records: List[dict],
    threshold: int = 10,
    min_delta: float = 0.001,
) -> Tuple[List[dict], List[StagnationPeriod]]:
    """
    Tags each record with streak_id and streak_position.
    Returns (tagged_records, stagnation_periods).
    """
```

**Progress definition:** `score_delta >= min_delta` AND `format_valid == True` AND `evaluation_status != "crash"`.

**Severity assignment:**

```python
def _assign_severity(period: StagnationPeriod, threshold: int) -> str:
    if period.length >= 2 * threshold:
        return "critical"
    if period.dominant_failure_type in ("identical_output", "compliance_violation"):
        return "critical" if period.length >= threshold else "high"
    return "warning"
```

---

### 3.4 Evaluator Analyzer (`evaluator_analyzer.py`)

**Failure mode classification:**

```python
FailureMode = Literal[
    "success",        # score_delta >= min_delta
    "partial",        # score_delta < min_delta but at least one sub-metric improved
    "worse",          # score_delta < 0, all sub-metrics same or worse
    "wrong_output",   # correctness metric == 0
    "timeout",        # evaluation_status == "timeout"
    "crash",          # evaluator raised exception
    "format_invalid", # diff/rewrite was unparseable, never reached evaluator
]

def classify_failure(record: dict) -> FailureMode: ...
```

**Sub-metric divergence detection:**

```python
def detect_sub_metric_divergence(record: dict) -> Optional[dict]:
    """
    Returns dict of metrics that improved even though combined_score did not.
    Returns None if combined_score improved (no divergence).
    """
```

**Cascade stage analysis (run-level):**

```python
def cascade_stage_summary(records: List[dict]) -> dict:
    """
    Returns count and % of failures at each cascade stage.
    Only meaningful when cascade_evaluation is enabled.
    """
```

**Evaluator variance check:**

```python
def flag_high_variance(record: dict, std_threshold: float = 0.1) -> bool:
    """
    Returns True if score_std across num_runs exceeds threshold.
    Indicates evaluator noise rather than LLM quality issue.
    """
```

**Artifact clustering (LLM-as-judge, Step 4):**

The LLM judge receives the aggregated `stderr` / `build_warnings` / `llm_feedback` across all failed iterations and returns:

```json
{
  "patterns": [
    {
      "description": "IndexError: list index out of range on line 42",
      "count": 23,
      "pct_of_failures": 60.5,
      "root_cause": "LLM modifies array indexing without preserving boundary conditions",
      "recommendation": "Add to system message: 'Preserve all boundary checks on array accesses'"
    }
  ]
}
```

---

### 3.5 Compliance Checker (`compliance_checker.py`)

All checks are deterministic and run before any LLM judge call.

```python
def check_evolve_block(parent_code: str, child_code: str) -> bool:
    """True if mutations are confined within EVOLVE-BLOCK markers."""

def check_format_valid(diff: Optional[str], child_code: Optional[str],
                        mutation_type: str) -> bool:
    """True if diff is parseable (diff mode) or child_code is a complete file (full mode)."""

def check_signature_preserved(parent_code: str, child_code: str,
                                language: str = "python") -> bool:
    """True if all function signatures in the parent are unchanged in the child."""

def check_semantic_compliance(
    system_message: str,
    diff_or_code: str,
    llm_client,
) -> dict:
    """
    LLM-as-judge. Returns:
    {
      "compliance_level": "fully_compliant" | "partially_compliant" | "non_compliant",
      "violations": [{"constraint": "...", "violating_code": "..."}]
    }
    """
```

---

### 3.6 Evolve Loop Use Case (`EvolveLoopUseCase.py`)

Implements the `BaseUseCase` interface used by CLEAR's existing pipeline.

```python
class EvolveLoopUseCase(BaseUseCase):

    def eval_records(self, df: pd.DataFrame, llm_client, config: dict) -> pd.DataFrame:
        """
        Runs Steps 3–5 (LLM judge calls) on the tagged+classified DataFrame.
        Returns DataFrame with judge columns appended.
        """

    def _eval_stagnation_periods(
        self, periods: List[StagnationPeriod], df: pd.DataFrame, llm_client
    ) -> List[dict]:
        """Step 3: per-stagnation-period root cause analysis."""

    def _eval_artifact_clusters(self, df: pd.DataFrame, llm_client) -> List[dict]:
        """Step 4: evaluator artifact clustering across all failures."""

    def _eval_mutation_quality(self, df: pd.DataFrame, llm_client) -> pd.DataFrame:
        """Step 5: per-mutation quality for non-stagnation iterations."""
```

**LLM judge prompt — stagnation root cause (Step 3):**

```
You are analyzing a stagnation period in an evolutionary code optimization experiment.

The algorithm was stuck for {N} consecutive iterations with no progress.

System message given to the LLM:
{SYSTEM_MESSAGE}

Parent code being mutated (unchanged across all iterations):
{PARENT_CODE}

Sequence of what was attempted:
{SEQUENCE_TABLE}

Evaluator artifacts from this period:
{ARTIFACTS}

Answer:
1. What approach was the LLM repeatedly trying? Was it exploring the same idea?
2. What constraint or structural issue prevented progress?
3. Was there a pattern in the failure types (e.g., always compliance failures
   → the instruction may be ambiguous)?
4. What eventually broke the stagnation (if recovery shown), and why did it work?
5. Categorize: LOCAL_OPTIMUM | INSTRUCTION_CONFUSION | APPROACH_EXHAUSTION |
               FORMAT_ISSUE | EVALUATOR_NOISE | OTHER
6. One concrete recommendation to prevent this stagnation.
```

---

### 3.7 Aggregate Synthesis

Reuses CLEAR's existing `synthesize_shortcomings_from_df` and `map_shortcomings_to_records` pattern, applied to the evolve loop domain.

**"What works" synthesis input:** records where `evaluation_status == "success"`, with their mutation type, model, compliance level, and judge quality assessment.

**"What doesn't" synthesis input:** all failure records, grouped by `dominant_failure_type`, with stagnation period annotations and evaluator artifact clusters.

**Cross-tabulation (new, run-level):**

```
Compliance level    │ Success rate │ Avg score_delta
────────────────────┼──────────────┼────────────────
fully_compliant     │    42%       │   +0.031
partially_compliant │    18%       │   +0.004
non_compliant       │     3%       │   -0.012
```

---

### 3.8 CLI Integration

New CLI entry point added to `cli.py`:

```bash
run-clear-evolve-analysis \
    --source skydiscover \
    --checkpoint-dir results/circle_packing/checkpoints/checkpoint_100 \
    --provider openai \
    --eval-model-name gpt-4o \
    --output-dir results/evolve_analysis/ \
    --stagnation-threshold 10 \
    --min-delta 0.001
```

`--source` accepts: `skydiscover` | `shinkaevolve` | `openevolve` | `jsonl`

---

### 3.9 Dashboard Extension (`evolve_dashboard.py`)

Extended Streamlit dashboard with evolve-loop-specific panels alongside the existing CLEAR views:

**New panels:**

1. **Score Progression** — line chart with stagnation periods shaded in red/orange, breakthrough iterations annotated
2. **Alert Panel** — list of all triggered alerts, sorted by severity, with expandable sequence and root cause
3. **Failure Budget** — stacked bar: % of iterations per failure mode (success / partial / worse / crash / timeout / format_invalid)
4. **Mutation Effectiveness** — success rate by mutation type, by LLM model, by island
5. **Compliance × Score** — cross-tabulation heatmap
6. **Cascade Stage Distribution** — bar chart showing failure counts per stage
7. **Evaluator Artifact Clusters** — ranked list of recurring error patterns with recommendation

---

## 4. Data Flow Detail

```
standard JSONL records
        │
        ├──► stagnation_detector.py   → tags: streak_id, streak_position, is_alert
        │                             → produces: List[StagnationPeriod]
        │
        ├──► evaluator_analyzer.py    → tags: failure_mode, divergent_metrics,
        │                                      cascade_stage_failed, high_variance
        │
        ├──► compliance_checker.py    → tags: evolved_block_only, format_valid,
        │                                      signature_preserved
        │
        └──► pd.DataFrame (fully tagged, no LLM cost so far)
                    │
                    ├──► LLM Judge: stagnation periods      (Step 3)
                    ├──► LLM Judge: artifact clustering     (Step 4)
                    ├──► LLM Judge: per-mutation quality    (Step 5)
                    │
                    └──► Aggregate synthesis                (Step 6)
                                    │
                                    └──► ZIP (parquet + metadata) → dashboard
```

All LLM judge calls use CLEAR's existing `LLMClient` abstraction, caching, and `run_parallel` threading — no new LLM infrastructure is needed.

---

## 5. Dependencies on Existing CLEAR Components

| CLEAR component | Used by |
|---|---|
| `LLMClient` / `get_llm_client` | Steps 3, 4, 5 (all LLM judge calls) |
| `run_parallel` | Parallel per-mutation judge calls (Step 5) |
| `caching_utils` | Resume support for all LLM judge steps |
| `synthesize_shortcomings_from_df` | Step 6 aggregate synthesis |
| `map_shortcomings_to_records` | Step 6 mapping back to individual iterations |
| `save_ui_input_results` | ZIP output for dashboard |
| `show_analysis_dashboard` | Base dashboard, extended with evolve panels |
| `load_config` / `merge_configs` | Configuration loading |

---

## 6. Configuration

New config keys under `default_config.yaml`:

```yaml
# Evolve loop analysis settings
evolve_loop:
  source: "jsonl"                    # jsonl | skydiscover | shinkaevolve | openevolve
  stagnation_threshold: 10           # alert when streak >= this value
  min_delta: 0.001                   # minimum score improvement to count as progress
  check_evolve_block: true           # run EVOLVE-BLOCK compliance check
  check_format: true                 # run format validity check
  check_signature: true              # run function signature preservation check
  check_semantic_compliance: true    # run LLM-as-judge semantic compliance
  analyze_stagnation: true           # run LLM root cause analysis on stagnation periods
  analyze_artifacts: true            # run LLM artifact clustering
  analyze_mutations: true            # run per-mutation quality judge
  score_variance_threshold: 0.1      # flag evaluator runs with std > this value
```
