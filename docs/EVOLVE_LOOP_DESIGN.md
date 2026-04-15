# Evolve Loop Analysis — Design

## 1. Overview

This document describes the design for extending CLEAR with an `evolve_loop` analysis mode. The extension reuses CLEAR's existing LLM-as-judge pipeline, inference backends, caching system, and dashboard infrastructure. New components are confined to input adapters, deterministic analyzers, LLM judge agents, a historical database, and dashboard extensions.

The architecture adopts a **coordinator + specialist agents** pattern: a thin coordinator orchestrates four specialist components — Ingestion Agent, Quantitative Analyzer, Qualitative Analyzer, Comparative Analyzer — and feeds their outputs into a Report Synthesizer. Steps with zero LLM cost always run first; LLM calls are deferred until the deterministic tagging is complete.

---

## 2. Architecture

### 2.1 High-Level Flow

```
Input (JSONL or framework checkpoint)
        │
        ▼
 [Ingestion Agent]
  checkpoint_adapter.py
  → auto-detects framework, normalizes to standard JSONL
  → validates schema, fills derived fields (score_delta, evaluation_status)
        │
        ▼
 [Quantitative Analyzer]  ─── deterministic, zero LLM cost ───────────────┐
  stagnation_detector.py      tags records: streak_id, streak_position     │
  evaluator_analyzer.py       tags records: failure_mode, divergence flags │
  compliance_checker.py       tags records: evolved_block_only, format     │
  convergence_analyzer.py     best-so-far curve, change-points, rate       │
  exploration_analyzer.py     diversity index, exploit/explore phases      │
  regression_analyzer.py      regression freq, severity, recovery time     │
  efficiency_analyzer.py      cost-per-improvement, Pareto frontier        │
  search_space_analyzer.py    param distributions, bound hits, dim-reduce  │
  meta_analyzer.py            suggestion compliance, follow rate           │
  ceiling_analyzer.py         plateau test, marginal improvement trend     │
  → produces QuantitativeBundle (fully tagged DataFrame + metric summaries)│
        │                                                                  │
        ▼                                                                  │
 [Qualitative Analyzer]  ─── LLM agents ──────────────────────────────────┤
  LLM Judge: stagnation root cause (per stagnation period)                 │
  LLM Judge: evaluator artifact clustering (across all failures)           │
  LLM Judge: per-mutation quality (non-stagnation iterations)              │
  LLM Judge: semantic compliance (per iteration or sampled)                │
  LLM Judge: exploration structure (code diff structural assessment)       │
  LLM Judge: meta-analysis quality (reasoning trace coherence)             │
        │
        ▼
 [Comparative Analyzer]  ─── LLM agent + historical DB ─────────────────
  Queries historical experiment database
  → "Is this convergence rate typical for this benchmark + tool?"
  → "Is this regression frequency abnormal?"
  → "Have we seen this failure pattern before?"
  → produces HistoricalComparison per dimension
        │
        ▼
 [Report Synthesizer]
  → per-dimension rating (1–5) + summary + evidence + historical + recommendation
  → cross-dimension interaction narrative
  → per-stagnation-period breakdown
  → per-run aggregate statistics
  → executive summary
        │
        ▼
 Dashboard / ZIP output
```

Deterministic analyzers always run first at zero LLM cost. They tag every record before any LLM judge call, so the judges focus on high-signal records only.

---

## 3. Directory Structure

```
src/clear_eval/
├── evolve_loop/
│   ├── __init__.py
│   ├── checkpoint_adapter.py        input normalization (Ingestion Agent)
│   │
│   ├── quantitative/
│   │   ├── __init__.py
│   │   ├── stagnation_detector.py   streak detection + alert classification
│   │   ├── evaluator_analyzer.py    failure classification + variance flags
│   │   ├── compliance_checker.py    EVOLVE-BLOCK, format, signature checks
│   │   ├── convergence_analyzer.py  best-so-far, change-points, rate
│   │   ├── exploration_analyzer.py  diversity index, exploit/explore phases
│   │   ├── regression_analyzer.py   frequency, severity, recovery time
│   │   ├── efficiency_analyzer.py   cost-per-improvement, Pareto frontier
│   │   ├── search_space_analyzer.py param distributions, bound detection
│   │   ├── meta_analyzer.py         suggestion compliance rate, follow rate
│   │   └── ceiling_analyzer.py      plateau test, marginal improvement trend
│   │
│   ├── historical_db.py             SQLite store for cross-experiment data
│   └── report_synthesizer.py        dimension ratings + structured report
│
├── pipeline/
│   └── use_cases/
│       └── EvolveLoopUseCase.py     use case class, LLM judge orchestration
│
└── dashboard/
    └── evolve_dashboard.py          extended Streamlit dashboard
```

---

## 4. Component Specifications

### 4.1 Ingestion Agent (`checkpoint_adapter.py`)

Converts framework-specific output into standard JSONL. One adapter function per framework.

```python
def adapt_skydiscover(checkpoint_dir: str) -> Iterator[dict]: ...
def adapt_shinkaevolve(db_path: str) -> Iterator[dict]: ...
def adapt_openevolve(checkpoint_dir: str, trace_path: str) -> Iterator[dict]: ...
def load_jsonl(path: str) -> Iterator[dict]: ...  # passthrough for direct JSONL

def load_evolve_records(source: str) -> Iterator[dict]:
    """Auto-detects format and returns normalized records."""
```

Each adapter produces records conforming to the standard JSONL schema (see Requirements §7). Derived fields (`score_delta`, `evaluation_status`) are computed during ingestion.

**ShinkaEvolve note:** reads from SQLite via the existing `dbase.py` schema. Framework stagnation events (dynamic island spawns) are mapped to `framework_stagnation_event`.

**OpenEvolve note:** requires `evolution_trace.enabled: true` in the run config. Without it, only checkpoint-derived fields are available.

**SkyDiscover note:** reads `checkpoints/checkpoint_N/` directories. AdaEvolve paradigm shift events are mapped to `framework_stagnation_event`.

---

### 4.2 Quantitative Analyzer

All components are pure functions — no side effects, no LLM calls. They run in sequence on the normalized records and produce a fully tagged DataFrame plus a `QuantitativeBundle` summary dict.

#### 4.2.1 Stagnation Detector (`stagnation_detector.py`)

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
    failure_type: str
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

#### 4.2.2 Evaluator Analyzer (`evaluator_analyzer.py`)

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

def detect_sub_metric_divergence(record: dict) -> Optional[dict]:
    """Returns dict of metrics that improved even though combined_score did not."""

def cascade_stage_summary(records: List[dict]) -> dict:
    """Count and % of failures at each cascade stage."""

def flag_high_variance(record: dict, std_threshold: float = 0.1) -> bool:
    """True if score_std across num_runs exceeds threshold (evaluator noise)."""
```

#### 4.2.3 Compliance Checker (`compliance_checker.py`)

```python
def check_evolve_block(parent_code: str, child_code: str) -> bool:
    """True if mutations are confined within EVOLVE-BLOCK markers."""

def check_format_valid(diff: Optional[str], child_code: Optional[str],
                        mutation_type: str) -> bool:
    """True if diff is parseable or child_code is a complete file."""

def check_signature_preserved(parent_code: str, child_code: str,
                                language: str = "python") -> bool:
    """True if all function signatures in the parent are unchanged in the child."""
```

#### 4.2.4 Convergence Analyzer (`convergence_analyzer.py`)

```python
@dataclass
class ConvergenceMetrics:
    best_so_far_curve: List[float]       # best score at each iteration
    rolling_mean: List[float]            # configurable window
    rolling_variance: List[float]
    change_points: List[int]             # iterations where trajectory shifted
    convergence_rate: float              # improvement per iteration
    improvement_per_eval: float          # improvement per evaluator call
    time_to_best_fraction: float         # fraction of compute before best found
    plateau_onset_iteration: Optional[int]

def analyze_convergence(records: List[dict], window: int = 10) -> ConvergenceMetrics: ...
```

#### 4.2.5 Exploration Analyzer (`exploration_analyzer.py`)

```python
@dataclass
class ExplorationMetrics:
    structural_diversity_index: float    # mean pairwise code distance
    exploit_phase_fraction: float        # fraction of iterations in exploit mode
    explore_phase_fraction: float
    distinct_strategy_clusters: int      # estimated via code clustering
    revert_frequency: float              # fraction of iterations reverting to prior best

def analyze_exploration(records: List[dict]) -> ExplorationMetrics: ...
```

#### 4.2.6 Regression Analyzer (`regression_analyzer.py`)

```python
@dataclass
class RegressionMetrics:
    regression_frequency: float          # fraction of iterations worse than prior best
    severity_distribution: dict          # histogram of score_delta for regressions
    mean_recovery_time: float            # iterations to recover from regression
    death_spiral_periods: List[Tuple[int, int]]  # (start, end) of consecutive regressions

def analyze_regressions(records: List[dict]) -> RegressionMetrics: ...
```

#### 4.2.7 Efficiency Analyzer (`efficiency_analyzer.py`)

```python
@dataclass
class EfficiencyMetrics:
    improvement_per_llm_call: float
    improvement_per_dollar: Optional[float]
    improvement_per_hour: Optional[float]
    improvement_per_eval_call: Optional[float]
    productive_phase_fraction: float     # compute before plateau
    wasted_phase_fraction: float         # compute after plateau
    pareto_frontier: List[Tuple[float, float]]  # (cost, best_score) over time

def analyze_efficiency(records: List[dict]) -> EfficiencyMetrics: ...
```

#### 4.2.8 Search Space Analyzer (`search_space_analyzer.py`)

```python
@dataclass
class SearchSpaceMetrics:
    param_distributions: dict            # per-parameter value distributions in top-K
    bound_hit_params: List[str]          # parameters repeatedly hitting bounds
    effective_dimensionality: int        # estimated non-redundant dimensions
    trial_to_param_ratio: Optional[float]

def analyze_search_space(records: List[dict], top_k: int = 10) -> SearchSpaceMetrics: ...
```

#### 4.2.9 Meta-Analyzer (`meta_analyzer.py`)

Only active when `reasoning_trace` fields are available.

```python
@dataclass
class MetaAnalysisMetrics:
    suggestion_follow_rate: Optional[float]
    conditional_improvement_rate: Optional[float]
    pattern_reuse_frequency: Optional[float]
    scratchpad_growth_rate: Optional[float]
    compaction_events: Optional[int]

def analyze_meta_quality(records: List[dict]) -> MetaAnalysisMetrics: ...
```

#### 4.2.10 Ceiling Analyzer (`ceiling_analyzer.py`)

```python
@dataclass
class CeilingMetrics:
    marginal_improvement_trend: str      # "declining" | "flat" | "improving"
    plateau_p_value: Optional[float]     # statistical test for flat region
    estimated_gain_probability: Optional[float]  # P(beat best in N more iters)
    early_stop_suggested_at: Optional[int]   # iteration meta-analyzer flagged
    early_stop_actual_at: Optional[int]      # iteration run actually ended
    wasted_iterations_after_suggestion: Optional[int]

def analyze_ceiling(records: List[dict]) -> CeilingMetrics: ...
```

---

### 4.3 Qualitative Analyzer (LLM Agents via `EvolveLoopUseCase.py`)

Implements the `BaseUseCase` interface used by CLEAR's existing pipeline.

```python
class EvolveLoopUseCase(BaseUseCase):

    def eval_records(self, df: pd.DataFrame, llm_client, config: dict) -> pd.DataFrame:
        """Runs all LLM judge steps on the tagged DataFrame."""

    def _eval_stagnation_periods(
        self, periods: List[StagnationPeriod], df: pd.DataFrame, llm_client
    ) -> List[dict]:
        """Step A: per-stagnation-period root cause analysis."""

    def _eval_artifact_clusters(self, df: pd.DataFrame, llm_client) -> List[dict]:
        """Step B: evaluator artifact clustering across all failures."""

    def _eval_mutation_quality(self, df: pd.DataFrame, llm_client) -> pd.DataFrame:
        """Step C: per-mutation quality for non-stagnation iterations."""

    def _eval_semantic_compliance(self, df: pd.DataFrame, llm_client) -> pd.DataFrame:
        """Step D: LLM-as-judge semantic compliance check."""

    def _eval_exploration_structure(self, df: pd.DataFrame, llm_client) -> dict:
        """Step E: structural diversity assessment from code diffs."""

    def _eval_meta_quality(self, df: pd.DataFrame, llm_client) -> dict:
        """Step F: reasoning trace coherence and self-contradiction analysis."""
```

**LLM judge prompt — stagnation root cause (Step A):**

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

**LLM judge prompt — evaluator artifact clustering (Step B):**

The judge receives aggregated `stderr` / `build_warnings` / `llm_feedback` across all failed iterations and returns:

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

**Semantic compliance (Step D):**

```python
def check_semantic_compliance(
    system_message: str,
    diff_or_code: str,
    llm_client,
) -> dict:
    """
    Returns:
    {
      "compliance_level": "fully_compliant" | "partially_compliant" | "non_compliant",
      "violations": [{"constraint": "...", "violating_code": "..."}]
    }
    """
```

---

### 4.4 Comparative Analyzer (`historical_db.py`)

Maintains a SQLite database of past experiment analyses, queryable by benchmark, tool, model, and algorithm.

```python
class HistoricalDB:

    def record_experiment(self, experiment_id: str, metrics: QuantitativeBundle) -> None:
        """Persists all dimension metrics for the completed experiment."""

    def compare(
        self,
        metric_name: str,
        value: float,
        filters: dict,          # e.g. {"tool": "skydiscover", "benchmark": "circle_packing"}
    ) -> HistoricalComparison:
        """
        Returns percentile rank and historical distribution summary.
        Falls back to absolute heuristic when n_experiments < min_history.
        """

    def get_recurring_patterns(self) -> List[str]:
        """Returns patterns that appeared in >= pattern_promotion_threshold experiments."""

@dataclass
class HistoricalComparison:
    metric_name: str
    current_value: float
    percentile: Optional[float]          # None when history insufficient
    historical_median: Optional[float]
    n_experiments: int
    rating_basis: str                    # "historical" | "heuristic"
    summary: str                         # e.g. "Worse than 80% of past runs"
```

Each new report is compared against the historical distribution per dimension. Ratings shift from absolute heuristics to relative norms once `min_history` experiments are accumulated (configurable, default = 5).

---

### 4.5 Report Synthesizer (`report_synthesizer.py`)

Combines quantitative metrics, qualitative judge outputs, and historical comparisons into the structured diagnostic report.

```python
@dataclass
class DimensionReport:
    name: str
    rating: int                          # 1–5
    rating_emoji: str                    # 🔴 🟠 🟡 🟢 ✅
    summary: str                         # one-line
    evidence: List[str]                  # iteration pointers + log excerpts
    historical: Optional[HistoricalComparison]
    recommendation: str

@dataclass
class EvolveLoopReport:
    experiment_id: str
    executive_summary: str
    dimensions: List[DimensionReport]
    cross_dimension_interactions: str    # narrative
    stagnation_periods: List[StagnationPeriodReport]
    aggregate_stats: AggregateStats
    novel_observations: List[str]        # patterns outside fixed dimensions

def synthesize_report(
    quant: QuantitativeBundle,
    qual: QualitativeBundle,
    historical: List[HistoricalComparison],
    config: dict,
) -> EvolveLoopReport: ...
```

**Dimension report format (text rendering):**

```
Dimension: Convergence Dynamics
Rating:    2/5  🔴
Summary:   Trajectory was flat from iteration 11 onward — 40 of 51 iterations
           produced no meaningful improvement over iteration 11.
Evidence:  Best-so-far at iter 11: 0.712. Best-so-far at iter 51: 0.731.
           Rolling variance (window=10) exceeded 0.05 from iter 15 onward.
Historical: Worse than 80% of past runs on this benchmark (median plateau onset: iter 28).
Recommendation: Add rolling improvement rate stopping criterion; would have saved
           ~35 iterations and ~$40 in this experiment.
```

**Aggregate synthesis** reuses CLEAR's existing `synthesize_shortcomings_from_df` and `map_shortcomings_to_records` pattern:

```
Compliance level    │ Success rate │ Avg score_delta
────────────────────┼──────────────┼────────────────
fully_compliant     │    42%       │   +0.031
partially_compliant │    18%       │   +0.004
non_compliant       │     3%       │   -0.012
```

---

## 5. Data Flow Detail

```
standard JSONL records
        │
        ├──► stagnation_detector.py    → tags: streak_id, streak_position, is_alert
        ├──► evaluator_analyzer.py     → tags: failure_mode, divergent_metrics,
        │                                       cascade_stage_failed, high_variance
        ├──► compliance_checker.py     → tags: evolved_block_only, format_valid,
        │                                       signature_preserved
        ├──► convergence_analyzer.py   → ConvergenceMetrics
        ├──► exploration_analyzer.py   → ExplorationMetrics
        ├──► regression_analyzer.py    → RegressionMetrics
        ├──► efficiency_analyzer.py    → EfficiencyMetrics
        ├──► search_space_analyzer.py  → SearchSpaceMetrics
        ├──► meta_analyzer.py          → MetaAnalysisMetrics (if reasoning_trace)
        └──► ceiling_analyzer.py       → CeilingMetrics
                    │
                    └──► QuantitativeBundle (fully tagged DataFrame + all metric summaries)
                                │
                                ├──► LLM Judge: stagnation root cause     (Step A)
                                ├──► LLM Judge: artifact clustering        (Step B)
                                ├──► LLM Judge: per-mutation quality       (Step C)
                                ├──► LLM Judge: semantic compliance        (Step D)
                                ├──► LLM Judge: exploration structure      (Step E)
                                ├──► LLM Judge: meta-analysis quality      (Step F)
                                │
                                ├──► Comparative Analyzer (historical DB)
                                │
                                └──► Report Synthesizer
                                                │
                                                └──► ZIP (parquet + metadata) → dashboard
```

All LLM judge calls use CLEAR's existing `LLMClient` abstraction, caching, and `run_parallel` threading — no new LLM infrastructure is needed.

---

## 6. Graceful Degradation

The system degrades gracefully based on available log richness:

| Available data | Active components |
|---|---|
| Scores only | Stagnation, Convergence, Regression, Efficiency, Ceiling |
| + Code diffs | + Compliance (deterministic), Exploration structure (LLM) |
| + Evaluator artifacts | + Artifact clustering (LLM) |
| + Reasoning traces | + Meta-analysis quality (LLM) |
| + Historical DB (n ≥ 5) | + Comparative ratings on all dimensions |

Dimension reports include a `data_available` flag so the dashboard can indicate which dimensions had full vs. degraded analysis.

---

## 7. CLI Integration

New CLI entry point added to `cli.py`:

```bash
run-clear-evolve-analysis \
    --source skydiscover \
    --checkpoint-dir results/circle_packing/checkpoints/checkpoint_100 \
    --provider openai \
    --eval-model-name gpt-4o \
    --output-dir results/evolve_analysis/ \
    --stagnation-threshold 10 \
    --min-delta 0.001 \
    --historical-db path/to/evolve_history.db
```

`--source` accepts: `skydiscover` | `shinkaevolve` | `openevolve` | `jsonl`

---

## 8. Dashboard Extension (`evolve_dashboard.py`)

Extended Streamlit dashboard with evolve-loop-specific panels:

**New panels:**

1. **Score Progression** — line chart with stagnation periods shaded (red/orange), breakthrough iterations annotated, change-points marked
2. **Alert Panel** — all triggered alerts sorted by severity, with expandable sequence table and LLM root cause
3. **Failure Budget** — stacked bar: % of iterations per failure mode
4. **Mutation Effectiveness** — success rate by mutation type, LLM model, island
5. **Compliance × Score** — cross-tabulation heatmap
6. **Cascade Stage Distribution** — bar chart of failure counts per stage
7. **Evaluator Artifact Clusters** — ranked recurring error patterns with recommendations
8. **Efficiency Curve** — improvement per LLM call / dollar over time
9. **Exploration vs. Exploitation Timeline** — exploit/explore phase visualization
10. **Dimension Ratings Overview** — 12-dimension radar or table with 1–5 ratings + traffic lights

---

## 9. Dependencies on Existing CLEAR Components

| CLEAR component | Used by |
|---|---|
| `LLMClient` / `get_llm_client` | All LLM judge steps |
| `run_parallel` | Parallel per-mutation judge calls (Step C) |
| `caching_utils` | Resume support for all LLM judge steps |
| `synthesize_shortcomings_from_df` | Report Synthesizer aggregate synthesis |
| `map_shortcomings_to_records` | Report Synthesizer iteration mapping |
| `save_ui_input_results` | ZIP output for dashboard |
| `show_analysis_dashboard` | Base dashboard, extended with evolve panels |
| `load_config` / `merge_configs` | Configuration loading |

---

## 10. Configuration

New config keys under `default_config.yaml`:

```yaml
# Evolve loop analysis settings
evolve_loop:
  source: "jsonl"                    # jsonl | skydiscover | shinkaevolve | openevolve
  stagnation_threshold: 10           # alert when streak >= this value
  min_delta: 0.001                   # minimum score improvement to count as progress
  convergence_window: 10             # rolling window for mean/variance
  top_k_solutions: 10                # top-K for search space analysis
  score_variance_threshold: 0.1      # flag evaluator runs with std > this value

  # Compliance checks (deterministic)
  check_evolve_block: true
  check_format: true
  check_signature: true
  check_semantic_compliance: true    # LLM-as-judge semantic compliance

  # LLM judge steps
  analyze_stagnation: true           # root cause analysis on stagnation periods
  analyze_artifacts: true            # evaluator artifact clustering
  analyze_mutations: true            # per-mutation quality
  analyze_exploration: true          # structural diversity from code diffs
  analyze_meta_quality: true         # reasoning trace coherence

  # Historical / comparative analysis
  historical_db_path: null           # path to SQLite DB; null disables comparison
  min_history_for_relative_rating: 5 # experiments needed before relative ratings
  pattern_promotion_threshold: 3     # recurring pattern → standing recommendation
```
