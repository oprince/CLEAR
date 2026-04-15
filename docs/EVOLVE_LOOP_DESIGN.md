# Evolve Loop Analysis — Design

## 1. Architectural Decision

### Decision: Standalone tool — shared code copied from CLEAR, no runtime dependency

The evolve loop analyzer is implemented as a **fully independent tool** (`evolve-analyzer`). The small set of LLM infrastructure modules shared with CLEAR is **vendored** (copied and owned) rather than imported as a package dependency.

### Rationale — why a separate tool

The question was whether the scope of shared logic justified embedding this tool within CLEAR's codebase. A systematic comparison revealed that the overlap is limited to the LLM infrastructure layer:

| CLEAR component | Reuse verdict | Reason |
|---|---|---|
| `LLMClient` / `get_llm_client` | ✅ Copied | Abstraction over LangChain/LiteLLM/Endpoint |
| `caching_utils` | ✅ Copied | Resume support, deduplication |
| `run_parallel` | ✅ Copied | Thread-pool for LLM judge calls |
| `BaseUseCase` | ❌ Not used | Forces CLEAR's linear pipeline shape; incompatible with coordinator pattern |
| `synthesize_shortcomings_from_df` | ❌ Not used | Tightly coupled to CLEAR's per-call CSV format |
| `map_shortcomings_to_records` | ❌ Not used | Same reason |
| `save_ui_input_results` / dashboard | ❌ Not used | Table browser for eval calls; incompatible with time-series UI paradigm |
| `load_config` / `merge_configs` | ❌ Not used | Any config library serves equally; no shared schema |

The divergences that make a shared codebase inappropriate:

1. **Data model is fundamentally different.** CLEAR's unit is an evaluation call: one row = `(input, expected_output, actual_output, judge_score)`, order irrelevant. The evolve loop's unit is an iteration in a time series: order is the entire point. Fields like `streak_id`, `plateau_onset_iteration`, `recovery_iteration`, and `change_points` have no analog in CLEAR's schema.

2. **The quantitative analyzer is larger than CLEAR's existing analysis codebase.** Ten deterministic analyzers covering stagnation, convergence, regression, exploration, efficiency, search space, meta-analysis, ceiling detection, evaluator behavior, and compliance share no logic with CLEAR's existing analysis pipeline. Locating them inside CLEAR adds weight without reuse.

3. **The multi-agent coordinator pattern conflicts with CLEAR's linear pipeline.** CLEAR's pipeline is: analyze → evaluate → aggregate → synthesize. The evolve loop requires a coordinator that runs deterministic analyzers first (zero LLM cost), dispatches multiple specialist LLM agents on tagged data, queries a historical database, and then synthesizes. Forcing this into `BaseUseCase` would produce an architectural pretense — the interface is implemented but immediately bypassed.

4. **The historical database is a new infrastructure concern.** Cross-experiment comparative analysis requires a persistent store that accumulates across runs. CLEAR has no concept of cross-experiment state. Adding it to CLEAR couples an evaluation framework to an optimization experiment registry with a different lifecycle.

5. **The dashboard paradigm is different.** CLEAR's dashboard is a table browser for evaluation calls. The evolve loop needs time-series charts, stagnation-shaded score progression curves, efficiency curves, and an exploration/exploitation timeline. These panels share no component logic with CLEAR's existing UI.

6. **The user populations are different.** CLEAR users are prompt engineers and ML engineers evaluating model output quality. Evolve loop users are researchers running evolutionary code optimization experiments. Bundling them in one CLI namespace creates a confusing surface for both groups.

### Rationale — why copy rather than depend on CLEAR as a package

Even for the three modules that are worth reusing, a pip dependency on `clear_eval` is the wrong mechanism:

1. **CLEAR is a tool, not a library.** Its internal APIs (`get_llm_client`, `run_parallel`, `cache_call`) are not versioned contracts — they are implementation details subject to refactor without notice. A dependency would couple `evolve-analyzer`'s stability to CLEAR's internal churn.

2. **Transitive dependency pollution.** Installing `clear_eval` pulls in its full dependency tree (LangChain, MLflow/Langfuse clients, Streamlit, etc.). `evolve-analyzer` needs none of those — only the LLM client abstractions. Vendoring copies only what is actually used.

3. **The shared surface is small and stable.** Three modules — LLM client abstraction, thread-pool wrapper, cache utility — total fewer than 300 lines. This is well within the threshold where vendoring is simpler than dependency management.

4. **Independent evolution.** Once copied, each tool can evolve its LLM layer independently. If CLEAR switches its caching backend or changes how it handles provider authentication, `evolve-analyzer` is not affected.

5. **Simpler installation for users.** `pip install evolve-analyzer` installs only what the tool actually needs. No CLEAR configuration files, no CLEAR CLI commands, no surprise dependencies on MLflow or Langfuse.

### Consequence

The three modules are copied at project creation into `evolve_analyzer/llm/` and maintained independently:

```
evolve_analyzer/llm/
├── client.py          # copied from clear_eval/llm/client.py — LLM provider abstraction
├── parallel.py        # copied from clear_eval/llm/parallel.py — run_parallel
└── cache.py           # copied from clear_eval/caching_utils.py — cache_call
```

A comment at the top of each copied file records the origin and the copy date, so future maintainers can diff against the CLEAR source if they want to pull in upstream improvements manually:

```python
# Copied from clear_eval/llm/client.py (CLEAR commit abc1234, 2025-01-01).
# Maintained independently. To sync upstream improvements, diff manually.
```

CLEAR does **not** appear in `pyproject.toml` at all. All other components — ingestion, analysis pipeline, historical DB, report synthesis, dashboard, and CLI — are implemented independently.

---

## 2. Overview

`evolve-analyzer` is a standalone diagnostic tool for post-hoc analysis of evolutionary code optimization experiments. It answers the question: *"What worked and what didn't in this evolve loop run?"* — using the same DL training diagnostics framing (convergence curve shape, exploration/exploitation balance, cost efficiency, ceiling detection) applied to evolutionary optimization.

The architecture adopts a **coordinator + specialist agents** pattern: a thin coordinator orchestrates four specialist components — Ingestion Agent, Quantitative Analyzer, Qualitative Analyzer, Comparative Analyzer — and feeds their outputs into a Report Synthesizer. Steps with zero LLM cost always run first; LLM calls are deferred until deterministic tagging is complete.

---

## 3. Architecture

### 3.1 High-Level Flow

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
 [Comparative Analyzer]  ─── historical DB ──────────────────────────────
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

## 4. Directory Structure

```
evolve-analyzer/
├── pyproject.toml               # no clear_eval dependency
├── config/
│   └── default_config.yaml
├── src/
│   └── evolve_analyzer/
│       ├── __init__.py
│       ├── cli.py               # run-evolve-analysis entry point
│       ├── coordinator.py       # thin orchestrator
│       │
│       ├── llm/                 # vendored from CLEAR — maintained independently
│       │   ├── __init__.py
│       │   ├── client.py        # LLM provider abstraction (OpenAI/Anthropic/LiteLLM/endpoint)
│       │   ├── parallel.py      # run_parallel thread-pool wrapper
│       │   └── cache.py         # cache_call — disk cache for LLM responses
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   └── checkpoint_adapter.py   # per-framework adapters + auto-detect
│       │
│       ├── quantitative/
│       │   ├── __init__.py
│       │   ├── bundle.py               # QuantitativeBundle dataclass
│       │   ├── stagnation_detector.py  # streak detection + alert classification
│       │   ├── evaluator_analyzer.py   # failure classification + variance flags
│       │   ├── compliance_checker.py   # EVOLVE-BLOCK, format, signature checks
│       │   ├── convergence_analyzer.py # best-so-far, change-points, rate
│       │   ├── exploration_analyzer.py # diversity index, exploit/explore phases
│       │   ├── regression_analyzer.py  # frequency, severity, recovery time
│       │   ├── efficiency_analyzer.py  # cost-per-improvement, Pareto frontier
│       │   ├── search_space_analyzer.py# param distributions, bound detection
│       │   ├── meta_analyzer.py        # suggestion compliance rate, follow rate
│       │   └── ceiling_analyzer.py     # plateau test, marginal improvement trend
│       │
│       ├── qualitative/
│       │   ├── __init__.py
│       │   └── qualitative_analyzer.py # LLM judge orchestration (Steps A–F)
│       │
│       ├── historical_db.py            # SQLite store for cross-experiment data
│       ├── report_synthesizer.py       # dimension ratings + structured report
│       └── dashboard.py                # Streamlit dashboard (standalone)
└── tests/
    ├── test_stagnation_detector.py
    ├── test_evaluator_analyzer.py
    ├── test_compliance_checker.py
    └── ...
```

**`pyproject.toml` — no CLEAR dependency:**

```toml
[project]
name = "evolve-analyzer"
dependencies = [
    "pandas",
    "numpy",
    "scipy",
    "streamlit",
    "click",
    "pyyaml",
    "openai",            # used by llm/client.py
    "anthropic",         # used by llm/client.py
    "litellm",           # used by llm/client.py
    "diskcache",         # used by llm/cache.py
]
```

---

## 5. Component Specifications

### 5.1 Ingestion Agent (`ingestion/checkpoint_adapter.py`)

Converts framework-specific output into standard JSONL. One adapter function per framework.

```python
def adapt_skydiscover(checkpoint_dir: str) -> Iterator[dict]: ...
def adapt_shinkaevolve(db_path: str) -> Iterator[dict]: ...
def adapt_openevolve(checkpoint_dir: str, trace_path: str) -> Iterator[dict]: ...
def load_jsonl(path: str) -> Iterator[dict]: ...  # passthrough for direct JSONL

def load_evolve_records(source: str, path: str) -> Iterator[dict]:
    """Auto-detects format and returns normalized records."""
```

Each adapter produces records conforming to the standard JSONL schema (see Requirements §7). Derived fields (`score_delta`, `evaluation_status`) are computed during ingestion.

**ShinkaEvolve:** reads from SQLite via the existing `dbase.py` schema. Framework stagnation events (dynamic island spawns) are mapped to `framework_stagnation_event`.

**OpenEvolve:** requires `evolution_trace.enabled: true` in the run config. Without it, only checkpoint-derived fields are available.

**SkyDiscover:** reads `checkpoints/checkpoint_N/` directories. AdaEvolve paradigm shift events are mapped to `framework_stagnation_event`.

---

### 5.2 Quantitative Analyzer (`quantitative/`)

All components are pure functions — no side effects, no LLM calls. They run in sequence on the normalized records and produce a fully tagged DataFrame plus a `QuantitativeBundle` summary dict.

#### 5.2.1 Stagnation Detector (`stagnation_detector.py`)

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

#### 5.2.2 Evaluator Analyzer (`evaluator_analyzer.py`)

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

#### 5.2.3 Compliance Checker (`compliance_checker.py`)

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

#### 5.2.4 Convergence Analyzer (`convergence_analyzer.py`)

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

#### 5.2.5 Exploration Analyzer (`exploration_analyzer.py`)

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

#### 5.2.6 Regression Analyzer (`regression_analyzer.py`)

```python
@dataclass
class RegressionMetrics:
    regression_frequency: float          # fraction of iterations worse than prior best
    severity_distribution: dict          # histogram of score_delta for regressions
    mean_recovery_time: float            # iterations to recover from regression
    death_spiral_periods: List[Tuple[int, int]]  # (start, end) of consecutive regressions

def analyze_regressions(records: List[dict]) -> RegressionMetrics: ...
```

#### 5.2.7 Efficiency Analyzer (`efficiency_analyzer.py`)

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

#### 5.2.8 Search Space Analyzer (`search_space_analyzer.py`)

```python
@dataclass
class SearchSpaceMetrics:
    param_distributions: dict            # per-parameter value distributions in top-K
    bound_hit_params: List[str]          # parameters repeatedly hitting bounds
    effective_dimensionality: int        # estimated non-redundant dimensions
    trial_to_param_ratio: Optional[float]

def analyze_search_space(records: List[dict], top_k: int = 10) -> SearchSpaceMetrics: ...
```

#### 5.2.9 Meta-Analyzer (`meta_analyzer.py`)

Only active when `reasoning_trace` fields are present.

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

#### 5.2.10 Ceiling Analyzer (`ceiling_analyzer.py`)

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

### 5.3 Qualitative Analyzer (`qualitative/qualitative_analyzer.py`)

Standalone class — does not implement `BaseUseCase` or any CLEAR interface. Uses the vendored LLM infrastructure from `evolve_analyzer.llm`.

```python
from evolve_analyzer.llm.client import get_llm_client
from evolve_analyzer.llm.parallel import run_parallel
from evolve_analyzer.llm.cache import cache_call

class QualitativeAnalyzer:

    def __init__(self, llm_client, config: dict):
        self.llm_client = llm_client
        self.config = config

    def run(self, quant: QuantitativeBundle) -> QualitativeBundle:
        """Orchestrates all LLM judge steps."""

    def _eval_stagnation_periods(
        self, periods: List[StagnationPeriod], df: pd.DataFrame
    ) -> List[dict]:
        """Step A: per-stagnation-period root cause analysis."""

    def _eval_artifact_clusters(self, df: pd.DataFrame) -> List[dict]:
        """Step B: evaluator artifact clustering across all failures."""

    def _eval_mutation_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step C: per-mutation quality for non-stagnation iterations."""

    def _eval_semantic_compliance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step D: LLM-as-judge semantic compliance check."""

    def _eval_exploration_structure(self, df: pd.DataFrame) -> dict:
        """Step E: structural diversity assessment from code diffs."""

    def _eval_meta_quality(self, df: pd.DataFrame) -> dict:
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

### 5.4 Comparative Analyzer (`historical_db.py`)

Standalone SQLite database, entirely new infrastructure with no CLEAR equivalent.

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

Ratings shift from absolute heuristics to relative norms once `min_history` experiments are accumulated (configurable, default = 5).

---

### 5.5 Report Synthesizer (`report_synthesizer.py`)

Standalone synthesis logic — not derived from CLEAR's `synthesize_shortcomings_from_df`. CLEAR's function is coupled to the per-evaluation-call CSV format and is incompatible with time-series iteration records.

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
    data_available: bool                 # False when degraded due to missing log fields

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

**Aggregate cross-tabulation** (implemented locally, not via CLEAR):

```
Compliance level    │ Success rate │ Avg score_delta
────────────────────┼──────────────┼────────────────
fully_compliant     │    42%       │   +0.031
partially_compliant │    18%       │   +0.004
non_compliant       │     3%       │   -0.012
```

---

## 6. Data Flow Detail

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
                                ├──► LLM Judge: per-mutation quality       (Step C)  ← run_parallel
                                ├──► LLM Judge: semantic compliance        (Step D)  ← run_parallel
                                ├──► LLM Judge: exploration structure      (Step E)
                                ├──► LLM Judge: meta-analysis quality      (Step F)
                                │       ↑ all via clear_eval.llm + cache_call
                                │
                                ├──► Comparative Analyzer (historical_db.py)
                                │
                                └──► Report Synthesizer
                                                │
                                                └──► ZIP (parquet + metadata) → dashboard
```

LLM judge calls use `get_llm_client`, `run_parallel`, and `cache_call` from the vendored `evolve_analyzer.llm` package. No CLEAR modules, pipeline classes, use case interfaces, or dashboard components are imported at runtime.

---

## 7. Graceful Degradation

The system degrades gracefully based on available log richness:

| Available data | Active components |
|---|---|
| Scores only | Stagnation, Convergence, Regression, Efficiency, Ceiling |
| + Code diffs | + Compliance (deterministic), Exploration structure (LLM) |
| + Evaluator artifacts | + Artifact clustering (LLM) |
| + Reasoning traces | + Meta-analysis quality (LLM) |
| + Historical DB (n ≥ 5) | + Comparative ratings on all dimensions |

`DimensionReport.data_available` is `False` for any dimension that lacked the required input fields, so the dashboard can surface which dimensions were fully vs. partially analyzed.

---

## 8. CLI

```bash
run-evolve-analysis \
    --source skydiscover \
    --checkpoint-dir results/circle_packing/checkpoints/checkpoint_100 \
    --provider openai \
    --model gpt-4o \
    --output-dir results/evolve_analysis/ \
    --stagnation-threshold 10 \
    --min-delta 0.001 \
    --historical-db path/to/evolve_history.db
```

`--source` accepts: `skydiscover` | `shinkaevolve` | `openevolve` | `jsonl`

---

## 9. Dashboard (`dashboard.py`)

Standalone Streamlit application — not an extension of CLEAR's dashboard. The UI paradigm is time-series and aggregate statistics, incompatible with CLEAR's per-call table browser.

**Panels:**

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

## 10. Vendored Code Origin

Three modules are copied from CLEAR at project creation and maintained independently under `evolve_analyzer/llm/`. Each file carries a header comment recording the source:

```python
# Vendored from clear_eval/llm/client.py (CLEAR commit <hash>, <date>).
# Maintained independently within evolve-analyzer.
# To incorporate upstream improvements, diff manually against the CLEAR source.
```

| Vendored module | Origin in CLEAR | Purpose |
|---|---|---|
| `evolve_analyzer/llm/client.py` | `clear_eval/llm/client.py` | LLM provider abstraction (OpenAI, Anthropic, LiteLLM, local endpoint) |
| `evolve_analyzer/llm/parallel.py` | `clear_eval/llm/parallel.py` | Thread-pool wrapper for concurrent LLM calls |
| `evolve_analyzer/llm/cache.py` | `clear_eval/caching_utils.py` | Disk cache — skip already-judged records on re-run |

CLEAR does not appear in `pyproject.toml`. There is no runtime dependency on CLEAR. After the initial copy, the two codebases evolve independently.

---

## 11. Configuration (`config/default_config.yaml`)

```yaml
# LLM backend (via clear_eval.llm)
llm:
  provider: "openai"              # openai | anthropic | litellm | endpoint
  model: "gpt-4o"
  temperature: 0.0
  max_tokens: 4096

# Ingestion
ingestion:
  source: "jsonl"                 # jsonl | skydiscover | shinkaevolve | openevolve

# Stagnation detection
stagnation:
  threshold: 10                   # alert when streak >= this value
  min_delta: 0.001                # minimum score improvement to count as progress

# Convergence
convergence:
  window: 10                      # rolling window for mean/variance

# Search space
search_space:
  top_k: 10                       # top-K solutions for parameter distribution

# Evaluator
evaluator:
  score_variance_threshold: 0.1   # flag runs with std > this value

# Compliance checks (deterministic)
compliance:
  check_evolve_block: true
  check_format: true
  check_signature: true
  check_semantic: true            # LLM-as-judge semantic compliance (Step D)

# LLM judge steps (can disable individually to save cost)
judges:
  stagnation_root_cause: true     # Step A
  artifact_clustering: true       # Step B
  mutation_quality: true          # Step C
  semantic_compliance: true       # Step D
  exploration_structure: true     # Step E
  meta_quality: true              # Step F

# Historical / comparative analysis
historical:
  db_path: null                   # path to SQLite DB; null disables comparison
  min_experiments: 5              # experiments needed before relative ratings activate
  pattern_promotion_threshold: 3  # pattern recurring in N experiments → standing recommendation
```
