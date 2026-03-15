# COMPILOT: LLM-Guided Loop Optimization via Agentic Auto-Scheduling

**Paper:** "Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization"
**Authors:** Massinissa Merouani, Islem Kara Bernou, Riyadh Baghdadi (NYU Abu Dhabi)
**Venue:** PACT 2025

---

## Core Idea

COMPILOT (Compiler Pilot) uses off-the-shelf LLMs (no fine-tuning) as interactive optimization agents that propose loop transformations to a compiler in a closed feedback loop. The LLM suggests transformations, the compiler checks legality and measures performance, and the feedback is returned to the LLM to iteratively refine its strategy.

---

## Architecture Overview

The system has two phases connected by an **optimization dialogue** — a conversation thread that serves as both the action/perception interface and the agent's episodic memory.

### Components

1. **Context Initializer** — Prepares the LLM with task context and the target loop nest.
2. **Interaction Loop Handler** — Manages the iterative optimization cycle.
   - **Response Parser** — Extracts transformation commands from LLM output.
   - **Feedback Generator** — Constructs feedback messages based on compiler/runtime outcomes.
3. **Compiler & Runtime Environment** — Tiramisu polyhedral compiler for legality checks, code generation, and execution.

---

## Phase 1: Context Initialization

Three sequential interactions between the Context Initializer and the LLM:

### 1.1 Context Prompt (System Instructions)

A constant message sent at the start of every session. It defines:

- **Task Description:** LLM's role as a compiler optimization assistant.
- **Input Loop Format:** Annotated C/C++ code with `// comp_ID: compXX` identifiers for each computation block.
- **Output Format:** Structured response with reasoning + transformations in `<schedule>...</schedule>` tags.
- **Transformation Repertoire:** Nine supported primitives (see below).
- **Action Space:** Combining, revoking, modifying transformations.
- **Hardware Context:** CPU model, cores, cache sizes of the target machine.
- **Crash Handling:** Guidelines for interpreting compiler crashes.

### 1.2 Target Loop Nest Presentation

The Context Initializer:
- Extracts the relevant loop nest from the input program.
- Annotates code with `comp_ID` comments (e.g., `// comp_ID: comp05`).
- **Anonymizes** iterator and buffer names (e.g., `a`, `b`, `c`, `buf_0`, `buf_1`) to prevent the LLM from being biased by naming conventions.
- Provides the **initial execution time** as a baseline.

Example format:
```
for(int a=0; a<250; a++){
  for(int b=0; b<220; b++){
    for(int c=0; c<270; c++){
      // comp_ID: comp00
      buf_2[c] = 0.0;
      for(int d=0; d<270; d++){
        // comp_ID: comp01
        buf_2[c] = buf_0[a][b][d]*buf_1[d][c] + buf_2[c];
      }
    }
    for(int e=0; e<270; e++){
      // comp_ID: comp02
      buf_0[a][b][e] = buf_2[e];
    }
  }
}
The initial execution time is 3030.17ms
```

### 1.3 Program Analysis (Chain-of-Thought)

The LLM is prompted to analyze the loop nest before proposing transformations. This serves as chain-of-thought reasoning and covers:
- Loop nest structure (depths, iteration domains).
- Purpose of each computation block.
- Data dependencies between computations.
- Potential optimization strategies (parallelization candidates, fusion opportunities, etc.).

---

## Phase 2: Iterative Optimization

### 2.1 LLM Action: Schedule Proposition

The LLM produces a structured response containing:

1. **Reasoning section:** Rationale for the proposed transformations based on strategy and previous feedback (explicit chain-of-thought).
2. **Schedule:** Transformation sequence in `<schedule>...</schedule>` tags.

Example:
```
Reasoning:
Skewing requires perfect loop nests, which is not the case here. Let's try tiling alone.

New full list of transformations:
<schedule>comp01.Tile2D(L1, L2, 32, 32)</schedule>
```

### 2.2 Supported Transformations (9 Primitives)

| Transformation | Syntax | Notes |
|---|---|---|
| **Fusion** | `comp01.Fuse(comp02, L2)` | Merges two computation blocks at a given level. May implicitly involve shifting. |
| **Shifting** | (handled internally) | Tiramisu's solver determines shifting factors. |
| **Interchange** | `comp01.Interchange(L0, L1)` | Swaps two loop levels. Requires perfect nesting. |
| **Parallelization** | `comp01.Parallelize(L0)` | Parallelizes a loop level using threads. |
| **2D Tiling** | `comp01.Tile2D(L1, L2, 32, 32)` | Tiles two loop levels with given factors. |
| **3D Tiling** | `comp01.Tile3D(L0, L1, L2, 16, 16, 16)` | Tiles three loop levels. |
| **Unrolling** | `comp01.Unroll(L3, 4)` | Unrolls a loop level by a given factor. `L-1` targets the innermost loop. |
| **Skewing** | `comp01.Skew(L1, L2)` | Skews two perfectly nested loops. Factors determined by Tiramisu's solver. |
| **Reversal** | `comp01.Reverse(L0)` | Reverses iteration direction of a loop. |

**Combination syntax:** Transformations are joined with `+`:
```
comp01.Interchange(L0,L1)+comp01.Tile2D(L1,L2,32,64)+comp01.Parallelize(L0)+comp01.Unroll(L1,4)
```

### 2.3 Two-Stage Correctness Check

**Stage 1 — Validity Check (lightweight, compiler-independent):**
- Syntactic correctness of transformation commands.
- Valid comp_IDs and loop levels.
- Fundamental preconditions (e.g., perfect nesting for interchange).
- Filters out malformed proposals before expensive compiler calls.

**Stage 2 — Legality Check (compiler-based):**
- Tiramisu performs polyhedral dependence analysis to guarantee semantic preservation.
- If legal: generates code, compiles, and executes on target hardware.
- Measures runtime and computes speedup ratio.

### 2.4 Feedback Categories

The Feedback Generator returns one of five outcome types:

| Category | Description | Example |
|---|---|---|
| **Invalid Schedule** | Syntactic/semantic error caught by pre-filter | `"Interchange cannot be applied on imperfectly nested loops"` |
| **Illegal Schedule** | Violates data dependencies (compiler check) | `"comp01.Tile2D(L2, L3, 32, 32)+comp01.Parallelize(L0) was not legal."` |
| **Solver Failure** | Tiramisu's solver can't find skewing/shifting factors | Reported with details. |
| **Compiler Crash** | Backend crash during code generation | Reported with error messages. |
| **Successful Execution** | Valid + legal + ran successfully | `"...yielded a speedup of 1.83x"` |

Feedback is appended to the conversation history, enabling the LLM to use **in-context learning (ICL)** — adapting strategy from the accumulated interaction without gradient updates.

### 2.5 Stopping Criteria

- **LLM-initiated:** The LLM issues a `no_further_transformations` command.
- **Premature stopping mitigation:** The framework can prompt the LLM to continue exploring (important — comparable speedup to full runs only achieved after pushing past 5+ quit attempts).
- **Iteration limit:** A predefined max iteration count (e.g., T=30).

### 2.6 Multi-Run Strategy

Due to LLM stochasticity, different runs converge to different local optima. The system supports two evaluation scenarios (see Metrics & Notation below).

Recommended defaults: **T=30 iterations, K=5 runs** (diminishing returns beyond these).

---

## Metrics and Notation

Performance depends on two parameters:

- **T (iterations per run):** The number of distinct schedules the LLM explores within a single optimization dialogue. Longer explorations (larger T) give the LLM more chances to refine schedules and discover better optima.
- **K (number of runs):** For the multi-run scenario, how many independent optimization dialogues are launched from scratch. More runs (larger K) increase the diversity of explored schedules, improving the chance of finding a high-performing one.

### Notation

- **COMPILOT@T** — The typical **single-run** speedup after T iterations. For a single benchmark, this is the **median** speedup across the pool of runs (e.g., 40 runs), each stopped after T iterations. When aggregated across benchmarks, it is the **geometric mean** of these medians.
- **COMPILOT_K@T** — The typical **best-of-K** speedup after T iterations per run. For a single benchmark, this is calculated by repeatedly sampling K runs from the pool, finding the **maximum** speedup within each sample of K, and taking the **median** of these maximums. This simulates the typical best result a user would get by running COMPILOT K times. Aggregation across benchmarks uses the geometric mean.
- By definition, **COMPILOT_1@T = COMPILOT@T**.

### Why these specific aggregation choices?

- **Median** (per benchmark): Robust to outliers and reflective of central tendency. A single run can land on a lucky or unlucky local optimum — the median captures what a user would "typically" experience.
- **Geometric mean** (across benchmarks): Standard for ratios like speedups. It weights relative improvements equally and is less skewed by extreme values than the arithmetic mean. A 100x speedup on one benchmark doesn't dominate the aggregate.
- **Best-of-K via resampling:** Rather than simply taking the max of all 40 runs (which would overestimate what a user with K=5 runs would see), the paper repeatedly samples K runs, takes the max of each sample, then reports the median of those maxes. This gives a realistic estimate of the "typical best" outcome.

### Scaling Behavior

- **Iterations (T):** Diminishing returns. Speedup climbs rapidly at first (1.41x at T=1, 2.15x at T=10) but saturates later (2.68x at T=30, only 3.06x at T=75). Most impactful transformations are found within the first ~30 iterations.
- **Runs (K):** Also diminishing returns. K=1 → 2.66x, K=5 → 3.54x (substantial boost), K=10 → 3.75x, K=13 → 3.82x. Different runs have considerable overlap in search space coverage.

---

## Addressing LLM Stochasticity

LLM outputs are inherently stochastic — repeated COMPILOT runs on the same program produce different dialogues, different transformation sequences, and different final speedups. The paper addresses this at multiple levels:

### 1. Large Pool of Independent Runs

For each of the 150 benchmark instances, **40 independent runs** are executed. This creates a pool of results large enough for robust statistical analysis. Each run starts from scratch with a fresh optimization dialogue.

### 2. Bootstrapped Confidence Intervals

To quantify uncertainty in the aggregate metrics, the paper uses **bootstrap resampling** (1000 iterations):
- For each bootstrap iteration: resample with replacement from each benchmark's 40-run pool → compute the median speedup per benchmark → compute the geometric mean of those 150 medians.
- The 2.5th and 97.5th percentiles of the 1000 bootstrap geometric means form the **95% confidence interval**.
- Bootstrapping is chosen because it is non-parametric (no distribution assumptions), handles the complex "geometric mean of medians" statistic, and is robust to outliers.

Example: COMPILOT@30 achieves a geomean of 2.66x with 95% CI [2.60, 2.77] — indicating good stability despite stochasticity.

### 3. Multi-Run Strategy (Best-of-K)

Rather than fighting stochasticity, the paper **embraces it** as a feature. Different runs explore different regions of the optimization space, converging to different local optima. The best-of-K strategy exploits this:
- Run COMPILOT K times independently.
- Select the best-performing schedule across all K runs.
- K=5 is recommended as the sweet spot (3.54x vs 2.66x for single-run).

### 4. Exploration Variability (Multi-Modal Distributions)

The paper documents that certain benchmarks exhibit **multi-modal speedup distributions** — distinct clusters of final speedups across 40 runs. Each cluster corresponds to a different set of effective optimization schedules the LLM discovered. For example, on `gramschmidt_LARGE`, runs visibly split into separate trajectory bundles over the 30 iterations. This is why multi-run is valuable: it increases the probability of landing in one of the better-performing clusters.

### 5. Premature Stopping Mitigation

LLMs tend to stop prematurely in two scenarios:
- **After a big speedup jump** (conservatism — afraid of regressing).
- **After repeated failures** (getting stuck in a local optimum).

The framework addresses this by:
- Prompting the LLM to continue exploring when it issues `no_further_transformations` too early.
- Performance comparable to COMPILOT@30 is only reached after pushing past the LLM's **5th quit attempt**.
- Beyond ~5 pushes, returns diminish (the LLM tends to quit again after just ~1 iteration or produces marginal gains).
- In ~2% of cases, the LLM insists on terminating despite prompts — the dialogue then runs until a predefined conversation length limit.

---

## Key Design Decisions & Rationale

### Why delegate code generation to the compiler?
- Direct code generation by the LLM yields ~14-16% lower speedups.
- 17.6% of LLM-generated "passing" code was actually incorrect (false positives from output comparison).
- ~5.3x more tokens consumed for direct code generation.
- Polyhedral dependence analysis **guarantees** correctness, unlike output-comparison testing.

### Why use feedback?
- Without feedback: ~23% lower speedup at T=30 (single-run), ~28% lower for best-of-5.
- Feedback enables the LLM to learn from mistakes (illegal patterns) and successes (refining promising schedules).
- Gap widens with more iterations — feedback becomes increasingly valuable over time.

### Why anonymize variable names?
- Prevents the LLM from being biased by potentially misleading naming conventions.
- Improves robustness across diverse codebases.

### Why chain-of-thought (program analysis + reasoning)?
- Removing initial analysis: ~8% speedup drop (single-run).
- Removing per-iteration reasoning: ~11% drop with GPT-4o; 4-7% drop in best-of-5 for both tested LLMs.

### Hardware context in the prompt?
- Surprisingly, no statistically significant impact on aggregate performance.
- Hypothesis: the iterative feedback loop provides a signal strong enough to overshadow initial hardware description; LLMs may not yet be proficient at translating hardware specs into concrete optimization decisions.

---

## Key Results

| Metric | Value |
|---|---|
| **Single-run geomean speedup** (COMPILOT@30) | 2.66x over original code |
| **Best-of-5 geomean speedup** (COMPILOT_5@30) | 3.54x over original code |
| **vs Pluto optimizer** (COMPILOT_5@30) | 2.94x faster than Pluto |
| **Schedule validity rate** | ~36% runnable, ~31% invalid, ~33% illegal |
| **Wall-clock time per benchmark** | ~8.9 min for 30 iterations |
| **Token consumption at T=30** | ~200k tokens (cumulative, non-linear growth) |
| **Peak speedups** | correlation_XLARGE: 339x, trmm_XLARGE: 183x |

### LLM Comparison (Single-Run @30)

| LLM | Geomean Speedup |
|---|---|
| gemini-2.0-flash | 2.66x |
| gpt-4o | 2.63x |
| gpt-o3-mini (@25) | 2.63x |
| llama3.3 (70B) | 2.47x |
| qwq (32B) | 2.36x |
| qwen2.5-coder (32B) | 2.14x |
| gemma3 (27B) | 2.03x |
| codestral-2501 (22B) | 1.75x |

Key finding: top general-purpose models outperform coding-specialized models for this task. Reasoning models (o3-mini, qwq) don't consistently outperform top non-reasoning models.

---

## Implementation Details

- **Compiler backend:** Tiramisu (polyhedral compiler, commit 041afad).
- **Hardware:** Dual-socket Intel Xeon E5-2695 v2 (24 cores, 48 threads, 128GB RAM).
- **Benchmarks:** PolyBench/C v4.2.1 — 30 kernels × 5 dataset sizes = 150 instances.
- **Runs per instance:** 40 independent runs for stable statistics.
- **Confidence intervals:** 95% bootstrapped CIs (1000 bootstrap iterations).
- **Primary LLM:** gemini-2.0-flash (chosen for performance/cost balance).
- **LLM access:** Proprietary models via cloud APIs; open-source models on local hardware.

---

## Limitations & Failure Cases

- Benchmarks with complex loop-carried dependencies (cholesky, durbin, ludcmp) show negligible improvement (~1x).
- ~64% of LLM proposals are unproductive (invalid or illegal).
- Token costs grow non-linearly due to growing context and unproductive exchanges in later iterations.
- LLMs tend to stop prematurely after big speedup jumps or repeated failures.
- The transformation space is limited to 9 primitives — loop distribution, computation reordering, etc. are not supported.

---

## Future Directions Identified

- Enrich feedback with **specific dependency violation details** to accelerate learning.
- Augment feedback with **hardware performance counters** (cache miss rates, vector utilization).
- Hybrid LLM guidance + systematic search algorithms to escape local optima.
- Dialogue summarization to manage context length and reduce cost.
- Extend to other compiler backends (GCC/Clang flags, LLVM IR pass sequences).
