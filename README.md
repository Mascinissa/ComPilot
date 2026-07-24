# ComPilot: LLM-Guided Loop Optimization via Agentic Auto-Scheduling

ComPilot uses a general-purpose large language model as the search policy for loop optimization. The model proposes sequences of loop transformations for a given loop nest; the Tiramisu polyhedral compiler checks each proposal for legality, executes the legal ones, and reports the measured speedup. The model uses that feedback to decide what to try next, and the process repeats until it stops or a turn budget is reached. The model is used as-is through a standard chat API, with no fine-tuning or gradient updates.

This repository is the reference implementation for the paper *Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization* (PACT 2025), by Massinissa Merouani, Islem Kara Bernou, and Riyadh Baghdadi (NYU Abu Dhabi).

---

## How it works

Each benchmark is optimized through a dialogue between the model and the compiler:

1. The loop nest is shown to the model as anonymized C code, annotated with an ID per computation and accompanied by its measured baseline runtime. The model is first asked to analyze the program.
2. On each turn the model proposes a schedule (a sequence of transformations). ComPilot parses it, and either rejects a malformed or structurally invalid proposal immediately with a specific reason, or applies it through TiraLib.
3. Tiramisu checks legality by polyhedral dependence analysis. A legal schedule is compiled and timed, and its speedup over the baseline is returned to the model; an illegal or failing one is returned with the reason instead.

The transformations available to the model are fusion, interchange, parallelization, 2D and 3D tiling, unrolling, skewing, and reversal.

Rejected proposals are returned with a concrete cause (unknown computation ID, out-of-range loop level, wrong argument count, dependence violation, and so on) rather than a generic error, and for an illegal combination of transformations ComPilot can optionally point out which one is at fault. Correctness is enforced first by a lightweight validity check before any compiler call, then by Tiramisu's polyhedral legality analysis. The system is built for long unattended runs: LLM requests retry on transient failures, one benchmark's crash never aborts the rest, and the dialogue is checkpointed each turn so an interrupted run resumes where it stopped. Independent runs launched with distinct `--run-id` values give a best-of-K search, and two interaction formats are supported, a conversational transcript and a token-efficient compact mode (see [Interaction modes](#interaction-modes)).

---

## Setup & Dependencies

ComPilot drives the [Tiramisu](https://github.com/Tiramisu-Compiler/tiramisu) polyhedral compiler through **TiraLib**. Both must be built and available before running ComPilot.

1. Build Tiramisu and TiraLib, and make sure TiraLib's `config.yaml` points to your Tiramisu install.
2. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy/edit `config.yaml` (see below) to match your setup.

---

## Configuration

All settings live in `config.yaml`:

| Key | Description |
|---|---|
| `llm_id` | Model id served by your OpenAI-compatible endpoint (e.g. `gpt-4o`, `gemini-2.0-flash`, a local vLLM/Ollama model). |
| `base_url` | Base URL of the OpenAI-compatible API. |
| `openai_api_key` | API key for the endpoint (any non-empty string for most local servers). |
| `interaction_mode` | `conversational` (default) or `compact` — see [Interaction modes](#interaction-modes). |
| `benchmarks_dir` | Directory of benchmark folders; each `<name>/` must contain `<name>_generator.cpp`. |
| `tiralib_dir` | Path to a compiled TiraLib checkout. |
| `lookup_dir_suffix` | Cache of already-tried schedules is stored under `lookup_dir_<suffix>/`. |
| `base_results_dir` | Where results and checkpoints are written. |
| `max_llm_responses` | Hard cap on LLM turns per benchmark. |
| `min_nb_suggestions` | Keep nudging the LLM to explore until it has tried at least this many schedules. |
| `nb_exec_init` / `min_runs_scheds` / `max_runs_scheds` / `time_budget_factor_scheds` | Execution-measurement budgets. |
| `temperature` / `top_p` / `nb_responses` / `enable_thinking` | LLM sampling settings. |
| `use_response_lookup` | Reuse cached compiler feedback for previously-seen schedules. Set to `false` to force fresh feedback for every schedule (e.g. when comparing feedback wording or running independent stochastic runs). The numeric caches (initial execution time, skewing factors) are always kept. |

> **Security note:** do not commit real endpoint URLs or credentials. Keep `config.yaml` pointed at a placeholder and override it locally.

---

## Interaction modes

Set `interaction_mode` in `config.yaml`:

- **`conversational`** (default) — a back-and-forth transcript. Every turn resends the full history (system prompt + all prior suggestions and feedback), so the prompt grows with each iteration.
- **`compact`** — every turn sends only two messages: the system prompt and a single message holding the loop nest, a *frozen* one-time program analysis, and a running **exploration log** (a de-duplicated, numbered summary of every schedule tried and its outcome, plus the current top schedules). The LLM's next suggestion is evaluated and appended to the log.

Both modes reach comparable speedups. Compact mode keeps the per-turn prompt small and roughly constant, so its **prompt-token cost grows far more slowly** with the number of turns. It trades this for slightly more reasoning per turn (the model re-reads the state each time), so the net token savings grow with the dialogue length — modest for very short runs, substantial for long ones. Compact runs are tagged with a `_compact` suffix on the experiment name.

---

## Usage

### Single exploration

```bash
python compilot.py --experiment-name my_first_run
```

### Best-of-K (multiple independent runs)

Because different runs converge to different optima, running several independent dialogues and keeping the best schedule is recommended (K=5 is a good default). Launch runs with distinct `--run-id` values (they can run in parallel; they safely share the schedule cache):

```bash
for i in $(seq 1 5); do
  python compilot.py --experiment-name my_experiment --run-id $i &
done
wait
```

### Resuming after a crash or interruption

Runs are resumable out of the box. If a run is interrupted (crash, timeout, LLM outage), simply re-launch the **same command**: completed benchmarks are skipped, and any benchmark that was mid-dialogue resumes from its checkpoint in `<results_dir>/checkpoints/`.

---

## Output

For each experiment, `base_results_dir/<experiment_name>/` contains:
- `results_<experiment_name>.json` — full optimization dialogue, explored schedules, best schedule, timings, and token usage per benchmark.
- `live_results_<experiment_name>.csv` — one summary row per benchmark (best speedup, counts, timing).
- `checkpoints/` — per-benchmark resume state (auto-deleted once a benchmark finishes).

---

## Evaluation

The paper reports two aggregate speedup metrics, both measured against each program's original runtime and as a function of the exploration budget `T` (the number of schedules tried):

- **COMPILOT@T** — the single-run result: for each benchmark take the best speedup found within the first `T` schedules, take the median of that across runs, then the geometric mean across benchmarks.
- **COMPILOT_K@T** — the best-of-K result: the typical speedup obtained by running `K` independent explorations and keeping the best, again aggregated across benchmarks.

Because the metrics aggregate over independent runs, first produce several runs of the same experiment with distinct `--run-id` values (see [Best-of-K](#best-of-k-multiple-independent-runs)), then aggregate:

```bash
python aggregate_json_results.py -r results/<experiment_name> --k 5
```

This prints a table of COMPILOT@T and COMPILOT_K@T over `T = 5, 10, ..., 30` and writes a per-benchmark CSV (`aggregate_results_*.csv`) alongside it. The benchmarks used in the paper are the [PolyBench](https://web.cse.ohio-state.edu/~pouchet.2/software/polybench/) kernels.

For a quick per-run sanity check without aggregating, `analyze_dialogue.py results/<...>/results_*.json` prints, per benchmark, how many proposed schedules were runnable vs. rejected vs. crashed, the runnable rate, and the best speedup (works for both interaction modes).

To regenerate the paper's figures, see `plotting.py` (requires `matplotlib`/`plotly`).

---

## File Structure

- `compilot.py` — main driver: runs the optimization dialogue for every benchmark, with checkpointing/resume.
- `compilot_utils.py` — parsing, validation, compiler feedback generation, and the TiraLib/Tiramisu backend interface.
- `prompts.py` — the system instructions and message templates used with the LLM.
- `config.yaml` — centralized configuration.
- `analyze_dialogue.py` — quick per-run outcome analysis.
- `aggregate_json_results.py` / `eval_utils.py` — the paper's aggregate metrics (COMPILOT@T and best-of-K) across runs.
- `plotting.py` — supplementary figure-generation code used for the paper's plots (not needed to run ComPilot or compute the metrics).

## Citation

If you use this code in your research, please cite:
> Merouani, M., Bernou, I. K., & Baghdadi, R. (2025). Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization. In *Proceedings of the 2025 International Conference on Parallel Architectures and Compilation Techniques (PACT)*.

## License

Released under the [Apache License 2.0](LICENSE).
