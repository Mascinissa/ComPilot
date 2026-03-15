# ComPilot: LLM-Guided Loop Optimization via Agentic Auto-Scheduling

ComPilot (Compiler Pilot) is an automated loop optimization framework that uses off-the-shelf Large Language Models (LLMs) as interactive optimization agents. It proposes loop transformations to a compiler in a closed feedback loop. By interacting iteratively, the LLM refines its strategy based on compiler feedback (validity, legality, and execution speedups), driving performance improvements without the need for gradient updates or fine-tuning.

This project is the official implementation of the paper:
**Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization** (PACT 2025).
*Authors: Massinissa Merouani, Islem Kara Bernou, Riyadh Baghdadi (NYU Abu Dhabi)*

---

## Architecture Overview

The system runs an **optimization dialogue** between an LLM agent and a polyhedral compiler (Tiramisu).
1. **Context Initializer**: Prepares the LLM with the C++ loop nest and initial execution time.
2. **Interaction Loop Handler**: Connects LLM responses to compiler API calls and generates feedback based on successful execution or errors.
3. **Compiler Backend (Tiramisu)**: Processes semantic validity, legality (via polyhedral dependence analysis), generates code, and evaluates the performance on target hardware.

## Features

- **Multi-run Best-of-K Strategy**: Embraces LLM stochasticity by allowing parallel exploration paths and selecting the most performant schedules.
- **Nine Transformation Primitives**: Fusion, Interchange, Parallelization, 2D/3D Tiling, Unrolling, Skewing, Reversal.
- **Robust Error Handling**: Syntactic and semantic feedback helps the LLM recover from poor schedule choices or invalid primitives.
- **Hardware Agnostic LLM Context**: Provides the LLM with the active hardware specs dynamically.

---

## Setup & Dependencies

*This project relies heavily on the Tiramisu Compiler API and python dependencies (like `openai` and `pyyaml`). Ensure your environment has Tiramisu compiled and accessible.*

1. Clone this repository.
2. Open `config.yaml` and update the required paths and LLM configuration parameters to match your setup:
   - `llm_id`: The LLM model to query (e.g. `gemini-2.0-flash`, `gpt-4o`).
   - `openai_api_key` & `base_url`: Pointing to your LLM provider.
   - `benchmarks_dir`: Path to your Polybench (or equivalent) C++ generator files.
   - `tiralib_dir`: Path to the compiled TiraLib.

---

## Usage

### Single Exploration
To start a single agentic optimization run on the provided loop nest benchmarks, simply configure `config.yaml` and run:

```bash
python compilot.py --experiment-name "my_first_run"
```

### Multi-Run Best-of-K (SLURM / HPC Deployment)
Because ComPilot uses a Best-of-K strategy to find peak optimizations, executing multiple independent dialogues simultaneously is recommended (K=5 is often ideal).

We provide `run_duplicated_compilot.sh` to seamlessly submit and monitor multiple Slurm jobs:

```bash
# Starts K=5 parallel optimization dialogues using Slurm/Screen
./run_duplicated_compilot.sh 5 my_experiment_prefix
```
The script uses `screen` to keep the runner alive and restarts runs automatically if they crash abruptly.

---

## File Structure

- `compilot.py`: The main script that configures the agent and runs the optimization loops for all benchmarks. 
- `compilot_utils.py`: Contains the helper logic for parsing, compiling, error handling, and driving the Tiramisu backend.
- `prompts.py`: Defines the context prompts and system instructions used by the LLM.
- `config.yaml`: Centralized configuration for the LLM, thresholds, parameters, and directories.
- `run_duplicated_compilot.sh`: A helper bash script designed for launching parallel best-of-K explorations on SLURM clusters.
- `all_utils.py`: Additional parser and expression tree functions imported by the framework.

## Citation

If you use this code in your research, please consider citing the underlying publication:
> Merouani, M., Bernou, I. K., & Baghdadi, R. (2025). Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization. In *Proceedings of the 2025 International Conference on Parallel Architectures and Compilation Techniques (PACT)*.
