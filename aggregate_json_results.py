import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import eval_utils

def get_best_speedup_over_search_json(explored_scheds, init_exec_time):
    cur_min = init_exec_time
    best_sched = 'None'
    speedups = [1.0] # t=0
    scheds = [best_sched]
    
    # Check if the first element is the injected baseline schedule
    start_idx = 1 if len(explored_scheds) > 0 and explored_scheds[0].get('schedule') == "" and explored_scheds[0].get('outcome') == 'run' else 0
    
    for elem in explored_scheds[start_idx:]:
        outcome = elem.get('outcome')
        exec_time = elem.get('execution_time')
        
        if outcome in ['ran', 'run'] and exec_time is not None and exec_time < cur_min:
            cur_min = exec_time
            best_sched = elem.get('schedule', '')
            
        speedups.append(init_exec_time / cur_min)
        scheds.append(best_sched)
        
    return speedups, scheds


def aggregate_json_files(files_list):
    full_dict = dict()
    for file_name in files_list:
        with open(file_name, 'r') as f:
            d = json.load(f)
        
        programs = d.get('programs', {})
        for bench, bench_data in programs.items():
            full_dict[bench] = full_dict.get(bench, {"explorations_list": [], "init_exec_times": []})
            full_dict[bench]["explorations_list"].append(bench_data.get('Explored_schedules', []))
            full_dict[bench]["init_exec_times"].append(bench_data.get('Initial_execution_time', 1.0))
            
    return full_dict


def main():
    parser = argparse.ArgumentParser(description="Aggregate json results from COMPILOT explorations.")
    parser.add_argument('--results-dir', '-r', required=True, help="Path to the results directory")
    parser.add_argument('--num-runs', type=int, default=None, help="Force aggregation using exactly first N runs. Default uses all available runs.")
    parser.add_argument('--k', type=int, default=5, help="Value of K for typical best-of-K metric.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} does not exist.")
        return
    experiment_name = results_dir.name
    # Find all json files matching the pattern
    files_list = sorted(list(results_dir.glob(f"results_{experiment_name}_run*.json")))
    if args.num_runs is not None:
        if len(files_list) < args.num_runs:
            print(f"Warning: Found {len(files_list)} files, but requested {args.num_runs}.")
        files_list = files_list[:args.num_runs]

    num_runs = len(files_list)
    if num_runs == 0:
        print(f"Error: No result files found for experiment {experiment_name} in {results_dir}.")
        return

    print(f"Aggregating {num_runs} run(s) for experiment {experiment_name}...")
    
    full_dict = aggregate_json_files(files_list)
    
    timestamps_to_report = [5, 10, 15, 20, 25, 30]
    max_iteration = max(timestamps_to_report)

    # Build the tensor
    per_bench_arrays = []
    benchmarks = sorted(list(full_dict.keys()))
    
    for bench_name in benchmarks:
        raw_speedup_list_list = []
        # Fallback to 1.0 if not found, but it should be uniform across runs usually
        init_time = full_dict[bench_name]["init_exec_times"][0] 
        
        for exploration_trace in full_dict[bench_name]['explorations_list']:
            speedups, _ = get_best_speedup_over_search_json(exploration_trace, init_time)
            raw_speedup_list_list.append(speedups)
            
        speedup_list_list = eval_utils.unify_list_sizes(raw_speedup_list_list, pad_or_trim='pad', pad_with_last=True, force_pad_len=max_iteration+1)
        
        nb_runs = len(speedup_list_list)
        benchmark_array = np.zeros((nb_runs, max_iteration+1))
        
        for iteration in range(1, max_iteration+1):
            benchmark_array[:, iteration] = [speedup_list_list[run_id][iteration] for run_id in range(nb_runs)]
            
        per_bench_arrays.append(benchmark_array)
        
    tensor = np.stack(per_bench_arrays)

    # Calculate and Print Top Level Summary Metrics
    print(f"\n{'='*50}")
    print(f"EVALUATION REPORT: {experiment_name}")
    print(f"{'='*50}")
    
    summary_df = pd.DataFrame(index=timestamps_to_report, columns=["COMPILOT@T (Single-Run)", f"COMPILOT_{args.k}@T (Best-of-{args.k})"])
    summary_df.index.name = "Iteration (T)"

    for T in timestamps_to_report:
        # Avoid crashing if a benchmark never ran that many iterations
        try:
            med_geomean = eval_utils.compute_geomean_medians(tensor, T)
            best_k = eval_utils.compute_gmean_typical_best_of_k(tensor, T, K=args.k)
            summary_df.loc[T] = [med_geomean, best_k]
        except Exception as e:
            summary_df.loc[T] = [np.nan, np.nan]

    print("\nAggregate Speedup Ratios (vs Original):")
    print(summary_df.to_markdown())

    # Build the per-benchmark dataframe to be consistent with aggregate_results.py format
    df = pd.DataFrame(columns=['benchmark'] + 
                              [f'med_{num_runs}@{i}' for i in timestamps_to_report] + 
                              [f'max_{num_runs}@{i}' for i in timestamps_to_report] + 
                              [f'geomean_{num_runs}@{i}' for i in timestamps_to_report])
    
    for idx, bench_name in enumerate(benchmarks):
        bench_slice = tensor[idx] # shape: (num_runs, max_iteration+1)
        
        medians = [np.median(bench_slice[:, i]) for i in timestamps_to_report]
        maxs = [np.max(bench_slice[:, i]) for i in timestamps_to_report]
        # eval_utils.geometric_mean fails on numpy floats or zeros sometimes, convert strictly
        geomeans = [eval_utils.geometric_mean(list(bench_slice[:, i])) if len(bench_slice[:, i]) > 0 else 1.0 for i in timestamps_to_report]
        
        row = [bench_name] + medians + maxs + geomeans
        df.loc[len(df)] = row

    csv_path = results_dir / f"aggregate_results_{num_runs}runs_{experiment_name}_json.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\nSaved detailed CSV report to: {csv_path}")

if __name__ == "__main__":
    main()
