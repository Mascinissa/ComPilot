import pickle
import numpy as np
from statistics import geometric_mean
import pandas as pd

# Parameters
experiment_name = 'experiment_name'
runs_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
num_runs = len(runs_list)
timestamps_to_report = [5, 10, 15, 20, 25, -1]
# timestamps_to_report = list(range(5,15+1,5)) + [-1]
results_dir = "./results/"+experiment_name



def get_best_speedup_over_search(explored_scheds, init_exec_time):
    cur_min = init_exec_time
    best_sched = 'None'
    speedups = [1.0]
    scheds = [best_sched]
    for elem in explored_scheds:
        if elem[1]=='ran' and elem[3]<cur_min:
            cur_min = elem[3]
            best_sched = elem[0]
        speedups.append(init_exec_time/cur_min)
        scheds.append(best_sched)
    return speedups, scheds

def aggregate_result_pickle_files(files_list):
    full_dict = dict()
    for file_name in files_list:
        with open(file_name, 'rb') as f:
            d = pickle.load(f)
        for bench in d:
            full_dict[bench] = full_dict.get(bench, {"explorations_list":[]})
            full_dict[bench]["explorations_list"].append(d[bench]['explored_scheds'])
    return full_dict

def unify_list_sizes(lists, pad_or_trim = 'trim', pad_with_last = False, force_pad_len = None):
    if pad_or_trim == 'trim':
        min_length = min(len(lst) for lst in lists)
        lists = [lst[:min_length] for lst in lists]
    elif pad_or_trim == 'pad':
        max_length = max(len(lst) for lst in lists)
        if force_pad_len is not None:
            max_length = max(force_pad_len, max_length)
        # Pad all lists with None to make them of equal length
        lists = [lst + [lst[-1] if pad_with_last else None] * (max_length - len(lst)) for lst in lists]
    else:
        return
    return lists
def compute_medians(lists):
    # Create the result list
    result = []
    for i in range(len(lists[0])):
        # Collect values at position i from all lists excluding the current one
        values = [lst[i] for lst in lists if lst[i] is not None]
        result.append(np.median(values) if values else None)
    return result
def compute_maxs(lists):
    # Create the result list
    result = []
    for i in range(len(lists[0])):
        # Collect values at position i from all lists excluding the current one
        values = [lst[i] for lst in lists if lst[i] is not None]
        result.append(np.max(values) if values else None)
    return result
def compute_geomeans(lists):
    # Create the result list
    result = []
    for i in range(len(lists[0])):
        # Collect values at position i from all lists excluding the current one
        values = [lst[i] for lst in lists if lst[i] is not None]
        result.append(geometric_mean(values) if values else None)
    return result


if __name__=="__main__":

    files_list = [f'{results_dir}/per_bench_results_{experiment_name}_run{i}.pkl' for i in runs_list]
    
    full_dict = aggregate_result_pickle_files(files_list)
    
    with open(f'{results_dir}/per_bench_results_{experiment_name}_{num_runs}runs_full.pkl', 'wb') as f:
        pickle.dump(full_dict, f)
    print(f"Generated {results_dir}/per_bench_results_{experiment_name}_{num_runs}runs_full.pkl")
    
    with open('init_exec_times_dict.pkl', 'rb') as f:
        init_times_dict = pickle.load(f)
    
    df = pd.DataFrame(columns=['benchmark'] + [f'med_{num_runs}@{i}' for i in timestamps_to_report] + [f'max_{num_runs}@{i}' for i in timestamps_to_report] + [f'geomean_{num_runs}@{i}' for i in timestamps_to_report])
    
    for bench_name in sorted(list(full_dict.keys())):
        raw_speedup_list_list = []
        for exploration_trace in full_dict[bench_name]['explorations_list']:
            speedups, scheds = get_best_speedup_over_search(exploration_trace, init_times_dict[bench_name])
            raw_speedup_list_list.append(speedups)
        speedup_list_list = unify_list_sizes(raw_speedup_list_list, pad_or_trim = 'pad', pad_with_last=True, force_pad_len = timestamps_to_report[-2]+1)
        maxs = compute_maxs(speedup_list_list)
        medians = compute_medians(speedup_list_list)
        geomeans = compute_geomeans(speedup_list_list)
    
        row = [bench_name] + [medians[i] for i in timestamps_to_report] + [maxs[i] for i in timestamps_to_report] + [geomeans[i] for i in timestamps_to_report]
    
        df.loc[len(df)]= row
    
    df.to_csv(f'{results_dir}/aggregate_results_{num_runs}runs_{experiment_name}.csv', index=False)
    print(f"Generated {results_dir}/aggregate_results_{num_runs}runs_{experiment_name}.csv")
