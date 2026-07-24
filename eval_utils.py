"""Aggregate metrics used to reproduce the paper's numbers from ComPilot result
JSONs: best-of-K and geometric-mean-of-medians over the search, with optional
bootstrap confidence intervals. Imported by aggregate_json_results.py; not needed
to run ComPilot itself. Figure-generation code lives in plotting.py."""
import numpy as np
from statistics import geometric_mean
from scipy.stats import gmean
from tqdm import tqdm

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


def bootstrap_confidence_interval_geomean_median(S, n_bootstrap=1000, confidence_level=0.95):
    """
    Computes the confidence interval for the geometric mean of medians using bootstrap.

    Args:
        S (numpy.ndarray): A 2D numpy array where S[i, j] is the speedup of run j on benchmark i.
                           Shape should be (num_benchmarks, num_runs_per_benchmark).
        n_bootstrap (int): Number of bootstrap samples to generate.
        confidence_level (float): The desired confidence level (e.g., 0.95 for 95% CI).

    Returns:
        tuple: A tuple containing the lower and upper bounds of the confidence interval.
    """
    assert S.ndim == 2 # takes the array of speedup at one iteration (not the tensor of speedups)
    num_benchmarks = S.shape[0]
    num_runs_per_benchmark = S.shape[1]
    bootstrap_geomean_medians = []

    for _ in tqdm(range(n_bootstrap)):
        bootstrap_medians = []
        for i in range(num_benchmarks):
            # Resample with replacement from runs for benchmark i
            resampled_speedups = np.random.choice(S[i, :], size=num_runs_per_benchmark, replace=True)
            bootstrap_medians.append(np.median(resampled_speedups))
        bootstrap_geomean_medians.append(gmean(bootstrap_medians))

    # Calculate percentiles for confidence interval
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 - (1 - confidence_level) / 2) * 100

    lower_bound = np.percentile(bootstrap_geomean_medians, lower_percentile)
    upper_bound = np.percentile(bootstrap_geomean_medians, upper_percentile)

    return lower_bound, upper_bound


def compute_geomean_medians(speedups_tensor, iteration, compute_ci=False):
    S = speedups_tensor[:,:,iteration]
    medians = np.median(S, axis=1)  # Shape: (150,)
    G = gmean(medians)
    if compute_ci:
        CI = bootstrap_confidence_interval_geomean_median(S)
        return G, CI
    return G


def typical_best_of_k_benchmark(runs, K, n_simulations, compute_ci=False, n_bootstraps=1000):
    """
    Simulate picking K runs with replacement, take the maximum, and return the median over n_simulations.
    
    Args:
        runs: Array of speedup values for a benchmark (length = n_runs)
        K: Number of runs to sample
        n_simulations: Number of times to repeat the simulation
    
    Returns:
        Median of the maximum speedups
    """
    best_of_k = np.median([np.max(np.random.choice(runs, K, replace=True)) for _ in range(n_simulations)])
    if compute_ci:
        bootstrap_medians = []
        for _ in range(n_bootstraps):
            resampled_runs = np.random.choice(runs, len(runs), replace=True)
            median_max = typical_best_of_k_benchmark(resampled_runs, K, n_simulations, compute_ci=False)
            bootstrap_medians.append(median_max)
        lower = np.percentile(bootstrap_medians, 2.5)
        upper = np.percentile(bootstrap_medians, 97.5)
        return best_of_k, lower, upper
    return best_of_k


def compute_gmean_typical_best_of_k(speedups_tensor, iteration, K, compute_ci=False, n_simulations=1000, n_bootstraps=100):
    S = speedups_tensor[:,:,iteration]
    medians = []
    n_benchmarks = S.shape[0]
    n_runs = S.shape[1]
    for i in range(n_benchmarks): # for each benchmark
        runs = S[i, :]  # all the runsfor benchmark i
        median_max = typical_best_of_k_benchmark(runs, K, n_simulations)
        medians.append(median_max)
    
    gmean_best_of_k = gmean(medians)  # Geometric mean of the medians
    if compute_ci:
        # Compute 95% Confidence Interval using Bootstrapping
        bootstrap_gmeans = []
        for _ in tqdm(range(n_bootstraps)):
            # Resample runs with replacement for each benchmark
            resampled_S = np.array([np.random.choice(S[i, :], n_runs, replace=True) for i in range(n_benchmarks)])
            
            # Compute medians for resampled benchmarks
            resampled_medians = []
            for i in range(n_benchmarks):
                runs = resampled_S[i, :]
                median_max = typical_best_of_k_benchmark(runs, K, n_simulations)
                resampled_medians.append(median_max)
            
            # Compute geometric mean for this bootstrap sample
            bootstrap_gmeans.append(gmean(resampled_medians))

        # Calculate the 2.5th and 97.5th percentiles for the 95% CI
        lower_bound = np.percentile(bootstrap_gmeans, 2.5)
        upper_bound = np.percentile(bootstrap_gmeans, 97.5)
        return gmean_best_of_k, (lower_bound, upper_bound)
    return gmean_best_of_k

