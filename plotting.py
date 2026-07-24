"""Figure generation and data-loading helpers used to produce the paper's plots
(speedup curves, grouped/stacked bars, heatmaps) from ComPilot result files. This
module is supplementary: it is not needed to run ComPilot or to compute the
aggregate metrics (see eval_utils.py), and it requires matplotlib/plotly."""
import pickle
import re
from collections import Counter
from pathlib import Path
from statistics import geometric_mean
import numpy as np
import pandas as pd
from scipy.stats import gmean, wilcoxon
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from eval_utils import unify_list_sizes, compute_geomean_medians, compute_gmean_typical_best_of_k

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
        assert len(d.keys())==150
        for bench in d:
            full_dict[bench] = full_dict.get(bench, {"explorations_list":[]})
            full_dict[bench]["explorations_list"].append(d[bench]['explored_scheds'])
    return full_dict


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


def get_expriment_results_dict(experiment_path):
    files_list = [file for file in list(Path(experiment_path).glob("*.pkl")) if not str(file).endswith('full.pkl')]
    full_dict = aggregate_result_pickle_files(files_list)
    return full_dict


def get_speedups_per_run_tensor(results_full_dict, init_exec_times_pkl, max_iteration=30, max_nb_runs=None):
    # init_exec_times_pkl: path to a pickle mapping benchmark name -> baseline runtime.
    # return a numpy array of shape (nb_benchmarks, nb_runs, nb_iterations) 
    # where each element i,j,k represents the speedup of benchmark i during run j at iteration k

    with open(init_exec_times_pkl, 'rb') as f:
        init_times_dict = pickle.load(f)

    explorations_lens = [len(exploration_trace) for bench_name in results_full_dict.keys() for exploration_trace in results_full_dict[bench_name]['explorations_list']]
    print(f'Number of exploration shorter than the desired max nb iters: {len([i for i in explorations_lens if i<max_iteration])} out of {len(explorations_lens)}')
    
    per_bench_arrays = []
    for bench_name in sorted(list(results_full_dict.keys())):
        raw_speedup_list_list = []
        for exploration_trace in results_full_dict[bench_name]['explorations_list']:
            speedups, scheds = get_best_speedup_over_search(exploration_trace, init_times_dict[bench_name])
            raw_speedup_list_list.append(speedups)
        speedup_list_list = unify_list_sizes(raw_speedup_list_list, pad_or_trim = 'pad', pad_with_last=True, force_pad_len = max_iteration+1)
        nb_runs = max_nb_runs if max_nb_runs else len(speedup_list_list)
        benchmark_array = np.zeros((nb_runs, max_iteration+1)) #+1 so that iteration 0 is always all 0
        
        for iteration in range(1, max_iteration+1):
            benchmark_array[:, iteration] = [speedup_list_list[run_id][iteration] for run_id  in range(nb_runs)] 
        per_bench_arrays.append(benchmark_array)
    
    tensor = np.stack(per_bench_arrays)
    benchmark_names = sorted(list(results_full_dict.keys()))
    return tensor, benchmark_names


def generate_comparison_table(speedups_tensor_A, speedups_tensor_B, iterations_to_compare, K=5):
    gmeans_medians_A = []
    gmeans_medians_B = []
    gmeans_best_of_k_A = []
    gmeans_best_of_k_B = []
    for iteration in tqdm(iterations_to_compare):
        gmeans_medians_A.append(compute_geomean_medians(speedups_tensor_A, iteration, compute_ci=False))
        gmeans_best_of_k_A.append(compute_gmean_typical_best_of_k(speedups_tensor_A, iteration, compute_ci=False, K=K))
        gmeans_medians_B.append(compute_geomean_medians(speedups_tensor_B, iteration, compute_ci=False))
        gmeans_best_of_k_B.append(compute_gmean_typical_best_of_k(speedups_tensor_B, iteration, compute_ci=False, K=K))

    results = np.stack([gmeans_medians_A,gmeans_medians_B,gmeans_best_of_k_A, gmeans_best_of_k_B])
    
    index = pd.MultiIndex.from_tuples(
    [("Geomean-Med Single-Run", "A"), ("Geomean-Med Single-Run", "B"), (f"Geomean-Med Best of {K}", "A"), (f"Geomean-Med Best of {K}", "B")], 
    names=["Metric", "Experiment"]
    )

    # Define column labels
    columns = [f'@{i}' for i in iterations_to_compare]
    
    # Create DataFrame
    df = pd.DataFrame(results, index=index, columns=columns)
    return df


def get_medians_ci(speedups_tensor, iteration, n_bootstrap=2000, confidence_level=0.95, relative_ci=False):
    """Calculates bootstrap confidence interval for the median for each benchmark.
    Caution! if relative_ci=True return relative CIs for plotting purposes not actual CIs"""
    median_estimates = []
    ci_errors_lower = []
    ci_errors_upper = []

    speedups_array = speedups_tensor[:,:,iteration]
    
    for i in tqdm(range(speedups_array.shape[0])): # For each benchmark
        bootstrap_medians = []
        row = speedups_array[i, :]
        for _ in range(n_bootstrap):
            resampled_row = np.random.choice(row, size=len(row), replace=True)
            bootstrap_medians.append(np.median(resampled_row))
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 - (1 - confidence_level) / 2) * 100
        lower_bound = np.percentile(bootstrap_medians, lower_percentile)
        upper_bound = np.percentile(bootstrap_medians, upper_percentile)
        if relative_ci:
            median_est, (lower_err, upper_err) = np.median(row), (np.median(row) - lower_bound, upper_bound - np.median(row)) # Return point estimate and error tuple
        else:
            median_est, (lower_err, upper_err) = np.median(row), (lower_bound, upper_bound) # Return point estimate and error tuple
            
        
        median_estimates.append(median_est)
        ci_errors_lower.append(lower_err)
        ci_errors_upper.append(upper_err)
    return median_estimates, ci_errors_lower, ci_errors_upper


def hex_to_rgba(hex_color, opacity=1.0):
    # Remove the '#' if present
    hex_color = hex_color.lstrip('#')

    # Convert 3-digit hex to 6-digit
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])

    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Return as RGBA tuple
    return (r, g, b, opacity)


def calculate_percentages(string_list):
    # Count occurrences of each string
    counter = Counter(string_list)
    
    # Calculate total number of elements
    total = len(string_list)
    
    # Calculate percentage for each string
    percentages = {item: (count / total) * 100 for item, count in counter.items()}
    
    return percentages


def plot_grouped_bar_char(median_speedups_per_benchmark, ci_errors_lower, ci_errors_upper, benchmark_names, ytickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500]):
    # Extract info
    program_names = [name.split('_')[1] for name in benchmark_names]
    sizes = [name.split('_')[-1].upper() for name in benchmark_names]
    unique_programs = sorted(set(program_names))
    size_order = ['MINI', 'SMALL', 'MEDIUM', 'LARGE', 'XLARGE']
    
    # Bar positioning
    bar_width = 0.4
    group_spacing = 0.5
    x_base = np.arange(len(unique_programs)) * (len(size_order) * bar_width + group_spacing)
    
    # Color palette (colorblind-friendly)
    # colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e'] # 2nd best
    # colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
    # colors = ['#EE7733', '#0077BB', '#33BBEE', '#CC3311', '#009988']
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854'] # pastel
    # colors = ['#EE7733', '#0077BB', '#33BBEE', '#CC3311', '#009988']
    # colors = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442']
    
    
    
    
    
    
    fig = go.Figure()
    
    # Add bars
    for idx, size in enumerate(size_order):
        indices = [i for i, s in enumerate(sizes) if s == size]
        speedup_vals = [median_speedups_per_benchmark[i] for i in indices]
        lower_err = [median_speedups_per_benchmark[i]-ci_errors_lower[i] for i in indices]
        upper_err = [ci_errors_upper[i]-median_speedups_per_benchmark[i] for i in indices]
        prog_subset = [program_names[i] for i in indices]
        pos = [x_base[unique_programs.index(prog)] + idx * bar_width for prog in prog_subset]
    
        fig.add_trace(go.Bar(
            x=pos,
            y=speedup_vals,
            name=size,
            width=bar_width,
            marker_color=colors[::-1][idx],
            marker=dict(cornerradius="30%"),
            error_y=dict(
                type='data',
                symmetric=False,
                array=upper_err,
                arrayminus=lower_err,
                color='rgba(0, 0, 0, 0.5)',
                thickness=1, 
                width=1.5
            ),
            hovertemplate='<b>%{y:.2f}×</b><br>Program: %{x}<br>Size: ' + size + '<extra></extra>'
        ))
    
    # Final layout polish
    fig.update_layout(
        xaxis=dict(
            tickvals=[x + bar_width * (len(size_order) - 1) / 2 for x in x_base],
            ticktext=unique_programs,
            tickangle=-45,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title='Median Speedup',
            type='log',
            gridcolor='lightgrey',
            tickfont=dict(size=10),
            titlefont=dict(size=12),
            tickvals=ytickvals,  # Specify ticks to include 1
    
        ),
        legend=dict(
            # title='Problem Sizes:',
            font=dict(size=10),
            orientation='h',
            yanchor='bottom',
            y=1,
            xanchor='right',
            x=1
        ),
        bargap=group_spacing / (bar_width * len(size_order) + group_spacing),
        margin=dict(t=0, b=0, l=0, r=0),
        height=330,
        width=1000,
        plot_bgcolor='white'
    )

    return fig


def plot_line_with_ci(data, xtickvals=None, ytickvals=None, x_title='x_title', trace_name='trace_name', color_id=0):
    x_values = [item[0] for item in data]
    y_values = [item[1] for item in data]
    lower_bounds = [item[2] for item in data]
    upper_bounds = [item[3] for item in data]
    
    # Calculate error bounds
    lower_errors = [y - lower for y, lower in zip(y_values, lower_bounds)]
    upper_errors = [upper - y for y, upper in zip(y_values, upper_bounds)]

    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
    color = colors[color_id]
    
    # Create figure
    fig = go.Figure()
    
    # Add line with markers
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines+markers',
        name=trace_name,
        line=dict(color=color, width=2),  # Using first color from your pastel palette
        marker=dict(
            size=8,
            color=color,
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>%{y:.2f}×</b><br>X: %{x}<extra></extra>'
    ))
    
    # Add confidence interval as shaded area
    fig.add_trace(go.Scatter(
        x=x_values + x_values[::-1],
        y=[y + err for y, err in zip(y_values, upper_errors)] + 
           [y - err for y, err in zip(y_values[::-1], lower_errors[::-1])],
        fill='toself',
        fillcolor= f'rgba{hex_to_rgba(color,0.4)}',  # Transparent version of line color
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Layout configuration to match your bar chart
    fig.update_layout(
        xaxis=dict(
            title=x_title,
            gridcolor='lightgrey',
            tickfont=dict(size=10),
            titlefont=dict(size=12),
            tickvals=xtickvals if xtickvals else list(range(1, len(data)+1))
        ),
        yaxis=dict(
            title='Geometric Mean Speedup',
            gridcolor='lightgrey',
            tickfont=dict(size=10),
            titlefont=dict(size=12),
            # Uncomment these if you want log scale like your first chart:
            type='log',
            # tickvals=ytickvals if ytickvals else [1, 2, 5, 10, 20, 50, 100],
            # range=[-0.1, np.log10(4.2)]
        ),
        legend=dict(
            font=dict(size=10),
            orientation='h',
            yanchor='bottom',
            y=1,
            xanchor='right',
            x=1
        ),
        margin=dict(t=0, b=0, l=0, r=0),
        height=330,
        width=500,
        plot_bgcolor='white'
    )
    
    return fig


def plot_heatmap(matrix, row_labels=None, col_labels=None, log_scale=False, annotation_format=None,
                 colorscale='pastel', width=1000, height=330, x_title="Number of Iterations", 
                 y_title="Number of Runs", colorbar_title="Value", layout_params=None):
    """
    Plots a heatmap of a numpy matrix using plotly, styled consistently with other charts.

    Args:
        matrix (numpy.ndarray): The input numpy matrix to visualize.
        row_labels (list or numpy.ndarray, optional): Labels for the rows (y-axis).
        col_labels (list or numpy.ndarray, optional): Labels for the columns (x-axis).
        log_scale (bool, optional): Whether to use a logarithmic color scale.
        annotation_format (str, optional): Format string for cell annotations (e.g., ".2f", "d").
        colorscale (str or list, optional): Colorscale for the heatmap. Options:
                                          - 'pastel': Matches the pastel colors from the bar chart
                                          - 'teal-orange': A teal to orange gradient
                                          - 'blue-red': A blue to red gradient
                                          - Any standard plotly colorscale name
        width (int, optional): Figure width in pixels. Defaults to 1000.
        height (int, optional): Figure height in pixels. Defaults to 330.
        x_title (str, optional): Title for the x-axis.
        y_title (str, optional): Title for the y-axis.
        colorbar_title (str, optional): Title for the colorbar.
        layout_params (dict, optional): Additional layout parameters to pass to update_layout.

    Returns:
        plotly.graph_objects.Figure: The plotly heatmap figure.
    """
    # Define custom colorscales to match the bar chart aesthetic
    custom_colorscales = {
        'pastel': [
            [0, '#66c2a5'],     # Light teal
            [0.25, '#fc8d62'],  # Light orange
            [0.5, '#8da0cb'],   # Light blue
            [0.75, '#e78ac3'],  # Light purple
            [1, '#a6d854']      # Light green
        ],
        'teal-orange': [
            [0, '#1b9e77'],     
            [0.7, '#66c2a5'],
            [0.85, '#fc8d62'],      
            [1, '#d95f02']
        ],
        'blue-red': [
            [0, '#8da0cb'],     # Light blue
            [1, '#e78ac3']      # Light purple
        ]
    }
    
    # Set the colorscale
    if isinstance(colorscale, str) and colorscale in custom_colorscales:
        colorscale = custom_colorscales[colorscale]
    
    if row_labels is None:
        row_labels = list(range(matrix.shape[0]))
    if col_labels is None:
        col_labels = list(range(matrix.shape[1]))

    z_matrix = np.copy(matrix)  # Create a copy to avoid modifying the original matrix
    original_values = np.copy(z_matrix)  # Store original values for annotations

    if log_scale:
        # Handle non-positive values for log scale
        min_positive = np.min(z_matrix[z_matrix > 0]) if np.any(z_matrix > 0) else 1e-9
        z_matrix[z_matrix <= 0] = min_positive
        z_matrix = np.log10(z_matrix)

    # Create heatmap trace
    heatmap_data = go.Heatmap(
        z=z_matrix,
        x=col_labels,
        y=row_labels,
        colorscale=colorscale,
        showscale=False,
        hovertemplate='<b>Value: %{z:.4g}</b><br>X: %{x}<br>Y: %{y}<extra></extra>',
        xgap=0.25,  # Add small gaps between cells
        ygap=0.25
    )

    # Add annotations if format is specified
    if annotation_format:
        annotation_text = np.array([["" for _ in range(matrix.shape[1])] for _ in range(matrix.shape[0])], dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                annotation_text[i, j] = format(original_values[i, j], annotation_format)

        heatmap_data.text = annotation_text
        heatmap_data.texttemplate = "%{text}"

    heatmap_data.textfont = {"color": "white"}

    # Create figure
    fig = go.Figure(data=[heatmap_data])

    # Basic layout matching the bar chart style
    default_layout = dict(
        xaxis=dict(
            title=x_title,
            titlefont=dict(size=9),
            tickfont=dict(size=7),
            gridcolor='lightgrey',
            title_standoff=3  # Reduced spacing between title and axis
        ),
        yaxis=dict(
            title=y_title,
            titlefont=dict(size=9),
            tickfont=dict(size=7),
            gridcolor='lightgrey',
            autorange="reversed",  # Keep y-axis reversed as in original
            title_standoff=2  # Reduced spacing between title and axis
        ),
        margin=dict(t=0, b=0, l=0, r=0),
        height=height,
        width=width,
        plot_bgcolor='white'
    )
    
    # Update with any additional layout parameters
    if layout_params:
        default_layout.update(layout_params)
    
    fig.update_layout(**default_layout)
    
    return fig


def plot_stacked_bar_chart(benchmark_data):
    """
    Create a stacked bar chart for benchmark results that shows percentage values in the bars.
    
    Args:
        benchmark_data: List of tuples where each tuple is (benchmark_name, counter_dict).
              counter_dict has keys 'crash', 'illegal', 'ran' with percentage values.
    """
    # Extract the benchmark names and create lists for each status
    benchmark_names = [item[0] for item in benchmark_data]
    crash_values = [item[1].get('crash', 0) for item in benchmark_data]
    illegal_values = [item[1].get('illegal', 0) for item in benchmark_data]
    ran_values = [item[1].get('ran', 0) for item in benchmark_data]
    
    # Format text labels (include % sign and round to 1 decimal place)
    crash_text = [f"{val:.0f}%" if val >= 5 else "" for val in crash_values]
    illegal_text = [f"{val:.0f}%" if val >= 5 else "" for val in illegal_values]
    ran_text = [f"{val:.0f}%" if val >= 5 else "" for val in ran_values]
    
    # Use the same color palette as in your original code (pastel, colorblind-friendly)
    colors = ['#e78ac3', '#fc8d62', '#66c2a5']  # Using colors for crash, illegal, ran
    
    # Create a Plotly stacked bar chart
    fig = go.Figure()
    
    # Add bars with the same styling as your original code
    fig.add_trace(go.Bar(
        x=benchmark_names,
        y=ran_values,
        name='Runnable Schedules',
        marker_color=colors[2],
        marker=dict(cornerradius="0%"),
        text=ran_text,
        textposition='inside',
        textfont=dict(size=11, color='white'),
        insidetextanchor='start',
        hovertemplate='<b>%{y:.1f}%</b><br>Benchmark: %{x}<br>Status: Ran<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=benchmark_names,
        y=illegal_values,
        name='Illegal Schedules',
        marker_color=colors[1],
        marker=dict(cornerradius="0%"),
        text=illegal_text,
        textposition='inside',
        textfont=dict(size=11, color='white'),
        insidetextanchor='middle',
        hovertemplate='<b>%{y:.1f}%</b><br>Benchmark: %{x}<br>Status: Illegal<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=benchmark_names,
        y=crash_values,
        name='Invalid Schedules',
        marker_color=colors[0],
        marker=dict(cornerradius="0%"),
        text=crash_text,
        textposition='inside',
        textfont=dict(size=11, color='white'),
        insidetextanchor='end',
        hovertemplate='<b>%{y:.1f}%</b><br>Benchmark: %{x}<br>Status: Crash<extra></extra>'
    ))
    
    # Consistent layout styling with your original plots
    fig.update_layout(
        barmode='stack',
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title='Percentage (%)',
            range=[0, 100],  # Fixed range for percentages
            gridcolor='lightgrey',
            tickfont=dict(size=8),
            titlefont=dict(size=10),
            title_standoff=2,
        ),
        legend=dict(
            font=dict(size=8),
            orientation='h',
            yanchor='bottom',
            y=1,
            xanchor='right',
            x=1
        ),
        margin=dict(t=30, b=0, l=0, r=0),
        height=330,
        width=500,
        plot_bgcolor='white'
    )
    
    # Add gridlines matching your style
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgrey',
    )
    
    return fig


def plot_stacked_area_chart(iteration_data):
    """
    Create a stacked area chart showing how percentages change across iterations.
    
    Args:
        iteration_data: List of tuples where each tuple is (iteration:int, counter:dict).
              counter_dict has keys 'crash', 'illegal', 'ran' with percentage values.
    """
    # Sort by iteration number to ensure correct order
    iteration_data = sorted(iteration_data, key=lambda x: x[0])
    
    # Extract the iterations and values
    iterations = [item[0] for item in iteration_data]
    crash_values = [item[1].get('crash', 0) for item in iteration_data]
    illegal_values = [item[1].get('illegal', 0) for item in iteration_data]
    ran_values = [item[1].get('ran', 0) for item in iteration_data]
    
    # Use the same color palette as before (pastel, colorblind-friendly)
    colors = ['#e78ac3', '#fc8d62', '#66c2a5']  # Using colors for crash, illegal, ran
    
    # Create a Plotly stacked area chart
    fig = go.Figure()
    
    # Add traces in reverse order for proper stacking (bottom to top)
    fig.add_trace(go.Scatter(
        x=iterations,
        y=ran_values,
        mode='lines',
        name='Runnable Schedules',
        line=dict(width=0, color=colors[2]),
        stackgroup='one',  # this makes it a stacked area chart
        fillcolor=colors[2],
        hovertemplate='<b>%{y:.1f}%</b><br>Iteration: %{x}<br>Status: Ran<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=illegal_values,
        mode='lines',
        name='Illegal Schedules',
        line=dict(width=0, color=colors[1]),
        stackgroup='one',
        fillcolor=colors[1],
        hovertemplate='<b>%{y:.1f}%</b><br>Iteration: %{x}<br>Status: Illegal<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=crash_values,
        mode='lines',
        name='Invalid Schedules',
        line=dict(width=0, color=colors[0]),
        stackgroup='one',
        fillcolor=colors[0],
        hovertemplate='<b>%{y:.1f}%</b><br>Iteration: %{x}<br>Status: Crash<extra></extra>'
    ))
    
    # Consistent layout styling with your original plots
    fig.update_layout(
        xaxis=dict(
            title='Iterations T',
            titlefont=dict(size=12),
            tickfont=dict(size=11),
            gridcolor='lightgrey',
            title_standoff = 3
        ),
        yaxis=dict(
            title='Percentage (%)',
            range=[0, 100],  # Fixed range for percentages
            gridcolor='lightgrey',
            tickfont=dict(size=10),
            titlefont=dict(size=12),
            title_standoff = 3
        ),
        legend=dict(
            font=dict(size=10),
            orientation='h',
            yanchor='bottom',
            y=1,
            xanchor='right',
            x=1
        ),
        margin=dict(t=0, b=0, l=0, r=0),
        height=330,
        width=500,
        plot_bgcolor='white'
    )
    
    # Add horizontal percentage lines at rounded intervals
    for y in [33.33, 66.66]:
        fig.add_shape(
            type="line",
            x0=min(iterations),
            y0=y,
            x1=max(iterations),
            y1=y,
            line=dict(
                color="lightgrey",
                width=1,
                dash="dot",
            )
        )
    
    # Add markers at each data point with percentage labels
    for i, x in enumerate(iterations):
        # Only add labels at key points to avoid clutter
        # if i == 1 or i == len(iterations)-1 or i % 5 == 1:  # first, last, and every 5th point
        if i % 5 == 1:  # first, last, and every 5th point
            ran_cum = ran_values[i]
            illegal_cum = ran_cum + illegal_values[i]
            crash_cum = illegal_cum + crash_values[i]
            
            # Add labels for percentages >= 5%
            if ran_values[i] >= 5:
                fig.add_annotation(
                    x=x, y=ran_values[i]/2,
                    text=f"{ran_values[i]:.1f}%",
                    showarrow=False,
                    font=dict(size=9, color="white"),
                    textangle=90
                )
            
            if illegal_values[i] >= 5:
                fig.add_annotation(
                    x=x, y=ran_cum + illegal_values[i]/2,
                    text=f"{illegal_values[i]:.1f}%",
                    showarrow=False,
                    font=dict(size=9, color="white"),
                    textangle=90
                )
                
            if crash_values[i] >= 5:
                fig.add_annotation(
                    x=x, y=illegal_cum + crash_values[i]/2,
                    text=f"{crash_values[i]:.1f}%",
                    showarrow=False,
                    font=dict(size=9, color="white"),
                    textangle=90
                )
    
    return fig

