import os
import argparse
import time
import json
import yaml
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run-id', '-i', help="run_ID to be appended to experiment name", type=str)
parser.add_argument('--experiment-name', '-n', help="name of the experiment", type=str)
parser.add_argument('--completion-file', '-c', help="a name for the completion signaling file", type=str, default="0")
parser.add_argument('--config', help="path to the yaml configuration file", type=str, default="config.yaml")
parser.add_argument('--worker-id', type=int, default=None, help="worker ID for parallel execution (0-indexed)")
parser.add_argument('--num-workers', type=int, default=None, help="total number of parallel workers")
args = parser.parse_args()

# Load config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

os.environ['OPENAI_API_KEY'] = config.get("openai_api_key", "<SET_OPENAI_API_KEY>")
os.environ['LLM_ID'] = config.get("llm_id", "<LLM_ID>")
os.environ['BASE_URL'] = config.get("base_url", "<BASE_URL>")
os.environ['BENCHMARKS_DIR'] = config.get("benchmarks_dir", "<BENCH_FOLDER_PATH>")
os.environ['TIRALIB_DIR'] = config.get("tiralib_dir", "<TIRALIB_PATH>")
os.environ['LOOKUP_DIR_SUFFIX'] = config.get("lookup_dir_suffix", "base")

import compilot_utils
from compilot_utils import (
    TiramisuProgram, Schedule, check_in_lookup_file, update_lookup_file,
    annots_to_c_prog, initialize_conversation, print_compiler_message, 
    print_llm_message, get_LLM_response, extract_transformation_commands,
    get_compiler_response, logger
)
from prompts import system_instructions, start_exploration_message, continue_optimization_messages

base_results_dir = config.get("base_results_dir", "./results")

experiment_name = config.get("experiment_name", "name_of_experiment")
max_llm_responses = config.get("max_llm_responses", 50)
min_nb_suggestions = config.get("min_nb_suggestions", 10)
nb_exec_init = config.get("nb_exec_init", 15)
min_runs_scheds = config.get("min_runs_scheds", 5)
max_runs_scheds = config.get("max_runs_scheds", 10)
time_budget_factor_scheds = config.get("time_budget_factor_scheds", 10)
top_p = config.get("top_p", 1)
temperature = config.get("temperature", 1)

if args.experiment_name is not None:
	base_experiment_name = args.experiment_name
else:
    base_experiment_name = experiment_name

# update experiment name with relevant params
model_name = os.environ['LLM_ID'].split('/')[1] if '/' in os.environ['LLM_ID'] else os.environ['LLM_ID']
experiment_name = f'{base_experiment_name}_{model_name}_T{temperature}_TP{top_p}_L{min_nb_suggestions}'

# update the results dir before adding the run's id
results_dir = base_results_dir + '/' + experiment_name
if args.run_id is not None: 
	experiment_name += '_run' + args.run_id

benchmarks_folder = os.environ["BENCHMARKS_DIR"]
benchmark_paths = sorted(list(Path(benchmarks_folder).iterdir()))

# Partition programs across workers (round-robin) when running in parallel mode
if args.worker_id is not None and args.num_workers is not None:
    assert 0 <= args.worker_id < args.num_workers, f"worker-id must be in [0, num-workers)"
    benchmark_paths = [p for i, p in enumerate(benchmark_paths) if i % args.num_workers == args.worker_id]
    print(f"Worker {args.worker_id}/{args.num_workers}: processing {len(benchmark_paths)} programs")

# Create the directory if it doesn't exist
Path(results_dir).mkdir(parents=True, exist_ok=True)

slurmd_nodename = os.environ.get('SLURMD_NODENAME', '')

# JSON results file — use worker-specific filename when running in parallel
if args.worker_id is not None:
    results_json_path = f'{results_dir}/results_{experiment_name}_worker{args.worker_id}.json'
else:
    results_json_path = f'{results_dir}/results_{experiment_name}.json'
if Path(results_json_path).is_file():
    with open(results_json_path, 'r') as f:
        results_data = json.load(f)
else:
    # Read config file contents for the results
    with open(args.config, 'r') as f:
        config_contents = yaml.safe_load(f)
    results_data = {
        'configuration': config_contents,
        'programs': {}
    }

for benchmark_path in benchmark_paths:
    if str(benchmark_path.parts[-1]) in results_data['programs']:
        continue
    start = time.time()
    benchmark_name = str(benchmark_path.parts[-1])
    print(benchmark_name)
    tiramisu_program = TiramisuProgram.from_file(
        str(benchmark_path)+'/'+benchmark_name+'_generator.cpp', load_annotations=True, load_tree=True
    )
    schedule = Schedule(tiramisu_program)
    if (lookup_res := check_in_lookup_file('', tiramisu_program.name, 'initial_exec')) != 'Not found':
        initial_exec_time = lookup_res
        logger.debug(f'Initial execution time for {tiramisu_program.name} retrieved from lookup is {initial_exec_time}.')
    else:
        initial_exec_time=min(schedule.execute(nb_exec_init))
        update_lookup_file('', tiramisu_program.name, 'initial_exec', initial_exec_time)
    print(initial_exec_time)
    annots = tiramisu_program.annotations

    loop_nest_str, new_loop_names_map, compute_blocks_IDs = annots_to_c_prog(annots, True)
    first_message = f'''
Here is the loop structure 

{loop_nest_str}


The initial execution time is {initial_exec_time}ms.

Before you start suggesting transformations. Tell me what are the different loops of the program and the nesting depth of each (L0, L1, ...) and give me a brief analysis of the loop nest. Do not suggest transformations yet.
    '''
    
    dialogue_messages, token_usage, full_message = initialize_conversation(system_instructions, first_message)
    print_compiler_message(0, dialogue_messages[0]['content']) # system instruction
    print_compiler_message(1, dialogue_messages[1]['content']) # first message about doing a program analysis
    print_llm_message(1, full_message) # program analysis
    print_compiler_message(2, start_exploration_message) # ask to start exploration

    nb_llm_responses = 1  # count LLM responses for the iteration limit
    full_message = get_LLM_response(start_exploration_message, dialogue_messages, token_usage, top_p=top_p, temperature=temperature)
    nb_llm_responses += 1
    print_llm_message(2, full_message) # first suggestion

    explored_scheds = [] # list of dicts with schedule, outcome, execution_time
    llm_sched_strs_dict = dict() # a dict of llm_sched_str:compiler response. Used to avoid rerunning same llm_sched_str
    stop_message = False
    giveups_counter = 0
    while (not stop_message) and nb_llm_responses < max_llm_responses:
        llm_sched_str = extract_transformation_commands(full_message)
        if llm_sched_str is None:
            compiler_response = 'The compiler could not parse your instruction. Please make sure to respect the described response format.'
            print_compiler_message(nb_llm_responses, compiler_response)
            full_message = get_LLM_response(compiler_response, dialogue_messages, token_usage, top_p=top_p, temperature=temperature)
            nb_llm_responses += 1
            print_llm_message(nb_llm_responses, full_message)
        elif 'no_further_transformations' in llm_sched_str:
            if len(explored_scheds) < min_nb_suggestions:
                compiler_message = continue_optimization_messages[giveups_counter%len(continue_optimization_messages)]
                giveups_counter += 1
                print_compiler_message(nb_llm_responses, compiler_message)
                full_message = get_LLM_response(compiler_message, dialogue_messages, token_usage, top_p=top_p, temperature=temperature)
                nb_llm_responses += 1
                print_llm_message(nb_llm_responses, full_message)
            else:
                stop_message = True
        else: # a schedule is provided
            if llm_sched_str not in llm_sched_strs_dict:
                compiler_response = get_compiler_response(
                    llm_sched_str, tiramisu_program, annots, compute_blocks_IDs,
                    initial_exec_time, explored_scheds,
                    min_runs=min_runs_scheds, max_runs=max_runs_scheds,
                    time_budget=time_budget_factor_scheds*initial_exec_time
                )
                llm_sched_strs_dict[llm_sched_str] = compiler_response
            else:
                compiler_response = "You already suggested " + llm_sched_str + ". The compiler's response was:\n" + llm_sched_strs_dict[llm_sched_str]

            print_compiler_message(nb_llm_responses, compiler_response)
            full_message = get_LLM_response(compiler_response, dialogue_messages, token_usage, top_p=top_p, temperature=temperature)
            nb_llm_responses += 1
            print_llm_message(nb_llm_responses, full_message)
            
    end = time.time()
    optim_time = (end-start)*1000 # in ms

    # Build explored_schedules list for JSON output
    # Include the initial schedule (identity / no transformations) as the first entry
    explored_schedules_json = [
        {
            'schedule': '',
            'outcome': 'ran',
            'execution_time': initial_exec_time
        }
    ]
    for sched_tuple in explored_scheds:
        entry = {
            'schedule': sched_tuple[0],
            'outcome': sched_tuple[1],  # 'ran', 'illegal', or 'crash'
            'execution_time': sched_tuple[3] if sched_tuple[1] == 'ran' else None
        }
        explored_schedules_json.append(entry)

    # Determine the best schedule
    ran_scheds = [s for s in explored_scheds if s[1] == 'ran']
    if ran_scheds:
        best = min(ran_scheds, key=lambda x: x[3])
        # If initial is still the best, report that
        if initial_exec_time <= best[3]:
            best_schedule = {'schedule': '', 'execution_time': initial_exec_time, 'speedup': 1.0}
        else:
            best_schedule = {'schedule': best[0], 'execution_time': best[3], 'speedup': round(initial_exec_time / best[3], 4)}
    else:
        best_schedule = {'schedule': '', 'execution_time': initial_exec_time, 'speedup': 1.0}

    results_data['programs'][benchmark_name] = {
        'initial_execution_time': initial_exec_time,
        'optimization_dialogue': dialogue_messages,
        'explored_schedules': explored_schedules_json,
        'optimization_time': optim_time,
        'token_usage': token_usage,
        'best_schedule': best_schedule,
        'program_annotation': annots
    }

    # Save after each benchmark for crash resilience
    with open(results_json_path, 'w') as f:
        json.dump(results_data, f, indent=2)

if args.completion_file != "0": # if a filename is provided for signaling completion, create it 
    with open(args.completion_file, "w") as f:
        pass
