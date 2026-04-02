import os
import sys
import argparse
import time
import json
import pickle
import yaml
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run-id', '-i', help="run_ID to be appended to experiment name", type= str)
parser.add_argument('--experiment-name', '-n', help="name of the experiment", type= str)
parser.add_argument('--completion-file', '-c', help="a name for the completion signaling file", type= str, default= "0")
parser.add_argument('--config', help="path to the yaml configuration file", type= str, default= "config.yaml")
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
nb_responses = config.get("nb_responses", 1)
enable_thinking = config.get("enable_thinking", True)

if args.experiment_name is not None:
	base_experiment_name = args.experiment_name
else:
    base_experiment_name = experiment_name

# update experiment name with relevant params
model_name = os.environ['LLM_ID'].split('/')[1] if '/' in os.environ['LLM_ID'] else os.environ['LLM_ID']
experiment_name = f'{base_experiment_name}_{model_name}_T{temperature}_TP{top_p}_L{min_nb_suggestions}_{'wT' if enable_thinking else 'woT'}'

# update the results dir before adding the run's id
results_dir = base_results_dir + '/' + experiment_name
if args.run_id is not None: 
	experiment_name += '_run' + args.run_id

benchmarks_folder = os.environ["BENCHMARKS_DIR"]
benchmark_paths = sorted(list(Path(benchmarks_folder).iterdir()))

# Create the directory if it doesn't exist
Path(results_dir).mkdir(parents=True, exist_ok=True)

# save the parameters into a json file
params_file = f"{results_dir}/params_run{args.run_id}.json" if args.run_id is not None else f"{results_dir}/params.json"
with open(params_file, 'w') as f:
    params_dict = {
        'experiment_name': experiment_name,
        'max_llm_responses': max_llm_responses,
        'min_nb_suggestions': min_nb_suggestions,
        'top_p': top_p,
        'temperature': temperature,
        'nb_responses': nb_responses,
        'benchmarks': os.environ['BENCHMARKS_DIR'],
        'LLM_ID': os.environ['LLM_ID'],
        'script_file': sys.argv[0],
        'creation_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.mktime(time.gmtime()) + 4 * 3600)), # in gmt+4
        'node_name': os.environ.get('SLURMD_NODENAME', 'localhost')
    }
    f.write(json.dumps(params_dict))

slurmd_nodename = os.environ.get('SLURMD_NODENAME', '')

per_bench_results = dict()
per_bench_results_pkl = f'{results_dir}/per_bench_results_{experiment_name}.pkl'
if Path(per_bench_results_pkl).is_file():
    with open(per_bench_results_pkl, 'rb') as f:
        per_bench_results = pickle.load(f)

for benchmark_path in benchmark_paths:
    if str(benchmark_path.parts[-1]) in per_bench_results:
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
    
    llm_reponses_trace, full_message = initialize_conversation(system_instructions, first_message, enable_thinking)
    curr_messages = llm_reponses_trace[-1]['all_prev_messages']
    print_compiler_message(0, curr_messages[0]['content']) # system instructin
    print_compiler_message(1, curr_messages[1]['content']) # first message about doing a program analysis
    print_llm_message(1, full_message) # program analysis
    print_compiler_message(2, start_exploration_message) # ask to start explortion
    
    last_line, full_message = get_LLM_response(start_exploration_message, llm_reponses_trace, top_p=top_p, temperature=temperature, enable_thinking=enable_thinking)
    print_llm_message(2, full_message) # first suggestion
    
    explored_scheds = [] # list of tuples (llm_sched_str, outcome_type={'ran', 'illegal', 'crashed'}, is_lookedup:Bool, outcome_data:{exec_time if ran, crash exception str if crashed, None if illegal})
    llm_sched_strs_dict = dict() # a dict of llm_sched_str:compiler repsponse. Used to avoid rerunning same llm_sched_str
    stop_message = False
    giveups_counter = 0
    while (not stop_message) and len(llm_reponses_trace) < max_llm_responses:
        llm_sched_str = extract_transformation_commands(full_message)
        if llm_sched_str is None:
            compiler_response = 'The compiler could not parse your instruction. Please make sure to respect the described response format.'
            print_compiler_message(len(llm_reponses_trace), compiler_response)
            last_line, full_message = get_LLM_response(compiler_response, llm_reponses_trace, top_p=top_p, temperature=temperature, enable_thinking=enable_thinking)
            print_llm_message(len(llm_reponses_trace), full_message)
        elif 'no_further_transformations' in llm_sched_str:
            if len(explored_scheds) < min_nb_suggestions:
                compiler_message = continue_optimization_messages[giveups_counter%len(continue_optimization_messages)]
                giveups_counter+=1
                print_compiler_message(len(llm_reponses_trace), compiler_message)
                last_line, full_message = get_LLM_response(compiler_message, llm_reponses_trace, top_p=top_p, temperature=temperature, enable_thinking=enable_thinking)
                print_llm_message(len(llm_reponses_trace), full_message)
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

            print_compiler_message(len(llm_reponses_trace), compiler_response)
            last_line, full_message = get_LLM_response(compiler_response, llm_reponses_trace, top_p=top_p, temperature=temperature, enable_thinking=enable_thinking)
            print_llm_message(len(llm_reponses_trace), full_message)
            
    end = time.time()
    optim_time = (end-start)*1000 # in ms
    per_bench_results[benchmark_name]= {'llm_reponses_trace':llm_reponses_trace,
                                      'explored_scheds':explored_scheds,
                                       'optim_time': optim_time ,# in ms
                                       }
    with open(f'{results_dir}/live_results_{experiment_name}.csv','a+') as f:
        # Move the pointer to the beginning of the file
        f.seek(0)
        if f.read(1) == '':
            #if emtpy write header
            f.write('benchmark, init_exec_time, nb_explored, nb_executed, optim_time, best_exec_time, best_speedup, best_sched,' + '\n')
        ran_scheds = [s for s in explored_scheds if s[1]=='ran']
        if ran_scheds:
            best_sched = sorted(ran_scheds, key=lambda x:x[3])[0]
            best_sched_str = best_sched[0]
            best_sched_exec = best_sched[3]
            best_sched_speedup = initial_exec_time/best_sched[3]
        else:
            best_sched_str = ''
            best_sched_exec = initial_exec_time
            best_sched_speedup = 1
        f.write(f'{benchmark_name}, {initial_exec_time}, {len(explored_scheds)}, {len(ran_scheds)}, {optim_time}, {best_sched_exec}, {best_sched_speedup}, "{best_sched_str}"\n')

    with open(per_bench_results_pkl, 'wb') as f:
        pickle.dump(per_bench_results, f)

if args.completion_file != "0": # if a filename is provided for signaling completion, create it 
    with open(args.completion_file, "w") as f:
        pass
