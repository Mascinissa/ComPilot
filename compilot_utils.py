from sympy import simplify, nsimplify
import re
from string import ascii_lowercase
import logging
import asyncio
import os
from pathlib import Path
import time
import pickle
import sys
from filelock import FileLock
import traceback
import random
import argparse
import json


from all_utils import *

sys.path.insert(1, os.getenv("TIRALIB_DIR"))
from tiralib.tiramisu import TiramisuProgram, Schedule, tiramisu_actions
from tiralib.config import BaseConfig

class ScheduleExecutionCrashed(Exception):
    pass

class KnownScheduleCreationCrash(Exception):
    """Exception class for know schedule creation errors."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
    
from openai import OpenAI, AsyncOpenAI
from IPython.display import display_markdown

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url = os.getenv("BASE_URL"), max_retries=10)
client_async = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url = os.getenv("BASE_URL"), max_retries=10)


BaseConfig.init(config_yaml= os.getenv("TIRALIB_DIR")+"/config.yaml")

lookup_files_dir = './lookup_dir_'+os.getenv("LOOKUP_DIR_SUFFIX")
Path(lookup_files_dir).mkdir(exist_ok=True)

llm_id = os.getenv("LLM_ID")

logger = logging.getLogger('__name__')
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("|%(asctime)s|%(levelname)s| %(message)s")
)
logger.addHandler(console_handler)

def expression_json_to_str(expr, iterator_names=None):
    operators = {
        'add': ('+', 2),  # Binary
        'sub': ('-', 2),  # Binary
        'mul': ('*', 2),  # Binary
        'div': ('/', 2),  # Binary
        'eq': ('==', 2),  # Binary
        'le': ('<=', 2),  # Binary
        'max': ('max', 2),  # Binary (with special formatting)
        'min': ('min', 2),  # Binary (with special formatting)
        'cast': ('cast', 1),  # Unary (with special formatting)
        'sqrt_f32': ('sqrt', 1),  # Unary (with special formatting)
        'sqrt_f64': ('sqrt', 1),  # Unary (with special formatting)
        'access': (None,0),
        'val': (None,0),
    }
    assert expr['expr_type'] in operators

     # Check if the token is an operator
    operator, arity = operators[expr['expr_type']]

    if arity == 1:  # Unary operator
        operand = expression_json_to_str(expr['children'][0],iterator_names)
        if expr['expr_type'].startswith('sqrt'):
            return f"sqrt({operand})"
        elif expr['expr_type'] == 'cast':
            return f"({expr['dtype']})({operand})"
            # return f"{operand}" # just not doing anything for know because sympy cant simplify

    elif arity == 2:  # Binary operator
        operand1 = expression_json_to_str(expr['children'][0],iterator_names)
        operand2 = expression_json_to_str(expr['children'][1],iterator_names)

        if expr['expr_type'] in ['max', 'min']:
            # Special case for max and min
            return f"{operator}({operand1}, {operand2})"
        else:
            # Wrap in parentheses for proper order of operations
            return f"({operand1} {operator} {operand2})"
    elif arity==0:
        if expr['expr_type'] == 'access':
            return get_access_str_from_access_matrix(expr['buffer_id'], expr['access_matrix'], iterator_names=iterator_names)
        elif expr['expr_type'] == 'val':
            # return expr['str']
            return str(int(expr['str'])) if 'int' in expr['dtype'] else str(float(expr['str']))
            
def expression_str_to_c_expression(expression_str):
    # print(expression_str)
    try:
        simplified_str = str(simplify(expression_str))
    except TypeError: # sympy can crash when there are casts or le 
        simplified_str = expression_str
    return re.sub(r'(\w+)\(([\s\w\d_,\-+*]+)\)', r'\g<1>[\g<2>]', simplified_str).replace(', ','][') 


def expression_json_to_c_expression(expr, iterator_names=None):
    return expression_str_to_c_expression(expression_json_to_str(expr, iterator_names))

def for_loop_str(iter_name, program_annotations):
    iter_dict = program_annotations['iterators'][iter_name]
    return f'for(int {iter_name}={str(simplify(iter_dict["lower_bound"]))}; {iter_name}<{str(simplify(iter_dict["upper_bound"]))}; {iter_name}++)'

def annots_to_c_prog(program_annotations, anonymize_loop_names = False):
    preorder_traversal = get_program_preorder_traversal(program_annotations)
    lines = []
    prev_depth = -1
    compute_blocks_IDs = dict() # a dict of comp_block_id:[list_comp_names_in_block]
    curr_comp_block_id = None
    
    for elem in preorder_traversal[1:]:
        while prev_depth>elem['depth']:
            prev_depth-=1
            lines.append('    '*prev_depth+'}')
            
        if elem['type'] == 'loop':
            curr_comp_block_id = None
                
            elem_str = '    '*elem['depth'] + for_loop_str(elem['name'],program_annotations) + '{'
            
            if program_annotations['iterators'][elem['name']]['computations_list']: # if loop has comps, make a comp block ID
                curr_comp_block_id = 'comp'+str(len(compute_blocks_IDs)).zfill(2)
                compute_blocks_IDs[curr_comp_block_id] = program_annotations['iterators'][elem['name']]['computations_list']
                elem_str += '\n'+'    '*(elem['depth']+1) + "// comp_ID: " + curr_comp_block_id

           
            
        else: # if comp
            elem_str = '    '*elem['depth'] + annots_to_c_comp(elem['name'], program_annotations) + ";"
        prev_depth = elem['depth']  
        lines.append(elem_str)
        
    while prev_depth>0:
        prev_depth-=1
        lines.append('    '*prev_depth+'}')
        
        
    loop_nest_str = '\n'.join(lines)
    
    if anonymize_loop_names:
        new_loop_names_map = {ascii_lowercase[d]:orig_name for d,orig_name in enumerate([elem['name'] for elem in preorder_traversal[1:] if elem['type'] == 'loop'])}
        for new_name, orig_name in new_loop_names_map.items():
            loop_nest_str = re.sub(fr'\b{orig_name}\b', new_name, loop_nest_str)
        return loop_nest_str, new_loop_names_map, compute_blocks_IDs
    else:
        return loop_nest_str, {}, compute_blocks_IDs

def annots_to_c_comp(comp_name, program_annotations):
    comp_dict = program_annotations['computations'][comp_name]
    write_acc_mat = isl_map_to_write_access_matrix(comp_dict['write_access_relation'])
    lhs_str = get_access_str_from_access_matrix(comp_dict['write_buffer_id'], write_acc_mat, comp_dict['iterators'])
    lhs_str = expression_str_to_c_expression(lhs_str)
    rhs_str = expression_json_to_c_expression(comp_dict['simplified_expression_representation'], comp_dict['iterators'])
    return lhs_str + ' = ' + rhs_str

def llm_sched_str_to_sched(llm_sched_str, program, program_annots, compute_blocks_IDs=None):
    # if new_loop_names_map is None:
    #     # new_loop_names_map = {loop_name:loop_name for loop_name in program.tree.iterators.keys()}
    #     new_loop_names_map = {loop_name:loop_name for loop_name in program_annots['iterators'].keys()}
    schedule = Schedule(program)
    llm_sched_str = llm_sched_str.replace(' ', '').replace('`','')
    llm_transfs = llm_sched_str.split('+')
    for transf_idx,transf in enumerate(llm_transfs):
        assert transf.count('.') == 1
        comp_id, transf_inst = tuple(transf.split('.'))
        real_comp_name = compute_blocks_IDs[comp_id][0]
        if transf_inst.lower().startswith('fuse'):
            transf_id = 'fuse'
            assert transf_inst[len(transf_id)]=='(' and transf_inst[-1]==')'
            transf_params = transf_inst[len(transf_id)+1:-1]
            split_transf_params = transf_params.split(',')
            assert len(split_transf_params)==2
            second_comp_id = split_transf_params[0]
            second_comp_real_name = compute_blocks_IDs[second_comp_id][0]
            loop = split_transf_params[1]
            assert loop[0] == 'L'
            loop_level = int(loop[1:])
            if loop_level == -1:
                loop_level = get_innermost_level(real_comp_name, schedule)
            loop_id1 = (real_comp_name,loop_level)
            loop_id2 = (second_comp_real_name,loop_level)
            new_optim = tiramisu_actions.Fusion([loop_id1,loop_id2])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])
            
        elif transf_inst.lower().startswith('parallelize'):
            transf_id = 'parallelize'
            assert transf_inst[len(transf_id)]=='(' and transf_inst[-1]==')'
            transf_params = transf_inst[len(transf_id)+1:-1]
            split_transf_params = transf_params.split(',')
            assert len(split_transf_params)==1
            loop = split_transf_params[0]
            assert loop[0] == 'L'
            loop_level = int(loop[1:])
            if loop_level == -1:
                loop_level= get_innermost_level(real_comp_name, schedule)
            if loop_level>=2:
                raise KnownScheduleCreationCrash(f'{transf_inst}: Deep loops (depth>=2) should not be parallelized.')
            loop_id = (real_comp_name, loop_level)
            new_optim = tiramisu_actions.Parallelization([loop_id])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])
        
        elif transf_inst.lower().startswith('reverse'):
            transf_id = 'reverse'
            assert transf_inst[len(transf_id)]=='(' and transf_inst[-1]==')'
            transf_params = transf_inst[len(transf_id)+1:-1]
            split_transf_params = transf_params.split(',')
            assert len(split_transf_params)==1
            loop = split_transf_params[0]
            assert loop[0] == 'L'
            loop_level = int(loop[1:])
            if loop_level == -1:
                loop_level= get_innermost_level(real_comp_name, schedule)
            loop_id = (real_comp_name, loop_level)
            new_optim = tiramisu_actions.Reversal([loop_id])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

            
        elif transf_inst.lower().startswith('interchange'):
            transf_id = 'interchange'
            assert transf_inst[len(transf_id)]=='(' and transf_inst[-1]==')'
            transf_params = transf_inst[len(transf_id)+1:-1]
            split_transf_params = transf_params.split(',')
            
            assert len(split_transf_params)==2
            loop1 = split_transf_params[0]
            loop2 = split_transf_params[1]
            assert loop1[0] == loop2[0] == 'L'
            loop_level1 = int(loop1[1:])
            if loop_level1==-1:
                loop_level1 = get_innermost_level(real_comp_name, schedule)
            loop_level2 = int(loop2[1:])
            if loop_level2==-1:
                loop_level2 = get_innermost_level(real_comp_name, schedule)
            loop_id1 = (compute_blocks_IDs[comp_id][0],loop_level1)
            loop_id2 = (compute_blocks_IDs[comp_id][0],loop_level2)
            new_optim = tiramisu_actions.Interchange([loop_id1,loop_id2])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])
            
        elif transf_inst.lower().startswith('skew'):
            transf_id = 'skew'
            assert transf_inst[len(transf_id)]=='(' and transf_inst[-1]==')'
            transf_params = transf_inst[len(transf_id)+1:-1]
            split_transf_params = transf_params.split(',')
            
            assert len(split_transf_params)==2
            loop1 = split_transf_params[0]
            loop2 = split_transf_params[1]
            assert loop1[0] == loop2[0] == 'L'
            loop_level1 = int(loop1[1:])
            loop_level2 = int(loop2[1:])
            loop_id1 = (compute_blocks_IDs[comp_id][0],loop_level1)
            loop_id2 = (compute_blocks_IDs[comp_id][0],loop_level2)

            # For skewing we need to do a partial check before getting the factors
            temp_skew_action = tiramisu_actions.Skewing([loop_id1,loop_id2,'X','X'])
            # temp hack: initialize manually
            temp_skew_action.initialize_action_for_tree(schedule.tree)
            assert is_new_optim_valid(schedule, temp_skew_action, llm_transfs[:transf_idx+1])

            
            lookup_res=check_in_lookup_file('+'.join(llm_transfs[:transf_idx+1]), program.name, 'skewing_solver')
            if lookup_res!='Not found':
                skewing_factors = lookup_res
                logger.debug(f'Skewing solu for {'+'.join(llm_transfs[:transf_idx+1])} retrieved from lookup are {skewing_factors}.')
            else:
                loop_level1_node = schedule.tree.get_iterator_of_computation(real_comp_name, 1)
                involved_computations = schedule.tree.get_iterator_subtree_computations(loop_level1_node.id) 
                skewing_factors = tiramisu_actions.Skewing.get_factors(schedule, [loop_level1, loop_level2], involved_computations)    
                update_lookup_file('+'.join(llm_transfs[:transf_idx+1]), program.name, 'skewing_solver', skewing_factors)
            
            if skewing_factors is None:
                raise KnownScheduleCreationCrash(f'Could not find any skewing solutions for the  transformation "{transf}".')
            factor1, factor2 = skewing_factors
            schedule.add_optimizations([tiramisu_actions.Skewing([loop_id1,loop_id2,factor1,factor2])])
        
        elif transf_inst.lower().startswith('tile2d'):
            transf_id = 'tile2d'
            assert transf_inst[len(transf_id)]=='(' and transf_inst[-1]==')'
            transf_params = transf_inst[len(transf_id)+1:-1]
            split_transf_params = transf_params.split(',')
            
            assert len(split_transf_params)==4
            loop1 = split_transf_params[0]
            loop2 = split_transf_params[1]
            factor1 = split_transf_params[2]
            factor2 = split_transf_params[3]
            assert loop1[0] == loop2[0] == 'L'
            loop_level1 = int(loop1[1:])
            loop_level2 = int(loop2[1:])
            if loop_level2==-1:
                loop_level2 = get_innermost_level(real_comp_name, schedule)
            loop_id1 = (compute_blocks_IDs[comp_id][0],loop_level1)
            loop_id2 = (compute_blocks_IDs[comp_id][0],loop_level2)
            new_optim = tiramisu_actions.TilingGeneral([loop_id1,loop_id2,int(factor1),int(factor2)])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

        elif transf_inst.lower().startswith('tile3d'):
            transf_id = 'tile3d'
            assert transf_inst[len(transf_id)]=='(' and transf_inst[-1]==')'
            transf_params = transf_inst[len(transf_id)+1:-1]
            split_transf_params = transf_params.split(',')
            
            assert len(split_transf_params)==6
            loop1 = split_transf_params[0]
            loop2 = split_transf_params[1]
            loop3 = split_transf_params[2]
            factor1 = split_transf_params[3]
            factor2 = split_transf_params[4]
            factor3 = split_transf_params[5]
            assert loop1[0] == loop2[0] == loop3[0] == 'L'
            loop_level1 = int(loop1[1:])
            loop_level2 = int(loop2[1:])
            loop_level3 = int(loop3[1:])
            if loop_level3==-1:
                loop_level3 = get_innermost_level(real_comp_name, schedule)
            loop_id1 = (compute_blocks_IDs[comp_id][0],loop_level1)
            loop_id2 = (compute_blocks_IDs[comp_id][0],loop_level2)
            loop_id3 = (compute_blocks_IDs[comp_id][0],loop_level3)
            new_optim = tiramisu_actions.TilingGeneral([loop_id1,loop_id2,loop_id3,int(factor1),int(factor2),int(factor3)])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

        elif transf_inst.lower().startswith('unroll'):
            transf_id = 'unroll'
            assert transf_inst[len(transf_id)]=='(' and transf_inst[-1]==')'
            transf_params = transf_inst[len(transf_id)+1:-1]
            split_transf_params = transf_params.split(',')
            
            assert len(split_transf_params)==2
            loop = split_transf_params[0]
            assert loop[0] == 'L'
            loop_level = int(loop[1:])
            if loop_level==-1:
                loop_level = get_innermost_level(real_comp_name, schedule)
            factor = int(split_transf_params[1])
            # if factor==1:
            #     raise Exception('tried to unroll with factor of 1')
            loop_id = (compute_blocks_IDs[comp_id][0],loop_level)
            new_optim = tiramisu_actions.Unrolling([loop_id,factor])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])
            # schedule.tree.get_iterator_id_from_name(new_loop_names_map[loop_name])
        else:
            raise Exception('Unsupported transformation ' + transf)
            
            
    return schedule

def get_innermost_level(real_comp_name, schedule):
    return schedule.tree.get_iterator_of_computation(real_comp_name).level

def check_in_lookup_file(llm_sched_str, prog_name, query_type):
    assert query_type in ['initial_exec', 'compiler_responses', 'skewing_solver']
    prog_lookup_file = f'{lookup_files_dir}/{prog_name}_lookup.pkl'
    prog_lookup_lock = FileLock(f'{lookup_files_dir}/{prog_name}_lookup.lock')
    with prog_lookup_lock:
        if Path(prog_lookup_file).is_file():
            with open(prog_lookup_file, 'rb') as f:
                lookup_dict = pickle.load(f)
        else: 
            lookup_dict = {'initial_exec':'Not found' , 'compiler_responses':{}, 'skewing_solver':{}}
    normalized_llm_sched_str = llm_sched_str.replace(' ', '').replace('`','')
    if query_type == 'initial_exec':
        return lookup_dict[query_type]
    elif normalized_llm_sched_str in lookup_dict[query_type]:
        return lookup_dict[query_type][normalized_llm_sched_str]
    else:
        return 'Not found'

def update_lookup_file(llm_sched_str, prog_name, query_type, value):
    # value should be of the format 
    #   (compliler_response_str, outcome_type={'ran', 'illegal', 'crashed'},  outcome_data:{exec_time if ran, crash exception str if crashed, None if illegal}) if query_type==compiler_responses
    #   exec_time as float if query_type==compiler_responses
    #   skewing_sols_tuple if query_type==skewing_solver and solutions exist, else 'Not found'
    assert query_type in ['initial_exec', 'compiler_responses', 'skewing_solver']
    prog_lookup_file = f'{lookup_files_dir}/{prog_name}_lookup.pkl'
    prog_lookup_lock = FileLock(f'{lookup_files_dir}/{prog_name}_lookup.lock')
    with prog_lookup_lock:
        # get the latest version of the file first
        if Path(prog_lookup_file).is_file():
            with open(prog_lookup_file, 'rb') as f:
                lookup_dict = pickle.load(f)
        else: 
            lookup_dict = {'initial_exec':'Not found' , 'compiler_responses':{}, 'skewing_solver':{}}
                
        normalized_llm_sched_str = llm_sched_str.replace(' ', '').replace('`','')
        if query_type == 'initial_exec':
            lookup_dict[query_type] = value
        elif normalized_llm_sched_str in lookup_dict[query_type]:
            # raise Exception(f'Trying to write an existing entry in lookup table {normalized_llm_sched_str},  prog_name:{prog_name}, query_type:{query_type} ')
            # return 
            # if it already exists means that two parallel runs collided on the same new schedule. One of them has to pass
            pass
        else:
            lookup_dict[query_type][normalized_llm_sched_str] = value
        with open(prog_lookup_file, 'wb') as f:
            pickle.dump(lookup_dict, f)

def get_compiler_response(llm_sched_str, tiramisu_program, annots, compute_blocks_IDs, initial_exec_time, explored_scheds, min_runs=10, max_runs=None, time_budget=None):
    prog_name = tiramisu_program.name
    if (lookup_res := check_in_lookup_file(llm_sched_str, prog_name, 'compiler_responses')) != 'Not found':
        logger.debug(f'Response for {llm_sched_str} retrieved from lookup.')
        message, outcome_type, outcome_data = lookup_res
        explored_scheds.append((llm_sched_str, outcome_type, True, outcome_data)) 
        return message
    try:
        schedule = llm_sched_str_to_sched(llm_sched_str, tiramisu_program, annots, compute_blocks_IDs)
    except KnownScheduleCreationCrash as e:
        explored_scheds.append((llm_sched_str, 'crash', False, traceback.format_exc())) 
        message = f"{llm_sched_str} crashed at creation with the following message:\n{e.message}"
        logger.exception(message)
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc()))
        return message
    except Exception as e:
        explored_scheds.append((llm_sched_str, 'crash', False, traceback.format_exc())) 
        message = f"{llm_sched_str} crashed at creation for unclear reasons."
        logger.exception(message)
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc()))
        return message
    try:
        is_legal = schedule.is_legal()
    except Exception as e:
        explored_scheds.append((llm_sched_str, 'crash', False, traceback.format_exc())) 
        message = f'{llm_sched_str} crashed at legality checking'
        logger.exception(message)
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc()))
        return message
    if is_legal:
        try:
            exec_time = min(schedule.execute(min_runs=min_runs,max_runs=max_runs,time_budget=time_budget))
            speedup = initial_exec_time/exec_time
            explored_scheds.append((llm_sched_str, 'ran', False, exec_time)) 
            
            if speedup>=1:
                message = f'{llm_sched_str} was legal and yielded a speedup of {round(speedup,2)}x'
            else:
                message = f'{llm_sched_str} was legal and yielded a speedup of {round(speedup,2)}x (i.e. a slowdown of {round(1/speedup,2)}x)'

            update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'ran', exec_time)) 
            return message
                
        except ScheduleExecutionCrashed as e:
            explored_scheds.append((llm_sched_str, 'crash', False, traceback.format_exc())) 
            message = f'{llm_sched_str} resulted in a segmentation fault'
            logger.exception(message)
            update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc())) 
            return message
            
        except Exception as e:
            explored_scheds.append((llm_sched_str, 'crash', False, traceback.format_exc())) 
            message = f'{llm_sched_str} resulted in a segmentation fault'
            logger.exception(message)
            update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc())) 
            
            return message
            
    else:
        explored_scheds.append((llm_sched_str, 'illegal', False, None))
        message = f'{llm_sched_str} was not legal'
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'illegal', None)) 
        return message



def get_LLM_response(compiler_response, llm_reponses_trace, top_p = 1, temperature=1):
    new_user_message = compiler_response
    messages = llm_reponses_trace[-1]['all_prev_messages'][:]
    messages.append({"role": "user", "content": new_user_message})
    got_a_response = False
    while not got_a_response:
        response = client.chat.completions.create(
            model=llm_id,
            top_p = top_p,
            temperature = temperature,
            messages=messages
        )
        if response.choices is not None:
            got_a_response = True
            new_assitant_message = response.choices[0].message.content
        else:
            print('No response recieved, retrying in 15s', response)
            time.sleep(15)
    
    
    messages.append({"role": "assistant", "content": new_assitant_message})
    
    # Doing the following to keep the same format as the one of the voting mechanism
    llm_reponses_trace_elem = {'all_prev_messages':messages, #includes all the conversation so far including the newly selected llm message
                              'grouped_answers_per_sched':'no voting', # includes all the answers gotten from the llm grouped my the suggested schedule
                               'new_selected_sched': 'no voting', # the sched that has newly been selected.
                               'tokens_usage': { # the number of tokens used so far
                                   'completion_tokens': response.usage.completion_tokens,
                                   'prompt_tokens': response.usage.prompt_tokens,
                                   'total_tokens': response.usage.total_tokens
                               }
                              }
    llm_reponses_trace.append(llm_reponses_trace_elem)
    last_line = new_assitant_message.splitlines()[-1]
    return last_line, new_assitant_message

async def async_chat_completion(messages, top_p = 1, temperature = 1):
    response = await client_async.chat.completions.create(
        model=llm_id,
        top_p = top_p,
        temperature = temperature,
        messages=messages
    )
    
    return response
    
async def get_LLM_response_async_voting(compiler_response, llm_reponses_trace, top_p = 1, temperature=1, nb_responses=1):
    new_user_message = compiler_response 
    messages = llm_reponses_trace[-1]['all_prev_messages'][:]
    messages.append({"role": "user", "content": new_user_message})
    responses = await asyncio.gather(*[async_chat_completion(messages=messages, temperature = temperature, top_p=top_p) for _ in range(nb_responses)])
    # response = async_chat_completion(messages=messages, temperature = temperature, top_p=top_p)
    llm_sched_choices = []
    grouped_messages_per_sched = group_responses_per_sched([response.choices[0] for response in responses])
    selected_sched = max(list(grouped_messages_per_sched.keys()), key = lambda x:len(grouped_messages_per_sched[x]))
    selected_message = grouped_messages_per_sched[selected_sched][0]
    new_assitant_message = selected_message
    messages.append({"role": "assistant", "content": new_assitant_message})
    llm_reponses_trace_elem = {'all_prev_messages':messages, #includes all the conversation so far including the newly selected llm message
                              'grouped_answers_per_sched':grouped_messages_per_sched, # includes all the answers gotten from the llm grouped my the suggested schedule
                               'new_selected_sched': selected_sched, # the sched that has newly been selected.
                               'tokens_usage': { # the number of tokens used so far
                                   'completion_tokens': responses[0].usage.completion_tokens, # just taking the tokens for response 0 as an estimate since the index of selected response not saved above
                                   'prompt_tokens': responses[0].usage.prompt_tokens,
                                   'total_tokens': responses[0].usage.total_tokens
                               }
                              }
    llm_reponses_trace.append(llm_reponses_trace_elem)
    last_line = new_assitant_message.splitlines()[-1]
    return last_line, new_assitant_message
    
# def get_LLM_response_w_voting(compiler_response, top_p, n):
#     new_user_message = compiler_response 
#     messages = llm_reponses_trace[-1]['all_prev_messages'][:]
#     messages.append({"role": "user", "content": new_user_message})
#     response = client.chat.completions.create(
#         model=llm_id,
#         top_p = top_p,
#         n = n,
#         messages=messages
#     )
#     llm_sched_choices = []
#     # for choice in response.choices:
        
#     #     last_line = choice.message.content.splitlines()[-1]
#     #     if 'no further transformations' in last_line.lower():
#     #         llm_sched_choices.append('No further transformations')
#     #     elif last_line.lower().startswith('full list of transformations:'):
#     #         assert last_line.count(':') == 1
#     #         llm_sched = last_line.split(':')[1].replace(' ', '')
#     #         llm_sched_choices.append(llm_sched)
#     #     else:
#     #         # unparsable response, just skip
#     #         pass
#     grouped_messages_per_sched = group_responses_per_sched(response.choices)
#     selected_sched = max(list(grouped_messages_per_sched.keys()), key = lambda x:len(grouped_messages_per_sched[x]))
#     selected_message = grouped_messages_per_sched[selected_sched][0]
#     new_assitant_message = selected_message
#     messages.append({"role": "assistant", "content": new_assitant_message})
#     llm_reponses_trace_elem = {'all_prev_messages':messages, #includes all the conversation so far including the newly selected llm message
#                               'grouped_answers_per_sched':grouped_messages_per_sched, # includes all the answers gotten from the llm grouped my the suggested schedule
#                                'new_selected_sched': selected_sched # the sched that has newly been selected.
#                               }
#     llm_reponses_trace.append(llm_reponses_trace_elem)
#     last_line = new_assitant_message.splitlines()[-1]
#     return last_line, new_assitant_message


def initialize_conversation(system_intructions, first_message, temperature=0, seed=42):
    messages=[
        {"role": "system", "content": system_intructions},
        {"role": "user", "content": first_message}
    ]
    # sending system message and first message
    response = client.chat.completions.create(
        model=llm_id,
        messages=messages,
        temperature = temperature,
        seed = seed
    )
    new_assitant_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": new_assitant_message})

    llm_reponses_trace = [{'all_prev_messages':messages, 
                           'grouped_answers_per_sched':None, 
                           'new_selected_sched': None, 
                           'tokens_usage': { # the number of tokens used so far
                               'completion_tokens': response.usage.completion_tokens,
                               'prompt_tokens': response.usage.prompt_tokens,
                               'total_tokens': response.usage.total_tokens
                           } 
                          }]
    return llm_reponses_trace, new_assitant_message
    
def print_llm_message(id, message, as_markdown=False):
    if as_markdown:
        display_markdown(f'{id}⬅️ Assistant: '+ message + '\n', raw=True)    
    else:
        print(f'\033[1m{id}⬅️ Assistant:\033[0m '+ message + '\n')  
        
def print_compiler_message(id, message, as_markdown=False):
    if as_markdown:
        display_markdown(f'{id}➡️ Compiler: '+ message + '\n', raw=True)    
    else:
        print(f'\033[1m{id}➡️ Compiler:\033[0m '+ message + '\n')  

def group_responses_per_sched(choices):
    responses_dict = dict()
    for choice in choices:
        last_line = choice.message.content.splitlines()[-1]
        if 'no further transformations' in last_line.lower():
            llm_sched = 'No further transformations'
        elif last_line.lower().startswith('full list of transformations:'):
            # assert last_line.count(':') == 1, f"Extra semi-column in last line? {last_line}"
            if last_line.count(':') != 1:
                print(f"Extra semi-column in last line? {last_line}")
            llm_sched = last_line.split(':')[-1].replace(' ', '')
        else:
            # unparsable response, just skip
            llm_sched='unparsable response?'
        responses_dict[llm_sched] = responses_dict.get(llm_sched,[]) + [choice.message.content]

    return responses_dict

# def schedule_is_valid(schedule, llm_transfs):
#     if not unrollings_applied_last_in_branch(schedule, llm_transfs):
#         return False
#     elif not unrollings_applied_once_in_branch(schedule, llm_transfs):
#         return False
#     elif has_unroll_factor_1(schedule, llm_transfs):
#         return False
#     elif has_skewing_interchange_on_imperfectly_nested(schedule, llm_transfs):
#         return False
#     elif has_skewing_tiling_on_non_consecutive(schedule, llm_transfs):
#         return False
#     elif has_tiling_interchange_after_parallelization(schedule, llm_transfs):
#         return False
#     return True

def is_new_optim_valid(prev_schedule, new_optim, llm_transfs):
    if new_optim.is_unrolling():
       
        if not unrollings_applied_once_in_branch(prev_schedule, new_optim, llm_transfs):
            return False
        elif has_unroll_factor_1(prev_schedule, new_optim, llm_transfs):
            return False
        
    elif new_optim.is_skewing():
        if not are_iterators_consecutive(new_optim.iterators):
            raise KnownScheduleCreationCrash(f'Skewing cannot be applied on non consecutive loops (command "{llm_transfs[-1]}"')
        if not are_iterators_perfectly_nested(prev_schedule.tree, new_optim.iterators):
            raise KnownScheduleCreationCrash(f'Skewing cannot be applied on imperfectly nested loops (command "{llm_transfs[-1]}")')
        # check if solution found?

    elif new_optim.is_interchange():
        iterators = sorted(new_optim.params, key=lambda x:x[1])
        if not are_iterators_perfectly_nested(prev_schedule.tree, iterators):
            raise KnownScheduleCreationCrash(f'Interchange cannot be applied on imperfectly nested loops (command "{llm_transfs[-1]}")')

        if has_interchange_on_parallelized_loops(prev_schedule, new_optim, llm_transfs):
            return False

    elif new_optim.is_any_tiling():
        if not are_iterators_consecutive(new_optim.iterators):
            raise KnownScheduleCreationCrash(f'Tiling cannot be applied on non consecutive loops (command "{llm_transfs[-1]}"')

        if has_tiling_above_parallelized_loops(prev_schedule, new_optim, llm_transfs):
            return False

    # checks that need do be done no matter what's the new optim
    if not unrollings_applied_last_in_branch(prev_schedule, new_optim, llm_transfs):
        return False
    return True


def unrollings_applied_last_in_branch(prev_schedule, new_optim, llm_transfs):
    new_optim_comps = new_optim.comps
    
    for i,optim in enumerate(prev_schedule.optims_list):
        if optim.is_unrolling():
            unrolled_comps = optim.comps
            if any(comp in unrolled_comps for comp in new_optim_comps):
                raise KnownScheduleCreationCrash(f'Transformation {llm_transfs[-1]} cannot be applied after unrolling the same computations (unrolled by {llm_transfs[i]}). You might consider reordering the transformations.')
                return False 
    return True

def unrollings_applied_once_in_branch(prev_schedule, new_optim, llm_transfs):
    assert new_optim.is_unrolling()

    new_optim_comps = new_optim.comps
    
    for i,optim in enumerate(prev_schedule.optims_list):
        if optim.is_unrolling():
            unrolled_comps = optim.comps
            if any(comp in unrolled_comps for comp in new_optim_comps):
                raise KnownScheduleCreationCrash(f'Unrolling {llm_transfs[-1]} cannot bet applied on already unrolled computations (previously unrolled by {llm_transfs[i]})')
                return False 
    return True

def has_unroll_factor_1(prev_schedule, new_optim, llm_transfs):
    assert new_optim.is_unrolling()
    if new_optim.unrolling_factor==1:
        raise KnownScheduleCreationCrash(f'{llm_transfs[-1]}: cannot unroll a loop with a factor of 1.')
        return True
    return False

# def _has_skewing_interchange_on_imperfectly_nested(schedule, llm_transfs):
#     for i,optim in enumerate(schedule.optims_list):
#         if optim.is_interchange(): 
#             iterators = sorted(optim.params, key=lambda x:x[1])
#             if notare_iterators_perfectly_nested(schedule.tree, iterators):
#                 raise KnownScheduleCreationCrash(f'Interchange cannot be applied on imperfectly nested loops (command "{llm_transfs[i]}")')
#                 return True
#         elif optim.is_skewing():      
#             if notare_iterators_perfectly_nested(schedule.tree, optim.iterators):
#                 raise KnownScheduleCreationCrash(f'Skewing cannot be applied on imperfectly nested loops (command "{llm_transfs[i]}")')
#                 return True

#     return False

def are_iterators_perfectly_nested(schedule_tree, iter_id_list_inp):
    # make sure that the ids are canonical (i.e. use the names that figure in tree.iterators)
    iter_id_list = []
    for iter_id in iter_id_list_inp:
        if iter_id in schedule_tree.iterators:
            iter_id_list.append(iter_id)
        else:
            iter_id_list.append(schedule_tree.get_iterator_of_computation(*iter_id).id)
        
        # assert iter_id in schedule_tree.iterators, f"{iter_id} not found in {schedule_tree.iterators}"

    for i in range(len(iter_id_list)-1):
        node = schedule_tree.iterators[iter_id_list[i]]
        if not (len(node.child_iterators)==1 and len(node.computations_list)==0 and iter_id_list[i+1] in node.child_iterators):
            return False
    return True

def are_iterators_consecutive(iter_id_list):
    if any((iter_id_list[i+1][1]-iter_id_list[i][1])!=1 for i in range(len(iter_id_list)-1)):
        return False
    return True
            
# def _has_skewing_tiling_on_non_consecutive(schedule, llm_transfs):
#     for i,optim in enumerate(schedule.optims_list):
#         # if optim.is_interchange(): 
#         #     if abs(optim.params[1][1]-optim.params[0][1])!=1:
#         #         raise KnownScheduleCreationCrash(f'Interchange cannot be applied on non consecutive loops (command "{llm_transfs[i]}"')
#         #         return True 
#         if optim.is_skewing():
#             if (optim.iterators[1][1]-optim.iterators[0][1])!=1:
#                 raise KnownScheduleCreationCrash(f'Skewing cannot be applied on non consecutive loops (command "{llm_transfs[i]}"')
#                 return True 
#         elif optim.is_any_tiling():
#             if any((optim.iterators[i+1][1]-optim.iterators[i][1])!=1 for i in range(len(optim.iterators)-1)):
#                 raise KnownScheduleCreationCrash(f'Tiling cannot be applied on non consecutive loops (command "{llm_transfs[i]}"')
#                 return True 
#     return False

# def _has_tiling_interchange_after_parallelization(schedule, llm_transfs):
#     parallelization_indices = []
#     for i,optim in enumerate(schedule.optims_list):
#         if optim.is_parallelization():
#             parallelization_indices.append(i)
            
#     for parallelization_idx in parallelization_indices:
#         parallelization_action = schedule.optims_list[parallelization_idx]
#         parallel_level = parallelization_action.iterator_id[1]
#         involved_comps = parallelization_action.comps
#         # alternative_ids = schedule.tree.get_alternative_iterator_identifers(schedule.tree.get_iterator_name_from_id(iterator_id))
#         for j, optim in enumerate(schedule.optims_list[parallelization_idx+1:]):
#             if optim.is_interchange():
#                 # if it affects parallelized comps and one of the interchanged levels is parallelized, this can cause a wrong legality evaluation
#                 if any(comp in involved_comps for comp in optim.comps) and (optim.params[0][1]==parallel_level or optim.params[1][1]==parallel_level):
#                     raise KnownScheduleCreationCrash(f'Interchange {llm_transfs[j+parallelization_idx+1]} cannot be applied after parallelization on the same loop (parallelized by {llm_transfs[parallelization_idx]}). You might consider reordering the transformations.')
#                     return True
#             elif optim.is_any_tiling():
#                 # if it affects parallelized comps and and the tiling level is higher than the parallel one, this can cause a wrong legality evaluation since new loops will be created while the parallel tag stays in the same level
#                 if any(comp in involved_comps for comp in optim.comps) and (optim.iterators[0][1]<parallel_level):
#                     raise KnownScheduleCreationCrash(f'Tiling {llm_transfs[j+parallelization_idx+1]} cannot be applied after parallelization on the same loop (parallelized by {llm_transfs[parallelization_idx]}). You might consider reordering the transformations.')
#                     return True
#     return False

def has_interchange_on_parallelized_loops(prev_schedule, new_optim, llm_transfs):
    assert new_optim.is_interchange()
    interchanged_levels = [new_optim.params[0][1], new_optim.params[1][1]]
    interchange_comps = new_optim.comps
    for i,optim in enumerate(prev_schedule.optims_list):
        if optim.is_parallelization():
            parallel_level = optim.iterator_id[1]
            parallel_comps = optim.comps
            # if it affects parallelized comps and one of the interchanged levels is parallelized, this can cause a wrong legality evaluation
            if any(comp in interchange_comps for comp in parallel_comps) and (interchanged_levels[0]==parallel_level or interchanged_levels[1]==parallel_level):
                raise KnownScheduleCreationCrash(f'Interchange {llm_transfs[-1]} cannot be applied after parallelization on the same loop (parallelized by {llm_transfs[i]}). You might consider reordering the transformations.')
                return True
    return False

def has_tiling_above_parallelized_loops(prev_schedule, new_optim, llm_transfs):
    assert new_optim.is_any_tiling()
    first_tiled_level = new_optim.iterators[0][1]
    tiled_comps = new_optim.comps
    for i,optim in enumerate(prev_schedule.optims_list):
        if optim.is_parallelization():
            parallel_level = optim.iterator_id[1]
            parallel_comps = optim.comps
            # if it affects parallelized comps and and the tiling level is higher than the parallel one, this can cause a wrong legality evaluation since new loops will be created while the parallel tag stays in the same level
            if any(comp in tiled_comps for comp in parallel_comps) and (first_tiled_level<parallel_level):
                raise KnownScheduleCreationCrash(f'Tiling {llm_transfs[-1]} cannot be applied on levels above parallelized loops (parallelized by {llm_transfs[i]}). You might consider reordering the transformations.')
                return True
    return False

def validate_optim_and_add_to_schedule(prev_schedule, new_optim, llm_transfs):
    new_optim.initialize_action_for_tree(prev_schedule.tree)
    assert is_new_optim_valid(prev_schedule, new_optim, llm_transfs)
    prev_schedule.add_optimizations([new_optim])
    return prev_schedule

def extract_transformation_commands(last_line):
    match = re.search(r"<schedule>(.*?)</schedule>", last_line)

    if match:
        llm_sched_str = match.group(1)
        return llm_sched_str.replace(' ', '').replace('`','')
    else:
        return None
