from sympy import simplify
import re
from string import ascii_lowercase
import logging
import os
from pathlib import Path
import time
import pickle
import sys
from filelock import FileLock
import traceback
import json

import numpy as np

def parse_isl_map(isl_map: str):
    # Extract the computation and buffer parts
    match = re.match(r'.*{\s*(\w+)\[([^\]]+)\]\s*->\s*(\w+)\[([^\]]+)\]\s*}', isl_map)
    if not match:
        raise ValueError("Invalid ISL map format")
    
    computation_name = match.group(1)
    comp_iterators = match.group(2).split(',')
    buffer_name = match.group(3)
    buffer_accesses = match.group(4).split(',')
    
    # Strip any spaces around iterators or accesses
    comp_iterators = [it.strip() for it in comp_iterators]
    buffer_accesses = [access.strip() for access in buffer_accesses]
    
    return comp_iterators, buffer_accesses

def extract_affine_coefficients(access, comp_iterators):
    """ 
    Extract coefficients of iterators in the affine expression.
    Example: access = 'i1 + 2*j2', comp_iterators = ['i1', 'j2', 'k']
    Output: [1, 2, 0, 0] (coefficients for i1, j2, k, and scalar term)
    """
    # remove all spaces
    access = access.replace(' ', '')
    
    # Initialize coefficients (including one for the scalar term)
    coefficients = [0] * (len(comp_iterators) + 1)
    
    # Match terms like '2*i1' or 'i1' or '-i1', and extract the coefficients
    for i, it in enumerate(comp_iterators):
        # Modify the pattern to match iterator names with numbers or underscores
        term_pattern = re.compile(r'([+-]?\d*)\s*\*?\s*(' + re.escape(it) + r')(\b|$)')
        match = term_pattern.search(access)
        if match:
            coeff = match.group(1)
            coefficients[i] = int(coeff) if coeff and coeff != '+' and coeff != '-' else 1 if coeff == '' or coeff == '+' else -1
    
    # Handle the scalar part, which is just a constant not attached to any iterator
    scalar_pattern = re.compile(r'([+-]?\b\d+\b)(?![\*\w])')  # Match isolated constants (scalars) not tied to iterators
    scalar_match = scalar_pattern.search(access)
    
    if scalar_match:
        coefficients[-1] = int(scalar_match.group(1))
    
    return coefficients

def isl_map_to_write_access_matrix(isl_map: str):
    # Parse the ISL map
    comp_iterators, buffer_accesses = parse_isl_map(isl_map)
    
    # Initialize the access matrix (rows = buffer dimensions, columns = iterators + scalar)
    access_matrix = []
    
    # Process each buffer access and extract affine coefficients
    for access in buffer_accesses:
        row = extract_affine_coefficients(access, comp_iterators)
        access_matrix.append(row)
    
    return np.array(access_matrix).tolist()

def get_program_preorder_traversal(program_annotations, comp_or_loop_name='program_root', depth=-1):
    """ Return a list the preorder traversal of the program. Each element in the list is a dict containing basic info about a loop or computation
    """
    if comp_or_loop_name=='program_root': # if this is the first call, the children are the roots
        iterators_dict = program_annotations['iterators']
        children_names = [iter for iter in iterators_dict if iterators_dict[iter]['parent_iterator'] == None]
        children_names = [iter for iter in children_names if not (iterators_dict[iter]['lower_bound']=='0' and iterators_dict[iter]['upper_bound']=='1')] # skip the dummy iterators (extent of 1)
        type_ = 'loop'
    elif comp_or_loop_name in program_annotations['iterators']:
        iter_dict = program_annotations['iterators'][comp_or_loop_name]
        children_names = iter_dict['child_iterators']+iter_dict['computations_list']
        type_ = 'loop'
    elif comp_or_loop_name in program_annotations['computations']:
        children_names = []
        type_ = 'comp'
    else:
        raise ValueError(f"{comp_or_loop_name} is neither a computation nor a loop name")

    pre_order_traversal = [{'name':comp_or_loop_name, 'type':type_, 'depth':depth, 'nb_children':len(children_names)}]

    children_names = sorted(children_names, key=lambda x:get_absolute_order(x,program_annotations))

    for child_name in children_names:
        pre_order_traversal+=get_program_preorder_traversal(program_annotations, child_name, depth+1)
    return pre_order_traversal

def get_absolute_order(comp_or_loop_name,program_annotations):
    """ Returns the absolute order of computation or loop w.r.t the preorder traversal of the program
    """
    if comp_or_loop_name in program_annotations['iterators']:
        iter_dict = program_annotations['iterators'][comp_or_loop_name]
        if 'absolute_order' in iter_dict: # if previously computed
            return iter_dict['absolute_order']
        # since we want to use the abs order for getting preorder traversal, we make sure the loop's order is smaller than all its children's. 
        # This is done by the -0.001. This works assuming that comp's abs order are always ints and a loop can't have >999 nested descendants 
        abs_order = min([get_absolute_order(child,program_annotations) for child in iter_dict['child_iterators']+iter_dict['computations_list']])-0.001
        iter_dict['absolute_order'] = abs_order # as an optimizatin to avoid recomputing during the recursion
        return iter_dict['absolute_order']
    elif comp_or_loop_name in program_annotations['computations']:
        return program_annotations['computations'][comp_or_loop_name]['absolute_order']
    else:
        raise ValueError(f"{comp_or_loop_name} is neither a computation nor a loop name")

# retruns a string that represents a memory access. eg buf_19(l0,l1+1). If iterator names are given, it uses iterator names instead of loop levels
def get_access_str_from_access_matrix(buf_id, acc_mat, iterator_names = None, c_style = False):
    """
    Extract access relation str (eg "(i1 , i1 + 2*j2, k + 10)") from access matrix.
    If iterator names are given, it uses iterator names instead of loop levels
    :param access_matrix: as a list of lists
    :param comp_iterator_names: a list of iterator names where the access occurs
    :return: buf_+id+ access relation str with parentheses
    """

    dim_accesses = []
    if iterator_names is None:
        iterator_names = ['L'+str(i) for i in range(len(acc_mat[0]))]
    for row in acc_mat:
        dim_acc = ''
        for col,e in enumerate(row):
            if e!=0:
                if col!=len(row)-1:
                    dim_acc += ('+' if dim_acc!='' else '') + (str(e)+'*' if e!=1 else '') + iterator_names[col]
                else:
                    dim_acc += ('+' if e>0 else '') + str(int(e))

        dim_accesses.append(dim_acc if dim_acc!='' else '0')
    if c_style:
        reconstructed_str = 'buf_' + str(buf_id) + '[' + ']['.join(dim_accesses) + ']'
    else:
        reconstructed_str = 'buf_' + str(buf_id) + '(' + ','.join(dim_accesses) + ')'
    return reconstructed_str

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


class InvalidScheduleError(KnownScheduleCreationCrash):
    """Raised when a proposed schedule is syntactically or structurally invalid
    (malformed command, unknown comp_ID, out-of-range loop level, non-integer
    factor, etc.). These problems are caught before any compiler call and are
    reported back to the LLM as precise, actionable validity errors instead of
    opaque "crashed for unclear reasons" messages."""
    pass


def _valid_comp_ids_str(compute_blocks_IDs):
    """Human-readable, comma-separated list of the valid comp_IDs of a program."""
    return ', '.join(sorted(compute_blocks_IDs.keys()))


def _resolve_comp(comp_id, compute_blocks_IDs, transf):
    """Map a comp_ID (e.g. comp00) to its underlying computation name, raising a
    clear InvalidScheduleError if the comp_ID does not exist in the program."""
    if comp_id not in compute_blocks_IDs:
        raise InvalidScheduleError(
            f'Unknown computation ID "{comp_id}" in "{transf}". The valid '
            f'computation IDs for this program are: {_valid_comp_ids_str(compute_blocks_IDs)}. '
            f'Use one of these IDs exactly as they appear in the loop nest annotations.'
        )
    return compute_blocks_IDs[comp_id][0]


def _parse_level(loop_token, real_comp_name, schedule, transf, comp_id):
    """Parse a loop-level token (e.g. "L1", "L-1") into an integer level and
    validate it against the loops actually enclosing the computation. Raises a
    descriptive InvalidScheduleError on any problem."""
    if not loop_token or loop_token[0] != 'L':
        raise InvalidScheduleError(
            f'Invalid loop level "{loop_token}" in "{transf}": a loop level must be written '
            f'as the letter L followed by its depth (e.g. L0 for the outermost loop, L1 for '
            f'the next one). Use L-1 to refer to the innermost loop.'
        )
    try:
        level = int(loop_token[1:])
    except ValueError:
        raise InvalidScheduleError(
            f'Invalid loop level "{loop_token}" in "{transf}": the value after L must be an '
            f'integer (e.g. L0, L1, L2). Use L-1 to refer to the innermost loop.'
        )
    innermost = get_innermost_level(real_comp_name, schedule)
    if level == -1:
        return innermost
    if level < 0 or level > innermost:
        depth = innermost + 1
        raise InvalidScheduleError(
            f'Loop level {loop_token} does not exist for {comp_id} in "{transf}". '
            f'{comp_id} is enclosed by {depth} loop level(s), so its valid loop levels are '
            f'L0 (outermost) to L{innermost} (innermost). You can also use L-1 for the innermost loop.'
        )
    return level


def _parse_call_args(transf_inst, transf_id, expected_n, transf, usage_example):
    """Validate the `Name(arg1, arg2, ...)` structure of a transformation command
    and return the list of raw argument strings. Raises InvalidScheduleError with
    a concrete usage example when the parentheses or argument count are wrong."""
    if not (len(transf_inst) > len(transf_id)
            and transf_inst[len(transf_id)] == '(' and transf_inst[-1] == ')'):
        raise InvalidScheduleError(
            f'The command "{transf}" is malformed: arguments must be enclosed in parentheses. '
            f'Expected format: {usage_example}.'
        )
    params = transf_inst[len(transf_id) + 1:-1]
    split_params = params.split(',')
    non_empty = [p for p in split_params if p != '']
    if len(split_params) != expected_n or len(non_empty) != expected_n:
        raise InvalidScheduleError(
            f'The command "{transf}" has the wrong number of arguments: expected exactly '
            f'{expected_n}, but got {len(non_empty)}. Expected format: {usage_example}.'
        )
    return split_params


def _parse_int_arg(token, transf, what='factor'):
    """Parse an integer transformation argument (tiling / unrolling factor)."""
    try:
        return int(token)
    except (ValueError, TypeError):
        raise InvalidScheduleError(
            f'Invalid {what} "{token}" in "{transf}": {what}s must be integers '
            f'(e.g. 4, 16, 32, 64).'
        )
    
import openai
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url = os.getenv("BASE_URL"), max_retries=10)


BaseConfig.init(config_yaml= os.getenv("TIRALIB_DIR")+"/config.yaml")

lookup_files_dir = './lookup_dir_'+os.getenv("LOOKUP_DIR_SUFFIX")
Path(lookup_files_dir).mkdir(exist_ok=True)

llm_id = os.getenv("LLM_ID")

# Whether to reuse cached compiler feedback messages across schedules/runs.
# The numeric caches (initial execution time and skewing solver factors) are
# always kept because they are format-agnostic and expensive to recompute; only
# the textual compiler-response cache is gated here. Disabling it guarantees that
# every schedule gets a freshly generated message (important when comparing runs
# after changing the feedback wording, and for honest stochastic multi-run
# experiments). Controlled by the USE_RESPONSE_LOOKUP env var ("0" to disable).
RESPONSE_LOOKUP_ENABLED = os.getenv("USE_RESPONSE_LOOKUP", "1") != "0"

# When an illegal *compound* schedule is found, optionally pinpoint the offending
# transformation by legality-checking each component on its own. This is helpful
# feedback but costs one extra legality check per component, which is noticeable
# on large programs, so it is off by default. Controlled by LOCALIZE_ILLEGALITY.
LOCALIZE_ILLEGALITY = os.getenv("LOCALIZE_ILLEGALITY", "0") == "1"

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
    """Parse an LLM schedule string into a TiraLib Schedule, validating each
    transformation and raising InvalidScheduleError with an actionable message on
    any malformed or structurally-invalid command."""
    schedule = Schedule(program)
    llm_sched_str = llm_sched_str.replace(' ', '').replace('`','')
    llm_transfs = llm_sched_str.split('+')
    for transf_idx,transf in enumerate(llm_transfs):
        if transf.count('.') != 1:
            raise InvalidScheduleError(
                f'The command "{transf}" is malformed: each transformation must have the form '
                f'comp_ID.Transformation(arguments), with exactly one dot separating the '
                f'computation ID from the transformation, e.g. comp00.Parallelize(L0). '
                f'If you meant to combine several transformations, join them with "+".'
            )
        comp_id, transf_inst = tuple(transf.split('.'))
        real_comp_name = _resolve_comp(comp_id, compute_blocks_IDs, transf)
        if transf_inst.lower().startswith('fuse'):
            transf_id = 'fuse'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 2, transf, 'comp_ID.Fuse(other_comp_ID, L<level>)')
            second_comp_id = split_transf_params[0]
            second_comp_real_name = _resolve_comp(second_comp_id, compute_blocks_IDs, transf)
            if second_comp_real_name == real_comp_name:
                raise InvalidScheduleError(
                    f'Cannot fuse {comp_id} with itself in "{transf}". Fusion merges two '
                    f'different computation blocks; the two comp_IDs must be distinct.')
            loop_level = _parse_level(split_transf_params[1], real_comp_name, schedule, transf, comp_id)
            second_innermost = get_innermost_level(second_comp_real_name, schedule)
            if loop_level > second_innermost:
                raise InvalidScheduleError(
                    f'Cannot fuse at loop level L{loop_level} in "{transf}": computation '
                    f'{second_comp_id} only has loop levels L0 to L{second_innermost}, so it has no '
                    f'loop at L{loop_level}. Fusion must be applied at a level that exists in both '
                    f'computations (fuse at the deepest level they share).')
            loop_id1 = (real_comp_name,loop_level)
            loop_id2 = (second_comp_real_name,loop_level)
            new_optim = tiramisu_actions.Fusion([loop_id1,loop_id2])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

        elif transf_inst.lower().startswith('parallelize'):
            transf_id = 'parallelize'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 1, transf, 'comp_ID.Parallelize(L<level>)')
            loop_level = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            if loop_level>=2:
                raise InvalidScheduleError(
                    f'Parallelization in "{transf}" targets loop level L{loop_level}, but only '
                    f'outer loops (L0, or L1) should be parallelized. Parallelizing deep loops '
                    f'(L2 and beyond) adds threading overhead that outweighs the benefit; '
                    f'parallelize an outer loop instead.')
            loop_id = (real_comp_name, loop_level)
            new_optim = tiramisu_actions.Parallelization([loop_id])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

        elif transf_inst.lower().startswith('reverse'):
            transf_id = 'reverse'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 1, transf, 'comp_ID.Reverse(L<level>)')
            loop_level = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            loop_id = (real_comp_name, loop_level)
            new_optim = tiramisu_actions.Reversal([loop_id])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])


        elif transf_inst.lower().startswith('interchange'):
            transf_id = 'interchange'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 2, transf, 'comp_ID.Interchange(L<level1>, L<level2>)')
            loop_level1 = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            loop_level2 = _parse_level(split_transf_params[1], real_comp_name, schedule, transf, comp_id)
            loop_id1 = (real_comp_name,loop_level1)
            loop_id2 = (real_comp_name,loop_level2)
            new_optim = tiramisu_actions.Interchange([loop_id1,loop_id2])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

        elif transf_inst.lower().startswith('skew'):
            transf_id = 'skew'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 2, transf, 'comp_ID.Skew(L<level1>, L<level2>)')
            loop_level1 = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            loop_level2 = _parse_level(split_transf_params[1], real_comp_name, schedule, transf, comp_id)
            loop_id1 = (real_comp_name,loop_level1)
            loop_id2 = (real_comp_name,loop_level2)

            # Validate the skewing preconditions before invoking the (expensive)
            # solver that computes the skewing factors. The action is initialized
            # against the current tree with placeholder factors (0, 0) purely for
            # this validity check.
            temp_skew_action = tiramisu_actions.Skewing([loop_id1,loop_id2,0,0])
            temp_skew_action.initialize_action_for_tree(schedule.tree)
            assert is_new_optim_valid(schedule, temp_skew_action, llm_transfs[:transf_idx+1])


            lookup_res=check_in_lookup_file('+'.join(llm_transfs[:transf_idx+1]), program.name, 'skewing_solver')
            if lookup_res!='Not found':
                skewing_factors = lookup_res
                logger.debug(f"Skewing solu for {'+'.join(llm_transfs[:transf_idx+1])} retrieved from lookup are {skewing_factors}.")
            else:
                loop_level1_node = schedule.tree.get_iterator_of_computation(real_comp_name, 1)
                involved_computations = schedule.tree.get_iterator_subtree_computations(loop_level1_node.id)
                skewing_factors = tiramisu_actions.Skewing.get_factors(schedule, [loop_level1, loop_level2], involved_computations)
                update_lookup_file('+'.join(llm_transfs[:transf_idx+1]), program.name, 'skewing_solver', skewing_factors)

            if skewing_factors is None:
                raise InvalidScheduleError(
                    f'The solver could not find valid skewing factors for "{transf}". '
                    f'Skewing is only useful/possible on a pair of perfectly nested loops that '
                    f'carry dependencies; try skewing a different loop pair, or drop the skewing.')
            factor1, factor2 = skewing_factors
            schedule.add_optimizations([tiramisu_actions.Skewing([loop_id1,loop_id2,factor1,factor2])])

        elif transf_inst.lower().startswith('tile2d'):
            transf_id = 'tile2d'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 4, transf,
                'comp_ID.Tile2D(L<level1>, L<level2>, <factor1>, <factor2>)')
            loop_level1 = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            loop_level2 = _parse_level(split_transf_params[1], real_comp_name, schedule, transf, comp_id)
            factor1 = _parse_int_arg(split_transf_params[2], transf, 'tiling factor')
            factor2 = _parse_int_arg(split_transf_params[3], transf, 'tiling factor')
            loop_id1 = (real_comp_name,loop_level1)
            loop_id2 = (real_comp_name,loop_level2)
            new_optim = tiramisu_actions.Tiling2D([loop_id1,loop_id2,factor1,factor2])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

        elif transf_inst.lower().startswith('tile3d'):
            transf_id = 'tile3d'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 6, transf,
                'comp_ID.Tile3D(L<level1>, L<level2>, L<level3>, <factor1>, <factor2>, <factor3>)')
            loop_level1 = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            loop_level2 = _parse_level(split_transf_params[1], real_comp_name, schedule, transf, comp_id)
            loop_level3 = _parse_level(split_transf_params[2], real_comp_name, schedule, transf, comp_id)
            factor1 = _parse_int_arg(split_transf_params[3], transf, 'tiling factor')
            factor2 = _parse_int_arg(split_transf_params[4], transf, 'tiling factor')
            factor3 = _parse_int_arg(split_transf_params[5], transf, 'tiling factor')
            loop_id1 = (real_comp_name,loop_level1)
            loop_id2 = (real_comp_name,loop_level2)
            loop_id3 = (real_comp_name,loop_level3)
            new_optim = tiramisu_actions.Tiling3D([loop_id1,loop_id2,loop_id3,factor1,factor2,factor3])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])

        elif transf_inst.lower().startswith('unroll'):
            transf_id = 'unroll'
            split_transf_params = _parse_call_args(
                transf_inst, transf_id, 2, transf, 'comp_ID.Unroll(L<level>, <factor>)')
            loop_level = _parse_level(split_transf_params[0], real_comp_name, schedule, transf, comp_id)
            factor = _parse_int_arg(split_transf_params[1], transf, 'unrolling factor')
            unroll_limit = _over_unroll_limit(real_comp_name, loop_level, schedule)
            if unroll_limit is not None and factor > unroll_limit:
                raise InvalidScheduleError(
                    f'Unrolling factor {factor} in "{transf}" is larger than the trip count '
                    f'({unroll_limit}) of the loop. Unrolling by more than the number of '
                    f'iterations is pointless and makes the code generator fail; use a factor '
                    f'between 2 and {unroll_limit} (ideally a divisor of the loop extent).')
            loop_id = (real_comp_name,loop_level)
            new_optim = tiramisu_actions.Unrolling([loop_id,factor])
            schedule = validate_optim_and_add_to_schedule(schedule, new_optim, llm_transfs[:transf_idx+1])
        else:
            transf_name = transf_inst.split('(')[0]
            raise InvalidScheduleError(
                f'Unknown transformation "{transf_name}" in "{transf}". The supported '
                f'transformations are: Fuse, Interchange, Parallelize, Tile2D, Tile3D, Unroll, '
                f'Skew, and Reverse. Please use one of these exactly.')


    return schedule

def get_innermost_level(real_comp_name, schedule):
    return schedule.tree.get_iterator_of_computation(real_comp_name).level


def _static_loop_extent(real_comp_name, loop_level, schedule):
    """Return the static trip count of the loop at `loop_level` enclosing the
    computation, if statically known, else None.

    NOTE on correctness: this reads the *current* schedule tree. The tree is
    regenerated from the real ISL AST after tiling/fusion/distribution, and
    reversal/parallelization/skewing do not reorder levels (skewing turns bounds
    non-rectangular, in which case this returns None). The only transformation
    that permutes level->extent without refreshing the tree is interchange; call
    sites must account for that (see `_max_enclosing_static_extent`)."""
    try:
        node = schedule.tree.get_iterator_of_computation(real_comp_name, loop_level)
    except Exception:
        return None
    lb, ub = node.lower_bound, node.upper_bound
    if isinstance(lb, int) and isinstance(ub, int):
        return ub - lb
    return None


def _max_enclosing_static_extent(real_comp_name, schedule):
    """Return the maximum statically-known trip count among all loops enclosing
    the computation, or None if none are statically known.

    This is invariant under interchange (which only permutes the enclosing
    loops), so `factor > max_enclosing_extent` is a *sound* over-unrolling test
    even when the tree's per-level mapping is stale after an interchange."""
    extents = []
    try:
        node = schedule.tree.get_iterator_of_computation(real_comp_name)
    except Exception:
        return None
    while node is not None:
        lb, ub = node.lower_bound, node.upper_bound
        if isinstance(lb, int) and isinstance(ub, int):
            extents.append(ub - lb)
        parent_id = node.parent_iterator
        node = schedule.tree.iterators.get(parent_id) if parent_id else None
    return max(extents) if extents else None


def _over_unroll_limit(real_comp_name, loop_level, schedule):
    """Return an upper bound U such that unrolling by a factor > U is guaranteed
    to exceed the target loop's trip count (and therefore crash the Tiramisu code
    generator). Returns None when no safe bound can be established, in which case
    the guard abstains (the schedule is passed through and any resulting backend
    crash is reported with recovery guidance instead of being wrongly rejected).

    When no interchange has been applied, the per-level extent is accurate and
    used directly (this catches e.g. tiling-then-unrolling a small tile loop).
    When an interchange is present the level->extent mapping in the tree may be
    stale, so the interchange-invariant max enclosing extent is used instead."""
    has_interchange = any(
        getattr(op, 'is_interchange', lambda: False)() for op in schedule.optims_list
    )
    if not has_interchange:
        level_extent = _static_loop_extent(real_comp_name, loop_level, schedule)
        if level_extent is not None:
            return level_extent
    return _max_enclosing_static_extent(real_comp_name, schedule)

def check_in_lookup_file(llm_sched_str, prog_name, query_type):
    assert query_type in ['initial_exec', 'compiler_responses', 'skewing_solver']
    if query_type == 'compiler_responses' and not RESPONSE_LOOKUP_ENABLED:
        return 'Not found'
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
    if query_type == 'compiler_responses' and not RESPONSE_LOOKUP_ENABLED:
        return
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
            # Entry already present: two parallel runs collided on the same new
            # schedule. Keep the existing value (idempotent).
            pass
        else:
            lookup_dict[query_type][normalized_llm_sched_str] = value
        with open(prog_lookup_file, 'wb') as f:
            pickle.dump(lookup_dict, f)

def _localize_illegality(llm_sched_str, tiramisu_program, annots, compute_blocks_IDs):
    """For a compound schedule found illegal, help the LLM localize the culprit by
    legality-checking each of its transformations on its own. Returns a short
    guidance string (starting with a space, or empty). Best-effort and cheap
    (legality checks do not execute code); never raises."""
    transfs = [t for t in llm_sched_str.split('+') if t]
    if len(transfs) < 2:
        return ''
    individually_illegal = []
    individually_legal = []
    try:
        for t in transfs:
            try:
                sched = llm_sched_str_to_sched(t, tiramisu_program, annots, compute_blocks_IDs)
                if sched.is_legal():
                    individually_legal.append(t)
                else:
                    individually_illegal.append(t)
            except Exception:
                # This single transformation can't be cleanly isolated; skip it.
                continue
    except Exception:
        return ''
    if individually_illegal:
        return (' Isolating the transformations shows that the following are already illegal on '
                'their own: ' + ', '.join(f'"{t}"' for t in individually_illegal) + '.')
    if individually_legal and len(individually_legal) == len(transfs):
        return (' Each transformation is legal on its own, so the illegality stems from their '
                'combination rather than a single one.')
    return ''


def get_compiler_response(llm_sched_str, tiramisu_program, annots, compute_blocks_IDs, initial_exec_time, explored_scheds, min_runs=10, max_runs=None, time_budget=None):
    prog_name = tiramisu_program.name
    if (lookup_res := check_in_lookup_file(llm_sched_str, prog_name, 'compiler_responses')) != 'Not found':
        logger.debug(f'Response for {llm_sched_str} retrieved from lookup.')
        message, outcome_type, outcome_data = lookup_res
        explored_scheds.append((llm_sched_str, outcome_type, True, outcome_data)) 
        return message
    try:
        schedule = llm_sched_str_to_sched(llm_sched_str, tiramisu_program, annots, compute_blocks_IDs)
    except InvalidScheduleError as e:
        # A schedule that is syntactically/structurally invalid. Caught before any
        # compiler call and reported to the LLM with a precise, actionable reason.
        explored_scheds.append((llm_sched_str, 'invalid', False, e.message))
        message = (f'The schedule {llm_sched_str} is invalid and was not applied. {e.message} '
                   f'None of its transformations were applied; please fix the issue and resend '
                   f'the full corrected schedule.')
        logger.info(message)
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'invalid', e.message))
        return message
    except KnownScheduleCreationCrash as e:
        explored_scheds.append((llm_sched_str, 'crash', False, traceback.format_exc()))
        message = (f'The schedule {llm_sched_str} could not be applied. {e.message} '
                   f'Please adjust the schedule accordingly.')
        logger.exception(message)
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc()))
        return message
    except Exception as e:
        explored_scheds.append((llm_sched_str, 'crash', False, traceback.format_exc()))
        message = (f'The schedule {llm_sched_str} made the compiler crash while it was being built '
                   f'({type(e).__name__}). This usually happens with aggressive or unusual '
                   f'combinations (e.g. large unrolling/tiling factors, or tiling then unrolling the '
                   f'same loop). Consider reducing the tiling/unrolling factors, reordering the '
                   f'transformations (apply fusion first and unrolling last), or dropping one of them.')
        logger.exception(message)
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc()))
        return message
    try:
        is_legal = schedule.is_legal()
    except Exception as e:
        explored_scheds.append((str(schedule), 'crash', False, traceback.format_exc()))
        message = (f'The schedule {llm_sched_str} caused the compiler to crash during the legality '
                   f'check. This can happen with unusual transformation combinations; consider '
                   f'simplifying the schedule or reordering its transformations (for instance, '
                   f'applying fusion first and unrolling last).')
        logger.exception(message)
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc()))
        return message
    if is_legal:
        try:
            exec_time = min(schedule.execute(min_runs=min_runs,max_runs=max_runs,time_budget=time_budget))
            speedup = initial_exec_time/exec_time
            explored_scheds.append((str(schedule), 'ran', False, exec_time)) 
            
            if speedup>=1:
                message = f'{llm_sched_str} was legal and yielded a speedup of {round(speedup,2)}x'
            else:
                message = f'{llm_sched_str} was legal and yielded a speedup of {round(speedup,2)}x (i.e. a slowdown of {round(1/speedup,2)}x)'

            update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'ran', exec_time)) 
            return message
                
        except ScheduleExecutionCrashed as e:
            explored_scheds.append((str(schedule), 'crash', False, traceback.format_exc()))
            message = (f'The schedule {llm_sched_str} was legal but crashed at runtime '
                       f'(segmentation fault) during execution. This often comes from an overly '
                       f'large unrolling or tiling factor, or unrolling a non-innermost loop. '
                       f'Consider reducing the factors, unrolling only the innermost loop (L-1), '
                       f'or removing the transformation likely responsible.')
            logger.exception(message)
            update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc()))
            return message

        except Exception as e:
            explored_scheds.append((str(schedule), 'crash', False, traceback.format_exc()))
            message = (f'The schedule {llm_sched_str} was legal but crashed at runtime during '
                       f'execution ({type(e).__name__}). This often comes from an overly large '
                       f'unrolling or tiling factor, or unrolling a non-innermost loop. Consider '
                       f'reducing the factors, unrolling only the innermost loop (L-1), or removing '
                       f'the transformation likely responsible.')
            logger.exception(message)
            update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'crash', traceback.format_exc()))

            return message

    else:
        explored_scheds.append((str(schedule), 'illegal', False, None))
        localization_hint = (_localize_illegality(llm_sched_str, tiramisu_program, annots, compute_blocks_IDs)
                             if LOCALIZE_ILLEGALITY else '')
        message = (f'The schedule {llm_sched_str} is illegal: it violates the data dependencies of '
                   f'the program (it would change the computed result), so it was not executed.'
                   f'{localization_hint} To make progress, revoke or replace the transformation(s) '
                   f'responsible for the violation, or try applying them in a different order.')
        update_lookup_file(llm_sched_str, prog_name, 'compiler_responses', (message, 'illegal', None))
        return message



# Error types that are always transient (network blips, timeouts, rate limits, 5xx).
_TRANSIENT_LLM_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)
# HTTP statuses worth retrying when returned by the endpoint or a gateway in front
# of it (request timeout, too-early, rate limit, and 5xx gateway errors).
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _is_transient(exc):
    """Whether an LLM request failure is worth retrying. Covers network/timeout/
    rate-limit/5xx errors and empty responses. A tunnel or gateway (e.g. ngrok, a
    load balancer) fronting the model can briefly return 404/5xx while the backend
    restarts, so those are retried too; a genuine 404 (wrong URL/model) is not."""
    if isinstance(exc, _TRANSIENT_LLM_ERRORS + (ValueError,)):
        return True
    status = getattr(exc, 'status_code', None)
    if status in _RETRYABLE_STATUS:
        return True
    if status == 404 and any(m in str(exc).lower()
                             for m in ('offline', 'ngrok', 'temporarily', 'unavailable', 'bad gateway')):
        return True
    return False


def _chat_completion(messages, max_attempts=60, **kwargs):
    """Call the chat completion endpoint, retrying transient failures with capped
    exponential backoff so external hiccups don't abort a run. Returns
    (assistant_message, tokens_usage). Raises after `max_attempts`, or immediately
    on a non-transient error (e.g. authentication or a malformed request)."""
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(model=llm_id, messages=messages, **kwargs)
            content = response.choices[0].message.content if response.choices else None
            if content is None:
                raise ValueError("empty response content")
            tokens_usage = {
                'completion_tokens': response.usage.completion_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'total_tokens': response.usage.total_tokens,
            }
            return content, tokens_usage
        except Exception as e:
            if attempt == max_attempts or not _is_transient(e):
                if _is_transient(e):
                    logger.error(f"LLM request still failing after {max_attempts} attempts: {e}")
                raise
            wait = min(2 ** min(attempt, 6), 60)
            logger.warning(f"LLM request failed ({type(e).__name__}: {e}); retry {attempt}/{max_attempts} in {wait}s")
            time.sleep(wait)


def get_LLM_response(compiler_response, messages, top_p = 1, temperature=1, enable_thinking = True):
    messages.append({"role": "user", "content": compiler_response})
    new_assitant_message, tokens_usage = _chat_completion(
        messages,
        top_p=top_p,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    messages.append({"role": "assistant", "content": new_assitant_message})
    last_line = new_assitant_message.splitlines()[-1] if new_assitant_message else ''
    return last_line, new_assitant_message, tokens_usage


def initialize_conversation(system_intructions, first_message, temperature=0, seed=42, enable_thinking=True):
    messages=[
        {"role": "system", "content": system_intructions},
        {"role": "user", "content": first_message}
    ]
    new_assitant_message, tokens_usage = _chat_completion(
        messages,
        temperature=temperature,
        seed=seed,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    messages.append({"role": "assistant", "content": new_assitant_message})
    return messages, new_assitant_message, tokens_usage
    
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

def is_new_optim_valid(prev_schedule, new_optim, llm_transfs):
    if new_optim.is_unrolling():
       
        if not unrollings_applied_once_in_branch(prev_schedule, new_optim, llm_transfs):
            return False
        elif has_unroll_factor_1(prev_schedule, new_optim, llm_transfs):
            return False
        
    elif new_optim.is_skewing():
        if not are_iterators_consecutive(new_optim.iterators):
            raise InvalidScheduleError(f'Skewing cannot be applied on non-consecutive loops (command "{llm_transfs[-1]}"). Skewing requires two adjacent loop levels (e.g. L1 and L2).')
        if not are_iterators_perfectly_nested(prev_schedule.tree, new_optim.iterators):
            raise InvalidScheduleError(f'Skewing cannot be applied on imperfectly nested loops (command "{llm_transfs[-1]}")')
        # check if solution found?

    elif new_optim.is_interchange():
        iterators = sorted(new_optim.params, key=lambda x:x[1])
        if not are_iterators_perfectly_nested(prev_schedule.tree, iterators):
            raise InvalidScheduleError(f'Interchange cannot be applied on imperfectly nested loops (command "{llm_transfs[-1]}")')

        if has_interchange_on_parallelized_loops(prev_schedule, new_optim, llm_transfs):
            return False

    elif new_optim.is_any_tiling():
        if not are_iterators_consecutive(new_optim.iterators):
            raise InvalidScheduleError(f'Tiling cannot be applied on non-consecutive loops (command "{llm_transfs[-1]}"). Tiling requires adjacent loop levels (e.g. Tile2D on L1 and L2).')

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
                raise InvalidScheduleError(f'{llm_transfs[-1]} cannot be applied after {llm_transfs[i]}: unrolling must be the last transformation applied to a computation. Move every Unroll(...) for these computations to the end of the schedule.')
    return True

def unrollings_applied_once_in_branch(prev_schedule, new_optim, llm_transfs):
    assert new_optim.is_unrolling()

    new_optim_comps = new_optim.comps
    
    for i,optim in enumerate(prev_schedule.optims_list):
        if optim.is_unrolling():
            unrolled_comps = optim.comps
            if any(comp in unrolled_comps for comp in new_optim_comps):
                raise InvalidScheduleError(f'Unrolling {llm_transfs[-1]} cannot be applied on already unrolled computations (previously unrolled by {llm_transfs[i]})')
    return True

def has_unroll_factor_1(prev_schedule, new_optim, llm_transfs):
    assert new_optim.is_unrolling()
    if new_optim.unrolling_factor==1:
        raise InvalidScheduleError(f'{llm_transfs[-1]}: cannot unroll a loop with a factor of 1.')
    return False



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
                raise InvalidScheduleError(f'Interchange {llm_transfs[-1]} cannot be applied after parallelization on the same loop (parallelized by {llm_transfs[i]}). You might consider reordering the transformations.')
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
                raise InvalidScheduleError(f'Tiling {llm_transfs[-1]} cannot be applied on levels above parallelized loops (parallelized by {llm_transfs[i]}). You might consider reordering the transformations.')
    return False

def validate_optim_and_add_to_schedule(prev_schedule, new_optim, llm_transfs):
    new_optim.initialize_action_for_tree(prev_schedule.tree)
    assert is_new_optim_valid(prev_schedule, new_optim, llm_transfs)
    prev_schedule.add_optimizations([new_optim])
    return prev_schedule

def extract_transformation_commands(message):
    if not message:
        return None
    match = re.search(r"<schedule>(.*?)</schedule>", message.replace('\n',''))

    if match:
        llm_sched_str = match.group(1)
        return llm_sched_str.replace(' ', '').replace('`','')
    else:
        return None
