import re
import numpy as np
import copy
import pandas as pd
import sympy

def get_preorder_expression(e): # Retrun the expression in preorder as a list of expr dicts
    expression = [e]
    for c in e['children']:
        expression.extend(get_preorder_expression(c))
    return expression

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

# checks if an iterator is a dummy one (ie has an extent of 1)
def is_loop_dummy(iter_name, program_annot):
    return  program_annot['iterators'][iter_name]['upper_bound']=='1' and program_annot['iterators'][iter_name]['lower_bound']=='0'\

# checks if a computation is a dummy one (ie occurs a loop that has an extent of 1)
def is_comp_loopless(comp_name,program_annot):
    root_iter = program_annot['computations'][comp_name]['iterators'][0]
    return is_loop_dummy(root_iter,program_annot)

# A constraint matrix is the set of linear inequalities that describes the iteration domain.
# Example:
# if the iteration domain D is the follwoing
#     {i >= 0
#      i < 128
# D =  j >= 0
#      j < 32
#      k >= 0
#      k < 64}
# The iterator vector is 
# x = [i, 
#      j, 
#      k]
# The coeffcients matrix A would be:
#     [-1,   0,   0,
#       1,   0,   0,
# A=    0,  -1,   0,
#       0,   1,   0,
#       0,   0,  -1,
#       0,   0,   1]
# The second hand side of the equation b (constants vector) is the vector:
#     b= [0,
#         127,
#         0,
#         31,
#         0,
#         63]
# Since:
#    D = Ax<=b
# Get the matrix describing the initial constraints for this program
# EDIT: Padding is done with zeros, this idea is no longer in use. ignore the following comment. --- For consistency, The padding of the coefficients part is done by adding -1,1 in a diagonal-ish manner. The constants vector is padded with zeros
def get_padded_initial_iteration_domain(program_json,comp_name, pad=True):
    #supports bounds of type ± i ± j ±...± cst and max(cst, iter)
    comp_dict = program_json['computations'][comp_name] 
    nb_dims = len(comp_dict['iterators'])
    
    coeff_mat = np.zeros((nb_dims*2,nb_dims),int)
    constants_col = np.zeros((nb_dims*2),int)
    
    for i,iterator_name_row in enumerate(comp_dict['iterators']):# rows loop
        
        # set the diagonal-ish values
        coeff_mat[i*2,i]=-1
        coeff_mat[i*2+1,i]=1
        
        iterator_row_dict = program_json['iterators'][iterator_name_row]
        
        #get the simplified expression of the bounds
        upper_bound = str(sympy.simplify(iterator_row_dict['upper_bound'])).replace(' ','')
        lower_bound = str(sympy.simplify(iterator_row_dict['lower_bound'])).replace(' ','')
        
        # in some special cases, the lower bound of i appears as max(cst, iter) when iter<=i. The following is a non-general solution
        if 'max' in lower_bound or 'Max' in lower_bound:
            lower_bound = re.findall(r'[mM]ax\(.+,(.+)\)',lower_bound)[0]
        
        # find the iterator names and constants used in the bounds
        iterators_in_upper = re.findall(r'[a-zA-Z]\w*', upper_bound)
        constants_in_upper = re.findall(r'(?:^|\+|-)\d+', upper_bound)
        iterators_in_lower = re.findall(r'[a-zA-Z]\w*', lower_bound)
        constants_in_lower = re.findall(r'(?:^|\+|-)\d+', lower_bound)
        
        #if no constants used, set to 0
        if not constants_in_upper:
            constants_in_upper = [0]
        else:
            assert len(constants_in_upper)==1
       
        if not constants_in_lower:
            constants_in_lower = [0]
        else:
            assert len(constants_in_lower)==1
        
        #for each iterator in the bounds expression, set the corresponding values in the matrix
        for iter_name in iterators_in_upper:
            if not iter_name in comp_dict['iterators']: # if the iterator has been renamed due to confilict, find the new name using the iter_name as a prefix
                matches = list(filter(lambda x: x.startswith(iter_name), comp_dict['iterators']))
                assert len(matches)==1 # there has to be one match only, otherwise it means that multiple iterators have the same prefix
                iter_name = matches[0]
            col_idx = comp_dict['iterators'].index(iter_name)
            if '-'+iter_name in upper_bound:
                coeff_mat[i*2+1,col_idx]=1
            else:
                coeff_mat[i*2+1,col_idx]=-1
        constants_col[i*2+1]= int(constants_in_upper[0])-1 #adding a -1 because we are representing a non-strict inequality
        
        for iter_name in iterators_in_lower:
            if not iter_name in comp_dict['iterators']: # if the iterator has been renamed due to confilict, find the new name using the iter_name as a prefix
                matches = list(filter(lambda x: x.startswith(iter_name), comp_dict['iterators']))
                assert len(matches)==1 # there has to be one match only, otherwise it means that multiple iterators have the same prefix
                iter_name = matches[0]
            col_idx = comp_dict['iterators'].index(iter_name)
            if '-'+iter_name in lower_bound:
                coeff_mat[i*2,col_idx]=-1
            else:
                coeff_mat[i*2,col_idx]=+1
        constants_col[i*2]= -int(constants_in_lower[0])
                
#     constants_col = constants_col.reshape(-1,1)
    
    
    # Add padding if requested
    if pad:
        padded_coeff_mat = np.pad(coeff_mat, [(0,MAX_DEPTH*2-nb_dims*2),(0,MAX_DEPTH-nb_dims)], mode='constant', constant_values=0)
        
#         #Edit: this idea has been dropped.!! For consistency, The padding of the coefficients part is done by adding -1,1 in a diagonal-ish manner
#         for i in range(nb_dims,MAX_DEPTH):
#             padded_coeff_mat[i*2,i]=-1
#             padded_coeff_mat[i*2+1,i]=1
            
        padded_constants_col = np.pad(constants_col, [(0,MAX_DEPTH*2-nb_dims*2)], mode='constant', constant_values=0)
        return padded_coeff_mat, padded_constants_col
    else:
        return coeff_mat, constants_col

def get_loop_depth(program_annots, iterator_name):
    loop_depth=0
    it = iterator_name
    while program_annots['iterators'][it]['parent_iterator']!=None:
        it = program_annots['iterators'][it]['parent_iterator']
        loop_depth+=1
    return loop_depth

def get_parents_chain(program_annots, iterator_name):
    it = iterator_name
    chain = [it]
    while program_annots['iterators'][it]['parent_iterator']!=None:
        it = program_annots['iterators'][it]['parent_iterator']
        chain.append(it)
    return chain[::-1]
def get_padded_initial_iteration_domain_rows(program_json, iterator_name, pad=True):
    #supports bounds of type ± i ± j ±...± cst and max(cst, iter)
    # comp_dict = program_json['computations'][comp_name] 
    # nb_dims = len(comp_dict['iterators'])

    loop_depth = get_loop_depth(program_json, iterator_name)
    parents_chain = get_parents_chain(program_json, iterator_name)
    coeff_mat = np.zeros((2,loop_depth+1),int)
    constants_col = np.zeros((2),int)
    
    # for i,iterator_name_row in enumerate([iterator_name]):# rows loop
    iterator_name_row = iterator_name
        
    # set the diagonal-ish values
    coeff_mat[0,loop_depth]=-1
    coeff_mat[1,loop_depth]=1
    
    iterator_row_dict = program_json['iterators'][iterator_name_row]
    
    #get the simplified expression of the bounds
    upper_bound = str(sympy.simplify(iterator_row_dict['upper_bound'])).replace(' ','')
    lower_bound = str(sympy.simplify(iterator_row_dict['lower_bound'])).replace(' ','')
    
    # in some special cases, the lower bound of i appears as max(cst, iter) when iter<=i. The following is a non-general solution
    if 'max' in lower_bound or 'Max' in lower_bound:
        lower_bound = re.findall(r'[mM]ax\(.+,(.+)\)',lower_bound)[0]
    
    # find the iterator names and constants used in the bounds
    iterators_in_upper = re.findall(r'[a-zA-Z]\w*', upper_bound)
    constants_in_upper = re.findall(r'(?:^|\+|-)\d+', upper_bound)
    iterators_in_lower = re.findall(r'[a-zA-Z]\w*', lower_bound)
    constants_in_lower = re.findall(r'(?:^|\+|-)\d+', lower_bound)
    
    #if no constants used, set to 0
    if not constants_in_upper:
        constants_in_upper = [0]
    else:
        assert len(constants_in_upper)==1
   
    if not constants_in_lower:
        constants_in_lower = [0]
    else:
        assert len(constants_in_lower)==1
    
    #for each iterator in the bounds expression, set the corresponding values in the matrix
    for iter_name in iterators_in_upper:
        if not iter_name in parents_chain: # if the iterator has been renamed due to confilict, find the new name using the iter_name as a prefix
            matches = list(filter(lambda x: x.startswith(iter_name), parents_chain))
            assert len(matches)==1 # there has to be one match only, otherwise it means that multiple iterators have the same prefix
            iter_name = matches[0]
                
        col_idx = parents_chain.index(iter_name)
        if '-'+iter_name in upper_bound:
            coeff_mat[1,col_idx]=1
        else:
            coeff_mat[1,col_idx]=-1
    constants_col[1]= int(constants_in_upper[0])-1 #adding a -1 because we are representing a non-strict inequality
    
    for iter_name in iterators_in_lower:
        if not iter_name in parents_chain: # if the iterator has been renamed due to confilict, find the new name using the iter_name as a prefix
            matches = list(filter(lambda x: x.startswith(iter_name), parents_chain))
            assert len(matches)==1 # there has to be one match only, otherwise it means that multiple iterators have the same prefix
            iter_name = matches[0]
        col_idx = parents_chain.index(iter_name)
        if '-'+iter_name in lower_bound:
            coeff_mat[0,col_idx]=-1
        else:
            coeff_mat[0,col_idx]=+1
    constants_col[0]= -int(constants_in_lower[0])
                
#     constants_col = constants_col.reshape(-1,1)
    
    
    # Add padding if requested
    if pad:
        padded_coeff_mat = np.pad(coeff_mat, [(0,0),(0,MAX_DEPTH-(loop_depth+1))], mode='constant', constant_values=0)
        
#         #Edit: this idea has been dropped.!! For consistency, The padding of the coefficients part is done by adding -1,1 in a diagonal-ish manner
#         for i in range(nb_dims,MAX_DEPTH):
#             padded_coeff_mat[i*2,i]=-1
#             padded_coeff_mat[i*2+1,i]=1
            
        # padded_constants_col = np.pad(constants_col, [(0,MAX_DEPTH*2-nb_dims*2)], mode='constant', constant_values=0)
        padded_constants_col = constants_col
        return padded_coeff_mat, padded_constants_col
    else:
        return coeff_mat, constants_col
    
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
        print("error:",comp_or_loop_name,"is not a computation name nor a loop name")
        return None
        
    pre_order_traversal = [{'name':comp_or_loop_name, 'type':type_, 'depth':depth, 'nb_children':len(children_names)}]
    # print(comp_or_loop_name,pre_order_traversal)

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
        print("error:",comp_or_loop_name,"is not a computation name nor a loop name")
        return None    
    
def get_access_matrix_form_access_str(access_str, comp_iterators):
    """
    Extract access matrix from access string.
    :param access_str: can be in the form of "bufXX(i1 , i1 + 2*j2, k + 10)" or "(i1 , i1 + 2*j2, k + 10)"
    :param comp_iterators: a list of iterator names where the access occurs
    :return: Access matrix as list of lists
    """
    access_strs = re.findall(r'\(([L\d,\-+*]+)\)',access_str)
    assert len(access_strs) == 1 #to make sure there is only one access in the str
    access_str = access_strs[0]
    per_dim_strs = access_str.split(',')
    access_matrix = [extract_affine_coefficients(str(simplify(per_dim_str)), comp_iterators) for per_dim_str in per_dim_strs]
    return access_matrix

# Recursive function to parse the tokens
def parse_expression(tokens,operators):
    if not tokens:
        return ""

    # Extract the current token
    token = tokens.pop(0)

    # Check if the token is an operator
    if token in operators:
        operator, arity = operators[token]

        if arity == 1:  # Unary operator
            operand = parse_expression(tokens,operators)
            if token.startswith('sqrt'):
                return f"expr(o_sqrt, {operand})"
            elif token == 'cast':
                return f"cast(#dtype#, {operand})"

        elif arity == 2:  # Binary operator
            operand1 = parse_expression(tokens,operators)
            operand2 = parse_expression(tokens,operators)

            if token in ['max', 'min', 'eq', 'le']:
                # Special case for max and min
                return f"expr(o_{token}, {operand1}, {operand2})"
            else:
                # Wrap in parentheses for proper order of operations
                return f"({operand1} {operator} {operand2})"

    # If it's not an operator, it's a function call or variable
    return token  # Return the operand as it is (e.g., buf_s0(L0))


def convert_to_inorder(expression):
    # Define the operator precedence and their behavior (unary or binary)
    operators = {
        'add': ('+', 2),  # Binary
        'sub': ('-', 2),  # Binary
        'mul': ('*', 2),  # Binary
        'div': ('/', 2),  # Binary
        'eq': ('eq', 2),  # Binary
        'le': ('le', 2),  # Binary
        'max': ('max', 2),  # Binary (with special formatting)
        'min': ('min', 2),  # Binary (with special formatting)
        'cast': ('cast', 1),  # Unary (with special formatting)
        'sqrt': ('sqrt', 1),  # Unary (with special formatting)
        'sqrt_f32': ('sqrt', 1),  # Unary (with special formatting)
        'sqrt_f64': ('sqrt', 1)  # Unary (with special formatting)
    }
    # Split the expression into tokens
    tokens = expression.split()
    return parse_expression(tokens,operators)

# def get_access_str_form_access_matrix(access_matrix, comp_iterator_names):
#     pass # reimplemented
#     """
#     Extract access relation str (eg "(i1 , i1 + 2*j2, k + 10)") from access matrix.
#     :param access_matrix: as a list of lists
#     :param comp_iterator_names: a list of iterator names where the access occurs
#     :return: access relation str with parentheses
#     """

#     dim_accesses = []
#     for row in access_matrix:
#         dim_acc = ''
#         for col, e in enumerate(row):
#             if e != 0:
#                 if col != len(row) - 1:
#                     dim_acc += ('+' if dim_acc != '' else '') + (str(e) + '*' if e != 1 else '') + \
#                                comp_iterator_names[col]
#                 else:
#                     dim_acc += ('+' if e > 0 else '') + str(int(e))

#         dim_accesses.append(dim_acc if dim_acc != '' else '0')
#     return '(' + ','.join(dim_accesses) + ')'

def is_int(s):
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True

def is_float(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True
    
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