import numpy as np
from z3 import Bool, Solver, Implies, Not, BoolRef, sat,print_matrix, Or, And, AtMost # type: ignore
from itertools import product

def ExactlyOne(vars):
    """Ensure exactly one of the variables in the list is True."""
    # At least one must be True
    at_least_one = Or(vars)
    # At most one must be True
    at_most_one = AtMost(*vars, 1)
    # Both conditions must be satisfied
    return And(at_least_one, at_most_one)

# adjacent = {0:[1,2], 1:[0,2,3],2:[0,1,4], 3:{1,4}, 4:{2,3} }
# n_nodes = 5
# n_colors = 3

# # we have a variable for each color for each node
# # x_i_j is the i-th color of the j-th node
# X = [[Bool('x_%s_%s'%(i,j)) for i in range(n_colors)] for j in range(n_nodes)] 
# print("X = ",X)
# s = Solver() # type: ignore

# # Each node can have only one color
# # if x_ij = 1, then x_ij = 0 for all other j's
# for i in range(n_nodes):
#     s.add(ExactlyOne([X[i][j] for j in range(n_colors)]))

# # Distinct colors
# for i in range(n_nodes):
#     for j in adjacent[i]:
#         for c in range(n_colors):
#             s.add(Implies(X[i][c], Not(X[j][c]) ))
                  
# s.check()
# if s.check() == sat:
#     print("Yup!")
# m = s.model()

# r = [[ m.evaluate(X[i][j]) for j in range(n_colors)] for i in range(n_nodes)]
# print_matrix(r)



def one_entry_per_row(B):
    cond = []
    for i in range(kappa):
        cond+= [ExactlyOne([B[i][j] for j in range(kappa)])]
    return cond

def boolean_matrix_vector_multiplication(A,b):
    # def boolean_matrix_vector_multiplication(matrix, vector):
    # Number of rows in matrix
    num_rows = len(A)
    # Number of columns in matrix (assuming non-empty matrix)
    num_cols = len(A[0])
    # print(f"The numerb of cols is {num_cols}")
    # Ensure the vector size matches the number of columns in the matrix
    assert len(b) == num_cols

    # Resulting vector after multiplication
    result = []

    # Perform multiplication
    for i in range(num_rows):
        # For each row in the matrix, compute the result using AND/OR operations
        # result_i = OR(AND(matrix[i][j], vector[j]) for all j)
        row_result = Or([And(A[i][j], b[j]) for j in range(num_cols)])
        result.append(row_result)
    
    return result


# Function for matrix-matrix boolean multiplication
def boolean_matrix_matrix_multiplication(A, B):
    # Number of rows in matrix A and columns in matrix B
    num_rows_A = len(A)
    num_cols_B = len(B[0])
    
    # Number of columns in A and rows in B (must match for matrix multiplication)
    num_cols_A = len(A[0])
    num_rows_B = len(B)
    assert num_cols_A == num_rows_B, "The number of columns in A must equal the number of rows in B."
    
    # Resulting matrix after multiplication
    result = [[None for _ in range(num_cols_B)] for _ in range(num_rows_A)]

    # Perform multiplication
    for i in range(num_rows_A):
        for j in range(num_cols_B):
            # Compute C[i][j] = OR(AND(A[i][k], B[k][j]) for all k)
            result[i][j] = Or([And(A[i][k], B[k][j]) for k in range(num_cols_A)])
    
    return result


def transpose_boolean_matrix(matrix):
    # Number of rows and columns in the input matrix
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Initialize the transposed matrix
    transposed = [[None for _ in range(num_rows)] for _ in range(num_cols)]

    # Transpose operation: Swap rows and columns
    for i in range(num_rows):
        for j in range(num_cols):
            transposed[j][i] = matrix[i][j]
    
    return transposed

# Function to compute the kth power of a boolean matrix
def boolean_matrix_power(matrix, k):
    # Get the size of the matrix (assuming square matrix)
    n = len(matrix)
    assert all(len(row) == n for row in matrix), "The input matrix must be square."

    # Initialize the result matrix as the input matrix (A^1)
    
    result = matrix
    if k == 0 or k == 1:
        return result

    # Multiply the matrix by itself k-1 times
    for _ in range(k - 1):
        result = boolean_matrix_matrix_multiplication(result, matrix)
    
    return result

# Function to compute the element-wise OR of a list of boolean matrices
def element_wise_or_boolean_matrices(matrices):
    # Ensure there is at least one matrix
    assert len(matrices) > 0, "There must be at least one matrix in the list."

    # Get the number of rows and columns from the first matrix
    num_rows = len(matrices[0])
    num_cols = len(matrices[0][0])

    # Ensure all matrices have the same dimensions
    for matrix in matrices:
        assert len(matrix) == num_rows and all(len(row) == num_cols for row in matrix), "All matrices must have the same dimensions."

    # Initialize the result matrix
    result = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    # Compute the element-wise OR for each element
    for i in range(num_rows):
        for j in range(num_cols):
            # OR all matrices at position (i, j)
            result[i][j] = Or([matrix[i][j] for matrix in matrices])
    
    return result

# Function to compute the element-wise OR of a list of boolean vectors
def element_wise_or_boolean_vectors(vectors):
    # Ensure there is at least one vector
    assert len(vectors) > 0, "There must be at least one vector in the list."

    # Get the length of the first vector
    vector_length = len(vectors[0])

    # Ensure all vectors have the same length
    for vector in vectors:
        assert len(vector) == vector_length, "All vectors must have the same length."

    # Initialize the result vector
    result = [None] * vector_length

    # Compute the element-wise OR for each element
    for i in range(vector_length):
        # OR all vectors at position i
        result[i] = Or([vector[i] for vector in vectors])
    
    return result

def bool_matrix_mult_from_indices(B,indices, x):
    # indices = [l0, l1 , l2 ,... , l_k]
    # Get the number of rows and columns from the first matrix
    num_rows = len(B[0])
    num_cols = len(B[0][0])
    # print(f"The B[0] matrix is of shape {num_rows} by {num_cols}")

    len_trace = len(indices)

    result = transpose_boolean_matrix(B[indices[0]])
    
    i = 0
    for i in range(1,len_trace):
        # print(f"i = {i}, len_Trace = {len_trace}")
        result = boolean_matrix_matrix_multiplication(transpose_boolean_matrix(B[indices[i]]), result)
        
    return boolean_matrix_vector_multiplication(result,x)

def element_wise_and_boolean_vectors(vector1, vector2):
    # Ensure both vectors have the same length
    assert len(vector1) == len(vector2), "Both vectors must have the same length."

    # Initialize the result vector
    result = [None] * len(vector1)

    # Compute the element-wise AND for each element
    for i in range(len(vector1)):
        # AND the corresponding elements from both vectors
        result[i] = And(vector1[i], vector2[i])
    
    return result

if __name__ == '__main__':
    kappa = 3
    AP = 3

    B = [[[Bool('x_%s_%s_%s'%(i,j,k) )for j in range(kappa)]for i in range(kappa)]for k in range(AP)]

    B_ = element_wise_or_boolean_matrices([b_k for b_k in B])
    x = [False]*kappa
    x[0] = True
    print(f"x = {x}")
    # for i, row in enumerate(B_):
    #     print(f"Row {i}: {[str(cell) for cell in row]}")
    
   
    B_T = transpose_boolean_matrix(B_)
    # for i, row in enumerate(B_T):
    #     print(f"B_T: Row {i}: {[str(cell) for cell in row]}")
    # print(boolean_matrix_vector_multiplication(boolean_matrix_power(B_T,2),x))

    powers_B_T = [boolean_matrix_power(B_T,k) for k in  range(1,kappa)]
    
    powers_B_T_x = [boolean_matrix_vector_multiplication(B,x) for B in powers_B_T]
    
    powers_B_T_x.insert(0, x)
    
    # print(powers_B_T_x[0])
    OR_powers_B_T_x = element_wise_or_boolean_vectors(powers_B_T_x)
    # print(OR_powers_B_T_x)
    s = Solver() # type: ignore

    # C1 and C2 from Notion Write-up
    for k in range(AP):
        s.add(one_entry_per_row(B[k]))

    # C3 from from Notion Write-up
    for element in OR_powers_B_T_x:
        s.add(element)
    

    proposition2index = {'A':0, 'B':1 , 'C':2}

    def prefix2indices(s):
        out = []
        for l in s:
            out.append(proposition2index[l])
        return out

    n_counter = 3

    counter_examples = {}
    for ce in range(n_counter):
        if ce not in counter_examples.keys():
            counter_examples[ce] = []
    
    # I will hard code the counter examples for
    # SET1 = ['A', 'AA', 'AAA', 'BA', 'BAA', 'BBA']
    # SET2 = ['B', 'BB', 'BBB','ABB']
    # SET3 = ['AB','AAB','BAB']
    # SET4 = ['ABA']
    

    SET1 = ['A', 'AA', 'BA','CA','AAA','ACA','BAA', 'BBA','BCA','CAA','CBA','CCA']
    SET2 = ['B','BB','CB','BBB','BCB', 'CBB', 'CCB']
    SET3 = ['C','BC', 'CC','ABC', 'BBC', 'BCC', 'CBC', 'CCC', 'ABAC','ABBC']
    SET4 = ['AB', 'CAB', 'ACB' , 'AAB', 'ABB', 'BAB','ABAB','ABBB']
    SET5 = ['AC', 'AAC', 'ACC', 'BAC', 'CAC']
    SET6 = ['ABA','ABAA','ABBA']

    counter_examples[0] = list(product(SET1, SET6))
    counter_examples[1] = list(product(SET3, SET5))
    counter_examples[2] = list(product(SET2, SET4))
    # counter_examples[3] = list(product(SET2, SET3))
    # print(f"The counter examples are: {counter_examples}")
    # C4 from from Notion Write-up 
    for state in range(n_counter):
        ce_set = counter_examples[state]
        # for each counter example in this set, add the correspodning constraint
        for ce in ce_set:
            p1 = prefix2indices(ce[0])
            p2 = prefix2indices(ce[1])

            # Now
            sub_B1 = bool_matrix_mult_from_indices(B,p1, x)
            sub_B2 = bool_matrix_mult_from_indices(B,p2, x)

            res_ = element_wise_and_boolean_vectors(sub_B1, sub_B2)

            for elt in res_:
                s.add(Not(elt))

    # s.add(B[0][2][2])
    # no
   

    s.check()
    if s.check() == sat:
        print("Yup!")
    else:
        print("NOT SAT")
    m = s.model()

    for ap in range(AP):
        r = [[ m.evaluate(B[ap][i][j]) for j in range(kappa)] for i in range(kappa)]
        print_matrix(r)

    # # print(B[0])