from SAT import boolean_matrix_matrix_multiplication
from SAT import boolean_matrix_vector_multiplication
from SAT import transpose_boolean_matrix
from SAT import boolean_matrix_power
from SAT import element_wise_or_boolean_matrices
from z3 import Bool
# Example usage
# Define matrix A (3x2) as a list of lists of Booleans
A = [
    [Bool('a11'), Bool('a12')],
    [Bool('a21'), Bool('a22')],
    [Bool('a31'), Bool('a32')]
]

# Define matrix B (2x2) as a list of lists of Booleans
B = [
    [False, Bool('b12')],
    [Bool('b21'), Bool('b22')]
]

# Perform boolean matrix-matrix multiplication
C = boolean_matrix_matrix_multiplication(A, B)

# Display the result
print("\nResult of Matrix Matrix multiplication ...")
for i, row in enumerate(C):
    print(f"Row {i}: {[str(cell) for cell in row]}")



A = [
        [Bool('a11'), Bool('a12')],
        [Bool('a21'), Bool('a22')],
        [Bool('a31'), Bool('a32')]
    ]

# Define vector x as a list of Booleans
x = [Bool('x1'), Bool('x2')]

# Perform boolean matrix-vector multiplication
y = boolean_matrix_vector_multiplication(A, x)


print("\nResult of Matrix vector multiplication ...")
# Display the result
for i, yi in enumerate(y):
    print(f"y[{i}] = {yi}")    


# Example usage
# Define a matrix A (3x2) as a list of lists of Booleans
A = [
    [Bool('a11'), Bool('a12')],
    [Bool('a21'), Bool('a22')],
    [Bool('a31'), Bool('a32')]
]

# Get the transpose of matrix A
A_transposed = transpose_boolean_matrix(A)
print("\nResult of Matrix transpose ...")
# Display the result
for i, row in enumerate(A_transposed):
    print(f"Row {i}: {[str(cell) for cell in row]}")




# Example usage
# Define a matrix A (3x3) as a list of lists of Booleans
A = [
    [Bool('a11'), Bool('a12'), Bool('a13')],
    [Bool('a21'), Bool('a22'), Bool('a23')],
    [Bool('a31'), Bool('a32'), Bool('a33')]
]

# Compute A^k where k = 2
k = 2
A_k = boolean_matrix_power(A, k)

# Display the result
print("\nResult of Boolean Matrix Power ...")
for i, row in enumerate(A_k):
    print(f"Row {i}: {[str(cell) for cell in row]}")




# Example usage
# Define matrix A (2x2)
A = [
    [Bool('a11')],
    [Bool('a21')]
]

# Define matrix B (2x2)
B = [
    [Bool('b11')],
    [Bool('b21')]
]

# Define matrix C (2x2)
C = [
    [Bool('c11')],
    [Bool('c21')]
]

# List of matrices
matrices = [A, B, C]

# Compute element-wise OR
result = element_wise_or_boolean_matrices(matrices)
print("\nResult of Boolean Matrix ELT Or ...")
# Display the result
for i, row in enumerate(result):
    print(f"Row {i}: {[str(cell) for cell in row]}")