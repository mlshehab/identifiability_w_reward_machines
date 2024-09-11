import numpy as np
from z3 import Bool, Solver, Implies, Not, BoolRef, sat,print_matrix, Or, And, AtMost # type: ignore

def ExactlyOne(vars):
    """Ensure exactly one of the variables in the list is True."""
    # At least one must be True
    at_least_one = Or(vars)
    # At most one must be True
    at_most_one = AtMost(*vars, 1)
    # Both conditions must be satisfied
    return And(at_least_one, at_most_one)

adjacent = {0:[1,2], 1:[0,2,3],2:[0,1,4], 3:{1,4}, 4:{2,3} }
n_nodes = 5
n_colors = 3

# we have a variable for each color for each node
# x_i_j is the i-th color of the j-th node
X = [[Bool('x_%s_%s'%(i,j)) for i in range(n_colors)] for j in range(n_nodes)] 
print("X = ",X)
s = Solver() # type: ignore

# Each node can have only one color
# if x_ij = 1, then x_ij = 0 for all other j's
for i in range(n_nodes):
    s.add(ExactlyOne([X[i][j] for j in range(n_colors)]))

# Distinct colors
for i in range(n_nodes):
    for j in adjacent[i]:
        for c in range(n_colors):
            s.add(Implies(X[i][c], Not(X[j][c]) ))
                  
s.check()
if s.check() == sat:
    print("Yup!")
m = s.model()

r = [[ m.evaluate(X[i][j]) for j in range(n_colors)] for i in range(n_nodes)]
print_matrix(r)

kappa = 3
AP = 2

B = [[[Bool('x_%s_%s_%s'%(i,j,k) )for j in range(kappa)]for i in range(kappa)]for k in range(AP)]

print(B[0])