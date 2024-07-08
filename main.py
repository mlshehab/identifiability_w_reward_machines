import numpy as np
from mdp import MDP, MDPRM
from reward_machine.reward_machine import RewardMachine
import scipy.linalg
import time 
from sample_mdps import sampleMDP1

def d(P):
    m,n = P.shape
    dP = np.zeros((m,m*n))
    for i in range(m):
        dP[i,n*i:n*(i+1)] = P[i]
    return dP

# t = np.array([[1,2,4],[3,2,4],[1,6,6],[1,2,0],[-1,-1,-1]])
# print(d(t))

def ppec(mdpRM: MDPRM):
    # Policy Preserving Equivalence Class
    mdp = mdpRM.mdp
    rm  =  mdpRM.rm
    L = mdpRM.L

    card_delta_u = rm.t_count

    print(f"Cardinality of delta_u is: {card_delta_u}")
    print(f"The reward machines has {rm.n_states} states ")
    # costruct Psi
    prodMDP = mdpRM.construct_product()
    P = prodMDP.P[0]
    E = np.eye(prodMDP.n_states)
    for a in range(1,prodMDP.n_actions):
        P = np.vstack((P,prodMDP.P[a]))
        E = np.vstack((E, np.eye(prodMDP.n_states)))
    
    Psi = d(P)
    # print(max(Psi@np.ones((Psi.shape[1],1))))
    # construct F
    row_F = (mdp.n_states**2)*(rm.n_states**2)*mdp.n_actions
    col_F = card_delta_u
    F = np.zeros(shape = (row_F,col_F))
    
    delta = rm.replace_values_with_positions()
    # print(delta)
    # print("The terminal state is : ", rm.terminal_states)

    
    for a in range(mdp.n_actions):
        for u in range(rm.n_states):
            for s in range(mdp.n_states):
                for u_prime in range(rm.n_states):
                    for s_prime in range(mdp.n_states):
                        # if not u in rm.terminal_states: 
                        try:
                            # if u == 1 and u_prime == 2:
                            #     print("YOL:", delta[u][u_prime])
                            F[s_prime + u_prime*mdp.n_states + s*mdp.n_states*rm.n_states + \
                                u*(mdp.n_states**2)*rm.n_states + a*(mdp.n_states**2)*(rm.n_states**2) , delta[u][u_prime] ] = 1
                        except KeyError:
                            continue
   
    # prune F and Psi
    keep_indices = ~np.all(Psi == 0, axis=0)
    F = F[keep_indices]
    Psi = Psi[:, keep_indices]
    
    print("The pruned F shape is: ", F.shape)
    print("The pruned Psi shape is: ", Psi.shape)
    start_time_prod = time.time()      
    prod = Psi@F
    end_time_prod = time.time()
    # print(prod)
    A = np.hstack((prod, -E + mdp.gamma*P))
    print("A shape: ", A.shape)
    print("A rank: ", np.linalg.matrix_rank(A, tol = 0.0001))

    kernel_A = scipy.linalg.null_space(A,rcond = 0.0001)
    P = kernel_A[:card_delta_u,:]

    # check that a constant vector of ones is in ran(P)
    print(f"The shape of P is {P.shape} and its rank is {np.linalg.matrix_rank(P,tol=0.0001)}")
    print("The projected matrix is: ", P)
    res = np.linalg.lstsq(P, np.ones((P.shape[0],1)), rcond=None)
    x = res[0]

    print("The norm is ", np.linalg.norm(P@x -np.ones((P.shape[0],1)) ))
    # assert np.linalg.lstsq(P, np.ones((P.shape[0],1)), rcond=None)[1] <= 1e-6 
    
    print("The projected matrix is: ", P)
    # print("The rank of the projection is:", np.linalg.matrix_rank(P,tol=0.0001))

    # compute product in closed form --> Section 3.7.1 of the write-up
    start_time_closed = time.time()
    test = np.zeros_like(prod)
    n_rows, n_cols = prod.shape
    for i in range(n_rows):
        (s,u,a) = mdpRM.sua_pair_from_i(i)
        for j in range(n_cols):
            # print(f"j = {j}")
            for s_prime in range(mdp.n_states):
                for u_prime in range(rm.n_states):
                    # print(f"u_prime= {u_prime}")
                    try:
                        if delta[u][u_prime] == j:
                            # print(f"delta[{u}][{u_prime}] = {delta[u][u_prime]} and j = {j} and i = {i}")
                            prod_mdp_s = mdpRM.s_from_su_pair((s,u))
                            prod_mdp_s_prime = mdpRM.s_from_su_pair((s_prime,u_prime))
                            test[i,j] += prodMDP.P[a][prod_mdp_s,prod_mdp_s_prime]
                    except KeyError:
                        continue
    
    
    
    end_time_closed = time.time()
    # print(f"Time taken by PsiF: {-start_time_prod + end_time_prod:.6f} seconds")
    # print(f"Time taken by Closed: {-start_time_closed + end_time_closed:.6f} seconds")
    



if __name__ == '__main__':


    P = sampleMDP1
    
    mdp = MDP(n_states=6,n_actions=4,P = P,gamma = 0.9,horizon=10)
    rm = RewardMachine("./rmc.txt")
    print(rm.delta_u)
    

    L = {}

    # TODO: specify what to do when a state doesn't have label

    L[0] = 'f'  # food
    L[1] = 'r1' # room 1
    L[2] = 'r2' # room 2
    L[3] = 'h'  # hallway
    L[4] = 'c'  # coffee
    L[5] = 'r3' # room 3

    mdpRM = MDPRM(mdp,rm,L)
    newMDP = mdpRM.construct_product()

    ppec(mdpRM)







