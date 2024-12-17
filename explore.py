import numpy as np
from mdp import MDP, MDPRM
from reward_machine.reward_machine import RewardMachine
import scipy.linalg
import time 
from sample_mdps import sampleMDP1
from scipy.special import softmax, logsumexp
from GridWorld import BasicGridWorld
from plot import plot_arrows
import pprint
import matplotlib.pyplot as plt

class Node:
    def __init__(self, label,state, u, policy, is_root = False):
        self.label = label
        self.parent = None
        self.state = state
        self.u = u
        self.policy = policy
        self.children = []
        self.is_root = is_root

    def __str__(self):
        return f"Node with ({self.label},{self.state}) and Parent's label is {self.parent.label}"
    
    # Method to add a child to the node
    def add_child(self, child_node):
        child_node.parent = self  # Set the parent of the child node
        self.children.append(child_node)  # Add the child node to the children list

    # Method to get the parent of the node
    def get_parent(self):
        return self.parent

    # Method to get the children of the node
    def get_children(self):
        return self.children
    

def get_future_states(s, mdp):
    P = mdp.P
    post_state = []
    for Pa in P:
        for index in np.argwhere(Pa[s]> 0.0):
            post_state.append(index[0])     
    return list(set(post_state))

def get_future_states_action(s,a, mdp):
    Pa = mdp.P[a]
    post_state = []
    
    for index in np.argwhere(Pa[s]> 0.0):
        post_state.append(index[0])   

    return list(set(post_state))


def u_from_obs(obs_str, mdpRM):
    # obs_traj : 'l0l1l2l3 ...'

    mdp = mdpRM.mdp
    rm  =  mdpRM.rm
    L = mdpRM.L

    # Given the observation trajectory, find the current reward machine state

    u0 = rm.u0
    current_u = u0
    T = len(obs_str)
    obs_traj = []
    for l in obs_str:
        obs_traj.append(l)

    for i in range(T):
        current_u = rm._compute_next_state(current_u,obs_traj[i])

    return current_u


def infinite_horizon_soft_bellman_iteration(mdprm:MDPRM, reward,  tol = 1e-4, logging = True, log_iter = 5, policy_test_iter = 20):
    
    MDP = mdprm.construct_product()

    gamma = MDP.gamma
    n_actions = MDP.n_actions
    n_states = MDP.n_states

    v_soft = np.zeros((n_states,1)) # value functions
    q_soft = np.zeros((n_states, n_actions))

    delta = np.inf 

    converged = delta < tol

    it = 0
    total_time = 0.0

    while not converged:
        
        it+=1

        start_time = time.time()

        for state in range(n_states): 
            for action in range(n_actions):

                p_ns = MDP.P[action][state]

                future_value_soft = 0.0
      

                for i in range(len(p_ns)):
                    future_value_soft += p_ns[i]*reward[state][action][i] + gamma*p_ns[i]*v_soft[i]

                q_soft[state,action] =   future_value_soft
                

        v_new_soft = logsumexp(q_soft,axis = 1)

        end_time = time.time()
        total_time += end_time - start_time

        if logging and not it%log_iter and it >1 :
            print(f"Total: {it} iterations -- iter time: {end_time - start_time:.2f} sec -- Total time: {total_time:.2f}")
            print(f"Soft Error Norm ||e||: {np.linalg.norm(v_new_soft -v_soft):.6f}")
        
      
        converged = np.linalg.norm(v_new_soft - v_soft) <= tol

        v_soft = v_new_soft
    

    # find the policy
    soft_policy  = softmax(q_soft,axis = 1)


    return q_soft,v_soft , soft_policy




if __name__ == '__main__':


    grid_size = 3
    wind = 0.1
    discount = 0.9
    horizon =   13   
    start_state = 4 
    feature_dim = 2

    theta = 5*np.ones((2,1))
    theta[1] += 5
    p1,p2 = 0,24
    feature_type = 'None'

    gw = BasicGridWorld(grid_size,wind,discount,horizon,start_state, \
                         feature_dim, p1,p2,theta, feature_type)
    
    n_states = gw.n_states
    n_actions = gw.n_actions

    P = []

    for a in range(n_actions):
        P.append(gw.transition_probability[:,a,:])

    mdp = MDP(n_states=n_states, n_actions=n_actions,P = P,gamma = gw.discount,horizon=10)
    
   
    rm = RewardMachine("./rm_examples/patrol_adv.txt")

    policy = {}
    for rms in range(rm.n_states):
        policy[rms] = f"p{rms}"
    # print("The policy is: ", policy)
    # print(rm.delta_u)
    

    # # The grid numbering and labeling is :
    # # 0 5 10 15 20     D D H C C
    # # 1 6 11 16 21     D D H C C
    # # 2 7 12 17 22     H H H H H
    # # 3 8 13 18 23     A A H B B
    # # 4 9 14 19 24     A A H B B 


    # L = {}

    # L[3], L[4], L[8], L[9]     = 'A', 'A', 'A', 'A'
    # L[18], L[19], L[23], L[24] = 'B', 'B', 'B', 'B'
    # L[20], L[21], L[15], L[16]   = 'C', 'C', 'C', 'C'
    # L[0], L[1], L[5], L[6]     = 'D', 'D', 'D', 'D'
    # L[2], L[7], L[10],L[11],L[12],L[13],L[14],L[17], L[22] = 'H','H','H','H','H','H','H','H','H'

    # The grid numbering and labeling is :
    # 0 3 6     D H C 
    # 1 4 7     H H H 
    # 2 5 8     A H B
    

    L = {}

    L[2]     = 'A'
    L[8] = 'B'
    L[6]= 'C'
    L[0]     = 'D'
    L[3], L[4], L[5],L[1],L[7] = 'H','H','H','H','H'

    mdpRM = MDPRM(mdp,rm,L)
    mdp_ =  mdpRM.construct_product()
    # now we need a state action state reward for the product MDP
    reward = np.zeros((mdp_.n_states, mdp_.n_actions, mdp_.n_states))
    print(f"Reward: {reward.shape}, S: {mdp.n_states}, A: {mdp.n_actions}, RM: {rm.n_states}")

    for bar_s in range(mdp_.n_states):
        for a in range(mdp_.n_actions):
            for bar_s_prime in range(mdp_.n_states):
                (s,u) = mdpRM.su_pair_from_s(bar_s)
                (s_prime,u_prime) = mdpRM.su_pair_from_s(bar_s_prime)

                is_possible = mdp_.P[a][bar_s][bar_s_prime] > 0.0

                if u == 4 and u_prime == 0 and L[s_prime] == 'D' and is_possible:

                    reward[bar_s, a, bar_s_prime] = 10.0



    q_soft,v_soft , soft_policy = infinite_horizon_soft_bellman_iteration(mdpRM,reward,logging = False)
    # print(f"The soft policy is of shape: {soft_policy.shape}")
    
    def get_matrix_from_policy(pi,gw):
        m,n = pi.shape
        m = gw.grid_size
        A = np.zeros((m,m))
        values = np.zeros((m,m))

        for i,row in enumerate(pi):
            (x,y) = gw.int_to_point(i)
            # Find the maximum value in the array
            max_value = np.max(row)
            A[x][y] = np.argmax(row)
            values[x][y] = max_value
        return A , values
    

    # Main function to generate subplots
    def plot_policies(soft_policy, mdp, gw, discount):
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))  # 3x2 grid
        axes = axes.flatten()  # Flatten to 1D array for easy indexing
        for u_des in range(5):
            p1 = soft_policy[u_des * mdp.n_states:(u_des + 1) * mdp.n_states, :]
            A, values = get_matrix_from_policy(p1, gw)
            print(f"The policy for u_des = {u_des} is: {A}")
            plot_arrows(A, values, f'u_{u_des}', discount, ax=axes[u_des])

        plt.tight_layout()
        plt.show()

    plot_policies(soft_policy, mdp, gw, discount)

    # u_des = 0
    # p1 = soft_policy[u_des*mdp.n_states:(u_des+1)*mdp.n_states,:]
    # A,values = get_matrix_from_policy(p1,gw)
    # print(f"The policy is: {A}")
    # plot_arrows(A,values, f'u_{u_des}', discount)

    # u_des = 1
    # p1 = soft_policy[u_des*mdp.n_states:(u_des+1)*mdp.n_states,:]
    # A,values = get_matrix_from_policy(p1,gw)
    # print(f"The policy is: {A}")
    # plot_arrows(A,values, f'u_{u_des}', discount)

    # u_des = 2
    # p1 = soft_policy[u_des*mdp.n_states:(u_des+1)*mdp.n_states,:]
    # A,values = get_matrix_from_policy(p1,gw)
    # print(f"The policy is: {A}")
    # plot_arrows(A,values, f'u_{u_des}', discount)

    # u_des = 5
    # p1 = soft_policy[u_des*mdp.n_states:(u_des+1)*mdp.n_states,:]
    # A,values = get_matrix_from_policy(p1,gw)
    # print(f"The policy is: {A}")
    # plot_arrows(A,values, f'u_{u_des}', discount)


    # for i,v in enumerate(v_soft):
    #     (s,u) = mdpRM.su_pair_from_s(i)
    #     if u == 0:
    #         print(f" u ={u},{u+1} | s = {s} | v_soft = {v:.2f} , {v_soft[(u+1)*mdp.n_states + s]:.2f}")



        







