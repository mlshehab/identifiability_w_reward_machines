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


    grid_size = 4
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
    
   
    rm = RewardMachine("./rm_examples/patrol.txt")

    policy = {}
    for rms in range(rm.n_states):
        policy[rms] = f"p{rms}"
    print("The policy is: ", policy)
    # print(rm.delta_u)
    

    # The grid numbering is :
    # 0 4 8 12
    # 1 5 9 13
    # 2 6 10 14
    # 3 7 11 15
    L = {}

    L[2], L[6], L[3], L[7]     = 'A', 'A', 'A', 'A'
    L[10], L[14], L[11], L[15] = 'B', 'B', 'B', 'B'
    L[8], L[9], L[12], L[13]   = 'C', 'C', 'C', 'C'
    L[0], L[1], L[4], L[5]     = 'D', 'D', 'D', 'D'

    mdpRM = MDPRM(mdp,rm,L)
    mdp_ =  mdpRM.construct_product()
    # now we need a state action state reward for the product MDP
    reward = np.zeros((mdp_.n_states, mdp_.n_actions, mdp_.n_states))
    print(f"Reward: {reward.shape}, S: {mdp.n_states}, A: {mdp.n_actions}, RM: {rm.n_states}")

    # for bar_s in range(mdp_.n_states):
    #     for a in range(mdp_.n_actions):
    #         for bar_s_prime in range(mdp_.n_states):
    #             (s,u) = mdpRM.su_pair_from_s(bar_s)
    #             (s_prime,u_prime) = mdpRM.su_pair_from_s(bar_s_prime)

    #             is_possible = mdp_.P[a][bar_s][bar_s_prime] > 0.0

    #             if u == 3 and u_prime == 4 and L[s_prime] == 'D' and is_possible:

    #                 reward[bar_s, a, bar_s_prime] = 10.0

    #             if u == 7 and u_prime == 0 and L[s_prime] == 'D' and is_possible:

    #                 reward[bar_s, a, bar_s_prime] = 10.0


    # q_soft,v_soft , soft_policy = infinite_horizon_soft_bellman_iteration(mdpRM,reward,logging = False)
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

    # u_des = 4
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




    #############
    #############
    # PREFIX TREE
    #############
    #############

    depth = 8
    Root = Node(label = None, state= None, u = None,policy = None , is_root= True)
    queue = [(Root, 0)]  # Queue of tuples (node, current_depth)

    # The first time step here is assuming a fully supported starting distribution
    current_node, current_depth = queue.pop(0)  # Dequeue the next node

    for s in range(mdp.n_states):
        # get label of the state
        label = L[s]
        # create a node for that state
        child_node = Node(label = label,state = s, u = u_from_obs(label,mdpRM), policy = None)
        child_node.policy = policy[child_node.u]
        current_node.add_child(child_node)
        queue.append((child_node, current_depth + 1))

    while queue:
        current_node, current_depth = queue.pop(0)  # Dequeue the next node

        if current_depth < depth:
            # get the state of the current node
            s = current_node.state
            # get the future possible states
            next_states = get_future_states(s,mdp)

            for nx_s in next_states:
                # get label of the next state
                label = L[nx_s]
                # create a node for that state
                nx_u = u_from_obs(current_node.label + label, mdpRM)
                child_node = Node(label = current_node.label + label,\
                                  state = nx_s, u = nx_u, policy = None)
                child_node.policy = policy[child_node.u]
                current_node.add_child(child_node)
                queue.append((child_node, current_depth + 1))
    
    # # for node in Root.children:
    # #     print(f"The children nodes of the root nodes are: {node.label}")

    # def print_tree(node, level=0):
    #     print(" " * (level * 4) + f"Node({node.label}, {node.u}, {node.policy})")
    #     for child in node.children:
    #         print_tree(child, level + 1)

    # print_tree(Root)

   

    class RM(object):
        def __init__(self,root):
            self.root = root
            self.u0 = 0
            self.delta_u = {} 
            self.delta_p = {}
            self.delta_u[self.u0] = {}
            self.delta_p[self.u0] = {}
        
      
        def add_transition(self, u, u_prime, label):
            if u not in self.delta_u:
                self.delta_u[u] = {}

            # check to see if the label is already in one of the transitions
            for u2 in self.delta_u[u]:
                if label in self.delta_u[u][u2]:
                    return False

            if u_prime not in self.delta_u[u]:
                self.delta_u[u][u_prime] = []

            # Append the label to the list if it's not already there
            if label not in self.delta_u[u][u_prime]:
                self.delta_u[u][u_prime].append(label)
            

            return True

        def _compute_next_state(self, u, label):
            for u_prime in self.delta_u[u]:
                if label in self.delta_u[u][u_prime]:
                    return u_prime
            
        def _compute_current_state(self, label):
            curr_u = self.u0

            for l in label:
                curr_u = self._compute_next_state(curr_u,l)
            
            return curr_u
            
    # # Now we are given a tree, and want to construct the reward machine from it
    
    # First, go to the root node, add it to a queue
    queue = [(Root, 0)]
    current_depth = 0

    RM_ = RM(Root)
    id = 1
    u0 = RM_.u0

    state_policy_dict = {}
    
    while queue:
        current_node , current_depth = queue.pop(0)

        for node in current_node.children:
            queue.append((node, current_depth + 1))
        
        if current_node.is_root:
            continue
        
        parent_node = current_node.parent
        parent_label = parent_node.label 

        # deal with the case when the parent node is the Root node:
        if parent_node.is_root:
            u_start = u0
            pi , label = current_node.policy, current_node.label 
          

            if pi not in RM_.delta_p[u_start]:
                
                RM_.delta_p[u_start][pi] = id
                RM_.add_transition(u_start,id, label)
                state_policy_dict[pi] = id
          
                id+=1
            else:
                up = RM_.delta_p[u_start][pi]
                RM_.add_transition(u_start,up, label)

        else:
            u_start = RM_._compute_current_state(parent_label)

            if not current_node.policy == parent_node.policy:
                # add a new node if this policy hasn't been seen before
                if current_node.policy not in state_policy_dict:

                    new_state_created = RM_.add_transition(u_start,id, current_node.label[-1])
                    state_policy_dict[current_node.policy] = id
                    
                    if new_state_created:# if this label hasnt been seen before
                        id+= 1

                else: # this policy has already been seen
                    corresponding_state  = state_policy_dict[current_node.policy] # find its corresponding state and append the current label
                    s = RM_.add_transition(u_start,corresponding_state , current_node.label[-1])

            else: # add a self transition with the seen label
               
                RM_.add_transition(u_start, u_start, label = current_node.label[-1] )

    print(state_policy_dict)
    # Pretty-print the dictionary
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(RM_.delta_u)
        







