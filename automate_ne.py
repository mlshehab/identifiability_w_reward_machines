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
import xml.etree.ElementTree as ET
from collections import deque
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
        # self.children = []
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


import argparse

# Define a function to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Automate script with depth option")
    parser.add_argument("-depth", type=int, help="Set the depth", required=True)
    return parser.parse_args()

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
    
   
    rm = RewardMachine("./rm_examples/patrol.txt")

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
    # print(f"Reward: {reward.shape}, S: {mdp.n_states}, A: {mdp.n_actions}, RM: {rm.n_states}")

 


    #############
    #############
    # PREFIX TREE
    #############
    #############

    # Parse command-line arguments
    args = parse_args()
    
    # Set the depth variable from the command line argument
    depth = args.depth
    print(f"The depth is: {depth}")
    Root = Node(label = None, state= None, u = None,policy = None , is_root= True)
    queue = [(Root, 0)]  # Queue of tuples (node, current_depth)

    # The first time step here is assuming a fully supported starting distribution
    current_node, current_depth = queue.pop(0)  # Dequeue the next node

    starting_states = [9]
    # for s in starting_states:
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

    def print_tree(node, level=0):
        print(" " * (level * 4) + f"Node({node.label}, {node.u}, {node.policy})")
        for child in node.children:
            print_tree(child, level + 1)
    
    # print_tree(Root)

    def collect_state_traces_iteratively(root):
        """
        Iteratively traverse the tree using BFS, starting from the root node,
        and collect proposition traces leading to each MDP state.

        Parameters:
        - root: The root node of the tree.

        Returns:
        - state_traces: A dictionary where keys are states and values are lists of proposition traces.
        """
        # Initialize the dictionary to store traces for each state
        state_traces = {}

        # Initialize the queue with the root node
        queue = deque([root])

        # Perform BFS traversal
        while queue:
            # Dequeue the next node
            current_node = queue.popleft()

            # If the node has a valid state, use the node's label to represent the trace
            if current_node.state is not None:
                if current_node.state not in state_traces:
                    state_traces[current_node.state] = []  # Initialize the list if the state is not in the dictionary
                state_traces[current_node.state].append((current_node.label,current_node.policy))

            # Enqueue all children
            for child in current_node.children:
                queue.append(child)

        return state_traces

    def remove_consecutive_duplicates(input_str):
        if not input_str:
            return ""
        
        result = [input_str[0]]  # Start with the first character
        
        for i in range(1, len(input_str)):
            if input_str[i] != input_str[i - 1]:
                result.append(input_str[i])
        
        return ''.join(result)

    # # Example usage:
    # input_str = 'CDCDCDCD'
    # output_str = remove_consecutive_duplicates(input_str)
    # print(output_str)  # Output: 'BADA'
    def get_unique_traces(proposition_traces):
        """
        Extract unique items from a list of tuples based on the label part of each tuple.

        Parameters:
        - proposition_traces: List of tuples where each tuple contains (label, policy).

        Returns:
        - unique_traces: List of unique tuples based on the label part.
        """
        # Use a set to track unique labels  
        unique_labels = set()
        # List to store unique tuples
        unique_traces = []

        for label, policy in proposition_traces:
            # Check if the label is already in the set
            if remove_consecutive_duplicates(label) not in unique_labels:
                # If not, add it to the set and add the tuple to the unique list
                unique_labels.add(remove_consecutive_duplicates(label))
                unique_traces.append((remove_consecutive_duplicates(label), policy))

        return unique_traces

    def group_traces_by_policy(proposition_traces):
        """
        Group labels from proposition traces based on their policies.

        Parameters:
        - proposition_traces: List of tuples where each tuple contains (label, policy).

        Returns:
        - grouped_traces: A dictionary where keys are policies and values are lists of labels.
        """
        # Initialize a dictionary to store lists of labels for each policy
        grouped_traces = {}

        for label, policy in proposition_traces:
            # If the policy is not yet in the dictionary, add it with an empty list
            if policy not in grouped_traces:
                grouped_traces[policy] = []
            # Append the label to the corresponding policy list
            grouped_traces[policy].append(label)

        # Convert the dictionary values to separate lists
        return list(grouped_traces.values())
        
    # Example usage:
    # Collect proposition traces for each state iteratively
    # state_traces = collect_state_traces_iteratively(Root)
   

    

    def write_traces_to_xml(state_traces, n_states, filename="state_traces.xml"):
        """
        Writes the grouped traces for each state to an XML file.

        Parameters:
        - state_traces: A dictionary where each key is a state and value is a list of (label, policy) tuples.
        - n_states: The number of states.
        - filename: The output XML file name.
        """
        # Create the root element
        root = ET.Element("StateTraces")

        for state in range(n_states):
            # Create a state element
            state_element = ET.SubElement(root, f"State_{state}")

            # Get unique traces for the current state
            unique_traces = get_unique_traces(state_traces[state])
            # Group the traces by their policy
            grouped_lists = group_traces_by_policy(unique_traces)

            # Add grouped lists to the state element
            for idx, group in enumerate(grouped_lists, 1):
                list_element = ET.SubElement(state_element, f"List_{idx}")
                list_element.text = ", ".join(group)  # Join the items into a single string

        # Convert the ElementTree to a string
        tree = ET.ElementTree(root)
        
        # Write the XML string to a file
        tree.write(filename, encoding='utf-8', xml_declaration=True)

        print(f"Traces written to {filename}.")


    # # Example usage
    state_traces = collect_state_traces_iteratively(Root)
    n_states = len(state_traces)  # Assuming this is defined elsewhere
    write_traces_to_xml(state_traces, n_states)
