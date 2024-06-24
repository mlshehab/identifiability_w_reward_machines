import numpy as np
from reward_machine import RewardMachine

class MDP(object):
    def __init__(self, n_states, n_actions, P, gamma, horizon):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
        self.gamma = gamma
        self.horizon = horizon 


class MDPRM(object):
    def __init__(self, mdp : MDP, rm : RewardMachine, L : list):
        self.mdp = mdp
        self.rm = rm
        self.L = L

        self.construct_product()
    
    def construct_product(self):

        n_mdp_states = self.mdp.n_states # number of mdp states
        n_rm_states = len(self.rm.U)     # number of RM states
        n_mdprm_states = n_mdp_states*n_rm_states # number of product states

        n_mdprm_actions = self.mdp.n_actions # number of product actions

        P_mdprm = []

        for a in range(n_mdprm_actions):
            P = np.zeros()




        print("The number is: ", n_mdprm_states)
        

mdp = MDP(5,1,1,1,1)
rm = RewardMachine("./rm1.txt")

rm = MDPRM(mdp,rm)