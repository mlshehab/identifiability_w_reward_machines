import numpy as np
from reward_machine.reward_machine import RewardMachine
import scipy.linalg
class MDP(object):
    def __init__(self, n_states, n_actions, P, gamma, horizon):
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
        self.gamma = gamma
        self.horizon = horizon 


class MDPRM(object):
    def __init__(self, mdp : MDP, rm : RewardMachine, L : dict):
        self.mdp = mdp 
        self.rm = rm
        self.L = L 
        
    
    def su_pair_from_s(self,i):
       
        return  ( i % self.mdp.n_states ,  i  // self.mdp.n_states)  # (s,u) pair

    def s_from_su_pair(self, su):

        (s,u) = su
        return u*self.mdp.n_states + s

    def construct_product(self):
        n_mdp_states = self.mdp.n_states # number of mdp states
        n_rm_states = len(self.rm.U)     # number of RM states
        n_mdprm_states = n_mdp_states*n_rm_states # number of product states

        n_mdprm_actions = self.mdp.n_actions # number of product actions

        P_mdprm = []

        for a in range(n_mdprm_actions):
            P = np.zeros(shape = (n_mdprm_states,n_mdprm_states))
            for s in range(n_mdprm_states):
                for s_prime in range(n_mdprm_states):
                    (s1,u1) = self.su_pair_from_s(s)
                    (s1_prime,u1_prime) = self.su_pair_from_s(s_prime)

                    if u1_prime == self.rm._compute_next_state(u1, L[s1_prime]):
                        P[s,s_prime] = self.mdp.P[a][s1,s1_prime] 
                   
            P_mdprm.append(P)

        return MDP(n_states=n_mdprm_states, n_actions=n_mdprm_actions,P = P_mdprm,gamma = self.mdp.gamma,horizon=self.mdp.horizon)
        
Pa1 = np.array([
    [0.17450356, 0.23063070, 0.09140879, 0.15965038, 0.16726377, 0.17654280],
    [0.21880141, 0.05086728, 0.19830625, 0.26830713, 0.16136689, 0.10235053],
    [0.22667067, 0.06500235, 0.20073538, 0.14525782, 0.27620895, 0.08612483],
    [0.22983631, 0.16930150, 0.20143408, 0.02948996, 0.18296449, 0.18697365],
    [0.19994916, 0.18180141, 0.19479883, 0.06005535, 0.17819130, 0.18520400],
    [0.08233311, 0.17491685, 0.22148033, 0.10214056, 0.28230491, 0.13682423]
])

Pa2 = np.array([
    [0.22916615, 0.23347260, 0.17677483, 0.10912436, 0.08410495, 0.16735711],
    [0.14404769, 0.15866721, 0.10156419, 0.22616263, 0.17935163, 0.19020666],
    [0.11637614, 0.10515980, 0.21783612, 0.22287861, 0.12093058, 0.21681875],
    [0.15292588, 0.17284777, 0.19524639, 0.09472642, 0.17304882, 0.21120472],
    [0.22049103, 0.17828885, 0.15461098, 0.13616755, 0.14866654, 0.16177406],
    [0.13696833, 0.21087032, 0.19716790, 0.19985495, 0.13389710, 0.12124139]
])

Pa3 = np.array([
    [0.13427358, 0.19849364, 0.15199673, 0.12576411, 0.18415667, 0.20531527],
    [0.12065782, 0.15901558, 0.22974395, 0.20683371, 0.18662230, 0.09712664],
    [0.18541564, 0.14928391, 0.11820370, 0.14145892, 0.22758133, 0.17805650],
    [0.20348675, 0.13709635, 0.19950796, 0.11973708, 0.21646012, 0.12371173],
    [0.16382877, 0.14567321, 0.20891424, 0.17062505, 0.13543788, 0.17552085],
    [0.19457896, 0.21021208, 0.13854420, 0.16183913, 0.17692410, 0.11790153]
])

Pa4 = np.array([
    [0.21950967, 0.18878493, 0.13529407, 0.19868784, 0.14420195, 0.11352154],
    [0.15321894, 0.19987710, 0.12329814, 0.20390876, 0.15247393, 0.16722313],
    [0.12634519, 0.16247501, 0.15709845, 0.13080561, 0.19873212, 0.22454362],
    [0.16826710, 0.18021139, 0.21459841, 0.15529001, 0.10971519, 0.17191790],
    [0.14209656, 0.14143835, 0.17864823, 0.14743918, 0.16750408, 0.22287360],
    [0.19056254, 0.12764522, 0.19033557, 0.16348461, 0.22737274, 0.10059933]
])

P = [Pa1,Pa2,Pa3,Pa4]

mdp = MDP(n_states=6,n_actions=4,P = P,gamma = 0.9,horizon=10)
rm = RewardMachine("./rm1.txt")
print(rm.delta_u)
print("rm_s: ", len(rm.U))

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


def S_prime_redistribution(mdpRM : MDPRM):
    mdp = mdpRM.mdp
    rm = mdpRM.rm
    L = mdpRM.L

    # find the number of transitions in the reward machine
    
    B = []

    for u in range(len(rm.U) - 1): # -1 because we're eliminating the terminal state
        # find the connectivity of u
        potential_u_prime = rm.delta_u[u]

        B_u = np.zeros((mdp.n_states*mdp.n_actions, len(potential_u_prime))) 
        
        for i, u_prime in enumerate(potential_u_prime):

            
            for s in range(mdp.n_states):
                for a in range(mdp.n_actions):

                    for s_prime in range(mdp.n_states):
                        if u_prime == rm._compute_next_state(u,L[s_prime]):
                            B_u[s + a*mdp.n_states, i] += mdp.P[a][s,s_prime]

        # print(f"B({u}): ", B_u)    
        # print(f"B({u}) is of shape: {mdp.n_states*mdp.n_actions}x{len(potential_u_prime)}")
        # print(f"B({u}) is of rank: {np.linalg.matrix_rank(B_u)}")

        B.append(B_u)
    
    return B

S_prime_redistribution(mdpRM)

def d(P):
    m,n = P.shape
    dP = np.zeros((m,m*n))
    for i in range(m):
        dP[i,n*i:n*(i+1)] = P[i]
    return dP

# t = np.array([[1,2,4],[3,2,4],[1,6,6],[1,2,0],[-1,-1,-1]])
# print(d(t))

def L4DC_equivalence_class(mdpRM : MDPRM):
    
    mdp = mdpRM.mdp
    rm  =  mdpRM.rm
    L = mdpRM.L

    prodMDP = mdpRM.construct_product()
    # construct the P matrix of the product
    P = prodMDP.P[0]
    E = np.eye(prodMDP.n_states)
    for a in range(1,prodMDP.n_actions):
        P = np.vstack((P,prodMDP.P[a]))
        E = np.vstack((E, np.eye(prodMDP.n_states)))

    Psi = d(P)
    dim_r = Psi.shape[1]
    print(f"dim_r  = {dim_r}")
    A = np.hstack((Psi, -E + prodMDP.gamma*P))
    print("A: ", A.shape)
    print("A rank: ", np.linalg.matrix_rank(A))
    K = scipy.linalg.null_space(A)
    print("K: ", K.shape)
    projected_K = K[:dim_r,:]

    print(f"The dimension of the equivalence is: {np.linalg.matrix_rank(projected_K)}")
  
L4DC_equivalence_class(mdpRM)
print(6*6*4*3*3)














