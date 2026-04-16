import numpy as np
import copy
from two_agent import agent, world, run, true_A
from two_agent_animator import animate


start_B = np.array(([[[0,0,0,0],
                    [1,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]],
                    [[1,1,0,0],
                    [0,0,0,0],
                    [0,0,1,0],
                    [0,0,0,1]],
                    [[0,0,0,1],
                    [0,1,0,0],
                    [1,0,1,0],
                    [0,0,0,0]],
                    [[0,0,1,0],
                    [0,1,0,0],
                    [0,0,0,0],
                    [1,0,0,1]]]))


start_s = np.zeros((2,2,4,4))
start_s[:,:,0,0] = [[0.25,0.25],[0.25,0.25]]

full_start_s = np.repeat(start_s[np.newaxis,:,:,:,:],repeats = 2, axis =0)

agent0_A, agent1_A = true_A(base = 0.1, pretrain_strength =10) #TRUE A

# use smaller Dirichlet prior so updates change probabilities faster
agent0_A = 0.1 * np.ones((11,2,2,4,4))
agent1_A = 0.1 * np.ones((11,2,2,4,4))



agent0_full_A = np.stack((agent0_A, agent1_A), axis = 0)
agent1_full_A = np.stack((agent1_A, agent0_A), axis = 0)


reward = 2
punishment = 5 

rp = np.exp(reward)
pp = np.exp(-punishment)

#rp = 1e1
#pp = 1e-10


agent0_goals = np.ones(start_s.shape)
agent0_goals[0,:,2,:] = rp
agent0_goals[1,:,3,:] = rp
agent0_goals[0,:,3,:] = pp
agent0_goals[1,:,2,:] = pp
agent0_goals = agent0_goals/pp
agent0_goals_cat = agent0_goals/np.sum(agent0_goals)

agent1_goals = np.ones(start_s.shape)
agent1_goals[:,0,:,2] = rp
agent1_goals[:,1,:,3] = rp
agent1_goals[:,0,:,3] = pp
agent1_goals[:,1,:,2] = pp
agent1_goals = agent1_goals/pp
agent1_goals_cat = agent1_goals/np.sum(agent1_goals)

assert(np.min(agent0_goals) == 1)
assert(np.min(agent1_goals) == 1)


uncertain_goals = np.ones((2,2,4,4))
"""
ug_sample = np.random.dirichlet(np.ones(64), 1)
uncertain_goals = ug_sample.reshape(uncertain_goals.shape)

print(uncertain_goals)"""

#test0_goals = np.ones((2,2,4,4))*1e-5
#test1_goals = np.ones((2,2,4,4))*1e-5
#test0_goals[0,:,3,:] = 1e-10
#test0_goals[1,:,2,:] = 1e-10
#test1_goals[:,0,:,3] = 1e-10
#test1_goals[:,1,:,2] = 1e-10

#test_goals[0,0,2,2] = 1e1
#test_goals[0,1,2,3] = 1e1
#test_goals[1,0,3,2] = 1e1
#test_goals[1,1,3,3] = 1e1
#test_goals0 = test_goals0/np.sum(test_goals0)
#test_goals1 = test_goals1/np.sum(test_goals1)


#agent0_fullgoals = np.stack((agent0_goals_cat, agent1_goals), axis = 0)
#agent1_fullgoals = np.stack((agent1_goals_cat, agent0_goals), axis = 0)

agent0_fullgoals = [agent0_goals_cat, agent1_goals]#uncertain_goals]
agent1_fullgoals = [agent1_goals_cat, agent0_goals]





agent0_goal_obs = np.ones(agent0_A.shape[0])
agent0_goal_obs[2] = 1e2
agent0_goal_obs[3] = 1e-5
agent0_goal_obs = agent0_goal_obs/np.sum(agent0_goal_obs)

agent1_goal_obs = np.ones(agent1_A.shape[0])
agent1_goal_obs[7] = 1e2
agent1_goal_obs[8] = 1e-5
agent1_goal_obs = agent1_goal_obs/np.sum(agent1_goal_obs)

agent0_fullgoals_obs = np.stack((agent0_goal_obs, agent1_goal_obs), axis = 0)
agent1_fullgoals_obs = np.stack((agent1_goal_obs, agent0_goal_obs), axis = 0)


U = [0,1,2,3]
reason_depth = 3
dec_temp = 0.01
little_o_imaginer = True
o_imagination_breadth = 3
o_prob_threshold = 0.1

little_u_imaginer = True
u_imagination_breadth = 3
u_prob_threshold = 0.1

social0 = True
social1 = False

terms0 = ["risk"]
terms1 = ["risk"]

sampling0 = "sample"
sampling1 = "sample"

branch_reductor0 = "cut"
branch_reductor1 = "cut"

mc_samples0 = 1000
mc_samples1 = 100

selfish0 = True
selfish1 = True

empathetic0 = False
empathetic1 = False


nr_moves = 3


cheeses = [[0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,1]]
#cheeses = [[0,0],[1,0],[0,1],[1,1]]
#cheeses = [[0,0]]
epochs = len(cheeses)


exploration = True
bayes_infer = False
A_learning = True
infer_only_on_last_action = False


agent0 = agent(agent0_full_A.copy(), start_B.copy(), copy.deepcopy(full_start_s), U = U.copy(), goal_states = copy.deepcopy(agent0_fullgoals), goal_obs = agent0_fullgoals_obs.copy(),
            reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
            little_o_imaginer = little_o_imaginer, o_imagination_breadth = o_imagination_breadth, o_prob_threshold = o_prob_threshold,
            little_u_imaginer = little_u_imaginer, u_imagination_breadth= u_imagination_breadth, u_prob_threshold = u_prob_threshold,
            agent_id = 0, moves_to_go = nr_moves, social = social0, terms = terms0, sampling = sampling0,
            branch_reductor = branch_reductor0, mc_samples = mc_samples0, selfish = selfish0, empathetic = empathetic0)

agent1 = agent(agent1_full_A.copy(), start_B.copy(), copy.deepcopy(full_start_s), U = U.copy(), goal_states = copy.deepcopy(agent1_fullgoals), goal_obs = agent1_fullgoals_obs.copy(),
            reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
            little_o_imaginer = little_o_imaginer, o_imagination_breadth = o_imagination_breadth, o_prob_threshold = o_prob_threshold,
            little_u_imaginer = little_u_imaginer, u_imagination_breadth= u_imagination_breadth, u_prob_threshold = u_prob_threshold,
            agent_id = 1, moves_to_go = nr_moves, social = social1, terms = terms1, sampling = sampling1,
            branch_reductor = branch_reductor1, mc_samples = mc_samples1, selfish = selfish1, empathetic = empathetic1)




epochs_positions, epochs_A_matrices, epochs_agent_worlds, epochs_beliefs, epochs_desires = run(agent0 = agent0, agent1 = agent1, epochs = epochs, nr_moves = nr_moves, reason_depth = reason_depth, start_beliefs = copy.deepcopy(full_start_s), cheeses = cheeses,
                                                                                                exploration_phase = exploration, bayes_infer = bayes_infer, A_learning = A_learning, infer_only_on_last_action = infer_only_on_last_action)

#print(epochs_beliefs[0][0][0])



animate(epochs_positions, cheeses, A_matrices = epochs_A_matrices, agent_worlds = epochs_agent_worlds, epochs_beliefs = epochs_beliefs, epochs_desires = epochs_desires)