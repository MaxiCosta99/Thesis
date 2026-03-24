import numpy as np
from single_agent import agent, pretrain, world, run
from single_agent_animator import animate


#start_B = np.array(([[[1,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,1]]]))
start_B = np.array(([[[0,0,0,0],[1,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]],[[0,0,0,1],[0,1,0,0],[1,0,1,0],[0,0,0,0]],[[0,0,1,0],[0,1,0,0],[0,0,0,0],[1,0,0,1]]]))
#start_A = np.random.uniform(0,1,(5,4,2))
#start_A = start_A/np.sum(start_A, axis = 0, keepdims = True)
start_A = np.ones((5,4,2))
start_A[0,1,0] = 10
start_A[1,1,1] = 10
start_A[2,2,0] = 10
start_A[2,3,1] = 10
start_A[3,2,1] = 10
start_A[3,3,0] = 10
start_A[4,0,:] = 10


start_s = np.array([[0.5,0.5],[0,0],[0,0],[0,0]])


goals = np.ones(start_s.shape)/8
goals[2,0] = 1e1
goals[3,1] = 1e1
goals[2,1] = 1e-10
goals[3,0] = 1e-10
goals = goals/np.sum(goals)

goal_obs = np.ones(start_A.shape[0])
goal_obs[2] = 1e2
goal_obs[3] = 1e-5
goal_obs = goal_obs/np.sum(goal_obs)

U = [0,1,2,3]
reason_depth = 3
dec_temp = 0.5
little_o_imaginer = True
o_imagination_breadth = 2
o_prob_threshold = 0.1

little_u_imaginer = True
u_imagination_breadth = 3
u_prob_threshold = 0.1

bayes_observe = True
terms = ["risk_obs", "nov"]


cheeses = [0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,0,1,1,0,0,0,1,0,1]
#cheeses = [0]
epochs = len(cheeses)
nr_moves = 3

maxi = agent(start_A, start_B, start_s, U = U, goal_states = goals,
             reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
             little_o_imaginer = little_o_imaginer, o_imagination_breadth = o_imagination_breadth, o_prob_threshold = o_prob_threshold,
             little_u_imaginer = little_u_imaginer, u_imagination_breadth= u_imagination_breadth, u_prob_threshold = u_prob_threshold,
             goal_obs = goal_obs, moves_to_go = nr_moves, bayes_observe = bayes_observe, terms = terms)




agent_positions, A_matrices = run(maxi, epochs, nr_moves, start_s, reason_depth, nr_moves, cheeses)

animate(agent_positions, cheeses, A_matrices)