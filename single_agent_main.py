import numpy as np
from single_agent import agent, pretrain, world, run
from single_agent_animator import animate
from utils import A_KL_plotter


#start_B = np.array(([[[1,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,1]]]))
start_B = np.array(([[[0,0,0,0],[1,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]],[[0,0,0,1],[0,1,0,0],[1,0,1,0],[0,0,0,0]],[[0,0,1,0],[0,1,0,0],[0,0,0,0],[1,0,0,1]]]))

true_A = np.zeros((5,4,2))
true_A[0,1,0] = 1
true_A[1,1,1] = 1
true_A[2,2,0] = 1
true_A[2,3,1] = 1
true_A[3,2,1] = 1
true_A[3,3,0] = 1
true_A[4,0,:] = 1

true_A = true_A/np.sum(true_A, axis = 0, keepdims = True)
start_A = true_A.copy()


#start_s = np.array([[0.5,0.5],[0,0],[0,0],[0,0]])
start_s = np.array([[0.5,0.5],[0,0],[0,0],[0,0]])

reward = 2
punishment = 10

beta = 1
rp = np.exp(reward/beta)
pp = np.exp(-punishment/beta)


goals = np.ones(start_s.shape)
goals[2,0] = rp
goals[3,1] = rp
goals[2,1] = pp
goals[3,0] = pp
goals = goals/np.sum(goals)

print(goals)

goal_obs = np.ones(start_A.shape[0])
goal_obs[2] = 1e2
goal_obs[3] = 1e-5
goal_obs = goal_obs/np.sum(goal_obs)

U = [0,1,2,3]
reason_depth = 1
dec_temp = 1000
little_o_imaginer = True
o_imagination_breadth = 2
o_prob_threshold = 0.00000001

little_u_imaginer = True
u_imagination_breadth = 3
u_prob_threshold = 0.00000001

bayes_observe = True
terms = ["epi_value", "risk"]

obs_model_given = False
infer_before_update_obs_model = False


#cheeses = [0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,1,1,0,0,1,0,1,0,1,0,1,1,0,1,1,0,0,0,1,0,1,0,0,1,0,0,0,1,1,0,1,0]
cheeses = [0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,1,1,0,0,1]
epochs = len(cheeses)
nr_moves = 10

maxi = agent(start_A, start_B, start_s, U = U, goal_states = goals,
             reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
             little_o_imaginer = little_o_imaginer, o_imagination_breadth = o_imagination_breadth, o_prob_threshold = o_prob_threshold,
             little_u_imaginer = little_u_imaginer, u_imagination_breadth= u_imagination_breadth, u_prob_threshold = u_prob_threshold,
             goal_obs = goal_obs, moves_to_go = nr_moves, bayes_observe = bayes_observe, terms = terms,
             obs_model_given = obs_model_given, infer_before_update_obs_model = infer_before_update_obs_model)




#risk = maxi.compute_risk()
#print(risk)
#1/0

def exp_learning_A_no_infer(num_tries = int):
    reason_depth = 1
    dec_temp = 1000
    nr_moves = 10
    obs_model_given = False
    infer_before_update_obs_model = False

    cheeses = []
    num_eps = 50

    for i in range(num_eps):
        cheeses.append(np.random.choice([0,1]))

    epochs = len(cheeses)

    diff_tries_A = []
    for i in range(num_tries):
        maxi = agent(start_A, start_B, start_s, U = U, goal_states = goals,
                reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
                little_o_imaginer = little_o_imaginer, o_imagination_breadth = o_imagination_breadth, o_prob_threshold = o_prob_threshold,
                little_u_imaginer = little_u_imaginer, u_imagination_breadth= u_imagination_breadth, u_prob_threshold = u_prob_threshold,
             goal_obs = goal_obs, moves_to_go = nr_moves, bayes_observe = bayes_observe, terms = terms,
             obs_model_given = obs_model_given, infer_before_update_obs_model = infer_before_update_obs_model)

        agent_positions, A_matrices = run(maxi, epochs, start_s, reason_depth, nr_moves, cheeses)

        diff_tries_A.append(A_matrices)

    #print(f"This is the learned matrix A: \n {np.round(A_matrices[-1],2)}")
    A_KL_plotter(true_A, diff_tries_A)

exp_learning_A_no_infer(num_tries = 10)


#animate(agent_positions, cheeses, A_matrices)