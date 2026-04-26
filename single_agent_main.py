import numpy as np
from single_agent import agent, pretrain, world, run
from single_agent_animator import animate
from utils import A_KL_plotter, make_goals_single_agent
import time


#start_B = np.array(([[[1,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,1]]]))
det_B = np.array(([[[0,0,0,0],
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

mdp_B = np.array(([[[0,0,0.1,0.1],
                    [0.8,1,0,0],
                    [0.1,0,0.9,0],
                    [0.1,0,0,0.9]],
                    [[0.8,0.8,0.1,0.1],
                    [0,0.2,0,0],
                    [0.1,0,0.9,0],
                    [0.1,0,0,0.9]],
                    [[0.1,0.1,0,0.8],
                    [0.1,0.9,0,0],
                    [0.8,0,1,0],
                    [0,0,0,0.2]],
                    [[0.1,0.1,0.8,0],
                    [0.1,0.9,0,0],
                    [0,0,0.2,0],
                    [0.8,0,0,1]]]))


"""true_A = np.zeros((5,4,2))
true_A[0,1,0] = 1
true_A[1,1,1] = 1
true_A[2,2,0] = 1
true_A[2,3,1] = 1
true_A[3,2,1] = 1
true_A[3,3,0] = 1
true_A[4,0,:] = 1"""

true_A = np.zeros((5,2,4))
true_A[0,0,1] = 1
true_A[1,1,1] = 1
true_A[2,0,2] = 1
true_A[2,1,3] = 1
true_A[3,1,2] = 1
true_A[3,0,3] = 1
true_A[4,:,0] = 1

true_A = true_A/np.sum(true_A, axis = 0, keepdims = True)

U = [0,1,2,3]

start_s = np.array([[0.5,0.5],[0,0],[0,0],[0,0]]).T



def default():
    reason_depth = 3
    dec_temp = 0.1
    little_o_imaginer = True
    o_prob_threshold = 0.1

    little_u_imaginer = True
    u_prob_threshold = 0.1

    bayes_observe = True
    terms = ["risk_obs", "epi_value"]

    obs_model_given = False
    infer_before_update_obs_model = False


    num_eps = 40
    nr_moves = 3

    A_update_method = "EM_all_ep"
    random_moves = False

    state_stabilizer = False
    det_dec_rule = True

    maxi = agent(start_A, start_B, start_s, U = U, goal_states = goals,
                reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
                little_o_imaginer = little_o_imaginer, o_prob_threshold = o_prob_threshold,
                little_u_imaginer = little_u_imaginer, u_prob_threshold = u_prob_threshold,
                goal_obs = goal_obs, moves_to_go = nr_moves, bayes_observe = bayes_observe, terms = terms,
                obs_model_given = obs_model_given, infer_before_update_obs_model = infer_before_update_obs_model,
                update_A_method = A_update_method, random_moves = random_moves, state_stabilizer = state_stabilizer, det_dec_rule = det_dec_rule)


    #agent_positions, A_matrices, cheeses, epoch_best_act_count = run(maxi, num_eps, start_s, reason_depth, nr_moves)




    #animate(agent_positions, cheeses, A_matrices)






def learn_obs_model_act():
    
    #world parameters
    B = mdp_B
    obs_model_given = False
    nr_moves = 3
    reward = 2
    punishment = 6
    goal_type = "default"
    goals = make_goals_single_agent(reward = reward, punishment = punishment, beta = 1, goal_type = goal_type)



    #agent parameters
    reason_depth = 3
    dec_temp = 0.1
    little_o_imaginer = True
    o_prob_threshold = 0.1
    little_u_imaginer = True
    u_prob_threshold = 0.1
    A_update_method = "EM_all_ep" #either "default", "EM_all_ep", "EM_current_ep"
    terms = ["risk", "epi value"]
    infer_before_update_obs_model = True #not needed in this case
    random_moves = False
    state_stabilizer = False
    det_dec_rule = True
    bayes_observe = True

    EM_inits = {"start":np.load('inits/sa_start.npy'),
                "end": np.load('inits/sa_end.npy'),
                "gm": np.load('inits/sa_gm.npy')}


    
    
    num_eps = 50
    num_tries = 1

    diff_tries_A = []
    for i in range(num_tries):
        maxi = agent(true_A.copy(), B.copy(), start_s.copy(), U = U, goal_states = goals.copy(),
             reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
             little_o_imaginer = little_o_imaginer, o_prob_threshold = o_prob_threshold,
             little_u_imaginer = little_u_imaginer, u_prob_threshold = u_prob_threshold,
             goal_obs = goals.copy(), moves_to_go = nr_moves, bayes_observe = bayes_observe, terms = terms,
             obs_model_given = obs_model_given, infer_before_update_obs_model = infer_before_update_obs_model,
             update_A_method = A_update_method, random_moves = random_moves, state_stabilizer = state_stabilizer, det_dec_rule = det_dec_rule,
             EM_inits = EM_inits)

        agent_positions, A_matrices, cheeses, epoch_best_act_count = run(maxi, num_eps, start_s, reason_depth, nr_moves)

        diff_tries_A.append(A_matrices)

        epoch_best_act_count.append(3)

    A_KL_plotter(true_A, diff_tries_A, epoch_best_act_count)

    animate(agent_positions, cheeses, A_matrices)



def get_em_inits():
    
    #world parameters
    B = mdp_B
    obs_model_given = False
    nr_moves = 3
    reward = 2
    punishment = 6
    goal_type = "obs"
    goals = make_goals_single_agent(reward = reward, punishment = punishment, beta = 1, goal_type = goal_type)



    #agent parameters
    reason_depth = 3
    dec_temp = 0
    little_o_imaginer = True
    o_prob_threshold = 0.1
    little_u_imaginer = True
    u_prob_threshold = 0.1
    A_update_method = "EM_all_ep" #either "default", "EM_all_ep", "EM_current_ep"
    terms = ["risk_obs", "epi_value"]

    infer_before_update_obs_model = True #not needed in this case
    state_stabilizer = False #not needed in this case
    bayes_observe = True #not needed in this case

    EM_inits = {"start": None,
                "end": None,
                "gm": None}


    
    #exp parameters
    num_eps = 50

    maxi = agent(true_A.copy(), B.copy(), start_s.copy(), U = U, goal_states = goals.copy(),
            reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
            little_o_imaginer = little_o_imaginer, o_prob_threshold = o_prob_threshold,
            little_u_imaginer = little_u_imaginer, u_prob_threshold = u_prob_threshold,
            goal_obs = goals.copy(), moves_to_go = nr_moves, bayes_observe = bayes_observe, terms = terms,
            obs_model_given = obs_model_given, infer_before_update_obs_model = infer_before_update_obs_model,
            update_A_method = A_update_method, state_stabilizer = state_stabilizer, EM_inits = EM_inits)

    agent_positions, A_matrices, cheeses, best_act_counts = run(maxi, num_eps, start_s, reason_depth, nr_moves)


    A_KL_plotter(true_A, A_matrices, best_act_counts)

    #animate(agent_positions, cheeses, A_matrices)


#learn_obs_model_act()
    
get_em_inits()



