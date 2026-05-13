import numpy as np
from single_agent import agent, pretrain, world, run
from single_agent_animator import animate
from utils import A_KL_plotter, make_goals_single_agent, A_KL, state_switcher, compute_norm_cov_A
import time
from claude_plotting2 import A_KL_plotter, A_diff_plotter


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
    
    #world parameters
    B = mdp_B
    obs_model_given = False
    A_learning = not obs_model_given
    intra_ep_A_learning = False
    nr_moves_A_learning = 20

    nr_moves = 3
    reward = 2
    punishment = 6
    goal_type = "obs"
    goals = make_goals_single_agent(reward = reward, punishment = punishment, beta = 1, goal_type = goal_type)
    
    num_eps = 30
    num_tries = 2


    #agent parameters
    reason_depth = 3
    dec_temp = 0
    little_o_imaginer = True
    o_prob_threshold = 0.1
    little_u_imaginer = True
    u_prob_threshold = 0.1
    A_update_method = "EM_all_ep" #either "default", "EM_all_ep", "EM_current_ep"
    terms = ["risk_obs", "epi_value"]
    sampling = "default" #either default or sample

    infer_before_update_obs_model = False #not needed in this case
    state_stabilizer = False #not needed in this case
    bayes_observe = True #not needed in this case

    EM_inits = {"start": None,
                "end": None,
                "gm": None}


    chatty_boy = True
 
    

    maxi = agent(true_A.copy(), B.copy(), start_s.copy(), U = U, goal_states = goals.copy(),
            reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = chatty_boy,
            little_o_imaginer = little_o_imaginer, o_prob_threshold = o_prob_threshold,
            little_u_imaginer = little_u_imaginer, u_prob_threshold = u_prob_threshold,
            goal_obs = goals.copy(), moves_to_go = nr_moves, bayes_observe = bayes_observe, terms = terms,
            obs_model_given = obs_model_given, infer_before_update_obs_model = infer_before_update_obs_model,
            update_A_method = A_update_method, state_stabilizer = state_stabilizer, EM_inits = EM_inits,
            sampling = sampling)



    A_matrices_eps = []
    bac_eps = []
    kls_eps = []
    var_eps = []
    final_A_matrices_ss = []

    for _ in range(num_tries):
        agent_positions, A_matrices, cheeses, best_act_counts = run(maxi, num_eps, start_s, reason_depth, nr_moves,
                                                                    A_learning, intra_ep_A_learning, nr_moves_A_learning = nr_moves_A_learning, terms = terms)
        
        
        A_matrices_eps.append(A_matrices)
        bac_eps.append(best_act_counts)


        kls = []
        var = []
        for i in range(num_eps):
            
            kls.append(A_KL(true_A, state_switcher(A_matrices[i], true_A)))
            var.append(compute_norm_cov_A(A_matrices[i]))
            
        
        final_A_matrices_ss.append(state_switcher(A_matrices[-1], true_A))


        kls_eps.append(kls)
        var_eps.append(var)

        
    


    A_KL_plotter(kls_eps, var_eps, bac_eps)
    
    A_diff_plotter(true_A, np.mean(np.array(final_A_matrices_ss), axis = 0))

    #animate(agent_positions, cheeses, A_matrices)


    
default()



