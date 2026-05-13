import numpy as np
import copy
from two_agent import agent, world, run, true_A, true_A_new
from two_agent_animator import animate
from utils import make_goals_two_agent, EM2, A_diff_plotter, efe_eps_plotter, A_KL, compute_norm_cov_A, state_switcher
import json
import os
from datetime import datetime
from claude_plotting_two_agents import two_agent_plotter


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

start_s = np.zeros((2,2,4,4))
start_s[:,:,0,0] = [[0.25,0.25],[0.25,0.25]]

full_start_s = np.repeat(start_s[np.newaxis,:,:,:,:],repeats = 2, axis =0)

true_A = true_A_new(base = 1e-8, pretrain_strength = 1.0, other_strength = 1.0) #TRUE A (0.1, 10,2) works
#agent0_A = np.load("inits/ta_learnedA.npy").copy()
#agent1_A = np.load("inits/ta_learnedA.npy").copy()

agent0_full_A = np.stack((true_A, np.ones((12,2,2,4,4))), axis = 0) #[agent0_A, agent1_A]
agent1_full_A = np.stack((true_A, true_A), axis = 0) #[agent1_A, agent0_A]

U = [0,1,2,3]

bayes_observe0 = True
bayes_observe1 = True

branch_reductor0 = "cut"
branch_reductor1 = "cut"

mc_samples0 = 1000 #use in case branch reductor is MCTS (it's not really MCTS HEHE)
mc_samples1 = 100 #use in case branch reductor is MCTS

little_o_imaginer = True
o_prob_threshold = 0.15

little_u_imaginer = True
u_prob_threshold = 0.2


B = mdp_B
beta = 1
nr_moves = 3
goal_type = "obs" #default or obs

dec_temp = 0

selfish0 = False #to use only one world goal states
selfish1 = False #to use only one world goal states

state_stabilizer0 = False
state_stabilizer1 = False

efe_discounting0 = "new"
efe_discounting1 = "new"

desires_other_showtype = "mean"




def default():

        #experiments params
        epochs = 1
        num_tries = 2
        nr_A_learning_moves = 50
        message_success_prob = 0.5



        #world params
        reward = 2 #states 3            #for mdp_B I need 3, for det_B I need 2
        punishment = 6 #states 8           #8 #for mdp_B i need 8, for det_B I need 5
        
        uncertain_goals = False
        agent0_goals, agent1_goals = make_goals_two_agent(reward = reward, punishment = punishment, beta = beta, goal_type = goal_type, uncertain_goals = uncertain_goals)
        
        obs_model_given = True #I am sorry for this
        A_learning = not obs_model_given
        exploration = False #not used anymore
        
        



        #agent params
        reason_depth = 3

        social0 = True
        social1 = False

        terms0 = ["risk_obs"]
        terms1 = ["risk_obs"]

        sampling0 = "mean" #don't use sample for now!!!!
        sampling1 = "mean"




        empathetic0 = False  #to learn goals of other
        empathetic1 = False #to learn goals of ohter

        infer_only_on_last_action = False
        beta1 = 1 - 10 ** np.random.uniform(-2, -0.5)   # roughly 0.68 to 0.99
        beta2 = 1 - 10 ** np.random.uniform(-4, -1)      # roughly 0.999 to 0.9
        num_iters = 100#np.random.randint(10,100)
        num_samples = 100#np.random.randint(10,100)
        step_size = 0.01#10**(np.random.uniform(-4,-2))
        optimizer = None
        reg_strength = 0
        goal_learning_beta = 1


        opt_params = {"natgrad" : False,
                        "log_space" : True,
                        "reg_strength" : reg_strength,
                        "beta1" : beta1,
                        "beta2" : beta2,
                        "num_iters" : num_iters,
                        "num_samples" : num_samples,
                        "optimizer" : optimizer,
                        "clipping" : False,
                        "clips" : None,
                        "step_size" : step_size,
                        "infer_only_on_last_action" : infer_only_on_last_action}


        """
        os.makedirs("hyperparams", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"hyperparams/run_{timestamp}.json"

        # Save at the start
        with open(filepath, "w") as f:
                json.dump({"hyperparams": opt_params}, f, indent=2)"""
                





        
        




        A_matrices_eps = []
        bac_eps_1 = []
        bac_eps_2 = []
        policy_eps_1 = []
        policy_eps_2 = []
        kls_eps = []
        var_eps = []

        for e in range(num_tries):

                agent0 = agent(agent0_full_A.copy(), B.copy(), copy.deepcopy(full_start_s), U = U.copy(), goal_states = copy.deepcopy(agent0_goals),
                reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
                little_o_imaginer = little_o_imaginer, o_prob_threshold = o_prob_threshold,
                little_u_imaginer = little_u_imaginer, u_prob_threshold = u_prob_threshold,
                agent_id = 0, moves_to_go = nr_moves, social = social0, terms = terms0, sampling = sampling0,
                branch_reductor = branch_reductor0, mc_samples = mc_samples0, selfish = selfish0, empathetic = empathetic0,
                bayes_observe = bayes_observe0, obs_model_given = obs_model_given, efe_discounting = efe_discounting0,
                state_stabilizer = state_stabilizer0, goal_learning_beta = goal_learning_beta)

                agent1 = agent(agent1_full_A.copy(), B.copy(), copy.deepcopy(full_start_s), U = U.copy(), goal_states = copy.deepcopy(agent1_goals),
                reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
                little_o_imaginer = little_o_imaginer, o_prob_threshold = o_prob_threshold,
                little_u_imaginer = little_u_imaginer, u_prob_threshold = u_prob_threshold,
                agent_id = 1, moves_to_go = nr_moves, social = social1, terms = terms1, sampling = sampling1,
                branch_reductor = branch_reductor1, mc_samples = mc_samples1, selfish = selfish1, empathetic = empathetic1,
                bayes_observe = bayes_observe1, obs_model_given = obs_model_given, efe_discounting = efe_discounting1,
                state_stabilizer = state_stabilizer1, goal_learning_beta = goal_learning_beta)
  
                epochs_positions, epochs_A_matrices, epochs_agent_worlds, epochs_beliefs, epochs_desires, cheeses, epochs_pus, epochs_best_act_counters = run(agent0 = agent0, agent1 = agent1, epochs = epochs, nr_moves = nr_moves, reason_depth = reason_depth, start_beliefs = copy.deepcopy(full_start_s),
                                                                                                        exploration_phase = exploration, A_learning = A_learning, infer_only_on_last_action = infer_only_on_last_action,
                                                                                                        opt_params = opt_params, nr_A_learning_moves = nr_A_learning_moves, message_success_prob = message_success_prob)
                
                bac1 = []
                bac2 = []
                pe1 = []
                pe2 = []
                A = []
                for i in range(epochs):
                        A.append(epochs_A_matrices[i][0][0]) #all A-matrices are the same for now
                        
                        bac1.append(epochs_best_act_counters[i][0])
                        bac2.append(epochs_best_act_counters[i][1])

                        pe1.append(epochs_pus[i][0][0])
                        pe2.append(epochs_pus[i][1][0])
                                

                kls = []
                var = []
                for i in range(epochs):
                
                        kls.append(A_KL(true_A, A[i]))
                        var.append(compute_norm_cov_A(A[i]))
                

                A_matrices_eps.append(A)
                bac_eps_1.append(bac1)
                bac_eps_2.append(bac2)
                policy_eps_1.append(pe1)
                policy_eps_2.append(pe2)
                kls_eps.append(kls)
                var_eps.append(var)


        

              



        two_agent_plotter(kls_eps,
                          var_eps,
                          policy_eps_1,
                          policy_eps_2,
                          bac_eps_1,
                          bac_eps_2
                          )







        """
        with open(filepath) as f:
                run_data = json.load(f)

        run_data["result"] = np.round(np.log(agent0.goal_states[1]),2).tolist()

        with open(filepath, "w") as f:
                json.dump(run_data, f, indent=2)"""

                

        """animate(epochs_positions, cheeses, A_matrices = epochs_A_matrices, agent_worlds = epochs_agent_worlds, epochs_beliefs = epochs_beliefs, epochs_desires = epochs_desires,
        desires_other_showtype = desires_other_showtype)"""






default()
















"""

ANDIAMOACULO = 1


for K in range(ANDIAMOACULO):

        beta1 = np.random.uniform(0.8,0.9)
        beta2 = np.random.uniform(0.9,1)
        num_iters = 1#np.random.randint(100,300)
        num_samples = 1#np.random.randint(50,100)
        step_size = np.random.uniform(0.001,0.1)
        optimizer = "adam"




        opt_params = {"natgrad" : False,
                        "log_space" : True,
                        "reg_strength" : 0,
                        "beta1" : beta1,
                        "beta2" : beta2,
                        "num_iters" : num_iters,
                        "num_samples" : num_samples,
                        "optimizer" : optimizer,
                        "clipping" : False,
                        "clips" : None,
                        "step_size" : step_size}
        print("")
        print("----------------------------------")
        print(f"NEW HYPERPARAMETER SEARCH ROUND")
        print("")
        print(f"These are the optimization parameters: \n {opt_params}")

        agent0 = agent(agent0_full_A.copy(), start_B.copy(), copy.deepcopy(full_start_s), U = U.copy(), goal_states = copy.deepcopy(agent0_fullgoals), goal_obs = agent0_fullgoals_obs.copy(),
                reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
                little_o_imaginer = little_o_imaginer, o_imagination_breadth = o_imagination_breadth, o_prob_threshold = o_prob_threshold,
                little_u_imaginer = little_u_imaginer, u_imagination_breadth= u_imagination_breadth, u_prob_threshold = u_prob_threshold,
                agent_id = 0, moves_to_go = nr_moves, social = social0, terms = terms0, sampling = sampling0,
                branch_reductor = branch_reductor0, mc_samples = mc_samples0, selfish = selfish0, empathetic = empathetic0,
                bayes_observe = bayes_observe0, obs_model_given = obs_model_given, futuro_prosperoso = futuro_prosperoso0,
                random_action = random_action1)

        agent1 = agent(agent1_full_A.copy(), start_B.copy(), copy.deepcopy(full_start_s), U = U.copy(), goal_states = copy.deepcopy(agent1_fullgoals), goal_obs = agent1_fullgoals_obs.copy(),
                reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
                little_o_imaginer = little_o_imaginer, o_imagination_breadth = o_imagination_breadth, o_prob_threshold = o_prob_threshold,
                little_u_imaginer = little_u_imaginer, u_imagination_breadth= u_imagination_breadth, u_prob_threshold = u_prob_threshold,
                agent_id = 1, moves_to_go = nr_moves, social = social1, terms = terms1, sampling = sampling1,
                branch_reductor = branch_reductor1, mc_samples = mc_samples1, selfish = selfish1, empathetic = empathetic1,
                bayes_observe = bayes_observe1, obs_model_given = obs_model_given, futuro_prosperoso = futuro_prosperoso1,
                random_action = random_action2)





        epochs_positions, epochs_A_matrices, epochs_agent_worlds, epochs_beliefs, epochs_desires, cheeses = run(agent0 = agent0, agent1 = agent1, epochs = epochs, nr_moves = nr_moves, reason_depth = reason_depth, start_beliefs = copy.deepcopy(full_start_s),
                                                                                                        exploration_phase = exploration, A_learning = A_learning, infer_only_on_last_action = infer_only_on_last_action,
                                                                                                        opt_params = opt_params)
        print("_________________________________________________________")
        print(f"These are the final learned goals: \n {np.mean(agent0.goal_states[1], axis = (0,2))}")
        print("_____________________________________________________________--")



print("I finished goal training")
agent0.random_action = False
agent0.empathetic = False
epochs = 1


new_desires_reshaped = np.array([[1.58686984, 1.57605153, 4.2516792,  1.4978339],
 [1.48872686, 1.30516358, 1.47279723, 6.09308952]])

temp = np.repeat(new_desires_reshaped[np.newaxis,:,:], axis = 0, repeats = 2)
new_desires_reshaped = np.repeat(temp[:,:,np.newaxis,:], axis = 2, repeats = 4)



agent0.goal_states[1] = new_desires_reshaped.copy()

print(np.mean(agent0.goal_states[1], axis = (0,2)))


epochs_positions, epochs_A_matrices, epochs_agent_worlds, epochs_beliefs, epochs_desires, cheeses = run(agent0 = agent0, agent1 = agent1, epochs = epochs, nr_moves = nr_moves, reason_depth = reason_depth, start_beliefs = copy.deepcopy(full_start_s),
                                                                                                exploration_phase = exploration, A_learning = A_learning, infer_only_on_last_action = infer_only_on_last_action)


"""

#animate(epochs_positions, cheeses, A_matrices = epochs_A_matrices, agent_worlds = epochs_agent_worlds, epochs_beliefs = epochs_beliefs, epochs_desires = epochs_desires,
     #   desires_other_showtype = desires_other_showtype)