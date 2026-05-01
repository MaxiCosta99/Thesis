import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy


def norm_om(A):
    return A/np.sum(A, axis = 0)

def flatten_om(A):
    return A.reshape(A.shape[0],-1)

def state_switcher(learned_A, true_A):
    """
    learnedA is a dirichlet counts matrix (can be of shape (num_obs, -1) as also (num_obs,2,4)
    true_A is a categorical of shape (num_obs, 2,2,4,4) or (num_obs, 4,2)

    output will be of shape of true_A (NOT THE INPUT LEARNED_A SHAPE!!!!!)
    """
    learned_A = flatten_om(learned_A)
    
    A_invstate = np.zeros(learned_A.shape)
    A_invstate[:,0:4] = learned_A[:,4:8]
    A_invstate[:,4:8] = learned_A[:,0:4]

    kl_state = A_KL(true_A, learned_A)
    kl_invstate = A_KL(true_A, A_invstate)

    if kl_invstate < kl_state: #states have been switched
        A = A_invstate.reshape(true_A.shape)
    else:
        A = learned_A.reshape(true_A.shape)

    return A


def compute_cov_dir(x):
    x = x.reshape(-1)
    x_norm = x/np.sum(x)
    x_norm = x_norm[:,np.newaxis]

    y = (np.diag(x_norm.reshape(-1))-(x_norm @ x_norm.T))/(np.sum(x)+1)

    E = scipy.linalg.eigh(y, eigvals_only = True)
    
    assert np.all(E > -1e-8)

    return y


def compute_norm_cov_A(A):
    """
    A is of shape (num_obs, ...)
    """
    
    A = A.reshape(A.shape[0],-1)
    

    cov_mat_sum = 0

    for a in A.T:
        cov_mat_sum =+ np.linalg.norm(compute_cov_dir(a))

    return cov_mat_sum



def make_goals_single_agent(reward = float, punishment = float, beta = float, goal_type = str):
  
    rp = np.exp(reward/beta)
    pp = np.exp(-punishment/beta)
    
    if goal_type == "default":

        """goals = np.ones((4,2))
        goals[2,0] = rp
        goals[3,1] = rp
        goals[2,1] = pp
        goals[3,0] = pp
        goals = goals/np.sum(goals)"""
        goals = np.ones((2,4))
        goals[0,2] = rp
        goals[1,3] = rp
        goals[1,2] = pp
        goals[0,3] = pp
        goals = goals/np.sum(goals)

    
    
    if goal_type == "obs":

        goals = np.ones((5))
        goals[2] = rp
        goals[3] = pp
        goals = goals/np.sum(goals)

    
    return goals


def make_goals_two_agent(reward = float, punishment = float, beta = float, goal_type = str, uncertain_goals = bool):
    rp = np.exp(reward/beta)
    pp = np.exp(-punishment/beta)

    
    if goal_type == "default":
        agent0_goals = np.ones((2,2,4,4))
        agent0_goals[0,:,2,:] = rp
        agent0_goals[1,:,3,:] = rp
        agent0_goals[0,:,3,:] = pp
        agent0_goals[1,:,2,:] = pp
        agent0_goals = agent0_goals/pp
        agent0_goals_cat = agent0_goals/np.sum(agent0_goals)

        agent1_goals = np.ones((2,2,4,4))
        agent1_goals[:,0,:,2] = rp
        agent1_goals[:,1,:,3] = rp
        agent1_goals[:,0,:,3] = pp
        agent1_goals[:,1,:,2] = pp
        agent1_goals = agent1_goals/pp
        agent1_goals_cat = agent1_goals/np.sum(agent1_goals)

        assert(np.min(agent0_goals) == 1)
        assert(np.min(agent1_goals) == 1)

        
        agent0_fullgoals = [agent0_goals_cat, agent1_goals]
        agent1_fullgoals = [agent1_goals_cat, agent0_goals]


        if uncertain_goals == True:
            uncertain_goals = np.ones((2,2,4,4))
            agent0_fullgoals[1] = uncertain_goals
            agent1_fullgoals[1] = uncertain_goals

        
    elif goal_type == "obs":

        agent0_goal_obs = np.ones(12)
        agent0_goal_obs[2] = rp
        agent0_goal_obs[3] = pp
        agent0_goal_obs = agent0_goal_obs/pp
        agent0_goal_obs_cat = agent0_goal_obs/np.sum(agent0_goal_obs)

        agent1_goal_obs = np.ones(12)
        agent1_goal_obs[9] = rp
        agent1_goal_obs[10] = pp
        agent1_goal_obs = agent1_goal_obs/pp
        agent1_goal_obs_cat = agent1_goal_obs/np.sum(agent1_goal_obs)

        
        assert(np.min(agent0_goal_obs) == 1)
        assert(np.min(agent1_goal_obs) == 1)
        
        
        agent0_fullgoals = [agent0_goal_obs_cat, agent1_goal_obs]
        agent1_fullgoals = [agent1_goal_obs_cat, agent0_goal_obs]

        if uncertain_goals == True:
            uncertain_goals = np.ones(11)
            agent0_fullgoals[1] = uncertain_goals
            agent1_fullgoals[1] = uncertain_goals

    
    else:
        assert 1/0, "Goal type not correctly specified fella"


    return agent0_fullgoals, agent1_fullgoals
    



def forward(obs, acts, start, tm, gm):
    T = len(obs)
    num_states = start.shape[0]
    forward_pass = np.zeros((T+1,num_states))
    forward_pass[0,:] = start

    for i in range(T):
        forward_pass[i+1] = gm[obs[i]] * (tm[acts[i]] @ forward_pass[i])
    
    return forward_pass


def backward(obs, acts, start, tm, gm):
    T = len(obs)
    num_states = start.shape[0]
    backward_pass = np.zeros((T+1,num_states))
    backward_pass[-1] = start

    for i in reversed(range(T)):
        backward_pass[i] = tm[acts[i]].T @ (gm[obs[i]] * backward_pass[i+1])
    
    return backward_pass


def Estep(obs_eps, acts_eps, start, end, tm, gm):

    num_eps = len(obs_eps)
    num_visits = len(obs_eps[0])+1
    num_states = gm.shape[1]
    
    #inf_states_eps = np.zeros((num_eps, num_visits, num_states))
    inf_states_eps = []

    for i, (obs, acts) in enumerate(zip(obs_eps, acts_eps)):
        
        fp = forward(obs, acts, start, tm, gm)
        bp = backward(obs, acts, end, tm, gm)
        unnorm_inf_states = fp*bp

        inf_states = unnorm_inf_states/np.sum(unnorm_inf_states, axis = 1, keepdims = True)

       
        inf_states_eps.append(inf_states)

    return inf_states_eps


def Mstep(inf_states_eps, obs_eps, flat_gm_shape):

    gm = np.zeros(flat_gm_shape)
    
    for inf_states, obs in zip(inf_states_eps, obs_eps): 
        for i in range(inf_states.shape[0]-1):
            
            gm[obs[i]] = gm[obs[i]] + inf_states[i+1]

    gm = gm+1e-6

    norm_gm = gm/np.sum(gm, axis = 0, keepdims = True)

    return norm_gm



def unnorm_Mstep(inf_states_eps, obs_eps, flat_gm_shape):

    gm = np.zeros(flat_gm_shape)
    
    for inf_states, obs in zip(inf_states_eps, obs_eps): 
        for i in range(inf_states.shape[0]-1):
            gm[obs[i]] = gm[obs[i]] + inf_states[i+1]

    unnorm_gm = gm+1e-12

    return unnorm_gm


def EM(obs_eps, acts_eps, tm_unres, A_param_shape, s_param_shape, gm = None, start = None, end = None):

    
    tm = np.zeros((tm_unres.shape[0],2*tm_unres.shape[1], 2*tm_unres.shape[2]))

    for i in range(tm_unres.shape[0]):
        tm[i] = np.block([[tm_unres[i], np.zeros(tm_unres[i].shape)],
                            [np.zeros(tm_unres[i].shape), tm_unres[i]]])

    

        
    #np.random.seed(0)

    if start is None:
        start_sample = np.random.beta(10,1,s_param_shape) #in case I don't want to use init
        start = start_sample.reshape(-1)
        start = start/np.sum(start)

    else:
        start_sample = start
        start = start.reshape(-1)

    
    if end is None:
        end_sample = np.random.beta(10,1,s_param_shape) #in case I don't want to use init
        end = end_sample.reshape(-1)
        end = end/np.sum(end)

    else:
        end_sample = end
        end = end.reshape(-1)

    
    if gm is None:
        
        """
        num_params = A_param_shape[0]
        num_states = np.prod(A_param_shape[1:])

        gm = np.zeros((num_params, num_states))

        alphas = np.ones(num_params)*2

        for i in range(num_states):
            gm[:,i] = np.random.dirichlet(alphas)"""

        gm_sample = np.random.beta(10,1, A_param_shape)
        gm = gm_sample/np.sum(gm_sample, axis = 0, keepdims = True)

        assert np.allclose(np.sum(gm, axis = 0),1)

        gm_shape = gm.shape

        gm = gm.reshape(gm.shape[0], -1)
        flat_gm_shape = gm.shape

    else:
        gm_sample = gm
        gm = gm
        gm = gm/np.sum(gm, axis = 0, keepdims = True)

        gm_shape = gm.shape

        gm = gm.reshape(gm.shape[0],-1)

        flat_gm_shape = gm.shape


    agent_gm = gm.copy() #which dirichlet GM the agent updates, for now the random start

    gm_old = np.zeros(gm.shape)
    #for _ in range(num_steps):
    while np.linalg.norm(gm - gm_old) > 0.001:
        gm_old = gm.copy()
        
        inf_states = Estep(obs_eps, acts_eps, start, end, tm, gm)

        gm = Mstep(inf_states, obs_eps, flat_gm_shape)


    gm = unnorm_Mstep(inf_states, obs_eps, flat_gm_shape)

    return gm, [start_sample, end_sample, gm_sample]


 
 
def forward2(obs, acts, start, tm, gm):
    T = len(obs)
    forward_pass = np.zeros((T+1,2,2,4,4))
    forward_pass[0,...] = start

    for i in range(T):
        if acts[i] < 4:
            forward_pass[i+1] = gm[obs[i]] * (np.einsum("mk, ijkl -> ijml", tm[acts[i]], forward_pass[i]))
        else:
            forward_pass[i+1] = gm[obs[i]] * (np.einsum("ml, ijkl -> ijkm", tm[acts[i]-4], forward_pass[i]))
    
    return forward_pass


def backward2(obs, acts, start, tm, gm):
    T = len(obs)
    backward_pass = np.zeros((T+1,2,2,4,4))
    backward_pass[-1,...] = start

    for i in reversed(range(T)):
        if acts[i] < 4:
            backward_pass[i] = np.einsum("mk, ijkl -> ijml", tm[acts[i]].T, gm[obs[i]] * backward_pass[i+1])
        else:
            backward_pass[i] = np.einsum("ml, ijkl -> ijkm", tm[acts[i]-4].T, gm[obs[i]] * backward_pass[i+1])
    
    return backward_pass


def Estep2(obs_eps, acts_eps, start, end, tm, gm):

    
    inf_states_eps = []

    for i, (obs, acts) in enumerate(zip(obs_eps, acts_eps)):
        
        fp = forward2(obs, acts, start, tm, gm)
        bp = backward2(obs, acts, end, tm, gm)
        unnorm_inf_states = fp*bp


        norm_c = np.einsum("ijkl -> ", unnorm_inf_states[0])
        
        assert np.allclose(np.einsum("ijklo -> i", unnorm_inf_states), norm_c, rtol=1e-4)

        inf_states = unnorm_inf_states/norm_c#summed_values

        inf_states_eps.append(inf_states)

    return inf_states_eps


def Mstep2(inf_states_eps, obs_eps, gm_shape):

    gm = np.zeros(gm_shape)
    
    for inf_states, obs in zip(inf_states_eps, obs_eps): 
        for i in range(inf_states.shape[0]-1):
            
            gm[obs[i]] = gm[obs[i]] + inf_states[i+1]

    gm = gm+1e-6

    norm_gm = gm/np.sum(gm, axis = 0, keepdims = True)

    return norm_gm



def unnoised_Mstep2(inf_states_eps, obs_eps, gm_shape):

    gm = np.zeros(gm_shape)
    
    for inf_states, obs in zip(inf_states_eps, obs_eps): 
        for i in range(inf_states.shape[0]-1):
            gm[obs[i]] = gm[obs[i]] + inf_states[i+1]

    unnorm_gm = gm/np.sum(gm, axis = 0, keepdims = True)

    return unnorm_gm


def EM2(obs_eps, acts_eps, tm, A_param_shape, s_param_shape, gm = None, start = None, end = None):

    if start is None:
        start = np.random.beta(10,1,s_param_shape) #in case I don't want to use init
        start = start/np.sum(start)

    else:
        start = start

    if end is None:
        end = np.random.beta(10,1,s_param_shape) #in case I don't want to use init
        end = end/np.sum(end)

    else:
        end = end

    if gm is None:
        gm_sample = np.random.beta(10,1, A_param_shape)
        gm = gm_sample/np.sum(gm_sample, axis = 0, keepdims = True)
        assert np.allclose(np.sum(gm, axis = 0),1)
        gm_shape = gm.shape

    else:
        gm = gm
        gm_shape = gm.shape




    agent_gm = gm.copy() #which dirichlet GM the agent updates, for now the random start

    gm_old = np.zeros(gm.shape)
    while np.linalg.norm(gm - gm_old) > 0.001:
        gm_old = gm.copy()
        
        inf_states = Estep2(obs_eps, acts_eps, start, end, tm, gm)

        gm = Mstep2(inf_states, obs_eps, gm_shape)


    gm = unnoised_Mstep2(inf_states, obs_eps, gm_shape)

    return gm, [start, end, gm]

 

def dir_VI(logl, data, prior, goal_learning_vars = dict, opt_params = dict):

    natgrad = opt_params["natgrad"]
    log_space = opt_params["log_space"]
    lambd = opt_params["reg_strength"]
    beta1 = opt_params["beta1"]
    beta2 = opt_params["beta2"]
    num_iters = opt_params["num_iters"]
    num_samples = opt_params["num_samples"]
    optimizer = opt_params["optimizer"]
    clipping = opt_params["clipping"]
    clips = opt_params["clips"]
    step_size = opt_params["step_size"]



    dim_params = prior.shape[0]

    var = prior#np.random.uniform(0.5,2,dim_params)

    sump = np.sum(prior)

    for j in range(num_iters):
        #print("-------------------------------------------")

        sumv = np.sum(var)

        #print(f"This is the sum of the dir parameters of the goals: {sumv}")

        kl_grad = - sp.polygamma(1, sumv)*(sumv-sump) + (var-prior)*sp.polygamma(1, var)

        samples = np.zeros((num_samples, dim_params))

        for i in range(num_samples):
            sample = np.random.dirichlet(var)
            score = -sp.polygamma(0, var) + sp.polygamma(0, sumv) + np.log(sample)
            samples[i] = logl(data, sample) * score

        exp_grad = np.sum(samples, axis = 0)/num_samples

        """#With baseline
        logls = np.zeros(num_samples)
        scores = np.zeros((num_samples, dim_params))

        for i in range(num_samples):
            sample = np.random.dirichlet(var)
            logls[i] = logl(data, sample)
            scores[i] = -sp.polygamma(0, var) + sp.polygamma(0, sumv) + np.log(sample)

        baseline = np.mean(logls)

        exp_grad = np.mean((logls - baseline)[:, None] * scores, axis=0)"""
        

        grad = exp_grad - kl_grad

        """#old version
        grad = np.clip(grad, -1, 1)
        var = var + grad*step_size
        var = np.clip(var, 0, 10)"""


        if natgrad == True: #In case I want to compute the gradients in natural parameter space
            d = sp.polygamma(1, var)
            c = sp.polygamma(1, sumv)

            g_over_d = grad / d
            correction = c / (1.0 + c * np.sum(1.0 / d))

            nat_grad = g_over_d - correction * np.sum(g_over_d)
            grad = nat_grad

        
        


        if log_space == True:

    
            phi = np.log(var)
            grad = grad * var
            grad = grad - lambd * phi #regularization


            m = goal_learning_vars["m"]
            v = goal_learning_vars["v"]
            t = goal_learning_vars["t"]


            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            goal_learning_vars["m"] = m
            goal_learning_vars["v"] = v
            goal_learning_vars["t"] = t+1

            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            if optimizer == "momentum":
                opt_grad = grad*m_hat

            elif optimizer == "adam_softstart":
                opt_grad = grad*m/(np.sqrt(v_hat) + 1e-5)

            elif optimizer == "momentum_softstart":
                opt_grad = grad*m

            elif optimizer == "adam":
                opt_grad = grad*m_hat/(np.sqrt(v_hat) + 1e-5)

            elif optimizer == None:
                opt_grad = grad
            
            else:
                assert 1/0, "What?? THERE IS NO OTHER OPTIMIZER"


            if clipping == True:
                grad = np.clip(grad, clips[0], clips[1])
            
            #print(f"These are the pre opt gradients: \n {grad} ")
            #print(f"this are the post opt gradients: \n {opt_grad}")

            
            #print(f"These are the old dirs in exp_space:\n  {phi}")
            phi = phi + step_size * grad
            #print(f"These are the new dirs in exp space: \n {phi}")
            var = np.exp(phi)
            
        else:
            assert 1/0, "Can do only things in exp space for now"


    

    
    return var, goal_learning_vars



def logl_cat(x, theta):

    value = np.sum(np.log(theta[x]))

    return value


def logl_action(a, agent, sampled_goal):

    gucp = agent.goal_states.copy()

    agent.change_goals(new_goal_states = sampled_goal, who = 1)

    actions, action_probs = agent.SFE(1)

    agent.goal_states = gucp.copy()

    if a in actions:
        idx = np.where(actions == a)[0][0]
        action_prob = action_probs[idx]

    else:
        action_prob = 0
    
    return np.log(action_prob + 1e-10)



def A_KL(A, learned_A):
    """
    A = (num_obs, ...)
    learned_A = (num_obs, ...) has to be normalized
    """

    learned_A = norm_om(flatten_om(learned_A))
    A = flatten_om(A)

    assert np.allclose(np.sum(A, axis = 0),1), f"The columns of A sum to {np.sum(A, axis = 0)}"

    kl = np.where(A != 0, A *np.log(np.divide(A, learned_A)), 0)
    return np.sum(kl)




def A_KL_plotter(true_A, learned_As, best_action_count):

    num_eps = len(learned_As)
    
    kls = []
    var = []

    for i in range(num_eps):

        kls.append(A_KL(true_A, state_switcher(learned_As[i], true_A)))
        var.append(compute_norm_cov_A(learned_As[i]))


    fig, ax = plt.subplots()
    ax.plot(kls, label = "KL between true OM and learned OM", linestyle = "-", color = "blue")
    ax.plot(np.zeros(num_eps), linestyle = "--")
    ax.plot(best_action_count, label = "Best action taken count", linestyle = "-.")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("KL & action count")

    ax1 = ax.twinx()
    ax1.plot(var, label = "Variances of the learned OM", linestyle = "-", color = "green")
    ax1.set_ylabel("Norm of OM covariances")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)
    plt.show()

    
def A_diff_plotter(true_A, learned_A):
    A = norm_om(flatten_om(learned_A))
    true_A = flatten_om(true_A)

    fig, ax = plt.subplots(1,2)
    
    im1 = ax[0].imshow(true_A,cmap ="viridis", vmin = 0, vmax = 1)
    ax[0].set_title("True A")
    im2 = ax[1].imshow(A,cmap = "viridis", vmin = 0, vmax = 1)
    ax[1].set_title("Average of final learned A")


    plt.colorbar(im1, ax=ax[0])
    plt.colorbar(im2, ax=ax[1])

    #plt.title("Comparison of the true OM and the average of the final learned OM")
    plt.show()


def efe_eps_plotter(efe_eps):
    efe_eps = np.array(efe_eps).T


    fig, ax = plt.subplots()
    labels = ["EFE down", "EFE up", "EFE left", "EFE right"]
    for i in range(4):
        ax.plot(efe_eps[i], label = labels[i])
    ax.legend()
    ax.set_xlabel("Episodes")
    ax.set_ylabel("EFE")
    plt.show()
    





def plotter(true_A, learned_As_eps, best_action_count):

    true_A_state = np.reshape(true_A, (5,8))
    true_A_invstate = np.zeros(true_A_state.shape)
    true_A_invstate[:,0:4] = true_A_state[:,4:8]
    true_A_invstate[:,4:8] = true_A_state[:,0:4] 

    kl_eps = []
    learned_As_eps_end = []
    var_eps = []
    for learned_As in learned_As_eps:

        kl = []
        var = []
        for A in learned_As:
            kl_state = A_KL(true_A_state, A)
            kl_invstate = A_KL(true_A_invstate, A)
            kl.append(min(kl_state, kl_invstate))

            var.append(np.linalg.norm(compute_cov_A(A)))


        if kl[-1] == kl_state: #states have not been switched
            learned_As_eps_end.append(A)
        else:
            temp = np.zeros(A.shape)
            temp[:,0:4] = A[:,4:8]
            temp[:,4:8] = A[:,0:4]
            learned_As_eps_end.append(temp)

        
        var_eps.append(var)
        kl_eps.append(kl)
    
    final_As = np.array(learned_As_eps_end)
    final_As_mean = np.mean(final_As, axis = 0, keepdims = False)
    #print(np.round(final_As_mean, 2))

    data = np.array(kl_eps)
    assert data.all() >= 0, "KL divergences should be positive"
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)

    var_data = np.array(var_eps)
    mean_var = np.mean(var_data, axis = 0)
    std_var = np.std(var_data, axis = 0)


    x = np.arange(len(kl))

    fig, ax1 = plt.subplots()


    ax1.plot(x, mean, label = "Mean KL divergence")
    ax1.fill_between(x, mean - std, mean + std, alpha = 0.2, label = "Std Dev")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("KL divergence")

    ax2 = ax1.twinx()
    ax2.plot(x, mean_var, linestyle="--", label = "Norm of the sum of the covariance matrices")
    ax2.fill_between(x, mean_var - std_var, mean_var + std_var, alpha = 0.2, label = "Std Dev")
    ax2.set_ylabel("Covariance norm")

    ax1.plot(x, best_action_count, linestyle=':', label="Best action count")


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)


    plt.title("KL divergence between true A and learned A over episodes")
    plt.xticks(np.arange(0, len(kl), step = 5), np.arange(0, len(kl), step = 5))
    plt.ylim(0,13)
    plt.show()




    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    im0 = axes[0].imshow(true_A_state, vmin=0, vmax=1)
    axes[0].set_title("True observation model A")

    im1 = axes[1].imshow(final_As_mean, vmin=0, vmax=1)
    axes[1].set_title("Average final learned observation model A")

    # Shared colorbar
    fig.colorbar(im1, ax=axes.ravel().tolist())

    plt.show()



if __name__ == "_main__":
    num_states = 10
    prior = np.ones(num_states)*10
    x = np.array([0,1,2,3,4,4])
    num_iters = 400
    num_samples = 300
    step_size = 0.01

    post = dir_VI(logl, x, num_iters, num_samples, prior, step_size, natgrad = False)


    print(np.round(post,2))

    1/0


if __name__ == "__a test__":

    x = np.array([[1,1,1,1,0,0,0.5,0.3],
                    [1,1,1,1,0,0.2,0,0.4],
                    [1,1,1,1,0,0,0,0],
                    [0,0.5,0,0,1,1,1,1]])

    
    lx = np.sum(x[:,0:4], axis = 1)
    rx = np.sum(x[:,4:8], axis = 1)

    conditions = lx>rx
 

    for cond, state in zip(conditions, x):
        if cond:
            state[4:] = 0
        else:
            state[:4] = 0

    print(x)







if __name__ == "__lol__":

    tm = np.array(([[[0,0,0,0],
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



    gm = np.random.uniform(0,1,(5,8))
    gm = gm/np.sum(gm, axis = 0, keepdims = True)

    """gm = np.ones((5,8))
    gm[0,1] = 10
    gm[1,5] = 10
    gm[2,2] = 10
    gm[2,7] = 10
    gm[3,3] = 10
    gm[3,6] = 10
    gm[4,0] = 10
    gm[4,4] = 10"""

    gm = gm/np.sum(gm, axis = 0, keepdims = True)

    acts = [[0,1,2,3],
            [0,1,3,2],
            [0,1,2,3],
            [0,1,3,2]]

    obs = [[0,4,2,4],
        [1,4,2,4],
        [1,4,3,4],
        [0,4,3,4]]

    num_steps = 10000

    gm = EM(obs, acts, tm, gm, None, None, num_steps)

    

    print(np.round(gm,0))

    #gm = EM(obs, acts, tm, gm, None, None, num_steps)



    #print(np.round(gm,0))



    
