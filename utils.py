import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

#np.set_printoptions(precision = 2)



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

    
    inf_states_eps = np.zeros((num_eps, num_visits, num_states))

    for i, (obs, acts) in enumerate(zip(obs_eps, acts_eps)):
        
        fp = forward(obs, acts, start, tm, gm)
        bp = backward(obs, acts, end, tm, gm)
        unnorm_inf_states = fp*bp

        inf_states = unnorm_inf_states/np.sum(unnorm_inf_states, axis = 1, keepdims = True)

        """
        lx = np.sum(inf_states[:,0:4], axis = 1)
        rx = np.sum(inf_states[:,4:8], axis = 1)

        conditions = lx>rx
 

        for cond, state in zip(conditions, inf_states):
            if cond:
                state[4:] = 0
            else:
                state[:4] = 0"""


        inf_states_eps[i] = inf_states

    return inf_states_eps

def Mstep(inf_states_eps, gm, obs_eps):
    
    for inf_states, obs in zip(inf_states_eps, obs_eps): 
        for i in range(inf_states.shape[0]-1):
            
            gm[obs[i]] = gm[obs[i]] + inf_states[i+1]

    gm = gm+1e-6

    norm_gm = gm/np.sum(gm, axis = 0, keepdims = True)

    return norm_gm



def unnorm_Mstep(inf_states_eps, gm, obs_eps):
    
    for inf_states, obs in zip(inf_states_eps, obs_eps): 
        for i in range(inf_states.shape[0]-1):
            gm[obs[i]] = gm[obs[i]] + inf_states[i+1]

    unnorm_gm = gm

    return unnorm_gm


def EM(obs_eps, acts_eps, tm_unres, num_steps, gm = None, start = None, end = None):

    
    tm = np.zeros((tm_unres.shape[0],2*tm_unres.shape[1], 2*tm_unres.shape[2]))

    for i in range(tm_unres.shape[0]):
        tm[i] = np.block([[tm_unres[i], np.zeros(tm_unres[i].shape)],
                            [np.zeros(tm_unres[i].shape), tm_unres[i]]])

    
    #np.random.seed(0)

    if start is None:
        start = np.random.beta(10,1,(4,2)) #in case I don't want to use init
        start = start.reshape(-1)
        start = start/np.sum(start)

    else:
        start = start.reshape(-1)

    
    if end is None:
        end = np.random.beta(10,1,(4,2)) #in case I don't want to use init
        end = end.reshape(-1)
        end = end/np.sum(end)

    else:
        end = end.reshape(-1)

    
    if gm is None:
        gm = np.random.beta(10,1,(5,4,2))
        gm = gm/np.sum(gm, axis = 0, keepdims = True)
        gm_shape = gm.shape
        gm = gm.reshape(gm.shape[0], -1)

    else:
        gm = gm
        gm = gm/np.sum(gm, axis = 0, keepdims = True)
        gm_shape = gm.shape
        gm = gm.reshape(gm.shape[0],-1)


    agent_gm = gm.copy() #which dirichlet GM the agent updates, for now the random start

    gm_old = np.zeros(gm.shape)
    #for _ in range(num_steps):
    while np.linalg.norm(gm - gm_old) > 0.001:
        gm_old = gm.copy()
        
        inf_states = Estep(obs_eps, acts_eps, start, end, tm, gm)

        gm = Mstep(inf_states, gm, obs_eps)


    gm = unnorm_Mstep(inf_states, agent_gm, obs_eps)

    reshaped_gm = np.reshape(gm,gm_shape, order = "F")

    return reshaped_gm


 
 

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
    kl = np.where(A != 0, A *np.log(np.divide(A, learned_A)), 0)
    return np.sum(kl)

def A_KL_plotter(true_A, learned_As_eps):

    true_A_state = np.reshape(true_A, (5,8), order = "F")
    true_A_invstate = np.zeros(true_A_state.shape)
    true_A_invstate[:,0:4] = true_A_state[:,4:8]
    true_A_invstate[:,4:8] = true_A_state[:,0:4] 
    kl_eps = []
    learned_As_eps_end = []
    for learned_As in learned_As_eps:

        kl = []
        for A in learned_As:
            kl_state = A_KL(true_A_state, A)
            kl_invstate = A_KL(true_A_invstate, A)
            kl.append(min(kl_state, kl_invstate))

        if kl[-1] == kl_state: #states have not been switched
            learned_As_eps_end.append(A)
        else:
            temp = np.zeros(A.shape)
            temp[:,0:4] = A[:,4:8]
            temp[:,4:8] = A[:,0:4]
            learned_As_eps_end.append(temp)

        kl_eps.append(kl)
    
    final_As = np.array(learned_As_eps_end)
    final_As_mean = np.mean(final_As, axis = 0, keepdims = False)
    #print(np.round(final_As_mean, 2))

    data = np.array(kl_eps)
    assert data.all() >= 0, "KL divergences should be positive"
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    x = np.arange(len(kl))

    plt.figure()
    plt.plot(x, mean, label = "Mean KL divergence")
    plt.fill_between(x, mean - std, mean + std, alpha = 0.2, label = "Std Dev")
    plt.xlabel("Episode")
    plt.ylabel("KL divergence")
    plt.title("KL divergence between true A and learned A over episodes")
    plt.xticks(np.arange(0, len(kl), step = 5), np.arange(0, len(kl), step = 5))
    plt.ylim(0,13)
    plt.legend()
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



    
