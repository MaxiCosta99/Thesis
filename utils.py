import numpy as np
import scipy.special as sp

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

    norm_gm = gm/np.sum(gm + 0.01, axis = 0, keepdims = True)

    return norm_gm



def unnorm_Mstep(inf_states_eps, gm, obs_eps):
    
    for inf_states, obs in zip(inf_states_eps, obs_eps): 
        for i in range(inf_states.shape[0]-1):
            gm[obs[i]] = gm[obs[i]] + inf_states[i+1]

    unnorm_gm = gm

    return unnorm_gm


def EM(obs_eps, acts_eps, tm_unres, gm, start, end, num_steps):

    
    tm = np.zeros((tm_unres.shape[0],2*tm_unres.shape[1], 2*tm_unres.shape[2]))

    for i in range(tm_unres.shape[0]):
        tm[i] = np.block([[tm_unres[i], np.zeros(tm_unres[i].shape)],
                            [np.zeros(tm_unres[i].shape), tm_unres[i]]])

    
    #np.random.seed(0)

    
    start = np.random.beta(10,1,(4,2)) #in case I don't want to use init
    start = start.reshape(-1)+0.01
    start = start/np.sum(start)

    end = np.random.beta(10,1,(4,2)) #in case I don't want to use init
    end = end.reshape(-1)+0.01 
    end = end/np.sum(end)

    
    gm = np.random.beta(10,1,(5,4,2))
    gm = gm/np.sum(gm, axis = 0, keepdims = True)

    gm_shape = gm.shape
    


    flat_gm = gm.reshape(gm.shape[0], -1)
    EM_gm = flat_gm/np.sum(flat_gm+0.01, axis = 0, keepdims = True)

    agent_gm = EM_gm.copy() #which dirichlet GM the agent updates, for now the random start


    #test_gm = np.zeros(gm_shape)

    for _ in range(num_steps):
        
        inf_states = Estep(obs_eps, acts_eps, start, end, tm, EM_gm)

        EM_gm = Mstep(inf_states, EM_gm, obs_eps)

    unnorm_gm = unnorm_Mstep(inf_states, agent_gm, obs_eps)

    reshaped_gm = unnorm_gm.reshape(gm_shape)

    return reshaped_gm


 
 


def dir_VI(logl, data, num_iters, num_samples, prior, step_size, natgrad):

    dim_params = prior.shape[0]

    var = prior#np.random.uniform(0.5,2,dim_params)

    sump = np.sum(prior)

  
    for _ in range(num_iters):

        
        print("I DID NOT CRASH YET")

        sumv = np.sum(var)

        print(f"This is the sum of the dir parameters of the goals: {sumv}")

        kl_grad = - sp.polygamma(1, sumv)*(sumv-sump) + (var-prior)*sp.polygamma(1, var)

        samples = np.zeros((num_samples, dim_params))

        for i in range(num_samples):
            sample = np.random.dirichlet(var)
            score = -sp.polygamma(0, var) + sp.polygamma(0, sumv) + np.log(sample)
            samples[i] = logl(data, sample) * score

        exp_grad = np.sum(samples, axis = 0)/num_samples

        """ With baseline
        logls = np.zeros(num_samples)
        scores = np.zeros((num_samples, dim_params))

        for i in range(num_samples):
            sample = np.random.dirichlet(var)
            logls[i] = logl(data, sample)
            scores[i] = -sp.polygamma(0, var) + sp.polygamma(0, sumv) + np.log(sample)

        baseline = np.mean(logls)

        exp_grad = np.mean((logls - baseline)[:, None] * scores, axis=0)
        """

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
            

        phi = np.log(var)

        grad_phi = grad * var
        print(f"this is the gradient: {grad_phi}")
        grad_phi = np.clip(grad_phi, -1, 1)
        phi = phi + grad_phi*step_size

        var = np.exp(phi)

    
    return var



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



    
