import numpy as np

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


def Estep(obs, acts, start, end, tm, gm):

    fp = forward(obs, acts, start, tm, gm)
    bp = backward(obs, acts, end, tm, gm)

    unnorm_inf_states = fp*bp
    inf_states = unnorm_inf_states/np.sum(unnorm_inf_states, axis = 1, keepdims = True)

    return inf_states

def Mstep(inf_states, gm, obs):
    

    for i in range(inf_states.shape[0]-1):
        gm[obs[i]] = gm[obs[i]] + inf_states[i+1]

    norm_gm = gm/np.sum(gm, axis = 0, keepdims = True)

    return norm_gm


def EM(obs, acts, start, end, tm, gm, num_steps):

    flattened_gm = gm.reshape(gm.shape[0], -1)
    gm = flattened_gm/np.sum(flattened_gm, axis = 0, keepdims = True)


    for _ in range(num_steps):
        
        inf_states = Estep(obs, acts, start, end, tm, gm)

        gm = Mstep(inf_states, gm, obs)


    return gm



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



tm = np.zeros((start_B.shape[0],2*start_B.shape[1], 2*start_B.shape[2]))


for i in range(start_B.shape[0]):
    tm[i] = np.block([[start_B[i], np.zeros(start_B[i].shape)],
                        [np.zeros(start_B[i].shape), start_B[i]]])


start = np.random.uniform(0,1,8)
start = start/np.sum(start)

end = np.random.uniform(0,1,8)
end = end/np.sum(end)

"""gm = np.random.uniform(0,1,(5,8))
gm = gm/np.sum(gm, axis = 0, keepdims = True)"""

gm = np.ones((5,8))
gm[0,1] = 10
gm[1,5] = 10
gm[2,2] = 10
gm[2,7] = 10
gm[3,3] = 10
gm[3,6] = 10
gm[4,0] = 10
gm[4,4] = 10

gm = gm/np.sum(gm, axis = 0, keepdims = True)

acts = [0,1,2,3,
        0,1,3,2,
        0,1,2,3,
        0,1,3,2]

obs = [0,4,2,4,
       1,4,2,4,
       0,4,3,4,
       1,4,3,4]

num_steps = 1000

gm = EM(obs, acts, start, end, tm, gm, num_steps)



print(np.round(gm,2))



    
