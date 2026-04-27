import numpy as np
from utils import EM, dir_VI, A_KL, state_switcher

def softmax(x, beta = 1):
  return np.exp(-x/beta)/np.sum(np.exp(-x/beta))


def pos_conv(pos):

    agent1pos = pos[:,0]
    agent2pos = pos[:,1]
    conv1 = int(np.squeeze(np.where(agent1pos == 1)))
    conv2 = int(np.squeeze(np.where(agent2pos == 1)))

    conv = [conv1, conv2]

    mapping = {
            0: (0, 1),
            1: (1, 1),
            2: (0, 0),
            3: (0, 2)
        }
    
    for i in range(2):
      conv[i] = mapping.get(conv[i])
 
    return conv

class agent:
  def __init__(self, A, B, s, U,
               goal_states, goal_obs, reason_depth = int, dec_temp = float, chatty_boy = bool,
               little_o_imaginer = bool, o_prob_threshold = float,
               little_u_imaginer = bool, u_prob_threshold = float,
               agent_id = int, moves_to_go = int, social = bool, terms = list, sampling = str,
               branch_reductor = str, mc_samples = int, selfish = bool, empathetic = bool,
               bayes_observe = bool, obs_model_given = bool, futuro_prosperoso = bool,
               random_moves = bool, det_dec_rule = bool, state_stabilizer = bool):
    if  obs_model_given == True:
        self.A = A
    else:
        my_A = np.random.beta(10,1, (11,2,2,4,4)) #my observation model (a catagorical)
        your_A = np.ones((11,2,2,4,4)) #your observation model (a dirichlet)
        self.A = np.stack((my_A, your_A), axis = 0)

    self.s = s #[my/your belief, agent0cheese, agent1cheese, agent0pos, agetn1pos]
    self.B = B
    self.U = U
    self.goal_states = goal_states #[my/your]
    self.goal_obs = goal_obs
    self.reason_depth = reason_depth
    self.chatty_boy = chatty_boy
    self.dec_temp = dec_temp
    self.little_o_imaginer = little_o_imaginer
    self.o_prob_threshold = o_prob_threshold
    self.little_u_imaginer = little_u_imaginer
    self.u_prob_threshold = u_prob_threshold


    self.agent_id = agent_id
    self.moves_to_go = moves_to_go
    self.social = social
    self.terms = terms
    self.sampling = sampling
    self.branch_reductor = branch_reductor
    self.mc_samples = mc_samples
    self.selfish = selfish
    self.empathetic = empathetic

    self.goal_learning_vars = {"m": 0,
                              "v": 0,
                              "t": 1
                              }

    self.bayes_observe = bayes_observe
    self.futuro_prosperoso = futuro_prosperoso



    self.true_A = A

    self.state_stabilizer = state_stabilizer

    if len(self.terms) == 0:
        self.random_moves = True
    else:
        self.random_moves = False

    if self.dec_temp == 0:
        self.det_dec_rule = True
    else:
        self.det_dec_rule = False


  def acting_randomly(self, new_action_type):
      self.random_moves = new_action_type
  
  def restart_agent(self, start_s, reason_depth, nr_moves):
      self.s = start_s.copy()
      self.reason_depth = reason_depth
      self.moves_to_go = nr_moves

  def sample(self, dcm):
      """
      dcm is dirichlet counts matrix
      """
      dcm_shape = dcm.shape
      dcm_flat = dcm.reshape(-1)
      
      if self.sampling == "mean":
          sample = dcm_flat/np.sum(dcm_flat)

      elif self.sampling == "sample":
          sample = np.random.dirichlet(dcm_flat)
      
      out = sample.reshape(dcm_shape)
      return out

  def SFE_call_other(self, sampled_goal = None):

      gcp = self.goal_states.copy()

      #print(sampled_goal.shape)
      if sampled_goal is None:

          if self.selfish == True:

              if self.agent_id == 0:
                  filtered_goals = np.mean(self.goal_states[1].copy(), axis = (0,2)) #np.einsum("ijkl -> jl", self.goal_states[1].copy()) 
                  #filtered_states = np.einsum("ijkl -> jl", self.s[1].copy())
              else:
                  filtered_goals =  np.mean(self.goal_states[1].copy(), axis = (1,3)) #np.einsum("ijkl -> ik", self.goal_states[1].copy()) 
                  #filtered_states = np.einsum("ijkl -> ik", self.s[1].copy())
          else:
              filtered_goals = self.goal_states[1].copy()
              #filtered_states = self.s[1].copy()
        
          sampled_goal = self.sample(filtered_goals).copy()
          #whose_s = filtered_states
      
      else:
          
          if self.selfish == True:
              sampled_goal = sampled_goal.reshape(self.goal_states[1].shape[1], self.goal_states[1].shape[3])
              

      #sampled_goal = self.sample(self.goal_states.copy())
      
      self.change_goals(new_goal_states = sampled_goal, who = 1)


      actions, actions_H = self.SFE(1)

      self.goal_states = gcp.copy()
      
      return actions, actions_H


  def SFE_call_self(self):

      gcp = self.goal_states.copy()

      
      if self.selfish == True:

          if self.agent_id == 0:
              filtered_goals = np.einsum("ijkl -> ik", self.goal_states[0].copy()) #np.mean(self.goal_states[0].copy(), axis = (1,3)) 
          else:
              filtered_goals = np.einsum("ijkl -> jl", self.goal_states[0].copy()) #np.mean(self.goal_states[0].copy(), axis = (0,2)) 
      else:
          filtered_goals = self.goal_states[0].copy()
        
       
      self.change_goals(new_goal_states = filtered_goals, who = 0)

      actions, actions_H = self.SFE(0)

      self.goal_states = gcp.copy()
      
      return actions, actions_H


  def update_A(self, ep_observations, ep_actions, agent_id):
    assert 1/0, f"Update A is DEAD"
    if self.update_A_method == "default":
        assert 1/0, f"This is not used anymore!!!"
    
        """backup = self.s.copy()

        self.s = s.copy()
        self.observe(o,0) #infer which state we were in

        
        print(f"This is the state before observing: {s[0]}")
        print(f"This is the state after observing: {self.s[0]}")
        print(f"This is the observation: {o}")
        print(f"This is A before update: {self.A[0,o,...]}")

        self.A[who,o,...] = self.A[who,o,...].copy() + self.s[0].copy()

        #print(f"This is the updated A: {self.A[0,o,...]}")

        self.s = backup"""

    elif self.update_A_method == "EM_all_ep":
        assert 1/0, f"This method is not in use anymore"
        

  


  def change_goals(self, new_goal_states, who):
      #new_gs_res = new_goal_states.reshape(self.goal_states[who].shape)
      self.goal_states[who] = new_goal_states.copy()


  def logl_action(self, a, sampled_goal):

      actions, actions_H = self.SFE_call_other(sampled_goal)

      if a in actions:
          idx = np.where(actions == a)[0][0]
          action_H = actions_H[idx]
          action_prob = softmax(action_H, 1)

      else:
          action_prob = 0.00001 
    
      return np.log(action_prob)


  def update_goals_other(self, a, belief_state, opt_params):

      self.reason_depth = self.reason_depth + 1 #need to go back at previous timepoint to see what I would have done if I were him using my belief about his goals
      
      belief_state_cp = self.s.copy()
      self.s = belief_state.copy()


      assert self.social == True, f"Can't update goals of other agents if agent is not social"
      """#old version, not bayesian
      #self.goal_states[1] = self.goal_states[1].copy()+self.s[1].copy()
      #self.goal_states[1] = self.goal_states[1]/np.sum(self.goal_states[1])"""

      #print(f"This is the action I observed: {a}")
      #print(f"These are my current beliefs about his location: {np.einsum("ijkl -> jl", self.s[1].copy())}")
      if self.selfish == True:
          prior = np.mean(self.goal_states[1].copy(), axis = (0,2)) #np.einsum("ijkl -> jl", self.goal_states[1].copy()) #

      else:
          prior = self.goal_states[1].copy()

      prior_shape = prior.shape
      prior = prior.reshape(-1)
      #num_iters = int(np.exp(3-self.reason_depth))
      

      goal_learning_vars = self.goal_learning_vars
      new_desires, goal_learning_vars = dir_VI(self.logl_action, a, prior, goal_learning_vars, opt_params = opt_params)
      self.goal_learning_vars = goal_learning_vars

      new_desires_reshaped = new_desires.reshape(prior_shape)

      if self.selfish == True:
          temp = np.repeat(new_desires_reshaped[np.newaxis,:,:], axis = 0, repeats = 2)
          new_desires_reshaped = np.repeat(temp[:,:,np.newaxis,:], axis = 2, repeats = 4)
          new_desires_reshaped = new_desires_reshaped
      else:
          assert 1/0, f"This should not be working, WHAT THE HELL? If no selfish sampling should crash"


      self.goal_states[1] = new_desires_reshaped.copy()

      self.reason_depth = self.reason_depth - 1
      self.s = belief_state_cp.copy()



  def observe(self, o, who):

    if self.bayes_observe == True: #this uses the observation model to infer states
        whose_A = self.A[who].copy()
        whose_A_flat = whose_A.reshape(whose_A.shape[0], -1)
        whose_A_norm = whose_A_flat/np.sum(whose_A_flat, axis = 0, keepdims = True)
        whose_s = self.s[who].copy()
        assert np.allclose(np.sum(whose_s), 1)
        whose_s_shape = whose_s.shape
        whose_s_flat = whose_s.reshape(-1)
       
        temp = whose_A_norm[o,:]*whose_s_flat
        whose_s_flat = temp/np.sum(temp)

        new_s = whose_s_flat

        assert new_s.shape == whose_s_flat.shape

        new_s_orig = new_s.reshape(whose_s_shape)

        self.s[who] = new_s_orig  


    else: #This one uses hard-coded inference, has a bug for observation 7 and 8 (and maybe also 2,3)
        assert 1/0, f"This should not be used, I don't have EM version yet for two agent setting \n it's an old hard coded inference version, not very nice and buggy"
        if o == 4:

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, after {who}'s observation {o}: {np.sum(self.s[who])}"

          if self.chatty_boy == 1:
            print("No new info")


        elif o == 1:

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, before {who}'s observation {o}: {np.sum(self.s[who])}"

          whose_s = self.s[who].copy()

          whose_s[1,:,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
          whose_s[0,:,:,:] = 0

          self.s[who] = whose_s.copy()

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, after {who}'s observation {o}: {np.sum(self.s[who])}"

          if self.chatty_boy == 1:
            print("Agent 0 cheese is on the right")


        elif o == 0:

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, before {who}'s observation {o}: {np.sum(self.s[who])}"

          whose_s = self.s[who].copy()

          whose_s[0,:,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
          whose_s[1,:,:,:] = 0

          self.s[who] = whose_s.copy()

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, after {who}'s observation {o}: {np.sum(self.s[who])}"

          if self.chatty_boy == 1:
            print("Agent 0 cheese is on the left")


        elif o == 2:

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, before {who}'s observation {o}: {np.sum(self.s[who])}"

          whose_s = self.s[who].copy()
          if np.einsum("ijkl -> k", whose_s)[2] == 1:
            whose_s[0,:,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
            whose_s[1,:,:,:] = 0

          elif np.einsum("ijkl -> k", whose_s)[3] == 1:
            whose_s[1,:,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
            whose_s[0,:,:,:] = 0

          self.s[who] = whose_s.copy()

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, after {who}'s observation {o}: {np.sum(self.s[who])}"

          if self.chatty_boy == 1:
            print("Agent 0 is on the cheese")


        elif o == 3:

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, before {who}'s observation {o}: {np.sum(self.s[who])}"

          whose_s = self.s[who].copy()
          if np.einsum("ijkl -> k", whose_s)[3] == 1:
            whose_s[0,:,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
            whose_s[1,:,:,:] = 0

          elif np.einsum("ijkl -> k", whose_s)[2] == 1:
            whose_s[1,:,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
            whose_s[0,:,:,:] = 0

          self.s[who] = whose_s.copy()

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, after {who}'s observation {o}: {np.sum(self.s[who])}"


          if self.chatty_boy == 1:
            print("Agent 0 is not on the cheese")


        elif o == 5:

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, before {who}'s observation {o}: {np.sum(self.s[who])}"

          whose_s = self.s[who].copy()

          whose_s[0,:,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
          whose_s[1,:,:,:] = 0
          whose_s[:,0,:,:] = np.sum(whose_s, axis = 1, keepdims = False)
          whose_s[:,1,:,:] = 0

          self.s[who] = whose_s.copy()

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, after {who}'s observation {o}: {np.sum(self.s[who])}"

          if self.chatty_boy ==1:
            print("Agent 0 cheese is on the left and Agent 1 cheese is on the left")


          """ THIS IS FOR THE CASE WHERE EACH REVEAL CHEESE TO OTHER
          idx = np.where(self.s[who,1,:,:]>0)[0][0]
          self.s[who,1,:,:] = np.zeros(self.s.shape[1:3])
          self.s[who,1,idx,0] = 1

          if self.chatty_boy == 1:
            print("Agent 1 cheese is on the left")"""


        elif o == 6:

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, before {who}'s observation {o}: {np.sum(self.s[who])}"

          whose_s = self.s[who].copy()

          whose_s[0,:,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
          whose_s[1,:,:,:] = 0
          whose_s[:,1,:,:] = np.sum(whose_s, axis = 1, keepdims = False)
          whose_s[:,0,:,:] = 0

          self.s[who] = whose_s.copy()

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, after {who}'s observation {o}: {np.sum(self.s[who])}"

          if self.chatty_boy ==1:
            print("Agent 0 cheese is on the left and Agent 1 cheese is on the right")


          """ THIS IS FOR THE CASE WHERE EACH REVEAL CHEESE TO OTHER
          idx = np.where(self.s[who,1,:,:]>0)[0][0]
          self.s[who,1,:,:] = np.zeros(self.s.shape[1:3])
          self.s[who,1,idx,1] = 1

          if self.chatty_boy == 1:
            print("Agent 1 cheese is on the right")"""


        elif o == 7:

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, before {who}'s observation {o}: {np.sum(self.s[who])}"

          whose_s = self.s[who].copy()
          if np.einsum("ijkl -> l", whose_s)[2] == 1:
            whose_s[:,0,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
            whose_s[:,1,:,:] = 0

          elif np.einsum("ijkl -> l", whose_s)[3] == 1:
            whose_s[:,1,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
            whose_s[:,0,:,:] = 0

          self.s[who] = whose_s.copy()

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, after {who}'s observation {o}: {np.sum(self.s[who])}"

          if self.chatty_boy == 1:
            print("Agent 1 is on the cheese")


        elif o == 8:

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, before {who}'s observation {o}: {np.sum(self.s[who])}"

          whose_s = self.s[who].copy()
          if np.einsum("ijkl -> l", whose_s)[3] == 1:
            whose_s[:,0,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
            whose_s[:,1,:,:] = 0

          elif np.einsum("ijkl -> l", whose_s)[2] == 1:
            whose_s[:,1,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
            whose_s[:,0,:,:] = 0

          self.s[who] = whose_s.copy()

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, after {who}'s observation {o}: {np.sum(self.s[who])}"


          if self.chatty_boy == 1:
            print("Agent 1 is not on the cheese")


        elif o == 9:

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, before {who}'s observation {o}: {np.sum(self.s[who])}"

          whose_s = self.s[who].copy()

          whose_s[1,:,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
          whose_s[0,:,:,:] = 0
          whose_s[:,0,:,:] = np.sum(whose_s, axis = 1, keepdims = False)
          whose_s[:,1,:,:] = 0

          self.s[who] = whose_s.copy()

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, after {who}'s observation {o}: {np.sum(self.s[who])}"


          if self.chatty_boy ==1:
            print("Agent 0 cheese is on the right and Agent 1 cheese is on the left")


        elif o == 10:

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, before {who}'s observation {o}: {np.sum(self.s[who])}"

          whose_s = self.s[who].copy()

          whose_s[1,:,:,:] = np.sum(whose_s, axis = 0, keepdims = False)
          whose_s[0,:,:,:] = 0
          whose_s[:,1,:,:] = np.sum(whose_s, axis = 1, keepdims = False)
          whose_s[:,0,:,:] = 0

          self.s[who] = whose_s.copy()

          assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum, after {who}'s observation {o}: {np.sum(self.s[who])}"


          if self.chatty_boy ==1:
            print("Agent 0 cheese is on the right and Agent 1 cheese is on the right")

        #self.s = self.s/np.sum(self.s)


  def act(self, u):

    my_s = self.s[0].copy()
    mybel_his_s = self.s[1].copy()

    action_matrix = self.B[u]

    if self.agent_id == 0:
      self.s[0] = np.einsum("ijkl, mk -> ijml", my_s, action_matrix).copy()
      self.s[1] = np.einsum("ijkl, mk -> ijml", mybel_his_s, action_matrix).copy() #he knows where I go

    elif self.agent_id == 1:
      self.s[0] = np.einsum("ijkl, ml -> ijkm", my_s, action_matrix).copy()
      self.s[1] = np.einsum("ijkl, ml -> ijkm", mybel_his_s, action_matrix).copy() #he knows where I go

    else:
      assert 1/0, "WHAT THE SHID? THERE IS ANOTHER???"


    assert abs(np.sum(self.s[0])-1)<0.01, f"My states probs after my action sum to {np.sum(self.s[0])}"
    assert abs(np.sum(self.s[1])-1)<0.01, f"His states probs after my action sum to {np.sum(self.s[1])}"


  def act_other(self, u):

    my_s = self.s[0].copy()
    mybel_his_s = self.s[1].copy()

    action_matrix = self.B[u].copy()

    if self.agent_id == 0:
      self.s[0] = np.einsum("ijkl, ml -> ijkm", my_s, action_matrix).copy() #where I believe he ended up
      self.s[1] = np.einsum("ijkl, ml -> ijkm", mybel_his_s, action_matrix).copy() #where I believe he believes he ended up

    elif self.agent_id == 1:
      self.s[0] = np.einsum("ijkl, mk -> ijml", my_s, action_matrix).copy() #where I believe he ended up
      self.s[1] = np.einsum("ijkl, mk -> ijml", mybel_his_s, action_matrix).copy() #where I believe he believes he ended up

    else:
      assert 1/0, "WHAT THE SHID? THERE IS ANOTHER???"


    assert abs(np.sum(self.s[0])-1)<0.01, f"My states probs after his action sum to {np.sum(self.s[0])}"
    assert abs(np.sum(self.s[1])-1)<0.01, f"His states probs after his action sum to {np.sum(self.s[1])}"


  def restart_beliefs(self, start_s):
    self.s = start_s.copy()


  def compute_ambiguity(self,who):
    whose_A = self.A[who]
    whose_s = self.s[who].copy()
    flat_A = whose_A.reshape(whose_A.shape[0],-1)
    avg_A = flat_A/np.sum(flat_A, axis = 0, keepdims = True)
    temp = -np.diag(np.dot(avg_A.T, np.log(avg_A)))
    s_flat = whose_s.reshape(-1)
    abg = np.dot(temp.T, s_flat)
    return abg

  def compute_risk(self,who):

    if who == 1:

        if self.selfish == True:
            if self.agent_id == 0:
                #filtered_goals = np.einsum("ijkl -> jl", self.goal_states[who].copy())
                filtered_states = np.einsum("ijkl -> jl", self.s[who].copy())
            else:
                #filtered_goals = np.einsum("ijkl -> ik", self.goal_states[who].copy())
                filtered_states = np.einsum("ijkl -> ik", self.s[who].copy())
        else:
            #filtered_goals = self.goal_states[who].copy()
            filtered_states = self.s[who].copy()
        
        
        whose_goals = self.goal_states[who]
        whose_s = filtered_states


    else:

        if self.selfish == True:
            if self.agent_id == 0:
                #whose_goals = np.einsum("ijkl -> ik", self.goal_states[who].copy())
                filtered_states = np.einsum("ijkl -> ik", self.s[who].copy())
            else:
                #whose_goals = np.einsum("ijkl -> jl", self.goal_states[who].copy())
                filtered_states = np.einsum("ijkl -> jl", self.s[who].copy())
        else:
            #whose_goals = self.goal_states[who].copy()
            filtered_states = self.s[who].copy()
        
  
        whose_goals = self.goal_states[who]
        whose_s = filtered_states


    #print(f"These is {self.agent_id}'s perspective of {who} goal : {whose_goals}")

    #print(f"this is the shape of the goals in the risk: {whose_goals.shape}")

    assert whose_s.shape == whose_goals.shape, f"The shape of the states is: {whose_s.shape} \n and that of the goals is: {whose_goals.shape}"

    s_flat = whose_s.reshape(-1)
    goals_flat = whose_goals.reshape(-1)

    assert s_flat.shape == goals_flat.shape

    temp = np.divide(s_flat, goals_flat)
    temp1 = np.where(temp != 0, np.log(temp), 0)

    #print(f"These are the states: {whose_s}")
    #print(f"These are the goals: {whose_goals}")

    assert np.allclose(np.sum(whose_s), 1)
    assert np.allclose(np.sum(goals_flat), 1)
    

    #temp1 = np.log(temp, where=(temp != 0), out=np.zeros_like(temp))

    risk = np.dot(s_flat, temp1)

    #print(f"This is the risk: {risk}")
    return risk

  def compute_novelty(self,who):
      whose_A = self.A[who]
      A_flat = whose_A.reshape(whose_A.shape[0], -1)
      A_flat_norm = A_flat/np.sum(A_flat, axis = 0)
      s_flat = self.s[who].reshape(-1)

      temp0 = np.power(A_flat, -1)/2
      temp1 = np.power(np.sum(A_flat, axis = 0), -1)/2

      W = temp0-temp1

      #print(A_flat)
      #print(W)
      dot = np.dot(A_flat_norm.T, W)
      diag = np.diag(dot)
      nov = np.dot(s_flat, diag)

      """
      temp2 = np.dot(A_flat.T, W)
      temp3 = np.dot(temp2, s_flat)
      nov = np.dot(s_flat, temp3)"""


      return nov

  def compute_risk_obs(self,who):


    A_probs = self.A[who]/np.sum(self.A[who], axis = 0)
    po = np.einsum("ijklo, jklo -> i", A_probs, self.s[who])

    assert abs(np.sum(po)-1) < 0.01, f"probs sum to {np.sum(po)}"

    temp = np.divide(po, self.goal_obs[who])
    temp1 = np.where(temp != 0, np.log(temp), 0)

    #temp1 = np.log(temp, where=(temp != 0), out=np.zeros_like(temp))

    risk_obs = np.dot(po, temp1)

    return risk_obs


  def compute_EFE(self, u = int, who = int):
    cp = self.s.copy()

    #print(f"These is {self.agent_id}'s perspective of {who} state before action {u} : {self.s[who]}")

    if who == 0:
      self.act(u)
    elif who == 1:
      self.act_other(u)

    #print(f"These is {self.agent_id}'s perspective of {who} state after action {u} : {self.s[who]}")

    toconsider = 0
    if "abg" in self.terms:
        toconsider = self.compute_ambiguity(who)
    elif "risk" in self.terms:
        toconsider = toconsider + self.compute_risk(who)
    elif "risk_obs" in self.terms:
        toconsider = toconsider + self.compute_risk_obs(who)
    elif "nov" in self.terms:
        toconsider = toconsider - self.compute_novelty(who)
    
    EFE = toconsider
    self.s = cp.copy()

    return EFE


  def imagine_u(self, who = int):

    if self.little_u_imaginer == True:

      pu = np.zeros(len(self.U))
      for u in self.U:
        pu[u] = self.compute_EFE(u, who)


      pu = softmax(pu,1)

      if self.branch_reductor == "cut":
          u_imagined = np.array(self.U)[pu>self.u_prob_threshold]
          pu_imagined = pu[pu>self.u_prob_threshold]
          pu_imagined = pu_imagined/np.sum(pu_imagined)

          
      elif self.branch_reductor == "sample":
          u_imagined = np.random.choice(len(self.U), size = 1, p = pu)
          pu_imagined = [1]


    else:
      u_imagined = self.U
      pu_imagined = np.ones(len(u_imagined))/len(u_imagined)

    
    assert len(u_imagined) >0, "No action was imagined"
    assert abs(np.sum(pu_imagined)-1)<0.01,f"Actions probabilities sum to {np.sum(pu_imagined)}"

    return u_imagined, pu_imagined


  def sop_imagine_u(self, who = int):
    
    if who == 1:
        u, H = self.SFE_call_other()
    
    else:
        assert 1/0, f"This should not happen, never call sop_imagine_u on yourself"

    pu = softmax(H, 1)

    if self.little_u_imaginer == True:

        if self.branch_reductor == "cut":
            
            u_imagined = u[pu>self.u_prob_threshold]
            pu_imagined = pu[pu>self.u_prob_threshold]
            pu_imagined = pu_imagined/np.sum(pu_imagined)

        elif self.branch_reductor == "sample":
            assert 1/0, "Sampling not implemented for two agent SFE, since for one agent it's already ass"
            u_imagined = np.random.choice(u, size = 1, p = pu)
            pu_imagined = [1]

        else:
            assert 1/0, "WHAT THE SHID IS THIS? THERE IS NO SUCH BRANCH REDUCTOR"

    else:
        u_imagined = u
        pu_imagined = pu

    #print(f"the actions imagined are: {u_imagined}")

    assert len(u_imagined)>0, f"No actions sampled"
    assert abs(np.sum(pu_imagined)-1) < 0.01, f"probabilities of actions sum to {np.sum(pu_imagined)}"
  

    return u_imagined, pu_imagined


  def imagine_o(self, who = int):
    
    A_norm = self.A[who]/np.sum(self.A[who], axis = 0)
    po = np.einsum("ijklm, jklm -> i", A_norm, self.s[who])

    if self.little_o_imaginer == True:

        if self.branch_reductor == "cut":

            o_imagined = np.arange(self.A[who].shape[0])[po > self.o_prob_threshold]
            po_imagined = po[o_imagined]/np.sum(po[o_imagined])

            if len(o_imagined) == 0:
                o_imagined = np.random.choice(self.A[who].shape[0], size = 1, p = po)
                po_imagined = [1]
            
        elif self.branch_reductor == "sample":
            o_imagined = np.random.choice(self.A[who].shape[0], size = 1, p = po)
            po_imagined = [1]



    else:
        o_imagined = np.arange(self.A[who].shape[0])
        po_imagined = po

    

    assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum to {np.sum(self.s[who])}"
    assert abs(np.sum(po_imagined)-1)<0.01, f"Observation probs sum to {np.sum(po_imagined)}"

    return o_imagined, po_imagined


  def SFE(self, who = int):

    if self.branch_reductor == "cut":
        num_samples = 1

    elif self.branch_reductor == "sample":
        num_samples = self.mc_samples

    else:
        assert 1/0, "WHAT THE SHID IS THIS? THERE IS NO SUCH BRANCH REDUCTOR"


    us = []
    sfes = []


    for i in range(num_samples):

        current_depth = 1

        u_imagined, _ = self.imagine_u(who)
        H = np.zeros(len(u_imagined))

        cp = self.s.copy()
        for i,u in enumerate(u_imagined):

          H[i]=self.compute_EFE(u,who)

          if current_depth < self.reason_depth:
            if self.social == True:
              
              if who == 0:
                self.act(u)
                future_EFE = self.SFEr2(current_depth + 1)
                
              else:
                self.act_other(u)
                future_EFE = self.SFEr(current_depth + 1,who = 1)

            else:
              self.act(u)
              future_EFE = self.SFEr(current_depth + 1, who = 0)
            
            if self.futuro_prosperoso == True:
                H[i] = future_EFE #PORCA TROIA PER N TAB NO JIVA NIA PLU DUTA STA MERDA
            else:
                H[i] = future_EFE + H[i]
            self.s = cp.copy()
    
        us.append(u_imagined)
        sfes.append(H)  



    if self.branch_reductor == "cut":
        u_imagined = us[0]
        H = sfes[0]

    elif self.branch_reductor == "sample":

        for i in range(len(us)):
            us[i] = int(us[i][0])
            sfes[i] = float(sfes[i][0])

        us = np.array(us)
        sfes = np.array(sfes)

        u_imagined = []
        H = []


        for u in range(len(self.U)):
            temp = sfes[us==u]
            if len(temp) > 0:
                sfes_mean = np.mean(temp)
                #sfes_mean = np.min(temp)


                u_imagined.append(u)
                H.append(sfes_mean)


    return u_imagined, np.array(H) 


  def SFEr(self, current_depth = int, who = int):


    o_imagined, po_imagined = self.imagine_o(who)
    Hro = np.zeros(len(o_imagined))

    cp1 = self.s.copy()
    for i,o in enumerate(o_imagined):

      self.observe(o,who)
      u_imagined, _ = self.imagine_u(who)
      Hrou = np.zeros(len(u_imagined))

      cp2 = self.s.copy()
      for j,u in enumerate(u_imagined):

        Hrou[j] = self.compute_EFE(u,who)

        if current_depth < self.reason_depth:

          if who == 0:
            self.act(u)
            future_EFE = self.SFEr(current_depth + 1, who = who)

          else:
            self.act_other(u)
            future_EFE = self.SFEr(current_depth + 1, who = who)

          if self.futuro_prosperoso == True:
              Hrou[j] = future_EFE
          else:
              Hrou[j] = future_EFE + Hrou[j]
          self.s = cp2.copy()

      
      if self.branch_reductor == "cut":
          pur = softmax(Hrou, 1) #makes more sense to have 1 decision temperature since this is counterfactual reasoning

      elif self.branch_reductor == "sample":
          assert len(Hrou) == 1
          assert len(u_imagined) == 1
          Hrou = Hrou[0]
          pur = 1

      else:
          assert 1/0, "WHAT THE SHID IS THIS? THERE IS NO SUCH BRANCH REDUCTOR"

      Hro[i] = np.dot(pur, Hrou)
      self.s = cp1.copy()

    future_EFE = np.dot(po_imagined, Hro)

    return future_EFE


  def SFEr2(self, current_depth = int):

    o_imagined, po_imagined = self.imagine_o(0) #0 is I imagine
    Hro = np.zeros(len(o_imagined))

    cp1 = self.s.copy()
    for i,o in enumerate(o_imagined):

      self.observe(o,0)
      self.observe(o,1)

      up_imagined, pup_imagined = self.sop_imagine_u(1)
      Hup = np.zeros(len(up_imagined))

      cp2 = self.s.copy()
      for j,up in enumerate(up_imagined):

        self.act_other(up) #this act both changes my beliefs and his beliefs

        op_imagined, pop_imagined = self.imagine_o(1)
        Hrop = np.zeros(len(op_imagined))

        cp3 = self.s.copy()
        for k,op in enumerate(op_imagined):
          self.observe(op,1)
          self.observe(op,0)

          u_imagined, _ = self.imagine_u(0)
          Hropu = np.zeros(len(u_imagined))

          for l,u in enumerate(u_imagined):

            Hropu[l] = self.compute_EFE(u,0)

            if current_depth < self.reason_depth:
              cp4 = self.s.copy()
              self.act(u)

              future_EFE = self.SFEr2(current_depth + 1)
              if self.futuro_prosperoso == True:
                  Hropu[l] = future_EFE
              else:
                  Hropu[l] = future_EFE + Hropu[l]
              
              self.s = cp4.copy()

          Hrop[k] = np.dot(softmax(Hropu, 1), Hropu)
          self.s = cp3.copy()

        Hup[j] = np.dot(pop_imagined, Hrop)
        self.s = cp2.copy()

      Hro[i] = np.dot(pup_imagined, Hup)
      self.s = cp1.copy()

    future_EFE = np.dot(po_imagined, Hro)

    return future_EFE


  def pick_action(self):
      
      if self.random_moves == False:

          u_imagined, H = self.SFE_call_self()
          print(f"These are the actions considered by agent {self.agent_id} with their EFE {u_imagined}, {np.round(H,2)}")
          assert len(u_imagined) > 0, "No action imagined"

          if self.det_dec_rule == True:

              action = u_imagined[np.argmin(H)]
      
          else:
              uprobs = softmax(H, self.dec_temp)
              action = np.random.choice(u_imagined, p = uprobs)

      
      else:
          action = np.random.choice(self.U)

    
      self.moves_to_go = self.moves_to_go - 1
      self.change_reason_depth()

      return action
  
  def explore(self):

    action = np.random.choice(self.U)

    """
    old_terms = self.terms.copy()
    #old_rd = self.reason_depth

    self.terms = ["nov"]
    #self.reason_depth = 1
    assert 1/0, f"This is not an error, just don't know why call SFE_other here, it's correct, just forgot why"
    u_imagined, uprobs = self.SFE_call_other()

    self.terms = old_terms.copy()
    #self.reason_depth = old_rd

    #action = np.random.choice(self.U)
    action = np.random.choice(u_imagined, p = uprobs)"""

    self.moves_to_go = self.moves_to_go - 1

    self.change_reason_depth()
          
    return action


  def change_reason_depth(self):

    if self.reason_depth > self.moves_to_go:
      self.reason_depth = self.reason_depth - 1
      assert self.reason_depth == self.moves_to_go, "SOMETHING IS SMELLY"

  def change_environment(self, env_id):
    self.agent_id = env_id


class world:
  def __init__(self, cheese_positions):
    self.true_s = np.zeros((4,2))
    self.true_s[0,:] = 1
    self.cheese_positions = cheese_positions
    self.B = np.array(([[[0,0,0,0],[1,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]],[[0,0,0,1],[0,1,0,0],[1,0,1,0],[0,0,0,0]],[[0,0,1,0],[0,1,0,0],[0,0,0,0],[1,0,0,1]]]))

  def change_true_s(self, agent, u):
    #print(agent.agent_id)
    if agent.agent_id == 0:
      self.true_s[:,0] = np.dot(self.B[u], self.true_s[:,0])

    else:
      self.true_s[:,1] = np.dot(self.B[u], self.true_s[:,1])


  def give_o(self, agent):
    if agent.agent_id == 0:
      if self.true_s[1,0] == 1:
        if self.cheese_positions[0] == 0: #put positions[1] if refer to mouse 1 cheese
          o = 0 #5 to tell mouse 1 where cheese is
        else:
          o = 1 #6 to tell mouse 1 where cheese is

      elif (self.true_s[2,0] == 1 and self.cheese_positions[0] == 0) or (self.true_s[3,0] == 1 and self.cheese_positions[0] == 1):
        o = 2 #agent 0 on the cheese

      elif (self.true_s[2,0] == 1 and self.cheese_positions[0] == 1) or (self.true_s[3,0] == 1 and self.cheese_positions[0] == 0):
        o = 3 #agent 0 not on the cheese

      else:
        o = 4

    else:

      if self.true_s[1,1] == 1: #first and clause is mouse 0 cheese, second clause if mouse 1 cheese
        if self.cheese_positions[0] == 0 and self.cheese_positions[1] == 0:
          o = 5 #both cheeses on the left
        elif self.cheese_positions[0] == 0 and self.cheese_positions[1] == 1:
          o = 6
        elif self.cheese_positions[0] == 1 and self.cheese_positions[1] == 0:
          o = 9
        elif self.cheese_positions[0] == 1 and self.cheese_positions[1] == 1:
          o = 10 #both cheeses on the right

        """o = self.cheese_positions[0] #which can be either 0 or 1 THIS OR THE PART ABOVE"""


      elif (self.true_s[2,1] == 1 and self.cheese_positions[1] == 0) or (self.true_s[3,1] == 1 and self.cheese_positions[1] == 1):
        o = 7 #agent 1 on the cheese
      elif (self.true_s[2,1] == 1 and self.cheese_positions[1] == 1) or (self.true_s[3,1] == 1 and self.cheese_positions[1] == 0):
        o = 8 #agent 1 not on the cheese

      else:
        o = 4 #no observation for anybody

    return o


  def reset(self):
    
    self.true_s = np.zeros((4,2))
    self.true_s[0,:] = 1


def true_A(base, pretrain_strength = float, other_strength = float):

    
    agent0_A = np.ones((11,2,2,4,4))*base

    agent0_A[0,0,:,1,:] = pretrain_strength

    agent0_A[1,1,:,1,:] = pretrain_strength

    agent0_A[2,0,:,2,:] = pretrain_strength
    agent0_A[2,1,:,3,:] = pretrain_strength

    agent0_A[3,1,:,2,:] = pretrain_strength
    agent0_A[3,0,:,3,:] = pretrain_strength

    agent0_A[4,:,:,0,:] = pretrain_strength
    agent0_A[4,:,:,:,0] = other_strength

    agent0_A[5,0,0,:,1] = other_strength

    agent0_A[6,0,1,:,1] = other_strength

    agent0_A[7,:,0,:,2] = other_strength
    agent0_A[7,:,1,:,3] = other_strength

    agent0_A[8,:,0,:,3] = other_strength
    agent0_A[8,:,1,:,2] = other_strength

    agent0_A[9,1,0,:,1] = other_strength

    agent0_A[10,1,1,:,1] = other_strength

    agent1_A = np.ones((11,2,2,4,4))*base

    agent1_A[0,0,:,1,:] = other_strength

    agent1_A[1,1,:,1,:] = other_strength

    agent1_A[2,0,:,2,:] = other_strength
    agent1_A[2,1,:,3,:] = other_strength

    agent1_A[3,1,:,2,:] = other_strength
    agent1_A[3,0,:,3,:] = other_strength

    agent1_A[4,:,:,0,:] = other_strength
    agent1_A[4,:,:,:,0] = pretrain_strength

    agent1_A[5,0,0,:,1] = pretrain_strength

    agent1_A[6,0,1,:,1] = pretrain_strength

    agent1_A[7,:,0,:,2] = pretrain_strength
    agent1_A[7,:,1,:,3] = pretrain_strength

    agent1_A[8,:,0,:,3] = pretrain_strength
    agent1_A[8,:,1,:,2] = pretrain_strength

    agent1_A[9,1,0,:,1] = pretrain_strength

    agent1_A[10,1,1,:,1] = pretrain_strength


    return agent0_A, agent1_A

def run(agent0, agent1, epochs = int, nr_moves = int, reason_depth = int,start_beliefs = np.array,
        exploration_phase = bool, A_learning = bool,
        infer_only_on_last_action = bool, opt_params = dict, acclimatization_episodes = int):



  epochs_agent_worlds = [] #save in which world each agent is
  epochs_positions = [] #true positions of each agent
  epochs_A_matrices = [] #A matrices of each agent (my, his)
  epochs_beliefs = []
  epochs_desires = []
  
  epochs_observations0 = {"lw": [], #leftworld is the proper agent0 env
                          "rw": []} #rightworld is the world of agent1

  epochs_observations1 = {"lw": [],
                          "rw": []}

  epochs_actions0 = {"lw": [],
                      "rw": []}

  epochs_actions1 = {"lw": [],
                      "rw": []}
  

  epochs_A_matrices.append((agent0.A.copy(), agent1.A.copy()))
  epochs_desires.append((agent0.goal_states.copy(),agent1.goal_states.copy())) #THIS NOT IN USE


  cheeses = []
  for i in range(epochs):
      cheeses.append([np.random.randint(0,2), np.random.randint(0,2)])

  for e in range(epochs):
    print("__________________________________")
    print("Starting epoch: " + str(e))
    print("__________________________________")


    if e < acclimatization_episodes:
        agent0.agent_id = 0
        agent1.agent_id = 1
        nr_moves = 10
        agent0.random_moves = True
        agent0.moves_to_go = nr_moves
        agent1.random_moves = True
        agent1.moves_to_go = nr_moves

    else:
        if exploration_phase == True:
            where_are_you = np.random.randint(0,2)
            agent0.change_environment(where_are_you)
            agent1.change_environment(abs(where_are_you-1))

        if agent0.agent_id == 0:
            nr_moves = 3
            agent0.random_moves = False
            agent0.moves_to_go = nr_moves
            agent1.random_moves = False
            agent1.moves_to_go = nr_moves

        else:
            nr_moves = 10
            agent0.random_moves = True
            agent0.moves_to_go = nr_moves
            agent1.random_moves = True
            agent1.moves_to_go = nr_moves


    epochs_agent_worlds.append((agent0.agent_id, agent1.agent_id))
    
  
    agent0.restart_agent(start_beliefs, reason_depth, nr_moves)
    agent1.restart_agent(start_beliefs, reason_depth, nr_moves)

    w = world(cheeses[e])


    positions = []
    beliefs = []
    agentselfposA = []
    agentselfposB = []
    
    agent0_acts = []
    agent1_acts = []
    agent0_obs = []
    agent1_obs = []



    beliefs.append((agent0.s.copy(),agent1.s.copy()))



    for k in range(nr_moves):

      if agent0.agent_id == 0: #if agent0 in env 1 he explores, if agent1 in env 0 he explores

    
        u = agent0.pick_action()
        agent0.act(u)
        agent1.act_other(u)
        

        w.change_true_s(agent0, u)
        o = w.give_o(agent0)
        
        positions.append(pos_conv(w.true_s))
        agentselfposA.append(agent0.s.copy())
     

        
        agent0.observe(o,0)
        agent0.observe(o,1)
        agent1.observe(o,0)
        agent1.observe(o,1)

        beliefs.append((agent0.s.copy(),agent1.s.copy()))
        
        agent0_acts.append(u)
        agent0_obs.append(o)
        




        
        u = agent1.pick_action()
        #u = 0


        agent1.act(u)
        agent0.act_other(u)
        
        #print("This is the action taken by agent 1: " + str(u))

        w.change_true_s(agent1, u)
        o = w.give_o(agent1)
        #print(f"This is the observation given by the world: {o}")
        positions.append(pos_conv(w.true_s))
        agentselfposB.append(agent1.s.copy())
    
      
        agent0.observe(o,0)
        agent0.observe(o,1)
        agent1.observe(o,0)
        agent1.observe(o,1)
          
        beliefs.append((agent0.s.copy(),agent1.s.copy()))

        agent1_acts.append(u)
        agent1_obs.append(o)



        if agent0.empathetic == True:
            if infer_only_on_last_action == True:
                if k == nr_moves-1:
                    agent0.update_goals_other(u, beliefs[-2][0], opt_params)
                    print(f"These are the learned goals at the end of the episode: \n {np.mean(agent0.goal_states[1], axis = (0,2))}")

            else:
                agent0.update_goals_other(u, beliefs[-2][0], opt_params) #use my beliefs about his beliefs before that we acted
            
            
        
        #print(f"This is the beliefs of agent 0 after observing: {agent0.s}")
        #print(f"This is the beliefs of agent 1 after observing: {agent1.s}")

      else:

        u = agent0.explore()
        agent0.act(u)
        #print("This is the action taken by agent 0: " + str(u))

        w.change_true_s(agent0, u)
        o = w.give_o(agent0)
        #print(f"This is the observation given by the world: {o}")
        positions.append(pos_conv(w.true_s))
        agentselfposA.append(agent0.s.copy())
 

        agent0.observe(o,0)
        agent0.observe(o,1)
        agent1.observe(o,0)
        agent1.observe(o,1)

        beliefs.append((agent0.s.copy(),agent1.s.copy()))

        agent0_acts.append(u)
        agent0_obs.append(o)


        


        u = agent1.explore()
        agent1.act(u)
        #print("This is the action taken by agent 1: " + str(u))

        w.change_true_s(agent1, u)
        o = w.give_o(agent1)
        #print(f"This is the observation given by the world: {o}")
        positions.append(pos_conv(w.true_s))
        agentselfposB.append(agent1.s.copy())
      

        agent0.observe(o,0)
        agent0.observe(o,1)
        agent1.observe(o,0)
        agent1.observe(o,1)

        beliefs.append((agent0.s.copy(),agent1.s.copy()))

        agent1_acts.append(u)
        agent1_obs.append(o)



    """NOT USED ANYMORE
    if A_learning == True:
        for (o,s) in zip(observationsA, agentselfposA):
            if agent0.agent_id == 0:
              agent0.update_A(o,s,who=0)
            else:
              agent0.update_A(o,s,who=1)

        for (o,s) in zip(observationsB, agentselfposB):
            if agent1.agent_id == 1:
              agent1.update_A(o,s,who=0)
            else:
              agent1.update_A(o,s,who=1)"""

    
    if A_learning == True:
        if agent0.agent_id == 0:
            epochs_observations0["lw"].append(agent0_obs)
            epochs_actions0["lw"].append(agent0_acts)
                
            A, _ = EM(epochs_observations0["lw"],
                        epochs_actions0["lw"],
                        agent0.B.copy(),
                        A_param_shape = (11,2,4),
                        s_param_shape = (2,4),
                        gm = None, start = None, end = None)

            true_A = np.einsum("ijklo -> ijl", agent0.true_A[0].copy())

            if agent0.state_stabilizer == True:
                A = state_switcher(learned_A = A, true_A = true_A)
            else:
                A = np.reshape(A, true_A.shape)
    
            
            temp = np.repeat(A[:,:,np.newaxis,:,np.newaxis], axis = 2, repeats = 2)
            temp = np.repeat(temp, axis = 4, repeats = 4)/8 #need to do this to keep dirichlet count proper
            agent0.A[0] = temp.copy()


            epochs_observations1["rw"].append(agent1_obs)
            epochs_actions1["rw"].append(agent1_acts)
            A, _ = EM(epochs_observations1["rw"],
                        epochs_actions1["rw"],
                        agent1.B.copy(),
                        A_param_shape = (11,2,4),
                        s_param_shape = (2,4),
                        gm = None, start = None, end = None)

            true_A = np.einsum("ijklo -> iko", agent1.true_A[0].copy())

            if agent1.state_stabilizer == True:
                A = state_switcher(learned_A = A, true_A = true_A)
            else:
                A = np.reshape(A, true_A.shape, order = "C")

            
            temp = np.repeat(A[:,np.newaxis,:,np.newaxis,:], axis = 1, repeats = 2)
            temp = np.repeat(temp, axis = 3, repeats = 4)/8 #need to do this to keep dirichlet count proper
            agent1.A[0] = temp.copy()


        else:
            epochs_observations0["rw"].append(agent0_obs)
            epochs_actions0["rw"].append(agent0_acts)

            A, _ = EM(epochs_observations0["rw"],
                        epochs_actions0["rw"],
                        agent0.B.copy(),
                        A_param_shape = (11,4,2),
                        s_param_shape = (4,2),
                        gm = None, start = None, end = None)

            true_A = np.einsum("ijklo -> iko", agent0.true_A[1].copy())

            if agent0.state_stabilizer == True:
                A = state_switcher(learned_A = A, true_A = true_A)
            else:
                A = np.reshape(A, true_A.shape, order = "C")
            
            temp = np.repeat(A[:,np.newaxis,:,np.newaxis,:], axis = 1, repeats = 2)
            temp = np.repeat(temp, axis = 3, repeats = 4)/8 #need to do this to keep dirichlet count proper
            agent0.A[1] = temp.copy()



            epochs_observations1["lw"].append(agent1_obs)
            epochs_actions1["lw"].append(agent1_acts)
            A, _ = EM(epochs_observations1["lw"],
                        epochs_actions1["lw"],
                        agent1.B.copy(),
                        A_param_shape = (11,4,2),
                        s_param_shape = (4,2),
                        gm = None, start = None, end = None)
            
            true_A = np.einsum("ijklo -> ijl", agent1.true_A[1].copy())

            if agent1.state_stabilizer == True:
                A = state_switcher(learned_A = A, true_A = true_A)
            else:
                A = np.reshape(A, true_A.shape, order = "C")
            
            temp = np.repeat(A[:,:,np.newaxis,:,np.newaxis], axis = 2, repeats = 2)
            temp = np.repeat(temp, axis = 4, repeats = 4)/8 #need to do this to keep dirichlet count proper
            agent1.A[1] = temp.copy()



  

    epochs_positions.append(positions)

    epochs_A_matrices.append((agent0.A.copy(), agent1.A.copy()))

    epochs_beliefs.append(beliefs)

    epochs_desires.append((agent0.goal_states.copy(),agent1.goal_states.copy()))



    #print(f"At the end of epoch {e}, agent0 A is: {agent0.A}")
    #print(f"At the end of epoch {e}, agent1 A is: {agent1.A}")
  

  return epochs_positions, epochs_A_matrices, epochs_agent_worlds, epochs_beliefs, epochs_desires, cheeses


def main():
    start_B = np.array(([[[0,0,0,0],[1,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]],[[0,0,0,1],[0,1,0,0],[1,0,1,0],[0,0,0,0]],[[0,0,1,0],[0,1,0,0],[0,0,0,0],[1,0,0,1]]]))


    start_s = np.zeros((2,2,4,4))
    start_s[:,:,0,0] = [[0.25,0.25],[0.25,0.25]]

    full_start_s = np.repeat(start_s[np.newaxis,:,:,:,:],repeats = 2, axis =0)

    agent0_A, agent1_A = true_A()

    agent0_full_A = np.stack((agent0_A, agent1_A), axis = 0)
    agent1_full_A = np.stack((agent1_A, agent0_A), axis = 0)



    agent0_goals = np.ones(start_s.shape)

    agent0_goals[0,:,2,:] = 1e1
    agent0_goals[1,:,3,:] = 1e1
    agent0_goals[0,:,3,:] = 1e-10
    agent0_goals[1,:,2,:] = 1e-10
    agent0_goals = agent0_goals/np.sum(agent0_goals)

    agent1_goals = np.ones(start_s.shape)

    agent1_goals[:,0,:,2] = 1e1
    agent1_goals[:,1,:,3] = 1e1
    agent1_goals[:,0,:,3] = 1e-10
    agent1_goals[:,1,:,2] = 1e-10
    agent1_goals = agent1_goals/np.sum(agent1_goals)

    agent0_fullgoals = np.stack((agent0_goals, agent1_goals), axis = 0)
    agent1_fullgoals = np.stack((agent1_goals, agent0_goals), axis = 0)


    U = [0,1,2,3]
    reason_depth = 3
    dec_temp = 0.05
    little_o_imaginer = True
    o_imagination_breadth = 3
    o_prob_threshold = 0.1

    little_u_imaginer = True
    u_imagination_breadth = 3
    u_prob_threshold = 0.1

    nr_moves = 3

    agent0 = agent(agent0_full_A.copy(), start_B, full_start_s.copy(), U = U, goal_states = agent0_fullgoals.copy(),
                reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
                little_o_imaginer = little_o_imaginer, o_imagination_breadth = o_imagination_breadth, o_prob_threshold = o_prob_threshold,
                little_u_imaginer = little_u_imaginer, u_imagination_breadth= u_imagination_breadth, u_prob_threshold = u_prob_threshold,
                agent_id = 0, moves_to_go = nr_moves)

    agent1 = agent(agent1_full_A.copy(), start_B, full_start_s.copy(), U = U, goal_states = agent1_fullgoals.copy(),
                reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
                little_o_imaginer = little_o_imaginer, o_imagination_breadth = o_imagination_breadth, o_prob_threshold = o_prob_threshold,
                little_u_imaginer = little_u_imaginer, u_imagination_breadth= u_imagination_breadth, u_prob_threshold = u_prob_threshold,
                agent_id = 1, moves_to_go = nr_moves)


    epochs = 1
    nr_actions = 3
    cheeses = [[0,1]]
    start_beliefs = full_start_s


    epochs_positions = run(agent0 = agent0, agent1 = agent1, epochs = epochs, nr_actions = nr_actions, start_beliefs = start_beliefs.copy(), cheeses = cheeses)
    
    #print(epochs_positions)

if __name__ == "__main__":
    #main()
    None



