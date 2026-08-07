import numpy as np
from utils import EM2, dir_VI, A_KL, state_switcher, norm_om, flatten_om

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
               goal_states, reason_depth = int, dec_temp = float, chatty_boy = bool,
               little_o_imaginer = bool, o_prob_threshold = float,
               little_u_imaginer = bool, u_prob_threshold = float,
               agent_id = int, moves_to_go = int, social = bool, terms = list, sampling = str,
               branch_reductor = str, mc_samples = int, selfish = bool, empathetic = bool,
               bayes_observe = bool, obs_model_given = bool, efe_discounting = str,
               state_stabilizer = bool, goal_learning_beta = float,
               imagine_u_temp = float, imagine_u_temp_other = float):
    
    self.true_A = A.copy()
    self.agent_id = agent_id
    
    if  obs_model_given == True:
        self.A = A.copy()
    else:
        baseline = 1e-5
        temp = np.zeros(self.true_A[0].shape)+baseline
        temp0 = temp.copy()
        temp1 = temp.copy()
        temp0[4] = 1
        temp1[11] = 1
        temp0 /= baseline
        temp1 /= baseline

        if self.agent_id == 0:
            my_A = temp0
            your_A = temp1
        else:
            my_A = temp1
            your_A = temp0
        #your_A = np.ones(A[1].shape) #your observation model (a dirichlet)

        self.A = np.stack((my_A, your_A), axis = 0)

    self.s = s #[my/your belief, agent0cheese, agent1cheese, agent0pos, agetn1pos]
    self.B = B
    self.U = U
    self.goal_states = goal_states #[my/your]
    self.reason_depth = reason_depth
    self.chatty_boy = chatty_boy
    self.dec_temp = dec_temp
    self.imagine_u_temp = imagine_u_temp
    self.imagine_u_temp_other = imagine_u_temp_other
    self.little_o_imaginer = little_o_imaginer
    self.o_prob_threshold = o_prob_threshold
    self.little_u_imaginer = little_u_imaginer
    self.u_prob_threshold = u_prob_threshold


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
    self.efe_discounting= efe_discounting



   

    self.state_stabilizer = state_stabilizer

    self.goal_learning_beta = goal_learning_beta



    if len(self.terms) == 0:
        self.random_moves = True
    else:
        self.random_moves = False

    if self.dec_temp == 0:
        self.det_dec_rule = True
    else:
        self.det_dec_rule = False

    implemented_terms = ["risk", "risk_obs"] #use "nov" and "abg" at your own risk, "epi_value" is not implemented
    for term in self.terms:
        assert term in implemented_terms, f"{term} is not in the implemented list of: {implemented_terms}"

        
    assert self.reason_depth >= 1, f"My man you want to do thinking depth of: {self.reason_depth}?"
    assert self.moves_to_go >= 1, f"You can always move"



    implemented_sampling = ["sample", "mean"]
    implemented_sampling_obj = ["om", "goals"]

    for key in sampling:
        assert key in implemented_sampling_obj, f"{key} is not in the implemented list {implemented_sampling_obj}"
    assert sampling["om"] in implemented_sampling, f"{sampling['om']} is not in the implemented list {implemented_sampling}"
    assert sampling["goals"] in implemented_sampling, f"{sampling['goals']} is not in the implemented list {implemented_sampling}"



  def discount_EFE(self, EFE, current_depth):

      if self.efe_discounting == "total":
          EFE = 0

      elif self.efe_discounting == "new":
          if current_depth == 1:
              EFE = EFE
          if current_depth == 2:
              EFE = EFE
          if current_depth == 3:
              EFE = EFE

      elif self.efe_discounting == "default":
          EFE = EFE
      else:
          assert 1/0, f"What the shid?"

      return EFE

  
  def restart_agent(self, start_s, reason_depth, nr_moves):
      self.s = start_s.copy()
      self.reason_depth = reason_depth
      self.moves_to_go = nr_moves
      self.random_moves = False


  def sample(self, dcm, sampler):
      """
      dcm is dirichlet counts matrix
      """
      dcm_shape = dcm.shape
      dcm_flat = dcm.reshape(-1)

      
      if sampler == "mean":
          sample = dcm_flat/np.sum(dcm_flat)

      elif sampler == "sample":
          sample = np.random.dirichlet(dcm_flat)

      else:
          assert 1/0, f"{sampler} does not exist"
  
      out = sample.reshape(dcm_shape)
      return out


  def SFE_call_other(self, sampled_goal = None):

        gcp = self.goal_states.copy()
        acp = self.A.copy()

        if sampled_goal is None:

            if self.selfish == True:

                if self.agent_id == 0:
                    filtered_goals = np.mean(self.goal_states[1].copy(), axis = (0,2)) #np.einsum("ijkl -> jl", self.goal_states[1].copy()) 
                    #filtered_states = np.einsum("ijkl -> jl", self.s[1].copy())
                else:
                      filtered_goals =  np.mean(self.goal_states[1].copy(), axis = (1,3)) #np.einsum("ijkl -> ik", self.goal_states[1].copy()) 
                    #filtered_states = np.einsum("ijkl -> ik", self.s[1].copy())
            else:
                sampled_goal = self.goal_states[1].copy()
                #filtered_states = self.s[1].copy()

          
         
        else:
            
            if self.selfish == True:
                  sampled_goal = sampled_goal.reshape(self.goal_states[1].shape[1], self.goal_states[1].shape[3])
              


        assert abs(np.sum(sampled_goal)-1)<1e-6
        self.change_goals(new_goal_states = sampled_goal, who = 1)


        actions, actions_H = self.SFE(1)

        #ATTENTION NOT SAMPLING OM IF CALLED DIRECTLY
            #NOT A PROBLEM IN GENERAL BECAUSE WE SAMPLE THEM IN SFE CALL SELF
            #IF LEARNING GOALS SFE CALL OTHER CALLED DIRECTLY AND THIS IS PROBLEM IS OM IS NOT GIVEN
        self.goal_states = gcp.copy()
        self.A = acp.copy()
      
      
        return actions, actions_H


  def SFE_call_self(self):

      gcp = self.goal_states.copy()
      acp = self.A.copy()


      #this bit is useless now, just kept because why not
      if self.selfish == True:

          if self.agent_id == 0:
              filtered_goals = np.einsum("ijkl -> ik", self.goal_states[0].copy()) #np.mean(self.goal_states[0].copy(), axis = (1,3)) 
          else:
              filtered_goals = np.einsum("ijkl -> jl", self.goal_states[0].copy()) #np.mean(self.goal_states[0].copy(), axis = (0,2)) 
      else:
          filtered_goals = self.goal_states[0].copy()
        

      #I don't sample my obs model and my goals (no uncertainty about them) 
      self.change_goals(new_goal_states = filtered_goals, who = 0)
      self.A[0] = norm_om(self.A[0].copy())
      
      

      self.goal_states[1] = self.sample(self.goal_states[1].copy(), self.sampling["goals"]).copy()
      self.A[1] = self.sample(self.A[1].copy(), self.sampling["om"]).copy()



      actions, actions_H = self.SFE(0)

      self.goal_states = gcp.copy()
      self.A = acp.copy()
      
      return actions, actions_H


  def update_A(self, ep_observations, ep_actions, agent_id):
    assert 1/0, f"Update A is DEAD"
    if self.update_A_method == "default":
        assert 1/0, f"This is not used anymore!!!"
    
        """backup = self.s.copy()

        self.s = s.copy()
        self.observe(o,0) #infer which state we were in

    

        self.A[who,o,...] = self.A[who,o,...].copy() + self.s[0].copy()

        self.s = backup"""

    elif self.update_A_method == "EM_all_ep":
        assert 1/0, f"This method is not in use anymore"
        

  


  def change_goals(self, new_goal_states, who):
      #new_gs_res = new_goal_states.reshape(self.goal_states[who].shape)
      self.goal_states[who] = new_goal_states.copy()


  def logl_action(self, a, sampled_goal, belief, res_dep):
        
        belief_cp = self.s.copy()
        resdep_cp = self.reason_depth

        self.s = belief.copy()
        self.reason_depth = res_dep

        actions, actions_H = self.SFE_call_other(sampled_goal)
        
        
        """actions_prob = softmax(actions_H, self.goal_learning_beta)
        
        if a in actions:
            idx = np.where(actions == a)[0][0]
            action_prob = actions_prob[idx]"""


        action = actions[np.argmin(actions_H)]
        if action == a:
           action_prob = 1

        else:
            action_prob = 0.01 
    
        self.s = belief_cp.copy()
        self.reason_depth = resdep_cp

        return np.log(action_prob)


  def update_goals_other(self, a, belief, opt_params, k):

      #self.reason_depth = self.reason_depth + 1 #need to go back at previous timepoint to see what I would have done if I were him using my belief about his goals
      
      #belief_state_cp = self.s.copy()
      #self.s = belief_state.copy()


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
      
      print(f'These are the previous goals; \n {np.round(prior,2)}')

      goal_learning_vars = self.goal_learning_vars
      new_desires, goal_learning_vars, logl = dir_VI(self.logl_action, a, prior, belief, goal_learning_vars, opt_params = opt_params)
      self.goal_learning_vars = goal_learning_vars

      print(f'These are the updated goals: \n {np.round(new_desires,2)}')

      new_desires_reshaped = new_desires.reshape(prior_shape)

      if self.selfish == True:
          temp = np.repeat(new_desires_reshaped[np.newaxis,:,:], axis = 0, repeats = 2)
          new_desires_reshaped = np.repeat(temp[:,:,np.newaxis,:], axis = 2, repeats = 4)
          new_desires_reshaped = new_desires_reshaped


      self.goal_states[1] = new_desires_reshaped.copy()







      avg_goals = new_desires_reshaped/np.sum(new_desires_reshaped)
      
      
      """if k == 0:
            lgl = self.logl_action(a, avg_goals)
            print(f'The Loglikelihood of action {a} at step {k} is {lgl}')
            if lgl==0:
                print('LETS GO????')"""






    

      #self.reason_depth = self.reason_depth - 1
      #self.s = belief_state_cp.copy()

      return logl



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
    
    assert abs(np.sum(self.s[who])-1) < 1e-4, f"The state probabilities of {self.agent_id} regarding {who} sum to {np.sum(self.s[who])} after seeing observation {o}"

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



    assert whose_s.shape == whose_goals.shape, f"The shape of the states is: {whose_s.shape} \n and that of the goals is: {whose_goals.shape}"

    s_flat = whose_s.reshape(-1)
    goals_flat = whose_goals.reshape(-1)

    assert s_flat.shape == goals_flat.shape

    temp = np.divide(s_flat, goals_flat)
    temp1 = np.where(temp != 0, np.log(temp), 0)


    assert np.allclose(np.sum(whose_s), 1)
    assert np.allclose(np.sum(goals_flat), 1)
    

    #temp1 = np.log(temp, where=(temp != 0), out=np.zeros_like(temp))

    risk = np.dot(s_flat, temp1)

    assert risk >= 0, f"The risk should be non negative, but instead is {risk}"
    

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

    temp = np.divide(po, self.goal_states[who])
    temp1 = np.where(temp != 0, np.log(temp), 0)

    #temp1 = np.log(temp, where=(temp != 0), out=np.zeros_like(temp))

    risk_obs = np.dot(po, temp1)

    assert risk_obs >= 0, f"The risk should be non negative, but instead is {risk_obs}"
    
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

      pu = softmax(pu,self.imagine_u_temp)
      #print(f"These are the imagined action probabilities: {pu}")

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

    pu = softmax(H, self.imagine_u_temp_other)

    #print(f"these are the probs of acts {u} imagined by {self.agent_id} regarding {who} \n {pu}")
   

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
  

    return u_imagined, pu_imagined, H



  def imagine_o(self, who = int):
    
    A_norm = self.A[who]/np.sum(self.A[who], axis = 0)
    po = np.einsum("ijklm, jklm -> i", A_norm, self.s[who])


    if self.agent_id == 0:
      if who == 0:
          po[5:] = 0
      else:
          po[0:5] = 0
    else:
      if who == 0:
          po[0:5] = 0
      else:
          po[5:] = 0

    po = po/np.sum(po)


    #print(f"This is the state {self.agent_id} things {who} is in: \n {np.einsum("ijkl -> jl", self.s[who])}")
    #print(f"These are the probs of {self.agent_id} regarding {who}: {po}")
    




    if self.little_o_imaginer == True:

        if self.branch_reductor == "cut":

            o_imagined = np.arange(self.A[who].shape[0])[po > self.o_prob_threshold]
            po_imagined = po[o_imagined]/np.sum(po[o_imagined])

            
            
            if len(o_imagined) == 0:
                
                o_imagined = np.random.choice(self.A[who].shape[0], size = 1, p = po)
                po_imagined = np.array([1])
                if self.agent_id == 0:
                    o_imagined = [4]
                else:
                    o_imagined = [11]
                #assert False, f"I HAD TO IMAGINE RANDOMLY BECAUSE this were the po imagined by {self.agent_id} according to {who} \n {po}"
            
        elif self.branch_reductor == "sample":
            o_imagined = np.random.choice(self.A[who].shape[0], size = 1, p = po)
            po_imagined = [1]



    else:
        o_imagined = np.arange(self.A[who].shape[0])
        po_imagined = po

    """
    if self.agent_id == 0:
      if who == 0:
          po_imagined = po_imagined[o_imagined<=4]
          o_imagined = o_imagined[o_imagined<=4]
      else:
          po_imagined = po_imagined[o_imagined>4]
          o_imagined = o_imagined[o_imagined>4]
    else:
        if who == 0:
            po_imagined = po_imagined[o_imagined>4]
            o_imagined = o_imagined[o_imagined>4]
        else:
            po_imagined = po_imagined[o_imagined<=4]
            o_imagined = o_imagined[o_imagined<=4]

    po_imagined = po_imagined/np.sum(po_imagined)"""
    
   

    assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum to {np.sum(self.s[who])}"
    assert abs(np.sum(po_imagined)-1)<0.01, f"Observation probs sum to {np.sum(po_imagined)}"
    assert len(o_imagined) == len(po_imagined), f"WHAT?"
    assert len(o_imagined) > 0, f"WHAT"

    #print(f"These are the final probs of {self.agent_id} regarding {who}: {po_imagined}")
    #print(f"These are the final obs imagined: {o_imagined}")



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

        u_imagined, pu_imagined = self.imagine_u(who)

        #if self.agent_id == 1:
            #print(f"These are the imagined u by {self.agent_id} regarding {who} with their probabilities: \n {u_imagined} \n {pu_imagined}")

        H = np.zeros(len(u_imagined))

        
        for i,u in enumerate(u_imagined):

          H[i]=self.compute_EFE(u,who)

          H[i] = self.discount_EFE(H[i], current_depth)

          """
          if who == 0:
              print(f"\n I am considering {who} did action {u} as first move")"""
          

          if current_depth < self.reason_depth:
            cp = self.s.copy()

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
            
            #if self.agent_id == 1:
                #print(f'this is the current EFE: {H[i]} \n and this is the future efe: {future_EFE}')
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
    """
    if current_depth == 2:
        print(f"These are the observations imagined by {self.agent_id} regarding {who} \n {o_imagined} \n {po_imagined}")
    """
    Hro = np.zeros(len(o_imagined))

    cp1 = self.s.copy()
    for i,o in enumerate(o_imagined):

      self.observe(o,who)
      u_imagined, _ = self.imagine_u(who)
      Hrou = np.zeros(len(u_imagined))

      
      for j,u in enumerate(u_imagined):

        Hrou[j] = self.compute_EFE(u,who)

        Hrou[j] = self.discount_EFE(Hrou[j], current_depth)
        
        cp2 = self.s.copy()
        if current_depth < self.reason_depth:

          if who == 0:
            self.act(u)
            future_EFE = self.SFEr(current_depth + 1, who = who)

          else:
            self.act_other(u)
            future_EFE = self.SFEr(current_depth + 1, who = who)

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

    
    """if current_depth == 2:
        print(f"These are the observations imagined by {self.agent_id} regarding {0} at depth {current_depth}: \n {o_imagined} \n {po_imagined} ")
    """

    Hro = np.zeros(len(o_imagined))

    cp1 = self.s.copy()
    for i,o in enumerate(o_imagined):
      
      """if current_depth == 2:
          print(f"I am considering {1} observed {o} at step 2 ")"""

      self.observe(o,0)
      self.observe(o,1)

      up_imagined, pup_imagined, H = self.sop_imagine_u(1)
      
      """if current_depth == 2:
          
          print(f"These are the actions imagined by {self.agent_id} regarding {1} at depth {current_depth}: \n {up_imagined} \n {pup_imagined} ")
          print(f"And these are the relative action efes \n {H}")
          print(f"_____________")"""

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

            Hropu[l] = self.discount_EFE(Hropu[l], current_depth)


            if current_depth < self.reason_depth:
              cp4 = self.s.copy()
              self.act(u)

              future_EFE = self.SFEr2(current_depth + 1)

              Hropu[l] = future_EFE + Hropu[l]
              self.s = cp4.copy()

          Hrop[k] = np.dot(softmax(Hropu, 1), Hropu)
          self.s = cp3.copy()

        Hup[j] = np.dot(pop_imagined, Hrop)
        self.s = cp2.copy()
      """if current_depth == 2:
          print(f"These are the EFES of each action imagined by {self.agent_id} regarding {1} \n {Hup} ")
      """
      Hro[i] = np.dot(pup_imagined, Hup)
      self.s = cp1.copy()

    """if current_depth == 2:
        print(f"This are the EFES of each observation imagined by {self.agent_id} regarding {0} \n {Hro}")"""
    future_EFE = np.dot(po_imagined, Hro)

    """if current_depth == 2:
        print(f"This is the future efe of the action considered {future_EFE}")"""

    return future_EFE


  def pick_action(self):
      
      if self.random_moves == False:

          u_imagined, H = self.SFE_call_self()
          print(f"These are the actions considered by agent {self.agent_id} with their dec 1 probs {u_imagined}, {np.round(softmax(H,1),2)}")
          assert len(u_imagined) > 0, "No action imagined"

          if self.det_dec_rule == True:
              action = u_imagined[np.argmin(H)]
      
          else:
              uprobs = softmax(H, self.dec_temp)
              action = np.random.choice(u_imagined, p = uprobs)

      
      else:
          action = np.random.choice(self.U)
          
          u_imagined = None 

    
      self.moves_to_go = self.moves_to_go - 1
      self.change_reason_depth()

      print(f"this is the action taken: {action}")

      #do this to put EFE to 0 of the actions not in use THIS IS WRONGGGGGGGG PROB NEED GO TO ZERO
      p_extended = np.zeros(4)
      p_extended[u_imagined] = softmax(H, 1)
      


      return action, p_extended
  
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

    assert self.reason_depth <= self.moves_to_go


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
          o = 7
        elif self.cheese_positions[0] == 1 and self.cheese_positions[1] == 1:
          o = 8 #both cheeses on the right

        """o = self.cheese_positions[0] #which can be either 0 or 1 THIS OR THE PART ABOVE"""


      elif (self.true_s[2,1] == 1 and self.cheese_positions[1] == 0) or (self.true_s[3,1] == 1 and self.cheese_positions[1] == 1):
        o = 9 #agent 1 on the cheese
      elif (self.true_s[2,1] == 1 and self.cheese_positions[1] == 1) or (self.true_s[3,1] == 1 and self.cheese_positions[1] == 0):
        o = 10 #agent 1 not on the cheese

      else:
        o = 11 #no observation for anybody

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

    agent0_A = norm_om(agent0_A)
    agent1_A = norm_om(agent1_A)


    return agent0_A, agent1_A


def true_A_new(base, pretrain_strength = float, other_strength = float):

    
    agent0_A = np.ones((12,2,2,4,4))*base

    agent0_A[0,0,:,1,:] = pretrain_strength

    agent0_A[1,1,:,1,:] = pretrain_strength

    agent0_A[2,0,:,2,:] = pretrain_strength
    agent0_A[2,1,:,3,:] = pretrain_strength

    agent0_A[3,1,:,2,:] = pretrain_strength
    agent0_A[3,0,:,3,:] = pretrain_strength

    agent0_A[4,:,:,0,:] = pretrain_strength

    agent0_A[5,0,0,:,1] = other_strength

    agent0_A[6,0,1,:,1] = other_strength

    agent0_A[7,1,0,:,1] = other_strength

    agent0_A[8,1,1,:,1] = other_strength

    agent0_A[9,:,0,:,2] = other_strength
    agent0_A[9,:,1,:,3] = other_strength

    agent0_A[10,:,0,:,3] = other_strength
    agent0_A[10,:,1,:,2] = other_strength

    agent0_A[11,:,:,:,0] = other_strength




    agent1_A = np.ones((12,2,2,4,4))*base

    agent1_A[0,0,:,1,:] = other_strength

    agent1_A[1,1,:,1,:] = other_strength

    agent1_A[2,0,:,2,:] = other_strength
    agent1_A[2,1,:,3,:] = other_strength

    agent1_A[3,1,:,2,:] = other_strength
    agent1_A[3,0,:,3,:] = other_strength

    agent1_A[4,:,:,0,:] = other_strength

    agent1_A[5,0,0,:,1] = pretrain_strength

    agent1_A[6,0,1,:,1] = pretrain_strength

    agent1_A[7,1,0,:,1] = pretrain_strength

    agent1_A[8,1,1,:,1] = pretrain_strength

    agent1_A[9,:,0,:,2] = pretrain_strength
    agent1_A[9,:,1,:,3] = pretrain_strength

    agent1_A[10,:,0,:,3] = pretrain_strength
    agent1_A[10,:,1,:,2] = pretrain_strength

    agent1_A[11,:,:,:,0] = pretrain_strength

    agent0_A = norm_om(agent0_A)
    agent1_A = norm_om(agent1_A)

    assert agent0_A.all() == agent1_A.all()

    true_A = agent0_A

    return true_A


def run(agent0, agent1, epochs = int, nr_moves = int, reason_depth = int,start_beliefs = np.array,
        exploration_phase = bool, A_learning = bool,
        infer_only_on_last_action = bool, opt_params = dict, nr_A_learning_moves = int, message_success_prob = float,
        selfish_episodes = int):

  prev_best = 100000

  epochs_agent_worlds = [] #save in which world each agent is
  epochs_positions = [] #true positions of each agent
  epochs_A_matrices = [] #A matrices of each agent (my, his)
  epochs_beliefs = []
  epochs_desires = []
  epochs_best_act_counts = []
  epochs_logls = []
  
  epochs_observations_agent_0 = {"0": [], #regarding agent 0
                                 "1": []} #regarding agent 1
  
  epochs_actions_agent_0 = {"0": [],
                            "1": []}

  epochs_observations_agent_1 = {"0": [],
                                 "1": []}

  epochs_actions_agent_1 = {"0": [],
                            "1": []}
  

  epochs_A_matrices.append((agent0.A.copy(), agent1.A.copy()))
  epochs_desires.append((agent0.goal_states.copy(),agent1.goal_states.copy())) #THIS NOT IN USE


  cheeses = []
  for i in range(epochs):
      cheeses.append([np.random.randint(0,2), np.random.randint(0,2)])



  epochs_pus = []

  best_move_0 = [1]
  best_move_1 = [0,1]


  for e in range(epochs):
    print("__________________________________")
    print("Starting epoch: " + str(e))
    print("__________________________________")


    epochs_agent_worlds.append((agent0.agent_id, agent1.agent_id))
    
  
    agent0.restart_agent(start_beliefs, reason_depth, nr_moves)
    agent1.restart_agent(start_beliefs, reason_depth, nr_moves)

    w = world(cheeses[e])


    best_act_counter0 = 0
    best_act_counter1 = 0

    pus0 = []
    pus1 = []

    positions = []
    beliefs = []
    agentselfposA = []
    agentselfposB = []

    acts1 = [] #now used for full action sequence goal learning
    bels = [] #now used for full action sequence goal learning

    beliefs.append((agent0.s.copy(),agent1.s.copy()))


    for k in range(nr_moves):



      if agent0.agent_id == 0: #if agent0 in env 1 he explores, if agent1 in env 0 he explores

        u0, pu0 = agent0.pick_action()

        pus0.append(pu0)


        agent0.act(u0)
        agent1.act_other(u0)


        w.change_true_s(agent0, u0)
        o = w.give_o(agent0)
        
        positions.append(pos_conv(w.true_s))
        agentselfposA.append(agent0.s.copy())

        
        agent0.observe(o,0)
        agent0.observe(o,1)
        agent1.observe(o,0)
        agent1.observe(o,1)


        beliefs.append((agent0.s.copy(),agent1.s.copy()))
        bels.append(agent0.s.copy()) #now used to full action sequence goal learning
        
   
        
        u1, pu1 = agent1.pick_action()

        pus1.append(pu1)


        acts1.append(u1) #now used to full action sequence goal learning


        agent1.act(u1)
        agent0.act_other(u1)
        

        w.change_true_s(agent1, u1)
        o = w.give_o(agent1)
   
        positions.append(pos_conv(w.true_s))
        agentselfposB.append(agent1.s.copy())
    
      
        agent0.observe(o,0)
        agent0.observe(o,1)
        agent1.observe(o,0)
        agent1.observe(o,1)
          
        beliefs.append((agent0.s.copy(),agent1.s.copy()))

        
        if k == 0 and u0 == best_move_0[k]:
            best_act_counter0 += 1
        
        elif cheeses[e][0] == 0 and k == 1 and u0 == 2:
            best_act_counter0 += 1
        
        elif cheeses[e][0] == 1 and k == 1 and u0 == 3:
            best_act_counter0 += 1

        elif cheeses[e][0] == 0 and k == 2 and u0 != 3:
            best_act_counter0 += 1

        elif cheeses[e][0] == 1 and k == 2 and u0 != 2:
            best_act_counter0 += 1

        if k < 2 and u1 == best_move_1[k]:
            best_act_counter1 = best_act_counter1 + 1

        elif cheeses[e][1] == 0 and k == 2 and u1 == 2:
            best_act_counter1 = best_act_counter1 + 1

        elif cheeses[e][1] == 1 and k == 2 and u1 == 3:
            best_act_counter1 = best_act_counter1 + 1


        



        #learning the goals of the other OLD VERSION, NOW TRY THE WHOLE ACTION SEQUENCE
        if 1 == 0:

            first = 20
            second = 20
            
            #agent0.goal_learning_beta = 1/(e+1)
            
            if infer_only_on_last_action == True:
                if k == nr_moves-1:
                    agent0.update_goals_other(u1, beliefs[-2][0], opt_params,k)
                    #print(f"These are the learned goals at the end of the episode: \n {np.round(agent0.goal_states[1],2)}")
            
        
            
            else:
                if e < first:
                    if k == nr_moves-1:
                        agent0.update_goals_other(u1, beliefs[-2][0], opt_params,k)

                elif e >= first and e < second:
                    if k == nr_moves-1 or k == nr_moves-2:
                        agent0.update_goals_other(u1, beliefs[-2][0], opt_params,k)
                    
                elif e >= second:
                    agent0.update_goals_other(u1, beliefs[-2][0], opt_params,k) #use my beliefs about his beliefs before that we acted


        


    

      else:
        1/0, f"Don't use this, exploration phase is now dead!!!"

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

    


    if agent0.empathetic == True:
        logl = agent0.update_goals_other(acts1, bels, opt_params, k)
        epochs_logls.append(logl)

        

    
    #learning the OM
    if A_learning == True:
        
        agent0_acts0 = []
        agent0_acts1 = []
        agent0_obs0 = []
        agent0_obs1 = []

        agent1_acts0 = []
        agent1_acts1 = []
        agent1_obs0 = []
        agent1_obs1 = []
        

        w = world(cheeses[e])

        messaging_ep = np.random.binomial(1,message_success_prob,1)




        if e < selfish_episodes:
            messaging_ep = 0
        else:
            messaging_ep = 1


        for k in range(nr_A_learning_moves):

            u = np.random.choice(agent0.U)
            w.change_true_s(agent0, u)
            o = w.give_o(agent0)

            agent0_acts0.append(u)
            agent0_obs0.append(o)
            agent1_acts1.append(u)
            agent1_obs1.append(o)


            if messaging_ep:
                agent1_acts0.append(u)
                agent1_obs0.append(o)
                agent0_acts1.append(u)
                agent0_obs1.append(o)

            


            u = np.random.choice(agent1.U)
            w.change_true_s(agent1, u)
            o = w.give_o(agent1)

            agent0_acts0.append(u+4)
            agent0_obs0.append(o)
            agent1_acts1.append(u+4)
            agent1_obs1.append(o)


            if messaging_ep:
                agent0_acts1.append(u+4)
                agent0_obs1.append(o)
                agent1_acts0.append(u+4)
                agent1_obs0.append(o)

            

        epochs_observations_agent_0["0"].append(agent0_obs0)
        epochs_observations_agent_0["1"].append(agent0_obs1) if agent0_obs1 else None

        epochs_actions_agent_0["0"].append(agent0_acts0)
        epochs_actions_agent_0["1"].append(agent0_acts1) if agent0_acts1 else None

        epochs_observations_agent_1["0"].append(agent1_obs0) if agent1_obs0 else None
        epochs_observations_agent_1["1"].append(agent1_obs1)

        epochs_actions_agent_1["0"].append(agent1_acts0) if agent1_acts0 else None
        epochs_actions_agent_1["1"].append(agent1_acts1)
  
        
        #agent0 learning his own environment
        if e < selfish_episodes:
            new_gm, [start, end, gm] = EM2(obs_eps = epochs_observations_agent_0["0"],
                          acts_eps = epochs_actions_agent_0["0"],
                          tm = agent0.B,
                          A_param_shape = (12,2,2,4,4),
                          s_param_shape = (2,2,4,4),
                          gm = np.load("inits/ta_gm.npy"),
                          start = np.load("inits/ta_start.npy"),
                          end = np.load("inits/ta_end.npy"))
            
            agent0.A[0] = new_gm.copy()
            
            #agent1 learning his own environment
            """new_gm, [start, end, gm] = EM2(obs_eps = epochs_observations_agent_1["1"],
                                acts_eps = epochs_actions_agent_1["1"],
                                tm = agent0.B,
                                A_param_shape = (12,2,2,4,4),
                                s_param_shape = (2,2,4,4),
                                gm = np.load("inits/ta_gm.npy"),
                                start = np.load("inits/ta_start.npy"),
                                end = np.load("inits/ta_end.npy"))"""
                  
            agent1.A[0] = new_gm.copy()
        

        #agent0 learning other agents environment
        else:
            new_gm, [start, end, gm] = EM2(obs_eps = epochs_observations_agent_0["1"],
                          acts_eps = epochs_actions_agent_0["1"],
                          tm = agent0.B,
                          A_param_shape = (12,2,2,4,4),
                          s_param_shape = (2,2,4,4),
                          gm = np.load("inits/ta_gm.npy"),
                          start = np.load("inits/ta_start.npy"),
                          end = np.load("inits/ta_end.npy"))
            
            agent0.A[1] = new_gm.copy()

        print("I did a GM learning update")
       
        """
        new_KL = A_KL(agent0.true_A[1], new_gm)
        if new_KL < prev_best:
            prev_best = new_KL
            np.save("inits/ta_start", start)
            np.save("inits/ta_end", end)
            np.save("inits/ta_gm", gm)
            print("I saved the params")"""

  

    epochs_positions.append(positions)

    epochs_A_matrices.append((agent0.A.copy(), agent1.A.copy()))

    epochs_beliefs.append(beliefs)

    epochs_desires.append((agent0.goal_states.copy(),agent1.goal_states.copy()))

    epochs_pus.append([pus0,pus1])
    
    epochs_best_act_counts.append([best_act_counter0, best_act_counter1])


  return epochs_positions, epochs_A_matrices, epochs_agent_worlds, epochs_beliefs, epochs_desires, cheeses, epochs_pus, epochs_best_act_counts, epochs_logls


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



