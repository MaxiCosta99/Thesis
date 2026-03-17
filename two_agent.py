import numpy as np

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
               little_o_imaginer = bool, o_imagination_breadth = int, o_prob_threshold = float,
               little_u_imaginer = bool, u_imagination_breadth = int, u_prob_threshold = float,
               agent_id = int, moves_to_go = int, social = bool, terms = list):
    self.A = A #[my/your]
    self.s = s #[my/your belief, agent0/agent1, loc, cheeseleft/cheeseright]
    self.B = B
    self.U = U
    self.goal_states = goal_states #[my/your]
    self.goal_obs = goal_obs
    self.reason_depth = reason_depth
    self.chatty_boy = chatty_boy
    self.dec_temp = dec_temp
    self.little_o_imaginer = little_o_imaginer
    self.o_imagination_breadth = o_imagination_breadth
    self.o_prob_threshold = o_prob_threshold
    self.little_u_imaginer = little_u_imaginer
    self.u_imagination_breadth = u_imagination_breadth
    self.u_prob_threshold = u_prob_threshold


    self.agent_id = agent_id
    self.moves_to_go = moves_to_go
    self.social = social
    self.terms = terms


  def update_A(self, o, s, who):
    

    self.s = s.copy()
    self.observe(o,0) #infer which state we were in

    """
    print(f"This is the state before observing: {s[0]}")
    print(f"This is the state after observing: {self.s[0]}")
    print(f"This is the observation: {o}")
    print(f"This is A before update: {self.A[0,o,...]}")"""

    self.A[who,o,...] = self.A[who,o,...].copy() + self.s[0].copy()

    #print(f"This is the updated A: {self.A[0,o,...]}")


  def observe(self, o, who):

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
    #mybel_his_s = self.s[1].copy()

    action_matrix = self.B[u]

    if self.agent_id == 0:
      self.s[0] = np.einsum("ijkl, mk -> ijml", my_s, action_matrix).copy()
      #self.s[1] = np.einsum("ijkl, mk -> ijml", mybel_his_s, action_matrix).copy() #he doesnt know where I go

    elif self.agent_id == 1:
      self.s[0] = np.einsum("ijkl, ml -> ijkm", my_s, action_matrix).copy()

    else:
      assert 1/0, "WHAT THE SHID? THERE IS ANOTHER???"


    assert abs(np.sum(self.s[0])-1)<0.01, f"My states probs after my action sum to {np.sum(self.s[0])}"
    assert abs(np.sum(self.s[1])-1)<0.01, f"His states probs after my action sum to {np.sum(self.s[1])}"


  def act_other(self, u):

    my_s = self.s[0].copy()
    mybel_his_s = self.s[1].copy()

    action_matrix = self.B[u]

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
    whose_s = self.s[who].copy()
    whose_goals = self.goal_states[who]


    #print(f"These is {self.agent_id}'s perspective of {who} goal : {whose_goals}")

    assert whose_s.shape == whose_goals.shape

    s_flat = whose_s.reshape(-1)
    goals_flat = whose_goals.reshape(-1)

    assert s_flat.shape == goals_flat.shape

    temp = np.divide(s_flat, goals_flat)
    temp1 = np.where(temp != 0, np.log(temp), 0)




    #temp1 = np.log(temp, where=(temp != 0), out=np.zeros_like(temp))

    risk = np.dot(s_flat, temp1)
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


      pu = softmax(pu)

      #print(f"Agent {self.agent_id} gives the following probs for each action: {pu}")


      #u_imagined = np.random.choice(len(self.U), size = self.u_imagination_breadth, replace = False, p = pu) to use if sampling
      u_imagined = np.array(self.U)[pu>self.u_prob_threshold]



      pu_imagined = pu[pu>self.u_prob_threshold]
      pu_imagined = pu_imagined/np.sum(pu_imagined)

      #print(f"the EFEs of these actions are: {pu}")

    else:
      u_imagined = self.U
      pu_imagined = np.ones(len(u_imagined))/len(u_imagined)

    #print(f"the actions imagined are: {u_imagined}")

    assert abs(np.sum(pu_imagined)-1)<0.01,f"Actions probabilities sum to {np.sum(pu_imagined)}"

    assert len(u_imagined) >0, "No action was imagined"

    return u_imagined, pu_imagined


  def sop_imagine_u(self, who = int):

    u, pu = self.SFE(who)

    if self.little_u_imaginer == True:

      #u_imagined = np.random.choice(len(self.U), size = self.u_imagination_breadth, replace = False, p = pu) to use if sampling
      u_imagined = u[pu>self.u_prob_threshold]
      pu_imagined = pu[pu>self.u_prob_threshold]
      pu_imagined = pu_imagined/np.sum(pu_imagined)
      #print(f"the EFEs of these actions are: {pu}")

    else:
      u_imagined = u
      pu_imagined = pu

    #print(f"the actions imagined are: {u_imagined}")

    assert len(u_imagined)>0, f"No actions sampled"
    assert abs(np.sum(pu_imagined)-1) < 0.01, f"probabilities of actions sum to {np.sum(pu_imagined)}"
  

    return u_imagined, pu_imagined


  def imagine_o(self, who = int):
    
    A_norm = self.A[who]/np.sum(self.A[who], axis = 0)
    po = np.einsum("ijklm, jklm -> i", A_norm, self.s[who]) #added l, maybe don't work


    if self.little_o_imaginer == True:

      o_imagined = np.arange(self.A[who].shape[0])[po > self.o_prob_threshold]
      #o_imagined = np.random.choice(self.A.shape[0], size = self.o_imagination_breadth, replace = True, p = po) to use if sampling
      #print(o_imagined)

      po_imagined = po[o_imagined]/np.sum(po[o_imagined])

    else:
      o_imagined = np.arange(self.A[who].shape[0])
      po_imagined = po

    if len(o_imagined)==0:
      #print(f"Agent {self.agent_id} didn't imagine any observation because all of them are below the threshold")
      o_imagined = [np.random.uniform(0,self.A[who].shape[0])]
      po_imagined = [1]

    assert abs(np.sum(self.s[who])-1)<0.01, f"States probs sum to {np.sum(self.s[who])}"
    assert abs(np.sum(po)-1)<0.01, f"Observation probs sum to {np.sum(po)}"

    return o_imagined, po_imagined


  def SFE(self, who = int):

    current_depth = 1

    u_imagined, _ = self.imagine_u(who)

    #print(f"The actions considered by agent{self.agent_id} at depth {current_depth} are: {u_imagined} with probability {u_probs_temp}")

    H = np.zeros(len(u_imagined))

    cp = self.s.copy()
    for i,u in enumerate(u_imagined):


      H[i]=self.compute_EFE(u,who)


      if current_depth < self.reason_depth:

        if self.social == True:

          if who == 0:
            #print(f"This is the action considered {u} at depth {current_depth}")
            self.act(u)
            future_EFE = self.SFEr2(current_depth + 1)
          else:
            self.act_other(u)
            future_EFE = self.SFEr(current_depth + 1,who = 1)

        else:
          self.act(u)
          future_EFE = self.SFEr(current_depth + 1, who = 0)

        H[i] = future_EFE #+ H[i]

        self.s = cp.copy()
    
      #print(f"THIS IS THE ACTION BEING CONSIDERED AT DETPH {current_depth}: {u} with EFE {H[i]}")


    return u_imagined, softmax(np.array(H), self.dec_temp) #1 is the decision temperature


  def SFEr(self, current_depth = int, who = int):


    o_imagined, po_imagined = self.imagine_o(who)
    Hro = np.zeros(len(o_imagined))

    #print(f"The observations considered by agent{self.agent_id} at depth {current_depth} are: {o_imagined} with probability {po_imagined}")


    cp1 = self.s.copy()
    for i,o in enumerate(o_imagined):
      
      #print(f"THIS IS THE OBSERVATION BEING CONSIDERED AT DEPTH {current_depth}: {o}")

      self.observe(o,who)

      u_imagined, _ = self.imagine_u(who)
      Hrou = np.zeros(len(u_imagined))

      #print(f"The actions considered by agent{self.agent_id} at depth {current_depth} are: {u_imagined} with probability {u_probs_temp_2}")
      

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

          Hrou[j] = future_EFE #+Hrou[j] 
          self.s = cp2.copy()

        #print(f"THIS IS THE ACTION BEING CONSIDERED AT DEPTH {current_depth}, {u}, WITH CUM_EFE {Hrou[j]}")


      pur = softmax(Hrou) #makes more sense to have 1 decision temperature since this is counterfactual reasoning


      Hro[i] = np.dot(pur, Hrou)
      self.s = cp1.copy()

    """if self.little_o_imaginer == True:
      future_EFE = sum(Go)/len(Go)
    else:
      future_EFE = np.dot(po, np.array(Go))"""

    future_EFE = np.dot(po_imagined, Hro)

    return future_EFE


  def SFEr2(self, current_depth = int):


    o_imagined, po_imagined = self.imagine_o(0) #0 is I imagine

    """
    if current_depth == 2:
      print(f"The obsevations considered by agent {self.agent_id} at depth {current_depth} are: {o_imagined} with probability: {po_imagined}")
    """

    Hro = np.zeros(len(o_imagined))

    cp1 = self.s.copy()
    for i,o in enumerate(o_imagined):


      self.observe(o,0)
      self.observe(o,1)

      """
      if current_depth == 2:
          print(f"This is the observation considered by me {o} at depth {current_depth}")"""
      #print(f"This is the depth before he acting {current_depth}")
      up_imagined, pup_imagined = self.sop_imagine_u(1) #he does the SFE CAN BE EXTENDED USING SAMPLE

      #### FOR DEBUGGGINIG

      #up_imagined = np.array(self.U)
      #pup_imagined = np.ones(len(self.U))/len(self.U)
      #up_imagined, pup_imagined = self.imagine_u(1)

      """if current_depth ==2:
          print(f"At depth {current_depth} agent {self.agent_id} imagined that in the subsequent action the other agent would perform action:\n {up_imagined} with probability: \n {pup_imagined}")
      """
      #print(f"This is the depth after he acting {current_depth}")
      ################

      Hup = np.zeros(len(up_imagined))

      cp2 = self.s.copy()

      for j,up in enumerate(up_imagined):

        #print(f"This is the action considered by him {up} at depth {current_depth}")
        self.act_other(up) #this act both changes my beliefs and his beliefs

        op_imagined, pop_imagined = self.imagine_o(1)
        Hrop = np.zeros(len(op_imagined))

        cp3 = self.s.copy()
        for k,op in enumerate(op_imagined):
          #print(f"This is the observation considered by him {op} at depth {current_depth}")
          self.observe(op,1)
          self.observe(op,0)

          u_imagined, _ = self.imagine_u(0)
          Hropu = np.zeros(len(u_imagined))

          for l,u in enumerate(u_imagined):
            Hropu[l] = self.compute_EFE(u,0)



            if current_depth < self.reason_depth:
              #print(f"This is the action considered {u} at depth {current_depth}")
              cp4 = self.s.copy() #he doesnt know what I do UP TO DEBATE
              self.act(u)

              future_EFE = self.SFEr2(current_depth + 1)
              Hropu[l] = future_EFE #+ Hropu[l]
              self.s = cp4.copy()


          Hrop[k] = np.dot(softmax(Hropu), Hropu)
          self.s = cp3.copy()

        Hup[j] = np.dot(pop_imagined, Hrop)
        self.s = cp2.copy()

      Hro[i] = np.dot(pup_imagined, Hup)
      """if current_depth == 2:
          print("----------------------------------")
          print(f"Agent {self.agent_id} considers the actions taken by the other agent \n with probability {pup_imagined} \n and EFE {Hup}")
          print("----------------------------------")"""
      self.s = cp1.copy()

    """if self.little_o_imaginer == True:
      future_EFE = sum(Go)/len(Go)
    else:
      future_EFE = np.dot(po, np.array(Go))"""

    future_EFE = np.dot(po_imagined, Hro)

    return future_EFE


  def pick_action(self):

    u_imagined, uprobs = self.SFE(0)

    assert len(u_imagined) > 0, "No action imagined"

    #print(f"This is the current belief state of agent {self.agent_id}: {np.einsum("ijkl -> jl", self.s[0].copy())}")

    print(f"These are the actions considered by agent {self.agent_id} with their probability {u_imagined}, {uprobs}")

    action = np.random.choice(u_imagined, p = uprobs)

    print(f"This is the real action taken by agent {self.agent_id}: {action}")

    self.moves_to_go = self.moves_to_go - 1

    self.change_reason_depth()

    return action
  
  def explore(self):
    old_terms = self.terms.copy()
    #old_rd = self.reason_depth

    self.terms = ["nov"]
    #self.reason_depth = 1
    u_imagined, uprobs = self.SFE(1)

    self.terms = old_terms.copy()
    #self.reason_depth = old_rd

    #action = np.random.choice(self.U)
    action = np.random.choice(u_imagined, p = uprobs)

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


def true_A(base, pretrain_strength = float):
    
    agent0_A = np.ones((11,2,2,4,4))*base

    agent0_A[0,0,:,1,:] = pretrain_strength

    agent0_A[1,1,:,1,:] = pretrain_strength

    agent0_A[2,0,:,2,:] = pretrain_strength
    agent0_A[2,1,:,3,:] = pretrain_strength

    agent0_A[3,1,:,2,:] = pretrain_strength
    agent0_A[3,0,:,3,:] = pretrain_strength

    agent0_A[4,:,:,0,:] = pretrain_strength


    agent1_A = np.ones((11,2,2,4,4))*base

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

def run(agent0, agent1, epochs = int, nr_actions = int, start_beliefs = np.array, cheeses = list, exploration_phase = bool):


  epochs_positions = []
  epochs_A_matrices = []

  epochs_A_matrices.append((agent0.A.copy(), agent1.A.copy()))

  epochs_agent_worlds = []

  for e in range(epochs):
    print("__________________________________")
    print("Starting epoch: " + str(e))
    print("__________________________________")

    if exploration_phase == True:
        where_are_you = np.random.randint(0,2)
        agent0.change_environment(where_are_you)
        agent1.change_environment(abs(where_are_you-1))

    epochs_agent_worlds.append((agent0.agent_id, agent1.agent_id))
    


    agent0.restart_beliefs(start_beliefs)
    agent1.restart_beliefs(start_beliefs)
    w = world(cheeses[e])


    positions = []
    observationsA = []
    agentselfposA = []
    
    observationsB = []
    agentselfposB = []

    agent0.reason_depth = nr_actions
    agent1.reason_depth = nr_actions
    agent0.moves_to_go = nr_actions
    agent1.moves_to_go = nr_actions


    for _ in range(nr_actions):

      if agent0.agent_id == 0: #if agent0 in env 1 he explores, if agent1 in env 0 he explores
  
        u = agent0.pick_action()
        agent0.act(u)
        #print("This is the action taken by agent 0: " + str(u))

        w.change_true_s(agent0, u)
        o = w.give_o(agent0)
        #print(f"This is the observation given by the world: {o}")
        positions.append(pos_conv(w.true_s))
        agentselfposA.append(agent0.s.copy())
        observationsA.append(o)

        agent0.observe(o,0)
        agent0.observe(o,1)
        agent1.observe(o,0)
        agent1.observe(o,1)

    


        
        u = agent1.pick_action()
        agent1.act(u)
        #print("This is the action taken by agent 1: " + str(u))

        w.change_true_s(agent1, u)
        o = w.give_o(agent1)
        #print(f"This is the observation given by the world: {o}")
        positions.append(pos_conv(w.true_s))
        agentselfposB.append(agent1.s.copy())
        observationsB.append(o)

        agent0.observe(o,0)
        agent0.observe(o,1)
        agent1.observe(o,0)
        agent1.observe(o,1)

      else:
        u = agent0.explore()
        agent0.act(u)
        #print("This is the action taken by agent 0: " + str(u))

        w.change_true_s(agent0, u)
        o = w.give_o(agent0)
        #print(f"This is the observation given by the world: {o}")
        positions.append(pos_conv(w.true_s))
        agentselfposA.append(agent0.s.copy())
        observationsA.append(o)

        agent0.observe(o,0)
        agent0.observe(o,1)
        agent1.observe(o,0)
        agent1.observe(o,1)
        


        u = agent1.explore()
        agent1.act(u)
        #print("This is the action taken by agent 1: " + str(u))

        w.change_true_s(agent1, u)
        o = w.give_o(agent1)
        #print(f"This is the observation given by the world: {o}")
        positions.append(pos_conv(w.true_s))
        agentselfposB.append(agent1.s.copy())
        observationsB.append(o)

        agent0.observe(o,0)
        agent0.observe(o,1)
        agent1.observe(o,0)
        agent1.observe(o,1)



    
    for (o,s) in zip(observationsA, agentselfposA):
        if agent0.agent_id == 0:
          agent0.update_A(o,s,who=0)
        else:
          agent0.update_A(o,s,who=1)

    for (o,s) in zip(observationsB, agentselfposB):
        if agent1.agent_id == 1:
          agent1.update_A(o,s,who=0)
        else:
          agent1.update_A(o,s,who=1)
    
    
    epochs_positions.append(positions)

    epochs_A_matrices.append((agent0.A.copy(), agent1.A.copy()))

    #print(f"At the end of epoch {e}, agent0 A is: {agent0.A}")
    #print(f"At the end of epoch {e}, agent1 A is: {agent1.A}")

  return epochs_positions, epochs_A_matrices, epochs_agent_worlds


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



