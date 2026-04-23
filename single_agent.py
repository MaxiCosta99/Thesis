import numpy as np
import math
from utils import EM, A_KL

def softmax(x, beta):
    return np.exp(-x/beta)/np.sum(np.exp(-x/beta))

def pos_conv(pos):
    """
    Converts a numerical position (0-3) to (row, col) coordinates on a 2x3 grid.
    Mapping: 0 -> (0,1), 1 -> (1,1), 2 -> (0,0), 3 -> (0,2)
    """

    agentpos = pos
    conv = int(np.squeeze(np.where(agentpos == 1)))

    mapping = {
        0: (0, 1),
        1: (1, 1),
        2: (0, 0),
        3: (0, 2)
    }
    conv = mapping.get(conv)
    return conv

class agent:
    def __init__(self, A, B, s, U,
               goal_states, reason_depth = int, dec_temp = float, chatty_boy = bool,
               little_o_imaginer = bool, o_imagination_breadth = int, o_prob_threshold = float,
               little_u_imaginer = bool, u_imagination_breadth = int, u_prob_threshold = float,
               goal_obs = None, moves_to_go = int, bayes_observe = bool, terms = list,
               obs_model_given = bool, infer_before_update_obs_model = bool,
               update_A_method = str, EM_steps = int, random_moves = bool):
        if obs_model_given == True:
            self.A = A.copy()
        else:
            self.A = np.random.beta(10,1,(5,4,2)).copy()
            #self.A = np.random.uniform(0,1,(5,4,2)).copy()
        self.s = s
        self.B = B
        self.U = U
        self.goal_states = goal_states
        self.reason_depth = reason_depth
        self.chatty_boy = chatty_boy
        self.dec_temp = dec_temp
        self.little_o_imaginer = little_o_imaginer
        self.o_imagination_breadth = o_imagination_breadth
        self.little_u_imaginer = little_u_imaginer
        self.u_imagination_breadth = u_imagination_breadth
        self.u_prob_threshold = u_prob_threshold
        self.o_prob_threshold = o_prob_threshold
        self.goal_obs = goal_obs
        self.moves_to_go = moves_to_go
        self.bayes_observe = bayes_observe
        self.terms = terms
        self.obs_model_given = obs_model_given
        self.infer_before_update_obs_model = infer_before_update_obs_model

        self.update_A_method = update_A_method
        self.EM_steps = EM_steps
        self.random_moves = random_moves
        self.true_A = A.copy()


    def update_A(self, ep_observations, ep_states, ep_actions):

        old_A_shape = self.A.shape

        if self.update_A_method == "default":
            for o,s in zip(ep_observations[-1], ep_states[-1]):
                self.s = s
                if self.infer_before_update_obs_model == True:
                    self.observe(o) #infer which state we were in before updating A
                self.A[o,:] = self.A[o,:] + self.s
        
        if self.update_A_method == "EM_all_ep":
            A = EM(ep_observations, ep_actions, self.B.copy(), num_steps = self.EM_steps, A_param_shape = (5,2,4), s_param_shape = (2,4), gm = None, start = None, end = None)

        if self.update_A_method == "EM_current_ep":
            A = EM(ep_observations[-1:], ep_actions[-1:], self.B.copy(), num_steps = self.EM_steps, A_param_shape = self.A.shape, s_param_shape = self.s.shape, gm = self.A.copy(), start = None, end = None)

        #THIS PART IS FOR ADJUST STATE SWITCHING
        A_flat = A/np.sum(A, axis = 0)

        A_invstate = np.zeros(A.shape)
        A_invstate[:,0:4] = A[:,4:8]
        A_invstate[:,4:8] = A[:,0:4]
        A_invstate_flat = A_invstate/np.sum(A_invstate, axis = 0)

        true_A = self.true_A.copy()
        true_A = true_A.reshape((5,8), order = "F")
        true_A_flat = true_A/np.sum(true_A, axis = 0)

        kl_state = A_KL(true_A_flat, A_flat)
        kl_invstate = A_KL(true_A_flat, A_invstate_flat)

        if kl_invstate < kl_state: #states have been switched
            self.A = A_invstate.reshape((5,4,2), order = "F")
        else:
            self.A = A.reshape((5,4,2), order = "F")


        assert self.obs_model_given == False, f"The obs model is given, can't update it"
        assert self.A.shape == old_A_shape, f"The shape of A changed from {old_A_shape} to {self.A.shape} after updating, something is wrong"


    def observe(self, o):
        if self.bayes_observe == False:

            if o == 4:
                if self.chatty_boy == 1:
                    print("No new info")

            elif o == 1:
                self.s = np.zeros(self.s.shape)
                self.s[1,o] = 1
                if self.chatty_boy == 1:
                    print("The cheese is on the right")

            elif o == 0:
                self.s = np.zeros(self.s.shape)
                self.s[1,o] = 1
                if self.chatty_boy == 1:
                    print("The cheese is on the left")

            elif o == 2:
              if np.sum(self.s[2,:]) == 1:
                self.s = np.zeros(self.s.shape)
                self.s[2,0] = 1

              elif np.sum(self.s[3,:]) == 1:
                self.s = np.zeros(self.s.shape)
                self.s[3,1] = 1

              if self.chatty_boy == 1:
                print("We are on the cheese")

            elif o == 3:
              if np.sum(self.s[2,:]) == 1:
                self.s = np.zeros(self.s.shape)
                self.s[2,1] = 1

              elif np.sum(self.s[3,:]) == 1:
                self.s = np.zeros(self.s.shape)
                self.s[3,0] = 1


              if self.chatty_boy == 1:
                print("We are not on the cheese")

        else:
            s = self.A[o]*self.s
            norm_s = s/np.sum(s)
            self.s = norm_s.copy()

    def act(self, u):
      self.s = np.dot(self.B[u,:], self.s)

    def restart_agent(self, start_s, reason_depth, nr_moves):
      self.s = start_s
      self.reason_depth = reason_depth
      self.moves_to_go = nr_moves

    def compute_ambiguity(self):
      flat_A = np.reshape(self.A, (self.A.shape[0], self.A.shape[1]*self.A.shape[2]))
      avg_A = flat_A/np.sum(flat_A, axis = 0, keepdims = True)
      temp = -np.diag(np.dot(avg_A.T, np.log(avg_A)))
      s_flat = np.reshape(self.s, (self.s.shape[0]*self.s.shape[1]))
      abg = np.dot(temp.T, s_flat)
      return abg

    def compute_risk(self):
      s_flat = np.reshape(self.s, (self.s.shape[0]*self.s.shape[1]))
      goals_flat = np.reshape(self.goal_states, (self.goal_states.shape[0]*self.goal_states.shape[1]))
      #risk = np.dot(s_flat.T, np.log(np.divide(s_flat, goals_flat)))
      #risk = np.sum(np.where(s_flat != 0, np.multiply(s_flat, np.log(np.divide(s_flat, goals_flat))), 0))
      #print(f"This is the log belief over states: {np.log(s_flat)} \n and these are the log goals: {np.log(goals_flat)}")
      #print(f" This is the divide term: {np.divide(s_flat, goals_flat)}")
      #print(f" This is the log divide term: {np.where(s_flat != 0, np.log(np.divide(s_flat,goals_flat)), 0)}")


      temp = np.multiply(s_flat, np.log(np.divide(s_flat,goals_flat)))
      risk = np.nansum(temp)
      return risk

    def compute_risk_obs(self):

      A_probs = self.A/np.sum(self.A, axis = 0)
      po = np.einsum("ijk, jk -> i", A_probs, self.s)

      assert abs(np.sum(po)-1) < 0.01, f"probs sum to {np.sum(po)}"

      temp = np.divide(po, self.goal_obs)
      temp1 = np.where(temp != 0, np.log(temp), 0)

      #temp1 = np.log(temp, where=(temp != 0), out=np.zeros_like(temp))

      risk_obs = np.dot(po, temp1)

      return risk_obs

    def compute_novelty(self):
      A_flat = self.A.reshape(self.A.shape[0], -1)
      A_flat_norm = A_flat/np.sum(A_flat, axis = 0)
      s_flat = self.s.reshape(-1)

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

      assert self.obs_model_given == False, f"Novelty is useless if the observation model is not being learned"

      return nov

    def compute_epistemic_value(self, start_belief):


      
        A_flat = self.A.reshape(self.A.shape[0], -1).copy()
        A_flat_norm = A_flat/np.sum(A_flat, axis = 0)
        s_flat = self.s.reshape(-1).copy()

        num_obs = self.A.shape[0]

        po = np.dot(A_flat_norm, s_flat)


        ponz = po[po!=0]
        O = np.arange(num_obs)
        Onz = O[po!=0]

        

        #print(f"These are the Onz: {Onz}")
        #print(f"These are the ponz: {ponz}")

        kls = np.zeros(Onz.shape[0])

        bbo = self.s.copy() #belief before observing
        bbo_flat = bbo.reshape(-1)
        bba_flat = start_belief.reshape(-1) #NO DON'T USE THIS, MAKE NO SENSE
        for j,o in enumerate(Onz):
            self.observe(o)
            bao = self.s.copy() #belief after observing
            bao_flat = bao.reshape(-1)
            kls[j] = np.dot(bao_flat, np.where(bao_flat != 0, np.log(np.divide(bao_flat, bbo_flat)), 0))

            self.s = bbo.copy()

        epi_value = np.dot(ponz, kls)

        assert not np.isnan(epi_value), "Epi value is NaN"
        assert np.allclose(np.sum(po), 1.0), f"Observation probabilities do not sum to 1, they sum to {np.sum(po)}"

        return epi_value



    def compute_EFE(self, u):
      cp = self.s.copy()
      self.act(u)

      toconsider = 0
      term_values = {}

      if "abg" in self.terms:
          abg = self.compute_ambiguity()
          term_values["abg"] = abg
          toconsider = toconsider + abg

      if "risk" in self.terms:
          risk = self.compute_risk()
          term_values["risk"] = risk
          toconsider = toconsider + risk

      if "risk_obs" in self.terms:
          risk_obs = self.compute_risk_obs()
          term_values["risk_obs"] = risk_obs
          toconsider = toconsider + risk_obs

      if "nov" in self.terms:
          novelty = self.compute_novelty()
          term_values["novelty"] = novelty
          toconsider = toconsider - novelty

      if "epi_value" in self.terms:
          epi_val = self.compute_epistemic_value(cp)
          term_values["epi_value"] = epi_val
          toconsider = toconsider - epi_val
      
      EFE = toconsider
      
      """abg = self.compute_ambiguity()
      risk = self.compute_risk()
      risk_obs = self.compute_risk_obs()
      novelty = self.compute_novelty()"""
      #print("this is the abg: " + str(abg))
      #print("this is the risk: " + str(risk))
      #EFE = risk + abg
      #print(f"This is the novelty {novelty} and this is the risk {risk_obs}")

      #EFE = -novelty

      self.s = cp
      return EFE, term_values


    def imagine_u(self):
      if self.little_u_imaginer == True:

        pu = np.zeros(len(self.U))
        for u in self.U:
          #print(f"This is the action I am considering {u}")
          pu[u], _ = self.compute_EFE(u)

        #print(f"pus inside the imagine_u before softmax {pu}")
        pu = softmax(pu, 1)
        #print(f"pus inside the imagine_u {pu}")

        #u_imagined = np.random.choice(len(self.U), size = self.u_imagination_breadth, replace = False, p = pu) to use if sampling
        u_imagined = np.array(self.U)[pu>self.u_prob_threshold]
        #print(f"the EFEs of these actions are: {pu}")

      else:
        u_imagined = self.U

      #print(f"the actions imagined are: {u_imagined}")

      return u_imagined


    def imagine_o(self):
      po = np.einsum("ijk, jk -> i", self.A/np.sum(self.A, axis = 0), self.s)

      if self.little_o_imaginer == True:
        o_imagined = np.arange(self.A.shape[0])[po > self.o_prob_threshold]
        #o_imagined = np.random.choice(self.A.shape[0], size = self.o_imagination_breadth, replace = True, p = po) to use if sampling
        #print(o_imagined)

        po_imagined = po[o_imagined]/np.sum(po[o_imagined])

      else:
        o_imagined = np.arange(self.A.shape[0])
        po_imagined = po

      return o_imagined, po_imagined


    def SFE(self):

      current_depth = 1

      u_imagined = self.imagine_u()
      H = np.zeros(len(u_imagined))

      cp = self.s
      for i,u in enumerate(u_imagined):
        H[i], term_values = self.compute_EFE(u)
        # print(f" The term values of action {u} are {term_values} \n The EFE of that action is {H[i]}")

        if current_depth < self.reason_depth:
          self.act(u)
          future_EFE = self.SFEr(current_depth + 1)
          H[i] = H[i] + future_EFE
          #print(f"The future term of that action is {future_EFE}")
          #print(f"The SFE of that action is {H[i]}")
          self.s = cp

      #print(f"these are the probabilities before picking and before softmax: {H}")

      return u_imagined, np.array(H)


    def SFEr(self, current_depth):


      o_imagined, po_imagined = self.imagine_o()
      Hro = np.zeros(len(o_imagined))

      cp1 = self.s
      for i,o in enumerate(o_imagined):

        self.observe(o)

        u_imagined = self.imagine_u()
        Hrou = np.zeros(len(u_imagined))

        cp2 = self.s
        for j,u in enumerate(u_imagined):
          Hrou[j], _ = self.compute_EFE(u)

          if current_depth < self.reason_depth:
            self.act(u)
            future_EFE = self.SFEr(current_depth + 1)
            Hrou[j] = Hrou[j] + future_EFE
            self.s = cp2

        #print(f"The are the EFEs {Hrou} \n of the imagined actions {u_imagined} \n after the imaginary observation {o} \n at depth {current_depth}")
        pur = softmax(Hrou, 1) #makes more sense to have 1 decision temperature since this is counterfactual reasoning
        #print(f"And these are the probabilities of those actions: {pur}")

        Hro[i] = np.dot(pur, Hrou)
        self.s = cp1

      """if self.little_o_imaginer == True:
        future_EFE = sum(Go)/len(Go)
      else:
        future_EFE = np.dot(po, np.array(Go))"""

      future_EFE = np.dot(po_imagined, Hro)

      return future_EFE


    def pick_action(self):
      
      if self.random_moves == True:
          action = np.random.choice(self.U)
          
      else:
          u_imagined, Hu = self.SFE()

          #print(f"These are the actions imagined {u_imagined} \n With their EFE's: {Hu}")


          pu = softmax(Hu, self.dec_temp)

          action = np.random.choice(u_imagined, p = pu)

          #print(f"This is the real action taken: {action}")



      self.moves_to_go = self.moves_to_go - 1

      self.change_reason_depth()

      return action


    def change_reason_depth(self):

        if self.reason_depth > self.moves_to_go:
          self.reason_depth = self.reason_depth - 1
          assert self.reason_depth == self.moves_to_go, "SOMETHING IS SMELLY"

class world:

  def __init__(self, cheese_position):
    self.true_s = np.zeros(4)
    self.true_s[0] = 1
    self.cheese_position = cheese_position
    self.B = np.array(([[[0,0,0,0],[1,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]],[[0,0,0,1],[0,1,0,0],[1,0,1,0],[0,0,0,0]],[[0,0,1,0],[0,1,0,0],[0,0,0,0],[1,0,0,1]]]))

  def change_true_s(self, u):
    self.true_s = np.dot(self.B[u], self.true_s)


  def give_o(self):
    if self.true_s[1] == 1:
      o = self.cheese_position #which can be either 0 or 1
    elif (self.true_s[2] == 1 and self.cheese_position == 0) or (self.true_s[3] == 1 and self.cheese_position == 1):
      o = 2 #on the cheese
    elif (self.true_s[2] == 1 and self.cheese_position == 1) or (self.true_s[3] == 1 and self.cheese_position == 0):
      o = 3 #not on the cheese
    else:
      o = 4 #null observation
    return o

  def reset(self):
    self.true_s = 0

def pretrain(me: agent, epochs, start_s_beliefs, cheeses):

    actions = [0,1,2,3,3,2,1,0,0,1,2,3,3,2,1,0,0,1,2,3]

    epoch_positions = []

    for e in range(epochs):

        observations = []
        states = []
        me.restart_agent(start_s_beliefs,3,3)
        w = world(cheeses[e])

        positions = [] 

        for u in actions:
            me.act(u)
            w.change_true_s(u)
            states.append(me.s)
            positions.append(pos_conv(w.true_s))

            o = w.give_o()
            me.observe(o)
            observations.append(o)


        for o,s in zip(observations, states):
            me.update_A(o,s)
    
        epoch_positions.append(positions)

    return epoch_positions


def run(me = agent, num_eps = int, start_beliefs = np.ndarray, reason_depth = int, nr_moves = int):
    
    epoch_positions = []
    epoch_acts = []
    epoch_obs = []
    epoch_states = []
    A_matrices = []
    A_matrices.append(me.A.reshape(me.A.shape[0], -1, order = "F").copy())

        
    
    cheeses = []
    for i in range(num_eps):
        cheeses.append(np.random.choice([0,1]))


    for e in range(num_eps):
        print(" ")
        print("__________________________________")
        print("New world: " + str(e))
        me.restart_agent(start_beliefs, reason_depth, nr_moves)
        w = world(cheeses[e])
        
        
        observations = []
        states = []
        positions = []
        acts = []


        for _ in range(nr_moves):
            u = me.pick_action()
            me.act(u)

            states.append(me.s)

            w.change_true_s(u)

            o = w.give_o()
            me.observe(o)

            
            positions.append(pos_conv(w.true_s))
            acts.append(u)
            observations.append(o)

        

        epoch_positions.append(positions)
        epoch_obs.append(observations)
        epoch_acts.append(acts)
        epoch_states.append(states)




        
        if me.obs_model_given == False:
            me.update_A(epoch_obs, epoch_states, epoch_acts)

            print(f"This is the updated A {np.round(np.reshape(me.A, (me.A.shape[0], -1)),2)}")

        tempA = np.reshape(me.A.copy(), (me.A.shape[0], -1), order = "F") #this is to convert matrix into classic (2,4) (instead of the (4,2) used inside the algorithm)
        tempA = tempA/np.sum(tempA, axis = 0, keepdims = True)
        A_matrices.append(tempA)


    return epoch_positions, A_matrices, cheeses


if __name__ == "__main__":
    #start_B = np.array(([[[1,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,1,1]]]))
    start_B = np.array(([[[0,0,0,0],[1,1,0,0],[0,0,1,0],[0,0,0,1]],[[1,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]],[[0,0,0,1],[0,1,0,0],[1,0,1,0],[0,0,0,0]],[[0,0,1,0],[0,1,0,0],[0,0,0,0],[1,0,0,1]]]))
    start_A = np.ones((5,4,2))
    start_s = np.array([[0.5,0.5],[0,0],[0,0],[0,0]])


    goals = np.ones(start_s.shape)/8
    goals[2,0] = 1e1
    goals[3,1] = 1e1
    goals[2,1] = 1e-10
    goals[3,0] = 1e-10
    goals = goals/np.sum(goals)

    goal_obs = np.ones(start_A.shape[0])
    goal_obs[2] = 1e2
    goal_obs[3] = 1e-5
    goal_obs = goal_obs/np.sum(goal_obs)

    U = [0,1,2,3]
    reason_depth = 3
    dec_temp = 0.5
    little_o_imaginer = False
    o_imagination_breadth = 2
    o_prob_threshold = 0.1

    little_u_imaginer = False
    u_imagination_breadth = 3
    u_prob_threshold = 0.1


    #cheeses = [0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0]
    cheeses = [0,1,0,1,0,1,0,1,0,1,1,1]
    epochs = len(cheeses)
    nr_moves = 5

    maxi = agent(start_A, start_B, start_s, U = U, goal_states = goals,
                reason_depth = reason_depth, dec_temp = dec_temp, chatty_boy = 0,
                little_o_imaginer = little_o_imaginer, o_imagination_breadth = o_imagination_breadth, o_prob_threshold = o_prob_threshold,
                little_u_imaginer = little_u_imaginer, u_imagination_breadth= u_imagination_breadth, u_prob_threshold = u_prob_threshold,
                goal_obs = goal_obs, moves_to_go = nr_moves)

    run(maxi, epochs, nr_moves, start_s, cheeses)