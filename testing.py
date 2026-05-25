import numpy as np
import scipy
from two_agent import true_A_new
import json
from claude_plotting_two_agents import two_agent_plotter



exp_name = r"experiments\two_agent_sampling_om_20260524_181553.json"

with open(exp_name, "r") as f:
    data = json.load(f)

experiment_data = data["results"]


bac_eps_1 = experiment_data["bac_eps_1"]
bac_eps_2 = experiment_data["bac_eps_2"]
policy_eps_1 = experiment_data["policy_eps_1"]
policy_eps_2 = experiment_data["policy_eps_2"]
kls_eps = experiment_data["kls_eps"]
var_eps = experiment_data["var_eps"]
kls_eps_other = experiment_data["kls_eps_other"]
var_eps_other = experiment_data["var_eps_other"]

two_agent_plotter(kls_eps,
                          kls_eps_other,
                          var_eps,
                          var_eps_other,
                          policy_eps_1,
                          policy_eps_2,
                          bac_eps_1,
                          bac_eps_2
                          )
