import json
from two_agent_plotting_utils import two_agent_plotter, unstable_EM_two_maze, two_agent_goal_learning_plotter



exp_name = r'experiments\two_agent_sampling_om_20260524_181553.json'


exp_name = r'experiments/two_agent_learn_om_150eps_20260614_154605.json'
exp_name = r'experiments/two_agent_mean_om_10runs_20260613_190939.json'

exp_name = r'experiments/no_init_test_20260615_102032.json'
exp_name = r'experiments/red_halluc_150_eps_20260622_174511.json'
exp_name = r'experiments/cheese_link_20260622_190518.json'

exp_name = r'experiments/two_agent_sample_om_10runs_20260614_120321.json'
exp_name = r'experiments\two_agent_sampling_om_20260524_181553.json'

exp_name = r'experiments/two_agent_mean_om_10runs_20260613_190939.json'
exp_name = r'experiments/learn_prefs_20260706_222120.json'


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
logls = experiment_data['logl_eps']


"""two_agent_plotter(kls_eps,
                  kls_eps_other,
                  var_eps,
                  var_eps_other,
                  policy_eps_1,
                  policy_eps_2,
                  bac_eps_1,
                  bac_eps_2)"""


"""

unstable_EM_two_maze(kls_eps,
                  kls_eps_other,
                  var_eps,
                  var_eps_other,
                  policy_eps_1,
                  policy_eps_2,
                  bac_eps_1,
                  bac_eps_2,
)"""

two_agent_goal_learning_plotter(kls_eps,
                  kls_eps_other,
                  var_eps,
                  var_eps_other,
                  policy_eps_1,
                  policy_eps_2,
                  bac_eps_1,
                  bac_eps_2,
                  logls)

