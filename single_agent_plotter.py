import json
from single_agent_plotting_utils import plotter
import numpy as np


true_A = np.zeros((5,2,4))
true_A[0,0,1] = 1
true_A[1,1,1] = 1
true_A[2,0,2] = 1
true_A[2,1,3] = 1
true_A[3,1,2] = 1
true_A[3,0,3] = 1
true_A[4,:,0] = 1

true_A = true_A/np.sum(true_A, axis = 0, keepdims = True)



exp_name = r'experiments/my_EM_resdep2_20260618_222424.json'
exp_name = r'experiments/my_EM_resdep3_20260618_222958.json'
exp_name = r'experiments/my_EM_resdep3-5acts_20260618_223921.json'
exp_name = r'experiments/my_EM_resdep3-5acts_nov_20260618_224641.json'

exp_name = r'experiments/Baseline_EM_20260618_221509.json'
exp_name = r'experiments/Dirichlet Acc_20260618_215233.json'


with open(exp_name, "r") as f:
    data = json.load(f)

experiment_data = data["results"]


A_matrices = experiment_data["A_matrices"]
kls = experiment_data["kls"]
vars = experiment_data["vars"]
bacs = experiment_data["bacs"]
final_om_ss = experiment_data['final_om_ss']

plotter(kls, vars, bacs, true_A, np.mean(np.array(final_om_ss), axis = 0))

