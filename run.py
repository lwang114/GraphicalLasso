from block_coordinate_descent import * 
from util import *
import json
import random

stage = 1
# Generate synthetic data according to Mazumder et. al. 2012
p = 10
n = 200
exp_dir = 'exp/may10_p=%d/' % p
syn_sample_file = 'synthetic_covariance.npy'
syn_prefix = exp_dir + 'syn_'
result_file = 'results.json' 
rho = 0.3
thres = 0.6
step = 0.1
T_max = 10
sparsity = 0.5

invS = np.zeros((p, p))  
if stage < 1:
  invS = generate_random_pd_matrix(p, filename = exp_dir + syn_sample_file, sparsity=sparsity)
  S = np.linalg.inv(invS)
  S /= (np.diag(S).reshape(-1, 1) @ np.diag(S).reshape(1, -1))**.5 
  plot_matrix(np.linalg.inv(S), exp_dir + 'real_inv_cov.png')

  print('max S_sample', np.max(np.diag(S)))
   
  X = joint_gaussian_samples(n, p, S)
  S_sample = 1 / n * X @ X.T

  print('Finish generating a sparse, positive definite concentration matrix and sample according to it')
  
if stage < 2:
  if syn_sample_file:
    S_sample = np.load(exp_dir + syn_sample_file)
  plot_matrix(np.linalg.inv(S_sample), exp_dir + 'syn_inv_cov_sample_estimate.png')
  
  S_ests = []
  invS_ests = []
  model_names = []
  timing_infos = []

  '''S_est, timing_info = lasso_and_or(S_sample, op='and', rho=rho, threshold=thres)
  plot_matrix(S_est, syn_prefix + 'lasso_and.png')
  print('Finish Lasso-and')

  S_est, timing_info = lasso_and_or(S_sample, op='or', rho=rho, threshold=thres)
  plot_matrix(S_est, syn_prefix + 'lasso_or.png')
  print('Finish Lasso-or')
  '''
  S_est, invS_est, timing_info = dglasso(S_sample, rho=rho, T_max=T_max)
  plot_matrix(invS_est, syn_prefix + 'dglasso.png')
  S_ests.append(S_est)
  invS_ests.append(invS_est)
  timing_infos.append(timing_info)
  model_names.append('D-GLasso')
  print('Finish D-GLasso')

  S_est, invS_est, timing_info = pglasso(S_sample, rho=rho, T_max=T_max)
  plot_matrix(invS_est, syn_prefix + 'pglasso.png')
  S_ests.append(S_est)
  invS_ests.append(invS_est)
  timing_infos.append(timing_info)
  model_names.append('P-Glasso')
  print('Finish P-GLasso')

  S_est, invS_est, timing_info = dpglasso(S_sample, rho=rho, T_max=T_max)
  plot_matrix(invS_est, syn_prefix + 'dpglasso.png')
  S_ests.append(S_est)
  invS_ests.append(invS_est)
  timing_infos.append(timing_info)
  model_names.append('DP-GLasso')
  print('Finish DP-GLasso')  

  begin_time = time.time()
  S_est, invS_est = graphical_lasso(S_sample, alpha=rho)
  tot_time = time.time() - begin_time
  #S_ests.append(S_est)
  #invS_ests.append(invS_est)
  #timing_infos.append(timing_info)
  #model_names.append('Sklearn Glasso')
  res_dict = None
  if result_file:
    res_dict = {model_name: [timing_info.tolist(), S_est.tolist(), invS_est.tolist()] for model_name, timing_info, S_est, invS_est in zip(model_names, timing_infos, S_ests, invS_ests)}
    with open(syn_prefix + result_file, 'w') as f:
      json.dump(res_dict, f, indent=4, sort_keys=True)

if stage < 3:
  res_dict = None
  if syn_sample_file:
    S_sample = np.load(exp_dir + syn_sample_file)

  if result_file:
    with open(syn_prefix + result_file, 'r') as f:
      res_dict = json.load(f)

  # 1) For each algorithm, plot the convergence curve (primal dual value vs. time +
  # primal dual value vs. number of sweeps)
  model_names = []
  for model_name, model_info in res_dict.items():
    plot_convergence(np.array(model_info[0]), x_axis='iteration')
    model_names.append(model_name)
  #plt.show()
  plt.legend(model_names, loc='best')
 
  plt.savefig(syn_prefix + 'convergence_comp.png')
  plt.close()

  print('Finish convergence plots')

  # 2) For each algorithm, plot the value histogram and compare it with the raw one 
  #plot_value_distribution(S_sample)
  for model_name, model_info in res_dict.items():
    plot_value_distribution(np.array(model_info[2]))
  plt.legend(model_names, loc='best')
  
  plt.savefig(syn_prefix + 'value_distribution_comp.png')
  plt.close()
  
  print('Finish plotting value distribution')
  # 3) For each algorithm, Binary classification comparison

if stage < 4:
  if syn_sample_file:
    S_sample = np.load(exp_dir + syn_sample_file)
  
  if result_file:
    with open(syn_prefix + result_file, 'r') as f:
      res_dict = json.load(f)
   
  # 4) Choose one algorithm, Plot sparsity as a function of thresholds
  s_max = np.max(S_sample - np.diag(np.diag(S_sample)))
  thres = [0.9 * s_max * (0.8 ** t) for t in range(5)]
  
  invS_ests = [dglasso(S_sample, rho=th, T_max=T_max)[1] for th in thres] 
  print('Finish estimating invS for various threshold')
  for invS_est in invS_ests:
    plot_value_distribution(invS_est)
  plt.legend(thres)
  
  plt.savefig(syn_prefix + 'dglasso_value_distribution_vs_thres.png')
  plt.close()
  print('Finish plotting value distribution as a function of penalty')
   
  for i, invS_est in enumerate(invS_ests):
    plot_matrix(invS_est, syn_prefix + 'dglasso_threshold=%0.5f.png' % (thres[i]))
  print('Finish threshold vs sparsity')
  plt.close()
  # 5) Choose two algorithm, Plot the change of minimum eigenvalue over time

  # 6) Choose one algorithm, Plot the change of reconstruction as number of samples increase
