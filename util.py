import numpy as np
import matplotlib.pyplot as plt
import random

# Used to generate the inverse covariance matrix
def generate_random_pd_matrix(p, sparsity, filename, step=0.1):
  S = np.zeros((p, p))
  dof = (p**2 - p) / 2
  N_nonzeros = int(sparsity * dof / 2)
  I_nonzeros = np.random.randint(0, p, size=(N_nonzeros, 2)) 
  for i in range(p):
    for j in range(i, p):
      if i == j:
        S[i, i] = random.random() + 0.001
      
  for i in range(N_nonzeros): 
    if I_nonzeros[i, 0] < I_nonzeros[i, 1]: 
      row, col = I_nonzeros[i, 0], I_nonzeros[i, 1]
      S[row, col] = random.random()
      S[col, row] = S[row, col] 
    elif I_nonzeros[i, 1] < I_nonzeros[i, 0]:
      col, row = I_nonzeros[i, 0], I_nonzeros[i, 1]
      S[row, col] = random.random()
      S[col, row] = S[row, col] 

  work = False
  n_step = 0
  while not work:
    w = np.linalg.eigvalsh(S)
    if np.min(w) <= 0: 
      S += step * np.eye(p)
      n_step += 1
    else:
      work = True
    
  if filename:
    np.save(filename, S)
  
  return S

# Sample based on a covariance matrix
def joint_gaussian_samples(n, p, S, mean=0.):
  scale = np.linalg.cholesky(S)
  return scale @ np.random.normal(size=(p, n)) + mean

def plot_convergence(timing_info, x_axis='real_time', filename=None):
  if x_axis == 'real_time':
    indices = [i for i in range(timing_info.shape[0]) if timing_info[i, 0] != 0]
    plt.plot(timing_info[indices, 0], timing_info[indices, 1])
    plt.xlabel('Time/s')
  elif x_axis == 'iteration':
    indices = [i for i in range(timing_info.shape[0]) if timing_info[i, 0] != 0]
    plt.plot(indices, timing_info[indices, 1])
    plt.xlabel('Number of iterations')
  
  plt.yscale('log')
  plt.ylabel('Duality gap')

  if filename:
    plt.savefig(filename)
  
def plot_value_distribution(A):
  n, m = A.shape
  A_list = sorted(A.flatten().tolist())
  plt.plot(np.arange(n * m), A_list)
  plt.xlabel('Sorted entry index')
  plt.ylabel('Entry values')
