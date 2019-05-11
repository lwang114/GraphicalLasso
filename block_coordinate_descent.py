import numpy as np
import matplotlib.pyplot as plt
import math
import time
from copy import deepcopy
import random
from sklearn.linear_model import Lasso
from sklearn.covariance import graphical_lasso

DEBUG = False
# Block coordinate descent methods for covariance selection
def permute_matrix(dim, col):
  P = np.eye(dim)
  for i in range(dim):
    if i >= col and i < dim-1:
      P[i, i] = 0.
      P[i+1, i] = 1.
    elif i == dim-1:
      P[dim-1, dim-1] = 0
      P[col, dim-1] = 1.
  return P

def permute_col(A, col):
  dim = A.shape[0]
  I = np.eye(dim)
  for i in range(dim):
    if i >= col and i < dim-1:
      I[i, i] = 0.
      I[i+1, i] = 1.
    elif i == dim-1:
      I[dim-1, dim-1] = 0
      I[col, dim-1] = 1.
  if len(A.shape) >= 2:
    return A @ I
  else:
    return np.squeeze(A.reshape(1, -1) @ I)

def permute_row(A, row):
  return permute_col(A.T, row).T

def revert_permute(A, col):
  dim = A.shape[0]
  I = np.eye(dim)
  for i in range(dim):
    if i == col:
      I[i, i] = 0.
      I[dim-1, col] = 1.
    elif i > col:
      I[i, i] = 0
      I[i-1, i] = 1.
  if len(A.shape) >= 2:
    return A @ I
  else:
    return np.squeeze(A.reshape(1, -1) @ I)

def soft_threshold(x, rho):
  return np.sign(x) * (np.abs(x) - rho) * ((np.abs(x) - rho) > 0)

# Assume symmetric matrix
def block_inv(invA, block):
  n, m = block.shape
  assert (n == m)
  #assert (block == block.T).all()
  
  inv_block = np.zeros((n, n)) 
  k = block[-1, -1] - block[-1, :-1] @ invA @ block[:-1, -1]
  inv_block[:-1, :-1] = invA + invA @ block[:-1, -1].reshape(-1, 1) @ block[-1, :-1].reshape(1, -1) @ invA / k
  inv_block[:-1, -1] = - invA @ block[-1, :-1] / k
  inv_block[-1, :-1] = inv_block[:-1, -1]
  inv_block[-1, -1] = 1 / k
  
  #if DEBUG:
  #  print('inv * orig: ', inv_block @ block) 
  
  return inv_block

# Solve the box-constrained QP: 1/2 (x + b)'A(x + b) s.t. ||x||_infty <= rho 
def box_constrained_qp(A, b, rho, tol=0.001):
  p = A.shape[0]
  x = np.ones((p,))
  x_prev = deepcopy(x)
  diff = float('inf')
  t = 0
  while diff > tol:
    for i in range(p):
      P = permute_matrix(p, i)
      x_p =  P.T @ x
      A_p = P.T @ A @ P
      b_p = P.T @ b
      if DEBUG:
        print('A_p: ', A_p)
        print('b_p: ', b_p)
      x[i] = max(min(-b_p[-1] - np.dot(A_p[:-1, -1], x_p[:-1] + b_p[:-1]) / A_p[-1, -1], rho), -rho) 
    
    diff = np.sum(x - x_prev)
    x_prev = deepcopy(x)
  return x

#
#  Solve the LASSO problem parameterized by the quadratic form 1/2 x^T A x + b^T x + \rho \|x\|_1
# using coordinate descent 
#
def lasso(A, b, rho, tol=0.001):
  p = A.shape[0]
  x = np.ones((p,))
  x_prev = deepcopy(x)
  diff = float('inf')
  t = 0
  while diff > tol: 
    #for t in range(10):
    for i in range(p):
      x_p = permute_col(x, i)
      A_p = permute_col(permute_col(A, i).T, i)
      b_p = permute_col(b, i)
      r = None
      if A_p.shape[0] == 1:
        r = A_p * x_p
        x[i] = soft_threshold(-b_p - r, rho) / A_p
      else: 
        r = np.dot(A_p[:-1, -1], x_p[:-1])
        x[i] = soft_threshold(-b_p[-1] - r, rho) / A_p[-1, -1]

    diff = np.sum((x - x_prev)**2)
    t += 1
    #print('Lasso iteration %d, residue %0.5f' % (t, diff))
    
    x_prev = deepcopy(x)  
  return x

def dglasso(S_sample, rho=0.1, eps=0.001, T_max=100):
  p, p_ = S_sample.shape
  assert p == p_ 
  assert np.linalg.det(S_sample) > 0
  assert np.trace(S_sample) > 0
  
  # Initialize the estimated covariance
  S_est = S_sample + rho * np.eye(p)

  # Repeatedly perform LASSO step until convergence
  t = 0
  
  #while abs(duality_gap) > eps:
  timing_info = np.zeros((T_max * p, 3))
  for t in range(T_max):
    C = np.zeros((p, p))
    iter_time = 0.
    for i in range(p):
      begin_time = time.time()
      S_permuted = permute_col(permute_col(S_est, i).T, i)
      S_sample_permuted = permute_col(permute_col(S_sample, i).T, i)
      W_ii, s_i = S_permuted[:-1, :-1], S_sample_permuted[-1, :-1]
      # Solve the problem 1/2 x^T S_ii x - s_i^T x + \rho \|x\|_1
      beta = lasso(W_ii, -s_i, rho, tol=1e-6)
      s_i_est = W_ii @ beta
      
      # TODO: work on the permutation matrix instead?
      S_permuted[:-1, -1] = s_i_est
      S_permuted[-1, :-1] = s_i_est
      S_est = revert_permute(revert_permute(S_permuted, i).T, i)
      iter_time += time.time() - begin_time

      C = permute_row(permute_col(C, i), i)
      
      C[-1, -1] = 1 / (S_permuted[-1, -1] + np.dot(beta, s_i_est))
      C[:-1, -1] = -beta * C[-1, -1]
      C[-1, :-1] = C[:-1, -1]
      C = revert_permute(revert_permute(C, i).T, i)
    
      C = np.linalg.inv(S_est)
      cost = math.log(np.linalg.det(C)) - np.trace(S_sample @ C) - rho * (np.sum(np.abs(C)))
      duality_gap = np.trace(C @ S_sample) + rho * (np.sum(np.abs(C))) - p 

      timing_info[t*p+i, 0] += timing_info[t*p+i-1, 0] + iter_time
      timing_info[t*p+i, 2] = cost  
      timing_info[t*p+i, 1] = duality_gap
          
    # TODO: Avoid recomputing the inverse; use other stopping criteria
    #C = np.linalg.inv(S_est)
    if DEBUG:
      print(np.trace(C * S_sample), rho * np.sum(np.abs(C)), p) 
     
    C = np.linalg.inv(S_est)
    cost = math.log(np.linalg.det(C)) - np.trace(S_sample @ C) - rho * (np.sum(np.abs(C)))
    duality_gap = np.trace(C @ S_sample) + rho * (np.sum(np.abs(C))) - p 
    #- p + np.trace(C * S_sample) + rho * np.sum(np.abs(C))
    t += 1
    print('Glasso iteraction %d, duality gap %0.5f: ' % (t, duality_gap)) 
    if duality_gap < eps:
      break

  return S_est, C, timing_info

def pglasso(S_sample, rho=0.1, eps=1e-4, T_max=100):
  p, p_ = S_sample.shape
  assert p == p_ and np.linalg.det(S_sample) > 0 and np.trace(S_sample) > 0
  
  begin_time = time.time()
  W = S_sample + rho * np.eye(p)
  invS = np.linalg.inv(W)
  init_time = time.time() - begin_time
  
  timing_info = np.zeros((T_max * p, 3))
  timing_info[0, 0] = init_time
  for t in range(T_max):
    for i in range(p):
      begin_time = time.time()
      # Permute the matrices
      P = permute_matrix(p, i)
      #if DEBUG:
      #  print('S_sample:\n', S_sample)
      #  print('S_permuted:\n', S_permuted)
      S_permuted = P.T @ S_sample @ P
      W_permuted = P.T @ W @ P
      invS_permuted = P.T @ invS @ P
      # Compute the inverse of invS_11 using schur complement
      s_i = S_permuted[:-1, -1]
      #if DEBUG:
      #  print('s_i: ', s_i)
      w_i = W_permuted[:-1, -1]
      invinvS_11 = W_permuted[:-1, :-1] - w_i.reshape(-1, 1) @ w_i.reshape(1, -1) / W_permuted[-1, -1]
      alpha = lasso(invinvS_11, s_i, rho=rho, tol=1e-6)
      
      invS_permuted[:-1, -1] = alpha / W_permuted[-1, -1] 
      invS_permuted[-1, :-1] = alpha / W_permuted[-1, -1]
      invS_permuted[-1, -1] = 1 / W_permuted[-1, -1] + invS_permuted[:-1, -1].reshape(1, -1) @ invinvS_11 @ invS_permuted[-1, :-1].reshape(-1, 1)
      
      #if DEBUG:
      #  print(invinvS_11 == invinvS_11.T, type(invS_permuted[0, 3]), type(invS_permuted[3, 0]), float(invS_permuted[0, 3]) == float(invS_permuted[3, 0])) 
      W_permuted = block_inv(invinvS_11, invS_permuted)
      
      W = P @ W_permuted @ P.T
      invS = P @ invS_permuted @ P.T
      
      for j in range(p):
        W[j, j] = S_sample[j, j] + rho  
      
      iter_time = time.time() - begin_time

      C = np.linalg.inv(W)
      cost = math.log(np.linalg.det(C)) - np.trace(S_sample @ C) - rho * (np.sum(np.abs(C)))
      duality_gap = np.trace(C @ S_sample) + rho * (np.sum(np.abs(C))) - p 
      
      timing_info[t*p+i, 0] += timing_info[t*p+i-1, 0] + iter_time
      timing_info[t*p+i, 2] = cost  
      timing_info[t*p+i, 1] = duality_gap
       
      #if DEBUG and i == 2:
      #print('invS:\n', invS)
      #print('invinvS_11 @ S_11:\n', invinvS_11 @ invS_permuted[:-1, :-1])
      #print(invS_permuted[-1, -1])
      #print('w_12_prev: ', -W_permuted[:-1, :-1] @ alpha)
      #print('w_12: ', W_permuted[:-1, -1])
    cost = math.log(np.linalg.det(invS)) - np.trace(S_sample @ invS) - rho * (np.sum(np.abs(invS)))
    duality_gap = np.trace(invS @ S_sample) + rho * (np.sum(np.abs(invS))) - p 
    #duality_gap = np.trace(S_inv @ S_sample) + rho * (np.sum(np.abs(S_inv))) - p 

    print('P-GLASSO iteration %d, duality gap %.5f' % (t, duality_gap))
    
    if duality_gap < eps:
      break

  return W, invS, timing_info

def dpglasso(S_sample, rho=0.1, eps=1e-4, T_max=100):
  p, p_ = S_sample.shape
  begin_time = time.time()
  invS = np.diag(np.diag(np.linalg.inv(S_sample + rho * np.eye(p))))
  S_est = S_sample + rho * np.eye(p) 
  init_time = time.time() - begin_time
  
  timing_info = np.zeros((T_max * p, 3))
  timing_info[0, 0] = init_time
  
  for t in range(T_max):
    for i in range(p):
      begin_time = time.time()
      
      P = permute_matrix(p, i)
      invS_permuted = P.T @ invS @ P
      S_permuted = P.T @ S_est @ P

      if DEBUG:
        print('invS:\n', invS)
        print('invS_permuted:\n', invS_permuted)
 
      S_sample_permuted = P.T @ S_sample @ P
      invS_11 = invS_permuted[:-1, :-1] 
      s_j = S_sample_permuted[:-1, -1]
      gamma = box_constrained_qp(invS_11, s_j, rho=rho)
      if DEBUG:
        print('gamma:\n', invS_11, s_j, gamma)

      invS_permuted[-1, :-1] = - invS_11 @ (gamma + s_j) / S_permuted[-1, -1]
      invS_permuted[:-1, -1] = invS_permuted[-1, :-1]
      invS_permuted[-1, -1] = (1 - np.dot(gamma + s_j, invS_permuted[-1, :-1])) / S_permuted[-1, -1] 
      S_permuted[:-1, -1] = s_j + gamma
      S_permuted[-1, :-1] = S_permuted[:-1, -1]

      S_est = P @ S_permuted @ P.T
      invS = P @ invS_permuted @ P.T

      for j in range(p):
        S_est[j, j] = S_sample[j, j] + rho  

      iter_time = time.time() - begin_time      
      
      cost = math.log(np.linalg.det(invS)) - np.trace(S_sample @ invS) - rho * (np.sum(np.abs(invS)))
      duality_gap = np.trace(invS @ S_sample) + rho * (np.sum(np.abs(invS))) - p 
      #duality_gap = np.trace(S_inv @ S_sample) + rho * (np.sum(np.abs(S_inv))) - p 

      timing_info[t*p+i, 0] += timing_info[t*p+i-1, 0] + iter_time
      timing_info[t*p+i, 2] = cost  
      timing_info[t*p+i, 1] = duality_gap
     
    print('DP-GLASSO iteration %d, duality gap %.5f' % (t, duality_gap))
    if duality_gap < eps:
      break
  return S_est, invS, timing_info 
      

def lasso_and_or(S, op='and', rho=0.1, threshold=1e-6):
  assert S.shape[0] == S.shape[1] and (S.T == S).all() and (np.diag(S) != 0).all()
  begin_time = time.time()
  p = S.shape[0]
  #S = normalize(S)
  S_sparse = np.zeros((p, p))

  # Iterate through each column of the sample covariance
  for i in range(p):
    P = permute_matrix(p, i)
    S_permuted = P.T @ S @ P 
    S_ii = S_permuted[:-1, :-1]
    s_i = S_permuted[:-1, -1]
    #print('min eigen value for the block matrix: ', np.min(np.linalg.eigvalsh(S_ii)))
    s_i_sparse = S_ii @ lasso(S_ii, -s_i, rho=rho)
    S_sparse = permute_row(permute_col(S_sparse, i), i)
    S_sparse[-1, -1] = S[i, i]
    S_sparse[:-1, -1] = s_i_sparse 
    S_sparse = P @ S_sparse @ P.T
    print('Iteration %d' % i)

  #invS_sparse = np.linalg.inv(S_sparse)
  for i in range(p):
    for j in range(i+1, p):
      if op == 'and':
        if S_sparse[i, j] <= threshold and S_sparse[j, i] <= threshold:
          S_sparse[i, j] = 0.
      elif op == 'or':
        if S_sparse[i, j] <= threshold or S_sparse[j, i] <= threshold:
          S_sparse[i, j] = 0.
  timing_infos = np.zeros((1, 3))
  timing_infos[0, 0] = time.time() - begin_time 
  print('Lasso %s takes %0.5f to finish' % (op, timing_infos[0, 0]))
  return S_sparse, timing_infos

def plot_matrix(X, filename=None):
  fig, ax = plt.subplots()
  plt.imshow(X, cmap=plt.get_cmap('gray'))
  plt.colorbar()
  #plt.show()
  if filename:  
    plt.savefig(filename)
  plt.close()

if __name__ == '__main__':
  n = 1000
  p = 100
  rho = 0.3
  thres = 0.001
  #print(soft_threshold(1.2, 0.5))
  #print(soft_threshold(1.2, 1.3))
  #print(soft_threshold(-1.2, 0.5))

  S_sample = np.load('exp/may10/synthetic_covariance.npy') #np.array([[3., 2., 1.], [2., 3., 0.], [1., 0., 3.]])
  A = S_sample[:-1, :-1] 
  b = S_sample[:-1, -1] #np.array([1, -1, 1])
  #sklearn_lasso = Lasso(alpha=0.1)
  #sklearn_lasso.fit(A, b)
  print('max A', np.max(A))  
  print('Dense solution: ', np.linalg.inv(A) @ (-b))
  print('Solution: ', lasso(A, b, rho=0.01))
  #X = np.random.normal(size=(n, p))
  #X_shift = np.zeros((n, p))
  #X_shift[:, -1] = X[:, 0]
  #X_shift[:, :-1] = X[:, 1:] 
  #S_sample = 1/n * (X + X_shift).T @ (X + X_shift)
  '''S_sample = np.zeros((p, p))
  for i in range(p):
    for j in range(i, p):
      if i == j:
        S_sample[i, i] = 1. #+ 0.1 * random.random()
      elif j - i == 1:
        S_sample[i, j] = 0.25 #+ 0.1 * random.random()
        S_sample[j, i] = S_sample[i, j]
      else:
        S_sample[i, j] = 0. #0.1 * random.random()
        S_sample[j, i] = S_sample[i, j]
  '''
  plot_matrix(S_sample, 'sample_estimate.png')
  print('Sample covariance:\n', S_sample)
  
  S_est, _ = lasso_and_or(S_sample, op='and', rho=rho, threshold=thres)
  print('LASSO-AND:\n', S_est)
  plot_matrix(S_est, 'lasso_and.png')
  
  S_est,_ = lasso_and_or(S_sample, op='or', rho=rho, threshold=thres)
  print('LASSO-OR:\n', S_est)
  plot_matrix(S_est, 'lasso_or.png')
  
  S_est, invS_est, timing_info = dglasso(S_sample, rho=rho)
  print('My implementation of D-GLASSO:\n', S_est)
  print('Total time: %.5f s' % max(timing_info[:, 0]))
  plot_matrix(S_est, 'dglasso.png')
  #plot_convergence(timing_info, 'dglasso_convergence.png')
  
  #S_sample = A #np.array([[2., 0.2], [0.2, 0.5]])
  S_est, invS_est, timing_info = pglasso(S_sample, rho=rho, T_max=10)
  print('P-GLASSO:\n', S_est)
  print(invS_est)
  print(S_est @ invS_est)
  print('Total time: %.5f s' % max(timing_info[:, 0]))
  plot_matrix(S_est, 'pglasso.png')
  
  S_est, invS_est, timing_info = dpglasso(S_sample, rho=rho, T_max=20)
  print('DP-GLASSO:\n', S_est)
  print(invS_est)
  print('Total time: %.5f s' % max(timing_info[:, 0]))
  plot_matrix(S_est, 'dpglasso.png')
  '''  
  begin_time = time.time()
  S_est, invS_est = graphical_lasso(S_sample, alpha=rho)
  tot_time = time.time() - begin_time
  print('Sklearn implementation of GLASSO:\n', S_est)
  print(invS_est)
  print('Total time: %.5f s' % tot_time)'''
