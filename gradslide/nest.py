import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
from helper import *
import sys

if len(sys.argv) != 5:
    print('python nagent.py --prob {1,2,3} --start {1,2}\n')
    print('{1,2,3}={rosen1,rosen2,nest}, {1,2}={random,fixed}')
    exit(0)

[prob_mode, x_0_mode] = [int(i) for i in sys.argv[2:5:2]]

######## SETUP #################################
n = 100
# Rosenbrock
if prob_mode == 1:
    def df(x): return df_r_comb(x)
    def f_eval(x): return f_r(x)
    L = 1
    xstar = np.ones(n, dtype=int)
    fstar = 0

# Alternative Rosenbrock
elif prob_mode == 2:
    def df(x): return df_ar_comb(x)
    def f_eval(x): return f_ar(x)
    L = 1
    xstar = np.ones(n, dtype=int)
    fstar = 0

# Nesterov
elif prob_mode == 3:
    def df(x): return df_nesterov_comb(x)
    def f_eval(x): return f_nesterov(x)
    L = 4
    xstar = nesterov_opt(n)
    fstar = f_nesterov(xstar)
else:
    print('Invalid prob {}'.format(prob_mode))
    exit(0)

if x_0_mode == 1:
    np.random.seed(0)
    x = 2*np.random.random(n)-1
elif x_0_mode == 2:
    x = np.tile([-1.2,1],int(n//2))
else:
    print('Invalid start {}'.format(x_0_mode))
    exit(0)

beta = L
beta_inv = 1.0/beta
eps = 1e-6
N = int((2*L/eps * la.norm(xstar-x,ord=2)**2)**0.5 + 1) 
lam_prev = 0
lam_curr = 1
y_curr = x.copy()
########### PARAMETERS ##########################

print('---- STARTING STATS ----')
gap = f_eval(x) - fstar
print('x={}\n'.format(x[:n]))
print('gap={}\n'.format(gap))

print('---- BEGIN OPTIMIZATION ({} iters) ----'.format(N))
for k in range(1,N+1):
    gamma = (1-lam_prev)/lam_curr
    lam_prev = lam_curr
    lam_curr = (1 + (1+4*lam_prev)**0.5)/2

    y_next = x - beta_inv*df(x)
    x = (1-gamma)*y_next + gamma*y_curr
    y_curr = y_next

    gap = f_eval(y_curr) - fstar

    print('Iter {}/{}'.format(k, N))
    print('gap={}'.format(gap))
    # print('x={}'.format(x))
    print('')

# print(x[:n])
print('---- FINAL STATS ----')
gap = f_eval(x) - fstar
print('x={}\n'.format(x[:n]))
print('gap={}\n'.format(gap))
