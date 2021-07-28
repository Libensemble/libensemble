import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
from helper import *
import sys

if len(sys.argv) != 5:
    print('python nagent.py --prob {1,2,3,4} --start {1,2,3}\n')
    print('{1,2,3,4}={rosen1,rosen2,nest,linreg}, {1,2,3}={random,fixed,zero}')
    exit(0)

[prob_mode, x_0_mode] = [int(i) for i in sys.argv[2:5:2]]

######## SETUP #################################
n = 100
np.random.seed(0)

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

elif prob_mode == 4:
    reg = 'l2'
    c = 0.1

    d = 10
    m = n
    n = d
    X = np.array([np.random.normal(loc=0, scale=1.0, size=d) for _ in range(m)]).T
    y = np.dot(X.T, np.ones(d)) + np.cos(np.dot(X.T, np.ones(d))) + np.random.normal(loc=0, scale=0.25, size=m)

    def df(theta): return df_regls_comb(theta, X, y, reg)
    def f_eval(theta): return f_regls(theta, X, y, reg)

    max_eig = la.eig(np.dot(X, X.T))[0][0]
    L = (2/m)*max_eig + c

    xstar = regls_opt(X, y)
    fstar = f_regls(xstar, X, y, reg)
    fstar = 0

else:
    print('Invalid prob {}'.format(prob_mode))
    exit(0)

if x_0_mode == 1:
    np.random.seed(0)
    x = 2*np.random.random(n)-1
elif x_0_mode == 2:
    x = np.tile([-1.2,1],int(n//2))
elif x_0_mode == 3:
    x = np.zeros(n)
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
    print('||grad_f(x)||={:.4f}'.format(la.norm(df(x), ord=2)))
    print('')

# print(x[:n])
print('---- FINAL STATS ----')
gap = f_eval(x) - fstar
print('x={}\n'.format(x[:n]))
print('gap={}\n'.format(gap))
