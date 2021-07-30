import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
from helper import *
import sys
from pycute_interface import Blackbox

if len(sys.argv) < 7:
    print('python nagent.py --graph {1,2,3} --prob {1,2,3,4,5} --start {1,2,3}\n')
    print('{1,2,3}={chain,random,complete}, {1,2,3}={rosen1,rosen2,nest}, {1,2}={random,fixed,zero}')
    exit(0)

seed_num = 0
if len(sys.argv)>=9 and sys.argv[7]=='--seed':
    seed_num = int(sys.argv[8])

[graph_mode,prob_mode, x_0_mode] = [int(i) for i in sys.argv[2:7:2]]

######## SETUP #################################
n = 100
np.random.seed(seed_num)
c = 0.1

# Rosenbrock
if prob_mode == 1:
    m = n//2
    def df(x): return df_r(x)
    def f_eval(x): return f_r_long(x)
    L = 1
    xstar = np.ones(m*n, dtype=int)
    fstar = 0

# Alternative Rosenbrock
elif prob_mode == 2:
    m = n-1
    def df(x): return df_ar(x)
    def f_eval(x): return f_ar_long(x)
    L = 1
    xstar = np.ones(m*n, dtype=int)
    fstar = 0

# Nesterov
elif prob_mode == 3:
    m = n+1
    def df(x): return df_nesterov(x)
    def f_eval(x): return f_nesterov_long(x)
    L = 4
    xstar = nesterov_opt(n)
    fstar = f_nesterov(xstar)
    xstar = np.kron(np.ones(m), xstar)

# Regularized LS
elif prob_mode == 4:
    m = n
    d = 10
    n = d
    X = np.array([np.random.normal(loc=0, scale=1.0, size=d) for _ in range(m)]).T
    assert X.shape[0] == d and X.shape[1] == m
    y = np.dot(X.T, np.ones(d)) + np.cos(np.dot(X.T, np.ones(d))) + np.random.normal(loc=0, scale=0.25, size=m)
    y = np.dot(X.T, np.ones(d))

    def df(theta): return df_regls(theta, X, y, reg='l2')
    def f_eval(theta): return f_regls_long(theta, X, y, reg='l2')

    # eigenvalue approach
    # eig_1 = la.eig(np.dot(X,X.T))[0][0]
    # L = eig_1/m + 2*c

    # each element approach
    X_norms = la.norm(X, ord=2, axis=0)**2
    L = (2/m)*(np.amax(X_norms)+c)

    xstar = regls_opt(X,y, reg='l2')
    fstar = f_regls(xstar, X, y, reg='l2')
    xstar = np.kron(np.ones(m), xstar)

# Regularized Log
elif prob_mode == 5:
    m = n
    d = 10
    n = d
    y = np.append(2*np.ones(m//2), np.zeros(m-m//2))-1
    X = np.array([np.random.normal(loc=y[i]*np.ones(d), scale=1.0, size=d) for i in range(m)]).T

    def df(theta): return df_log(theta, X, y, reg='l2')
    def f_eval(theta): return f_log_long(theta, X, y, reg='l2')

    XXT_sum = np.outer(X[:,0], X[:,0])
    for i in range(1,m):
        XXT_sum += np.outer(X[:,i],X[:,i])
    eig_max = np.amax(la.eig(XXT_sum)[0].real)
    L = eig_max/m

    reg = 'l2'
    xstar = log_opt(X, y, reg)
    fstar = f_log(xstar, X, y, reg)
    xstar = np.kron(np.ones(m), xstar)

# CUTEr
elif prob_mode == 6:
    n = 100
    m = 20

    bbox = Blackbox()
    bbox.setup_new_prob(seed_num)
    bbox.set_scale()
    L = 1

    def df(theta): return bbox.df_long(theta)
    def f_eval(theta): return bbox.f_long(theta)/bbox.get_scale()

    [fstar, xstar] = optimize_blackbox(bbox)
    fstar = fstar/bbox.get_scale()
    xstar = np.kron(np.ones(m), xstar)

else:
    print('Invalid prob {}'.format(prob_mode))
    exit(0)

if x_0_mode == 1:
    np.random.seed(0)
    x = 2*np.random.random(m*n)-1
elif x_0_mode == 2:
    x = np.tile([-1.2,1],int(m*n//2))
elif x_0_mode == 3:
    x = np.zeros(m*n)
else:
    print('Invalid start {}'.format(x_0_mode))
    exit(0)

k = 1
if graph_mode==1:
    A = spp.diags(np.append(1, np.append(2*np.ones(m-2), 1))) - get_k_reach_chain_matrix(m,k)
elif graph_mode==2:
    p = 0.1 if prob_mode == 2 else 0.15
    A = get_er_graph(m, p, seed=0)
elif graph_mode==3: 
    k = m-1
    A = k*spp.eye(m) - get_k_reach_chain_matrix(m,k)
else:
    print('Invalid graph {}'.format(graph_mode))
    exit(0)

assert la.norm(A.dot(np.ones(A.shape[1]))) < 1e-15
W = get_doubly_stochastic(A)
rho = la.norm(W - n**-1*np.ones((m,m)), ord=2)
eta = 1.0/L * min(1/6, (1-rho**2)**2/(4*rho**2*(3+4*rho**2)))
W = spp.kron(W, spp.eye(n)).toarray()
Lap = spp.kron(A, spp.eye(n))

eps = 0.1
N_const = 50000
N = int(N_const / eps + 1)
# TEMP
N = min(N, 500000)
######## SETUP #################################

s = np.zeros(len(x), dtype=float)
g_prv = s.copy()

print('x={}\n'.format(x))
for k in range(1,N+1):
    g = df(x)
    s = np.dot(W, s + g - g_prv)
    x = np.dot(W, x-eta*s)
    g_prv = g

    gap = f_eval(x) - fstar
    print('Iter {}/{}'.format(k, N))
    print('gap={:.8e}'.format(gap))
    print('consensus={:.3e}'.format(np.dot(x, Lap.dot(x))))
    # print('x={}'.format(x))
    # print('g={}\n'.format(g))
    # print('!!score={:.4e}\n'.format(1/m*np.sum(np.log(1+np.exp(-y*np.dot(X.T,x[:4])))) + c*la.norm(x[:4],ord=2)))
    print('')

# print('\n!! x*={}'.format(regls_opt(X,y)))
# print('\n!! x*={}'.format(log_opt(X,y)))
# print('\n!! X^Tt={}, y={}'.format(np.dot(X.T,x[:4]), y))
