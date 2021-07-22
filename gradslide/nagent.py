import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
from helper import *
import sys

if len(sys.argv) != 7:
    print('python nagent.py --graph {1,2,3} --prob {1,2,3} --start {1,2}\n')
    print('{1,2,3}={chain,random,complete}, {1,2,3}={rosen1,rosen2,nest}, {1,2}={random,fixed}')
    exit(0)

[graph_mode,prob_mode, x_0_mode] = [int(i) for i in sys.argv[2:7:2]]

######## SETUP #################################
n = 100
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
else:
    print('Invalid prob {}'.format(prob_mode))
    exit(0)

if x_0_mode == 1:
    np.random.seed(0)
    x = 2*np.random.random(m*n)-1
elif x_0_mode == 2:
    x = np.tile([-1.2,1],int(m*n//2))
else:
    print('Invalid start {}'.format(x_0_mode))
    exit(0)

k = 1
if graph_mode==1:
    A = spp.diags(np.append(1, np.append(2*np.ones(m-2), 1))) - get_k_reach_chain_matrix(m,k)
elif graph_mode==2:
    p = 0.1
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
N_const = 1500
N = int(N_const / eps + 1)
######## SETUP #################################

s = np.zeros(len(x), dtype=float)
g_prv = s.copy()

print('x={}\n'.format(x))
for k in range(1,N+1):
    g = df(x)
    s = np.dot(W, s + g - g_prv)
    x = np.dot(W, x-eta*s)
    g_prv = g

    # CHANGE ME
    print('Iter {}/{}'.format(k, N))
    print('gap={}'.format(f_eval(x) - fstar))
    print('consensus={}'.format(np.dot(x, Lap.dot(x))))
    # print('x={}'.format(x))
    # print('g={}\n'.format(g))
    # print('!!score={:.4e}\n'.format(1/m*np.sum(np.log(1+np.exp(-y*np.dot(X.T,x[:4])))) + c*la.norm(x[:4],ord=2)))
    print('')

# print('\n!! x*={}'.format(regls_opt(X,y)))
# print('\n!! x*={}'.format(log_opt(X,y)))
# print('\n!! X^Tt={}, y={}'.format(np.dot(X.T,x[:4]), y))
