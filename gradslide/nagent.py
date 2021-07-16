import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
from helper import *

"""
Change
- Size of problem: m, n, (Line 18)
- Initial starting point x and xstar (Line 26)
- df() as well as gradient Lipschitz constant L (Line 44)
- fstar (optimal solution, if known) (Line 69)
- f() for gap computation (Line 45)
- k and communication graph (Line 50)
"""

########### PARAMETERS ##########################
# Size of problem
n = 100
m = n-1

# Lipschitz constant
L = 1

# Initialize
x = np.zeros(m*n, dtype=float)
np.random.seed(0)
x = 2*np.random.random(m*n)-1
xstar = nesterov_opt(n)
# x = np.append(np.ones(n), -np.ones(n))
# x = np.array([0,0,1,1])
# x = get_square_x0()
# y_curr = x.copy()
########### PARAMETERS ##########################

########### FUNCTIONS ##########################
d = n
# b = 2*np.random.random(m*n)-1 
ones_d = np.ones(d)

def V(x,y): return 0.5*la.norm(x-y,ord=2)**2

# CHANGE ME
def df(x): return df_ar(x)
def f_eval(x): return f_ar_long(x)
L = 1
########### FUNCTIONS ##########################
 
########### PARAMETERS (pt2) ####################
k = 1
if k == m-1: A = k*spp.eye(m) - get_k_reach_chain_matrix(m,k)
else: 
    assert k==1; 
    A = spp.diags(np.append(1, np.append(2*np.ones(m-2), 1))) - get_k_reach_chain_matrix(m,k)
assert la.norm(A.dot(np.ones(A.shape[1]))) < 1e-15
W = get_doubly_stochastic(A)
rho = la.norm(W - n**-1*np.ones((m,m)), ord=2)
eta = 1.0/L * min(1/6, (1-rho**2)**2/(4*rho**2*(3+4*rho**2)))
W = spp.kron(W, spp.eye(n)).toarray()
Lap = spp.kron(A, spp.eye(n))

lam_prev = 0
lam_curr = 1

eps = 0.1
N_const = 1500
N = int(N_const / eps + 1)

# CHANGE ME
fstar = 0
########### PARAMETERS ##########################


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
