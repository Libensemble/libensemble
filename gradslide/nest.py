import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
from helper import *

"""
Change
- Size of problem: m, n (Line 18)
- Initial starting point x and xstar (Line 24)
- Gradient Lipschitz constant L (Line 30)
- df() (Line 43)
- fstar (optimal solution, if known) (Line 45)
- f() for gap computation (Line 42)
"""

########### PARAMETERS ##########################
# Size of problem
n = 100

# Initialize
# x = get_square_x0()
np.random.seed(0)
x = 2*np.random.random(n)-1
y_curr = x.copy()
lam_prev = 0
lam_curr = 1
xstar = np.ones(len(x))

beta = L = 1

beta_inv = 1.0/beta
eps = 1e-6
N = int((2*L/eps * la.norm(xstar-x,ord=2)**2)**0.5 + 1) 
########### PARAMETERS ##########################

########### FUNCTIONS ##########################
# b = np.ones(n)
# b = 2*np.random.random(n)-1 
def f_eval(x): return f_ar(x)
def df(x): return df_ar_comb(x)
# fstar = f_nesterov(xstar)
fstar = 0
########### FUNCTIONS ##########################

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
