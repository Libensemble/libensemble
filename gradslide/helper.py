import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
# from sklearn.linear_model import LogisticRegression

"""
Function definitions
"""

# GEOMETRIC MEDIAN
def f_gm(x,b,m):
    """ Geometric median """
    assert len(x)==len(b)
    n = len(x)//m
    z = x-b
    Z = np.reshape(z, newshape=(m,n))
    norms_z = la.norm(Z, axis=1, ord=2)
    return 1.0/m * np.sum(norms_z)

def df_gm(x,b,m):
    """ Geometric median """
    n = len(x)//m
    z = x-b
    Z = np.reshape(z, newshape=(m,n))
    norms_z = la.norm(Z, axis=1, ord=2)
    norms_z[norms_z == 0] = np.inf
    assert len(norms_z) == m
    norms_Z = np.tile(norms_z,(n,1)).T
    Z = np.divide(Z, norms_Z)
    return 1.0/m * np.reshape(Z, newshape=(-1,))

# ROSENBROCK
const = 1000
def f_r(x):
    """ Rosenbrock """
    assert len(x) % 2 == 0, 'must be even lengthed input vector'
    f1 = 1 * np.power( np.power(x[::2],2) - x[1::2], 2)
    f2 = np.power(x[::2]-np.ones(len(x)//2), 2)
    # return 1/const*np.sum(f1 + f2)
    return np.sum(f1 + f2)

def f_r_long(x):
    """ Rosenbrock eval with concatenated  x's, i.e., [x_1, x_2, ..., x_m] """
    N = len(x)
    m = int((N//2)**0.5)
    n = m*2
    assert 2*m**2 == N
    idx = (n+2)*np.arange(m, dtype=int)
    idx = np.repeat(idx,2)
    idx[1::2] += 1
    x_cut = x[idx]
    return f_r(x_cut)

def df_r(x):
    """ Rosenbrock 

    We assume each agent has ahold of one f_i.
    """
    n = len(x)
    m = int((n//2)**0.5)
    assert 2*m**2 == n
    idxs = 2*(m+1)*np.arange(m, dtype=int)
    df = np.zeros(len(x), dtype=float)
    df[idxs] = 400 * np.multiply(x[idxs], np.power(x[idxs], 2) - x[idxs+1]) + 2 * ( x[idxs] - 1)
    df[idxs+1] = -200 * (np.power(x[idxs], 2) - x[idxs+1])
    return (1/const)*df

def df_r_comb(x):
    """ Derivative Rosenbrock (no separating each f_i) """
    assert len(x)%2==0
    df = np.zeros(len(x), dtype=float)
    df[::2]  = 400*np.multiply(x[::2], np.power(x[::2],2)-x[1::2]) + 2*(x[::2]-1)
    df[1::2] = -200*(np.power(x[::2],2) - x[1::2])
    return (1/const)*df

# ALTERNATIVE ROSENBROCK
const = 1500
def f_ar(x):
    """ Rosenbrock """
    return 100*np.sum(np.power(np.power(x[:-1],2)-x[1:], 2)) + np.sum(np.power(x[:-1]-1,2))

def f_ar_long(x):
    """ Rosenbrock """
    N = len(x)
    n = int((1+(1+4*N)**0.5)/2)
    m = n-1
    assert n*m==N
    idx = (n+1)*np.arange(m, dtype=int)
    f = 100*np.sum(np.power(x[idx]-x[idx+1], 2))
    f += 0.5*np.sum(np.power(x[idx]-1, 2))
    return f

def df_ar(x):
    """ Rosenbrock 

    We assume each agent has ahold of one f_i.
    """
    N = len(x)
    n = int((1+(1+4*N)**0.5)/2)
    assert n*(n-1)==N
    m = n-1
    idxs = (n+1)*np.arange(m, dtype=int)
    df = np.zeros(N, dtype=float)
    df[idxs] = 400 * np.multiply(x[idxs], np.power(x[idxs],2)-x[idxs+1]) + 2 * (x[idxs]-1)
    df[idxs+1] = -200 * (np.power(x[idxs],2) - x[idxs+1])
    return 1/const*df

def df_ar_comb(x):
    """ Derivative Rosenbrock (no separating each f_i) """
    df = np.zeros(len(x), dtype=float)
    df[:-1] = 400*np.multiply(x[:-1], np.power(x[:-1],2)-x[1:]) + 2*(x[:-1]-1)
    df[1:] += -200*(np.power(x[:-1],2) - x[1:])
    return 1/const*df

# REGULARIZED LEAST SQUARES
c = 0.1
def regls_opt(X, y):
    A = np.dot(X,X.T)
    m = A.shape[0]
    x = la.lstsq(A+m*c*np.eye(m), np.dot(X,y))[0]
    ones_d = np.ones(X.shape[0])
    # print('{} vs {}'.format(1/m*la.norm(y-X.T@x,ord=2) + c*la.norm(x,ord=2), 1/m*la.norm(y-X.T@ones_d,ord=2) + c*la.norm(ones_d,ord=2)))
    return x

def df_regls(theta, X, y):
    d,m = X.shape
    ones_d = np.ones(d)

    # broadcast multiply
    X_arr = np.reshape(X, newshape=(-1,))
    df = -np.multiply( X_arr, np.kron(y, ones_d) )  
    Theta = np.reshape(theta, newshape=(m,d))  # unfold
    temp  = np.einsum("ij,ij->i", X.T, Theta)  # size m array
    # broadcast multiply
    df += np.multiply( X_arr, np.kron( temp, ones_d ) )
    df += c/m * theta

    df *= 2/m
    return df

# SIMPLE
def df_silly(x_in):
    """ Minimize Solves |x_1| + |x_2-1| """
    assert len(x_in) == 4
    df = np.zeros(4)
    df[0] = 2*x_in[0]
    df[3] = 2*(x_in[3]-1)
    return df

# LOGISTIC REGRESSION
def df_log(theta, X, y):
    d,m = X.shape
    XTTheta = np.einsum('ij,ij->i', X.T, np.reshape(theta, newshape=X.T.shape))
    df_scalar = np.divide(-y, 1 + np.exp(-np.multiply(y, XTTheta)) )
    df = np.multiply( np.reshape(X.T, newshape=(-1,)),  np.kron(df_scalar, np.ones(d)) )
    return 1/m * df + 2*c/m*theta

"""
def log_opt(X, y, reg='l2', reg_strength=0):
    if reg_strength == 0: reg_strength = c
    solver = 'newton-cg' if reg=='l2' or reg_strength == np.inf else 'liblinear' 
    clf = LogisticRegression(penalty=reg, 
                             C=1.0/reg_strength,  # regularizer strength (inverse)
                             random_state=0, 
                             solver=solver).fit(X, y)
    return clf.predict(X)
"""

# NONCONVEX
def df_noncvx(x,a,b,nu,Xi):
    m = len(b)
    d = len(x)//m

    base = np.exp(-np.dot(Xi.T,x)-nu)
    df_scalar = np.divide(a, 1+base)
    df = np.multiply( np.reshape(Xi, newshape=(-1,)), np.kron(df_scalar, np.ones(d)) )

    X = np.reshape(x, newshape=(m,d))
    df2_scalar = np.divide(2*b, 1+np.power(la.norm(X, ord=2, axis=1), 2))
    df = df + np.kron(x, np,kron(df2_scalar, np.ones(d)) )

    return 1/d*df

# NESTEROV'S QUADRATIC FUNCTION
def f_nesterov(x):
    assert len(x) >=1
    return 0.5*(x[0]**2 + x[-1]**2) - x[0] + 0.5*np.sum(np.power(x[1:]-x[:-1], 2))

def f_nesterov_long(x):
    assert len(x) >=1
    N = len(x)
    n = int((-1 + (1+4*N)**0.5)/2)
    assert n*(n+1)==N, 'Size must be N=n(n+1) for some n>=1'
    m = n+1

    x_1 = x[0]
    x_n = x[-1]
    f = 0.5*(x_1**2 + x_n**2) - x_1

    idx = (n+1)*np.arange(1,n, dtype=int)
    f += 0.5*np.sum(np.power(x[idx]-x[idx-1], 2))

    return f

def df_nesterov(x):
    """ Nesterov's quadratic function. 

    We assume each agent has one of the m=n+1 f_i's, where n=len(x) (before copies)
    """
    N = len(x)
    n = int((-1 + (1+4*N)**0.5)/2)
    assert n*(n+1)==N, 'Size must be N=n(n+1) for some n>=1'
    m = n+1

    idxs = (n+1)*np.arange(n, dtype=int)
    df = np.zeros(len(x), dtype=float)
    df[0] = x[0]-1
    df[-1] = x[-1]

    df[idxs[1:m-1]] = x[idxs[1:m-1]] - x[idxs[1:m-1]-1]
    df[idxs[1:m-1]-1] = -df[idxs[1:m-1]]
    return df

def df_nesterov_comb(x):
    df = np.zeros(len(x), dtype=float)
    df[:] = 2*x[:]
    df[:-1] -= x[1:]
    df[1:] -= x[:-1]
    df[0] -= 1
    return df

def nesterov_opt(n):
    return np.ones(n) - 1/(n+1) * np.arange(1,n+1)

def get_k_reach_chain_matrix(n, k):
    """ Constructs adjacency matrix for a chain matrix where the ith vertex can
        reach vertices that are at most @k distances from them (does not wrap around),
        where the distance is based on the absoluate difference between vertices'
        indexes.
    """
    assert 1 <= k <= n-1

    half_of_diagonals = [np.ones(n-k+j) for j in range(k)]
    half_of_indices = np.arange(1,k+1)
    all_of_diagonals = half_of_diagonals + half_of_diagonals[::-1]
    all_of_indices = np.append(-half_of_indices[::-1], half_of_indices)
    A = spp.csr_matrix( spp.diags(all_of_diagonals, all_of_indices) )
    return A

def get_doubly_stochastic(A):
    """ Generates a doubly stochastic matrix where
    (i) S_ii > 0 for all i
    (ii) S_ij > 0 if and only if (i,j) \in E

    Parameter
    ---------
    A : np.ndarray
        - adjacency matrix

    Returns
    -------
    x : scipy.sparse.csr_matrix
    """
    np.random.seed(0)
    n = A.shape[0]
    x = np.multiply( A.toarray() != 0,  np.random.random((n,n)))
    x = x + np.diag(np.random.random(n) + 1e-4)

    rsum = np.zeros(n)
    csum = np.zeros(n)
    tol=1e-15

    while (np.any(np.abs(rsum - 1) > tol)) | (np.any(np.abs(csum - 1) > tol)):
        x = x / x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)

    return spp.csr_matrix(x)

def get_square_x0():
    n = 2
    m = 12
    x = np.zeros(m*n, dtype=float)
    x[:8:2]   = 1
    x[1:8:2]  = np.linspace(start=-1, stop=1, num=4, endpoint=True)
    x[8:16:2] = -1
    x[9:16:2] = np.linspace(start=-1, stop=1, num=4, endpoint=True)
    x[16:20]  = [1/3,-1,-1/3,-1]
    x[20:24]  = [1/3,1,-1/3,1]
    return x

def split_idx(n,m):
    return (int(n//m) + int(n))*np.arange(m, dtype=int)

def get_er_graph(n,p,seed=-1):
    """ Generates Erdos-Reyni random graph """

    p_control = (1.05*np.log(n)/np.log(2))/n
    if p < p_control:
        print('{} < {:.4f}; Unlikely graph will be connected...'.format(p, p_control))

    A = np.zeros((n,n), dtype=int)

    if seed >= 0:
        np.random.seed(seed)

    for i in range(n):
        for j in range(i):
            if np.random.random() < p:
                A[i,j] = 1
    A = A + A.T
    d = np.sum(A, axis=0)
    L = np.diag(d) - A

    assert la.norm(np.dot(L, np.ones(n))) < 1e-15

    x = np.append(1, np.zeros(n-1))
    niters = int(np.log(n)/np.log(2)+1)
    # Breadth first search
    for _ in range(niters):
        x = x + np.dot(A, x)
        x = (x != 0).astype('int')
    is_connected = np.count_nonzero(x) == n

    print('Graph is {}connected'.format('' if is_connected else 'dis'))

    return spp.csr_matrix(L)

def V(x,u): 
    """ Bregman divergence with $\omega=\| \cdot \|_2^2$ """
    return 0.5*la.norm(x-u, ord=2)**2
