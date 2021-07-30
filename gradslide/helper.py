import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
# from sklearn.linear_model import LogisticRegression
# import cvxopt 
import cvxpy as cp
import scipy.optimize as sciopt

"""
Function definitions
"""

# GEOMETRIC MEDIAN
def f_gm_comb(x,b,m):
    return f_gm(np.kron(np.ones(m),x),b,m)

def f_gm(x,b,m):
    """ Geometric median """
    assert len(x)==len(b)
    n = len(x)//m
    z = x-b
    Z = np.reshape(z, newshape=(m,n))
    norms_z = la.norm(Z, axis=1, ord=2)
    return (1/m) * np.sum(norms_z)

def df_gm(x,b,m):
    assert len(x)*m==len(b)
    return f_gm_long(np.kron(np.ones(m),x), b)

def df_gm_long(x,b,m):
    """ Geometric median """
    n = len(x)//m
    z = x-b
    Z = np.reshape(z, newshape=(m,n))
    norms_z = la.norm(Z, axis=1, ord=2)
    norms_z[norms_z == 0] = np.inf
    assert len(norms_z) == m
    norms_Z = np.tile(norms_z,(n,1)).T
    Z = np.divide(Z, norms_Z)
    return (1/m) * np.reshape(Z, newshape=(-1,))

def gm_opt(b,m):

    n = len(b)//m
    assert len(b) == m*n

    ones_m = np.ones((m,1))
    def obj_fn(x,B,m):
        X = ones_m @ x
        return (1/m) * cp.sum( cp.norm(X-B, 2, axis=1) )

    beta = cp.Variable((1,n))
    B = np.reshape(b, newshape=(m,n))
    problem = cp.Problem(cp.Minimize(obj_fn(beta, B, m)))
    # print('Problem is DCP: {}'.format(problem.is_dcp()))
    # problem.solve(verbose=True)
    problem.solve()

    print('Value: {}'.format(problem.value))
    return np.reshape(beta.value, newshape=(-1,))

    ########################################################

    # TODO: Get this working ...
    d,m = X.shape

    theta = cp.Variable((d,1))
    # loss = cp.sum([
    reg = cp.norm(theta, 1)
    lambd = cp.Parameter(nonneg=True)
    prob = cp.Problem(cp.Minimize(loss/m + lambd*reg))
    lambd.value = c
    prob.solve()
    return theta.value

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
"""
def regls_opt_old(X, y):
    A = np.dot(X,X.T)
    m = A.shape[0]
    xstar = la.lstsq(A+m*c*np.eye(m), np.dot(X,y))[0]
    return xstar
"""

def regls_opt(X, y, reg=None):
    assert reg=='l1' or reg=='l2' or reg is None
    if reg=='l1': p = 1
    elif reg=='l2': p = 2
    elif reg is None: p = -1
    else: assert False, 'illegal regularization "{}"'.format(reg)

    def obj_fn(X,y,beta,c,p):
        m = X.shape[0]
        if p==1:
            return (1/m) * cp.pnorm(X @ beta - y, p=2)**2 + c * cp.pnorm(beta, p=1)
        return (1/m) * cp.pnorm(X @ beta - y, p=2)**2 + c * cp.pnorm(beta, p=2)**2

    # already X.T
    d,m = X.shape
    beta = cp.Variable(d)
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(obj_fn(X.T, y, beta, c, p)))
    print('Problem is DCP: {}'.format(problem.is_dcp()))
    # problem.solve(verbose=True)
    problem.solve()

    print('Value: {}'.format(problem.value))
    return beta.value

def df_regls(theta, X, y, reg=None):
    assert reg=='l2' or reg=='l1' or reg is None

    d,m = X.shape
    ones_d = np.ones(d)

    # broadcast multiply
    X_arr = np.reshape(X.T, newshape=(-1,))
    Theta = np.reshape(theta, newshape=(m,d))    # unfold
    XT_theta = np.einsum("ij,ij->i", X.T, Theta) # size m array
    # broadcast multiply
    df = (2/m) * np.multiply( X_arr, np.kron(XT_theta-y, ones_d ) )

    if reg is None:
        pass
    elif reg=='l2':
        df += (2*c/m) * theta
    else:
        df += (c/m) * np.sign(theta)

    return df

def df_regls_comb(theta, X, y, reg=None):
    assert reg=='l2' or reg=='l1' or reg is None

    d,m = X.shape

    df = (2/m)*np.dot(X, np.dot(X.T,theta)-y)

    if reg is None:
        pass
    elif reg=='l2':
        df += (2*c) * theta
    else:
        df += c * np.sign(theta)

    return df

def f_regls_long(theta, X, y, reg=None):
    assert reg=='l2' or reg=='l1' or reg is None

    d,m = X.shape
    ones_d = np.ones(d)

    # broadcast multiply
    Theta = np.reshape(theta, newshape=(m,d))  # unfold
    XT_theta = np.einsum("ij,ij->i", X.T, Theta)  # size m array

    f = (1/m)*la.norm(y-XT_theta,ord=2)**2

    # Problem: Forgot to divide regularizer by m
    if reg is None:
        pass
    elif reg=='l2':
        f += (c/m)*np.dot(theta, theta)
    else:
        f += (c/m)*la.norm(theta,1)

    return f

"""
def _f_regls_long(theta, X, y, reg=None):
    assert reg=='l2' or reg=='l1' or reg is None

    d,m = X.shape

    # broadcast multiply
    Theta = np.reshape(theta, newshape=(m,d))  # unfold
    XT_theta = np.einsum("ij,ij->i", X.T, Theta)  # size m array

    # (X_i^Ttheta)^2
    f = np.dot(XT_theta, XT_theta)

    yT_XT_theta = np.dot(y, XT_theta)
    # -2y_i (X_i^Ttheta)
    f += -2*yT_XT_theta

    # add y^Ty
    f += np.dot(y,y)

    # scale down 
    f *= (1/m)

    if reg is None:
        pass
    elif reg=='l2':
        f += c*la.norm(theta,2)**2
    else:
        f += c*la.norm(theta,1)

    return f

def f_regls_opt(A,b,reg=None):
    s = 1/(c**0.5)
    U = cvxopt.matrix(s*A)
    y = cvxopt.matrix(s*b)
    xstar = l1regls(U, y)
    xstar = np.array(xstar)
    return xstar
"""

def f_regls(theta, X, y, reg=None):
    d,m = X.shape
    assert len(theta)==d
    return f_regls_long(np.kron(np.ones(m), theta), X, y, reg)

def f_regls2(theta, X, y, reg=None):
    d,m = X.shape
    return (1/m)*la.norm(y-np.dot(X.T,theta), ord=2)**2 + c*la.norm(theta,ord=2)**2

# SIMPLE
def df_silly(x_in):
    """ Minimize Solves |x_1| + |x_2-1| """
    assert len(x_in) == 4
    df = np.zeros(4)
    df[0] = 2*x_in[0]
    df[3] = 2*(x_in[3]-1)
    return df

# LOGISTIC REGRESSION
def df_log(theta, X, y, reg=None):
    assert reg=='l2' or reg=='l1' or reg is None

    d,m = X.shape
    XT_theta = np.einsum('ij,ij->i', X.T, np.reshape(theta, newshape=X.T.shape))
    base = np.exp(-np.multiply(y, XT_theta))
    sigmoid_base = np.divide(base, 1+base)
    df_scalar = np.multiply(-y, sigmoid_base)
    df = np.multiply( np.reshape(X.T, newshape=(-1,)),  np.kron(df_scalar, np.ones(d)) )

    if reg is None:
        df = (1/m)*df
    elif reg=='l2':
        df = (1/m)*df + (2*c/m)*theta
    else:
        df = (1/m)*df + (c/m)*np.sign(theta)

    return df

def df_log_comb(theta, X, y, reg=None):
    assert reg=='l2' or reg=='l1' or reg is None

    d,m = X.shape
    df_long = df_log(np.kron(np.ones(m), theta), X, y, reg)
    DF = np.reshape(df_long, newshape=(m,d))
    df = np.sum(DF, axis=0)

    return df

def f_log_long(theta, X, y, reg=None):
    assert reg=='l2' or reg=='l1' or reg is None

    d,m = X.shape
    assert len(y)==m
    XT_theta = np.einsum('ij,ij->i', X.T, np.reshape(theta, newshape=X.T.shape))
    z = np.multiply(y, XT_theta)
    summands = np.log(1 + np.exp(-z))
    f = 1/m * np.sum(summands)

    if reg is None:
        pass
    elif reg=='l2':
        f += (c/m) * la.norm(theta,ord=2)**2
    else:
        f += (c/m) * la.norm(theta,ord=1)

    return f

def f_log(theta, X, y, reg=None):
    d,m = X.shape
    assert len(theta)==d
    return f_log_long(np.kron(np.ones(m), theta), X, y, reg)

def log_opt(X, y, reg=None):
    """ https://www.cvxpy.org/examples/machine_learning/logistic_regression.html """
    assert reg=='l1' or reg=='l2' or reg is None
    if reg=='l1': p = 1
    elif reg=='l2': p = 2
    elif reg is None: p = 0
    else: assert False, 'illegal regularization mode, "{}"'.format(reg)

    def obj_fn(X,y,beta,c,p):
        m = X.shape[0]
        if p==0: reg = 0
        if p==1: reg = c * cp.norm(beta, 1)
        elif p==2: reg = c * cp.norm(beta, 2)**2
        # cp.logistic(x) == log(1+e^x)
        return (1/m) * cp.sum(cp.logistic( cp.multiply(-y, X @ beta))) + reg
        
    d,m = X.shape
    beta = cp.Variable(d)
    problem = cp.Problem(cp.Minimize(obj_fn(X.T, y, beta, c, p)))
    # print('Problem is DCP: {}'.format(problem.is_dcp()))
    problem.solve()

    print('Value: {}'.format(problem.value))
    return beta.value

# SUPPORT VECTOR MACHINE
def f_svm_long(theta, X, b, reg=None):
    """ Returns SVM score where

    Parameter
    theta : np.ndarray
        - weight (decision) vector
    b : np.ndarray
        - class labels
    X : np.ndarray
        - 2D array of feastures
    """
    assert reg=='l2' or reg=='l1' or reg is None

    d,m = X.shape
    assert len(b)==m    
    Theta = np.reshape(theta, newshape=X.T.shape)
    XT_theta = np.einsum('ij,ij->i', X.T, Theta)
    h = np.maximum(0, 1-np.multiply(b, XT_theta))
    
    # S = np.hstack((b, Theta.T))
    # S_norms = la.norm(S, ord=2, axis=1)
    # CHANGE THE NORM

    if reg is None:
        theta_norms = 0
    elif reg=='l2':
        theta_norms = (c/m) * la.norm(Theta, ord=2, axis=1)**2
    else:
        theta_norms = (c/m) * la.norm(Theta, ord=1, axis=1)

    # f = np.sum(h) + np.sum(np.divide(theta_norms, S_norms))
    f = np.sum(h) + np.sum(theta_norms)

    return f

def f_svm(theta,X,b,reg=None):
    d,m = X.shape
    assert len(theta)==d
    return f_svm_long(np.kron(np.ones(m), theta), X, b, reg)

def df_svm(theta, X, b, reg=None):
    assert reg=='l2' or reg=='l1' or reg is None

    d,m = X.shape
    Theta = np.reshape(theta, newshape=X.T.shape)
    XT_theta = np.einsum('ij,ij->i', X.T, Theta)
    # will be zero if gradient is zero, else 1 
    zero_df = np.kron((np.multiply(b, XT_theta) < 1).astype('int'), np.ones(d))
    nonzero_df = -np.multiply(np.kron(b, np.ones(d)), np.reshape(X.T, newshape=(-1,)))
    df = np.multiply(zero_df, nonzero_df)

    # S = np.hstack((b, Theta.T))
    # S_norms = la.norm(S, ord=2, axis=1)

    if reg is None:
        reg = 0
    elif reg=='l2':
        # reg = 2*np.divide(theta, np.kron(S_norms, np.ones(d)))
        reg = (2*c/m)*theta
    else:
        # reg = np.divide(np.sign(theta), np.kron(S_norms, np.ones(d)))
        reg = (c/m)*np.sign(theta)

    df = df + reg

    return df

def svm_opt(X, b, reg='l1'):
    if reg=='l1': p = 1
    elif reg=='l2': p = 2
    elif reg is None: p = 0
    else: assert False, 'illegal regularization mode, "{}"'.format(reg)

    def obj_fn(X,b,theta,c,p):
        if p==0: reg = 0
        if p==1: reg = c * cp.norm(theta, 1)
        if p==2: reg = c * cp.norm(theta, 2)**2
        return cp.sum(cp.pos(1-cp.multiply(b, X @ theta))) + reg

    d,m = X.shape
    theta = cp.Variable(d)
    problem = cp.Problem(cp.Minimize(obj_fn(X.T, b, theta, c, p)))
    # print('Problem is DCP: {}'.format(problem.is_dcp()))
    problem.solve()

    print('Value: {}'.format(problem.value))
    return theta.value

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

    assert is_connected, 'Graph must be connected, increase either @m or @p'

    # print('Graph is {}connected'.format('' if is_connected else 'dis'))

    return spp.csr_matrix(L)

def V(x,u): 
    """ Bregman divergence with $\omega=\| \cdot \|_2^2$ """
    return 0.5*la.norm(x-u, ord=2)**2

def readin_csv(fname):
    fp = open(fname, 'r+')

    n = 569
    label = np.zeros(n, dtype='int')
    datas = np.zeros((n,30))
    i = 0
    
    for line in fp.readlines():
        line = line.rsplit()[0]
        data = line.split(',')
        label[i] = data[1]=='M'
        datas[i,:] = [float(val) for val in data[2:32]]
        i += 1

    assert i==n, 'Expected {} datapoints, recorded'.format(n, i)

    return label, datas

def optimize_blackbox(bbox):
    """ Given a blackbox object (see ```pytest_interface.py'''), solves using
        SciPy's optimizer 
    """
    eps = 1e-06
    gtol = eps
    n = 100
    x0 = np.zeros(n)
    res = sciopt.minimize(bbox.f, x0, jac=bbox.df, method="BFGS", tol=eps, options={'gtol': gtol, 'norm': 2, 'maxiter': None})

    xstar = res.x
    fstar = bbox.f(xstar)

    return [fstar, xstar]
