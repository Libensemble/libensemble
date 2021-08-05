import numpy as np
import numpy.linalg as la
import scipy.sparse as spp
import cvxpy as cp
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg

def print_final_score(x, f_i_idxs, gen_specs, libE_info):

   # evaluate { f_i(x) } first
   l = len(f_i_idxs)
   H_o = np.zeros(l, dtype=gen_specs['out'])
   H_o['x'][:] = x
   H_o['consensus_pt'][:] = False
   H_o['obj_component'][:] = f_i_idxs
   H_o['get_grad'][:] = False
   H_o = np.reshape(H_o, newshape=(-1,))      

   tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)
   f_is = calc_in['f_i']
   F_i  = np.sum(f_is)

   # get alloc to print sum of f_i
   H_o = np.zeros(1, dtype=gen_specs['out'])
   H_o['x'][0] = x
   H_o['f_i'][0] = F_i
   H_o['eval_pt'][0] = True
   H_o['consensus_pt'][0] = True

   sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

def get_grad(x, f_i_idxs, gen_specs, libE_info):
    l = len(f_i_idxs)
    H_o = np.zeros(l, dtype=gen_specs['out'])
    H_o['x'][:] = x
    H_o['consensus_pt'][:] = False
    H_o['obj_component'][:] = f_i_idxs
    H_o['get_grad'][:] = True
    H_o = np.reshape(H_o, newshape=(-1,))      # unfold into 1d array

    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

    gradf_is = calc_in['gradf_i']
    gradf    = np.sum(gradf_is, axis=0)

    return gradf

def get_grad_locally(x, f_i_idxs, df):
    gradf = np.zeros(len(x), dtype=float)
    for i in f_i_idxs:
        gradf += df(x, i)

    return gradf

def get_neighbor_vals(x, local_gen_id, A_gen_ids_no_local, gen_specs, libE_info):
    """ Sends local gen data (@x) and retrieves neighbors local data.
        Sorts the data so the gen ids are in increasing order

    Parameters
    ----------
    x : np.ndarray
        - local input variable

    local_gen_id : int
        - this gen's gen_id

    A_gen_ids_local : int
        - expected neighbor's gen ids, not including local gen id

    gen_specs, libE_info : ?
        - objects to communicate and construct mini History array

    Returns
    -------
    X : np.ndarray 
        - 2D array of neighbors and local x values sorted by gen_ids
    """
    H_o = np.zeros(1, dtype=gen_specs['out'])
    H_o['x'][0] = x
    H_o['consensus_pt'][0] = True

    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

    neighbor_X = calc_in['x']
    neighbor_gen_ids = calc_in['gen_worker']

    assert local_gen_id not in neighbor_gen_ids, 'Local data should not be ' + \
                                                 'sent back from manager'
    assert np.array_equal(A_gen_ids_no_local, neighbor_gen_ids),'Expected ' + \
                'gen_ids {}, received {}'.format(A_gen_ids, gen_ids)

    X = np.vstack((neighbor_X, x))
    gen_ids = np.append(neighbor_gen_ids, local_gen_id)

    # sort data (including local) in corresponding gen_id increasing order
    X[:] = X[np.argsort(gen_ids)]

    return X

def get_consensus_gradient(x, gen_specs, libE_info):
    """ Sends local gen data (@x) and retrieves neighbors local data,
        and takes sum of the neighbors' x's, which is equivalent to taking
        the gradient of consensus term for this particular node/agent. 

        This function is equivalent to the @get_neighbor_vals function, but is
        less general, i.e., when we need only take a sum rather than a linear
        combination of our neighbors.

    Parameters
    ----------
    x : np.ndarray
        - local input variable

    Returns
    -------
     : np.ndarray 
        - Returns this node's corresponding gradient of consensus 
    """
    H_o = np.zeros(1, dtype=gen_specs['out'])
    H_o['x'][0] = x
    H_o['consensus_pt'][0] = True

    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], H_o)

    neighbor_X = calc_in['x']
    num_neighbors = len(neighbor_X)

    return (num_neighbors*x) - np.sum(neighbor_X, axis=0)

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

    X = spp.csr_matrix(x)
    return X

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

"""
The remaining functions below don't help with consensus, but help with the
regression tests. One can move these functions to a different Python file
if need be.
"""
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

    # print('F(x*)={:.4f}'.format(problem.value))
    return problem.value
    # return np.reshape(beta.value, newshape=(-1,))

def regls_opt(X, y, c, reg=None):
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

    # print('F(x*)={:.4f}'.format(problem.value))
    return problem.value
    return beta.value

def log_opt(X, y, c, reg=None):
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

    # print('F(x*)={:.4f}'.format(problem.value))
    return problem.value
    return beta.value

def svm_opt(X, b, c, reg='l1'):
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

    # print('F(x*)={:.4f}'.format(problem.value))
    return problem.value
    return theta.value
