import pycutest, os, random
import numpy as np
import numpy.linalg as la
import scipy.optimize as sciopt

"""
This interface requires particular text file named ``output.txt'', which 
holds metadata for all the qualified PYCUTEST functions. To create this
function, you must run

```
python print_metadata.py > output.txt.
```

For ease of access, we've included the function ``print_metadata()`` below,
which must be run as described above to create `output.txt`. For example,
you can simply create your own `print_metadata.py` file, paste the function
below into it, and call it like above.

Then, place the text file `output.txt` in the same diretory where you are
running the script.

Make sure you have pycutest installed. For instructions, please refer to
https://github.com/jfowkes/pycutest.
"""

def print_metadata():
    """ Imports unconstrainted problems with variable input size """
    probs = pycutest.find_problems(objective='LQS', constraints='U', userN=True, regular=True)
    probs = sorted(probs)
    print('List of {} Possible problems: {}'.format(len(probs), probs))

    for prob in probs:
        print('Name={}'.format(prob))
        # print(pycutest.problem_properties(prob))
        pycutest.print_available_sif_params(prob)
        print('End=')

def gather_metadata():
    """ Parses through problems, saves only those which have input size of 100 """
    fp = open('output.txt', 'r+')
    active = False
    viable = False
    setting = ''
    num_probs = 0
    settings = {}
    for line in fp.readlines():
        line = line.rstrip()
        if not active and ('Name=' in line):
            prob_name = line[len('Name='):]
            active = True
            viable = False
            setting = ''
        elif active and (('N = 100 ' in line) or ('n = 100 ' in line) or ('N = 100)' in line) or ('n = 100)' in line)):
            viable=True
            setting = line
        elif active and ('End=' in line):
            # Problem sizes for exceptions below are wrong
            if viable and ('YATP' not in prob_name) and ('INTE' not in prob_name):
                # break apart
                idx = setting.find(' = ')
                key = setting[:idx]
                idx2 = setting[idx + 3:].find(' ')
                val = setting[idx+3: idx+3+idx2]
                settings.update({prob_name: '{}:{}'.format(key,val)})
                num_probs += 1
            active=False

    return settings

def import_problems(settings, silent=True):
    """ Caches problems by importing """
    probs = settings.keys()
    prob_ptr = []
    prob_ids = []
    for prob in probs:
        key,val = settings[prob].split(':')
        val = int(val)
        params = {key : val}
        if not silent:
            print('Importing {}...'.format(prob))
        pr = pycutest.import_problem(prob, sifParams=params)
        L = 0; n = 100
        for _ in range(5):
            u = np.random.random(n)
            v = np.random.random(n)
            _, dfu = pr.obj(u, gradient=True)
            _, dfv = pr.obj(v, gradient=True)
            L = max(L, la.norm(dfu-dfv, ord=2)/la.norm(u-v,ord=2))
        if L < 1e5:
            prob_ptr.append(pr)
            prob_ids.append(prob)
            if not silent:
                print('Estimated L={:.4e}'.format(L))
                print('Finished importing\n')
        elif not silent:
            print('Skipping {} due to Lipschitz term {:.4e}\n'.format(prob, L))

    if not silent:
        print('Importing {} problems'.format(len(prob_ids)))
    return prob_ids, prob_ptr

# os.system('python print_metadata.py > output.txt')
# settings = gather_metadata()
# prob_ids, prob_ptr = import_problems(settings)
# num_probs = len(prob_ids)

class Blackbox:
    """ Blackbox function that will return derivatives and function """

    def __init__(self, seed_num=110497, k=20, scale=1):
        """
        Parmeters
        ---------
        - seed_num : int
            Seed number of problem since we take random {f_i}
        - k : int
            Number of summands defining the problem, i.e. f(x)=\sum\limits_{i=1}^k f_i(x)
        - scale : float
            What to scale function by (to improve smoothness, etc.)
        """

        # os.system('python print_metadata.py > output.txt')
        settings = gather_metadata()
        prob_ids, prob_ptr = import_problems(settings)

        # stores problems
        self.prob_ptr = prob_ptr
        self.num_probs = len(prob_ids)

        self.n = 100
        self.k = k
        self.scale = scale

        random.seed(seed_num)
        self.idxs = random.choices(np.arange(self.num_probs), k=self.k)

    def setup_new_prob(self, seed_num):
        """ Setups problems (and resets scale) """
        random.seed(seed_num)
        self.idxs = random.choices(np.arange(self.num_probs), k=self.k)
        self.scale = 1

    def set_scale(self, scale=None):
        """ If no scale, scale down by inverse of estimated Lipschitz smoothness """
        if scale is None:
            # use Lipschitz smoothness as estimate
            self.scale = 1.0/self.estimate_L()
        else:
            self.scale = scale

    def get_scale(self):
        return self.scale

    def f_df(self,x):
        assert len(x) == self.n, 'Input must be size {}, recieved {}'.format(self.n, len(x))
        f = 0
        df = np.zeros(self.n)

        for idx in self.idxs:
            _f,_df = self.prob_ptr[idx].obj(x, gradient=True)
            f += _f
            df+= _df

        return [self.scale*f, self.scale*df]

    def f(self,x):
        return self.f_df(x)[0]

    def df(self,x):
        return self.f_df(x)[1]

    def f_i(self,x,i):
        _f,_df = self.prob_ptr[i].obj(x, gradient=True)
        return self.scale * _f

    def df_i(self,x,i):
        _f,_df = self.prob_ptr[i].obj(x, gradient=True)
        return self.scale * _df

    def f_df_long(self,x):
        assert len(x) == self.n*self.k, 'Input must be size {}, recieved {}'.format(self.n*self.k, len(x))
        f = 0
        df = np.zeros(self.k*self.n)

        for i,idx in enumerate(self.idxs):
            u = x[i*self.n:(i+1)*self.n]
            _f,_df = self.prob_ptr[idx].obj(u, gradient=True)
            f += _f
            df[i*self.n:(i+1)*self.n] = _df

        return [self.scale*f, self.scale*df]

    def f_long(self,x):
        return self.f_df_long(x)[0]

    def df_long(self,x):
        return self.f_df_long(x)[1]

    def estimate_L(self, nsamples=10):
        """ Randomly selects points to estimates Lipschitz smoothness constant """
        L = 0
        for _ in range(nsamples):
            u = 5*(2*np.random.random()-1)*np.random.random(self.n)
            v = 5*(2*np.random.random()-1)*np.random.random(self.n)
            dfu = self.df(u)
            dfv = self.df(v)
            L = max(L, la.norm(dfu-dfv, ord=2)/la.norm(u-v,ord=2))

        return L

    def get_optimal_sol(self):
        """ Given a blackbox object (see ```pytest_interface.py'''), solves using
            SciPy's optimizer 
        """
        eps = 1e-06
        gtol = eps
        n = 100
        x0 = np.zeros(n)
        res = sciopt.minimize(self.f, x0, jac=self.df, method="BFGS", tol=eps, options={'gtol': gtol, 'norm': 2, 'maxiter': None})

        xstar = res.x
        fstar = self.f(xstar)

        return [fstar, xstar]


def print_metadata():
    # Find unconstrained, variable-dimension problems
    probs = pycutest.find_problems(objective='LQS', constraints='U', userN=True, regular=True)
    probs = sorted(probs)
    # print('List of {} Possible problems: {}'.format(len(probs), probs))

    for prob in probs:
        print('Name={}'.format(prob))
        # print(pycutest.problem_properties(prob))
        pycutest.print_available_sif_params(prob)
        print('End=')

def main():
    pass
    """
    pint = Blackbox()
    for _ in range(10):
        print(pint.estimate_L())
    pint.set_scale()
    for _ in range(10):
        print(pint.estimate_L())
    """

if __name__ == '__main__':
    main()
