"""
This module contains methods for APOSMM to interface with various local
optimization routines.
"""

__all__ = [
    "LocalOptInterfacer",
    "run_local_nlopt",
    "run_local_tao",
    "run_local_dfols",
    "run_local_ibcdfo_pounders",
    "run_local_ibcdfo_manifold_sampling",
    "run_local_scipy_opt",
    "run_external_localopt",
]

from multiprocessing import Event, Process, Queue

import numpy as np
import psutil

import libensemble.gen_funcs
from libensemble.message_numbers import EVAL_GEN_TAG, STOP_TAG  # Only used to simulate receiving from manager


class APOSMMException(Exception):
    """Raised for any exception in APOSMM"""


optimizer_list = ["petsc", "nlopt", "dfols", "scipy", "ibcdfo_pounders", "ibcdfo_manifold_sampling", "external"]
optimizers = libensemble.gen_funcs.rc.aposmm_optimizers

if optimizers is not None:
    assert isinstance(optimizers, list), "Must have a list"
    unrec = set(optimizers) - set(optimizer_list)
    if unrec:
        raise APOSMMException(f"APOSMM Error: unrecognized optimizers {unrec}")

    # Preferable to import globally in most cases
    if "petsc" in optimizers:
        from petsc4py import PETSc  # noqa: F401
    if "nlopt" in optimizers:
        import nlopt  # noqa: F401
    if "dfols" in optimizers:
        import dfols  # noqa: F401
    if "ibcdfo_pounders" in optimizers:
        from ibcdfo.pounders import pounders  # noqa: F401
    if "ibcdfo_manifold_sampling" in optimizers:
        from ibcdfo.manifold_sampling import manifold_sampling_primal  # noqa: F401
    if "scipy" in optimizers:
        from scipy import optimize as sp_opt  # noqa: F401
    if "external_localopt" in optimizers:
        pass


class ConvergedMsg(object):
    """
    Message communicated when a local optimization is converged.
    """

    def __init__(self, x, opt_flag):
        self.x = x
        self.opt_flag = opt_flag


class ErrorMsg(object):
    """
    Message communicated when a local optimization has an exception.
    """

    def __init__(self, x):
        self.x = x


class LocalOptInterfacer(object):
    """
    This class defines the APOSMM interface to various local optimization routines.

    Currently supported routines are

    - NLopt [``'LN_SBPLX'``, ``LN_BOBYQA'``, ``'LN_COBYLA'``, ``'LN_NEWUOA'``, ``'LN_NELDERMEAD'``, ``'LD_MMA'``]
    - PETSc/TAO [``'pounders'``, ``'blmvm'``, ``'nm'``]
    - SciPy [``'scipy_Nelder-Mead'``, ``'scipy_COBYLA'``, ``'scipy_BFGS'``]
    - DFOLS [``'dfols'``]
    - IBCDFO [``'pounders'``, ``'manifold_sampling_primal'``]
    - External local optimizer [``'external_localopt'``] (which use files to pass/receive ``x/f`` values)
    """

    def __init__(self, user_specs, x0, f0, grad0=None):
        """
        :param x0: A numpy array of the initial guess solution. This guess
            should be scaled to a unit cube.
        :param f0: A numpy array of the initial function value.


        .. warning:: In order to have correct functioning of the local
            optimization child processes. ~self.iterate~ should be called
            immediately after creating the class.

        """
        self.parent_can_read = Event()
        self.comm_queue = Queue()
        self.child_can_read = Event()

        self.x0 = x0.copy()
        self.f0 = f0.copy()
        if grad0 is not None:
            self.grad0 = grad0.copy()
        else:
            self.grad0 = None

        # Setting the local optimization method
        if user_specs["localopt_method"] in [
            "LN_SBPLX",
            "LN_BOBYQA",
            "LN_COBYLA",
            "LN_NEWUOA",
            "LN_NELDERMEAD",
            "LD_MMA",
        ]:
            run_local_opt = run_local_nlopt
        elif user_specs["localopt_method"] in ["pounders", "blmvm", "nm"]:
            run_local_opt = run_local_tao
        elif user_specs["localopt_method"] in ["scipy_Nelder-Mead", "scipy_COBYLA", "scipy_BFGS"]:
            run_local_opt = run_local_scipy_opt
        elif user_specs["localopt_method"] in ["dfols"]:
            run_local_opt = run_local_dfols
        elif user_specs["localopt_method"] in ["ibcdfo_pounders"]:
            run_local_opt = run_local_ibcdfo_pounders
        elif user_specs["localopt_method"] in ["ibcdfo_manifold_sampling"]:
            run_local_opt = run_local_ibcdfo_manifold_sampling
        elif user_specs["localopt_method"] in ["external_localopt"]:
            run_local_opt = run_external_localopt
        else:
            raise APOSMMException(f"APOSMM Error: unrecognized method {user_specs['localopt_method']}")

        self.parent_can_read.clear()
        self.process = Process(
            target=opt_runner,
            args=(run_local_opt, user_specs, self.comm_queue, x0, f0, self.child_can_read, self.parent_can_read),
        )

        self.process.start()
        self.is_running = True
        self.parent_can_read.wait()
        x_new = self.comm_queue.get()
        if isinstance(x_new, ErrorMsg):
            raise APOSMMException(x_new.x)

        assert np.allclose(
            x_new, x0, rtol=1e-15, atol=1e-15
        ), "The first point requested by this run does not match the starting point. Exiting"

    def iterate(self, data):
        """
        Returns an instance of either ``numpy.ndarray`` corresponding to the next
        iterative guess or ``ConvergedMsg`` when the solver has completed its run.

        :param x_on_cube: A numpy array of the point being evaluated (for a handshake)
        :param f: A numpy array of the function evaluation.
        :param grad: A numpy array of the function's gradient.
        :param fvec: A numpy array of the function's component values.
        """

        self.parent_can_read.clear()

        if "grad" in data.dtype.names:
            self.comm_queue.put((data["x_on_cube"], data["f"], data["grad"]))
        elif "fvec" in data.dtype.names:
            self.comm_queue.put((data["x_on_cube"], data["fvec"]))
        else:
            self.comm_queue.put((data["x_on_cube"], data["f"]))

        self.child_can_read.set()
        self.parent_can_read.wait()

        x_new = self.comm_queue.get()
        if isinstance(x_new, ErrorMsg):
            raise APOSMMException(x_new.x)
        elif isinstance(x_new, ConvergedMsg):
            self.close()
        else:
            x_new = np.atleast_2d(x_new)

        return x_new

    def destroy(self):
        """Recursively kill any optimizer processes still running"""
        if self.process.is_alive():
            process = psutil.Process(self.process.pid)
            for child in process.children(recursive=True):
                child.kill()
            process.kill()
        self.close()

    def close(self):
        """Join process and close queue"""
        self.process.join()
        self.comm_queue.close()
        self.comm_queue.join_thread()
        self.is_running = False


def run_local_nlopt(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):
    """
    Runs an NLopt local optimization run starting at ``x0``, governed by the
    parameters in ``user_specs``.
    """

    import nlopt  # noqa: F811

    # print('[Child]: Started local opt at {}.'.format(x0), flush=True)
    n = len(user_specs["ub"])

    opt = nlopt.opt(getattr(nlopt, user_specs["localopt_method"]), n)

    lb = np.zeros(n)
    ub = np.ones(n)

    if not user_specs.get("periodic"):
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)

    # Care must be taken here because a too-large initial step causes nlopt to move the starting point!
    dist_to_bound = min(min(ub - x0), min(x0 - lb))
    assert dist_to_bound > np.finfo(np.float64).eps, "The distance to the boundary is too small for NLopt to handle"

    if "dist_to_bound_multiple" in user_specs:
        opt.set_initial_step(dist_to_bound * user_specs["dist_to_bound_multiple"])
    else:
        opt.set_initial_step(dist_to_bound)

    run_max_eval = user_specs.get("run_max_eval", 1000 * n)
    opt.set_maxeval(run_max_eval)

    opt.set_min_objective(
        lambda x, grad: nlopt_callback_fun(x, grad, comm_queue, child_can_read, parent_can_read, user_specs)
    )

    if "xtol_rel" in user_specs:
        opt.set_xtol_rel(user_specs["xtol_rel"])
    if "ftol_rel" in user_specs:
        opt.set_ftol_rel(user_specs["ftol_rel"])
    if "xtol_abs" in user_specs:
        opt.set_xtol_abs(user_specs["xtol_abs"])
    if "ftol_abs" in user_specs:
        opt.set_ftol_abs(user_specs["ftol_abs"])

    # FIXME: Do we need to do something of the final 'x_opt'?
    # print('[Child]: Started my optimization', flush=True)
    x_opt = opt.optimize(x0)
    return_val = opt.last_optimize_result()

    if return_val >= 1 and return_val <= 4:
        # These return values correspond to an optimium being identified
        # https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#return-values
        opt_flag = 1
    elif return_val >= 5:
        print(
            "[APOSMM] The run started from " + str(x0) + " reached its maximum number "
            "of function evaluations: " + str(run_max_eval) + ". No point from "
            "this run will be ruled as a minimum! APOSMM may start a new run "
            "from some point in this run."
        )
        opt_flag = 0
    else:
        print("[APOSMM] NLopt returned with a negative return value, which indicates an error")
        opt_flag = 0

    if user_specs.get("periodic"):
        # Shift x_opt to be in the correct location in the unit cube
        # (not the domain user_specs['lb'] - user_specs['ub'])
        x_opt = x_opt % 1

    finish_queue(x_opt, opt_flag, comm_queue, parent_can_read, user_specs)


def run_local_scipy_opt(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):
    """
    Runs a SciPy local optimization run starting at ``x0``, governed by the
    parameters in ``user_specs``.
    """
    from scipy import optimize as sp_opt  # noqa: F811

    # Construct the bounds in the form of constraints
    cons = []
    for factor in range(len(x0)):
        lo = {"type": "ineq", "fun": lambda x, lb=user_specs["lb"][factor], i=factor: x[i] - lb}
        up = {"type": "ineq", "fun": lambda x, ub=user_specs["ub"][factor], i=factor: ub - x[i]}
        cons.append(lo)
        cons.append(up)

    method = user_specs["localopt_method"][6:]
    jac_flag = method in ["BFGS"]
    # print('[Child]: Started my optimization', flush=True)
    res = sp_opt.minimize(
        lambda x: scipy_dfols_callback_fun(x, comm_queue, child_can_read, parent_can_read, user_specs),
        x0,
        # constraints=cons,
        method=method,
        jac=jac_flag,
        **user_specs.get("scipy_kwargs", {}),
    )

    if res["status"] in user_specs["opt_return_codes"]:
        opt_flag = 1
    else:
        print(
            "[APOSMM] The SciPy localopt run started from " + str(x0) + " stopped"
            " without finding a local min.\nThe 'status' of the run is "
            + str(res["status"])
            + ' and the message is: "'
            + res["message"]
            + '".\nNo point from this run will be ruled as a minimum! APOSMM may '
            "start a new run from some point in this run.\n"
        )
        opt_flag = 0

    if user_specs.get("periodic"):
        x_opt = res["x"] % 1
    else:
        x_opt = res["x"]

    finish_queue(x_opt, opt_flag, comm_queue, parent_can_read, user_specs)


def run_external_localopt(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):
    """
    Runs an external local optimization run starting at ``x0``, governed by the
    parameters in ``user_specs``.
    """

    import os
    import subprocess
    from uuid import uuid4

    run_id = uuid4().hex

    x_file = "x_" + run_id + ".txt"
    y_file = "y_" + run_id + ".txt"
    x_done_file = "x_done_" + run_id + ".txt"
    y_done_file = "y_done_" + run_id + ".txt"
    opt_file = "opt_" + run_id + ".txt"

    # cmd = ["matlab", "-nodisplay", "-nodesktop", "-nojvm", "-nosplash", "-r",
    cmd = [
        "octave",
        "--no-window-system",
        "--eval",
        "x0=[" + " ".join([f"{x:18.18f}" for x in x0]) + "];"
        "opt_file='" + opt_file + "';"
        "x_file='" + x_file + "';"
        "y_file='" + y_file + "';"
        "x_done_file='" + x_done_file + "';"
        "y_done_file='" + y_done_file + "';"
        "call_matlab_octave_script",
    ]

    p = subprocess.Popen(cmd, shell=False, stdout=subprocess.DEVNULL)

    while p.poll() is None:  # Process still going
        if os.path.isfile(x_done_file):  # x file exists
            x = np.loadtxt(x_file)
            os.remove(x_done_file)

            x_recv, f_recv = put_set_wait_get(x, comm_queue, parent_can_read, child_can_read, user_specs)

            np.savetxt(y_file, [f_recv])
            open(y_done_file, "w").close()

    x_opt = np.loadtxt(opt_file)
    opt_flag = np.loadtxt(opt_file + "_flag")

    for f in [x_file, y_file, opt_file]:
        os.remove(f)

    finish_queue(x_opt, opt_flag, comm_queue, parent_can_read, user_specs)


def run_local_dfols(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):
    """
    Runs a DFOLS local optimization run starting at ``x0``, governed by the
    parameters in ``user_specs``.
    """
    import dfols  # noqa: F811

    # Define bound constraints (lower <= x <= upper)
    lb = np.zeros(len(x0))
    ub = np.ones(len(x0))

    # Set random seed (for reproducibility)
    np.random.seed(0)

    # Care must be taken here because a too-large initial step causes DFO-LS to move the starting point!
    dist_to_bound = min(min(ub - x0), min(x0 - lb))
    assert dist_to_bound > np.finfo(np.float64).eps, "The distance to the boundary is too small"
    assert "bounds" not in user_specs.get("dfols_kwargs", {}), "APOSMM must set the bounds for DFO-LS"
    assert "rhobeg" not in user_specs.get("dfols_kwargs", {}), "APOSMM must set rhobeg for DFO-LS"
    assert "x0" not in user_specs.get("dfols_kwargs", {}), "APOSMM must set x0 for DFO-LS"

    # Call DFO-LS
    soln = dfols.solve(
        lambda x: scipy_dfols_callback_fun(x, comm_queue, child_can_read, parent_can_read, user_specs),
        x0,
        bounds=(lb, ub),
        rhobeg=0.5 * dist_to_bound,
        **user_specs.get("dfols_kwargs", {}),
    )

    x_opt = soln.x

    if soln.flag == soln.EXIT_SUCCESS:
        opt_flag = 1
    else:
        print(
            "[APOSMM] The DFO-LS run started from " + str(x0) + " stopped with an exit "
            "flag of " + str(soln.flag) + ". No point from this run will be "
            "ruled as a minimum! APOSMM may start a new run from some point "
            "in this run."
        )
        opt_flag = 0

    finish_queue(x_opt, opt_flag, comm_queue, parent_can_read, user_specs)


def run_local_ibcdfo_manifold_sampling(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):
    """
    Runs a IBCDFO local optimization run starting at ``x0``, governed by the
    parameters in ``user_specs``.

    Although IBCDFO methods can receive previous evaluations, few other methods
    support that, so APOSMM assumes the first point will be re-evaluated (but
    not be sent back to the manager).
    """
    n = len(x0)
    # Define bound constraints (lower <= x <= upper)
    lb = np.zeros(n)
    ub = np.ones(n)

    # Set random seed (for reproducibility)
    np.random.seed(0)

    # dist_to_bound = min(min(ub - x0), min(x0 - lb))
    # assert dist_to_bound > np.finfo(np.float64).eps, "The distance to the boundary is too small"

    run_max_eval = user_specs.get("run_max_eval", 100 * (n + 1))
    # g_tol = 1e-8
    # delta_0 = 0.5 * dist_to_bound
    # m = len(f0)
    subprob_switch = "linprog"

    [X, F, hF, xkin, flag] = manifold_sampling_primal(
        user_specs["hfun"],
        lambda x: scipy_dfols_callback_fun(x, comm_queue, child_can_read, parent_can_read, user_specs),
        x0,
        lb,
        ub,
        run_max_eval,
        subprob_switch,
    )

    assert flag >= 0 or flag == -6, "IBCDFO errored"

    x_opt = X[xkin]

    if flag > 0:
        opt_flag = 1
    else:
        print(
            "[APOSMM] The IBCDFO run started from " + str(x0) + " stopped with an exit "
            "flag of " + str(flag) + ". No point from this run will be "
            "ruled as a minimum! APOSMM may start a new run from some point "
            "in this run."
        )
        opt_flag = 0

    finish_queue(x_opt, opt_flag, comm_queue, parent_can_read, user_specs)


def run_local_ibcdfo_pounders(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):
    """
    Runs a IBCDFO local optimization run starting at ``x0``, governed by the
    parameters in ``user_specs``.

    Although IBCDFO methods can receive previous evaluations, few other methods
    support that, so APOSMM assumes the first point will be re-evaluated (but
    not be sent back to the manager).
    """
    n = len(x0)
    # Define bound constraints (lower <= x <= upper)
    lb = np.zeros(n)
    ub = np.ones(n)

    # Set random seed (for reproducibility)
    np.random.seed(0)

    dist_to_bound = min(min(ub - x0), min(x0 - lb))
    assert dist_to_bound > np.finfo(np.float64).eps, "The distance to the boundary is too small"

    run_max_eval = user_specs.get("run_max_eval", 100 * (n + 1))
    g_tol = 1e-8
    delta_0 = 0.5 * dist_to_bound
    m = len(f0)

    if "hfun" in user_specs:
        Options = {"hfun": user_specs["hfun"], "combinemodels": user_specs["combinemodels"]}
    else:
        Options = None

    [X, F, hF, flag, xkin] = pounders(
        lambda x: scipy_dfols_callback_fun(x, comm_queue, child_can_read, parent_can_read, user_specs),
        x0,
        n,
        run_max_eval,
        g_tol,
        delta_0,
        m,
        lb,
        ub,
        Options=Options,
    )

    assert flag >= 0 or flag == -6, "IBCDFO errored"

    x_opt = X[xkin]

    if flag == 0 or flag == -6:
        opt_flag = 1
    else:
        print(
            "[APOSMM] The IBCDFO run started from " + str(x0) + " stopped with an exit "
            "flag of " + str(flag) + ". No point from this run will be "
            "ruled as a minimum! APOSMM may start a new run from some point "
            "in this run."
        )
        opt_flag = 0

    finish_queue(x_opt, opt_flag, comm_queue, parent_can_read, user_specs)


def run_local_tao(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):
    """
    Runs a PETSc/TAO local optimization run starting at ``x0``, governed by the
    parameters in ``user_specs``.
    """

    from petsc4py import PETSc  # noqa: F811

    assert isinstance(x0, np.ndarray)

    tao_comm = PETSc.COMM_SELF
    (n,) = x0.shape
    if f0.shape == ():
        m = 1
    else:
        (m,) = f0.shape

    # Create starting point, bounds, and tao object
    x = PETSc.Vec().create(tao_comm)
    x.setSizes(n)
    x.setFromOptions()
    x.array = x0
    lb = x.duplicate()
    ub = x.duplicate()
    lb.array = 0 * np.ones(n)
    ub.array = 1 * np.ones(n)
    tao = PETSc.TAO().create(tao_comm)
    tao.setType(user_specs["localopt_method"])

    if user_specs["localopt_method"] == "pounders":
        f = PETSc.Vec().create(tao_comm)
        f.setSizes(m)
        f.setFromOptions()

        if hasattr(tao, "setResidual"):
            tao.setResidual(
                lambda tao, x, f: tao_callback_fun_pounders(
                    tao, x, f, comm_queue, child_can_read, parent_can_read, user_specs
                ),
                f,
            )
        else:
            tao.setSeparableObjective(
                lambda tao, x, f: tao_callback_fun_pounders(
                    tao, x, f, comm_queue, child_can_read, parent_can_read, user_specs
                ),
                f,
            )
        delta_0 = user_specs["dist_to_bound_multiple"] * np.min(
            [np.min(ub.array - x.array), np.min(x.array - lb.array)]
        )
        PETSc.Options().setValue("-tao_pounders_delta", str(delta_0))

    elif user_specs["localopt_method"] == "nm":
        tao.setObjective(
            lambda tao, x: tao_callback_fun_nm(tao, x, comm_queue, child_can_read, parent_can_read, user_specs)
        )

    elif user_specs["localopt_method"] == "blmvm":
        g = PETSc.Vec().create(tao_comm)
        g.setSizes(n)
        g.setFromOptions()
        tao.setObjectiveGradient(
            lambda tao, x, g: tao_callback_fun_grad(tao, x, g, comm_queue, child_can_read, parent_can_read, user_specs),
            None,
        )

    # Set everything for tao before solving
    PETSc.Options().setValue("-tao_max_funcs", str(user_specs.get("run_max_eval", 1000 * n)))
    tao.setFromOptions()
    tao.setVariableBounds((lb, ub))

    tao.setTolerances(grtol=user_specs.get("grtol", 1e-8), gatol=user_specs.get("gatol", 1e-8))
    tao.setInitial(x)

    # print('[Child]: Started my optimization', flush=True)
    tao.solve(x)

    x_opt = tao.getSolution().getArray()
    exit_code = tao.getConvergedReason()

    if exit_code > 0:
        opt_flag = 1
    else:
        # https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Tao/TaoGetConvergedReason.html
        print(
            "[APOSMM] The run started from " + str(x0) + " exited with a nonpositive reason. No point from "
            "this run will be ruled as a minimum! APOSMM may start a new run from some point in this run."
        )
        opt_flag = 0

    if user_specs["localopt_method"] == "pounders":
        f.destroy()
    elif user_specs["localopt_method"] == "blmvm":
        g.destroy()

    lb.destroy()
    ub.destroy()
    x.destroy()
    tao.destroy()

    finish_queue(x_opt, opt_flag, comm_queue, parent_can_read, user_specs)


def opt_runner(run_local_opt, user_specs, comm_queue, x0, f0, child_can_read, parent_can_read):
    try:
        run_local_opt(user_specs, comm_queue, x0, f0, child_can_read, parent_can_read)
    except Exception as e:
        comm_queue.put(ErrorMsg(e))
        parent_can_read.set()


# Callback functions and routines
def nlopt_callback_fun(x, grad, comm_queue, child_can_read, parent_can_read, user_specs):
    if user_specs["localopt_method"] in ["LD_MMA"]:
        x_recv, f_recv, grad_recv = put_set_wait_get(x, comm_queue, parent_can_read, child_can_read, user_specs)
        grad[:] = grad_recv
    else:
        assert user_specs["localopt_method"] in [
            "LN_SBPLX",
            "LN_BOBYQA",
            "LN_NEWUOA",
            "LN_COBYLA",
            "LN_NELDERMEAD",
            "LD_MMA",
        ]
        x_recv, f_recv = put_set_wait_get(x, comm_queue, parent_can_read, child_can_read, user_specs)

    return f_recv


def scipy_dfols_callback_fun(x, comm_queue, child_can_read, parent_can_read, user_specs):
    if user_specs["localopt_method"] in ["scipy_BFGS"]:
        (
            x_recv,
            f_x_recv,
            grad_recv,
        ) = put_set_wait_get(x, comm_queue, parent_can_read, child_can_read, user_specs)

        return f_x_recv, grad_recv

    (
        x_recv,
        f_x_recv,
    ) = put_set_wait_get(x, comm_queue, parent_can_read, child_can_read, user_specs)

    return f_x_recv


def tao_callback_fun_nm(tao, x, comm_queue, child_can_read, parent_can_read, user_specs):
    (
        x_recv,
        f_recv,
    ) = put_set_wait_get(x.array_r, comm_queue, parent_can_read, child_can_read, user_specs)

    return f_recv


def tao_callback_fun_pounders(tao, x, f, comm_queue, child_can_read, parent_can_read, user_specs):
    (
        x_recv,
        f_recv,
    ) = put_set_wait_get(x.array_r, comm_queue, parent_can_read, child_can_read, user_specs)
    f.array[:] = f_recv

    return f


def tao_callback_fun_grad(tao, x, g, comm_queue, child_can_read, parent_can_read, user_specs):
    x_recv, f_recv, grad_recv = put_set_wait_get(x.array_r, comm_queue, parent_can_read, child_can_read, user_specs)
    g.array[:] = grad_recv

    return f_recv


def finish_queue(x_opt, opt_flag, comm_queue, parent_can_read, user_specs):
    if user_specs.get("print") and opt_flag:
        print("[APOSMM] Local optimum on the [0,1]^n domain", x_opt, flush=True)
    comm_queue.put(ConvergedMsg(x_opt, opt_flag))
    parent_can_read.set()


def put_set_wait_get(x, comm_queue, parent_can_read, child_can_read, user_specs):
    """This routine is used by children process callback functions. It:
    - puts x into a comm_queue,
    - tells the parent it can read,
    - tells the child to wait
    - receives the values put in the comm_queue by the parent
    - checks that the first value received matches x
    - removes the wait on the child
    - returns values"""

    comm_queue.put(x)
    # print('[Child]: I just put x_on_cube =', x.array, flush=True)
    # print('[Child]: Parent should no longer wait.', flush=True)
    parent_can_read.set()
    # print('[Child]: I have started waiting', flush=True)
    child_can_read.wait()
    # print('[Child]: Wohooo.. I am free folks', flush=True)
    values = comm_queue.get()
    child_can_read.clear()

    if user_specs.get("periodic"):
        assert np.allclose(x % 1, values[0] % 1, rtol=1e-15, atol=1e-15), "The point I gave is not the point I got back"
    else:
        assert np.allclose(x, values[0], rtol=1e-15, atol=1e-15), "The point I gave is not the point I got back"

    return values


def simulate_recv_from_manager(local_H, gen_specs):
    # This function goes through any entries of local_H and if they have not
    # "returned", then it performs all function/gradient evaluations and makes
    # output as if the calculations were performed externally by libEnsemble.
    user = gen_specs["user"]["standalone"]

    if np.sum(local_H["sim_ended"]) >= user["eval_max"]:
        return STOP_TAG, {}, {}

    H_rows = np.where(~local_H["sim_ended"])[0]
    H_fields = [i[0] for i in gen_specs["out"]]

    Work = {"libE_info": {"H_rows": H_rows}, "H_fields": H_fields}

    calc_in = np.zeros(len(H_rows), dtype=gen_specs["out"] + [("f", float), ("grad", float, len(local_H["x"][0]))])

    for name in H_fields:
        calc_in[name] = local_H[name][H_rows]

    assert "obj_func" in user or "obj_and_grad_func" in user, "Must have some way to calculate objective values"

    if "obj_func" in user:
        for i, row in enumerate(H_rows):
            calc_in["f"][i] = user["obj_func"](local_H["x"][row])

        if "grad" in local_H.dtype.names:
            for i, row in enumerate(H_rows):
                calc_in["grad"][i] = user["grad_func"](local_H["x"][row])
    else:
        for i, row in enumerate(H_rows):
            out = user["obj_and_grad_func"](local_H["x"][row])
            calc_in["f"][i] = out[0]
            calc_in["grad"][i] = out[1]

    return EVAL_GEN_TAG, Work, calc_in
