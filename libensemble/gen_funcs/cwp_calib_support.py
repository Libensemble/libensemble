"""Contains parameter selection and obviation methods for cwpCalibration."""

import numpy as np
from gemulator.emulation import emulation_prediction, emulation_builder, emulation_draws


def gen_thetas(n, persis_info):
    """Generates and returns n parameters for the Borehole function, according to distributions
    outlined in Harper and Gupta (1983).

    input:
      n: number of parameter to generate
    output:
      matrix of (n, 6), input as to borehole_func(t, x) function
    """

    randstream = persis_info['rand_stream']

    Tu = randstream.uniform(63070, 115600, n)
    Tl = randstream.uniform(63.1, 116, n)
    Hu = randstream.uniform(990, 1110, n)
    Hl = randstream.uniform(700, 820, n)
    r = randstream.lognormal(7.71, 1.0056, n)
    Kw = randstream.uniform(9855, 12045, n)
    thetas = np.column_stack((Tu, Tl, Hu, Hl, r, Kw))
    return thetas, persis_info

def gen_new_thetas(n, persis_info):
    thetas, _ = gen_thetas(n, persis_info)
    return thetas, persis_info


def select_next_theta(model, cur_thetas, n_explore_theta, expect_impr_exit, persis_info):   # !!! add step_add_theta
    new_thetas, persis_info = gen_new_thetas(n_explore_theta, persis_info)

    fpred, _ = emulation_prediction(model, cur_thetas)

    fdraws = emulation_draws(model, new_thetas, options={'numsamples': 500})

    cur_chi2 = np.sum(fpred ** 2, axis=1)
    best_chi2 = np.min(cur_chi2)

    new_chi2 = np.zeros((fdraws.shape[0], fdraws.shape[2]))
    for k in np.arange(fdraws.shape[2]):
        new_chi2[:, k] = np.sum(fdraws[:, :, k] ** 2, axis=1)

    expect_improvement = ((best_chi2 > new_chi2)*(best_chi2 - new_chi2)).mean(axis=1)
    # prob_improvement = (best_chi2 > new_chi2).mean(axis=1)
    print('No. of thetas = {:d}'.format(cur_thetas.shape[0]))
    print('MAX EI = {:.2f}'.format(np.max(expect_improvement)))
    print('Best chi^2 = {:.2f}'.format(best_chi2))

    if np.max(expect_improvement) < 0.001 * fpred.shape[1]:  # > 0.95:  tolerance?
        stop_flag = True
        new_theta = None
    else:
        stop_flag = False
        new_theta = np.copy(new_thetas[np.argmax(expect_improvement)])
        # print(new_theta)
        # print('new chi2: {:.2f}'.format(np.min(new_chi2)))

    return np.atleast_2d(new_theta), stop_flag, persis_info