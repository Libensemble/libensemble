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
    print('\nNo. of thetas = {:d}'.format(cur_thetas.shape[0]))
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


def obviate_pend_thetas(model, cur_theta, data_status):
    assert cur_theta.shape[0] == data_status.shape[0]

    complete_theta_bool = np.any(data_status != 0, axis=1)

    fpred, _ = emulation_prediction(model, cur_theta[complete_theta_bool])
    fdraws = emulation_draws(model, cur_theta[~complete_theta_bool], options={'numsamples': 500})

    cur_chi2 = np.sum(fpred ** 2, axis=1)
    best_chi2 = np.min(cur_chi2)

    incomplete_draw_chi2 = np.zeros((fdraws.shape[0], fdraws.shape[2]))
    for k in np.arange(fdraws.shape[2]):
        incomplete_draw_chi2[:, k] = np.sum(fdraws[:, :, k] ** 2, axis=1)

    expect_improvement = np.full_like(complete_theta_bool, 0)
    # meaningful when there are pending thetas
    expect_improvement[~complete_theta_bool] = ((best_chi2 > incomplete_draw_chi2)*(best_chi2 - incomplete_draw_chi2)).mean(axis=1)
    r_obviate = np.full_like(expect_improvement, False)
    r_obviate[np.where(np.logical_and(expect_improvement < 10 ** (-9), ~complete_theta_bool))] = True
    r_obviate[np.argmin(cur_chi2)] = False

    # data_status[r_obviate, :] = -2  # assign cancellation
    return np.array(np.where(r_obviate))  #, data_status
