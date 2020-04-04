import os
import numpy as np
import yt ; yt.funcs.mylog.setLevel(50)
import scipy.constants as scc

def read_sim_output( workdir ):
    # Get initial charge
    datafile = 'diags/plotfiles/plt00000/'
    filepath = os.path.join(workdir, datafile)
    ds = yt.load( filepath )
    ad = ds.all_data()
    w = ad['beam', 'particle_weight'].v
    charge_i = np.sum(w)

    # Get properties of beam at the end
    datafile = 'diags/plotfiles/plt01830/'
    filepath = os.path.join(workdir, datafile)
    ds = yt.load( filepath )
    ad = ds.all_data()
    w = ad['beam', 'particle_weight'].v
    ux = ad['beam', 'particle_momentum_x'].v/scc.m_e/scc.c
    uy = ad['beam', 'particle_momentum_y'].v/scc.m_e/scc.c
    uz = ad['beam', 'particle_momentum_z'].v/scc.m_e/scc.c
    gamma = np.sqrt( 1. + ux**2 + uy**2 + uz**2 )
    energy_MeV = .511 * (gamma - 1.)
    energy_avg = np.mean( energy_MeV )
    energy_std = np.std( energy_MeV ) / energy_avg
    charge = np.sum(w) / charge_i
    warpX_out = np.array([energy_std, energy_avg, charge])
    return warpX_out
