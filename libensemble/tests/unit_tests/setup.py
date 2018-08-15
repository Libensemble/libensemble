import numpy as np

def make_criteria_and_specs_0(simx=10):
    sim_specs={'sim_f': np.linalg.norm, 'in':['x_on_cube'], 'out':[('f',float),('fvec',float,3)], }
    gen_specs={'gen_f': np.random.uniform, 'in':[], 'out':[('x_on_cube',float),('priority',float),('local_pt',bool)], 'ub':np.ones(1), 'nu':0}
    exit_criteria={'sim_max':simx}

    return sim_specs, gen_specs, exit_criteria

def make_criteria_and_specs_1(simx=10):
    sim_specs={'sim_f': np.linalg.norm, 'in':['x'], 'out':[('g',float)], }
    gen_specs={'gen_f': np.random.uniform, 'in':[], 'out':[('x',float),('priority',float)], }
    exit_criteria={'sim_max':simx, 'stop_val':('g',-1), 'elapsed_wallclock_time':0.5}

    return sim_specs, gen_specs, exit_criteria

def make_criteria_and_specs_1A(simx=10):
    sim_specs={'sim_f': np.linalg.norm, 'in':['x'], 'out':[('g',float)], }
    gen_specs={'gen_f': np.random.uniform, 'in':[], 'out':[('x',float),('priority',float),('sim_id',int)], }
    exit_criteria={'sim_max':simx, 'stop_val':('g',-1), 'elapsed_wallclock_time':0.5}

    return sim_specs, gen_specs, exit_criteria
