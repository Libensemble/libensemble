from libensemble.message_numbers import WORKER_DONE
from libensemble.executors.executor import Executor
import numpy as np


def exp_nodelist_for_worker(exp_list, workerID, nodes_per_worker, persis_gens):
    """Modify expected node-lists based on workerID"""
    comps = exp_list.split()
    new_line = []
    for comp in comps:
        if comp.startswith('node-'):
            new_node_list = []
            node_list = comp.split(',')
            for node in node_list:
                node_name, node_num = node.split('-')
                offset = workerID - (1 + persis_gens)
                new_num = int(node_num) + int(nodes_per_worker*offset)
                new_node = '-'.join([node_name, str(new_num)])
                new_node_list.append(new_node)
            new_list = ','.join(new_node_list)
            new_line.append(new_list)
        else:
            new_line.append(comp)
    return ' '.join(new_line)


def runline_check(H, persis_info, sim_specs, libE_info):
    """Check run-lines produced by executor provided by a list"""
    calc_status = 0
    x = H['x'][0][0]
    exctr = Executor.executor
    test_list = sim_specs['user']['tests']
    exp_list = sim_specs['user']['expect']
    npw = sim_specs['user']['nodes_per_worker']
    p_gens = sim_specs['user'].get('persis_gens', 0)

    for i, test in enumerate(test_list):
        task = exctr.submit(calc_type='sim',
                            num_procs=test.get('nprocs', None),
                            num_nodes=test.get('nnodes', None),
                            ranks_per_node=test.get('ppn', None),
                            extra_args=test.get('e_args', None),
                            app_args='--testid ' + test.get('testid', None),
                            stdout='out.txt',
                            stderr='err.txt',
                            hyperthreads=test.get('ht', None),
                            dry_run=True)

        outline = task.runline
        new_exp_list = exp_nodelist_for_worker(exp_list[i], libE_info['workerID'], npw, p_gens)

        if outline != new_exp_list:
            print('outline is: {}\nexp     is: {}'.format(outline, new_exp_list), flush=True)

        assert(outline == new_exp_list)

    calc_status = WORKER_DONE
    output = np.zeros(1, dtype=sim_specs['out'])
    output['f'][0] = np.linalg.norm(x)
    return output, persis_info, calc_status
