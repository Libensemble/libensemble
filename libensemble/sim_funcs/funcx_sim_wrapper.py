from libensemble.message_numbers import EVAL_SIM_TAG, FINISHED_PERSISTENT_SIM_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.utils.runners import Runners


def persistent_funcx_submitter(H, persis_info, sim_specs, libE_info):
    """
    Submits batched work together as a funcX batch
    """

    ps = PersistentSupport(libE_info, EVAL_SIM_TAG)
    runners = Runners(sim_specs, {})
    # runner_fs = runners.make_runners()
    client = runners.funcx_client
    # session_id = client.session_task_group_id

    # Either start with a work item to process - or just start and wait for data
    if H.size > 0:
        tag = None
        Work = None
        calc_in = H
    else:
        tag, Work, calc_in = ps.recv()

    while tag not in [STOP_TAG, PERSIS_STOP]:
        if Work is not None:
            persis_info = Work.get("persis_info", persis_info)
            libE_info = Work.get("libE_info", libE_info)
        libE_info["comm"] = None

        batch = client.create_batch()

        from libensemble.tools import ForkablePdb

        ForkablePdb().set_trace()

        func_args = runners._truncate_args(calc_in, persis_info, sim_specs, libE_info, sim_specs["sim_f"])

        for entry in calc_in:
            batch.add(runners.funcx_simfid, sim_specs.get("funcx_endpoint"), args=func_args)

        H_o, persis_info = print("helo")
        tag, Work, calc_in = ps.send_recv(H_o)

    final_return = None

    return final_return, persis_info, FINISHED_PERSISTENT_SIM_TAG
