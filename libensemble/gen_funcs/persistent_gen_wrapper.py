import inspect

from libensemble.message_numbers import EVAL_GEN_TAG, FINISHED_PERSISTENT_GEN_TAG, PERSIS_STOP, STOP_TAG
from libensemble.tools.persistent_support import PersistentSupport
from libensemble.utils.misc import list_dicts_to_np, np_to_list_dicts


def persistent_gen_f(H, persis_info, gen_specs, libE_info):

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    U = gen_specs["user"]
    b = U.get("initial_batch_size") or U.get("batch_size")

    generator = U["generator"]
    if inspect.isclass(generator):
        gen = generator(H, persis_info, gen_specs, libE_info)
    else:
        gen = generator

    tag = None
    calc_in = None
    while tag not in [STOP_TAG, PERSIS_STOP]:
        H_o = gen.ask(b)
        if isinstance(H_o, list):
            H_o = list_dicts_to_np(H_o)
        tag, Work, calc_in = ps.send_recv(H_o)
        gen.tell(np_to_list_dicts(calc_in))

        if hasattr(calc_in, "__len__"):
            b = len(calc_in)

    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG