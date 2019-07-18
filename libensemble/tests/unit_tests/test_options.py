# Tests ==============
from libensemble.options import GlobalOptions as go
import ast


def test_init():
    opts = go({'comms': 'local'}, {'nprocesses': 4}, logging='info')
    assert opts


def test_to_str():
    assert ast.literal_eval(str(go.current_options())) == ast.literal_eval("{'comms': 'local', 'nprocesses': 4, 'logging': 'info'}")
    # Prevent errors based on elements becoming unordered when becoming string


def test_get_1():
    assert go.current_options().get() == {'comms': 'local', 'nprocesses': 4, 'logging': 'info'}


def test_get_2():
    assert go.current_options().get('comms') == 'local'


def test_get_3():
    assert go.current_options().get('comms', 'logging') == {'comms': 'local',
                                                            'logging': 'info'}


def test_set():
    go.current_options().set({'gen_f': 'urs'}, logging='debug')
    assert go.current_options().get('gen_f', 'logging') == {'gen_f': 'urs', 'logging': 'debug'}

# Can this even be tested on Travis? Need to rearrange parse_args() in any case
# Is this even something that Options should do?
# def test_parse_args():  # also tests get_libE_specs()
#     opts = go()
#     is_master = opts.parse_args()
#     assert is_master
#     assert opts.get_libE_specs()['comms'] == 'mpi'


def test_get_libE_specs():
    comm = 'FAKE COMM'
    opts = go(comms='local', nprocesses=4, logging='info')
    assert opts.get_libE_specs() == {'comms': 'local', 'nprocesses': 4}
    opts = go(comms='mpi', comm=comm, color=0, logging='info')
    assert opts.get_libE_specs() == {'comms': 'mpi', 'comm': 'FAKE COMM', 'color': 0}


def test_to_from_file():
    filename = go.current_options().to_file()
    opts = go()
    opts.from_file(filename)
    assert str(opts.get())
    import os
    os.remove(filename)
    # TODO: reimplement exact testing


# def test_get_logger():
#     assert go.current_options().get_logger()
# The logger is already tested by test_logger.py in /unit_tests_logger


# def test_set_logging():  # Might need dedicated regression test?
#     pass


if __name__ == '__main__':
    test_init()
    test_to_str()
    test_get_1()
    test_get_2()
    test_get_3()
    test_set()
    test_get_libE_specs()
    # test_parse_args()
    test_to_from_file()
