# Tests ==============
from libensemble.options import GlobalOptions as go


def test_init():
    topt = go({'comms': 'mpi'}, {'nworkers': 4}, logging='info')
    assert topt


def test_to_str():
    assert str(go.get_current_options()) == "{'comms': 'mpi', 'nworkers': 4, 'logging': 'info'}"


def test_get_1():
    assert go.get_current_options().get() == {'comms': 'mpi', 'nworkers': 4, 'logging': 'info'}


def test_get_2():
    assert go.get_current_options().get('comms') == 'mpi'


def test_get_3():
    assert go.get_current_options().get('comms', 'logging') == {'comms': 'mpi',
                                                                'logging': 'info'}


def test_set():
    go.get_current_options().set({'gen_f': 'urs'}, logging='debug')
    assert go.get_current_options().get('gen_f', 'logging') == {'gen_f': 'urs', 'logging': 'debug'}


def test_parse_args():  # also tests get_libE_specs()
    opts = go()
    is_master = opts.parse_args()
    assert is_master
    assert opts.get_libE_specs()['comms'] == 'mpi'


def test_get_logger():
    assert go.get_current_options().get_logger()
    # The logger is already tested by test_logger.py in /unit_tests_logger


def test_set_logging():  # TODO: Might need dedicated regression test?
    pass


def test_to_file():  # TODO: What format will this be?
    file = go.get_current_options().to_file()
    with open(file, 'r') as f:
        assert f.readline() == str(go.get_current_options().get())


if __name__ == '__main__':
    test_init()
    test_to_str()
    test_get_1()
    test_get_2()
    test_get_3()
    test_set()
    test_parse_args()
    test_get_logger()
    test_set_logging()
    test_to_file()
