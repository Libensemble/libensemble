# Tests ==============
from libensemble.options import GlobalOptions as go


def test_init():
    opts = go({'comm': 'mpi'}, {'nworkers': 4}, logging='info')
    assert opts


def test_to_str():
    assert str(go.current_options()) == "{'comm': 'mpi', 'nworkers': 4, 'logging': 'info'}"


def test_get_1():
    assert go.current_options().get() == {'comm': 'mpi', 'nworkers': 4, 'logging': 'info'}


def test_get_2():
    assert go.current_options().get('comm') == 'mpi'


def test_get_3():
    assert go.current_options().get('comm', 'logging') == {'comm': 'mpi',
                                                           'logging': 'info'}


def test_set():
    go.current_options().set({'gen_f': 'urs'}, logging='debug')
    assert go.current_options().get('gen_f', 'logging') == {'gen_f': 'urs', 'logging': 'debug'}


def test_parse_args():  # also tests get_libE_specs()
    opts = go()
    is_master = opts.parse_args()
    assert is_master
    assert opts.get_libE_specs()['comms'] == 'mpi'


def test_to_from_file():
    filename = go.current_options().to_file()
    with open(filename, 'r') as f:
        lines = f.readline()
        options_no_comm = go.current_options().get().copy()
        options_no_comm.pop('comm')
        assert lines == str(options_no_comm)
        f.close()
    opts = go()
    opts.from_file(filename)
    assert str(opts.get()) == lines

    import os
    os.remove(filename)


def test_current_options():
    opts = go()
    assert opts is go.current_options()


def test_get_logger():
    assert go.current_options().get_logger()
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
    test_parse_args()
    test_to_from_file()
    test_current_options()
