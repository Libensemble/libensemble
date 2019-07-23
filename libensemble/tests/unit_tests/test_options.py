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
    assert opts.get() == {'comms': 'mpi', 'color': 0, 'logging': 'info'}
    import os
    os.remove(filename)
    TODO: reimplement exact testing



if __name__ == '__main__':
    test_init()
    test_to_str()
    test_get_1()
    test_get_2()
    test_get_3()
    test_set()
    test_get_libE_specs()
    test_to_from_file()
