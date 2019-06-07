# Tests ==============
from libensemble.options import Options


def test_init():
    topt = Options({'comms': 'mpi'}, {'factor': 2}, logging='info')
    assert topt


def test_to_str():
    topt = Options(logging='info')

    assert str(topt) == "{'logging': 'info'}"


def test_get_options_1():
    topt = Options({'comms': 'mpi'}, {'factor': 2}, logging='info')

    assert topt.get_options() == {'comms': 'mpi', 'factor': 2, 'logging': 'info'}


def test_get_options_2():
    topt1 = Options({'comms': 'mpi'})
    topt2 = Options(comms='mpi')

    assert topt1.get_options('comms') == topt2.get_options('comms') == 'mpi'


def test_get_options_3():
    topt1 = Options({'comms': 'mpi'}, {'nworkers': 4}, logging='info')

    assert topt1.get_options('comms', 'logging') == {'comms': 'mpi',
                                                     'logging': 'info'}


def test_set_logging():
    pass


def test_set_option_1():
    pass


def as_libE_specs():
    pass


def to_python():
    pass


if __name__ == '__main__':
    test_init()
    test_to_str()
    test_get_options_1()
    test_get_options_2()
    test_get_options_3()
    print('Passed')
