# Tests ==============
from libensemble.options import Options
# from libensemble.tests.regression_tests.common import parse_args
# from mpi4py import MPI
# Will write tests for these


def test_init():
    topt = Options({'comms': 'mpi'}, {'factor': 2}, logging='info')
    assert topt


def test_to_str():
    topt = Options(logging='info')

    assert str(topt) == "{'logging': 'info'}"


def test_get_1():
    topt = Options({'comms': 'mpi'}, {'factor': 2}, logging='info')

    assert topt.get() == {'comms': 'mpi', 'factor': 2, 'logging': 'info'}


def test_get_2():
    topt1 = Options({'comms': 'mpi'})
    topt2 = Options(comms='mpi')

    assert topt1.get('comms') == topt2.get('comms') == 'mpi'


def test_get_3():
    topt1 = Options({'comms': 'mpi'}, {'nworkers': 4}, logging='info')

    assert topt1.get('comms', 'logging') == {'comms': 'mpi',
                                             'logging': 'info'}


def test_set_1():
    topt = Options()
    topt.set({'comms': 'mpi'}, logging='info')
    assert topt.get() == {'comms': 'mpi', 'logging': 'info'}


def test_get_libE_specs():
    pass


def test_set_logging():  # TODO: What logging? Normal logging or libEnsemble logging?
    pass


def test_local_parse_args():
    pass


def test_mpi_parse_args():
    pass


def test_to_output():  # TODO: What format will this be?
    pass


# TODO: Begin theorizing MPI testing for Options class

if __name__ == '__main__':
    test_init()         # Maybe these basic tests aren't necessary
    test_to_str()
    test_get_1()
    test_get_2()
    test_get_3()
    test_set_1()
