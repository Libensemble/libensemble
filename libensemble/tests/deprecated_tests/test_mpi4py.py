import pytest


@pytest.mark.extra
def test_mpi4py():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    assert size == 1
    assert rank == 0
    assert len(name)
