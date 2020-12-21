## Distributed ML Example with Horovod and libEnsemble

Requires Tensorflow, Keras, and Horovod.

This test launches a Python Machine Learning training routine as a ``gen_f``
application through the Executor, then evaluates the resulting model in a ``sim_f``
using test data. The number of training epochs, MPI processes, arguments for the
Tensorflow application, and other parameters can be easily adjusted.

### Running with libEnsemble

Like other multiprocessing libEnsemble routines:

    python run_distrib_mnist.py --comms local --nworkers 4
