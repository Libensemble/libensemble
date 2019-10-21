.. libEnsemble documentation master file, created by
   sphinx-quickstart on Fri Aug 18 11:52:31 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. image:: images/libE_logo.png
 :alt: libEnsemble


.. only::html
  |

  .. image:: https://img.shields.io/pypi/v/libensemble.svg?color=blue
    :target: https://pypi.org/project/libensemble

  .. image::  https://travis-ci.org/Libensemble/libensemble.svg?branch=master
    :target: https://travis-ci.org/Libensemble/libensemble

  .. image:: https://coveralls.io/repos/github/Libensemble/libensemble/badge/?maxAge=2592000/?branch=master
    :target: https://coveralls.io/github/Libensemble/libensemble?branch=master

  .. image::  https://readthedocs.org/projects/libensemble/badge/?maxAge=2592000
    :target: https://libensemble.readthedocs.org/en/latest/
    :alt: Documentation Status

  |

=======================================
Welcome to libEnsemble's documentation!
=======================================

libEnsemble is a library for managing ensemble-like collections of computations.


* New to libEnsemble? Start :doc:`here<quickstart>`.
* Try out libEnsemble with a :doc:`tutorial<tutorials/local_sine_tutorial>`.
* Go in-depth by reading the :doc:`User Guide<user_guide>`.
* Check the :doc:`FAQ<FAQ>` for common questions and answers, errors and resolutions.

.. only:: (latex or latexpdf)

  .. toctree::
     :maxdepth: 2

     Quickstart<quickstart>
     user_guide
     tutorials/local_sine_tutorial
     platforms/bebop
     platforms/theta
     contributing
     FAQ
     libE_module
     data_structures/data_structures
     user_funcs
     job_controller/jc_index
     logging
     dev_guide/release_management/release_index.rst
     dev_guide/dev_API/developer_API.rst
     release_notes


.. only:: html

  .. toctree::
     :maxdepth: 2
     :caption: Getting Started:

     Quickstart<quickstart>
     contributing
     release_notes
     FAQ


  .. toctree::
     :maxdepth: 1
     :caption: Tutorials:

     tutorials/local_sine_tutorial


  .. toctree::
     :maxdepth: 2
     :caption: Running on:

     platforms/bebop
     platforms/theta


  .. toctree::
     :maxdepth: 2
     :caption: User Guide:

     user_guide
     libE_module
     data_structures/data_structures
     user_funcs
     job_controller/jc_index
     logging


  .. toctree::
     :maxdepth: 2
     :caption: Developer Guide:

     dev_guide/release_management/release_index.rst
     dev_guide/dev_API/developer_API.rst




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
