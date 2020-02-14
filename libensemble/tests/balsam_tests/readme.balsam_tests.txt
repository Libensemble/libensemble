List of balsam test jobs for libEnsemble.

test_balsam_1__runjobs.py:
Launches parallel parent job. Each job iterates over a loop launching sub-jobs
and waiting for completion. Does 3 iterations by default.

test_balsam_2__workerkill.py:
As first test but each top-level jobs kills its middle job.

test_balsam_3__managerkill.py:
Process 0 sends out a kill for jobs which include sim_id_1 in their name. This
will kill any such jobs in the database at the time of the kill. Note that if a
job has not yet been added to database at the time of the kill, it will still
run.

Note that test3 exploits the fact that balsam is built on a Django database,
and all Django functionality for manipulating the database can easily be
exposed.
