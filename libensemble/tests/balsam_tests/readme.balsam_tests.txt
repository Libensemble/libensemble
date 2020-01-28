List of balsam test tasks for libEnsemble.

test_balsam_1__runtasks.py:
Launches parallel parent job. Each task iterates over a loop launching sub-tasks
and waiting for completion. Does 3 iterations by default.

test_balsam_2__workerkill.py:
As first test but each top-level task kills its middle job.

test_balsam_3__managerkill.py:
Process 0 sends out a kill for tasks which include sim_id_1 in their name. This
will kill any such tasks in the database at the time of the kill. Note that if a
job has not yet been added to database at the time of the kill, it will still
run.

Note that test3 exploits the fact that balsam is built on a Django database,
and all Django functionality for manipulating the database can easily be
exposed.
