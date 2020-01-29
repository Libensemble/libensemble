#!/usr/bin/env python

import balsam.launcher.dag as dag

dag.BalsamTask.objects.filter(name__contains='outfile').delete()

for job in dag.BalsamTask.objects.filter(name__contains='job_test_balsam'):
    job.update_state('CREATED')
    job.save()
