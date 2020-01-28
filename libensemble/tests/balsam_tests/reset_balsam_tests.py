#!/usr/bin/env python

import balsam.launcher.dag as dag

dag.BalsamTask.objects.filter(name__contains='outfile').delete()

for task in dag.BalsamTask.objects.filter(name__contains='task_test_balsam'):
    task.update_state('CREATED')
    task.save()
