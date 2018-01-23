#!/usr/bin/env python
import sys, os             # for adding to path

import balsam.launcher.dag as dag

dag.add_job(name = "helloworld", workflow = "libe_workflow", application="helloworld", num_nodes=1, ranks_per_node=8)

print ("done")
