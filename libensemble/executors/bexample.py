from balsam.api import ApplicationDefinition, BatchJob, Job
import time

class VecNorm(ApplicationDefinition):
    site = "one"

    def run(self, vec):
        return sum(x**2 for x in vec)**0.5

job = VecNorm.submit(workdir="test/1", vec=[3, 4])

batchjob = BatchJob.objects.create(
    site_id=job.site_id,
    num_nodes=1,
    wall_time_min=10,
    job_mode="mpi",
    queue="local",
    project="local",
)

import ipdb; ipdb.set_trace()



print('hello')
print(job.result())

for job in Job.objects.as_completed(jobs):
    print(job.workdir, job.result())
