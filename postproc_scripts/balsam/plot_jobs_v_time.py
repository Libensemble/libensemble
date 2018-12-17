import matplotlib
matplotlib.use('Agg')

from balsam.core import models
from matplotlib import pyplot as plt

times, throughputs = models.throughput_report()
plt.step(times, throughputs, where='post')
plt.savefig('balsam_jobs_v_time.png')

