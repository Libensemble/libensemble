import matplotlib
matplotlib.use('Agg')

from balsam.core import models
from matplotlib import pyplot as plt

times, utilization = models.utilization_report()
plt.step(times, utilization, where='post')
plt.savefig('balsam_util_v_time.png')

