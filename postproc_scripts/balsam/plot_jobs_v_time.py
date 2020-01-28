import matplotlib
matplotlib.use('Agg')

from balsam.core import models
from matplotlib import pyplot as plt

fig, ax = plt.subplots()

plt.title('Balsam: Jobs completed v Time')
plt.xlabel('Time of Day (H:M)')
plt.ylabel('Num. Jobs Completed (Accum)')

times, throughputs = models.throughput_report()
plt.step(times, throughputs, where='post')

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(myFmt)
fig.autofmt_xdate()

plt.savefig('balsam_tasks_v_time.png')
