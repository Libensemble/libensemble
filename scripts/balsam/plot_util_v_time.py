import matplotlib

matplotlib.use("Agg")

from balsam.core import models
from matplotlib import pyplot as plt

fig, ax = plt.subplots()

plt.title("Balsam Utilization: Running Tasks v Date/Time")
plt.xlabel("Time of Day (H:M)")
plt.ylabel("Num. Tasks Running")

times, utilization = models.utilization_report()
plt.step(times, utilization, where="post")

import matplotlib.dates as mdates

myFmt = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(myFmt)
fig.autofmt_xdate()

plt.savefig("balsam_util_v_time.png")
