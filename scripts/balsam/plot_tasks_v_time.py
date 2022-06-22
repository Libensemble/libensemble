import matplotlib

matplotlib.use("Agg")

from balsam.core import models
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()

plt.title("Balsam: Tasks completed v Time")
plt.xlabel("Time of Day (H:M)")
plt.ylabel("Num. Tasks Completed (Accum)")

times, throughputs = models.throughput_report()
plt.step(times, throughputs, where="post")

myFmt = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(myFmt)
fig.autofmt_xdate()

plt.savefig("balsam_tasks_v_time.png")
