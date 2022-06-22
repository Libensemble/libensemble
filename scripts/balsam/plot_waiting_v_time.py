import matplotlib

matplotlib.use("Agg")

from balsam.core import models
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()

plt.title("Balsam: Tasks in waiting state v Date/Time")
plt.xlabel("Time of Day (H:M)")
plt.ylabel("Num. Tasks Waiting")

times, waiting = models.job_waiting_report()
plt.step(times, waiting, where="post")

myFmt = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_formatter(myFmt)
fig.autofmt_xdate()

plt.savefig("balsam_waiting_v_time.png")
