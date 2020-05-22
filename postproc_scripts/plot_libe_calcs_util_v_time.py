infile = 'libE_stats.txt'
import pandas as pd

run_stats = []
with open(infile) as f:
    # content = f.readlines()
    for line in f:
        lst = line.split()
        foundstart = False
        foundend = False
        for i, val in enumerate(lst):
            if val == 'Start:':
                startdate = lst[i+1]
                starttime = lst[i+2]
                foundstart = True
            if val == 'End:':
                enddate = lst[i+1]
                endtime = lst[i+2]
                foundend = True
            if foundstart and foundend:
                run_datetime = {'start': startdate + ' ' + starttime, 'end': enddate + ' ' + endtime}
                run_stats.append(run_datetime)
                break

df = pd.DataFrame(run_stats)
df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])
df = df.sort_values(by='start')

time_start = df['start'][0]
dend = df.sort_values(by='end')
time_end = dend['end'].iloc[-1]
date_series = pd.date_range(time_start, time_end, freq='1S')

counts = []
for i in range(len(date_series)):
    # Inclusive/exclusive to avoid multiple accounting - need high resolution
    count = sum((df['start'] <= date_series[i]) & (date_series[i] < df['end']))
    counts.append(count)

df_list = pd.DataFrame(date_series, columns=['datetime'])

# df_count = pd.DataFrame([counts], columns=['count']) # List goes to single row by default
df_count = pd.DataFrame({'count': counts})  # Transpose to columns like this


final = df_list.join(df_count)
# print(final)

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
final.plot(x='datetime', y='count', legend=None, linewidth=2, fontsize=12)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Active calculations', fontsize=14)
# plt._show()
plt.savefig('calcs_util_v_time.png', bbox='tight', transparent=True)
