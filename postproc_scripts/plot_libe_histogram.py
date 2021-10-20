#!/usr/bin/env python

# infile='testfile.txt'
infile = 'libE_stats.txt'
time_key = 'Time:'
status_key = 'Status:'
sim_only = True  # Ignore generator times
num_bins = 40

# States - could add multiple lines - eg Failed.
ran_ok = ['Completed']  # list of ok states

# run_killed = ['killed', 'Exception'] # Currently searches for this word in string
run_killed = ['killed']  # Currently searches for this word in string
run_exception = ['Exception', 'Failed']

import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# import csv


def search_for_keyword(in_list, kw_list):
    for i, val in enumerate(in_list):
        if val.endswith(':'):
            break  # New key word found
        else:
            if val in kw_list:
                return True
    return False


def append_to_list(mylst, glob_list, found_time):
    # Assumes Time comes first - else have to modify
    if found_time:
        mylst.append(glob_list[-1])
    else:
        print('Error Status found before time - exiting')
        sys.exit()


active_line_count = 0
in_times = []
in_times_ran = []
in_times_kill = []
in_times_exception = []
exceptions = False


# Read straight from libEnsemble summary file.
with open(infile) as f:
    for line in f:
        lst = line.split()
        found_time = False
        found_status = False
        for i, val in enumerate(lst):
            if val == time_key:
                if sim_only and lst[i-1] != 'sim':
                    break
                in_times.append(lst[i+1])
                found_time = True
            if val == status_key:
                if lst[i+1] in ran_ok:
                    append_to_list(in_times_ran, in_times, found_time)  # Assumes Time comes first
                elif search_for_keyword(lst[i+1:len(lst)], run_killed):
                    append_to_list(in_times_kill, in_times, found_time)  # Assumes Time comes first
                elif search_for_keyword(lst[i+1:len(lst)], run_exception):
                    exceptions = True
                    append_to_list(in_times_exception, in_times, found_time)  # Assumes Time comes first
                else:
                    print('Error: Unknown status - rest of line: {}'.format(lst[i+1:len(lst)]))
                    sys.exit()
                found_status = True
            if found_time and found_status:
                active_line_count += 1
                break

print('Processed {} calcs'.format(active_line_count))

times = np.asarray(in_times, dtype=float)
times_ran = np.asarray(in_times_ran, dtype=float)
times_kill = np.asarray(in_times_kill, dtype=float)

binwidth = (times.max() - times.min()) / num_bins
bins = np.arange(min(times), max(times) + binwidth, binwidth)

# plt.hist(times, bins, histtype='bar', rwidth=0.8)
p1 = plt.hist(times_ran, bins, label='Completed')
p2 = plt.hist(times_kill, bins, label='Killed')

if exceptions:
    times_exc = np.asarray(in_times_exception, dtype=float)
    p3 = plt.hist(times_exc, bins, label='Except/Failed', color='C3')  # red

# plt.title('Theta Opal/libEnsemble Times: 127 Workers - sim_max 508')
if sim_only:
    calc = 'sim'
else:
    calc = 'calc'

titl = ('Histogram of ' + calc + ' times' + ' (' + str(active_line_count) + ' user calcs)' + str(num_bins) + ' bins')

plt.title(titl)
plt.xlabel('Calculation run time (sec)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(True)
plt.legend(loc='best', fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.show()
plt.savefig('hist_completed_v_killed.png', bbox='tight', transparent=True)
