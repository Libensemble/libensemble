#!/usr/bin/env python

#infile='testfile.txt'
infile='libE_stats.txt'
time_key='Time:'
status_key='Status:'
sim_only = True # Ignore generator times

#States - could add multiple lines - eg Failed.
ran_ok = ['Completed'] # list of ok states
run_killed = ['killed'] # Currently searches for this word in string

import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
#import csv

#todo: Make general search for any word from keyword list in a list.
def search_for_killed(sublist):
    for i, val in enumerate(sublist):
        if val.endswith(':'):
            break # New key word found
        else:
            if val in run_killed:
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
in_times=[]
in_times_ran=[]
in_times_kill=[]

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
                    append_to_list(in_times_ran,in_times,found_time) # Assumes Time comes first
                elif search_for_killed(lst[i+1:len(lst)]):
                    append_to_list(in_times_kill,in_times,found_time) # Assumes Time comes first
                else:
                    print('Error: Unkown status - rest of line: {}'.format(lst[i+1:len(lst)]))
                    sys.exit()
                found_status = True
            if found_time and found_status:
                active_line_count += 1
                break

# Read from modified csv file.
#with open('histo.csv', newline='') as csvfile:
    #cin = csv.reader(csvfile, delimiter=',')
    #for row in cin:
        #in_times.append(row[0])
        #if row[1] == 'killed':
            #in_times_kill.append(row[0])
        #else:
            #in_times_ran.append(row[0])

print('Processed {} calcs'.format(active_line_count))

times = np.asarray(in_times, dtype=float)
times_ran = np.asarray(in_times_ran, dtype=float)
times_kill = np.asarray(in_times_kill, dtype=float)

num_bins = 40
binwidth = (times.max() - times.min()) / num_bins
bins=np.arange(min(times), max(times) + binwidth, binwidth)

#plt.hist(times, bins, histtype='bar', rwidth=0.8)
p1 = plt.hist(times_ran, bins, label='Completed')
p2 = plt.hist(times_kill, bins, label='Killed')

#plt.title('Theta Opal/libEnsemble Times: 127 Workers - sim_max 508')
if sim_only:
    calc_type = 'sim'
else:
    calc_type = 'calc'
title = 'libEnsemble histogram of ' + calc_type  + ' times' + ' (' + str(active_line_count) + ' user calcs)'   

plt.title(title)
plt.xlabel('Calc run-time (sec)')
plt.ylabel('Count')
plt.grid(True)
plt.legend(loc='upper left')

#plt.show()
plt.savefig('hist_completed_v_killed.png')
