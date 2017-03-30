from __future__ import division
from __future__ import absolute_import

import numpy as np
import scipy as sp
from scipy import spatial
import time 

import sys
class PriorityQueue():
    """
    Class to help the manager manage the queue

    Parameters
    ----------
    pt_id: numpy array
        Vector of pt_ids 
    priority_levels: numpy array
        Vector of priority levels for the pt_ids
    run_numbers: numpy array
        Vector with integers telling which run is associated with the point
    """

    def __init__(self, pt_ids, priority_levels, run_numbers):
        self.pt_ids = np.atleast_1d(pt_ids[:])
        self.priority_levels = np.atleast_1d(priority_levels[:])
        self.run_numbers = np.atleast_1d(run_numbers[:])

    def __len__(self):
        return len(self.pt_ids)

    def __cleanup__(self, inds):
        run_number = np.atleast_1d(self.run_numbers[inds])[0]
        pt_ids = np.atleast_1d(self.pt_ids[inds,:].squeeze())

        self.priority_levels = np.delete(self.priority_levels,inds)
        self.run_numbers = np.delete(self.run_numbers,inds)
        self.pt_ids = np.delete(self.pt_ids,inds,0) 
        return (pt_ids, run_number)

    def add_to_queue(self, new_pt_ids, new_priority_levels, new_run_numbers):
        new_pt_ids = np.atleast_2d(new_pt_ids)
        new_priority_levels = np.atleast_1d(new_priority_levels)
        new_run_numbers = np.atleast_1d(new_run_numbers)

        if not isinstance(new_pt_ids,np.ndarray):
            raise TypeError("Unknown Data Type")

        assert new_pt_ids.shape[0] == len(new_priority_levels),\
                "Not equal number of new pt_ids and new levels"

        assert new_pt_ids.shape[0] == len(new_run_numbers),\
                "Not equal number of new pt_ids and new run numbers"

        if len(self) == 0:
            self.pt_ids = new_pt_ids[:]
            self.priority_levels = new_priority_levels[:]
            self.run_numbers = new_run_numbers[:]
        else:

            self.pt_ids = np.vstack((self.pt_ids,new_pt_ids))
            self.priority_levels = np.append(self.priority_levels,new_priority_levels)
            self.run_numbers = np.append(self.run_numbers,new_run_numbers)

    def get_all_from_highest_priority_run(self):
        """Return all pt_ids from the highest priority run """

        if len(self) == 0:
            return (None,-1,0,0)
        else:
            max_run_ind = self.priority_levels.argmax()
            return self.__cleanup__(np.where(self.run_numbers==self.run_numbers[max_run_ind]))

    def get_item_i(self,i):
        """Remove the ith queue point and return it """

        if len(self) == 0:
            return (None,-1,0,0)
        elif i >= len(self.pt_ids):
            raise IndexError("No point i in queue")
        else:
            return self.__cleanup__(i)

    def get_all_from_first_run(self):
        """Return the point in the queue with the smallest run_number"""

        if len(self) == 0:
            return (None,-1,0,0)
        else:
            min_run = self.run_numbers.min()
            return self.__cleanup__(np.where(self.run_numbers==min_run))

    def get_all_from_first_run_with_no_point_within_rk_of_a_min(self, mins, r_k, H):

        if len(self) == 0:
            return (None,-1,0,0)
        else:
            if mins.shape[0]==0:
                return self.__cleanup__(np.where(self.run_numbers==self.run_numbers.min()))
            else:
                sorted_run_nums = np.unique(self.run_numbers)
                for run_num in sorted_run_nums:
                    pts = self.pt_ids[self.run_numbers==run_num]
                    pdist = sp.spatial.distance.cdist(pts, mins, 'euclidean')
                    if np.min(pdist) > r_k:
                        return self.__cleanup__(np.where(self.run_numbers==run_num))
                    else:
                        continue
            return (None,-1,0,0)


    def get_all_from_random_run(self):
        """Return the queue pt_ids from a random run """

        if len(self) == 0:
            return (None,-1,0,0)
        elif isinstance(self.pt_ids,np.ndarray):
            rand_run = np.random.choice(self.run_numbers)
            return self.__cleanup__(np.where(self.run_numbers==rand_run))

    # @profile
    def use_these_rules(self, rules, tie_break, H, LH, r_k, last_run_num_evaled):
        """Return the queue point that satisfies all rules and tie_break 

        Parameters
        ----------
        self: The priority queue instance
        rules: 9x1 bool vector
            True in entry i telling not to take a point out of the localopt queue...
                0: ... if next point is with r_k of any active localopt run point
                1: ... if next point is with r_k of any past localopt run point
                2: ... if next point is with r_k of any identified minima
                3: ... if best point is with r_k of any active localopt run point
                4: ... if best point is with r_k of any past localopt run point
                5: ... if best point is with r_k of any identified minima
                6: ... if best point is with r_k of a better active localopt run point
                7: ... if best point is with r_k of a better past localopt run point
                8: ... if best point is with r_k of a better identified minima
        tie_break: int
            0: uniformly random
            1: cycle through all runs
            2: lowest function value 
        H: numpy structured array
            History array
        r_k: scalar
            Critical distance
        last_run_num_evaled: int
            Last run evaluated (only needed for tie_break 1)
        """
        
        if len(self) == 0:
            return (None,-1,0,0)
        
        runs = np.unique(self.run_numbers).astype('int')
        possible_runs = np.ones(len(runs),dtype='bool')  

        if np.any(rules):
            # Set up the arrays
            active_within_rk_of_next = np.zeros(len(runs))
            past_local_within_rk_of_next = np.zeros(len(runs))
            min_within_rk_of_next = np.zeros(len(runs))

            active_within_rk_of_best = np.zeros(len(runs))
            past_local_within_rk_of_best = np.zeros(len(runs))
            min_within_rk_of_best = np.zeros(len(runs))

            better_active_within_rk_of_best = np.zeros(len(runs))
            better_past_local_within_rk_of_best = np.zeros(len(runs))
            better_min_within_rk_of_best = np.zeros(len(runs))

            # active_within_rk_of_next_2 = np.zeros(len(runs))
            # past_local_within_rk_of_next_2 = np.zeros(len(runs))
            # min_within_rk_of_next_2 = np.zeros(len(runs))

            # active_within_rk_of_best_2 = np.zeros(len(runs))
            # past_local_within_rk_of_best_2 = np.zeros(len(runs))
            # min_within_rk_of_best_2 = np.zeros(len(runs))

            # better_active_within_rk_of_best_2 = np.zeros(len(runs))
            # better_past_local_within_rk_of_best_2 = np.zeros(len(runs))
            # better_min_within_rk_of_best_2 = np.zeros(len(runs))


            best_ind = np.zeros(len(runs))
            for run_num,i in zip(runs,range(0,len(runs))):
                pdist_next = sp.spatial.distance.cdist(self.pt_ids[self.run_numbers==run_num], H['x'], 'euclidean')
                best_ind[i] = LH['run_order'][run_num][np.argmin(H['f'][LH['run_order'][run_num,LH['run_order'][run_num]>=0]])]
                pdist_best = sp.spatial.distance.cdist([H['x'][best_ind[i]]], H['x'], 'euclidean').flatten()


                # # For the next point(s) in the run is there any point...
                # active_within_rk_of_next[i] = np.any(np.logical_and.reduce((
                #                                 np.any(pdist_next <= r_k, axis=0), #... within r_k 
                #                                ~np.in1d(H['pt_id'],LH['run_order'][run_num]), # not from this run
                #                                 H['active'])))                      #... and active?
                # past_local_within_rk_of_next[i] = np.any(np.logical_and.reduce((
                #                                 np.any(pdist_next <= r_k, axis=0), #... within r_k 
                #                                ~np.in1d(H['pt_id'],LH['run_order'][run_num]), # not from this run
                #                                 H['local_pt'])))                    #... and a local point?
                # min_within_rk_of_next[i] = np.any(np.logical_and.reduce((
                #                                 np.any(pdist_next <= r_k, axis=0), #... within r_k 
                #                                 H['local_min'])))                   #... and a local min?

                # # For the best point in the run is there any point...
                # active_within_rk_of_best[i] = np.any(np.logical_and.reduce((         
                #                                 pdist_best <= r_k,            #... within r_k 
                #                                ~np.in1d(H['pt_id'],LH['run_order'][run_num]), # not from this run
                #                                 H['active'])))                 #... and active?
                # past_local_within_rk_of_best[i] = np.any(np.logical_and.reduce((                             
                #                                 pdist_best <= r_k,            #... within r_k 
                #                                ~np.in1d(H['pt_id'],LH['run_order'][run_num]), # not from this run
                #                                 H['local_pt'])))               #... and a local point?
                # min_within_rk_of_best[i] = np.any(np.logical_and.reduce((                                    
                #                                 pdist_best <= r_k,            #... within r_k 
                #                                 H['local_min'])))              #... and a local min?

                # # For the best point in the run is there any point...
                # better_active_within_rk_of_best[i] = np.any(np.logical_and.reduce((
                #                                 pdist_best <= r_k,                #... within r_k 
                #                                ~np.in1d(H['pt_id'],LH['run_order'][run_num]), # not from this run
                #                                 H['active'],                      #... and active
                #                                 H['f'] < H['f'][best_ind[i]])))       #... and better?          
                # better_past_local_within_rk_of_best[i] = np.any(np.logical_and.reduce((  
                #                                 pdist_best <= r_k,                #... within r_k 
                #                                ~np.in1d(H['pt_id'],LH['run_order'][run_num]), # not from this run
                #                                 H['local_pt'],                    #... and a local point
                #                                 H['f'] < H['f'][best_ind[i]])))       #... and better?                                 
                # better_min_within_rk_of_best[i] = np.any(np.logical_and.reduce((         
                #                                 pdist_best <= r_k,                #... within r_k 
                #                                 H['local_min'],                   #... and a local min
                #                                 H['f'] < H['f'][best_ind[i]])))       #... and better?


                if np.any(rules[[0,1,3,4,6,7]]):
                    not_from_this_run = ~np.in1d(H['pt_id'],LH['run_order'][run_num])

                if np.any(rules[0:3]):
                    next_within_rk = np.any(pdist_next <= r_k, axis=0)

                if np.any(rules[3:9]): 
                    best_within_rk = pdist_best <= r_k

                if np.any(rules[6:9]):
                    better = H['f'] < H['f'][best_ind[i]]

                # For the next point(s) in the run is there any point...
                if rules[0]:
                    active_within_rk_of_next[i] = np.any(np.logical_and.reduce((
                                                    next_within_rk,    #... within r_k 
                                                    not_from_this_run, # not from this run
                                                    H['active'])))     #... and active?
                if rules[1]:
                    past_local_within_rk_of_next[i] = np.any(np.logical_and.reduce((
                                                    next_within_rk,    #... within r_k 
                                                    not_from_this_run, # not from this run
                                                    H['local_pt'])))   #... and a local point?
                if rules[2]:
                    min_within_rk_of_next[i] = np.any(np.logical_and.reduce((
                                                    next_within_rk,   #... within r_k 
                                                    H['local_min']))) #... and a local min?

                # For the best point in the run is there any point...
                if rules[3]:
                    active_within_rk_of_best[i] = np.any(np.logical_and.reduce((         
                                                    best_within_rk,    #... within r_k 
                                                    not_from_this_run, #... not from this run
                                                    H['active'])))     #... and active?
                if rules[4]:
                    past_local_within_rk_of_best[i] = np.any(np.logical_and.reduce((                             
                                                    best_within_rk,    #... within r_k 
                                                    not_from_this_run, #... not from this run
                                                    H['local_pt'])))   #... and a local point?
                if rules[5]:
                    min_within_rk_of_best[i] = np.any(np.logical_and.reduce((                                    
                                                    best_within_rk,   #... within r_k 
                                                    H['local_min']))) #... and a local min?

                # For the best point in the run is there any point...
                if rules[6]:
                    better_active_within_rk_of_best[i] = np.any(np.logical_and.reduce((
                                                    best_within_rk,    #... within r_k 
                                                    not_from_this_run, #... not from this run
                                                    H['active'],       #... and active
                                                    better)))          #... and better?          
                if rules[7]:
                    better_past_local_within_rk_of_best[i] = np.any(np.logical_and.reduce((  
                                                    best_within_rk,    #... within r_k 
                                                    not_from_this_run, #... not from this run
                                                    H['local_pt'],     #... and a local point
                                                    better)))          #... and better?                                 
                if rules[8]:
                    better_min_within_rk_of_best[i] = np.any(np.logical_and.reduce((         
                                                    best_within_rk,  #... within r_k 
                                                    H['local_min'],  #... and a local min
                                                    better)))        #... and better?

        # Remove any runs if their next or best pt_ids violate any rules
        if rules[0]:
            possible_runs = np.logical_and.reduce(( possible_runs, 
                                np.logical_not(active_within_rk_of_next)))
        if rules[1]:
            possible_runs = np.logical_and.reduce(( possible_runs, 
                                np.logical_not(past_local_within_rk_of_next)))
        if rules[2]:
            possible_runs = np.logical_and.reduce(( possible_runs, 
                                np.logical_not(min_within_rk_of_next)))
        if rules[3]:
            possible_runs = np.logical_and.reduce(( possible_runs, 
                                np.logical_not(active_within_rk_of_best)))
        if rules[4]:
            possible_runs = np.logical_and.reduce(( possible_runs, 
                                np.logical_not(past_local_within_rk_of_best)))
        if rules[5]:
            possible_runs = np.logical_and.reduce(( possible_runs, 
                                np.logical_not(min_within_rk_of_best)))
        if rules[6]:
            possible_runs = np.logical_and.reduce(( possible_runs, 
                                np.logical_not(better_active_within_rk_of_best)))
        if rules[7]:
            possible_runs = np.logical_and.reduce(( possible_runs, 
                                np.logical_not(better_past_local_within_rk_of_best)))
        if rules[8]:
            possible_runs = np.logical_and.reduce(( possible_runs, 
                                np.logical_not(better_min_within_rk_of_best)))

        # If there are any remaining pt_ids
        if np.any(possible_runs):
            if tie_break == 0:
                return self.__cleanup__(np.where(self.run_numbers==np.random.choice(runs[possible_runs])))
            elif tie_break == 1: 
                next_run_number = runs[possible_runs][runs[possible_runs]>last_run_num_evaled]
                if len(next_run_number)==0:
                    next_run_number = runs[possible_runs][0]
                else: 
                    next_run_number = next_run_number[0]
                return self.__cleanup__(np.where(self.run_numbers==next_run_number))
            elif tie_break == 2:
                if 'best_ind' not in locals():
                    best_ind = np.zeros(len(runs),dtype='int')
                    for run,i in zip(runs,range(0,len(runs))):
                        best_ind[i] = LH['run_order'][run][np.argmin(H['f'][LH['run_order'][run,LH['run_order'][run]>=0]])]
                sorted_inds = best_ind[np.argsort(H['f'][best_ind])]
                best_run_satisfying_rules = runs[np.argsort(H['f'][best_ind.astype('int')])][possible_runs][0]
                return self.__cleanup__(np.where(self.run_numbers==best_run_satisfying_rules))
        else:
            return (None,-1,0,0)
        
