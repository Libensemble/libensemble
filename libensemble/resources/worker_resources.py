import logging
from collections import Counter
from collections import OrderedDict
import numpy as np
from libensemble.resources.base_worker_class import BaseWorkerResources

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class WorkerResourcesException(Exception):
    "Worker resources module exception."


class ResourceManager(BaseWorkerResources):

    rset_dtype = [('assigned', int),  # Holds worker ID assigned to or zero
                  ('group', int)      # Group ID this resource set belongs to
                  # ('pool', int),    # Pool ID (eg. separate gen/sim resources) - not yet used.
                  ]

    def __init__(self, num_workers, resources):
        """Initializes a new WorkerResources instance

        Determines the compute resources available for current worker, including
        node list and cores/hardware threads available within nodes.

        Parameters
        ----------

        workerID: int
            workerID of current process

        comm: Comm
            The Comm object for manager/worker communications

        resources: Resources
            A Resources object containing global nodelist and intranode information

        """
        super().__init__(num_workers, resources)

        # SH May change name from index_list - something like "default_mapping"
        # SH Should default mapping use 1 resource set each or divide up?
        self.index_list = ResourceManager.get_index_list(self.num_workers, resources.zero_resource_workers)
        print('index list:', self.index_list)  # SH TODO: Remove when done testing

        #SH TODO: Need to update to make uneven distribution of rsets to nodes work as does on develop
        self.rsets = np.zeros(self.num_rsets, dtype=ResourceManager.rset_dtype)
        self.rsets['assigned'] = 0
        self.rsets['group'] = ResourceManager.get_group_list(self.split_list)
        self.num_groups = self.rsets['group'][-1]
        unique, counts = np.unique(self.rsets['group'], return_counts=True)
        self.rsets_free = self.num_rsets

        # SH TODO: Useful for scheduling tasks with different sized groups (resource sets per node).
        self.group_sizes = dict(zip(unique, counts))
        self.ngroups_by_size = Counter(counts)
        self.even_groups = True if len(self.ngroups_by_size) == 1 else False
        print('\nrsets are {} even groups is {}\n'.format(self.rsets,self.even_groups))
        #import pdb;pdb.set_trace()


    def assign_rsets(self, rset_team, worker_id):
        """Mark the resource sets given by rset_team as assigned to worker_id"""

        if rset_team:
            rteam = self.rsets['assigned'][rset_team]
            for i, wid in enumerate(rteam):
                if wid == 0:
                    self.rsets['assigned'][rset_team[i]] = worker_id
                    self.rsets_free -= 1
                elif wid != worker_id:
                    # Raise error if rsets are already assigned to a different worker.
                    raise WorkerResourcesException("Error: Attempting to assign rsets {} already assigned to workers: {}".
                                             format(rset_team, rteam))

            #self.rsets['assigned'][rset_team] = worker_id
            #self.rsets_free -= len(rset_team)

            #print('rsets free', self.rsets_free)
            #print('resource ids assigned', np.where(self.rsets['assigned'])[0])  # SH TODO: Remove
            #print('resource worker assignment', self.rsets['assigned'])  # SH TODO: Remove
            # print('resources unassigned', np.where(self.rsets['assigned'] == 0)[0])  # SH TODO: Remove


    def free_rsets(self, worker=None):
        """Free up assigned resource sets"""
        if worker is None:
            self.rsets['assigned'] = 0
            #self.num_assigned = 0
            self.rsets_free = self.num_rsets
        else:
            rsets_to_free = np.where(self.rsets['assigned'] == worker)[0]
            self.rsets['assigned'][rsets_to_free] = 0
            self.rsets_free += len(rsets_to_free)
            #print('\nWorker {} returned - freed up rsets {}'.format(worker, rsets_to_free))

        # print('resources assigned', np.where(self.rsets['assigned'])[0])  # SH TODO: Remove
        # print('resources unassigned', np.where(self.rsets['assigned'] == 0)[0])  # SH TODO: Remove

    @staticmethod
    def get_group_list(split_list):
        group = 1
        group_list = []
        node = split_list[0]

        # SH What to do when multiple nodes in each entry........ what is group then.
        for i in range(len(split_list)):
            if split_list[i] == node:
                group_list.append(group)
            else:
                node = split_list[i]
                group += 1
                group_list.append(group)
        return group_list

    @staticmethod
    def get_index_list(num_workers, zero_resource_list):
        """Map WorkerID to index into a nodelist"""
        index = 0
        index_list = []
        for i in range(1, num_workers+1):
            if i in zero_resource_list:
                index_list.append(None)
            else:
                index_list.append(index)
                index += 1
        return index_list



class WorkerResources(BaseWorkerResources):
    """Provide system resources per worker to libEnsemble and executor.

    **Object Attributes:**

    These are set on initialisation.

    :ivar int num_workers: Total number of workers
    :ivar int workerID: workerID
    :ivar list local_nodelist: A list of all nodes assigned to this worker
    :ivar int local_node_count: The number of nodes available to this worker (rounded up to whole number)
    :ivar int workers_per_node: The number of workers per node (if using subnode workers)
    """

    #def __init__(self, workerID, comm, resources):
    def __init__(self, num_workers, resources, workerID):

        """Initializes a new WorkerResources instance

        Determines the compute resources available for current worker, including
        node list and cores/hardware threads available within nodes.

        Parameters
        ----------

        workerID: int
            workerID of current process

        comm: Comm
            The Comm object for manager/worker communications

        resources: Resources
            A Resources object containing global nodelist and intranode information

        """
        super().__init__(num_workers, resources)

        #self.num_workers = comm.get_num_workers()

        self.workerID = workerID
        self.rset_team = None
        self.slots = None
        self.even_slots = None
        self.slot_count = None
        self.slots_on_node = None

        #SH TODO: Maybe call total_num_rsets or global_num_rsets - as its not rsets for this worker.
        #self.num_rsets = resources.num_resource_sets or self.num_workers_2assign2  ######Now in baseclass

        ######do i need this??? - use below - but when restrucutre resources - will be all better
        self.zero_resource_workers = resources.zero_resource_workers
        self.local_nodelist = []
        self.local_node_count = len(self.local_nodelist)
        self.set_slot_count()

    def set_rset_team(self, rset_team):
        """Update worker team and local attributes

        Updates: rset_team
                 local_nodelist
                 slots (dictionary with list of partitions for each node)
                 slot_count - number of slots on each node # SH TODO: Make a list if uneven?
                 local_node_count
        """
        if self.workerID in self.zero_resource_workers:
            return

        if rset_team != self.rset_team:  # Order matters
            self.rset_team = rset_team
            self.local_nodelist, self.slots = \
                WorkerResources.get_local_nodelist(self.workerID, self.rset_team,
                                                   self.split_list, self.rsets_per_node)
            self.set_slot_count()
            self.local_node_count = len(self.local_nodelist)

    # SH TODO: Same count, but I want same list...
    #          This needs checking... what is slot_count/slots_on_node when uneven
    #          May be more efficient to do when create slot list.
    def set_slot_count(self):
        if self.slots:
            # Check if even distribution
            # lens = set(map(len, self.slots.values()))
            # lens = set(map(len, self.slots.values()))
            # self.even_slots = True if len(lens) == 1 else False

            # Check if same slots on each node (not just lengths)
            # SH TODO: Maybe even_slots v equal_slots?
            first_node_slots = list(self.slots.values())[0]
            all_match = True
            for node_list in self.slots.values():
                if node_list != first_node_slots:
                    all_match = False
                    break

            # SH TODO: Need to update for uneven scenarios
            self.even_slots = True if all_match else False
            if self.even_slots:
                self.slots_on_node = first_node_slots
                self.slot_count = len(self.slots_on_node)
            else:
                self.slots_on_node = None  # SH TODO: What should this be
                self.slot_count = None  # SH TODO: Could be list of lengths


    # Will work out rset siblings - not just yet - or mayb can use get_rsets_by_group
    #@staticmethod
    #def get_num_siblings_by_rset(split_list):
        #group = 1
        #num_siblings_by_rset = []
        #node = split_list[0]

        #for i in range(len(split_list)):
            #if split_list[i] == node:
                #group_list.append(group)
            #else:
                #node = split_list[i]
                #group += 1
                #group_list.append(group)
        #return group_list

    # Not used currently
    #@staticmethod
    #def get_rsets_by_group(split_list):
        #"""Returns a dictionary where key is groupID and values
        #are a list of resource set IDs (indices into rsets array).
        #"""
        #rsets_by_group = {}
        #group = 0
        #loc_nodes = None
        #for i in range(len(split_list)):
            #if split_list[i] == loc_nodes:
                #rsets_by_group[group].append(i)
            #else:
                #loc_nodes = split_list[i]
                #group += 1
                #rsets_by_group[group] = [i]
        #return rsets_by_group


    # Not used currently
    #@staticmethod
    #def map_workerid_to_index(num_workers, workerID, zero_resource_list):
        #"""Map WorkerID to index into a nodelist"""
        #index = workerID - 1
        #if zero_resource_list:
            #for i in range(1, num_workers+1):
                #if i in zero_resource_list:
                    #index -= 1
                #if index < i:
                    #return index
            #raise WorkerResourcesException("Error mapping workerID {} to nodelist index {}".format(workerID, index))
        #return index

    @staticmethod
    def get_local_nodelist(workerID, rset_team, split_list, rsets_per_node):
        """Returns the list of nodes available to the current worker"""

        # SH May update to do in two stages - get nodelist then get slots
        # but still have to merge with uneven layout handling (as on develop).

        # SH TODO: Update docstring - or split function (also returns slots dictionary)
        #          Remove print comments when done testing
        if workerID is None:
            raise WorkerResourcesException("Worker has no workerID - aborting")

        # print('Worker {}. rsets_per_node {}'.format(workerID, rsets_per_node), flush=True)  # SH TODO: Remove
        team_list = []
        for index in rset_team:
            team_list += split_list[index]

        # print('Worker {} team_list {}'.format(workerID, team_list),flush=True)  # SH TODO: Remove

        local_nodelist = list(OrderedDict.fromkeys(team_list))  # Maintain order of nodes
        # print("Worker {} Worker's local_nodelist is {}".format(workerID, local_nodelist),flush=True) # SH TODO:Remove
        logger.debug("Worker's local_nodelist is {}".format(local_nodelist))

        # SH TODO: Maybe can essentailly do this at mapping stage with group list or create a structure to reference.
        slots = {}
        for node in local_nodelist:
            slots[node] = []

        for index in rset_team:
            mynodes = split_list[index]
            # If rset has > 1 node, then all nodes just have a single slot (slot 0).
            if len(mynodes) > 1:
                for node in  mynodes:
                    slots[node].append(0)
            else:
                mynode = split_list[index][0]
                # rsets_per_node = local_rsets_list[index]  # SH TODO Support uneven rsets per node
                pos_in_node = index % rsets_per_node  # SH TODO: check/test this
                slots[mynode].append(pos_in_node)
                # SH TODO: Can potentially create a machinefile from slots if/when support uneven lists

        # print("Worker {} slots are {}".format(workerID, slots),flush=True) # SH TODO:Remove

        return local_nodelist, slots
