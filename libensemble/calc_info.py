import time
import datetime
import itertools
import os
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG

#Maybe these should be set in here - and make manager signals diff? Would mean called by sim_func...
#Currently get from message_numbers
from libensemble.message_numbers import UNSET_TAG
from libensemble.message_numbers import WORKER_KILL
from libensemble.message_numbers import WORKER_KILL_ON_ERR
from libensemble.message_numbers import WORKER_KILL_ON_TIMEOUT
from libensemble.message_numbers import JOB_FAILED 
from libensemble.message_numbers import WORKER_DONE
from libensemble.message_numbers import MAN_SIGNAL_FINISH
from libensemble.message_numbers import MAN_SIGNAL_KILL
from libensemble.message_numbers import CALC_EXCEPTION

class CalcInfo():
    
    newid = itertools.count()
    stat_file = 'libe_summary.txt'
    worker_statfile = None
    keep_worker_stat_files = False

    calc_type_strings = {
        EVAL_SIM_TAG: 'sim',
        EVAL_GEN_TAG: 'gen',
        None: 'No type set'
    }

    calc_status_strings = {
        MAN_SIGNAL_FINISH: "Manager killed on finish",
        MAN_SIGNAL_KILL: "Manager killed job",
        WORKER_KILL_ON_ERR: " Worker killed job on Error",
        WORKER_KILL_ON_TIMEOUT: "Worker killed job on Timeout",
        WORKER_KILL: "Worker killed",
        JOB_FAILED: "Job Failed",
        WORKER_DONE: "Completed",
        CALC_EXCEPTION: "Exception occurred",
        None: "Unknown Status"
    }

    @staticmethod
    def set_statfile_name(name):
        CalcInfo.stat_file = name
    
    @staticmethod
    def smart_sort(l):
        import re
        """ Sort the given iterable in the way that humans expect.
        
        For example: Worker10 comes after Worker9. No padding required
        """ 
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(l, key = alphanum_key)

    @staticmethod
    def create_worker_statfile(workerID):
        CalcInfo.worker_statfile = CalcInfo.stat_file + '.w' + str(workerID)
        with open(CalcInfo.worker_statfile,'w') as f:
            f.write("Worker %d:\n" % (workerID))        

    @staticmethod
    def add_calc_worker_statfile(calc):
        with open(CalcInfo.worker_statfile,'a') as f:
            calc.print_calc(f)

    @staticmethod
    def merge_statfiles():
        import glob
        worker_stat_files = CalcInfo.stat_file + '.w'
        stat_files = CalcInfo.smart_sort(glob.glob(worker_stat_files + '*'))        
        with open(CalcInfo.stat_file, 'w') as outfile:
            for fname in stat_files:
                with open(fname) as infile:
                    outfile.write(infile.read())
        for file in stat_files:
            if not CalcInfo.keep_worker_stat_files:
                os.remove(file)        
    
    def __init__(self):
        self.time = 0.0
        self.start = 0.0
        self.end = 0.0
        self.date_start = None
        self.date_end = None        
        self.calc_type = None
        self.id = next(CalcInfo.newid)
        self.status = "Not complete" 
    
    def start_timer(self):
        self.start = time.time()
        self.date_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    def stop_timer(self):
        self.end = time.time()
        self.date_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        #increment so can start and stop repeatedly
        self.time += self.end - self.start
        
    def print_calc(self, fileH):
        fileH.write("   Calc %d: %s Time: %.2f Start: %s End: %s Status: %s\n" % (self.id, self.get_type() ,self.time, self.date_start, self.date_end, self.status))

    #Should use message_numbers - except i want to separate type for being just a tag.

    def get_type(self):
        return CalcInfo.calc_type_strings.get(self.calc_type, "Unknown type")

    def set_calc_status(self, calc_status_flag):
        """Set status description for this calc"""
        #Prob should store both flag and description (as string)
        #For now assuming if not got an error - it was ok
        self.status = CalcInfo.calc_status_strings.get(calc_status_flag, "Completed")
