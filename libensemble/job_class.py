import time
import datetime
from libensemble.message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG

class Job():
        
    def __init__(self):
        self.time = 0.0
        self.start = 0.0
        self.end = 0.0
        self.date_start = None
        self.date_end = None        
        self.pid = 0 #process ID - not currently used.
        self.calc_type = None
        
        #temp - this is a libe feature that is to be reviewed for best solution
        #Includes use of strings/descriptions for job.status.
        self.status = "Not complete" 
        #self.jobid = 
        
    def start_timer(self):
        self.start = time.time()
        self.date_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    
    def stop_timer(self):
        self.end = time.time()
        self.date_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        #increment so can start and stop repeatedly
        self.time += self.end - self.start
        
#    def get_time():
#        return self.time
    
    #Should use message_numbers - except i want to separate type for being just a tag.
    def get_type(self):
        if self.calc_type==EVAL_SIM_TAG:
            return 'sim'
        elif self.calc_type==EVAL_GEN_TAG:
            return 'gen' 
        elif self.calc_type==None:
            return 'No type set'
        else:
            return 'Unknown type'
