import os
import random
import subprocess

import psutil


class Plopper:
    def __init__(self, sourcefile, outputdir):
        # Initializing global variables
        self.sourcefile = sourcefile
        self.outputdir = outputdir + "/tmp_files"

        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    # Creating a dictionary using parameter label and value
    def createDict(self, x, params):
        dictVal = {}
        for p, v in zip(params, x):
            dictVal[p] = v
        return dictVal

    # Replace the Markers in the source file with the corresponding values
    def plotValues(self, dictVal, inputfile, outputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()

        with open(outputfile, "w") as f2:
            for line in buf:
                modify_line = line
                for key, value in dictVal.items():
                    if key in modify_line:
                        if value != "None":  # For empty string options
                            modify_line = modify_line.replace("#" + key, str(value))

                if modify_line != line:
                    f2.write(modify_line)
                else:
                    # To avoid writing the Marker
                    f2.write(line)

    # Function to find the execution time of the interim file, and return the
    # execution time as cost to the search module
    def findRuntime(self, x, params, worker):
        interimfile = ""
        # exetime = float('inf')
        # exetime = sys.maxsize
        exetime = -1
        counter = random.randint(1, 10001)  # To reduce collision increasing the sampling intervals

        interimfile = self.outputdir + "/" + str(counter) + ".sh"

        # Generate intermediate file
        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, interimfile)

        # compile and find the execution time
        # tmpbinary = interimfile[:-2]
        tmpbinary = interimfile
        # tmpbinary = interimfile[:-3] + '_w' + str(worker)+".sh"

        kernel_idx = self.sourcefile.rfind("/")
        kernel_dir = self.sourcefile[:kernel_idx]

        cmd2 = kernel_dir + "/exe.pl " + tmpbinary

        # Find the execution time

        execution_status = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE)
        app_timeout = 120

        try:
            outs, errs = execution_status.communicate(timeout=app_timeout)
        except subprocess.TimeoutExpired:
            execution_status.kill()
            for proc in psutil.process_iter(attrs=["pid", "name"]):
                if "exe.pl" in proc.info["name"]:
                    proc.kill()
            outs, errs = execution_status.communicate()
            return app_timeout

        exetime = float(outs.strip())
        # exetime = execution_status.stdout.decode('utf-8')
        # if exetime == 0:
        #   exetime = -1

        return exetime  # return execution time as cost
