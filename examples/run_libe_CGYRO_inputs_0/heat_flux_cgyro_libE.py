import subprocess
import numpy  as np

#result_shell = subprocess.run('ls -l | grep "dd.dat"', capture_output=True, text=True, shell=True)
#print("Output using shell:", result_shell.stdout)

subprocess.run('cgyro_plot -e .  -plot flux  -ext nox > dd1.dat', capture_output=True, text=True, shell=True)

d = {}
with open("input.cgyro.gen") as f:
    for line in f:
       (val,key) = line.split()
       d[key] = val   
  
n_species = int( d['N_SPECIES'])

#print (n_species)

file1 = open('dd1.dat', 'r')
Lines = file1.readlines()
#Lines = file1.read()
#Lines =Lines.splitlines()      


#print (Lines)

count = 0
# Strips the newline character
for line in Lines:
       count += 1
    #if (count >5)  :
       line=line.split()
       #if(line[0]!='INFO:'): 
       #    print("Line{}: {}".format(count, line))
       if (line[0] =='Q' ) :
            #print (repr(np.array(line[2:2+n_species])))
           print( np.array2string(np.array(line[2:2+n_species]), separator=', '))
