import os, time

"""
Keyword: [WXYZ_vV]
W, Algorithm: A (GD), B (PDS), C (GS), D (DGD)
X, Comm Graph: 0 (None), [A-C]
Y, Problem: [A-E]
Z, Start: [A-C]
V, Version (Optional): Any integer to indicate which iteration we are
"""
KEYWORDS = [chr(ord('A') + i) for i in range(5)]
v = '0'
fp = open('progress.txt', 'w+')
fp.truncate(0)
fp.write('Keyword: [WXYZ_vV]\nW, Algorithm: A (GD), B (PDS), C (GS), D (DGD)\nX, Comm Graph: 0 (None), [A-C]\nY, Problem: [A-E]\nZ, Start: [A-C]\nV, Version (Optional): Any integer to indicate which iteration we are\n\n')
fp.close()

# Gradient Descent
num_probs = 4
num_start = 3

W = 'A'
X = '0'
for j in range(1,num_probs+1):
    for k in range(1,num_start+1):
        Y = KEYWORDS[j-1]
        Z = KEYWORDS[k-1]
        fname = '{}{}{}{}_v{}.log'.format(W,X,Y,Z,v)

        os.system('echo "Starting {}" >> progress.txt'.format(fname))
        os.system('python nest.py --prob {} --start {}'.format(j,k))
        os.system('echo "Finished {}" >> progress.txt'.format(fname))

# Primal-dual sliding
num_graph = 3
num_probs = 5

W = 'B'
for i in range(1,num_graph+1):
    for j in range(1,num_probs+1):
        for k in range(1,num_start+1):
            X = KEYWORDS[i-1]
            Y = KEYWORDS[j-1]
            Z = KEYWORDS[k-1]
            fname = '{}{}{}{}_v{}.log'.format(W,X,Y,Z,v)

            os.system('echo "Starting {}" >> progress.txt'.format(fname))
            os.system('python pds.py --graph {} --prob {} --start {}'.format(i,j,k))
            os.system('echo "Finished {}" >> progress.txt'.format(fname))

# Gradient sliding
num_probs = 4

W = 'C'
for i in range(1,num_graph+1):
    for j in range(1,num_probs+1):
        for k in range(1,num_start+1):
            X = KEYWORDS[i-1]
            Y = KEYWORDS[j-1]
            Z = KEYWORDS[k-1]
            fname = '{}{}{}{}_v{}.log'.format(W,X,Y,Z,v)

            os.system('echo "Starting {}" >> progress.txt'.format(fname))
            os.system('python zosa.py --graph {} --prob {} --start {}'.format(i,j,k))
            os.system('echo "Finished {}" >> progress.txt'.format(fname))

# Distributed GD
num_probs = 5

W = 'D'
for i in range(1,num_graphs+1):
    for j in range(1,num_probs+1):
        for k in range(1,num_start+1):
            X = KEYWORDS[i-1]
            Y = KEYWORDS[j-1]
            Z = KEYWORDS[k-1]
            fname = '{}{}{}{}_v{}.log'.format(W,X,Y,Z,v)

            os.system('echo "Starting {}" >> progress.txt'.format(fname))
            os.system('python nagent.py --graph {} --prob {} --start {}'.format(i,j,k))
            os.system('echo "Finished {}" >> progress.txt'.format(fname))
