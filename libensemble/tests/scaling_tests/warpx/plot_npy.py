#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    results = np.load(sys.argv[1])
else:
    print('You need to supply an .npy file - aborting')
    sys.exit()
    
names = results.dtype.names
print(names)
results_dict = {}
for i in range(len(names)):
    results_dict[names[i]] = np.array([task[i] for task in results])

plt.figure(figsize=(12,4))
plt.subplot(131)
# plt.plot(-results_dict['x'], results_dict['energy_std'], '.')
plt.scatter(results_dict['x'][:,0], results_dict['energy_std'], c=results_dict['given_time'], cmap='inferno')
plt.xlim(1.e-13, 3.e-12)
plt.colorbar()
plt.grid()
plt.xlabel('Initial charge (C)')
plt.clabel('start time (arb. units)')
plt.title('Energy spread')
plt.title('Energy spread (%)')
plt.subplot(132)
plt.plot(results_dict['x'][:,0], results_dict['energy_avg'], '.')
plt.grid()
plt.xlabel('Initial charge (C)')
plt.ylabel('Energy (MeV)')
plt.title('Average energy')
plt.subplot(133)
plt.plot(results_dict['x'][:,0], results_dict['charge'], '.')
plt.grid()
plt.xlabel('Initial charge (C)')
plt.ylabel('final charge / initial charge')
plt.title('charge ratio')
plt.tight_layout()
plt.savefig('results.pdf', bbox_inches='tight')
