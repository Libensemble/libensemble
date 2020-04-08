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

# import pickle
# with open(sys.argv[2], 'rb') as f:
#     data_pickle = pickle.load(f)
# print(data_pickle)

print(results_dict['emittance'])
print(results_dict['x'][:,0])

plt.figure(figsize=(8,8))
plt.subplot(221)
# plt.plot(-results_dict['x'], results_dict['energy_std'], '.')
# plt.scatter(results_dict['x'][:,0], results_dict['energy_std'], c=results_dict['given_time'], cmap='inferno')
plt.scatter(results_dict['x'][:,0], results_dict['energy_std'], c=results_dict['sim_id'], cmap='inferno')
cbar = plt.colorbar()
cbar.ax.set_ylabel('start time (arb. units)')
plt.xlim(1.e-13, 3.e-12)
plt.grid()
plt.xlabel('Initial charge (C)')
plt.title('Energy spread')
plt.title('Energy spread (%)')
plt.subplot(222)
plt.plot(results_dict['x'][:,0], results_dict['energy_avg'], '.')
plt.grid()
plt.xlabel('Initial charge (C)')
plt.ylabel('Energy (MeV)')
plt.title('Average energy')
plt.subplot(223)
plt.plot(results_dict['x'][:,0], results_dict['charge'], '.')
plt.grid()
plt.xlabel('Initial charge (C)')
plt.ylabel('final charge / initial charge')
plt.title('charge ratio')
plt.subplot(224)
plt.plot(results_dict['x'][:,0], results_dict['emittance'], '.')
plt.grid()
plt.xlabel('Initial charge (C)')
plt.ylabel('emittance')
plt.title('emittance')
plt.tight_layout()
plt.savefig('results.pdf', bbox_inches='tight')
