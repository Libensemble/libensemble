from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import csv

# Call this script to plot the contents of 'pareto_pts.txt' when P=3.
obj1 = []
obj2 = []
obj3 = []
with open('pareto_pts.txt') as file:
   reader = csv.reader(file, delimiter=' ',skipinitialspace=True)
   for row in reader:
      obj1.append(float(row[0]))
      obj2.append(float(row[1]))
      obj3.append(float(row[2]))

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.scatter(obj1, obj2, obj3, c=obj3, cmap='hsv')
ax.set_xlabel('F1')
ax.set_ylabel('F2')
ax.set_zlabel('F3')

plt.title('Tradeoff curve between objectives F1, F2, and F3')

# Uncomment the code below to save a .eps image of the Pareto front.

#ax.view_init(elev=20,azim=45)
#plt.savefig('sphere1.eps',format='eps')
#ax.view_init(elev=50,azim=20)
#plt.savefig('sphere2.eps',format='eps')

plt.show()
