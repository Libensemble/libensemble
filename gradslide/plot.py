import sys
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import animation
# plt.style.use('seaborn-whitegrid')

def readin(fname, lim=-1, key='x='):
    fp = open(fname, 'r+')
    temp = ''
    active = False
    data = np.array([], dtype=float)
    ct = 0
    sz = 1

    for line in fp.readlines():
        line = line.rstrip()
        if lim < 0 or ct > lim:
            break
        if key in line:
            temp = line[len(key):]
            active=True
        elif active:
            temp += line
        if active and ']' in line:
            temp += line
            temp = temp[temp.find('[')+1:temp.find(']')]
            arr = np.array([e for e in temp.split(' ') if len(e) > 0], dtype=float)

            # when adding new data, make sure the size is the same
            if ct == 0:
                sz = len(arr)
                print(arr)
            else:
                assert sz == len(arr)
            data = np.append(data, arr)
            ct += 1
            active=False

    return np.reshape(data, newshape=(len(data)//sz, -1))

def readin_vals(fname, lim=-1, key='x='):
    fp = open(fname, 'r+')
    temp = ''
    active = False
    data = np.array([], dtype=float)
    ct = 0
    sz = 1

    for line in fp.readlines():
        line = line.rstrip()
        if lim < 0 or ct > lim:
            break
        if key in line:
            curr_data = float(line[len(key):])
            data = np.append(data, curr_data)

    return data

def print_friendly(data, i=0):
    assert len(data.shape) == 1
    print('d{} = np.array(['.format(i), end='')
    for d in data[:-1]:
        print('{:.8f}'.format(d), end=', ')
    print('{:.8f}])\n'.format(data[-1]))

def anim_agents(data, num_slow=0, outname='basic_animation'):
    """
    Given input log files (@data), creates animation of agents

    author: Jake Vanderplas
    email: vanderplas@astro.washington.edu
    website: http://jakevdp.github.com
    license: BSD
    Please feel free to use and modify this, but keep the above information. Thanks!
    """
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    D = np.reshape(data, newshape=(-1,))
    x_min = np.amin(D[::2])
    x_max = np.amax(D[::2])
    y_min = np.amin(D[1::2])
    y_max = np.amax(D[1::2])
    ax = plt.axes(xlim=(x_min, x_max), ylim=(y_min, y_max), title="Agent i's x value", xlabel=r'$[x_i]_0$', ylabel=r'$[x_i]_1$')
    line, = ax.plot([], [], ls='', marker='o', color='black')
    line_sol, = ax.plot([1], [1], ls='', marker='*', color='red')
    global prog, N
    prog = 0
    
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line, line_sol
    
    # calculates number of frames, including slowmo ones
    global slowf
    slowf = 40 # slowing factor
    num_slow = min(num_slow, len(data))
    tot_num_frames = max(0, (slowf-1))*num_slow + len(data)-1
    N = tot_num_frames = int(tot_num_frames)

    # animation function.  This is called sequentially
    def animate(i):
        global prog, N
        if (i+1)/N > prog:
            print('{}% '.format(int(100*(i+1)//N)), end='', flush=True)
            prog += 0.1
            if prog < 1:
                print('>> ', end='', flush=True)
            else:
                print('')
        # slowdown first @num_slow by 10x
        global slowf
        if i < num_slow*slowf:
            i = i//slowf
        else:
            i=i-(slowf-1)*num_slow
        curr = data[i]
        x = curr[::2]
        y = curr[1::2]
        line.set_data(x, y)
        return line, line_sol
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=tot_num_frames, interval=33, blit=True)

    print('Saving {} frames'.format(N))
    
    # for more info, see http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save('{}.mp4'.format(outname), fps=30, extra_args=['-vcodec', 'libx264'])

def anim_grads(data, outname='basic_animation'):
    """ Given input log files (@data), creates animation of gradients """

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    t = 100

    y = data[:100]
    Y = np.reshape(y, newshape=(-1,2))
    norm_y = la.norm(Y, axis=1)
    ytop = 1.01*np.amax(norm_y)

    ax = plt.axes(xlim=(0,t), ylim=(0, ytop), title="Agent i's gradient", ylabel=r'$\nabla f_i(x)$', xlabel='Iter')
    line, = ax.plot([], [], ls='', marker='.', color='black')
    prog = 0
    
    def init():
        line.set_data([], [])
        return line,
    
    def animate(i):
        i = i+1
        i0 = max(0, i-75)
        ax.set(xlim=(i0,i0+100))
        x = np.arange(i0, i)
        x = np.repeat(x, 12)   # 12 agents

        y = data[i0:i]
        Y = np.reshape(y, newshape=(-1,2))
        norm_y = la.norm(Y, axis=1)
        line.set_data(x, norm_y)
        return line,
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(data), interval=33, blit=True)

    print('Saving {} frames'.format(len(data)))
    
    # for more info, see http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save('{}.mp4'.format(outname), fps=30, extra_args=['-vcodec', 'libx264'])
    
def main():
    f_in = 'run.log'
    f_out = 'run'

    if len(sys.argv) >= 2:
        f_in = sys.argv[1]
    if len(sys.argv) >= 3:
        f_out = sys.argv[2]

    num_frames = 750
    plot_x = True
    just_read = False

    if just_read:
        data = readin_vals(f_in, num_frames, key='gap=')
        print_friendly(data,5)
        return
    if plot_x:
        data = readin(f_in, num_frames, key='x=')
        anim_agents(data, 20, f_out)
    else:
        data = readin(f_in, num_frames, key='g=')
        anim_grads(data, f_out)

from ddata import *
def custom_plot():
    _, ax = plt.subplots()
    ds = [d1,d2,d3,d4,d5]
    names = ['Nesterov (Nest)', r'Nest Dist ($R=1$)', r'Nest Dist ($R=100$)', 'N-agent', 'N-agent Fast']
    lss   =["solid","dashed","dotted","dashdot",(1,(3,5,1,5,1,5))]
    colors=['gray','blue', 'red', 'orange', 'green']
    for i,di in enumerate(ds):
        n = min(15000, len(di))
        ax.plot(np.arange(n), di[:n], label=names[i], linestyle=lss[i], color=colors[i])
    ax.set(yscale='log', 
           ylabel=r'$f(x_k)-f(x_*)$', 
           xlabel=r'Iteration ($k$)', 
           title=r"Optimality gap for solving alternative Rosenbrock function ($n=100, m=999999999999999999$)")
    ax.legend()
    plt.show()

if __name__ == '__main__':
    # custom_plot()
    main()


