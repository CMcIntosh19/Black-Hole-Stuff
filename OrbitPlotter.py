# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:37:02 2022

@author: hepiz
"""

import numpy as np
import matplotlib.pyplot as plt
import MetricMath as mm

def get_index(array, time):
    idx = np.abs(array - time).argmin()
    val = array.flat[idx]
    return np.where(array == val)[0][0]

def sph2cart(pos):
    x = pos[0] * np.sin(pos[1]) * np.cos(pos[2])
    y = pos[0] * np.sin(pos[1]) * np.sin(pos[2])
    z = pos[0] * np.cos(pos[1])
    return [x, y, z]

def plotvalue(data, value, start=0, end=-1):
    if (type(value) == str) and (value in data.keys()):
        fig, ax = plt.subplots()
        if len(data[value]) == len(data["time"]):
            to = get_index(data["time"], start)
            if end > 0.0:
                tf = get_index(data["time"], end)
            else:
                tf = get_index(data["time"], data["time"][-1])
            ax.plot(data["time"][to:tf], data[value][to:tf])
            
        elif len(data[value]) == len(data["tracktime"]):
            to = get_index(data["tracktime"], start)
            if end > 0.0:
                tf = get_index(data["tracktime"], end)
            else:
                tf = get_index(data["tracktime"], data["tracktime"][-1])
            ax.plot(data["tracktime"][to:tf], data[value][to:tf])
        print(to, tf) 
        ax.set_title(value + " vs time")
            
    else:
        print("Not a valid plottable")
    return True

def comparevalues(data, values, start=0, end=-1, leg=True):
    clean_list = []
    if type(values) != list:
        print("Must be a list of variables")
        return False
    
    for value in values:
        if value in data.keys():
            clean_list.append(value)
        else:
            print(value + "is not a valid plottable")
    
    fig, ax = plt.subplots()
    for value in clean_list:
        if len(data[value]) == len(data["time"]):
            time = data["time"]
            to = get_index(data["time"], start)
            if end > 0.0:
                tf = get_index(data["time"], end)
            else:
                tf = get_index(data["time"], data["time"][-1])
        elif len(data[value]) == len(data["tracktime"]):
            time = data["tracktime"]
            to = get_index(data["tracktime"], start)
            if end > 0.0:
                tf = get_index(data["tracktime"], end)
            else:
                tf = get_index(data["tracktime"], data["tracktime"][-1])
        
        raw = data[value][to:tf]
        clean = []
        scales = []
        if len(np.shape(data[value])) > 1:
            for i in range(np.shape(raw)[1]):
                rawsub = data[value][to:tf, i]
                clean.append((rawsub - min(rawsub))/(max(rawsub) - min(rawsub)))
                scales.append((round(min(rawsub),3), round(100*(max(rawsub) - min(rawsub))/min(rawsub), 13)))
        else:
            rawsub = data[value][to:tf]
            clean.append((rawsub - min(rawsub))/(max(rawsub) - min(rawsub)))
            scales.append((round(min(rawsub),3), round(100*(max(rawsub) - min(rawsub))/min(rawsub), 13)))

        for i in range(len(clean)):
            ax.plot(time[to:tf], clean[i], label=value+str(scales[i]))
        
    ax.set_title("normalized values vs time")
    if leg == True:
        ax.legend()
    
    return True

def orthoplots(datalist, merge=False, start=0, end=-1, leg=True):
    if type(datalist) != list:
        datalist = [datalist]
    if merge == True:
        fig, ax_list = plt.subplots(1,3)
    else:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        ax_list = [ax1, ax2, ax3]
    
    cap = 0
    for data in datalist:
        to = get_index(data["time"], start)
        if end > 0.0:
            tf = get_index(data["time"], end)
        else:
            tf = get_index(data["time"], data["time"][-1])
        cap = max(max(data["pos"][to:tf,0])*1.05, cap)
        carts = np.array([sph2cart(pos) for pos in data["pos"]])
        ax_list[0].plot(carts[to:tf,0], carts[to:tf,1], label=data["name"])  #XY Plot
        ax_list[0].set_title('XY')
        ax_list[1].plot(carts[to:tf,0], carts[to:tf,2], label=data["name"])  #XZ Plot
        ax_list[1].set_title('XZ')
        ax_list[2].plot(carts[to:tf,1], carts[to:tf,2], label=data["name"])  #ZY Plot
        ax_list[2].set_title('YZ')

    for i in ax_list:
        i.label_outer()
        i.set_xlim(-cap, cap)
        i.set_ylim(-cap, cap)
        i.set_aspect('equal')
        if leg == True:
            i.legend()
    return ax_list

def physplots(datalist, merge=False, start=0, end=-1, fit=True, leg=True):
    if type(datalist) != list:
        datalist = [datalist]
    if merge == True:
        fig1, ax_list1 = plt.subplots(3)
        fig1a, ax_list1a = plt.subplots(3)
        fig2, ax_list2 = plt.subplots(4)
    else:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        ax_list1 = [ax1, ax2, ax3]
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        ax_list1a = [ax4, ax5, ax6]
        fig7, ax7 = plt.subplots()
        fig8, ax8 = plt.subplots()
        fig9, ax9 = plt.subplots()
        fig10, ax10 = plt.subplots()
        ax_list2 = [ax7, ax8, ax9, ax10]
    
    elapse_max = -(10**(30))
    elapse_min = 10**(30)
    max_time = 0
    min_time = 10**(30)
    for data in datalist:
        try:
            to1 = get_index(data["time"], start)
            #print(to1)
            if end == -1:
                tf1 = get_index(data["time"], data["time"][-1])
            else: 
                tf1 = get_index(data["time"], end)
                
            to2 = get_index(data["tracktime"], start)
            if end == -1:
                tf2 = get_index(data["tracktime"], data["tracktime"][-1])
            else: 
                tf2 = get_index(data["tracktime"], end)
            
            min_time = min(data["time"][to1], min_time)
            max_time = max(data["time"][tf1], max_time)
            
            ax_list1[0].plot(data["time"][to1:tf1], data["pos"][to1:tf1, 0], label=data["name"])
            ax_list1[0].set_title('Radius vs Time')
            ax_list1[1].plot(data["time"][to1:tf1], data["pos"][to1:tf1, 1], label=data["name"])
            ax_list1[1].set_title('Theta vs Time')
            ax_list1[2].plot(data["time"][to1:tf1], data["pos"][to1:tf1, 2], label=data["name"])
            ax_list1[2].set_title('Phi vs Time')
            
            ax_list1a[0].plot(data["time"][to1:tf1], data["Lx_momentum"][to1:tf1], label=data["name"])
            ax_list1a[0].set_title('Psuedo Lx vs Time')
            ax_list1a[1].plot(data["time"][to1:tf1], data["Ly_momentum"][to1:tf1], label=data["name"])
            ax_list1a[1].set_title('Psuedo Ly vs Time')
            ax_list1a[2].plot(data["time"][to1:tf1], data["Lz_momentum"][to1:tf1], label=data["name"])
            ax_list1a[2].set_title('Psuedo Lz vs Time')
            elapse_min = min(elapse_min, min(data["pos"][to1:tf1, 2]))
            elapse_max = max(elapse_max, max(data["pos"][to1:tf1, 2]))
            #print(elapse_min, elapse_max)
    
            ax_list2[0].plot(data["tracktime"][to2:tf2], data["energy"][to2:tf2], label=data["name"])
            ax_list2[0].set_title('Energy vs Time')
            ax_list2[1].plot(data["tracktime"][to2:tf2], data["phi_momentum"][to2:tf2], label=data["name"])
            ax_list2[1].set_title('L_phi vs Time')
            ax_list2[2].plot(data["tracktime"][to2:tf2], data["carter"][to2:tf2], label=data["name"])
            ax_list2[2].set_title('Carter(C) vs Time')
            ax_list2[3].plot(data["tracktime"][to2:tf2], data["e"][to2:tf2], label=data["name"])
            ax_list2[3].set_title('Eccentricity vs Time')
            if fit == True:
                b, m = np.polynomial.polynomial.polyfit(list(data["tracktime"][to2:tf2]), data["energy"][to2:tf2], 1)
                ax_list2[0].plot(data["tracktime"][to2:tf2], b + m * data["tracktime"][to2:tf2], '-', label= str(m))
                b, m = np.polynomial.polynomial.polyfit(list(data["tracktime"][to2:tf2]), data["phi_momentum"][to2:tf2], 1)
                ax_list2[1].plot(data["tracktime"][to2:tf2], b + m * data["tracktime"][to2:tf2], '-', label= str(m))
                b, m = np.polynomial.polynomial.polyfit(list(data["tracktime"][to2:tf2]), data["carter"][to2:tf2], 1)
                ax_list2[2].plot(data["tracktime"][to2:tf2], b + m * data["tracktime"][to2:tf2], '-', label= str(m))
        except:
            print("Orbit labelled " + data["name"] + " did a bad")

    step = max(1, int((elapse_max - elapse_min)//(20*np.pi)))
    all_lines = np.arange(elapse_min, elapse_max, step*2*np.pi)
    #all_lines = np.append(all_lines, np.arange(0, elapse_min, -step*2*np.pi))
    #print(all_lines)
    #print(min_time)
    ax_list1[2].hlines(all_lines, min_time, max_time, color='black')
    ax_list1[2].set_title('Phi vs Time (Marked Per ' + str(step) + ' Orbits)')

    for i in ax_list1:
        i.label_outer()
        #i.set_aspect('equal')
        if leg == True:
            i.legend()
    for i in ax_list1a:
        i.label_outer()
        #i.set_aspect('equal')
        if leg == True:
            i.legend()
    for i in ax_list2:
        i.label_outer()
        #i.set_aspect('equal')
        if leg == True:
            i.legend()
    return ax_list1, ax_list1a, ax_list2

'''
def animation_thing(data):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import animation
    int_sphere, int_time = mm.interpolate(data["raw"][:,1:4], data["raw"][:,0])
    x = int_sphere[::10,0]*np.sin(int_sphere[::10,1])*np.cos(int_sphere[::10,2])
    y = int_sphere[::10,0]*np.sin(int_sphere[::10,1])*np.sin(int_sphere[::10,2])
    z = int_sphere[::10,0]*np.cos(int_sphere[::10,1])
    t = int_time[::10]
    dataSet = np.array([x, y, z])  # Combining our position coordinates
    numDataPoints = len(t)
    int_sphere[0]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    line_ani = animation.FuncAnimation(fig, animate_func, interval=10,   
                                       frames=numDataPoints)
    #this bit saves it
    f = r"c://Users/hepiz/Documents/Github/Black-Hole-Stuff/animate_func.gif"
    writergif = animation.PillowWriter(fps=numDataPoints)
    line_ani.save(f, writer=writergif)
'''

def ani_thing2(data):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import animation
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    line = ax.plot([], [], [])[0]
    print(type(line))

    # initialization function: plot the background of each frame
    def init():
        line.set_data_3d([], [], [])
        return line,
    
    int_sphere, int_time = mm.interpolate(data["raw"][:,1:4], data["raw"][:,0])
    X = int_sphere[::10,0]*np.sin(int_sphere[::10,1])*np.cos(int_sphere[::10,2])
    Y = int_sphere[::10,0]*np.sin(int_sphere[::10,1])*np.sin(int_sphere[::10,2])
    Z = int_sphere[::10,0]*np.cos(int_sphere[::10,1])
    t = int_time[::10]
    # animation function.  This is called sequentially
    def animate(i):       
        x = X[:i]
        y = Y[:i]
        z = Z[:i]
        line.set_data(x, y, z)
        return line,
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    line_ani = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=200, interval=20, blit=True)
    
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    f = r"c://Users/hepiz/Documents/Github/Black-Hole-Stuff/animate_func.gif"
    numDataPoints = len(t)
    writergif = animation.PillowWriter(fps=numDataPoints)
    line_ani.save(f, writer=writergif)
    
    plt.show()

def ani_thing3(data, name, threeD=True):
    import matplotlib.animation as animation
    
    jump = 20
    int_sphere, int_time = mm.interpolate(data["raw"][:,1:4], data["raw"][:,0])
    X = int_sphere[::jump,0]*np.sin(int_sphere[::jump,1])*np.cos(int_sphere[::jump,2])
    Y = int_sphere[::jump,0]*np.sin(int_sphere[::jump,1])*np.sin(int_sphere[::jump,2])
    Z = int_sphere[::jump,0]*np.cos(int_sphere[::jump,1])
    t = int_time[::jump]
    num_steps = len(t)
    
    if threeD == True:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        line = ax.plot([], [], [])[0]
        # Setting the axes properties
        rbound = max(data["pos"][:,0])*1.05
        ax.set(xlim3d=(-rbound, rbound), xlabel='X')
        ax.set(ylim3d=(-rbound, rbound), ylabel='Y')
        ax.set(zlim3d=(-rbound, rbound), zlabel='Z')
        
        def update_line(num, xdata, ydata, zdata, line):
            line.set_data_3d(xdata[:num], ydata[:num], zdata[:num])
            return line
    else:
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
        (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
        line = [ax1.plot([], [])[0], ax3.plot([], [])[0], ax4.plot([], [])[0]]
        # Setting the axes properties
        rbound = max(data["pos"][:,0])*1.05
        ax1.set(xlim=(-rbound, rbound), ylim=(-rbound, rbound), ylabel="Y")
        ax2.set_axis_off()
        ax3.set(xlim=(-rbound, rbound), ylim=(-rbound, rbound), xlabel="X", ylabel="Z")
        ax4.set(xlim=(-rbound, rbound), ylim=(-rbound, rbound), xlabel="Y")
    
        def update_line(num, xdata, ydata, zdata, line):
            line[0].set_data(xdata[:num], ydata[:num])
            line[1].set_data(xdata[:num], zdata[:num])
            line[2].set_data(ydata[:num], zdata[:num])
            #line.set_data_3d(xdata[:num], ydata[:num], zdata[:num])
            return line
        
    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_line, num_steps, fargs=(X, Y, Z, line), interval=10)
    
    f = r"c://Users/hepiz/Documents/Github/Black-Hole-Stuff/" + str(name) + ".gif"
    numDataPoints = num_steps
    writergif = animation.PillowWriter(fps=numDataPoints)
    ani.save(f, writer=writergif)
    
    plt.show()
    

def potentplotter(E, L, C, a, rbounds=[-1, -1]):
    
    thetbounds = np.linspace(0.0, 2*np.pi, num=180)
    #tri, sig = rbounds**2 - 2*rbounds + a**2, rbounds**2 + a**2
    
    R = lambda r: ((r**2 + a**2)*E - a*L)**2 - (r**2 - 2*r + a**2)*(r**2 + (L - a*E)**2 + C)
    T = lambda t: C - ((1 - E**2)*(a**2) + (L**2)/(np.sin(t)**2))*(np.cos(t)**2)
        
    print(R(50.0))
    print(R(49.9999999))
    print(R(50.0000001))
    print(T(np.pi/2))
    rx, rn, blah, blee = np.roots([E**2 - 1, 2, (a**2)*(E**2 - 1) - L**2 - C, 2*((a*E - L)**2 + C), -(a**2)*C])
    r0, bloh, bluh = np.roots([4*(E**2 - 1), 6, 2*((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C)])
    ecc = (rx - rn)/(rx + rn)
    if -1 in rbounds:
        p = 1/(1 - E**2)
        rbounds = np.linspace(rn*0.95, rx*1.05, num=100)
    else:
        rbounds = np.linspace(rbounds[0], rbounds[1], num=100)
    print(rx, rn, r0, ecc)
    print(r0/(1-ecc), r0/(1+ecc))
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.plot(rbounds, rbounds*0.0)
    ax1.plot(rbounds, (E**2 - R(rbounds)/(rbounds**2 + a**2))**(1/2) - E)
    #ax2.plot(thetbounds, thetbounds*0.0)
    #ax2.plot(thetbounds, T(thetbounds))

def fouriercountourthing(data, wavedis, num=1000):
    from scipy.fft import rfft, rfftfreq
    waves, time = mm.full_transform(data, wavedis)
    x, z = [], []
    d = 0
    i = 0
    while d < len(waves)-1:
        c, d = i*(len(waves)//num), min((i+2)*(len(waves)//num), len(waves)-1)
        #print(time[c], time[d])
        N = len(waves[c:d, 0, 0])
        samprate = N/(time[d] - time[c])
        x.append((time[c] + time[d])/2)
        xf = rfftfreq(N, 1 / samprate)
        z.append(rfft(waves[c:d, 0, 0])[0:np.where(xf <= 0.10)[0][-1]])
        i += 1
    print("good?")
    x = np.array(x)
    print("good?")
    z = np.abs(np.array(z)**2)
    print("good?")
    y = xf[0:np.where(xf <= 0.10)[0][-1]]
    print("good?")
    X, Y = np.meshgrid(x, y)
    print("good?")
    Z = z.transpose()
    print("good?")
    print(X, Y, Z)
    plt.contourf(X, Y, Z)
    plt.show()
