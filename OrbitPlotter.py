# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:37:02 2022

@author: hepiz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import MetricMathStreamline as mm
import os
from scipy.fftpack import fft
import time
import matplotlib.animation as animation
import pywt

def get_index(array, time):
    '''
    Searches an array for the closest number to the given value, returns the lowest applicable index

    Parameters
    ----------
    array : N element array of floats
        list of coordinate time values
    time : float
        desired time

    Returns
    -------
    ind : int
        index of the value in array closest to time
    '''
    idx = np.abs(array - time).argmin()
    val = array.flat[idx]
    return np.where(array == val)[0][0]

def sph2cart(pos):
    '''
    Converts spherical coords to cartesian
    
    Parameters
    ----------
    pos : 3-element array of floats
        r, theta, and phi position

    Returns
    -------
    new_pos : 3-element array of floats
        x, y, and z position
    '''
    x = pos[0] * np.sin(pos[1]) * np.cos(pos[2])
    y = pos[0] * np.sin(pos[1]) * np.sin(pos[2])
    z = pos[0] * np.cos(pos[1])
    return [x, y, z]

def plotvalue(data, value, vsindex=False, start=0, end=-1):
    '''
    Parameters
    ----------
    data : dictionary
        the thing
    value : string
        thing to plot
    vsindex : bool, optional
        decide whether you're plotting against coordinate time or index. The default is False.
    start : int, optional
        starting time or index. The default is 0.
    end : int, optional
        ending time or index. The default is -1.

    Returns
    -------
    bool
        True!

    '''

    termdict = {"time": [data["time"], "Coordinate Time"],
                "radius": [data["pos"][:,0], "Radius"],
                "theta": [data["pos"][:,1], "Theta"],
                "phi": [data["pos"][:,2], "Phi"],
                "r0": [data["r0"], "Effective Potential Minimum"],
                "ecc": [data["e"], "Eccentricity"],
                "inc": [data["inc"], "Inclination"],
                "periapse": [data["it"], "Periapse"],
                "apoapse": [data["ot"], "Apoapse"],
                "omega": [data["omega"], "Phi Position of Periapse"],
                "otime": [data["otime"], "Time of Periapse"],
                "semi_maj": [0.5*(data["it"] + data["ot"]), "Semimajor Axis"],
                "semi_lat": [0.5*(data["it"] + data["ot"])*(1 - data["e"]**2), "Semilatus Rectum"],
                "radial_v": [data["all_vel"][:,1], "Radial Velocity"],
                "theta_v": [data["all_vel"][:,2], "Theta Velocity"],
                "phi_v": [data["all_vel"][:,3], "Phi Velocity"],
                "total_v": [data["vel"], "Velocity"],
                "energy": [data["energy"], "Specific Energy"],
                "l_momentum": [data["phi_momentum"], "Specific Angular Momentum"],
                "carter": [data["carter"], "Carter Constant"],
                "qarter": [data["qarter"], "Carter Constant (Unnormalized)"],
                "l_momentumx": [data["Lx_momentum"], "Specific Angular Momentum (x-component)"],
                "l_momentumy": [data["Ly_momentum"], "Specific Angular Momentum (y-component)"],
                "l_momentumz": [data["Lz_momentum"], "Specific Angular Momentum (z-component)"]}
    
    if (type(value) == str) and (value in termdict):
        fig, ax = plt.subplots()
        if len(termdict[value][0]) == len(data["time"]):
            if vsindex == True:
                ax.plot(termdict[value][0][start:end]) 
            else:
                to = get_index(data["time"], start)
                if end > 0.0:
                    tf = get_index(data["time"], end)
                else:
                    tf = get_index(data["time"], data["time"][-1])
                ax.plot(data["time"][to:tf], termdict[value][0][to:tf])   
        elif len(termdict[value][0]) == len(data["tracktime"]):
            if vsindex == True:
                inds = [ind for ind in range(len(data["time"])) if (data["time"][ind] in data["tracktime"]) and ((ind >= start) and (ind < (end%len(data["time"]))-1))]
                ax.plot(inds, termdict[value][0])
            else:
                to = get_index(data["tracktime"], start)
                if end > 0.0:
                    tf = get_index(data["tracktime"], end)
                else:
                    tf = get_index(data["tracktime"], data["tracktime"][-1])
                ax.plot(data["tracktime"][to:tf], termdict[value][0][to:tf])
        ax.set_title(termdict[value][1] + " vs Time")
        
    else:
        print("Not a valid plottable. Chose one of the following:")
        for name in termdict:
            print("'" + name + "':", termdict[name][1])
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

def orthoplots(datalist, ortho=False, zoom=1.0, start=0.0, end=-1.0, leg=True, ele=30, azi=-60, cb=False):
    '''
    Plots one or more test particles' path through space
    
    Parameters
    ----------
    datalist : N element list of 30 element dictionaries OR single 30 element dictionary
        dictionary MUST be output of clean_inspiral
    ortho : bool
        determines plot type - False creates a single 3D plot, True creates 3 orthogonal 2D plots from POV of positve x, y, and z axes
        defaults to False
    zoom : float
        determines how tightly plot focuses on origin
        defaults to 1.0 - bounds of plot are just wide enough to include furthest point on orbital path
    start : float
        determines starting coordinate time in whatever units the dictionary is in
        defaults to 0.0
    end : float
        determines final coordinate time in whatever units the dictionary is in
        defaults to -1.0 - gives largest value
    leg : bool
        determines whether or not to include legend
        defaults to True
    ele : float
        determines elevation viewing angle when plotting in 3D, in degrees above or below equator of central body
        defaults to 30 - 30 degrees above equator
    azi : float
        determines azimuthal viewing angle when plotting in 3D, in degrees relative to positive x axis
        defaults to -60 - 60 degrees behind positive x axis
    cb : bool
        determines whether or not to visualize event horizon and ergosphere (if applicable) of central body
        defaults to False

    Returns
    -------
    True
    '''
    if type(datalist) != list:
        datalist = [datalist]
    if ortho == True:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8,8))
        ax2.set_axis_off()
        ax_list = [ax1, ax3, ax4]
        
        cap = 0
        for data in datalist:
            to = get_index(data["time"], start)
            if end > 0.0:
                tf = get_index(data["time"], end)
            else:
                tf = get_index(data["time"], data["time"][-1])
            cap = max(max(data["pos"][to:tf,0])*1.05, cap)
        scale_dict = {0: "", 1: "Thousands of ", 2: "Millions of ", 3: "Billions of ", 4: "Trillions of "}
        scaler = np.floor(np.log10(cap))//3
        scale_word = scale_dict[min(4, scaler)]
        cap = cap/(10**(3*scaler))
        for data in datalist:
            to = get_index(data["time"], start)
            if end > 0.0:
                tf = get_index(data["time"], end)
            else:
                tf = get_index(data["time"], data["time"][-1])

            carts = np.array([sph2cart(pos)/(10**(3*scaler)) for pos in data["pos"][to:tf]])
            cartsxy = np.copy(carts)
            cartsxz = np.copy(carts)
            cartsyz = np.copy(carts)
            
            if cb == True:
                rb = (1 + (1 - data["spin"]**2)**(0.5))/(10**(3*scaler))
                circle1 = plt.Circle((0, 0), rb, color='black')
                circle2 = plt.Circle((0, 0), rb, color='black')
                circle3 = plt.Circle((0, 0), rb, color='black')
                
                al = (azi)*np.pi/180. -np.pi
                el = (ele)*np.pi/180. - np.pi/2.
                Xxy = [ 0.0, 0.0, 1.0]
                Xxz = [ 0.0, 1.0, 0.0]
                Xyz = [ 1.0, 0.0, 0.0]

                A = np.pi - np.arctan(20*cap/rb)
                B_ = A - np.pi/2 + np.arcsin((rb/data["pos"][to:tf,0])*np.sin(A))
                condxy = (np.arccos(np.dot(carts, Xxy)/data["pos"][to:tf,0]) < np.pi - B_)
                condxz = (np.arccos(np.dot(carts, Xxz)/data["pos"][to:tf,0]) < np.pi - B_)
                condyz = (np.arccos(np.dot(carts, Xyz)/data["pos"][to:tf,0]) < np.pi - B_)
                cartsxy = np.array([carts[i] if condxy[i] == True else [np.nan, np.nan, np.nan] for i in range(len(condxy))])
                cartsxz = np.array([carts[i] if condxz[i] == True else [np.nan, np.nan, np.nan] for i in range(len(condxy))])
                cartsyz = np.array([carts[i] if condyz[i] == True else [np.nan, np.nan, np.nan] for i in range(len(condxy))])
                ax_list[0].add_patch(circle1)
                ax_list[1].add_patch(circle2)
                ax_list[2].add_patch(circle3)
            
            ax_list[0].plot(cartsxy[:,0], cartsxy[:,1], label=data["name"], zorder=10)  #XY Plot
            ax_list[1].plot(cartsxz[:,0], cartsxz[:,2], label="_nolabel_", zorder=10)  #XZ Plot
            ax_list[2].plot(cartsyz[:,1], cartsyz[:,2], label="_nolabel_", zorder=10)  #ZY Plot
            
        
        if datalist[0]["inputs"][-1] == "grav":
            unit = "Geometric Units"
        elif datalist[0]["inputs"][-1] == "mks":
            unit = "Meters"
        elif datalist[0]["inputs"][-1] == "mks":
            unit = "Centimeters"
        ax1.set(ylabel="Y")
        ax2.set_axis_off()
        ax3.set(xlabel="X", ylabel="Z")
        ax4.set( xlabel="Y")
        #(ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
        ax2.set_xlim(-cap/zoom, cap/zoom)
        ax2.set_ylim(-cap/zoom, cap/zoom)
        ax2.set_aspect('equal')
        ax2.text(0, 0.62*cap/zoom, "Orthographic View", fontsize=20, ha="center", va="top")
        ax2.text(0, 0.40*cap/zoom, "Scale: " + scale_word + unit, fontsize=15, ha="center", va="top")
        print(cap)
        fig.subplots_adjust(wspace=0, hspace=0)
        legend = fig.legend(loc=(0.75,0.5))
        hor_ratio = legend.get_window_extent().width/ fig.get_window_extent().width
        ver_ratio = legend.get_window_extent().height/ fig.get_window_extent().height
        #print(fig.get_window_extent().width)
        #print(legratio)
        legend.set_bbox_to_anchor(bbox=(0.666 - 0.5*hor_ratio, 0.55 - 0.5*ver_ratio))
        
        
    else:
        fig = plt.figure(figsize=(10,12))
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=ele, azim=azi)
        
        rbound = 0
        for data in datalist:
            to = get_index(data["time"], start)
            if end > 0.0:
                tf = get_index(data["time"], end)
            else:
                tf = get_index(data["time"], data["time"][-1])
                
            rbound = max(max(data["pos"][to:tf,0])*1.05, rbound)
            carts = np.array([sph2cart(pos) for pos in data["pos"][to:tf]])
            
            if cb == True:
                rb = 1 + (1 - data["spin"]**2)**(0.5)
                theta, phi = np.linspace(0, 2*np.pi), np.linspace(0, np.pi)
                phi, theta = np.meshgrid(phi, theta)
                x, y, z = rb*np.sin(theta)*np.sin(phi), rb*np.sin(theta)*np.cos(phi), rb*np.cos(theta)
                ax.plot_surface(x, y, z, color="black", zorder=1, shade=False)
                
                re = 1 + (1 - (data["spin"]*np.cos(theta))**2)**(0.5)
                x, y, z = re*np.sin(theta)*np.sin(phi), re*np.sin(theta)*np.cos(phi), re*np.cos(theta)
                ax.plot_surface(x, y, z, color="darksalmon", zorder=2, alpha = 0.3)
                
                al = (azi)*np.pi/180. -np.pi
                el = (ele)*np.pi/180. - np.pi/2.
                X = [ np.sin(el) * np.cos(al),np.sin(el) * np.sin(al),np.cos(el)]

                A = np.pi - np.arctan(20*rbound/(rb*zoom))
                B_ = A - np.pi/2 + np.arcsin((rb/data["pos"][to:tf,0])*np.sin(A))
                
                blockedcheck = np.arccos(np.dot(carts, X)/data["pos"][to:tf,0]) < np.pi - B_
                boundboxcheck = [False not in piece for piece in np.abs(carts) <= rbound/zoom]
                
                cond = np.logical_and(blockedcheck, boundboxcheck)
                carts = np.array([carts[i] if cond[i] == True else [np.nan, np.nan, np.nan] for i in range(len(cond))])
                
            ax.plot3D(carts[:, 0], carts[:, 1], carts[:, 2], label=data["name"], zorder=10)
            
        
        ax.set(xlim3d=(-rbound/zoom, rbound/zoom), xlabel='X')
        ax.set(ylim3d=(-rbound/zoom, rbound/zoom), ylabel='Y')
        ax.set(zlim3d=(-rbound/zoom, rbound/zoom), zlabel='Z')
        ax.set_box_aspect((rbound, rbound, rbound))
        if leg == True:
            ax.legend()
    return True

def physplots(datalist, merge=False, start=0.0, end=-1.0, fit=True, leg=True):
    '''
    Plots various parameters of one or more test particles' orbits across time
    
    Parameters
    ----------
    datalist : N element list of 30 element dictionaries OR single 30 element dictionary
        dictionary MUST be output of clean_inspiral
    merge : bool
        determines whether to combine certain plots into subplots
        defaults to False
    start : float
        determines starting coordinate time in whatever units the dictionary is in
        defaults to 0.0
    end : float
        determines final coordinate time in whatever units the dictionary is in
        defaults to -1.0 - gives largest value
    fit : bool
        determines whether or not to generate linear fit for certain plots
        defaults to True
    leg : bool
        determines whether or not to include legends
        defaults to True

    Returns
    -------
    N x 5 array of floats (if fit == True)
        derivatives of E, L, C, r0, e w.r.t time
    False (if fit == False)
    '''
    if type(datalist) != list:
        datalist = [datalist]
    if merge == True:
        fig1, ax_list1 = plt.subplots(3)
        fig1a, ax_list1a = plt.subplots(3)
        fig2, ax_list2 = plt.subplots(5)
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
        fig11, ax11 = plt.subplots()
        ax_list2 = [ax7, ax8, ax9, ax10, ax11]
    
    elapse_max = -(10**(30))
    elapse_min = 10**(30)
    max_time = 0
    min_time = 10**(30)
    dervs = []
    for data in datalist:

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
        
        try:
            ax_list2[0].plot(data["tracktime"][to2:tf2], data["energy"][to2:tf2], label=data["name"])
            ax_list2[0].set_title('Energy vs Time')
            ax_list2[1].plot(data["tracktime"][to2:tf2], data["phi_momentum"][to2:tf2], label=data["name"])
            ax_list2[1].set_title('L_phi vs Time')
            ax_list2[2].plot(data["tracktime"][to2:tf2], data["carter"][to2:tf2], label=data["name"])
            ax_list2[2].set_title('Carter(C) vs Time')
            ax_list2[3].plot(data["tracktime"][to2:tf2], data["r0"][to2:tf2], label=data["name"])
            ax_list2[3].set_title('r_0 vs Time')
            ax_list2[4].plot(data["tracktime"][to2:tf2], data["e"][to2:tf2], label=data["name"])
            ax_list2[4].set_title('Eccentricity vs Time')
            if fit == True:
                ax_list1[0].plot(data["tracktime"][to2:tf2], data["r0"][to2:tf2])
                b, mE = np.polynomial.polynomial.polyfit(list(data["tracktime"][to2:tf2]), data["energy"][to2:tf2], 1)
                ax_list2[0].plot(data["tracktime"][to2:tf2], b + mE * data["tracktime"][to2:tf2], '-', label= str(mE))
                print("Edot", mE)
                b, mL = np.polynomial.polynomial.polyfit(list(data["tracktime"][to2:tf2]), data["phi_momentum"][to2:tf2], 1)
                ax_list2[1].plot(data["tracktime"][to2:tf2], b + mL * data["tracktime"][to2:tf2], '-', label= str(mL))
                print("Ldot", mL)
                b, mC = np.polynomial.polynomial.polyfit(list(data["tracktime"][to2:tf2]), data["carter"][to2:tf2], 1)
                ax_list2[2].plot(data["tracktime"][to2:tf2], b + mC * data["tracktime"][to2:tf2], '-', label= str(mC))
                print("Cdot", mC)
                b, mr = np.polynomial.polynomial.polyfit(list(data["tracktime"][to2:tf2]), np.float64(data["r0"][to2:tf2]), 1)
                ax_list2[3].plot(data["tracktime"][to2:tf2], b + mr * data["tracktime"][to2:tf2], '-', label= str(mr))
                print("r0dot", mr)
                b, me = np.polynomial.polynomial.polyfit(list(data["tracktime"][to2:tf2]), np.float64(data["e"][to2:tf2]), 1)
                ax_list2[4].plot(data["tracktime"][to2:tf2], b + me * data["tracktime"][to2:tf2], '-', label= str(me))
                print("edot", me)
                dervs.append([mE, mL, mC, mr, me])
        except:
            pass

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
    if fit == True:
        return dervs
    else:
        return True

def ani_thing3(data, name, ortho=False, zoom=1.0, ele=30, azi=-60, cb=True, numturns=10, fid=1):
    '''
    Creates an animation of a test particle's path through space
    
    Parameters
    ----------
    data: 30 element dictionary
        dictionary MUST be output of clean_inspiral
    name : string
        name of final animation - will be saved as cwd/name.gif
    ortho : bool
        determines plot type - False creates a single 3D plot, True creates 3 orthogonal 2D plots from POV of positve x, y, and z axes
        defaults to False
    zoom : float
        determines how tightly plot focuses on origin
        defaults to 1.0 - bounds of plot are just wide enough to include furthest point on orbital path
    ele : float
        determines elevation viewing angle when plotting in 3D, in degrees above or below equator of central body
        defaults to 30 - 30 degrees above equator
    azi : float
        determines azimuthal viewing angle when plotting in 3D, in degrees relative to positive x axis
        defaults to -60 - 60 degrees behind positive x axis
    cb : bool
        determines whether or not to visualize event horizon and ergosphere (if applicable) of central body
        defaults to False
    numturns : float
        determines approximatelt how many phi-orbits to include at any one time - how long the "tail" is
        defaults to 10
    fid : positive float
        determines how many frames to make the animation - "fidelity"
        defaults to 1 - multiplied by 100 gives 100 frames

    Returns
    -------
    True
    '''
    import matplotlib.animation as animation
    
    int_sphere, int_time = mm.interpolate(data["pos"], data["time"], supress = False)
    X = int_sphere[:,0]*np.sin(int_sphere[:,1])*np.cos(int_sphere[:,2])
    Y = int_sphere[:,0]*np.sin(int_sphere[:,1])*np.sin(int_sphere[:,2])
    Z = int_sphere[:,0]*np.cos(int_sphere[:,1])

    num_steps = int(100*fid)
    
    first_turn = np.where(data["pos"][:,2] > 2*np.pi)[0][0]
    print(first_turn)
    

    if ortho == False:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=ele, azim=azi)
        line = ax.plot([], [], [], zorder=10)[0]

        # Setting the axes properties
        rbound = max(data["pos"][:,0])*1.05/zoom

        if cb == True:
            rb = 1 + (1 - data["spin"]**2)**(0.5)
            theta, phi = np.linspace(0, 2*np.pi), np.linspace(0, np.pi)
            phi, theta = np.meshgrid(phi, theta)
            x, y, z = rb*np.sin(theta)*np.sin(phi), rb*np.sin(theta)*np.cos(phi), rb*np.cos(theta)
            ax.plot_surface(x, y, z, color="black", zorder=2)
            
            re = 1 + (1 - (data["spin"]*np.cos(theta))**2)**(0.5)
            x, y, z = re*np.sin(theta)*np.sin(phi), re*np.sin(theta)*np.cos(phi), re*np.cos(theta)
            ax.plot_surface(x, y, z, color="darksalmon", zorder=1, alpha = 0.3)
            
            al = (azi)*np.pi/180. -np.pi
            el = (ele)*np.pi/180. - np.pi/2.
            V = [ np.sin(el) * np.cos(al),np.sin(el) * np.sin(al),np.cos(el)]
            carts = np.transpose(np.array([X, Y, Z]))
            r = (X**2 + Y**2 + Z**2)**(0.5)
            
            #Hide things behind black hole
            A = np.pi - np.arctan(20*rbound/(rb*zoom))
            B_ = A - np.pi/2 + np.arcsin((rb/r)*np.sin(A))
            cond = (np.arccos(np.dot(carts, V)/r) < np.pi - B_)
            X, Y, Z = np.transpose(np.array([carts[i] if cond[i] == True else [np.nan, np.nan, np.nan] for i in range(len(cond))]))
            
        
        ax.set(xlim3d=(-rbound, rbound), xlabel='X')
        ax.set(ylim3d=(-rbound, rbound), ylabel='Y')
        ax.set(zlim3d=(-rbound, rbound), zlabel='Z')
        ax.set_box_aspect((rbound, rbound, rbound))
        
        def update_line(num, xdata, ydata, zdata, line):
            full = int(len(xdata)/num_steps)
            if numturns == False:
                beg = 0
            else:
                beg = max(0, full*num - first_turn*numturns)
            print(len(xdata), num_steps, full, num)
            #print(beg, full*num, full*num - beg, first_turn)
            line.set_data_3d(xdata[beg:full*num], ydata[beg:full*num], zdata[beg:full*num])
            return line
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8,8))
        ax2.set_axis_off()
        line = [ax1.plot([], [])[0], ax3.plot([], [])[0], ax4.plot([], [])[0]]
        # Setting the axes properties
        ax1.set(ylabel="Y")
        ax2.set_axis_off()
        ax3.set(xlabel="X", ylabel="Z")
        ax4.set( xlabel="Y")
        ax2.text(0.5, 0.81, "Orthographic View", fontsize=20, ha="center", va="top", transform=ax2.transAxes)
        fig.subplots_adjust(wspace=0, hspace=0)
        legend = fig.legend(loc=(0.75,0.5))
        hor_ratio = legend.get_window_extent().width/ fig.get_window_extent().width
        ver_ratio = legend.get_window_extent().height/ fig.get_window_extent().height
        legend.set_bbox_to_anchor(bbox=(0.666 - 0.5*hor_ratio, 0.55 - 0.5*ver_ratio))
        def update_line(num, xdata, ydata, zdata, line):
            full = len(xdata)//num_steps
            if numturns == False:
                beg = 0
            else:
                beg = max(0, full*num - first_turn*numturns)
            end = int(np.round(num*len(xdata)/num_steps))
            line[0].set_data(xdata[beg:end], ydata[beg:end])
            line[1].set_data(xdata[beg:end], zdata[beg:end])
            line[2].set_data(ydata[beg:end], zdata[beg:end])
            
            try:
                rbound = max((xdata[beg:end]**2 + ydata[beg:end]**2 + zdata[beg:end]**2)**0.5)*1.05  
            except:
                rbound = max((xdata**2 + ydata**2 + zdata**2)**0.5)*1.05 

            ax2.set_xlim(-rbound, rbound)
            ax2.set_ylim(-rbound, rbound)
            ax2.set_aspect('equal')
            print(len(xdata)-beg)
            return line
        
    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_line, frames=num_steps + 10, fargs=(X, Y, Z, line), interval=10)
    
    cwd = os.getcwd()
    f = os.path.join(cwd, name + ".gif")
    writergif = animation.PillowWriter(fps=10)
    ani.save(f, writer=writergif)
    
    plt.show()
    return True

def gimme_startpot(data, rbounds = [-1, 1]):
    a, mu = data["inputs"][1], data["inputs"][2]
    E, L, C = data["energy"][0]/mu, data["phi_momentum"][0]/mu, data["carter"][0]/(mu**2)
    print(E, L,C)
    potentplotter(E, L, C, a, rbounds)

def potentplotter(E, L, C, a, rbounds=[-1, -1]):
    if type(E) == np.ndarray:
        pass
    elif type(E) == list:
        E, L, C = np.array(E), np.array(L), np.array(C)
    else:
        E, L, C = np.array([E]), np.array([L]), np.array([C])
        
    thetbounds = np.linspace(0.0, 2*np.pi, num=180)

    R = lambda r: ((r**2 + a**2)*E - a*L)**2 - (r**2 - 2*r + a**2)*(r**2 + (L - a*E)**2 + C)
    T = lambda t: C - ((1 - E**2)*(a**2) + (L**2)/(np.sin(t)**2))*(np.cos(t)**2)
        
    
    rx, rn, blah, blee = np.transpose(np.array([np.roots([E[i]**2 - 1, 2, (a**2)*(E[i]**2 - 1) - L[i]**2 - C[i], 2*((a*E[i] - L[i])**2 + C[i]), -(a**2)*C[i]]) for i in range(len(E))]))
    r0, bloh, bluh = np.transpose(np.array([np.roots([4*(E[i]**2 - 1), 6, 2*((a**2)*(E[i]**2 - 1) - L[i]**2 - C[i]), 2*((a*E[i] - L[i])**2 + C[i])]) for i in range(len(E))]))
    print(r0, bloh, bluh)
    ecc = (rx - rn)/(rx + rn)
    if -1 in rbounds:
        p = 1/(1 - E**2)
        rbounds = np.linspace(rn*0.95, rx*1.05, num=100)
    else:
        rbounds = np.linspace(rbounds[0]*np.ones((len(rn))), rbounds[-1]*np.ones((len(rx))), num=100)

    fig1, ax1 = plt.subplots()
    ax1.plot(rbounds, rbounds*0.0)
    ax1.plot(rbounds, R(rbounds))
    return(R(bloh))

def potentplotter2(cons, a, rbounds=[-1, -1]):
    if len(np.shape(cons)) == 1:
        cons = [cons]
    fig1, ax1 = plt.subplots()
    maxbounds = [1e12, -1e12]
    for E, L, C in cons:
        R = lambda r: ((r**2 + a**2)*E - a*L)**2 - (r**2 - 2*r + a**2)*(r**2 + (L - a*E)**2 + C)
        coeff = np.array([E**2 - 1.0, 2.0, (a**2)*(E**2 - 1.0) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
        coeff2 = np.polyder(coeff)
        rx, rn, blah, blee = np.roots(coeff)
        r0, bloh, bluh = np.roots(coeff2)
        if True in np.iscomplex([rx, rn, blah, blee, r0, bluh]):
            print("HEY")
            print([rx, rn, r0])
        if -1 in rbounds:
            rbounds2 = np.linspace(rn*0.95, rx*1.05, num=100)
        else:
            rbounds2 = np.linspace(rbounds[0], rbounds[-1], num=100)
        maxbounds = [min(maxbounds[0], rbounds2[0]), max(maxbounds[1], rbounds2[-1])]
        ax1.plot(rbounds2, R(rbounds2))
    ax1.hlines(0.0, maxbounds[0], maxbounds[-1], color="black", zorder=1)

def fouriercountourthing(data, wavedis, num=1000):
    from scipy.fft import rfft, rfftfreq
    waves, time = mm.full_transform(data, wavedis)
    x, z = [], []
    d = 0
    i = 0
    while d < len(waves)-1:
        #print(d, len(waves)-1)
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
    print("good?", x)
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

def orbitchecker(data, mu, r0, e):
    dEdt = -(32/5)*(mu**2)*(1+mu)*(1 + (73/24)*(e**2) + (37/96)*(e**4))/((r0**5)*((1-e**2)**(7/2)))
    dLdt = -(32/5)*(mu**2)*((1+mu)**(1/2))*(1 + (7/8)*(e**2))/((r0**(7/2))*((1-e**2)**2))
    dr0dt = -(64/5)*(mu)*(1+mu)*(1 + (73/24)*(e**2) + (37/96)*(e**4))/((r0**3)*((1-e**2)**(7/2)))
    dedt = -(304/15)*e*(mu)*(1+mu)*(1 + (121/304)*(e**2))/((r0**4)*((1-e**2)**(5/2)))
    
    b, mE = np.polynomial.polynomial.polyfit(list(data["tracktime"]), data["energy"], 1)
    b, mL = np.polynomial.polynomial.polyfit(list(data["tracktime"]), data["phi_momentum"], 1)
    b, mC = np.polynomial.polynomial.polyfit(list(data["tracktime"]), data["carter"], 1)
    b, mr = np.polynomial.polynomial.polyfit(list(data["tracktime"]), np.float64(data["r0"]), 1)
    b, me = np.polynomial.polynomial.polyfit(list(data["tracktime"]), np.float64(data["e"]), 1)
    
    print("Peters Expected | Linear Fit | Percent Error")
    print(dEdt, mE, round(100*abs(dEdt - mE)/dEdt, 3))
    print(dLdt, mL, round(100*abs(dLdt - mL)/dLdt, 3))
    print(dr0dt, mr, round(100*abs(dr0dt - mr)/dr0dt, 3))
    print(dedt, me, round(100*abs(dedt - me)/dedt, 3))
    
def peterscheck(mu, r0, e):
    dEdt = -(32/5)*(mu**2)*(1+mu)*(1 + (73/24)*(e**2) + (37/96)*(e**4))/((r0**5)*((1-e**2)**(7/2)))
    dLdt = -(32/5)*(mu**2)*((1+mu)**(1/2))*(1 + (7/8)*(e**2))/((r0**(7/2))*((1-e**2)**2))
    dr0dt = -(64/5)*(mu)*(1+mu)*(1 + (73/24)*(e**2) + (37/96)*(e**4))/((r0**3)*((1-e**2)**(7/2)))
    dedt = -(304/15)*e*(mu)*(1+mu)*(1 + (121/304)*(e**2))/((r0**4)*((1-e**2)**(5/2)))
    return [dEdt, dLdt, dr0dt, dedt]

def top_and_fourier(datalist, start=0, end=-1, width=12, height=0, space=0.01):
    num = len(datalist)
    if num < 2:
        print("For comparisons only")
        return False
    if width == 0:
        width = (10/3)*num
    if height == 0:
        height = 3*num + 1
    fig, ax = plt.subplots(num, 2, figsize=(width, height))
    fig.subplots_adjust(wspace=space)
    #start, end = 0, 20000
    for i in range(num):
        to = get_index(datalist[i]["time"], start)
        if end > 0.0:
            tf = get_index(datalist[i]["time"], end)
        else:
            tf = get_index(datalist[i]["time"], datalist[i]["time"][-1])
        cap = max(datalist[i]["pos"][to:tf,0])*1.05

        scaler = np.floor(np.log10(cap))//3
        carts = np.array([sph2cart(pos)/(10**(3*scaler)) for pos in datalist[i]["pos"]])
        ax[i,0].plot(carts[to:tf,0], carts[to:tf,1])
        ax[i,0].set_aspect('equal')
        wave, time = mm.full_transform(datalist[i], cap*1000)
        x = np.copy(time)
        y1 = np.copy(wave[:,0,0])
        y2 = np.copy(wave[:,0,1])
        N = time.size
        T = (x[-1] - x[0])/N
        yf1 = fft(y1)
        yf2 = fft(y2)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        ax[i,1].plot(xf, 2.0/N * np.abs(yf2[0:N//2]), label = "hx")
        ax[i,1].plot(xf, 2.0/N * np.abs(yf1[0:N//2]), label = "h+")
        #plt.xscale('log')
        ax[i,1].set_yscale('log')
        ax[i,1].set_xscale('log')
        ax[i,1].grid()
        ax[i,1].legend()
        
def orth_and_fourier(data, start=0, end=-1):
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(2,3,4)
    ax2 = fig.add_subplot(2,1,1)
    ax3 = fig.add_subplot(2,3,5)
    ax4 = fig.add_subplot(2,3,6)
    ax_list = [ax1, ax3, ax4]
    
    to = get_index(data["time"], start)
    if end > 0.0:
        tf = get_index(data["time"], end)
    else:
        tf = get_index(data["time"], data["time"][-1])
    cap = max(data["pos"][to:tf,0])*1.05
    scale_dict = {0: "", 1: "Thousands of ", 2: "Millions of ", 3: "Billions of ", 4: "Trillions of "}
    scaler = np.floor(np.log10(cap))//3
    scale_word = scale_dict[min(4, scaler)]
    #cap = cap/(10**(3*scaler))
    to = get_index(data["time"], start)
    if end > 0.0:
        tf = get_index(data["time"], end)
    else:
        tf = get_index(data["time"], data["time"][-1])

    carts = np.array([sph2cart(pos)/(10**(3*scaler)) for pos in data["pos"]])
    ax_list[0].plot(carts[to:tf,0], carts[to:tf,1], label=data["name"])  #XY Plot
    ax_list[1].plot(carts[to:tf,0], carts[to:tf,2], label="_nolabel_")  #XZ Plot
    ax_list[2].plot(carts[to:tf,1], carts[to:tf,2], label="_nolabel_")  #ZY Plot
    #ax1.set_xlim(-cap, cap)
    #ax1.set_ylim(-cap, cap)
    #ax3.set_xlim(-cap, cap)
    #ax3.set_ylim(-cap, cap)
    #ax4.set_xlim(-cap, cap)
    #ax4.set_ylim(-cap, cap)
    ax1.set(xlim=(-cap, cap), ylim=(-cap, cap), xlabel='XY Plot')
    ax2.set(xlabel='Waveform Frequency (G\u209C\u207B\u00B9)')
    ax3.set(xlim=(-cap, cap), ylim=(-cap, cap), xlabel='XZ Plot')
    ax4.set(xlim=(-cap, cap), ylim=(-cap, cap), xlabel='YZ Plot')
    ax1.set_aspect('equal')
    ax3.set_aspect('equal')
    ax4.set_aspect('equal')
    
    wave, time = mm.full_transform(data, cap*1000)
    to = get_index(time, start)
    if end > 0.0:
        tf = get_index(time, end)
    else:
        tf = get_index(time, data["time"][-1])
    x = np.copy(time[to:tf])
    y1 = np.copy(wave[to:tf,0,0])
    y2 = np.copy(wave[to:tf,0,1])
    y0 = np.sqrt(y1**2 + y2**2)
    N = x.size
    T = (x[-1] - x[0])/N
    yf1 = fft(y1)
    yf2 = fft(y2)
    yf0 = fft(y0)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    ax2.plot(xf, 2.0/N * np.abs(yf1[0:N//2]), label = "h+")
    ax2.plot(xf, 2.0/N * np.abs(yf2[0:N//2]), label = "hx")
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    ax2.set_title(data["name"])
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.grid()
    ax2.legend()
    
    return 0

def wavelething(data):
    #It hates you and it's not even what you want, leave it alone
    rad = data["pos"][0,0]
    waves, tim = mm.full_transform(data, rad*100)
    samper = tim[1]-tim[0]
    print(samper)
    period = 2*np.pi*np.sqrt(data["r0"][0]**3)
    print(period)
    freq = 2*np.pi/period
    print(freq)
    #0.06804175435239163
    print(pywt.frequency2scale('morl',samper*freq
*10), pywt.frequency2scale('morl',samper*freq/10))
    scalelow, scalehigh = max(1, pywt.frequency2scale('morl',samper*0.06804175435239163
*10)), pywt.frequency2scale('morl',samper*freq/100)
    coef, freqs = pywt.cwt(waves[:,0,0], np.linspace(scalelow, scalehigh, 200), 'morl',
                       sampling_period=samper) 

    # Show w.r.t. time and frequency
    plt.figure()
    #plt.pcolor(tim, freqs, (coef+np.min(coef))**1/2)
    plt.pcolor(tim, freqs, coef)

    # Set yscale, ylim and labels
    plt.title(data["name"])
    plt.yscale('log')
    #plt.hlines([2*np.pi/period, 1/period], 0, 50000)
    #plt.ylim([1, 100])
    #plt.hlines(pywt.scale2frequency('morl', np.linspace(scalelow, scalehigh, 200))/samper, 0, 50000)
    plt.ylabel('Frequency (GU)')
    plt.xlabel('Time (GU)')
    plt.show()
    return(coef)

#Thing for plotting not-contour plots
'''
import matplotlib.tri as tri
r0, e, i, mu, a = 100, 0.1, 1, 1e-6, 0.0
x_base, y_base = mus2[:-1], a_s2
x = np.sort(np.tile(x_base, len(y_base)))
y = np.tile(y_base, len(x_base))
z = np.array([dictfill(all_dots2, (r0, e, i, x[j], y[j]))[-1] for j in range(len(x))])
Z = my_symlog10(-z)
levels = np.linspace(Z.min(), Z.max(), 28)
fig, ax = plt.subplots()
fig.set(figwidth=5, figheight=3)
plt.scatter(x, y, c=my_symlog10(-z), cmap="viridis")
ax.set_xscale("log")
c = plt.colorbar(label="Powers of 10")
plt.title("Time Derivative of Eccentricity (e=0.1)", fontsize=16)
plt.xlabel("Mass Ratio (q)", fontsize=14)
plt.ylabel("Dimensionless Black Hole Spin (a)", fontsize=14)
plt.show()
'''

#thing for fiulling up big dictionary
'''
#holds2 = {}
count = 1
import time
start = time.time()
es2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
mus2 = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
a_s2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
r0s2 = [10, 20, 50, 100]
is2 = [1, 2, 3, 4]
count = 1
skip = 0
total = len(list(product(r0s2, es2, is2, mus2, a_s2))) - len(list(holds2.keys()))
for r0, e, i, mu, a in product(r0s2, es2, is2, mus2, a_s2):
    try:
        print(r0, e, i, mu, a)
        if (r0, e, i, mu, a) not in list(holds2.keys()):
            lab = "r" + str(r0) + "e" + str(e) + "i" + str(i/2) + "mu" + str(mu) + "a" + str(a)
            holds2[r0, e, i, mu, a] = clean_inspiral3(1, a, mu, "phi_orbit > 25", 10**(-15), lab, params= [r0, e, np.pi/(2**i)], verbose=False)
            try:
                if holds2[r0, e, i, mu, a]["pos"][-1, 2] < 25*2*np.pi:
                    print("Too short!")
                    del holds2[r0, e, i, mu, a]
            except:
                del holds2[r0, e, i, mu, a]
            print("average runtime", (time.time() - start)/(count-skip))
            print("Completed:", count)
            print("Total:", total)
            print("estimated time remaining:", ((time.time() - start)/(count-skip))*(total - count)/60, "min")
        else:
            skip += 1
        count += 1
    except:
        break
for key in list(holds2.keys()):
    try:
        all_dots2[key] = get_alldots(holds2[key])
    except:
        print(key, "acting fucky")
dict_saver(holds2, "holds2")
dict_saver(all_dots2, "newdots")
'''