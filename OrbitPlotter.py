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
import time

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
    #print(time, idx, val, "yo")
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

def plotvalue(data, value, vsphase=False, linefit=True, start=0, end=-1):
    '''
    Parameters
    ----------
    data : dictionary
        the thing
    value : string
        thing to plot
    vsphase : bool, optional
        decide whether you're plotting against coordinate time or phase. The default is False, which corresponds to time.
    linefit : bool, optional
        toggle linear fitting. Defaults to True
    start : int, optional
        starting time or phase. The default is 0.
    end : int, optional
        ending time or phase. The default is -1.

    Returns
    -------
    bool
        True!

    '''

    termdict = {"time": [data["time"], "Coordinate Time"],
                "radius": [data["pos"][:,0], "Radius"],
                "theta": [data["pos"][:,1], "Theta"],
                "phase": [data["pos"][:,2]/(2*np.pi), "Phi"],
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
                "radial_freq": [data["freqs"][:, 0], "Radial Frequency"],
                "theta_freq": [data["freqs"][:, 1], "Theta Frequency"],
                "phi_freq": [data["freqs"][:, 2], "Phi Frequency"],
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
            if vsphase == True:
                title = "%s vs Phase"%(termdict[value][1])
                xvals = termdict["phase"][0][start:end]
                yvals = termdict[value][0][start:end]
            else:
                title = "%s vs Time"%(termdict[value][1])
                to = get_index(data["time"], start)
                if end > 0.0:
                    tf = get_index(data["time"], end)
                else:
                    tf = len(data["time"])
                xvals = termdict["time"][0][to:tf]
                yvals = termdict[value][0][to:tf]
        elif len(termdict[value][0]) == len(data["tracktime"]):
            to = get_index(data["tracktime"], start)
            if end > 0.0:
                tf = get_index(data["tracktime"], end)
            else:
                tf = len(data["tracktime"])
            if vsphase == True:
                title = "%s vs Phase"%(termdict[value][1])
                newphase = np.interp(data["tracktime"], data["time"], termdict["phase"][0])
                xvals = newphase[to:tf]
                yvals = termdict[value][0][to:tf]
            else:
                title = "%s vs Time"%(termdict[value][1])
                xvals = data["tracktime"][to:tf]
                yvals = termdict[value][0][to:tf]
        elif value == "omega":
            to = get_index(data["otime"], start)
            if end > 0.0:
                tf = get_index(data["otime"], end)
            else:
                tf = len(data["otime"])
            if vsphase == True:
                title = "%s vs Phase"%(termdict[value][1])
                newphase = np.interp(data["otime"], data["time"], termdict["phase"][0])
                xvals = newphase[to:tf]
                yvals = termdict[value][0][to:tf]
            else:
                title = "%s vs Time"%(termdict[value][1])
                xvals = data["otime"][to:tf]
                yvals = termdict[value][0][to:tf]
        ax.plot(xvals, yvals)
        if linefit == True:
            stuff = np.polyfit(xvals, yvals, 1)
            ax.plot(xvals, np.polyval(stuff, xvals), linestyle="dashed", label="Slope: {res:.3e}".format(res=stuff[0]))
            ax.legend()
        ax.set_title(title)
        
    else:
        print("Not a valid plottable. Chose one of the following:")
        for name in termdict:
            print("'" + name + "':", termdict[name][1])
    return True
    
def plotvalue2(datalist, value, vsphase=False, linefit=True, start=0, end=-1, xscale='linear', yscale='linear'):
    '''
    Parameters
    ----------
    data : single dict OR list/array of dicts
        orbit dictionar(y/ies). Inputting a single dict will turn it into a list.
    value : string
        variable to plot
    vsphase : bool, optional
        decide whether you're plotting against coordinate time or phase. The default is False, which corresponds to time.
    linefit : bool, optional
        toggle linear fitting. Defaults to True
    start : int, optional
        starting time or phase. The default is 0.
    end : int, optional
        ending time or phase. The default is -1.

    Returns
    -------
    bool
        True!

    '''
    # The time thing becomes an issue, since I'm using geometric time that could be different for each orbit
    # Although now that I think about it that's been an issue from the beginning
    # Hadn't even considered it until now
    # Actually it's all based on the central body so?? Shut up??
    
    if type(datalist) != list:
        datalist = [datalist]
    fig, ax = plt.subplots()
    colors = list(mcolors.TABLEAU_COLORS)
    for thing in range(len(datalist)):
        data = datalist[thing]
        # ["value": [location in data dict, Value name, extra bit if timing is weird]]
        termdict = {"time": [data["time"], "Coordinate Time"],
                    "radius": [data["pos"][:,0], "Radius"],
                    "theta": [data["pos"][:,1], "Theta"],
                    "phase": [data["pos"][:,2]/(2*np.pi), "Phase"],
                    "true_anom": [data["true_anom"], "True Anomaly"],
                    "r0": [data["r0"], "Semimajor Axis"],
                    "pot_min": [data["pot_min"], "Effective Potential Minimum"],
                    "ecc": [data["e"], "Eccentricity"],
                    "semilat": [data["r0"]*(1 - data["e"]**2), "Semilatus-Rectum"],
                    "inc": [data["inc"], "Inclination"],
                    "periapse": [data["it"], "Periapse"],
                    "apoapse": [data["ot"], "Apoapse"],
                    "omega": [data["omega"], "Phi Position of Periapse", "otime"],
                    "otime": [data["otime"], "Time of Periapse", "otime"],
                    "omegadot": [np.diff(data["omega"])/np.diff(data["otime"]), "Advance of Periapse", "odottime"],
                    "odottime": [0.5*data["otime"][:-1] + 0.5*data["otime"][1:], "Periadvance time", "odottime"],
                    "asc_node": [data["asc_node"], "Phi Position of Ascending Node", "asc_node_time"],
                    "asc_node_time": [data["asc_node_time"], "Time of Ascending Node", "asc_node_time"],
                    "semi_maj": [0.5*(data["it"] + data["ot"]), "Semimajor Axis"],
                    "semi_lat": [0.5*(data["it"] + data["ot"])*(1 - data["e"]**2), "Semilatus Rectum"],
                    "radial_v": [data["all_vel"][:,1], "Radial Velocity"],
                    "theta_v": [data["all_vel"][:,2], "Theta Velocity"],
                    "phi_v": [data["all_vel"][:,3], "Phi Velocity"],
                    "total_v": [data["vel"], "Velocity"],
                    "radial_freq": [data["freqs"][:, 0], "Radial Frequency"],
                    "theta_freq": [data["freqs"][:, 1], "Theta Frequency"],
                    "phi_freq": [data["freqs"][:, 2], "Phi Frequency"],
                    "energy": [data["energy"], "Specific Energy"],
                    "l_momentum": [data["phi_momentum"], "Specific Angular Momentum"],
                    "carter": [data["carter"], "Carter Constant"],
                    "qarter": [data["qarter"], "Carter Constant (Unnormalized)"],
                    "l_momentumx": [data["Lx_momentum"], "Specific Angular Momentum (x-component)"],
                    "l_momentumy": [data["Ly_momentum"], "Specific Angular Momentum (y-component)"],
                    "l_momentumz": [data["Lz_momentum"], "Specific Angular Momentum (z-component)"]}
        
        if (type(value) == str) and (value in termdict):
            if len(termdict[value][0]) == len(data["time"]):
                if vsphase == True:
                    title = "%s vs Phase"%(termdict[value][1])
                    xvals = termdict["phase"][0][start:end]
                    yvals = termdict[value][0][start:end]
                else:
                    title = "%s vs Time"%(termdict[value][1])
                    to = get_index(data["time"], start)
                    if end > 0.0:
                        tf = get_index(data["time"], end)
                    else:
                        tf = len(data["time"])
                    xvals = termdict["time"][0][to:tf]
                    yvals = termdict[value][0][to:tf]
            elif len(termdict[value][0]) == len(data["tracktime"]):
                to = get_index(data["tracktime"], start)
                if end > 0.0:
                    tf = get_index(data["tracktime"], end)
                else:
                    tf = len(data["tracktime"])
                if vsphase == True:
                    title = "%s vs Phase"%(termdict[value][1])
                    newphase = np.interp(data["tracktime"], data["time"], termdict["phase"][0])
                    xvals = newphase[to:tf]
                    yvals = termdict[value][0][to:tf]
                else:
                    title = "%s vs Time"%(termdict[value][1])
                    xvals = data["tracktime"][to:tf]
                    yvals = termdict[value][0][to:tf]
            else:
                timething = termdict[termdict[value][2]][0]
                #change otime to timething
                to = get_index(timething, start)
                if end > 0.0:
                    tf = get_index(timething, end)
                else:
                    tf = len(timething)
                if vsphase == True:
                    title = "%s vs Phase"%(termdict[value][1])
                    newphase = np.interp(timething, data["time"], termdict["phase"][0])
                    xvals = newphase[to:tf]
                    yvals = termdict[value][0][to:tf]
                else:
                    title = "%s vs Time"%(termdict[value][1])
                    xvals = timething[to:tf]
                    yvals = termdict[value][0][to:tf]
            ax.plot(xvals, yvals, color=colors[thing%len(colors)])
            if linefit == True:
                stuff = np.polyfit(xvals, yvals, 1)
                ax.plot(xvals, np.polyval(stuff, xvals), linestyle="dashed", label=data["name"]+": {res:.3e}".format(res=stuff[0]), color=colors[thing%len(colors)])
                ax.legend()
        
        else:
            print("Not a valid plottable. Chose one of the following:")
            for name in termdict:
                print("'" + name + "':", termdict[name][1])
            return False
        
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title)
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

def orthoplots(datalist, ortho=False, zoom=1.0, start=0.0, end=-1.0, leg=True, ele=30, azi=-60, cb=False, stitch=False):
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
            flipto = get_index(data["tracktime"], start)
            if end > 0.0:
                tf = get_index(data["time"], end)
                fliptf = get_index(data["tracktime"], end)
            else:
                tf = get_index(data["time"], data["time"][-1])
                fliptf = get_index(data["tracktime"], data["tracktime"][-1])

            carts = np.array([sph2cart(pos)/(10**(3*scaler)) for pos in data["pos"][to:tf]])
            cartsxy = np.copy(carts)
            cartsxz = np.copy(carts)
            cartsyz = np.copy(carts)
            flippoints = np.array([sph2cart(pos)/(10**(3*scaler)) for pos in data["pos"][data["trackix"][flipto:fliptf].astype(int)]])
            flipsxy = np.copy(flippoints)
            flipsxz = np.copy(flippoints)
            flipsyz = np.copy(flippoints)
            
            if cb == True:
                rb = (1 + (1 - data["spin"]**2)**(0.5))/(10**(3*scaler))
                circle1 = plt.Circle((0, 0), rb, color='black')
                circle2 = plt.Circle((0, 0), rb, color='black')
                circle3 = plt.Circle((0, 0), rb, color='black')
                ax_list[0].add_patch(circle1)
                ax_list[1].add_patch(circle2)
                ax_list[2].add_patch(circle3)
                
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
                
                try:
                    flipB_ = A - np.pi/2 + np.arcsin((rb/data["pos"][data["trackix"][flipto:fliptf].astype(int),0])*np.sin(A))
                    flipcondxy = (np.arccos(np.dot(flippoints, Xxy)/data["pos"][data["trackix"][flipto:fliptf].astype(int),0]) < np.pi - flipB_)
                    flipcondxz = (np.arccos(np.dot(flippoints, Xxz)/data["pos"][data["trackix"][flipto:fliptf].astype(int),0]) < np.pi - flipB_)
                    flipcondyz = (np.arccos(np.dot(flippoints, Xyz)/data["pos"][data["trackix"][flipto:fliptf].astype(int),0]) < np.pi - flipB_)
                    flipsxy = np.array([flippoints[i] if flipcondxy[i] == True else [np.nan, np.nan, np.nan] for i in range(len(flipcondxy))])
                    flipsxz = np.array([flippoints[i] if flipcondxz[i] == True else [np.nan, np.nan, np.nan] for i in range(len(flipcondxy))])
                    flipsyz = np.array([flippoints[i] if flipcondyz[i] == True else [np.nan, np.nan, np.nan] for i in range(len(flipcondxy))])
                except:
                    pass
                
            ax_list[0].plot(cartsxy[:,0], cartsxy[:,1], label=data["name"], zorder=10)  #XY Plot
            ax_list[1].plot(cartsxz[:,0], cartsxz[:,2], label="_nolabel_", zorder=10)  #XZ Plot
            ax_list[2].plot(cartsyz[:,1], cartsyz[:,2], label="_nolabel_", zorder=10)  #ZY Plot
            if stitch == True:
                ax_list[0].scatter(flipsxy[:,0], flipsxy[:,1], label=data["name"], zorder=9, marker="*", s=300)  #XY Plot
                ax_list[1].scatter(flipsxz[:,0], flipsxz[:,2], label="_nolabel_", zorder=9, marker="*", s=300)  #XZ Plot
                ax_list[2].scatter(flipsyz[:,1], flipsyz[:,2], label="_nolabel_", zorder=9, marker="*", s=300) 
            
        
        if datalist[0]["inputs"][-1] == "grav":
            unit = "Geometric Units"
        elif datalist[0]["inputs"][-1] == "mks":
            unit = "Meters"
        elif datalist[0]["inputs"][-1] == "cgs":
            unit = "Centimeters"
        ax1.set(ylabel="Y")
        ax2.set_axis_off()
        ax3.set(xlabel="X", ylabel="Z")
        ax4.set( xlabel="Y")
        ax2.set_xlim(-cap/zoom, cap/zoom)
        ax2.set_ylim(-cap/zoom, cap/zoom)
        ax2.set_aspect('equal')
        ax2.text(0, 0.62*cap/zoom, "Orthographic View", fontsize=20, ha="center", va="top")
        ax2.text(0, 0.40*cap/zoom, "Scale: " + scale_word + unit, fontsize=15, ha="center", va="top")
        fig.subplots_adjust(wspace=0, hspace=0)
        if leg == True:
            legend = fig.legend(loc=(0.75,0.5))
            hor_ratio = legend.get_window_extent().width/ fig.get_window_extent().width
            ver_ratio = legend.get_window_extent().height/ fig.get_window_extent().height
            legend.set_bbox_to_anchor(bbox=(0.666 - 0.5*hor_ratio, 0.55 - 0.5*ver_ratio))
        
        
    else:
        fig = plt.figure(figsize=(10,12))
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=ele, azim=azi)
        
        rbound = 0
        for data in datalist:
            to = get_index(data["time"], start)
            flipto = get_index(data["tracktime"], start)
            if end > 0.0:
                tf = get_index(data["time"], end)
                fliptf = get_index(data["tracktime"], end)
            else:
                tf = get_index(data["time"], data["time"][-1])
                fliptf = get_index(data["tracktime"], data["tracktime"][-1])
                
            rbound = max(max(data["pos"][to:tf,0])*1.05, rbound)
            carts = np.array([sph2cart(pos) for pos in data["pos"][to:tf]])
            flippoints = np.array([sph2cart(pos) for pos in data["pos"][data["trackix"][flipto:fliptf].astype(int)]])
            
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
                
                try:
                    flipB_ = A - np.pi/2 + np.arcsin((rb/data["pos"][data["trackix"][flipto:fliptf].astype(int),0])*np.sin(A))
                    flipblockedcheck = np.arccos(np.dot(flippoints, X)/data["pos"][data["trackix"][flipto:fliptf].astype(int),0]) < np.pi - flipB_
                    flipboundboxcheck = [False not in piece for piece in np.abs(flippoints) <= rbound/zoom]
                    flipcond = np.logical_and(flipblockedcheck, flipboundboxcheck)
                    flippoints = np.array([flippoints[i] if flipcond[i] == True else [np.nan, np.nan, np.nan] for i in range(len(flipcond))])
                except:
                    pass
                
            ax.plot3D(carts[:, 0], carts[:, 1], carts[:, 2], label=data["name"], zorder=10)
            if stitch == True:
                ax.scatter(flippoints[:, 0], flippoints[:, 1], flippoints[:, 2], label=data["name"], zorder=9, marker="*", s=300)
            
        
        ax.set(xlim3d=(-rbound/zoom, rbound/zoom), xlabel='X')
        ax.set(ylim3d=(-rbound/zoom, rbound/zoom), ylabel='Y')
        ax.set(zlim3d=(-rbound/zoom, rbound/zoom), zlabel='Z')
        ax.set_box_aspect((rbound, rbound, rbound))
        if leg == True:
            ax.legend()
    return fig

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
            tf1 = len(data["time"]) #get_index(data["time"], data["time"][-1])
        else: 
            tf1 = get_index(data["time"], end)
            
        to2 = get_index(data["tracktime"], start)
        if end == -1:
            tf2 = len(data["time"]) #get_index(data["tracktime"], data["tracktime"][-1])
        else: 
            tf2 = get_index(data["tracktime"], end)
        
        min_time = min(data["time"][to1], min_time)
        max_time = max(data["time"][tf1], max_time) if end != -1 else max(data["time"][-1], max_time)
        
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

def ani_thing3(data, name=False, ortho=False, zoom=1.0, ele=30, azi=-60, scroll=True, cb=True, numturns=10, fid=1):
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
    scroll : bool
        determines whether the bounds of the plot will shift to track the orbit during its evolution
        defaults to True
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
    
    if name == False:
        name=data["name"][:10] + time.strftime("%y_%m_%d_%H", time.localtime())
    
    int_sphere, int_time = mm.interpolate(data["pos"], data["time"], supress = False)
    X = int_sphere[:,0]*np.sin(int_sphere[:,1])*np.cos(int_sphere[:,2])
    Y = int_sphere[:,0]*np.sin(int_sphere[:,1])*np.sin(int_sphere[:,2])
    Z = int_sphere[:,0]*np.cos(int_sphere[:,1])

    num_steps = int(100*fid)
    
    #print(np.where(data["pos"][:,2] > 2*np.pi))
    turn_ind = np.where(data["pos"][:,2] > 2*np.pi)[0][0]
    first_turn = get_index(int_time, data["time"][turn_ind])
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
            
        if scroll == False:
            ax.set(xlim3d=(-rbound, rbound), xlabel='X')
            ax.set(ylim3d=(-rbound, rbound), ylabel='Y')
            ax.set(zlim3d=(-rbound, rbound), zlabel='Z')
            ax.set_box_aspect((rbound, rbound, rbound))
        
        def update_line(num, xdata, ydata, zdata, line):
            full = int(len(xdata)/num_steps)
            beg = max(0, int(full*num - first_turn*numturns))
            #print(len(xdata), num_steps, full, num)
            #print(beg, full*num, full*num - beg, first_turn)
            line.set_data_3d(xdata[beg:full*num], ydata[beg:full*num], zdata[beg:full*num])
            
            if scroll == True:
                try:
                    rbound = max(max((xdata[beg:full*num]**2 + ydata[beg:full*num]**2 + zdata[beg:full*num]**2)**0.5)*1.05, 3)
                except:
                    rbound = max(max((xdata**2 + ydata**2 + zdata**2)**0.5)*1.05, 3)
                
                try:
                    ax.set(xlim3d=(-rbound, rbound), xlabel='X')
                    ax.set(ylim3d=(-rbound, rbound), ylabel='Y')
                    ax.set(zlim3d=(-rbound, rbound), zlabel='Z')
                    ax.set_box_aspect((rbound, rbound, rbound))
                except:
                    pass
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
        
        if scroll == False:
            rbound = max(data["pos"][:,0])*1.05/zoom
            ax2.set_xlim(-rbound, rbound)
            ax2.set_ylim(-rbound, rbound)
            ax2.set_aspect('equal')
        
        def update_line(num, xdata, ydata, zdata, line):
            full = len(xdata)//num_steps
            beg = max(0, int(full*num - first_turn*numturns))
            end = int(np.round(num*len(xdata)/num_steps))
            line[0].set_data(xdata[beg:end], ydata[beg:end])
            line[1].set_data(xdata[beg:end], zdata[beg:end])
            line[2].set_data(ydata[beg:end], zdata[beg:end])
            
            if scroll == True:
                try:
                    rbound = np.nanmax(np.nanmax((xdata[beg:end]**2 + ydata[beg:end]**2 + zdata[beg:end]**2)**0.5)*1.05, 3) 
                except:
                    rbound = np.nanmax(np.nanmax((xdata**2 + ydata**2 + zdata**2)**0.5)*1.05, 3)
    
                ax2.set_xlim(-rbound, rbound)
                ax2.set_ylim(-rbound, rbound)
                ax2.set_aspect('equal')

            return line
        
    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_line, frames=num_steps + 10, fargs=(X, Y, Z, line), interval=10)
    
    #HEY, CAN YOU JUST MAKE MULTIPLE ANIMATION OBJECTS ON THE SAME FIGURE??? EXPERIMENT ON SOMETHING SIMPLE
    
    cwd = os.getcwd()
    f = os.path.join(cwd, name + ".gif")
    writergif = animation.PillowWriter(fps=10)
    ani.save(f, writer=writergif)
    
    plt.show()
    print(name + '.gif')
    return True

def ani_test():
    # initializing a figure in  
    # which the graph will be plotted 
    fig = plt.figure()  
       
    # marking the x-axis and y-axis 
    axis = plt.axes(xlim =(0, 4),  
                    ylim =(-2, 2))  
      
    # initializing a line variable 
    line, = axis.plot([], [], lw = 3) 
    line2, = axis.plot([], [], lw = 3) 
       
    # data which the line will  
    # contain (x, y) 
    def init():  
        line.set_data([], []) 
        line2.set_data([], []) 
        return line, line2
       
    def animate(i): 
        x = np.linspace(0, 4, 1000) 
       
        # plots a sine graph 
        y = np.sin(2 * np.pi * (x - 0.01 * i)) 
        line.set_data(x, y) 
        y2 = np.cos(2 * np.pi * (x - 0.03 * i)) 
        line2.set_data(x, y2) 
          
        return line, line2, 
       
    anim = animation.FuncAnimation(fig, animate, init_func = init, 
                         frames = 200, interval = 20, blit = True)  
       
    anim.save('continuousSineWave.gif', fps = 30) 
    return 0

def ani_thing4(datalist, name=False, ortho=False, zoom=1.0, ele=30, azi=-60, scroll=True, cb=True, numturns=10, fid=1):
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
    scroll : bool
        determines whether the bounds of the plot will shift to track the orbit during its evolution
        defaults to True
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
    
    if name == False:
        name="False"

    num_steps = int(100*fid)
    
    #print(np.where(data["pos"][:,2] > 2*np.pi))
    #print(first_turn)
    
    if ortho == False:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=ele, azim=azi)
        lines, paths = [], []
        
        rbound = 0
        for data in datalist:
            lines.append(ax.plot([], [], [], zorder=10)[0])
            int_sphere, int_time = mm.interpolate(data["pos"], data["time"], supress = False)
            X = int_sphere[:,0]*np.sin(int_sphere[:,1])*np.cos(int_sphere[:,2])
            Y = int_sphere[:,0]*np.sin(int_sphere[:,1])*np.sin(int_sphere[:,2])
            Z = int_sphere[:,0]*np.cos(int_sphere[:,1])
            paths.append(np.array([X, Y, Z, int_time]))
            rbound = max(rbound, max(data["pos"][:,0]))
            
        if cb == True:
            rb = 1 + (1 - datalist[0]["spin"]**2)**(0.5)
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
            A = np.pi - np.arctan(20*rbound/(rb*zoom))
            
            #hide paths behind black hole
            print(paths[0])
            for i in range(len(paths)):
                carts = np.transpose(paths[i][:3])
                r = np.sqrt(carts[:,0]**2 + carts[:,1]**2 + carts[:,2]**2)
                B_ = A - np.pi/2 + np.arcsin((rb/r)*np.sin(A))
                #print(len(carts), len(V), len(r), len(B_))
                cond = (np.arccos(np.dot(carts, V)/r) < np.pi - B_)
                #print(list(cond).count(False))
                #newpath = 
                paths[i] = np.append(np.transpose(np.array([carts[i] if cond[i] == True else [np.nan, np.nan, np.nan] for i in range(len(cond))])), [paths[i][-1]], axis=0)
            print(paths[0])
            
        if scroll == False:
            ax.set(xlim3d=(-rbound, rbound), xlabel='X')
            ax.set(ylim3d=(-rbound, rbound), ylabel='Y')
            ax.set(zlim3d=(-rbound, rbound), zlabel='Z')
            ax.set_box_aspect((rbound, rbound, rbound))
        
        def update_lines(num, paths, lines):
            rbound = 0
            for i in range(len(lines)):
                turn_ind = np.where(datalist[i]["pos"][:,2] > 2*np.pi)[0][0]
                #print(datalist[i]["time"][turn_ind])
                first_turn = get_index(paths[i][-1], datalist[i]["time"][turn_ind])
                #rbound = max(datalist[i]["pos"][:,0])*1.05/zoom
                full = int(len(paths[i][0])/num_steps)
                beg = max(0, int(full*num - first_turn*numturns)) #gonna have to change this whole turns thing, maybe time based?
                lines[i].set_data_3d(paths[i][0,beg:full*num], paths[i][1,beg:full*num], paths[i][2,beg:full*num])
                rbound = max(rbound, max(datalist[i]["pos"][:,0]))*1.05/zoom
            
            if scroll == True:
                try:
                    ax.set(xlim3d=(-rbound, rbound), xlabel='X')
                    ax.set(ylim3d=(-rbound, rbound), ylabel='Y')
                    ax.set(zlim3d=(-rbound, rbound), zlabel='Z')
                    ax.set_box_aspect((rbound, rbound, rbound))
                except:
                    pass
            return lines
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
        
        if scroll == False:
            rbound = max(data["pos"][:,0])*1.05/zoom
            ax2.set_xlim(-rbound, rbound)
            ax2.set_ylim(-rbound, rbound)
            ax2.set_aspect('equal')
        
        def update_line(num, xdata, ydata, zdata, line):
            full = len(xdata)//num_steps
            beg = max(0, int(full*num - first_turn*numturns))
            end = int(np.round(num*len(xdata)/num_steps))
            line[0].set_data(xdata[beg:end], ydata[beg:end])
            line[1].set_data(xdata[beg:end], zdata[beg:end])
            line[2].set_data(ydata[beg:end], zdata[beg:end])
            
            if scroll == True:
                try:
                    rbound = np.nanmax(np.nanmax((xdata[beg:end]**2 + ydata[beg:end]**2 + zdata[beg:end]**2)**0.5)*1.05, 3) 
                except:
                    rbound = np.nanmax(np.nanmax((xdata**2 + ydata**2 + zdata**2)**0.5)*1.05, 3)
    
                ax2.set_xlim(-rbound, rbound)
                ax2.set_ylim(-rbound, rbound)
                ax2.set_aspect('equal')

            return line
        
    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_lines, frames=num_steps + 10, fargs=(paths, lines), interval=10)
    
    #HEY, CAN YOU JUST MAKE MULTIPLE ANIMATION OBJECTS ON THE SAME FIGURE??? EXPERIMENT ON SOMETHING SIMPLE
    
    cwd = os.getcwd()
    f = os.path.join(cwd, name + ".gif")
    writergif = animation.PillowWriter(fps=10)
    ani.save(f, writer=writergif)
    
    plt.show()
    print(name + '.gif')
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
    ecc = (rx - rn)/(rx + rn)
    print(rx, rn, ecc)
    print(r0, bloh, bluh)
    if -1 in rbounds:
        p = 1/(1 - E**2)
        rbounds = np.linspace(0.0, rx*1.05, num=100)
    else:
        rbounds = np.linspace(rbounds[0]*np.ones((len(rn))), rbounds[-1]*np.ones((len(rx))), num=100)

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Radius (Geometric Units)")
    ax1.set_ylabel("Effective Potential")
    #ax1.set_title("Effective Potential")
    ax1.plot(rbounds, rbounds*0.0)
    ax1.plot(rbounds, -R(rbounds))
    print(R(np.array([r0, bloh, bluh])))
    ext = False
    if r0 >= rbounds[0]:
        ax1.vlines(r0, -R(r0), 0)
        ax1.scatter(r0, 0.0, label="Potential Minimum")
        ext = True
    if bloh >= rbounds[0] and abs(R(bloh)) < 1e-5:
        ax1.vlines(bloh, -R(bloh), 0)
        ax1.scatter(bloh, 0.0, marker="*", label="Unstable Circular orbit")
        ext = True
    if ext == True:
        ax1.legend()
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
        print("zah",rbounds2[0],rbounds2[-1])
        maxbounds = [min(maxbounds[0], rbounds2[0]), max(maxbounds[1], rbounds2[-1])]
        print(maxbounds)
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
    print(np.shape(X), np.shape(Y), np.shape(Z))
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

    carts = np.array([sph2cart(pos) for pos in data["pos"]])
    #carts = np.array([sph2cart(pos)/(10**(3*scaler)) for pos in data["pos"]])
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
    if data["inputs"][-1] == "grav":
        freq_unit = '(G\u209C\u207B\u00B9)'
    else:
        freq_unit = '(Hz)'
    #ax2.set(xlabel='Waveform Frequency ' + freq_unit)
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
    ax2.set_title('Waveform Frequency ' + freq_unit)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.grid()
    ax2.legend()
    return 0

def justfourier(data, start=0, end=-1):
    cap = max(data["pos"][:,0])
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
    
    fig, ax = plt.subplots()
    h_plus = 2.0/N * np.abs(yf1[0:N//2])
    h_min = 2.0/N * np.abs(yf2[0:N//2])
    ax.plot(xf, h_plus/max(h_plus), label = "h+")
    ax.plot(xf, h_min/max(h_min), label = "hx")
    #plt.setp(ax3.get_yticklabels(), visible=False)
    #plt.setp(ax4.get_yticklabels(), visible=False)
    ax.set_title("Waveform Fourier Transform")
    #ax.set_title('Waveform Frequency ' + freq_unit)
    ax.set_yscale('log')
    ax.set_xscale('log')
    if data["inputs"][-1] == "grav":
        freq_unit = "(Geometric Units)"#'(G\u209C\u207B\u00B9)'
    else:
        freq_unit = '(Hz)'
    ax.set_xlabel("Frequency " + freq_unit)
    ax.set_ylabel("Relative Intensity")
    ax.grid()
    ax.legend()
    return 0
    

def wavelething(data):
    #It hates you and it's not even what you want, leave it alone
    rad = data["pos"][0,0]
    waves, tim = mm.full_transform(data, rad*100)
    samper = tim[1]-tim[0]
    print(samper)
    period = np.real(2*np.pi*np.sqrt(data["r0"][0]**3))
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

def peters_compare(data, plot=True):
    timen = data["tracktime"]
    mu = data["inputs"][2]
    ecc = data["e"]
    r0 = data["r0"]
    dadt = (-64/5)*mu*(1+mu)*(1 + (73/24)*(ecc**2) + (37/96)*(ecc**4))/((r0**3)*((1-ecc**2)**(7/2)))
    dedt = (-304/15)*ecc*mu*(1+mu)*(1 + (121/304)*(ecc**2))/((r0**4)*((1-ecc**2)**(5/2)))
    dade = (12/19)*(r0/ecc)*(1 + (73/24)*(ecc**2) + (37/96)*(ecc**4))/((1-ecc**2)*(1 + (121/304)*(ecc**2)))
    con = r0[0]/((ecc[0]**(12/19))*(1 + (121/304)*(ecc[0]**2))/(1-ecc[0]**2))
    aofe = con*((ecc**(12/19))*(1 + (121/304)*(ecc**2))/(1-ecc**2))
    
    num = 10
    modtime = 0.5*(timen[:-num] + timen[num:])
    moddadt = (r0[num:] - r0[:-num])/(timen[num:] - timen[:-num])
    moddedt = (ecc[num:] - ecc[:-num])/(timen[num:] - timen[:-num])
    moddade = (r0[num:] - r0[:-num])/(ecc[num:] - ecc[:-num])
    
    if plot==True:
        fig1 = plt.figure()
        ax11 = fig1.add_subplot(111, label="1")
        ax11.set_title("r0")
        ax12 = fig1.add_subplot(111, label="2", frame_on=False)
        ax12.set_xticks([])
        ax12.yaxis.tick_right()
        ax12.set_ylabel("Percent Difference", color="C3")
        ax12.yaxis.set_label_position('right')
        ax11.plot(modtime, moddadt, label="true r0 deriv")
        ax11.plot(timen, dadt, label="peters r0 deriv")
        ax12.plot(modtime, 100*np.abs(moddadt - np.interp(modtime, timen, dadt))/np.abs(np.interp(modtime, timen, dadt)), label="dadt %diff", color="C3")
        
        fig2 = plt.figure()
        ax21 = fig2.add_subplot(111, label="1")
        ax21.set_title("e")
        ax22 = fig2.add_subplot(111, label="2", frame_on=False)
        ax22.set_xticks([])
        ax22.yaxis.tick_right()
        ax22.set_ylabel("Percent Difference", color="C3")
        ax22.yaxis.set_label_position('right')
        ax21.plot(modtime, moddedt, label="true ecc deriv")
        ax21.plot(timen, dedt, label="peters ecc deriv")
        ax22.plot(modtime, 100*np.abs(moddedt - np.interp(modtime, timen, dedt))/np.abs(np.interp(modtime, timen, dedt)), label="dadt %diff", color="C3")
        
        fig3 = plt.figure()
        ax31 = fig3.add_subplot(111, label="1")
        ax31.set_title("r0/e")
        ax32 = fig3.add_subplot(111, label="2", frame_on=False)
        ax32.set_xticks([])
        ax32.yaxis.tick_right()
        ax32.set_ylabel("Percent Difference", color="C3")
        ax32.yaxis.set_label_position('right')
        ax31.plot(0.5*(ecc[num:] + ecc[:-num]), moddade, label="true dade")
        ax31.plot(ecc, dade, label="peters dade")
        ax32.plot((ecc[num:] - ecc[:-num]), 100*np.abs(moddade - np.interp((ecc[num:] - ecc[:-num]), ecc, dade))/np.abs(np.interp((ecc[num:] - ecc[:-num]), ecc, dade)), label="dade %diff", color="C3")
    
        fig4 = plt.figure()
        ax41 = fig4.add_subplot(111, label="1")
        ax41.set_title("aofe")
        ax42 = fig4.add_subplot(111, label="2", frame_on=False)
        ax42.set_xticks([])
        ax42.yaxis.tick_right()
        ax42.set_ylabel("Percent Difference", color="C3")
        ax42.yaxis.set_label_position('right')
        ax41.plot(ecc, r0, label="true ecc deriv")
        ax41.plot(ecc, aofe, label="peters ecc deriv")
        ax42.plot(ecc, 100*np.abs(r0 - aofe)/np.abs(aofe), label="aofe %diff", color="C3")
        plt.show()
    else:
        return [np.mean(100*np.abs(moddadt - np.interp(modtime, timen, dadt))/np.abs(np.interp(modtime, timen, dadt))),
                np.mean(100*np.abs(moddedt - np.interp(modtime, timen, dedt))/np.abs(np.interp(modtime, timen, dedt))),
                np.mean(100*np.abs(moddade - np.interp((ecc[num:] - ecc[:-num]), ecc, dade))/np.abs(np.interp((ecc[num:] - ecc[:-num]), ecc, dade))),
                np.mean(100*np.abs(r0 - aofe)/np.abs(aofe))]

def peters_compare2(data, plot=True):
    timen = data["tracktime"]
    mu = data["inputs"][2]
    ecc = data["e"]
    r0 = data["r0"]
    dadt = (-64/5)*mu*(1+mu)*(1 + (73/24)*(ecc**2) + (37/96)*(ecc**4))/((r0**3)*((1-ecc**2)**(7/2)))
    dedt = (-304/15)*ecc*mu*(1+mu)*(1 + (121/304)*(ecc**2))/((r0**4)*((1-ecc**2)**(5/2)))
    dade = (12/19)*(r0/(ecc+1e-15))*(1 + (73/24)*(ecc**2) + (37/96)*(ecc**4))/((1-ecc**2)*(1 + (121/304)*(ecc**2)))
    con = r0[0]/((ecc[0]**(12/19))*(1 + (121/304)*(ecc[0]**2))/(1-ecc[0]**2))
    aofe = con*((ecc**(12/19))*(1 + (121/304)*(ecc**2))/(1-ecc**2))
    
    calcr0 = r0[:-1] + dadt[:-1]*np.diff(timen)
    calce = ecc[:-1] + dedt[:-1]*np.diff(timen)
    calcaofe = r0[:-1] + dade[:-1]*np.diff(ecc)
    
    calc2r0, calc2e, calc2aofe = np.array([r0[0]]), np.array([ecc[0]]), np.array([r0[0]])
    for dt in np.diff(timen):
        r, e, a = calc2r0[-1], calc2e[-1], calc2aofe[-1]
        calc2r0 = np.append(calc2r0, r + dt*(-64/5)*mu*(1+mu)*(1 + (73/24)*(e**2) + (37/96)*(e**4))/((r**3)*((1-e**2)**(7/2))))
        calc2e = np.append(calc2e, e + dt*(-304/15)*e*mu*(1+mu)*(1 + (121/304)*(e**2))/((r**4)*((1-e**2)**(5/2))))
        calc2aofe = np.append(calc2aofe, a + (calc2e[-1] - calc2e[-2])*(12/19)*(a/e)*(1 + (73/24)*(e**2) + (37/96)*(e**4))/((1-e**2)**(1 + (121/304)*(e**2))))
    
    if plot==True:
        fig1 = plt.figure()
        ax11 = fig1.add_subplot(111, label="1")
        ax11.set_title("r0")
        ax12 = fig1.add_subplot(111, label="2", frame_on=False)
        ax12.set_xticks([])
        ax12.yaxis.tick_right()
        ax12.set_ylabel("Percent Difference", color="C3")
        ax12.yaxis.set_label_position('right')
        ax11.plot(timen[1:], r0[1:], label="r0")
        ax11.plot(timen[1:], calcr0, label="calcr0")
        ax11.plot(timen, calc2r0, label="calc2r0")
        ax12.plot(timen[1:], 100*np.abs(r0[1:] - calcr0)/calcr0, label="dadt %diff", color="C3")
        #ax12.plot(timen[1:], 100*np.abs(r0[1:] - calc2r0[1:])/calc2r0[1:], label="dadt %diff", color="C4")
        
        fig2 = plt.figure()
        ax21 = fig2.add_subplot(111, label="1")
        ax21.set_title("e")
        ax22 = fig2.add_subplot(111, label="2", frame_on=False)
        ax22.set_xticks([])
        ax22.yaxis.tick_right()
        ax22.set_ylabel("Percent Difference", color="C3")
        ax22.yaxis.set_label_position('right')
        ax21.plot(timen[1:], ecc[1:], label="e")
        ax21.plot(timen[1:], calce, label="calce")
        ax21.plot(timen, calc2e, label="calc2e")
        ax22.plot(timen[1:], 100*np.abs(ecc[1:] - calce)/calce, label="dadt %diff", color="C3")
        #ax22.plot(timen[1:], 100*np.abs(ecc[1:] - calc2e[1:])/calc2e[1:], label="dadt %diff", color="C4")
        
        fig3 = plt.figure()
        ax31 = fig3.add_subplot(111, label="1")
        ax31.set_title("r0/e")
        ax32 = fig3.add_subplot(111, label="2", frame_on=False)
        ax32.set_xticks([])
        ax32.yaxis.tick_right()
        ax32.set_ylabel("Percent Difference", color="C3")
        ax32.yaxis.set_label_position('right')
        ax31.plot(ecc[1:], aofe[1:], label="r0")
        ax31.plot(ecc[1:], calcaofe, label="calcr0")
        ax31.plot(ecc, calc2aofe, label="calc2r0")
        ax32.plot(ecc[1:], 100*np.abs(aofe[1:] - calcaofe)/calcaofe, label="dadt %diff", color="C3")
        #ax32.plot(ecc[1:], 100*np.abs(aofe[1:] - calc2aofe[1:])/calc2aofe[1:], label="dadt %diff", color="C4")
        '''
        fig4 = plt.figure()
        ax41 = fig4.add_subplot(111, label="1")
        ax41.set_title("aofe")
        ax42 = fig4.add_subplot(111, label="2", frame_on=False)
        ax42.set_xticks([])
        ax42.yaxis.tick_right()
        ax42.set_ylabel("Percent Difference", color="C3")
        ax42.yaxis.set_label_position('right')
        ax41.plot(ecc, r0, label="true ecc deriv")
        ax41.plot(ecc, aofe, label="peters ecc deriv")
        ax42.plot(ecc, 100*np.abs(r0 - aofe)/np.abs(aofe), label="aofe %diff", color="C3")
        '''
        plt.show()
    else:
        #print(dade)
        return ["{calc:.5e}, {calc2:.5e}".format(calc=np.mean(100*np.abs(r0[1:] - calcr0)/calcr0), calc2=np.mean(100*np.abs(r0[1:] - calc2r0[1:])/calc2r0[1:])),
                "{calc:.5e}, {calc2:.5e}".format(calc=np.mean(200*np.abs(ecc[1:] - calce)/(np.abs(ecc[1:]) + np.abs(calce))), calc2=np.mean(200*np.abs(ecc[1:] - calc2e[1:])/(np.abs(ecc[1:]) + np.abs(calc2e[1:])))),
                "{calc:.5e}, {calc2:.5e}".format(calc=np.mean(100*np.abs(aofe[1:] - calcaofe)/calcaofe), calc2=np.mean(100*np.abs(aofe[1:] - calc2aofe[1:])/calc2aofe[1:]))]

def get_peter_diffs(r0, ecc, mu):
    dadt = (-64/5)*mu*(1+mu)*(1 + (73/24)*(ecc**2) + (37/96)*(ecc**4))/((r0**3)*((1-ecc**2)**(7/2)))
    dedt = (-304/15)*ecc*mu*(1+mu)*(1 + (121/304)*(ecc**2))/((r0**4)*((1-ecc**2)**(5/2)))
    return np.array([dadt, dedt])

def new_RK(r0, ecc, mu, butcher, dt):
    k = [get_peter_diffs(r0, ecc, mu)]
    for i in range(len(butcher["nodes"])):                                        
        param = np.array([r0, ecc])                                                     
        for j in range(len(butcher["coeff"][i])):                                   
            param += np.array(butcher["coeff"][i][j] * dt * k[j])                   
        k.append(get_peter_diffs(*param, mu))                          
    new_state = np.array([r0, ecc])
    for val in range(len(k)):                                                     
        new_state += k[val] * butcher["weights"][val] * dt                       
    return new_state

def peters_comp(data, dt):
    mu = data["inputs"][2]
    perc = 1
    vals, T = [np.array([data["r0"][0], data["e"][0]])], [0.0]
    end = data["tracktime"][-1]
    state = vals[-1]
    while T[-1] < end:
        new_step = new_RK(*state, mu, mm.ck4, dt)
        vals.append(new_step)
        T.append(T[-1] + dt)
        if round(100*T[-1]/end) > perc:
            #print(T[-1])
            perc += 1
        state = np.copy(new_step)
    vals = np.array(vals)
    return T, vals[:,0], vals[:,1]

def compbig(data, dt=False):
    if dt == False:
        dt = data["tracktime"][-1]/1000.0
    T, r0, e = peters_comp(data, dt)
    check = np.concatenate((np.where(r0 <= 0)[0], np.where(e <= 0)[0]))
    if len(check) > 0:
        end = min(check)
        T = T[:end]
        r0 = r0[:end]
        e = e[:end]
    dataT = np.real(data["tracktime"])
    datar0 = np.real(data["r0"])
    datae = np.real(data["e"])
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, label="1")
    ax1.set_title(data["name"])
    ax1.plot(data["tracktime"], data["r0"])
    ax1.plot(T, r0)
    ax12 = fig1.add_subplot(111, label="2", frame_on=False)
    ax12.set_xticks([])
    ax12.yaxis.tick_right()
    ax12.set_ylabel("Percent Difference", color="C3")
    ax12.yaxis.set_label_position('right')
    ax12.plot(dataT, 100*np.abs(datar0 - np.interp(dataT, T, r0))/np.interp(dataT, T, r0), label="dadt %diff", color="C3")
    #
    fig2, ax2 = plt.subplots()
    ax2.plot(data["tracktime"], data["e"])
    ax2.plot(T, e)
    ax22 = fig2.add_subplot(111, label="2", frame_on=False)
    ax22.set_xticks([])
    ax22.yaxis.tick_right()
    ax22.set_ylabel("Percent Difference", color="C3")
    ax22.yaxis.set_label_position('right')
    ax22.plot(dataT, 200*np.abs(datae - np.interp(dataT, T, e))/(np.abs(datae) + np.abs(np.interp(dataT, T, e))), label="dadt %diff", color="C3")
    #
    fig3, ax3 = plt.subplots()
    ax3.plot(data["e"], data["r0"])
    ax3.plot(e, r0)
    ax32 = fig3.add_subplot(111, label="2", frame_on=False)
    ax32.set_xticks([])
    ax32.yaxis.tick_right()
    ax32.set_ylabel("Percent Difference", color="C3")
    ax32.yaxis.set_label_position('right')
    ax32.plot(datae, 100*np.abs(datar0 - np.interp(dataT, T, r0))/np.interp(dataT, T, r0), label="dadt %diff", color="C3")

def compsmall(data, dt=False):
    if dt == False:
        dt = data["tracktime"][-1]/1000.0
    T, r0, e = peters_comp(data, dt)

    r0_pd = 100*np.abs(r0 - np.interp(T, np.real(data["tracktime"]), np.real(data["r0"])))/np.interp(T, np.real(data["tracktime"]), np.real(data["r0"]))
    e_pd = 200*np.abs(e - np.interp(T, np.real(data["tracktime"]), np.real(data["e"])))/(np.abs(e) + np.abs(np.interp(T, np.real(data["tracktime"]), np.real(data["e"]))))
    print("r0_pd mean/median/max:", np.mean(r0_pd), np.median(r0_pd), max(r0_pd))
    print("r0_pd error linear slope:", np.polyfit(T, r0_pd/100.0, 1)[0])
    print("e_pd mean/median/max:", np.mean(e_pd), np.median(e_pd), max(e_pd))
    print("r0_pd error linear slope:", np.polyfit(T, np.abs(e - np.interp(T, np.real(data["tracktime"]), np.real(data["e"]))), 1)[0])
    
def compsmall2(data, dt=False):
    if dt == False:
        dt = data["tracktime"][-1]/1000.0
    T, r0, e = peters_comp(data, dt)

    r0_pd = np.abs(r0 - np.interp(T, np.real(data["tracktime"]), np.real(data["r0"])))/np.interp(T, np.real(data["tracktime"]), np.real(data["r0"]))
    e_pd = np.abs(e - np.interp(T, np.real(data["tracktime"]), np.real(data["e"])))
    #print("r0_pd error linear slope:", np.polyfit(T, r0_pd, 1)[0])
    #print("e_pd error linear slope:", np.polyfit(T, e_pd, 1)[0])
    return [np.polyfit(T, r0_pd, 1)[0], np.polyfit(T, e_pd, 1)[0]]
        

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