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
from IPython.display import HTML
#import pywt
from tqdm import tqdm
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.gridspec import GridSpec

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
    idx = np.nanargmin(np.abs(array - time))
    val = array.flat[idx]
    return np.where(array == val)[0][0]

def sph2cart(pos, a):
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
    x = np.sqrt(pos[0]**2 + a**2) * np.sin(pos[1]) * np.cos(pos[2])
    y = np.sqrt(pos[0]**2 + a**2) * np.sin(pos[1]) * np.sin(pos[2])
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
    
def plotvalue2(datalist, value, vsphase=False, linefit=True, start=0, end=-1, xscale='linear', yscale='linear', filename=False):
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
                    "true_anom": [data["true_anom"]/(2*np.pi), "True Anomaly"],
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
                    "semi_lat": [data["p"], "Semilatus Rectum"],
                    "radial_v": [data["all_vel"][:,1], "Radial Velocity"],
                    "theta_v": [data["all_vel"][:,2], "Theta Velocity"],
                    "phi_v": [data["all_vel"][:,3], "Phi Velocity"],
                    "total_v": [data["vel"], "Velocity"],
                    "radial_freq": [data["freqs"][:, 0], "Radial Frequency"],
                    "theta_freq": [data["freqs"][:, 1], "Theta Frequency"],
                    "phi_freq": [data["freqs"][:, 2], "Phi Frequency"],
                    "all_freq": [data["freqs"], "All Frequencies"],
                    "energy": [data["energy"], "Specific Energy"],
                    "l_momentum": [data["phi_momentum"], "Specific Angular Momentum"],
                    "carter": [data["carter"], "Carter Constant"],
                    "qarter": [data["qarter"], "Carter Constant (Unnormalized)"],
                    "l_momentumx": [data["Lx_momentum"], "Specific Angular Momentum (x-component)"],
                    "l_momentumy": [data["Ly_momentum"], "Specific Angular Momentum (y-component)"],
                    "l_momentumz": [data["Lz_momentum"], "Specific Angular Momentum (z-component)"],
                    "interval": [data["interval"], "Spacetime Interval"],
                    "energy2": [data["energy2"], "other energy"],
                    "CartLx": [data["Lx_momentum"], "cartLx"],
                    "CartLy": [data["Ly_momentum"], "cartLy"],
                    "CartLz": [data["Lz_momentum"], "cartLz"]}
        
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
                ax.legend(title="Linear Fit")
        
        else:
            print("Not a valid plottable. Chose one of the following:")
            for name in termdict:
                print("'" + name + "':", termdict[name][1])
            return False
        
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title)
    if filename == False:
        plt.show()
    else:
        plt.savefig('%s.png'%(str(filename)), bbox_inches='tight')

def plotvalue3(datalist, xvalue="time", yvalue="r0", legend=True, select_legend=None, polyfit=False, start=False, end=False, xscale='linear', yscale='linear', filename=False, derv=0, grid=False):
    '''
    Parameters
    ----------
    data : single dict OR list/array of dicts
        orbit dictionar(y/ies). Inputting a single dict will turn it into a list.
    xvalue : string, optional
        variable to plot along the x-axis. Defaults to time
    yvalue : string, optional
        variable to plot along the y-axis. Defaults to semimajor axis
    legend : bool, optional
        toggle legend, defaults to True
    polyfit : int/bool
        toggle fitting to a n-th degree polynomial, defaults to False/0
    start : bool/int, optional
        desired starting value of xvalue. The default is False, which corresponds to the initial value
    end : bool/int, optional
        desired ending value of xvalue. The default is False, which corresponds to the final value
    xscale : string, optional
        scaling for x-axis, matches options for matplotlib.set_xscale(). Defaults to linear
    yscale : string, optional
        scaling for y-axis, matches options for matplotlib.set_yscale(). Defaults to linear
    filename : bool/string, optional
        Saves the output as a png file if a string is provided. Defaults to False

    Returns
    -------
    bool
        True!

    '''
    # The time thing becomes an issue, since I'm using geometric time that could be different for each orbit
    # Although now that I think about it that's been an issue from the beginning
    # Hadn't even considered it until now
    # Actually it's all based on the central body so?? Shut up??
    
    polyfit = int(polyfit)
    if type(datalist) != list:
        datalist = [datalist]

    if not select_legend:
        select_legend = np.arange(len(datalist))
    select_legend = np.array(select_legend)
    select_legend = np.where(select_legend < 0, select_legend + len(datalist), select_legend)

    fig, ax = plt.subplots()
    colors = list(mcolors.TABLEAU_COLORS)
    for thing in range(len(datalist)):
        data = datalist[thing]
        # ["value": [location in data dict, Value name, extra bit if timing is weird]]
        termdict = {
            "index": ['np.arange(len(data["raw"]))', "Index", 'np.arange(len(data["raw"]))'],
            "time": ['data["raw"][:, 0]', "Coordinate Time", 'data["raw"][:, 0]'],
            "tracktime": ['data["tracktime"]', "Coordinate Time", 'data["tracktime"]'],
            "radius": ['data["raw"][:,1]', "Radius", 'data["raw"][:, 0]'],
            "theta": ['data["raw"][:,2]', "Theta", 'data["raw"][:, 0]'],
            "phi": ['data["raw"][:,3]%(2*np.pi)', "Phi", 'data["raw"][:, 0]'],
            "phase": ['data["raw"][:,3]/(2*np.pi)', "Phase", 'data["raw"][:, 0]'],
            "x": ['np.sqrt(data["raw"][:,1]**2 + data["spin"]**2)*np.sin(data["raw"][:,2])*np.cos(data["raw"][:,3])', "X", 'data["raw"][:, 0]'],
            "y": ['np.sqrt(data["raw"][:,1]**2 + data["spin"]**2)*np.sin(data["raw"][:,2])*np.sin(data["raw"][:,3])', "Y", 'data["raw"][:, 0]'],
            "z": ['data["raw"][:,1]*np.cos(data["raw"][:,3])', "Z", 'data["raw"][:, 0]'],
            "true_anom": ['data["true_anom"]/(2*np.pi)', "True Anomaly", 'data["tracktime"]'],
            "r0": ['data["r0"]', "Semimajor Axis", 'data["tracktime"]'],
            "pot_min": ['data["pot_min"]', "Effective Potential Minimum", 'data["tracktime"]'],
            "ecc": ['data["e"]', "Eccentricity", 'data["tracktime"]'],
            "semilat": ['data["p"]', "Semilatus-Rectum", 'data["tracktime"]'],
            "inc": ['data["inc"]', "Inclination", 'data["tracktime"]'],
            "inc_deg": ['np.rad2deg(data["inc"])', "Inclination (degrees)", 'data["tracktime"]'],
            "periapse": ['data["it"]', "Periapse", 'data["tracktime"]'],
            "apoapse": ['data["ot"]', "Apoapse", 'data["tracktime"]'],
            "omega": ['data["omega"]', "Phi Position of Periapse", 'otime'],
            "otime": ['data["otime"]', "Time of Periapse", 'otime'],
            "omegadot": ['np.diff(data["omega"])/np.diff(data["otime"])', "Advance of Periapse", 'odottime'],
            "odottime": ['0.5*data["otime"][:-1] + 0.5*data["otime"][1:]', "Periadvance time", 'odottime'],
            "asc_node": ['data["asc_node"]', "Phi Position of Ascending Node", 'asc_node_time'],
            "asc_node_time": ['data["asc_node_time"]', "Time of Ascending Node", 'asc_node_time'],
            "semi_maj": ['0.5*(data["it"] + data["ot"])', "Semimajor Axis", 'data["tracktime"]'],
            "semi_lat": ['data["p"]', "Semilatus Rectum", 'data["tracktime"]'],
            "radial_v": ['data["all_vel"][:,1]', "Radial Velocity", 'data["raw"][:, 0]'],
            "theta_v": ['data["all_vel"][:,2]', "Theta Velocity", 'data["raw"][:, 0]'],
            "phi_v": ['data["all_vel"][:,3]', "Phi Velocity", 'data["raw"][:, 0]'],
            "total_v": ['data["vel"]', "Velocity", 'data["raw"][:, 0]'],
            "energy": ['data["energy"]', "Specific Energy", 'data["tracktime"]'],
            "L_z": ['data["phi_momentum"]', "Specific Axial Angular Momentum", 'data["tracktime"]'],
            "carter": ['data["carter"]', "Carter Constant", 'data["tracktime"]'],
            "qarter": ['data["qarter"]', "Carter Constant (Unnormalized)", 'data["tracktime"]'],
            "approx_L": ['np.sqrt(data["carter"] + data["phi_momentum"]**2)', "Full Angular Momentum sqrt(C + L\u2080\u00B2)", 'data["tracktime"]'],
            "interval": ['data["interval"]', "Spacetime Interval", 'data["raw"][:, 0]'],
            "cosi": ['data["phi_momentum"]/np.sqrt(data["carter"] + data["phi_momentum"]**2)', "Inclination (cosi)", 'data["tracktime"]'],
            "cosi_deg": ['np.rad2deg(np.arccos(data["phi_momentum"]/np.sqrt(data["carter"] + data["phi_momentum"]**2)))', "Inclination (cosi, degrees)", 'data["tracktime"]']
            }
        
        if ((type(xvalue) == str) and (xvalue in termdict)) and ((type(yvalue) == str) and (yvalue in termdict)):
            xstuff = [eval(termdict[xvalue][0]), termdict[xvalue][1], termdict[xvalue][2]]
            ystuff = [eval(termdict[yvalue][0]), termdict[yvalue][1], termdict[yvalue][2]]
            title = "%s vs %s"%(termdict[yvalue][1], termdict[xvalue][1])
            title_add = ""
            if len(xstuff[0]) == len(ystuff[0]):
                #if the values have the same length, just grab the data
                xo = 0 if start==False else get_index(xstuff[0], start)
                xf = len(xstuff[0]) if end==False else get_index(xstuff[0], end)
                xvals, yvals = xstuff[0][xo:xf], ystuff[0][xo:xf]
            else: 
                #if the values don't have the same length, interpolate using time
                xvals, yvals = np.real_if_close(xstuff[0], 1000), np.real_if_close(ystuff[0], 1000)
                if len(xvals) != len(data["raw"][:, 0]):
                    xvals = np.interp(data["raw"][:, 0], eval(xstuff[2]), xstuff[0])
                if len(yvals) != len(data["raw"][:, 0]):
                    yvals = np.interp(data["raw"][:, 0], eval(ystuff[2]), ystuff[0])
                xo = 0 if start==False else get_index(xvals, start)
                xf = len(xvals) if end==False else get_index(xvals, end)
                xvals, yvals = xvals[xo:xf], yvals[xo:xf]
            if derv > 0:
                try:
                    title_add = " (%s%s derivative)"%(derv, ["st", "nd", "rd"][derv-1] if derv%10 < 4 else "th")
                    xvals_sub = np.copy(xvals)
                    yvals_sub = np.copy(yvals)
                    for i in range(derv):
                        yvals_sub = np.diff(yvals_sub)/np.diff(xvals_sub)
                        xvals_sub = 0.5*(xvals_sub[0:-1] + xvals_sub[1:])
                    #yvals = np.interp(xvals, xvals_sub, yvals_sub)
                    yvals = yvals_sub
                    xvals = xvals_sub
                except:
                    print("Could not calculate derivative! Maybe use a different x value?")
                    title_add = ""
            try: 
                lab_add = ""
                if polyfit:
                    try:
                        stuff = np.polyfit(xvals, yvals, polyfit)
                        ax.plot(xvals, np.polyval(stuff, xvals), linestyle="dashed", color=colors[thing%len(colors)])
                        lab_add = ": {res:.3e}".format(res=stuff[0])
                    except:
                        print(f"Could not plot linear fit for {datalist[thing]['name']}")
                if len(xvals) > 1:
                    ax.plot(xvals, yvals, color=colors[thing%len(colors)], label = (data["name"] + lab_add if thing in select_legend else None))
                else:
                    ax.scatter(xvals, yvals, color=colors[thing%len(colors)], label = (data["name"] + lab_add if thing in select_legend else None))
            except Exception as e:
                print(e)
                ax.plot(np.real(xvals), np.real(yvals), color=colors[thing%len(colors)], linestyle="dashed", label = (data["name"] + lab_add if thing in select_legend else None))
            
        
        else:
            print("Not a valid plottable. Chose one of the following:")
            for name in termdict:
                print("'" + name + "':", termdict[name][1])
            return False
        
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title+title_add)
    if grid:
        plt.grid()

    if legend:
        leg_add = ""
        if polyfit:
            leg_add = f" (with Leading {polyfit}-degree fit)"
        ax.legend(title="Datasets")
    
    if not filename:
        plt.show()
    else:
        plt.savefig('%s.png'%(str(filename)), bbox_inches='tight')

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

def orthoplots(datalist, ortho=False, zoom=1.0, start=0.0, end=-1.0, leg=True, ele=30, azi=-60, cb=False, stitch=False, filename=False, colors_list=None):
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
    if colors_list == None:
        from matplotlib.colors import TABLEAU_COLORS
        tab_cols = list(TABLEAU_COLORS.values())
    else:
        tab_cols = colors_list
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
        i = -1
        for data in datalist:
            i += 1
            a = data["spin"]
            to = get_index(data["time"], start)
            flipto = get_index(data["tracktime"], start)
            if end > 0.0:
                tf = get_index(data["time"], end)
                fliptf = get_index(data["tracktime"], end)
            else:
                tf = get_index(data["time"], data["time"][-1])
                fliptf = get_index(data["tracktime"], data["tracktime"][-1])

            carts = np.array([sph2cart(pos, a)/(10**(3*scaler)) for pos in data["pos"][to:tf]])
            cartsxy = np.copy(carts)
            cartsxz = np.copy(carts)
            cartsyz = np.copy(carts)
            flippoints = np.array([sph2cart(pos, a)/(10**(3*scaler)) for pos in data["pos"][data["trackix"][flipto:fliptf].astype(int)]])
            flipsxy = np.copy(flippoints)
            flipsxz = np.copy(flippoints)
            flipsyz = np.copy(flippoints)
            
            from matplotlib.path import Path
            from matplotlib.patches import PathPatch
            rb = (1 + (1 - data["spin"]**2)**(0.5))/(10**(3*scaler))
            ev_hor1 = plt.Circle((0, 0), rb, color='black')
            ev_hor2 = plt.Circle((0, 0), rb, color='black')
            ev_hor3 = plt.Circle((0, 0), rb, color='black')

            theta = np.linspace(0, 2*np.pi, 100)
            re = (1 + (1 - (data["spin"]*np.cos(theta))**2)**(0.5))/(10**(3*scaler))
            path = Path(np.transpose([re*np.sin(theta), re*np.cos(theta)]))
            top_erg = plt.Circle((0, 0), 2/(10**(3*scaler)), color='darksalmon', alpha=0.6)
            side_erg1 = PathPatch(path, color ='darksalmon', alpha=0.6)
            side_erg2 = PathPatch(path, color ='darksalmon', alpha=0.6)
            
            
            al = (azi)*np.pi/180. -np.pi
            el = (ele)*np.pi/180. - np.pi/2
            Xxy = [ 0.0, 0.0, 1.0]
            Xxz = [ 0.0, 1.0, 0.0]
            Xyz = [ 1.0, 0.0, 0.0]

            A = np.pi - np.arctan(20*cap/rb)
            B_ = A - np.pi/2 + np.arcsin((rb/data["pos"][to:tf,0])*np.sin(A))
            
            condxy = (np.arccos(np.dot(carts, Xxy)/data["pos"][to:tf,0]) < np.pi - B_)
            condxz = (np.arccos(np.dot(carts, Xxz)/data["pos"][to:tf,0]) < np.pi - B_)
            condyz = (np.arccos(np.dot(carts, Xyz)/data["pos"][to:tf,0]) < np.pi - B_)
            if cb == False:
                condxy = np.ones_like(condxy)
                condxz = np.ones_like(condxz)
                condyz = np.ones_like(condyz)

            cartsxy = np.array([carts[i] if condxy[i] == True else [np.nan, np.nan, np.nan] for i in range(len(condxy))])
            cartsxz = np.array([carts[i] if condxz[i] == True else [np.nan, np.nan, np.nan] for i in range(len(condxz))])
            cartsyz = np.array([carts[i] if condyz[i] == True else [np.nan, np.nan, np.nan] for i in range(len(condyz))])
            
            try:
                flipB_ = A - np.pi/2 + np.arcsin((rb/data["pos"][data["trackix"][flipto:fliptf].astype(int),0])*np.sin(A))
                flipcondxy = (np.arccos(np.dot(flippoints, Xxy)/data["pos"][data["trackix"][flipto:fliptf].astype(int),0]) < np.pi - flipB_)
                flipcondxz = (np.arccos(np.dot(flippoints, Xxz)/data["pos"][data["trackix"][flipto:fliptf].astype(int),0]) < np.pi - flipB_)
                flipcondyz = (np.arccos(np.dot(flippoints, Xyz)/data["pos"][data["trackix"][flipto:fliptf].astype(int),0]) < np.pi - flipB_)
                flipsxy = np.array([flippoints[i] if flipcondxy[i] == True else [np.nan, np.nan, np.nan] for i in range(len(flipcondxy))])
                flipsxz = np.array([flippoints[i] if flipcondxz[i] == True else [np.nan, np.nan, np.nan] for i in range(len(flipcondxz))])
                flipsyz = np.array([flippoints[i] if flipcondyz[i] == True else [np.nan, np.nan, np.nan] for i in range(len(flipcondyz))])
            except:
                pass

            #XY Plane
            frontxy = np.array([carts[i] if carts[i][2] >= 0 else [np.nan, np.nan, np.nan] for i in range(len(carts))])
            backxy = np.array([carts[i] if carts[i][2] <= 0 else [np.nan, np.nan, np.nan] for i in range(len(carts))])
            frontxz = np.array([carts[i] if carts[i][1] >= 0 else [np.nan, np.nan, np.nan] for i in range(len(carts))])
            backxz = np.array([carts[i] if carts[i][1] <= 0 else [np.nan, np.nan, np.nan] for i in range(len(carts))])
            frontyz = np.array([carts[i] if carts[i][0] >= 0 else [np.nan, np.nan, np.nan] for i in range(len(carts))])
            backyz = np.array([carts[i] if carts[i][0] <= 0 else [np.nan, np.nan, np.nan] for i in range(len(carts))])

            ax_list[0].plot(frontxy[:,0], frontxy[:,1], label=data["name"], c = tab_cols[i])
            ax_list[0].plot(backxy[:,0], backxy[:,1], label="_nolabel_", c = tab_cols[i])
            ax_list[1].plot(frontxz[:,0], frontxz[:,2], label="_nolabel_", c = tab_cols[i])
            ax_list[1].plot(backxz[:,0], backxz[:,2], label="_nolabel_", c = tab_cols[i])
            ax_list[2].plot(frontyz[:,1], frontyz[:,2], label="_nolabel_", c = tab_cols[i])
            ax_list[2].plot(backyz[:,1], backyz[:,2], label="_nolabel_", c = tab_cols[i])

            if cb == True:
                ax_list[0].add_patch(top_erg)
                ax_list[0].add_patch(ev_hor1)

                ax_list[1].add_patch(side_erg1)
                ax_list[1].add_patch(ev_hor2)

                ax_list[2].add_patch(side_erg2)
                ax_list[2].add_patch(ev_hor3)

            #ax_list[0].plot(cartsxy[:,0], cartsxy[:,1], label=data["name"], zorder=10)  #XY Plot
            #ax_list[1].plot(cartsxz[:,0], cartsxz[:,2], label="_nolabel_", zorder=10)  #XZ Plot
            #ax_list[2].plot(cartsyz[:,1], cartsyz[:,2], label="_nolabel_", zorder=10)  #ZY Plot


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
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=ele, azim=azi)
        
        rbound = 0
        i = -1
        for data in datalist:
            i += 1
            a = data["spin"]
            to = get_index(data["time"], start)
            flipto = get_index(data["tracktime"], start)
            if end > 0.0:
                tf = get_index(data["time"], end)
                fliptf = get_index(data["tracktime"], end)
            else:
                tf = get_index(data["time"], data["time"][-1])
                fliptf = get_index(data["tracktime"], data["tracktime"][-1])
                
            rbound = max(max(data["pos"][to:tf,0])*1.05, rbound)
            carts = np.array([sph2cart(pos, a) for pos in data["pos"][to:tf]])
            flippoints = np.array([sph2cart(pos, a) for pos in data["pos"][data["trackix"][flipto:fliptf].astype(int)]])
            
            if cb == True:
                rb = 1 + (1 - data["spin"]**2)**(0.5)
                theta, phi = np.linspace(0, 2*np.pi), np.linspace(0, np.pi)
                phi, theta = np.meshgrid(phi, theta)
                xS, yS, zS = rb*np.sin(theta)*np.sin(phi), rb*np.sin(theta)*np.cos(phi), rb*np.cos(theta)
                
                re = 1 + (1 - (data["spin"]*np.cos(theta))**2)**(0.5)
                xE, yE, zE = re*np.sin(theta)*np.sin(phi), re*np.sin(theta)*np.cos(phi), re*np.cos(theta)
                
                al = (azi)*np.pi/180. -np.pi
                el = (ele)*np.pi/180. - np.pi/2.
                X = [ np.sin(el) * np.cos(al),np.sin(el) * np.sin(al),np.cos(el)]

                A = np.pi - np.arctan(20*rbound/(rb*zoom))
                B_ = A - np.pi/2 + np.arcsin((rb/data["pos"][to:tf,0])*np.sin(A))
                
                blockedcheck = np.arccos(np.dot(carts, X)/data["pos"][to:tf,0]) < np.pi - B_
                boundboxcheck = [False not in piece for piece in np.abs(carts) <= rbound/zoom]
                 
                cond = np.logical_and(blockedcheck, boundboxcheck)
                #carts = np.array([carts[i] if cond[i] == True else [np.nan, np.nan, np.nan] for i in range(len(cond))])
                
                try:
                    flipB_ = A - np.pi/2 + np.arcsin((rb/data["pos"][data["trackix"][flipto:fliptf].astype(int),0])*np.sin(A))
                    flipblockedcheck = np.arccos(np.dot(flippoints, X)/data["pos"][data["trackix"][flipto:fliptf].astype(int),0]) < np.pi - flipB_
                    flipboundboxcheck = [False not in piece for piece in np.abs(flippoints) <= rbound/zoom]
                    flipcond = np.logical_and(flipblockedcheck, flipboundboxcheck)
                    flippoints = np.array([flippoints[i] if flipcond[i] == True else [np.nan, np.nan, np.nan] for i in range(len(flipcond))])
                except:
                    pass
            else:
                cond = np.ones_like(carts[:,0])
                
            rad_ele, rad_azi = ele*np.pi/180, azi*np.pi/180
            A, B, C = np.cos(rad_ele)*np.cos(rad_azi), np.cos(rad_ele)*np.sin(rad_azi), np.sin(rad_ele)
            front = np.array([carts[i] if -(A/C)*carts[i][0] - (B/C)*carts[i][1] - C*5 <= carts[i][2] else [np.nan, np.nan, np.nan] for i in range(len(cond))])
            back = np.array([carts[i] if -(A/C)*carts[i][0] - (B/C)*carts[i][1] - C*5 >= carts[i][2] else [np.nan, np.nan, np.nan] for i in range(len(cond))])

            ax.plot3D(back[:, 0], back[:, 1], back[:, 2], label=data["name"], c = tab_cols[i])
            if cb == True:
                ax.plot_surface(xS, yS, zS, color="black", shade=False)
                ax.plot_surface(xE, yE, zE, color="darksalmon", alpha = 0.3)
            ax.plot3D(front[:, 0], front[:, 1], front[:, 2], zorder=10, c = tab_cols[i])
            if stitch == True:
                ax.scatter(flippoints[:, 0], flippoints[:, 1], flippoints[:, 2], label=data["name"], zorder=9, marker="*", s=300)
            
        
        ax.set(xlim3d=(-rbound/zoom, rbound/zoom), xlabel='X')
        ax.set(ylim3d=(-rbound/zoom, rbound/zoom), ylabel='Y')
        ax.set(zlim3d=(-rbound/zoom, rbound/zoom), zlabel='Z')
        ax.set_box_aspect((rbound, rbound, rbound))
        if leg == True:
            ax.legend()
    if filename == False:
        plt.show()
    else:
        plt.savefig('%s.png'%(str(filename)), bbox_inches='tight')

def multi3D(datalist, grid=None, zoom=1.0, start=0, end=-1, ele=30, azi=30, cb=True, filename=False, colors_list=None):
    if colors_list == None:
        from matplotlib.colors import TABLEAU_COLORS
        tab_cols = list(TABLEAU_COLORS.values())
    else:
        tab_cols = colors_list

    if type(datalist) != list:
        datalist = [datalist]

    if grid == None:
        c, r = np.ceil(np.sqrt(len(datalist))), 0
        while c*r < len(datalist):
            r += 1
        grid = (int(r), int(c))

    scale = np.ceil(2*np.sqrt(len(datalist)))/2
    fig = plt.figure(figsize=(grid[1]*8/scale, grid[0]*8/scale))
    gs = GridSpec(
        grid[0], grid[1],
        figure=fig,
        left=0, right=1,
        bottom=0, top=1,
        wspace=0, hspace=0
    )

    for i in range(len(datalist)):
        ax = fig.add_subplot(gs[i//grid[1], i % grid[1]], projection="3d")
        ax.set_axis_off()
        print(i//grid[1], i % grid[1], grid)
        ax.view_init(elev=ele, azim=azi)

        data = datalist[i]
        to = get_index(data["time"], start)
        tf = get_index(data["time"], end) if end > 0 else len(data["time"])

        carts = np.array([sph2cart(pos, data["spin"]) for pos in data["pos"][to:tf]])

        # View mask
        T = np.pi / 180
        view_norm = np.array([
            np.cos(ele*T)*np.cos(azi*T),
            np.cos(ele*T)*np.sin(azi*T),
            np.sin(ele*T)
            ])
        mask = np.sign(carts @ view_norm)

        # Horizons
        rb = 1 + np.sqrt(1 - data["spin"]**2)
        theta, phi = np.linspace(0, 2*np.pi), np.linspace(0, np.pi)
        phi, theta = np.meshgrid(phi, theta)

        xS = rb*np.sin(theta)*np.sin(phi)
        yS = rb*np.sin(theta)*np.cos(phi)
        zS = rb*np.cos(theta)

        re = 1 + np.sqrt(1 - (data["spin"]*np.cos(theta))**2)
        xE = re*np.sin(theta)*np.sin(phi)
        yE = re*np.sin(theta)*np.cos(phi)
        zE = re*np.cos(theta)

        rbound = np.max(np.abs(carts)) * 0.9
        ax.set_xlim(-rbound, rbound)
        ax.set_ylim(-rbound, rbound)
        ax.set_zlim(-rbound, rbound)
        ax.set_box_aspect((1, 1, 1))

        dr = 10
        max_r = dr * max(1, int(rbound // dr))
        radii = np.arange(dr, max_r + dr, dr)
        while len(radii) < 3:
            dr = dr//2
            max_r = int(max_r * 1.5)
            radii = np.arange(dr, max_r + dr, dr)
            #print(dr, radii)
        while len(radii) > 5:
            radii = radii[1::2]

        radial_reference_circles(ax, radii, angle=azi+90)
        radial_axes_with_ticks(ax, rbound, ticks=())

        # Back half
        ax.plot3D(
            np.where(mask <= 0, carts[:,0], np.nan),
            np.where(mask <= 0, carts[:,1], np.nan),
            np.where(mask <= 0, carts[:,2], np.nan),
            c=tab_cols[i%len(tab_cols)], lw=1.2, alpha=0.85
        )

        ax.plot_surface(xS, yS, zS, color="black", alpha=0.85)
        ax.plot_surface(xE, yE, zE, color="darksalmon", alpha=0.25)

        # Front half
        ax.plot3D(
            np.where(mask > 0, carts[:,0], np.nan),
            np.where(mask > 0, carts[:,1], np.nan),
            np.where(mask > 0, carts[:,2], np.nan),
            c=tab_cols[i%len(tab_cols)], lw=1.3
        )

        ax.margins(0)
        ax.set_box_aspect((1, 1, 1))
        ax.set_anchor('C')
    
    draw_panel_dividers(fig, gridvals=(grid[1], grid[0]))
    plt.show()
    

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

def ani_thing3(data, name=None, display=True, ortho=False, zoom=1.0, ele=30, azi=-60, scroll=True, cb=True, numturns=10, fid=1):
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
    
    int_sphere, int_time = mm.interpolate(data["pos"], data["time"], supress = False)
    a = data["spin"]
    X = np.sqrt(int_sphere[:,0]**2 + a**2)*np.sin(int_sphere[:,1])*np.cos(int_sphere[:,2])
    Y = np.sqrt(int_sphere[:,0]**2 + a**2)*np.sin(int_sphere[:,1])*np.sin(int_sphere[:,2])
    Z = int_sphere[:,0]*np.cos(int_sphere[:,1])

    num_steps = int(100*fid)
    
    #print(np.where(data["pos"][:,2] > 2*np.pi))
    turn_ind = np.where(data["pos"][:,2] > 2*np.pi)[0][0]
    first_turn = get_index(int_time, data["time"][turn_ind])
    #print(first_turn)
    
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
    
    #pbar = tqdm(total=num_steps+10, position=0)
    ani = animation.FuncAnimation(
        fig, update_line, frames=tqdm(range(num_steps+10), position=0, initial=1), fargs=(X, Y, Z, line), interval=10)
    
    #HEY, CAN YOU JUST MAKE MULTIPLE ANIMATION OBJECTS ON THE SAME FIGURE??? EXPERIMENT ON SOMETHING SIMPLE
    '''
    if name == False:
        name=data["name"][:10] + time.strftime("%y_%m_%d_%H", time.localtime())

    cwd = os.getcwd()
    f = os.path.join(cwd, name + ".gif")
    writergif = animation.PillowWriter(fps=10)
    ani.save(f, writer=writergif)
    
    plt.show()
    print("\n")
    print(name + '.gif')'''
    # Case 1: Save if requested
    if name:
        cwd = os.getcwd()
        f = os.path.join(cwd, name if name.endswith(".gif") else name + ".gif")
        writergif = animation.PillowWriter(fps=10)
        ani.save(f, writer=writergif)
        print(f"Saved animation as: {f}")

    # Case 2: Display inline if requested
    if display:
        plt.close(fig)  # prevent duplicate static plot
        return HTML(ani.to_jshtml())

    # Case 3: Neither save nor display
    plt.close(fig)
    return None

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

def ani_thing4(datalist, name=None, display=True, ortho=False, zoom=1.0, ele=30, azi=-60, scroll=True, cb=True, numturns=10, fid=1):
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

    if type(datalist) != list:
        datalist = [datalist]

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
            #print(data)
            #print(data["time"][0])
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
            #print(paths[0])
            for i in range(len(paths)):
                carts = np.transpose(paths[i][:3])
                r = np.sqrt(carts[:,0]**2 + carts[:,1]**2 + carts[:,2]**2)
                B_ = A - np.pi/2 + np.arcsin((rb/r)*np.sin(A))
                #print(len(carts), len(V), len(r), len(B_))
                cond = (np.arccos(np.dot(carts, V)/r) < np.pi - B_)
                #print(list(cond).count(False))
                #newpath = 
                paths[i] = np.append(np.transpose(np.array([carts[i] if cond[i] == True else [np.nan, np.nan, np.nan] for i in range(len(cond))])), [paths[i][-1]], axis=0)
            #print(paths[0])
            
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
            turn_ind = np.where(datalist[i]["pos"][:,2] > 2*np.pi)[0][0]
            first_turn = get_index(paths[i][-1], datalist[i]["time"][turn_ind])
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
    
    # Case 1: Save if requested
    if name:
        cwd = os.getcwd()
        f = os.path.join(cwd, name if name.endswith(".gif") else name + ".gif")
        writergif = animation.PillowWriter(fps=10)
        ani.save(f, writer=writergif)
        print(f"Saved animation as: {f}")

    # Case 2: Display inline if requested
    if display:
        plt.close(fig)  # prevent duplicate static plot
        return HTML(ani.to_jshtml())

    # Case 3: Neither save nor display
    plt.close(fig)
    return None

def ani_thing5(data, name=False, ortho=False, start=0.0, end=-1.0, zoom=1.0, ele=30, azi=-60, scroll=True, cb=True, numturns=10, fid=1):
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
    
    to = get_index(data["time"], start)
    if end == -1.0:
        tf = len(data["time"]) #get_index(data["time"], data["time"][-1])
    else: 
        tf = get_index(data["time"], end)
    
    if name == False:
        name=data["name"][:10] + time.strftime("%y_%m_%d_%H", time.localtime())
    
    int_sphere, int_time = mm.interpolate(data["pos"][to:tf], data["time"][to:tf], supress = False)
    X = int_sphere[:,0]*np.sin(int_sphere[:,1])*np.cos(int_sphere[:,2])
    Y = int_sphere[:,0]*np.sin(int_sphere[:,1])*np.sin(int_sphere[:,2])
    Z = int_sphere[:,0]*np.cos(int_sphere[:,1])
    num_steps = int(100*fid)
    first_turn = np.where(int_sphere[:,2] - int_sphere[0,2] > 2*np.pi)[0][0]
    
    if ortho == False:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=ele, azim=azi)
        line = ax.plot([], [], [], zorder=10)[0]

        # Setting the axes properties
        rbound = max(data["pos"][to:tf,0])*1.05/zoom

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
                    rbound = max(max((xdata[beg:full*num]**2 + ydata[beg:full*num]**2 + zdata[beg:full*num]**2)**0.5)*1.05/zoom, 3)
                except:
                    rbound = max(max((xdata**2 + ydata**2 + zdata**2)**0.5)*1.05/zoom, 3)
                
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
            rbound = max(data["pos"][to:tf,0])*1.05/zoom
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
                    rbound = np.nanmax(np.nanmax((xdata[beg:end]**2 + ydata[beg:end]**2 + zdata[beg:end]**2)**0.5)*1.05/zoom, 3)
                except:
                    rbound = np.nanmax(np.nanmax((xdata**2 + ydata**2 + zdata**2)**0.5)*1.05/zoom, 3)
    
                ax2.set_xlim(-rbound, rbound)
                ax2.set_ylim(-rbound, rbound)
                ax2.set_aspect('equal')

            return line
        
    # Creating the Animation object
    
    #pbar = tqdm(total=num_steps+10, position=0)
    ani = animation.FuncAnimation(
        fig, update_line, frames=tqdm(range(num_steps+10), position=0, initial=1), fargs=(X, Y, Z, line), interval=10)
    
    #HEY, CAN YOU JUST MAKE MULTIPLE ANIMATION OBJECTS ON THE SAME FIGURE??? EXPERIMENT ON SOMETHING SIMPLE
    
    cwd = os.getcwd()
    f = os.path.join(cwd, name + ".gif")
    writergif = animation.PillowWriter(fps=10)
    ani.save(f, writer=writergif)#, bbox_inches='tight')
    
    plt.show()
    print("\n")
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

def gimme_startpot(data, rbounds = [-1, 1]):
    a, mu = data["inputs"][1], data["inputs"][2]
    E, L, C = data["energy"][0], data["phi_momentum"][0], data["carter"][0]
    print(E, L, C)
    potentplotter(E, L, C, a, rbounds)

def potentplotter(E, L, C, a, rbounds=[-1, -1]):
    if type(E) == np.ndarray:
        pass
    elif type(E) == list:
        E, L, C = np.array(E), np.array(L), np.array(C)
    else:
        E, L, C = np.array([E]), np.array([L]), np.array([C])
        
    R = lambda r: ((r**2 + a**2)*E - a*L)**2 - (r**2 - 2*r + a**2)*(r**2 + (L - a*E)**2 + C)
    rx, rn, blah, blee = np.transpose(np.array([np.roots([E[i]**2 - 1, 2, (a**2)*(E[i]**2 - 1) - L[i]**2 - C[i], 2*((a*E[i] - L[i])**2 + C[i]), -(a**2)*C[i]]) for i in range(len(E))]))
    r0, bloh, bluh = np.transpose(np.array([np.roots([4*(E[i]**2 - 1), 6, 2*((a**2)*(E[i]**2 - 1) - L[i]**2 - C[i]), 2*((a*E[i] - L[i])**2 + C[i])]) for i in range(len(E))]))

    if -1 in rbounds:
        rbounds = np.linspace(0.0, rx*1.05, num=100)
    else:
        rbounds = np.linspace(rbounds[0]*np.ones((len(rn))), rbounds[-1]*np.ones((len(rx))), num=100)

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Radius (Geometric Units)")
    ax1.set_ylabel("Effective Potential")
    #ax1.set_title("Effective Potential")
    ax1.plot(rbounds, rbounds*0.0)
    ax1.plot(rbounds, -R(rbounds))
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
    #plt.show()
    return(R(bloh))

def potentplotter4(E, L, C, a, rbounds=[-1, -1]):
    if type(E) == np.ndarray:
        pass
    elif type(E) == list:
        E, L, C = np.array(E), np.array(L), np.array(C)
    else:
        E, L, C = np.array([E]), np.array([L]), np.array([C])
        
    R = lambda r: ((r**2 + a**2)*E - a*L)**2 - (r**2 - 2*r + a**2)*(r**2 + (L - a*E)**2 + C)
    rx, rn, blah, blee = np.transpose(np.array([np.roots([E[i]**2 - 1, 2, (a**2)*(E[i]**2 - 1) - L[i]**2 - C[i], 2*((a*E[i] - L[i])**2 + C[i]), -(a**2)*C[i]]) for i in range(len(E))]))
    r0, bloh, bluh = np.transpose(np.array([np.roots([4*(E[i]**2 - 1), 6, 2*((a**2)*(E[i]**2 - 1) - L[i]**2 - C[i]), 2*((a*E[i] - L[i])**2 + C[i])]) for i in range(len(E))]))

    if -1 in rbounds:
        rbounds = np.linspace(0.0, rx*1.05, num=100)
    else:
        rbounds = np.linspace(rbounds[0]*np.ones((len(rn))), rbounds[-1]*np.ones((len(rx))), num=100)

    #fig1, ax1 = plt.subplots()
    #ax1.set_xlabel("Radius (Geometric Units)")
    #ax1.set_ylabel("Effective Potential")
    #ax1.set_title("Effective Potential")
    #ax1.plot(rbounds, rbounds*0.0)
    #ax1.plot(rbounds, -R(rbounds))
    #ext = False
    #if r0 >= rbounds[0]:
        #ax1.vlines(r0, -R(r0), 0)
        #ax1.scatter(r0, 0.0, label="Potential Minimum")
     #   ext = True
    #if bloh >= rbounds[0] and abs(R(bloh)) < 1e-5:
        #ax1.vlines(bloh, -R(bloh), 0)
        #ax1.scatter(bloh, 0.0, marker="*", label="Unstable Circular orbit")
     #   ext = True
    #if ext == True:
     #   ax1.legend()
    #plt.show()
    return(rbounds, -R(rbounds))

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
        #print("zah",rbounds2[0],rbounds2[-1])
        maxbounds = [min(maxbounds[0], rbounds2[0]), max(maxbounds[1], rbounds2[-1])]
        #print(maxbounds)
        ax1.plot(rbounds2, R(rbounds2))
    ax1.hlines(0.0, maxbounds[0], maxbounds[-1], color="black", zorder=1)

def potentplotter3(cons, a, rbounds=[-1, -1]):
    if len(np.shape(cons)) == 1:
        cons = [cons]
    fig1, ax1 = plt.subplots()
    maxbounds = [1e12, -1e12]
    for E, L, C in cons:   
        R = lambda r: ((r**2 + a**2)*E - a*L)**2 - (r**2 - 2*r + a**2)*(r**2 + (L - a*E)**2 + C)
        rx, rn, blah, blee = np.transpose(np.array([np.roots([E[i]**2 - 1, 2, (a**2)*(E[i]**2 - 1) - L[i]**2 - C[i], 2*((a*E[i] - L[i])**2 + C[i]), -(a**2)*C[i]]) for i in range(len(E))]))
        r0, bloh, bluh = np.transpose(np.array([np.roots([4*(E[i]**2 - 1), 6, 2*((a**2)*(E[i]**2 - 1) - L[i]**2 - C[i]), 2*((a*E[i] - L[i])**2 + C[i])]) for i in range(len(E))]))
    
        if -1 in rbounds:
            rbounds = np.linspace(0.0, rx*1.05, num=100)
        else:
            rbounds = np.linspace(rbounds[0]*np.ones((len(rn))), rbounds[-1]*np.ones((len(rx))), num=100)

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Radius (Geometric Units)")
    ax1.set_ylabel("Effective Potential")
    #ax1.set_title("Effective Potential")
    ax1.plot(rbounds, rbounds*0.0)
    ax1.plot(rbounds, -R(rbounds))
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
    plt.show()
    return(R(bloh))

def fouriercountourthing(datalist, wavedis, num=1000):
    from scipy.fft import rfft, rfftfreq
    if type(datalist) != list:
        datalist = [datalist]
    for data in datalist:
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
        print(np.shape(x), np.shape(y), np.shape(z))
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
    if type(datalist) != list:
        datalist = [datalist]
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
    xmin, xmax = 10**10, 10**(-10)
    size_vals = []
    for i in range(num):
        to = get_index(datalist[i]["time"], start)
        if end > 0.0:
            tf = get_index(datalist[i]["time"], end)
        else:
            tf = get_index(datalist[i]["time"], datalist[i]["time"][-1])
        cap = max(datalist[i]["pos"][to:tf,0])*1.05

        scaler = np.floor(np.log10(cap))//3
        carts = np.array([sph2cart(pos, datalist[i]["spin"])/(10**(3*scaler)) for pos in datalist[i]["pos"]])
        ax[i,0].plot(carts[to:tf,0], carts[to:tf,1])
        ax[i,0].set_aspect('equal')
        size_vals.append([*ax[i,0].set_xlim(), *ax[i, 0].set_ylim(), 0.0])
        size_vals[-1][4] = (size_vals[-1][3] - size_vals[-1][2])*(size_vals[-1][1] - size_vals[-1][0])
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
        thismin, thismax = ax[i, 1].set_xlim()
        xmin, xmax = min(xmin, thismin), max(xmax, thismax)
    size_ix = np.where(np.array(size_vals)[:,-1] == max(np.array(size_vals)[:,-1]))[0][0]
    for i in range(num):
        ax[i, 0].set_xlim(*size_vals[size_ix][:2])
        ax[i, 0].set_ylim(*size_vals[size_ix][2:4])
        ax[i, 1].set_xlim(xmin, xmax)
    plt.show()

def full_and_fourier(datalist, start=0, end=-1, width=12, height=0, space=0.01):
    from matplotlib.colors import TABLEAU_COLORS
    tab_cols = list(TABLEAU_COLORS.values())
    if type(datalist) != list:
        datalist = [datalist]
    num = len(datalist)
    if num < 2:
        print("For comparisons only")
        return False
    if width == 0:
        width = (10/3)*num
    if height == 0:
        height = 3*num + 1
    fig = plt.figure(figsize=(width, height))
    
    #start, end = 0, 20000
    xmin, xmax = 10**10, 10**(-10)
    size_vals = []
    for i in range(num):
        ax = fig.add_subplot(num, 2, i*2 + 1, projection="3d")
        to = get_index(datalist[i]["time"], start)
        if end > 0.0:
            tf = get_index(datalist[i]["time"], end)
        else:
            tf = get_index(datalist[i]["time"], datalist[i]["time"][-1])
        cap = max(datalist[i]["pos"][to:tf,0])*1.05

        #Actual path data
        elev = ax.elev
        azim = ax.azim
        carts = np.array([sph2cart(pos, datalist[i]["spin"]) for pos in datalist[i]["pos"]])
        T = np.pi/180
        view_norm = np.array([np.cos(elev*T)*np.cos(azim*T), np.cos(elev*T)*np.sin(azim*T), np.sin(elev*T)])
        mask = np.sign(np.matmul(carts, view_norm))
        #Event horizon
        rb = 1 + (1 - datalist[i]["spin"]**2)**(0.5)
        theta, phi = np.linspace(0, 2*np.pi), np.linspace(0, np.pi)
        phi, theta = np.meshgrid(phi, theta)
        xS, yS, zS = rb*np.sin(theta)*np.sin(phi), rb*np.sin(theta)*np.cos(phi), rb*np.cos(theta)
        #Ergosphere
        re = 1 + (1 - (datalist[i]["spin"]*np.cos(theta))**2)**(0.5)
        xE, yE, zE = re*np.sin(theta)*np.sin(phi), re*np.sin(theta)*np.cos(phi), re*np.cos(theta)
        #Plot the 3D bits
        ax.plot3D(np.where(mask<=0, carts[:, 0], np.nan), np.where(mask<=0, carts[:, 1], np.nan), np.where(mask<=0, carts[:, 2], np.nan), c=tab_cols[0], zorder=0)
        ax.plot_surface(xS, yS, zS, color="black", shade=False, zorder=1)
        ax.plot_surface(xE, yE, zE, color="darksalmon", alpha = 0.3, zorder=2)
        ax.plot3D(np.where(mask>0, carts[:, 0], np.nan), np.where(mask>0, carts[:, 1], np.nan), np.where(mask>0, carts[:, 2], np.nan), c=tab_cols[0], zorder=3)
        size_vals.append([*ax.set_xlim(), *ax.set_ylim(), 0.0])
        size_vals[-1][4] = (size_vals[-1][3] - size_vals[-1][2])*(size_vals[-1][1] - size_vals[-1][0])
        rbound = np.max(carts)*1.01
        ax.set(xlim3d=(-rbound, rbound), xlabel='X')
        ax.set(ylim3d=(-rbound, rbound), ylabel='Y')
        ax.set(zlim3d=(-rbound, rbound), zlabel='Z')
        ax.set_box_aspect((rbound, rbound, rbound))
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            labels = axis.get_ticklabels()
            for lbl in labels[0::2]:
                lbl.set_visible(False)


        ax = fig.add_subplot(num, 2, i*2 + 2)
        wave, time = mm.full_transform(datalist[i], cap*1000)
        x = np.copy(time)
        y1 = np.copy(wave[:,0,0])
        y2 = np.copy(wave[:,0,1])
        N = time.size
        T = (x[-1] - x[0])/N
        yf1 = fft(y1)
        yf2 = fft(y2)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        ax.plot(xf, 2.0/N * np.abs(yf2[0:N//2]), label = "hx")
        ax.plot(xf, 2.0/N * np.abs(yf1[0:N//2]), label = "h+")
        #plt.xscale('log')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        ax.legend(title=datalist[i]["name"])
        thismin, thismax = ax.set_xlim()
        xmin, xmax = min(xmin, thismin), max(xmax, thismax)
    size_ix = np.where(np.array(size_vals)[:,-1] == max(np.array(size_vals)[:,-1]))[0][0]
    fig.subplots_adjust(wspace=0*space)
    for ax in fig.axes[1::2]:
        ax.set_xlim(xmin, xmax)
    plt.show()

def orth_and_fourier(datalist, start=0, end=-1, filename=False, leg_title=False):
    if type(datalist) != list:
        datalist = [datalist]
    for data in datalist:
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

        a = data["spin"]
        carts = np.array([sph2cart(pos, a) for pos in data["pos"]])
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
        if leg_title != False:
            ax2.legend(title=leg_title)
        if filename == False:
            plt.show()
        else:
            plt.savefig("%s.png"%(filename), bbox_inches="tight")

def justfourier(datalist, start=0, end=-1, filename=False, supress=True, m_bh=False, distance=False):
    if type(datalist) != list:
        datalist = [datalist]
    num = len(datalist)
    for data in datalist:
        if not distance:
            distance = max(data["raw"][:,1])*100000  # distance is in GU
        h_plus, h_cross, time = mm.full_transform(data, distance, supress=supress, m_bh=m_bh)
        to = get_index(time, start)
        if end > 0.0:
            tf = get_index(time, end)
        else:
            tf = None
        print(start, end, to, tf)
        x = np.copy(time[to:tf])
        y1 = np.copy(h_plus)
        y2 = np.copy(h_cross)
        N = x.size
        T = x[1] - x[0]
        yf1 = np.fft.rfft(y1)
        yf2 = np.fft.rfft(y2)
        xf = np.fft.rfftfreq(N, d=T)
        h_plus_fft = 2.0/N * np.abs(yf1)
        h_cross_fft = 2.0/N * np.abs(yf2)
        
        fig, ax = plt.subplots()
        ax.plot(xf[1:], h_plus_fft[1:], label = "h+", alpha=0.8)
        ax.plot(xf[1:], h_cross_fft[1:], label = "hx", alpha=0.8)
        #plt.setp(ax3.get_yticklabels(), visible=False)
        #plt.setp(ax4.get_yticklabels(), visible=False)
        ax.set_title("Waveform Fourier Transform")
        #ax.set_title('Waveform Frequency ' + freq_unit)
        ax.set_yscale('log')
        ax.set_xscale('log')
        if data["inputs"][-1] == "grav" and not m_bh:
            freq_unit = "(Geometric Units)"#'(G\u209C\u207B\u00B9)'
        else:
            freq_unit = '(Hz)'
        ax.set_xlabel("Frequency " + freq_unit)
        ax.set_ylabel("Strain Intensity")
        ax.grid()
        ax.legend()
        if filename:
            plt.savefig("%s.png"%(filename), bbox_inches="tight")
    if not filename:
        plt.show(block=False)
    
'''
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
    print(pywt.frequency2scale('morl',samper*freq*10), pywt.frequency2scale('morl',samper*freq/10))
    scalelow, scalehigh = max(1, pywt.frequency2scale('morl',samper*0.06804175435239163*10)), pywt.frequency2scale('morl',samper*freq/100)
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
'''
    
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


# Pretty stuff

def radial_axes_with_ticks(ax, r, ticks=(-0.5, -1, 0.5, 1),
                           color="0.55", lw=1.0, label_color="0.45"):
    '''
    Draw radial axes with ticks and labels
    
    :param ax: matplotlib axes object
    :param r: maximum radius value for plot
    :param ticks: tick mark locations as fractions of r, make empty to not include ticks
    :param color: axis color (default gray)
    :param lw: line width
    :param label_color: axis label color (default slightly darker gray)
    '''

    # Axis lines
    axes = {
        "x": ([ -r,  r], [0, 0], [0, 0]),
        "y": ([0, 0], [ -r,  r], [0, 0]),
        "z": ([0, 0], [0, 0], [ -r,  r]),
    }

    for x, y, z in axes.values():
        ax.plot(x, y, z, color=color, lw=lw, zorder=1)

    tick_len = 0.035 * r

    for f in ticks:
        if abs(f) > 1:
            continue
        t = f * r

        # X-axis ticks
        ax.plot([t, t], [-tick_len, tick_len], [0, 0],
                color=color, lw=lw)
        ax.text(t, -3*tick_len, 0, f"{t:.0f}",
                color=label_color, fontsize=8,
                ha="center", va="top")

        # Y-axis ticks
        ax.plot([-tick_len, tick_len], [t, t], [0, 0],
                color=color, lw=lw)
        ax.text(3*tick_len, t, 0, f"{t:.0f}",
                color=label_color, fontsize=8,
                ha="left", va="center")

        # Z-axis ticks
        ax.plot([0, 0], [-tick_len, tick_len], [t, t],
                color=color, lw=lw)
        ax.text(0, 3*tick_len, t, f"{t:.0f}",
                color=label_color, fontsize=8,
                ha="center", va="bottom")

    # Axis labels
    ax.text(r*1.08, 0, 0, r"$x/M$", fontsize=10, color=label_color)
    ax.text(0, r*1.08, 0, r"$y/M$", fontsize=10, color=label_color)
    ax.text(0, 0, r*1.08, r"$z/M$", fontsize=10, color=label_color)

def draw_panel_dividers(fig, gridvals=(2,2), color="0.25", lw=0.8):
    '''
    Docstring for draw_panel_dividers
    
    :param fig: matplotlib figure object
    :param gridvals: Tuple in the form (rows, columns), assumes 2x2
    :param color: divider color, default dark gray
    :param lw: line width, default kinda skinny
    '''
    # Vertical dividers
    for i in range(gridvals[0] - 1):
        y_pos = (i + 1)/gridvals[0]
        fig.add_artist(plt.Line2D(
            [y_pos, y_pos], [0, 1],
            transform=fig.transFigure,
            color=color, lw=lw
        ))
    # Horizontal divider
    for i in range(gridvals[1] - 1):
        x_pos = (i + 1)/gridvals[1]
        fig.add_artist(plt.Line2D(
            [0, 1], [x_pos, x_pos],
            transform=fig.transFigure,
            color=color, lw=lw
        ))

def radial_reference_circles(ax, radii, color="0.55", lw=0.8, alpha=0.7,
                             z=0.0, label=True, label_color="0.45",
                             fontsize=10, angle=120):
    '''
    Draw concentric circles in the XY plane at given radii.
    
    :param ax: matplotlib axes object
    :param radii: radii of concentric circles
    :param color: color of circles, default light gray
    :param lw: line width
    :param alpha: circle transparency
    :param z: height of circles, defaults to 0 for XY plane
    :param label: True if including radius labels
    :param label_color: default gray
    :param fontsize: Font size
    :param angle: Angle at which labels appear on XY plane
    '''

    phi = np.linspace(0, 2*np.pi, 400)

    for r in radii:
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        zc = np.full_like(x, z)

        ax.plot(x, y, zc, color=color, lw=lw, alpha=alpha * (1 - r / radii[-1]) + 0.2, zorder=5)

        if label:
            ax.text(r * np.cos(angle*np.pi/180), r * np.sin(angle*np.pi/180), z,
                    f"{r:.0f}",
                    color=label_color,
                    fontsize=fontsize,
                    ha="left", va="bottom")

