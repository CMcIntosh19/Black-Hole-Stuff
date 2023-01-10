# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:37:02 2022

@author: hepiz
"""

import numpy as np
import matplotlib.pyplot as plt

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

def physplots(datalist, merge=False, start=0, end=-1, leg=True):
    if type(datalist) != list:
        datalist = [datalist]
    if merge == True:
        fig1, ax_list1 = plt.subplots(6)
        fig2, ax_list2 = plt.subplots(4)
    else:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        ax_list1 = [ax1, ax2, ax3, ax4, ax5, ax6]
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
        
        ax_list1[0].plot(data["time"][to1:tf1], data["pos"][to1:tf1, 0], label=data["name"])  #XY Plot
        ax_list1[0].set_title('Radius vs Time')
        ax_list1[1].plot(data["time"][to1:tf1], data["pos"][to1:tf1, 1], label=data["name"])  #XZ Plot
        ax_list1[1].set_title('Theta vs Time')
        ax_list1[2].plot(data["time"][to1:tf1], data["pos"][to1:tf1, 2], label=data["name"])  #ZY Plot
        ax_list1[2].set_title('Phi vs Time')
        ax_list1[3].plot(data["time"][to1:tf1], data["Lx_momentum"][to1:tf1], label=data["name"])  #XY Plot
        ax_list1[3].set_title('Psuedo Lx vs Time')
        ax_list1[4].plot(data["time"][to1:tf1], data["Ly_momentum"][to1:tf1], label=data["name"])  #XY Plot
        ax_list1[4].set_title('Psuedo Ly vs Time')
        ax_list1[5].plot(data["time"][to1:tf1], data["Lz_momentum"][to1:tf1], label=data["name"])  #XY Plot
        ax_list1[5].set_title('Psuedo Lz vs Time')
        elapse_min = min(elapse_min, min(data["pos"][to1:tf1, 2]))
        elapse_max = max(elapse_max, max(data["pos"][to1:tf1, 2]))
        #print(elapse_min, elapse_max)

        ax_list2[0].plot(data["tracktime"][to2:tf2], data["energy"][to2:tf2], label=data["name"])  #XY Plot
        ax_list2[0].set_title('Energy vs Time')
        b, m = np.polynomial.polynomial.polyfit(list(data["tracktime"][to2:tf2]), data["energy"][to2:tf2], 1)
        ax_list2[0].plot(data["tracktime"][to2:tf2], b + m * data["tracktime"][to2:tf2], '-', label= str(m))
        ax_list2[1].plot(data["tracktime"][to2:tf2], data["phi_momentum"][to2:tf2], label=data["name"])  #XY Plot
        ax_list2[1].set_title('L_phi vs Time')
        b, m = np.polynomial.polynomial.polyfit(list(data["tracktime"][to2:tf2]), data["phi_momentum"][to2:tf2], 1)
        ax_list2[1].plot(data["tracktime"][to2:tf2], b + m * data["tracktime"][to2:tf2], '-', label= str(m))
        ax_list2[2].plot(data["tracktime"][to2:tf2], data["carter"][to2:tf2], label=data["name"])  #XZ Plot
        ax_list2[2].set_title('Carter(C) vs Time')
        b, m = np.polynomial.polynomial.polyfit(list(data["tracktime"][to2:tf2]), data["carter"][to2:tf2], 1)
        ax_list2[2].plot(data["tracktime"][to2:tf2], b + m * data["tracktime"][to2:tf2], '-', label= str(m))
        ax_list2[3].plot(data["tracktime"][to2:tf2], data["e"][to2:tf2], label=data["name"])  #ZY Plot
        ax_list2[3].set_title('Eccentricity vs Time')
        #print(data["e"][to2:tf2])
        #b, m = np.polynomial.polynomial.polyfit(list(data["tracktime"][to2:tf2]), data["e"][to2:tf2], 1)
        #ax_list2[3].plot(data["tracktime"][to2:tf2], b + m * data["tracktime"][to2:tf2], '-', label= str(m))
    

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
    for i in ax_list2:
        i.label_outer()
        #i.set_aspect('equal')
        if leg == True:
            i.legend()
    return ax_list1, ax_list2
