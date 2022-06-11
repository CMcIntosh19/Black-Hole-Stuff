# -*- coding: utf-8 -*-
"""
Created on Fri May 27 18:20:12 2022

@author: hepiz
"""


import numpy as np
import MainLoops as ml
import matplotlib.pyplot as plt
import OrbitPlotter as op
import operator

def per_diff(upp, low):
    return 2*abs(upp-low)/(upp+low) * 100


def run_it(radius, spin, err): 
    if err < 10**(-13):
        print("Desired percent error too small, reverting to 10^(-13)")
        err = 10**(-13)
    upper, lower = 10.0, 0.0
    mult = 2.0
    
    if spin >= 0.0:
        pro = 1
    else:
        pro = -1

    spin = abs(spin)
    launch = 1/(radius**(3/2) + spin)
    bound = (1 + (1 - pro*spin)**(1/2))**(2)
    small = radius
    
    orb_list = []
    ang_list = []
    
    while per_diff(upper, lower) > err:
        if small <= bound:
            lower = mult
            mult = (upper + lower)/2
        else:
            upper = mult
            mult = (upper + lower)/2
            
        print(str(mult) + " HEY LOOK OVER HERE ---------------------------------")
        print(upper, lower)
        print(per_diff(upper, lower))
        
        initial  = np.array([ [0.00, radius, np.pi/2, 0.00, 1.0, 0.00, 0.00, launch*mult] ])
        test0 = ml.inspiral_long(initial, 1, spin, 0, 1, np.pi/launch + 10000*(launch**(1/4)), 0.1, True, min(err, 10**(-11)), 90, pro*90, 'blah', verbose=False)
        
        small = min(test0["pos"][:, 0])
        print(small, "small")
        print(bound, "bound")
        
        if small >= bound:
            peaks = np.where( abs(test0["pos"][:,0] - radius) < (radius-bound)/1000)
            print(peaks)
            try:
                idx = peaks[0][np.where(np.diff(peaks)[0] > 10)[0][0] + 1]
                orbs = test0["pos"][idx, 2] - pro*2*np.pi
                orb_list.append(orbs)
                ang_list.append(test0["phi_momentum"][0])
            except:
                print("unsuited launch speed")
        

    
    diff = (upper - mult)/2
    
    while small < bound:
        mult = mult + diff
        initial  = np.array([ [0.00, radius, np.pi/2, 0.00, 1.0, 0.00, 0.00, launch*mult] ])
        test0 = ml.inspiral_long(initial, 1, spin, 0, 1, 2*(np.pi/launch + 10000*(launch**(1/4))), 0.1, True, min(err, 10**(-11)), 90, pro*90, 'blah', verbose=False)        
        small = min(test0["pos"][:, 0])
        print(str(mult) + " HEY WHAT THE HECK ++++++++++++++++++++++++++++++++++")
        print(small)
    
    diff2 = (1 - mult)/30
    hold = mult
    while hold <= 1:
        initial  = np.array([ [0.00, radius, np.pi/2, 0.00, 1.0, 0.00, 0.00, launch*hold] ])
        test0 = ml.inspiral_long(initial, 1, spin, 0, 1, 2*(np.pi/launch + 10000*(launch**(1/4))), 0.1, True, min(err, 10**(-11)), 90, pro*90, 'blah', verbose=False)
        peaks = np.where( abs(test0["pos"][:,0] - radius) < (radius-bound)/1000)
        print(peaks)
        try:
            idx = peaks[0][np.where(np.diff(peaks)[0] > 10)[0][0] + 1]
            orbs = test0["pos"][idx, 2] - pro*2*np.pi
            orb_list.append(orbs)
            ang_list.append(test0["phi_momentum"][0])
        except:
            print("unsuited launch speed")
        hold += diff2
        
    L = sorted(zip(ang_list, orb_list), key=operator.itemgetter(0))
    ang_list, orb_list = zip(*L)
    ang_list = np.array(ang_list)
    orb_list = np.array(orb_list)


    return (ang_list, orb_list, launch*mult)




'''
def run_big(radii, spins):
    radii = np.arange(12, 20, 5)
    spins = np.linspace(0, 1, num=11)
    data = np.empty([18, 11])
    for i in range(len(radii)):
        for j in range(len(spins)):
            test00 = run_it(radii[i], spins[j])
            data[i,j] = test00["raw"][0,-1]
    return radii, spins, data


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#r, s, thing = run_big()

#ax.plot_surface(r, s, thing)
'''
#X = np.arange(-5, 5, 0.25)
#Y = np.arange(-5, 5, 0.25)
#X, Y = np.meshgrid(X, Y)
#print(type(X), type(Y))
#R = np.sqrt(X**2 + Y**2)
#print(type(R))
#print(R)
#Z = np.sin(R)

#X = np.arange(10, 51, 3)
#Y = np.linspace(0, 0.9, num=10)
#Y = [0]
#Z = np.array([[run_it(radius, spin)["raw"][0,-1] for radius in X] for spin in Y])

# Plot the surface.
#surf = ax.plot_surface(X, Y, Z)

def plotter(X, Y):
    fig, ax = plt.subplots()
    for i in Y:
        ax.plot(X,i, marker="o", linestyle="none")
    return ax


#for plotting z info real quick
def quick_make(radii, rad_idx, spins, spn_idx, data):
    initial  = np.array([ [0.00, radii[rad_idx], np.pi/2, 0.00, 1.0, 0.00, 0.00, data[spn_idx,rad_idx] ]])
    test = ml.inspiral_long(initial, 1, spins[spn_idx], 0, 1, 5000, 0.1, True, 10**(-11), 90, 90, str(radii[rad_idx]) + "/" + str(spins[spn_idx]))
    op.orthoplots(test, merge=False)
    op.physplots(test, merge=False)

'''
spin = 0 - launch speed seems proportional to somewhere between r^(-1.96) and r^(-2)



X = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
Y = np.array([0. , 0.2788, 0.4472, 0.5682, 0.6628, 0.7404, 0.8062, 0.8633, 0.9138, 0.959])

x, y = np.meshgrid(X, Y)

Z = np.array([[0.03535533, 0.0161394 , 0.009245  , 0.00599205, 0.00419961, 
        0.00310701, 0.00239182, 0.00189814, 0.00154303],
       [0.03348007, 0.01519376, 0.0086724 , 0.00560754, 0.00392341,
        0.00289895, 0.00222943, 0.00176785, 0.0014362 ],
       [0.03206781, 0.01449787, 0.00825574, 0.00532969, 0.00372503,
        0.00275029, 0.00211391, 0.00167551, 0.00136069],
       [0.03085947, 0.0139104 , 0.00790709, 0.00509926, 0.00356131,
        0.00262789, 0.00201891, 0.00159962, 0.00129866],
       [0.02975218, 0.01337817, 0.00759457, 0.00489319, 0.00341505,
        0.00251864, 0.00193417, 0.00153197, 0.00124339],
       [0.02868999, 0.01287499, 0.00729973, 0.00469904, 0.0032774 ,
        0.00241593, 0.00185456, 0.00146845, 0.00119153],
       [0.02763316, 0.01237773, 0.00700886, 0.0045078 , 0.00314199,
        0.00231497, 0.00177638, 0.00140611, 0.00114066],
       [0.02653899, 0.01186354, 0.0067087 , 0.00431078, 0.00300266,
        0.0022112 , 0.00169609, 0.00134214, 0.00108848],
       [0.02533399, 0.01129839, 0.00637957, 0.00409514, 0.00285039,
        0.00209792, 0.00160853, 0.00127242, 0.00103165],
       [0.023857  , 0.01060772, 0.00597851, 0.00383298, 0.00266558,
        0.00196062, 0.00150249, 0.00118806, 0.00096294]])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x, y, Z, 50, cmap='viridis')
ax.set_xlabel('radius')
ax.set_ylabel('spin')
ax.set_zlabel('launch');
#ax.view_init(90, -90)

plotter(X, Z)
plotter(Y, Z.transpose())
'''

def whirl_range(radii, spins, launches, rad, spn):
    r, a, l = radii[rad], spins[spn], launches[spn, rad]
    freq = 1/(r**(3/2) + a)
    time = 6*np.pi/freq
    initial  = np.array([ [0.00, r, np.pi/2, 0.00, 1.0, 0.00, 0.00, l] ])
    test = ml.inspiral_long(initial, 1, a, 0, 1, time, 0.1, True, 10**(-11), 90, 90, 'blah', verbose=False)
    peaks = np.where( abs(test["pos"][:,0] - r)/r < 0.0001)

    try:
        idx = peaks[0][np.where(np.diff(peaks)[0] > 10)[0][0] + 1]
        orbs = test["pos"][idx, 2]/(2*np.pi)
    except:
        if min(test["pos"][:,0]) == r:
            orbs = 10
        else:
            orbs = 0
    mult = 1
    fail = 0
    
    upper, lower = 1.5, 0.5
    while (abs(orbs - 2)/2 > 0.0001) and (fail < 9):
        if orbs > 2:
            lower = mult
        else:
            upper = mult
        mult = (upper + lower)/2
        
        initial  = np.array([ [0.00, r, np.pi/2, 0.00, 1.0, 0.00, 0.00, l*mult] ])
        test = ml.inspiral_long(initial, 1, a, 0, 1, time, 0.1, True, 10**(-11), 90, 90, 'blah', verbose=False)
        peaks = np.where( abs(test["pos"][:,0] - r)/r < 0.0002)
        try:
            idx = peaks[0][np.where(np.diff(peaks)[0] > 10)[0][0] + 1]
            orbs = test["pos"][idx, 2]/(2*np.pi)
        except:
            if min(test["pos"][:,0]) == r:
                orbs = 10
            else:
                orbs = 0
            fail += 1
    
    if fail < 9:
        min_whirl = l*mult
    else:
        min_whirl = False
    
    return min_whirl
    
    
