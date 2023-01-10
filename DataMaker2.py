# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:54:06 2022

@author: hepiz
"""

import numpy as np
import matplotlib.pyplot as plt
import MetricMath as mm
import MainLoops as ml
import datetime as dt
import time
from scipy.optimize import fsolve


def get_big_data(t):
    #t = 1200000 should give at least one full precession
    energy = 0.985
    lmoms = np.linspace(3.88, 5.97, 5)
    spins = [0.0, 0.01, 0.1]
    
    orbitdict = {}
    runs = 1
    print(dt.datetime.now().time())
    for lel in lmoms:
        maxcart = (lel**2)/2.0
        carts = (np.linspace(0.0, 1.0, 5)**2)*maxcart
        for cart in carts:
            truelel = np.sqrt(lel**2 - cart)
            initial = np.array([ [0.00, 32.0, np.pi/2, 0.00, energy, truelel, cart] ])
            for a in spins:
                label = str(a)+ "," + str(np.round(truelel,2)) + "," + str(np.round(cart,2))
                inc = round(np.arccos(truelel/np.sqrt(truelel**2 + cart))*(180/np.pi), 1)
                petexp = ml.inspiral_long1(initial, 1.0, a, 10**(-4), 1, t, 0.1, True, 10**(-13), 90, 90, "exp,"+label, verbose=False)
                petint = ml.inspiral_long5(initial, 1.0, a, 10**(-4), 1, t, 0.1, True, 10**(-13), 90, 90, "int,"+label, verbose=False)
                orbitdict["exp", a, np.round(lel,2), inc] = petexp
                orbitdict["int", a, np.round(lel,2), inc] = petint
                print("Done with", runs, "out of 75 total runs")
                print(dt.datetime.now().time())
                runs += 1
    return orbitdict

def l_locator(ene, a, dec):
    low = 2.0
    A, B, C, D = ene**2 - 1, 2, (a*ene + low)*(a*ene - low) - a**2, 2*(a*ene - low)**2
    val = (B*C)**2 - 4*A*(C**3) - 4*(B**3)*D - 27*((A*D)**2) + 18*A*B*C*D
    count = 0
    for i in range(dec):
        hold = low
        while val <= 0.0:
            low += (10**(-i-1))
            low = round(low, dec)
            A, B, C, D = ene**2 - 1, 2, (a*ene + low)*(a*ene - low) - a**2, 2*(a*ene - low)**2
            val = (B*C)**2 - 4*A*(C**3) - 4*(B**3)*D - 27*((A*D)**2) + 18*A*B*C*D
            #print(low, val)
            count += 1
            #print("fuck")
            if low > 4.1:
                print("boo")
                low = hold
                print(low)
                break
        if i+1 < dec:
            low -= (10**(-i-1))
            low = round(low, dec)
            A, B, C, D = ene**2 - 1, 2, (a*ene + low)*(a*ene - low) - a**2, 2*(a*ene - low)**2
            val = (B*C)**2 - 4*A*(C**3) - 4*(B**3)*D - 27*((A*D)**2) + 18*A*B*C*D
    high = low
    A, B, C, D = ene**2 - 1, 2, (a*ene + high)*(a*ene - high) - a**2, 2*(a*ene - high)**2
    val = (B*C)**2 - 4*A*(C**3) - 4*(B**3)*D - 27*((A*D)**2) + 18*A*B*C*D
    count = 0
    for i in range(dec ):
        while val > 0.0:
            high += (10**(-i-1))
            high = round(high, dec)
            A, B, C, D = ene**2 - 1, 2, (a*ene + high)*(a*ene - high) - a**2, 2*(a*ene - high)**2
            val = (B*C)**2 - 4*A*(C**3) - 4*(B**3)*D - 27*((A*D)**2) + 18*A*B*C*D
            #print(high, val)
            count +=1
        high -= (10**(-i-1))
        high = round(high, dec)
        A, B, C, D = ene**2 - 1, 2, (a*ene + low)*(a*ene - low) - a**2, 2*(a*ene - low)**2
        val = (B*C)**2 - 4*A*(C**3) - 4*(B**3)*D - 27*((A*D)**2) + 18*A*B*C*D
    return low, high

def new_big_data(t):
    #t = 1200000 should give at least one full precession
    orbitdict = {}
    energies = np.linspace(0.95, 0.995, 5)
    spins = np.linspace(0.0, 1.0, 5)
    incs = np.linspace(np.pi/4, np.pi/2, 5)
    mus = 10**(-np.array([3.0, 4.0, 5.0, 6.0, 7.0]))
    
    runs = 1
    start = time.time()
    for ene in energies:
        for a in spins:
            low, high = l_locator(ene, a, 2)
            lmoms = np.linspace(low, high, 5)
            for lel in lmoms:
                for theta in incs:
                    lz = np.sin(theta)*np.sqrt(lel**2 + (a**2)*(ene**2 - 1)*(np.cos(theta)**2))
                    cart = (np.cos(theta)**2)*(lel**2 - (a**2)*(ene**2 - 1)*(np.sin(theta)**2))
                    for mu in mus:
                        start = -0.5*(ene/(ene-1.0) + (1+mu)**(-2))
                        label = str(round(ene,2)) + ", " + str(round(lz,2)) + ", " + str(round(cart,2))
                        print("Running", str((ene, lz, cart)))
                        initial  = np.array([ [0.00, start, np.pi/2, 0.00, ene, lz, cart] ])
                        orbitdict[round(ene,3), round(lel, 3), round(theta*180/np.pi)] = ml.inspiral_long5(initial, 1, a, mu, 1, t, 0.1, True, 10**(-13), 90, 90, label, verbose=False)
                        current = time.time()
                        avg = (current - start)/runs
                        remain, unit = avg*(3125-runs), "seconds"
                        if remain > 60:
                            remain, unit = remain/60, "minutes"
                            if remain > 60:
                                remain, unit = remain/60, "hours"
                                if remain > 24:
                                    remain, unit = remain/24, "days"
                        print("Estimated time remaining:", str(round(remain,2)), unit)
                        runs += 1
    return orbitdict

def l_vectorplot(data):
    ax = plt.axes(projection='3d')
    time = data["time"]
    xdata = data["Lx_momentum"]
    ydata = data["Ly_momentum"]
    zdata = data["Lz_momentum"]
    xy = [*xdata, *ydata]
    xymin, xymax = min(xy), max(xy)
    diff = max(zdata) - min(zdata)
    ax.set_xlim3d(xymin, xymax)
    ax.set_ylim3d(xymin, xymax)
    ax.set_zlim3d(min(zdata)-diff, max(zdata)+diff)
    ax.scatter3D(xdata, ydata, zdata, c=time, cmap='coolwarm')
    
    return(ax)

def set_u_kerr2(**kwargs):
    '''
    set_u function normalizes an input state vector according to kerr metric

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
            OR
            7 element list/numpy array
        4-position, Energy, Phi-momentum, and Carter constant (C) of the test particle at a particular moment
    mass : int/float
        mass of black hole in arbitrary units. Generally set to 1
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    timelike : boolean
        specifies type of spacetime trajectory. Generally set to True
    eta : float
        angular displacement between desired trajectory and positive r direction,
        measured in radians
    xi : float
        angular displacement between desired trajectory and positive phi direction,
        measured in radians
    special : boolean/string
        if false, run normalization with data as given
        if "circle", modify parameters to create a circular orbit
            at the given radius
        if "zoom", modify parameters to create a 'zoom-whirl' orbit
            (in progress)

    Returns
    -------
    new : 8 element numpy array
        4-position and 4-velocity of the test particle at a particular moment

    '''
    
    try:
        energy = kwargs["energy"]
        lmom = kwargs["lmom"]
        cart = kwargs["cart"]
        a = kwargs["a"]
        vers = 0
    except:
        try:
            v, q, y, e = np.sqrt(1/kwargs["r0"]), kwargs["a"], np.tan(kwargs["i"])**2, kwargs["e"]
            energy = 1 - 0.5*(v**2) + 0.375*(v**4) - q*(v**5) - (0.5*(v**2) - 0.25*(v**4) + 2*q*(v**5))*(e**2) + 0.5*q*(v**5)*y + q*(v**5)*(e**2)*y
            lmom = kwargs["r0"]*v*(1 + 1.5*(v**2) - 3*q*(v**3) + 3.375*(v**4) + (q**2)*(v**4) - 7.5*q*(v**5)
                                   + (-1 + 1.5*(v**2) - 6*q*(v**3) + 10.125*(v**4) + 3.5*(q**2)*(v**4) - 31.5*q*(v**5))*(e**2)
                                   + (-0.5 - 0.75*(v**2) + 3*q*(v**3) - 1.6875*(v**4) - 1.5*(q**2)*(v**4) + 7.5*q*(v**5))*y
                                   + (0.5 - 0.75*(v**2) + 6*q*(v**3) - 5.0625*(v**4) - 4.75*(q**2)*(v**4) - 31.5*q*(v**5))*(e**2)*y)
            cart = y*(lmom**2)
            vers = 1
        except:
            try:
                pos, vel = kwargs["pos"], np.array([1.0, kwargs["vx"], kwargs["vy"], kwargs["vz"]])
                a = kwargs["a"]
                vers = 2
            except:
                print("Insufficient starting data. Refer to documentation for viable data combinations")
                return False
    
    if vers == 0:
        lrange = l_locator(energy, a, 4)
        psuedoL = np.sqrt(lmom**2 + cart)
        if psuedoL >= lrange[1]*1.03:
            print("L and C combination nonviable, attempting to scale down while maintaining inclination.")
            lmom = lmom*(lrange[1]/psuedoL)
            cart = lrange[1]**2 - lmom**2
        try:
            pos = kwargs["pos"]
        except:
            try:
                pos = np.array([0.0, kwargs["r0"], np.pi/2, 0.0])
            except:
                pos = np.array([0.0, 1/(2*(1-energy)), np.pi/2, 0.0])
        new = recalc_state(np.array([energy, lmom, cart]), np.array([*pos, 0.0, 0.0, 0.0]), 1.0, a)
    elif vers == 1:
        try:
            pos = kwargs["pos"]
        except:
            pos = np.array([0.0, kwargs["r0"], np.pi/2, 0.0])
        new = recalc_state(np.array([energy, lmom, cart]), np.array([*pos, 0.0, 0.0, 0.0]), 1.0, a)
    elif vers == 2:
        r, theta = pos[1], pos[2]
        metric, chris = kerr(np.array([*pos,*vel]), 1.0, a)
        #various defined values that make math easier
        rho2 = r**2 + (a**2)*(np.cos(theta)**2)
        tri = r**2 - 2*r + a**2
        al2 = (rho2*tri)/(rho2*tri + 2*r*((a**2) + (r**2)))
        w = (2*r*a)/(rho2*tri + 2*r*((a**2) + (r**2)))
        wu2 = ((rho2*tri + 2*r*((a**2) + (r**2)))/rho2)*(np.sin(theta)**2)
        new = np.array([*pos[:4], 0, 0, 0, 0])
        tetrad_matrix = np.array([[1/(np.sqrt(al2)), 0,                 0,               0],
                                  [0,                np.sqrt(tri/rho2), 0,               0],
                                  [0,                0,                 1/np.sqrt(rho2), 0],
                                  [w/np.sqrt(al2),   0,                 0,               1/np.sqrt(wu2)]])

        rdot, thetadot, phidot = vel[1]/vel[0], vel[2]/vel[0], vel[3]/vel[0],
        vel_2 = (rdot**2 + (r * thetadot)**2 + (r * np.sin(theta) * phidot)**2)
        beta = np.sqrt(vel_2)
        if (beta >= 1):
            print("Velocity greater than or equal to speed of light. Setting beta to 0.1")
            beta = 0.1
        gamma = 1/np.sqrt(1 - beta**2)
        tilde = np.array([ gamma,
                           gamma*beta*(rdot/beta),
                          -gamma*beta*(np.sqrt((r * thetadot)**2 + (r * np.sin(theta) * phidot)**2)/beta)*(r*np.sin(theta)*phidot/beta),
                           gamma*beta*(np.sqrt((r * thetadot)**2 + (r * np.sin(theta) * phidot)**2)/beta)*((np.sqrt(rdot**2 + (r * np.sin(theta) * phidot)**2))/beta)])
        test = np.copy(new)
        test[4:] = tilde
        final = np.matmul(tetrad_matrix, tilde)
        new[4:] = final
    return new
        