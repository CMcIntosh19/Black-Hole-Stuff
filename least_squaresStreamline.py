# -0*- coding: utf-8 -*-
"""
Created on Thu Oct 13 00:09:16 2022

@author: hepiz
"""

import numpy as np
import MetricMathStreamline as mm
import matplotlib.pyplot as plt
import sympy as sp
import warnings
from random import uniform
from scipy.optimize import curve_fit

def getEnergy(state, a):
    metric, chris = mm.kerr(state, 1, a)
    stuff = np.matmul(metric, state[4:])
    ene = -stuff[0]
    return ene
    
def getLs(state, mu):
    r, theta, phi, vel4 = *state[1:4], state[4:]
    sint, cost = np.sin(theta), np.cos(theta)
    #print(sint, cost)
    #print(np.sin(theta), np.cos(theta))
    sinp, cosp = np.sin(phi), np.cos(phi)
    sph2cart = np.array([[1.0, 0.0,       0.0,         0.0    ],
                         [0.0, sint*cosp, r*cost*cosp, -r*sinp],
                         [0.0, sint*sinp, r*cost*sinp, r*cosp ],
                         [0.0, cost,      -r*sint,     0.0    ]])
    #print(sph2cart)
    vel4cart = np.matmul(sph2cart, vel4)
    vel3cart = vel4cart[1:4]
    pos3cart = np.array([r*sint*cosp, r*sint*sinp, r*cost])
    #print(pos3cart)
    #print(vel3cart)
    Lmom = np.cross(pos3cart, vel3cart)
    #print(Lmom)
    return Lmom
        
def new_recalc_state3(state, con_derv, mu, mass, a):
    r, theta, phi, vel4 = *state[1:4], state[4:]
    sint, cost = np.sin(theta), np.cos(theta)
    sinp, cosp = np.sin(phi), np.cos(phi)
    
    evhor = 1 - 2/r
    tet2cor = np.array([[evhor**(-1/2), 0,            0,   0         ],
                        [0,             evhor**(1/2), 0,   0         ],
                        [0,             0,            1/r, 0         ],
                        [0,             0,            0,   1/(r*sint)]])
    cor2tet = np.linalg.inv(tet2cor)
    sph2cart = np.array([[1, 0,         0,         0         ],
                         [0, sint*cosp, cost*cosp, -sint*sinp],
                         [0, sint*sinp, cost*sinp, sint*cosp ],
                         [0, cost,      -sint,     0         ]])
    cart2sph = np.linalg.inv(sph2cart)
    metric, chris = mm.kerr(state, 1, a)
    bigA = np.zeros([4,3])
    
    cart_tet_state = np.matmul(sph2cart, np.matmul(cor2tet, vel4))
    strip_ct_state = (cart_tet_state[1:4])/(cart_tet_state[0])
    
    for i in range(3):
        dvel = np.array([0.0, 0.0, 0.0])
        dvel[i] = 10**(-7)
        print(dvel)
        new_strip_ct_state = strip_ct_state + dvel
        newgamma = (1 - np.linalg.norm(new_strip_ct_state)**2)**(-1/2)
        new_ct_state = newgamma*np.array([1, *new_strip_ct_state])
        new_vel = np.matmul(tet2cor, np.matmul(cart2sph, new_ct_state))
        new_state = np.array([*state[:4], *new_vel])
        bigA[:, i] = [getEnergy(new_state, a), *getLs(new_state, mu)]

    dcons = con_derv[0:4]
    bigAt = np.transpose(bigA)
    block = np.linalg.inv(np.matmul(bigAt, bigA))
    dvel = np.matmul(block, np.matmul(bigAt, dcons))
    new_strip_ct_state = strip_ct_state + dvel
    newgamma = (1 - np.linalg.norm(new_strip_ct_state)**2)**(-1/2)
    new_ct_state = newgamma*np.array([1, *new_strip_ct_state])
    new_vel = np.matmul(tet2cor, np.matmul(cart2sph, new_ct_state))
    new_state = np.array([*state[:4], *new_vel])
    
    return(new_state)

def newLs(data):
    metrics = np.array([mm.kerr(state, 1.0, 0.1)[0] for state in data["raw"]])
    Lz = np.array([np.matmul(data["raw"][i][4:], np.matmul(metrics[i], [0, 0, 0, 1])) for i in range(len(metrics))])
    carts = np.array([[state[1]*np.sin(state[2])*np.cos(state[3]),
                       state[1]*np.sin(state[2])*np.sin(state[3]),
                       state[1]*np.cos(state[2]) ]  for state in data["raw"]])
    vels = np.array([[state[5]*np.sin(state[2])*np.cos(state[3]) + state[1]*state[6]*np.cos(state[2])*np.cos(state[3]) - state[1]*state[7]*np.sin(state[2])*np.sin(state[3]),
                      state[5]*np.sin(state[2])*np.sin(state[3]) + state[1]*state[6]*np.cos(state[2])*np.cos(state[3]) + state[1]*state[7]*np.sin(state[2])*np.cos(state[3]),
                      state[5]*np.cos(state[2]) - state[1]*state[6]*np.sin(state[2])] for state in data["raw"]])
    L_props = np.array([np.cross(carts[i], vels[i]) for i in range(len(carts))])
    Lx = (Lz/L_props[:,2])*L_props[:,0]
    Ly = (Lz/L_props[:,2])*L_props[:,1]
    return(Lx, Ly, Lz, L_props)

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

def l_locator2(energy, a):
    err = 10**(-12)
    lz, cart, mass = 4.0, 0.0, 1.0
    coeff = np.array([energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*mass*((a*energy - lz)**2 + cart), -cart*(a**2)])
    coeff2 = np.array([4*(energy**2 - 1), 3*2*mass, 2*((a**2)*(energy**2 - 1) - lz**2 - cart), 2*mass*((a*energy - lz)**2 + cart)])
    
    r0 = np.max(np.roots(coeff2))
    Rval = np.sum(coeff*np.array([r0**4, r0**3, r0**2, r0, 1.0]))
    Rvalst = []
    lastBest = (10000.0, 4.0)
    while Rval < 0.0 or Rval > err:
        if (Rval > 0.0) and (Rval < lastBest[0]):
            lastBest = (Rval, lz)
        Rvalst.append(Rval)
        lz += Rval*max(1.0/(2*r0*(lz*r0 + mass*(a*energy - lz))),  10**(-15) )
        coeff = np.array([energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*mass*((a*energy - lz)**2 + cart), -cart*(a**2)])
        coeff2 = np.array([4*(energy**2 - 1), 3*2*mass, 2*((a**2)*(energy**2 - 1) - lz**2 - cart), 2*mass*((a*energy - lz)**2 + cart)])
        while True in np.iscomplex(np.roots(coeff2)):
            lz = lz/2.0
            coeff = np.array([energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*mass*((a*energy - lz)**2 + cart), -cart*(a**2)])
            coeff2 = np.array([4*(energy**2 - 1), 3*2*mass, 2*((a**2)*(energy**2 - 1) - lz**2 - cart), 2*mass*((a*energy - lz)**2 + cart)])
        r0 = np.max(np.roots(coeff2))
        Rval = np.sum(coeff*np.array([r0**4, r0**3, r0**2, r0, 1.0]))
        #print(r0, Rval, lz)
        if Rvalst.count(Rval) >= 10:
            break
    if Rval < 0.0:
        lz = lastBest[1]
    circL = lz
    #print("yo")
    
    
    rL = np.sort(np.roots(coeff2))[-2]
    Rval = np.sum(coeff*np.array([rL**4, rL**3, rL**2, rL, 1.0]))
    Rvalst = []
    lastBest = (10000.0, 4.0)
    #print(np.sort(np.roots(coeff2)))
    while Rval < 0.0 or Rval > err:
        if (Rval > 0.0) and (Rval < lastBest[0]):
            lastBest = (Rval, lz)
        Rvalst.append(Rval)
        lz += 1.5*Rval*max(1.0/(2*rL*(lz*rL + mass*(a*energy - lz))),  10**(-15) )
        coeff = np.array([energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*mass*((a*energy - lz)**2 + cart), -cart*(a**2)])
        coeff2 = np.array([4*(energy**2 - 1), 3*2*mass, 2*((a**2)*(energy**2 - 1) - lz**2 - cart), 2*mass*((a*energy - lz)**2 + cart)])
        rL = np.sort(np.roots(coeff2))[-2]
        Rval = np.sum(coeff*np.array([rL**4, rL**3, rL**2, rL, 1.0]))
        #print(rL, Rval, lz)
        if Rvalst.count(Rval) >= 10:
            break
    if Rval < 0.0:
        lz = lastBest[1]
    critL = lz
    
    return critL, circL

def l_locator3(energy, a, inc=np.pi/2):
    #inc goes from pi/2 (equatorial) to 0.0 (polar)
    if inc == np.pi/2:
        return l_locator2(energy, a), (0.0, 0.0)
    mass, err = 1.0, 10**(-10)
    trueLmag = 4.0
    cart, lz =  (trueLmag*np.sin(inc))**2, trueLmag*np.cos(inc)
    print(cart, lz)
    coeff = np.array([energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*mass*((a*energy - lz)**2 + cart), -cart*(a**2)])
    coeff2 = np.array([4*(energy**2 - 1), 3*2*mass, 2*((a**2)*(energy**2 - 1) - lz**2 - cart), 2*mass*((a*energy - lz)**2 + cart)])
    
    r0 = np.max(np.roots(coeff2))
    Rval = np.sum(coeff*np.array([r0**4, r0**3, r0**2, r0, 1.0]))
    Rvalst = []
    lastBest = (10000.0, lz, cart)
    while Rval < 0.0 or Rval > err:
        if (Rval > 0.0) and (Rval < lastBest[0]):
            lastBest = (Rval, lz, cart)
        Rvalst.append(Rval)
        A = (cart/(lz**2))*(-(r0**2 - 2*mass*r0 + a**2))
        B = (-2*r0*(lz*r0 + mass*(a*energy - lz)) + 2*(cart/lz)*(-(r0**2 - 2*mass*r0 + a**2)))
        C = Rval*0.95
        dlz = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
        dcart = cart*(2*(dlz/lz) + (dlz/lz)**2)
        lz += dlz
        cart += dcart
        coeff = np.array([energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*mass*((a*energy - lz)**2 + cart), -cart*(a**2)])
        coeff2 = np.array([4*(energy**2 - 1), 3*2*mass, 2*((a**2)*(energy**2 - 1) - lz**2 - cart), 2*mass*((a*energy - lz)**2 + cart)])
        r0 = np.max(np.roots(coeff2))
        Rval = np.sum(coeff*np.array([r0**4, r0**3, r0**2, r0, 1.0]))
        print(Rval)
        if Rvalst.count(Rval) >= 10:
            break
    if Rval < 0.0:
        lz, cart = lastBest[1], lastBest[2]
    circL, circC = lz, cart
    
    rL = np.sort(np.roots(coeff2))[-2]
    Rval = np.sum(coeff*np.array([rL**4, rL**3, rL**2, rL, 1.0]))
    Rvalst = []
    lastBest = (10000.0, lz, cart)
    while Rval < 0.0 or Rval > err:
        if (Rval > 0.0) and (Rval < lastBest[0]):
            lastBest = (Rval, lz, cart)
        Rvalst.append(Rval)
        A = (cart/(lz**2))*(-(rL**2 - 2*mass*rL + a**2))
        B = (-2*rL*(lz*rL + mass*(a*energy - lz)) + 2*(cart/lz)*(-(rL**2 - 2*mass*rL + a**2)))
        C = Rval
        dlz = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
        dcart = cart*(2*(dlz/lz) + (dlz/lz)**2)
        lz += dlz
        cart += dcart
        coeff = np.array([energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*mass*((a*energy - lz)**2 + cart), -cart*(a**2)])
        coeff2 = np.array([4*(energy**2 - 1), 3*2*mass, 2*((a**2)*(energy**2 - 1) - lz**2 - cart), 2*mass*((a*energy - lz)**2 + cart)])
        rL = np.sort(np.roots(coeff2))[-2]
        Rval = np.sum(coeff*np.array([rL**4, rL**3, rL**2, rL, 1.0]))
        if Rvalst.count(Rval) >= 10:
            break
    if Rval < 0.0:
        lz, cart = lastBest[1], lastBest[2]
    critL, critC = lz, cart
    return (critL, circL), (critC, circC)

def getparams(vx, vy, vz, tet_mat, metric, r0, e0, i0, a, showme = False):
    warnings.filterwarnings("ignore")

    gamma = 1.0/np.sqrt(1 - (vx**2 + vy**2 + vz**2))
    tilde = np.array([ gamma,
                       gamma*vx,
                      -gamma*vz,
                       gamma*vy])
    final = np.matmul(tet_mat, tilde)
    new = np.array([0.0, r0, np.pi/2, 0.0, *final])
    
    E = -np.matmul(new[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
    L = np.matmul(new[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
    Q = np.matmul(np.matmul(mm.kill_tensor(new, 1.0, a), new[4:]), new[4:])               #new Carter constant Q
    C = abs(Q - (a*E - L)**2)
    
    #print(E, L, C)
    Rco = [E**2 - 1, 2, (a**2)*(E**2 - 1) - L**2 - C, 2*((a*E - L)**2 + C), -(a**2)*C]
    Rdco = [4*(E**2 - 1), 6, 2*((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C)]
    Tco = [(a**2)*(1 - E**2), -(C + (a**2)*(1 - E**2) + L**2), C]
    turns = np.roots(Rco)
    flats = np.roots(Rdco)
    incs = np.roots(Tco)
    incs = incs[(incs >= -1) & (incs <= 1)]
    
    rmax1, rmin1 = turns[0], turns[1]
    rc = (1 + np.sqrt(1 + a))**2
    e1 = (rmax1 - rmin1)/(rmax1 + rmin1 - 2*rc)
    r1 = np.real(flats[0])
    i1 = abs(np.pi/2 - np.arccos(np.sqrt(incs[0])))
    while (i1 > np.pi/2) or (i1 < 0.0):
        fix = np.sign(i1)
        i1 -= np.pi/2*fix
    
    rbound1, rbound2 = rmin1*0.7, rmax1*1.1
    rads1 = np.linspace(rbound1, rbound2, num = int(r0*2))
    
    if showme == True:
        R = lambda r: (E**2 - 1)*(r**4) + 2*(r**3) + ((a**2)*(E**2 - 1) - L**2 - C)*(r**2) + 2*((a*E - L)**2 + C)*r - (a**2)*C
        T = lambda t: C - (C + (a**2)*(1 - E**2) + L**2)*(np.cos(t)**2) + (a**2)*(1 - E**2)*(np.cos(t)**4)
        e = e0
        rads = np.linspace(0, r0*(1+e)*1.15, num = int(r0**2))
        thets = np.linspace(0.0, np.pi, num=50)
        fig, ax = plt.subplots()
        ax.plot(rads, R(rads))
        ax.plot(rads, np.zeros(int(r0**2)))
        ax.vlines([rmax1, rmin1], min(R(rads1)), max(R(rads1)), color='red')
        ax.vlines([r1], min(R(rads1)), max(R(rads1)), color='green')
        ax.vlines([r0*(1 + e), r0*(1 - e)], min(R(rads1)), max(R(rads1)), color='red', linestyle='dashed')
        ax.vlines([r0], min(R(rads1)), max(R(rads1)), color='green', linestyle='dashed')
        ax.set_title("err(r0, rmin) = (" + str(round(100*abs(r1-r0)/r0, 6)) + ", " + str(round(100*abs(rmin1-r0*(1 - e))/(r0*(1 - e)), 6)) + ")")
        ax.set_xlim(min(rbound1, r0*(1 - e)*0.9), max(rbound2, r0*(1+e)*1.05))
        ax.set_ylim(min(R(rads1)), max(R(rads1)) - 0.1*min(R(rads1)))
        
        fig1, ax1 = plt.subplots()
        ax1.plot(thets, T(thets))
        ax1.plot(thets, np.zeros(50))
        ax1.vlines([np.pi/2 + i1, np.pi/2 - i1], min(T(thets)), max(T(thets)), color='black')
        ax1.vlines([np.pi/2 + i0, np.pi/2 - i0], min(T(thets)), max(T(thets)), color='black', linestyle='dashed')
        ax1.set_title("err =" + str(100*abs(i1 - i0)/(np.pi/2)))

    stuff = np.array([r1, e1, i1])
    return stuff

def leastsquaresparam(r0, e, i, a, showme=False):
    #rmin = r0*(1 - e)
    #params = np.array([r0, rmin, i])
    icor = (i%(2*np.pi))//(np.pi/2)
    icor1 = i%(np.pi/2) if (icor%2 == 0) else np.pi - i%(np.pi)
    params = np.array([r0, e, icor1])
    r, theta = r0, np.pi/2
    pre1 = 1 if (icor < 1) or (icor > 2) else -1
    pre2 = 1 if (icor < 2) else -1
    
    v0 = (1/np.sqrt(r0-2) + 1/np.sqrt(r0))/2
    vx, vy, vz = v0/10.0, pre1*v0, pre2*v0/10.0
    vel = np.array([vx, vy, vz])
    pos = np.array([0.0, r0, theta, 0.0])
    
    #various defined values that make math easier
    metric, chris = mm.kerr(np.array([*pos, 0, 0, 0, 0]), 1.0, a)
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

    #get first guess
    velguess = getparams(*vel, tetrad_matrix, metric, r0, e, icor1, a)
    ack = 0
    while (np.linalg.norm(abs((velguess - params)/np.array([r0, 1, np.pi/2])))*100 > (10**(-10))) and (ack < 100):
        go = False
        eps = 10**(-5)
        while go == False and eps > 10**(-9):
            bigD = np.array([getparams(*(vel + np.array([eps, 0.0, 0.0])), tetrad_matrix, metric, r0, e, i, a) - velguess,
                             getparams(*(vel + np.array([0.0, eps, 0.0])), tetrad_matrix, metric, r0, e, i, a) - velguess,
                             getparams(*(vel + np.array([0.0, 0.0, eps])), tetrad_matrix, metric, r0, e, i, a) - velguess])/eps
            bigD2 = np.transpose(bigD)
            invBlock = np.linalg.inv(bigD2)
            delpam = params - velguess
            delvel = np.matmul(invBlock, delpam)
            if np.linalg.norm(vel + delvel) > 1.0:
                eps *= 0.25
            else: 
                go=True
        if np.linalg.norm(delvel)/np.linalg.norm(vel) > 0.15:   
            delvel *= 0.10*np.linalg.norm(vel)/np.linalg.norm(delvel)
        eps = 10**(-5)
        tryvel = vel + delvel
        velguess = getparams(*tryvel, tetrad_matrix, metric, r0, e, icor1, a)
        #print(velguess)
        flip = 0
        while True in np.iscomplex(velguess) and flip < 15:
            delvel *= 0.5
            tryvel = vel + delvel
            velguess = getparams(*tryvel, tetrad_matrix, metric, r0, e, icor1, a)
            flip += 1
        if flip < 15:
            vel = tryvel
        else:
            print("Exact match not found, giving best approximation")
            ack = 99
        velguess = getparams(*vel, tetrad_matrix, metric, r0, e, icor1, a, showme=showme)
        ack += 1
    #print(ack, (np.linalg.norm(abs((velguess - params)/np.array([r0, 1, np.pi/2])))*100))
    gamma = 1/np.sqrt(1 - np.linalg.norm(vel)**2)
    tilde = np.array([ gamma,
                       gamma*vel[0],
                      -gamma*vel[2],
                       gamma*vel[1]])
    final = np.matmul(tetrad_matrix, tilde)
    new = np.array([0.0, r0, np.pi/2, 0.0, *final])
    return new, velguess

def schwtest():
    r0, e, i = uniform(5, 500), uniform(0, 1), uniform(0, np.pi/2)
    state, ans = leastsquaresparam(r0, e, i, 0)
    val = np.linalg.norm(100*(np.array([r0, r0*(1-e), i]) - ans)/np.array([r0, r0*(1-e), 1]))
    if val > 10**(-6):
        metric, chris = mm.kerr(state, 1.0, 0.0)
        E = -np.matmul(state[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
        L = np.matmul(state[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
        Q = np.matmul(np.matmul(mm.kill_tensor(state, 1.0, 0.0), state[4:]), state[4:])               #new Carter constant Q
        C = abs(Q - (0.0*E - L)**2)
    return val

def kerrtest():
    r0, e, i, a = uniform(5, 500), uniform(0, 1), uniform(0, np.pi/2), uniform(0, 1)
    state, ans = leastsquaresparam(r0, e, i, a)
    val = np.linalg.norm(100*(np.array([r0, r0*(1-e), i]) - ans)/np.array([r0, r0*(1-e), 1]))
    if val > 100:
        metric, chris = mm.kerr(state, 1.0, a)
        E = -np.matmul(state[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
        L = np.matmul(state[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
        Q = np.matmul(np.matmul(mm.kill_tensor(state, 1.0, a), state[4:]), state[4:])               #new Carter constant Q
        C = abs(Q - (a*E - L)**2)
    return val

def errorplot(x, y, num=1000, binns=50):
    x = np.log(np.array([schwtest() for i in range(num)]))
    y = np.log(np.array([kerrtest() for i in range(num)]))
    xcounts, xbins = np.histogram(x, bins=binns)
    ycounts, ybins = np.histogram(y, bins=xbins)
    fig, ax = plt.subplots()
    ax.set_title("Histogram of Errors for Least-Squares Parameter Calculation")
    ax.set_ylabel("Count")
    ax.set_xlabel("Logarithm of error")
    ax.stairs(xcounts, xbins, label="Random r0, e, i; a=0")
    ax.stairs(ycounts, ybins, label="Random r0, e, i, a")
    ax.vlines(-6, 0, max(np.concatenate((xcounts, ycounts))), label="Maximum Accepted Error")
    ax.legend()

def schmidtparam3(r0, e, i, a, inner=False):
    p = r0*(1 - (e**2))  #p is semi-latus rectum, r0 is semimajor axis 
    rp, ra = p/(1 + e), p/(1 - e)
    polar = False
    j = i
    if i == 0.0 or i == np.pi:
        vals = np.transpose([[inc, *schmidtparam3(r0, e, inc, a, inner=True)] for inc in np.linspace(np.pi/2, i, endpoint=False)])
        Efit, Lfit, Cfit = np.polyfit(vals[0], vals[1], 10), np.polyfit(vals[0], vals[2], 10), np.polyfit(vals[0], vals[3], 10)
        E, L, C = np.polyval(Efit, i), np.polyval(Lfit, i), np.polyval(Cfit, i)
        coeff = np.array([E**2 - 1.0, 2.0, (a**2)*(E**2 - 1.0) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
        coeff2 = np.polyder(coeff)
        turns = np.roots(coeff)
        flats = np.roots(coeff2)
        r02, e2 = (turns[0] + turns[1])/2.0, (turns[0] - turns[1])/(turns[0] + turns[1])
        r_err, e_err = ((r0 - r02)/r0)*100, ((e2 - e)/(2-e))*100
        r_err *= np.sign(r_err)
        e_err *= np.sign(e_err)
        return [E, 0.0, C]
    z = abs(np.cos(j))
    
    def rfuncs(r):
        tri = r**2 - 2*r + a**2
        f = r**4 + (a**2)*(r*(r + 2) + tri*(z**2))
        g = 2*a*r
        h = r*(r - 2) + tri*(z**2)/(1 - z**2)
        d = (r**2 + (a*z)**2)*tri
        return f, g, h, d
    def r_funcs(r):
        f_ = 4*(r**3) + 2*(a**2)*((1 + z**2)*r + (1 - z**2))
        g_ = 2*a
        h_ = 2*(r - 1)/(1 - z**2)
        d_ = 2*(2*r - 3)*(r**2) + 2*(a**2)*((1 + z**2)*r - z**2)
        return f_, g_, h_, d_   
    
    if e == 0.0:
        f1, g1, h1, d1 = rfuncs(p)
        f2, g2, h2, d2 = r_funcs(p)
    else:
        f1, g1, h1, d1 = rfuncs(rp)
        f2, g2, h2, d2 = rfuncs(ra)
    
    def newC(E, L, a, z):
        return (z**2)*((a**2)*(1 - E**2) + (L**2)/(1 - z**2))
    
    x, y = sp.symbols("x y", real=True)
    eq1 = sp.Eq(f1*(x**2) - 2*g1*x*y - h1*(y**2), d1)
    eq2 = sp.Eq(f2*(x**2) - 2*g2*x*y - h2*(y**2), d2)
    symsols = sp.solve([eq1, eq2])

    full_sols = []
    E, L, C = (1 - (1 - e**2)/p)**0.5, ((1 - z**2)*p)**(0.5), p*(z**2)
    for thing in symsols:
        ene, lel = np.array([thing[x], thing[y]]).astype(float)
        if ene > 0.0 and ene < 1.0: 
            full_sols.append([ene, lel, newC(ene, lel, a, z)])
            E, L, C = [ene, lel, newC(ene, lel, a, z)]
            if np.product(np.sign([E, L, C])) == np.sign(np.sin(j)):
                break
            else:
                E, L, C = (1 - (1 - e**2)/p)**0.5, ((1 - z**2)*p)**(0.5), p*(z**2)

    coeff = np.array([E**2 - 1.0, 2.0, (a**2)*(E**2 - 1.0) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
    coeff2 = np.polyder(coeff)
    turns = np.roots(coeff)
    flats = np.roots(coeff2)
    ro = max(flats)
    while np.polyval(coeff, ro) < -1e-12:
        E += 10**(-16)
        coeff = np.array([E**2 - 1.0, 2.0, (a**2)*(E**2 - 1.0) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
        coeff2 = np.polyder(coeff)
        turns = np.roots(coeff)
        flats = np.roots(coeff2)
        ro = max(flats)
        #print("uh", np.polyval(coeff, ro), E)
    turns = np.sort(turns)
    
    r02, e2 = (turns[-1] + turns[-2])/2.0, (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
    r_err, e_err = ((r0 - r02)/r0)*100, ((e2 - e)/(2-e))*100
    r_err *= np.sign(r_err)
    e_err *= np.sign(e_err)

    if r_err < 1e-6 and e_err < 1e-6:
        return [E, L, C]
    else:
        def r__funcs(r):
            f__ = 12*(r**2) + 2*a*(1 + z**2)
            g__ = 0.0
            h__ = 2/(1 - z**2)
            d__ = 4*(r - 1)*(r + 3) + 2*(a**2)*(1 + z**2)
            return f__, g__, h__, d__  
        
        if False not in np.isreal([r_err, e_err]):
            f1, g1, h1, d1 = r_funcs(rp)
        else:
            f1, g1, h1, d1 = r__funcs(rp)

        f2, g2, h2, d2 = rfuncs(ra)
        
        eq1 = sp.Eq(f1*(x**2) - 2*g1*x*y - h1*(y**2), d1)
        eq2 = sp.Eq(f2*(x**2) - 2*g2*x*y - h2*(y**2), d2)

        symsols = sp.solve([eq1, eq2])
        
        full_sols = []
        for thing in symsols:
            ene, lel = np.array([thing[x], thing[y]]).astype(float)
            if ene > 0.0: 
                full_sols.append([ene, lel, newC(ene, lel, a, z)])
        
        for solution in full_sols:
            if (np.product(np.sign(solution)) == np.sign(np.sin(i))):
                E, L, C = solution
                break     
        if polar == True:
            L, C = 0.0, C/(z**2) - (a**2)*(1 - E**2)
        return [E, L, C]
    
def seper_locator3(r0, inc, a):
    print(r0)
    test_r = find_rmb(a)[1]
    e = 1 - test_r/r0
    small, big = 0.0, 1.0
    E, L, C = schmidtparam3(r0, e, np.pi/2, a)
    R = lambda r: ((r**2 + a**2)*E - a*L)**2 - (r**2 - 2*r + a**2)*(r**2 + (L - a*E)**2 + C)
    r1, bloh, bluh = np.array(np.roots([4*(E**2 - 1), 6, 2*((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C)]))
    val = R(bloh)
    while ((val > 0) or (val < -1e-12)) and (big-small > 1e-14):
        #print(val, e, small, big, big-small, bloh)
        if val > 0:
            big = e
        else:
            small = e
        e = (2*big + small)/3
        E, L, C = schmidtparam3(r0, e, np.pi/2, a)
        R = lambda r: ((r**2 + a**2)*E - a*L)**2 - (r**2 - 2*r + a**2)*(r**2 + (L - a*E)**2 + C)
        r1, bloh, bluh = np.array(np.roots([4*(E**2 - 1), 6, 2*((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C)]))
        val = R(bloh)
    return e