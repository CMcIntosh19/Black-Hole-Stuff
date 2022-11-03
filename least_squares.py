# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 00:09:16 2022

@author: hepiz
"""

import numpy as np
import MetricMath as mm

def getEnergy(state, a):
    metric, chris = mm.kerr(state, 1, a)
    stuff = np.matmul(metric, state[4:])
    ene = -stuff[0]
    return ene
    
def getLs(state, mu):
    t, r, theta, phi, vel4 = *state[:4], state[4:]
    sint, cost = mm.fix_sin(theta), mm.fix_cos(theta)
    #print(sint, cost)
    #print(np.sin(theta), np.cos(theta))
    sinp, cosp = mm.fix_sin(phi), mm.fix_cos(phi)
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

def getC(state, mu):
    Lmom = getLs(state,mu)
    Lnorm = np.linalg.norm(Lmom)
    return(Lmom[0]**2 + Lmom[1]**2)

def makeAmatrix(state, a, mu):
    t, r, theta, phi, vel4 = *state[:4], state[4:]
    sint, cost = mm.fix_sin(theta), mm.fix_cos(theta)
    sinp, cosp = mm.fix_sin(phi), mm.fix_cos(phi)
    
    evhor = 1 - 2/r
    tet2cor = np.array([[evhor**(-1/2), 0,            0,   0         ],
                        [0,             evhor**(1/2), 0,   0         ],
                        [0,             0,            1/r, 0         ],
                        [0,             0,            0,   1/(r*sint)]])
    cor2tet = np.linalg.inv(tet2cor)
    sph2cart = np.array([[1.0, 0.0,       0.0,       0.0  ],
                         [0.0, sint*cosp, cost*cosp, -sinp],
                         [0.0, sint*sinp, cost*sinp, cosp ],
                         [0.0, cost,      -sint,     0.0  ]])
    cart2sph = np.linalg.inv(sph2cart)
    metric, chris = mm.kerr(state, 1, a)
    bigA = np.zeros([4,3])
    
    cart_tet_state = np.matmul(sph2cart, np.matmul(cor2tet, vel4))
    strip_ct_state = (cart_tet_state[1:4])/(cart_tet_state[0])
    print(strip_ct_state)
    
    for i in range(3):
        dvel = np.array([0.0, 0.0, 0.0])
        dvel[i] = 10**(-10)
        #print("woo")
        #print(dvel)
        new_strip_ct_state = strip_ct_state + dvel
        print("new strip state")
        print(new_strip_ct_state)
        newgamma = (1 - np.linalg.norm(new_strip_ct_state)**2)**(-1/2)
        new_ct_state = newgamma*np.array([1, *new_strip_ct_state])
        new_vel = np.matmul(tet2cor, np.matmul(cart2sph, new_ct_state))
        new_state = np.array([*state[:4], *new_vel])
        print(state)
        print(new_state)
        old_cons = np.array([getEnergy(state, a), *getLs(state, mu)])
        new_cons = np.array([getEnergy(new_state, a), *getLs(new_state, mu)])
        print(new_cons)
        print(old_cons)
        del_cons = new_cons - old_cons                                         #Using newcons - oldcons instead of assuming delcons is linear like Jeremy said
        print(del_cons)
        bigA[:, i] = del_cons/dvel[i]
        #print(bigA[:, i])
        print(" ")
        
    return bigA
        
def new_recalc_state3(state, con_derv, mu, mass, a):
    t, r, theta, phi, vel4 = *state[:4], state[4:]
    sint, cost = mm.fix_sin(theta), mm.fix_cos(theta)
    sinp, cosp = mm.fix_sin(phi), mm.fix_cos(phi)
    
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
    carts = np.array([[state[1]*mm.fix_sin(state[2])*mm.fix_cos(state[3]),
                       state[1]*mm.fix_sin(state[2])*mm.fix_sin(state[3]),
                       state[1]*mm.fix_cos(state[2]) ]  for state in data["raw"]])
    vels = np.array([[state[5]*mm.fix_sin(state[2])*mm.fix_cos(state[3]) + state[1]*state[6]*mm.fix_cos(state[2])*mm.fix_cos(state[3]) - state[1]*state[7]*mm.fix_sin(state[2])*mm.fix_sin(state[3]),
                      state[5]*mm.fix_sin(state[2])*mm.fix_sin(state[3]) + state[1]*state[6]*mm.fix_cos(state[2])*mm.fix_cos(state[3]) + state[1]*state[7]*mm.fix_sin(state[2])*mm.fix_cos(state[3]),
                      state[5]*mm.fix_cos(state[2]) - state[1]*state[6]*mm.fix_sin(state[2])] for state in data["raw"]])
    L_props = np.array([np.cross(carts[i], vels[i]) for i in range(len(carts))])
    Lx = (Lz/L_props[:,2])*L_props[:,0]
    Ly = (Lz/L_props[:,2])*L_props[:,1]
    return(Lx, Ly, Lz, L_props)