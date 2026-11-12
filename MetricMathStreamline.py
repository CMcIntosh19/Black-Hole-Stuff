# -*- coding: utf-8 -*-
"""
Metric Math stuff
"""

import numpy as np
from scipy import optimize
from scipy.optimize import brentq
from scipy.signal import argrelmin
import scipy.interpolate as spi
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sympy as sp
from numba import njit
from tqdm import tqdm
import OrbitPlotter as op
import warnings
from scipy.optimize import brentq
from itertools import product

def find_rmb(spin):
    '''
    Brief calculations for marginally bound orbits

    Parameters
    ----------
    spin : float
        Dimensionless spin constant of black hole, between -1 and 1 inclusive

    Returns
    -------
    r_mb: float
        Periapse of an equatorial marginally BOUND orbit
    '''

    return (1 + np.sqrt(1 - spin))**2

def find_rms(spin):
    '''
    Brief calculations for marginally stable orbits (ISCO)

    Parameters
    ----------
    spin : float
        Dimensionless spin constant of black hole, between -1 and 1 inclusive

    Returns
    -------
    r_ms: float
        Radius of an equatorial marginally STABLE bound orbit
    '''
    if spin >= 0.0:
        pro = 1.0
    else:
        pro = -1.0
        
    z1 = 1 + ((1 - spin**2)**(1/3))*(((1 + spin)**(1/3)) + ((1 - spin)**(1/3)))
    z2 = np.sqrt(3*(spin**2) + z1**2)
    r_ms = 3 + z2 - pro*np.sqrt((3 - z1)*(3 + z1 + 2*z2))
    return r_ms

def find_rph(spin):
    '''
    Brief calculations for photon orbit

    Parameters
    ----------
    spin : float
        Dimensionless spin constant of black hole, between -1 and 1 inclusive

    Returns
    -------
    r_ph: float
        Radius of an equatorial photon orbit
    '''
    return 2*(1 + np.cos((2/3)*np.arccos(-spin)))

def mink(state):
    '''
    mink function generates metric and christoffel symbols
    for the minkowski metric. Mostly for testing purposes

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment

    Returns
    -------
    metric : 4x4 list
        Spacetime metric in terms of coordinate directions
    chris : dictionary {string: float}
        List of connection terms between coordinate directions

    '''
    metric = [[-1,   0,   0,    0],
              [0,    1,   0,    0],
              [0,    0,   1,    0],
              [0,    0,   0,    1]]
    chris_tens = np.zeros((4,4,4))
    return (metric, chris_tens)

def schwarz(state):
    '''
    schwarz function generates metric and christoffel symbols
    for the schwarzschild metric.

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment

    Returns
    -------
    metric : 4x4 list
        Spacetime metric in terms of coordinate directions
    chris : 4x4x4 numpy array
        List of connection terms between coordinate directions

    '''
    r, theta = state[1], state[2]
    metric = np.array([[-(1-(2/r)),          0,                        0,    0],
                       [0,                   (1-((2)/r))**(-1),        0,    0],
                       [0,                   0,                        r**2, 0],
                       [0,                   0,                        0,    (r**2) * (np.sin(theta))**2]])
    chris_tens = np.zeros((4,4,4))
    chris_tens[0,0,1] = 1 / (r * (r - 2))
    chris_tens[0,1,0] = 1 / (r * (r - 2))
    chris_tens[1,0,0] = (1 / r**3) * (r - 2)
    chris_tens[1,1,1] = -1 / (r * (r - 2))
    chris_tens[1,2,2] = -(r - 2)
    chris_tens[1,3,3] = -(r - 2) * np.sin(theta)**2
    chris_tens[2,1,2] = 1/r
    chris_tens[2,2,1] = 1/r
    chris_tens[2,3,3] = -np.sin(theta) * np.cos(theta)
    chris_tens[3,1,3] = 1/r
    chris_tens[3,3,1] = 1/r
    chris_tens[3,2,3] = np.cos(theta) / np.sin(theta)
    chris_tens[3,3,2] = np.cos(theta) / np.sin(theta)
    return (metric, chris_tens)

def kerr(state, a):
    '''
    kerr function generates metric and christoffel symbols
    for the kerr metric. should be identical to schwarz for a=0

    Parameters
    ----------
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : float
        dimensionless spin constant of black hole, between -1 and 1 inclusive

    Returns
    -------
    metric : 4x4 numpy array
        Spacetime metric in terms of coordinate directions t, r, theta, phi
    chris : 4x4x4 numpy array
        List of connection terms between coordinate directions

    '''
    r, theta, a = np.real(state[1]), np.real(state[2]), np.real(a)
    sine, cosi = np.sin(theta), np.cos(theta)
    cota = cosi/sine if sine != 0.0 else 1e100
    #various defined values that make math easier
    rho2, tri = r**2 + (a*cosi)**2, r**2 - 2*r + a**2
    al2, w = (rho2*tri)/(rho2*tri + 2*r*(a**2 + r**2)), (2*r*a)/(rho2*tri + 2*r*(a**2 + r**2))
    wu2 = ((rho2*tri + 2*r*(a**2 + r**2))/(rho2))*sine**2
    bigA = (r**2 + a**2)**2 - tri*(a*sine)**2
    metric = np.array([[-al2 + wu2*(w**2), 0.0,             0.0,    -w*wu2 ],
                       [0.0,               rho2/tri,        0.0,    0.0    ],
                       [0.0,               0.0,             rho2,   0.0    ],
                       [-w*wu2,            0.0,             0.0,    wu2    ]])
    chris_tens = np.zeros((4,4,4))
    chris_tens[0,0,1] = 2*(r**2 + a**2)*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2))
    chris_tens[0,1,0] = 2*(r**2 + a**2)*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2))
    chris_tens[0,0,2] = -2*(a**2)*r*sine*cosi/(rho2**2)
    chris_tens[0,2,0] = -2*(a**2)*r*sine*cosi/(rho2**2)
    chris_tens[0,1,3] = 2*a*(sine**2)*(((a*cosi)**2)*(a**2 - r**2) - (r**2)*(a**2 + 3*(r**2)))/(2*(rho2**2)*tri)
    chris_tens[0,3,1] = 2*a*(sine**2)*(((a*cosi)**2)*(a**2 - r**2) - (r**2)*(a**2 + 3*(r**2)))/(2*(rho2**2)*tri)
    chris_tens[0,2,3] = 2*r*cosi*((a*sine)**3)/(rho2**2)
    chris_tens[0,3,2] = 2*r*cosi*((a*sine)**3)/(rho2**2)
    chris_tens[1,0,0] = 2*tri*(r**2 - (a*cosi)**2)/(2*(rho2**3))
    chris_tens[1,0,3] = -tri*2*a*(sine**2)*(r**2 - (a*cosi)**2)/(2*(rho2**3))
    chris_tens[1,3,0] = -tri*2*a*(sine**2)*(r**2 - (a*cosi)**2)/(2*(rho2**3))
    chris_tens[1,1,1] = (2*r*((a*sine)**2) - 2*(r**2 - (a*cosi)**2))/(2*rho2*tri)
    chris_tens[1,1,2] = -(a**2)*sine*cosi/rho2
    chris_tens[1,2,1] = -(a**2)*sine*cosi/rho2
    chris_tens[1,2,2] = -r*tri/rho2
    chris_tens[1,3,3] = (tri*(sine**2)/(2*(rho2**3)))*(-2*r*(rho2**2) + 2*((a*sine)**2)*(r**2 - (a*cosi)**2))
    chris_tens[2,0,0] = -2*(a**2)*r*sine*cosi/(rho2**3)
    chris_tens[2,0,3] = 2*a*r*(r**2 + a**2)*sine*cosi/(rho2**3)
    chris_tens[2,3,0] = 2*a*r*(r**2 + a**2)*sine*cosi/(rho2**3)
    chris_tens[2,1,1] = (a**2)*sine*cosi/(rho2*tri)
    chris_tens[2,1,2] = r/rho2
    chris_tens[2,2,1] = r/rho2
    chris_tens[2,2,2] = -(a**2)*sine*cosi/rho2
    chris_tens[2,3,3] = -(sine*cosi/(rho2**3))*(bigA*rho2 + (r**2 + a**2)*2*r*((a*sine)**2))
    chris_tens[3,0,1] = 2*a*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2))
    chris_tens[3,1,0] = 2*a*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2))
    chris_tens[3,0,2] = -2*a*r*cota/(rho2**2)
    chris_tens[3,2,0] = -2*a*r*cota/(rho2**2)
    chris_tens[3,1,3] = (2*r*(rho2**2) + 2*(((a**2)*sine*cosi)**2 - (r**2)*(rho2 + r**2 + a**2)))/(2*tri*(rho2**2))
    chris_tens[3,3,1] = (2*r*(rho2**2) + 2*(((a**2)*sine*cosi)**2 - (r**2)*(rho2 + r**2 + a**2)))/(2*tri*(rho2**2))
    chris_tens[3,2,3] = (cota/(rho2**2))*((rho2**2) + 2*r*((a*sine)**2))
    chris_tens[3,3,2] = (cota/(rho2**2))*((rho2**2) + 2*r*((a*sine)**2))
    return (metric, chris_tens)

@njit
def kerr_2(state, a):
    r, theta, a = np.real(state[1]), np.real(state[2]), np.real(a)
    sine, cosi = np.sin(theta), np.cos(theta)
    cota = cosi/sine if abs(sine) >= 1e-15 else cosi/(np.sign(sine)*1e-15)
    rho2, tri = r**2 + (a*cosi)**2, r**2 - 2*r + a**2
    al2 = (rho2*tri)/(rho2*tri + 2*r*(a**2 + r**2))
    w = (2*r*a)/(rho2*tri + 2*r*(a**2 + r**2))
    wu2 = ((rho2*tri + 2*r*(a**2 + r**2))/(rho2))*sine**2
    bigA = (r**2 + a**2)**2 - tri*(a*sine)**2

    metric = np.array([[-al2 + wu2*(w**2), 0.0,             0.0,    -w*wu2 ],
                       [0.0,               rho2/tri,        0.0,    0.0    ],
                       [0.0,               0.0,             rho2,   0.0    ],
                       [-w*wu2,            0.0,             0.0,    wu2    ]])

    chris_tens = np.zeros((4,4,4))
    chris_tens[0,0,1] = 2*(r**2 + a**2)*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2))
    chris_tens[0,1,0] = 2*(r**2 + a**2)*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2))
    chris_tens[0,0,2] = -2*(a**2)*r*sine*cosi/(rho2**2)
    chris_tens[0,2,0] = -2*(a**2)*r*sine*cosi/(rho2**2)
    chris_tens[0,1,3] = 2*a*(sine**2)*(((a*cosi)**2)*(a**2 - r**2) - (r**2)*(a**2 + 3*(r**2)))/(2*(rho2**2)*tri)
    chris_tens[0,3,1] = 2*a*(sine**2)*(((a*cosi)**2)*(a**2 - r**2) - (r**2)*(a**2 + 3*(r**2)))/(2*(rho2**2)*tri)
    chris_tens[0,2,3] = 2*r*cosi*((a*sine)**3)/(rho2**2)
    chris_tens[0,3,2] = 2*r*cosi*((a*sine)**3)/(rho2**2)
    chris_tens[1,0,0] = 2*tri*(r**2 - (a*cosi)**2)/(2*(rho2**3))
    chris_tens[1,0,3] = -tri*2*a*(sine**2)*(r**2 - (a*cosi)**2)/(2*(rho2**3))
    chris_tens[1,3,0] = -tri*2*a*(sine**2)*(r**2 - (a*cosi)**2)/(2*(rho2**3))
    chris_tens[1,1,1] = (2*r*((a*sine)**2) - 2*(r**2 - (a*cosi)**2))/(2*rho2*tri)
    chris_tens[1,1,2] = -(a**2)*sine*cosi/rho2
    chris_tens[1,2,1] = -(a**2)*sine*cosi/rho2
    chris_tens[1,2,2] = -r*tri/rho2
    chris_tens[1,3,3] = (tri*(sine**2)/(2*(rho2**3)))*(-2*r*(rho2**2) + 2*((a*sine)**2)*(r**2 - (a*cosi)**2))
    chris_tens[2,0,0] = -2*(a**2)*r*sine*cosi/(rho2**3)
    chris_tens[2,0,3] = 2*a*r*(r**2 + a**2)*sine*cosi/(rho2**3)
    chris_tens[2,3,0] = 2*a*r*(r**2 + a**2)*sine*cosi/(rho2**3)
    chris_tens[2,1,1] = (a**2)*sine*cosi/(rho2*tri)
    chris_tens[2,1,2] = r/rho2
    chris_tens[2,2,1] = r/rho2
    chris_tens[2,2,2] = -(a**2)*sine*cosi/rho2
    chris_tens[2,3,3] = -(sine*cosi/(rho2**3))*(bigA*rho2 + (r**2 + a**2)*2*r*((a*sine)**2))
    chris_tens[3,0,1] = 2*a*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2))
    chris_tens[3,1,0] = 2*a*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2))
    chris_tens[3,0,2] = -2*a*r*cota/(rho2**2)
    chris_tens[3,2,0] = -2*a*r*cota/(rho2**2)
    chris_tens[3,1,3] = (2*r*(rho2**2) + 2*(((a**2)*sine*cosi)**2 - (r**2)*(rho2 + r**2 + a**2)))/(2*tri*(rho2**2))
    chris_tens[3,3,1] = (2*r*(rho2**2) + 2*(((a**2)*sine*cosi)**2 - (r**2)*(rho2 + r**2 + a**2)))/(2*tri*(rho2**2))
    chris_tens[3,2,3] = (cota/(rho2**2))*((rho2**2) + 2*r*((a*sine)**2))
    chris_tens[3,3,2] = (cota/(rho2**2))*((rho2**2) + 2*r*((a*sine)**2))
    return metric, chris_tens

def check_interval(solution, state, *args):
    '''
    Returns the spacetime interval for a state vector given a particular spacetime solution. 

    Parameters
    ----------
    solution : function
        One of the solution functions mink, schwarz, or kerr
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    *args : int/float
        args required for different solutions, depends on specific function.

    Returns
    -------
    interval : float
        spacetime interval. Returns -1 for velocities, -m^2 for 4-momenta
    '''
    metric, chris = solution(state, *args)
    interval = np.einsum("ij,i,j -> ", metric, state[4:], state[4:])
    return interval

def check_interval_w_metric(metric, state, *args):
    '''
    Returns the spacetime interval for a state vector given a particular spacetime solution. 

    Parameters
    ----------
    solution : function
        One of the solution functions mink, schwarz, or kerr
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    *args : int/float
        args required for different solutions, depends on specific function.

    Returns
    -------
    interval : float
        spacetime interval. Returns -1 for velocities, -m^2 for 4-momenta
    '''
    interval = np.einsum("ij,i,j -> ", metric, state[4:], state[4:])
    return interval

@njit
def check_interval_vec(states, a):
    '''
    Returns the spacetime interval for a state vector given a particular spacetime solution. 

    Parameters
    ----------
    solution : function
        One of the solution functions mink, schwarz, or kerr
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    *args : int/float
        args required for different solutions, depends on specific function.

    Returns
    -------
    interval : float
        spacetime interval. Returns -1 for velocities, -m^2 for 4-momenta
    '''
    interval = np.zeros(len(states))
    for i in range(len(states)):
        metric, chris = kerr_2(states[i], a)
        for mu in range(4):
            for nu in range(4):
                interval[i] += metric[mu, nu]*states[i][4 + mu]*states[i][4 + nu]
    return interval

def set_u_kerr(a, cons=False, velorient=False, vel4=False, params=False, pos=False):
    '''
    Creates and normalizes an input state vector according to kerr metric given a variety of inputs

    Parameters
    ----------
    a : float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    cons : 3-element list/array of floats
        energy, angular momentum, and carter constant per unit mass
    velorient : 3-element list/array of floats
        ratio of velocity/speed of light (beta), angle between r-hat and trajectory (eta - radians), angle between phi hat and trajectory (xi - radians)
    vel4 : 4-element list/array of floats
        tetrad component velocities [t, r, theta, phi]
    params : 3-element list/array of floats
        minimum of effective radial potential (GU distance), eccentricity, inclination (pi/2 as equatorial, negative values correspond to retrograde motion)
    pos : 4-element list/array of floats
        initial 4-position of particle
    Returns
    -------
    new : 8 element numpy array
        4-position and 4-velocity of the test particle at a particular moment
    cons : 3-element list/array of floats
        energy, angular momentum, and carter constant per unit mass
    '''
    if np.shape(cons) == (3,):
        cons = np.real(cons)
        #print("Calculating initial velocity from constants E,L,C")
        if np.shape(pos) == (4,):
            pos = np.real(pos)
            new = recalc_state3(cons, pos, a)
            if type(new) == bool:
                rho2 = pos[1]**2 + a**2*np.cos(pos[2])**2
                tri  = pos[1]**2 - 2*pos[1] + a**2
                al2 = (rho2*tri / (rho2*tri + 2*pos[1]*(a*a + pos[1]*pos[1])))
                w = (2*pos[1]*a / (rho2*tri + 2*pos[1]*(a*a + pos[1]*pos[1])))
                new = np.array([*pos[:4], 1/np.sqrt(al2), 0.0, 0.0, w/np.sqrt(al2)])  
        else:
            E, L, C = cons
            R = np.array([ E**2 - 1, 2, (a**2)*(E**2 - 1) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
            turns = np.roots(R)
            turns = np.sort(turns.real[abs(turns.imag)<1e-14])
            flats = np.sort(np.real(np.roots(np.polyder(R))))
            pos = np.array([0.0, flats[-1], np.pi/2, 0.0])
            new = recalc_state3(cons, pos, a)
    elif (np.shape(velorient) == (3,)) and np.shape(pos) == (4,):
        velorient, pos = np.real(velorient), np.real(pos)
        #print("Calculating intial velocity from tetrad velocity and orientation")
        beta, eta, xi = velorient
        #eta is radial angle - 0 degrees is radially outwards, 90 degrees is in the phi direction
        #xi is up/down - 0 degrees is in the theta , 90 degrees is no up/down component
        eta, xi = eta*np.pi/180, xi*np.pi/180
        if (beta > 1):
            #print("Tetrad velocity exceeds c. Normalizing to 0.05")
            beta = 0.05
        gamma = 1/np.sqrt(1 - beta**2)
        r, theta = pos[1], pos[2]
            
        #various defined values that make math easier
        rho2 = r**2 + (a**2)*(np.cos(theta)**2)
        tri = r**2 - 2*r + a**2
        al2 = (rho2*tri)/(rho2*tri + 2*r*((a**2) + (r**2)))
        w = (2*r*a)/(rho2*tri + 2*r*((a**2) + (r**2)))
        wu2 = ((rho2*tri + 2*r*((a**2) + (r**2)))/rho2)*(np.sin(theta)**2)
        tetrad_matrix = np.array([[1/(np.sqrt(al2)), 0,                 0,               0],
                                  [0,                np.sqrt(tri/rho2), 0,               0],
                                  [0,                0,                 1/np.sqrt(rho2), 0],
                                  [w/np.sqrt(al2),   0,                 0,               1/np.sqrt(wu2)]])
        
        tilde = np.array([gamma, gamma*beta*np.cos(eta), -gamma*beta*np.sin(eta)*np.cos(xi), gamma*beta*np.sin(eta)*np.sin(xi)])
        new = np.matmul(tetrad_matrix, tilde)
        new = np.array([*pos, *new])
    elif (np.shape(vel4) == (4,)) and np.shape(pos) == (4,):
        vel4, pos = np.real(vel4), np.real(pos)
        #print("Calculating initial velocity from tetrad component velocities")
        r, theta = pos[1], pos[2]
        metric, chris = kerr(pos, a)
        #various defined values that make math easier
        rho2 = r**2 + (a**2)*(np.cos(theta)**2)
        tri = r**2 - 2*r + a**2
        al2 = (rho2*tri)/(rho2*tri + 2*r*((a**2) + (r**2)))
        w = (2*r*a)/(rho2*tri + 2*r*((a**2) + (r**2)))
        wu2 = ((rho2*tri + 2*r*((a**2) + (r**2)))/rho2)*(np.sin(theta)**2)
        tetrad_matrix = np.array([[1/(np.sqrt(al2)), 0,                 0,               0],
                                  [0,                np.sqrt(tri/rho2), 0,               0],
                                  [0,                0,                 1/np.sqrt(rho2), 0],
                                  [w/np.sqrt(al2),   0,                 0,               1/np.sqrt(wu2)]])
        rdot, thetadot, phidot = vel4[1]/vel4[0], vel4[2]/vel4[0], vel4[3]/vel4[0]
        vel_2 = (rdot**2 + thetadot**2 + phidot**2)
        beta = np.sqrt(vel_2)
        #print("uhhh", beta, vel_2)
        if beta >= 1.0:
            #print("Tetrad velocity exceeds or equals c, Normalizing to 0.05")
            rdot, thetadot, phidot = np.array([rdot, thetadot, phidot])*(0.05/beta)
            vel_2 = (rdot**2 + thetadot**2 + phidot**2)
            beta = np.sqrt(vel_2)
            #print(vel2, beta, "wonk")
        #print("yo?", vel_2)
        gamma = 1/np.sqrt(1 - vel_2)
        #eta = np.arccos(np.sqrt((r * np.sin(theta) * phidot)**2)/beta)
        #xi = np.arccos(np.sqrt(rdot**2)/(beta*np.sin(eta)))
        #tilde = np.array([gamma, gamma*beta*np.cos(eta), -gamma*beta*np.sin(eta)*np.cos(xi), gamma*beta*np.sin(eta)*np.sin(xi)])
        tilde = np.array([gamma, gamma*rdot, gamma*thetadot, gamma*phidot])
        new = np.matmul(tetrad_matrix, tilde)
        new = np.array([*pos, *new])
    elif np.shape(params) == (3,):
        params = np.real(params)
        #print("Calculating initial velocity from orbital parameters r0, e, i (WIP)")
        cons = schmidtparam3(*params, a)
        if cons == False:
            print("Non-viable parameters, plung")
            return set_u_kerr(a, params=[find_rms(a), 0.2, np.pi/2])
        if np.shape(pos) != (4,):
            E, L, C = cons
            R = np.array([ E**2 - 1, 2, (a**2)*(E**2 - 1) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
            flats = np.real(np.roots(np.polyder(R)))
            pos = np.array([0.0, max(flats), np.pi/2, 0.0])
        new = recalc_state3(cons, pos, a)
        tick = 0
        while type(new) == bool:
            tick += 1
            cons[0] *= 1 + 1e-15
            new = recalc_state3(cons, pos, a)

    else:
        print("Insufficent information provided, have a plunge")
        return set_u_kerr(a, params=[find_rms(a), 0.2, np.pi/2])

    if np.shape(cons) != (3,):
        metric, chris = kerr(new, a)
        energy = -np.matmul(new[4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
        lz = np.matmul(new[4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
        qarter = np.matmul(np.matmul(kill_tensor(new, a), new[4:]), new[4:])    #initial Carter constant Q
        cart = qarter - (a*energy - lz)**2                                          #initial adjusted Carter constant 
        cons = [energy, lz, cart]
    return new, cons

def schmidtparam3(r0, e, i, a):
    '''
    Returns characteristic constants of an orbit given observable parameters + spin
    Cannot generate unbound orbits (e >= 1.0)

    Parameters
    ----------
    r0 : int/float
        semi-major axis of the approximate Keplerian orbit
    e : int/float
        eccentricity of approximate Keplerian orbit, between 0 and 1
    i : int/float
        inclination of orbit w.r.t. angular momentum of black hole, where pi/2 is a prograde equatorial orbit, 0 is a polar orbit, and -pi/2 is a retrograde equatorial orbit
        parameter space extends in both directions s.t. any i + 2*n*pi is equivalent for any integer n
    a : int/float
        dimensionlesss spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    cons : 3-element list/array of floats
        energy, angular momentum, and carter constant per unit mass
    '''
    #sympy doesn't like tiny e?? idk what the issue is
    e = 0.0 if e < 1e-15 else e
    p = r0*(1 - (e**2))  #p is semi-latus rectum 
    rp, ra = p/(1 + e), p/(1 - e)
    polar = False
    i = i%(2*np.pi)
    if i > np.pi:
        i -= 2*np.pi
    j = i
    if i == 0.0 or i == np.pi:
        vals = np.transpose([[inc, *schmidtparam3(r0, e, inc, a)] for inc in np.linspace(np.pi/2, i, endpoint=False)])
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
        h = r*(r - 2) + tri*(z**2)/(max(1e-15, 1 - z**2)) #include a bias for divide by zero error
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
        return (z**2)*((a**2)*(1 - E**2) + (L**2)/(max(1e-15, 1 - z**2)))
    
    x, y = sp.symbols("x y", real=True)
    eq1 = sp.Eq(f1*(x**2) - 2*g1*x*y - h1*(y**2), d1)
    eq2 = sp.Eq(f2*(x**2) - 2*g2*x*y - h2*(y**2), d2)
    symsols = sp.solve([eq1, eq2])

    full_sols = []
    E, L, C = [np.sqrt(1-((1-e**2)/p)), (1 - z**2)*p, p*(z**2)]
    for thing in symsols:
        ene, lel = np.array([thing[x], thing[y]]).astype(float)
        if ene > 0.0 and ene < 1.0: 
            full_sols.append([ene, lel, newC(ene, lel, a, z)])
            E, L, C = [ene, lel, newC(ene, lel, a, z)]
            if np.prod(np.sign([E, L, C])) == np.sign(np.sin(j)):
                break
            else:
                E, L, C = (1 - (1 - e**2)/p)**0.5, ((1 - z**2)*p)**(0.5), p*(z**2)
                
    coeff, ro, count = [-1], 1, 0
    while np.polyval(coeff, ro) < -1e-12 and count < 20:
        count += 1
        E += 10**(-16)
        coeff = np.array([ E**2 - 1, 2, (a**2)*(E**2 - 1) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
        coeff2 = np.polyder(coeff)
        turns = np.roots(coeff)
        flats = np.roots(coeff2)
        turns = turns.real[abs(turns.imag)<(1e-6)*r0]
        flats = flats.real[abs(flats.imag)<(1e-6)*r0]
        try:
            ro = max(flats)
        except:
            ro = np.max(np.roots(coeff2)).real

    turns = np.sort(turns)
    r02, e2 = (turns[-1] + turns[-2])/2.0, (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
    r_err, e_err = np.abs((r0 - r02)/r0)*100, np.abs((e2 - e)/(2-e))*100
    
    if r_err < 1e-6 and e_err < 1e-6:
        if (np.sqrt(L**2 + C) - np.abs(L))/np.abs(L) < 1e-15:
            C = 0.0
        return [E, L, C]
    else:
        E = False
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
            if ene > 0.0 and ene < 1.0: 
                full_sols.append([ene, lel, newC(ene, lel, a, z)])
        
        for solution in full_sols:
            if (np.prod(np.sign(solution)) == np.sign(np.sin(j))):
                E, L, C = solution
                break     
        if polar == True:
            L, C = 0.0, C/(z**2) - (a**2)*(1 - E**2)
            
        if (np.sqrt(L**2 + C) - np.abs(L))/np.abs(L) < 1e-15:
            C = 0.0

        if E:
            return [E, L, C]
        else:
            return False

@njit
def getCons(state, a):
    metric, _ = kerr_2(state, a)  # assuming this returns (metric, chris), and we only need metric
    u = state[4:]

    # Replace np.matmul(metric, u) with explicit dot product
    stuff = np.zeros(4)
    for i in range(4):
        for j in range(4):
            stuff[i] += metric[i, j] * u[j]

    ene = -stuff[0]
    Lz = stuff[3]

    kill = kill_tensor_njit(state, a)
    temp = 0.0
    for i in range(4):
        for j in range(4):
            temp += kill[i, j] * u[i] * u[j]

    cart = temp - (a * ene - Lz)**2
    return np.array([ene, Lz, cart])

def kill_tensor(state, a):
    '''
    kill_tensor function calculates killing Kerr killing tensor for a given system

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    ktens : 4 element numpy array
        Describes symmetry in spacetime associated with the Carter Constant

    '''
    r, theta = state[1], state[2]
    metric, chris = kerr(state, a)
    rho2, tri = r**2 + (a*np.cos(theta))**2, r**2 - 2*r + a**2
    l_up = np.array([(r**2 + a**2)/tri, 1.0, 0.0, a/tri])
    n_up = np.array([(r**2 + a**2)/(2*rho2), -tri/(2*rho2), 0.0, a/(2*rho2)])
    l = np.matmul(metric, l_up)
    n = np.matmul(metric, n_up)
    l_n = np.outer(l, n)
    ktens = 2 * rho2 * 0.5*(l_n + np.transpose(l_n)) + (r**2) * np.array(metric)
    return ktens

@njit
def kill_tensor_njit(state, a):
    r, theta = state[1], state[2]
    metric, chris = kerr_2(state, a)
    rho2, tri = r**2 + (a*np.cos(theta))**2, r**2 - 2*r + a**2
    l_up = np.array([(r**2 + a**2)/tri, 1.0, 0.0, a/tri])
    n_up = np.array([(r**2 + a**2)/(2*rho2), -tri/(2*rho2), 0.0, a/(2*rho2)])

    # Lower indices: l = g @ l_up, n = g @ n_up
    l = np.zeros(4)
    n = np.zeros(4)
    for mu in range(4):
        for nu in range(4):
            l[mu] += metric[mu, nu] * l_up[nu]
            n[mu] += metric[mu, nu] * n_up[nu]

    # Symmetric outer product: l_n + l_n^T
    ktens = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            l_n_sym = 0.5 * (l[mu] * n[nu] + l[nu] * n[mu])
            ktens[mu, nu] = 2.0 * rho2 * l_n_sym + r**2 * metric[mu, nu]

    return ktens

def gr_diff_eq(solution, state, *args):
    '''
    gr_diff_eq function calculates the instantaneous proper time derivative for
    a given state in a given system

    Parameters
    ----------
    solution : function
        One of the GR solution functions (mink, schwarz, or kerr)
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    *args : int/float
        args required for different solutions, depends on specific function.
        generally mass and possibly spin.

    Returns
    -------
    d_state: 8 element numpy array
        4-velocity and 4-acceleration for the test particle at a particular moment

    '''

    d_state = np.zeros((8), dtype=float)                                         #create empty array to be the derivative of the state
    d_state[0:4] = state[4:]                                                      #derivative of position is velocity
    metric, chris = solution(state, *args)
    d_state[4:] = -np.einsum("ijk, j, k -> i", chris, state[4:], state[4:])
    return d_state                                                                #return derivative of state

@njit
def gr_diff_eq2(solution, state, a):
    '''
    gr_diff_eq function calculates the instantaneous proper time derivative for
    a given state in a given system

    Parameters
    ----------
    solution : function
        One of the GR solution functions (mink, schwarz, or kerr)
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    *args : int/float
        args required for different solutions, depends on specific function.
        generally mass and possibly spin.

    Returns
    -------
    d_state: 8 element numpy array
        4-velocity and 4-acceleration for the test particle at a particular moment

    '''

    d_state = np.zeros((8), dtype=float)                                         #create empty array to be the derivative of the state
    d_state[0:4] = state[4:]                                                      #derivative of position is velocity
    metric, chris = solution(state, a)
    acc = np.zeros(4)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                acc[i] += chris[i, j, k] * state[4 + j] * state[4 + k]
    d_state[4:] = -acc
    return d_state                                                                #return derivative of state

#Butcher table for standard RK4 method
rk4 = {"label": "Standard RK4",
       "nodes": [1/2, 1/2, 1],
       "weights": [1/6, 1/3, 1/3, 1/6],
       "coeff": [[1/2], 
                 [0, 1/2],
                 [0, 0, 1]]}   
rk4_2 = [np.array([1/2, 1/2, 1]),  #nodes
       np.array([1/6, 1/3, 1/3, 1/6]), #weights
       np.array([[1/2,   0, 0],  #coefficients
                 [  0, 1/2, 0],
                 [  0,   0, 1]])]                                                

#Butcher table for 5th order Cash-Karp method
ck5 = {"label": "Cash-Karp 5th Order",
       "nodes": [1/5, 3/10, 3/5, 1, 7/8],
       "weights": [37/378, 0, 250/621, 125/594, 0, 512/1771],
       "coeff": [[1/5],
                 [3/40, 9/40], 
                 [3/10, -9/10, 6/5], 
                 [-11/54, 5/2, -70/27, 35/27],
                 [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]]} 
ck5_2 = [np.array([1/5, 3/10, 3/5, 1, 7/8]),  #nodes
       np.array([37/378, 0, 250/621, 125/594, 0, 512/1771]), #weights
       np.array([[       1/5,       0,         0,            0,        0], #coefficients
                 [      3/40,    9/40,         0,            0,        0], 
                 [      3/10,   -9/10,       6/5,            0,        0], 
                 [    -11/54,     5/2,    -70/27,        35/27,        0],
                 [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]])]      

#Butcher table for 4th order Cash-Karp method    
ck4 = {"label": "Cash-Karp 4th Order",
       "nodes": [1/5, 3/10, 3/5, 1, 7/8],
       "weights": [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4],
       "coeff": [[1/5],
                 [3/40, 9/40], 
                 [3/10, -9/10, 6/5], 
                 [-11/54, 5/2, -70/27, 35/27],
                 [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]]}   
ck4_2 = [np.array([1/5, 3/10, 3/5, 1, 7/8]),  #nodes
       np.array([2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4]), #weights
       np.array([[       1/5,       0,         0,            0,        0], #coefficients
                 [      3/40,    9/40,         0,            0,        0], 
                 [      3/10,   -9/10,       6/5,            0,        0], 
                 [    -11/54,     5/2,    -70/27,        35/27,        0],
                 [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]])]    

#Butcher table for 4th order Runge-Kutta-Fehlberg method
rkf4_2 = [np.array([1/4, 3/8, 12/13, 1, 1/2]),  #nodes
       np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]), #weights
       np.array([[       1/4,          0,          0,            0,     0], #coefficients
                 [      3/32,       9/32,          0,            0,     0], 
                 [ 1932/2197, -7200/2197,  7296/2197,            0,     0], 
                 [   439/216,         -8,   3680/513,    -845/4104,     0],
                 [     -8/27,          2, -3544/2565,    1859/4104, 11/40]])]    

#Butcher table for 5th order Runge-Kutta-Fehlberg method
rkf5_2 = [np.array([1/4, 3/8, 12/13, 1, 1/2]),  #nodes
       np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0]), #weights
       np.array([[       1/4,          0,          0,            0,     0], #coefficients
                 [      3/32,       9/32,          0,            0,     0], 
                 [ 1932/2197, -7200/2197,  7296/2197,            0,     0], 
                 [   439/216,         -8,   3680/513,    -845/4104,     0],
                 [     -8/27,          2, -3544/2565,    1859/4104, 11/40]])]  

def gen_RK(butcher, solution, state, dTau, *args):
    '''
    gen_RK function applies a given Runge-Kutta method to calculate whatever the new state
    of an orbit will be after some given amount of proper time

    Parameters
    ----------
    butcher : dictionary
        Butcher table information for a given Runge-Kutta method. 
    solution : function
        One of the solution functions mink, schwarz, or kerr
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    dTau : float
        proper time between current state and state-to-be-calculated
    *args : int/float
        args required for different solutions, depends on specific function.
        generally mass and possibly spin.

    Returns
    -------
    new_state : 8 element numpy array
        4-position and 4-velocity of the test particle at the next moment
    '''
    k = [gr_diff_eq(solution, state, *args)]                                      #start with k1, based on initial conditions
    for i in range(len(butcher["nodes"])):                                        #iterate through each non-zero node as defined by butcher table
        param = np.copy(state)                                                      #start with the basic state, then
        for j in range(len(butcher["coeff"][i])):                                   #interate through each coeffiecient
            param += np.array(butcher["coeff"][i][j] * dTau * k[j])                   #in order to obtain the approximated state based on previously defined k values
        k.append(gr_diff_eq(solution, param, *args))                          #which is then used to find the next k value
    new_state = np.copy(state)
    for val in range(len(k)):                                                     #another for loop to add all the weights and find the final state
        new_state += k[val] * butcher["weights"][val] * dTau                        #can probably be simplified but this works for now
    return new_state

@njit
def gen_RK2(nodes, weights, coeff, solution, state, dTau, a):   #passing in a instead of *args because numba doesn't like it? figure something else out later
    '''
    gen_RK function applies a given Runge-Kutta method to calculate whatever the new state
    of an orbit will be after some given amount of proper time

    Parameters
    ----------
    butcher : dictionary
        Butcher table information for a given Runge-Kutta method. 
    solution : function
        One of the solution functions mink, schwarz, or kerr
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    dTau : float
        proper time between current state and state-to-be-calculated
    *args : int/float
        args required for different solutions, depends on specific function.
        generally mass and possibly spin.

    Returns
    -------
    new_state : 8 element numpy array
        4-position and 4-velocity of the test particle at the next moment
    '''
    #nodes = butcher[0]
    #weights = butcher[1]
    #coeff = butcher[2]
    node_size = nodes.shape[0]
    coeff_size = coeff.shape[0]

    k = np.empty((node_size + 1, 8))
    k[0] = gr_diff_eq2(solution, state, a)                                     #start with k1, based on initial conditions
    for i in range(node_size):                                        #iterate through each non-zero node as defined by butcher table
        param = np.copy(state)                                                      #start with the basic state, then
        for j in range(i + 1): 
            coeff_ij = coeff[i, j]
            if coeff_ij != 0.0:                                   #interate through each coeffiecient
                param += coeff_ij * dTau * k[j]                  #in order to obtain the approximated state based on previously defined k values
        k[i+1] = gr_diff_eq2(solution, param, a)                         #which is then used to find the next k value
    new_state = np.copy(state)
    for val in range(len(k)):                                                     #another for loop to add all the weights and find the final state
        new_state += k[val] * weights[val] * dTau                        #can probably be simplified but this works for now
    return new_state

def recalc_state3(constants, state, a, g=None, tol=1e-12):
    '''
    Calculates a new state vector based on a given position and constants of motion
    (numerically stabilized version)

    Parameters
    ----------
    constants : [E, L, C]
    state : 4- or 8-element array
    a : spin (0 ≤ a ≤ 1)
    tol : tolerance for numerical errors

    Returns
    -------
    new_state : 8-element numpy array
    '''
    state = np.real(state)
    energy, lmom, cart = constants
    rad, theta = state[1], state[2]

    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    # Metric functions
    sig = rad**2 + (a**2)*(cos_th**2)
    tri = rad**2 - 2*rad + a**2

    # ===== Radial potential R(r) =====
    p_r = np.array([energy, 0, a*(a*energy - lmom)])
    r_r = np.array([
        energy**2 - 1,
        2,
        (a**2)*(energy**2 - 1) - lmom**2 - cart,
        2*((a*energy - lmom)**2 + cart),
        -cart*(a**2)
    ])

    R = np.polyval(r_r, rad)
    if R < -tol:
        #print(f"Radial potential negative: {R}")
        #print(f"turns = {np.roots(r_r)}")
        #print(f"rad {rad}")
        #print(f"cons {[energy, lmom, cart]}")
        #print(f"Locking to psuedo-spherical orbit.")
        #rs = np.linspace(0.999999*rad, 1.000001*rad)
        #plt.plot(rs, np.polyval(r_r, rs))
        #plt.axhline(0.0)
        #plt.show()
        return False
    R = max(R, 0.0)

    # ===== Polar potential Θ(θ) (stable form) =====
    if abs(sin_th) < 1e-14:
        print(f"Theta too close to pole; numerical instability: {theta:.4e}")

    Theta = cart - cos_th**2 * (a**2 * (1 - energy**2) + (lmom**2) / (sin_th**2))

    if Theta < -tol:
        eps = np.finfo(float).eps
        term = cos_th**2 * (a*a*(1-energy*energy) + lmom*lmom/(sin_th*sin_th))
        theta_err = eps * (abs(cart) + abs(term))
        #print(f"Theta potential negative: {Theta}, {theta_err}, {tol}")
        #print(f"Extending geodesic section.")
        return None
    Theta = max(Theta, 0.0)

    # ===== First integrals =====
    tlam = -a*(a*energy*(sin_th**2) - lmom) + ((rad**2 + a**2)/tri)*np.polyval(p_r, rad)
    rlam = np.sqrt(R)
    thelam = np.sqrt(Theta)
    philam = -(a*energy - (lmom/(sin_th**2))) + (a/tri)*np.polyval(p_r, rad)

    # Convert to proper time
    ttau = tlam / sig
    rtau = rlam / sig
    thetau = thelam / sig
    phitau = philam / sig

    # ===== Sign handling =====
    if len(state) != 8:
        # default: inward and downward
        rtau = -abs(rtau)
        thetau = -abs(thetau)

        new_state = np.zeros(8)
        new_state[:4] = state[:4]

    else:
        new_state = np.copy(state)

        # Radial direction
        if abs(state[5]) > tol:
            rtau = abs(rtau) * np.sign(state[5])
        else:
            # fallback: use slope of potential
            dR = np.polyval(np.polyder(r_r), rad)
            rtau = abs(rtau) * np.sign(dR)

        # Theta direction
        if abs(state[6]) > tol:
            thetau = abs(thetau) * np.sign(state[6])
        else:
            thetau = abs(thetau)

    new_state[4:] = np.array([ttau, rtau, thetau, phitau])

    # ===== Enforce normalization u^μ u_μ = -1 =====
    # You MUST have a metric function defined elsewhere
    u = new_state[4:]
    if g is None:
        try:
            g, _ = kerr_2(state, a)
        except:
            g, _ = kerr(state, a)

    norm = np.einsum('ij,i,j->', g, u, u)

    if norm >= 0:
        #print(f"Non-timelike 4-velocity detected: norm = {norm}")
        #print(f"Here vals! {energy}, {lmom}, {cart}, {a}")
        return None

    u *= np.sqrt(-1.0 / norm)
    new_state[4:] = u

    # ===== Optional consistency check (VERY useful for debugging) =====
    # Uncomment if needed
    """
    E_check = -np.dot(g[0], u)  # depends on your conventions
    L_check = np.dot(g[3], u)
    # Carter check depends on your implementation

    print("ΔE:", E_check - energy, "ΔL:", L_check - lmom)
    """

    return new_state

def interpolate2(data, time, supress=True, samp_freq=1):
    '''
    interpolates coordinate data to be evenly spaced in coordinate time
    Fail state plots time against index twice?

    Parameters
    ----------
    data : N x 3 numpy array of floats
        r, theta, and phi position of test particle
    time : N element numpy array of floats
        coordinate time of test particle
    supress : bool
        If True, limit the total number of samples.
    samp_freq : float
        sampling frequency

    Returns
    -------
    new_data : M x 3 numpy array of floats
        r, theta, and phi position of test particle, interpolated to be evenly spaced along new_time
    new_time : M element numpy array of floats
        coordinate time of test particle, interpolated to be evenly spaced
        M is maximum of the length of the original time array or the integerized number of time units that have passed
    '''
    # Use this one for gwaves
    data = np.array(data)
    if supress == True:
        # Reduce to 360 samples per azimuthal orbit OR 10000 samples, whichever is larger
        phi_per = 2 * np.pi * abs((time[-1] - time[0])/(data[-1, 2] - data[0, 2]))
        new_time = np.arange(time[0], time[-1], min(phi_per/360, (time[-1] - time[0])/10000))
    else:
        # Sample frequency defaults to 1 per geometric time unit, which is ~41mHz for a 5e6 solar mass body
        new_time = np.arange(time[0], time[-1], 1/samp_freq)
    

    j = np.searchsorted(time, new_time, side='right') - 1
    j = np.clip(j, 0, len(time)-2)
    w = (new_time - time[j]) / (time[j + 1] - time[j])

    new_data = np.stack((new_time, new_time, new_time), axis=-1)
    for i in range(3):
        new_data[:, i]  = data[j, i]  + w*(data[j+1, i] - data[j, i])

    return new_data, new_time

def interpolate(data, time, supress=True):
    '''
    interpolates coordinate data to be evenly spaced in coordinate time
    Fail state plots time against index twice?

    Parameters
    ----------
    data : N x 3 numpy array of floats
        r, theta, and phi position of test particle
    time : N element numpy array of floats
        coordinate time of test particle
    suppress : bool, defaults to True
        limits the size of the final array to 10000 entries or ~20 samples per phase (assuming circular orbit), whichever is greater

    Returns
    -------
    new_data : M x 3 numpy array of floats
        r, theta, and phi position of test particle, interpolated to be evenly spaced along new_time
    new_time : M element numpy array of floats
        coordinate time of test particle, interpolated to be evenly spaced
        M is maximum of the length of the original time array or the integerized number of time units that have passed
    '''
    data = np.array(data)
    if supress == True:
        try:
            rad = data[argrelmin(data[:,0])[0][0], 0]
        except:
            rad = data[0,0]
        new_time = np.arange(time[0], time[-1], min(2*np.pi*np.sqrt(rad**3)/20, (time[-1] - time[0])/10000))
    else:
        new_time = np.linspace(time[0], time[-1], max(len(time), 10*int(time[-1] - time[0])))
    try:
        r_poly = spi.CubicSpline(time, data[:,0])
        theta_poly = spi.CubicSpline(time, data[:,1])
        phi_poly = spi.CubicSpline(time, data[:,2])
        new_data = np.transpose(np.array([r_poly(new_time), theta_poly(new_time), phi_poly(new_time)]))
        return new_data, new_time
    except ValueError:
        fig, ax = plt.subplots(4, 1)
        ax[0].plot(time)
        ax[1].plot(data[:,0])
        ax[2].plot(data[:,1])
        ax[3].plot(data[:,2])
        print("yo" if -1 in np.sign(np.diff(time)) else "no negs")
        print("yop" if 0.0 in np.diff(time) else "no zos")
        r_poly = spi.CubicSpline(time, data[:,0])
        return False

def hav_angle(theta1, phi1, theta2, phi2):
    dtheta, dphi = theta2 - theta1, phi2 - phi1
    dist = 0.5*(1 - np.cos(dtheta)) + 0.5*np.sin(theta1)*np.sin(theta2)*(1 - np.cos(dphi))
    return np.arccos(1 - 2*dist)

def interpolate3(data, time):
    '''
    interpolates coordinate data to be evenly spaced in coordinate time
    Fail state plots time against index twice?
    automatically sets it to traversed angular distance in degrees (integerized) or length of original data, whichever is shorter

    Parameters
    ----------
    data : N x 3 numpy array of floats
        r, theta, and phi position of test particle
    time : N element numpy array of floats
        coordinate time of test particle


    Returns
    -------
    new_data : M x 3 numpy array of floats
        r, theta, and phi position of test particle, interpolated to be evenly spaced along new_time
    new_time : M element numpy array of floats
        coordinate time of test particle, interpolated to be evenly spaced
        M is maximum of the length of the original time array or the integerized number of time units that have passed
    '''
    data = np.array(data)
    num = int(np.rad2deg(np.sum(hav_angle(data[:-1,1], data[:-1,2], data[1:,1], data[1:,2]))))
    new_time = np.linspace(time[0], time[-1], min(len(data), num))
    try:
        r_poly = spi.CubicSpline(time, data[:,0])
        theta_poly = spi.CubicSpline(time, data[:,1])
        phi_poly = spi.CubicSpline(time, data[:,2])
        new_data = np.transpose(np.array([r_poly(new_time), theta_poly(new_time), phi_poly(new_time)]))
        return new_data, new_time
    except ValueError:
        fig, ax = plt.subplots(4, 1)
        ax[0].plot(np.diff(time))
        ax[1].plot(data[:,0])
        ax[2].plot(data[:,1])
        ax[3].plot(data[:,2])
        print("yo" if -1 in np.sign(np.diff(time)) else "no negs")
        print("yop" if 0.0 in np.diff(time) else "no zos")
        r_poly = spi.CubicSpline(time, data[:,0])
        return False

def interpolate3_linear(data, time):
    """
    Linearly interpolate r, theta, phi onto an evenly-spaced coordinate-time grid.
    """

    data = np.asarray(data)
    time = np.asarray(time)

    num = int(np.rad2deg(np.sum(
        hav_angle(data[:-1,1], data[:-1,2],
                  data[1:,1],  data[1:,2])
    )))

    new_time = np.linspace(time[0], time[-1], min(len(data), num))

    new_data = np.column_stack([
        np.interp(new_time, time, data[:,0]),
        np.interp(new_time, time, data[:,1]),
        np.interp(new_time, time, data[:,2]),
    ])

    return new_data, new_time

@njit
def linear_interp(x_vals, y_vals, new_xs):
    ix_low = np.searchsorted(x_vals, new_xs, side="right") - 1
    ix_high = np.searchsorted(x_vals, new_xs, side="right")
    x_low = x_vals[ix_low]
    x_high = x_vals[ix_high]
    y_low = y_vals[ix_low]
    y_high = y_vals[ix_high]
    return (y_low*(x_high - new_xs) + y_high*(new_xs - x_low))/(x_high - x_low)

@njit
def cubic_interp(x_vals, y_vals, new_xs):
    result = np.empty(len(new_xs))
    n = len(x_vals)

    for i in range(len(new_xs)):
        x = new_xs[i]
        ix = np.searchsorted(x_vals, x)  # right by default

        # Clamp to valid interior range
        if ix < 1:
            ix = 1
        elif ix > n - 3:
            ix = n - 3

        # Select 4 points: x[i-1], x[i], x[i+1], x[i+2]
        x0 = x_vals[ix - 1]
        x1 = x_vals[ix]
        x2 = x_vals[ix + 1]
        x3 = x_vals[ix + 2]

        y0 = y_vals[ix - 1]
        y1 = y_vals[ix]
        y2 = y_vals[ix + 1]
        y3 = y_vals[ix + 2]

        # Lagrange basis interpolation
        denom0 = (x0 - x1)*(x0 - x2)*(x0 - x3)
        denom1 = (x1 - x0)*(x1 - x2)*(x1 - x3)
        denom2 = (x2 - x0)*(x2 - x1)*(x2 - x3)
        denom3 = (x3 - x0)*(x3 - x1)*(x3 - x2)

        L0 = ((x - x1)*(x - x2)*(x - x3)) / denom0
        L1 = ((x - x0)*(x - x2)*(x - x3)) / denom1
        L2 = ((x - x0)*(x - x1)*(x - x3)) / denom2
        L3 = ((x - x0)*(x - x1)*(x - x2)) / denom3

        result[i] = y0*L0 + y1*L1 + y2*L2 + y3*L3

    return result

@njit
def interpolate_njit(data, time, supress=True):
    if supress == True:
        rad = np.mean(data[:,0])
        dt = min(2 * np.pi * np.sqrt(rad**3) / 20, (time[-1] - time[0]) / 10000)
        new_time = np.arange(time[0], time[-1], dt)
    else:
        new_time = np.linspace(time[0], time[-1], max(len(time), 10*int(time[-1] - time[0])))
    r_interp = cubic_interp(time, data[:,0], new_time)
    theta_interp = cubic_interp(time, data[:,1], new_time)
    phi_interp = cubic_interp(time, data[:,2], new_time)
    new_data = np.stack((r_interp, theta_interp, phi_interp), axis=1)
    return new_data, new_time

@njit
def interpolate_njit2(data, time, real_len, supress=True):
    time = time[:real_len]
    data = data[:real_len]
    if supress == True:
        rad = np.mean(data[:,0])
        dt = min(2 * np.pi * np.sqrt(rad**3) / 20, (time[-1] - time[0]) / 10000)
        new_time = np.arange(time[0], time[-1], dt)
    else:
        new_time = np.linspace(time[0], time[-1], max(len(time), 10*int(time[-1] - time[0])))
    r_interp = cubic_interp(time, data[:,0], new_time)
    theta_interp = cubic_interp(time, data[:,1], new_time)
    phi_interp = cubic_interp(time, data[:,2], new_time)
    new_data = np.stack((r_interp, theta_interp, phi_interp), axis=1)
    return new_data, new_time

@njit
def traceless_quad(quad):
    N = quad.shape[0]
    result = np.empty((N, 3, 3))
    for i in range(N):
        trace = quad[i, 0, 0] + quad[i, 1, 1] + quad[i, 2, 2]
        for j in range(3):
            for k in range(3):
                result[i, j, k] = quad[i, j, k] - (1/3) * trace * (1 if j == k else 0)
    
    return result

def sphr2quad(pos, a):
    '''
    Calculates quadrupole moment of a test particle

    Parameters
    ----------
    pos : 3-element numpy array of floats
        r, theta, and phi position of test particle

    Returns
    -------
    qmom : 3 x 3 numpy array of floats
        quadrupole moment of test particle
    '''
    rho = np.sqrt(pos[0]**2 + a**2)
    x = rho * np.sin(pos[1]) * np.cos(pos[2])
    y = rho * np.sin(pos[1]) * np.sin(pos[2])
    z = pos[0] * np.cos(pos[1])
    qmom = np.array([[2*x*x - (y**2 + z**2), 3*x*y,                 3*x*z],
                     [3*y*x,                 2*y*y - (x**2 + z**2), 3*y*z],
                     [3*z*x,                 3*z*y,                 2*z*z - (x**2 + y**2)]], dtype=np.float64)
    return qmom

def sphr2quad_vec(pos, a):
    '''
    Calculates quadrupole moment of a test particle

    Parameters
    ----------
    pos : N x 3-element numpy array of floats
        r, theta, and phi position of test particle

    Returns
    -------
    qmom : N x 3 x 3 numpy array of floats
        quadrupole moment of test particle
    '''
    rho = np.sqrt(pos[:, 0]**2 + a**2)
    carts = np.zeros((len(rho), 3))
    carts[:, 0] = rho * np.sin(pos[:, 1]) * np.cos(pos[:, 2])
    carts[:, 1] = rho * np.sin(pos[:, 1]) * np.sin(pos[:, 2])
    carts[:, 2] = pos[:, 0] * np.cos(pos[:, 1])
    r2 = np.sum(carts * carts, axis=1)
    qmom = np.zeros((len(rho), 3, 3))
    qmom = carts[:, :, None] * carts[:, None, :]
    qmom -= np.eye(3)[None,:,:] * (r2[:,None,None]/3)
    return qmom

def matrix_derive(data, time, degree):
    '''
    Calculates degree-th time derivative of a series of 3x3 matrices, assuming they are interpolated across time

    Parameters
    ----------
    data : N x 3 x 3 numpy array of floats
        x, y, z quadrupole moment of test particle, assumed interpolated
    time : N element numpy array of floats
        coordinate time of test particle, assumed interpolated
    degree : int
        desired degree of the resulting derivative

    Returns
    -------
    new_data : N x 3 x 3 numpy array of floats
        degree-th derivative of quadrupole moment
    '''
    new_data = np.zeros((len(time), 3, 3))
    
    for i in range(3):
        for j in range(3):
            spline = spi.CubicSpline(time, data[:,i,j])
            new_data[:, i, j] = spline(time, degree)
    return new_data

def matrix_derive2(data, old_time, time, degree):
    '''
    Calculates degree-th time derivative of a series of 3x3 matrices, assuming they are interpolated across time

    Parameters
    ----------
    data : N x 3 x 3 numpy array of floats
        x, y, z quadrupole moment of test particle, assumed interpolated
    time : N element numpy array of floats
        coordinate time of test particle, assumed interpolated
    degree : int
        desired degree of the resulting derivative

    Returns
    -------
    new_data : N x 3 x 3 numpy array of floats
        degree-th derivative of quadrupole moment
    '''
    polys = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            polys[i][j] = spi.CubicSpline(old_time, data[:,i,j])
    new_data = np.transpose(np.array([[polys[0][0](time, degree), polys[0][1](time, degree), polys[0][2](time, degree)],
                                      [polys[1][0](time, degree), polys[1][1](time, degree), polys[1][2](time, degree)],
                                      [polys[2][0](time, degree), polys[2][1](time, degree), polys[2][2](time, degree)]]))
    return new_data

def matrix_derive3(data, old_time, time):
    '''
    fhdjknslnfj
    Parameters
    ----------
    data : N x 3 x 3 numpy array of floats
        x, y, z quadrupole moment of test particle, assumed interpolated
    time : N element numpy array of floats
        coordinate time of test particle, assumed interpolated
    degree : int
        desired degree of the resulting derivative

    Returns
    -------
    new_data : N x 3 x 3 numpy array of floats
        degree-th derivative of quadrupole moment
    '''
    polysd2 = [[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]
    polysd3 = [[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]
    dt = np.mean(np.diff(time))
    for i in range(3):
        for j in range(3):
            u = spi.CubicSpline(old_time, data[:,i,j])
            polysd2[i][j] = (-u(time + 2*dt) + 16*u(time + dt) - 30*u(time) + 16*u(time - dt) - u(time - 2*dt))/(12*dt*dt)
            polysd3[i][j] = (-u(time + 3*dt) + 8*u(time + 2*dt) - 13*u(time + dt) + 13*u(time - dt) - 8*u(time - 2*dt) + u(time - 3*dt))/(8*dt*dt*dt)
    polysd2 = np.transpose(polysd2)
    polysd3 = np.transpose(polysd3)
    return polysd2, polysd3

@njit
def matrix_derive3_numba(data, dt):
    N = data.shape[0]
    d2 = np.zeros_like(data)
    d3 = np.zeros_like(data)

    for i in range(3):
        for j in range(3):
            f = data[:, i, j]
            for k in range(3, N-3):  # Avoid boundaries for 7-point stencil
                d2[k, i, j] = (-f[k+2] + 16*f[k+1] - 30*f[k] + 16*f[k-1] - f[k-2]) / (12 * dt**2)
                d3[k, i, j] = (-f[k+3] + 8*f[k+2] - 13*f[k+1] + 13*f[k-1] - 8*f[k-2] + f[k-3]) / (8 * dt**3)
    
    return d2, d3

def gwaves(quad_moment, time, distance, n_hat=None):
    """
    Calculate the plus and cross gravitational-wave polarizations
    from the second time derivative of the quadrupole moment.

    Parameters
    ----------
    quad_moment : ndarray, shape (N, 3, 3)
        Mass quadrupole moment as a function of time.
    time : ndarray, shape (N,)
        Coordinate time corresponding to quad_moment.
    distance : float
        Distance to the source in geometric units.
    n_hat : array-like, shape (3,), optional
        Unit vector pointing from the source toward the observer.
        If omitted, the observer is placed along the +x direction.

    Returns
    -------
    h_plus : ndarray, shape (N,)
        Plus polarization of the gravitational wave.
    h_cross : ndarray, shape (N,)
        Cross polarization of the gravitational wave.
    """

    # Second time derivative of the quadrupole moment.
    der_2 = matrix_derive(quad_moment, time, 2)

    # Define the observer's line-of-sight direction.
    # n_hat points from the source toward the observer; default is in the positive z-direction.
    if n_hat is None:
        n_hat = [0, 0, 1]

    # Normalize n_hat so that it is a unit vector.
    n_hat = np.array(n_hat) / np.linalg.norm(n_hat)

    # Construct two orthonormal basis vectors spanning the plane perpendicular to the observer's line of sight.
    if np.allclose(n_hat, [0, 0, 1]):
        ref = np.array([0, 1, 0])
    else:
        ref = np.array([0, 0, 1])
    e_x = ref - np.dot(ref, n_hat) * n_hat
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(n_hat, e_x)

    # Projection operator onto the plane perpendicular to the observer's line of sight;
    # This removes the component of a tensor parallel to the direction of propagation.
    P = np.eye(3) - np.outer(n_hat, n_hat)

    # Allocate an array for the transverse-traceless part of the
    # second time derivative of the quadrupole moment.
    h_TT = np.zeros_like(der_2)
    for k in range(len(der_2)):
        # Project the quadrupole derivative into the plane of the sky.
        term = P @ der_2[k] @ P
        # Calculate the trace of the projected tensor.
        trace = np.trace(P @ der_2[k])
        # Remove the transverse trace to obtain the transverse-traceless (TT) component.
        h_TT[k] = term - 0.5 * P * trace

    # Apply the leading-order quadrupole scaling for the strain.
    # The factor of 2/distance comes from the standard far-field quadrupole expression in geometric units.
    h_TT *= 2 / distance

    # Project the TT tensor onto the two polarization basis tensors.
    h_plus = 0.5 * (np.einsum("i, j, kij -> k", e_x, e_x, h_TT) - np.einsum("i, j, kij -> k", e_y, e_y, h_TT))
    h_cross = 0.5 * (np.einsum("i, j, kij -> k", e_x, e_y, h_TT) + np.einsum("i, j, kij -> k", e_y, e_x, h_TT))

    return h_plus, h_cross

def full_transform(data, distance, supress=True, samp_freq=1, m_bh=False, n_hat = None):    #defunctish??
    '''
    Calculates gravitational wave moment from orbit dictionary

    Parameters
    ----------
    data : 30 element dictionary
        full data package of an orbit given by clean_inspiral
    distance : float
        distance from GW source in geometric units
    m_bh : float
        central body mass in solar masses - optional

    Returns
    -------
    waves : N x 3 x 3 numpy array of floats
        GW moment in cartesian coords over interpolated time; waves[:,0,0] is h+ polarization, waves[:,0,1]=waves[:,1,0] is hx polarization
    int_time : N element numpy array of floats
        coordinate time of test particle, interpolated to be evenly spaced
        N is maximum of the length of the original time array or the integerized number of time units that have passed
    '''
    sphere, time = data["raw"][:, 1:], data["raw"][:, 0]
    int_sphere, int_time = interpolate2(sphere, time, supress, samp_freq)

    if data["inputs"][-1] == "grav":
        # Convert to mks units if it's grav
        if m_bh is False:
            G, c, mass = 1, 1, 1
        else:
            G, c, mass = 6.67e-11, 3e8, 1.989e30 * m_bh
            int_time *= G * mass / (c**3)
            int_sphere[:,0] *= G * mass / (c**2)

    elif data["inputs"][-1] != "grav":
        if data["inputs"][-1] == "mks":
            G, c, M_sun = 6.67e-11, 3e8, 1.989e30
        elif data["inputs"][-1] == "cgs":
            G, c, M_sun = 6.67e-8, 3e10, 1.989e33
        mass = m_bh * M_sun if m_bh else data["inputs"][0]

    distance *= G * mass / (c**2)
    rho = np.sqrt(int_sphere[:, 0]**2 + (G * mass * data["inputs"][1] / (c**2))**2)
    carts = np.zeros((len(rho), 3))
    carts[:, 0] = rho * np.sin(int_sphere[:, 1]) * np.cos(int_sphere[:, 2])
    carts[:, 1] = rho * np.sin(int_sphere[:, 1]) * np.sin(int_sphere[:, 2])
    carts[:, 2] = int_sphere[:, 0] * np.cos(int_sphere[:, 1])
    r2 = np.sum(carts * carts, axis=1)
    qmom = np.zeros((len(rho), 3, 3))
    qmom = carts[:, :, None] * carts[:, None, :]
    qmom -= np.eye(3)[None,:,:] * (r2[:,None,None]/3)

    h_plus, h_cross = gwaves(qmom, int_time, distance, n_hat)
    if data["inputs"][2]:
        return data["inputs"][2]*h_plus, data["inputs"][2]*h_cross, int_time
    else:
        return 1e-4*h_plus, 1e-4*h_cross, int_time

def full_transform2(data, distance, supress=True, samp_freq=1, m_bh=False, n_hat = None):    #defunctish??
    '''
    Calculates gravitational wave moment from orbit dictionary

    Parameters
    ----------
    data : 30 element dictionary
        full data package of an orbit given by clean_inspiral
    distance : float
        distance from GW source in geometric units
    m_bh : float
        central body mass in solar masses - optional

    Returns
    -------
    waves : N x 3 x 3 numpy array of floats
        GW moment in cartesian coords over interpolated time; waves[:,0,0] is h+ polarization, waves[:,0,1]=waves[:,1,0] is hx polarization
    int_time : N element numpy array of floats
        coordinate time of test particle, interpolated to be evenly spaced
        N is maximum of the length of the original time array or the integerized number of time units that have passed
    '''
    sphere, time = data["raw"][:, 1:], data["raw"][:, 0]
    int_time = np.arange(int(time[-1]))
    int_sphere = np.transpose([spi.CubicSpline(time, sphere[:, 0])(int_time),
                               spi.CubicSpline(time, sphere[:, 1])(int_time),
                               spi.CubicSpline(time, sphere[:, 2])(int_time)])

    if data["inputs"][-1] == "grav":
        # Convert to mks units if it's grav
        if m_bh is False:
            G, c, mass = 1, 1, 1
        else:
            G, c, mass = 6.67e-11, 3e8, 1.989e30 * m_bh
            int_time *= G * mass / (c**3)
            int_sphere[:,0] *= G * mass / (c**2)

    elif data["inputs"][-1] != "grav":
        if data["inputs"][-1] == "mks":
            G, c, M_sun = 6.67e-11, 3e8, 1.989e30
        elif data["inputs"][-1] == "cgs":
            G, c, M_sun = 6.67e-8, 3e10, 1.989e33
        mass = m_bh * M_sun if m_bh else data["inputs"][0]

    distance *= G * mass / (c**2)
    rho = np.sqrt(int_sphere[:, 0]**2 + (G * mass * data["inputs"][1] / (c**2))**2)
    carts = np.zeros((len(rho), 3))
    carts[:, 0] = rho * np.sin(int_sphere[:, 1]) * np.cos(int_sphere[:, 2])
    carts[:, 1] = rho * np.sin(int_sphere[:, 1]) * np.sin(int_sphere[:, 2])
    carts[:, 2] = int_sphere[:, 0] * np.cos(int_sphere[:, 1])
    r2 = np.sum(carts * carts, axis=1)
    qmom = np.zeros((len(rho), 3, 3))
    qmom = carts[:, :, None] * carts[:, None, :]
    qmom -= np.eye(3)[None,:,:] * (r2[:,None,None]/3)

    h_plus, h_cross = gwaves(qmom, int_time, distance, n_hat)
    if data["inputs"][2]:
        return data["inputs"][2]*h_plus, data["inputs"][2]*h_cross, int_time
    else:
        return 1e-5*h_plus, 1e-5*h_cross, int_time

def trace_ortholize_old(pos_list):
    '''
    Calculates quadrupole moment in cartesian coords from position in spherical coords 

    Parameters
    ----------
    pos : N x 3 numpy array of floats
        full data package of an orbit given by clean_inspiral, assumed interpolated to be evenly spaced across time

    Returns
    -------
    qmom : N x 3 x 3 numpy array of floats
        quadrupole moment of test particle per unit mass
    '''
    x = pos_list[:,0] * np.sin(pos_list[:,1]) * np.cos(pos_list[:,2])
    y = pos_list[:,0] * np.sin(pos_list[:,1]) * np.sin(pos_list[:,2])
    z = pos_list[:,0] * np.cos(pos_list[:,1]) 
    
    qmom = np.transpose(np.array([[x*x, x*y, x*z],
                                  [y*x, y*y, y*z],
                                  [z*x, z*y, z*z]]))
    return qmom

def trace_ortholize(pos_list, a=None):
    '''
    Calculates quadrupole moment in cartesian coords from position in spherical coords 

    Parameters
    ----------
    pos : N x 3 numpy array of floats
        full data package of an orbit given by clean_inspiral, assumed interpolated to be evenly spaced across time

    Returns
    -------
    qmom : N x 3 x 3 numpy array of floats
        quadrupole moment of test particle per unit mass
    '''
    if a == None:
        #print('old')
        return trace_ortholize_old(pos_list)
    x = np.sqrt(pos_list[:,0]**2 + a**2) * np.sin(pos_list[:,1]) * np.cos(pos_list[:,2])
    y = np.sqrt(pos_list[:,0]**2 + a**2) * np.sin(pos_list[:,1]) * np.sin(pos_list[:,2])
    z = pos_list[:,0] * np.cos(pos_list[:,1]) 
    
    qmom = np.transpose(np.array([[x*x, x*y, x*z],
                                  [y*x, y*y, y*z],
                                  [z*x, z*y, z*z]]))
    return qmom

@njit
def trace_ortholize_njit(pos_list, a=0):
    N = pos_list.shape[0]
    qmom = np.empty((N, 3, 3))
    
    r = pos_list[:, 0]
    theta = pos_list[:, 1]
    phi = pos_list[:, 2]

    r2_a2 = r**2 + a**2
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    x = np.sqrt(r2_a2) * sin_theta * cos_phi
    y = np.sqrt(r2_a2) * sin_theta * sin_phi
    z = r * cos_theta

    for i in range(N):
        xi = x[i]
        yi = y[i]
        zi = z[i]

        qmom[i, 0, 0] = xi * xi
        qmom[i, 0, 1] = xi * yi
        qmom[i, 0, 2] = xi * zi
        qmom[i, 1, 0] = yi * xi
        qmom[i, 1, 1] = yi * yi
        qmom[i, 1, 2] = yi * zi
        qmom[i, 2, 0] = zi * xi
        qmom[i, 2, 1] = zi * yi
        qmom[i, 2, 2] = zi * zi

    return qmom

def RR_radial(states, a, mu, ind1, ind2, state, constants, label, tol):
    '''
    Calculates change in characteristic orbital values from path of test particle through space 

    Parameters
    ----------
    states : N x 8 numpy array of floats
        list of state vectors - [4-position, 4-velocity] in geometric units
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    mu : float
        mass ratio of test particle to central body
    ind1 : int
        index value of the first entry in states relative to the master state list in clean_inspiral
    ind2 : int
        index value of the last entry in states relative to the master state list in clean_inspiral

    Returns
    -------
     4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    '''
    if (ind2 - ind1 - 10) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0] - states[0,0]
        int_sphere, int_time = interpolate3(sphere, time)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = matrix_derive3_numba(coolquad, dt) 
        # Get reference point values - radius and velocity from index 5
        ref_r = int_sphere[5, 0]
        drdt = np.zeros_like(int_time)
        drdt[1:-1] = (int_sphere[2:, 0] - int_sphere[:-2, 0]) / (int_time[2:] - int_time[:-2])
        ref_drdt = drdt[5]
        # Create a phase space mapping of points on the BACK HALF of the orbit
        #  (exclude the last 5 points, since we want a buffer for the derivative stuff)
        r_scale = np.std(int_sphere[:,0])
        v_scale = np.std(drdt[1:-1])
        r_candidates = int_sphere[len(int_time)//2:-5, 0]
        drdt_candidates = drdt[len(int_time)//2:-5]
        phase_space_diffs = ((r_candidates - ref_r)/r_scale)**2 + ((drdt_candidates - ref_drdt)/v_scale)**2
        # Estimate the radial period
        rad = max(find_rms(a*np.sign(constants[1]))*1.01, np.mean(int_sphere[:,0]))
        Omega_phi = 1.0 / (rad**1.5 + a)
        Omega_r = Omega_phi * np.sqrt(1 - 6/rad + 8*a/(rad**1.5) - 3*a*a/(rad**2))
        T_r = 2*np.pi / Omega_r
        # Calculate a number of anchor points based on how many radial periods have passed (rounding down)
        N = int(np.nanmax([1, np.floor(int_time[-1]/T_r)]))
        # Get the indexes of the candidates, sorted by proximity in phase space (and adjust the indexes to match the original data)
        sorted_ix = np.argsort(phase_space_diffs) + len(int_time)//2
        # Grab the first N indexes, since these correspond to the best matches
        candidates = sorted_ix[:N]
        # Grab the BIGGEST index, since that corresponds to the one FURTHEST from the start and will give us as many full orbits as possible
        # (with the assumption that it will usually just be one, but could be more)
        ref_ix = np.max(candidates)                       

        # If the sign of the radial velocity doesn't match (fully in the wrong phase of the orbit)
        # Just grab index -5 and take basically the whole thing
        if np.sign(drdt[5]) != np.sign(drdt[ref_ix]):
            ref_ix = -5
        dt2 = dt2[5:ref_ix]
        dt3 = dt3[5:ref_ix]
        int_time = int_time[5:ref_ix]   
        int_sphere = int_sphere[5:ref_ix]    

        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        dE = mu*np.sum(dedt*div)
        dLx, dLy, dLz = mu*np.sum(dldt*div, axis=0)

        E0, L0, C0 = constants
        if a == 0:
            z2 = C0/(L0**2 + C0)
        else:
            A = (a**2)*(1 - E0**2)
            sig = A + L0**2 + C0
            z2 = (sig - (sig**2 - 4*A*C0)**(1/2))/(2*A)

        P = E0*(int_sphere[:, 0]**2 + a*a) - a*L0
        D = int_sphere[:, 0]*(int_sphere[:, 0] - 2) + a*a
        dC = np.sum(((2*P*(int_sphere[:, 0]**2 + a*a)/D - 2*a*(a*E0 - L0))*dedt + (-2*P*a/D + 2*(a*E0 - L0))*dldt[:, 2])*div*mu)    

        E, L, C = E0 + dE, L0 + dLz, C0 + dC

        new_state = recalc_state3([E, L, C], state, a, tol=tol)
        return new_state, np.array([E, L, C])
    else:
        #print("----------- kaput!")
        return state, constants

def RR_least_squares(states, a, mu, ind1, ind2, state, constants, tol):
    '''
    Calculates change in characteristic orbital values from path of test particle through space 

    Parameters
    ----------
    states : N x 8 numpy array of floats
        list of state vectors - [4-position, 4-velocity] in geometric units
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    mu : float
        mass ratio of test particle to central body
    ind1 : int
        index value of the first entry in states relative to the master state list in clean_inspiral
    ind2 : int
        index value of the last entry in states relative to the master state list in clean_inspiral

    Returns
    -------
     4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    '''
    if (ind2 - ind1 - 10) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0] - states[0,0]
        #int_sphere, int_time = interpolate3(sphere, time)
        int_sphere, int_time = interpolate3(sphere, time)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = matrix_derive3_numba(coolquad, dt) 
        # Get reference point values - radius and velocity from index 5
        ref_r = int_sphere[5, 0]
        drdt = np.zeros_like(int_time)
        drdt[1:-1] = (int_sphere[2:, 0] - int_sphere[:-2, 0]) / (int_time[2:] - int_time[:-2])
        ref_drdt = drdt[5]
        # Create a phase space mapping of points on the BACK HALF of the orbit
        #  (exclude the last 5 points, since we want a buffer for the derivative stuff)
        r_scale = np.std(int_sphere[:,0])
        v_scale = np.std(drdt[1:-1])
        r_candidates = int_sphere[len(int_time)//2:-5, 0]
        drdt_candidates = drdt[len(int_time)//2:-5]
        phase_space_diffs = ((r_candidates - ref_r)/r_scale)**2 + ((drdt_candidates - ref_drdt)/v_scale)**2
        # Estimate the radial period
        rad = max(find_rms(a*np.sign(constants[1]))*1.01, np.mean(int_sphere[:,0]))
        Omega_phi = 1.0 / (rad**1.5 + a)
        Omega_r = Omega_phi * np.sqrt(1 - 6/rad + 8*a/(rad**1.5) - 3*a*a/(rad**2))
        T_r = 2*np.pi / Omega_r
        # Calculate a number of anchor points based on how many radial periods have passed (rounding down)
        N = int(np.nanmax([1, np.floor(int_time[-1]/T_r)]))
        # Get the indexes of the candidates, sorted by proximity in phase space (and adjust the indexes to match the original data)
        sorted_ix = np.argsort(phase_space_diffs) + len(int_time)//2
        # Grab the first N indexes, since these correspond to the best matches
        candidates = sorted_ix[:N]
        # Grab the BIGGEST index, since that corresponds to the one FURTHEST from the start and will give us as many full orbits as possible
        # (with the assumption that it will usually just be one, but could be more)
        ref_ix = np.max(candidates)                       

        # If the sign of the radial velocity doesn't match (fully in the wrong phase of the orbit)
        # Just grab index -5 and take basically the whole thing
        if np.sign(drdt[5]) != np.sign(drdt[ref_ix]):
            ref_ix = -5
        dt2 = dt2[5:ref_ix]
        dt3 = dt3[5:ref_ix]
        int_time = int_time[5:ref_ix]   
        int_sphere = int_sphere[5:ref_ix]    

        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        dE = mu*np.sum(dedt*div)
        dLx, dLy, dLz = mu*np.sum(dldt*div, axis=0)

        E0, L0, C0 = constants
        if a == 0:
            z2 = C0/(L0**2 + C0)
        else:
            A = (a**2)*(1 - E0**2)
            sig = A + L0**2 + C0
            z2 = (sig - (sig**2 - 4*A*C0)**(1/2))/(2*A)

        def get_observables(test_state, a):

            metric, _ = kerr_2(test_state, a)
            fmom = metric @ test_state[4:]

            E = -fmom[0]
            L = np.sqrt(fmom[2]**2 + fmom[3]**2/(np.sin(test_state[2])**2))

            cart_state = sph2cart(test_state, a)
            L_dir = np.cross(cart_state[1:4], cart_state[5:])

            L_vec = L * L_dir / np.linalg.norm(L_dir)
            Q = np.matmul(np.matmul(kill_tensor(test_state, a), test_state[4:]), test_state[4:])
            C = Q - (a*E - fmom[3])**2

            return np.array([E, *L_vec]), C


        def perturb_state(test_state, a, dv):

            r, theta = test_state[1], test_state[2]

            rho2 = r**2 + a**2*np.cos(theta)**2
            tri  = r**2 - 2*r + a**2
            al2 = (rho2*tri / (rho2*tri + 2*r*(a*a + r*r)))
            w = (2*r*a / (rho2*tri + 2*r*(a*a + r*r)))
            wbar2 = ((rho2*tri + 2*r*(a*a + r*r)) / rho2) * np.sin(theta)**2

            tet2kerr = np.array([[1/np.sqrt(al2), 0.0, 0.0, 0.0],
                                 [0.0, np.sqrt(tri/rho2), 0.0, 0.0],
                                 [0.0, 0.0, 1/np.sqrt(rho2), 0.0],
                                 [w/np.sqrt(al2), 0.0, 0.0, 1/np.sqrt(wbar2)]])
            kerr2tet = np.linalg.inv(tet2kerr)
            tetrad = kerr2tet @ test_state[4:]

            v = tetrad[1:] / tetrad[0]
            new_v = v + dv
            v2 = np.dot(new_v, new_v)
            if v2 >= 1:
                raise ValueError(f"Superluminal tetrad velocity encountered: v²={v2, v, dv}")

            gamma = 1/np.sqrt(1 - v2)
            new_tet = np.array([gamma, *(gamma*new_v)])
            new_vel = tet2kerr @ new_tet

            return np.array([*test_state[:4], *new_vel])

        obs0, C_init = get_observables(state, a)

        #E0  = obs0[0]
        Lx0 = obs0[1]
        Ly0 = obs0[2]
        Lz0 = L0 #obs0[3]

        target = np.array([dE, dLx, dLy, dLz])

        # ============================================================
        # Construct Jacobian
        # ============================================================

        eps = 1e-6

        A = np.zeros((4,3))
        gradC = np.zeros(3)

        for i in range(3):

            dv = np.zeros(3)
            dv[i] = eps

            state_p = perturb_state(state, a, dv)
            state_m = perturb_state(state, a, -dv)

            obs_p, C_p = get_observables(state_p, a)
            obs_m, C_m = get_observables(state_m, a)

            A[:, i] = (obs_p - obs_m) / (2*eps)
            gradC[i] = (C_p - C_m) / (2*eps)

        # ============================================================
        # Solve least squares problem
        # ============================================================
        targ_scale = 1
        while True:
            true_div, residuals, rank, singular_vals = np.linalg.lstsq(A, targ_scale*target, rcond=None)
            if np.linalg.norm(true_div) > 1e-8:
                break
            else:
                targ_scale *= 2

        # ============================================================
        # Linear prediction
        # ============================================================

        predicted = A @ true_div
        predicted_dC = gradC @ true_div

        # ============================================================
        # Apply perturbation
        # ============================================================

        new_state = perturb_state(state, a, true_div)
        obs1, C1 = get_observables(new_state, a)
        actual = obs1 - obs0
        actual_dC = C1 - C_init

        # ============================================================
        # Recover final constants
        # ============================================================

        metric, _ = kerr_2(new_state, a)
        fmom = metric @ new_state[4:]

        newE = -fmom[0]
        newLz = fmom[3]
        newQ = np.matmul(np.matmul(kill_tensor(new_state, a), new_state[4:]),new_state[4:])
        newC = C1
        predC = newC - C_init
        dC = predC/targ_scale

        E, L, C = E0 + dE, L0 + dLz, C0 + dC

        new_state = recalc_state3([E, L, C], state, a, tol=tol)
        return new_state, np.array([E, L, C])
    else:
        #print("----------- kaput!")
        return state, constants

def RR_const_inc(states, a, mu, ind1, ind2, state, constants, tol):
    '''
    Calculates change in characteristic orbital values from path of test particle through space 

    Parameters
    ----------
    states : N x 8 numpy array of floats
        list of state vectors - [4-position, 4-velocity] in geometric units
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    mu : float
        mass ratio of test particle to central body
    ind1 : int
        index value of the first entry in states relative to the master state list in clean_inspiral
    ind2 : int
        index value of the last entry in states relative to the master state list in clean_inspiral

    Returns
    -------
     4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    '''
    if (ind2 - ind1 - 10) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0] - states[0,0]
        int_sphere, int_time = interpolate3(sphere, time)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = matrix_derive3_numba(coolquad, dt) 
        # Get reference point values - radius and velocity from index 5
        ref_r = int_sphere[5, 0]
        drdt = np.zeros_like(int_time)
        drdt[1:-1] = (int_sphere[2:, 0] - int_sphere[:-2, 0]) / (int_time[2:] - int_time[:-2])
        ref_drdt = drdt[5]
        # Create a phase space mapping of points on the BACK HALF of the orbit
        #  (exclude the last 5 points, since we want a buffer for the derivative stuff)
        r_scale = np.std(int_sphere[:,0])
        v_scale = np.std(drdt[1:-1])
        r_candidates = int_sphere[len(int_time)//2:-5, 0]
        drdt_candidates = drdt[len(int_time)//2:-5]
        phase_space_diffs = ((r_candidates - ref_r)/r_scale)**2 + ((drdt_candidates - ref_drdt)/v_scale)**2
        # Estimate the radial period
        rad = max(find_rms(a*np.sign(constants[1]))*1.01, np.mean(int_sphere[:,0]))
        Omega_phi = 1.0 / (rad**1.5 + a)
        Omega_r = Omega_phi * np.sqrt(1 - 6/rad + 8*a/(rad**1.5) - 3*a*a/(rad**2))
        T_r = 2*np.pi / Omega_r
        # Calculate a number of anchor points based on how many radial periods have passed (rounding down)
        N = int(np.nanmax([1, np.floor(int_time[-1]/T_r)]))
        # Get the indexes of the candidates, sorted by proximity in phase space (and adjust the indexes to match the original data)
        sorted_ix = np.argsort(phase_space_diffs) + len(int_time)//2
        # Grab the first N indexes, since these correspond to the best matches
        candidates = sorted_ix[:N]
        # Grab the BIGGEST index, since that corresponds to the one FURTHEST from the start and will give us as many full orbits as possible
        # (with the assumption that it will usually just be one, but could be more)
        ref_ix = np.max(candidates)                       

        # If the sign of the radial velocity doesn't match (fully in the wrong phase of the orbit)
        # Just grab index -5 and take basically the whole thing
        if np.sign(drdt[5]) != np.sign(drdt[ref_ix]):
            ref_ix = -5
        dt2 = dt2[5:ref_ix]
        dt3 = dt3[5:ref_ix]
        int_time = int_time[5:ref_ix]   
        int_sphere = int_sphere[5:ref_ix]    

        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        dE = mu*np.sum(dedt*div)
        dLx, dLy, dLz = mu*np.sum(dldt*div, axis=0)

        E0, L0, C0 = constants
        if a == 0:
            z2 = C0/(L0**2 + C0)
        else:
            A = (a**2)*(1 - E0**2)
            sig = A + L0**2 + C0
            z2 = (sig - (sig**2 - 4*A*C0)**(1/2))/(2*A)

        # assume d/dt(cosi) = 0
        if L0 != 0.0:
            dC = 2*dLz*C0/L0 
        else:
            #polarrr
            metric, chris = kerr_2(state, a)
            theta = state[2]
            fmom = np.matmul(metric, state[4:])
            L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
            cart_state = sph2cart(state, a)
            L_dir = np.cross(cart_state[1:4], cart_state[5:])
            L_vec = L*L_dir/np.linalg.norm(L_dir)
            dC = 2*(L_vec[0]*dLx + L_vec[1]*dLy 
                    - (a*a)*((1 - E0**2)*np.sin(theta)*np.cos(theta)*(int_sphere[-1,1] - int_sphere[0,1])
                             + E0*dE*(np.cos(theta)**2))) 

        E, L, C = E0 + dE, L0 + dLz, C0 + dC

        new_state = recalc_state3([E, L, C], state, a, tol=tol)
        return new_state, np.array([E, L, C])
    else:
        #print("----------- kaput!")
        return state, constants

def peters_integrate6_6_4_7(states, a, mu, ind1, ind2, state, constants, label, tol):
    '''
    Calculates change in characteristic orbital values from path of test particle through space 

    Parameters
    ----------
    states : N x 8 numpy array of floats
        list of state vectors - [4-position, 4-velocity] in geometric units
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    mu : float
        mass ratio of test particle to central body
    ind1 : int
        index value of the first entry in states relative to the master state list in clean_inspiral
    ind2 : int
        index value of the last entry in states relative to the master state list in clean_inspiral

    Returns
    -------
     4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    '''
    if (ind2 - ind1 - 10) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0] - states[0,0]
        int_sphere, int_time = interpolate3(sphere, time)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = matrix_derive3_numba(coolquad, dt) 
        # Get reference point values - radius and velocity from index 5
        ref_r = int_sphere[5, 0]
        drdt = np.zeros_like(int_time)
        drdt[1:-1] = (int_sphere[2:, 0] - int_sphere[:-2, 0]) / (int_time[2:] - int_time[:-2])
        ref_drdt = drdt[5]
        # Create a phase space mapping of points on the BACK HALF of the orbit
        #  (exclude the last 5 points, since we want a buffer for the derivative stuff)
        r_scale = np.std(int_sphere[:,0])
        v_scale = np.std(drdt[1:-1])
        r_candidates = int_sphere[len(int_time)//2:-5, 0]
        drdt_candidates = drdt[len(int_time)//2:-5]
        #plt.plot((int_sphere[2:, 0] - int_sphere[:-2, 0]))
        #plt.plot((int_time[2:] - int_time[:-2]))
        phase_space_diffs = ((r_candidates - ref_r)/r_scale)**2 + ((drdt_candidates - ref_drdt)/v_scale)**2
        # Estimate the radial period
        rad = max(find_rms(a*np.sign(constants[1]))*1.01, np.mean(int_sphere[:,0]))
        Omega_phi = 1.0 / (rad**1.5 + a)
        Omega_r = Omega_phi * np.sqrt(1 - 6/rad + 8*a/(rad**1.5) - 3*a*a/(rad**2))
        T_r = 2*np.pi / Omega_r
        # Calculate a number of anchor points based on how many radial periods have passed (rounding down)
        N = int(np.nanmax([1, np.floor(int_time[-1]/T_r)]))
        # Get the indexes of the candidates, sorted by proximity in phase space (and adjust the indexes to match the original data)
        sorted_ix = np.argsort(phase_space_diffs) + len(int_time)//2
        # Grab the first N indexes, since these correspond to the best matches
        candidates = sorted_ix[:N]
        # Grab the BIGGEST index, since that corresponds to the one FURTHEST from the start and will give us as many full orbits as possible
        # (with the assumption that it will usually just be one, but could be more)
        ref_ix = np.max(candidates)                       

        # If the sign of the radial velocity doesn't match (fully in the wrong phase of the orbit)
        # Just grab index -5 and take basically the whole thing
        if np.sign(drdt[5]) != np.sign(drdt[ref_ix]):
            ref_ix = -5
        dt2 = dt2[5:ref_ix]
        dt3 = dt3[5:ref_ix]
        int_time = int_time[5:ref_ix]   
        int_sphere = int_sphere[5:ref_ix]    

        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        dE = mu*np.sum(dedt*div)
        dLx, dLy, dLz = mu*np.sum(dldt*div, axis=0)

        E0, L0, C0 = constants
        if a == 0:
            z2 = C0/(L0**2 + C0)
        else:
            A = (a**2)*(1 - E0**2)
            sig = A + L0**2 + C0
            z2 = (sig - (sig**2 - 4*A*C0)**(1/2))/(2*A)

        if "radial" in label:
            P = E0*(int_sphere[:, 0]**2 + a*a) - a*L0
            D = int_sphere[:, 0]*(int_sphere[:, 0] - 2) + a*a
            dC = np.sum(((2*P*(int_sphere[:, 0]**2 + a*a)/D - 2*a*(a*E0 - L0))*dedt + (-2*P*a/D + 2*(a*E0 - L0))*dldt[:, 2])*div*mu)

        elif "polial" in label:
            oop = np.min(int_sphere[:, 1])
            dC = np.sum((np.cos(oop)**2)*(-2*a*a*E0*dedt + (2*L0/(1 - np.cos(oop)**2))*dldt[:, 2])*div*mu)

        elif "poliar" in label:
            cosi_mod = np.arcsin(L0/np.sqrt(L0**2 + C0))
            dC = np.sum((np.cos(cosi_mod)**2)*(-2*a*a*E0*dedt + (2*L0/(1 - np.cos(cosi_mod)**2))*dldt[:, 2])*div*mu)

        elif "redial" in label:
            poly1 = np.array([E0**2 - 1, 2.0, (a**2)*(E0**2 - 1) - L0**2 - C0, 2*((a*E0 - L0)**2) + 2*C0, -C0*(a**2)])
            if "redialp" in label:
                ix = -2
            else:
                ix = -1
            r1 = np.sort(np.roots(poly1))[ix]
            P = E0*(r1**2 + a*a) - a*L0
            D = r1*(r1 - 2) + a*a
            dC0 = (2*P*(r1**2 + a*a)/D - 2*a*(a*E0 - L0))*dE + (-2*P*a/D + 2*(a*E0 - L0))*dLz
            E1, L1, C1 = E0 + dE, L0 + dLz, C0 + dC0
            poly2 = np.array([E1**2 - 1, 2.0, (a**2)*(E1**2 - 1) - L1**2 - C1, 2*((a*E1 - L1)**2) + 2*C1, -C1*(a**2)])
            r2 = np.sort(np.roots(poly2))[ix]
            dC = dC0 + ((2*P/(D**2))*(2*E0*r1*D + (1-r1)*P) - 2*r1)*(r2 - r1)

        elif "lystsq" in label:
            metric, _ = kerr_2(np.real(state), a)
            fmom = metric @ state[4:]
            r, theta, phi = state[1], state[2], state[3]
            sint, cost, sinp, cosp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
            sig = r**2 + a**2*np.cos(theta)**2
            tri  = r**2 - 2*r + a**2
            al2 = (sig*tri / (sig*tri + 2*r*(a*a + r*r)))
            w = (2*r*a / (sig*tri + 2*r*(a*a + r*r)))
            wbar2 = ( (sig*tri + 2*r*(a*a + r*r)) / sig) * np.sin(theta)**2
            tet2kerr = np.array([
                [1/np.sqrt(al2), 0.0,               0.0,             0.0],
                [0.0,            np.sqrt(tri/sig),  0.0,             0.0],
                [0.0,            0.0,               1/np.sqrt(sig),  0.0],
                [w/np.sqrt(al2), 0.0,               0.0,             1/np.sqrt(wbar2)]])
            kerr2tet = np.linalg.inv(tet2kerr)
            tetvec = kerr2tet @ state[4:]
            g = tetvec[0]
            v = tetvec/g
            # dE/dv^i = (dE/du^mu) * (du^mu/dv^i)
            # dL^j/dv^i = (dL^j/du^mu) * (du^mu/dv^i)
            rho = np.sqrt(r**2 + a**2)
    
            A = np.zeros((4, 3))
            kdeck = np.identity(4)
            for i in range(1, 4):
                valE = 0.0
                valLx = 0.0
                valLy = 0.0
                valLz = 0.0
                for mu in range(4):
                    dudv_mui = tet2kerr[mu, 0] * (g**3) * v[i]
                    for j in range(1,4):
                        dudv_mui += tet2kerr[mu, j] * ((g**3) * v[i] * v[j] + g * kdeck[i, j])
                    valE  += -metric[mu, 0] * dudv_mui

                    valLx += (a*a*rho/sig) * sint * cost * sinp * metric[mu, 1]
                    valLx -= (r*rho/sig) * sinp * metric[mu, 2]
                    valLx -= (r/rho) * cost * cosp * metric[mu, 3] / sint
                    valLx *= dudv_mui

                    valLy -= (a*a*rho/sig) * sint * cost * cosp * metric[mu, 1]
                    valLy += (r*rho/sig) * cosp * metric[mu, 2]
                    valLy -= (r/rho) * cost * sinp * metric[mu, 3] / sint
                    valLy *= dudv_mui

                    valLz +=  metric[mu, 3] * dudv_mui

                A[0, i-1] += valE
                A[1, i-1] += valLx
                A[2, i-1] += valLy
                A[3, i-1] += valLz
    
            #dv = (np.linalg.inv(A.T @ A) @ A.T) @ [dE, dLx, dLy, dLz]
            dv = np.linalg.lstsq(A, np.array([dE, dLx, dLy, dLz]), rcond=None)[0]
            new_v = v[1:] + dv
            v2 = np.dot(new_v, new_v)
            if v2 >= 1:
                raise ValueError(f"Superluminal tetrad velocity encountered gufgufguf: v²={v2, v[1:], dv}")
            new_g = 1/np.sqrt(1 - np.dot(new_v, new_v))
            new_u = tet2kerr @ (new_g * np.array([1.0, *new_v]))
            new_fmom = metric @ new_u
            newE, newLz = -new_fmom[0], new_fmom[3]
            newQ = (kill_tensor([*state[:4], *new_u], a) @ new_u) @ new_u
            newC = newQ - (a*newE - newLz)**2
            dC = newC - C0

        elif "lestsq" in label:
            metric, _ = kerr_2(np.real(state), a)
            fmom = metric @ state[4:]
            r, theta = state[1], state[2]
            rho2 = r**2 + a**2*np.cos(theta)**2
            tri  = r**2 - 2*r + a**2
            al2 = (rho2*tri / (rho2*tri + 2*r*(a*a + r*r)))
            w = (2*r*a / (rho2*tri + 2*r*(a*a + r*r)))
            wbar2 = ( (rho2*tri + 2*r*(a*a + r*r)) / rho2) * np.sin(theta)**2
            tet2kerr = np.array([
                [1/np.sqrt(al2), 0.0,               0.0,             0.0],
                [0.0,            np.sqrt(tri/rho2), 0.0,             0.0],
                [0.0,            0.0,               1/np.sqrt(rho2), 0.0],
                [w/np.sqrt(al2), 0.0,               0.0,             1/np.sqrt(wbar2)]])
            kerr2tet = np.linalg.inv(tet2kerr)
            tetvec = kerr2tet @ state[4:]
            g = tetvec[0]
            v = tetvec/g
            # dE/dv^i = (dE/du^mu) * (du^mu/dv^i)
            # dL^j/dv^i = (dL^j/du^mu) * (du^mu/dv^i)

            A = np.zeros((4, 3))
            kdeck = np.identity(4)
            for i in range(1, 4):
                valE = 0.0
                valLx = 0.0
                valLy = 0.0
                valLz = 0.0
                for mu in range(4):
                    dudv_mui = 0.0
                    for j in range(4):
                        dudv_mui += (tet2kerr[mu, 0] * (g**3) * v[j] + tet2kerr[mu, j] * ((g**3) * v[i] * v[j] + g * kdeck[i, j]))
                    valE  += -metric[mu, 0] * dudv_mui
                    valLx += -metric[mu, 2] * dudv_mui * np.sin(state[3]) - metric[mu, 3] * dudv_mui * np.cos(state[3]) / np.tan(state[2])
                    valLy +=  metric[mu, 2] * dudv_mui * np.cos(state[3]) - metric[mu, 3] * dudv_mui * np.sin(state[3]) / np.tan(state[2])
                    valLz +=  metric[mu, 3] * dudv_mui
                A[0, i-1] += valE
                A[1, i-1] += valLx
                A[2, i-1] += valLy
                A[3, i-1] += valLz

            dv = (np.linalg.inv(A.T @ A) @ A.T) @ [dE, dLx, dLy, dLz]
            new_v = v[1:] + dv
            v2 = np.dot(new_v, new_v)
            if v2 >= 1:
                raise ValueError(f"Superluminal tetrad velocity encountered MKNJBHVKCGF: v²={v2, v[1:], dv}")
            new_g = 1/np.sqrt(1 - np.dot(new_v, new_v))
            new_u = tet2kerr @ (new_g * np.array([1.0, *new_v]))
            new_fmom = metric @ new_u
            newE, newLz = -new_fmom[0], new_fmom[3]
            newQ = (kill_tensor([*state[:4], *new_u], a) @ new_u) @ new_u
            newC = newQ - (a*newE - newLz)**2
            dC = newC - C0

        elif "liestsq" in label:
            metric, _ = kerr_2(np.real(state), a)
            fmom = metric @ state[4:]
            r, theta, phi = state[1], state[2], state[3]
            sint, cost, sinp, cosp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
            rho2 = r**2 + (a*cost)**2
            tri  = r**2 - 2*r + a**2
            al2 = (rho2*tri / (rho2*tri + 2*r*(a*a + r*r)))
            w = (2*r*a / (rho2*tri + 2*r*(a*a + r*r)))
            wbar2 = ( (rho2*tri + 2*r*(a*a + r*r)) / rho2) * sint**2
            tet2kerr = np.array([
                [1/np.sqrt(al2), 0.0,               0.0,             0.0],
                [0.0,            np.sqrt(tri/rho2), 0.0,             0.0],
                [0.0,            0.0,               1/np.sqrt(rho2), 0.0],
                [w/np.sqrt(al2), 0.0,               0.0,             1/np.sqrt(wbar2)]])
            kerr2tet = np.linalg.inv(tet2kerr)
            tetvec = kerr2tet @ state[4:]
            g = tetvec[0]
            v = tetvec/g
            lev_civ = np.zeros((4, 4, 4))
            for ix in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]:
                lev_civ[ix] = 1
            for ix in [(1, 3, 2), (2, 1, 3), (3, 2, 1)]:
                lev_civ[ix] = -1
            fmom = np.matmul(metric, state[4:])
            L2 = fmom[2]**2 + (fmom[3]**2)/(sint**2)
            L = np.sqrt(L2)
            cart_state = sph2cart(state, a)
            rc = cart_state[:4]
            rxu = np.cross(cart_state[1:4], cart_state[5:])
            rxu_norm = np.linalg.norm(rxu)
            L_vec = L * rxu/rxu_norm
            rho = np.sqrt(r**2 + a**2)
            K = np.array([[0.0, 0.0,                   0.0,               0.0        ],
                          [0.0, r * sint * cosp / rho, rho * cost * cosp, -rho * sinp],
                          [0.0, r * sint * sinp / rho, rho * cost * sinp, -rho * cosp],
                          [0.0, cost,                  r * sint,          0.0        ]])

            # dE/dv^i = (dE/du^mu) * (du^mu/dv^i)
            # dL^j/dv^i = (dL^j/du^mu) * (du^mu/dv^i)
            # dE/du^mu = -g_{mu, 0} 
            # p = fmom, W=Levi-civita, (rc, uc) = cartesian (position, velocity)
            # K^i_mu = matrix for converting between uc and u
            # L_vec^j = L rxu^j / rxu_norm
            # dL_vec^j/du^mu = (L_vec^j/(2 L^2)) d(L^2)/du^mu + (L/rxu_norm)(drxu^j/du^mu - (L_vec^j L_vec^m/L^2) dArxu^m/du^mu)
            # d(L^2)/du^mu = (2 p_2 g_{mu, 2} + 2 p_3 g_{mu, 3}/sin^2(theta))
            # drxu^j/du^mu = W_{jkl} rc^k K^l_mu

            A = np.zeros((4, 3))
            kdeck = np.identity(4)
            for i in range(1, 4):
                valE = 0.0
                valLs = np.zeros(3)  #[valLx, valLy, valLz]
                for mu in range(4):
                    dudv_mui = tet2kerr[mu, 0] * g**3 * v[i]
                    for j in range(1, 4):
                        dudv_mui += tet2kerr[mu, j] * ((g**3) * v[i] * v[j] + g * kdeck[i, j])
                    valE  += -metric[mu, 0] * dudv_mui

                    for j in range(3):
                        first_term = (L_vec[j]/(2*L2)) * (2 * fmom[2] * metric[mu, 2] + 2 * fmom[3] * metric[mu, 3] / (sint**2))
                        second_term = 0.0
                        drxudu_jmu = 0.0
                        for k, l in product(range(3), range(3)):
                            drxudu_jmu += lev_civ[j+1, k+1, l+1] * rc[k+1] * K[l+1, mu]
                        second_term += drxudu_jmu
                        for m in range(3):
                            drxudu_mmu = 0.0
                            for n, o in product(range(3), range(3)):
                                drxudu_mmu += lev_civ[m+1, n+1, o+1] * rc[n+1] * K[o+1, mu]
                            second_term -= (L_vec[j] * L_vec[m] / L2) * drxudu_mmu
                        second_term *= (L/rxu_norm)
                        valLs[j] += (first_term + second_term) * dudv_mui
                            
                A[0, i-1] += valE
                A[1, i-1] += valLs[0]
                A[2, i-1] += valLs[1]
                A[3, i-1] += valLs[2]

            dv = (np.linalg.inv(A.T @ A) @ A.T) @ [dE, dLx, dLy, dLz]
            new_v = v[1:] + dv
            v2 = np.dot(new_v, new_v)
            if v2 >= 1:
                raise ValueError(f"Superluminal tetrad velocity encountered scooby: v²={v2, v[1:], dv}")
            new_g = 1/np.sqrt(1 - np.dot(new_v, new_v))
            new_u = tet2kerr @ (new_g * np.array([1.0, *new_v]))
            new_fmom = metric @ new_u
            newE, newLz = -new_fmom[0], new_fmom[3]
            newQ = (kill_tensor([*state[:4], *new_u], a) @ new_u) @ new_u
            newC = newQ - (a*newE - newLz)**2
            dC = newC - C0
        
        elif "lastsq" in label:
            # ============================================================
            # Least-squares radiation reaction update
            # ============================================================

            def get_observables(test_state, a):

                metric, _ = kerr_2(test_state, a)

                fmom = metric @ test_state[4:]

                E = -fmom[0]

                L = np.sqrt(
                    fmom[2]**2 +
                    fmom[3]**2/(np.sin(test_state[2])**2)
                )

                cart_state = sph2cart(test_state, a)

                L_dir = np.cross(
                    cart_state[1:4],
                    cart_state[5:]
                )

                L_vec = L * L_dir / np.linalg.norm(L_dir)

                Q = np.matmul(
                    np.matmul(
                        kill_tensor(test_state, a),
                        test_state[4:]
                    ),
                    test_state[4:]
                )

                C = Q - (a*E - fmom[3])**2

                return np.array([
                    E,
                    L_vec[0],
                    L_vec[1],
                    L_vec[2]
                ]), C


            def perturb_state(test_state, a, dv):

                r, theta = test_state[1], test_state[2]

                rho2 = r**2 + a**2*np.cos(theta)**2
                tri  = r**2 - 2*r + a**2

                al2 = (
                    rho2*tri /
                    (rho2*tri + 2*r*(a*a + r*r))
                )

                w = (
                    2*r*a /
                    (rho2*tri + 2*r*(a*a + r*r))
                )

                wbar2 = (
                    (rho2*tri + 2*r*(a*a + r*r))
                    / rho2
                ) * np.sin(theta)**2

                tet2kerr = np.array([
                    [1/np.sqrt(al2), 0.0, 0.0, 0.0],
                    [0.0, np.sqrt(tri/rho2), 0.0, 0.0],
                    [0.0, 0.0, 1/np.sqrt(rho2), 0.0],
                    [w/np.sqrt(al2), 0.0, 0.0, 1/np.sqrt(wbar2)]
                ])

                kerr2tet = np.linalg.inv(tet2kerr)

                tetrad = kerr2tet @ test_state[4:]

                v = tetrad[1:] / tetrad[0]

                new_v = v + dv

                v2 = np.dot(new_v, new_v)

                if v2 >= 1:
                    raise ValueError(f"Superluminal tetrad velocity encountered: v²={v2, v, dv}")

                gamma = 1/np.sqrt(1 - v2)

                new_tet = np.array([
                    gamma,
                    gamma*new_v[0],
                    gamma*new_v[1],
                    gamma*new_v[2]
                ])

                new_vel = tet2kerr @ new_tet

                return np.array([
                    *test_state[:4],
                    *new_vel
                ])


            # ============================================================
            # Initial quantities
            # ============================================================

            obs0, C_init = get_observables(state, a)

            #E0  = obs0[0]
            Lx0 = obs0[1]
            Ly0 = obs0[2]
            Lz0 = L0 #obs0[3]

            target = np.array([
                dE,
                dLx,
                dLy,
                dLz
            ])

            # ============================================================
            # Construct Jacobian
            # ============================================================

            eps = 1e-6

            A = np.zeros((4,3))
            gradC = np.zeros(3)

            for i in range(3):

                dv = np.zeros(3)
                dv[i] = eps

                state_p = perturb_state(state, a, dv)
                state_m = perturb_state(state, a, -dv)

                obs_p, C_p = get_observables(state_p, a)
                obs_m, C_m = get_observables(state_m, a)

                A[:, i] = (
                    obs_p - obs_m
                )/(2*eps)

                gradC[i] = (
                    C_p - C_m
                )/(2*eps)

            # ============================================================
            # Solve least squares problem
            # ============================================================
            targ_scale = 1
            while True:
                true_div, residuals, rank, singular_vals = np.linalg.lstsq(
                    A,
                    targ_scale*target,
                    rcond=None
                )
                if np.linalg.norm(true_div) > 1e-8:
                    break
                else:
                    targ_scale *= 2

            # ============================================================
            # Linear prediction
            # ============================================================

            predicted = A @ true_div
            predicted_dC = gradC @ true_div

            # ============================================================
            # Apply perturbation
            # ============================================================

            new_state = perturb_state(
                state,
                a,
                true_div
            )

            obs1, C1 = get_observables(
                new_state,
                a
            )

            actual = obs1 - obs0
            actual_dC = C1 - C_init

            # ============================================================
            # Recover final constants
            # ============================================================

            metric, _ = kerr_2(new_state, a)

            fmom = metric @ new_state[4:]

            newE = -fmom[0]
            newLz = fmom[3]

            newQ = np.matmul(
                np.matmul(
                    kill_tensor(new_state, a),
                    new_state[4:]
                ),
                new_state[4:]
            )

            newC = C1

            predC = newC - C_init
            dC = predC/targ_scale

            # ============================================================
            # Diagnostics
            # ============================================================

            '''print("\n================ LSQ DIAGNOSTICS ================")

            print("\nTarget:")
            print(target)

            print("\nApplied scaling: ", targ_scale)

            print("\nLinear prediction:")
            print(predicted)

            print("\nActual achieved:")
            print(actual)

            print("\nTarget - prediction:")
            print(target - predicted)

            print("\nPrediction - actual:")
            print(predicted - actual)

            print("\nJacobian:")
            print(A)

            print("\nC gradient:")
            print(gradC)

            print("\nSingular values:")
            print(singular_vals)

            if singular_vals[-1] > 0:
                print(
                    "Condition number =",
                    singular_vals[0]/singular_vals[-1]
                )

            print("\nRank =", rank)

            print("\nVelocity correction:")
            print(true_div)

            print(
                "||dv|| =",
                np.linalg.norm(true_div)
            )

            print(
                "max(|dv|) =",
                np.max(np.abs(true_div))
            )

            print("\nC diagnostics:")
            print("predicted dC =", predicted_dC)
            print("actual dC    =", actual_dC)

            print("\nFinal constants:")
            print("E0    =", E0)
            print("newE  =", newE)
            print("dE    =", newE - E0)

            print("C0    =", C0)
            print("newC  =", newC)
            print("dC    =", predC)

            print("\nTrue changes to be applied:")
            print("Original dEq:", dE)
            print("Original dLz:", dLz)
            print("Final dC", dC)

            print("================================================\n")'''

        elif "leastsq" in label:
            metric, chris = kerr_2(state, a)
            true_fmom = metric @ state[4:]
            E0, L0, Q0 = -true_fmom[0], true_fmom[3], np.matmul(np.matmul(kill_tensor(state, a), state[4:]), state[4:])
            C0 = Q0 - (a*E0 - L0)**2
            r, theta, phi = state[1:4]
            sint, cost, sinp, cosp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
            rho2, tri = r**2 + (a**2)*(cost**2), r**2 - 2*r + a**2
            al2 = (rho2*tri)/(rho2*tri + 2*r*(a**2 + r**2))
            w = (2*r*a)/(rho2*tri + 2*r*(a**2 + r**2))
            wbar2 = ((rho2*tri + 2*r*(a**2 + r**2))/rho2)*(sint**2)
            tet2kerr = np.array([[1/np.sqrt(al2), 0.0,               0.0,             0.0],
                                [0.0,            np.sqrt(tri/rho2), 0.0,             0.0],
                                [0.0,            0.0,               1/np.sqrt(rho2), 0.0],
                                [w/np.sqrt(al2), 0.0,               0.0,             1/np.sqrt(wbar2)]])
            kerr2tet = np.linalg.inv(tet2kerr)
            tetrad = kerr2tet @ state[4:]
            v = tetrad[1:]/tetrad[0]
            fmom = np.matmul(metric, state[4:])
            L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
            cart_state = sph2cart(state, a)
            L_dir = np.cross(cart_state[1:4], cart_state[5:])
            L_vec = L*L_dir/np.linalg.norm(L_dir)

            A = np.zeros((4,3))
            A_C = np.zeros(3)
            eps = 1e-5/state[1]
            for i in range(3):
                div = np.zeros(3)
                div[i] = eps
                gamma = 1/np.sqrt(1 - np.dot(v + div, v + div))
                new_tet = np.array([gamma, *(gamma*(v + div))])
                new_vel = tet2kerr @ new_tet
                new_fmom = metric @ new_vel
                new_L = np.sqrt(new_fmom[2]**2 + (new_fmom[3]**2)/(np.sin(theta)**2))
                new_cart_state = sph2cart([*state[:4], *new_vel], a)
                new_L_dir = np.cross(new_cart_state[1:4], new_cart_state[5:])
                new_L_vec = new_L * new_L_dir/np.linalg.norm(new_L_dir)
                fplus = np.array([-new_fmom[0] - E0, *(new_L_vec - L_vec)])
                newQ = np.matmul(np.matmul(kill_tensor(np.array([*state[:4], *new_vel]), a), new_vel), new_vel)
                newC = newQ - (-a*new_fmom[0] - new_fmom[3])**2
                fC_plus = newC - C0
                div[i] = -eps
                gamma = 1/np.sqrt(1 - np.dot(v + div, v + div))
                new_tet = np.array([gamma, *(gamma*(v + div))])
                new_vel = tet2kerr @ new_tet
                new_fmom = metric @ new_vel
                new_L = np.sqrt(new_fmom[2]**2 + (new_fmom[3]**2)/(np.sin(theta)**2))
                new_cart_state = sph2cart([*state[:4], *new_vel], a)
                new_L_dir = np.cross(new_cart_state[1:4], new_cart_state[5:])
                new_L_vec = new_L * new_L_dir/np.linalg.norm(new_L_dir)
                fminus = np.array([-new_fmom[0] - E0, *(new_L_vec - L_vec)])
                newQ = np.matmul(np.matmul(kill_tensor(np.array([*state[:4], *new_vel]), a), new_vel), new_vel)
                newC = newQ - (-a*new_fmom[0] - new_fmom[3])**2
                fC_minus = newC - C0
                A[:, i] = (fplus - fminus)/(2*eps)
                A_C[i] = (fC_plus - fC_minus)/(2*eps)
                    
            u, s, vt = np.linalg.svd(A)
            true_div = np.linalg.lstsq(A, [dE, dLx, dLy, dLz], rcond=None)[0]
            gamma = 1/np.sqrt(1 - np.dot(v + true_div, v + true_div))
            new_tet = np.array([gamma, *(gamma*(v + true_div))])
            new_vel = tet2kerr @ new_tet
            test_fmom = metric @ new_vel
            new_L = np.sqrt(test_fmom[2]**2 + (test_fmom[3]**2)/(np.sin(theta)**2))
            new_cart_state = sph2cart([*state[:4], *new_vel], a)
            new_L_dir = np.cross(new_cart_state[1:4], new_cart_state[5:])
            new_L_vec = new_L * new_L_dir/np.linalg.norm(new_L_dir)
            predicted_from_mapping = [-test_fmom[0] - E0, *(new_L_vec - L_vec)]
            #print("predicted from mapping =", predicted_from_mapping)
            A[i%4, i] = (-new_fmom[0] - E0)/eps
            new_L = np.sqrt(new_fmom[2]**2 + (new_fmom[3]**2)/(np.sin(theta)**2))
            new_cart_state = sph2cart([*state[:4], *new_vel], a)
            new_L_dir = np.cross(new_cart_state[1:4], new_cart_state[5:])
            new_L_vec = new_L * new_L_dir/np.linalg.norm(new_L_dir)
            A[1:, i] = (new_L_vec - L_vec)/eps
            target = np.array([dE, dLx, dLy, dLz])
            achieved = A @ true_div

            new_v = v + true_div
            true_gamma = 1/np.sqrt(1 - np.dot(new_v, new_v))
            true_tet = np.array([true_gamma, *(true_gamma * new_v)])
            true_vel = tet2kerr @ true_tet
            new_state = np.array([*state[:4], *true_vel])
            true_fmom = metric @ true_vel
            newE, newLz, newQ = -true_fmom[0], true_fmom[3], np.matmul(np.matmul(kill_tensor(new_state, a), new_state[4:]), new_state[4:])
            newC = newQ - (a*newE - newLz)**2
            dC = C0 - newC
            Q0 = np.matmul(np.matmul(kill_tensor(state, a), state[4:]), state[4:])

            actual_L = np.sqrt(
                true_fmom[2]**2 +
                true_fmom[3]**2/(np.sin(theta)**2)
            )

            actual_cart = sph2cart(new_state, a)
            actual_L_dir = np.cross(actual_cart[1:4], actual_cart[5:])
            actual_L_vec = actual_L * actual_L_dir/np.linalg.norm(actual_L_dir)
            test_div = true_div

            gamma = 1/np.sqrt(1 - np.dot(v + test_div, v + test_div))
            new_tet = np.array([gamma, *(gamma*(v + test_div))])
            new_vel = tet2kerr @ new_tet

            test_fmom = metric @ new_vel

            test_E = -test_fmom[0]

            test_L = np.sqrt(
                test_fmom[2]**2 +
                test_fmom[3]**2/(np.sin(theta)**2)
            )

            test_cart = sph2cart([*state[:4], *new_vel], a)
            test_L_dir = np.cross(test_cart[1:4], test_cart[5:])
            test_L_vec = test_L*test_L_dir/np.linalg.norm(test_L_dir)

        elif "bracket" in label:
            def R(E, L, C, a):
                return np.array([ E**2 - 1, 2, (a**2)*(E**2 - 1) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])

            def T(E, L, C, a):
                return np.array([(a**2)*(1 - E**2), 0.0, -(C + (a**2)*(1 - E**2) + L**2), 0.0, C])
            
            def constants_to_pei3(E, L, C, a):
                # Get roots and extrema
                R2 = R(E, L, C, a)
                turns, flats = np.sort(np.roots(R2)), np.sort(np.roots(np.polyder(R2)))
                T2 = T(E, L, C, a)
                zs = np.sort(np.roots(T2))
                pot_min = flats[-1]
                inner_turn, outer_turn = turns[-2:]
                inc = np.sign(L)*np.arccos(min(1.0, np.mean(np.abs(zs[1:3]))))
                dr = np.linspace(turns[-2]*(1 + 1e-5), turns[-1]*(1 - 1e-5), 500)
                return inner_turn, outer_turn, inc, *np.polyval(np.polyder(R2), turns)

            def get_Js(E, L, C, a):
                R2 = R(E, L, C, a)
                turns, flats = np.sort(np.roots(R2)), np.sort(np.roots(np.polyder(R2)))
                T2 = T(E, L, C, a)
                zs = np.sort(np.roots(T2))

                def integrand_r(r):
                    D = r*r - 2*r + a*a
                    R3 = np.polyval(R2, r)
                    return np.sqrt(np.maximum(R3, 0.0))/D

                def integrand_z(z):
                    T3 = np.polyval(T2, z)
                    return np.sqrt(np.maximum(T3,0.0)) / np.sqrt(np.maximum(1-z*z,1e-15))

                val_r, _ = integrate.quad(integrand_r, turns[-2], turns[-1], limit=200)
                val_z, _ = integrate.quad(integrand_z, 0, min(1.0, np.mean(np.abs(zs[1:3]))), limit=200)
                return [val_r/np.pi, 2*val_z/np.pi]

            original = np.array(constants_to_pei3(E0, L0, C0, a))
            basic = np.array(constants_to_pei3(E0 + dE, L0 + dLz, C0, a))
            maxim = 20*np.abs(dLz*np.sqrt(L0**2 + C0)/L0)
            length = 100
            dCs = -np.linspace(0, maxim, length)
            full = np.array([constants_to_pei3(E0 + dE, L0 + dLz, C0 + trial, a) for trial in dCs])
            mask = np.where((full[:, 0] <= original[0])*(full[:, 1] <= original[1])*(full[:,2] <= original[2])*(np.abs(full[:,5]) <= np.abs(original[5]))*(np.abs(full[:,6]) <= np.abs(original[6])))
            while True:
                if length > 1000:
                    break
                elif (len(mask[0]) < 3):
                    length *= 2
                elif mask[0][-1] == len(dCs) - 1:
                    maxim *= 2
                else:
                    break
                dCs = -np.linspace(0, maxim, length)
                full = np.array([constants_to_pei3(E0 + dE, L0 + dLz, C0 + trial, a) for trial in dCs])
                mask = np.where((full[:, 0] <= original[0])*(full[:, 1] <= original[1])*(full[:,2] <= original[2])*(np.abs(full[:,5]) <= np.abs(original[5]))*(np.abs(full[:,6]) <= np.abs(original[6])))
            if len(mask) == 0:
                dCs = [[0]]
            dCs = np.linspace(dCs[mask[0][-1]], dCs[mask[0][0]], length)

            if "j" in label:
                OJs = np.array(get_Js(E0, L0, C0, a))
                full_Js = np.array([get_Js(E0 + dE, L0 + dLz, C0 + boop, a) for boop in dCs])
                if "jtot" in label:
                    diffy = ((full_Js[:, 0] - OJs[0])/OJs[0])**2 + ((full_Js[:, 1] - OJs[1])/OJs[1])**2   #Get minimum change of [Jr, Jthet]
                elif "jratio" in label:
                    diffy = (full_Js[:, 0]/full_Js[:, 1] - OJs[0]/OJs[1])**2  #Get minimum change of Jr/Jthet
                ix = np.argmin(diffy)
            elif "rri" in label:    # norm of [d(inc), d(peri)/peri, d(apo)/apo]
                full = np.array([constants_to_pei3(E0 + dE, L0 + dLz, C0 + trial, a) for trial in dCs])
                diffy = np.sqrt(((full[:,2] - original[2]))**2 + ((full[:,0] - original[0])/full[:,0])**2 + ((full[:,1] - original[1])/full[:,1])**2)
                ix = np.argmin(diffy)
            elif "big" in label:
                ix = np.argmin(dCs)
            elif "small" in label:
                ix = np.argmax(dCs)
            elif "half" in label:
                ix = op.get_index(dCs, np.mean(dCs))
    
            dC = dCs[ix]

        elif "incy0" in label:
            # assume d/dt(cosi) = 0
            if L0 != 0.0:
                dC = 2*dLz*C0/L0 
            else:
                #polarrr
                metric, chris = kerr_2(state, a)
                theta, phi = state[2], state[3]
                fmom = np.matmul(metric, state[4:])
                Lx = -fmom[2] * np.sin(phi) - fmom[3] * np.cos(theta) * np.cos(phi) / np.sin(theta)
                Ly =  fmom[2] * np.cos(phi) - fmom[3] * np.cos(theta) * np.sin(phi) / np.sin(theta)
                dC = 2*(Lx*dLx + Ly*dLy 
                        - (a*a)*((1 - E0**2)*np.sin(theta)*np.cos(theta)*(int_sphere[-1,1] - int_sphere[0,1])
                                    + E0*dE*(np.cos(theta)**2)))

        elif "inci0" in label:
            # assume d/dt(cosi) = 0
            if L0 != 0.0:
                dC = 2*dLz*C0/L0 
            else:
                #polarrr
                metric, chris = kerr_2(state, a)
                theta, phi = state[2], state[3]
                fmom = np.matmul(metric, state[4:])
                Lx = -fmom[2] * np.sin(phi) - fmom[3] * np.cos(theta) * np.cos(phi) / np.sin(theta)
                Ly =  fmom[2] * np.cos(phi) - fmom[3] * np.cos(theta) * np.sin(phi) / np.sin(theta)
                dC = 2*(Lx*dLx + Ly*dLy)
        
        else:
            # assume d/dt(cosi) = 0
            if L0 != 0.0:
                dC = 2*dLz*C0/L0 
            else:
                #polarrr
                metric, chris = kerr_2(state, a)
                theta = state[2]
                fmom = np.matmul(metric, state[4:])
                L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
                cart_state = sph2cart(state, a)
                L_dir = np.cross(cart_state[1:4], cart_state[5:])
                L_vec = L*L_dir/np.linalg.norm(L_dir)
                dC = 2*(L_vec[0]*dLx + L_vec[1]*dLy 
                        - (a*a)*((1 - E0**2)*np.sin(theta)*np.cos(theta)*(int_sphere[-1,1] - int_sphere[0,1])
                                 + E0*dE*(np.cos(theta)**2)))

        #dC = min(0.0, dC)
        E, L, C = E0 + dE, L0 + dLz, C0 + dC

        new_state = recalc_state3([E, L, C], state, a, tol=tol)
        return new_state, np.array([E, L, C])
    else:
        #print("----------- kaput!")
        return state, constants

@njit
def compute_dldt(dt2, dt3):
    N = dt2.shape[0]
    dldt = np.zeros((N, 3))
    
    # Levi-Civita symbol
    levciv = np.zeros((3, 3, 3))
    levciv[0, 1, 2] = levciv[1, 2, 0] = levciv[2, 0, 1] = 1
    levciv[0, 2, 1] = levciv[1, 0, 2] = levciv[2, 1, 0] = -1

    for l in range(N):
        for i in range(3):
            acc = 0.0
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        acc += levciv[i, j, k] * dt2[l, j, m] * dt3[l, k, m]
            dldt[l, i] = (-2.0 / 5.0) * acc
    return dldt

def sph2cart(vec, a):
    x = np.sqrt(vec[1]**2 + a**2) * np.sin(vec[2]) * np.cos(vec[3])
    y = np.sqrt(vec[1]**2 + a**2) * np.sin(vec[2]) * np.sin(vec[3])
    z = vec[1] * np.cos(vec[2]) 
    vx = vec[1]*vec[5]/(np.sqrt(vec[1]**2 + a**2)) * np.sin(vec[2]) * np.cos(vec[3]) + np.sqrt(vec[1]**2 + a**2) * vec[6] * np.cos(vec[2]) * np.cos(vec[3]) + np.sqrt(vec[1]**2 + a**2) * np.sin(vec[2]) * vec[7] * (-np.sin(vec[3]))
    vy = vec[1]*vec[5]/(np.sqrt(vec[1]**2 + a**2)) * np.sin(vec[2]) * np.sin(vec[3]) + np.sqrt(vec[1]**2 + a**2) * vec[6] * np.cos(vec[2]) * np.sin(vec[3]) + np.sqrt(vec[1]**2 + a**2) * np.sin(vec[2]) * vec[7] * np.cos(vec[3])
    vz = vec[5] * np.cos(vec[2]) + vec[1] * vec[6] * (-np.sin(vec[2]))
    new_vec = np.array([vec[0], x, y, z, vec[4], vx, vy, vz])
    return new_vec

def sph2cart_vec(raw, a):
    rho = np.sqrt(raw[:, 1]**2 + a**2)
    sin_th, cos_th = np.sin(raw[:, 2]), np.cos(raw[:, 2])
    sin_ph, cos_ph = np.sin(raw[:, 3]), np.cos(raw[:, 3])
    x = rho * sin_th * cos_ph
    y = rho * sin_th * sin_ph
    z = raw[:, 1] * cos_th 
    vx = raw[:, 1]*raw[:, 5]/(rho) * sin_th * cos_ph + rho * raw[:, 6] * cos_th * cos_ph + rho * sin_th * raw[:, 7] * (-sin_ph)
    vy = raw[:, 1]*raw[:, 5]/(rho) * sin_th * sin_ph + rho * raw[:, 6] * cos_th * sin_ph + rho * sin_th * raw[:, 7] * cos_ph
    vz = raw[:, 5] * cos_th + raw[:, 1] * raw[:, 6] * (-sin_th)
    new_raw = np.transpose([raw[:, 0], x, y, z, raw[:, 4], vx, vy, vz])
    return new_raw

def cart2sph(vec, a):
    t, x, y, z, tdot, vx, vy, vz = vec

    # --- Position ---
    R2 = x*x + y*y
    term = R2 + z*z - a*a
    r2 = 0.5 * (term + np.sqrt(term*term + 4*a*a*z*z))
    r = np.sqrt(r2)

    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    sinth = np.sin(theta)
    costh = np.cos(theta)
    rho = np.sqrt(r*r + a*a)

    # --- Jacobian ---
    J = np.array([
        [r/rho * sinth * np.cos(phi), rho * costh * np.cos(phi), -rho * sinth * np.sin(phi)],
        [r/rho * sinth * np.sin(phi), rho * costh * np.sin(phi),  rho * sinth * np.cos(phi)],
        [                      costh,                -r * sinth,                        0.0]
    ])

    v_cart = np.array([vx, vy, vz])

    # Solve for (rdot, thetadot, phidot)
    rdot, thetadot, phidot = np.linalg.solve(J, v_cart)

    return np.array([
        t, r, theta, phi,
        tdot, rdot, thetadot, phidot
    ])

def Jfunc(x, r0, e, i, a, E, L, C):
    '''
    Supplementary function for freqs_finder. Separated because it uses recursion to iterate

    Parameters
    ----------
    x : float
        Radius in terms of gravitational radii
    r0 : float
        Semimajor axis corresponding to given orbit, in terms of gravitational radii
    e : float
        Eccentrity of orbit
    i : float
        Inclination of orbit, with pi/2 as equatorial
    a : float
        Dimensionless spin of central black hole
    E : float
        Specific energy of orbit
    L : float
        Specific angular momentum of orbit
    C : float
        Specific Carter constant of orbit

    Returns
    -------
    float
        Intermediate value used for calculating orbital frequencies

    '''
    #E, L, C = schmidtparam3(r0, e, i, a)
    p = r0*(1 - e**2)
    J = lambda x: (1-E**2)*(1-e**2)+2*(1-E**2-1/r0)*(1+e*np.cos(x))+((1-E**2)*(3+e**2)/(1-e**2)-(4/p)+((a**2)*(1-E**2)+L**2+C)*(1/(r0*p)))*((1+e*np.cos(x))**2)
    z1 = 1 + ((1+a)**(1/3) + (1-a)**(1/3))*(1 - a**2)**(1/3)
    z2 = np.sqrt(3*(a**2) + z1**2)
    rms = 3 + z2 - np.sign(a)*np.sqrt((3-z1)*(3 + z1 + 2*z2))
    if J(0) < 0.0 and r0*(1-e) > rms:
        E1, L1, C1 = schmidtparam3(r0/10, e, i, a)
        return Jfunc(x, r0/10, e, i, a, E1, L1, C1)/10
    else:
        return J(x)
    
def freqs_finder(E, L, C, a):
    '''
    Calculates characteristic frequencies of a given orbit

    Parameters
    ----------
    E : float
        Specific energy of orbit
    L : float
        Specific axial angular momentum of orbit
    C : float
        Specific Carter constant of orbit
    a : float
        Dimensionless spin of central black hole

    Returns
    -------
    3-element numpy array
        [Radial frequency, theta frequency, phi frequency] in geometric units

    '''
    B2 = (a**2)*(1 - E**2)
    roots = np.sort_complex(np.roots([B2, 0, -(B2 + C + L**2), 0, C]))
    if len(roots) == 4:
        zm, zp = roots[-2], roots[-1]
    else:
        zm, zp = roots[-1], 1e151
    k, i = (zm**2)/(zp**2), np.arccos(zm)
    
    Rcoeff = np.array([E**2 - 1.0, 2.0, (a**2)*(E**2 - 1.0) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
    ri, ro = np.sort(np.roots(Rcoeff))[-2:]
    if abs(ri.imag) > 1e-8 or abs(ro.imag) > 1e-8:
        return np.full(3, np.nan)
    ri = ri.real
    ro = ro.real

    r0, e = 0.5*(ro + ri), (ro - ri)/(ro + ri)
    p = r0*(1 - e**2)

    J2 = lambda x: np.real(Jfunc(x, r0, e, i, a, E, L, C))
    H = lambda x: np.real(1 - (2/p)*(1 + e*np.cos(x)) + ((a/p)**2)*(1 + e*np.cos(x))**2)
    G = lambda x: np.real(L - 2*(L - a*E)*(1 + e*np.cos(x))/p)
    F = lambda x: np.real(E + ((a/p)**2)*E*((1 + e*np.cos(x))**2) - 2*a*(L - a*E)*((1 + e*np.cos(x))/p)**3)
    
    Xt = integrate.quad(lambda x: 1/(J2(x)**0.5), 0.0, np.pi)[0]
    Yt = integrate.quad(lambda x: (p**2)/(((1+e*np.cos(x))**2)*(J2(x)**0.5)), 0.0, np.pi)[0]
    Zt = integrate.quad(lambda x: G(x)/(H(x)*(J2(x)**0.5)), 0.0, np.pi)[0]
    Wt = integrate.quad(lambda x: (p**2)*F(x)/(((1 + e*np.cos(x))**2)*H(x)*(J2(x)**0.5)), 0.0, np.pi)[0]

    Kk = integrate.quad(lambda p: 1/np.sqrt(1 - k*(np.sin(p)**2)), 0, np.pi/2)[0]
    Ek = integrate.quad(lambda p: np.sqrt(1 - k*(np.sin(p)**2)), 0, np.pi/2)[0]
    Pk = integrate.quad(lambda p: 1/((1-(zm*np.sin(p))**2)*np.sqrt(1 - k*(np.sin(p)**2))), 0, np.pi/2)[0]

    Lam = (Yt + Xt*(a*zp)**2)*Kk - Xt*Ek*(a*zp)**2
    wr, wt, wp = np.pi*p*Kk/((1-e**2)*Lam), np.pi*(B2**0.5)*zp*Xt/(2*Lam), (1/Lam)*((Zt - L*Xt)*Kk + L*Xt*Pk)
    g = (1/Lam)*((Wt + E*Xt*(a*zp)**2)*Kk - E*Xt*Ek*(a*zp)**2)
    wr, wt, wp, g = np.where(np.array([wr, wt, wp, g]).imag < 1e-11, np.array([wr, wt, wp, g]).real, np.array([wr, wt, wp, g]))
    if a == 0.0:
        wt = wp
    return np.array([wr, wt, wp])/g

def freqs_finder2(E, L, C, a):
    '''
    Calculates characteristic frequencies of a given orbit

    Parameters
    ----------
    E : float
        Specific energy of orbit
    L : float
        Specific angular momentum of orbit
    C : float
        Specific Carter constant of orbit
    a : float
        Dimensionless spin of central black hole

    Returns
    -------
    3-element numpy array
        [Radial frequency, theta frequency, phi frequency] in geometric units

    '''
    B2 = (a**2)*(1 - E**2)
    roots = np.round(np.sort(np.roots([B2, 0, -(B2 + C + L**2), 0, C])), 15)
    if len(roots) == 4:
        zm, zp = roots[-2], roots[-1]
    else:
        zm, zp = roots[-1], 1e151
    k, i = (zm**2)/(zp**2), np.arccos(zm)
    
    Rcoeff = np.array([E**2 - 1.0, 2.0, (a**2)*(E**2 - 1.0) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
    ri, ro = np.sort(np.roots(Rcoeff))[-2:]
    if ri.imag/ri.real > 1e-8:
        return np.array([np.nan, np.nan, np.nan])

    r0, e = 0.5*(ro + ri), (ro - ri)/(ro + ri)
    p = r0*(1 - e**2)

    J2 = lambda x: np.real(Jfunc(x, r0, e, i, a, E, L, C))
    H = lambda x: np.real(1 - (2/p)*(1 + e*np.cos(x)) + ((a/p)**2)*(1 + e*np.cos(x))**2)
    G = lambda x: np.real(L - 2*(L - a*E)*(1 + e*np.cos(x))/p)
    F = lambda x: np.real(E + ((a/p)**2)*E*((1 + e*np.cos(x))**2) - 2*a*(L - a*E)*((1 + e*np.cos(x))/p)**3)
    
    Xt = integrate.quad(lambda x: 1/(J2(x)**0.5), 0.0, np.pi)[0]
    Yt = integrate.quad(lambda x: (p**2)/(((1+e*np.cos(x))**2)*(J2(x)**0.5)), 0.0, np.pi)[0]
    Zt = integrate.quad(lambda x: G(x)/(H(x)*(J2(x)**0.5)), 0.0, np.pi)[0]
    Wt = integrate.quad(lambda x: (p**2)*F(x)/(((1 + e*np.cos(x))**2)*H(x)*(J2(x)**0.5)), 0.0, np.pi)[0]

    Kk = integrate.quad(lambda p: 1/np.sqrt(1 - k*(np.sin(p)**2)), 0, np.pi/2)[0]
    Ek = integrate.quad(lambda p: np.sqrt(1 - k*(np.sin(p)**2)), 0, np.pi/2)[0]
    Pk = integrate.quad(lambda p: 1/((1-(zm*np.sin(p))**2)*np.sqrt(1 - k*(np.sin(p)**2))), 0, np.pi/2)[0]

    Lam = (Yt + Xt*(a*zp)**2)*Kk - Xt*Ek*(a*zp)**2
    wr, wt, wp = np.pi*p*Kk/((1-e**2)*Lam), np.pi*(B2**0.5)*zp*Xt/(2*Lam), (1/Lam)*((Zt - L*Xt)*Kk + L*Xt*Pk)
    g = (1/Lam)*((Wt + E*Xt*(a*zp)**2)*Kk - E*Xt*Ek*(a*zp)**2)
    wr, wt, wp, g = np.where(np.array([wr, wt, wp, g]).imag < 1e-11, np.array([wr, wt, wp, g]).real, np.array([wr, wt, wp, g]))
    if a == 0.0:
        wt = wp
    return np.array([g, wr, wt, wp])

def freqs_finder3(a, cons = False, params=False):
    '''
    Calculates characteristic frequencies of a given orbit

    Parameters
    ----------
    a : float
        Dimensionless spin of central black hole

    Returns
    -------
    3-element numpy array
        [Radial frequency, theta frequency, phi frequency] in geometric units

    '''
    if cons:
        E, L, C = cons
    elif params:
        r0, e, inc = params
        E, L, C = schmidtparam3(r0, e, inc, a)
    else:
        print("No input")
        return np.array([0.0, 0.0, 0.0])
    
    B2 = (a**2)*(1 - E**2)
    roots = np.sort_complex(np.roots([B2, 0, -(B2 + C + L**2), 0, C]))
    if len(roots) == 4:
        zm, zp = roots[-2], roots[-1]
    else:
        zm, zp = roots[-1], 1e151
    k, i = (zm**2)/(zp**2), np.arccos(zm)
    
    Rcoeff = np.array([E**2 - 1.0, 2.0, (a**2)*(E**2 - 1.0) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
    ri, ro = np.sort(np.roots(Rcoeff))[-2:]
    if abs(ri.imag) > 1e-8 or abs(ro.imag) > 1e-8:
        return np.full(3, np.nan)
    ri = ri.real
    ro = ro.real

    r0, e = 0.5*(ro + ri), (ro - ri)/(ro + ri)
    p = r0*(1 - e**2)

    J2 = lambda x: np.real(Jfunc(x, r0, e, i, a, E, L, C))
    H = lambda x: np.real(1 - (2/p)*(1 + e*np.cos(x)) + ((a/p)**2)*(1 + e*np.cos(x))**2)
    G = lambda x: np.real(L - 2*(L - a*E)*(1 + e*np.cos(x))/p)
    F = lambda x: np.real(E + ((a/p)**2)*E*((1 + e*np.cos(x))**2) - 2*a*(L - a*E)*((1 + e*np.cos(x))/p)**3)
    
    Xt = integrate.quad(lambda x: 1/(J2(x)**0.5), 0.0, np.pi)[0]
    Yt = integrate.quad(lambda x: (p**2)/(((1+e*np.cos(x))**2)*(J2(x)**0.5)), 0.0, np.pi)[0]
    Zt = integrate.quad(lambda x: G(x)/(H(x)*(J2(x)**0.5)), 0.0, np.pi)[0]
    Wt = integrate.quad(lambda x: (p**2)*F(x)/(((1 + e*np.cos(x))**2)*H(x)*(J2(x)**0.5)), 0.0, np.pi)[0]

    Kk = integrate.quad(lambda p: 1/np.sqrt(1 - k*(np.sin(p)**2)), 0, np.pi/2)[0]
    Ek = integrate.quad(lambda p: np.sqrt(1 - k*(np.sin(p)**2)), 0, np.pi/2)[0]
    Pk = integrate.quad(lambda p: 1/((1-(zm*np.sin(p))**2)*np.sqrt(1 - k*(np.sin(p)**2))), 0, np.pi/2)[0]

    Lam = (Yt + Xt*(a*zp)**2)*Kk - Xt*Ek*(a*zp)**2
    wr, wt, wp = np.pi*p*Kk/((1-e**2)*Lam), np.pi*(B2**0.5)*zp*Xt/(2*Lam), (1/Lam)*((Zt - L*Xt)*Kk + L*Xt*Pk)
    g = (1/Lam)*((Wt + E*Xt*(a*zp)**2)*Kk - E*Xt*Ek*(a*zp)**2)
    wr, wt, wp, g = np.where(np.array([wr, wt, wp, g]).imag < 1e-11, np.array([wr, wt, wp, g]).real, np.array([wr, wt, wp, g]))
    if a == 0.0:
        wt = wp
    return np.array([wr, wt, wp])/g

def root_getter(E, L, C, spin):
    E, L, C = np.fix(1e13*np.array([E, L, C]))*1e-13
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)

            #print("orgio")
            a, b, c, d, e = np.array(np.fix(1e12*np.array([(E**2 - 1.0), 2.0, ((spin**2)*(E**2 - 1.0) - L**2 - C),  (2*((L - spin*E)**2) + 2*C), -(spin**2)*C]))*1e-12).astype(complex)
            #print(E, L, C, spin, L**2 + C)
            #print(a, b, c, d, e)
            p1 = 2*(c**3) - 9*b*c*d + 27*a*(d**2) + 27*(b**2)*e - 72*a*c*e
            p2 = p1 + (-4*(c**2 - 3*b*d + 12*a*e)**3 + p1**2)**0.5
            p3 = (c**2 - 3*b*d + 12*a*e)/(3*a*((0.5*p2)**(1/3))) + ((0.5*p2)**(1/3))/(3*a)
            p4 = ((b**2)/(4*(a**2)) - (2*c)/(3*a) + p3)**(0.5)
            p5 = (b**2)/(2*(a**2)) - (4*c)/(3*a) - p3
            p6 = (-(b**3)/(a**3) + (4*b*c)/(a**2) - 8*d/a)/(4*p4)
            x1 = -b/(4*a) - p4/2 - 0.5*((p5 - p6)**0.5)
            x2 = -b/(4*a) - p4/2 + 0.5*((p5 - p6)**0.5)
            x3 = -b/(4*a) + p4/2 - 0.5*((p5 + p6)**0.5)
            x4 = -b/(4*a) + p4/2 + 0.5*((p5 + p6)**0.5) 
            turns = np.array([np.real(num) if np.abs(np.imag(num)) < 1e-8 else num for num in [x1, x2, x3, x4]])
            #print("HELLO")
    except RuntimeWarning:
        # Default to this if it freaked out
        turns = np.roots([a, b, c, d, e])
        turns = np.array([np.real(num) if np.abs(np.imag(num)) < 1e-8 else num for num in turns])
    
    flats = np.roots(np.polyder([a,b,c,d,e])) #np.roots([4*a,3*b,2*c,d])
    flats = np.array([np.real(num) if np.abs(np.imag(num)) < 1e-8 else num for num in flats])
    
    #print(np.array([(a**2)*(1 - E**2), 0.0, -(C + (a**2)*(1 - E**2) + L**2), 0.0, C]))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)

            #print("orgio")
            a, b, c, d, e = np.array(np.fix(1e13*np.array([(a**2)*(1 - E**2), 0.0, -(C + (a**2)*(1 - E**2) + L**2), 0.0, C]).real)*1e-13).astype(complex)
            p1 = 2*(c**3) - 9*b*c*d + 27*a*(d**2) + 27*(b**2)*e - 72*a*c*e
            p2 = p1 + (-4*(c**2 - 3*b*d + 12*a*e)**3 + p1**2)**0.5
            p3 = (c**2 - 3*b*d + 12*a*e)/(3*a*((0.5*p2)**(1/3))) + ((0.5*p2)**(1/3))/(3*a)
            p4 = ((b**2)/(4*(a**2)) - (2*c)/(3*a) + p3)**(0.5)
            p5 = (b**2)/(2*(a**2)) - (4*c)/(3*a) - p3
            p6 = (-(b**3)/(a**3) + (4*b*c)/(a**2) - 8*d/a)/(4*p4)
            x1 = -b/(4*a) - p4/2 - 0.5*((p5 - p6)**0.5)
            x2 = -b/(4*a) - p4/2 + 0.5*((p5 - p6)**0.5)
            x3 = -b/(4*a) + p4/2 - 0.5*((p5 + p6)**0.5)
            x4 = -b/(4*a) + p4/2 + 0.5*((p5 + p6)**0.5) 
            zs = np.array([np.real(num) if np.abs(np.imag(num)) < 1e-5 else num for num in [x1, x2, x3, x4]])
            #print("SIR")
    except RuntimeWarning:
        # Default to this if it freaked out
        zs = np.roots([a, b, c, d, e])
        zs = np.array([np.real(num) if np.abs(np.imag(num)) < 1e-8 else num for num in zs])
    
    return np.round(np.sort(turns), 10), np.round(np.sort(flats), 10), np.round(np.sort(zs), 10)

def root_getter_vec(E, L, C, spin):
    a, b, c, d, e = np.array([(E**2 - 1.0), 2.0*np.ones_like(E), ((spin**2)*(E**2 - 1.0) - L**2 - C),  (2*((L - spin*E)**2) + 2*C), -(spin**2)*C*np.ones_like(E)]).astype(complex)
    p1 = 2*(c**3) - 9*b*c*d + 27*a*(d**2) + 27*(b**2)*e - 72*a*c*e
    p2 = p1 + (-4*(c**2 - 3*b*d + 12*a*e)**3 + p1**2)**0.5
    p3 = (c**2 - 3*b*d + 12*a*e)/(3*a*((0.5*p2)**(1/3))) + ((0.5*p2)**(1/3))/(3*a)
    p4 = ((b**2)/(4*(a**2)) - (2*c)/(3*a) + p3)**(0.5)
    p5 = (b**2)/(2*(a**2)) - (4*c)/(3*a) - p3
    p6 = (-(b**3)/(a**3) + (4*b*c)/(a**2) - 8*d/a)/(4*p4)
    x1 = -b/(4*a) - p4/2 - 0.5*((p5 - p6)**0.5)
    x2 = -b/(4*a) - p4/2 + 0.5*((p5 - p6)**0.5)
    x3 = -b/(4*a) + p4/2 - 0.5*((p5 + p6)**0.5)
    x4 = -b/(4*a) + p4/2 + 0.5*((p5 + p6)**0.5)
    turns = np.array([x1, x2, x3, x4])
    turns = np.transpose(np.where(np.abs(turns.imag) < 1e-8, turns.real, turns))

    flats = np.array([np.roots(group) for group in np.transpose([4*a,3*b,2*c,d])])    
    flats = np.where(np.abs(flats.imag) < 1e-8, flats.real, flats) 
    
    a, b, c, d, e = np.array([(a**2)*(1 - E**2), np.zeros_like(E), -(C + (a**2)*(1 - E**2) + L**2), np.zeros_like(E), C]).astype(complex)
    p1 = 2*(c**3) - 9*b*c*d + 27*a*(d**2) + 27*(b**2)*e - 72*a*c*e
    p2 = p1 + (-4*(c**2 - 3*b*d + 12*a*e)**3 + p1**2)**0.5
    p3 = (c**2 - 3*b*d + 12*a*e)/(3*a*((0.5*p2)**(1/3))) + ((0.5*p2)**(1/3))/(3*a)
    p4 = ((b**2)/(4*(a**2)) - (2*c)/(3*a) + p3)**(0.5)
    p5 = (b**2)/(2*(a**2)) - (4*c)/(3*a) - p3
    p6 = (-(b**3)/(a**3) + (4*b*c)/(a**2) - 8*d/a)/(4*p4)
    x1 = -b/(4*a) - p4/2 - 0.5*((p5 - p6)**0.5)
    x2 = -b/(4*a) - p4/2 + 0.5*((p5 - p6)**0.5)
    x3 = -b/(4*a) + p4/2 - 0.5*((p5 + p6)**0.5)
    x4 = -b/(4*a) + p4/2 + 0.5*((p5 + p6)**0.5) 
    zs = np.array([x1, x2, x3, x4])
    zs = np.transpose(np.where(np.abs(zs.imag) < 1e-5, zs.real, zs))
    
    return np.round(np.sort(turns), 10), np.round(np.sort(flats), 10), np.round(np.sort(zs), 10)

def sig_q(r, a, inc):
    return r**2 + (a*np.cos(inc))**2, np.sqrt(r**2 - (a*np.cos(inc))**2)

def circE_inc(r, a, inc):
    p = 1 if a >= 0 else -1
    a = abs(a)
    sig, q = sig_q(r, a, inc)
    denom = np.sqrt(1 - 3*r/sig + p*2*a*q*np.sin(inc)/(sig*np.sqrt(r)) + ((a*np.cos(inc))**2)/(sig*r))
    numer = 1 - 2*r/sig + p*a*q*np.sin(inc)/(sig*np.sqrt(r))
    return numer / denom

def circL_inc(r, a, inc):
    p = 1 if a >= 0 else -1
    a = abs(a)
    sig, q = sig_q(r, a, inc)
    denom = np.sqrt(1 - 3*r/sig + p*2*a*q*np.sin(inc)/(sig*np.sqrt(r)) + ((a*np.cos(inc))**2)/(sig*r))
    numer = p*q*(r**2 + a**2)*np.sin(inc)/(sig*np.sqrt(r)) - 2*a*r*(np.sin(inc)**2)/sig
    return numer / denom

def circC_inc(r, a, inc):
    E, L = circE_inc(r, a, inc), circL_inc(r, a, inc)
    tri = r**2 - 2*r + a**2
    C = (((r**2 + a**2)*E - a*L)**2)/tri - r**2 - (L - a*E)**2
    return C

def Yfunc(r, a, C):
    #From Teo 2021
    return r**5 - C*(r - 3)*r**3 + (a*C)**2

def circE_C(r, a, C, p=1):
    #From Teo 2021
    p = float(np.sign(p))**np.sign(p)
    Y = Yfunc(r, a, C)
    sqrt_Y = np.sqrt(Y)
    denom = (r**2) * np.sqrt(r**3*(r - 3) - 2*a*(a*C - p*sqrt_Y))
    numer = r**3*(r - 2) - a*(a*C - p*sqrt_Y)
    return numer / denom

def circL_C(r, a, C, p=1):
    #From Teo 2021
    p = float(np.sign(p))**np.sign(p)
    Y = Yfunc(r, a, C)
    sqrt_Y = np.sqrt(Y)
    denom = (r**2) * np.sqrt(r**3*(r - 3) - 2*a*(a*C - p*sqrt_Y))
    numer = -2*a*r**3 - (r**2 + a**2)*(a*C - p*sqrt_Y)
    return numer / denom

def circE_L_C(r, a, C, p=1):
    #From Teo 2021
    """
    Vectorized: returns tuple (E, L) for arrays r.
    Assumes r is numpy array (or scalar).
    p should be +1 or -1 (or a sign-like value).
    """
    p = 1.0 if p >= 0 else -1.0
    Y = r**5 - C*(r - 3)*r**3 + (a*C)**2         # same as Yfunc
    sqrt_Y = np.sqrt(Y)
    inner = r**3*(r - 3) - 2*a*(a*C - p*sqrt_Y)
    denom = (r**2) * np.sqrt(inner)
    numerE = r**3*(r - 2) - a*(a*C - p*sqrt_Y)
    E = numerE / denom
    numerL = -2*a*r**3 - (r**2 + a**2)*(a*C - p*sqrt_Y)
    L = numerL / denom
    return E, L

def get_EL_curve(a, C, rmax=100.0, npts=1000):
    r1min = find_rmb(a)*0.5        # Marginally bound orbit, smallest possible periapse for an equatorial orbit
    r1_isco = find_rms(a)      # Innermost stable circular orbit - NOT the same as innermost stable spherical orbit
    r2min = find_rmb(-a)*0.5        # Marginally bound orbit, smallest possible periapse for an equatorial orbit
    r2_isco = find_rms(-a)      # Innermost stable circular orbit - NOT the same as innermost stable spherical orbit

    # Sample more densely near ISCO
    n_half = npts // 2
    r1_inner = np.geomspace(r1min, r1_isco, n_half)
    r1_outer = np.geomspace(r1_isco, rmax, npts - n_half)
    r2_inner = np.geomspace(r2min, r2_isco, n_half)
    r2_outer = np.geomspace(r2_isco, rmax, npts - n_half)

    r = np.array([np.concatenate([r1_inner, r1_outer]), np.concatenate([r2_inner, r2_outer])])

    E_1, E_2 = circE_C(r[0], a, C), circE_C(r[1], a, C, -1)
    L_1, L_2 = circL_C(r[0], a, C), circL_C(r[1], a, C, -1)
    mask1, mask2 = (0 <= E_1) & (E_1 <= 1), (0 <= E_2) & (E_2 <= 1)
    E_1, E_2 = E_1[mask1], E_2[mask2]
    L_1, L_2 = L_1[mask1], L_2[mask2]
    return E_1, L_1, E_2, L_2

def is_in_ELC_region(E_test, L_test, C_test, a, tol=1e-4, rmax=1000):
    
    #Starter radii, roughly covers the ranges of viable extremal orbits (psuedo-circular or marginally bound) where branch splits occur
    #First half is for upper branch, second half is for lower branch
    rys = [np.linspace(find_rph(a), 6, 50000), 
           np.linspace(max(12, C_test)/2, max(12, C_test), 50000),
           np.linspace(find_rph(-a), 6, 50000), 
           np.linspace(max(12, C_test)/2, max(12, C_test), 50000)]
    #Get the energies and angular momenta associated with these radii
    getEs, getLs = [], []
    for i in range(4):
        p = (-1)**(i//2)
        E_arr, L_arr = circE_L_C(rys[i], a, C_test, p=p)
        getEs.append(E_arr)
        getLs.append(L_arr)
    getEs = np.array(getEs)  
    getLs = np.array(getLs)  #We need these for later

    #Get the minima of each set of energies - they will either correspond to the actual minima OR the split point
    Emins, Emins_ix = np.nanmin(getEs, axis=1), np.nanargmin(getEs, axis=1)
    if E_test < np.nanmin(Emins):
        #print("E is too low!")
        return False
    #Get the radii associated with THOSE energies (lots of back and forth, can I fix that?)
    key_radii = [rys[i][Emins_ix[i]] for i in range(4)]

    r_mb = max(key_radii[0], key_radii[2])      # Split point for marginally bound curve
    r_cir = min(key_radii[1], key_radii[3])     # Split point for psuedo-circular curve
    #When C >= 12, the upper and lower branches connect, r_mb != r_cir, and the values in between are not viable extremal orbits
    #When C < 12, the upper and lower branches DON'T connect, r_mb = r_cir
    
    # Sample more densely near split points
    npts = 50000
    n_half = npts // 2
    r1_inner = r_mb - np.geomspace(find_rph(a), r_mb, n_half) + find_rph(a)         # Start from upper branch photon orbit, go to mb split
    r1_outer = np.geomspace(r_cir, rmax, npts - n_half)                             # Start psuedo-circular split, go to upper bound
    r2_inner = r_mb - np.geomspace(find_rph(-a), r_mb, n_half) + find_rph(-a)       # Start from lower branch photon orbit, go to mb split
    r2_outer = np.geomspace(r_cir, rmax, npts - n_half)                             # Start psuedo-circular split, go to upper bound
    #The 'outer' ranges are identical, and for C >= 16 the 'inner' ranges all correspond to E > 1, but!! I don't care!! It's gonna work for all cases so there
    r = np.array([r1_inner, r1_outer, r2_inner, r2_outer])
    #Get the energies and angular momenta
    Es1, Ls1 = [], []
    for i in range(4):
        p = (-1)**(i//2)
        E_arr, L_arr = circE_L_C(r[i], a, C_test, p=p)
        Es1.append(E_arr)
        Ls1.append(L_arr)
    Es1 = np.array(Es1)
    Ls1 = np.array(Ls1)

    # For some values of C, a, we need to move values from the upper branch to the lower branch or vice versa so that we
        # can have nicely defined upper and lower bounds for later
    #Grab the angular momenta associated with the actual minimum energy points, so our splits are more symmetrical
    Lturn = [np.concatenate((getLs[0], getLs[2]))[np.nanargmin(np.concatenate((getEs[0], getEs[2])))],    # marginally bound curve
              np.concatenate((getLs[1], getLs[3]))[np.nanargmin(np.concatenate((getEs[1], getEs[3])))]]    # psuedo-circular curve 
    #Reorganize Es1 and Ls1 to move the values
    Es, Ls = [[],[],[],[]], [[],[],[],[]]
    for i in range(2): #inner, then outer
        great_mask = Ls1[0 + i] >= Lturn[i], Ls1[2 + i] >= Lturn[i]
        less_mask = Ls1[0 + i] < Lturn[i], Ls1[2 + i] < Lturn[i]
        Es[0 + i] = np.concatenate((Es1[0 + i][great_mask[0]], Es1[2 + i][great_mask[1]]))
        Es[2 + i] = np.concatenate((Es1[0 + i][less_mask[0]], Es1[2 + i][less_mask[1]]))
        Ls[0 + i] = np.concatenate((Ls1[0 + i][great_mask[0]], Ls1[2 + i][great_mask[1]]))
        Ls[2 + i] = np.concatenate((Ls1[0 + i][less_mask[0]], Ls1[2 + i][less_mask[1]]))

    #If for whatever reason there are no energies, just break everything I guess
    if np.all(Es == None):
        #print("E_curve is None")
        return False

    # Only consider points where |E - E_test| < tol
    close_indices = [np.abs(E_group - E_test) <= tol for E_group in Es]
    if sum(close_indices[0]) + sum(close_indices[1]) < 2 and sum(close_indices[2]) + sum(close_indices[3]) < 2:
        #print("close_indices too small")
        return False  # Either no valid E or not enough to get bounds
    #Get corresponding Ls
    Ls_at_E = [Ls[i][close_indices[i]] for i in range(4)]

    if len(Ls_at_E[0])*len(Ls_at_E[2]) == 0:  #If E = E_test does not intersect the marginally bound curve
        Lmax, Lmin = np.mean(Ls_at_E[1]), np.mean(Ls_at_E[3])
        val = Lmin <= L_test <= Lmax
    else:
        Lmin, Lmax = [np.mean(Ls_at_E[0]), np.mean(Ls_at_E[3])], [np.mean(Ls_at_E[1]), np.mean(Ls_at_E[2])]
        val = (Lmin[0] <= L_test <= Lmax[0]) or (Lmin[1] <= L_test <= Lmax[1])

    #for i in range(4):
    #    plt.scatter(Es[i][close_indices[i]], Ls[i][close_indices[i]])
    #plt.scatter(E_test, L_test, marker="x")

    # Also ensure E < 1 for bound orbits
    val = val and (E_test < 1)

    return val 

def get_sep_inc(a, inc, mult=1):
    flip = False
    if a < 0:
        flip = True
        a *= -1
        inc *= -1
    mult = max(1, int(mult))
    x = np.sin(inc)
    a2, x2 = a**2, x**2
    r = np.geomspace(2*(1 + np.cos((2/3)*np.arccos(-a))), 10, 50*mult)
    sqrt_r = np.sqrt(r)
    D = r*r - 2*r + a2
    Lamb = np.sqrt(r*r - a2*(1 - x2)) - a*x*sqrt_r
    O = Lamb**4 - a2*(1 - x2)*D*D
    sqrt_O = np.sqrt(O)
    denom = (r*r + a2*(1 - x2))*(-Lamb**2 + sqrt_O + r*D)
    e_sep = ((3*r*r - a2*(1 - x2))*(Lamb**2) + (r*r + a2*(1 - x2))*(sqrt_O - r*D))/denom
    p_sep = 2*r*((r*r - a2*(1 - x2))*(Lamb**2) + (r*r + a2*(1 - x2))*sqrt_O)/denom
    high  = r[0 >= e_sep]
    low = r[1 <= e_sep]

    count = 0
    while (len(high) > 1 or len(low) > 1) and count < 20:
        min_r = max(low) if len(low) else min(r)
        max_r = min(high) if len(high) else max(r)
        r = np.geomspace(min_r, max_r, 50*mult)
        sqrt_r = np.sqrt(r)
        D = r*r - 2*r + a2
        Lamb = np.sqrt(r*r - a2*(1 - x2)) - a*x*sqrt_r
        O = Lamb**4 - a2*(1 - x2)*D*D
        sqrt_O = np.sqrt(O)
        denom = (r*r + a2*(1 - x2))*(-Lamb**2 + sqrt_O + r*D)
        e_sep = ((3*r*r - a2*(1 - x2))*(Lamb**2) + (r*r + a2*(1 - x2))*(sqrt_O - r*D))/denom
        p_sep = 2*r*((r*r - a2*(1 - x2))*(Lamb**2) + (r*r + a2*(1 - x2))*sqrt_O)/denom
        high  = r[0 >= e_sep]
        low = r[1 <= e_sep]
        count += 1

    r2 = r*r
    Y = r*(r2 - a2*(1 - x2))
    G = r*(r2 + a2*(1 - x2))*(r2*(r - 3) + a2*(r + 1)*(1 - x2) + 2*a*x*np.sqrt(Y))
    E_sep = (r2*(r - 2) + a2*r*(1 - x2) + a*x*np.sqrt(Y))/np.sqrt(G)
    L_sep = (-2*a*r2*x2 + (r2 + a2)*x*np.sqrt(Y))/np.sqrt(G)

    mask = (E_sep <= 1.0)*(e_sep >= 0.0) 
    p_sep = p_sep[mask]
    e_sep = e_sep[mask]
    L_sep = L_sep[mask]
    r     =     r[mask]
    r2 = r*r
    E_sep = E_sep[mask]

    r_out = np.geomspace(max(r), 100, len(r))
    r_out2 = r_out*r_out
    sig, q = r_out2 + a2*(1 - x2), np.sqrt(r_out2 - a2*(1 - x2))
    p = 1 if a >= 0 else -1
    A = abs(a)
    # E
    denom = np.sqrt(1 - 3*r_out/sig + p*2*A*q*x/(sig*np.sqrt(r_out)) + a2*(1 - x2)/(sig*r_out))
    numer_E = 1 - 2*r_out/sig + p*A*q*x/(sig*np.sqrt(r_out))
    numer_L = p*q*(r_out2 + a2)*x/(sig*np.sqrt(r_out)) - 2*A*r_out*x2/sig
    E_out, L_out = numer_E/denom, numer_L/denom
    tri_sep = r2 - 2*r + a2
    tri_out = r_out2 - 2*r_out + a2
    if np.isclose(x, 0.0, atol=1e-13):
        C_sep = (1/tri_sep)*((r2 + a2)*E_sep)**2 - r2 - a2*(E_sep**2)
        C_out = (1/tri_out)*((r_out2 + a2)*E_out)**2 - r_out2 - a2*(E_out**2)
    else:
        C_sep = (1 - x2)*(a*(1 - E_sep**2) + (L_sep**2)/x2)
        C_out = (1 - x2)*(a*(1 - E_out**2) + (L_out**2)/x2)

    if flip:
        L_sep *= -1
        L_out *= -1
    
    return {"p": np.concatenate((p_sep, r_out)),
            "e": np.concatenate((e_sep, 0 * e_sep)),
            "it": np.concatenate((r, r_out)), 
            "ot": np.concatenate((p_sep / (1 - e_sep), r_out)), 
            "energy": np.concatenate((E_sep, E_out)),
            "phi_momentum": np.concatenate((L_sep, L_out)), 
            "carter": np.concatenate((C_sep, C_out))}

def get_sep_cosi(a, cosi, mult=1):
    if np.isclose(cosi, 0.0, atol=1e-12):
        # for small cosi it should be basically the same as inc, even with spin
        return get_sep_inc(a, cosi, mult=mult)
    
    flip = False
    if a < 0:
        flip = True
        a *= -1
        cosi *= -1
    
    mult = max(1, int(mult))
    cosi2 = cosi**2
    sini2, sini = 1 - cosi2, np.sqrt(1 - cosi2)
    a2 = a*a
    r = np.geomspace(2*(1 + np.cos((2/3)*np.arccos(-a))), 10, 50*mult)
    r2 = r*r

    D = r2 - 2*r + a2
    Xi = r2*r2 + 2*a2*r2 - 4*a2*r + a2*a2
    Lamb = (r2 - a2)*np.sqrt(r2*r2 + (2*r2 - 4*r + a2)*a2*sini2) - np.sqrt(r*r2)*D*a*cosi
    O = (r2 - a2)*((r2 - a2)*(Lamb**4) + (4*r*(Lamb**2) - (r2 - a2*sini2)*Xi*Xi)*a2*sini2*Xi*Xi)
    sqrt_O = np.sqrt(O)
    denom = -(3*r2 + a2)*(Lamb**2) + sqrt_O + r*(r2 - a2*sini2)*Xi*Xi
    e_sep = ((5*r2 - a2)*(Lamb**2) + sqrt_O - r*(r2 - a2*sini2)*Xi*Xi)/denom  # WHACHAA
    p_sep = 2*r*((r2 - a2)*(Lamb**2) + sqrt_O)/denom
    high  = r[0 >= e_sep]
    low = r[1 <= e_sep]

    count = 0
    while (len(high) > 1 or len(low) > 1) and count < 20:
        min_r = max(low) if len(low) else min(r)
        max_r = min(high) if len(high) else max(r)
        r = np.geomspace(min_r, max_r, 50*mult)
        r2 = r*r
        D = r2 - 2*r + a2
        Xi = r2*r2 + 2*a2*r2 - 4*a2*r + a2*a2
        Lamb = (r2 - a2)*np.sqrt(r2*r2 + (2*r2 - 4*r + a2)*a2*sini2) - np.sqrt(r*r2)*D*a*cosi
        O = (r2 - a2)*((r2 - a2)*(Lamb**4) + (4*r*(Lamb**2) - (r2 - a2*sini2)*Xi*Xi)*a2*sini2*Xi*Xi)
        sqrt_O = np.sqrt(O)
        denom = -(3*r2 + a2)*(Lamb**2) + sqrt_O + r*(r2 - a2*sini2)*Xi*Xi
        e_sep = ((5*r2 - a2)*(Lamb**2) + sqrt_O - r*(r2 - a2*sini2)*Xi*Xi)/denom
        p_sep = 2*r*((r2 - a2)*(Lamb**2) + sqrt_O)/denom
        high  = r[0 >= e_sep]
        low = r[1 <= e_sep]
        count += 1

    r2 = r*r
    Y = r*(r2*r2 + (2*r2 - 4*r + a2)*a2*sini2)
    G = (r2*(r*r2 - 3*r2 - 2*a2) + (2*r*r2 + a2*r + a2)*a2*sini2)*Xi + 2*r*(3*r2 + a2)*(r*(3*r2 - 4*r + a2)*a*cosi + D*np.sqrt(Y))*a*cosi
    E_sep = (r*(3*r2 - 4*r + a2)*a*cosi + D*np.sqrt(Y))/np.sqrt(G)
    L_sep = r*cosi*Xi/np.sqrt(G)

    mask = (E_sep <= 1.0)*(e_sep >= 0.0) 
    p_sep = p_sep[mask]
    e_sep = e_sep[mask]
    L_sep = L_sep[mask]
    r     =     r[mask]
    E_sep = E_sep[mask]
    C_sep = (L_sep**2)*sini2/cosi2

    r_out = np.geomspace(max(r), 100, len(r))
    r_out2 = r_out*r_out
    f1, f2 = r_out2*r_out2 + a2*r_out2 + 2*a2*r_out, 4*r_out2*r_out + 2*a2*r_out + 2*a2
    g1, g2 = 2*a*r_out, 2*a
    h1, h2 = (1/cosi2)*(r_out2 - 2*r_out + a2*sini2), (1/cosi2)*(2*r_out - 2)
    d1, d2 = r_out2*r_out2 - 2*r_out2*r_out + a2*r_out2, 4*r_out2*r_out - 6*r_out2 + 2*a2*r_out
    A, B, C, D_ = f1 - f2, g1 - g2, h1 - h2, d1 - d2
    alpha, beta, gamma = D_*f1 - d1*A, -2*(D_*g1 - d1*B), -(D_*h1 - d1*C)
    k = (- beta - np.sign(cosi)*np.sqrt(beta**2 - 4*alpha*gamma))/(2 * alpha)
    L_out = np.sqrt(d1/(f1*k*k - 2*g1*k - h1))
    E_out = np.sign(cosi)*k*L_out
    L_out *= np.sign(cosi)
    C_out = (L_out**2)*sini2/cosi2

    if flip:
        L_sep *= -1
        L_out *= -1
    
    return {"p": np.concatenate((p_sep, r_out)),
            "e": np.concatenate((e_sep, 0 * e_sep)),
            "it": np.concatenate((r, r_out)), 
            "ot": np.concatenate((p_sep / (1 - e_sep), r_out)), 
            "energy": np.concatenate((E_sep, E_out)),
            "phi_momentum": np.concatenate((L_sep, L_out)), 
            "carter": np.concatenate((C_sep, C_out))}

def peters_sim_EL(a, q, cons=False, params=False, endflag="False"):
    '''
    LESS DUMB BUT A LITTLE DUMB
    Reproduce 2-body system results from Peters 1964
    
    :param a: Black Hole Spin - between -1 and 1 (Peters assumes spin=0)
    :param q: Mass Ratio - <= 10^-4 for EMRIs
    :param cons: Orbital Constants (optional) - [Energy, Axial Angular Momentum, Carter Constant] per unit mass
    :param params: Orbital Parameters (optional) - [Semimajor Axis, Eccentricity, Inclination]
    :param endflag: End state of orbit (optional) - Boolean statement to end simulation at a certain condition.
                                                    Defaults to False, so the sim ends at plunge
    '''
    # Use Orbital Constants
    if cons != False:
        E, L, C = cons
        turns, flats, zs = root_getter(E, L, C, a)
        p, e = 2*turns[-1]*turns[-2]/(turns[-1] + turns[-2]), (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
        inc = np.arccos(min(1, np.mean(np.abs(zs[1:3]))))
        r0 = p/(1 - e**2)
    # Use Orbital Parameters
    elif params != False:
        r0, e, inc = params
        p = r0*(1 - e**2)
        E, L, C = schmidtparam3(*params, a)
        turns, flats, zs = root_getter(E, L, C, a)
    # Nuffin
    else:
        print("No starting input.")
        return 0
    
    # E and L as defined by Peters, not the same as the calculated ones
    Ep, Lp = -q/(2*r0), np.sqrt(q*q*r0*(1 - e*e)/(1 + q))
    # Inclination approximation for cosine of inclination
    A = (a**2)*(1 - E**2)
    cosi = L/np.sqrt(C + L**2)
    # Get changes to values as Peters wrote them (G = c = 1)
    dEpdt = lambda maj, ecc: (-32/5)*q*q*(1+q)*(1 + 73*(ecc**2)/24 + 37*(ecc**4)/96)/((maj**5)*((1 - ecc**2)**(7/2)))
    dLpdt = lambda maj, ecc: (-32/5)*q*q*np.sqrt(1+q)*(1 + 7*(ecc**2)/8)/((maj**(7/2))*((1 - ecc**2)**2))

    # Get initial dt - this is supposed to be ~1/100 of the full inspiral according to Peters
    beta = (64/5)*q*(1 + q)
    dt = 0.001*(r0**4)/(4*beta)

    # Initialize final data
    vals = [[0, E, L, C, r0, e, Ep, Lp]]#, inc, cosi]]
    
    # A way to track our progress so we don't get bored out of our minds
    pbar = tqdm(total = 10000, position=0)
    pbar.set_postfix_str(f"Semilat: {r0*(1 - e**2):.4f}, Ecc {e:.4f}, Peri: {r0*(1 - e):.4f}")
    prog = 0
    # Counter for how many times we find complex orbital constants in a row
    wack = 0
    # Continue while periapse is greater than marginally bound radius for a=0 and 
    #  semilatus rectum is not super complex
    while r0*(1 - e) > 4 and np.abs(np.imag(r0*(1 - e*e))/np.real(r0*(1 - e*e))) < 1e-5:
        try:
            # Get changes to values
            dEp = dEpdt(r0, e)*dt
            dLp = dLpdt(r0, e)*dt
            new_r0 = -q/(2 * (Ep + dEp))
            new_e = np.sqrt(1 - ((Lp + dLp)**2)*(1 + q)/(new_r0 * q * q))
            if -(new_r0 - r0)/r0 > 0.25:
                dt *= 0.5
                continue
            # Assume inclination remains constant because a=0
            E, L, C = schmidtparam3(new_r0, new_e, inc, a)
            # If we get complex orbital constants, try taking a smaller step size
            # Use Q = C + (a*E - L)**2 so that equatorial orbits don't make division by zero errors
            tester = np.array([E, L, C + (a*E - L)**2])
            if True in np.abs(np.imag(tester)/np.real(tester)) < 1e-5:
                wack += 1
                if wack < 3:
                    dt *= 0.5
                    continue
                # If you get the complexity issue three times in a row, just stop
                else:
                    break
            else:
                wack = 0

            # Get all your new values    
            # We actuall don't really do anything with Ep or Lp, we just keep track of them
            Ep, Lp, r0, e = Ep + dEp, Lp + dLp, new_r0, new_e
            # Update so that the user knows what's going on
            pbar.set_postfix_str(f"Semilat: {r0*(1 - e**2):.4f}, Ecc {e:.4f}, Peri: {r0*(1 - e):.4f}")
            pbar.update(int(10000*(p - r0*(1 - e**2))/(p - (6 + 2*e))) - prog)
            prog = int(10000*(p - r0*(1 - e**2))/(p - (6 + 2*e)))
            vals.append([vals[-1][0] + dt, E, L, C, r0, e, Ep, Lp])
            # Make the step size a little bigger, just as a treat
            dt *= 1.001
        except KeyboardInterrupt:
            break
    pbar.close()
    vals = np.real(np.array(vals))
    A = (a**2)*(1 - vals[:, 0]**2)
    dct = {"name": "peters1964_EL",
           "tracktime": vals[:, 0],
           "energy": vals[:, 1],
           "phi_momentum": vals[:, 2],
           "carter": vals[:, 3],
           "r0": vals[:, 4],
           "p": vals[:, 4]*(1 - vals[:, 5]**2),
           "e": vals[:, 5],
           "Ep": vals[:, 6],
           "Lp": vals[:, 7],
           "inc": inc*np.ones_like(vals[:,0]),
           "cosi": np.where(A != 0, ((A + vals[:, 2]**2 + vals[:, 3]) - ((A + vals[:, 2]**2 + vals[:, 3])**2 - 4*A*vals[:, 3])**(1/2))/(2*A), vals[:, 2]/np.sqrt(vals[:, 3] + vals[:, 2]**2)),
           "it": vals[:, 4]*(1 - vals[:, 5]),
           "ot": vals[:, 4]*(1 + vals[:, 5]),
           "spin": a}
    return dct

def peters_sim_EL2(a, q, cons=False, params=False, endflag="False"):
    '''
    DUMB
    Reproduce 2-body system results from Peters 1964
    
    :param a: Black Hole Spin - between -1 and 1 (Peters assumes spin=0)
    :param q: Mass Ratio - <= 10^-4 for EMRIs
    :param cons: Orbital Constants (optional) - [Energy, Axial Angular Momentum, Carter Constant] per unit mass
    :param params: Orbital Parameters (optional) - [Semimajor Axis, Eccentricity, Inclination]
    :param endflag: End state of orbit (optional) - Boolean statement to end simulation at a certain condition.
                                                    Defaults to False, so the sim ends at plunge
    '''
    # Use Orbital Constants
    if cons != False:
        E, L, C = cons
        turns, flats, zs = root_getter(E, L, C, a)
        p, e = 2*turns[-1]*turns[-2]/(turns[-1] + turns[-2]), (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
        inc = np.arccos(min(1, np.mean(np.abs(zs[1:3]))))
        r0 = p/(1 - e**2)
    # Use Orbital Parameters
    elif params != False:
        r0, e, inc = params
        p = r0*(1 - e**2)
        E, L, C = schmidtparam3(*params, a)
        turns, flats, zs = root_getter(E, L, C, a)
    # Nuffin
    else:
        print("No starting input.")
        return 0
    
    # E and L as defined by Peters, not the same as the calculated ones
    Ep, Lp = -q/(2*r0), np.sqrt(q*q*r0*(1 - e*e)/(1 + q))
    # Inclination approximation for cosine of inclination
    A = (a**2)*(1 - E**2)
    cosi = L/np.sqrt(C + L**2)
    # Get changes to values as Peters wrote them (G = c = 1)
    dEpdt = lambda maj, ecc: (-32/5)*q*q*(1+q)*(1 + 73*(ecc**2)/24 + 37*(ecc**4)/96)/((maj**5)*((1 - ecc**2)**(7/2)))
    dLpdt = lambda maj, ecc: (-32/5)*q*q*np.sqrt(1+q)*(1 + 7*(ecc**2)/8)/((maj**(7/2))*((1 - ecc**2)**2))

    # Get initial dt - this is supposed to be ~1/100 of the full inspiral according to Peters
    beta = (64/5)*q*(1 + q)
    dt = 0.001*(r0**4)/(4*beta)

    # Initialize final data
    vals = [[0, E, L, C, r0, e, Ep, Lp]]#, inc, cosi]]
    
    # A way to track our progress so we don't get bored out of our minds
    pbar = tqdm(total = 10000, position=0)
    pbar.set_postfix_str(f"Semilat: {r0*(1 - e**2):.4f}, Ecc {e:.4f}, Peri: {r0*(1 - e):.4f}")
    prog = 0
    # Counter for how many times we find complex orbital constants in a row
    wack = 0
    # Continue while periapse is greater than marginally bound radius for a=0 and 
    #  semilatus rectum is not super complex
    while r0*(1 - e) > 4 and np.abs(np.imag(r0*(1 - e*e))/np.real(r0*(1 - e*e))) < 1e-5:
        try:
            # Get changes to values
            dEp = dEpdt(r0, e)*dt
            dLp = dLpdt(r0, e)*dt
            dr0 = (q/2)*(1/(Ep*Ep))*dEp
            de = (Lp*Lp/(2*r0*r0))*((1+q)/q)*(q/2)*(1/(Ep*Ep))*dEp/np.sqrt(1 - (Lp*Lp/r0)*((1+q)/q)) - (Lp/r0)*((1+q)/q)*dLp/np.sqrt(1 - (Lp*Lp/r0)*((1+q)/q))
            if -dr0/r0 > 0.25:
                dt *= 0.5
                continue
            # Assume inclination remains constant because a=0
            E, L, C = schmidtparam3(r0 + dr0, e + de, inc, a)
            # If we get complex orbital constants, try taking a smaller step size
            # Use Q = C + (a*E - L)**2 so that equatorial orbits don't make division by zero errors
            tester = np.array([E, L, C + (a*E - L)**2])
            if True in np.abs(np.imag(tester)/np.real(tester)) < 1e-5:
                wack += 1
                if wack < 3:
                    dt *= 0.5
                    continue
                # If you get the complexity issue three times in a row, just stop
                else:
                    break
            else:
                wack = 0

            # Get all your new values    
            # We actuall don't really do anything with Ep or Lp, we just keep track of them
            Ep, Lp, r0, e = Ep + dEp, Lp + dLp, r0 + dr0, e + de
            # Update so that the user knows what's going on
            pbar.set_postfix_str(f"Semilat: {r0*(1 - e**2):.4f}, Ecc {e:.4f}, Peri: {r0*(1 - e):.4f}")
            pbar.update(int(10000*(p - r0*(1 - e**2))/(p - (6 + 2*e))) - prog)
            prog = int(10000*(p - r0*(1 - e**2))/(p - (6 + 2*e)))
            vals.append([vals[-1][0] + dt, E, L, C, r0, e, Ep, Lp])
            # Make the step size a little bigger, just as a treat
            dt *= 1.001
        except KeyboardInterrupt:
            break
    pbar.close()
    vals = np.real(np.array(vals))
    A = (a**2)*(1 - vals[:, 0]**2)
    dct = {"name": "peters1964_EL2",
           "tracktime": vals[:, 0],
           "energy": vals[:, 1],
           "phi_momentum": vals[:, 2],
           "carter": vals[:, 3],
           "r0": vals[:, 4],
           "p": vals[:, 4]*(1 - vals[:, 5]**2),
           "e": vals[:, 5],
           "Ep": vals[:, 6],
           "Lp": vals[:, 7],
           "inc": inc*np.ones_like(vals[:,0]),
           "cosi": np.where(A != 0, ((A + vals[:, 2]**2 + vals[:, 3]) - ((A + vals[:, 2]**2 + vals[:, 3])**2 - 4*A*vals[:, 3])**(1/2))/(2*A), vals[:, 2]/np.sqrt(vals[:, 3] + vals[:, 2]**2)),
           "it": vals[:, 4]*(1 - vals[:, 5]),
           "ot": vals[:, 4]*(1 + vals[:, 5]),
           "spin": a}
    return dct

def peters_sim_re(a, q, cons=False, params=False, endflag="False"):
    '''
    FOR COOL KIDS
    Reproduce 2-body system results from Peters 1964
    
    :param a: Black Hole Spin - between -1 and 1 (Peters assumes spin=0)
    :param q: Mass Ratio - <= 10^-4 for EMRIs
    :param cons: Orbital Constants (optional) - [Energy, Axial Angular Momentum, Carter Constant] per unit mass
    :param params: Orbital Parameters (optional) - [Semimajor Axis, Eccentricity, Inclination]
    :param endflag: End state of orbit (optional) - Boolean statement to end simulation at a certain condition.
                                                    Defaults to False, so the sim ends at plunge
    '''
    # Use Orbital Constants
    G = 6.67e-11
    c = 3e8
    M = 1.989e37
    m = q*M
    if cons != False:
        E, L, C = cons
        turns, flats, zs = root_getter(E, L, C, a)
        p, e = 2*turns[-1]*turns[-2]/(turns[-1] + turns[-2]), (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
        inc = np.sign(L)*np.arccos(min(1, np.mean(np.abs(zs[1:3]))))
        r0 = p/(1 - e**2)
    # Use Orbital Parameters
    elif params != False:
        print(params)
        r0, e, inc = params
        print(r0, e, inc)
        p = r0*(1 - e**2)
        E, L, C = schmidtparam3(*params, a)
        turns, flats, zs = root_getter(E, L, C, a)
    # Nuffin
    else:
        print("No starting input.")
        return 0
    time_mult, dist_mult = G*M/(c*c*c), G*M/(c*c)
    r0 *= dist_mult
    p *= dist_mult
    turns *= dist_mult
    flats *= dist_mult

    # E and L as defined by Peters, not the same as the calculated ones
    Ep, Lp = -G*m*M/(2*r0), np.sqrt(G*m*m*M*M*a*(1 - e*e)/(m + M))
    # Inclination approximation for cosine of inclination
    #A = (a**2)*(1 - E**2)
    cosi =  L/np.sqrt(C + L**2)
    # Get changes to values as Peters wrote them (G = c = 1)
    dr0dt = lambda maj, ecc: (-64/5)*(G**3)*m*M*(m+M)*(1 + 73*ecc*ecc/24 + 37*(ecc**4)/96)/((c**5)*(maj**3)*((1 - ecc*ecc)**3.5))
    dedt = lambda maj, ecc: (-304/15)*ecc*(G**3)*m*M*(m+M)*(1 + 121*ecc*ecc/304)/((c**5)*(maj**4)*((1 - ecc*ecc)**2.5))

    # Get initial dt - this is supposed to be ~1/100 of the full inspiral according to Peters
    beta = (64/5)*G*G*G*m*M*(m+M)/(c**5)
    #dt = r0**0.5 #0.000001*((dist_mult*r0)**4)/(4*beta)
    #dt = G*M/(c*c*c)
    #dt = np.sqrt((p**1.5)/(G*M))
    B = (64/5) * 1 * q * (1 + q) 
    c0 = (p/(e**(12/19)))/((1 + (121/304)*(e*e))**(870/2299))
    circ_dt = (c0**4)*(e**(48/19))/(4*B)
    ecc_dt = (768/425)*((1-e**2)**3.5)*(r0**4)/(4*B)
    max_dt = time_mult * circ_dt/10
    #dt = time_mult * circ_dt/100000
    dt = circ_dt/100000
    dt = np.sqrt(p/(1 + e))/100

    # Initialize final data
    E, L, C = schmidtparam3((r0)/dist_mult, e, inc, a)
    vals = [[0, E, L, C, r0, e, Ep, Lp]]#, inc, cosi]]
    

    # A way to track our progress so we don't get bored out of our minds
    sep_stuff = get_sep_inc(a, inc, 50)
    #line = np.polyfit(sep_stuff["p"][:len(sep_stuff["p"])//2]*dist_mult, sep_stuff["e"][:len(sep_stuff["p"])//2], 1)
    line = np.polyfit(sep_stuff["e"][:len(sep_stuff["p"])//2], sep_stuff["p"][:len(sep_stuff["p"])//2]*dist_mult, 2)
    pbar = tqdm(total = 10000, position=0)
    pbar.set_postfix_str(f"Semilat: {r0*(1 - e**2)/dist_mult:.4f}, Ecc {e:.4f}, Peri: {r0*(1 - e)/dist_mult:.4f}")
    prog = 0
    # Counter for how many times we find complex orbital constants in a row
    wack = 0
    # Continue while periapse is greater than marginally bound radius for a=0 and 
    #  semilatus rectum is not super complex
    while (np.polyval(line, e) - (r0*(1 - e*e)) < - 1 and np.abs(np.imag(r0*(1 - e*e))/np.real(r0*(1 - e*e))) < 1e-5) and not eval(endflag):
        try:
            # Get changes to values
            #dEp = dEpdt(r0, e)*dt
            #dLp = dLpdt(r0, e)*dt
            dr0 = dr0dt(r0, e)*dt
            de = dedt(r0, e)*dt
            # FIX THESE
            dEp = (-G*m*M/(2*r0*r0)) * dr0
            dLp = 0.5*np.sqrt((G*M*M*m*m*(1 - e*e))/((m+M)*r0))*dr0 + np.sqrt((G*M*M*m*m*r0*e*e)/((m+M)*(1-e*e)))*de
            # If the change in dr0 is way too big, shrink the step size and try again
            if abs(dr0/r0) > 0.1:
                #print(dr0, r0, dr0/r0, dt)
                dt *= 0.5
                continue
            # Assume inclination remains constant because a=0
            E, L, C = schmidtparam3((r0 + dr0)/dist_mult, e + de, inc, a)
            # If we get complex orbital constants, try taking a smaller step size
            # Use Q = C + (a*E - L)**2 so that equatorial orbits don't make division by zero errors
            tester = np.array([E, L, C + (a*E - L)**2])
            if True in np.abs(np.imag(tester)/np.real(tester)) < 1e-5:
                wack += 1
                if wack < 3:
                    dt *= 0.5
                    continue
                # If you get the complexity issue three times in a row, just stop
                else:
                    break
            else:
                wack = 0

            # Get all your new values    
            # We actuall don't really do anything with Ep or Lp, we just keep track of them
            Ep, Lp, r0, e = Ep + dEp, Lp + dLp, r0 + dr0, e + de
            # Update so that the user knows what's going on
            pbar.set_postfix_str(f"Semilat: {r0*(1 - e**2)/dist_mult:.4f}, Ecc {e:.4f}, Peri: {r0*(1 - e)/dist_mult:.4f}")
            pbar.update(int(10000*(p - r0*(1 - e**2))/(p - line[0]*(e - line[1]))) - prog)
            prog = int(10000*(p - r0*(1 - e**2))/(p - line[0]*(e - line[1])))
            vals.append([vals[-1][0] + dt, E, L, C, r0, e, Ep, Lp])
            # Make the step size a little bigger, just as a treat
            dt *= 1.05
        except Exception as e:
            print(e)
            break
    pbar.close()
    vals = np.real(np.array(vals))
    #If the last value is past the separatrix, interpolate back?
    # Convert the final point to semilatus rectum
    p_last = vals[-1, 4] * (1 - vals[-1, 5]**2)/dist_mult
    e_last = vals[-1, 5]

    # Get nearby separatrix points
    e_tol = 0.01
    e_ix = np.where(np.abs(sep_stuff["e"] - e_last) < e_tol)[0]
    while len(e_ix) < 2:
        e_tol += 0.01
        e_ix = np.where(np.abs(sep_stuff["e"] - e_last) < e_tol)[0]

    #print("es", sep_stuff["e"][e_ix])
    #print("ps", sep_stuff["p"][e_ix])
    # Local linear approximation to separatrix: e_sep(p)
    poly = np.polyfit(sep_stuff["p"][e_ix], sep_stuff["e"][e_ix], 1)
    e_sep_last = np.polyval(poly, p_last)
    # If the final point is still physically allowed, do nothing.
    if abs((e_sep_last - e_last)/e_sep_last) > 0.01:
        # Second to last two points
        old = vals[-3]
        new = vals[-2]

        # Corresponding p and e values
        p_old = old[4] * (1 - old[5]**2)/dist_mult,
        e_old = old[5]
        p_new = new[4] * (1 - new[5]**2)/dist_mult,
        e_new = new[5]
        slope, intercept = poly
        denom = (e_new - e_old) - slope * (p_new[0] - p_old[0])

        if abs(denom) >= 1e-14:
            t = (slope * p_old[0] + intercept - e_old) / denom
            # Make sure the intersection actually lies between the two points
            # Actually extend the range to 2 since I'm using second to last
            if 0 <= t <= 2:
                # Interpolate the entire state vector
                new_point = old + t * (new - old)
                # Replace the old final point with the intersection
                vals[-1] = new_point
    
    A = (a**2)*(1 - vals[:, 0]**2)
    dct = {"name": "Peters",
           "tracktime": vals[:, 0]/time_mult,
           "energy": vals[:, 1],
           "phi_momentum": vals[:, 2],
           "carter": vals[:, 3],
           "r0": vals[:, 4]/dist_mult,
           "p": vals[:, 4]*(1 - vals[:, 5]**2)/dist_mult,
           "e": vals[:, 5],
           "Ep": vals[:, 6],
           "Lp": vals[:, 7],
           "inc": inc*np.ones_like(vals[:,0]),
           "cosi": vals[:, 2]/np.sqrt(vals[:, 3] + vals[:, 2]**2),
           "it": vals[:, 4]*(1 - vals[:, 5])/dist_mult,
           "ot": vals[:, 4]*(1 + vals[:, 5])/dist_mult,
           "spin": a}
    return dct

def glamp_2002(a, q, cons=False, params=False, endflag="False", eps=1e-3, verbose=True):
    # matches glampedakis 2002
    if cons != False:
        # Use constants to start
        E, L, C = cons
        turns, flats, zs = root_getter(E, L, C, a)
        p, e = 2*turns[-1]*turns[-2]/(turns[-1] + turns[-2]), (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
        inc = np.sign(L)*np.arccos(min(1, np.mean(np.abs(zs[1:3]))))
        r0 = p/(1 - e**2)
    elif params != False:
        # Use orbital parameters to start
        r0, e, inc = params
        p = r0*(1 - e**2)
        E, L, C = schmidtparam3(*params, a)
        turns, flats, zs = root_getter(E, L, C, a)
    else:
        # Nuffin
        print("No starting input.")
        return 0
    
    cosi = L/np.sqrt(C + L**2)
    f1 = lambda x: 1 + (73/24)*(x**2) + (37/96)*(x**4)
    f2 = lambda x: 73/12 + (823/24)*(x**2) + (949/32)*(x**4) + (491/192)*(x**6)
    f3 = lambda x: 1 + (7/8)*(x**2)
    f4 = lambda x: 61/24 + (63/8)*(x**2) + (95/64)*(x**4)
    f5 = lambda x: 61/8 + (91/4)*(x**2) + (461/64)*(x**4)
    f6 = lambda x: 97/12 + (37/2)*(x**2) + (211/32)*(x**4)
    B = (64/5) * 1 * q * (1 + q) 
    c0 = (p/(e**(12/19)))/((1 + (121/304)*(e*e))**(870/2299))
    circ_dt = (c0**4)*(e**(48/19))/(4*B)
    ecc_dt = (768/425)*((1-e**2)**3.5)*(r0**4)/(4*B)
    max_dt = circ_dt/10
    dt = circ_dt/1000
    vals = [[0, E, L, C, p, e, cosi, inc, *turns]]
    oldp = p
    if verbose == True:
        pbar = tqdm(total = 10000, position=0)
        pbar.set_postfix_str(f"Semilat: {r0*(1 - e**2):.4f}, Ecc {e:.4f}, Peri: {r0*(1 - e):.4f}")
        prog = 0
    r_isco, r_mb = find_rms(a), find_rmb(a)
    # Approximate slope of the separatrix for a given spin - not exact
    stop = False
    eep = False
    #while (np.abs(np.imag(p)/np.real(p)) < 1e-3 and np.real(p)/(1 + np.real(e)) > find_rmb(a * np.sign(cosi))) and not eval(endflag):
    while (np.abs(np.imag(p)/np.real(p)) < 1e-3) and not eval(endflag):
        try:
            E_dot = -(32/5)*(q**2)*(p**-5)*((1 - e**2)**1.5)*(f1(e) - a*(p**-1.5)*cosi*f2(e))
            L_dot = -(32/5)*(q**2)*(p**-3.5)*((1 - e**2)**1.5)*(cosi*f3(e) + a*(p**-1.5)*(f4(e) - cosi*cosi*f5(e)))
            C_dot = 2*C*L_dot/L
            E, L, C = np.real([E + E_dot*dt, L + L_dot*dt, C + C_dot*dt])
            turns, flats, zs = root_getter(E, L, C, a)
            p_1, e_1 = 2*turns[-1]*turns[-2]/(turns[-1] + turns[-2]), (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
            inc = np.sign(L)*np.arccos(min(1, np.mean(np.abs(zs[1:3]))))
            cosi_1 = L/np.sqrt(C + L**2)
            # If r_a or r_p are too complex, shrink that biz
            if max(np.abs(np.imag(turns[-2:]))/np.abs(np.real(turns[-2:]))) > 5e-4:
                dt *= 0.5
                if dt < p_1**1.5 + a:
                    print("Turning Point Complexity Exceeded Allowed Threshold - glamp") #if verbose == True else None
                    stop = True
                    break
            else:
                p, e, cosi = p_1, e_1, cosi_1
                vals.append([vals[-1][0] + dt, E, L, C, p, e, cosi, inc, *turns])
                # Tell the user where we're at right now
                if verbose == True:
                    pbar.set_postfix_str(f"Semilat: {np.real(p_1):.4f}, Ecc {np.real(e_1):.4f}, Peri: {np.real(p_1/(1 + e_1)):.4f}, dt={dt:.4e}")
                    pbar.update(max(0, int(np.real(10000*(oldp - p_1)/(oldp - (r_isco + e*(2*r_mb - r_isco)))))) - prog)
                    prog = max(0, int(np.real(10000*(oldp - p_1)/(oldp - (r_isco + e*(2*r_mb - r_isco))))))
                # Calculate distance and rate of approach to separatrix
                #delta = max(abs(np.array(vals[-1][-2:]) - turns[-2:])/np.array(vals[-1][-2:])) 
                delta = np.sqrt((vals[-1][4] - vals[-2][4])**2 + 100*(vals[-1][5] - vals[-2][5])**2)/100
                
                if delta < eps / 10:
                    dt *= 1.05
                elif delta > eps / 5:
                    dt *= 0.5
                else:
                    dt *= (0.5 - 1.05)*(delta - eps/5)/(eps/5 - eps/10) + 0.5
                #dt = min(dt, max_dt)
                #dt = max(dt, np.real(((M*p)**4)/(4*b))/1e37)
                #dt = max(dt, 10*(2 * np.pi * (6.67e-11 * M / ((3e8)**3)) * (p**(1.5) - a*np.sign(L))))
        except Exception as bad:
            print(bad)
            stop = True
            break
    pbar.close() if verbose == True else None
    if not stop:
        print("All Gucci - glamp", f"({np.abs(np.imag(p)/np.real(p)) < 1e-3} and {np.real(p)/(1 + np.real(e)) > find_rmb(a)}) and {not eval(endflag)}") #if verbose == True else None
        print((vals[-1][4]/(1 + vals[-1][5]) - p_1/(1 + e_1) <= 0))
        print(max(abs(np.array(vals[-1][-2:]) - turns[-2:])/np.array(vals[-1][-2:])) > eps)
        print("EEEp", eep)

    vals = np.real(np.array(vals))
    # Return stuff in the same format as EMRIGenerator for convenience
    dct = {"name": "GHK",
           "tracktime": vals[:, 0]*q,
           "energy": vals[:, 1],
           "phi_momentum": vals[:, 2],
           "carter": vals[:, 3],
           "p": vals[:, 4],
           "e": vals[:, 5],
           "cosi": vals[:, 6],
           "inc": vals[:, 7],
           "it": vals[:, 10],
           "ot": vals[:, 11],
           "r0": vals[:, 4]/(1 - vals[:, 5]**2),
           "spin": a}
    return dct

def gair_glamp_2006(a, q, cons=False, params=False, endflag="False", eps=1e-3, verbose=True):
    '''
    Reproduce results for Gair + Glampedakis 2006
    
    :param a: Black Hole Spin - between -1 and 1
    :param q: Mass Ratio - <= 10^-4 for EMRIs
    :param cons: Orbital Constants (optional) - [Energy, Axial Angular Momentum, Carter Constant] per unit mass
    :param params: Orbital Parameters (optional) - [Semimajor Axis, Eccentricity, Inclination]
    :param endflag: End state of orbit (optional) - Boolean statement to end simulation at a certain condition.
                                                    Defaults to False, so the sim ends at plunge
    '''

    flip = False
    if cons != False:
        # Use constants to start
        E, L, C = cons
        if a < 0:
            flip = True
            a = abs(a)
            L *= -1
        turns, flats, zs = root_getter(E, L, C, a)
        p, e = 2*turns[-1]*turns[-2]/(turns[-1] + turns[-2]), (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
        inc = np.sign(L)*np.arccos(min(1, np.mean(np.abs(zs[1:3]))))
        r0 = p/(1 - e**2)
    elif params != False:
        # Use orbital parameters to start
        r0, e, inc = params
        if a < 0:
            flip = True
            a = abs(a)
            inc *= -1
        p = r0*(1 - e**2)
        E, L, C = schmidtparam3(r0, e, inc, a)
        turns, flats, zs = root_getter(E, L, C, a)
    else:
        # Nuffin
        print("No starting input.")
        return 0

    # inclination approximation for cosine of inclination
    cosi =  L/np.sqrt(C + L**2)
    sini = np.sqrt(1 - cosi**2)
    # Functions in terms of e
    g1 = lambda x: 1 + (73/24)*(x**2) + (37/96)*(x**4)
    g2 = lambda x: 73/12 + (823/24)*(x**2) + (949/32)*(x**4) + (491/192)*(x**6)
    g3 = lambda x: 1247/336 + (9181/672)*x*x
    g4 = lambda x: 4 + (1375/48)*x*x
    g5 = lambda x: 44711/9072 + (172157/2592)*x*x
    g6 = lambda x: 33/16 + (359/32)*x*x
    g7 = lambda x: 8191/672 + (44531/336)*x*x
    g8 = lambda x: 3749/336 - (5143/168)*x*x
    g9 = lambda x: 1 + (7/8)*(x**2)
    g10= lambda x: 61/12 + (119/8)*(x**2) + (183/32)*(x**4)
    g10a=lambda x: 61/24 + (63/8)*(x**2) + (95/64)*(x**4)
    g10b=lambda x: 61/8 + (91/4)*(x**2) + (461/64)*(x**4)
    g11= lambda x: 1247/336 + (425/336)*x*x
    g12= lambda x: 4 + (97/8)*x*x
    g13= lambda x: 44711/9072 + (302893/6048)*x*x
    g14= lambda x: 33/16 + (95/16)*x*x
    g15= lambda x: 8191/672 + (48361/1344)*x*x
    g16= lambda x: 417/56 - (37241/672)*x*x

    
    # Get initial dt 
    dt = p**0.5
    dt = np.sqrt((p**1.5))
    B = (64/5) * 1 * q * (1 + q) 
    c0 = (p/(e**(12/19)))/((1 + (121/304)*(e*e))**(870/2299))
    circ_dt = (c0**4)*(e**(48/19))/(4*B)
    ecc_dt = (768/425)*((1-e**2)**3.5)*(r0**4)/(4*B)
    max_dt = circ_dt/10
    dt = circ_dt/1000
    

    # Initialize final output and also save the first semilatus rectum for comparison
    vals = [[0, E, L, C, p, e, inc, cosi, *turns[-2:], 0]]
    oldp = p
    
    # A way to track our progress so we don't get bored out of our minds
    if verbose == True:
        pbar = tqdm(total = 10000, position=0)
        pbar.set_postfix_str(f"Semilat: {p:.4f}, Ecc {e:.4f}, Peri: {(p/(1 + e)):.4f}")
        prog = 0
    # Useful to keep track of the ISCO and marginally bound radius for later

    sep_stuff = get_sep_cosi(a, cosi, mult=500)
    dist = p - sep_stuff["p"][op.get_index(sep_stuff["e"], 1)]
    line = np.polyfit(sep_stuff["p"][:len(sep_stuff["p"])//2], sep_stuff["e"][:len(sep_stuff["p"])//2], 1)

    r_isco, r_mb = find_rms(a*np.sign(L)), find_rmb(a*np.sign(L))
    # Approximate slope of the separatrix for a given spin - not exact
    approx_m = 1/(2*r_mb - r_isco)
    # Keep going while p is not super complex and the periapsis is above the marginally bound radius
    stop = False
    delta = 1
    deltert = 0.0
    while (np.abs(np.imag(p)/np.real(p)) < eps and (deltert < 1e-4)) and not eval(endflag):
        try:
            E, L, C = vals[-1][1:4]
            # Define the predicted changes in E, L, C, Q
            E_dot = lambda p, e, cosi, sini: -(32/5)*((q)**2)*((1/p)**5)*((1 - e**2)**1.5)*(g1(e) - (a/1)*((1/p)**1.5)*g2(e)*cosi - (1/p)*g3(e) + np.pi*((1/p)**1.5)*g4(e) - ((1/p)**2)*g5(e) + ((a/p)**2)*g6(e) - (527/96)*((a/p)**2)*(sini**2))
            L_dot = lambda p, e, cosi, sini: -(32/5)*(q*q/1)*((1/p)**3.5)*((1 - e**2)**1.5)*(g9(e)*cosi + (a/1)*((1/p)**1.5)*(g10a(e) - cosi*cosi*g10b(e)) - (1/p)*g11(e)*cosi + np.pi*((1/p)**1.5)*g12(e)*cosi - ((1/p)**2)*g13(e)*cosi + ((a/p)**2)*cosi*(g14(e) - (45/8)*sini*sini))
            C_dot = lambda p, e, cosi, sini: -(64/5)*(q*q/1)*((1/p)**3.5)*np.sqrt(C)*sini*((1 - e**2)**1.5)*(g9(e) - (a/1)*((1/p)**1.5)*cosi*g10b(e) - (1/p)*g11(e) + np.pi*((1/p)**1.5)*g12(e) - ((1/p)**2)*g13(e) + ((a/p)**2)*(g14(e) - (45/8)*sini*sini))
            inc = np.sign(L)*np.arccos(min(1, np.mean(np.abs(zs[1:3]))))
            
            # Save current parameters for comparison, make them real so they're useful
            #  really only using p_0 at this point anyway, and if it were too complex we wouldn't be here
            p_0, e_0, cosi_0, sini_0 = np.real(p), np.real(e), np.real(cosi), np.real(sini)
            
            # Get the true change in E from the corrections made to the 2006 paper
            E_0, L_0 = circE_inc(p_0, a, inc), circL_inc(p_0, a, inc)
            N1 = E_0*(p_0**4) + (a**2)*E_0*(p_0**2) - 2*a*(L_0 - a*E_0)*p_0
            N4 = (2*p_0 - p_0**2)*L_0 - 2*a*E_0*p_0
            N5 = (2*p_0 - p_0**2 - a**2)/2
            E_dot_true = ((1 - e_0**2)**1.5)*(((1 - e_0**2)**(-1.5))*E_dot(p_0, e_0, cosi_0, sini_0) - E_dot(p_0, 0, cosi_0, sini_0) - (N4/N1)*L_dot(p_0, 0, cosi_0, sini_0) - (N5/N1)*C_dot(p_0, 0.0, cosi_0, sini_0))
            # If it's complex we have a problem! Bail!
            if not np.real(E_dot_true):
                print(E_dot_true, "grah", E_dot(p_0, 0, cosi_0, sini_0), L_dot(p_0, 0, cosi_0, sini_0), C_dot(p_0, 0, cosi_0, sini_0), len(vals))
                break
            # Apply changes
            E, L, C = np.real([E + E_dot_true*dt, L + L_dot(p_0, e_0, cosi_0, sini_0)*dt, C + C_dot(p_0, e_0, cosi_0, sini_0)*dt])
            # If they don't make a viable orbit we have a problem! Bail!
            try:
                turns, flats, zs = root_getter(E, L, C, a)
                r_p, r_a = turns[-2:]
            except:
                print(E, L, C, a, E_0, L_0, E_dot_true, dt, p, e, cosi)
                break
            # Get preliminary new parameters
            p_1, e_1 = 2*r_a*r_p/(r_a + r_p), (r_a - r_p)/(r_a + r_p)
            cosi_1 = L/np.sqrt(np.abs(C + L**2))
            sini_1 = np.sqrt(np.abs(1 - cosi_1**2))

            # If r_a or r_p are too complex, shrink that biz
            if max(np.abs(np.imag(turns[-2:]))/np.abs(np.real(turns[-2:]))) > eps:
                dt *= 0.5
                if dt < p_0**0.25:
                    print("Turning Point Complexity Exceeded Allowed Threshold - gg") #if verbose == True else None
                    stop = True
                    break
            # r_a and r_p should both decrease
            #elif (e_0 > 1e-3 and min(np.array(vals[-1][8:10]) - turns[-2:]) < -1e-5) or (e_0 <= 1e-3 and vals[-1][4] - p_0 < 0):
            #    dt *= 0.999
            #    if dt < p_0**1.5 + a:
            #        print("Turning Point Monotonic Descent Violated - gg") #if verbose == True else None
            #        stop = True
            #        break
            # don't let r_a or r_p change too much
            elif max(abs(np.array(vals[-1][8:10]) - turns[-2:])/np.array(vals[-1][8:10])) > eps:
                dt *= 0.5
                if dt < p_0**0.25:
                    print("Turning Point Descent Perturbation Model Violated - gg") #if verbose == True else None
                    print(abs(np.array(vals[-1][8:10]) - turns[-2:])/np.array(vals[-1][8:10]))
                    stop = True
                    break
            # If it's not any of those, you're not done yet. Keep going!
            else:
                # Accept new values of parameters
                p, e, cosi, sini = p_1, e_1, cosi_1, sini_1
                delta = max(abs(np.array(vals[-1][8:10]) - turns[-2:])/np.array(vals[-1][8:10]))
                if delta == 0.0:
                    deltert = 0.0
                else:
                    deltert = np.abs(np.imag(delta)/np.real(delta))

                if delta < eps / 10:
                    tweak = 1
                    dt *= 1.05
                elif delta > eps / 5:
                    tweak = 2
                    dt *= 0.5
                else:
                    tweak = 3
                    dt *= (0.5 - 1.05)*(delta - eps/5)/(eps/5 - eps/10) + 0.5

                #dt = max(dt, np.real(((M*p)**4)/(4*b))/1e37)
                #dt = np.real(max(dt, 1 ))

                # Tell the user where we're at right now
                if verbose == True:
                    pbar.set_postfix_str(f"Semilat: {np.real(p):.4f}, Ecc {np.real(e):.4f}, Peri: {np.real((p/(1 + e))):.4f}, dt={dt:.4e}")
                    new_prog = np.real(max(prog, 1 - (p - sep_stuff["p"][op.get_index(sep_stuff["e"], 1)])/dist))
                    pbar.update(int((new_prog - prog)*10000))
                    prog = new_prog

                r0 = p/(1 - e**2)
                vals.append([vals[-1][0] + dt, E, L, C, p, e, inc, cosi, *turns[-2:], tweak])

        except Exception as e:
            print(e)
            stop = True
            break
    if not stop:
        print(np.abs(np.imag(p)/np.real(p)) < eps, np.iscomplex(delta), delta)
    pbar.close() if verbose == True else None
    vals = np.real(np.array(vals))
    # Return stuff in the same format as EMRIGenerator for convenience
    dct = {"name": "GG",
           "tracktime": vals[:, 0]*q,
           "energy": vals[:, 1],
           "phi_momentum": vals[:, 2]*((-1)**flip),
           "carter": vals[:, 3],
           "p": vals[:, 4],
           "e": vals[:, 5],
           "inc": vals[:, 6]*((-1)**flip),
           "cosi": vals[:, 7]*((-1)**flip),
           "it": vals[:, 8],
           "ot": vals[:, 9],
           "spin": a*((-1)**flip),
           "tweak": vals[:, 10]}
    return dct

def compy(a, q, cons=False, params=False):
    peters2 = peters_sim_re(a, q, cons=cons, params=params)
    glamp = glamp_2002(a, q, cons=cons, params=params)
    gg3 = gair_glamp_2006(a, q, cons=cons, params=params)
    return [peters2, glamp, gg3]