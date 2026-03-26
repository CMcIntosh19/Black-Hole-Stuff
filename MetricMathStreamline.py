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
    r, theta = state[1], state[2]
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
    r, theta = state[1], state[2]
    sine, cosi = np.sin(theta), np.cos(theta)
    cota = cosi/sine if sine != 0.0 else 1e100  # optionally replace with np.inf
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
            new = recalc_state(cons, pos, a)
        else:
            E, L, C = cons
            R = [(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -a*a*C]
            turns = np.roots(R)
            turns = np.sort(turns.real[abs(turns.imag)<1e-14])
            pos = [0.0, 2*turns[-1]*turns[-2]/(turns[-1] + turns[-2]), np.pi/2, 0.0]
            new = recalc_state(cons, pos, a)
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
            print("Non-viable parameters")
        if np.shape(pos) != (4,):
            pos = [0.0, params[0]*(1 - params[1]**2), np.pi/2, 0.0]
        new = recalc_state(cons, pos, a)
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
        coeff = np.array([E**2 - 1.0, 2.0, (a**2)*(E**2 - 1.0) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
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
            if (np.prod(np.sign(solution)) == np.sign(np.sin(j))):
                E, L, C = solution
                break     
        if polar == True:
            L, C = 0.0, C/(z**2) - (a**2)*(1 - E**2)
            
        if (np.sqrt(L**2 + C) - np.abs(L))/np.abs(L) < 1e-15:
            C = 0.0
        return [E, L, C]

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

def recalc_state(constants, state, a):
    '''
    Calculates a new state vector based on a given position and constants of motion

    Parameters
    ----------
    constants : 3-element list/array of floats
        energy, angular momentum, and carter constant per unit mass
    state : 8 or 4 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
        Only the 4-position is explicitly required, not specifying the 4-velocity will make the resulting vector default to decreasing r and theta
        specifying the 4 velocity will maintain the r and theta directions
    dTau : float
        proper time between current state and state-to-be-calculated
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array
        4-position and 4-velocity of the test particle
    '''
    energy, lmom, cart = constants[0], constants[1], constants[2]
    rad, theta = state[1], state[2]
    sig, tri = rad**2 + (a**2)*(np.cos(theta)**2), rad**2 - 2*rad + a**2

    p_r = np.array([energy, 0, a*(a*energy - lmom)])
    r_r = np.array([energy**2 - 1, 2, (a**2)*(energy**2 - 1) - lmom**2 - cart, 2*((a*energy - lmom)**2 + cart), -cart*(a**2)])
    the_the = np.array([(a**2)*(1 - energy**2), 0, - (cart + (a**2)*(1 - energy**2) + lmom**2), 0, cart])
    
    tlam = -a*(a*energy*(np.sin(theta)**2) - lmom) + ((rad**2 + a**2)/tri)*np.polyval(p_r, rad)
    rlam = np.sqrt(abs(np.polyval(r_r, rad)))
    cothelam = np.sqrt(abs(np.polyval(the_the, np.cos(theta))))
    thelam = (-1/np.sin(theta))*cothelam
    philam = -( a*energy - ( lmom/(np.sin(theta)**2) ) ) + (a/tri)*np.polyval(p_r, rad)
    
    ttau = tlam/sig
    rtau = rlam/sig
    thetau = thelam/sig
    phitau = philam/sig
    
    #sign correction and initialization
    if (len(state) != 8):
        rtau = abs(rtau) * -1
        thetau = abs(thetau) * -1
        new_state = np.zeros(8)
        new_state[:4] = state[:4]
    else:
        roots = np.sort(np.roots(r_r))
        '''
        #If current radius is between the inner and outer turning points, maintain direction
        if (rad - roots[-2])*(roots[-1] - rad) > 0:
            direc = np.sign(state[5])
        #If current radius is somehow outside that range, follow the potential to go back in
        else:
            direc = np.sign(np.polyval(np.polyder(r_r), rad))
        '''
        direc = np.sign(state[5])
        rtau = abs(rtau) * direc
        thetau = abs(thetau) * np.sign(state[6])
        new_state = np.copy(state)
        
    new_state[4:] = np.array([ttau, rtau, thetau, phitau])
    return new_state

def recalc_state2(constants, state, a):
    '''
    Calculates a new state vector based on a given position and constants of motion

    Parameters
    ----------
    constants : 3-element list/array of floats
        energy, angular momentum, and carter constant per unit mass
    state : 8 or 4 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
        Only the 4-position is explicitly required, not specifying the 4-velocity will make the resulting vector default to decreasing r and theta
        specifying the 4 velocity will maintain the r and theta directions
    dTau : float
        proper time between current state and state-to-be-calculated
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array
        4-position and 4-velocity of the test particle
    '''
    energy, lmom, cart = constants[0], constants[1], constants[2]
    if abs(cart) < 1e-11:
        cart = 0.0
    if abs(lmom) < 1e-15:
        lmom = 0.0
    rad, theta = state[1], state[2]
    sig, tri = rad**2 + (a**2)*(np.cos(theta)**2), rad**2 - 2*rad + a**2

    p_r = np.array([energy, 0, a*(a*energy - lmom)])
    r_r = np.array([energy**2 - 1, 2, (a**2)*(energy**2 - 1) - lmom**2 - cart, 2*((a*energy - lmom)**2 + cart), -cart*(a**2)])
    the_the = np.array([(a**2)*(1 - energy**2), 0, - (cart + (a**2)*(1 - energy**2) + lmom**2), 0, cart])
    
    tlam = -a*(a*energy*(np.sin(theta)**2) - lmom) + ((rad**2 + a**2)/tri)*np.polyval(p_r, rad)
    rlam = np.sqrt(abs(np.polyval(r_r, rad)))
    cothelam = np.sqrt(abs(np.polyval(the_the, np.cos(theta))))
    thelam = (-1/np.sin(theta))*cothelam
    philam = -( a*energy - ( lmom/(np.sin(theta)**2) ) ) + (a/tri)*np.polyval(p_r, rad)
    
    ttau = tlam/sig
    rtau = rlam/sig
    thetau = thelam/sig
    phitau = philam/sig
    
    #sign correction and initialization
    if (len(state) != 8):
        rtau = abs(rtau) * -1
        thetau = abs(thetau) * -1
        new_state = np.zeros(8)
        new_state[:4] = state[:4]
    else:
        roots = np.sort(np.roots(r_r))
        '''
        #If current radius is between the inner and outer turning points, maintain direction
        if (rad - roots[-2])*(roots[-1] - rad) > 0:
            direc = np.sign(state[5])
        #If current radius is somehow outside that range, follow the potential to go back in
        else:
            direc = np.sign(np.polyval(np.polyder(r_r), rad))
        '''
        direc = np.sign(state[5])
        rtau = abs(rtau) * direc
        thetau = abs(thetau) * np.sign(state[6])
        new_state = np.copy(state)
        
    new_state[4:] = np.array([ttau, rtau, thetau, phitau])
    return new_state

def interpolate2(data, time, supress=True):
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
    # Use this one for gwaves
    data = np.array(data)
    if supress == True:
        phi_per = 2 * np.pi * abs((time[-1] - time[0])/(data[-1, 2] - data[0, 2]))
        new_time = np.arange(time[0], time[-1], min(phi_per/200, (time[-1] - time[0])/10000))
    else:
        new_time = np.linspace(time[0], time[-1], max(len(time), 10*int(time[-1] - time[0])))
    

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

def interpolate3(data, time):
    '''
    interpolates coordinate data to be evenly spaced in coordinate time
    Fail state plots time against index twice?
    automatically sets it to 400 points or length of original data, whichever is shorter

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
    new_time = np.linspace(time[0], time[-1], min(len(data), 400))
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

def gwaves(quad_moment, time, distance, e_r = None):
    '''
    Calculates gravitational wave moment from quadrupole moment and distance (but not mass ratio)

    Parameters
    ----------
    quad_moment : N x 3 x 3 numpy array of floats
        x, y, z quadrupole moment of test particle, assumed interpolated
    time : N element numpy array of floats
        coordinate time of test particle, assumed interpolated
    distance : float
        distance from GW source in geometric units

    Returns
    -------
    waves : N x 3 x 3 numpy array of floats
        GW moment over time; waves[:,0,0] is h+ polarization, waves[:,0,1]=waves[:,1,0] is hx polarization
    '''
    der_2 = matrix_derive(quad_moment, time, 2)

    if e_r is None:
        e_r = [1, 0, 0]
    e_r = np.array(e_r) / np.linalg.norm(e_r)
    # choose any vector not parallel to e_r
    if np.allclose(e_r, [0,0,1]):
        ref = np.array([0,1,0])
    else:
        ref = np.array([0,0,1])
    e_th = ref - np.dot(ref, e_r)*e_r
    e_th /= np.linalg.norm(e_th)
    e_ph = np.cross(e_th, e_r)
    print(e_r, e_th, e_ph)
    P = np.eye(3) - np.outer(e_r, e_r)
    h_TT = np.zeros_like(der_2)
    for k in range(len(der_2)):
        term = P @ der_2[k] @ P
        trace = np.trace(P @ der_2[k])
        h_TT[k] = term - 0.5 * P * trace
    h_TT *= 2/distance

    h_plus = 0.5 * (np.einsum("i, j, kij -> k", e_th, e_th, h_TT) - np.einsum("i, j, kij -> k", e_ph, e_ph, h_TT))
    h_cross = 0.5 * (np.einsum("i, j, kij -> k", e_th, e_ph, h_TT) + np.einsum("i, j, kij -> k", e_ph, e_th, h_TT))
    return h_plus, h_cross

def full_transform(data, distance, supress=True, m_bh=False, e_r = None):    #defunctish??
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
    int_sphere, int_time = interpolate2(sphere, time, supress)

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

    h_plus, h_cross = gwaves(qmom, int_time, distance, e_r)
    return data["inputs"][2]*h_plus, data["inputs"][2]*h_cross, int_time

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

def peters_integrate6(states, a, mu, ind1, ind2):
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
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(int_sphere)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad - (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2 = matrix_derive(coolquad, int_time, 2)
        dt3 = matrix_derive(coolquad, int_time, 3)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*np.einsum('ijk,ijk ->i', dt3, dt3)
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
        #print(len(time), time[-1]-time[0], dE, np.linalg.norm([dLx, dLy, dLz]))
    #print([dE, dLx, dLy, dLz], "org")
    return np.array([dE, dLx, dLy, dLz])
    #return quad

def peters_integrate6_2(states, a, mu, ind1, ind2):
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
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(int_sphere)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = np.copy(quad)#3*quad - np.array([delta*r*r for r,th,ph in int_sphere])
        dt2 = matrix_derive(coolquad, int_time, 2)
        dt3 = matrix_derive(coolquad, int_time, 3)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
        #print(len(time), time[-1]-time[0], dE, np.linalg.norm([dLx, dLy, dLz]))
    return np.array([dE, dLx, dLy, dLz])
    #return quad

def peters_integrate6_3(states, a, mu, ind1, ind2):
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
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(sphere)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad - (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2 = matrix_derive2(coolquad, time, int_time, 2)
        dt3 = matrix_derive2(coolquad, time, int_time, 3)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
        #print(len(time), time[-1]-time[0], dE, np.linalg.norm([dLx, dLy, dLz]))
    #print([dE, dLx, dLy, dLz], "6_3")
    return np.array([dE, dLx, dLy, dLz])
    #return quad

def peters_integrate6_4(states, a, mu, ind1, ind2):
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
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad - (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2, dt3 = matrix_derive3(coolquad, time, int_time)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu
        #print([dE, dLx, dLy, dLz])
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
        #print(len(time), time[-1]-time[0], dE, np.linalg.norm([dLx, dLy, dLz]))
    #print([dE, dLx, dLy, dLz], "6_4")
    return np.array([dE, dLx, dLy, dLz])
    #return quad

def peters_integrate6_5(states, a, mu, ind1, ind2):
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
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1 - 10) > 2:
        states = np.array(states)
        sphere, time = states[5:-5, 1:4], states[5:-5, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad - (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2, dt3 = matrix_derive3(coolquad, time, int_time)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
            #scale both changes to make up for the bits that got cut off
        #print([dE, dLx, dLy, dLz])
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
        #print(len(time), time[-1]-time[0], dE, np.linalg.norm([dLx, dLy, dLz]))
        #print([dE, dLx, dLy, dLz], "6_5")
        return np.array([dE, dLx, dLy, dLz])
    else:
        #print("gorp", "6_5")
        return np.array([0.0, 0.0, 0.0, 0.0])
    #return quad

def peters_integrate6_6(states, a, mu, ind1, ind2):
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
        sphere, time = states[5:-5, 1:4], states[5:-5, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = matrix_derive3_numba(coolquad, dt) 
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        #print((states[-1,0] - states[0,0]), (int_time[-1] - int_time[0]), "wa")
        dE = mu*mu*np.sum(dedt*div)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        #print(dE, dLx, dLy, dLz)
        #print(states[-1,0] - states[0,0])
        return np.array([dE, dLx, dLy, dLz])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

def peters_integrate6_6_2(states, real_len, a, mu, ind1, ind2):
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
        #states = np.array(states[:real_len])
        states = np.array(states)
        #move clip to after derivatives since numba matrix derive does some weird stuff
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate_njit2(sphere, time, real_len, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = matrix_derive3_numba(coolquad, dt) 
        dt2 = dt2[5:-5]
        dt3 = dt3[5:-5]
        int_time = int_time[5:-5]
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        #print("dt2 check", np.isnan(np.sum(dt2)))
        #print("dt3 check", np.isnan(np.sum(dt3)))
        #print("div check", np.isnan(np.sum(dt2)))
        #print("states check", np.isnan(np.sum(states)))
        #print("time check", np.isnan(np.sum(time)))
        #print("int_time check", np.isnan(np.sum(int_time)))
        if np.isnan(np.sum(states)):
            print(np.where(np.isnan(states)))
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        #print((states[real_len-1,0] - states[0,0]), (int_time[-1] - int_time[0]), "wo")
        dE = mu*mu*np.sum(dedt*div)*(states[real_len-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)*(states[real_len-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        return np.array([dE, dLx, dLy, dLz])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

def peters_integrate6_6_3(states, real_len, a, mu, ind1, ind2):
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
        sphere, time = states[5:-5, 1:4], states[5:-5, 0]
        int_sphere, int_time = interpolate_njit(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        #coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = matrix_derive3_numba(quad, dt) 
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        #print((states[-1,0] - states[0,0]), (int_time[-1] - int_time[0]), "wa")
        dE = mu*mu*np.sum(dedt*div)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        #print(dE, dLx, dLy, dLz)
        return np.array([dE, dLx, dLy, dLz])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

def peters_integrate6_6_4(states, a, mu, ind1, ind2):
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
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = matrix_derive3_numba(coolquad, dt) 
        dt2 = dt2[5:-5]
        dt3 = dt3[5:-5]
        int_time = int_time[5:-5]
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        #print((states[-1,0] - states[0,0]), (int_time[-1] - int_time[0]), "wa")
        dE = mu*mu*np.sum(dedt*div)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        #print(dE, dLx, dLy, dLz)
        #print(states[-1,0] - states[0,0])
        return np.array([dE, dLx, dLy, dLz])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

def peters_integrate6_6_4_2(states, a, mu, ind1, ind2):
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
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt1 = np.gradient(coolquad, dt, axis=0)
        dt2 = np.gradient(dt1, dt, axis=0)
        dt3 = np.gradient(dt2, dt, axis=0)
        dt2 = dt2[5:-5]
        dt3 = dt3[5:-5]
        int_time = int_time[5:-5]

        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        #print((states[-1,0] - states[0,0]), (int_time[-1] - int_time[0]), "wa")
        dE = mu*mu*np.sum(dedt*div)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        #print(dE, dLx, dLy, dLz)
        #print(states[-1,0] - states[0,0])
        return np.array([dE, dLx, dLy, dLz])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

from scipy.signal import savgol_filter
def peters_integrate6_6_4_3(states, a, mu, ind1, ind2):
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
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2 = savgol_filter(coolquad, window_length=21, polyorder=3, deriv=2, delta=dt, axis=0)
        dt3 = savgol_filter(coolquad, window_length=21, polyorder=3, deriv=3, delta=dt, axis=0)
        dt2 = dt2[5:-5]
        dt3 = dt3[5:-5]
        int_time = int_time[5:-5]
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        #print((states[-1,0] - states[0,0]), (int_time[-1] - int_time[0]), "wa")
        dE = mu*mu*np.sum(dedt*div)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        #print(dE, dLx, dLy, dLz)
        #print(states[-1,0] - states[0,0])
        return np.array([dE, dLx, dLy, dLz])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

from scipy.interpolate import UnivariateSpline
def peters_integrate6_6_4_4(states, a, mu, ind1, ind2):
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
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2 = np.zeros_like(coolquad)
        dt3 = np.zeros_like(coolquad)
        for i in range(3):
            for j in range(i+1):
                spl = UnivariateSpline(int_time, coolquad[:, i, j], s=10)

                d2 = spl.derivative(2)
                d3 = spl.derivative(3)

                dt2[:, i, j] = d2(int_time)
                dt3[:, i, j] = d3(int_time)
                if i != j:
                    dt2[:, j, i] = d2(int_time)
                    dt3[:, j, i] = d3(int_time)

        dt2 = dt2[5:-5]
        dt3 = dt3[5:-5]
        int_time = int_time[5:-5]
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        #print((states[-1,0] - states[0,0]), (int_time[-1] - int_time[0]), "wa")
        dE = mu*mu*np.sum(dedt*div)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        #print(dE, dLx, dLy, dLz)
        #print(states[-1,0] - states[0,0])
        return np.array([dE, dLx, dLy, dLz])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
def peters_integrate6_6_4_5(states, a, mu, ind1, ind2):
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
        dt2 = dt2[5:-5]
        dt3 = dt3[5:-5]
        int_time = int_time[5:-5]

        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        #print((states[-1,0] - states[0,0]), (int_time[-1] - int_time[0]), "wa")
        dE = mu*mu*np.sum(dedt*div)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        #print(dE, dLx, dLy, dLz)
        #print(states[-1,0] - states[0,0])
        return np.array([dE, dLx, dLy, dLz])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

def peters_integrate6_6_5(states, a, mu, ind1, ind2):
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
    # CHECK AND SEE IF THIS IS ACTUALLY ANY GOOD
    if (ind2 - ind1 - 10) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0] - states[0,0]
        int_sphere, int_time = interpolate(sphere, time, False)
        dt = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        coolquad = traceless_quad(quad)
        dt2, dt3 = matrix_derive3_numba(coolquad, dt) 

        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        #print((states[-1,0] - states[0,0]), (int_time[-1] - int_time[0]), "wa")
        dE = mu*mu*np.trapz(dedt, x=int_time)
        dLx, dLy, dLz = mu*mu*np.trapz(dldt, x=int_time, axis=0)
        #print(dE, dLx, dLy, dLz)
        #print(states[-1,0] - states[0,0])
        return np.array([dE, dLx, dLy, dLz])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])

def peters_integrate_differential(states, a, mu, cons, state, ind1, ind2):
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
        E0, L0, C0 = cons
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0] - states[0,0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = matrix_derive3_numba(coolquad, dt) 
        dt2 = dt2[5:-5]
        dt3 = dt3[5:-5]
        int_time = int_time[5:-5]
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        P = E0*(int_sphere[5:-5,0]**2 + a*a) - a*L0
        D = int_sphere[5:-5,0]*(int_sphere[5:-5,0] - 2) + a*a
        dcdt_radial = ((2*P*(int_sphere[5:-5,0]**2 + a*a)/D - 2*a*(a*E0 - L0))*dedt + (-2*P*a/D + 2*(a*E0 - L0))*dldt[:,2])*div*mu*mu
        dE = mu*mu*np.sum(dedt*div)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dcons = np.array([dE, dLz, np.sum(dcdt_radial)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])])
    else:
        dcons = np.array([0.0, 0.0, 0.0])

    E, L, C = E0 + dcons[0], L0 + dcons[1], C0 + dcons[2]
    C = max(0.0, C)
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def peters_integrate_differential2(states, a, mu, cons, state, ind1, ind2):
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
        E0, L0, C0 = cons
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0] - states[0,0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = matrix_derive3_numba(coolquad, dt) 
        dt2 = dt2[5:-5]
        dt3 = dt3[5:-5]
        int_time = int_time[5:-5]
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        dE = mu*mu*np.sum(dedt*div)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])

        if a == 0:
            z2 = C0/(L0**2 + C0)
        else:
            A = (a**2)*(1 - E0**2)
            sig = A + L0**2 + C0
            z2 = (sig - (sig**2 - 4*A*C0)**(1/2))/(2*A)
        x2 = 1 - z2
        x = np.sign(L0)*np.sqrt(x2)
        dC0 = -2*a*a*z2*E0*dE + 2*z2*L0/(1 - z2)*dLz
        dC_dx = - 2*(a*a*x*(1 - E0*E0) + L0*L0/(x**3))
        dx = dC0 + 2*(1 - x2)*a*a*E0*dE - 2*L0*dLz*(1 - x2)/x2
        dx /= -2*(x*(a*a*(1 - E0*E0) + L0*L0/x2) + L0*L0*(1 - x2)/(x**3))
        dC = dC0 + dC_dx*dx
        dcons = np.array([dE, dLz, dC])
    else:
        dcons = np.array([0.0, 0.0, 0.0])

    E, L, C = E0 + dcons[0], L0 + dcons[1], C0 + dcons[2]
    C = max(0.0, C)
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def peters_integrate_differential3(states, a, mu, cons, state, ind1, ind2):
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
        E0, L0, C0 = cons
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0] - states[0,0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = matrix_derive3_numba(coolquad, dt) 
        dt2 = dt2[5:-5]
        dt3 = dt3[5:-5]
        int_time = int_time[5:-5]
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        dE = mu*mu*np.sum(dedt*div)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])

        if a == 0:
            z2 = C0/(L0**2 + C0)
        else:
            A = (a**2)*(1 - E0**2)
            sig = A + L0**2 + C0
            z2 = (sig - (sig**2 - 4*A*C0)**(1/2))/(2*A)
        x2 = 1 - z2
        x = np.sign(L0)*np.sqrt(x2)
        dC0 = -2*a*a*z2*E0*dE + 2*z2*L0/(1 - z2)*dLz
        dC_dx = - 2*(a*a*x*(1 - E0*E0) + L0*L0/(x**3))
        dx = dC0 + 2*(1 - x2)*a*a*E0*dE - 2*L0*dLz*(1 - x2)/x2
        dx /= -2*(x*(a*a*(1 - E0*E0) + L0*L0/x2) + L0*L0*(1 - x2)/(x**3))
        dC_theta = dC0 + dC_dx*dx
        P = E0*(int_sphere[5:-5,0]**2 + a*a) - a*L0
        D = int_sphere[5:-5,0]*(int_sphere[5:-5,0] - 2) + a*a
        dcdt_radial = ((2*P*(int_sphere[5:-5,0]**2 + a*a)/D - 2*a*(a*E0 - L0))*dedt + (-2*P*a/D + 2*(a*E0 - L0))*dldt[:,2])*div*mu*mu
        dC = (np.sum(dcdt_radial)*z2 + dC_theta*x2)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dcons = np.array([dE, dLz, dC])
    else:
        dcons = np.array([0.0, 0.0, 0.0])
        
    E, L, C = E0 + dcons[0], L0 + dcons[1], C0 + dcons[2]
    C = max(0.0, C)
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def peters_integrate_differential4(states, a, mu, cons, state, ind1, ind2):
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
        E0, L0, C0 = cons
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0] - states[0,0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize_njit(int_sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = matrix_derive3_numba(coolquad, dt) 
        dt2 = dt2[5:-5]
        dt3 = dt3[5:-5]
        int_time = int_time[5:-5]
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = compute_dldt(dt2, dt3)
        dE = mu*mu*np.sum(dedt*div)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])

        if a == 0:
            z2 = C0/(L0**2 + C0)
        else:
            A = (a**2)*(1 - E0**2)
            sig = A + L0**2 + C0
            z2 = (sig - (sig**2 - 4*A*C0)**(1/2))/(2*A)
        x2 = 1 - z2
        x = np.sign(L0)*np.sqrt(x2)
        dC0 = -2*a*a*z2*E0*dE + 2*z2*L0/(1 - z2)*dLz
        dC_dx = - 2*(a*a*x*(1 - E0*E0) + L0*L0/(x**3))
        dx = dC0 + 2*(1 - x2)*a*a*E0*dE - 2*L0*dLz*(1 - x2)/x2
        dx /= -2*(x*(a*a*(1 - E0*E0) + L0*L0/x2) + L0*L0*(1 - x2)/(x**3))
        dC_theta = dC0 + dC_dx*dx
        P = E0*(int_sphere[5:-5,0]**2 + a*a) - a*L0
        D = int_sphere[5:-5,0]*(int_sphere[5:-5,0] - 2) + a*a
        dcdt_radial = ((2*P*(int_sphere[5:-5,0]**2 + a*a)/D - 2*a*(a*E0 - L0))*dedt + (-2*P*a/D + 2*(a*E0 - L0))*dldt[:,2])*div*mu*mu
        dC = (np.sum(dcdt_radial)*(1 - (x2**10)) + dC_theta*(x2**10))*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dcons = np.array([dE, dLz, dC])
    else:
        dcons = np.array([0.0, 0.0, 0.0])
        
    E, L, C = E0 + dcons[0], L0 + dcons[1], C0 + dcons[2]
    C = max(0.0, C)
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

import numpy as np
from numba import njit, prange

@njit
def lagrange_derivative_numba(data, t, stencil=5):
    """
    Second or third derivative using Lagrange interpolation on nonuniform timesteps.

    Parameters
    ----------
    data : (N, 3, 3) array
        Trajectory data (or previous derivative) per time step
    t : (N,) array
        Time array (nonuniform)
    stencil : int, odd
        Number of points in Lagrange stencil

    Returns
    -------
    deriv : (N-2*half, 3, 3) array
        Derivative (second or third) of input
    """
    N = data.shape[0]
    half = stencil // 2
    deriv = np.zeros((N - 2*half, 3, 3))
    
    for k in range(half, N - half):
        for i in range(3):
            for j in range(3):
                y = np.zeros(stencil)
                x = np.zeros(stencil)
                for s in range(stencil):
                    idx = k - half + s
                    x[s] = t[idx]
                    y[s] = data[idx, i, j]
                # Lagrange derivative
                d = 0.0
                for m in range(stencil):
                    prod = 1.0
                    for n in range(stencil):
                        if n != m:
                            prod *= (x[m] - x[n])
                    for n in range(stencil):
                        if n != m:
                            sum_term = 1.0
                            for l in range(stencil):
                                if l != m and l != n:
                                    sum_term *= (0.0 - x[l])  # derivative at 0? use t=0 offset
                            d += y[m] * sum_term / prod
                deriv[k-half, i, j] = d
    return deriv


@njit(parallel=True)
def compute_dldt_numba_woo(dt2, dt3):
    N = dt2.shape[0]
    dldt = np.zeros((N, 3))
    # Levi-Civita tensor
    levciv = np.zeros((3,3,3))
    levciv[0,1,2] = levciv[1,2,0] = levciv[2,0,1] = 1
    levciv[0,2,1] = levciv[1,0,2] = levciv[2,1,0] = -1

    for l in prange(N):
        for i in range(3):
            acc = 0.0
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        acc += levciv[i,j,k] * dt2[l,j,m] * dt3[l,k,m]
            dldt[l,i] = -2.0/5.0 * acc
    return dldt


def peters_integrate_numba(states, a, mu, stencil=5):
    """
    Numba-accelerated Peters flux for nonuniform timesteps.
    """
    states = np.array(states)
    sphere = states[:, 1:4]
    time = states[:, 0] - states[0,0]

    quad = trace_ortholize_njit(sphere, a)
    coolquad = traceless_quad(quad)

    # derivatives
    dt2 = lagrange_derivative_numba(coolquad, time, stencil=stencil)
    dt3 = lagrange_derivative_numba(dt2, time[stencil//2:-(stencil//2)], stencil=stencil)

    # fluxes
    N = dt3.shape[0]
    dedt = np.zeros(N)
    for idx in range(N):
        t3 = dt3[idx]
        dedt[idx] = -1/5 * (np.sum(t3**2) - (1/3) * np.sum(np.diagonal(t3)**2))
    
    dldt = compute_dldt_numba_woo(dt2, dt3)

    # integrate using trapezoid rule
    t_int = time[stencil-1:-(stencil-1)]
    dE = mu*mu * np.trapz(dedt, x=time[stencil-1:-(stencil-1)])
    dLx, dLy, dLz = mu*mu * np.trapz(dldt, x=time[stencil//2:-(stencil//2)], axis=0)

    return np.array([dE, dLx, dLy, dLz])

def peters_integrate6_7(states, a, mu, ind1, ind2):
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
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1 - 10) > 2:
        states = np.array(states)
        sphere, time = states[5:-5, 1:4], states[5:-5, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(int_sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad - (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2, dt3 = matrix_derive3(coolquad, int_time, int_time)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
            #scale both changes to make up for the bits that got cut off
        #print([dE, dLx, dLy, dLz])
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
        #print(len(time), time[-1]-time[0], dE, np.linalg.norm([dLx, dLy, dLz]))
        #print([dE, dLx, dLy, dLz], "6_5")
        return np.array([dE, dLx, dLy, dLz])
    else:
        #print("gorp", "6_5")
        return np.array([0.0, 0.0, 0.0, 0.0])
    #return quad

def peters_integrate6_7_2(states, a, mu, ind1, ind2):
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
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1 - 10) > 2:
        states = np.array(states)
        sphere, time = states[5:-5, 1:4], states[5:-5, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(int_sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad - (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2, dt3 = matrix_derive3(coolquad, int_time, int_time)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu*mu*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu*mu*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
            #scale both changes to make up for the bits that got cut off
        #print([dE, dLx, dLy, dLz])
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
        #print(len(time), time[-1]-time[0], dE, np.linalg.norm([dLx, dLy, dLz]))
        #print([dE, dLx, dLy, dLz], "6_5")
        return np.array([dE, dLx, dLy, dLz])
    else:
        #print("gorp", "6_5")
        return np.array([0.0, 0.0, 0.0, 0.0])
    #return quad

def peters_integrate7(states, a, mu, ind1, ind2):
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
    path = np.array(states)
    #print(path)
    int_time = np.arange(int(path[0, 0]), int(path[-1, 0] + 1))
    #print(int_time)
    path = np.transpose([np.interp(int_time, path[:,0], path[:,1]),
                         np.interp(int_time, path[:,0], path[:,2]),
                         np.interp(int_time, path[:,0], path[:,3])])
    cartpath = np.transpose([path[:,0]*np.sin(path[:,1])*np.cos(path[:,2]), 
                             path[:,0]*np.sin(path[:,1])*np.sin(path[:,2]),
                             path[:,0]*np.cos(path[:,1])])
    quad = mu*np.einsum("ij, ik -> ijk", cartpath, cartpath)
    dt2 = matrix_derive(quad, int_time, 2)
    dt3 = matrix_derive(quad, int_time, 3)
    levciv = np.array([[[0, 0, 0],   #Levi-civita tensor - np.array([[[int(not((i+1)*(j+1)*(k+1)-6))*(int(j-i==1)*2-1) for k in range(3)] for j in range(3)] for i in range(3)])
                        [0, 0, 1],
                        [0, -1, 0]],
                       [[0, 0, -1],
                        [0, 0, 0],
                        [1, 0, 0]],
                       [[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 0]]])
    #dE = -(1/2)*np.sum(np.transpose([[dt3[:,i,j]**2 - (1/3)*dt3[:,i,i]*dt3[:,j,j] for j in range(3)] for i in range(3)]))
    #dL = np.transpose([[[[levciv[i,j,k]*dt2[:,j,m]*dt3[:,k,m] for m in range(3)] for k in range(3)] for j in range(3)] for i in range(3)])
    dE = (-1/5)*(np.einsum("ijk, ijk", dt3, dt3) - (1/3)*np.einsum("ijj, ikk", dt3, dt3))
    dL = -(2/5)*(np.einsum("ijk, ljm, lkm -> i", levciv, dt2, dt3))
    return np.array([dE, dL[0], dL[1], dL[2]])
    #return quad
    
def peters_integrate8(states, a, mu, ind1, ind2):
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
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = mu*trace_ortholize(int_sphere)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad #- (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2 = matrix_derive(coolquad, int_time, 2)
        dt3 = matrix_derive(coolquad, int_time, 3)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk -> i', dt3, dt3) - (1/3)*np.einsum('ijj, ikk -> i', dt3, dt3))
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)
    return np.array([dE, dLx, dLy, dLz])

def mat_derv2(data, time, degree):
    new = np.copy(data)
    devtime = np.copy(time)
    for i in range(degree):
        new = np.diff(new, axis = 0)
        devtime = 0.5*(devtime[:-1] + devtime[1:])
    final = np.transpose([[np.interp(time, devtime, new[:,a,b]) for b in range(3)] for a in range(3)])
    #print(data)
    #print(final)
    #print(np.shape(data))
    #print(np.shape(final))
    return final

def peters_integrate9(states, a, mu, ind1, ind2):
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
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(int_sphere)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad - (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2 = mat_derv2(coolquad, int_time, 2)
        dt3 = mat_derv2(coolquad, int_time, 3)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*np.einsum('ijk,ijk ->i', dt3, dt3)
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
    return np.array([dE, dLx, dLy, dLz])
    #return quad

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

def new_recalc_state5(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
            
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    #dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    dC = 2*z2*(L0*dLz/(1-z2) - (a**2)*E0*dE)  
    if np.isnan(dC):
        dC = -2*z2*(a**2)*E0*dE 

    # Step 4
    E, L = E0 + dE, L0 + dLz*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
    
    # Step 5
    C = C0 + dC
    #print(dE, dLz, dC)
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    test = max(np.roots(np.polyder(potent)))
    count = 0
    #while (np.polyval(potent, test) < 0.0):
    #    count += 1
    #    dR = -np.polyval(potent, test)
    #    E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
    #    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    #    test = max(np.roots(np.polyder(potent)))
    #print(count)
    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state6(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
            
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    #dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    dC = 2*z2*(L0*dLz/(1-z2) - (a**2)*E0*dE)  
    if np.isnan(dC):
        dC = -2*z2*(a**2)*E0*dE 

    # Step 4
    E, L = E0 + dE, L0 + dLz*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
    
    # Step 5
    C = C0 + dC
    #print(dE, dLz, dC)
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    test = max(np.roots(np.polyder(potent)))
    count = 0
    #while (np.polyval(potent, test) < 0.0):
    #    count += 1
    #    dR = -np.polyval(potent, test)
    #    E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
    #    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    #    test = max(np.roots(np.polyder(potent)))
    #print(count)
    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state7(cons, con_derv, state, a, loop=0):
    metric, chris = kerr(state, a)
    E0 = -np.matmul(metric, state[4:])[0]
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
    tetrad = np.linalg.solve(tet2kerr, state[4:])
    tetcart = np.array([*tetrad[:2], tetrad[3], -tetrad[2]])
    vel, cartpos = tetcart[1:]/tetcart[0], np.array([r*sint*cosp, r*sint*sinp, r*cost])
    L0 = np.cross(cartpos, vel)
    A, eps = np.zeros((4,3)), np.max(abs(con_derv))
    #print("---")
    #print(con_derv)
    #print(eps)
    if eps == 0.0:
        eps = 1e-7
    #print(eps)
    
    def getNewCons(j):
        intvel = np.array([0.0,0.0,0.0])
        intvel[j] += eps
        intL, gamma = np.cross(cartpos, intvel+vel), 1/np.sqrt(1 - np.linalg.norm(intvel + vel)**2)
        inttetrad = gamma*np.array([1, intvel[0], -intvel[2], intvel[1]])
        intkerr = np.matmul(tet2kerr, inttetrad)
        intE = -np.matmul(metric, intkerr)[0]
        return np.array([intE - E0, *(intL - L0)])/eps
    
    A[:,0], A[:,1], A[:,2] = getNewCons(0), getNewCons(1), getNewCons(2)
    try:
        #print("org")
        bigD = np.linalg.inv(np.matmul(np.transpose(A), A))
        dvel = np.matmul(bigD, np.matmul(np.transpose(A), con_derv[:4]))
    except:
        #print(A)
        #print(np.matmul(np.transpose(A), A))
        dvel = np.linalg.solve(np.matmul(np.transpose(A), A), np.matmul(np.transpose(A), con_derv[:4]))
    #print(dvel)
    newvel = vel + dvel
    gamma = 1/np.sqrt(1 - np.linalg.norm(newvel)**2)
    newtetrad = gamma*np.array([1, newvel[0], -newvel[2], newvel[1]])
    newkerr = np.matmul(tet2kerr, newtetrad)
    holdstate = np.array([*state[0:4], *newkerr])
    newE = -np.matmul(metric, newkerr)[0]        #initial energy
    newLz = np.matmul(metric, newkerr)[3]        #initial angular momentum
    newQ = np.matmul(np.matmul(kill_tensor(holdstate, a), newkerr), newkerr)
    newC = newQ - (a*newE - newLz)**2  
    #print(con_derv)
    #print([newE - E0, *(np.cross(cartpos, newvel) - L0)])
    #print(con_derv)
    #print(con_derv - np.array([newE - E0, *(np.cross(cartpos, newvel) - L0)]))
    #print(np.linalg.norm(con_derv - np.array([newE - E0, *(np.cross(cartpos, newvel) - L0)])))
    #print([newE, newLz, newC])
    #print("-----")
    #print(np.linalg.norm(con_derv - np.array([newE - E0, *(np.cross(cartpos, newvel) - L0)])))
    #if loop < 10 and np.linalg.norm(con_derv - np.array([newE - E0, *(np.cross(cartpos, newvel) - L0)])) >= 1e-15:
        #print("HELLO WAIT WHAT")
        #print(np.linalg.norm(con_derv - np.array([newE - E0, *(np.cross(cartpos, newvel) - L0)])))
        #holdstate, [newE, newLz, newC] = new_recalc_state7([newE, newLz, newC], np.array([newE - E0, *(np.cross(cartpos, newvel) - L0)]), holdstate, a, loop=loop+1)
    return holdstate, [newE, newLz, newC]

def new_recalc_state8(cons, con_derv, state, a):
    metric, chris = kerr(state, a)
    E0 = -np.matmul(metric, state[4:])[0]
    
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
    
    tetrad = np.linalg.solve(tet2kerr, state[4:])
    tetcart = np.array([*tetrad[:2], tetrad[3], -tetrad[2]])
    vel, cartpos = tetcart[1:]/tetcart[0], np.array([r*sint*cosp, r*sint*sinp, r*cost])
    hold = con_derv
    L0 = np.cross(cartpos, vel)
    A = np.zeros((4,3))
    loop = -1
    #print(con_derv)
    sol_list = [vel]
    hold_derv = con_derv
    target = np.array([E0, *L0]) + con_derv
    thing = 1e10
    tog = 0
    #print(target)
    while thing > 6e-15:
        thing = np.linalg.norm(hold)
        eps = 1e-8 #np.sqrt(abs(max(hold)))

        def getNewCons(j):
            intvel = np.array([0.0,0.0,0.0])
            intvel[j] += eps
            intL, gamma = np.cross(cartpos, intvel+vel), 1/np.sqrt(1 - np.linalg.norm(intvel + vel)**2)
            inttetrad = gamma*np.array([1, intvel[0], -intvel[2], intvel[1]])
            intkerr = np.matmul(tet2kerr, inttetrad)
            intE = -np.matmul(metric, intkerr)[0]
            return np.array([intE - E0, *(intL - L0)])/eps
    
        A[:,0], A[:,1], A[:,2] = getNewCons(0), getNewCons(1), getNewCons(2)
        dvel = np.linalg.solve(np.matmul(np.transpose(A), A), np.matmul(np.transpose(A), hold_derv[:4]))
        #print(dvel)
        vel = vel + dvel + tog*np.random.randn(3)*1e-17
        sol_list.append(vel)
        L1 = np.cross(cartpos, vel)
        gamma = 1/np.sqrt(1 - np.linalg.norm(vel)**2)
        newtetrad = gamma*np.array([1, vel[0], -vel[2], vel[1]])
        newkerr = np.matmul(tet2kerr, newtetrad)
        E1 = -np.matmul(metric, newkerr)[0]
        hold = (np.array([E1 - E0, *(L1 - L0)]) - con_derv)
        loop += 1
        #print(hold, np.linalg.norm(hold))
        hold_derv = target - np.array([E1, *L1])
        #if np.linalg.norm(hold) > thing:
        #    tog = 1
            #print("KACHOW")
        #else:
        #    tog = 0
        if loop > 100:
            #print("oop!")
            #print(con_derv)
            #print(np.array([E1 - E0, *(L1 - L0)]))
            #print(hold)
            break
    #print(np.array([E1 - E0, *(L1 - L0)]))
    #print(100*(np.array([E1 - E0, *(L1 - L0)]) - con_derv)/con_derv)
    newstate = np.array([*state[0:4], *newkerr])
    newLz = np.matmul(metric, newkerr)[3]        #initial angular momentum
    newQ = np.matmul(np.matmul(kill_tensor(newstate, a), newkerr), newkerr)
    newC = newQ - (a*E1 - newLz)**2 
    return newstate, [E1, newLz, newC]

def new_recalc_state8b(cons, con_derv, state, a):
    metric, chris = kerr_2(state, a)
    
    # make orthonormal vectors
    basis = [0,0,0]
    for i in range(3):
        vec = state[4:].copy()
        vec[i+1] = 0.0
        vec = vec - ((metric @ vec) @ state[4:]) * state[4:]
        vec = vec/np.sqrt(-((metric @ vec) @ vec))
        #disc = (2*metric[0,3]*vec[3])**2 - 4 * metric[0,0] * (np.sum([metric[j, j]*vec[j]*vec[j] for j in range(1, 4)]) + 1)
        #vec[0] = (1/metric[0,0])*(-2*metric[0,3]*vec[3] - np.sqrt(disc))
        basis[i] = vec

    J = np.zeros((2,3))
    eps = 1/(state[1]**6)
    for i in range(3):
        pert_state = state[4:] + eps * basis[i]
        # no renormalization
        dE = -(metric @ pert_state)[0] + (metric @ state[4:])[0]
        dLz =  (metric @ pert_state)[3] - (metric @ state[4:])[3]
        J[:, i] = [dE/eps, dLz/eps]

    coeffs = np.linalg.pinv(J) @ np.array([con_derv[0], con_derv[-1]])
    new_vel = state[4:] + sum(coeffs[i] * basis[i] for i in range(3))
    new_vel = new_vel/np.sqrt(-((metric @ new_vel) @ new_vel))

    new_state = np.array([*state[:4], *new_vel])
    stuff = np.matmul(metric, new_state[4:])
    newE, newLz, newQ = -stuff[0], stuff[3], np.matmul(np.matmul(kill_tensor(new_state, a), new_state[4:]), new_state[4:])
    newC = newQ - (a*newE - newLz)**2
    return new_state, [newE, newLz, newC]

def new_recalc_state8c(cons, con_derv, state, a):
    """
    Enforces dE, dLz exactly while minimizing |dC| to linear order.
    Uses a true orthonormal spatial triad in the particle rest frame.
    """
    metric, chris = kerr_2(state, a)
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
    tetrad = np.linalg.solve(kerr2tet, state[4:])
    u = tetrad[1:]/tetrad[0].copy()

    A = np.zeros((3,3))
    eps = 1e-7
    for i in range(9):
        div = u.copy()
        div[i//3] += eps
        gamma = 1/np.sqrt(1 - np.dot(div, div))
        new_tet = np.array([gamma, *(gamma*div)])
        new_vel = tet2kerr @ new_tet
        stuff = metric @ new_vel
        if i%3 == 0:
            val = -stuff[0]
        if i%3 == 1:
            val = stuff[3]
        else:
            newE, newLz = -stuff[0], stuff[3]
            val = np.matmul(np.matmul(kill_tensor([*state[:4], *new_vel], a), new_vel), new_vel)
            val = val - (a*newE - newLz)**2
        A[i%3, i//3] = (cons[i%3] - val)/eps
    
    # Build reduced system (2 equations)
    A12 = A[:2, :]          # first two rows
    b12 = np.array([con_derv[0], con_derv[-1]])

    # Minimum-norm solution
    new_u = u + np.linalg.pinv(A12) @ b12
    print(u)
    print(new_u)
    gamma = 1/np.sqrt(1 - np.dot(new_u, new_u))
    new_tet = np.array([gamma, *(gamma*new_u)])
    new_vel = tet2kerr @ new_tet
    new_state = np.array([*state[:4], *new_vel])
    print(state[4:])
    print(new_vel)
    stuff = np.matmul(metric, new_state[4:])
    newE, newLz, newQ = -stuff[0], stuff[3], np.matmul(np.matmul(kill_tensor(new_state, a), new_state[4:]), new_state[4:])
    newC = newQ - (a*newE - newLz)**2
    print(cons)
    print(np.array([newE, newLz, newC]))
    print(cons - np.array([newE, newLz, newC]))
    print(con_derv)
    print("************")
    return new_state, [newE, newLz, newC]

def new_recalc_state8d(cons, con_derv, state, a, eps=1e-7, svd=False):
    """
    Enforces dE, dLz exactly while minimizing |dC| to linear order.
    Uses a true orthonormal spatial triad in the particle rest frame.
    """
    metric, chris = kerr_2(state, a)
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
    v = tetrad[1:]/tetrad[0].copy()
    fmom = np.matmul(metric, state[4:])
    L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    cart_state = sph2cart(state, a)
    L_dir = np.cross(cart_state[1:4], cart_state[5:])
    L_vec = L*L_dir/np.linalg.norm(L_dir)

    A = np.zeros((4,3))
    for i in range(3):
        div = np.zeros(3)
        div[i] += eps
        gamma = 1/np.sqrt(1 - np.dot(v + div, v + div))
        new_tet = np.array([gamma, *(gamma*(v + div))])
        new_vel = tet2kerr @ new_tet
        new_fmom = metric @ new_vel
        A[i%4, i] = (-new_fmom[0] - cons[0])/eps
        new_L = np.sqrt(new_fmom[2]**2 + (new_fmom[3]**2)/(np.sin(theta)**2))
        new_cart_state = sph2cart([*state[:4], *new_vel], a)
        new_L_dir = np.cross(new_cart_state[1:4], new_cart_state[5:])
        new_L_vec = new_L * new_L_dir/np.linalg.norm(new_L_dir)
        A[1:, i] = (new_L_vec[i] - L_vec[i])/eps
            
    try:
        if svd:
            true_div = np.linalg.lstsq(A, con_derv, rcond=None)[0]
        else:
            true_div = np.linalg.inv(A.T @ A) @ (A.T @ con_derv)
        new_v = v + true_div
        true_gamma = 1/np.sqrt(1 - np.dot(new_v, new_v))
        true_tet = np.array([true_gamma, *(true_gamma * new_v)])
        true_vel = tet2kerr @ true_tet
        new_state = np.array([*state[:4], *true_vel])
        true_fmom = metric @ true_vel
        newE, newLz, newQ = -true_fmom[0], true_fmom[3], np.matmul(np.matmul(kill_tensor(new_state, a), new_state[4:]), new_state[4:])
        newC = newQ - (a*newE - newLz)**2
        return new_state, [newE, newLz, newC]
    except Exception as e:
        return np.nan * np.arange(8), np.nan * np.arange(3)

def new_recalc_state9(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    theta = state[2]
    fmom = np.matmul(metric, state[4:])
    L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
    cosz, sinz = np.sqrt(min(1.0, np.abs(z2))), np.sqrt(1 - min(1.0, np.abs(z2)))
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    #dC = 2*z2*(L0*dLz/(1-z2) - (a**2)*E0*dE)  
    #if np.isnan(dC):
    #    dC = -2*z2*(a**2)*E0*dE 
    dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(state[2]))**2)*(1 - E0**2)*dE)# if C0 != 0 else 0.0
        #From glamp A3, thetadot term goes away because I don't change position!

    #dC = 2*(L*dL_vec - L0*dLz - (a**2)*cosz*(sinz*state[6]*(1 - E0**2) + cosz*E0*dE))
    #dC = 2*(L0 - a*E0*np.sin(theta)**2)*(dLz - a*dE*np.sin(theta)**2 - 2*a*E0*np.sin(theta)*np.cos(theta)*state[6])*np.sin(theta)**2
    #dC -= 2*np.sin(theta)*np.cos(theta)*(L0 - a*E0*np.sin(theta)**2)*state[6]
    #dC /= np.sin(theta)**4
    #dC -=2*(a**2)*np.cos(theta)*np.sin(theta)*state[6]
    #dC += 2*(state[1]**2 + (a*np.cos(theta))**2)*(2*state[1]*state[5] - 2*(a**2)*np.sin(theta)*np.cos(theta)*state[6])*(state[6]**2)

    # Step 4
    E, L = E0 + dE, L0 + dLz    #*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                        #I actually don't think I need to make that correction
    
    # Step 5
    C = C0 + dC
    
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    #test = max(np.roots(np.polyder(potent)))
    count = 0
    #Make sure point we're AT is viable, the rest is maybe lame?
    '''
    while (np.polyval(potent, state[1]) < 0.0):
        dR = -np.polyval(potent, test)
        #scale all the variables
        #E -= dE*(1e-5)
        L += dLz*(1e-5)
        C += dC*(1e-5)
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
        count += 1
        if count == 1000:
            print("gotdamn")
        if count >= 1e6:
            break
    if count > 0:
        print("L, C adjust: %s, %s (%s times)"%( count*dLz*1e-5, count*dC*1e-5, count))
    '''
    # Step 6
    #print(E, L, C)
    #print(dE, dLz*np.sign(L0), dC)
    #print(cosz, sinz, np.abs(z2))
    #print(E0, L0, C0)
    #print(E, L, C)
    #print("yo!", E0, L0, C0)
    #print("to!", E, L, C)
    #print("mo!", con_derv)
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

import numpy as np
from scipy.optimize import brentq

import numpy as np
from scipy.optimize import brentq

def compute_J_theta(E, Lz, C, a, N=32,
                    eps_C=1e-14,
                    eps_denom=1e-14):
    """
    Computes the polar action J_theta for Kerr geodesics using the convention
    that C >= 0 and C = 0 corresponds to equatorial orbits.
    """

    # -----------------------------
    # 0. Equatorial shortcut
    # -----------------------------
    # Under your convention: C = 0 ⇔ equatorial ⇔ J_theta = 0
    if C <= eps_C:
        return 0.0

    # -----------------------------
    # 1. Polar potential in z = cos(theta)
    # -----------------------------
    # Theta(z) = C - z^2 [ a^2 (1 - E^2) + Lz^2 / (1 - z^2) ]
    def Theta_z(z):
        denom = 1.0 - z*z
        if denom <= 0.0:
            return -np.inf
        return C - z*z * (a*a*(1.0 - E*E) + Lz*Lz / denom)

    # -----------------------------
    # 2. Find polar turning point z_max
    # -----------------------------
    # Physical root lies in z ∈ [0, 1)
    z_lo = 0.0
    z_hi = 1.0 - 1e-12

    f_lo = Theta_z(z_lo)
    f_hi = Theta_z(z_hi)

    if f_lo < 0.0:
        # Should not happen for physical C >= 0
        return 0.0

    if f_hi > 0.0:
        # Nearly polar orbit: turning point extremely close to z = 1
        z_max = z_hi
    else:
        z_max = brentq(Theta_z, z_lo, z_hi)

    # -----------------------------
    # 3. Gauss–Legendre quadrature
    # -----------------------------
    nodes, weights = np.polynomial.legendre.leggauss(N)

    # Map nodes from [-1, 1] → [0, z_max]
    z = 0.5 * z_max * (nodes + 1.0)
    w = 0.5 * z_max * weights

    denom = 1.0 - z*z
    denom = np.maximum(denom, eps_denom)

    integrand = np.sqrt(np.maximum(
        C - z*z * (a*a*(1.0 - E*E) + Lz*Lz / denom),
        0.0
    )) / np.sqrt(denom)

    integral = np.sum(w * integrand)

    # -----------------------------
    # 4. Symmetry factor
    # -----------------------------
    # J_theta = (2 / pi) ∫ p_theta dθ
    return 2.0 * integral / np.pi

def solve_for_C(E, Lz, Jtheta0, a,
                Cmin=0.0,
                C_init=1.0,
                max_expand=60):
    """
    Solve J_theta(E, Lz, C) = Jtheta0 for C >= 0.
    Safe for equatorial, polar, and near-polar orbits.
    """

    # Equatorial shortcut
    if Jtheta0 <= 1e-14:
        return 0.0

    def fC(C):
        return compute_J_theta(E, Lz, C, a) - Jtheta0

    C_lo = Cmin
    f_lo = fC(C_lo)

    if f_lo > 0:
        # Numerical noise, but physically C=0
        return 0.0

    C_hi = C_init
    f_hi = fC(C_hi)

    n = 0
    while f_hi < 0 and n < max_expand:
        C_hi *= 2.0
        f_hi = fC(C_hi)
        n += 1

    if f_hi < 0:
        raise RuntimeError(
            "Failed to bracket C root — check J_theta consistency"
        )

    return brentq(fC, C_lo, C_hi)

def new_recalc_state9n(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    E0, L0, C0 = cons

    # Step 2
    # z = cos(thet_min)
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        sig = A + L0**2 + C0
        z2 = (sig - (sig**2 - 4*A*C0)**(1/2))/(2*A)
    z = np.sqrt(min(1.0, np.abs(z2)))
    dE, dLx, dLy, dLz = con_derv[:4]

    
    # Near-circular orbits! This feels a little like cheating?
    # The theta-potential becomes useless for polar orbits
    # And it's very easy for near-circular orbits to go out of wack
    # so in both cases we switch to the radial potential, rearrange that to solve for C
    # then get dC = (dC_dE)*dE + (dC_dLz)*dLz + (dC_dr)*dr
    # the especially cheaty part is saying (dC_dr) = (dC_dR)(dR_dr), and that since
    # we're doing this calculation at roughly r ~ potential minimum, (dR_dr) ~ 0 -> (dC_dr) ~ 0
    # dr would also be fairly small here (it's not r_dot, but the change in potential minimum, which is slow in most cases)
    # but again, feels cheaty, might not work, especially in strong field
    # NUHHH CALCULATE AT APOAPSE
    poly1 = np.array([(E0**2 - 1), 2, ((a**2)*(E0**2 - 1) - L0**2 - C0), 2*((a*E0 - L0)**2 + C0), -C0*(a**2)])
    r = np.real(max(np.roots(poly1)))
    P, D = E0*(r*r + a*a) - a*L0, r*r - 2*r + a*a
    dC_dE = 2*P*(r*r + a*a)/D - 2*a*(a*E0 - L0)
    dC_dLz = -2*a*P/D + 2*(a*E0 - L0)
    dC0 = dC_dE*dE + dC_dLz*dLz
    dC_dr = (4*E0*r*P - P*P*(2*r - 2))/(D*D) - 2*r
    poly2 = np.array([((E0 + dE)**2 - 1), 2, ((a**2)*((E0 + dE)**2 - 1) - (L0 + dLz)**2 - (C0 + dC0)), 2*((a*(E0 + dE) - (L0 + dLz))**2 + (C0 + dC0)), -(C0 + dC0)*(a**2)])
    dr = np.real(max(np.roots(poly2))) - r
    dC = dC0 + dC_dr*dr

    # Step 4
    E, L = E0 + dE, L0 + dLz    #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                #I actually don't think I need to make that correction
    # Step 5
    C = max(C0 + dC, 0.0)

    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state9m(cons, con_derv, state, a, ecc):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    E0, L0, C0 = cons

    # Step 2
    # z = cos(thet_min)
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        sig = A + L0**2 + C0
        z2 = (sig - (sig**2 - 4*A*C0)**(1/2))/(2*A)
    z = np.sqrt(min(1.0, np.abs(z2)))
    dE, dLx, dLy, dLz = con_derv[:4]

    if ecc <= 1e-3 or np.isclose(z2, 1):
        # Near-circular orbits! This feels a little like cheating?
        # The theta-potential becomes useless for polar orbits
        # And it's very easy for near-circular orbits to go out of wack
        # so in both cases we switch to the radial potential, rearrange that to solve for C
        # then get dC = (dC_dE)*dE + (dC_dLz)*dLz + (dC_dr)*dr
        # the especially cheaty part is saying (dC_dr) = (dC_dR)(dR_dr), and that since
        # we're doing this calculation at roughly r ~ potential minimum, (dR_dr) ~ 0 -> (dC_dr) ~ 0
        # dr would also be fairly small here (it's not r_dot, but the change in potential minimum, which is slow in most cases)
        # but again, feels cheaty, might not work, especially in strong field
        r = state[1]
        P, D = E0*(r*r + a*a) - a*L0, r*r - 2*r + a*a
        dC_dE = 2*P*(r*r + a*a)/D - 2*a*(a*E0 - L0)
        dC_dLz = -2*a*P/D + 2*(a*E0 - L0)
        dC = dC_dE*dE + dC_dLz*dLz
    else:
        # All other cases! Use the theta potential for this, but same concept:
        # solve for C, get partials
        # a bit of extra work because z acts funky, but I gave up and used an approximation
        dC_dE = -2*a*a*z2*E0
        dC_dLz = 2*z2*L0/(1 - z2)
        # Now you would think we would include a z or z2 term, but naw! z actually destroys sign information
        # so instead let's try x = sin(theta_min)! Now we retain sign info, which means we can actually tell if we're retrograde or not
        # now z2 = 1 - x2, so see what that does?
        x2 = 1 - z2
        x = np.sign(L0)*np.sqrt(x2)
        dC_dx = - 2*(a*a*x*(1 - E0*E0) + L0*L0/(x**3))
        # Okay but dx?? If you differentiate, you get something dependent on dC, which is pretty circular
        # So let's treat this as a correction term on to the rest of dC, and just use dC0 = dC_dE*dE + dC_dLz*dLz
        dC0 = dC_dE*dE + dC_dLz*dLz
        dx = dC0 + 2*(1 - x2)*a*a*E0*dE - 2*L0*dLz*(1 - x2)/x2
        dx /= -2*(x*(a*a*(1 - E0*E0) + L0*L0/x2) + L0*L0*(1 - x2)/(x**3))
        dC = dC0 + dC_dx*dx

    # Step 4
    E, L = E0 + dE, L0 + dLz    #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                #I actually don't think I need to make that correction
    # Step 5
    C = max(C0 + dC, 0.0)

    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state9l(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Uses C = Lx^2 + Ly^2 + (a_term)
    # instead of C = L^2 - Lz^2 + (a_term)
    # It seemed like mixing up L_vec and L0 caused issues? But also just made different ones
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr_2(state, a)
    theta = state[2]
    fmom = np.matmul(metric, state[4:])
    L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    cart_state = sph2cart(state, a)
    L_dir = np.cross(cart_state[1:4], cart_state[5:])
    L_vec = L*L_dir/np.linalg.norm(L_dir)

    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
    cosz, sinz = np.sqrt(min(1.0, np.abs(z2))), np.sqrt(1 - min(1.0, np.abs(z2)))

    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = [dLx, dLy, dLz]
    cosinc = np.cos(np.mean(np.abs(root_getter(E0, L0, C0, a)[2][1:3])))
    dC = 2*(L_vec[0]*dLx + L_vec[1]*dLy - ((a*np.cos(state[2]))**2)*E0*dE)
        #From glamp A3, thetadot term goes away because I don't change position!

    # Step 4
    E, L = E0 + dE, L0 + dLz    #*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                        #I actually don't think I need to make that correction
    # Step 5
    C = max(C0 + dC, 0.0)

    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state9k(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Tried using J_theta integral
    # Step 1
    E0, Lz0, C0 = cons
    Jtheta0 = compute_J_theta(E0, Lz0, C0, a)
    
    dE, dLx, dLy, dLz = con_derv[:4]

    # Step 4
    E, Lz = E0 + dE, Lz0 + dLz    #*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                        #I actually don't think I need to make that correction
    # Step 5
    C = solve_for_C(E, Lz, Jtheta0, a)

    # Step 6
    new_state = recalc_state([E, Lz, C], state, a)
    return new_state, [E, Lz, C]

def new_recalc_state9j(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr_2(state, a)
    theta = state[2]
    fmom = np.matmul(metric, state[4:])
    L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    cart_state = sph2cart(state, a)
    L_dir = np.cross(cart_state[1:4], cart_state[5:])
    L_vec = L*L_dir/np.linalg.norm(L_dir)

    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
    cosz, sinz = np.sqrt(min(1.0, np.abs(z2))), np.sqrt(1 - min(1.0, np.abs(z2)))

    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = [dLx, dLy, dLz]
    cosinc = np.cos(np.mean(np.abs(root_getter(E0, L0, C0, a)[2][1:3])))
    dC = 2*(np.dot(L_vec, dL_vec) - L0*dLz - ((a*np.cos(state[2]))**2)*E0*dE)
        #From glamp A3, thetadot term goes away because I don't change position!

    # Step 4
    E, L = E0 + dE, L0 + dLz    #*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                        #I actually don't think I need to make that correction
    # Step 5
    C = max(C0 + dC, 0.0)

    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state9i(cons, con_derv, state, a, name):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    theta = state[2]
    fmom = np.matmul(metric, state[4:])
    L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
    cosz, sinz = np.sqrt(min(1.0, np.abs(z2))), np.sqrt(1 - min(1.0, np.abs(z2)))
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    cosinc = np.cos(np.mean(np.abs(root_getter(E0, L0, C0, a)[2][1:3])))
    if "s_-_0" in name:
        dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(state[2]))**2)*(1 - E0**2)*dE)
    elif "i_-_0" in name:
        dC = 2*(L*dL_vec - L0*dLz - ((a*cosinc)**2)*(1 - E0**2)*dE)
    elif "s_E_0" in name:
        dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(state[2]))**2)*E0*dE)
    elif "i_E_0" in name:
        dC = 2*(L*dL_vec - L0*dLz - ((a*cosinc)**2)*E0*dE)
    elif "s_-_d" in name:
        dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(state[2]))**2)*(1 - E0**2)*dE - (a**2)*np.sin(state[2])*np.cos(state[2])*(1 - E0**2)*state[6])
    elif "i_-_d" in name:
        alph, bet = 2*L*dL_vec - 2*L0*dLz - 2*a*a*cosinc*(1 - E0**2)*dE, a*a*(1 - E0**2)
        A = (2*L0*dLz - 2*a*a*E0*dE)*cosinc + 2*a*a*E0*dE*(cosinc**2) - C0
        B = 2*a*a*(1 - E0**2)*cosinc - (C0 + a*a*(1 - E0**2) + L0**2)
        dC = (alph + bet*cosinc/B)*(B/(B - bet*cosinc))
    elif "s_E_d" in name:
        dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(state[2]))**2)*E0*dE - (a**2)*np.sin(state[2])*np.cos(state[2])*(1 - E0**2)*state[6])
    elif "i_E_d" in name:
        alph, bet = 2*L*dL_vec - 2*L0*dLz - 2*a*a*cosinc*E0*dE, a*a*(1 - E0**2)
        A = (2*L0*dLz - 2*a*a*E0*dE)*cosinc + 2*a*a*E0*dE*(cosinc**2) - C0
        B = 2*a*a*(1 - E0**2)*cosinc - (C0 + a*a*(1 - E0**2) + L0**2)
        dC = (alph + bet*cosinc/B)*(B/(B - bet*cosinc))
    
        #From glamp A3, thetadot term goes away because I don't change position!

    # Step 4
    E, L = E0 + dE, L0 + dLz + np.sqrt(np.abs(dC))*np.sign(dC)    #*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                        #I actually don't think I need to make that correction
    
    # Step 5
    C = C0 
    
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    #test = max(np.roots(np.polyder(potent)))
    count = 0
    #Make sure point we're AT is viable, the rest is maybe lame?

    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state9h(cons, con_derv, state, a, name):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    theta = state[2]
    fmom = np.matmul(metric, state[4:])
    L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
    cosz, sinz = np.sqrt(min(1.0, np.abs(z2))), np.sqrt(1 - min(1.0, np.abs(z2)))
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    cosinc = np.cos(np.mean(np.abs(root_getter(E0, L0, C0, a)[2][1:3])))
    if "s_-_0" in name:
        dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(state[2]))**2)*(1 - E0**2)*dE)
    elif "i_-_0" in name:
        dC = 2*(L*dL_vec - L0*dLz - ((a*cosinc)**2)*(1 - E0**2)*dE)
    elif "s_E_0" in name:
        dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(state[2]))**2)*E0*dE)
    elif "i_E_0" in name:
        dC = 2*(L*dL_vec - L0*dLz - ((a*cosinc)**2)*E0*dE)
    elif "s_-_d" in name:
        dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(state[2]))**2)*(1 - E0**2)*dE - (a**2)*np.sin(state[2])*np.cos(state[2])*(1 - E0**2)*state[6])
    elif "i_-_d" in name:
        alph, bet = 2*L*dL_vec - 2*L0*dLz - 2*a*a*cosinc*(1 - E0**2)*dE, a*a*(1 - E0**2)
        A = (2*L0*dLz - 2*a*a*E0*dE)*cosinc + 2*a*a*E0*dE*(cosinc**2) - C0
        B = 2*a*a*(1 - E0**2)*cosinc - (C0 + a*a*(1 - E0**2) + L0**2)
        dC = (alph + bet*cosinc/B)*(B/(B - bet*cosinc))
    elif "s_E_d" in name:
        dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(state[2]))**2)*E0*dE - (a**2)*np.sin(state[2])*np.cos(state[2])*(1 - E0**2)*state[6])
    elif "i_E_d" in name:
        alph, bet = 2*L*dL_vec - 2*L0*dLz - 2*a*a*cosinc*E0*dE, a*a*(1 - E0**2)
        A = (2*L0*dLz - 2*a*a*E0*dE)*cosinc + 2*a*a*E0*dE*(cosinc**2) - C0
        B = 2*a*a*(1 - E0**2)*cosinc - (C0 + a*a*(1 - E0**2) + L0**2)
        dC = (alph + bet*cosinc/B)*(B/(B - bet*cosinc))
    
        #From glamp A3, thetadot term goes away because I don't change position!

    # Step 4
    E, L = E0 + dE, L0 + dLz    #*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                        #I actually don't think I need to make that correction
    
    # Step 5
    C = C0 + dC
    
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    #test = max(np.roots(np.polyder(potent)))
    count = 0
    #Make sure point we're AT is viable, the rest is maybe lame?

    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state9g(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    theta = state[2]
    fmom = np.matmul(metric, state[4:])
    L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
    cosz, sinz = np.sqrt(min(1.0, np.abs(z2))), np.sqrt(1 - min(1.0, np.abs(z2)))
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    inc = np.mean(np.abs(root_getter(E0, L0, C0, a)[2][1:3]))
    dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(inc))**2)*E0*dE - (a**2)*np.sin(state[2])*np.cos(state[2])*(1 - E0**2)*state[6] )# if C0 != 0 else 0.0
        #From glamp A3, thetadot term goes away because I don't change position!

    # Step 4
    E, L = E0 + dE, L0 + dLz    #*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                        #I actually don't think I need to make that correction
    
    # Step 5
    C = C0 + dC
    
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    #test = max(np.roots(np.polyder(potent)))
    count = 0
    #Make sure point we're AT is viable, the rest is maybe lame?

    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state9f(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    theta = state[2]
    fmom = np.matmul(metric, state[4:])
    L_full = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
    cosz, sinz = np.sqrt(min(1.0, np.abs(z2))), np.sqrt(1 - min(1.0, np.abs(z2)))
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    #dC = 2*z2*(L0*dLz/(1-z2) - (a**2)*E0*dE)  
    #if np.isnan(dC):
    #    dC = -2*z2*(a**2)*E0*dE 
    #dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(state[2]))**2)*E0*dE)# if C0 != 0 else 0.0
        #From glamp A3, thetadot term goes away because I don't change position!

    dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(inc))**2)*E0*dE - (a**2)*np.sin(state[2])*np.cos(state[2])*(1 - E0**2)*state[6] )# if C0 != 0 else 0.0
        #From glamp A3, thetadot term goes away because I don't change position!
    #dC = 2*(L*dL_vec - L0*dLz - (a**2)*cosz*(sinz*state[6]*(1 - E0**2) + cosz*E0*dE))
    #dC = 2*(L0 - a*E0*np.sin(theta)**2)*(dLz - a*dE*np.sin(theta)**2 - 2*a*E0*np.sin(theta)*np.cos(theta)*state[6])*np.sin(theta)**2
    #dC -= 2*np.sin(theta)*np.cos(theta)*(L0 - a*E0*np.sin(theta)**2)*state[6]
    #dC /= np.sin(theta)**4
    #dC -=2*(a**2)*np.cos(theta)*np.sin(theta)*state[6]
    #dC += 2*(state[1]**2 + (a*np.cos(theta))**2)*(2*state[1]*state[5] - 2*(a**2)*np.sin(theta)*np.cos(theta)*state[6])*(state[6]**2)

    # Step 4
    E, L = E0 + dE, L0 + dLz    #*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                        #I actually don't think I need to make that correction
    
    # Step 5
    #C = C0 + dC
    #if a == 0.0:
    #    C = L_full**2 - L**2
    #else:
    #    a_, b_, c_ = 1, 2*(L**2) - L_full**2 - (a**2)*(1 - E**2), -(L_full*L)**2 + L**4
    #    C = (-b_ + np.sqrt(b_**2 - 4*a_*c_))/(2*a_)
    
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    #test = max(np.roots(np.polyder(potent)))
    count = 0
    #Make sure point we're AT is viable, the rest is maybe lame?
    '''
    while (np.polyval(potent, state[1]) < 0.0):
        dR = -np.polyval(potent, test)
        #scale all the variables
        #E -= dE*(1e-5)
        L += dLz*(1e-5)
        C += dC*(1e-5)
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
        count += 1
        if count == 1000:
            print("gotdamn")
        if count >= 1e6:
            break
    if count > 0:
        print("L, C adjust: %s, %s (%s times)"%( count*dLz*1e-5, count*dC*1e-5, count))
    '''
    # Step 6
    #print(E, L, C)
    #print(dE, dLz*np.sign(L0), dC)
    #print(cosz, sinz, np.abs(z2))
    #print(E0, L0, C0)
    #print(E, L, C)
    #print("yo!", E0, L0, C0)
    #print("to!", E, L, C)
    #print("mo!", con_derv)
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state9e(cons, con_derv, state, a, label):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    theta = state[2]
    fmom = np.matmul(metric, state[4:])
    L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
    cosz, sinz = np.sqrt(min(1.0, np.abs(z2))), np.sqrt(1 - min(1.0, np.abs(z2)))
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    if "glamp" in label:
        inc = np.mean(np.abs(root_getter(E0, L0, C0, a)[2][1:3]))
        dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(inc))**2)*(1 - E0**2)*dE)# if C0 != 0 else 0.0
        #From glamp A3, thetadot term goes away because I don't change position!
    if "dcosi_0" in label:
        #based on cosi = L/sqrt(L^2 + C) or (tani)^2 = C/(L^2)
        if L0 != 0:
            dC = 2*(C0/L0)*dLz
        else:
            dC = 2*L*dL_vec  #assumes dL_vec is proportional to L
    if "dinc_0" in label:
        #based on (tani)^2 = C/(L^2 + (a^2)(1 - E^2))
        dC = (C/(L0**2 + (a**2)*(1 - E**2)))*(2*L0*dLz -2*E*dE*a*a)
        

    # Step 4
    E, L = E0 + dE, L0 + dLz    #*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                        #I actually don't think I need to make that correction
    
    # Step 5
    C = C0 + dC
    
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    test = max(np.roots(np.polyder(potent)))
    count = 0
    #Make sure point we're AT is viable, the rest is maybe lame?
 
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state9d(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    theta = state[2]
    fmom = np.matmul(metric, state[4:])
    L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
    cosz, sinz = np.sqrt(min(1.0, np.abs(z2))), np.sqrt(1 - min(1.0, np.abs(z2)))
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    #dC = 2*z2*(L0*dLz/(1-z2) - (a**2)*E0*dE)  
    #if np.isnan(dC):
    #    dC = -2*z2*(a**2)*E0*dE 
    inc = np.mean(np.abs(root_getter(E0, L0, C0, a)[2][1:3]))
    dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(inc))**2)*(1 - E0**2)*dE)# if C0 != 0 else 0.0
        #From glamp A3, thetadot term goes away because I don't change position!

    # Step 4
    E, L = E0 + dE, L0 + dLz    #*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                        #I actually don't think I need to make that correction
    
    # Step 5
    C = C0 + dC
    
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    test = max(np.roots(np.polyder(potent)))
    count = 0
    #Make sure point we're AT is viable, the rest is maybe lame?
    '''
    while (np.polyval(potent, state[1]) < 0.0):
        dR = -np.polyval(potent, test)
        #scale all the variables
        #E -= dE*(1e-5)
        L += dLz*(1e-5)
        C += dC*(1e-5)
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
        count += 1
        if count == 1000:
            print("gotdamn")
        if count >= 1e6:
            break
    if count > 0:
        print("L, C adjust: %s, %s (%s times)"%( count*dLz*1e-5, count*dC*1e-5, count))
    '''
    # Step 6
    #print(E, L, C)
    #print(dE, dLz*np.sign(L0), dC)
    #print(cosz, sinz, np.abs(z2))
    #print(E0, L0, C0)
    #print(E, L, C)
    #print("yo!", E0, L0, C0)
    #print("to!", E, L, C)
    #print("mo!", con_derv)
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state9a(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    theta = state[2]
    thedot = state[6]
    #thedotdot = -((thedot/np.sqrt(1 - theta**2))**2)*theta  #an approximation, check that
    fmom = np.matmul(metric, state[4:])
    L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    Lx, Ly = fmom[2]*np.sin(state[3]), -L*np.cos(state[3])
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
    cosz, sinz = np.sqrt(np.abs(z2)), np.sqrt(1 - np.abs(z2))
    sint, cost = np.sin(theta), np.cos(theta)
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    def get_Ls(state):
        x, y, z = state[1]*np.sin(state[2])*np.cos(state[3]), state[1]*np.sin(state[2])*np.sin(state[3]), state[1]*np.cos(state[2])
        vx = state[5]*np.sin(state[2])*np.cos(state[3]) + state[1]*state[6]*np.cos(state[2])*np.cos(state[3]) - state[1]*state[7]*np.sin(state[2])*np.sin(state[3]) 
        vy = state[5]*np.sin(state[2])*np.sin(state[3]) + state[1]*state[6]*np.cos(state[2])*np.sin(state[3]) + state[1]*state[7]*np.sin(state[2])*np.cos(state[3]) 
        vz = state[5]*np.cos(state[2]) - state[1]*state[6]*np.sin(state[2])
        return np.cross([x,y,z], [vx, vy, vz])
    Lx, Ly, Lz = get_Ls(state)
    #print("WHAT IS HAPPENING")
    #print(dLx, dLy, dLz)
    #print(Lx, Ly, L0)
    #print(np.sqrt(Lx**2 + Ly**2 + L0**2))
    #print(L)
    #print(Lx**2 + Ly**2)
    #print(C0)
    #dC = 2*(L*dL_vec - L0*dLz - (a**2)*cosz*(sinz*state[6]*(1 - E0**2) + cosz*E0*dE))
    #dC = 2*L*dL_vec - 2*L0*dLz - 2*(a**2)*(thedot*np.sin(theta)*np.cos(theta)*(1 - E0**2) + (np.cos(theta)**2)*E0*dE)
    dC = 2*Lx*dLx + 2*Ly*dLy - 2*(a**2)*(thedot*np.sin(theta)*np.cos(theta)*(1 - E0**2) + (np.cos(theta)**2)*E0*dE)
    #print(2*Lx*dLx + 2*Ly*dLy)
    #print(dC)
    #sig = state[1]**2 + (a*cost)**2
    #sig_dot = 2*state[1]*state[5] - 2*(a**2)*thedot*sint*cost
    #dQ = 2*(L0 - a*E0*(sint**2))*(dLz - a*(dE*(sint**2) + 2*E0*thedot*sint*cost))/(sint**2)
    #dQ += -((L0 - a*E0*(sint**2))**2)*(2*cost/(sint**3))*thedot
    #dQ += -2*(a**2)*thedot*sint*cost + 2*sig*sig_dot*(thedot**2) #+ 2*(sig**2)*thedot*thedotdot second derivative, real small
    #dC = dQ - 2*(a*E0 - L0)*(a*dE - dLz)

    # Step 4
    E, L = E0 + dE, L0 + dLz*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
    
    # Step 5
    C = C0 + dC
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    test = max(np.roots(np.polyder(potent)))
    '''
    while (np.polyval(potent, test) < 0.0):
        #print("WEIRD THING:", np.polyval(potent, test))
        dR = -np.polyval(potent, test)
        E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
        #C -= 0.0001*C
        #L -= 0.01*L
        #potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        #test = max(np.roots(np.polyder(potent)))
    '''
    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    #print(E0 - E)
    #print(dE)
    #print(L0 - L)
    #print(dLz)
    #print(C0 - C)
    #print(dC)
    return new_state, [E, L, C]

def new_recalc_state9b(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    #Get the ZAMO Kerr tetrad matrix
    metric, chris = kerr(state, a)
    hold = np.linalg.inv(metric)
    w = hold[0,3]/hold[0,0]
    #E_u_d converts tetrads to BL
    E_u_d = np.array([[np.sqrt(-hold[0,0]),   0,                  0,                  0],
                      [0,                     np.sqrt(hold[1,1]), 0,                  0],
                      [0,                     0,                  np.sqrt(hold[2,2]), 0],
                      [np.sqrt(-w*hold[3,0]), 0,                  0,                  1/np.sqrt(metric[3,3])]])
    #E_d_u converts BL to tetrad
    E_d_u = np.linalg.inv(E_u_d)
    tetrad = np.linalg.matmul(E_d_u, state[4:])
    #Convert spherical tetrad to cartesian
    st, ct, sp, cp = np.sin(state[2]), np.cos(state[2]), np.sin(state[3]), np.cos(state[3])
    rho = np.array([sp*st, cp*st, ct])
    phi = np.linalg.cross([0,0,1], rho)/np.linalg.norm(np.linalg.cross([0,0,1], rho))
    thet = -np.linalg.cross(rho, phi)/np.linalg.norm(np.linalg.cross(rho, phi))
    #conversion matrices
    bl2cart, cart2bl = np.array([rho, thet, phi]), np.linalg.inv([rho, thet, phi])
    #cartesian tetrad
    cart_tetrad = np.array([*state[:4], tetrad[0], *np.linalg.matmul(bl2cart, tetrad[1:])])
    #Get psuedoconstants energy and cartesian L components [E, Lx, Ly, Lz]
    r = np.array([np.sqrt(state[1]**2 + a**2) * np.sin(state[2]) * np.cos(state[3]),
                  np.sqrt(state[1]**2 + a**2) * np.sin(state[2]) * np.sin(state[3]),
                  state[1] * np.cos(state[2]) ])
    v = np.array([(state[5]*state[1]/(np.sqrt(state[1]**2 + a**2)) * np.sin(state[2]) * np.cos(state[3]) + np.sqrt(state[1]**2 + a**2) * state[6]*np.cos(state[2]) * np.cos(state[3]) - np.sqrt(state[1]**2 + a**2) * np.sin(state[2]) * state[7] * np.sin(state[3])),
                  (state[5]*state[1]/(np.sqrt(state[1]**2 + a**2)) * np.sin(state[2]) * np.sin(state[3]) + np.sqrt(state[1]**2 + a**2) * state[6]*np.cos(state[2]) * np.sin(state[3]) + np.sqrt(state[1]**2 + a**2) * np.sin(state[2]) * state[7] * np.cos(state[3])),
                  state[5]*np.cos(state[2]) - state[1]*state[6]*np.sin(state[2])])
    newL = np.cross(r, v)
    four_mom = np.matmul(metric, state[4:])
    psuedo1 = np.array([-four_mom[0], *newL])
    #Create least squares matrix
    # A*d(v) = d(cons), At = transpose(A) 
    At = []
    for i in range(3):
        eps = np.zeros(3)
        eps[i] = 1e-7
        vec = cart_tetrad[5:]/cart_tetrad[4] + eps
        b = np.linalg.norm(vec)
        test_vec = np.array([1/np.sqrt(1 - b**2), *(vec/np.sqrt(1 - b**2))])
        #[new_vec[0], *np.linalg.matmul(cart2bl, new_vec[1:])]
        test_state = np.array([*cart_tetrad[:4], *np.linalg.matmul(E_u_d, [test_vec[0], *np.linalg.matmul(cart2bl, test_vec[1:])])])
        test_v = np.array([(test_state[5]*test_state[1]/(np.sqrt(test_state[1]**2 + a**2)) * np.sin(test_state[2]) * np.cos(test_state[3]) + np.sqrt(test_state[1]**2 + a**2) * test_state[6]*np.cos(test_state[2]) * np.cos(test_state[3]) - np.sqrt(test_state[1]**2 + a**2) * np.sin(test_state[2]) * test_state[7] * np.sin(test_state[3])),
                           (test_state[5]*test_state[1]/(np.sqrt(test_state[1]**2 + a**2)) * np.sin(test_state[2]) * np.sin(test_state[3]) + np.sqrt(test_state[1]**2 + a**2) * test_state[6]*np.cos(test_state[2]) * np.sin(test_state[3]) + np.sqrt(test_state[1]**2 + a**2) * np.sin(test_state[2]) * test_state[7] * np.cos(test_state[3])),
                           test_state[5]*np.cos(test_state[2]) - test_state[1]*test_state[6]*np.sin(test_state[2])])
        test_psuedo = np.array([-np.matmul(metric, test_state[4:])[0], *np.cross(r, test_v)])
        #new_cons1 = gurf2(bl_conv(np.array([*cartstate[:4], *new_vec]), a), a)
        At.append((test_psuedo - psuedo1)/1e-7)
    At, A = np.array(At), np.array(At).T
    #Calculate necessary change to cart_tetrad
    try:
        #original least squares, use At
        dv = np.matmul(np.linalg.inv(np.matmul(At, A)), np.matmul(At, con_derv))
    except:
        #try this?? just a random matrix instead of At
        print("At.A is singular or something")
        print(A)

        B = np.random.random((3,4))
        dv = np.matmul(np.linalg.inv(np.matmul(B, A)), np.matmul(B, con_derv))
    #Calculate new state
    vec = cart_tetrad[5:]/cart_tetrad[4] + dv
    #get beta
    b = np.linalg.norm(vec)
    #new tetrad
    new_vec = np.array([1/np.sqrt(1 - b**2), *(vec/np.sqrt(1 - b**2))])
    #actual new state
    new_state = np.array([*cart_tetrad[:4], *np.linalg.matmul(E_u_d, [new_vec[0], *np.linalg.matmul(cart2bl, new_vec[1:])])])
    #get new psuedoconstants
    new_v = np.array([(state[5]*state[1]/(np.sqrt(state[1]**2 + a**2)) * np.sin(state[2]) * np.cos(state[3]) + np.sqrt(state[1]**2 + a**2) * state[6]*np.cos(state[2]) * np.cos(state[3]) - np.sqrt(state[1]**2 + a**2) * np.sin(state[2]) * state[7] * np.sin(state[3])),
                           (state[5]*state[1]/(np.sqrt(state[1]**2 + a**2)) * np.sin(state[2]) * np.sin(state[3]) + np.sqrt(state[1]**2 + a**2) * state[6]*np.cos(state[2]) * np.sin(state[3]) + np.sqrt(state[1]**2 + a**2) * np.sin(state[2]) * state[7] * np.cos(state[3])),
                           state[5]*np.cos(state[2]) - state[1]*state[6]*np.sin(state[2])])
    four_mom2 = np.matmul(metric, new_state[4:])
    E = -four_mom2[0]
    L = four_mom2[3]
    Q = np.matmul(np.matmul(kill_tensor(new_state, a), new_state[4:]), new_state[4:])
    psuedo2 = np.array([-four_mom2[0], *np.cross(r, new_v)])
    cart = Q - (a*E - L)**2
    return new_state, [E, L, cart]

def new_recalc_state10(cons, con_derv, state, a):
    metric, chris = kerr(state, a)
    E0 = -np.matmul(metric, state[4:])[0]
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
    tetrad = np.linalg.solve(tet2kerr, state[4:])
    tetcart = np.array([*tetrad[:2], tetrad[3], -tetrad[2]])
    vel, cartpos = tetcart[1:]/tetcart[0], np.array([r*sint*cosp, r*sint*sinp, r*cost])
    L0 = np.cross(cartpos, vel)
    A, eps = np.zeros((4,3)), 1e-7
    loop, err, target = 0, 100, np.array([E0, *L0]) + con_derv
    #print(con_derv)
    #print(np.linalg.norm(con_derv)**2)
    #print(np.array([E0, *L0]))
    diff = con_derv
    
    def getNewCons(j, vel, eps):
        intvel = np.array([0.0,0.0,0.0])
        intvel[j] += eps
        intL, gamma = np.cross(cartpos, intvel+vel), 1/np.sqrt(1 - np.linalg.norm(intvel + vel)**2)
        inttetrad = gamma*np.array([1, intvel[0], -intvel[2], intvel[1]])
        intkerr = np.matmul(tet2kerr, inttetrad)
        intE = -np.matmul(metric, intkerr)[0]
        return np.array([intE - E0, *(intL - L0)])/eps
    
    while err > 1e-5 and loop < 100:
        A[:,0], A[:,1], A[:,2] = getNewCons(0, vel, eps), getNewCons(1, vel, eps), getNewCons(2, vel, eps)
        dvel = np.linalg.solve(np.matmul(np.transpose(A), A), np.matmul(np.transpose(A), diff))
        vel = vel + dvel
        gamma, newL = 1/np.sqrt(1 - np.linalg.norm(vel)**2), np.cross(cartpos, vel)
        newtetrad = gamma*np.array([1, vel[0], -vel[2], vel[1]])
        newkerr = np.matmul(tet2kerr, newtetrad)
        newE = -np.matmul(metric, newkerr)[0]        #initial energy
        err = 100*np.linalg.norm((np.array([newE, *newL]) - target)/target)
        #print(np.array([newE, *newL]), err)
        loop += 1
        diff = target - np.array([newE, *newL])
    #print("___")
    #print(target)
    #print(np.array([newE, *newL]))
    #print(np.linalg.norm(con_derv)**2)
    holdstate = np.array([*state[0:4], *newkerr])
    newLz = np.matmul(metric, newkerr)[3]        #initial angular momentum
    newQ = np.matmul(np.matmul(kill_tensor(holdstate, a), newkerr), newkerr)
    newC = newQ - (a*newE - newLz)**2  

    return holdstate, [newE, newLz, newC]

def new_recalc_state10b(cons, con_derv, state, a):
    metric, chris = kerr(state, a)
    E0 = -np.matmul(metric, state[4:])[0]
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
    tetrad = np.linalg.solve(tet2kerr, state[4:])
    tetcart = np.array([*tetrad[:2], tetrad[3], -tetrad[2]])
    vel, cartpos = tetcart[1:]/tetcart[0], np.array([r*sint*cosp, r*sint*sinp, r*cost])
    L0 = np.cross(cartpos, vel)
    A, eps = np.zeros((4,3)), 1e-7
    loop, err, target = 0, 100, np.array([E0, *L0]) + con_derv
    diff = con_derv
    
    def getNewCons(j, vel, eps):
        intvel = np.array([0.0,0.0,0.0])
        intvel[j] += eps
        intL, gamma = np.cross(cartpos, intvel+vel), 1/np.sqrt(1 - np.linalg.norm(intvel + vel)**2)
        inttetrad = gamma*np.array([1, intvel[0], -intvel[2], intvel[1]])
        intkerr = np.matmul(tet2kerr, inttetrad)
        intE = -np.matmul(metric, intkerr)[0]
        return np.array([intE - E0, *(intL - L0)])/eps
    
    while err > 1e-5 and loop < 100:
        A[:,0], A[:,1], A[:,2] = getNewCons(0, vel, eps), getNewCons(1, vel, eps), getNewCons(2, vel, eps)
        dvel = np.linalg.solve(np.matmul(np.transpose(A), A), np.matmul(np.transpose(A), diff))
        vel = vel + dvel
        gamma, newL = 1/np.sqrt(1 - np.linalg.norm(vel)**2), np.cross(cartpos, vel)
        newtetrad = gamma*np.array([1, vel[0], -vel[2], vel[1]])
        newkerr = np.matmul(tet2kerr, newtetrad)
        newE = -np.matmul(metric, newkerr)[0]        #initial energy
        err = 100*np.linalg.norm((np.array([newE, *newL]) - target)/target)
        #print(np.array([newE, *newL]), err)
        loop += 1
        diff = target - np.array([newE, *newL])
    #print("___")
    #print(target)
    #print(np.array([newE, *newL]))
    #print(np.linalg.norm(con_derv)**2)
    holdstate = np.array([*state[0:4], *newkerr])
    newLz = np.matmul(metric, newkerr)[3]        #initial angular momentum
    newQ = np.matmul(np.matmul(kill_tensor(holdstate, a), newkerr), newkerr)
    newC = newQ - (a*newE - newLz)**2  

    return holdstate, [newE, newLz, newC]

def new_recalc_state11(cons, con_derv, state, a, mu, path):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons

    cosi = L0/np.sqrt(L0**2 + C0)
    path = np.array(path)
    e = 1 - min(path[:,1])/max(path[:,1]) if path[0,5] < 0 else max(path[:,1])/min(path[:,1]) - 1
    r0 = max(path[:,1]) if path[0,5] < 0 else min(path[:,1])
    p = r0*(1 - e**2)
    R = lambda r: (E0**2 - 1.0)*(r**4) + 2.0*(r**3) + ((a**2)*(E0**2 - 1.0) - L0**2 - C0)*(r**2) + 2*((a*E0 - L0)**2 + C0)*r - C0*(a**2)
    turns = optimize.fsolve(R, [(a**2)*C0, (0.3*(a**2)*C0 + 0.7*p/(1 + e)), p/(1 + e), p/(1 - e)])
    e = (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
    p = np.sqrt(turns[-1]*turns[-2]*(1 - e**2))
    f1 = lambda x: 1 + (73/24)*(x**2) + (37/96)*(x**4)
    f2 = lambda x: 73/12 + (823/24)*(x**2) + (949/32)*(x**4) + (491/192)*(x**6)
    f3 = lambda x: 1 + (7/8)*(x**2)
    f4 = lambda x: 61/24 + (63/8)*(x**2) + (94/64)*(x**4)
    f5 = lambda x: 61/8 + (91/4)*(x**2) + (461/64)*(x**4)
    f6 = lambda x: 97/12 + (37/2)*(x**2) + (211/32)*(x**4)
    
    r0 = p/(1 - e**2)
    
    dEdt = ((-32/5)*(mu**2)*(p**(-5))*((1 - e**2)**(3/2))*(f1(e) - a*(p**(-3/2))*cosi*f2(e)))
    dLdt = ((-32/5)*(mu**2)*(p**(-7/2))*((1 - e**2)**(3/2))*(cosi*f3(e) + a*(p**(-3/2))*(f4(e) - (cosi**2)*f5(e))))
    dQdt = ((-64/5)*(mu**3)*(p**(-3))*((1 - e**2)**(3/2))*(f3(e) - a*(p**(-3/2))*cosi*f6(e)))
    dCdt = dQdt - 2*L0*dLdt
    dt = path[-1,0] - path[0,0]
    #print(dQ, dC)
    #print(dC, dC + 2*L0*dL, dC - 2*L0*dL)
    
    #print(path[-1,0] - path[0,0])
    #print("no")
    #print(r0*(1-e), r0*(1+e), r0, e)
    E, L, C = E0 + dEdt*dt, L0 + dLdt*dt, C0 + 0*dCdt*dt
    #print("HEY", E0, L0, C0)
    #print("HEY", E, L, C)
    #print(E)
    '''
    R2 = lambda r: (E**2 - 1.0)*(r**4) + 2.0*(r**3) + ((a**2)*(E**2 - 1.0) - L**2 - C)*(r**2) + 2*((a*E - L)**2 + C)*r - C*(a**2)
    test = optimize.fsolve(R2, turns)
    while R2(max(test)) < 0.0:
        dR = -np.polyval(potent, test)
        E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
    '''
    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state12(cons, con_derv, state, a, mu, path):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    
    roots = root_getter(E0, L0, C0, a)[0]
    e = (roots[-1] - roots[-2])/(roots[-1] + roots[-2])
    p = 0.5*(roots[-1] + roots[-2])*(1 - e**2)
    cosi = L0/np.sqrt(L0**2 + C0)

    f1 = lambda x: 1 + (73/24)*(x**2) + (37/96)*(x**4)
    f2 = lambda x: 73/12 + (823/24)*(x**2) + (949/32)*(x**4) + (491/192)*(x**6)
    f3 = lambda x: 1 + (7/8)*(x**2)
    f4 = lambda x: 61/24 + (63/8)*(x**2) + (94/64)*(x**4)
    f5 = lambda x: 61/8 + (91/4)*(x**2) + (461/64)*(x**4)
    f6 = lambda x: 97/12 + (37/2)*(x**2) + (211/32)*(x**4)
    
    dEdt = ((-32/5)*(mu**2)*(p**(-5))*((1 - e**2)**(3/2))*(f1(e) - a*(p**(-3/2))*cosi*f2(e)))
    dLdt = ((-32/5)*(mu**2)*(p**(-7/2))*((1 - e**2)**(3/2))*(cosi*f3(e) + a*(p**(-3/2))*(f4(e) - (cosi**2)*f5(e))))
    dQdt = ((-64/5)*(mu**3)*(p**(-3))*((1 - e**2)**(3/2))*(f3(e) - a*(p**(-3/2))*cosi*f6(e)))
    dCdt = dQdt - 2*L0*dLdt
    dt = path[-1][0] - path[0][0]
    #print(dQ, dC)
    #print(dC, dC + 2*L0*dL, dC - 2*L0*dL)
    
    #print(path[-1,0] - path[0,0])
    #print("no")
    #print(r0*(1-e), r0*(1+e), r0, e)

    E, L, C = E0 + dEdt*dt/mu, L0 + dLdt*dt/mu, C0 + 0*dCdt*dt/mu
    #print("HEY", E0, L0, C0)
    #print("HEY", E, L, C)
    #print(E)
    '''
    R2 = lambda r: (E**2 - 1.0)*(r**4) + 2.0*(r**3) + ((a**2)*(E**2 - 1.0) - L**2 - C)*(r**2) + 2*((a*E - L)**2 + C)*r - C*(a**2)
    test = optimize.fsolve(R2, turns)
    while R2(max(test)) < 0.0:
        dR = -np.polyval(potent, test)
        E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
    '''
    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state13(cons, con_derv, state, a, mu, path):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    
    roots = root_getter(E0, L0, C0, a)[0]
    e = (roots[-1] - roots[-2])/(roots[-1] + roots[-2])
    p = 0.5*(roots[-1] + roots[-2])*(1 - e**2)
    cosi = L0/np.sqrt(L0**2 + C0)

    f1 = lambda x: 1 + (73/24)*(x**2) + (37/96)*(x**4)
    f2 = lambda x: 73/12 + (823/24)*(x**2) + (949/32)*(x**4) + (491/192)*(x**6)
    f3 = lambda x: 1 + (7/8)*(x**2)
    f4 = lambda x: 61/24 + (63/8)*(x**2) + (94/64)*(x**4)
    f5 = lambda x: 61/8 + (91/4)*(x**2) + (461/64)*(x**4)
    f6 = lambda x: 97/12 + (37/2)*(x**2) + (211/32)*(x**4)
    
    dEdt = ((-32/5)*(mu**2)*(p**(-5))*((1 - e**2)**(3/2))*(f1(e) - a*(p**(-3/2))*cosi*f2(e)))
    dLdt = ((-32/5)*(mu**2)*(p**(-7/2))*((1 - e**2)**(3/2))*(cosi*f3(e) + a*(p**(-3/2))*(f4(e) - (cosi**2)*f5(e))))
    #dQdt = ((-64/5)*(mu**3)*(p**(-3))*((1 - e**2)**(3/2))*(f3(e) - a*(p**(-3/2))*cosi*f6(e)))
    #dCdt = dQdt - 2*L0*dLdt
    dCdt = 2*(C0/L0)*dLdt
    dt = path[-1][0] - path[0][0]
    #print(path[0], "PATH")
    #print(dQ, dC)
    #print(dC, dC + 2*L0*dL, dC - 2*L0*dL)
    
    #print(path[-1,0] - path[0,0])
    #print("no")
    #print(r0*(1-e), r0*(1+e), r0, e)
    yomp = 1
    if path[0][0] == 0.0:
        yomp = 0

    #print(dEdt, dLdt, dQdt, yomp*dCdt, L0, mu*np.sqrt(p), "AUGH")
    E, L, C = E0 + dEdt*dt/mu, L0 + dLdt*dt/mu, C0 + dCdt*dt/mu
    #print("HEY", E0, L0, C0)
    #print("HEY", E, L, C)
    #print(E)
    '''
    R2 = lambda r: (E**2 - 1.0)*(r**4) + 2.0*(r**3) + ((a**2)*(E**2 - 1.0) - L**2 - C)*(r**2) + 2*((a*E - L)**2 + C)*r - C*(a**2)
    test = optimize.fsolve(R2, turns)
    while R2(max(test)) < 0.0:
        dR = -np.polyval(potent, test)
        E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
    '''
    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    #print(len(path), path[-1][0] - path[0][0], dEdt*dt/mu, (2*L0*dLdt*dt/mu + dCdt*dt/mu)/(2*np.sqrt(L0**2 + C)))
    return new_state, [E, L, C]

def new_recalc_state14(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    r, theta, phi, vel4 = *state[1:4], state[4:]
    sint, cost = np.sin(theta), np.cos(theta)
    sinp, cosp = np.sin(phi), np.cos(phi)
    sph2cart = np.array([[1, 0,         0,           0           ],
                         [0, (r/np.sqrt(r**2 + a**2))*sint*cosp, np.sqrt(r**2 + a**2)*cost*cosp, -np.sqrt(r**2 + a**2)*sint*sinp],
                         [0, (r/np.sqrt(r**2 + a**2))*sint*sinp, np.sqrt(r**2 + a**2)*cost*sinp, np.sqrt(r**2 + a**2)*sint*cosp ],
                         [0,                             r*cost,                        -r*sint, 0                              ]])
    vel4cart = np.matmul(sph2cart, vel4)
    vel3cart = vel4cart[1:4]
    pos3cart = np.array([np.sqrt(r**2 + a**2)*sint*cosp, np.sqrt(r**2 + a**2)*sint*sinp, r*cost])
    Lx, Ly, Lz = np.cross(pos3cart, vel3cart)
    dE, dLx, dLy, dLz = con_derv[:4]
    dC = 2*(Lx*dLx + Ly*dLy)
    E, L, C = E0 + dE, L0 + dLz, C0 + dC
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state15(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    r, theta, phi, vel4 = *state[1:4], state[4:]
    sint, cost = np.sin(theta), np.cos(theta)
    sinp, cosp = np.sin(phi), np.cos(phi)
    sph2cart = np.array([[1, 0,         0,           0           ],
                         [0, sint*cosp, r*cost*cosp, -r*sint*sinp],
                         [0, sint*sinp, r*cost*sinp, r*sint*cosp ],
                         [0, cost,      -r*sint,     0           ]])
    vel4cart = np.matmul(sph2cart, vel4)
    vel3cart = vel4cart[1:4]
    pos3cart = np.array([r*sint*cosp, r*sint*sinp, r*cost])
    Lx, Ly, Lz = np.cross(pos3cart, vel3cart)
    dE, dLx, dLy, dLz = con_derv[:4]
    dC = 2*(C0/L0)*dLz
    #dC = 2*(Lx*dLx + Ly*dLy)
    #dC = 2*(Lx*dLx + Ly*dLy)
    E, L, C = E0 + dE, L0 + dLz, C0 + dC
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

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

def seper_locator(r0, inc, a):
    '''
    Locates seperatrix for a given semimajor axis, inclination, and black hole spin

    Parameters
    ----------
    r0 : float
        Semimajor axis in gravitational units
    inc : float
        Inclination of orbit, with pi/2 as equatorial
    a : float
        Dimensionless spin of central black hole

    Returns
    -------
    3-element numpy array
        [Specific Energy, Specific Angular Momentum, Specific Carter constant] of seperatrix orbit
    float
        Eccentricity of orbit

    '''
    r2, r3 = 1, 0
    rmb = find_rmb(a)
    e = (1 - (rmb/r0))*0.5
    e_list = [e]
    loops = 1
    while (r2 - r3 > 1e-11 or r2 - r3 < 0) and loops < 100:
        r1, r2 = r0*(1 + e), r0*(1 - e)
        E, L, C = schmidtparam3(r0, e, inc, a)
        A_B = 2/(1 - E**2) - (r1 + r2)
        AB = (a**2)*C/((1 - E**2)*r1*r2)
        r3 = (A_B + np.sqrt(A_B**2 - 4*AB))/2.0
        #print(r2, r3, r2-r3, e)
        if r2 - r3 > 1e-11:
            new_e = e*(1 + (np.abs(r2 - r3)/np.abs(r1 - r3))**1.0)
            if new_e == e:
                new_e += 5e-13
        elif r2 - r3 < 0:
            if r1 < r3:
                print("r0 gives plunge for all values of e")
                return [False, False, False]
            else:
                new_e = e*(1 + (r2 - r3)/10)
        e = new_e
        e_list.append(e)
        loops += 1
    return [E, L, C], e

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
    return r**5 - C*(r - 3)*r**3 + (a*C)**2

def circE_C(r, a, C, p=1):
    p = float(np.sign(p))**np.sign(p)
    Y = Yfunc(r, a, C)
    sqrt_Y = np.sqrt(Y)
    denom = (r**2) * np.sqrt(r**3*(r - 3) - 2*a*(a*C - p*sqrt_Y))
    numer = r**3*(r - 2) - a*(a*C - p*sqrt_Y)
    return numer / denom

def circL_C(r, a, C, p=1):
    p = float(np.sign(p))**np.sign(p)
    Y = Yfunc(r, a, C)
    sqrt_Y = np.sqrt(Y)
    denom = (r**2) * np.sqrt(r**3*(r - 3) - 2*a*(a*C - p*sqrt_Y))
    numer = -2*a*r**3 - (r**2 + a**2)*(a*C - p*sqrt_Y)
    return numer / denom

def circE_L_C(r, a, C, p=1):
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
    '''Y = Yfunc(r, a, C)
    sqrt_Y = np.sqrt(np.maximum(Y, 0))
    radicand = r**3 * (r - 3) - 2 * a * (a * C - sqrt_Y)
    valid1 = (Y[0] >= 0) & (radicand[0] > 0)
    valid2 = (Y[1] >= 0) & (radicand[1] > 0)
    r[0] = r[0][valid1]
    r[1] = r[1][valid2]
    if len(r) == 0:
        return None, None'''
    E_1, E_2 = circE_C(r[0], a, C), circE_C(r[1], a, C, -1)
    L_1, L_2 = circL_C(r[0], a, C), circL_C(r[1], a, C, -1)
    mask1, mask2 = (0 <= E_1) & (E_1 <= 1), (0 <= E_2) & (E_2 <= 1)
    E_1, E_2 = E_1[mask1], E_2[mask2]
    L_1, L_2 = L_1[mask1], L_2[mask2]
    return E_1, L_1, E_2, L_2

def is_in_ELC_region(E_test, L_test, C_test, a, tol=1e-3):
    E_curve_1, L_curve_1, E_curve_2, L_curve_2 = get_EL_curve(a, C_test, rmax=1000, npts=5000)
    
    if E_curve_1 is None and E_curve_2 is None:
        #print("E_curve is None")
        return False

    # Only consider points where |E - E_test| < tol
    E_diff_1 = np.abs(E_curve_1 - E_test)
    E_diff_2 = np.abs(E_curve_2 - E_test)
    close_indices_1 = np.where(E_diff_1 < tol)[0]
    close_indices_2 = np.where(E_diff_2 < tol)[0]

    if len(close_indices_1) < 2 and len(close_indices_2) < 2:
        #print("close_indices too small")
        return False  # Either no valid E or not enough to get bounds

    Ls_at_E = [L_curve_1[close_indices_1], L_curve_2[close_indices_2]]
    #print(Ls_at_E[0])
    Lmin, Lmax = [0,0], [0,0]

    for i in range(2):
        try:
            mid = circL_C(find_rms(a), a, C_test, 0-i)
            print(find_rmb(a), find_rms(a))
            Lmin[i], Lmax[i] = np.mean(Ls_at_E[i][np.where(Ls_at_E[i] <= mid)]), np.mean(Ls_at_E[i][np.where(Ls_at_E[i] >= mid)])
        except:
            pass
    
    #Lmin, Lmax = [np.min(Ls) for Ls in Ls_at_E], [np.max(Ls) for Ls in Ls_at_E] #np.min(Ls_at_E[1], axis=1), np.max(Ls_at_E[1], ax) #
    #print(Lmin, Lmax)
    #print(Ls_at_E[i][np.where(Ls_at_E[0] <= mid)])

    # Also ensure E < 1 for bound orbits
    val = ((Lmin[0] <= L_test <= Lmax[0]) or (Lmin[1] <= L_test <= Lmax[1])) and (E_test < 1)
 
    #if val == False and (E_test < 1):
    #    print(Lmin[0], L_test, Lmax[0], "starts", E_test)
    #    print(Lmin[1], L_test, Lmax[1])
    return val #((Lmin[0] <= L_test <= Lmax[0]) or (Lmin[1] <= L_test <= Lmax[1])) and (E_test < 1)

def is_in_ELC_region2(E_test, L_test, C_test, a, tol=1e-4):
    #E_curve_1, L_curve_1, E_curve_2, L_curve_2 = get_EL_curve(a, C_test, rmax=1000, npts=5000)
    rmax, npts = 1000, 50000
    r1min = find_rmb(a)        # Marginally bound orbit, smallest possible periapse for an equatorial orbit
    r1_isco = find_rms(a)      # Innermost stable circular orbit - NOT the same as innermost stable spherical orbit
    r2min = find_rmb(-a)        # Marginally bound orbit, smallest possible periapse for an equatorial orbit
    r2_isco = find_rms(-a)      # Innermost stable circular orbit - NOT the same as innermost stable spherical orbit
    #I NEED BETTER BOUNDS, THESE FUCK UP WHEN C != 0

    # Sample more densely near ISCO
    n_half = npts // 2
    r1_inner = np.geomspace(r1min, r1_isco, n_half, endpoint=False)
    r1_outer = np.geomspace(r1_isco, rmax, npts - n_half)
    r2_inner = np.geomspace(r2min, r2_isco, n_half, endpoint=False)
    r2_outer = np.geomspace(r2_isco, rmax, npts - n_half)

    r = np.array([r1_inner, r1_outer, r2_inner, r2_outer])
    Y = Yfunc(r, a, C_test)
    sqrt_Y = np.sqrt(np.maximum(Y, 0))
    radicand = r**3 * (r - 3) - 2 * a * (a * C_test - sqrt_Y)
    valids = [(Y[i] >= 0) & (radicand[i] > 0) for i in range(4)]
    rs = [r[i][valids[i]] for i in range(4)]
    #print(rs[0])
    #print(rs[1])
    #print(rs[2])
    #print(rs[3])
    #print("----")
    print(r1min, r1_isco)
    plt.plot(r1_inner)
    plt.show()
    if len(rs[0]) == 0:
        return None, None
    Es1 = [circE_C(r[i], a, C_test, p=(-1)**(i//2)) for i in range(4)]#, circE_C(r[1], a, C, -1)
    Ls1 = [circL_C(r[i], a, C_test, p=(-1)**(i//2)) for i in range(4)]

    # For some values of C, a, we need to move values from the upper branch to the lower branch or vice versa so we can check things properly!
    # I need to check for the minimum values of E between the inner sections and the outer sections (the points)
    # Then get the corresponding L values
    # Everything HIGHER than that value needs to go in the upper branch, and everything LOWER needs to go in the lower branch
    #Get the minimum E value in each section, plus their indices
    Emins, Emins_ix = np.nanmin(Es1, axis=1), np.nanargmin(Es1, axis=1)
    #Get the corresponding Ls
    Lmins = np.array([Ls1[i][Emins_ix[i]] for i in range(4)])
    Es, Ls = [[],[],[],[]], [[],[],[],[]]
    print(Emins, "HUH", np.nanmin(Es1[0]), np.nanmax(Es1[0]))
    
    for i in range(2): #inner, then outer
        if Emins[0 + i] < Emins[2 + i]:
            Lturn = Lmins[0 + i]
        else:
            Lturn = Lmins[2 + i]
        print(Lturn)
        Es[0 + i] = np.concatenate((Es1[0 + i][Ls1[0 + i] >= Lturn], Es1[2 + i][Ls1[2 + i] >= Lturn]))
        Es[2 + i] = np.concatenate((Es1[0 + i][Ls1[0 + i] < Lturn], Es1[2 + i][Ls1[2 + i] < Lturn]))
        Ls[0 + i] = np.concatenate((Ls1[0 + i][Ls1[0 + i] >= Lturn], Ls1[2 + i][Ls1[2 + i] >= Lturn]))
        Ls[2 + i] = np.concatenate((Ls1[0 + i][Ls1[0 + i] < Lturn], Ls1[2 + i][Ls1[2 + i] < Lturn]))

    if np.all(Es == None):
        print("E_curve is None")
        return False

    # Only consider points where |E - E_test| < tol
    E_diffs = [np.abs(E_group - E_test) for E_group in Es]
    close_indices = [np.where(E_group < tol)[0] for E_group in E_diffs]
    #print([len(thing) for thing in close_indices])

    if len(close_indices[0]) + len(close_indices[1]) < 2 and len(close_indices[2]) + len(close_indices[3]) < 2:
        print("close_indices too small")
        return False  # Either no valid E or not enough to get bounds

    #print(Ls[0][close_indices[0]])
    Ls_at_E = [Ls[i][close_indices[i]] for i in range(4)]
    #print(Ls_at_E[0])
    Lmin, Lmax = [np.mean(Ls_at_E[0]), np.mean(Ls_at_E[2])], [np.mean(Ls_at_E[1]), np.mean(Ls_at_E[3])]
    
    for i in range(4):
        plt.scatter(Es[i][close_indices[i]], Ls[i][close_indices[i]])

    # Also ensure E < 1 for bound orbits
    val = ((Lmin[0] <= L_test <= Lmax[0]) or (Lmin[1] <= L_test <= Lmax[1])) and (E_test < 1)
 
    #if val == False and (E_test < 1):
    #    print(Lmin[0], L_test, Lmax[0], "starts", E_test)
    #    print(Lmin[1], L_test, Lmax[1])
    return val #((Lmin[0] <= L_test <= Lmax[0]) or (Lmin[1] <= L_test <= Lmax[1])) and (E_test < 1)

def is_in_ELC_region3(E_test, L_test, C_test, a, tol=1e-4, rmax=1000):
    
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

def get_sep_inc(a, inc, mult=1, getELC=False):
    mult = max(1, int(mult))
    x = np.sin(inc)
    a2, x2 = a**2, x**2
    r = np.linspace(2*(1 + np.cos((2/3)*np.arccos(-a))), 10, 50*mult)
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
        r = np.linspace(min_r, max_r, 100)
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

    if not getELC:
        return p_sep, e_sep, r

    r2 = r*r
    Y = r*(r2 - a2*(1 - x2))
    G = r*(r2 + a2*(1 - x2))*(r2*(r - 3) + a2*(r + 1)*(1 - x2) + 2*a*x*np.sqrt(Y))
    E_sep = (r2*(r - 2) + a2*r*(1 - x2) + a*x*np.sqrt(Y))/np.sqrt(G)
    L_sep = (-2*a*r2*x2 + (r2 + a2)*x*np.sqrt(Y))/np.sqrt(G)

    mask = E_sep <= 1.0
    p_sep = p_sep[mask]
    e_sep = e_sep[mask]
    L_sep = L_sep[mask]
    r     =     r[mask]
    r2 = r*r
    E_sep = E_sep[mask]

    r_out = np.linspace(max(r), max(r)*10, len(r))
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
    
    return p_sep, e_sep, r, E_sep, L_sep, C_sep, r_out, E_out, L_out, C_out

def get_sep_cosi(a, cosi, mult=1, getELC=False):
    if np.isclose(cosi, 0.0, atol=1e-12):
        # for small cosi it should be basically the same as inc, even with spin
        return get_sep_inc(a, cosi, mult=mult, getELC=getELC)
    
    mult = max(1, int(mult))
    cosi2 = cosi**2
    sini2, sini = 1 - cosi2, np.sqrt(1 - cosi2)
    a2 = a*a
    r = np.linspace(2*(1 + np.cos((2/3)*np.arccos(-a))), 10, 50*mult)
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
        r = np.linspace(min_r, max_r, 100)
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

    if not getELC or np.isclose(cosi, 0.0, atol=1e-12):
        if getELC and np.isclose(cosi, 0.0, atol=1e-12):
            print("No E solution for cosi = 0")
        return p_sep, e_sep, r

    r2 = r*r
    Y = r*(r2*r2 + (2*r2 - 4*r + a2)*a2*sini2)
    G = (r2*(r*r2 - 3*r2 - 2*a2) + (2*r*r2 + a2*r + a2)*a2*sini2)*Xi + 2*r*(3*r2 + a2)*(r*(3*r2 - 4*r + a2)*a*cosi + D*np.sqrt(Y))*a*cosi
    E_sep = (r*(3*r2 - 4*r + a2)*a*cosi + D*np.sqrt(Y))/np.sqrt(G)
    L_sep = r*cosi*Xi/np.sqrt(G)

    mask = E_sep <= 1.0
    p_sep = p_sep[mask]
    e_sep = e_sep[mask]
    L_sep = L_sep[mask]
    r     =     r[mask]
    E_sep = E_sep[mask]
    C_sep = (L_sep**2)*sini2/cosi2

    r_out = np.linspace(max(r), max(r)*10, len(r))
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
    
    return p_sep, e_sep, r, E_sep, L_sep, C_sep, r_out, E_out, L_out, C_out

def peters_sim(a, q, cons=False, params=False, endflag="False"):
    '''
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
    cosi = ((A + L**2 + C) - ((A + L**2 + C)**2 - 4*A*C)**(1/2))/(2*A) if A != 0 else L/np.sqrt(C + L**2)
    # Get changes to values as Peters wrote them (G = c = 1)
    dEpdt = lambda maj, ecc: (-32/5)*q*q*(1+q)*(1 + 73*(ecc**2)/24 + 37*(ecc**4)/96)/((maj**5)*((1 - ecc**2)**(7/2)))
    dLpdt = lambda maj, ecc: (-32/5)*q*q*np.sqrt(1+q)*(1 + 7*(ecc**2)/8)/((maj**(7/2))*((1 - ecc**2)**2))
    dr0dt = lambda maj, ecc: (-64/5)*q*(1+q)*(1 + 73*(ecc**2)/24 + 37*(ecc**4)/96)/((maj**3)*((1 - ecc**2)**(7/2)))
    dedt = lambda maj, ecc: (-304/15)*ecc*q*(1+q)*(1 + 121*(ecc**2)/304)/((maj**4)*((1 - ecc**2)**(5/2)))

    # Get initial dt - this is supposed to be ~1/100 of the full inspiral according to Peters
    beta = (64/5)*q*(1 + q)
    dt = 0.01*(r0**4)/(4*beta)

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
            dr0 = dr0dt(r0, e)*dt
            de = dedt(r0, e)*dt
            # If the change in dr0 is way too big, shrink the step size and try again
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
    dct = {"tracktime": vals[:, 0],
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

def glamp_2002(a, q, cons=False, params=False, endflag="False", eps=1e-3, verbose=True):
    M = 1.989e30 * 1e7
    mu = q*M
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
    #b = (64/5)*mu*M*(mu + M)
    #dt = eps*np.real(((M*r0)**4)/(4*b))/1e37
    dt = (2 * np.pi * (6.67e-11 * M / ((3e8)**3)) * (p**(1.5) - a*np.sign(L))) * 100 * eps
    vals = [[0, E, L, C, p, e, cosi, inc, *turns]]
    oldp = p
    if verbose == True:
        pbar = tqdm(total = 10000, position=0)
        pbar.set_postfix_str(f"Semilat: {r0*(1 - e**2):.4f}, Ecc {e:.4f}, Peri: {r0*(1 - e):.4f}")
        prog = 0
    r_isco, r_mb = find_rms(a), find_rmb(a)
    # Approximate slope of the separatrix for a given spin - not exact
    approx_m = 1/(2*r_mb - r_isco)
    while (np.abs(np.imag(p)/np.real(p)) < 1e-3 and np.real(p)/(1 + np.real(e)) > find_rmb(a)) and not eval(endflag):
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
            if max(np.abs(np.imag(turns[-2:]))/np.abs(np.real(turns[-2:]))) > eps/100:
                dt *= 0.5
                if dt < p_1**1.5 + a:
                    print("Turning Point Complexity Exceeded Allowed Threshold") if verbose == True else None
                    break
            # r_a and r_p should both decrease
            elif (e_1 > 1e-3 and min(np.array(vals[-1][-2:]) - turns[-2:]) < 0) or (e <= 1e-3 and vals[-1][4] - p_1 < 0):
                dt *= 0.999
                if dt < p_1**1.5 + a:
                    print("Turning Point Monotonic Descent Violated") if verbose == True else None
                    break
            # don't let r_a or r_p change too much
            elif max(abs(np.array(vals[-1][-2:]) - turns[-2:])/np.array(vals[-1][-2:])) > eps:
                dt *= 0.5
                if dt < p_1**1.5 + a:
                    print("Turning Point Descent Perturbation Model Violated") if verbose == True else None
                    break
            else:
                p, e, cosi = p_1, e_1, cosi_1
                # Tell the user where we're at right now
                if verbose == True:
                    pbar.set_postfix_str(f"Semilat: {np.real(p_1):.4f}, Ecc {np.real(e_1):.4f}, Peri: {np.real(p_1*(1 - e_1)):.4f}, dt={dt:.4e}")
                    pbar.update(int(np.real(10000*(oldp - p_1)/(oldp - (r_isco + e*(2*r_mb - r_isco))))) - prog)
                    prog = int(np.real(10000*(oldp - p_1)/(oldp - (r_isco + e*(2*r_mb - r_isco)))))
                # Calculate distance and rate of approach to separatrix
                delta = max(abs(np.array(vals[-1][-2:]) - turns[-2:])/np.array(vals[-1][-2:])) 
                
                if delta < eps / 10:
                    dt *= 1.05
                elif delta > eps / 5:
                    dt *= 0.5
                else:
                    dt *= -5.5*(delta) + 1.6

                #dt = max(dt, np.real(((M*p)**4)/(4*b))/1e37)
                dt = max(dt, 10*(2 * np.pi * (6.67e-11 * M / ((3e8)**3)) * (p**(1.5) - a*np.sign(L))))
                vals.append([vals[-1][0] + dt, E, L, C, p, e, cosi, inc, *turns])
        except Exception as bad:
            print(bad)
            break
    pbar.close() if verbose == True else None

    vals = np.real(np.array(vals))
    # Return stuff in the same format as EMRIGenerator for convenience
    dct = {"name": "g2002",
           "tracktime": vals[:, 0],
           "energy": vals[:, 1],
           "phi_momentum": vals[:, 2],
           "carter": vals[:, 3],
           "p": vals[:, 4],
           "e": vals[:, 5],
           "cosi": vals[:, 6],
           "inc": vals[:, 7],
           "it": vals[:, 8],
           "ot": vals[:, 9],
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
    :param dt_cap: Maximum Multiplier for Step Size (optional) - limits step size to maintain accuracy. Should be <= 1
    '''
    M = 1.989e30 * 1e7
    mu = q*M
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
    # Functions in terms of e
    f1 = lambda x: 1 + (73/24)*(x**2) + (37/96)*(x**4)
    f2 = lambda x: 73/12 + (823/24)*(x**2) + (949/32)*(x**4) + (491/192)*(x**6)
    f3 = lambda x: 1 + (7/8)*(x**2)
    f4 = lambda x: 61/24 + (63/8)*(x**2) + (95/64)*(x**4)
    f5 = lambda x: 61/8 + (91/4)*(x**2) + (461/64)*(x**4)
    f6 = lambda x: 85/8 + (211/8)*(x**2) + (517/64)*(x**4)
    
    # Get initial dt - this is supposed to be ~1/10 of the full inspiral according to Peters
    #  but it seems to be way smaller
    #b = (64/5)*mu*M*(mu + M)
    #dt = eps*np.real(((M*r0)**4)/(4*b))/1e37
    dt = (2 * np.pi * (6.67e-11 * M / ((3e8)**3)) * (p**(1.5) - a*np.sign(L))) * 100 * eps

    # Initialize final output and also save the first semilatus rectum for comparison
    vals = [[0, E, L, C, p, e, inc, cosi, *turns[-2:]]]
    oldp = p
    
    # A way to track our progress so we don't get bored out of our minds
    if verbose == True:
        pbar = tqdm(total = 10000, position=0)
        pbar.set_postfix_str(f"Semilat: {r0*(1 - e**2):.4f}, Ecc {e:.4f}, Peri: {r0*(1 - e):.4f}")
        prog = 0
    # Useful to keep track of the ISCO and marginally bound radius for later
    p_ref, e_ref, r_ref = get_sep_cosi(a, cosi, mult=4)
    p_close1, e_close1 = p_ref.flat[np.nanargmin(np.abs(p_ref - p))], e_ref.flat[np.nanargmin(np.abs(e_ref - e))]
    dist1 = np.sqrt((p_close1 - p)**2 + (e_close1 - e)**2)

    r_isco, r_mb = find_rms(a*np.sign(L)), find_rmb(a*np.sign(L))
    # Approximate slope of the separatrix for a given spin - not exact
    approx_m = 1/(2*r_mb - r_isco)
    # Keep going while p is not super complex and the periapsis is above the marginally bound radius
    while np.abs(np.imag(p)/np.real(p)) < eps and not eval(endflag):# and np.real(p)/(1 + np.real(e)) > r_mb) and not eval(endflag):
        try:
            # Define the predicted changes in E, L, C, Q
            E_dot = lambda p, e, cosi: -(32/5)*(q**2)*(p**-5)*((1 - e**2)**1.5)*(f1(e) - a*(p**-1.5)*cosi*f2(e))
            L_dot = lambda p, e, cosi: (-32/5)*q*q*(p**-3.5)*((1 - e**2)**1.5)*(cosi*f3(e) + a*(p**-1.5)*(f4(e) - (cosi**2)*f5(e)))
            C_dot = lambda p, e, cosi, duh: (-64/5)*(q*q)*(p**-3)*((1 - e**2)**1.5)*(1 - cosi**2)*(f3(e) - a*(p**-1.5)*cosi*f6(e))
            #C_dot = lambda p, e, cosi, L: Q_dot(p, e, cosi) - 2*L_dot(p, e, cosi)*L
            inc = np.sign(L)*np.arccos(min(1, np.mean(np.abs(zs[1:3]))))
            
            # Save current parameters for comparison, make them real so they're useful
            #  really only using p_0 at this point anyway, and if it were too complex we wouldn't be here
            p_0, e_0, cosi_0 = np.real(p), np.real(e), np.real(cosi)
            
            # Get the true change in E from the corrections made to the 2006 paper
            E_0, L_0 = circE_inc(p_0, a, inc), circL_inc(p_0, a, inc)
            N1 = E_0*(p_0**4) + (a**2)*E_0*(p_0**2) - 2*a*(L_0 - a*E_0)*p_0
            N4 = (2*p_0 - p_0**2)*L_0 - 2*a*E_0*p_0
            N5 = (2*p_0 - p_0**2 - a**2)/2
            E_dot_true = ((1 - e_0**2)**1.5)*(((1 - e_0**2)**(-1.5))*E_dot(p_0, e_0, cosi_0) - E_dot(p_0, 0, cosi_0) - (N4/N1)*L_dot(p_0, 0, cosi_0) - (N5/N1)*C_dot(p_0, 0.0, cosi_0, L_0))
            # If it's complex we have a problem! Bail!
            if not np.real(E_dot_true):
                print(E_dot_true, "grah", E_dot(p_0, 0, cosi_0), L_dot(p_0, 0, cosi_0), C_dot(p_0, 0, cosi_0, L), len(vals))
                break
            # Apply changes
            E, L, C = np.real([E + E_dot_true*dt, L + L_dot(p_0, e_0, cosi_0)*dt, C + C_dot(p_0, e_0, cosi_0, L)*dt])
            # If they don't make a viable orbit we have a problem! Bail!
            try:
                turns, flats, zs = root_getter(E, L, C, a)
                r_p, r_a = turns[-2:]
            except:
                print(E, L, C, a, E_0, L_0, E_dot_true, dt, p, e, cosi)
                print(dt, E_dot_true, E_dot(p, e, cosi), E_dot(p, 0, cosi), C_dot(p, e, cosi, L), N1, N4, N5, E_0, L_0, p, p_0, a, C_0, np.sign(cosi), cosi)
                break
            # Get preliminary new parameters
            p_1, e_1 = 2*r_a*r_p/(r_a + r_p), (r_a - r_p)/(r_a + r_p)
            #print("WUGUGGH", E_dot(p, e, cosi), L_dot(p, e, cosi), C_dot(p, e, cosi, 3), dt)
            #print(E, L, C)
            cosi_1 = L/np.sqrt(C + L**2)

            # If r_a or r_p are too complex, shrink that biz
            if max(np.abs(np.imag(turns[-2:]))/np.abs(np.real(turns[-2:]))) > eps/100:
                dt *= 0.5
                if dt < p_0**1.5 + a:
                    print("Turning Point Complexity Exceeded Allowed Threshold") if verbose == True else None
                    break
            # r_a and r_p should both decrease
            elif (e_0 > 1e-3 and min(np.array(vals[-1][-2:]) - turns[-2:]) < 0) or (e <= 1e-3 and vals[-1][4] - p_0 < 0):
                dt *= 0.999
                if dt < p_0**1.5 + a:
                    print("Turning Point Monotonic Descent Violated") if verbose == True else None
                    break
            # don't let r_a or r_p change too much
            elif max(abs(np.array(vals[-1][-2:]) - turns[-2:])/np.array(vals[-1][-2:])) > eps:
                dt *= 0.5
                if dt < p_0**1.5 + a:
                    print("Turning Point Descent Perturbation Model Violated") if verbose == True else None
                    break
            # If it's not any of those, you're not done yet. Keep going!
            else:
                # Accept new values of parameters
                p, e, cosi = p_1, e_1, cosi_1
                delta = max(abs(np.array(vals[-1][-2:]) - turns[-2:])/np.array(vals[-1][-2:])) 
                
                if delta < eps / 10:
                    dt *= 1.05
                elif delta > eps / 5:
                    dt *= 0.5
                else:
                    dt *= -5.5*(delta) + 1.6

                #dt = max(dt, np.real(((M*p)**4)/(4*b))/1e37)
                dt = max(dt, 10*(2 * np.pi * (6.67e-11 * M / ((3e8)**3)) * (p**(1.5) - a*np.sign(L))))

                # Tell the user where we're at right now
                if verbose == True:
                    pbar.set_postfix_str(f"Semilat: {np.real(p):.4f}, Ecc {np.real(e):.4f}, Peri: {np.real(p*(1 - e)):.4f}, dt={dt:.4e}")
                    #pbar.update(int(np.real(10000*(oldp - r0*(1 - e**2))/(oldp - (r_isco + e*(2*r_mb - r_isco))))) - prog)
                    #prog = int(np.real(10000*(oldp - r0*(1 - e**2))/(oldp - (r_isco + e*(2*r_mb - r_isco)))))
                    p_close, e_close = p_ref.flat[np.nanargmin(np.abs(p_ref - p))], e_ref.flat[np.nanargmin(np.abs(e_ref - e))]
                    dist = np.sqrt((p_close - p)**2 + (e_close - e)**2)
                    pbar.update(int(max(0, np.real(10000*(dist1 - dist)/dist1) - prog)))
                    prog = int(np.real(10000*((dist1 - dist)/dist1)))

                r0 = p/(1 - e**2)
                vals.append([vals[-1][0] + dt, E, L, C, p, e, inc, cosi, *turns[-2:]])

        except Exception as e:
            print(e)
            break
    pbar.close() if verbose == True else None
    vals = np.real(np.array(vals))
    # Return stuff in the same format as EMRIGenerator for convenience
    dct = {"name": "gg2006",
           "tracktime": vals[:, 0],
           "energy": vals[:, 1],
           "phi_momentum": vals[:, 2]*((-1)**flip),
           "carter": vals[:, 3],
           "p": vals[:, 4],
           "e": vals[:, 5],
           "inc": vals[:, 6]*((-1)**flip),
           "cosi": vals[:, 7]*((-1)**flip),
           "it": vals[:, 8],
           "ot": vals[:, 9],
           "spin": a*((-1)**flip)}
    return dct
