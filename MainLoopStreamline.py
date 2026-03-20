# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:30:25 2023

@author: camcinto
"""

import numpy as np
import MetricMathStreamline as mm
from scipy.signal import argrelmin
from tqdm import tqdm
import scipy.interpolate as spi
import OrbitPlotter as op
from scipy import optimize
import sympy as sp
from numba import njit
import h5py
import json
import zlib
import base64
import os
import statistics as st
import time
from dataclasses import dataclass
import operator

def getEnergy(state, a):
    '''
    Calculates energy per unit mass for a given position, trajectory, and black hole spin

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    a : float
        Dimensionless spin parameter of the central body. Valid for values between -1 and 1.

    Returns
    -------
    ene : float
        Energy per unit mass
    '''
    metric, chris = mm.kerr(state, a)
    stuff = np.matmul(metric, state[4:])
    ene = -stuff[0]
    #print(stuff)
    return ene

def getCons(state, a):
    '''
    Calculates energy per unit mass for a given position, trajectory, and black hole spin

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    a : float
        Dimensionless spin parameter of the central body. Valid for values between -1 and 1.

    Returns
    -------
    ene : float
        Energy per unit mass
    '''
    metric, chris = mm.kerr(state, a)
    stuff = np.matmul(metric, state[4:])
    ene = -stuff[0]
    Lz = stuff[3]
    Q = np.matmul(np.matmul(mm.kill_tensor(state, a), state[4:]), state[4:])
    cart = Q - (a*ene - Lz)**2
    return np.array([ene, Lz, cart])
    
@njit
def getCons_2(state, a):
    metric = mm.kerr_2(state, a)[0]  # assuming this returns (metric, chris), and we only need metric
    u = state[4:]

    # Replace np.matmul(metric, u) with explicit dot product
    stuff = np.zeros(4)
    for i in range(4):
        for j in range(4):
            stuff[i] += metric[i, j] * u[j]

    ene = -stuff[0]
    Lz = stuff[3]

    kill = mm.kill_tensor_njit(state, a)
    temp = 0.0
    for i in range(4):
        for j in range(4):
            temp += kill[i, j] * u[i] * u[j]

    cart = temp - (a * ene - Lz)**2
    return np.array([ene, Lz, cart])

@njit
def getCons_3(state, a):
    metric = mm.kerr_2(state, a)[0]  # assuming this returns (metric, chris), and we only need metric
    u = state[4:]

    # Replace np.matmul(metric, u) with explicit dot product
    stuff = np.zeros(4)
    for i in range(4):
        for j in range(4):
            stuff[i] += metric[i, j] * u[j]

    ene = -stuff[0]
    Lz = stuff[3]

    kill = mm.kill_tensor_njit(state, a)
    temp = 0.0
    for i in range(4):
        for j in range(4):
            temp += kill[i, j] * u[i] * u[j]

    cart = temp - (a * ene - Lz)**2

    # interval = g_{ij} u^i u^j
    interval = 0.0
    for i in range(4):
        for j in range(4):
            interval += metric[i, j] * u[i] * u[j]
    return np.array([ene, Lz, cart, interval])

def getLs(state, mu):
    '''
    Returns Cartesian angular momentum given position, trajectory, and mass ratio

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    mu : float
        Mass ratio between secondary body and central body. EMRI systems require mu to be less than or equal to 10^-4.

    Returns
    -------
    Lmom : 3 element numpy array
        x, y, and z components of Cartesian angular momentum, where the z-component is parallel to the central body's rotational axis

    '''
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
    Lmom = np.cross(pos3cart, vel3cart)
    return Lmom

def big_sph2cart(vec, a):
    t, r, th, ph, t_t, r_t, th_t, ph_t = vec
    new_vel = [r_t*np.sin(th)*np.cos(ph) + r*th_t*np.cos(th)*np.cos(ph) - r*ph_t*np.sin(th)*np.sin(ph),
               r_t*np.sin(th)*np.sin(ph) + r*th_t*np.cos(th)*np.sin(ph) + r*ph_t*np.sin(th)*np.cos(ph),
               r_t*np.cos(th) - r*th_t*np.sin(th)]
    new_vec = np.array([t, r*np.sin(th)*np.cos(ph), r*np.sin(th)*np.sin(ph), r*np.cos(th), t_t, *new_vel])
    return new_vec

def new_sph2cart(vec, a):
    x = np.sqrt(vec[1]**2 + a**2) * np.sin(vec[2]) * np.cos(vec[3])
    y = np.sqrt(vec[1]**2 + a**2) * np.sin(vec[2]) * np.sin(vec[3])
    z = vec[1] * np.cos(vec[2]) 
    vx = vec[1]*vec[5]/(np.sqrt(vec[1]**2 + a**2)) * np.sin(vec[2]) * np.cos(vec[3]) + np.sqrt(vec[1]**2 + a**2) * vec[6] * np.cos(vec[2]) * np.cos(vec[3]) + np.sqrt(vec[1]**2 + a**2) * np.sin(vec[2]) * vec[7] * (-np.sin(vec[3]))
    vy = vec[1]*vec[5]/(np.sqrt(vec[1]**2 + a**2)) * np.sin(vec[2]) * np.sin(vec[3]) + np.sqrt(vec[1]**2 + a**2) * vec[6] * np.cos(vec[2]) * np.sin(vec[3]) + np.sqrt(vec[1]**2 + a**2) * np.sin(vec[2]) * vec[7] * np.cos(vec[3])
    vz = vec[5] * np.cos(vec[2]) + vec[1] * vec[6] * (-np.sin(vec[2]))
    new_vec = np.array([vec[0], x, y, z, vec[4], vx, vy, vz])
    return new_vec

def cart2sph(cart_vec, a):   #inaccurate, kind of
    t, x, y, z, tdot, vx, vy, vz = cart_vec
    
    # coordinates
    R = np.sqrt(x**2 + y**2 + (z*a/r if False else 0)**2)  # keep safe
    r = np.sqrt(x**2 + y**2 + z**2 - a**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)%(np.pi) + (0 if cart_vec[2] >= 0 else np.pi)
    
    # matrix relating (ṙ, θ̇, φ̇) to (vx, vy, vz)
    Rtot = np.sqrt(r**2 + a**2)
    M = np.array([
        [ (r/Rtot)*np.sin(theta)*np.cos(phi),  Rtot*np.cos(theta)*np.cos(phi),  -Rtot*np.sin(theta)*np.sin(phi) ],
        [ (r/Rtot)*np.sin(theta)*np.sin(phi),  Rtot*np.cos(theta)*np.sin(phi),   Rtot*np.sin(theta)*np.cos(phi) ],
        [ (r/Rtot)*np.cos(theta),             -r*np.sin(theta),                  0                              ]
    ])
    
    rhs = np.array([vx, vy, vz])
    rdot, thetadot, phidot = np.linalg.solve(M, rhs)
    
    sph_vec = np.array([t, r, theta, phi, tdot, rdot, thetadot, phidot])
    return sph_vec


def vec_rot(vec, axis, angle):
    posvec, velvec = vec[1:4], vec[5:8]
    new_posvec = posvec*np.cos(angle) + (np.cross(axis, posvec))*np.sin(angle) + axis*np.dot(axis, posvec)*(1 - np.cos(angle))
    new_velvec = velvec*np.cos(angle) + (np.cross(axis, velvec))*np.sin(angle) + axis*np.dot(axis, velvec)*(1 - np.cos(angle))
    new_vec = np.array([vec[0], *new_posvec, vec[4], *new_velvec])
    return new_vec

def new_rot(vec, angle):
    # Rotates the thing by some angle around the x-axis
    t, r, th, ph, t_t, r_t, th_t, ph_t = vec
    st, ct, sp, cp, sa, ca = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph), np.sin(angle), np.cos(angle)
    new_th = np.arccos(ct*ca+st*sp*sa)
    new_ph = np.arctan((sp/cp)*ca - ct*sa/(st*cp))
    gam = (sp/cp)*ca - ct*sa/(st*cp)
    new_th_t = -(-th_t*st*ca + (th_t*ct*sp + ph_t*st*cp)*sa)*((1 - (ct*ca+st*sp*sa)**2)**(-1/2))
    new_ph_p = (ph_t*(ca/(cp**2)) + (th_t/st - ph_t*ct*sp/cp)*(sa/(st*cp)))/(gam**2 + 1)
    return np.array([t, r, new_th, new_ph, t_t, r_t, new_th_t, new_ph_p])

@njit
def fast_err_with_constants(state1, state2, a):
    # Basic difference in position and 4-velocity components (excluding t and phi)
    coords1 = np.array([state1[1], state1[2], state1[4], state1[5], state1[6], state1[7]])
    coords2 = np.array([state2[1], state2[2], state2[4], state2[5], state2[6], state2[7]])
    delt_coords = coords1 - coords2

    # Simple scaling for position/velocity
    scales_coords = np.array([1e1, 1e-1, 1e0, 1e-1, 1e-1, 1e0])
    scaled_coords = delt_coords / scales_coords

    # Constants calculation (you'd want to inline this part)
    E1, L1, C1 = getCons_2(state1, a)
    E2, L2, C2 = getCons_2(state2, a)
    delt_consts = np.array([E1 - E2, L1 - L2, C1 - C2])

    scales_consts = np.array([1e-4, 1e-2, 1e-1])
    scaled_consts = delt_consts / scales_consts

    # Combine and compute RMS
    all_errors = np.concatenate((scaled_coords, scaled_consts))
    return np.sqrt(np.mean(all_errors**2))

@njit
def fast_err_with_constants2(state1, state2, a):
    # Basic difference in position and 4-velocity components (excluding t and phi)
    coords1 = np.array([state1[1], state1[2], state1[4], state1[5], state1[6], state1[7]])
    coords2 = np.array([state2[1], state2[2], state2[4], state2[5], state2[6], state2[7]])
    delt_coords = coords1 - coords2

    # Simple scaling for position/velocity
    scales_coords = np.array([1e1, 1e-1, 1e0, 1e-1, 1e-1, 1e0])
    scaled_coords = delt_coords / scales_coords

    # Constants calculation (you'd want to inline this part)
    E1, L1, C1 = getCons_2(state1, a)
    E2, L2, C2 = getCons_2(state2, a)
    delt_consts = np.array([E1 - E2, L1 - L2, C1 - C2])

    scales_consts = np.array([1e-4, 1e-2, 1e-1])
    scaled_consts = delt_consts / scales_consts

    # Combine and compute RMS
    all_errors = np.concatenate((scaled_coords, scaled_consts))
    return np.min(all_errors**2)

@njit
def quick_state(constants, state, a):
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


@dataclass
class EndContext:
    i: int
    all_states: list
    true_anom: list
    tracker: list
    j: int

def EMRIGenerator(a, mu, endflag="min_radius < 0.5", mass=1.0, err_target=1e-15, label="default", cons=False, velorient=False, vel4=False, params=False, pos=False, veltrue=False, units="grav", verbose=1, eps=1e-5, trigger=2, override=False, bonk2=True, skip_tar=0, nofix=True, time_reverse=False, weird=False, force_stop=None):
    '''
    Generates orbit

    Parameters
    ----------
    a : float
        Dimensionless spin parameter of the central body. Valid for values between -1 and 1.
    mu : float
        Mass ratio between secondary body and central body. EMRI systems require mu to be less than or equal to 10^-4.
    endflag : string, optional
        Condition for ending the simulation, written in the form '(variable) (comp.operator) (value)'
        Current valid variables:
            time - time, measured in geometric units
            phi_orbit - absolute phi displacement from original position, measured in radians
            rad_orbit - number of completed radial oscillations
            radius - distance from central body, measured in geometric units
            inclination - maximum displacement from north pole of central body, measured in radians
            semilat - semilatus rectum of orbit, measured in geometric units
            eccentricity - eccentricity of orbit
        The default is "radius < 2"
    mass : float, optional
        Mass of the central body. The default is 1.0.
    err_target : float, optional
        Maximum error allowed during the geodesic evaluation. The default is 1e-15.
    label : string, optional
        Internal label for the simulation. The default is "default", which gives it a label based on Keplerian paramters.
    cons : 3-element list of floats, optional
        Energy, Angular Momentum, and Carter Constant per unit mass. The default is False.
    velorient : 3-element list/array of floats, optional
        Ratio of velocity/speed of light (beta), angle between r-hat and trajectory (eta - radians), angle between phi hat and trajectory (xi - radians)
    vel4 : 4-element list/array of floats, optional
        Tetrad component velocities [t, r, theta, phi].
    params : 3-element list/array of floats, optional
        Semimajor axis, eccentricity, and inclination of orbit.
    pos : 4-element list/array of floats, optional
        Initial 4-position of particle. The default is False
    veltrue : 4-element list/array of floats, optional
        Initial 4-velocity of particle. The default is False.
    units : string, optional
        System of units for final output. The default is "grav".
        Current valid units:
            'grav' - Geometric units, with G, c, and M (central body mass) all set to 1.0.
            'mks' - Standard SI units, with G = 6.67e-11 N*m^2*kg^-2, c = 3e8 m*s^-1, and M in kg
            'cgs' - Standard cgs units, with G = 6.67e-8 dyn*cm^2*g^-2, c = 3e11 cm*s^-1, and M in g
    verbose : int, optional
        Toggle for progress updates as program runs. The default is 1.
        0 - No output
        1 - progress bar
        2 - full output

    Returns
    -------
    final: 35 element dict
        Various tracked and record-keeping values for the resulting orbit
        "name": Label for orbit if plotted, defaults to a list of Keplerian values for initial trajectory
        "raw": 8 element state of the orbiting body from beginning to end [time, radius, theta, phi, dt, dradius, dtheta, dphi]
        "inputs": initial input for function
        "pos": Subset of "raw", only includes radius, theta position, and phi positions
        "all_vel": Subset of "raw", only includes time, radius, theta position, and phi velocities
        "time": Subset of "raw", only includes time
        "true_anom": True anomaly measured at every moment in "time"; approximate
        "interval": Derived from "raw", spacetime interval at every point measured in "time"; should equal -1 at all times
        "vel": Derived from "raw", absolute velocity w.r.t. Mino time
        "dTau_change": Change in timestep 
        "energy": Energy of orbiting body at points of recalculation
        "phi_momentum": Angular momentum of orbiting body at points of recalculation
        "carter": Carter Constant of orbiting body (set to 0 for equatorial orbits) at points of recalculation
        "qarter": Carter Constant of orbiting body at points of recalculation
        "energy2": Specific Energy of orbiting body at all points in "time"
        "Lx_momentum": X-component of Specific Angular Momentum of orbiting body at all points in "time"
        "Ly_momentum": Y-component of Specific Angular Momentum of orbiting body at all points in "time"
        "Lz_momentum": Z-component of Specific Angular Momentum of orbiting body at all points in "time"
        "spin": Dimensionless spin of central body
        "freqs": Characteristic frequencies of orbit w.r.t. time at points of recalculation [radial, theta, phi]
        "pot_min": Radial distance of potential minimum at points of recalculation
        "e": Eccentricity at points of recalculation
        "inc": Inclination at points of recalculation
        "it": Inner turning point at points of recalculation
        "ot": Outer turning point at points of recalculation
        "r0": Semimajor axis at points of recalculation
        "tracktime": Value of time corresponding to points of recalculation
        "trackix": Indices of "raw" corresponding to points of recalculation
        "omega": Phi position of periapse
        "otime": Time at periapse
        "asc_node": Phi position of ascending node
        "asc_node_time": Time at ascending node
        "stop": 'True' if simulation was aborted before reaching end condition, False otherwise
        "plunge": 'True' if simulation ended in a plunge, False otherwise
        "issues": index and state corresponding to any point where Keplerian values read as complex
    '''
    #termdict = {"time": "abs(all_states[i][0] - all_states[0][0])",
    #            "radius": "all_states[i][1]",
    #            "rad_orbit": "abs((true_anom[i] - true_anom[0])/(2*np.pi))",
    #            "phi_orbit": "abs(abs(all_states[i][3]/(2*np.pi)) - abs(all_states[0][3]/(2*np.pi)))",
    #            "semilat": "2*tracker[j][3]*tracker[j][4]/(tracker[j][3] + tracker[j][4])",
    #            "eccentricity": "tracker[j][1]"}
    
    endvars = {"time": lambda c: abs(c.all_states[c.i][0] - c.all_states[0][0]),
               "min_radius": lambda c: c.last_chunk[:,1].min(),
               "max_radius": lambda c: c.last_chunk[:,1].max(),
               "rad_orbit": lambda c: abs((c.true_anom[c.i] - c.true_anom[0]) / (2*np.pi)),
               "phi_orbit": lambda c: abs(abs(c.all_states[c.i][3] / (2*np.pi)) - abs(c.all_states[0][3] / (2*np.pi))),
               "semilat": lambda c: (2 * c.tracker[c.j][3] * c.tracker[c.j][4] / (c.tracker[c.j][3] + c.tracker[c.j][4])),
               "eccentricity": lambda c: c.tracker[c.j][1],
               "inclination": lambda c: c.tracker[c.j][2]}
    
    OPS = {
        ">=": operator.ge,
        "<=": operator.le,
        ">": operator.gt,
        "<": operator.lt,
    }

    def parse_endflag(flag: str):
        for op_str, op in OPS.items():
            if op_str in flag:
                lhs, rhs = flag.split(op_str)
                lhs = lhs.strip()
                rhs = float(rhs)
                return endvars[lhs], op, rhs
        raise ValueError("Invalid end condition")
    
    #try:
    #    if "custom" in endflag:
    #        terms = ["custom"]
    #        newflag = input("Input custom endflag:\n")
    #    else:
    #        terms = endflag.split(" ")
    #        newflag = termdict[terms[0]] + terms[1] + terms[2]
    #except:
    #    print("Endflag should be a valid variable name, comparison operator, and numerical value, all separated by spaces")
    #    return 0

    time_check = (-1)**time_reverse
    inputs = [mass, a, mu, endflag, err_target, label, cons, velorient, vel4, params, pos, veltrue, units]          #Grab initial input in case you want to run the continue function
    inputs = [entry.tolist() if type(entry) == np.ndarray else entry for entry in inputs]                           #Convert numpy arrays to lists so that JSON doesn't complain
    all_states = [[np.zeros(8)]]                                                  #Grab that initial state         
    err_calc = err_target*1.01
    i = 0                                                                         #initialize step counter
    if (np.shape(veltrue) == (4,)) and (np.shape(pos) == (4,)):
        all_states[0] = np.array([*pos, *veltrue])
    else:
        if verbose == True:
            print("Normalizing initial state")
        all_states[0], cons = mm.set_u_kerr(a, cons, velorient, vel4, params, pos)      #normalize initial state so it's actually physical
    
    metric, chris = mm.kerr_2(all_states[0], a)                                     #initial metric and christoffel symbols
    interval = [mm.check_interval_w_metric(metric, all_states[0], a)]           #create interval tracker
    
    def viable_cons0(new_cons, old_cons, state, a, scream=False):
        #print("----")
        E1, L1, C1 = old_cons
        E2, L2, C2 = new_cons
        R = lambda r, E, L, C, a: ((r**2 + a**2)*E - a*L)**2 - (r**2 - 2*r + a**2)*(r**2 + (L - a*E)**2 + C)
        Rpoly = lambda E, L, C, a: np.array([(E**2 - 1.0), 2.0*np.ones_like(E), ((a**2)*(E**2 - 1.0) - L**2 - C),  (2*((L - a*E)**2) + 2*C), -(a**2)*C])
        if scream == True:
            import matplotlib.pyplot as plt
            turns1, flats1, zs1 = mm.root_getter(E1, L1, C1, a)
            turns2, flats2, zs2 = mm.root_getter(E2, L2, C2, a)
            low, high = min(turns1[-2], turns2[-2]), max(turns1[-1], turns2[-1])
            low_b, high_b = low - 0.01*(high - low), high + 0.01*(high - low) 
            r_vals = np.real(np.linspace(low_b, high_b, num=100))
            fig, ax = plt.subplots()
            ax.hlines(0, r_vals[0], r_vals[-1])
            #ax.plot(r_vals, np.polyval(Rpoly(*old_cons, a), r_vals))
            #ax.plot(r_vals, np.polyval(Rpoly(*new_cons, a), r_vals))
            #ax.scatter(state[1], np.polyval(Rpoly(*old_cons, a), state[1]))
            #ax.scatter(state[1], np.polyval(Rpoly(*new_cons, a), state[1]))
            ax.plot(r_vals, R(r_vals, *old_cons, a))
            ax.plot(r_vals, R(r_vals, *new_cons, a))
            ax.scatter(state[1], R(state[1], *old_cons, a))
            ax.scatter(state[1], R(state[1], *new_cons, a))
            ax.set_title("Viable Cons scream")
            plt.show()
            #print(turns1)
            #print(turns2)
            #print(old_cons)
            #print(new_cons)
            #print(new_cons - old_cons)
        potential_min = R(mm.root_getter(*new_cons, a)[1][-1], *new_cons, a)
        #print("uh", *new_cons, a)
        #print("HEWWO??", mm.root_getter(*new_cons, a))
        return potential_min
    
    def viable_cons(new_cons, old_cons, state, a,
                    scream=False,
                    rtol=1e-12):
        """
        Checks whether updated constants of motion admit a viable bound orbit
        consistent with the current radial phase.

        Returns
        -------
        Rmin : float
            Minimum of radial potential between turning points.
            Negative values indicate unphysical constants.
        """

        E2, L2, C2 = new_cons
        r0 = state[1]

        # Radial potential
        def R(r, E, L, C, a):
            return ((r**2 + a**2)*E - a*L)**2 \
                - (r**2 - 2*r + a**2)*(r**2 + (L - a*E)**2 + C)

        # Get roots and extrema
        turns, flats, _ = mm.root_getter(E2, L2, C2, a)

        # Keep sufficiently real roots
        turns = np.real(turns[np.abs(np.imag(turns)/np.real(turns + 1e-4)) < 1e-4]).astype(float)
        flats = np.real(flats[np.abs(np.imag(flats)/np.real(flats + 1e-4)) < 1e-4]).astype(float)

        if len(turns) < 2:
            return -np.inf

        # Outer turning points
        r_peri, r_apo = turns[-2], turns[-1]

        # Phase consistency check
        if not (r_peri - rtol <= r0 <= r_apo + rtol):
            return -np.inf

        # Valid extrema inside the allowed region
        valid_flats = flats[(flats > r_peri) & (flats < r_apo)]

        if len(valid_flats) == 0:
            return -np.inf

        # Evaluate R at all candidate minima
        Rvals = R(valid_flats, E2, L2, C2, a)
        Rmin = np.min(Rvals)

        # Optional diagnostic plot
        if scream:
            import matplotlib.pyplot as plt
            r_plot = np.linspace(r_peri*0.98, r_apo*1.02, 400)
            plt.figure()
            plt.axhline(0, color='k', lw=0.5)
            plt.plot(r_plot, R(r_plot, E2, L2, C2, a))
            plt.scatter(valid_flats, Rvals, color='red')
            plt.scatter([r0], [R(r0, E2, L2, C2, a)], color='blue')
            plt.title("Radial potential viability check")
            plt.xlabel("r")
            plt.ylabel("R(r)")
            plt.show()

        return Rmin


    def bl2cart_oof(state, a):
        t, r, thet, phi, ut, ur, uthet, uphi = state
        sint, cost, sinp, cosp = np.sin(thet), np.cos(thet), np.sin(phi), np.cos(phi)
        new = [t, np.sqrt(r**2 + a**2)*sint*cosp, np.sqrt(r**2 + a**2)*sint*sinp, r*cost,
                ut, r*ur*sint*cosp/np.sqrt(r**2 + a**2) + np.sqrt(r**2 + a**2)*(uthet*cost*cosp - uphi*sint*sinp),
                r*ur*sint*sinp/np.sqrt(r**2 + a**2) + np.sqrt(r**2 + a**2)*(uthet*cost*sinp + uphi*sint*cosp),
                ur*cost - r*uthet*sint]
        return np.array(new)

    def get_true_anom(state, r0, e):
        pre = np.sign((r0*(1 - e**2)/state[1] - 1)) #e is always positive
        val = np.arccos(pre*min(1.0, abs((r0*(1 - e**2)/state[1] - 1)/(e + 1e-15)))) #add a little tiny bias to get rid of divide by zero errors
        if state[5] < 0:
            val = 2*np.pi - val
        return val
    
    def get_true_anom_vec(r, r_dot, r0, e, orbcount):
        pre = np.sign((r0*(1 - e**2)/r - 1)) #e is always positive
        val = np.arccos(pre*np.minimum(1.0, abs((r0*(1 - e**2)/r - 1)/(e + 1e-15)))) #add a little tiny bias to get rid of divide by zero errors
        val = np.where(r_dot < 0, 2*np.pi - val, val) + orbcount*2*np.pi
        ix = np.where(np.diff(val) < -np.pi)[0]
        for num in ix:
            val[num+1:] += 2*np.pi
            orbcount += 1
        return val, orbcount
    
    if np.shape(cons) == (3,):
        initE, initLz, initC = cons
        initQ = initC + (a*initE - initLz)**2
    else:
        initE, initLz, initC = getCons_2(all_states[0], a)
        #initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
        #initLz = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
        #initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
        #initC = initQ - (a*initE - initLz)**2                                          #initial adjusted Carter constant 
        initQ = initC + (a*initE - initLz)**2
    pot_min = viable_cons([initE, initLz, initC], [initE, initLz, initC], all_states[0], a)
    count = 0
    while pot_min < 0.0:
        #print(pot_min)
        count += 1
        initE += err_target
        pot_min = viable_cons([initE, initLz, initC], [initE, initLz, initC], all_states[0], a)
        if count >= 21:
            print("Don't trust this!", pot_min, inputs)
            initE -= count*err_target
            break
                
    coeff = np.array([initE**2 - 1, 2.0, (a**2)*(initE**2 - 1) - initLz**2 - initC, 2*((a*initE - initLz)**2) + 2*initC, -initC*(a**2)])
    coeff2 = np.polyder(coeff)
    keps = np.array([np.sort(np.roots(coeff2))[-1], *np.sort(np.real(np.roots(coeff)))[-2:]])
    pot_min, inner_turn, outer_turn = keps.real[abs(keps.imag)<(1e-6)*abs(keps[0])]
    e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
    A = (a**2)*(1 - initE**2)
    z2 = np.round(((A + initLz**2 + initC) - ((A + initLz**2 + initC)**2 - 4*A*initC)**(1/2))/(2*A), 13) if A != 0 else np.round(initC/(initLz**2 + initC), 13)
    inc = np.sign(initLz)*np.arccos(np.sqrt(z2))
    tracker = [[pot_min, e, inc, inner_turn, outer_turn, all_states[0][0], 0]]
    if True in np.iscomplex(tracker[0]):
        initE = (4*a*initLz*pot_min + ((4*a*initLz*pot_min)**2 - 4*(pot_min**4 + 2*pot_min*(a**2))*((a*initLz)**2 - (pot_min**2 - 2*pot_min + a**2)*(pot_min**2 + initLz**2 + initC)))**(0.5))/(2*(pot_min**4 + 2*pot_min*(a**2)))
        coeff = np.array([initE**2 - 1, 2.0, (a**2)*(initE**2 - 1) - initLz**2 - initC, 2*((a*initE - initLz)**2) + 2*initC, -initC*(a**2)])
        coeff2 = np.polyder(coeff)
        keps = np.array([np.sort(np.roots(coeff2))[-1], *np.sort(np.real(np.roots(coeff)))[-2:]])
        pot_min, inner_turn, outer_turn = keps.real[abs(keps.imag)<(1e-6)*abs(keps[0])]
        e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
        A = (a**2)*(1 - initE**2)
        z2 = ((A + initLz**2 + initC) - ((A + initLz**2 + initC)**2 - 4*A*initC)**(1/2))/(2*A) if A != 0 else initC/(initLz**2 + initC)
        inc = np.sign(initLz)*np.arccos(np.sqrt(z2))
        tracker = [[pot_min, e, inc, inner_turn, outer_turn, all_states[0][0], 0]]

    #basically change how I handle all of this??
    turns, flats, zs = mm.root_getter(initE, initLz, initC, a)
    pot_min = flats[-1]
    inner_turn, outer_turn = turns[-2:]
    e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
    inc = np.sign(initLz)*np.arccos(min(1.0, np.mean(np.abs(zs[1:3]))))
    
    j = 0
    constants = []
    tracker = []
    qarter = []
    constants.append([initE,      #energy   
                      initLz,      #angular momentum (axial)
                      initC])
    #gorfed
    tracker.append([pot_min, e, inc, inner_turn, outer_turn, all_states[0][0], 0])
    qarter.append(initQ)
    
    
    #false_constants = [np.array([getEnergy(all_states[0], a), *getLs(all_states[0], mu)])]  #Cartesian approximation of L vector
    
    #freqs = [mm.freqs_finder(initE, initLz, initC, a)]

    cond = ['((S2-compl) > 0 and (compl-S1) > 0)',                                         #outgoing r0
            '((S2-compl) > 0 and (compl-S1) > 0) or ((S2-comph) > 0 and (comph-S1) > 0)',  #both r0s
            '((S2-comph) > 0 and (comph-S1) > 0)',                                         #ingoing r0
            '(S1 > np.pi and S2 < np.pi)',                                                 #at r_min
            '(S1 < np.pi and S2 > np.pi)',                                                 #at r_max
            '(S1 > np.pi and S2 < np.pi) or (S1 < np.pi and S2 > np.pi)',                  #at extrema
            '((S2-np.pi/2) > 0 and (np.pi/2-S1) > 0)',                                     #outgoing p
            '((S2-np.pi/2) > 0 and (np.pi/2-S1) > 0) or ((S2-1.5*np.pi) > 0 and (1.5*np.pi-S1) > 0)',  #both ps
            '((S2-1.5*np.pi) > 0 and (1.5*np.pi-S1) > 0)',                                 #ingoing p
            '((S2-comph) > 0 and (comph-S1) > 0) and (new_step[3] - all_states[int(tracker[j][-1])][3] >= 6*np.pi)',  #ingoing r0 + 3 phi orbits (deprecated?)
            'new_step[2] < np.pi/2 and all_states[-1][2] > np.pi/2',                       #ascending node
            '(new_step[3] + np.pi)%(2*np.pi) >= np.pi and (all_states[-1][3] + np.pi)%(2*np.pi) < np.pi']        #passing phi=0

    
    compErr = 0
    milestone = 0
    issues = [(None, None)]
    orbitside = np.sign(all_states[0][1] - pot_min)
    if orbitside == 0:
        orbitside = -1
    
    orbCount = 0
    val = get_true_anom(all_states[0], 0.5*(outer_turn + inner_turn), e)
    P0, ECC = 2*(inner_turn*outer_turn)/(inner_turn + outer_turn), (outer_turn - inner_turn)/(outer_turn + inner_turn)
    POT_MIN = max(flats)
    true_anom = [val if np.isnan(val) == False else 0.0]
    stop = False
    
    if label == "default":
        label = "r" + str(pot_min) + "e" + str(e) + "zU+03C0" + str(inc/np.pi) + "mu" + str(mu) + "a" + str(a)
    
    #Main Loop
    dTau = 0.1*np.abs(np.real((inner_turn/200)**(2)))
    if "dumb" in label:
        dTau *= -1
    dTau_change = [0]                                                #create dTau tracker
    borken = 0
    #if terms[0] != "custom":
    #    initflagval = eval(termdict[terms[0]])
    plunge, unbind = False, False
    def anglething(angle):
        return 0.5*np.pi - np.abs(angle%np.pi - np.pi/2)

    if verbose > 0 and verbose < 3:
        pbar = tqdm(total = 10000000, position=0)
        pbar.set_postfix_str("Semilat: %s, Ecc %s, Peri: %s, Theta_min: %s*pi" %(np.round( 0.5*(tracker[0][3] + tracker[0][4])*(1 - tracker[0][1]**2), 3), np.round(tracker[0][1], 3), np.round(tracker[0][3], 3), np.round(tracker[0][2]/np.pi, 3)))
    progress = 0
    rkfull = []
    confull = []
    upfull = []
    skip_count = 0
    gorf = 0
    getval, op, threshold = parse_endflag(endflag)
    ctx = EndContext(i, all_states, true_anom, tracker, j)
    ctx.last_chunk = np.asarray(all_states)
    #min(np.array(ctx.all_states)[ctx.tracker[ctx.j][-2]:c.i, 1])
    start_val, ctx_val = getval(ctx), getval(ctx)
    #while (not(eval(newflag)) and (i < 10**7 or override)):
    while not(ctx_val and op(ctx_val, threshold)) and (i < 10**7 or override):
        try:
            update = False
            condate = False
            first = True
          
            #Grab the current state
            state = all_states[i]  
            pot_min = tracker[j][0]   
          
            #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
            if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001):
                plunge = True
                break
            
            if (state[1] >= (1 + np.sqrt(1 - a**2))*1e15):
                unbind = True
                break

            #break if something stops making sense
            if (np.nan in state or constants[j][0] < 0) or (np.isnan(state[0])):
                print("HEWWO", np.nan in state, constants[j][0] < 0, np.isnan(state[0]))
                plunge = True
                unbind = True
                break

            #Runge-Kutta update using geodesic
            old_dTau = dTau
            skip = False
            rkcount = time.perf_counter()
            counter = 0
            while ((err_calc >= err_target) or (first == True)) and (skip == False):
                counter += 1
                if counter%20 == 0:
                    print(f"A lot! {counter} {err_calc}")
                if counter > 30:
                    raise KeyboardInterrupt
                # Generate 4th and 5th order calculations for next step
                if "fool1" in label:
                    step_check = mm.gen_RK2(*mm.ck4_2, mm.kerr_2, state, dTau, a)
                    new_step = mm.gen_RK2(*mm.ck5_2, mm.kerr_2, state, dTau, a)
                elif "fool2" in label:
                    step_check = mm.gen_RK2(*mm.rkf4_2, mm.kerr_2, state, dTau, a)
                    new_step = mm.gen_RK2(*mm.rkf5_2, mm.kerr_2, state, dTau, a)
                else:
                    new_step = mm.gen_RK2(*mm.ck4_2, mm.kerr_2, state, dTau, a)
                    step_check = mm.gen_RK2(*mm.ck5_2, mm.kerr_2, state, dTau, a) 
                if "ridic" in label:
                    new_step = mm.recalc_state(constants[j], new_step, a)
                    step_check = mm.recalc_state(constants[j], step_check, a)
                # Calculate the error
                if weird == False:
                    #delt = new_step[1:3] - step_check[1:3]
                    #delt = np.concatenate((delt, new_step[4:] - step_check[4:]))
                    #mod_r = np.array([*new_step[1:3], *new_step[4:]])
                    #err_calc = np.sqrt(np.dot(delt, delt)/np.dot(mod_r, mod_r))
                    ###
                    err_calc = new_step[1:3] - step_check[1:3]
                    err_calc = np.concatenate((err_calc, new_step[4:] - step_check[4:]))
                    mod_r = np.array([*new_step[1:3], *new_step[4:]])
                    err_calc = np.sqrt(np.dot(err_calc, err_calc)/np.dot(mod_r, mod_r))

                else:
                    new_cons = getCons_3(new_step, a)
                    check_cons = getCons_3(step_check, a)
                    if weird == 10:
                        new_step = np.array(new_step)
                        step_check = np.array(step_check)
                        state = np.array(state)

                        # Difference between RK4 and RK5
                        delt = new_step - step_check

                        # Actual motion from previous accepted state
                        motion = new_step - state

                        # --- Physics-aware scaling ---
                        # Characteristic orbital scales
                        r_orbit = max(1e-12, state[1])  # radial scale
                        theta_scale = max(1e-12, 0.1)   # angles typically ~ radians, small floor
                        phi_scale = max(1e-12, 0.1)
                        
                        # Typical velocity scale ~ sqrt(M/r) in geometric units, or just norm of spatial velocities
                        vel_scale = max(1e-12, np.linalg.norm(state[4:]))  

                        # Time scaling: t changes can be huge, so scale with radial motion
                        t_scale = max(1e-12, r_orbit)

                        # Floor to prevent division by near-zero
                        floor = np.array([t_scale, r_orbit, theta_scale, phi_scale, vel_scale, vel_scale, vel_scale, vel_scale])

                        # Safe motion
                        motion_safe = np.where(np.abs(motion) < floor, floor, motion)

                        # Fractional error per component
                        frac_err = delt / motion_safe

                        # Weighted mean: we can give more weight to velocities if you want energy/momentum preservation
                        # For example: 50% coords, 50% velocities
                        weights = np.array([1,1,1,1,2,2,2,2])  # simple weighting, adjust as desired
                        err_calc = np.sqrt(np.sum(weights * frac_err**2) / np.sum(weights))
                    if weird == 1:
                        new_cons = np.array([new_cons[0], np.sqrt(new_cons[1]**2 + abs(new_cons[2])), np.pi + np.arccos(new_cons[1]/np.sqrt(new_cons[1]**2 + abs(new_cons[2])))])
                        check_cons = np.array([check_cons[0], np.sqrt(check_cons[1]**2 + abs(check_cons[2])), np.pi + np.arccos(check_cons[1]/np.sqrt(check_cons[1]**2 + abs(check_cons[2])))])
                        err_calc = np.linalg.norm((new_cons - check_cons)/new_cons)
                    elif weird == 2:
                        new_cons = np.array([(new_cons[1]**2 + abs(new_cons[2]))*new_cons[0], new_cons[1], np.sign(new_cons[2])*np.sqrt(np.abs(new_cons[2]))])
                        check_cons = np.array([(check_cons[1]**2 + abs(check_cons[2]))*check_cons[0], check_cons[1], np.sign(check_cons[2])*np.sqrt(np.abs(check_cons[2]))])
                        err_calc = np.linalg.norm(new_cons - check_cons)/np.linalg.norm(new_cons)
                    elif weird == 3:
                        new_cons = np.array([(new_cons[1]**2 + abs(new_cons[2]))*new_cons[0], new_cons[1], np.sign(new_cons[2])*np.sqrt(np.abs(new_cons[2])), new_cons[3]])
                        check_cons = np.array([(check_cons[1]**2 + abs(check_cons[2]))*check_cons[0], check_cons[1], np.sign(check_cons[2])*np.sqrt(np.abs(check_cons[2])), check_cons[3]])
                        err_calc = np.linalg.norm(new_cons - check_cons)/np.linalg.norm(new_cons)
                    elif weird == 4:
                        new_cons = np.array([new_cons[0], np.sqrt(new_cons[1]**2 + abs(new_cons[2])), np.pi + np.arccos(new_cons[1]/np.sqrt(new_cons[1]**2 + abs(new_cons[2]))), 100*new_cons[3]])
                        check_cons = np.array([check_cons[0], np.sqrt(check_cons[1]**2 + abs(check_cons[2])), np.pi + np.arccos(check_cons[1]/np.sqrt(check_cons[1]**2 + abs(check_cons[2]))), 100*check_cons[3]])
                        err_calc = np.linalg.norm((new_cons - check_cons)/new_cons)
                    elif weird == 5:
                        new_cons = np.array([new_cons[0], np.sqrt(new_cons[1]**2 + new_cons[2]), new_cons[2] + (a*new_cons[0] - new_cons[1])**2, new_cons[3]])
                        check_cons = np.array([check_cons[0], np.sqrt(check_cons[1]**2 + check_cons[2]), check_cons[2] + (a*check_cons[0] - check_cons[1])**2, check_cons[3]])
                        #new_cons[2] += (a*new_cons[0] - new_cons[1])**2
                        #check_cons[2] += (a*check_cons[0] - check_cons[1])**2
                        err_calc = np.nanmax(np.abs(((new_cons - check_cons)/new_cons)))
                        #print(np.abs(((new_cons - check_cons)/new_cons)), op.get_index(np.abs(((new_cons - check_cons)/new_cons)), err_calc), err_calc<= err_target)
                    elif weird == 6:
                        metric, _ = mm.kerr_2(state, a)
                        new_metric, _ = mm.kerr_2(new_step, a)
                        diff_rk, diff = new_step - step_check, new_step - state
                        ds2_rk, ds2 = np.matmul(np.matmul(new_metric, diff_rk[:4]), diff_rk[:4]), np.matmul(np.matmul(metric, diff[:4]), diff[:4])
                        vds2_rk, vds2 = np.matmul(np.matmul(new_metric, dTau*diff_rk[4:]), dTau*diff_rk[4:]), np.matmul(np.matmul(metric, dTau*diff[4:]), dTau*diff[4:])
                        #print(ds2_rk, vds2_rk, ds2, vds2)
                        ##print(np.sqrt(abs(ds2_rk) + abs(vds2_rk)), abs(ds2_rk), abs(vds2_rk), dTau, state[1], state[2]/np.pi)
                        ##print("----", diff_rk[:4])
                        #print("wonk", np.sqrt(abs(ds2_rk) + abs(vds2_rk))/np.sqrt(abs(ds2) + abs(vds2)), np.sqrt(ds2_rk + vds2_rk)/np.sqrt(ds2 + vds2), dTau)
                        #print("work", np.sqrt(abs(ds2_rk) + abs(vds2_rk)))#/np.sqrt(abs(ds2) + abs(vds2)))#, np.sqrt(ds2_rk + vds2_rk)/np.sqrt(ds2 + vds2), dTau)
                        err_calc = np.sqrt(abs(ds2_rk) + abs(vds2_rk))#/np.sqrt(abs(ds2) + abs(vds2))
                    elif weird == 7:
                        new_metric, _ = mm.kerr_2(new_step, a)
                        diff_rk = new_step - step_check
                        ds2_rk, vds2_rk = np.matmul(np.matmul(new_metric, diff_rk[:4]), diff_rk[:4]), np.matmul(np.matmul(metric, dTau*diff_rk[4:]), dTau*diff_rk[4:])
                        #print(ds2_rk, vds2_rk, ds2, vds2)
                        #print(abs(vds2_rk)/abs(ds2_rk), abs(vds2_rk), abs(ds2_rk), dTau, state[1])
                        #print("----", diff_rk)
                        #print("wonk", np.sqrt(abs(ds2_rk) + abs(vds2_rk))/np.sqrt(abs(ds2) + abs(vds2)), np.sqrt(ds2_rk + vds2_rk)/np.sqrt(ds2 + vds2), dTau)
                        #print("work", np.sqrt(abs(ds2_rk) + abs(vds2_rk)))#/np.sqrt(abs(ds2) + abs(vds2)))#, np.sqrt(ds2_rk + vds2_rk)/np.sqrt(ds2 + vds2), dTau)
                        err_calc = (abs(vds2_rk)/abs(vds2_rk))#/np.sqrt(abs(ds2) + abs(vds2))
                    elif weird == 8:
                        mod_new = np.array([new_step[0] - state[0], *new_step[1:3], new_step[3] - state[3], *new_step[4:]])
                        mod_check = np.array([step_check[0] - state[0], *step_check[1:3], step_check[3] - state[3], *step_check[4:]])
                        delt = mod_new - mod_check
                        if i == 100000:
                            print(np.sqrt(np.dot(delt[[1,2,4,5,6,7]], delt[[1,2,4,5,6,7]])/np.dot(mod_new[[1,2,4,5,6,7]], mod_new[[1,2,4,5,6,7]])))
                            print(np.sqrt(np.dot(delt, delt)/np.dot(mod_new, mod_new)))
                        err_calc = np.sqrt(np.dot(delt, delt)/np.dot(mod_new, mod_new))
                    elif weird == 9:
                        mod_new = np.array([new_step[0] - state[0], *new_step[1:3], new_step[3] - state[3], *new_step[4:]*dTau])
                        mod_check = np.array([step_check[0] - state[0], *step_check[1:3], step_check[3] - state[3], *step_check[4:]*dTau])
                        delt = mod_new - mod_check
                        if i == 100000:
                            print(np.sqrt(np.dot(delt[[1,2,4,5,6,7]], delt[[1,2,4,5,6,7]])/np.dot(mod_new[[1,2,4,5,6,7]], mod_new[[1,2,4,5,6,7]])))
                            print(np.sqrt(np.dot(delt, delt)/np.dot(mod_new, mod_new)))
                        err_calc = np.sqrt(np.dot(delt, delt)/np.dot(mod_new, mod_new))

                # Correct for pole effects
                E, L, C = constants[j]
                '''
                Old version, busted apparently?
                # if ((within ~0.5 degrees of pole) and (moving closer to pole)) and (average of last few dTau_change values is much smaller than the average):
                if ((np.sin(new_step[2])**2 <= 8e-5) and np.sign(new_step[6]*np.cos(new_step[2])) < 0) and (np.mean(np.diff(dTau_change[-10:])) <= 0.001*np.mean(np.diff(dTau_change))):
                    old_step = new_step[0]
                    new_step[0] += ((new_step[0] - state[0])/abs(new_step[2] - state[2]))*(2*anglething(new_step[2]))
                    new_step[3] += 2*np.arccos(np.sin(abs(np.pi/2 - np.arccos(L/np.sqrt(L**2 + C))))/ np.sin(new_step[2]))
                    new_step[6] = -new_step[6]
                    dTau = dTau*abs((new_step[0] - state[0])/(old_step - state[0]))
                '''
                # if ((within ~0.5 degrees of pole) and (moving closer to pole)) and (average of last few dTau_change values is much smaller than the average):
                if ((np.sin(new_step[2])**2 <= 8e-5) and np.sign(new_step[6]*np.cos(new_step[2])) < 0) and (np.mean(np.diff(dTau_change[-10:])) <= 0.001*np.mean(np.diff(dTau_change))):
                    old_step = new_step
                    # inc is the minimum value of theta, pretend the particle is traveling on a straight line between
                    # current theta value and inc, then flash to the other side (same theta). Find phi!
                    k = abs(inc)/new_step[2] if new_step[2] < np.pi/2 else abs(inc)/(np.pi - new_step[2])
                    if k == 0.0:
                        # Orbit is polar!
                        phi_dist = np.pi/2
                    else:
                        cosp, sinp = np.cos(new_step[3]), np.sin(new_step[3])
                        m = (cosp*sinp + np.array([1, -1])*k*np.sqrt(1 - k*k))/(cosp**2 - k*k)
                        H = (sinp - m*cosp)/k
                        phi_min = (np.arctan(m) + np.arcsin(np.clip(H/np.sqrt(1 + m*m), -1, 1)))%(2*np.pi)
                        # This gives us 2 values of phi_min. We need the one that corresponds to the correct direction of motion
                        arr = (phi_min - new_step[3]%(2*np.pi))/new_step[7]
                        phi_min_real = phi_min[0] if arr[0] > 0 else phi_min[1]
                        # Now get the distance from the current phi, add double that to get the NEW phi
                        phi_dist = 2*min(abs(phi_min - old_step[3]%(2*np.pi)))*np.sign(old_step[7])
                    new_step[3] += phi_dist
                    # Estimate the elapsed time as d(phi)/phi_dot
                    t_change = phi_dist/new_step[7]
                    if np.isnan(t_change):
                        # Same polar issue
                        t_change = 2*old_step[2]/old_step[6]
                    new_step[0] += t_change
                    # Scale up dTau so we aren't liars
                    dTau = dTau*(new_step[0] - state[0])/(old_step[0] - state[0])
                    # Flip the sign on theta_dot so now it's moving away from the pole
                    new_step[6] *= -1
                    #THIS IS STILL FUCKED TEST THIS
        
                # Get new dTau value 
                old_dTau, dTau = dTau, np.sign(dTau)*min(abs(dTau) * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
                if "stupid" in label and (np.mean(dTau_change[-10:]) <= 0.001*np.mean(dTau_change)):
                    dTau *= 10
                if ((-1)**("dumb" in label))*dTau <= 0.0:
                    err_calc = 1
                    print("a")
                    dTau = old_dTau
                if ((-1)**("dumb" in label))*(new_step[0] - state[0]) < 0:
                    err_calc = 1
                    dTau = 10*abs(old_dTau)
                    print("b")
                if new_step[0] - state[0] > 100 and new_step[0] - state[0] < 100:
                    print("what the hell??")
                    print(dTau)
                    print(state, mm.check_interval(mm.kerr, state, a))
                    print(new_step, mm.check_interval(mm.kerr, new_step, a))
                    print(step_check, mm.check_interval(mm.kerr, step_check, a))
                    print(old_dTau, dTau)
                    print(err_calc)
                    oof = input("Type x to try unfucking this: ")
                    if "x" in oof:
                        err_calc = 1
                        print("c")
                if np.isnan(np.sum(new_step)):
                    print("d", new_step, step_check, err_calc, dTau, state, old_dTau)
                    err_calc = 1
                    dTau = old_dTau*0.9
                if np.isnan(dTau):
                    dTau = 0.1*np.abs(np.real((inner_turn/200)**(2)))
                    
                first = False
            
            metric = mm.kerr(new_step, a)[0]
            test = mm.check_interval_w_metric(metric, new_step, a)
            looper = 0
            while ((abs(test+1)>(err_target) or new_step[4] < 0.0) and looper < 10) and nofix:
                borken = borken + 1
                og_new_step = np.copy(new_step)
                if bonk2 == True:
                    gtt, gtp = metric[0,0], metric[0,3]
                    disc = 4*(gtp*new_step[4]*new_step[7])**2 - 4*gtt*(new_step[4]**2)*(np.einsum('ij, i, j ->', metric[1:,1:], new_step[5:], new_step[5:]) + 1)
                    delt = (-2*gtp*new_step[4]*new_step[7] - np.sqrt(disc))/(2*gtt*new_step[4]*new_step[4])
                    new_step[4] *= delt
                else:
                    new_step = mm.recalc_state(constants[j], new_step, a)
                test = mm.check_interval_w_metric(metric, new_step, a)
                looper += 1
            if (test+1) > err_target or new_step[4] < 0.0:
                new_step = np.copy(og_new_step)
            if looper > 0:
                issues.append((i, new_step[0]))

            if "stupid" in label and i%5==0:
                E, L, C = constants[j]
                LC = np.sqrt(L**2 + C)
                ang = np.cosh(np.arccos(np.sqrt(C)/LC))
                E1, L1, C1 = getCons_2(new_step, a)
                LC1 = np.sqrt(L1**2 + C1)
                ang1 = np.cosh(np.arccos(np.sqrt(C1)/LC1))
                thing = np.array([(E - E1)/E, (LC - LC1)/LC, (ang - ang1)/ang])
                thing1 = np.abs(thing[~np.isnan(thing)])
                #print(thing)
                if len(np.where(thing1 > err_target)[0]) > 0:
                    #print(thing)
                    new_step = mm.recalc_state(constants[j], new_step, a)
                    gorf += 1
            rkfull.append(time.perf_counter() - rkcount)

            #constant modifying section
            
            compl, comph = np.arccos(-ECC), 2*np.pi - np.arccos(-ECC)
            #S1, S2 = get_true_anom(state, R0, ECC), get_true_anom(new_step, R0, ECC)
            # if (chosen trigger evals true) AND (it has been it least 11 steps since the last constant modification)
            #if ((S2-comph) > 0 and (comph-S1) > 0) and i - int(tracker[j][-1]) > 10:
            if (state[1] > POT_MIN and new_step[1] <= POT_MIN) and i - int(tracker[j][-1]) > 10:
                if skip_count >= skip_tar:  #Allows you to average over some integer number of orbit
                    skip_count = 0
                    update = True
                    concount = time.perf_counter()
                    if ( np.sign(new_step[1] - pot_min) != orbitside):
                        orbitside *= -1
                    if mu != 0.0:
                        if force_stop is not None:
                            if force_stop():
                                raise KeyboardInterrupt
                        condate = True
                        if "differy" in label:
                            new_step_hold, ch_cons = mm.peters_integrate_differential(all_states[int(tracker[j][-1]):i], a, mu,
                                                                                   constants[j], new_step, ctx.j, i)
                        elif "differo" in label:
                            new_step_hold, ch_cons = mm.peters_integrate_differential2(all_states[int(tracker[j][-1]):i], a, mu,
                                                                                   constants[j], new_step, ctx.j, i)
                        elif "differa" in label:
                            new_step_hold, ch_cons = mm.peters_integrate_differential3(all_states[int(tracker[j][-1]):i], a, mu,
                                                                                   constants[j], new_step, ctx.j, i)
                        elif "differe" in label:
                            new_step_hold, ch_cons = mm.peters_integrate_differential4(all_states[int(tracker[j][-1]):i], a, mu,
                                                                                   constants[j], new_step, ctx.j, i)
                        else:
                            if "mark1" in label:
                                dcons = mm.peters_integrate6_6_4_2(all_states[int(tracker[j][-1]):i], a, mu, ctx.j, i)
                            elif "mark2" in label:
                                dcons = mm.peters_integrate6_6_4_3(all_states[int(tracker[j][-1]):i], a, mu, ctx.j, i)
                            else:
                                dcons = mm.peters_integrate6_6_4(all_states[int(tracker[j][-1]):i], a, mu, ctx.j, i)
                            if "dumb" in label:
                                dcons *= -1
                            if "hoopa" in label:
                                new_step_hold, ch_cons = mm.new_recalc_state8(constants[j], dcons, new_step, a)
                            elif "chatty" in label:
                                new_step_hold, ch_cons = mm.new_recalc_state9k(constants[j], dcons, new_step, a)
                            elif "hooper" in label:
                                new_step_hold, ch_cons = mm.new_recalc_state8b(constants[j], dcons, new_step, a)
                            elif "hoopla" in label:
                                new_step_hold, ch_cons = mm.new_recalc_state8c(constants[j], dcons, new_step, a)
                            elif "power" in label:
                                new_step_hold, ch_cons = mm.new_recalc_state9j(constants[j], dcons, new_step, a)
                            elif "zampow" in label:
                                new_step_hold, ch_cons = mm.new_recalc_state9l(constants[j], dcons, new_step, a)
                            elif "sketchy" in label:
                                new_step_hold, ch_cons = mm.new_recalc_state9n(constants[j], dcons, new_step, a)
                            else:
                                new_step_hold, ch_cons = mm.new_recalc_state9m(constants[j], dcons, new_step, a, ECC)
                        pot_min = viable_cons(ch_cons, constants[j], new_step, a)
                        subcount = 0
                        if pot_min < -err_target:
                            if "woosh" in label:
                                update = False
                                condate = False
                                #print("try just going for another loop?", i)
                            else:
                                viable_cons(ch_cons, constants[j], new_step, a, True)
                                print("BADVALS", *ch_cons)
                                raise KeyError
                        else:
                            new_step = new_step_hold
                        if subcount > 0:
                            print(subcount, "oof", pot_min)
                    confull.append(time.perf_counter() - concount)
                else:
                    skip_count += 1

            #Initializing for the next step
            #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
            #Only happens the step before the derivatives are recalculated.
            
            upcount = time.perf_counter()
            #Update stuff!
            if (update == True):
                if condate == False:
                    #metric = mm.kerr(new_step, a)[0]
                    newE, newLz, newC = getCons_2(state, a)
                    newQ = newC + (a*newE - newLz)**2  
                    turns, flats, zs = mm.root_getter(newE, newLz, newC, a)
                    pot_min = flats[-1]
                    inner_turn, outer_turn = turns[-2:]
                    e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
                    inc = np.sign(newLz)*np.arccos(min(1.0, np.mean(np.abs(zs[1:3]))))
                    j += 1
                    constants.append([newE, newLz, newC])
                    tracker.append([pot_min, e, inc, inner_turn, outer_turn, new_step[0], i])
                    qarter.append(newQ)
                    #freqs.append(mm.freqs_finder(newE, newLz, newC, a))
                    P0, ECC = 2*(inner_turn*outer_turn)/(inner_turn + outer_turn), (outer_turn - inner_turn)/(outer_turn + inner_turn)
                    POT_MIN = max(flats)
                else:
                    turns, flats, zs = mm.root_getter(*ch_cons, a)
                    pot_min = flats[-1]
                    inner_turn, outer_turn = turns[-2:]
                    e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
                    inc = np.sign(ch_cons[1])*np.arccos(min(1.0, np.mean(np.abs(zs[1:3]))))
                    #freqs.append(mm.freqs_finder(*ch_cons, a))
                    j += 1
                    constants.append(ch_cons)
                    tracker.append([pot_min, e, inc, inner_turn, outer_turn, new_step[0], i])
                    qarter.append(ch_cons[2] + (a*ch_cons[0] - ch_cons[1])**2)
                    P0, ECC = 2*(inner_turn*outer_turn)/(inner_turn + outer_turn), (outer_turn - inner_turn)/(outer_turn + inner_turn)
                    POT_MIN = max(flats)
                    if verbose > 0 and verbose < 3:
                        pbar.set_postfix_str("Semilat: %s, Ecc %s, Peri: %s, Theta_min: %s*pi" %(np.round( 0.5*(tracker[j][3] + tracker[j][4])*(1 - tracker[j][1]**2), 3), np.round(tracker[j][1], 3), np.round(tracker[j][3], 3), np.round(tracker[j][2]/np.pi, 3)))
                ctx.j = j
                ctx.tracker = tracker
                ctx.last_chunk = np.asarray(all_states[tracker[j-1][-1]:])
                if True in np.iscomplex(tracker[j]):
                    compErr += 1
                    issues.append((i, new_step[0]))  
            #print("not stuck!")
            #interval.append(mm.check_interval_w_metric(metric, new_step, a))
            #false_constants.append([getEnergy(new_step, a), *getLs(new_step, mu)])
            dTau_change.append(dTau_change[-1] + old_dTau)
            all_states.append(new_step )    #update position and velocity
            anomval = get_true_anom(new_step, 0.5*(outer_turn + inner_turn), e) + orbCount*2*np.pi
            if anomval - true_anom[-1] < -np.pi:
                anomval += 2*np.pi
                orbCount += 1
            true_anom.append(anomval)
            i += 1
            #last_chunk = np.array(all_states[tracker[j-1][-1]:])
            #r, r_dot = last_chunk[:,1], last_chunk[:,5]
            #anomval = get_true_anom(new_step, 0.5*(outer_turn + inner_turn), e, orbCount)
            #true_anom.extend(anomvals)
            ctx.i = i
            ctx.all_states = all_states
            ctx.true_anom = true_anom
            ctx_val = getval(ctx)
            if verbose == 3:
                #if terms[0] != "custom":
                #progress = max( abs((eval(termdict[terms[0]]) - initflagval)/(eval(terms[2]) - initflagval)), i/(10**7)) * 100
                progress = max(abs((ctx_val - start_val) / (threshold - start_val)), i/1e7)
                if (progress >= milestone):
                    print(f"Program has completed {ctx_val}, {progress*100}% of maximum run: Index = {i}")
                    milestone = int(progress) + 1
            elif verbose > 0 and verbose < 3:
                #if terms[0] != "custom":
                if update:
                    #print(abs((ctx_val - start_val) / (threshold - start_val)), progress)
                    #print((np.exp(-abs(ctx_val - threshold)) - np.exp(-abs(start_val - threshold))), "new?")
                    val = max(abs((ctx_val - start_val) / (threshold - start_val)) - progress, i/1e7 - progress, 0)
                    #print(ctx_val, start_val, threshold, start_val, progress, "stuff")
                    #print(val)
                    pbar.update(val*1e7)
                    #progress = max( (10**7)*abs((eval(termdict[terms[0]]) - initflagval)/(eval(terms[2]) - initflagval)), i)
                    progress = max(abs((ctx_val - start_val) / (threshold - start_val)), i/1e7)
            #print("maybe even finished?")
            upfull.append(time.perf_counter() - upcount)
            

        #Lets you end the program before the established end without breaking anything
        except KeyboardInterrupt:
            print("\nEnding program", flush=True)
            stop = True
            cap = len(all_states) - 1
            all_states = all_states[:cap]
            #interval = interval[:cap]
            dTau_change = dTau_change[:cap]
            constants = constants[:cap]
            qarter = qarter[:cap]
            true_anom = true_anom[:cap]
            #freqs = freqs[:cap]
            break
        except KeyError:
            print("\nEnding program", flush=True)
            stop = True
            cap = len(all_states) - 1
            all_states = all_states[:cap]
            #interval = interval[:cap]
            dTau_change = dTau_change[:cap]
            constants = constants[:cap]
            qarter = qarter[:cap]
            true_anom = true_anom[:cap]
            #freqs = freqs[:cap]
            break
        
        except Exception as e:
            print("\nEnding program - ERROR", flush=True)
            print(type(e), e)
            stop = True
            cap = len(all_states) - 1
            all_states = all_states[:cap]
        #    interval = interval[:cap]
            dTau_change = dTau_change[:cap]
            constants = constants[:cap]
            qarter = qarter[:cap]
        #    freqs = freqs[:cap]
            break
    if verbose > 0 and verbose < 3:
        pbar.close()
    #print(len(issues), len(all_states))
    #unit conversion stuff
    if units == "mks":
        G, c = 6.67e-11, 3e8
    elif units == "cgs":
        G, c = 6.67e-8,  3e10
    else:
        G, mass, c = 1.0, 1.0, 1.0
        
    if mu == 0.0:
        #so it gives actual numbers for pure geodesics
        mu = 1.0
    
    if j != -1:
        constants = constants[:j+1]
        tracker = tracker[:j+1]
        qarter = qarter[:j+1]
    constants = np.array(constants)
    constants[:,0] *= mass*(c**2)
    constants[:,1] *= mass*mass*G/c
    constants[:,2] *= (mass*mass*G/c)**2
    #constants = np.array([entry*np.array([mass*(c**2), mass*mass*G/c, (mass*mass*G/c)**2]) for entry in np.array(constants)], dtype=np.float64)
    #false_constants = np.array(false_constants)
    qarter = np.array(qarter)
    #freqs = np.array(freqs)*(c**3)/(G*mass)
    #interval = np.array(interval)
    dTau_change = np.array(dTau_change) * (G*mass)/(c**3)
    #           np.array([entry * (G*mass)/(c**3) for entry in dTau_change])
    all_states = np.array(all_states)
    all_states[:,0] *= (G*mass)/(c**3) 
    all_states[:,1] *= (G*mass)/(c**2)
    all_states[:,5] *= c
    all_states[:,6] *= (c**3)/(G*mass)
    all_states[:,7] *= (c**3)/(G*mass)
    #all_states = np.array([entry*np.array([(G*mass)/(c**3), (G*mass)/(c**2), 1.0, 1.0, 1.0, c, (c**3)/(G*mass), (c**3)/(G*mass)]) for entry in np.array(all_states)])
    tracker = np.array(tracker)
    tracker[:,0] *= (G*mass)/(c**2)
    tracker[:,3] *= (G*mass)/(c**2)
    tracker[:,4] *= (G*mass)/(c**2)
    tracker[:,5] *= (G*mass)/(c**3)
    #tracker = np.array([entry*np.array([(G*mass)/(c**2), 1.0, 1.0, (G*mass)/(c**2), (G*mass)/(c**2), (G*mass)/(c**3), 1]) for entry in tracker])
    
    ind = argrelmin(all_states[:,1])[0]
    omega, otime = all_states[ind,3] - 2*np.pi*np.arange(len(ind)), all_states[ind,0]
    asc_node, asc_node_time = np.array([]), np.array([])
    des_node, des_node_time = np.array([]), np.array([])
    interval = mm.check_interval_vec(all_states, a)
    true_anom = np.array(true_anom)
    if max(all_states[:,2]) - min(all_states[:,2]) > 1e-15:
        theta_derv = np.interp(all_states[:,0], 0.5*(all_states[:,0][:-1] + all_states[:,0][1:]), np.diff(all_states[:,2])/np.diff(all_states[:,0]))
        ind2 = argrelmin(theta_derv)[0] #indices for the ascending node
        ind3 = argrelmin(-theta_derv)[0] #indices for the descending node
        asc_node, asc_node_time = all_states[ind2,3] - 2*np.pi*np.arange(len(ind2)), all_states[ind2,0] #subtract the normal phi advancement
        des_node, des_node_time = all_states[ind3,3] - 2*np.pi*np.arange(len(ind3)), all_states[ind3,0] #subtract the normal phi advancement
        if asc_node.size == 0:
            asc_node = np.array([])
            asc_node_time = np.array([])
    if verbose == 3:
        print("There were " + str(compErr) + " issues with complex roots/turning points.")
    if verbose >= 2:
        try:
            print("rkstats: min %s, max %s, mean %s, stdev%s, median %s, mode %s, total %s"%(min(rkfull), max(rkfull), np.mean(rkfull), np.std(rkfull), np.median(rkfull), st.mode(rkfull), np.sum(rkfull)))
            print("constats: min %s, max %s, mean %s, stdev%s, median %s, mode %s, total %s"%(min(confull), max(confull), np.mean(confull), np.std(confull), np.median(confull), st.mode(confull), np.sum(confull)))
            print("upstats: min %s, max %s, mean %s, stdev%s, median %s, mode %s, total %s"%(min(upfull), max(upfull), np.mean(upfull), np.std(upfull), np.median(upfull), st.mode(upfull), np.sum(upfull)))
        except:
            print("stats borken")
    #print(f"Steps:{len(all_states)}, Corrections:{gorf}")
    final = {"name": label,
             "raw": all_states,
             "inputs": inputs,
             "pos": all_states[:,1:4],
             "all_vel": all_states[:,4:], 
             "time": all_states[:,0],
             "true_anom": true_anom,
             "interval": interval,
             "vel": (np.square(all_states[:,5]) + np.square(all_states[:,1]) * (np.square(all_states[:,6]) + (np.sin(all_states[:,2])**2)*np.square(all_states[:,7])))**(0.5),
             "dTau_change": dTau_change,
             "energy": constants[:, 0],
             "phi_momentum": constants[:, 1],
             "carter": constants[:, 2],
             "qarter":qarter,
             #"energy2": false_constants[:, 0],
             #"Lx_momentum": false_constants[:, 1],
             #"Ly_momentum": false_constants[:, 2],
             #"Lz_momentum": false_constants[:, 3],
             "spin": a,
             #"freqs": freqs,
             "pot_min": tracker[:,0],
             "e": tracker[:,1],
             "inc": tracker[:,2],
             "it": tracker[:,3],
             "ot": tracker[:,4],
             "r0": 0.5*(tracker[:,3] + tracker[:,4]),
             "p": 0.5*(tracker[:,3] + tracker[:,4])*(1 - tracker[:,1]**2),
             "tracktime": tracker[:,5],
             "trackix": np.array([int(np.real(num)) for num in tracker[:,6]]),
             "omega": omega,
             "otime": otime,
             "asc_node": asc_node,
             "asc_node_time": asc_node_time,
             "des_node": des_node,
             "des_node_time": des_node_time,
             "stop": stop,
             "plunge": plunge,
             "unbind": unbind,
             "issues": issues}
    return final

def encode_filename():
    now = time.time()
    return hex(int(40587.0 + now//86400.0))[2:] + "_" + hex(int((now/86400.0 - now//86400.0)*60*24))[2:]

def decode_filename(name):
    splits = os.path.splitext(name)[0].split("_")
    mjd = int(splits[1], 16) + int(splits[2], 16)/(60*24)
    unix = (mjd - 40587)*86400
    return unix

def update_index(filename, metadata, index_path="D:/EMRIData/saved_sims/index.json"):
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
    else:
        new_path = input(f"{index_path} is not a valid index path.\nWould you like to create a new index at this location? (y/n): ").lower()
        if "y" in new_path:
            print("Creating new index.")
            index = {}
        else:
            print("Index Update Aborted.")
            return False

    index[filename] = metadata

    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

def load_index(index_path="D:/EMRIData/saved_sims/index.json"):
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            return json.load(f)
    else:
        print(f"{index_path} does not exist.")

def save_emri_data(final, filename=False, folder="D:/EMRIData/saved_sims/", auto=False):
    if filename == False:
        filename = "auto_" + encode_filename()

    if not filename.endswith('.h5') and not filename.endswith('.hdf5'):
        filename += '.h5' 

    if os.path.exists(f"{folder}{filename}"):
        if auto == False:
            overwrite = input(f"Data for {filename} already exists. Overwrite? (y/n): ").strip().lower()
        else:
            overwrite = 'n'
        if overwrite == 'y':
            print(f"Overwriting {filename}...")
        else:
            num = 1
            base, ext = os.path.splitext(filename)
            while os.path.exists(f"{folder}{filename}"):
                filename = f"{base}_{num}{ext}"
                num += 1
            print(f"File now saved as {filename}")

    with h5py.File(folder + filename, 'w') as f:
        # String attributes
        f.attrs['name'] = final['name']

        # Boolean flags
        f.attrs['stop'] = final['stop']
        f.attrs['plunge'] = final['plunge']
        f.attrs['unbind'] = final['unbind']

        # Raw numerical data
        f.create_dataset('raw', data=final['raw'], compression='gzip')
        f.create_dataset('dTau_change', data=final['dTau_change'], compression='gzip')
        f.create_dataset('energy', data=final['energy'], compression='gzip')
        f.create_dataset('phi_momentum', data=final['phi_momentum'], compression='gzip')
        f.create_dataset('carter', data=final['carter'], compression='gzip')
        f.create_dataset('trackix', data=final['trackix'], compression='gzip')

        # Store inputs and issues as JSON strings
        inputs_json = json.dumps(final['inputs'])
        issues_json = json.dumps(final['issues'])

        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('inputs', data=inputs_json, dtype=dt)
        f.create_dataset('issues', data=issues_json, dtype=dt)

    input_labels = ["Central Body Mass", "Spin", "Mass Ratio", "Endflag", "Target Error", "Label", "Constants", "Orientation Velocity", "Tetrad Velocity", "Kepler Parameters", "Position", "True Velocity", "Units"]
    metadata = dict(zip(input_labels, final["inputs"]))
    moredata = {
        "Halted": final["stop"],
        "Plunged": final["plunge"],
        "Escaped": final["unbind"],
        "Created": time.strftime("%a, %d %b %Y %H:%M:%S UTC", time.gmtime())
    }
    for key, value in moredata.items():
        metadata[key] = value

    update_index(filename, metadata)
    return filename

def load_emri_data(filename, folder="D:/EMRIData/saved_sims/", quiet=False, reconstruct=True):
    print(f"Loading {folder + filename}") if not quiet else None
    with h5py.File(folder + filename, 'r') as f:
        final = {}

        # Basic attributes
        final['name'] = f.attrs['name']
        final['stop'] = f.attrs['stop']
        final['plunge'] = f.attrs['plunge']
        final['unbind'] = f.attrs['unbind']

        # Arrays
        final['raw'] = f['raw'][:]
        final['dTau_change'] = f['dTau_change'][:]
        final['energy'] = f['energy'][:]
        final['phi_momentum'] = f['phi_momentum'][:]
        final['carter'] = f['carter'][:]
        final['trackix'] = f['trackix'][:]

        # Decode JSONs back into Python objects
        final['inputs'] = json.loads(f['inputs'][()].decode('utf-8', errors='replace'))
        final['issues'] = [tuple(x) for x in json.loads(f['issues'][()].decode('utf-8'))]
        
        if reconstruct:
            # Reconstruct derived values
            raw = final["raw"]
            trackix = final["trackix"]
            final["pos"] = raw[:,1:4]
            final["all_vel"] = raw[:,4:]
            final["time"] = raw[:,0]
            time = final["time"]
            final["vel"] = (np.square(raw[:,5]) + np.square(raw[:,1]) * (np.square(raw[:,6]) + (np.sin(raw[:,2])**2)*np.square(raw[:,7])))**(0.5)
            final["spin"] = final["inputs"][1]
            final["qarter"] = final["carter"] + (final["spin"]*final["energy"] - final["phi_momentum"])**2
            final["tracktime"] = time[trackix + 1]
            final["tracktime"][0] = time[0]
                # Numba + vectorized version of check_interval
            final["interval"] = mm.check_interval_vec(raw, final["spin"])
                # Vectorized version of root_getter
            stuff = mm.root_getter_vec(final["energy"], final["phi_momentum"], final["carter"], final["spin"])
            final["pot_min"] = stuff[1][:, -1]
            incstuff = np.mean(np.abs(stuff[2][:, 1:3]), axis=1)
            final["inc"] = np.arccos(np.where(incstuff <= 1.0, incstuff, 1.0))
            final["it"] = stuff[0][:, -2]
            final["ot"] = stuff[0][:, -1]
            it, ot = final["it"], final["ot"]
            final["e"] = (ot - it)/(ot + it)
            final["r0"] = 0.5*(ot + it)
            final["p"] = 2*ot*it/(ot + it)
                # A list of indices that shows how elements of "raw" correspond to the indices in "trackix"
            ix = np.searchsorted(trackix, np.arange(len(raw)), side='right') - 1
                    # Make sure the values in here actually match "trackix" values
            ix = np.clip(ix, 0, len(trackix) - 1)
                    # Don't know where the off-by-1 issue keeps coming from (probably original sim) but this fixes it
            ix = np.insert(ix[:-1], 0, 0)
                # Vectorized version of get_true_anom
            denom = final["r0"][ix] * (1 - final["e"][ix]**2) / raw[:,1] - 1
            pre = np.sign(denom)
                    # Padding for divide by zero errors
            clipped = np.clip(np.abs(denom / (final["e"][ix] + 1e-15)), 0, 1)
            val = np.arccos(pre * clipped)
                    # Correct for inward motion (when r_dot < 0)
            val = np.where(raw[:,5] < 0, 2 * np.pi - val, val)
                    # Unwrap to make values monotonically increasing
            final["true_anom"] = np.unwrap(val.real)
                # Get omega and ascending/descinding node info
            ind = argrelmin(raw[:,1])[0]
            omega, otime = raw[ind,3] - 2*np.pi*np.arange(len(ind)), raw[ind,0]
            asc_node, asc_node_time = np.array([]), np.array([])
            des_node, des_node_time = np.array([]), np.array([])
            if max(raw[:,2]) - min(raw[:,2]) > 1e-15:
                theta_derv = np.interp(raw[:,0], 0.5*(raw[:,0][:-1] + raw[:,0][1:]), np.diff(raw[:,2])/np.diff(raw[:,0]))
                ind2 = argrelmin(theta_derv)[0] #indices for the ascending node
                ind3 = argrelmin(-theta_derv)[0] #indices for the descending node
                asc_node, asc_node_time = raw[ind2,3] - 2*np.pi*np.arange(len(ind2)), raw[ind2,0] #subtract the normal phi advancement
                des_node, des_node_time = raw[ind3,3] - 2*np.pi*np.arange(len(ind3)), raw[ind3,0] #subtract the normal phi advancement
                if asc_node.size == 0:
                    asc_node = np.array([])
                    asc_node_time = np.array([])
            final["omega"] = omega
            final["otime"] = otime
            final["asc_node"] = asc_node
            final["asc_node_time"] = asc_node_time
            final["des_node"] = des_node
            final["des_node_time"] = des_node_time
        else:
            print("Skipping derived value reconstruction") if not quiet else None
        print(f"Done") if not quiet else None
        return final

def delete_emri_data(filename, index_path="D:/EMRIData/saved_sims/index.json", folder="D:/EMRIData/saved_sims/", auto=False):
    full_path = os.path.join(folder, filename)
    
    # Delete the file
    if os.path.exists(full_path):
        if auto == True:
            confirm = "y"
        else:
            confirm = input(f"Are you sure you want to delete {filename}? (y/n): ").strip().lower()

        if confirm == "y":
            os.remove(full_path)
            print(f"Deleted file: {full_path}")
        else:
            print("File deletion aborted.")
    else:
        print(f"File not found: {full_path}")

    # Remove from index
    if os.path.exists(index_path):
        if confirm == "y":
            with open(index_path, 'r') as f:
                index = json.load(f)

            if filename in index:
                del index[filename]
                with open(index_path, 'w') as f:
                    json.dump(index, f, indent=2)
                print(f"Removed {filename} from index.")
            else:
                print(f"{filename} not found in index.")
        else:
            pass
    else:
        print("Index file does not exist.")


def EMRIGenMin(a, mu, endflag="radius < 2", mass=1.0, err_target=1e-15, label="default", cons=False, velorient=False, vel4=False, params=False, pos=False, veltrue=False, units="grav", verbose=1, eps=1e-5, override=False, bonk2=True):
    '''
    Generates orbit

    Parameters
    ----------
    a : float
        Dimensionless spin parameter of the central body. Valid for values between -1 and 1.
    mu : float
        Mass ratio between secondary body and central body. EMRI systems require mu to be less than or equal to 10^-4.
    endflag : string
        Condition for ending the simulation, written in the form '(variable) (comp.operator) (value)'
        Current valid variables:
            time - time, measured in geometric units
            phi_orbit - absolute phi displacement from original position, measured in radians
            rad_orbit - number of completed radial oscillations
            radius - distance from central body, measured in geometric units
            inclination - maximum displacement from north pole of central body, measured in radians
    mass : float, optional
        Mass of the central body. The default is 1.0.
    err_target : float, optional
        Maximum error allowed during the geodesic evaluation. The default is 1e-15.
    label : string, optional
        Internal label for the simulation. The default is "default", which gives it a label based on Keplerian paramters.
    cons : 3-element list of floats, optional
        Energy, Angular Momentum, and Carter Constant per unit mass. The default is False.
    velorient : 3-element list/array of floats, optional
        Ratio of velocity/speed of light (beta), angle between r-hat and trajectory (eta - radians), angle between phi hat and trajectory (xi - radians)
    vel4 : 4-element list/array of floats, optional
        Tetrad component velocities [t, r, theta, phi].
    params : 3-element list/array of floats, optional
        Semimajor axis, eccentricity, and inclination of orbit.
    pos : 4-element list/array of floats, optional
        Initial 4-position of particle. The default is False
    veltrue : 4-element list/array of floats, optional
        Initial 4-velocity of particle. The default is False.
    units : string, optional
        System of units for final output. The default is "grav".
        Current valid units:
            'grav' - Geometric units, with G, c, and M (central body mass) all set to 1.0.
            'mks' - Standard SI units, with G = 6.67e-11 N*m^2*kg^-2, c = 3e8 m*s^-1, and M in kg
            'cgs' - Standard cgs units, with G = 6.67e-8 dyn*cm^2*g^-2, c = 3e11 cm*s^-1, and M in g
    verbose : int, optional
        Toggle for progress updates as program runs. The default is 1.
        0 - No output
        1 - progress bar
        2 - full output

    Returns
    -------
    final: 35 element dict
        Various tracked and record-keeping values for the resulting orbit
        "name": Label for orbit if plotted, defaults to a list of Keplerian values for initial trajectory
        "raw": 8 element state of the orbiting body from beginning to end [time, radius, theta, phi, dt, dradius, dtheta, dphi]
        "inputs": initial input for function
        "pos": Subset of "raw", only includes radius, theta position, and phi positions
        "all_vel": Subset of "raw", only includes time, radius, theta position, and phi velocities
        "time": Subset of "raw", only includes time
        "true_anom": True anomaly measured at every moment in "time"; approximate
        "interval": Derived from "raw", spacetime interval at every point measured in "time"; should equal -1 at all times
        "vel": Derived from "raw", absolute velocity w.r.t. Mino time
        "dTau_change": Change in timestep 
        "energy": Energy of orbiting body at points of recalculation
        "phi_momentum": Angular momentum of orbiting body at points of recalculation
        "carter": Carter Constant of orbiting body (set to 0 for equatorial orbits) at points of recalculation
        "qarter": Carter Constant of orbiting body at points of recalculation
        "energy2": Specific Energy of orbiting body at all points in "time"
        "Lx_momentum": X-component of Specific Angular Momentum of orbiting body at all points in "time"
        "Ly_momentum": Y-component of Specific Angular Momentum of orbiting body at all points in "time"
        "Lz_momentum": Z-component of Specific Angular Momentum of orbiting body at all points in "time"
        "spin": Dimensionless spin of central body
        "freqs": Characteristic frequencies of orbit w.r.t. time at points of recalculation [radial, theta, phi]
        "pot_min": Radial distance of potential minimum at points of recalculation
        "e": Eccentricity at points of recalculation
        "inc": Inclination at points of recalculation
        "it": Inner turning point at points of recalculation
        "ot": Outer turning point at points of recalculation
        "r0": Semimajor axis at points of recalculation
        "tracktime": Value of time corresponding to points of recalculation
        "trackix": Indices of "raw" corresponding to points of recalculation
        "omega": Phi position of periapse
        "otime": Time at periapse
        "asc_node": Phi position of ascending node
        "asc_node_time": Time at ascending node
        "stop": 'True' if simulation was aborted before reaching end condition, False otherwise
        "plunge": 'True' if simulation ended in a plunge, False otherwise
        "issues": index and state corresponding to any point where Keplerian values read as complex
    '''
    termdict = {"time": "all_states[i][0]",
                "phi_orbit": "abs(all_states[i][3]/(2*np.pi))",
                "rad_orbit": "(true_anom[i] - true_anom[0])/(2*np.pi)",
                "radius": "all_states[i][1]",
                "inclination": "tracker[-1][2]"}
    
    try:
        terms = endflag.split(" ")
        if terms[0] == "rad_orbit":
            print("Busted, default to phi_orbit")
            terms[0] = "phi_orbit"
        newflag = termdict[terms[0]] + terms[1] + terms[2]
        
    except:
        print("Endflag should be a valid variable name, comparison operator, and numerical value, all separated by spaces")
        return 0
    
    inputs = [mass, a, mu, endflag, err_target, label, cons, velorient, vel4, params, pos, veltrue, units]          #Grab initial input in case you want to run the continue function
    all_states = [[np.zeros(8)]]                                                  #Grab that initial state         
    err_calc = 1 
    i = 0                                                                         #initialize step counter
    
    if (np.shape(veltrue) == (4,)) and (np.shape(pos) == (4,)):
        all_states[0] = [*pos, *veltrue]
    else:
        if verbose == True:
            print("Normalizing initial state")
        all_states[0], cons = mm.set_u_kerr(a, cons, velorient, vel4, params, pos)      #normalize initial state so it's actually physical
    
    interval = [mm.check_interval(mm.kerr, all_states[0], a)]           #create interval tracker
    metric = mm.kerr(all_states[0], a)[0]                                      #initial metric
    
    def viable_cons(new_cons, old_cons, state, a, scream=False):
        #print("----")
        E1, L1, C1 = old_cons
        E2, L2, C2 = new_cons
        R = lambda r, E, L, C, a: ((r**2 + a**2)*E - a*L)**2 - (r**2 - 2*r + a**2)*(r**2 + (L - a*E)**2 + C)
        if scream == True:
            import matplotlib.pyplot as plt
            turns1, flats1, zs1 = mm.root_getter(E1, L1, C1, a)
            turns2, flats2, zs2 = mm.root_getter(E2, L2, C2, a)
            low, high = min(turns1[-2], turns2[-2]), max(turns1[-1], turns2[-1])
            low_b, high_b = low - 0.01*(high - low), high + 0.01*(high - low) 
            r_vals = np.linspace(low_b, high_b, num=100)
            fig, ax = plt.subplots()
            ax.hlines(0, r_vals[0], r_vals[-1])
            ax.plot(r_vals, R(r_vals, *old_cons, a))
            ax.plot(r_vals, R(r_vals, *new_cons, a))
            ax.scatter(state[1], R(state[1], *old_cons, a))
            ax.scatter(state[1], R(state[1], *new_cons, a))
        potential_min = R(mm.root_getter(*new_cons, a)[1][-1], *new_cons, a)
        return potential_min
    
    def get_true_anom(state, r0, e):
        pre = np.sign((r0*(1 - e**2)/state[1] - 1)) #e is always positive
        val = np.arccos(pre*min(1.0, abs((r0*(1 - e**2)/state[1] - 1)/(e + 1e-15)))) #add a little tiny bias to get rid of divide by zero errors
        if state[5] < 0:
            val = 2*np.pi - val
        return val
    
    def get_true_anom2(state):
        E, L, C = getCons(state, a)
        turns, flats, zs = mm.root_getter(E, L, C, a)
        r0, e = 0.5*(turns[-1] + turns[-2]), (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
        pre = np.sign((r0*(1 - e**2)/state[1] - 1)) #e is always positive
        val = np.arccos(pre*min(1.0, abs((r0*(1 - e**2)/state[1] - 1)/(e + 1e-15)))) #add a little tiny bias to get rid of divide by zero errors
        if state[5] < 0:
            val = 2*np.pi - val
        return val
    
    if np.shape(cons) == (3,):
        initE, initLz, initC = cons
    else:
        initE, initLz, initC = getCons(all_states[0], a)
    
    pot_min = viable_cons([initE, initLz, initC], [initE, initLz, initC], all_states[0], a)
    count = 0
    while pot_min < 0.0:
        count += 1
        initE += err_target
        pot_min = viable_cons([initE, initLz, initC], [initE, initLz, initC], all_states[0], a)
        if count >= 21:
            print("Don't trust this!", pot_min, inputs)
            break
                
    coeff = np.array([initE**2 - 1, 2.0, (a**2)*(initE**2 - 1) - initLz**2 - initC, 2*((a*initE - initLz)**2) + 2*initC, -initC*(a**2)])

    constants = [[0,                  #index
                  all_states[0][0],   #time
                  initE,              #energy   
                  initLz,             #angular momentum (axial)
                  initC]]           #Carter constant (C)

    compErr = 0
    milestone = 0
    issues = [(None, None)]
    orbitside = np.sign(all_states[0][1] - pot_min)
    if orbitside == 0:
        orbitside = -1
    
    orbCount = 0
    stop = False
    
    if label == "default":
        checks = [cons, velorient, vel4, params, pos, veltrue]   
        if cons != False:
            label = "con_%s_%s_%s"%(*cons, )
        elif velorient != False:
            label = "vor_%s_%s_%s"%(*velorient, )
        elif vel4 != False:
            label = "tet_%s_%s_%s_%s"%(*vel4, )
        elif params != False:
            label = "par_%s_%s_%s"%(params[0], params[1], params[2]/np.pi)
        elif veltrue != False:
            label = "vel_%s_%s_%s_%s"%(*veltrue, )
        if pos != False:
            label += "_%s_%s_%s_%s"%(*pos, )
        label += "_%s"%(a)
    
    #Main Loop 
    dTau = 0.1*np.abs(np.real((all_states[0][1]/200)**(2)))
    dTau_change = [dTau]                                                #create dTau tracker
    borken = 0
    initflagval = eval(termdict[terms[0]])
    plunge, unbind = False, False
    def anglething(angle):
        return 0.5*np.pi - np.abs(angle%np.pi - np.pi/2)

    if verbose == 1:
        pbar = tqdm(total = 10000000, position=0)
        turns, flats, zs = mm.root_getter(initE, initLz, initC, a)
        pbar.set_postfix_str("Semilat: %s, Ecc %s, Peri: %s" %(np.round( 2*(turns[-1]*turns[-2])/(turns[-1] + turns[-2]), 3), np.round((turns[-1] - turns[-2])/(turns[-1] + turns[-2]), 3), np.round(turns[-2], 3)))
    progress = 0
    while (not(eval(newflag)) and (i < 10**7 or override)):
        try:
            update = False
            condate = False
            first = True
          
            #Grab the current state
            state = all_states[i]  
          
            #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
            if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001):
                plunge = True
                break
            
            if (state[1] >= (1 + np.sqrt(1 - a**2))*1e15):
                unbind = True
                break

            #break if something stops making sense
            if (np.nan in state or constants[-1][0] < 0) or (np.isnan(state[0])):
                print("HEWWO")
                plunge = True
                unbind = True
                break

            #Runge-Kutta update using geodesic
            old_dTau = dTau
            skip = False
            while ((err_calc >= err_target) or (first == True)) and (skip == False):
                new_step = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, a)
                step_check = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, a) 
                #jeremy with mods
                mod_new = np.array([*new_step[1:3], *new_step[4:]])
                mod_check = np.array([*step_check[1:3], *step_check[4:]])
                delt = mod_new - mod_check
                mod_r = np.array([*new_step[1:3], *new_step[4:]])
                err_calc = np.sqrt(np.dot(delt, delt)/np.dot(mod_r, mod_r))

                E, L, C = constants[-1][2:]
                # if (high inclination) AND ((very close to pole AND approaching pole) OR (dTau is very small AND dTau is monotonically non-increasing))
                if np.sign(new_step[6])*(np.pi/2 - new_step[2]%np.pi) <= -89.5*(np.pi/180) and np.mean(dTau_change[-10:]) <= 0.001*np.mean(dTau_change):
                    new_step[0] += ((new_step[0] - state[0])/abs(new_step[2] - state[2]))*(2*anglething(new_step[2]))
                    new_step[3] += 2*np.arccos(np.sin(abs(np.pi/2 - np.arccos(L/np.sqrt(L**2 + C))))/ np.sin(new_step[2]))
                    new_step[6] = -new_step[6]
                    #break
                
                speed = np.sqrt(new_step[5]**2 + (new_step[1]**2)*(new_step[6]**2 + (np.sin(new_step[2])*new_step[7])**2))
                old_dTau, dTau = dTau, min(dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
                #old_dTau, dTau = dTau, min(dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2/speed)
                if dTau <= 0.0:
                    err_calc = 1
                    dTau = old_dTau
                if new_step[0] - state[0] < 0:
                    err_calc = 1
                    dTau = 10*abs(old_dTau)
                if new_step[0] - state[0] > 100 and new_step[0] - state[0] < 100:
                    print("what the hell??")
                    print(dTau)
                    print(state, mm.check_interval(mm.kerr, state, a))
                    print(new_step, mm.check_interval(mm.kerr, new_step, a))
                    print(step_check, mm.check_interval(mm.kerr, step_check, a))
                    print(old_dTau, dTau)
                    print(err_calc)
                    oof = input("Type x to try unfucking this: ")
                    if "x" in oof:
                        err_calc = 1
                first = False

            metric = mm.kerr(new_step, a)[0]
            test = mm.check_interval(mm.kerr, new_step, a)
            looper = 0
            while (abs(test+1)>(err_target) or new_step[4] < 0.0) and looper < 10:
                borken = borken + 1
                og_new_step = np.copy(new_step)
                gtt, gtp = metric[0,0], metric[0,3]
                disc = 4*(gtp*new_step[4]*new_step[7])**2 - 4*gtt*(new_step[4]**2)*(np.einsum('ij, i, j ->', metric[1:,1:], new_step[5:], new_step[5:]) + 1)
                delt = (-2*gtp*new_step[4]*new_step[7] - np.sqrt(disc))/(2*gtt*new_step[4]*new_step[4])
                new_step[4] *= delt
                test = mm.check_interval(mm.kerr, new_step, a)
                looper += 1
            if (test+1) > err_target or new_step[4] < 0.0:
                new_step = np.copy(og_new_step)
   
            #constant modifying section
            turns, flats, zs = mm.root_getter(E, L, C, a)
            R0, ECC = 0.5*(turns[-1] + turns[-2]), (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
            compl, comph = np.arccos(-ECC), 2*np.pi - np.arccos(-ECC)
            S1, S2 = get_true_anom(state, R0, ECC), get_true_anom(new_step, R0, ECC)
            #print("hoo", constants[-1])#, all_states[constants[-1][0]])
            S1, S2, S3 = get_true_anom2(state), get_true_anom2(new_step), get_true_anom2(all_states[constants[-1][0]])
            if (((S2-comph) > 0 and (comph-S1) > 0) and np.abs(S1 - S3) > 0.5*np.pi) or (np.abs(new_step[3] - all_states[constants[-1][0]][3]) > 6*np.pi):
            #if (((S2-comph) > 0 and (comph-S1) > 0) and true_anom[-1] - true_anom[tracker[-1][-1]] > 0.5*np.pi) or ((S2 - true_anom[tracker[-1][-1]] > 4*np.pi and abs(new_step[1] - pot_min) < 0.5*max(outer_turn - pot_min, pot_min - inner_turn))):
                if (i - constants[-1][-1] > 10):
                    update = True
                    if ( np.sign(new_step[1] - pot_min) != orbitside):
                        orbitside *= -1
                    if mu != 0.0:
                        condate = True
                        dcons = mm.peters_integrate6_6(all_states[constants[-1][0]:i], a, mu, constants[-1][0], i)
                        new_step, ch_cons = mm.new_recalc_state9(constants[-1][2:], dcons, new_step, a)
                        pot_min = viable_cons(ch_cons, constants[-1][2:], new_step, a)
                        subcount = 0
                        while pot_min < -err_target:
                            viable_cons(ch_cons, constants[-1], new_step, a, True)
                            raise KeyError
                            if (subcount < 10) or subcount%10000000 == 0:
                                print(dcons, pot_min, "HEWWO??", subcount)
                            Lphi, ro = ch_cons[1], pot_min
                            ch_cons[0] += max(10**(-16), 2*(-pot_min)*((2*ro*((ro**3 + ro*(a**2) + 2*(a**2))*ch_cons[0] - 2*Lphi*a))**(-1)))
                            #ch_cons[0] += 10**(-16)
                            new_step = mm.recalc_state(ch_cons, new_step, a)
                            pot_min = viable_cons(ch_cons, new_step, a)
                            subcount += 1
                        if subcount > 0:
                            print(subcount, "oof", pot_min)
                        
            #Initializing for the next step
            #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
            #Only happens the step before the derivatives are recalculated.
            
            #Update stuff!
            if (update == True):
                if condate == False:
                    newE, newLz, newC = getCons(new_step, a)
                    constants.append([i, new_step[0], newE, newLz, newC])
                else:
                    constants.append([i, new_step[0], *ch_cons])
                    if verbose == 1:
                        turns, flats, zs = mm.root_getter(*constants[-1][2:], a)
                        pbar.set_postfix_str("Semilat: %s, Ecc %s, Peri: %s" %(np.round( 2*(turns[-1]*turns[-2])/(turns[-1] + turns[-2]), 3), np.round((turns[-1] - turns[-2])/(turns[-1] + turns[-2]), 3), np.round(turns[-2], 3)))
                if True in np.iscomplex(constants[-1]):
                    compErr += 1
                    issues.append((i, new_step[0]))  
            all_states.append(new_step)    #update position and velocity
            dTau_change.append(old_dTau)
            i += 1
            if verbose == 2:
                progress = max( abs((eval(termdict[terms[0]]) - initflagval)/(eval(terms[2]) - initflagval)), i/(10**7)) * 100
                if (progress >= milestone):
                    print("Program has completed " + str(round(eval(termdict[terms[0]]), 2)), ",", str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
                    milestone = int(progress) + 1
            elif verbose == 1:
                val = max( (10**7)*abs((eval(termdict[terms[0]]) - initflagval)/(eval(terms[2]) - initflagval)), i) - progress
                if val > 0:
                    pbar.update(val)
                    progress = max( (10**7)*abs((eval(termdict[terms[0]]) - initflagval)/(eval(terms[2]) - initflagval)), i)
            #print("maybe even finished?")
        #Lets you end the program before the established end without breaking anything
        except KeyboardInterrupt:
            print("\nEnding program")
            stop = True
            cap = len(all_states) - 1
            all_states = all_states[:cap]
            constants = constants[:cap]
            break
        except KeyError:
            print("\nEnding program")
            stop = True
            cap = len(all_states) - 1
            all_states = all_states[:cap]
            constants = constants[:cap]
            break
        
        '''
        except Exception as e:
            print("\nEnding program - ERROR")
            print(type(e), e)
            stop = True
            cap = len(all_states) - 1
            all_states = all_states[:cap]
            interval = interval[:cap]
            dTau_change = dTau_change[:cap]
            constants = constants[:cap]
            qarter = qarter[:cap]
            freqs = freqs[:cap]
            break
        '''
    if verbose == 1:
        pbar.close()
    #print(len(issues), len(all_states))
    #unit conversion stuff
    if units == "mks":
        G, c = 6.67*(10**-11), 3*(10**8)
    elif units == "cgs":
        G, c = 6.67*(10**-8),  3*(10**10)
    else:
        G, mass, c = 1.0, 1.0, 1.0
        
    if mu == 0.0:
        #so it gives actual numbers for pure geodesics
        mu = 1.0
        
    freqs = np.array([mm.freqs_finder(E, L, C, a) for i, t, E, L, C in constants]) 
    bulk = [mm.root_getter(E, L, C, a) for i, t, E, L, C in constants]
    vals = np.array([[flats[-1], (turns[-1] - turns[-2])/(turns[-1] + turns[-2]), 2*turns[-1]*turns[-2]/(turns[-1] + turns[-2]), turns[-2], turns[-1], np.arccos(np.mean(np.abs(zs[1:3])))] for turns, flats, zs in bulk])
    vals *= [(G*mass)/(c**2), 1, (G*mass)/(c**2), (G*mass)/(c**2), (G*mass)/(c**2), 1]
    constants = np.array([entry*np.array([1, (G*mass)/(c**3), mass*(c**2), mass*mass*G/c, (mass*mass*G/c)**2]) for entry in np.array(constants)], dtype=np.float64)
    all_states = np.array([entry*np.array([(G*mass)/(c**3), (G*mass)/(c**2), 1.0, 1.0, 1.0, c, (c**3)/(G*mass), (c**3)/(G*mass)]) for entry in np.array(all_states)])

    if verbose == 2:
        print("There were " + str(compErr) + " issues with complex roots/turning points.")
    final = {"name": label,
             "raw": all_states,
             "inputs": inputs,
             "pos": all_states[:,1:4],
             "all_vel": all_states[:,4:], 
             "time": all_states[:,0],
             "vel": (np.square(all_states[:,5]) + np.square(all_states[:,1]) * (np.square(all_states[:,6]) + (np.sin(all_states[:,2])**2)*np.square(all_states[:,7])))**(0.5),
             "energy": constants[:, 2],
             "phi_momentum": constants[:, 3],
             "carter": constants[:, 4],
             "spin": a,
             "freqs": freqs,
             "r0": vals[:,0],
             "e": vals[:,1],
             "p": vals[:,2],
             "it": vals[:,3],
             "ot": vals[:,4],
             "inc": vals[:,5],
             "tracktime": constants[:,1],
             "trackix": constants[:, 0],
             "stop": stop,
             "plunge": plunge,
             "unbind": unbind,
             "issues": issues}
    return final

def guessmaker2(cons, old_state, a):
    import scipy.interpolate as interp
    state = mm.recalc_state(cons, old_state[:4], a)
    #state[5] *= -1
    freqs = mm.freqs_finder(*cons, a)
    turns, flats, zs = mm.root_getter(*cons, a)
    #print(turns)
    vals = np.sort(np.arccos(zs))
    #print(vals)
    p, e, inc = 2*turns[-1]*turns[-2]/(turns[-1] + turns[-2]), (turns[-1] - turns[-2])/(turns[-1] + turns[-2]), vals[1]
    r_min, r_max, r0 = p/(1+e), p/(1-e), p/(1-e**2)
    #print(p,e,inc)
    
    ano1 = np.real(np.arccos((1/e)*(p/state[1] - 1))*180/np.pi)
    thetano1 = np.real(np.arcsin((2*state[2] - np.pi)/(2*inc - np.pi))) if inc != np.pi/2 else 0.0
    ano1 = 360 - ano1 if state[5] < 0 else ano1
    thetano1 = thetano1%(2*np.pi) if state[6] < 0 else (thetano1 + np.pi)%(2*np.pi)
    theta_cor = not(state[2] == np.pi/2 and np.abs(state[6]) < 1e-15)
    print(theta_cor)
    print(state[1], p)

    if state[1] > p or (state[1] == p and state[5] >= 0):                 #outer orbit
        anomaly = np.linspace(ano1, 270, int(270 - ano1) + 1)*np.pi/180   #true anomaly
        rads = p/(1 + e*np.cos(anomaly))                                  #radial position
        print(rads)
        thetanoms = thetano1 + anomaly - anomaly[0]                       #theta anomaly
        thets = np.pi/2 - (inc - np.pi/2)*np.sin(thetanoms)*theta_cor               #theta position
        uph = (((1 + e*np.cos(anomaly))/(1 - e**2))**2)/(r0**(3/2) + a)
        #kep2con = 0.5*(state[1]**2)*np.sqrt(state[7]**2 + state[6]**2)    #approx "area per unit time" (keplers 2nd law)
        ut = (1 - 2*rads/(rads**2 + (a**2)*(np.cos(thets))))**(-1)
        #times = (np.cumsum(ut) - ut[0])*np.pi*(((p/(1-e))**(3/2)))/np.sum(ut) + state[0]

    elif state[1] < p or (state[1] == p and state[5] < 0):                #inner orbit (bound)
        #print("this?")
        print(inc)
        anomaly = np.linspace(ano1, 450, int(450 - ano1) + 1)*np.pi/180
        rads = p/(1 + e*np.cos(anomaly))
        thetanoms = thetano1 + anomaly - anomaly[0]                       #theta anomaly
        #print(inc, np.pi/2 - inc)
        thets = np.pi/2 - (inc - np.pi/2)*np.sin(thetanoms)*theta_cor               #theta position
        uph = (((1 + e*np.cos(anomaly))/(1 - e**2))**2)/(r0**(3/2) + a)    
        #approx ratio between frequencies: r0 circular orbit/r_min circular orbit
        ut = (1 - 2*rads/(rads**2 + (a**2)*(np.cos(thets))))**(-1)
        #times = (np.cumsum(ut) - ut[0])*np.pi*(((p/(1+e))**(3/2)))/np.sum(ut) + state[0]
        
    hold = np.arange(len(rads))
    
    #np.cumsum(ut - ut[0])*(np.pi*((p)**(3/2)))/np.sum(ut) + state[0]
    #print((np.pi*((p)**(3/2))))
    #print(np.sum(ut))
    #print((np.cumsum(ut) - ut[0])[-1])
    ut *= state[4]/ut[0]
    uth = interp.CubicSpline(hold, thets)(hold, 1)
    #uph = interp.CubicSpline(hold, phis)(hold, 1)
    times = np.append([0], np.cumsum(ut)[:-1])*np.pi*(((p/(1+e))**(3/2)))/np.sum(ut[:-1]) + state[0]
    print(np.imag(times))
    phis = np.cumsum(np.append(uph[:-1]*np.diff(times), uph[-1]*np.diff(times)[-1])) + state[3]
    ur = interp.CubicSpline(hold, rads)(hold, 1)/(np.diff(times)/np.diff(hold))[0]
    #print(uth)
    #ut *= state[4]/ut[0] if not(np.isinf(1/ut[0])) else 
    #ur *= state[5]/ur[0]
    #uth *= state[6]/uth[0]
    #uph *= state[7]/uph[0] 
    #plt.plot(gtt)
    #plt.plot(ut)
    return np.real(np.transpose([times, rads, thets, phis, ut, ur, uth, uph]))

def guessmaker3(cons, old_state, a):
    bound = True
    import scipy.interpolate as interp
    print("---")
    print(old_state)
    state = mm.recalc_state(cons, old_state, a)
    print(state)
    #state[5] *= -1
    freqs = mm.freqs_finder(*cons, a)
    turns, flats, zs = mm.root_getter(*cons, a)
    print(turns)
    print(np.where(np.isreal(turns)==True))
    if len(np.where(np.isreal(turns)==True)[0]) < 4:
        print("unbound or plunge!")
        bound = False

    #print(turns)
    vals = np.sort(np.arccos(zs))
    #print(vals)
    p, e, inc = 2*turns[-1]*turns[-2]/(turns[-1] + turns[-2]), (turns[-1] - turns[-2])/(turns[-1] + turns[-2]), vals[1]
    r_min, r_max, r0 = p/(1+e), p/(1-e), p/(1-e**2)
    print(p,e,inc)
    
    ano1 = np.real(np.arccos((1/e)*(p/state[1] - 1))*180/np.pi)
    thetano1 = np.real(np.arcsin((2*state[2] - np.pi)/(2*inc - np.pi))) if inc != np.pi/2 else 0.0
    ano1 = 360 - ano1 if state[5] < 0 else ano1
    thetano1 = thetano1%(2*np.pi) if state[6] < 0 else (thetano1 + np.pi)%(2*np.pi)
    theta_cor = not(state[2] == np.pi/2 and np.abs(state[6]) < 1e-15)
    print(ano1)

    if state[1] > p or (state[1] == p and state[5] >= 0):                 #bound orbit
        anomaly = np.linspace(ano1, 270, int(270 - ano1) + 1)*np.pi/180   #true anomaly
        rads = p/(1 + e*np.cos(anomaly))                                  #radial position
        thetanoms = thetano1 + anomaly - anomaly[0]                       #theta anomaly
        thets = np.pi/2 - (inc - np.pi/2)*np.sin(thetanoms)*theta_cor               #theta position
        uph = (((1 + e*np.cos(anomaly))/(1 - e**2))**2)/(r0**(3/2) + a)
        #kep2con = 0.5*(state[1]**2)*np.sqrt(state[7]**2 + state[6]**2)    #approx "area per unit time" (keplers 2nd law)
        ut = (1 - 2*rads/(rads**2 + (a**2)*(np.cos(thets))))**(-1)
        #times = (np.cumsum(ut) - ut[0])*np.pi*(((p/(1-e))**(3/2)))/np.sum(ut) + state[0]

    elif state[1] < p or (state[1] == p and state[5] < 0):                #inner orbit (bound)
        #print("this?")
        anomaly = np.linspace(ano1, 450, int(450 - ano1) + 1)*np.pi/180
        rads = p/(1 + e*np.cos(anomaly))
        thetanoms = thetano1 + anomaly - anomaly[0]                       #theta anomaly
        #print(inc, np.pi/2 - inc)
        thets = np.pi/2 - (inc - np.pi/2)*np.sin(thetanoms)*theta_cor               #theta position
        uph = (((1 + e*np.cos(anomaly))/(1 - e**2))**2)/(r0**(3/2) + a)    
        #approx ratio between frequencies: r0 circular orbit/r_min circular orbit
        ut = (1 - 2*rads/(rads**2 + (a**2)*(np.cos(thets))))**(-1)
        #times = (np.cumsum(ut) - ut[0])*np.pi*(((p/(1+e))**(3/2)))/np.sum(ut) + state[0]
        
    hold = np.arange(len(rads))
    
    #np.cumsum(ut - ut[0])*(np.pi*((p)**(3/2)))/np.sum(ut) + state[0]
    #print((np.pi*((p)**(3/2))))
    #print(np.sum(ut))
    #print((np.cumsum(ut) - ut[0])[-1])
    ut *= state[4]/ut[0]
    uth = interp.CubicSpline(hold, thets)(hold, 1)
    #uph = interp.CubicSpline(hold, phis)(hold, 1)
    times = np.append([0], np.cumsum(ut)[:-1])*np.pi*(((p/(1+e))**(3/2)))/np.sum(ut[:-1]) + state[0]
    phis = np.cumsum(np.append(uph[:-1]*np.diff(times), uph[-1]*np.diff(times)[-1])) + state[3]
    ur = interp.CubicSpline(hold, rads)(hold, 1)/(np.diff(times)/np.diff(hold))[0]
    #print(uth)
    #ut *= state[4]/ut[0] if not(np.isinf(1/ut[0])) else 
    #ur *= state[5]/ur[0]
    #uth *= state[6]/uth[0]
    #uph *= state[7]/uph[0] 
    #plt.plot(gtt)
    #plt.plot(ut)
    return np.real(np.transpose([times, rads, thets, phis, ut, ur, uth, uph]))

def corrector(cons, guess, a): #doesn't work
    new = guess.copy()
    def dervs(cons, state, a):
        E, L, C = cons
        r, T = state[1], state[2]
        sint, cost = np.sin(T), np.cos(T)
        sig, delt = r**2 + (a*cost)**2, r**2 - 2*r + a**2
        u0, u2, u3 = state[4], state[6], state[7]
        dEdr = -2*(a*(sint**2)*u3 - u0)*(r**2 - (a*cost)**2)/(sig**2)
        dEdT = 4*a*r*((r**2 + a**2)*u3 - a*u0)*cost*sint/(sig**2)
        dLdr = (2*(sint**2)/(sig**2))*(a*(r**2 - (a*cost)**2)*u0 + (r**5 + 2*(a**2)*(r**3) - (a*r*sint)**2 + r*((a*cost)**4) + (a**4)*((sint*cost)**2))*u3)
        dLdT = (2*sint*cost/(sig**2))*(-2*a*r*(r**2 + a**2)*u0 + (delt*((a*sint)**2)*((a*sint)**2 - 2*(r**2 + a**2)) + (r**2 + a**2)**3)*u3)
        dQdr = (2/sint**2)*(dLdr - a*dEdr*(sint**2))*(L - a*E*(sint**2)) + 4*r*sig*(u2**2)
        dQdT = (1/(sint**3))*(2*(dLdT - a*(dEdT*(sint**2) + 2*E*sint*cost))*(L - a*E*(sint**2))*sint - 2*cost*((L - a*E*(sint**2))**2)) - 2*(a**2)*sint*cost - 4*(a**2)*sint*cost*sig*(u2**2)
        dCdr = dQdr - 2*(a*E - L)*(a*dEdr - dLdr)
        dCdT = dQdT - 2*(a*E - L)*(a*dEdT - dLdT)
        return np.array([dEdr, dLdr, dCdr, dEdT, dLdT, dCdT])
        
    def intderv(state, a):
        #print(state)
        r, T = state[1], state[2]
        sint, cost = np.sin(T), np.cos(T)
        sig, delt = r**2 + (a*cost)**2, r**2 - 2*r + a**2
        u0, u1, u2, u3 = state[4:]
        #print(r, T, sig, u0, a, sint, sig, u3)
        du0 = -2*(1 - 2*r/sig)*u0 - (4*a*r*(sint**2)/sig)*u3
        du1 = 2*(sig/delt)*u1
        du2 = 2*sig*u2
        du3 = 2*(r**2 + a**2 + 2*r*((a*sint)**2)/sig)*(sint**2)*u3
        return np.array([du0, du1, du2, du3])
    #print(new[:2])
    dcons = np.array([np.array(cons) - getCons(state, a) for state in new])
    print(dcons[0:2])
    print(input("hhhe"))
    dervs = np.array([dervs(cons, state, a) for state in new])
    print(dervs[:2])
    dervs = np.where(np.isinf(1/dervs), 0.0, 1/dervs)
    #return dervs[0]

    print(dervs[:2])
    print(input("hhhe"))
    delt_r, delt_T = np.sum(dcons*dervs[:,:3], axis=1), np.sum(dcons*dervs[:,3:], axis=1)
    print(dervs[0,:3]*dcons[0])
    print(dervs[0,3:]*dcons[0])
    print(input("check this"))
    
    print(delt_r[:2])
    print(delt_T[:2])
    #print(delt_T[:2]%(
    
    new[:, 1] += delt_r
    new[:, 2] += delt_T
    new[:, 2] = np.arccos(np.cos(new[:, 2]))
    int_diffs = -1 - np.array([mm.check_interval(mm.kerr, i, a) for i in new])
    int_dervs = np.array([intderv(state, a) for state in new])
    delt_vel = np.array([(-1 - mm.check_interval(mm.kerr, state, a))/intderv(state, a) for state in new])
    #new[:, 4:] += delt_vel
    return new

def corrector2(cons, guess, a):
    dcons = np.array([cons - getCons(state) for state in guess])
    #rho, z
    def dervs(state, a):
        E, L, C = getCons(state, a)
        r, T = state[1], state[2]
        sint, cost = np.sin(T), np.cos(T)
        sig, delt = r**2 + (a*cost)**2, r**2 - 2*r + a**2
        u0, u2, u3 = state[4], state[6], state[7]
        dEdr = -2*(a*(sint**2)*u3 - u0)*(r**2 - (a*cost)**2)/(sig**2)
        dEdT = 4*a*r*((r**2 + a**2)*u3 - a*u0)*cost*sint/(sig**2)
        dLdr = (2*(sint**2)/(sig**2))*(a*(r**2 - (a*cost)**2)*u0 + (r**5 + 2*(a**2)*(r**3) - (a*r*sint)**2 + r*((a*cost)**4) + (a**4)*((sint*cost)**2))*u3)
        dLdT = (2*sint*cost/(sig**2))*(-2*a*r*(r**2 + a**2)*u0 + (delt*((a*sint)**2)*((a*sint)**2 - 2*(r**2 + a**2)) + (r**2 + a**2)**3)*u3)
        dQdr = (2/sint**2)*(dLdr - a*dEdr*(sint**2))*(L - a*E*(sint**2)) + 4*r*sig*(u2**2)
        dQdT = (1/(sint**3))*(2*(dLdT - a*(dEdT*(sint**2) + 2*E*sint*cost))*(L - a*E*(sint**2))*sint - 2*cost*((L - a*E*(sint**2))**2)) - 2*(a**2)*sint*cost - 4*(a**2)*sint*cost*sig*(u2**2)
        dCdr = dQdr - 2*(a*E - L)*(a*dEdr - dLdr)
        dCdT = dQdT - 2*(a*E - L)*(a*dEdT - dLdT)
        return np.array([[dEdr, dEdT],
                         [dLdr, dLdT],
                         [dCdr, dCdT]])
        def dervs(state, a0):
            E0, L0, C0 = getCons(state, a0)
            r, T, a, u0, u2, u3 = sp.symbols('r T a u0 u2 u3', real=True)
            sig = r**2 + (a*sp.cos(T))**2
            delt = r**2 - 2*r + a**2
            E = (1 - 2*r/sig)*u0 + 2*a*r*(sp.sin(T)**2)*u3/sig
            L = -2*a*r*(sp.sin(T)**2)*u0/sig + ((r**2 + a**2)**2 - delt*((a*sp.sin(T))**2))*(sp.sin(T)**2)*u3/sig
            Q = ((L - a*E*(sp.sin(T)**2))**2)/(sp.sin(T)**2) + (a*sp.cos(T))**2 + (sig*u2)**2
            C = Q - (a*E - L)**2
            
            dEdr, dLdr, dCdr = sp.diff(E, r), sp.diff(L, r), sp.diff(C, r)
            dEdT, dLdT, dCdT = sp.diff(E, T), sp.diff(L, T), sp.diff(C, T)
            d2Edr2, d2Ldr2, d2Cdr2 = sp.diff(dEdr, r), sp.diff(dLdr, r), sp.diff(dCdr, r)
            d2EdT2, d2LdT2, d2CdT2 = sp.diff(dEdT, T), sp.diff(dLdT, T), sp.diff(dCdT, T)
            d2EdrdT, d2LdrdT, d2CdrdT = sp.diff(dEdr, T), sp.diff(dLdr, T), sp.diff(dCdr, T)
            
            #d1_block = np.array([

            dEdr = -2*(a*(sint**2)*u3 - u0)*(r**2 - (a*cost)**2)/(sig**2)
            dEdT = 4*a*r*((r**2 + a**2)*u3 - a*u0)*cost*sint/(sig**2)
            dLdr = (2*(sint**2)/(sig**2))*(a*(r**2 - (a*cost)**2)*u0 + (r**5 + 2*(a**2)*(r**3) - (a*r*sint)**2 + r*((a*cost)**4) + (a**4)*((sint*cost)**2))*u3)
            dLdT = (2*sint*cost/(sig**2))*(-2*a*r*(r**2 + a**2)*u0 + (delt*((a*sint)**2)*((a*sint)**2 - 2*(r**2 + a**2)) + (r**2 + a**2)**3)*u3)
            dQdr = (2/sint**2)*(dLdr - a*dEdr*(sint**2))*(L - a*E*(sint**2)) + 4*r*sig*(u2**2)
            dQdT = (1/(sint**3))*(2*(dLdT - a*(dEdT*(sint**2) + 2*E*sint*cost))*(L - a*E*(sint**2))*sint - 2*cost*((L - a*E*(sint**2))**2)) - 2*(a**2)*sint*cost - 4*(a**2)*sint*cost*sig*(u2**2)
            dCdr = dQdr - 2*(a*E - L)*(a*dEdr - dLdr)
            dCdT = dQdT - 2*(a*E - L)*(a*dEdT - dLdT)
            return np.array([[dEdr, dEdT],
                             [dLdr, dLdT],
                             [dCdr, dCdT]])
    Amat = np.array([dervs(cons, state, a) for state in guess])
    Atrans = np.transpose(Amat, axes=(0,2,1))
    Dblock = np.einsum("ijk, ikl -> ijl", Atrans, Amat)
    Amat = np.transpose(np.append([dcons], [dcons], axis=0), axes=(1,2,0))/1e-7
    #this looks weird? Shouldn't Amat bet the derivatives?

def clean_continue(data, endflag = False, verbose=False):
    #continue from the last crossing, I think?
    verbose_new = verbose
    mass, a, mu, endflag_old, err_target, label_old, cons, velorient, vel4, params, pos, units = data["inputs"]
    if endflag == False:
        endflag = endflag_old
    #inputs = [mass, a, mu, endflag, err_target, label, cons, velorient, vel4, params, pos, units]          #Grab initial input in case you want to run the continue function
    lastcross = int(data["trackix"][-1])
    if units == "mks":
        G, c = 6.67*(10**-11), 3*(10**8)
    elif units == "cgs":
        G, c = 6.67*(10**-8),  3*(10**10)
    else:
        G, mass, c = 1.0, 1.0, 1.0
    
    newstart = data["raw"][lastcross]/np.array([(G*mass)/(c**3), (G*mass)/(c**2), 1.0, 1.0, 1.0, c, (c**3)/(G*mass), (c**3)/(G*mass)])
    pos_new, vel_new = newstart[:4], newstart[4:]
    newdata = EMRIGenerator(a, mu, endflag, mass, err_target, label=label_old, pos=pos_new, veltrue=vel_new, verbose=verbose_new)
    
    final = {"name": label_old,
             "raw": np.concatenate((data["raw"][:lastcross], newdata["raw"])),
             "inputs": data["inputs"],
             "pos": np.concatenate((data["pos"][:lastcross], newdata["pos"])),
             "all_vel": np.concatenate((data["all_vel"][:lastcross], newdata["all_vel"])), 
             "time": np.concatenate((data["time"][:lastcross], newdata["time"])),
             "interval": np.concatenate((data["interval"][:lastcross], newdata["interval"])),
             "vel": np.concatenate((data["vel"][:lastcross], newdata["vel"])),
             "dTau_change": np.concatenate((data["dTau_change"][:lastcross], newdata["dTau_change"])),
             "energy": np.concatenate((data["energy"], newdata["energy"])),
             "phi_momentum": np.concatenate((data["phi_momentum"], newdata["phi_momentum"])),
             "carter": np.concatenate((data["carter"], newdata["carter"])),
             "qarter": np.concatenate((data["qarter"], newdata["qarter"])),
             "energy2": np.concatenate((data["energy2"], newdata["energy2"])),
             "Lx_momentum": np.concatenate((data["Lx_momentum"], newdata["Lx_momentum"])),
             "Ly_momentum": np.concatenate((data["Ly_momentum"], newdata["Ly_momentum"])),
             "Lz_momentum": np.concatenate((data["Lz_momentum"], newdata["Lz_momentum"])),
             "spin": a,
             "freqs": np.concatenate((data["freqs"], newdata["freqs"])),
             "pot_min":np.concatenate((data["pot_min"], newdata["pot_min"])),
             "e": np.concatenate((data["e"], newdata["e"])),
             "inc": np.concatenate((data["inc"], newdata["inc"])),
             "it": np.concatenate((data["it"], newdata["it"])),
             "ot": np.concatenate((data["ot"], newdata["ot"])),
             "r0": np.concatenate((data["r0"], newdata["r0"])),
             "tracktime": np.concatenate((data["tracktime"], newdata["tracktime"])),
             "trackix": np.concatenate((data["trackix"], newdata["trackix"])),
             "omega": np.concatenate((data["omega"][:lastcross], newdata["omega"] - 2*np.pi*len(data["omega"][:lastcross]))),
             "otime": np.concatenate((data["otime"][:lastcross], newdata["otime"])),
             "asc_node": np.concatenate((data["asc_node"][:lastcross], newdata["asc_node"] - 2*np.pi*len(data["asc_node"][:lastcross]))),
             "asc_node_time": np.concatenate((data["asc_node_time"][:lastcross], newdata["asc_node_time"])),
             "stop": newdata["stop"],
             "plunge": newdata["plunge"],
             "issues": np.concatenate((data["issues"], newdata["issues"]))}
    return final

def dict_saver(data, filename):
    np.save(filename, data) 
    return True
        
def dict_from_file(filename):
    if ".npy" not in filename:
        filename = filename+".npy"
    data = np.load(filename, allow_pickle='TRUE').item()
    return data

def EGTimer(a, mu, endflag="radius < 2", mass=1.0, err_target=1e-15, label="default", cons=False, velorient=False, vel4=False, params=False, pos=False, veltrue=False, units="grav", verbose=False, eps=1e-5, conch=6, trigger=2, override=False, bonk=1, bonk2=True):
    '''
    Generates orbit

    Parameters
    ----------
    a : float
        Dimensionless spin parameter of the central body. Valid for values between -1 and 1.
    mu : float
        Mass ratio between secondary body and central body. EMRI systems require mu to be less than or equal to 10^-4.
    endflag : string
        Condition for ending the simulation, written in the form '(variable) (comp.operator) (value)'
        Current valid variables:
            time - time, measured in geometric units
            phi_orbit - absolute phi displacement from original position, measured in radians
            rad_orbit - number of completed radial oscillations
            radius - distance from central body, measured in geometric units
            inclination - maximum displacement from north pole of central body, measured in radians
    mass : float, optional
        Mass of the central body. The default is 1.0.
    err_target : float, optional
        Maximum error allowed during the geodesic evaluation. The default is 1e-15.
    label : string, optional
        Internal label for the simulation. The default is "default", which gives it a label based on Keplerian paramters.
    cons : 3-element list of floats, optional
        Energy, Angular Momentum, and Carter Constant per unit mass. The default is False.
    velorient : 3-element list/array of floats, optional
        Ratio of velocity/speed of light (beta), angle between r-hat and trajectory (eta - radians), angle between phi hat and trajectory (xi - radians)
    vel4 : 4-element list/array of floats, optional
        Tetrad component velocities [t, r, theta, phi].
    params : 3-element list/array of floats, optional
        Semimajor axis, eccentricity, and inclination of orbit.
    pos : 4-element list/array of floats, optional
        Initial 4-position of particle. The default is False
    veltrue : 4-element list/array of floats, optional
        Initial 4-velocity of particle. The default is False.
    units : string, optional
        System of units for final output. The default is "grav".
        Current valid units:
            'grav' - Geometric units, with G, c, and M (central body mass) all set to 1.0.
            'mks' - Standard SI units, with G = 6.67e-11 N*m^2*kg^-2, c = 3e8 m*s^-1, and M in kg
            'cgs' - Standard cgs units, with G = 6.67e-8 dyn*cm^2*g^-2, c = 3e11 cm*s^-1, and M in g
    verbose : bool, optional
        Toggle for progress updates as program runs. The default is False.

    Returns
    -------
    final: 35 element dict
        Various tracked and record-keeping values for the resulting orbit
        "name": Label for orbit if plotted, defaults to a list of Keplerian values for initial trajectory
        "raw": 8 element state of the orbiting body from beginning to end [time, radius, theta, phi, dt, dradius, dtheta, dphi]
        "inputs": initial input for function
        "pos": Subset of "raw", only includes radius, theta position, and phi positions
        "all_vel": Subset of "raw", only includes time, radius, theta position, and phi velocities
        "time": Subset of "raw", only includes time
        "true_anom": True anomaly measured at every moment in "time"; approximate
        "interval": Derived from "raw", spacetime interval at every point measured in "time"; should equal -1 at all times
        "vel": Derived from "raw", absolute velocity w.r.t. Mino time
        "dTau_change": Change in timestep 
        "energy": Energy of orbiting body at points of recalculation
        "phi_momentum": Angular momentum of orbiting body at points of recalculation
        "carter": Carter Constant of orbiting body (set to 0 for equatorial orbits) at points of recalculation
        "qarter": Carter Constant of orbiting body at points of recalculation
        "energy2": Specific Energy of orbiting body at all points in "time"
        "Lx_momentum": X-component of Specific Angular Momentum of orbiting body at all points in "time"
        "Ly_momentum": Y-component of Specific Angular Momentum of orbiting body at all points in "time"
        "Lz_momentum": Z-component of Specific Angular Momentum of orbiting body at all points in "time"
        "spin": Dimensionless spin of central body
        "freqs": Characteristic frequencies of orbit w.r.t. time at points of recalculation [radial, theta, phi]
        "pot_min": Radial distance of potential minimum at points of recalculation
        "e": Eccentricity at points of recalculation
        "inc": Inclination at points of recalculation
        "it": Inner turning point at points of recalculation
        "ot": Outer turning point at points of recalculation
        "r0": Semimajor axis at points of recalculation
        "tracktime": Value of time corresponding to points of recalculation
        "trackix": Indices of "raw" corresponding to points of recalculation
        "omega": Phi position of periapse
        "otime": Time at periapse
        "asc_node": Phi position of ascending node
        "asc_node_time": Time at ascending node
        "stop": 'True' if simulation was aborted before reaching end condition, False otherwise
        "plunge": 'True' if simulation ended in a plunge, False otherwise
        "issues": index and state corresponding to any point where Keplerian values read as complex
    '''
    termdict = {"time": "all_states[i][0]",
                "phi_orbit": "abs(all_states[i][3]/(2*np.pi))",
                "rad_orbit": "(true_anom[i] - true_anom[0])/(2*np.pi)",
                "radius": "all_states[i][1]",
                "inclination": "tracker[-1][2]"}
    
    try:
        terms = endflag.split(" ")
        newflag = termdict[terms[0]] + terms[1] + terms[2]
    except:
        print("Endflag should be a valid variable name, comparison operator, and numerical value, all separated by spaces")
        return 0
    
    inputs = [mass, a, mu, endflag, err_target, label, cons, velorient, vel4, params, pos, units]          #Grab initial input in case you want to run the continue function
    all_states = [[np.zeros(8)]]                                                  #Grab that initial state         
    err_calc = 1 
    i = 0                                                                         #initialize step counter
    
    if (np.shape(veltrue) == (4,)) and (np.shape(pos) == (4,)):
        all_states[0] = [*pos, *veltrue]
    else:
        if verbose == True:
            print("Normalizing initial state")
        all_states[0], cons = mm.set_u_kerr(a, cons, velorient, vel4, params, pos)      #normalize initial state so it's actually physical
    
    interval = [mm.check_interval(mm.kerr, all_states[0], a)]           #create interval tracker
    metric = mm.kerr(all_states[0], a)[0]                                      #initial metric
    
    def viable_cons(constants, state, a, scream=False):
        #print("----")
        energy, lz, cart = constants
        coeff = np.array([energy**2 - 1, 2, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*((a*energy - lz)**2 + cart), -cart*(a**2)])
        coeff2 = np.polyder(coeff)
        coeff_2 = lambda r: 4*(energy**2 - 1)*(r**3) + 6*(r**2) + 2*((a**2)*(energy**2 - 1) - lz**2 - cart)*r +  2*((a*energy - lz)**2 + cart)
        flats = np.roots(coeff2)
        #op.plt.plot(np.linspace(9.9, 10.1), np.polyval(coeff2, np.linspace(9.9,10.1)))
        #op.plt.plot(np.linspace(9.9, 10.1), np.polyval(coeff, np.linspace(9.9,10.1)))
        #op.plt.hlines(0, 9.9, 10.1)
        if scream == True:
            print(flats)
            print(np.real(flats))
            print(coeff2)
            try:
                flat_check = optimize.fsolve(coeff_2, np.real(flats))
            except:
                flat_check = "arg!!"
            print(flat_check)
            print("sta")
        #print(flats)
        #flat_check = optimize.fsolve(coeff_2, flats)
        #print(flat_check)
        flats = flats.real[abs(flats.imag)<1e-11]
        #print(flats)
        if len(flats) == 0:
            return 0
        try:
            pot_min = max(flats)
        except:
            print("HELLOP")
            print(constants)
            op.potentplotter(energy, lz, cart, a)
        if scream == True:
            print(pot_min, flats)
            print("ROOTER")
            print(mm.root_getter(energy, lz, cart, a))
        pot_min = mm.root_getter(energy, lz, cart, a)[1][-1]
        #print(pot_min, "hellur?")
        potential_min = np.polyval(coeff, pot_min)
        return potential_min
    
    def bl2cart_oof(state, a):
        t, r, thet, phi, ut, ur, uthet, uphi = state
        sint, cost, sinp, cosp = np.sin(thet), np.cos(thet), np.sin(phi), np.cos(phi)
        new = [t, np.sqrt(r**2 + a**2)*sint*cosp, np.sqrt(r**2 + a**2)*sint*sinp, r*cost,
                ut, r*ur*sint*cosp/np.sqrt(r**2 + a**2) + np.sqrt(r**2 + a**2)*(uthet*cost*cosp - uphi*sint*sinp),
                r*ur*sint*sinp/np.sqrt(r**2 + a**2) + np.sqrt(r**2 + a**2)*(uthet*cost*sinp + uphi*sint*cosp),
                ur*cost - r*uthet*sint]
        return np.array(new)

    def get_true_anom(state, r0, e):
        pre = np.sign((r0*(1 - e**2)/state[1] - 1)) #e is always positive
        val = np.arccos(pre*min(1.0, abs((r0*(1 - e**2)/state[1] - 1)/(e + 1e-15)))) #add a little tiny bias to get rid of divide by zero errors
        if state[5] < 0:
            val = 2*np.pi - val
        return val
    
    if np.shape(cons) == (3,):
        initE, initLz, initC = cons
        initQ = initC + (a*initE - initLz)**2
    else:
        initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
        initLz = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
        initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
        initC = initQ - (a*initE - initLz)**2                                          #initial adjusted Carter constant 
    pot_min = viable_cons([initE, initLz, initC], all_states[0], a)
    count = 0
    while pot_min < 0.0:
        count += 1
        initE += err_target
        pot_min = viable_cons([initE, initLz, initC], all_states[0], a)
        if count >= 21:
            print("Don't trust this!", inputs)
            break
                
    coeff = np.array([initE**2 - 1, 2.0, (a**2)*(initE**2 - 1) - initLz**2 - initC, 2*((a*initE - initLz)**2) + 2*initC, -initC*(a**2)])
    coeff2 = np.polyder(coeff)
    keps = np.array([np.sort(np.roots(coeff2))[-1], *np.sort(np.real(np.roots(coeff)))[-2:]])
    pot_min, inner_turn, outer_turn = keps.real[abs(keps.imag)<(1e-6)*abs(keps[0])]
    e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
    A = (a**2)*(1 - initE**2)
    z2 = ((A + initLz**2 + initC) - ((A + initLz**2 + initC)**2 - 4*A*initC)**(1/2))/(2*A) if A != 0 else initC/(initLz**2 + initC)
    inc = np.arccos(np.sqrt(z2))
    tracker = [[pot_min, e, inc, inner_turn, outer_turn, all_states[0][0], 0]]
    if True in np.iscomplex(tracker[0]):
        initE = (4*a*initLz*pot_min + ((4*a*initLz*pot_min)**2 - 4*(pot_min**4 + 2*pot_min*(a**2))*((a*initLz)**2 - (pot_min**2 - 2*pot_min + a**2)*(pot_min**2 + initLz**2 + initC)))**(0.5))/(2*(pot_min**4 + 2*pot_min*(a**2)))
        coeff = np.array([initE**2 - 1, 2.0, (a**2)*(initE**2 - 1) - initLz**2 - initC, 2*((a*initE - initLz)**2) + 2*initC, -initC*(a**2)])
        coeff2 = np.polyder(coeff)
        keps = np.array([np.sort(np.roots(coeff2))[-1], *np.sort(np.real(np.roots(coeff)))[-2:]])
        pot_min, inner_turn, outer_turn = keps.real[abs(keps.imag)<(1e-6)*abs(keps[0])]
        e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
        A = (a**2)*(1 - initE**2)
        z2 = ((A + initLz**2 + initC) - ((A + initLz**2 + initC)**2 - 4*A*initC)**(1/2))/(2*A) if A != 0 else initC/(initLz**2 + initC)
        inc = np.arccos(np.sqrt(z2))
        tracker = [[pot_min, e, inc, inner_turn, outer_turn, all_states[0][0], 0]]
    constants = [ np.array([initE,      #energy   
                            initLz,      #angular momentum (axial)
                            initC]) ]    #Carter constant (C)
    qarter = [initQ]           #Carter constant (Q)
    
    false_constants = [np.array([getEnergy(all_states[0], a), *getLs(all_states[0], mu)])]  #Cartesian approximation of L vector
    
    freqs = [mm.freqs_finder(initE, initLz, initC, a)]

    compErr = 0
    milestone = 0
    issues = [(None, None)]
    orbitside = np.sign(all_states[0][1] - pot_min)
    if orbitside == 0:
        orbitside = -1
    
    orbCount = 0
    val = get_true_anom(all_states[0], 0.5*(outer_turn + inner_turn), e)
    true_anom = [val if np.isnan(val) == False else 0.0]
    stop = False
    
    if label == "default":
        label = "r" + str(pot_min) + "e" + str(e) + "zU+03C0" + str(inc/np.pi) + "mu" + str(mu) + "a" + str(a)
    
    #Main Loop
    dTau = np.abs(np.real((inner_turn/200)**(2)))
    dTau_change = [dTau]                                                #create dTau tracker
    borken = 0
    initflagval = eval(termdict[terms[0]])
    plunge, unbind = False, False
    def anglething(angle):
        return 0.5*np.pi - np.abs(angle%np.pi - np.pi/2)
    '''
    if bonk == True:
        print("old")
    else:
        print("new")
    '''
    if verbose == False:
        pbar = tqdm(total = 10000000, position=0)
    progress = 0
    diag_times = []
    while (not(eval(newflag)) and (i < 10**7 or override)):
        try:
            geostart = time.time()
            update = False
            condate = False
            first = True
          
            #Grab the current state
            state = all_states[i]  
            pot_min = tracker[-1][0]   
          
            #Break if you fall inside event horizon
            if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001):
                plunge = True
                break
            
            #break if you get really far away (orbit is unbound)
            if (state[1] >= (1 + np.sqrt(1 - a**2))*1e15):
                unbind = True
                break
          
            #Runge-Kutta update using geodesic
            old_dTau = dTau
            skip = False
            while ((err_calc >= err_target) or (first == True)) and (skip == False):
                new_step = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, a)
                step_check = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, a) 
                if bonk == 0:
                    #preferred for long time? jeremy thing
                    delt = new_step - step_check
                    mod_r = np.array([*new_step[1:3], *new_step[4:]])
                    err_calc = np.sqrt(np.dot(delt, delt)/np.dot(mod_r, mod_r))
                    #angle = np.random.rand()*np.pi #some random angle between 0 and pi radians
                    #new_step_morph, step_check_morph = new_rot(new_step, angle), new_rot(step_check, angle)   
                    #err_calc = abs(1 - np.dot(new_step_morph, step_check_morph)/np.dot(new_step_morph, new_step_morph))
                elif bonk == 1:
                    err_calc = abs(1 - np.dot(new_step, step_check)/np.dot(new_step, new_step))
                elif bonk == 2:
                    #Halfway thing between original (bonk1) and jeremy (bonk2)
                    #Actually it's definitely closer to the original than jeremy
                    #but jeremy's takes forever for whatever reason?
                    err_calc = abs(1 - np.sqrt(np.dot(new_step[1:], step_check[1:])/np.dot(new_step[1:], new_step[1:])))
                elif bonk == 3:
                    #jeremy with mods
                    mod_new = np.array([*new_step[1:3], *new_step[4:]])
                    mod_check = np.array([*step_check[1:3], *step_check[4:]])
                    delt = mod_new - mod_check
                    mod_r = np.array([*new_step[1:3], *new_step[4:]])
                    err_calc = np.sqrt(np.dot(delt, delt)/np.dot(mod_r, mod_r))
                elif bonk == 4:
                    #my thing with long time mods? and a tweak
                    mod_new = np.array([*new_step[1:3], *new_step[4:]])
                    mod_check = np.array([*step_check[1:3], *step_check[4:]])
                    err_calc = abs(1 - np.sqrt(np.dot(mod_new, mod_check)/np.dot(mod_new, mod_new)))
                elif bonk == 5:
                    #try a new thing
                    r, thet = state[1], state[2]
                    opp = (new_step - step_check)*np.array([1, 1, r, r*np.sin(thet), dTau, dTau, r*dTau, r*np.sin(thet)*dTau])
                    hyp = (new_step - state)*np.array([1, 1, r, r*np.sin(thet), dTau, dTau, r*dTau, r*np.sin(thet)*dTau])
                    err_calc = 100*abs(np.arcsin(np.linalg.norm(opp)/np.linalg.norm(hyp)) - np.linalg.norm(opp)/np.linalg.norm(hyp))/np.linalg.norm(opp)/np.linalg.norm(hyp)
                elif bonk == 6:
                    #preferred for long time? jeremy thing carted??
                    delt = bl2cart_oof(new_step, a) - bl2cart_oof(step_check, a)
                    garp = bl2cart_oof(new_step, a)
                    mod_r = np.array([*garp[1:3], *garp[4:]])
                    err_calc = np.sqrt(np.dot(delt, delt)/np.dot(mod_r, mod_r))
        
                E, L, C = constants[-1]
                # if (high inclination) AND ((very close to pole AND approaching pole) OR (dTau is very small AND dTau is monotonically non-increasing))
                if np.sign(new_step[6])*(np.pi/2 - new_step[2]%np.pi) <= -1.55 and np.mean(dTau_change[-10:]) <= 0.001*np.mean(dTau_change):
                    new_step[0] += ((new_step[0] - state[0])/abs(new_step[2] - state[2]))*(2*anglething(new_step[2]))
                    new_step[3] += 2*np.arccos(np.sin(abs(np.pi/2 - np.arccos(L/np.sqrt(L**2 + C))))/ np.sin(new_step[2]))
                    new_step[6] = -new_step[6]
                    break
                
                speed = np.sqrt(new_step[5]**2 + (new_step[1]**2)*(new_step[6]**2 + (np.sin(new_step[2])*new_step[7])**2))
                old_dTau, dTau = dTau, min(dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
                #old_dTau, dTau = dTau, min(dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2/speed)
                if dTau <= 0.0:
                    err_calc = 1
                    dTau = old_dTau
                if new_step[0] - state[0] < 0:
                    err_calc = 1
                    dTau = 10*abs(old_dTau)
                if new_step[0] - state[0] > 100 and new_step[0] - state[0] < 100:
                    print("what the hell??")
                    print(dTau)
                    print(state, mm.check_interval(mm.kerr, state, a))
                    print(new_step, mm.check_interval(mm.kerr, new_step, a))
                    print(step_check, mm.check_interval(mm.kerr, step_check, a))
                    print(old_dTau, dTau)
                    print(err_calc)
                    oof = input("Type x to try unfucking this: ")
                    if "x" in oof:
                        err_calc = 1
                first = False
            #if np.nan in new_step:
            #    print("HEY")
            metric = mm.kerr(new_step, a)[0]
            test = mm.check_interval(mm.kerr, new_step, a)
            looper = 0
            while (abs(test+1)>(err_target) or new_step[4] < 0.0) and looper < 10:
                borken = borken + 1
                og_new_step = np.copy(new_step)
                if bonk2 == True:
                    gtt, gtp = metric[0,0], metric[0,3]
                    disc = 4*(gtp*new_step[4]*new_step[7])**2 - 4*gtt*(new_step[4]**2)*(np.einsum('ij, i, j ->', metric[1:,1:], new_step[5:], new_step[5:]) + 1)
                    delt = (-2*gtp*new_step[4]*new_step[7] - np.sqrt(disc))/(2*gtt*new_step[4]*new_step[4])
                    new_step[4] *= delt
                else:
                    new_step = mm.recalc_state(constants[-1], new_step, a)
                test = mm.check_interval(mm.kerr, new_step, a)
                looper += 1
            if (test+1) > err_target or new_step[4] < 0.0:
                new_step = np.copy(og_new_step)

            geostart = time.time() - geostart
            constart = time.time()
   
            #constant modifying section
            #Whenever you pass from one side of pot_min to the other, mess with the effective potential.
            #if ( np.sign(new_step[1] - pot_min) != orbitside) or ((new_step[3] - all_states[tracker[-1][-1]][3] > np.pi*(3/2)) and (np.std([state[1] for state in all_states[tracker[-1][-1]:]]) < 0.01*np.mean([state[1] for state in all_states[tracker[-1][-1]:]]))):
            R0, ECC = 0.5*(inner_turn + outer_turn), (outer_turn - inner_turn)/(outer_turn + inner_turn)
            compl, comph = np.arccos(-ECC), 2*np.pi - np.arccos(-ECC)
            S1, S2 = get_true_anom(state, R0, ECC), get_true_anom(new_step, R0, ECC)
            #if ((S2-compl) > 0 and (compl-S1) > 0) or ((S2-comph) > 0 and (comph-S1) > 0):   #cross the r0 on both sides
            cond = [((S2-compl) > 0 and (compl-S1) > 0),                                         #outgoing r0
                    ((S2-compl) > 0 and (compl-S1) > 0) or ((S2-comph) > 0 and (comph-S1) > 0),  #both r0s
                    ((S2-comph) > 0 and (comph-S1) > 0),                                         #ingoing r0
                    (S1 > np.pi and S2 < np.pi),                                                 #at r_min
                    (S1 < np.pi and S2 > np.pi),                                                 #at r_max
                    (S1 > np.pi and S2 < np.pi) or (S1 < np.pi and S2 > np.pi),                  #at extrema
                    ((S2-np.pi/2) > 0 and (np.pi/2-S1) > 0),                                     #outgoing p
                    ((S2-np.pi/2) > 0 and (np.pi/2-S1) > 0) or ((S2-1.5*np.pi) > 0 and (1.5*np.pi-S1) > 0),  #both ps
                    ((S2-1.5*np.pi) > 0 and (1.5*np.pi-S1) > 0),                                 #ingoing p
                    ((S2-comph) > 0 and (comph-S1) > 0) and (new_step[3] - all_states[tracker[-1][-1]][3] >= 6*np.pi)]
            smooth = np.all(np.diff(true_anom[tracker[-1][-1]:]) > 0)
            #if cond[trigger] == True:
            if (smooth and cond[trigger]) or (not smooth and (state[3]%(2*np.pi) < np.pi and new_step[3]%(2*np.pi) > np.pi)):
                if (i - tracker[-1][-1] > 10):
                    #if not smooth:
                        #print("heyy", new_step[1])
                    update = True
                    if ( np.sign(new_step[1] - pot_min) != orbitside):
                        orbitside *= -1
                    if mu != 0.0:
                        condate = True
                        #print(inner_turn, new_step[1], outer_turn)
                        dcons = mm.peters_integrate6(all_states[tracker[-1][-1]:i], a, mu, tracker[-1][-1], i)
                        if "wonk" in label:
                            dcons = mm.peters_integrate6_3(all_states[tracker[-1][-1]:i], a, mu, tracker[-1][-1], i)
                        elif "wink" in label:
                            dcons = mm.peters_integrate6_4(all_states[tracker[-1][-1]:i], a, mu, tracker[-1][-1], i)
                        if conch == 5:
                            new_step, ch_cons = mm.new_recalc_state5(constants[-1], dcons, new_step, a)
                        elif conch == 6:
                            new_step, ch_cons = mm.new_recalc_state6(constants[-1], dcons, new_step, a)#, eps=1e-5)#, eps)#, eps=1e-1)
                        elif conch == 7:
                            new_step, ch_cons = mm.new_recalc_state7(constants[-1], dcons, new_step, a)#, eps=1e-5)#, eps)#, eps=1e-1)
                        elif conch == 8:
                            new_step, ch_cons = mm.new_recalc_state8(constants[-1], dcons, new_step, a)#, eps=1e-5)#, eps)#, eps=1e-1
                        elif conch == 9:
                            new_step, ch_cons = mm.new_recalc_state9a(constants[-1], dcons, new_step, a)#, eps=1e-5)#, eps)#, eps=1e-1)
                        elif conch == 10:
                            new_step, ch_cons = mm.new_recalc_state10(constants[-1], dcons, new_step, a)#, eps=1e-5)#, eps)#, eps=1e-1
                        elif conch == 11:
                            new_step, ch_cons = mm.new_recalc_state11(constants[-1], dcons, new_step, a, mu, all_states[tracker[-1][-1]:i])
                        elif conch == 12:
                            new_step, ch_cons = mm.new_recalc_state12(constants[-1], dcons, new_step, a, mu, all_states[tracker[-1][-1]:i])
                        elif conch == 13:
                            new_step, ch_cons = mm.new_recalc_state13(constants[-1], dcons, new_step, a, mu, all_states[tracker[-1][-1]:i])
                        elif conch == 14:
                            new_step, ch_cons = mm.new_recalc_state14(constants[-1], dcons, new_step, a)
                        elif conch == 15:
                            new_step, ch_cons = mm.new_recalc_state15(constants[-1], dcons, new_step, a)
                        else:
                            new_step, ch_cons = mm.new_recalc_state9(constants[-1], dcons, new_step, a)#, eps=1e-5)#, eps)#, eps=1e-1)
                        pot_min = viable_cons(ch_cons, new_step, a)
                        subcount = 0
                        while pot_min < -err_target:
                            viable_cons(ch_cons, new_step, a, True)
                            print(pot_min, -err_target, "whoops")
                            op.potentplotter(*constants[-1], a)
                            op.potentplotter(*ch_cons, a)
                            raise KeyboardInterrupt
                            if (subcount < 10) or subcount%10000000 == 0:
                                print(dcons, pot_min, "HEWWO??", subcount)
                            Lphi, ro = ch_cons[1], pot_min
                            ch_cons[0] += max(10**(-16), 2*(-pot_min)*((2*ro*((ro**3 + ro*(a**2) + 2*(a**2))*ch_cons[0] - 2*Lphi*a))**(-1)))
                            #ch_cons[0] += 10**(-16)
                            new_step = mm.recalc_state(ch_cons, new_step, a)
                            pot_min = viable_cons(ch_cons, new_step, a)
                            subcount += 1
                        if subcount > 0:
                            print(subcount, "oof", pot_min)
                        

            constart = time.time() - constart
            upstart = time.time()
            #Initializing for the next step
            #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
            #Only happens the step before the derivatives are recalculated.
            
            #Update stuff!
            if (update == True):
                if condate == False:
                    metric = mm.kerr(new_step, a)[0]
                    newE = -np.matmul(new_step[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
                    newLz = np.matmul(new_step[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
                    newQ = np.matmul(np.matmul(mm.kill_tensor(new_step, a), new_step[4:]), new_step[4:])    #new Carter constant Q
                    newC = newQ - (a*newE - newLz)**2                                                             #initial adjusted Carter constant  
                    coeff = np.array([newE**2 - 1, 2.0, (a**2)*(newE**2 - 1) - newLz**2 - newC, 2*((a*newE - newLz)**2 + newC), -newC*(a**2)])
                    coeff2 = np.array([4*(newE**2 - 1), 6.0, 2*((a**2)*(newE**2 - 1) - newLz**2 - newC), 2*((a*newE - newLz)**2 + newC)])
                    pot_min, inner_turn, outer_turn = max(np.roots(coeff2)), *np.sort(np.roots(coeff))[-2:]
                    e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
                    A = (a**2)*(1 - newE**2)
                    z2 = ((A + newLz**2 + newC) - ((A + newLz**2 + newC)**2 - 4*A*newC)**(1/2))/(2*A) if A != 0 else newC/(newLz**2 + newC)
                    inc = np.arccos(np.sqrt(z2))
                    tracker.append([pot_min, e, inc, inner_turn, outer_turn, new_step[0], i])
                    constants.append([newE, newLz, newC])
                    qarter.append(newQ)
                    freqs.append(mm.freqs_finder(newE, newLz, newC, a))
                else:
                    constants.append(ch_cons)
                    qarter.append(ch_cons[2] + (a*ch_cons[0] - ch_cons[1])**2)
                    coeff = np.array([ch_cons[0]**2 - 1, 2.0, (a**2)*(ch_cons[0]**2 - 1) - ch_cons[1]**2 - ch_cons[2], 2*((a*ch_cons[0] - ch_cons[1])**2 + ch_cons[2]), -ch_cons[2]*(a**2)])
                    coeff2 = np.array([4*(ch_cons[0]**2 - 1), 6.0, 2*((a**2)*(ch_cons[0]**2 - 1) - ch_cons[1]**2 - ch_cons[2]), 2*((a*ch_cons[0] - ch_cons[1])**2 + ch_cons[2])])
                    pot_min, inner_turn, outer_turn = max(np.roots(coeff2)), *np.sort(np.roots(coeff))[-2:]
                    inner_turn, outer_turn = np.real(inner_turn), np.real(outer_turn)
                    e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
                    A = (a**2)*(1 - ch_cons[0]**2)
                    z2 = ((A + ch_cons[1]**2 + ch_cons[2]) - ((A + ch_cons[1]**2 + ch_cons[2])**2 - 4*A*ch_cons[2])**(1/2))/(2*A) if A != 0 else ch_cons[2]/(ch_cons[1]**2 + ch_cons[2])
                    inc = np.arccos(np.sqrt(z2))
                    tracker.append([pot_min, e, inc, inner_turn, outer_turn, new_step[0], i])
                    freqs.append(mm.freqs_finder(*ch_cons, a))
                if True in np.iscomplex(tracker[-1]):
                    compErr += 1
                    issues.append((i, new_step[0]))  
            #print("not stuck!")
            interval.append(mm.check_interval(mm.kerr, new_step, a))
            false_constants.append([getEnergy(new_step, a), *getLs(new_step, mu)])
            dTau_change.append(old_dTau)
            all_states.append(new_step )    #update position and velocity
            anomval = get_true_anom(new_step, 0.5*(outer_turn + inner_turn), e) + orbCount*2*np.pi
            if anomval < true_anom[-1]:
                anomval += 2*np.pi
                orbCount += 1
            true_anom.append(anomval)
            i += 1
            if verbose == True:
                progress = max( abs((eval(termdict[terms[0]]) - initflagval)/(eval(terms[2]) - initflagval)), i/(10**7)) * 100
                if (progress >= milestone):
                    print("Program has completed " + str(round(eval(termdict[terms[0]]), 2)), ",", str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
                    milestone = int(progress) + 1
            else:
                val = max( (10**7)*abs((eval(termdict[terms[0]]) - initflagval)/(eval(terms[2]) - initflagval)), i) - progress
                if val > 0:
                    pbar.update(val)
                    progress = max( (10**7)*abs((eval(termdict[terms[0]]) - initflagval)/(eval(terms[2]) - initflagval)), i)
            #print("maybe even finished?")
            upstart = time.time() - upstart
            diag_times.append([geostart, constart, upstart])
        #Lets you end the program before the established end without breaking anything
        except KeyboardInterrupt:
            print("\nEnding program")
            stop = True
            cap = len(all_states) - 1
            all_states = all_states[:cap]
            interval = interval[:cap]
            dTau_change = dTau_change[:cap]
            constants = constants[:cap]
            qarter = qarter[:cap]
            freqs = freqs[:cap]
            break
        '''
        except Exception as e:
            print("\nEnding program - ERROR")
            print(type(e), e)
            stop = True
            cap = len(all_states) - 1
            all_states = all_states[:cap]
            interval = interval[:cap]
            dTau_change = dTau_change[:cap]
            constants = constants[:cap]
            qarter = qarter[:cap]
            freqs = freqs[:cap]
            break
        '''
    if verbose == False:
        pbar.close()
    #print(len(issues), len(all_states))
    #unit conversion stuff
    if units == "mks":
        G, c = 6.67*(10**-11), 3*(10**8)
    elif units == "cgs":
        G, c = 6.67*(10**-8),  3*(10**10)
    else:
        G, mass, c = 1.0, 1.0, 1.0
        
    if mu == 0.0:
        #so it gives actual numbers for pure geodesics
        mu = 1.0
        
    constants = np.array([entry*np.array([mass*(c**2), mass*mass*G/c, (mass*mass*G/c)**2]) for entry in np.array(constants)], dtype=np.float64)
    false_constants = np.array(false_constants)
    qarter = np.array(qarter)
    freqs = np.array(freqs)*(c**3)/(G*mass)
    interval = np.array(interval)
    dTau_change = np.array([entry * (G*mass)/(c**3) for entry in dTau_change])
    all_states = np.array([entry*np.array([(G*mass)/(c**3), (G*mass)/(c**2), 1.0, 1.0, 1.0, c, (c**3)/(G*mass), (c**3)/(G*mass)]) for entry in np.array(all_states)]) 
    tracker = np.array([entry*np.array([(G*mass)/(c**2), 1.0, 1.0, (G*mass)/(c**2), (G*mass)/(c**2), (G*mass)/(c**3), 1]) for entry in tracker])
    ind = argrelmin(all_states[:,1])[0]
    omega, otime = all_states[ind,3] - 2*np.pi*np.arange(len(ind)), all_states[ind,0]
    asc_node, asc_node_time = np.array([]), np.array([])
    des_node, des_node_time = np.array([]), np.array([])
    true_anom = np.array(true_anom)
    diag_times = np.array(diag_times)
    if max(all_states[:,2]) - min(all_states[:,2]) > 1e-15:
        theta_derv = np.interp(all_states[:,0], 0.5*(all_states[:,0][:-1] + all_states[:,0][1:]), np.diff(all_states[:,2])/np.diff(all_states[:,0]))
        ind2 = argrelmin(theta_derv)[0] #indices for the ascending node
        ind3 = argrelmin(-theta_derv)[0] #indices for the descending node
        asc_node, asc_node_time = all_states[ind2,3] - 2*np.pi*np.arange(len(ind2)), all_states[ind2,0] #subtract the normal phi advancement
        des_node, des_node_time = all_states[ind3,3] - 2*np.pi*np.arange(len(ind3)), all_states[ind3,0] #subtract the normal phi advancement
        try:
            #if ind2[0] > ind3[0]: #if the ascending node occurs after the descending node
                #ascending node should be first because of how the program starts on default
            #    asc_node = asc_node - np.ones(len(ind2))*2*np.pi #subtract a bit more for when comparing
            if type(asc_node) != np.ndarray:
                asc_node, asc_node_time = np.array([asc_node]), np.array([asc_node_time])
        except:
            pass
    if verbose == True:
        print("There were " + str(compErr) + " issues with complex roots/turning points.")
    final = {"name": label,
             "raw": all_states,
             "inputs": inputs,
             "pos": all_states[:,1:4],
             "all_vel": all_states[:,4:], 
             "time": all_states[:,0],
             "true_anom": true_anom,
             "interval": interval,
             "vel": (np.square(all_states[:,5]) + np.square(all_states[:,1]) * (np.square(all_states[:,6]) + (np.sin(all_states[:,2])**2)*np.square(all_states[:,7])))**(0.5),
             "dTau_change": dTau_change,
             "energy": constants[:, 0],
             "phi_momentum": constants[:, 1],
             "carter": constants[:, 2],
             "qarter":qarter,
             "energy2": false_constants[:, 0],
             "Lx_momentum": false_constants[:, 1],
             "Ly_momentum": false_constants[:, 2],
             "Lz_momentum": false_constants[:, 3],
             "spin": a,
             "freqs": freqs,
             "pot_min": tracker[:,0],
             "e": tracker[:,1],
             "inc": tracker[:,2],
             "it": tracker[:,3],
             "ot": tracker[:,4],
             "r0": 0.5*(tracker[:,3] + tracker[:,4]),
             "p": 0.5*(tracker[:,3] + tracker[:,4])*(1 - tracker[:,1]**2),
             "tracktime": tracker[:,5],
             "trackix": np.array([int(num) for num in tracker[:,6]]),
             "omega": omega,
             "otime": otime,
             "asc_node": asc_node,
             "asc_node_time": asc_node_time,
             "des_node": des_node,
             "des_node_time": des_node_time,
             "stop": stop,
             "plunge": plunge,
             "unbind": unbind,
             "issues": issues,
             "diag_times": diag_times}
    return final

def guessmaker2(cons, old_state, a):
    import scipy.interpolate as interp
    state = mm.recalc_state(cons, old_state[:4], a)
    #state[5] *= -1
    freqs = mm.freqs_finder(*cons, a)
    turns, flats, zs = mm.root_getter(*cons, a)
    #print(turns)
    vals = np.sort(np.arccos(zs))
    #print(vals)
    p, e, inc = 2*turns[-1]*turns[-2]/(turns[-1] + turns[-2]), (turns[-1] - turns[-2])/(turns[-1] + turns[-2]), vals[1]
    r_min, r_max, r0 = p/(1+e), p/(1-e), p/(1-e**2)
    #print(p,e,inc)
    
    ano1 = np.real(np.arccos((1/e)*(p/state[1] - 1))*180/np.pi)
    thetano1 = np.real(np.arcsin((2*state[2] - np.pi)/(2*inc - np.pi))) if inc != np.pi/2 else 0.0
    ano1 = 360 - ano1 if state[5] < 0 else ano1
    thetano1 = thetano1%(2*np.pi) if state[6] < 0 else (thetano1 + np.pi)%(2*np.pi)
    theta_cor = not(state[2] == np.pi/2 and np.abs(state[6]) < 1e-15)
    print(theta_cor)

    if state[1] > p or (state[1] == p and state[5] >= 0):                 #outer orbit
        anomaly = np.linspace(ano1, 270, int(270 - ano1) + 1)*np.pi/180   #true anomaly
        rads = p/(1 + e*np.cos(anomaly))                                  #radial position
        thetanoms = thetano1 + anomaly - anomaly[0]                       #theta anomaly
        thets = np.pi/2 - (inc - np.pi/2)*np.sin(thetanoms)*theta_cor               #theta position
        uph = (((1 + e*np.cos(anomaly))/(1 - e**2))**2)/(r0**(3/2) + a)
        #kep2con = 0.5*(state[1]**2)*np.sqrt(state[7]**2 + state[6]**2)    #approx "area per unit time" (keplers 2nd law)
        ut = (1 - 2*rads/(rads**2 + (a**2)*(np.cos(thets))))**(-1)
        #times = (np.cumsum(ut) - ut[0])*np.pi*(((p/(1-e))**(3/2)))/np.sum(ut) + state[0]

    elif state[1] < p or (state[1] == p and state[5] < 0):                #inner orbit (bound)
        #print("this?")
        print(inc)
        anomaly = np.linspace(ano1, 450, int(450 - ano1) + 1)*np.pi/180
        rads = p/(1 + e*np.cos(anomaly))
        thetanoms = thetano1 + anomaly - anomaly[0]                       #theta anomaly
        #print(inc, np.pi/2 - inc)
        thets = np.pi/2 - (inc - np.pi/2)*np.sin(thetanoms)*theta_cor               #theta position
        uph = (((1 + e*np.cos(anomaly))/(1 - e**2))**2)/(r0**(3/2) + a)    
        #approx ratio between frequencies: r0 circular orbit/r_min circular orbit
        ut = (1 - 2*rads/(rads**2 + (a**2)*(np.cos(thets))))**(-1)
        #times = (np.cumsum(ut) - ut[0])*np.pi*(((p/(1+e))**(3/2)))/np.sum(ut) + state[0]
        
    hold = np.arange(len(rads))
    
    #np.cumsum(ut - ut[0])*(np.pi*((p)**(3/2)))/np.sum(ut) + state[0]
    #print((np.pi*((p)**(3/2))))
    #print(np.sum(ut))
    #print((np.cumsum(ut) - ut[0])[-1])
    ut *= state[4]/ut[0]
    uth = interp.CubicSpline(hold, thets)(hold, 1)
    #uph = interp.CubicSpline(hold, phis)(hold, 1)
    times = np.append([0], np.cumsum(ut)[:-1])*np.pi*(((p/(1+e))**(3/2)))/np.sum(ut[:-1]) + state[0]
    print(np.imag(times))
    phis = np.cumsum(np.append(uph[:-1]*np.diff(times), uph[-1]*np.diff(times)[-1])) + state[3]
    ur = interp.CubicSpline(hold, rads)(hold, 1)/(np.diff(times)/np.diff(hold))[0]
    #print(uth)
    #ut *= state[4]/ut[0] if not(np.isinf(1/ut[0])) else 
    #ur *= state[5]/ur[0]
    #uth *= state[6]/uth[0]
    #uph *= state[7]/uph[0] 
    #plt.plot(gtt)
    #plt.plot(ut)
    return np.real(np.transpose([times, rads, thets, phis, ut, ur, uth, uph]))

def corrector(cons, guess, a): #doesn't work
    new = guess.copy()
    def dervs(cons, state, a):
        E, L, C = cons
        r, T = state[1], state[2]
        sint, cost = np.sin(T), np.cos(T)
        sig, delt = r**2 + (a*cost)**2, r**2 - 2*r + a**2
        u0, u2, u3 = state[4], state[6], state[7]
        dEdr = -2*(a*(sint**2)*u3 - u0)*(r**2 - (a*cost)**2)/(sig**2)
        dEdT = 4*a*r*((r**2 + a**2)*u3 - a*u0)*cost*sint/(sig**2)
        dLdr = (2*(sint**2)/(sig**2))*(a*(r**2 - (a*cost)**2)*u0 + (r**5 + 2*(a**2)*(r**3) - (a*r*sint)**2 + r*((a*cost)**4) + (a**4)*((sint*cost)**2))*u3)
        dLdT = (2*sint*cost/(sig**2))*(-2*a*r*(r**2 + a**2)*u0 + (delt*((a*sint)**2)*((a*sint)**2 - 2*(r**2 + a**2)) + (r**2 + a**2)**3)*u3)
        dQdr = (2/sint**2)*(dLdr - a*dEdr*(sint**2))*(L - a*E*(sint**2)) + 4*r*sig*(u2**2)
        dQdT = (1/(sint**3))*(2*(dLdT - a*(dEdT*(sint**2) + 2*E*sint*cost))*(L - a*E*(sint**2))*sint - 2*cost*((L - a*E*(sint**2))**2)) - 2*(a**2)*sint*cost - 4*(a**2)*sint*cost*sig*(u2**2)
        dCdr = dQdr - 2*(a*E - L)*(a*dEdr - dLdr)
        dCdT = dQdT - 2*(a*E - L)*(a*dEdT - dLdT)
        return np.array([dEdr, dLdr, dCdr, dEdT, dLdT, dCdT])
        
    def intderv(state, a):
        #print(state)
        r, T = state[1], state[2]
        sint, cost = np.sin(T), np.cos(T)
        sig, delt = r**2 + (a*cost)**2, r**2 - 2*r + a**2
        u0, u1, u2, u3 = state[4:]
        #print(r, T, sig, u0, a, sint, sig, u3)
        du0 = -2*(1 - 2*r/sig)*u0 - (4*a*r*(sint**2)/sig)*u3
        du1 = 2*(sig/delt)*u1
        du2 = 2*sig*u2
        du3 = 2*(r**2 + a**2 + 2*r*((a*sint)**2)/sig)*(sint**2)*u3
        return np.array([du0, du1, du2, du3])
    #print(new[:2])
    dcons = np.array([np.array(cons) - getCons(state, a) for state in new])
    print(dcons[0:2])
    print(input("hhhe"))
    dervs = np.array([dervs(cons, state, a) for state in new])
    print(dervs[:2])
    dervs = np.where(np.isinf(1/dervs), 0.0, 1/dervs)
    #return dervs[0]

    print(dervs[:2])
    print(input("hhhe"))
    delt_r, delt_T = np.sum(dcons*dervs[:,:3], axis=1), np.sum(dcons*dervs[:,3:], axis=1)
    print(dervs[0,:3]*dcons[0])
    print(dervs[0,3:]*dcons[0])
    print(input("check this"))
    
    print(delt_r[:2])
    print(delt_T[:2])
    #print(delt_T[:2]%(
    
    new[:, 1] += delt_r
    new[:, 2] += delt_T
    new[:, 2] = np.arccos(np.cos(new[:, 2]))
    int_diffs = -1 - np.array([mm.check_interval(mm.kerr, i, a) for i in new])
    int_dervs = np.array([intderv(state, a) for state in new])
    delt_vel = np.array([(-1 - mm.check_interval(mm.kerr, state, a))/intderv(state, a) for state in new])
    #new[:, 4:] += delt_vel
    return new

def corrector2(cons, guess, a):
    dcons = np.array([cons - getCons(state) for state in guess])
    #rho, z
    def dervs(state, a):
        E, L, C = getCons(state, a)
        r, T = state[1], state[2]
        sint, cost = np.sin(T), np.cos(T)
        sig, delt = r**2 + (a*cost)**2, r**2 - 2*r + a**2
        u0, u2, u3 = state[4], state[6], state[7]
        dEdr = -2*(a*(sint**2)*u3 - u0)*(r**2 - (a*cost)**2)/(sig**2)
        dEdT = 4*a*r*((r**2 + a**2)*u3 - a*u0)*cost*sint/(sig**2)
        dLdr = (2*(sint**2)/(sig**2))*(a*(r**2 - (a*cost)**2)*u0 + (r**5 + 2*(a**2)*(r**3) - (a*r*sint)**2 + r*((a*cost)**4) + (a**4)*((sint*cost)**2))*u3)
        dLdT = (2*sint*cost/(sig**2))*(-2*a*r*(r**2 + a**2)*u0 + (delt*((a*sint)**2)*((a*sint)**2 - 2*(r**2 + a**2)) + (r**2 + a**2)**3)*u3)
        dQdr = (2/sint**2)*(dLdr - a*dEdr*(sint**2))*(L - a*E*(sint**2)) + 4*r*sig*(u2**2)
        dQdT = (1/(sint**3))*(2*(dLdT - a*(dEdT*(sint**2) + 2*E*sint*cost))*(L - a*E*(sint**2))*sint - 2*cost*((L - a*E*(sint**2))**2)) - 2*(a**2)*sint*cost - 4*(a**2)*sint*cost*sig*(u2**2)
        dCdr = dQdr - 2*(a*E - L)*(a*dEdr - dLdr)
        dCdT = dQdT - 2*(a*E - L)*(a*dEdT - dLdT)
        return np.array([[dEdr, dEdT],
                         [dLdr, dLdT],
                         [dCdr, dCdT]])
        def dervs(state, a0):
            E0, L0, C0 = getCons(state, a0)
            r, T, a, u0, u2, u3 = sp.symbols('r T a u0 u2 u3', real=True)
            sig = r**2 + (a*sp.cos(T))**2
            delt = r**2 - 2*r + a**2
            E = (1 - 2*r/sig)*u0 + 2*a*r*(sp.sin(T)**2)*u3/sig
            L = -2*a*r*(sp.sin(T)**2)*u0/sig + ((r**2 + a**2)**2 - delt*((a*sp.sin(T))**2))*(sp.sin(T)**2)*u3/sig
            Q = ((L - a*E*(sp.sin(T)**2))**2)/(sp.sin(T)**2) + (a*sp.cos(T))**2 + (sig*u2)**2
            C = Q - (a*E - L)**2
            
            dEdr, dLdr, dCdr = sp.diff(E, r), sp.diff(L, r), sp.diff(C, r)
            dEdT, dLdT, dCdT = sp.diff(E, T), sp.diff(L, T), sp.diff(C, T)
            d2Edr2, d2Ldr2, d2Cdr2 = sp.diff(dEdr, r), sp.diff(dLdr, r), sp.diff(dCdr, r)
            d2EdT2, d2LdT2, d2CdT2 = sp.diff(dEdT, T), sp.diff(dLdT, T), sp.diff(dCdT, T)
            d2EdrdT, d2LdrdT, d2CdrdT = sp.diff(dEdr, T), sp.diff(dLdr, T), sp.diff(dCdr, T)
            
            #d1_block = np.array([

            dEdr = -2*(a*(sint**2)*u3 - u0)*(r**2 - (a*cost)**2)/(sig**2)
            dEdT = 4*a*r*((r**2 + a**2)*u3 - a*u0)*cost*sint/(sig**2)
            dLdr = (2*(sint**2)/(sig**2))*(a*(r**2 - (a*cost)**2)*u0 + (r**5 + 2*(a**2)*(r**3) - (a*r*sint)**2 + r*((a*cost)**4) + (a**4)*((sint*cost)**2))*u3)
            dLdT = (2*sint*cost/(sig**2))*(-2*a*r*(r**2 + a**2)*u0 + (delt*((a*sint)**2)*((a*sint)**2 - 2*(r**2 + a**2)) + (r**2 + a**2)**3)*u3)
            dQdr = (2/sint**2)*(dLdr - a*dEdr*(sint**2))*(L - a*E*(sint**2)) + 4*r*sig*(u2**2)
            dQdT = (1/(sint**3))*(2*(dLdT - a*(dEdT*(sint**2) + 2*E*sint*cost))*(L - a*E*(sint**2))*sint - 2*cost*((L - a*E*(sint**2))**2)) - 2*(a**2)*sint*cost - 4*(a**2)*sint*cost*sig*(u2**2)
            dCdr = dQdr - 2*(a*E - L)*(a*dEdr - dLdr)
            dCdT = dQdT - 2*(a*E - L)*(a*dEdT - dLdT)
            return np.array([[dEdr, dEdT],
                             [dLdr, dLdT],
                             [dCdr, dCdT]])
    Amat = np.array([dervs(cons, state, a) for state in guess])
    Atrans = np.transpose(Amat, axes=(0,2,1))
    Dblock = np.einsum("ijk, ikl -> ijl", Atrans, Amat)
    Amat = np.transpose(np.append([dcons], [dcons], axis=0), axes=(1,2,0))/1e-7
    #this looks weird? Shouldn't Amat bet the derivatives?

def clean_continue(data, endflag = False, verbose=False):
    #continue from the last crossing, I think?
    verbose_new = verbose
    mass, a, mu, endflag_old, err_target, label_old, cons, velorient, vel4, params, pos, units = data["inputs"]
    if endflag == False:
        endflag = endflag_old
    #inputs = [mass, a, mu, endflag, err_target, label, cons, velorient, vel4, params, pos, units]          #Grab initial input in case you want to run the continue function
    lastcross = int(data["trackix"][-1])
    if units == "mks":
        G, c = 6.67*(10**-11), 3*(10**8)
    elif units == "cgs":
        G, c = 6.67*(10**-8),  3*(10**10)
    else:
        G, mass, c = 1.0, 1.0, 1.0
    
    newstart = data["raw"][lastcross]/np.array([(G*mass)/(c**3), (G*mass)/(c**2), 1.0, 1.0, 1.0, c, (c**3)/(G*mass), (c**3)/(G*mass)])
    pos_new, vel_new = newstart[:4], newstart[4:]
    newdata = EMRIGenerator(a, mu, endflag, mass, err_target, label=label_old, pos=pos_new, veltrue=vel_new, verbose=verbose_new)
    
    final = {"name": label_old,
             "raw": np.concatenate((data["raw"][:lastcross], newdata["raw"])),
             "inputs": data["inputs"],
             "pos": np.concatenate((data["pos"][:lastcross], newdata["pos"])),
             "all_vel": np.concatenate((data["all_vel"][:lastcross], newdata["all_vel"])), 
             "time": np.concatenate((data["time"][:lastcross], newdata["time"])),
             "interval": np.concatenate((data["interval"][:lastcross], newdata["interval"])),
             "vel": np.concatenate((data["vel"][:lastcross], newdata["vel"])),
             "dTau_change": np.concatenate((data["dTau_change"][:lastcross], newdata["dTau_change"])),
             "energy": np.concatenate((data["energy"], newdata["energy"])),
             "phi_momentum": np.concatenate((data["phi_momentum"], newdata["phi_momentum"])),
             "carter": np.concatenate((data["carter"], newdata["carter"])),
             "qarter": np.concatenate((data["qarter"], newdata["qarter"])),
             "energy2": np.concatenate((data["energy2"], newdata["energy2"])),
             "Lx_momentum": np.concatenate((data["Lx_momentum"], newdata["Lx_momentum"])),
             "Ly_momentum": np.concatenate((data["Ly_momentum"], newdata["Ly_momentum"])),
             "Lz_momentum": np.concatenate((data["Lz_momentum"], newdata["Lz_momentum"])),
             "spin": a,
             "freqs": np.concatenate((data["freqs"], newdata["freqs"])),
             "pot_min":np.concatenate((data["pot_min"], newdata["pot_min"])),
             "e": np.concatenate((data["e"], newdata["e"])),
             "inc": np.concatenate((data["inc"], newdata["inc"])),
             "it": np.concatenate((data["it"], newdata["it"])),
             "ot": np.concatenate((data["ot"], newdata["ot"])),
             "r0": np.concatenate((data["r0"], newdata["r0"])),
             "tracktime": np.concatenate((data["tracktime"], newdata["tracktime"])),
             "trackix": np.concatenate((data["trackix"], newdata["trackix"])),
             "omega": np.concatenate((data["omega"][:lastcross], newdata["omega"] - 2*np.pi*len(data["omega"][:lastcross]))),
             "otime": np.concatenate((data["otime"][:lastcross], newdata["otime"])),
             "asc_node": np.concatenate((data["asc_node"][:lastcross], newdata["asc_node"] - 2*np.pi*len(data["asc_node"][:lastcross]))),
             "asc_node_time": np.concatenate((data["asc_node_time"][:lastcross], newdata["asc_node_time"])),
             "stop": newdata["stop"],
             "plunge": newdata["plunge"],
             "issues": np.concatenate((data["issues"], newdata["issues"]))}
    return final

def dict_saver(data, filename):
    np.save(filename, data) 
    return True
        
def dict_from_file(filename):
    if ".npy" not in filename:
        filename = filename+".npy"
    data = np.load(filename, allow_pickle='TRUE').item()
    return data
