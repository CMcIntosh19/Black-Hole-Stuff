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
import traceback
import matplotlib.pyplot as plt
import warnings

@njit
def getCons(state, a):
    metric, _ = mm.kerr_2(state, a)  # assuming this returns (metric, chris), and we only need metric
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
    
    endvars = {"time": lambda c: c.all_states[c.i][0],
               "elapsed_time": lambda c: abs(c.all_states[c.i][0] - c.all_states[0][0]),
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

    time_check = (-1)**time_reverse
    inputs = [mass, a, mu, endflag, err_target, label, cons, velorient, vel4, params, pos, veltrue, units]          #Grab initial input in case you want to run the continue function
    inputs = [entry.tolist() if type(entry) == np.ndarray else entry for entry in inputs]                           #Convert numpy arrays to lists so that JSON doesn't complain
    all_states = [[np.zeros(8)]]                                                  #Grab that initial state         
    err_calc = err_target*1.01
    i = 0                                                                         #initialize step counter
    if (np.shape(veltrue) == (4,)) and (np.shape(pos) == (4,)):
        all_states[0] = np.array([*pos, *veltrue])
    else:
        if verbose:
            print("Normalizing initial state")
        all_states[0], cons = mm.set_u_kerr(a, cons, velorient, vel4, params, pos)      #normalize initial state so it's actually physical
    
    metric, chris = mm.kerr_2(all_states[0], a)                                     #initial metric and christoffel symbols
    interval = [mm.check_interval_w_metric(metric, all_states[0], a)]           #create interval tracker
    
    def viable_cons(new_cons, old_cons, state, a,
                    scream=False,
                    rtol=err_target):
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
        def R(E, L, C, a):
            return np.array([ E**2 - 1, 2, (a**2)*(E**2 - 1) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])

        # Get roots and extrema
        R2 = R(E2, L2, C2, a)
        turns, flats = np.sort(np.roots(R2)), np.sort(np.roots(np.polyder(R2)))

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
        Rvals = np.polyval(R(E2, L2, C2, a), valid_flats)
        Rmin = np.min(Rvals)

        # Optional diagnostic plot
        if scream:
            import matplotlib.pyplot as plt
            r_plot = np.linspace(r_peri*0.98, r_apo*1.02, 400)
            plt.figure()
            plt.axhline(0, color='k', lw=0.5)
            plt.plot(r_plot, np.polyval(R(E2, L2, C2, a), r_plot))
            plt.scatter(valid_flats, Rvals, color='red')
            plt.scatter(r0, np.polyval(R(E2, L2, C2, a), r0), color='blue')
            plt.title("Radial potential viability check")
            plt.xlabel("r")
            plt.ylabel("R(r)")
            plt.show()

        return Rmin

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
        initE, initLz, initC = getCons(all_states[0], a)
        initQ = initC + (a*initE - initLz)**2
    pot_min = viable_cons(np.array([initE, initLz, initC]), np.array([initE, initLz, initC]), all_states[0], a)
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
    inc = np.sign(initLz)*np.arccos(min(1.0, np.sqrt(z2)))
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
    issues = []
    orbitside = np.sign(all_states[0][1] - pot_min)
    if orbitside == 0:
        orbitside = -1
    
    orbCount = 0
    val = get_true_anom(all_states[0], 0.5*(outer_turn + inner_turn), e)
    P0, ECC = 2*(inner_turn*outer_turn)/(inner_turn + outer_turn), (outer_turn - inner_turn)/(outer_turn + inner_turn)
    POT_MIN = max(flats)
    PERIOD = 2*np.pi / ((1/(POT_MIN**1.5 + a)) * np.sqrt(1 - 6/POT_MIN + 8*a/(POT_MIN**1.5) - 3*a*a/(POT_MIN**2)))
    true_anom = [val if np.isnan(val) == False else 0.0]
    stop = False
    radial_fail = 0
    
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
    start_val, ctx_val = getval(ctx), getval(ctx)
    trigger = False
    wonky_test = 0
    while not(ctx_val and op(ctx_val, threshold)) and (i < 1e7 or override):
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
            rkcount = time.perf_counter()
            counter = 0

            while (err_calc >= err_target) or (first == True):
                #print("yoooo?????")
                counter += 1
                if counter%20 == 0:
                    print(f"A lot! {counter} {err_calc}")
                if counter > 30:
                    raise KeyboardInterrupt
                # Generate 4th and 5th order calculations for next step
                new_step = mm.gen_RK2(*mm.ck4_2, mm.kerr_2, state, dTau, a)
                step_check = mm.gen_RK2(*mm.ck5_2, mm.kerr_2, state, dTau, a) 

                # Calculate the error
                delt = new_step - step_check
                err_calc = np.concatenate((delt[1:3], delt[4:]))
                err_calc = np.linalg.norm(err_calc/6)

                # Correct for pole effects
                # if ((within ~0.5 degrees of pole) and (moving closer to pole)) and (average of last few dTau_change values is much smaller than the average):
                if ((np.sin(new_step[2]) <= 0.009) and np.sign(new_step[6]*np.cos(new_step[2])) < 0) and (np.mean(np.diff(dTau_change[-10:])) <= 0.001*np.mean(np.diff(dTau_change))):
                    #print(new_step)
                    #print("here?", i, counter, new_step[2], new_step[6])
                    old_step = new_step
                    # inc is the minimum value of theta, pretend the particle is traveling on a straight line between
                    # current theta value and inc, then flash to the other side (same theta). Find phi!
                    k = abs(inc)/min((np.pi - new_step[2]), new_step[2])
                    if k == 0.0:
                        # Orbit is polar!
                        phi_dist = np.pi if new_step[7] >= 0.0 else -np.pi
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
                    t_change = 2 * min((np.pi - np.abs(new_step[2]%(2*np.pi) - np.pi)), np.abs(new_step[2]%(2*np.pi) - np.pi)) / abs(new_step[6])
                    new_step[0] += t_change
                    # Scale up dTau so we aren't liars
                    dTau = dTau*(new_step[0] - state[0])/(old_step[0] - state[0])
                    # Flip the sign on theta_dot so now it's moving away from the pole
                    new_step[6] *= -1
                    #print(new_step)
                    #THIS IS STILL FUCKED TEST THIS

                old_dTau, dTau = dTau, np.sign(dTau)*min(abs(dTau) * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)

                if "stupid" in label and (np.mean(dTau_change[-10:]) <= 0.001*np.mean(dTau_change)):
                    dTau *= 10
                if ((-1)**("dumb" in label))*dTau <= 0.0:
                    err_calc = 1
                    print(dTau)
                    print("a womp")
                    dTau = old_dTau
                if ((-1)**("dumb" in label))*(new_step[0] - state[0]) < 0:
                    err_calc = 1
                    print(state)
                    print(new_step)
                    print("--")
                    print("b", dTau, new_step[0] - state[0], i, counter)
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
                        print("c")
                if np.isnan(np.sum(new_step)):
                    print("d", new_step, step_check, err_calc, dTau, state, old_dTau)
                    err_calc = 1
                    dTau = old_dTau*0.9
                if np.isnan(err_calc):
                    #print("you??")
                    dTau = old_dTau * (10**counter)
                    err_calc = 1
                    #dTau = 0.1*np.abs(np.real((inner_turn/200)**(2)))   
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
                    new_step = mm.recalc_state3(constants[j], new_step, a)
                    delt = np.nan
                test = mm.check_interval_w_metric(metric, new_step, a)
                looper += 1
            if (test+1) > err_target or new_step[4] < 0.0:
                new_step = np.copy(og_new_step)
                looper = 0
            if looper > 0:
                issues.append((i, new_step[0], delt))

            rkfull.append(time.perf_counter() - rkcount)

            #constant modifying section

            if i%50 == 0:
                if force_stop is not None:
                    if force_stop():
                        raise KeyboardInterrupt
            
            if trigger == False and (((state[1] > POT_MIN and new_step[1] <= POT_MIN) and new_step[0] - tracker[j][-2] > PERIOD*0.75) or (new_step[0] - tracker[j][-2] > PERIOD*2 and ((new_step[5] - state[5] >= 0 and all_states[i-1][5] - state[5] > 0) and new_step[1] < state[1]))):
                #register the spot where we cross pot_min
                trigger = True
                tracker.append([pot_min, e, inc, inner_turn, outer_turn, new_step[0], i])

            if (trigger == True) and abs(new_step[0] - tracker[-1][-2]) > 0.03*PERIOD:
                #register when we've gone 1% of a radial orbit past pot_min
                trigger = False
                concount = time.perf_counter()
                update = True
                if np.sign(new_step[1] - pot_min) != orbitside:
                    orbitside *= -1
                if force_stop is not None:
                    if force_stop():
                        raise KeyboardInterrupt
                if mu != 0.0:
                    condate = True
                    new_step_hold, ch_cons = mm.peters_integrate6_6_4_7(all_states[tracker[j][-1]:], a, mu, ctx.j, i, all_states[tracker[-1][-1]], constants[j], label, err_target)

                    if radial_fail == 0 and new_step_hold is False:
                        #Radial issue, actual problem
                        E_hold, L_hold, C_hold = constants[j]
                        sep_cosi = mm.get_sep_cosi(a, L_hold/np.sqrt(L_hold*L_hold + C_hold), 100)
                        circ_ixs = np.where(sep_cosi["e"] == 0.0)
                        circEs, circLs, circCs, circrs = sep_cosi["energy"][circ_ixs], sep_cosi["phi_momentum"][circ_ixs], sep_cosi["carter"][circ_ixs], sep_cosi["p"][circ_ixs]
                        radial_fail = 1

                    wonky_test = max(0, wonky_test - 1)
                    if wonky_test >= 9:
                        print("Too many wonkies!")
                        raise KeyboardInterrupt
                    
                    if radial_fail == 1 and new_step_hold is False:
                        ixs = np.sort(np.argsort(np.abs(circEs - ch_cons[0]))[:5])
                        with warnings.catch_warnings():
                            warnings.filterwarnings('error')
                            fit_err, fit_num = True, 3
                            while fit_err and fit_num:
                                try:
                                    ch_cons[1] = np.polyval(np.polyfit(circEs[ixs], circLs[ixs], fit_num), ch_cons[0])
                                    fit_err = False
                                except np.exceptions.RankWarning:
                                    fit_num -= 1
                            if fit_err:
                                wonky_test += 1

                            fit_err, fit_num = True, 3
                            while fit_err and fit_num:
                                try:
                                    ch_cons[2] = np.polyval(np.polyfit(circEs[ixs], circCs[ixs], fit_num), ch_cons[0])
                                    fit_err = False
                                except np.exceptions.RankWarning:
                                    fit_num -= 1
                            if fit_err:
                                wonky_test += 1

                            fit_err, fit_num = True, 3
                            while fit_err and fit_num:
                                try:
                                    good_r = np.polyval(np.polyfit(circEs[ixs], circrs[ixs], fit_num), ch_cons[0])
                                    fit_err = False
                                except np.exceptions.RankWarning:
                                    fit_num -= 1
                            if fit_err:
                                wonky_test += 1

                        poly = np.array([ch_cons[0]**2 - 1, 2, (a**2)*(ch_cons[0]**2 - 1) - ch_cons[1]**2 - ch_cons[2], 2*((a*ch_cons[0] - ch_cons[1])**2 + ch_cons[2]), -ch_cons[2]*(a**2)])
                        tick = 0
                        flats = np.roots(np.polyder(poly))
                        turns = np.roots(poly)[:2]
                        while np.polyval(poly, max(flats)) < 0.0:
                            tick += 1
                            if ch_cons[1] != 0:
                                dL = - ch_cons[1] * err_target
                                ch_cons[2] = max(0.0, ch_cons[2] + 2 * ch_cons[2] * dL/ch_cons[1])
                                ch_cons[1] += dL
                            else:
                                ch_cons[2] = max(0.0, ch_cons[2] - ch_cons[2] * err_target)

                            poly = np.array([ ch_cons[0]**2 - 1, 2, (a**2)*(ch_cons[0]**2 - 1) - ch_cons[1]**2 - ch_cons[2], 2*((a*ch_cons[0] - ch_cons[1])**2 + ch_cons[2]), -ch_cons[2]*(a**2)])
                            flats = np.roots(np.polyder(poly))
                        turns = np.roots(poly)[:2]

                        new_step_hold = np.copy(all_states[tracker[-1][-1]])
                        new_step_hold[1] = good_r
                        new_step_hold = mm.recalc_state3(ch_cons, new_step_hold, a, tol=err_target)

                    if new_step_hold is None or new_step_hold is False:
                        print(f"skip! radial fail? {radial_fail == 1}")
                        #Theta issue, just try running longer
                        new_step_hold = new_step
                        update = False
                        condate = False
                        tracker.pop()
                    else:
                        new_step = np.copy(new_step_hold)

                    confull.append(time.perf_counter() - concount)

            #Initializing for the next step
            #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
            #Only happens the step before the derivatives are recalculated.
            
            upcount = time.perf_counter()
            #Update stuff!
            if (update == True):
                if condate == False:
                    newE, newLz, newC = getCons(state, a)
                else:
                    newE, newLz, newC = ch_cons

                newQ = newC + (a*newE - newLz)**2  
                turns, flats, zs = mm.root_getter(newE, newLz, newC, a)
                pot_min = flats[-1]
                inner_turn, outer_turn = turns[-2:]
                e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
                inc = np.sign(newLz)*np.arccos(min(1.0, np.mean(np.abs(zs[1:3]))))
                P0, ECC = 2*(inner_turn*outer_turn)/(inner_turn + outer_turn), (outer_turn - inner_turn)/(outer_turn + inner_turn)
                POT_MIN = pot_min
                try:
                    PERIOD = 2*np.pi / ((1/(POT_MIN**1.5 + a)) * np.sqrt(1 - 6/POT_MIN + 8*a/(POT_MIN**1.5) - 3*a*a/(POT_MIN**2)))
                except:
                    PERIOD = tracker[-1][-2] - tracker[-2][-2]
                constants.append([newE, newLz, newC])
                i = tracker[-1][-1] 
                tracker[-1] = [pot_min, e, inc, inner_turn, outer_turn, new_step[0], i]
                qarter.append(newQ)
                del all_states[i:]
                del dTau_change[i:]

                del true_anom[i+1:]
                if verbose > 0 and verbose < 3:
                    pbar.set_postfix_str("Semilat: %s, Ecc %s, Peri: %s, Theta_min: %s*pi" %(np.round( 0.5*(tracker[j][3] + tracker[j][4])*(1 - tracker[j][1]**2), 3), np.round(tracker[j][1], 3), np.round(tracker[j][3], 3), np.round(tracker[j][2]/np.pi, 3)))
                j += 1
                i -= 1
                ctx.j = j
                ctx.tracker = tracker
                ctx.last_chunk = np.asarray(all_states[tracker[j-1][-1]:])
                if True in np.iscomplex(tracker[j]):
                    compErr += 1
                    issues.append((i, new_step[0]))  
            i += 1
            dTau_change.append(dTau_change[-1] + old_dTau)
            all_states.append(new_step)    #update position and velocity
            anomval = get_true_anom(new_step, 0.5*(outer_turn + inner_turn), e) + orbCount*2*np.pi
            if anomval - true_anom[-1] < -np.pi:
                anomval += 2*np.pi
                orbCount += 1
            true_anom.append(anomval)
            ctx.i = i
            ctx.all_states = all_states
            ctx.true_anom = true_anom
            ctx_val = getval(ctx)
            if verbose == 3:
                progress = max(abs((ctx_val - start_val) / (threshold - start_val)), i/1e7)
                if (progress >= milestone):
                    print(f"Program has completed {ctx_val}, {progress*100}% of maximum run: Index = {i}")
                    milestone = int(progress) + 1
            elif verbose > 0 and verbose < 3:
                if update or trigger:
                    val = max(abs((ctx_val - start_val) / (threshold - start_val)) - progress, i/1e7 - progress, 0)
                    pbar.update(val*1e7)
                    progress = max(abs((ctx_val - start_val) / (threshold - start_val)), i/1e7)
            upfull.append(time.perf_counter() - upcount)
            

        #Lets you end the program before the established end without breaking anything
        except KeyboardInterrupt:
            print("\nEnding program - Halted", flush=True)
            stop = True
            cap = len(all_states) - 1
            del all_states[cap:]
            del dTau_change[cap:]
            del true_anom[cap:]
            break

        except KeyError:
            print("\nEnding program - Bad Values", flush=True)
            stop = True
            cap = len(all_states) - 1
            del all_states[cap:]
            del dTau_change[cap:]
            del true_anom[cap:]
            break
        
        except Exception as e:
            print("\nEnding program - ERROR", flush=True)
            print(type(e), e)
            print(traceback.format_exc())
            stop = True
            cap = len(all_states) - 1
            del all_states[cap:]
            del dTau_change[cap:]
            del true_anom[cap:]
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
        
    if mu == 1e-100:
        #so it gives actual numbers for pure geodesics
        mu = 1.0
    
    if j < len(tracker) - 1:
        del constants[j+1:]
        del tracker[j+1:]
        del qarter[j+1:]
    constants = np.array(constants)
    constants[:,0] *= mass*(c**2)
    constants[:,1] *= mass*mass*G/c
    constants[:,2] *= (mass*mass*G/c)**2
    qarter = np.array(qarter)
    t_scale, r_scale = (G*mass)/(c**3), (G*mass)/(c**2) 
    dTau_change = np.array(dTau_change) * t_scale
    all_states = np.array(all_states)
    all_states[:,0] *= t_scale 
    all_states[:,1] *= r_scale
    all_states[:,5] *= c
    all_states[:,6] /= t_scale
    all_states[:,7] /= t_scale
    tracker = np.array(tracker)
    tracker[:,0] *= r_scale
    tracker[:,3] *= r_scale
    tracker[:,4] *= r_scale
    tracker[:,5] *= t_scale
    
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

from filelock import FileLock
def update_index(filename, metadata, index_path="D:/EMRIData/saved_sims/index.json"):
    with FileLock(index_path + ".lock"):
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            print(f"{index_path} is not a valid index path. Creating new index.")
            index = {}

        index[filename] = metadata

        tmp_path = index_path + ".tmp"

        with open(tmp_path, 'w') as f:
            json.dump(index, f, indent=2)

        os.replace(tmp_path, index_path)

def load_index(index_path="D:/EMRIData/saved_sims/index.json"):
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            return json.load(f)
    else:
        print(f"{index_path} does not exist.")

def save_emri_data(final, filename=False, folder="D:/EMRIData/saved_sims/", auto=False, lock=None):
    if filename == False:
        filename = "auto_" + encode_filename()

    if not filename.endswith('.h5') and not filename.endswith('.hdf5'):
        filename += '.h5' 

    full = os.path.join(folder, filename)
    if lock:
        with lock:
            if not os.path.exists(full):
                pass
            else:
                if not auto:
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
                    print(f"File now saving as {filename}")
    else:
        if not os.path.exists(full):
            pass
        else:
            if not auto:
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
                print(f"File now saving as {filename}")

    with h5py.File(folder + filename, 'w') as f:
        # String attributes
        f.attrs['name'] = final['name']

        # Boolean flags
        f.attrs['stop'] = final['stop']
        f.attrs['plunge'] = final['plunge']
        f.attrs['unbind'] = final['unbind']

        # Raw numerical data
        f.create_dataset('raw', data=final['raw'], compression='gzip', compression_opts=9, shuffle=True)
        f.create_dataset('dTau_change', data=final['dTau_change'], compression='gzip', compression_opts=9, shuffle=True)
        f.create_dataset('energy', data=final['energy'], compression='gzip', compression_opts=9, shuffle=True)
        f.create_dataset('phi_momentum', data=final['phi_momentum'], compression='gzip', compression_opts=9, shuffle=True)
        f.create_dataset('carter', data=final['carter'], compression='gzip', compression_opts=9, shuffle=True)
        f.create_dataset('trackix', data=final['trackix'], compression='gzip', compression_opts=9, shuffle=True)

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

    if lock:
        with lock:
            update_index(filename, metadata)
    else:
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
            final["inc"] = np.arccos(np.where(incstuff <= 1.0, incstuff, 1.0))*np.sign(final['phi_momentum'])
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

import glob
import re

def normalize(s: str) -> str:
    return re.sub(r"[ _]+", "", s.lower())


def clean_logs(log_folder="D:/EMRIDATA/", json_path="D:/EMRIDATA/saved_sims/index.json", delete=False):
    # Load JSON
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Precompute normalized labels
    labels = []
    for entry in data.values():
        if isinstance(entry, dict) and "Label" in entry:
            labels.append(normalize(entry["Label"]))

    # Scan logs
    pattern = os.path.join(log_folder, "inspiral_*.log")
    files = glob.glob(pattern)
    
    if delete:
        print("Deleting files.")
    else:
        print("Evaluating files.")

    for path in files:
        filename = os.path.basename(path)

        # extract x from inspiral_x.log
        x = filename.replace("inspiral_", "").replace(".log", "")
        nx = normalize(x)

        # match check
        match = any(nx in label for label in labels)

        if not match:
            if delete:
                os.remove(path)
                print(f"Deleted: {filename}")
            else:
                print(f"Would delete: {filename}")
    print("Done.")

