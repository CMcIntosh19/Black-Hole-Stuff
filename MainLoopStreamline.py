# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:30:25 2023

@author: camcinto
"""

import numpy as np
import MetricMathStreamline as mm
from scipy.signal import argrelmin
from tqdm import tqdm
import h5py
import json
import os
import sys
import statistics as st
import time
from dataclasses import dataclass
import operator
import traceback
import warnings
import glob
import re
from filelock import FileLock
from pathlib import Path

@dataclass
class EndContext:
    i: int
    all_states: list
    true_anom: list
    tracker: list
    j: int

def EMRIGenerator(a, mu, 
                  endflag="min_radius < 0.5", 
                  err_target=1e-12, 
                  label="default", 
                  cons=False, 
                  velorient=False, 
                  vel4=False, 
                  params=False, 
                  pos=False, 
                  veltrue=False, 
                  verbose=1, 
                  skip_tar=0, 
                  force_stop=None):
    '''
    Generates orbit

    Parameters
    ----------
    a : float
        Dimensionless spin parameter of the central body. Valid for values between -1 and 1.
    mu : float
        Mass ratio between secondary body and central body. EMRI systems implicitly require mu to be less than or equal to 10^-4, but there are no explicit restrictions on this value.
    endflag : string, optional
        Condition for ending the simulation, written in the form '(variable) (comp.operator) (value)'
        Current valid variables:
            time - final time, measured in geometric units
            index - number of steps run, regardless of time
            elapsed_time - difference between final and initial time, measured in geometric units
            min_radius - minimum radius of the last radial orbit, in geometric units
            max_radius - maximum radius of the last radial orbit, in geometric units
            rad_orbit - number of completed radial orbits
            phi_orbit - number of completed azimuthal orbits
            semilat - semilatus rectum of orbit, measured in geometric units
            eccentricity - eccentricity of orbit
            inclination - maximum displacement from north pole of central body, measured in radians
        The default is "min_radius < 0.5", which places the requested stopping radius inside the event horizon. The simulation 
            normally terminates upon detecting plunge or reaching the maximum of 10^7 integration steps, whichever occurs first.
    err_target : float, optional
        Maximum error allowed during the geodesic evaluation. The default is 1e-12.
    label : string, optional
        Internal label for the simulation. The default option automatically generates label based on Keplerian parameters.
    cons : 3-element list of floats, optional
        Specific Energy, Specific Angular Momentum, and Specific Carter Constant. The default is False.
        The version of the Carter Constant used in EMRIGenerator vanishes for equatorial orbits and is non-negative for the physical timelike trajectories considered.
    velorient : 3-element list/array of floats, optional
        Ratio of velocity/speed of light (beta), angle from r_hat direction (eta - radians), angle from phi_hat (xi - radians),
            as measured from the Zero Angular Momentum (ZAMO) Frame
    vel4 : 4-element list/array of floats, optional
         4-velocity [gamma, gamma * v_r, gamma * v_theta, gamma * v_phi] as measured from the ZAMO frame.
         3-velocity components must be in fractions of c, and should satisfy gamma = (1 - (v_r^2 + v_theta^2 + v_phi^2))^(-1/2)
    params : 3-element list/array of floats, optional
        Semilatus rectum (geometric units), eccentricity, and inclination (radians) of orbit.
        Inclination is defined as the minimum polar coordinate theta_min reached along the trajectory, with a negative sign assigned to retrograde orbits. 
            pi/2 corresponds to an equatorial prograde orbit, 0 to a polar orbit, and -pi/2 to an equatorial retrograde orbit.
    pos : 4-element list/array of floats, optional
        Initial 4-position [t (geometric time), r (geometric distance), theta (radians), phi (radians)] of particle in Boyer-Lindquist coordinates. The default is False
    veltrue : 4-element list/array of floats, optional
        Initial contravariant four-velocity [u^t, u^r, u^theta, u^phi] in Boyer-Lindquist coordinates, where u^mu = dx^mu/dtau. 
            Components are expressed in geometric units (G = c = M = 1). The default is False.
    verbose : int, optional
        Toggle for progress updates as program runs. The default is 1.
        0 - No output
        1 - progress bar
        2 - full output
    skip_tar : int, optional
        Allows the geodesic to run for N + 1 radial orbits before evaluating losses due to radiation reaction. 
        The default is 0, which corresponds to a single orbit.
    force_stop : function, optional
        really only used for the multiprocessing stuff

    Returns
    -------
    final: 35 element dict
        Various tracked and record-keeping values for the resulting orbit
        "name": String, default label for orbit if plotted with OrbitPlotter functions
        "raw": N x 8 element numpy array of position and velocity of the secondary body
        "inputs": 11 element list of initial inputs for function call
        "pos": N x 4 subset of "raw", only includes 4-positions
        "vel": N x 4 Subset of "raw", only includes 4-velocities
        "time": N element array, Subset of "raw", only includes coordinate time
        "true_anom": N element array, True anomaly measured at every moment in "time"; approximate
        "interval": N element array, Spacetime interval at every point measured in "time"; should equal -1 at all times
        "proper_time": N element array, Proper time along trajectory
        "energy": M element array, Specific Energy of orbiting body at the beginning of each radial orbit
        "phi_momentum": M element array, Specific Angular Momentum of orbiting body at the beginning of each radial orbit
        "carter": M element array, Specific Carter Constant of orbiting body at the beginning of each radial orbit
        "spin": float, Dimensionless spin of central body
        "pot_min": M element array, radial potential minimum of orbiting body at the beginning of each radial orbit
        "e": M element array, eccentricity of orbiting body at the beginning of each radial orbit
        "inc": M element array, inclination of orbiting body at the beginning of each radial orbit
        "it": M element array, inner turning point of orbiting body at the beginning of each radial orbit
        "ot": M element array, outer turning point of orbiting body at the beginning of each radial orbit
        "r0": M element array, semimajor axis of orbiting body at the beginning of each radial orbit
        "p": M element array, semilatus rectum of orbiting body at the beginning of each radial orbit
        "tracktime": M element array, coordinate time at the beginning of each radial orbit
        "trackix": M element array, indices for the elements of all N-length arrays corresponding to the values in M-length arrays
        "omega": J element array, phi position of secondary body at periapse
        "otime": J element array, coordinate time at periapse
        "asc_node": K element array, phi position of secondary body at ascending node
        "asc_node_time": K element array, coordinate time at ascending node
        "des_node": L element array, phi position of secondary body at ascending node
        "des_node_time": L element array, coordinate time at ascending node
        "stop": bool, 'True' if simulation was aborted before reaching end condition, False otherwise
        "plunge": 'True' if simulation ended in a plunge, False otherwise
        "unbind": 'True' if secondary body moves out of primary body's approximated region of gravitational dominance (assumed unbound), False otherwise
        "issues": index and state corresponding to any point where Keplerian values read as complex
    '''

    def parse_endflag(flag: str):
        for op_str, op in OPS.items():
            if op_str in flag:
                lhs, rhs = flag.split(op_str)
                lhs = lhs.strip()
                rhs = float(rhs)
                if "radius" in lhs and rhs == 0.5:
                    rhs = mm.find_rms(a)
                return endvars[lhs], op, rhs
        raise ValueError("Invalid end condition")

    def get_true_anom(state, r0, e):
        pre = np.sign((r0*(1 - e**2)/state[1] - 1)) #e is always positive
        val = np.arccos(pre*min(1.0, abs((r0*(1 - e**2)/state[1] - 1)/(e + 1e-15)))) #add a little tiny bias to get rid of divide by zero errors
        if state[5] < 0:
            val = 2*np.pi - val
        return val

    endvars = {"time": lambda dat: dat.all_states[dat.i][0],
               "index": lambda dat: dat.i,
               "elapsed_time": lambda dat: abs(dat.all_states[dat.i][0] - dat.all_states[0][0]),
               "min_radius": lambda dat: dat.last_chunk[:,1].min(),
               "max_radius": lambda dat: dat.last_chunk[:,1].max(),
               "rad_orbit": lambda dat: abs((dat.true_anom[dat.i] - dat.true_anom[0]) / (2*np.pi)),
               "phi_orbit": lambda dat: abs(abs(dat.all_states[dat.i][3] / (2*np.pi)) - abs(dat.all_states[0][3] / (2*np.pi))),
               "semilat": lambda dat: (2 * dat.tracker[dat.j][3] * dat.tracker[dat.j][4] / (dat.tracker[dat.j][3] + dat.tracker[dat.j][4])),
               "eccentricity": lambda dat: dat.tracker[dat.j][1],
               "inclination": lambda dat: dat.tracker[dat.j][2]}
    
    OPS = {
        ">=": operator.ge,
        "<=": operator.le,
        ">": operator.gt,
        "<": operator.lt,
    }

    err_flag = 1
    err_flags = ["Too Many Runge-Kutta Loops",
                 "Forced Stop",
                 "Cannot Fit Radial Fail Polynomials",
                 "Cannot Correct Radial Fail Constants"]
    
    inputs = [a, mu, endflag, err_target, label, cons, velorient, vel4, params, pos, veltrue]          #Grab initial input in case you want to run the continue function
    inputs = [entry.tolist() if type(entry) == np.ndarray else entry for entry in inputs]              #Convert numpy arrays to lists so that JSON doesn't complain      

    capacity = 1000000
    all_states = np.full((capacity, 8), np.nan)                                                                       #initialize step counter
    if (np.shape(veltrue) == (4,)) and (np.shape(pos) == (4,)):
        start_check = mm.check_interval(mm.kerr_2, [*pos, *veltrue], a)
        if abs(start_check + 1) <= err_target:
            all_states[0] = np.array([*pos, *veltrue])
        else:
            print("Nonviable initial state provided, defaulting to plunge.")
            all_states[0], cons = mm.set_u_kerr(a, pos, velorient=[0.01, np.pi, np.pi/2])

    else:
        if verbose:
            print("Normalizing initial state")
        all_states[0], cons = mm.set_u_kerr(a, cons, velorient, vel4, params, pos)      #normalize initial state so it's actually physical

    metric, chris = mm.kerr_2(all_states[0], a)                                     #initial metric and christoffel symbols
    if type(cons) == bool:
        cons = mm.getCons(all_states[0], a)

    turns, flats, zs = mm.root_getter(*cons, a)
    pot_min = max(flats)
    inner_turn, outer_turn = turns[-2:]
    r0, e = 0.5 * (inner_turn + outer_turn), (outer_turn - inner_turn)/(outer_turn + inner_turn)
    inc = np.sign(cons[1])*np.arccos(min(1.0, np.mean(np.abs(zs[1:3]))))
    val = get_true_anom(all_states[0], r0, e)
    # Get the approximate radial period for circular orbits
    PERIOD = 2 * np.pi / (np.sqrt(1 - 6/r0 + 8*a/(r0**1.5) + 8*(a**2)/(r0*r0)) / (r0**1.5 + a))
    # But it blows up at ISCO, so cap it when necessary (probably won't come up)
    PERIOD = min(PERIOD, 7*(r0**1.5) + 1000, r0)

    true_anom = np.full(capacity, np.nan, dtype=complex)
    true_anom[0] = (val if np.isnan(val) == False else 0.0)
    constants = np.full((capacity//100, 3), np.nan)
    constants[0] = [*cons]
    tracker = np.full((capacity//100, 7), np.nan, dtype=complex)
    tracker[0] = [pot_min, e, inc, inner_turn, outer_turn, all_states[0][0], 0]

    compErr = 0
    milestone = 0
    issues = []
    orbitside = np.sign(all_states[0][1] - pot_min)
    if orbitside == 0:
        orbitside = -1
    
    orbCount = 0
    stop = False
    radial_fail = 0
    
    if label == "default":
        label = f"p{r0 * (1 - e**2):.3f}e{e:.3E}\u03C0{inc/np.pi:.3f}mu{mu:.3E}a{a}"
    
    #Main Loop
    dTau = PERIOD / (360 * 60)
    proper_time = np.full(capacity, np.nan)
    proper_time[0] = 0
    plunge, unbind = False, False

    if verbose > 0 and verbose < 3:
        pbar = tqdm(total = 10000001, position=0)
        pbar.set_postfix_str("Semilat: %s, Ecc %s, Peri: %s, Theta_min: %s*pi" %(np.round( 0.5*(tracker[0][3] + tracker[0][4])*(1 - tracker[0][1]**2), 3), np.round(tracker[0][1], 3), np.round(tracker[0][3], 3), np.round(tracker[0][2]/np.pi, 3)))

    progress = 0
    rkfull = []
    confull = []
    upfull = []
    i = 0
    j = 0
    getval, op, threshold = parse_endflag(endflag)
    ctx = EndContext(i, all_states, true_anom, tracker, j)
    ctx.last_chunk = np.asarray(all_states[:i+1])
    start_val, ctx_val = getval(ctx), getval(ctx)
    trigger = False
    con_fit_error = 0
    skip_count = 0
    photon_ring = mm.find_rph(a)

    while not(ctx_val and op(ctx_val, threshold)) and i < 1e7:
        try:
            if capacity - i < 100 and capacity < 10000001:
                all_states = np.concatenate((all_states, np.full((capacity, 8), np.nan)))
                constants = np.concatenate((constants, np.full((capacity//100, 3), np.nan)))
                tracker = np.concatenate((tracker, np.full((capacity//100, 7), np.nan, dtype=complex)))
                true_anom = np.concatenate((true_anom, np.full(capacity, np.nan, dtype=complex)))
                proper_time = np.concatenate((proper_time, np.full(capacity, np.nan)))
                capacity = min(10000001, len(true_anom))
                
            update = False
            condate = False
            err_calc = 1
          
            #Grab the current state
            state = all_states[i]  
            pot_min = tracker[j][0] 
          
            #Break if you fall below photon ring, or if you get really far away (orbit is probably unbound)
            if (state[1] <= photon_ring):
                plunge = True
                break
            
            if (state[1] >= 4.872e5):    #Kind of an arbitrary distance, ~0.1 pc if M_bh = Sagittarius A*
                unbind = True
                break

            #break if something stops making sense
            if (True in np.isnan(state) in state or constants[j][0] < 0):
                plunge = True
                unbind = True
                break

            # Runge-Kutta update using geodesic
            old_dTau = dTau
            rkcount = time.perf_counter()
            counter = 0
            while (err_calc >= err_target):
                # If you go through the RK loop too many times, assume it's stalled and jump ship
                counter += 1
                if counter > (int(-np.log10(err_target))**2)//2:
                    # More forgiving for tighter tolerances
                    err_flag = 0
                    raise KeyboardInterrupt
                
                # Generate 4th and 5th order calculations for next step
                new_step = mm.gen_RK2(*mm.ck4_2, mm.kerr_2, state, dTau, a)
                step_check = mm.gen_RK2(*mm.ck5_2, mm.kerr_2, state, dTau, a) 

                # Calculate the error
                delt = new_step - step_check
                delt = np.concatenate((delt[1:3], delt[4:]))
                err_calc = np.linalg.norm(delt/6)

                # Correct for pole effects
                # if ((within ~0.5 degrees of pole) and (moving closer to pole)) and (at least fifteen steps have been evaluated):
                if ((np.sin(new_step[2]) <= 0.009) and new_step[6]*np.cos(new_step[2]) < 0) and i >= 16:
                    # Average of the last 10 accepted dTau values
                    recent_mean = (proper_time[i-1] - proper_time[i-10]) / 9
                    # Average of all accepted dTau values
                    overall_mean = (proper_time[i-1] - proper_time[0]) / (i-1)
                    # if average dTau value is rapidly shrinking:
                    if recent_mean <= 0.001 * overall_mean:
                        old_step = new_step
                        # This is happening close enough to the pole that we will assume:
                        #   1) We can treat the area around the pole as a flat plane
                        #   2) We can approximate the movement of the particle in this plane as linear
                        # We treat angular distance from the pole as our radius analogue, then define two circles:
                        # Circle 1 has a 'radius' equal to the inclination of the orbit (minimum distance from the pole)
                        # Circle 2 has a 'radius' equal to the current distance from the pole
                        # The path the particle travels on is secant to Circle 2 and tangent to Circle 1 
                        #   such that a ray from the origin to the tangent point on Circle 1 bisects the path
                        # k is the ratio between these circles
                        k = abs(inc)/min((np.pi - new_step[2]), new_step[2])
                        if k == 0.0:
                            # Orbit is polar, so the particle passes straight through the origin. Trivial
                            phi_dist = np.pi if new_step[7] >= 0.0 else -np.pi
                        else:
                            cosp, sinp = np.cos(new_step[3]), np.sin(new_step[3])
                            # m = the slopes of lines that are tangent to Circle 1 and intersect Circle 2 at our current position
                            # There are two! Deeply Inconvenient!
                            m = (cosp*sinp + np.array([1, -1])*k*np.sqrt(1 - k*k))/(cosp**2 - k*k)
                            # phi_min = the phi values that correspond to the tangent points
                            numerator = (sinp - m*cosp)/k
                            phi_min = (np.arctan(m) + np.arcsin(np.clip(numerator/np.sqrt(1 + m*m), -1, 1)))%(2*np.pi)
                            # Use phi_dot to determine which solution is right
                            arr = (phi_min - new_step[3]%(2*np.pi))/new_step[7]
                            phi_min_real = phi_min[0] if arr[0] > 0 else phi_min[1]
                            # Now get the angular distance from the tangent to the current phi, add double that to get the NEW phi
                            phi_dist = 2*min(abs(phi_min - old_step[3]%(2*np.pi)))*np.sign(old_step[7])
                        new_step[3] += phi_dist
                        # Estimate the elapsed time as d(phi)/phi_dot
                        t_change = 2 * min((np.pi - np.abs(new_step[2]%(2*np.pi) - np.pi)), np.abs(new_step[2]%(2*np.pi) - np.pi)) / abs(new_step[6])
                        new_step[0] += t_change
                        # Scale up dTau so that we account for travel time across Circle 2
                        dTau = dTau*(new_step[0] - state[0])/(old_step[0] - state[0])
                        # Flip the sign on theta_dot so now it's moving away from the pole
                        new_step[6] *= -1

                # Pass the current dTau along to old_dTau so we can actually record it
                old_dTau = dTau
                # Adjust dTau for our amazing adaptive timestep; the smaller our error, the larger we'll make dTau and vice versa
                # Except we don't want to accidentally skip over cool stuff, so we choose an approximate maximum of 4% of a circular period (very rough estimate)
                dTau = np.sign(dTau) * min(abs(dTau) * abs(err_target / (err_calc + (err_target/100)))**(0.2), PERIOD * 0.04)

                # If some kind of nonsense happened in the Runge-Kutta, try increasing dTau by a bunch
                # In the hopes it gives you a real answer that will shrink to a good one if necessary
                if np.isnan(err_calc):
                    err_calc = 1
                    dTau = old_dTau * 100

            metric = mm.kerr(new_step, a)[0]
            # This shouldn't generally trigger, but just in case let's make sure we're on the mass shell
            # This also makes sure we're travelling foreward in time but maybe there's a use case for allowing backwards time? Later
            test = mm.check_interval_w_metric(metric, new_step, a)
            counter2 = 0
            while (abs(test+1) > (err_target) or new_step[4] < 0.0) and counter2 < 10:
                old_step = np.copy(new_step)
                gtt, gtp = metric[0,0], metric[0,3]
                # Try to change t_dot to put us back on the mass shell
                # This does actually change the energy, but usually (hopefully) not by much if this ever triggers
                disc = 4*(gtp*new_step[4]*new_step[7])**2 - 4*gtt*(new_step[4]**2)*(np.einsum('ij, i, j ->', metric[1:,1:], new_step[5:], new_step[5:]) + 1)
                delt = (-2*gtp*new_step[4]*new_step[7] - np.sqrt(disc))/(2*gtt*new_step[4]*new_step[4])
                new_step[4] *= delt
                test = mm.check_interval_w_metric(metric, new_step, a)
                counter2 += 1

            # If tweaking t_dot didn't work, just put it back the way it was - no reason to have a new bad answer when we have an old one that's consistent
            if (test+1) > err_target or new_step[4] < 0.0:
                new_step = np.copy(old_step)
                counter2 = 0
                
            if counter2 > 0:
                issues.append((i, new_step[0], delt))

            rkfull.append(time.perf_counter() - rkcount)

            #constant modifying section
            if i%50 == 0:
                if force_stop is not None:
                    if force_stop():
                        err_flag = 1
                        raise KeyboardInterrupt

            # (hasn't triggered yet) and (((has just crossed potential minimum) and (it's been >75% of an orbit)) or ((orbit is very circular) and (it's been more than one orbit))
            if trigger == False and (((state[1] > pot_min and new_step[1] <= pot_min) and new_step[0] - tracker[j][-2] > PERIOD*0.75) or ((e <= 1e-4) and (new_step[0] - tracker[j][-2] > PERIOD))):
                if skip_count == skip_tar:
                    orbCount += 1
                    trigger = True
                    tracker[j+1] = [pot_min, e, inc, inner_turn, outer_turn, new_step[0], i]
                else:
                    skip_count += 1

            if (trigger == True) and abs(new_step[0] - tracker[j+1][-2]) > 0.03*PERIOD:
                #register when we've gone 1% of a radial orbit past pot_min
                trigger = False
                concount = time.perf_counter()
                update = True
                if np.sign(new_step[1] - pot_min) != orbitside:
                    orbitside *= -1
                if force_stop is not None:
                    if force_stop():
                        err_flag = 1
                        raise KeyboardInterrupt
                if mu != 0.0:
                    condate = True
                    new_step_hold, ch_cons = mm.peters_integrate(all_states[int(np.real(tracker[j][-1])):i], 
                                                                 a, 
                                                                 mu, 
                                                                 ctx.j, 
                                                                 i, 
                                                                 all_states[int(np.real(tracker[j+1][-1]))], 
                                                                 constants[j], 
                                                                 label, 
                                                                 err_target)

                    if radial_fail == 0 and new_step_hold is False:
                        #Radial issue, actual problem
                        E_hold, L_hold, C_hold = constants[j]
                        sep_cosi = mm.get_sep_cosi(a, L_hold/np.sqrt(L_hold*L_hold + C_hold), 100)
                        circ_ixs = np.where(sep_cosi["e"] == 0.0)
                        circEs, circLs, circCs, circrs = sep_cosi["energy"][circ_ixs], sep_cosi["phi_momentum"][circ_ixs], sep_cosi["carter"][circ_ixs], sep_cosi["p"][circ_ixs]
                        radial_fail = 1

                    con_fit_error = max(0, con_fit_error - 1)
                    if con_fit_error >= 9:
                        err_flag = 2
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
                                con_fit_error += 1

                            fit_err, fit_num = True, 3
                            while fit_err and fit_num:
                                try:
                                    ch_cons[2] = np.polyval(np.polyfit(circEs[ixs], circCs[ixs], fit_num), ch_cons[0])
                                    fit_err = False
                                except np.exceptions.RankWarning:
                                    fit_num -= 1
                            if fit_err:
                                con_fit_error += 1

                            fit_err, fit_num = True, 3
                            while fit_err and fit_num:
                                try:
                                    good_r = np.polyval(np.polyfit(circEs[ixs], circrs[ixs], fit_num), ch_cons[0])
                                    fit_err = False
                                except np.exceptions.RankWarning:
                                    fit_num -= 1
                            if fit_err:
                                con_fit_error += 1

                        poly = np.array([ch_cons[0]**2 - 1, 2, (a**2)*(ch_cons[0]**2 - 1) - ch_cons[1]**2 - ch_cons[2], 2*((a*ch_cons[0] - ch_cons[1])**2 + ch_cons[2]), -ch_cons[2]*(a**2)])
                        tick = 0
                        flats = np.roots(np.polyder(poly))
                        turns = np.roots(poly)[:2]
                        while np.polyval(poly, max(flats)) < 0.0:
                            tick += 1

                            if tick > 1000:
                                err_flag = 3
                                raise KeyboardInterrupt
                            
                            if ch_cons[1] != 0:
                                dL = - ch_cons[1] * err_target
                                ch_cons[2] = max(0.0, ch_cons[2] + 2 * ch_cons[2] * dL/ch_cons[1])
                                ch_cons[1] += dL
                            else:
                                ch_cons[2] = max(0.0, ch_cons[2] - ch_cons[2] * err_target)

                            poly = np.array([ ch_cons[0]**2 - 1, 2, (a**2)*(ch_cons[0]**2 - 1) - ch_cons[1]**2 - ch_cons[2], 2*((a*ch_cons[0] - ch_cons[1])**2 + ch_cons[2]), -ch_cons[2]*(a**2)])
                            flats = np.roots(np.polyder(poly))
                        turns = np.roots(poly)[:2]

                        new_step_hold = np.copy(all_states[int(np.real(tracker[j+1][-1]))])
                        new_step_hold[1] = good_r
                        new_step_hold = mm.recalc_state3(ch_cons, new_step_hold, a, tol=err_target)

                    if new_step_hold is None or new_step_hold is False:
                        #print(f"skip! radial fail? {radial_fail == 1}")
                        #Theta issue, just try running longer
                        new_step_hold = new_step
                        update = False
                        condate = False
                    else:
                        new_step = np.copy(new_step_hold)

                    confull.append(time.perf_counter() - concount)
                else:
                    new_step = all_states[int(np.real(tracker[j+1][-1]))]
            #Initializing for the next step
            #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
            #Only happens the step before the derivatives are recalculated.
            
            upcount = time.perf_counter()
            #Update stuff!
            if (update == True):
                skip_count = 0
                if condate:
                    new_cons = ch_cons
                else:
                    new_cons = constants[j]
                turns, flats, zs = mm.root_getter(*new_cons, a)
                pot_min = flats[-1]
                inner_turn, outer_turn = turns[-2:]
                r0, e = 0.5 * (inner_turn + outer_turn), (outer_turn - inner_turn)/(outer_turn + inner_turn)
                inc = np.sign(new_cons[1])*np.arccos(min(1.0, np.mean(np.abs(zs[1:3]))))
                PERIOD = 2 * np.pi / (np.sqrt(1 - 6/r0 + 8*a/(r0**1.5) + 8*(a**2)/(r0*r0)) / (r0**1.5 + a))
                PERIOD = min(PERIOD, 7*(r0**1.5) + 1000)
                if condate == False:
                    new_cons = mm.getCons(state, a)
                else:
                    new_cons = ch_cons
                constants[j+1] = [*new_cons]

                i = int(np.real(tracker[j+1][-1])) 
                tracker[j+1] = [pot_min, e, inc, inner_turn, outer_turn, new_step[0], i]

                if verbose > 0 and verbose < 3:
                    pbar.set_postfix_str(f"Semilat: {np.real(0.5*(tracker[j][3] + tracker[j][4])*(1 - tracker[j][1]**2)):.3f}, Ecc {np.real(tracker[j][1]):.3f}, Peri: {np.real(tracker[j][3]):.3f}, Theta_min: {np.real(tracker[j][2]/np.pi):.3f}*pi")
                j += 1
                i -= 1
                ctx.j = j
                ctx.tracker = tracker
                ctx.last_chunk = np.asarray(all_states[int(np.real(tracker[j-1][-1])):i+1])
                if True in np.iscomplex(tracker[j]):
                    compErr += 1
                    issues.append((i, new_step[0]))  
            
            i += 1
            proper_time[i] = proper_time[i-1] + old_dTau
            all_states[i] = new_step    #update position and velocity
            anomval = get_true_anom(new_step, r0, e) + orbCount*2*np.pi
            if anomval - true_anom[i-1] < -np.pi:
                anomval += 2*np.pi
            true_anom[i] = anomval
            ctx.i = i
            ctx.all_states = all_states[:i+1]
            ctx.true_anom = true_anom[:i+1]
            ctx_val = getval(ctx)
            if verbose == 3:
                progress = max(abs((ctx_val - start_val) / (threshold - start_val)), i/1e7)
                if (progress >= milestone):
                    print(f"Program has completed {ctx_val}, {progress*100}% of maximum run: Index = {i}")
                    milestone = int(progress) + 1
            elif verbose > 0 and verbose < 3:
                if update or trigger:
                    #print(ctx_val, start_val, threshold, progress, abs((ctx_val - start_val) / (threshold - start_val)) - progress, i/1e7 - progress)
                    val = max(abs((ctx_val - start_val) / (threshold - start_val)) - progress, i/1e7 - progress, 0)
                    pbar.update(int(val*1e7))
                    progress = max(abs((ctx_val - start_val) / (threshold - start_val)), i/1e7)
            upfull.append(time.perf_counter() - upcount)
            
        #Lets you end the program before the established end without breaking anything
        except KeyboardInterrupt:
            print(f"\nEnding program - {err_flags[err_flag]}", flush=True)
            stop = True
            break

        except KeyError:
            print("\nEnding program - Bad Values", flush=True)
            stop = True
            break
        
        except Exception as e:
            print("\nEnding program - ERROR", flush=True)
            print(type(e), e)
            print(traceback.format_exc())
            stop = True
            break


    if verbose > 0 and verbose < 3:
        pbar.close()
    
    constants = constants[:j+1]
    tracker = tracker[:j+1]
    all_states = all_states[:i]
    proper_time = proper_time[:i]
    true_anom = true_anom[:i]
    
    ind = argrelmin(all_states[:,1])[0]
    omega, otime = all_states[ind,3] - 2*np.pi*np.arange(len(ind)), all_states[ind,0]
    asc_node, asc_node_time = np.array([]), np.array([])
    des_node, des_node_time = np.array([]), np.array([])
    interval = mm.check_interval_vec(all_states, a)
    if max(all_states[:,2]) - min(all_states[:,2]) > 1e-15:
        cos2theta = np.cos(all_states[:, 2])**2
        ind_cross = argrelmin(cos2theta)
        ind2 = ind_cross[0][::2] if all_states[ind_cross[0][0], 6] < 0.0 else ind_cross[0][1::2]
        ind3 = ind_cross[0][1::2] if all_states[ind_cross[0][0], 6] < 0.0 else ind_cross[0][::2]

        asc_node, asc_node_time = all_states[ind2,3] - 2*np.pi*np.arange(len(ind2)), all_states[ind2,0] #subtract the normal phi advancement
        des_node, des_node_time = all_states[ind3,3] - 2*np.pi*np.arange(len(ind3)), all_states[ind3,0] #subtract the normal phi advancement

    if verbose == 3:
        print("There were " + str(compErr) + " issues with complex roots/turning points.")
    if verbose >= 2:
        try:
            print("rkstats: min %s, max %s, mean %s, stdev%s, median %s, mode %s, total %s"%(min(rkfull), max(rkfull), np.mean(rkfull), np.std(rkfull), np.median(rkfull), st.mode(rkfull), np.sum(rkfull)))
            print("constats: min %s, max %s, mean %s, stdev%s, median %s, mode %s, total %s"%(min(confull), max(confull), np.mean(confull), np.std(confull), np.median(confull), st.mode(confull), np.sum(confull)))
            print("upstats: min %s, max %s, mean %s, stdev%s, median %s, mode %s, total %s"%(min(upfull), max(upfull), np.mean(upfull), np.std(upfull), np.median(upfull), st.mode(upfull), np.sum(upfull)))
        except:
            print("stats borken")

    final = {"name": label,
             "raw": all_states,
             "inputs": inputs,
             "pos": all_states[:, 1:4],
             "vel": all_states[:, 4:], 
             "time": all_states[:,0],
             "true_anom": true_anom,
             "interval": interval,
             "proper_time": proper_time,
             "energy": constants[:, 0],
             "phi_momentum": constants[:, 1],
             "carter": constants[:, 2],
             "spin": a,
             "pot_min": np.real(tracker[:,0]),
             "e": np.real(tracker[:,1]),
             "inc": np.real(tracker[:,2]),
             "it": tracker[:,3],
             "ot": tracker[:,4],
             "r0": np.real(0.5*(tracker[:,3] + tracker[:,4])),
             "p": np.real(0.5*(tracker[:,3] + tracker[:,4])*(1 - tracker[:,1]**2)),
             "tracktime": np.real(tracker[:,5]),
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
    """Creates a filename based on UTC date and time"""
    now = time.time()
    return hex(int(40587.0 + now//86400.0))[2:] + "_" + hex(int((now/86400.0 - now//86400.0)*60*24))[2:]

def decode_filename(name):
    """Turns filenames created by encode_filename back into unix dates"""
    splits = os.path.splitext(name)[0].split("_")
    mjd = int(splits[1], 16) + int(splits[2], 16)/(60*24)
    unix = (mjd - 40587)*86400
    return unix

def get_size(obj, seen=None):
    """Recursive function that returns the size of obj in bytes"""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(get_size(v, seen) for v in obj.values())
        size += sum(get_size(k, seen) for k in obj.keys())
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_size(item, seen) for item in obj)
    return size

def resolve_data_folder(folder=None, create=False):
    """Resolve a data directory, optionally creating it."""
    if folder is None:
        # Track a folder called "saved_sims" on the same level as MainLoopStreamline
        path = Path(__file__).resolve().parent / "saved_sims"
    else:
        # Accepts absolute and relative paths
        path = Path(folder).expanduser()
    path = path.resolve()

    if create:
        # Creates a new directory, or silently does nothing if the directory already exists
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise FileNotFoundError(f"EMRI data directory does not exist: {path}")

    return path

def resolve_data_file(filename, folder=None, create_folder=False):
    """Resolve an EMRI HDF5 filename within a data directory."""
    directory = resolve_data_folder(folder, create=create_folder)

    filename = Path(filename).name
    if not filename.endswith((".h5", ".hdf5")):
        filename += ".h5"

    return directory / filename

INPUT_LABELS = ["Spin", "Mass Ratio", "Endflag", "Target Error", 
                "Label", "Constants", "Orientation Velocity",
                "Tetrad Velocity", "Kepler Parameters", "Position",
                "True Velocity"]

def make_index_metadata(inputs, stop, plunge, unbind, uncompressed, compressed, created=None):
    """Creates metadata for EMRI data in an index"""
    metadata = dict(zip(INPUT_LABELS, inputs))
    metadata.update({
        "Halted": bool(stop),
        "Plunged": bool(plunge),
        "Escaped": bool(unbind),
        "Dataset Size (MiB)": uncompressed,
        "File Size (MiB)": compressed, 
        "Created": created or time.strftime("%a, %d %b %Y %H:%M:%S UTC",
                                            time.gmtime())})

    return metadata

def update_index(filename, metadata, folder=None):
    """Adds an entry for a file created by save_emri_data in an index"""
    directory = resolve_data_folder(folder, create=True)
    index_path = directory / "index.json"

    with FileLock(str(index_path) + ".lock"):
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
        else:
            print(f"{index_path} is not a valid index path. Creating new index.")
            index = {}

        index[filename] = metadata

        tmp_path = directory / "index.json.tmp"

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        os.replace(tmp_path, index_path)

def rebuild_index(folder=None, overwrite=False):
    """
    Scan a directory for EMRICode HDF5 files and rebuild its index.

    Parameters
    ----------
    folder : str or Path, optional
        Directory to scan. Defaults to EMRICode/saved_sims/.
    overwrite : bool, optional
        If True, replace the existing index with entries found
        in the directory. Otherwise, preserve existing entries
        and add/update entries for files discovered.

    Returns
    -------
    dict
        The resulting index.
    """
    directory = resolve_data_folder(folder, create=False)
    index_path = directory / "index.json"

    # Coordinate with save_emri_data so files aren't being
    #   published while we scan the directory.
    with FileLock(str(directory / ".emri_save.lock")):

        # Also coordinate with update_index/load_index.
        with FileLock(str(index_path) + ".lock"):

            if index_path.exists() and not overwrite:
                with index_path.open("r", encoding="utf-8") as f:
                    index = json.load(f)
            else:
                index = {}

            files = sorted(path for path in directory.iterdir()
                           if path.is_file()
                           and path.suffix.lower() in (".h5", ".hdf5"))

            added = 0
            skipped = 0

            for path in files:
                try:
                    with h5py.File(path, "r") as f:
                        required = {"raw", "energy", "phi_momentum",
                                    "carter", "trackix", "inputs", "issues"}

                        if not required.issubset(f.keys()):
                            skipped += 1
                            continue

                        if not all(key in f.attrs
                                   for key in ("name", "stop", "plunge", "unbind")):
                            skipped += 1
                            continue

                        inputs_raw = f["inputs"][()]
                        if isinstance(inputs_raw, bytes):
                            inputs_raw = inputs_raw.decode("utf-8")

                        inputs = json.loads(inputs_raw)

                        if not isinstance(inputs, list) or len(inputs) != len(INPUT_LABELS):
                            skipped += 1
                            continue

                        # Use the file modification time
                        # when no previous index entry exists.
                        old_created = index.get(path.name, {}).get("Created")
                        created = old_created or time.strftime("%a, %d %b %Y %H:%M:%S UTC", 
                                                               time.gmtime(path.stat().st_mtime))

                        # Calculate total uncompressed dataset size
                        uncompressed_bytes = sum(obj.nbytes for obj in f.values() if isinstance(obj, h5py.Dataset))

                        # Get actual file size on disk
                        compressed_bytes = path.stat().st_size

                        metadata = make_index_metadata(inputs, 
                                                       f.attrs["stop"],
                                                       f.attrs["plunge"],
                                                       f.attrs["unbind"],
                                                       round(uncompressed_bytes / 1024**2, 2),
                                                       round(compressed_bytes / 1024**2, 2),
                                                       created=created)

                        index[path.name] = metadata
                        added += 1

                except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    print(f"Skipping {path.name}: {exc}")
                    skipped += 1

            # Atomically replace the index.
            tmp_path = directory / "index.json.tmp"

            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)

            os.replace(tmp_path, index_path)

    print(f"Indexed {added} EMRI files; skipped {skipped} files.")
    return index

def load_index(folder=None):
    """Load a chosen index to a dict object"""
    directory = resolve_data_folder(folder)
    index_path = directory / "index.json"

    if not index_path.exists():
        print(f"{index_path.resolve()} does not exist.")
        return {}

    with FileLock(str(index_path) + ".lock"):
        with index_path.open("r", encoding="utf-8") as f:
            return json.load(f)

def save_emri_data(final, filename=False, folder=None, auto=False, lock=None):
    """
    Save the output of EMRIGenerator to an HDF5 file and record it in an index.

    Parameters
    ----------
    final : dict
        result of EMRIGenerator
    filename : str, optional
        name of HDF5 file data is saved to. Defaults to a UTC-derived
            string generated by encode_filename
    folder : str or Path, optional
        Directory to save data to. Defaults to EMRICode/saved_sims/.
    auto : bool, optional
        If True, automatically adjust filename in order to prevent overwrite
        of an existing file with the same name. Otherwise, ask the user.
    lock : lock object, optional
        for multiprocessing

    Returns
    -------
    Path
        The resulting file path.
    """
    directory = resolve_data_folder(folder, create=True)

    if filename is False or filename is None:
        filename = "auto_" + encode_filename()

    path = resolve_data_file(filename, directory)

    def choose_path(path):
        if not path.exists():
            return path

        if not auto:
            overwrite = input(f"Data for {path.name} already exists. Overwrite? (y/n): ").strip().lower()
        else:
            overwrite = "n"

        if overwrite == "y":
            print(f"Overwriting {path.name}...")
            return path

        base = path.stem
        ext = path.suffix
        num = 1

        while True:
            candidate = path.with_name(f"{base}_{num}{ext}")
            if not candidate.exists():
                print(f"File now saving as {candidate.name}")
                return candidate
            num += 1

    # Choose the output path
    if lock is not None:
        with lock:
            path = choose_path(path)
    else:
        path = choose_path(path)

    # Write the HDF5 file
    with h5py.File(path, "w") as f:
        f.attrs["name"] = final["name"]
        f.attrs["stop"] = final["stop"]
        f.attrs["plunge"] = final["plunge"]
        f.attrs["unbind"] = final["unbind"]

        for key in ("raw", "proper_time", "energy",
                    "phi_momentum", "carter", "trackix"):
            f.create_dataset(key, data=final[key],
                             compression="gzip", compression_opts=9, shuffle=True)

        dt = h5py.string_dtype(encoding="utf-8")

        f.create_dataset("inputs", data=json.dumps(final["inputs"]), dtype=dt)
        f.create_dataset("issues", data=json.dumps(final["issues"]), dtype=dt)

    uncompressed_bytes = get_size(final)
    #for key in final.keys():
    #    size = get_size(final[key])
    #    print(key, size)
    #    uncompressed_bytes += size

    compressed_bytes = path.stat().st_size

    metadata = make_index_metadata(final["inputs"],
                                   final["stop"],
                                   final["plunge"],
                                   final["unbind"],
                                   round(uncompressed_bytes / 1024**2, 2),
                                   round(compressed_bytes / 1024**2, 2))

    if lock is not None:
        with lock:
            update_index(path.name, metadata, folder=directory)
    else:
        update_index(path.name, metadata, folder=directory)

    return path.name

def load_emri_data(filename, folder=None, quiet=False, reconstruct=True):
    """
    Load an HDF5 file and save the data into a new EMRIGenerator object.

    Parameters
    ----------
    filename : str
        name of HDF5 file data is saved to.
    folder : str or Path, optional
        Directory to search. Defaults to EMRICode/saved_sims/.
    quiet : bool, optional
        If False, announce progress as the function runs.
        Otherwise, remain silent.
    reconstruct : bool, optional
        If True, rebuild the original EMRIGenerator object in its entirety.
        Otherwise, only load values that cannot be reconstructed from the others.
        This includes:
            name - str, Internally resolved object label
            stop - bool, tracks forced stop
            plunge - bool, tracks if simulation ended in a plunge
            unbind - bool, tracks if secondary object moved too far from primary
            raw - N x 8 numpy array, position and velocity data throughout orbit
            proper_time - length N numpy array, proper time along orbit
            energy - length M numpy array, specific energy once per orbit
            phi_momentum - length M numpy array, specific axial angular momentum once per orbit
            carter - length M numpy array, specific Carter constant once per orbit
            trackix - length M numpy array, indices that map M-length arrays to N-length arrays
            inputs - mixed object list, original inputs to EMRIGenerator
            issues - mixed object list, lists errors in original geodesic run
            spin - float, dimensionless spin of black hole

    Returns
    -------
    dict
        EMRIGenerator results
    """
    path = resolve_data_file(filename, folder)

    if not path.is_file():
        raise FileNotFoundError(f"EMRI data file does not exist: {path}")

    if not quiet:
        print(f"Loading {path}")

    with h5py.File(path, 'r') as f:
        final = {}

        # Basic attributes
        final['name'] = f.attrs['name']
        final['stop'] = f.attrs['stop']
        final['plunge'] = f.attrs['plunge']
        final['unbind'] = f.attrs['unbind']

        # Arrays
        final['raw'] = f['raw'][:]
        try:
            final['proper_time'] = f['proper_time'][:]
        except:
            final['proper_time'] = f['dTau_change'][:]
        final['energy'] = f['energy'][:]
        final['phi_momentum'] = f['phi_momentum'][:]
        final['carter'] = f['carter'][:]
        final['trackix'] = f['trackix'][:]

        # Decode JSONs back into Python objects
        final['inputs'] = json.loads(f['inputs'][()].decode('utf-8', errors='replace'))
        final['issues'] = [tuple(x) for x in json.loads(f['issues'][()].decode('utf-8'))]
        final["spin"] = final["inputs"][0]
        
        if reconstruct:
            # Reconstruct derived values
            raw = final["raw"]
            trackix = final["trackix"]
            final["pos"] = raw[:,1:4]
            final["vel"] = raw[:,4:]
            final["time"] = raw[:,0]
            time = final["time"]
            final["tracktime"] = time[trackix + 1]
            final["tracktime"][0] = time[0]
                # Numba + vectorized version of check_interval
            final["interval"] = mm.check_interval_vec(raw, final["spin"])
                # Vectorized version of root_getter
            roots = mm.root_getter_vec(final["energy"], final["phi_momentum"], final["carter"], final["spin"])
            final["pot_min"] = roots[1][:, -1]
            incstuff = np.mean(np.abs(roots[2][:, 1:3]), axis=1)
            final["inc"] = np.arccos(np.where(incstuff <= 1.0, incstuff, 1.0))*np.sign(final['phi_momentum'])
            final["it"] = roots[0][:, -2]
            final["ot"] = roots[0][:, -1]
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
                cos2theta = np.cos(raw[:, 2])**2
                ind_cross = argrelmin(cos2theta)
                ind2 = ind_cross[0][::2] if raw[ind_cross[0][0], 6] < 0.0 else ind_cross[0][1::2]
                ind3 = ind_cross[0][1::2] if raw[ind_cross[0][0], 6] < 0.0 else ind_cross[0][::2]
                
                asc_node, asc_node_time = raw[ind2,3] - 2*np.pi*np.arange(len(ind2)), raw[ind2,0] #subtract the normal phi advancement
                des_node, des_node_time = raw[ind3,3] - 2*np.pi*np.arange(len(ind3)), raw[ind3,0] #subtract the normal phi advancement
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

def delete_emri_data(filename, folder=None, auto=False):
    """
    Delete a saved HDF5 file and its associated index entry

    Parameters
    ----------
    filename : str
        name of HDF5 file data is saved to.
    folder : str or Path, optional
        Directory to search. Defaults to EMRICode/saved_sims/.
    auto : bool, optional
        If False, requires user to manually verify deletion request.
        Otherwise, deletes file without user input.

    Returns
    -------
    bool
        True for successful deletion, False otherwise.
    """
    path = resolve_data_file(filename, folder)

    if not path.exists():
        raise FileNotFoundError(f"EMRI data file does not exist: {path}")

    if not auto:
        confirm = input(
            f"Are you sure you want to delete {path.name}? (y/n): "
        ).strip().lower()

        if confirm != "y":
            print("File deletion aborted.")
            return False

    path.unlink()
    print(f"Deleted file: {path}")

    directory = path.parent
    index_path = directory / "index.json"

    with FileLock(str(index_path) + ".lock"):
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)

            if path.name in index:
                del index[path.name]

                tmp_path = directory / "index.json.tmp"
                with tmp_path.open("w", encoding="utf-8") as f:
                    json.dump(index, f, indent=2)

                os.replace(tmp_path, index_path)

    return True

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

