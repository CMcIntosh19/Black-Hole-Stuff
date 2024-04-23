# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:30:25 2023

@author: camcinto
"""

import numpy as np
import MetricMathStreamline as mm
import matplotlib.pyplot as plt
from itertools import product
from scipy.signal import argrelmin

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
    return ene
    
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

def EMRIGenerator(a, mu, endflag, mass=1.0, err_target=1e-15, label="default", cons=False, velorient=False, vel4=False, params=False, pos=False, veltrue=False, units="grav", verbose=False):
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
    
    def viable_cons(constants, state, a):
        energy, lz, cart = constants
        coeff = np.array([energy**2 - 1, 2, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*((a*energy - lz)**2 + cart), -cart*(a**2)])
        coeff2 = np.polyder(coeff)
        flats = np.roots(coeff2)
        flats = flats.real[abs(flats.imag)<err_target]
        pot_min = max(flats)
        potential_min = np.polyval(coeff, pot_min)
        return potential_min
    
    def get_true_anom(state, r0, e):
        pre = np.sign((r0*(1 - e**2)/state[1] - 1)/e)
        val = np.arccos(pre*min(1.0, abs((r0*(1 - e**2)/state[1] - 1)/e)))
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
        if count < 20:
            print("RAGH", pot_min)
        elif count < 21:
            print(inputs)
        else:
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
    plunge = False
    def anglething(angle):
        return 0.5*np.pi - np.abs(angle%np.pi - np.pi/2)
    while (not(eval(newflag)) and i < 10**7):
        try:
            update = False
            condate = False
            first = True
          
            #Grab the current state
            state = all_states[i]  
            pot_min = tracker[-1][0]   
          
            #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
            if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001):
                plunge = True
                break
          
            #Runge-Kutta update using geodesic
            old_dTau = dTau
            skip = False
            while ((err_calc >= err_target) or (first == True)) and (skip == False):
                new_step = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, a)
                step_check = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, a) 
                err_calc = abs(1 - np.dot(new_step, step_check)/np.dot(new_step, new_step))
                E, L, C = constants[-1]
                # if (high inclination) AND ((very close to pole AND approaching pole) OR (dTau is very small AND dTau is monotonically non-increasing))
                if np.sign(new_step[6])*(np.pi/2 - new_step[2]%np.pi) <= -1.55 and np.mean(dTau_change[-10:]) <= 0.001*np.mean(dTau_change):
                    new_step[0] += ((new_step[0] - state[0])/abs(new_step[2] - state[2]))*(2*anglething(new_step[2]))
                    new_step[3] += 2*np.arccos(np.sin(abs(np.pi/2 - np.arccos(L/np.sqrt(L**2 + C))))/ np.sin(new_step[2]))
                    new_step[6] = -new_step[6]
                    break
        
                old_dTau, dTau = dTau, min(dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
                if dTau <= 0.0:
                    dTau = old_dTau
                first = False
            
            metric = mm.kerr(new_step, a)[0]
            test = mm.check_interval(mm.kerr, new_step, a)
            looper = 0
            while (abs(test+1)>(err_target) or new_step[4] < 0.0) and looper <10:
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
            #Whenever you pass from one side of pot_min to the other, mess with the effective potential.
            #if ( np.sign(new_step[1] - pot_min) != orbitside) or ((new_step[3] - all_states[tracker[-1][-1]][3] > np.pi*(3/2)) and (np.std([state[1] for state in all_states[tracker[-1][-1]:]]) < 0.01*np.mean([state[1] for state in all_states[tracker[-1][-1]:]]))):
            R0, ECC = 0.5*(inner_turn + outer_turn), (outer_turn - inner_turn)/(outer_turn + inner_turn)
            compl, comph = np.arccos(-ECC), 2*np.pi - np.arccos(-ECC)
            S1, S2 = get_true_anom(state, R0, ECC), get_true_anom(new_step, R0, ECC)
            if (np.sign(new_step[1] - pot_min) != orbitside) or ((S2-compl)*(compl-S1) > 0 or (S2-comph)*(comph-S1) > 0):
                if (i - tracker[-1][-1] > 2):
                    update = True
                    if ( np.sign(new_step[1] - pot_min) != orbitside):
                        orbitside *= -1
                    if mu != 0.0:
                        condate = True
                        dcons = mm.peters_integrate6(all_states[tracker[-1][-1]:i], a, mu, tracker[-1][-1], i)
                        new_step, ch_cons = mm.new_recalc_state6(constants[-1], dcons, new_step, a)
                        pot_min = viable_cons(ch_cons, new_step, a)
                        while pot_min < -err_target:
                            print("tick?")
                            Lphi, ro = *ch_cons[1], pot_min[1]
                            ch_cons[0] += max(10**(-16), 2*(-pot_min[0])*((2*ro*((ro**3 + ro*(a**2) + 2*(a**2))*ch_cons[0] - 2*Lphi*a))**(-1)))
                            new_step = mm.recalc_state(ch_cons, new_step, a)
                            pot_min = viable_cons(ch_cons, new_step, a)
                        
                    
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
            progress = max(1 - abs(eval(terms[2]) - eval(termdict[terms[0]]))/eval(terms[2]), i/(10**7) ) * 100
            progress = max( abs((eval(termdict[terms[0]]) - initflagval)/(eval(terms[2]) - initflagval)), i/(10**7)) * 100
            if verbose == True:
                if (progress >= milestone):
                    print("Program has completed " + str(round(eval(termdict[terms[0]]), 2)), ",", str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
                    milestone = int(progress) + 1
        
        #Lets you end the program before the established end without breaking anything
        except KeyboardInterrupt:
            print("Ending program")
            stop = True
            cap = len(all_states) - 1
            all_states = all_states[:cap]
            interval = interval[:cap]
            dTau_change = dTau_change[:cap]
            constants = constants[:cap]
            qarter = qarter[:cap]
            freqs = freqs[:cap]
            break
        except Exception as e:
            print("Ending program - ERROR")
            print(e)
            stop = True
            cap = len(all_states) - 1
            all_states = all_states[:cap]
            interval = interval[:cap]
            dTau_change = dTau_change[:cap]
            constants = constants[:cap]
            qarter = qarter[:cap]
            freqs = freqs[:cap]
            break
    print(len(issues), len(all_states))
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
        
    constants = np.array([entry*np.array([mass*mu*(c**2), mass*mass*mu*G/c, (mass*mass*mu*G/c)**2]) for entry in np.array(constants)], dtype=np.float64)
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
    true_anom = np.array(true_anom)
    if max(all_states[:,2]) - min(all_states[:,2]) > 1e-15:
        theta_derv = np.interp(all_states[:,0], 0.5*(all_states[:,0][:-1] + all_states[:,0][1:]), np.diff(all_states[:,2])/np.diff(all_states[:,0]))
        ind2 = argrelmin(theta_derv)[0] #indices for the ascending node
        ind3 = argrelmin(-theta_derv)[0] #indices for the descending node
        asc_node, asc_node_time = all_states[ind2,3] - 2*np.pi*np.arange(len(ind2)), all_states[ind2,0] #subtract the normal phi advancement
        try:
            if ind2[0] > ind3[0]: #if the ascending node occurs after the descending node
                #ascending node should be first because of how the program starts on default
                asc_node = asc_node - np.ones(len(ind2))*2*np.pi #subtract a bit more for when comparing
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
             "tracktime": tracker[:,5],
             "trackix": tracker[:,6],
             "omega": omega,
             "otime": otime,
             "asc_node": asc_node,
             "asc_node_time": asc_node_time,
             "stop": stop,
             "plunge": plunge,
             "issues": issues}
    return final

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