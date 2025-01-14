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

def EMRIGenerator(a, mu, endflag="radius < 2", mass=1.0, err_target=1e-15, label="default", cons=False, velorient=False, vel4=False, params=False, pos=False, veltrue=False, units="grav", verbose=False, eps=1e-5, conch=6, trigger=2, override=False, bonk=1, bonk2=True):
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
    dTau = 0.1*np.abs(np.real((inner_turn/200)**(2)))
    dTau_change = [dTau]                                                #create dTau tracker
    borken = 0
    initflagval = eval(termdict[terms[0]])
    plunge, unbind = False, False
    def anglething(angle):
        return 0.5*np.pi - np.abs(angle%np.pi - np.pi/2)

    if verbose == False:
        pbar = tqdm(total = 10000000, position=0)
        pbar.set_postfix_str("Semilat: %s, Ecc %s" %(np.round( 0.5*(tracker[0][3] + tracker[0][4])*(1 - tracker[0][1]**2), 3), np.round(tracker[0][1], 3)))
    progress = 0
    while (not(eval(newflag)) and (i < 10**7 or override)):
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
                if np.sign(new_step[6])*(np.pi/2 - new_step[2]%np.pi) <= -89.5*(np.pi/180) and np.mean(dTau_change[-10:]) <= 0.001*np.mean(dTau_change):
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
            #if (smooth and cond[trigger]) or (not smooth and (state[3]%(2*np.pi) < np.pi and new_step[3]%(2*np.pi) > np.pi)):
            if (cond[trigger] and true_anom[-1] - true_anom[tracker[-1][-1]] > 0.5*np.pi) or ((S2 - true_anom[tracker[-1][-1]] > 4*np.pi and abs(new_step[1] - pot_min) < 0.5*max(outer_turn - pot_min, pot_min - inner_turn))):
                if (i - tracker[-1][-1] > 10):
                    #if not smooth:
                        #print("heyy", new_step[1])
                    update = True
                    if ( np.sign(new_step[1] - pot_min) != orbitside):
                        orbitside *= -1
                    if mu != 0.0:
                        #print("(%s and %s) or (%s and (%s and %s))"%(smooth, cond[trigger], (not smooth), (state[3]%(2*np.pi) < np.pi), (new_step[3]%(2*np.pi) > np.pi)))
                        #print("(%s) or ((%s and %s))"%(cond[trigger], (S2 - true_anom[tracker[-1][-1]] > 2*np.pi), (abs(state[1] - pot_min) < 0.5*max(outer_turn - pot_min, pot_min - inner_turn))), state[0])
                        condate = True
                        #print(inner_turn, new_step[1], outer_turn)
                        if "wonk" in label:
                            dcons = mm.peters_integrate6_3(all_states[tracker[-1][-1]:i], a, mu, tracker[-1][-1], i)
                        elif "wink" in label:
                            #print("hello")
                            dcons = mm.peters_integrate6_4(all_states[tracker[-1][-1]:i], a, mu, tracker[-1][-1], i)
                        elif "wunk" in label:
                            #print("hello")
                            dcons = mm.peters_integrate6_5(all_states[tracker[-1][-1]:i], a, mu, tracker[-1][-1], i)
                        else:
                            dcons = mm.peters_integrate6(all_states[tracker[-1][-1]:i], a, mu, tracker[-1][-1], i)
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
                    if verbose == False:
                        pbar.set_postfix_str("Semilat: %s, Ecc %s" %(np.round( 0.5*(tracker[-1][3] + tracker[-1][4])*(1 - tracker[-1][1]**2), 3), np.round(tracker[-1][1], 3)))
                if True in np.iscomplex(tracker[-1]):
                    compErr += 1
                    issues.append((i, new_step[0]))  
            #print("not stuck!")
            interval.append(mm.check_interval(mm.kerr, new_step, a))
            false_constants.append([getEnergy(new_step, a), *getLs(new_step, mu)])
            dTau_change.append(old_dTau)
            all_states.append(new_step )    #update position and velocity
            anomval = get_true_anom(new_step, 0.5*(outer_turn + inner_turn), e) + orbCount*2*np.pi
            if anomval - true_anom[-1] < -np.pi:
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

import time

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
    if bonk == True:
        print("old")
    else:
        print("new")
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
          
            #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
            if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001):
                plunge = True
                break
            
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
