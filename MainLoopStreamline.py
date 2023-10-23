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
    metric, chris = mm.kerr(state, a)
    stuff = np.matmul(metric, state[4:])
    ene = -stuff[0]
    #print(stuff)
    return ene
    
def getLs(state, mu):
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

def clean_inspiral3(mass, a, mu, endflag, err_target, label="default", cons=False, velorient=False, vel4=False, params=False, pos=False, veltrue=False, units="grav", verbose=False):
    #basically the same as schwarz, except references to schwarz changed to kerr
    
    def viable_cons(constants, state, a):
        energy, lz, cart = constants
        coeff = np.array([energy**2 - 1, 2, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*((a*energy - lz)**2 + cart), -cart*(a**2)])
        coeff2 = np.array([(energy**2 - 1)*4, (2.0)*3, ((a**2)*(energy**2 - 1) - lz**2 - cart)*2, 2*((a*energy - lz)**2) + 2*cart])
        r0 = max(np.roots(coeff2))
        potential_min = np.polyval(coeff, r0)
        return potential_min
    
    termdict = {"time": "all_states[i][0]",
                "phi_orbits": "abs(all_states[i][3]/(2*np.pi))",
                "radial_orbits": "orbitCount",
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
        print("Normalizing initial state")
        all_states[0], cons = mm.set_u_kerr(a, cons, velorient, vel4, params, pos, units)      #normalize initial state so it's actually physical
    pro = np.sign(all_states[0][7])
    
    interval = [mm.check_interval(mm.kerr, all_states[0], a)]           #create interval tracker
    metric = mm.kerr(all_states[0], a)[0]                                      #initial metric
    
    if np.shape(cons) == (3,):
        initE, initLz, initC = cons
        initQ = initC + (a*initE - initLz)**2
    else:
        initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
        initLz = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
        initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
        initC = initQ - (a*initE - initLz)**2                                          #initial adjusted Carter constant 
    pot_min = viable_cons([initE, initLz, initC], all_states[0], a)
    if pot_min < 0.0:
        print("RAGH")
        return False
                
    coeff = np.array([initE**2 - 1, 2.0, (a**2)*(initE**2 - 1) - initLz**2 - initC, 2*((a*initE - initLz)**2) + 2*initC, -initC*(a**2)])
    coeff2 = np.array([(initE**2 - 1)*4, 6.0, ((a**2)*(initE**2 - 1) - initLz**2 - initC)*2, 2*((a*initE - initLz)**2) + 2*initC])
    r0, inner_turn, outer_turn = np.sort(np.roots(coeff2))[-1], *np.sort(np.real(np.roots(coeff)))[-2:]
    e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
    A = (a**2)*(1 - initE**2)
    z2 = ((A + initLz**2 + initC) - ((A + initLz**2 + initC)**2 - 4*A*initC)**(1/2))/(2*A) if A != 0 else initC/(initLz**2 + initC)
    inc = np.arccos(np.sqrt(z2))
    tracker = [[r0, e, inc, inner_turn, outer_turn, all_states[0][0], 0]]
    if True in np.iscomplex(tracker[0]):
        initE = (4*a*initLz*r0 + ((4*a*initLz*r0)**2 - 4*(r0**4 + 2*r0*(a**2))*((a*initLz)**2 - (r0**2 - 2*r0 + a**2)*(r0**2 + initLz**2 + initC)))**(0.5))/(2*(r0**4 + 2*r0*(a**2)))
        coeff = np.array([initE**2 - 1, 2.0, (a**2)*(initE**2 - 1) - initLz**2 - initC, 2*((a*initE - initLz)**2) + 2*initC, -initC*(a**2)])
        coeff2 = np.array([(initE**2 - 1)*4, (2.0)*3, ((a**2)*(initE**2 - 1) - initLz**2 - initC)*2, 2*((a*initE - initLz)**2) + 2*initC])
        r0, inner_turn, outer_turn = np.sort(np.roots(coeff2))[-1], *np.sort(np.roots(coeff))[-2:]
        e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
        A = (a**2)*(1 - initE**2)
        z2 = ((A + initLz**2 + initC) - ((A + initLz**2 + initC)**2 - 4*A*initC)**(1/2))/(2*A) if A != 0 else initC/(initLz**2 + initC)
        inc = np.arccos(np.sqrt(z2))
        tracker = [[r0, e, inc, inner_turn, outer_turn, all_states[0][0], 0]]
    constants = [ np.array([initE,      #energy   
                            initLz,      #angular momentum (axial)
                            initC]) ]    #Carter constant (C)
    qarter = [initQ]           #Carter constant (Q)
    
    false_constants = [np.array([getEnergy(all_states[0], a), *getLs(all_states[0], mu)])]  #Cartesian approximation of L vector
               
    #tracker = [[r0, e, inner_turn, outer_turn, all_states[0][0], 0]]
               #r0, eccentricity, turning points, timestamp, index
    
    compErr = 0
    milestone = 0
    issues = []
    orbitside = np.sign(all_states[0][1] - r0)
    if orbitside == 0:
        orbitside = -1
    
    orbitCount = all_states[-1][0]/(2*np.pi/(((r0**(3/2) + pro*a)**(-1))*(1 - (6/r0) + pro*(8*a*(r0**(-3/2))) - (3*((a/r0)**(2))))**(0.5)))
    stop = False
    
    if label == "default":
        label = "r" + str(r0) + "e" + str(e) + "zU+03C0" + str(inc/np.pi) + "mu" + str(mu) + "a" + str(a)
    
    #Main Loop
    dTau = np.abs(np.real((inner_turn/200)**(2)))
    dTau_change = [dTau]                                                #create dTau tracker
    borken = 0
    def anglething(angle):
        return 0.5*np.pi - np.abs(angle%np.pi - np.pi/2)
    while (not(eval(newflag)) and i < 10**7):
        try:
            update = False
            condate = False
            first = True
          
            #Grab the current state
            state = all_states[i]  
            r0 = tracker[-1][0]   
          
            #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
            if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001):
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
                    print("POW, ind=", i)
                    print(new_step)
                    new_step[0] += ((new_step[0] - state[0])/abs(new_step[2] - state[2]))*(2*anglething(new_step[2]))
                    new_step[3] += 2*np.arccos(np.sin(abs(np.pi/2 - np.arccos(L/np.sqrt(L**2 + C))))/ np.sin(new_step[2]))
                    new_step[6] = -new_step[6]
                    print(new_step)
                    break
        
                old_dTau, dTau = dTau, min(dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
                if dTau <= 0.0:
                    dTau = old_dTau
                first = False

            metric = mm.kerr(new_step, a)[0]
            test = mm.check_interval(mm.kerr, new_step, a)
            looper = 0
            while (abs(test+1)>(err_target) or new_step[4] < 0.0) and looper <5:
                borken = borken + 1
                og_new_step = np.copy(new_step)
                gtt, gtp = metric[0,0], metric[0,3]
                disc = 4*(gtp*new_step[4]*new_step[7])**2 - 4*gtt*(new_step[4]**2)*(np.einsum('ij, i, j ->', metric[1:,1:], new_step[5:], new_step[5:]) + 1)
                delt = (-2*gtp*new_step[4]*new_step[7] - np.sqrt(disc))/(2*gtt*new_step[4]*new_step[4])
                new_step[4] *= delt
                test = mm.check_interval(mm.kerr, new_step, a)
                looper += 1
            if (test+1) > err_target or new_step[4] < 0.0:
                print("borked", looper)
                print(test+1, delt)
                new_step = np.copy(og_new_step)
   
            #constant modifying section
            #Whenever you pass from one side of r0 to the other, mess with the effective potential.
            if ( np.sign(new_step[1] - r0) != orbitside) or ((new_step[3] - all_states[tracker[-1][-1]][3] > np.pi*(3/2)) and (np.std([state[1] for state in all_states[tracker[-1][-1]:]]) < 0.01*np.mean([state[1] for state in all_states[tracker[-1][-1]:]]))):
                if (i - tracker[-1][-1] > 2):
                    update = True
                    if ( np.sign(new_step[1] - r0) != orbitside):
                        orbitside *= -1
                    orbitCount += 0.5
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
            #updatetime = time.time()
            if (update == True):
                if condate == False:
                    metric = mm.kerr(new_step, a)[0]
                    newE = -np.matmul(new_step[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
                    newLz = np.matmul(new_step[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
                    newQ = np.matmul(np.matmul(mm.kill_tensor(new_step, a), new_step[4:]), new_step[4:])    #new Carter constant Q
                    newC = newQ - (a*newE - newLz)**2                                                             #initial adjusted Carter constant  
                    coeff = np.array([newE**2 - 1, 2.0, (a**2)*(newE**2 - 1) - newLz**2 - newC, 2*((a*newE - newLz)**2 + newC), -newC*(a**2)])
                    coeff2 = np.array([4*(newE**2 - 1), 6.0, 2*((a**2)*(newE**2 - 1) - newLz**2 - newC), 2*((a*newE - newLz)**2 + newC)])
                    r0, inner_turn, outer_turn = max(np.roots(coeff2)), *np.sort(np.roots(coeff))[-2:]
                    e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
                    A = (a**2)*(1 - newE**2)
                    z2 = ((A + newLz**2 + newC) - ((A + newLz**2 + newC)**2 - 4*A*newC)**(1/2))/(2*A) if A != 0 else newC/(newLz**2 + newC)
                    inc = np.arccos(np.sqrt(z2))
                    tracker.append([r0, e, inc, inner_turn, outer_turn, new_step[0], i])
                    constants.append([newE, newLz, newC])
                    qarter.append(newQ)
                else:
                    constants.append(ch_cons)
                    qarter.append(ch_cons[2] + (a*ch_cons[0] - ch_cons[1])**2)
                    coeff = np.array([ch_cons[0]**2 - 1, 2.0, (a**2)*(ch_cons[0]**2 - 1) - ch_cons[1]**2 - ch_cons[2], 2*((a*ch_cons[0] - ch_cons[1])**2 + ch_cons[2]), -ch_cons[2]*(a**2)])
                    coeff2 = np.array([4*(ch_cons[0]**2 - 1), 6.0, 2*((a**2)*(ch_cons[0]**2 - 1) - ch_cons[1]**2 - ch_cons[2]), 2*((a*ch_cons[0] - ch_cons[1])**2 + ch_cons[2])])
                    r0, inner_turn, outer_turn = max(np.roots(coeff2)), *np.sort(np.roots(coeff))[-2:]
                    inner_turn, outer_turn = np.real(inner_turn), np.real(outer_turn)
                    e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
                    A = (a**2)*(1 - ch_cons[0]**2)
                    z2 = ((A + ch_cons[1]**2 + ch_cons[2]) - ((A + ch_cons[1]**2 + ch_cons[2])**2 - 4*A*ch_cons[2])**(1/2))/(2*A) if A != 0 else ch_cons[2]/(ch_cons[1]**2 + ch_cons[2])
                    inc = np.arccos(np.sqrt(z2))
                    tracker.append([r0, e, inc, inner_turn, outer_turn, new_step[0], i])
           
                if True in np.iscomplex(tracker[-1]):
                    compErr += 1
                    print("issue")
                    print(tracker[-1])
                    issues.append((i, new_step[0]))  
            interval.append(mm.check_interval(mm.kerr, new_step, a))
            false_constants.append([getEnergy(new_step, a), *getLs(new_step, mu)])
            dTau_change.append(old_dTau)
            all_states.append(new_step )    #update position and velocity
            i += 1
            progress = max(1 - abs(eval(terms[2]) - eval(termdict[terms[0]]))/eval(terms[2]), i/(10**7) ) * 100
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
    interval = np.array(interval)
    dTau_change = np.array([entry * (G*mass)/(c**3) for entry in dTau_change])
    all_states = np.array([entry*np.array([(G*mass)/(c**3), (G*mass)/(c**2), 1.0, 1.0, 1.0, c, (c**3)/(G*mass), (c**3)/(G*mass)]) for entry in np.array(all_states)]) 
    tracker = np.array([entry*np.array([(G*mass)/(c**2), 1.0, 1.0, (G*mass)/(c**2), (G*mass)/(c**2), (G*mass)/(c**3), 1]) for entry in tracker])
    r = all_states[0][1]
    ind = argrelmin(all_states[:,1])[0]
    omega, otime = np.diff(all_states[:,2][ind]) - 2*np.pi, np.diff(all_states[:,0][ind])
    
    if verbose == True:
        print("There were " + str(compErr) + " issues with complex roots/turning points.")
    final = {"name": label,
             "raw": all_states,
             "inputs": inputs,
             "pos": all_states[:,1:4],
             "all_vel": all_states[:,4:], 
             "time": all_states[:,0],
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
             "freqs": np.array([((r**(3/2) + pro*a)**(-1))*np.sqrt(1 - (6/r) + pro*(8*a*(r**(-3/2))) - (3*((a/r)**(2)))),
                                ((r**(3/2) + pro*a)**(-1))*np.sqrt(1 - pro*(4*a*(r**(-3/2))) + (3*((a/r)**(2)))),
                                ((r**(3/2) + pro*a)**(-1))]) * (c**3)/(G*mass),
             "r0": tracker[:,0],
             "e": tracker[:,1],
             "inc": tracker[:,2],
             "it": tracker[:,3],
             "ot": tracker[:,4],
             "tracktime": tracker[:,5],
             "trackix": tracker[:,6],
             "omegadot": omega/otime,
             "otime": all_states[:,0][ind][1:],
             "stop": stop,
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
    newdata = clean_inspiral3(mass, a, mu, endflag, err_target, label=label_old, pos=pos_new, veltrue=vel_new, verbose=verbose_new)
    
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
             "freqs": data["freqs"],
             "r0": np.concatenate((data["r0"], newdata["r0"])),
             "e": np.concatenate((data["e"], newdata["e"])),
             "it": np.concatenate((data["it"], newdata["it"])),
             "ot": np.concatenate((data["ot"], newdata["ot"])),
             "tracktime": np.concatenate((data["tracktime"], newdata["tracktime"])),
             "trackix": np.concatenate((data["trackix"], newdata["trackix"])),
             "omegadot": np.concatenate((data["omegadot"][:lastcross], newdata["omegadot"])),
             "otime": np.concatenate((data["otime"][:lastcross], newdata["otime"])),
             "stop": newdata["stop"],
             "issues": np.concatenate((data["issues"], newdata["issues"]))}
    return final

def dict_saver(data, filename):
    np.save(filename, data) 
    return True
        
def dict_from_file(filename):
    if ".npy" not in filename:
        filename = filename+".npy"
    data = np.load(filename, allow_pickle='TRUE').item()
    #print(read_dictionary['hello']) # displays "world"
    return data

def wrapsnap(data):
    #be, me = np.polynomial.polynomial.polyfit(list(data["tracktime"]), data["energy"], 1)
    #bl, ml = np.polynomial.polynomial.polyfit(list(data["tracktime"]), data["phi_momentum"], 1)
    #bc, mc = np.polynomial.polynomial.polyfit(list(data["tracktime"]), data["carter"], 1)
    br, mr = np.polynomial.polynomial.polyfit(list(data["tracktime"]), data["r0"], 1)
    be, me = np.polynomial.polynomial.polyfit(list(data["tracktime"]), data["e"], 1)
    bi, mi = np.polynomial.polynomial.polyfit(list(data["tracktime"]), data["inc"], 1)
    return mr, me, mi

def wrapwrap(rml, e0l, inl, spl, mul, skip=[]):
    try:
        total = len(rml)*len(e0l)*len(inl)*len(spl)*len(mul)
    except:
        print("Invalid parameter list detected")
        return {}, {}, {}
    if total == 0:
        print("Invalid parameter list detected")
        return {}, {}, {}
    condt = {}
    for i, j, k, l, m in product(rml, e0l, inl, spl, mul):
        rm, e0, inc, spin, mu = i, j, k, l, m
        if (rm, e0, inc, spin, mu) not in skip:
            t = np.sqrt(4*(np.pi**2)*(rm**3))
            try:
                test = clean_inspiral3(1.0, spin, mu, t*4, 0.1, 10**(-13), "test", params = [rm, e0, inc*np.pi/2], verbose=False)
                if test["time"][-1] < t*4:
                    if test["stop"] == True:
                        skip.append(False)
                        break
                    else:
                        print("Something's wonky.")
                else:
                    #edt[rm, e0, inc, spin, mu], ldt[rm, e0, inc, spin, mu], cdt[rm, e0, inc, spin, mu] = wrapsnap(test)
                    condt[rm, e0, inc, spin, mu] = wrapsnap(test)
            except:
                print("Didn't work?")
        else:
            print("Already run!")
        print("---")
    skip = skip + list(condt.keys())
    return condt, skip

def wrapwrap2(rml, e0l, inl, spl, mul, endflag, skip=[]):
    try:
        total = len(rml)*len(e0l)*len(inl)*len(spl)*len(mul)
    except:
        print("Invalid parameter list detected")
        return {}, {}, {}
    if total == 0:
        print("Invalid parameter list detected")
        return {}, {}, {}
    paramdt = {}
    for i, j, k, l, m in product(rml, e0l, inl, spl, mul):
        rm, e0, inc, spin, mu = i, j, k, l, m
        if (rm, e0, inc, spin, mu) not in skip:
            try:  
                print(rm, e0, inc, spin, mu)
                test = clean_inspiral3(1.0, spin, mu, endflag, 10**(-15), str((rm, e0, inc, spin, mu)), params = [rm, e0, inc], verbose=False)
                if test["stop"] == True:
                    break
                else:
                    paramdt[rm, e0, inc, spin, mu] = wrapsnap(test)
            except:
                print("Didn't work?")
        else:
            print("Already run!")
        print("---")
    skip = skip + list(paramdt.keys())
    return paramdt, skip

def dictslice(data, con, rm, e0, inc, spin, mu):
    #con = 0 for edot, 1 for ldot, 2 for cdot
    info = data.keys()
    params = [rm, e0, inc, spin, mu]
    itr = params.index("A")
    for i in range(len(params)):
        if params[i] != "A":
            info = [keys for keys in info if keys[i] == params[i]]
    xdat = [ind[itr] for ind in info]
    ydat = [data[ind][con] for ind in info]
    xdat, ydat = (list(t) for t in zip(*sorted(zip(xdat, ydat))))
    return xdat, ydat

def multislice(data, con, rm, e0, inc, spin, mu, legend = True, logx=False, logy=False):
    info = data.keys()
    params = [rm, e0, inc, spin, mu]
    conname = ["dE/dt", "dL/dt", "dC/dt"]
    paramname = ["R0", "Eccentricity", "Inclination", "Spin", "Mu"]
    itr1, itr2 = params.index("A"), params.index("B")
    spec = [paramname[i][:3] + " = " + str(params[i]) for i in range(5) if (params[i]!="A" and params[i]!="B")]
    blist = []
    pairs = []
    labs = []
    for i in range(len(params)):
        if (params[i] != "A") and (params[i] != "B"):
            info = [keys for keys in info if keys[i] == params[i]]
    for index in info:
        if index[itr2] not in blist:
            blist.append(index[itr2])
    temp = dict((k, data[k]) for k in info if k in data)
    for bval in blist:
        params[itr2] = bval
        pairs.append((*dictslice(temp, con, *params), paramname[itr2][:3] + " = " + str(bval)))
        labs.append(bval)
    
    labs, pairs = (list(t) for t in zip(*sorted(zip(labs, pairs))))
    fig, ax = plt.subplots()
    for pair in pairs:
        x = pair[0]
        y = pair[1]
        if logx == True:
            x = np.log(pair[0])/np.log(10)
        if logy == True:
            y = np.log(pair[1])/np.log(10)
        ax.plot(x, y, label=pair[2])
    if logx == True:
        paramname[itr1] = "Log(" + paramname[itr1] + ")"
    if logy == True:
        conname[con] = "Log(" + conname[con] + ")"
    ax.set_title("Variation in " + conname[con] + " vs " + paramname[itr1])
    if legend != False:
        ax.legend(title = ', '.join(spec)) if legend == True else ax.legend()
    return pairs