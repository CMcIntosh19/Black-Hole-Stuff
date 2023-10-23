# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:14:26 2022

@author: hepiz
"""

import numpy as np
import MetricMath as mm
import time
import matplotlib.pyplot as plt
from itertools import product
import json
from ast import literal_eval
import random
from scipy.signal import argrelmin
from scipy.signal import argrelmax


def just_constants(state, mass, a, mu, vstep, max_time, dTau, timelike, err_target, eta, xi, label, spec = False):
  inputs = (mass, mu, vstep, max_time, timelike, err_target, spec)          #Grab initial input in case you want to run the continue function
  start = np.copy(state)                                                        #Grab that initial state
  if (mm.check_interval(mm.kerr, start[0], mass, a) + 1) > err_target:
    print("Normalizing initial state")
    start = [mm.set_u_kerr(start[0], mass, a, timelike, eta*np.pi/180, xi*np.pi/180, special = spec)]      
                                                                                #normalize initial state so it's actually physical
  
  metric = mm.kerr(start[0], mass, a)[0]                                           #initial metric
  initE = -np.matmul(start[0][4:], np.matmul(metric, [1, 0, 0, 0]))             #initial energy
  initL = np.matmul(start[0][4:], np.matmul(metric, [0, 0, 0, 1]))              #initial angular momentum
  initQ = np.matmul(np.matmul(mm.kill_tensor(start[0], mass, a), start[0][4:]), start[0][4:])    
                                                                                #initial Carter constant Q
  initC = initQ - (a*initE - initL)**2                                          #initial adjusted Carter constant
                                     
  constants = [ np.array([initE,                                                #energy
                          initL,                                                #angular momentum
                          initC]) ]                                             #Carter constant (C)
  qarter = [initQ]                                                   #Carter constant (Q)

  r0_list = [0]
  e_list = [0]
  ot_list = [0]
  it_list = [0]

  all_time = [0]
  con_derv = [0, np.array([0,0,0]), 0, 0, 0, 0, 0, 0, 0]                                  #[index, [dedt, dldt, dcdt], r0, y, v, q, e, outer_turn, inner_turn]

  i, time, dt = 0, 0, start[0][4]*dTau                                          #initialize step counter and time
  milestone = 0
  while (time < max_time and i < 50*max_time):
    try:
      con_derv = [i, *mm.constant_derivatives(constants[i], mass, a, mu, start[0][1])]
      constants.append(constants[i] + con_derv[1]*dt)
      qarter.append(constants[i+1][2] + (a*constants[i+1][0] - constants[i+1][1])**2)
      r0_list.append(con_derv[2])
      e_list.append(con_derv[6])
      ot_list.append(con_derv[7])
      it_list.append(con_derv[8])
      i += 1
      time += dt
      all_time.append(time)
      progress = max(time/max_time, i/(50*max_time)) * 100
      if (progress >= milestone):
        print("Program has completed " + str(round(progress, 4)) + "% of full run: Index =" + str(i))
        milestone += 1

    except KeyboardInterrupt:
      print("Ending program")
      cap = len(constants) - 1
      constants = constants[:cap]
      qarter = qarter[:cap]
      r0_list = r0_list[:cap]
      e_list = e_list[:cap]
      ot_list = ot_list[:cap]
      it_list = it_list[:cap]
      all_time = all_time[:cap]
      break
  
  con_derv = [i-1, *mm.constant_derivatives(constants[i-1], mass, a, mu, start[0][1])]
  r0_list.append(con_derv[2])
  e_list.append(con_derv[6])
  ot_list.append(con_derv[7])
  it_list.append(con_derv[8])
  
  constants = np.array(constants)
  return {"name": label,
          "tracktime": np.array(all_time),
          "energy": constants[:,0],
          "phi_momentum": constants[:,1],
          "carter": constants[:,2],
          "qarter": np.array(qarter),
          "spin": a,
          "r0": np.array(r0_list[1:]),
          "e": np.array(e_list[1:]),
          "ot": np.array(ot_list[1:]),
          "it": np.array(it_list[1:])}

#version that uses sago explicit formulas, updates through default recalc_state function    
def inspiral_long(state, mass, a, mu, vstep, max_time, dTau, timelike, err_target, eta, xi, label, spec = False, verbose=True):  #basically the same as schwarz, except references to schwarz changed to kerr
  inputs = (mass, mu, vstep, max_time, timelike, err_target, spec)          #Grab initial input in case you want to run the continue function
  all_states = [np.copy(state)[0]]                                                   #Grab that initial state         
  err_calc = 1                                                                  #initialize calculated error
  pro = np.sign(float(eta))**(np.sign(float(eta)))                              # +1 if prograde/polar, -1 if retrograde
  i = 0                                                                         #initialize step counter

  print("Normalizing initial state")
  all_states[0] = mm.set_u_kerr(all_states[0], mass, a, timelike, eta*np.pi/180, xi*np.pi/180, special = spec)      #normalize initial state so it's actually physical

  interval = [mm.check_interval(mm.kerr, all_states[0], mass, a)]           #create interval tracker
  dTau_change = [dTau]                                                #create dTau tracker
  true_u_t = [all_states[0][4]]                                         #create u_t tracker (what it should be, not what it is)
  metric = mm.kerr(all_states[0], mass, a)[0]                                      #initial metric
  initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
  initL = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
  initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], mass, a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
  initC = initQ - (a*initE - initL)**2                                          #initial adjusted Carter constant                                   
  constants = [ np.array([initE,      #energy   
                          initL,      #angular momentum
                          initC]) ]    #Carter constant (C)
  qarter = [initQ]           #Carter constant (Q)
  '''
  various = np.array([[0,             #r0 - Semimajor axis of the orbit, equal to the radius if circular
                       0,             #y - Carter constant C divided by angular momentum squared
                       0,             #v - square root of mass divided by r0
                       0,             #q - spin divided by mass
                       0,             #e - eccentricity of the orbit
                       0,             #ot - apoapsis of the orbit, the "outer turn"
                       0,             #it - periapsis of the orbit, the "inner turn"
                       0]])           #tracktime - timestamp
  '''

  con_derv = [[0, *mm.constant_derivatives_long(constants[0], mass, a, mu), all_states[0][0]]]    # returns [index, [dedt, dldt, dcdt], r0, y, v, q, e, outer_turn, inner_turn, compErr, timestamp]
  r0 = con_derv[0][2]
  #con_derv[0][-1] = np.pi*(r0**(3/2) + a)/2
  #timestamp is set to a quarter-period to keep the half-orbit forced update from happening at max or min r
  #forced update at max/min r causes problem, near r0 is better
  #doing this because more circular orbits are wonky
  #but this might break stuff for non-circular orbits??
  #problem for later
  phys_avgs = [ np.array([all_states[0][1],
                          all_states[0][2],
                          all_states[0][3]]) ]
  
  compErr = 0
  milestone = 0
  rmin = all_states[0][1]
  orbitCount = np.round(20/(rmin))
  issues = []
  checker = []
  orbitside = np.sign(all_states[0][1] - con_derv[0][2])
  #Main Loop
  while (all_states[i][0] < max_time and i < 50*max_time):
    try:
      update = False
      first = True
      loop2 = False

      #Grab the current state
      state = all_states[i]  
      r0 = con_derv[-1][2]      

      #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
      if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001) :
        break
      elif state[1] >= state[1]*1000:    #Currently based on event horizon, might change it to be based on r0
        break
        
      #Runge-Kutta update using geodesic
      while (err_calc >= err_target) or (first == True):
        step_check = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, mass, a)
        new_step = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, mass, a) 
        mod_step = mm.np.append(step_check[0:6], mm.fix_sin(step_check[6])*mm.fix_cos(step_check[7]))
        mod_new = np.append(new_step[0:6], mm.fix_sin(new_step[6])*mm.fix_cos(new_step[7]))
        mod_state = np.append(state[0:6], mm.fix_sin(state[6])*mm.fix_cos(state[7]))
        mod_pre = mod_step - mod_new
        mod_pre[:4] = mod_pre[:4]/np.linalg.norm(mod_state[:4]) 
        mod_pre[4:] = mod_pre[4:]/np.linalg.norm(mod_state[4:])
        err_calc = np.linalg.norm(mod_pre)
        old_dTau, dTau = dTau, min(0.95 * dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
        first = False

      #new_step = recalc_state(constants[-1], new_step, mass, a)


      #Whenever you pass from one side of r0 to the other, mess with the effective potential.
      if i == 0:
        flip = False       
      else:
        old_accel = np.sign((state[5] - all_states[i-1][5])/(state[0] - all_states[i-1][0]))
        new_accel = np.sign((new_step[5] - state[5])/(new_step[0] - state[0]))
        if old_accel == new_accel:
          flip = False
        else:
          flip = True
      if (( np.sign(new_step[1] - r0) != orbitside ) or (flip)):  #(new_step[0] - con_derv[-1][10]) > np.pi*(r0**(3/2) + a) or
        orbitside = np.sign(new_step[1] - r0)
        update = True
        dt = new_step[0] - con_derv[-1][10]
        if dt == 0:
          print("holler", new_step[0])
        new_con = constants[-1] + con_derv[-1][1]*dt                                #Figure out how the constants have changed!
        dir1 = new_step[5]
        new_step = mm.recalc_state(new_con, new_step, mass, a)                           #Change the velocity so it actually makes sense with those constants
        test = mm.check_interval(mm.kerr, new_step, mass, a)
        if abs(test+1)>(err_target):
          print("HEY SOMETHING'S UP")
          print(i)
          print("behold")
          print(test)
          print(new_step)
          print(constants[-1])
          print("press enter to continue")
          #input()
          loop2 = True
          #break
        if dir1 != new_step[5]:
          checker.append([dir1, new_step[5], con_derv[-1][1]*dt  ])
      #Initializing for the next step
      #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
      #Only happens the step before the derivatives are recalculated.
      
      #new interval thing
      trial = 0
      while loop2 == True:
        ut, up = new_step[4], new_step[7]
        err = mm.check_interval(mm.kerr, new_step, mass, a)
        metric = mm.kerr(new_step, mass, a)[0]
        gtt, gtp = metric[0][0], metric[0][3]
        disc = np.sqrt(ut**2 + ((gtp/gtt)**2)*(up**2) + 2*(gtp/gtt)*ut*up - (1.0 + err)/gtt)
        int_steps = [np.copy(new_step), np.copy(new_step)]
        int_steps[0][4] += -ut - (gtp/gtt)*up + disc
        int_steps[1][4] += -ut - (gtp/gtt)*up - disc
        checks = [(mm.check_interval(mm.kerr, int_steps[0], mass, a)+1)**2, (mm.check_interval(mm.kerr, int_steps[1], mass, a)+1)**2]
        test = mm.check_interval(mm.kerr, int_steps[checks.index(min(checks))], mass, a)
        trial += 1
        if (abs(test+1)<=(err_target)) and (int_steps[checks.index(min(checks))][4] > 0):
          new_step = int_steps[checks.index(min(checks))]
          loop2 = False
        if trial >= 10:
          print("giving up!", trial, test)
          break 
              
      #Update stuff!
      if update == True:
        constants.append(new_con)
        qarter.append(new_con[2] + (a*new_con[0] - new_con[1])**2)
        con_derv.append([i, *mm.constant_derivatives_long(constants[-1], mass, a, mu), new_step[0]])
        if con_derv[-1][9] == True:
          compErr += 1
          issues.append((i, new_step[0]))

      interval.append(mm.check_interval(mm.kerr, new_step, mass, a))
      dTau_change.append(old_dTau)
      #true_u_t.append(set_u_kerr(new_step, mass, a, timelike, 0, 0, special = "circle")[4])
      metric = mm.kerr(new_step, mass, a)[0]
      all_states.append(new_step )    #update position and velocity
      #phys_avgs.append([np.average(all_states[:][1]),
      #                  np.average(all_states[:][2]),
      #                  np.average(all_states[:][3])])
      i += 1

      if new_step[1] < rmin:
        rmin = new_step[1]
      progress = max(new_step[0]/max_time, i/(50*max_time)) * 100
      if verbose == True:
          if (progress >= milestone):
            print("Program has completed " + str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
            milestone += 1
          if (new_step[3] >= orbitCount*2*np.pi):
            print("Number of orbits = " + str(round(new_step[3]/(2*np.pi))) + ", t = " + str(new_step[0]) + ", r_min = " + str(rmin))
            orbitCount += np.round(20/rmin)

    #Lets you end the program before the established end without breaking anything
    except KeyboardInterrupt:
      print("Ending program")
      cap = len(all_states) - 1
      all_states = all_states[:cap]
      interval = interval[:cap]
      dTau_change = dTau_change[:cap]
      true_u_t = true_u_t[:cap]
      constants = constants[:cap]
      qarter = qarter[:cap]
      #phys_avgs = phys_avgs[:cap]
      break

  r = all_states[0][1]
  constants = np.array(constants)
  qarter = np.array(qarter)
  con_derv = np.array(con_derv, dtype=object)
  interval = np.array(interval)
  dTau_change = np.array(dTau_change)
  true_u_t = np.array(true_u_t)
  all_states = np.array(all_states)
  phys_avgs = np.array(phys_avgs)
  if verbose == True:
      print("There were " + str(compErr) + " issues with complex roots/turning points.")
      for i in issues:
        print("Complex value at index " + str(i[0]) + ", t = " + str(i[1]))
      #print("Here's some data!")
      #print(checker)
  final = {"name": label,
           "raw": all_states,
           "inputs": inputs,
           "dTau": dTau,
           "r_av": phys_avgs[:,0],
           "theta_av": phys_avgs[:,1],
           "phi_av": phys_avgs[:,2],
           "pos": all_states[:,1:4],
           "all_vel": all_states[:,4:], 
           "time": all_states[:,0],
           "interval": interval,
           "vel": (np.square(all_states[:,5]) + np.square(all_states[:,1]) * (np.square(all_states[:,6]) + np.square(all_states[:,7])))**(0.5),
           "dTau_change": dTau_change,
           "ut check": true_u_t,
           "energy": constants[:, 0],
           "phi_momentum": constants[:, 1],
           "qarter":qarter,
           "carter": constants[:, 2],
           "spin": a,
           "freqs": np.array([((r**(3/2) + pro*a)**(-1))*np.sqrt(1 - (6/r) + pro*(8*a*(r**(-3/2))) - (3*((a/r)**(2)))),
                              ((r**(3/2) + pro*a)**(-1))*np.sqrt(1 - pro*(4*a*(r**(-3/2))) + (3*((a/r)**(2)))),
                              ((r**(3/2) + pro*a)**(-1))]),
           "r0": con_derv[:,2],
           "y": con_derv[:,3],
           "v": con_derv[:,4],
           "q": con_derv[:,5],
           "e": con_derv[:,6],
           "ot": con_derv[:,7],
           "it": con_derv[:,8],
           "tracktime": con_derv[:,10]}
  return final

#version that uses peters explicit formulas, updates through default recalc_state function
def inspiral_long1(state, mass, a, mu, vstep, max_time, dTau, timelike, err_target, eta, xi, label, spec = False, verbose=True):  #basically the same as schwarz, except references to schwarz changed to kerr
  inputs = (mass, mu, vstep, max_time, timelike, err_target, spec)          #Grab initial input in case you want to run the continue function
  all_states = [np.copy(state)[0]]                                                   #Grab that initial state         
  err_calc = 1                                                                  #initialize calculated error
  pro = np.sign(float(eta))**(np.sign(float(eta)))                              # +1 if prograde/polar, -1 if retrograde
  i = 0                                                                         #initialize step counter

  print("Normalizing initial state")
  all_states[0] = mm.set_u_kerr(all_states[0], mass, a, timelike, eta*np.pi/180, xi*np.pi/180, special = spec)      #normalize initial state so it's actually physical

  interval = [mm.check_interval(mm.kerr, all_states[0], mass, a)]           #create interval tracker
  dTau_change = [dTau]                                                #create dTau tracker
  true_u_t = [all_states[0][4]]                                         #create u_t tracker (what it should be, not what it is)
  metric = mm.kerr(all_states[0], mass, a)[0]                                      #initial metric
  initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
  initL = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
  initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], mass, a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
  initC = initQ - (a*initE - initL)**2                                          #initial adjusted Carter constant                                   
  constants = [ np.array([initE,      #energy   
                          initL,      #angular momentum
                          initC]) ]    #Carter constant (C)
  qarter = [initQ]           #Carter constant (Q)
  '''
  various = np.array([[0,             #r0 - Semimajor axis of the orbit, equal to the radius if circular
                       0,             #y - Carter constant C divided by angular momentum squared
                       0,             #v - square root of mass divided by r0
                       0,             #q - spin divided by mass
                       0,             #e - eccentricity of the orbit
                       0,             #ot - apoapsis of the orbit, the "outer turn"
                       0,             #it - periapsis of the orbit, the "inner turn"
                       0]])           #tracktime - timestamp
  '''

  con_derv = [[0, *mm.constant_derivatives_long4(constants[0], mass, a, mu), all_states[0][0]]]    # returns [index, [dedt, dldt, dcdt], r0, y, v, q, e, outer_turn, inner_turn, compErr, timestamp]
  r0 = con_derv[0][2]
  #con_derv[0][-1] = np.pi*(r0**(3/2) + a)/2
  #timestamp is set to a quarter-period to keep the half-orbit forced update from happening at max or min r
  #forced update at max/min r causes problem, near r0 is better
  #doing this because more circular orbits are wonky
  #but this might break stuff for non-circular orbits??
  #problem for later
  phys_avgs = [ np.array([all_states[0][1],
                          all_states[0][2],
                          all_states[0][3]]) ]
  
  compErr = 0
  milestone = 0
  rmin = all_states[0][1]
  orbitCount = np.round(20/(rmin))
  issues = []
  checker = []
  orbitside = np.sign(all_states[0][1] - con_derv[0][2])
  #Main Loop
  while (all_states[i][0] < max_time and i < 50*max_time):
    try:
      update = False
      first = True
      loop2 = False

      #Grab the current state
      state = all_states[i]  
      r0 = con_derv[-1][2]      

      #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
      if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001) :
        break
      elif state[1] >= state[1]*1000:    #Currently based on event horizon, might change it to be based on r0
        break
        
      #Runge-Kutta update using geodesic
      while (err_calc >= err_target) or (first == True):
        step_check = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, mass, a)
        new_step = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, mass, a) 
        mod_step = mm.np.append(step_check[0:6], mm.fix_sin(step_check[6])*mm.fix_cos(step_check[7]))
        mod_new = np.append(new_step[0:6], mm.fix_sin(new_step[6])*mm.fix_cos(new_step[7]))
        mod_state = np.append(state[0:6], mm.fix_sin(state[6])*mm.fix_cos(state[7]))
        mod_pre = mod_step - mod_new
        mod_pre[:4] = mod_pre[:4]/np.linalg.norm(mod_state[:4]) 
        mod_pre[4:] = mod_pre[4:]/np.linalg.norm(mod_state[4:])
        err_calc = np.linalg.norm(mod_pre)
        old_dTau, dTau = dTau, min(0.95 * dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
        first = False

      #new_step = recalc_state(constants[-1], new_step, mass, a)


      #Whenever you pass from one side of r0 to the other, mess with the effective potential.
      if i == 0:
        flip = False       
      else:
        old_accel = np.sign((state[5] - all_states[i-1][5])/(state[0] - all_states[i-1][0]))
        new_accel = np.sign((new_step[5] - state[5])/(new_step[0] - state[0]))
        if old_accel == new_accel:
          flip = False
        else:
          flip = True
      if (( np.sign(new_step[1] - r0) != orbitside ) or (flip)):  #(new_step[0] - con_derv[-1][10]) > np.pi*(r0**(3/2) + a) or
        orbitside = np.sign(new_step[1] - r0)
        update = True
        dt = new_step[0] - con_derv[-1][10]
        if dt == 0:
          print("holler", new_step[0])
        new_con = constants[-1] + con_derv[-1][1]*dt                                #Figure out how the constants have changed!
        dir1 = new_step[5]
        new_step = mm.recalc_state(new_con, new_step, mass, a)                           #Change the velocity so it actually makes sense with those constants
        test = mm.check_interval(mm.kerr, new_step, mass, a)
        if abs(test+1)>(err_target):
          print("HEY SOMETHING'S UP")
          print(i)
          print("behold")
          print(test)
          print(new_step)
          print(constants[-1])
          print("press enter to continue")
          #input()
          loop2 = True
          #break
        if dir1 != new_step[5]:
          checker.append([dir1, new_step[5], con_derv[-1][1]*dt  ])
      #Initializing for the next step
      #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
      #Only happens the step before the derivatives are recalculated.
      
      #new interval thing
      trial = 0
      while loop2 == True:
        ut, up = new_step[4], new_step[7]
        err = mm.check_interval(mm.kerr, new_step, mass, a)
        metric = mm.kerr(new_step, mass, a)[0]
        gtt, gtp = metric[0][0], metric[0][3]
        disc = np.sqrt(ut**2 + ((gtp/gtt)**2)*(up**2) + 2*(gtp/gtt)*ut*up - (1.0 + err)/gtt)
        int_steps = [np.copy(new_step), np.copy(new_step)]
        int_steps[0][4] += -ut - (gtp/gtt)*up + disc
        int_steps[1][4] += -ut - (gtp/gtt)*up - disc
        checks = [(mm.check_interval(mm.kerr, int_steps[0], mass, a)+1)**2, (mm.check_interval(mm.kerr, int_steps[1], mass, a)+1)**2]
        test = mm.check_interval(mm.kerr, int_steps[checks.index(min(checks))], mass, a)
        trial += 1
        if (abs(test+1)<=(err_target)) and (int_steps[checks.index(min(checks))][4] > 0):
          new_step = int_steps[checks.index(min(checks))]
          loop2 = False
        if trial >= 10:
          print("giving up!", trial, test)
          break
              
      #Update stuff!
      if update == True:
        constants.append(new_con)
        qarter.append(new_con[2] + (a*new_con[0] - new_con[1])**2)
        con_derv.append([i, *mm.constant_derivatives_long4(constants[-1], mass, a, mu), new_step[0]])
        if con_derv[-1][9] == True:
          compErr += 1
          issues.append((i, new_step[0]))

      interval.append(mm.check_interval(mm.kerr, new_step, mass, a))
      dTau_change.append(old_dTau)
      #true_u_t.append(set_u_kerr(new_step, mass, a, timelike, 0, 0, special = "circle")[4])
      metric = mm.kerr(new_step, mass, a)[0]
      all_states.append(new_step )    #update position and velocity
      #phys_avgs.append([np.average(all_states[:][1]),
      #                  np.average(all_states[:][2]),
      #                  np.average(all_states[:][3])])
      i += 1

      if new_step[1] < rmin:
        rmin = new_step[1]
      progress = max(new_step[0]/max_time, i/(50*max_time)) * 100
      if verbose == True:
          if (progress >= milestone):
            print("Program has completed " + str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
            milestone += 1
          if (new_step[3] >= orbitCount*2*np.pi):
            print("Number of orbits = " + str(round(new_step[3]/(2*np.pi))) + ", t = " + str(new_step[0]) + ", r_min = " + str(rmin))
            orbitCount += np.round(20/rmin)

    #Lets you end the program before the established end without breaking anything
    except KeyboardInterrupt:
      print("Ending program")
      cap = len(all_states) - 1
      all_states = all_states[:cap]
      interval = interval[:cap]
      dTau_change = dTau_change[:cap]
      true_u_t = true_u_t[:cap]
      constants = constants[:cap]
      qarter = qarter[:cap]
      #phys_avgs = phys_avgs[:cap]
      break

  r = all_states[0][1]
  constants = np.array(constants)
  qarter = np.array(qarter)
  con_derv = np.array(con_derv, dtype=object)
  interval = np.array(interval)
  dTau_change = np.array(dTau_change)
  true_u_t = np.array(true_u_t)
  all_states = np.array(all_states)
  phys_avgs = np.array(phys_avgs)
  if verbose == True:
      print("There were " + str(compErr) + " issues with complex roots/turning points.")
      for i in issues:
        print("Complex value at index " + str(i[0]) + ", t = " + str(i[1]))
      #print("Here's some data!")
      #print(checker)
  final = {"name": label,
           "raw": all_states,
           "inputs": inputs,
           "dTau": dTau,
           "r_av": phys_avgs[:,0],
           "theta_av": phys_avgs[:,1],
           "phi_av": phys_avgs[:,2],
           "pos": all_states[:,1:4],
           "all_vel": all_states[:,4:], 
           "time": all_states[:,0],
           "interval": interval,
           "vel": (np.square(all_states[:,5]) + np.square(all_states[:,1]) * (np.square(all_states[:,6]) + np.square(all_states[:,7])))**(0.5),
           "dTau_change": dTau_change,
           "ut check": true_u_t,
           "energy": constants[:, 0],
           "phi_momentum": constants[:, 1],
           "qarter":qarter,
           "carter": constants[:, 2],
           "spin": a,
           "freqs": np.array([((r**(3/2) + pro*a)**(-1))*np.sqrt(1 - (6/r) + pro*(8*a*(r**(-3/2))) - (3*((a/r)**(2)))),
                              ((r**(3/2) + pro*a)**(-1))*np.sqrt(1 - pro*(4*a*(r**(-3/2))) + (3*((a/r)**(2)))),
                              ((r**(3/2) + pro*a)**(-1))]),
           "r0": con_derv[:,2],
           "y": con_derv[:,3],
           "v": con_derv[:,4],
           "q": con_derv[:,5],
           "e": con_derv[:,6],
           "ot": con_derv[:,7],
           "it": con_derv[:,8],
           "tracktime": con_derv[:,10]}
  return final

def inspiral_long_resume(original, add_time):
  state, a, dTau = [original["raw"][-1]], original["spin"], original["dTau"]
  mass, mu, vstep, max_time, timelike, err_target, circular = original["inputs"]
  eta, xi = np.arctan(state[0][7]/state[0][5]), np.arctan(state[0][7]/state[0][5])
  ene, lel, car = original["energy"][-1], original["phi_momentum"][-1], original["carter"][-1]
  constate = [[*original["raw"][-1, :4], ene, lel, car]]
  print(constate)
  print(circular, err_target, timelike, max_time, vstep, mu, mass, ene, lel, car)
  
  next = inspiral_long(constate, mass, a, mu, vstep, original["time"][-1] + add_time, dTau, timelike, err_target, circular, eta, xi)

  if len(next["energy"]) < 2:
    print("")
    return original

  final = {"name": original["name"],
           "raw": np.append(original["raw"][:-1], next["raw"], axis=0),
           "inputs": original["inputs"],
           "dTau": next["dTau"],
           "r_av": np.append(original["r_av"][:-1], next["r_av"]),
           "theta_av": np.append(original["theta_av"][:-1], next["theta_av"]),
           "phi_av": np.append(original["phi_av"][:-1], next["phi_av"]),
           "pos": np.append(original["pos"][:-1], next["pos"], axis=0),
           "all_vel": np.append(original["all_vel"][:-1], next["all_vel"], axis=0), 
           "time": np.append(original["time"][:-1], next["time"], axis=0),
           "interval": np.append(original["interval"][:-1], next["interval"], axis=0),
           "vel": np.append(original["vel"][:-1], next["vel"], axis=0),
           "dTau_change": np.append(original["dTau_change"][:-1], next["dTau_change"], axis=0),
           "ut check": np.append(original["ut check"][:-1], next["ut check"], axis=0),
           "energy": np.append(original["energy"][:-1], next["energy"], axis=0),
           "phi_momentum": np.append(original["phi_momentum"][:-1], next["phi_momentum"], axis=0),
           "qarter":np.append(original["qarter"][:-1], next["qarter"], axis=0),
           "carter": np.append(original["carter"][:-1], next["carter"], axis=0),
           "spin": a,
           "freqs": original["freqs"],
           "r0": np.append(original["r0"][:-1], next["r0"], axis=0),
           "y": np.append(original["y"][:-1], next["y"], axis=0),
           "v": np.append(original["v"][:-1], next["v"], axis=0),
           "q": np.append(original["q"][:-1], next["q"], axis=0),
           "e": np.append(original["e"][:-1], next["e"], axis=0),
           "ot": np.append(original["ot"][:-1], next["ot"], axis=0),
           "it": np.append(original["it"][:-1], next["it"], axis=0),
           "tracktime": np.append(original["tracktime"][:-1], next["tracktime"], axis=0)}
  return final

#version that uses integrated peters formulas, updates through default recalc_state function
def inspiral_long2(state, mass, a, mu, vstep, max_time, dTau, timelike, err_target, eta, xi, label, spec = False, verbose=True):  #basically the same as schwarz, except references to schwarz changed to kerr
  inputs = (mass, mu, vstep, max_time, timelike, err_target, spec)          #Grab initial input in case you want to run the continue function
  all_states = [np.copy(state)[0]]                                                   #Grab that initial state         
  err_calc = 1                                                                  #initialize calculated error
  pro = np.sign(float(eta))**(np.sign(float(eta)))                              # +1 if prograde/polar, -1 if retrograde
  i = 0                                                                         #initialize step counter

  print("Normalizing initial state")
  all_states[0] = mm.set_u_kerr(all_states[0], mass, a, timelike, eta*np.pi/180, xi*np.pi/180, special = spec)      #normalize initial state so it's actually physical

  interval = [mm.check_interval(mm.kerr, all_states[0], mass, a)]           #create interval tracker
  dTau_change = [dTau]                                                #create dTau tracker
  true_u_t = [all_states[0][4]]                                         #create u_t tracker (what it should be, not what it is)
  metric = mm.kerr(all_states[0], mass, a)[0]                                      #initial metric
  initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
  initL = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
  initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], mass, a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
  initC = initQ - (a*initE - initL)**2                                          #initial adjusted Carter constant                                   
  constants = [ np.array([initE,      #energy   
                          initL,      #angular momentum
                          initC]) ]    #Carter constant (C)
  qarter = [initQ]           #Carter constant (Q)
  '''
  various = np.array([[0,             #r0 - Semimajor axis of the orbit, equal to the radius if circular
                       0,             #y - Carter constant C divided by angular momentum squared
                       0,             #v - square root of mass divided by r0
                       0,             #q - spin divided by mass
                       0,             #e - eccentricity of the orbit
                       0,             #ot - apoapsis of the orbit, the "outer turn"
                       0,             #it - periapsis of the orbit, the "inner turn"
                       0]])           #tracktime - timestamp
  '''

  con_derv = [[0, *mm.peters_integrate(constants[0], a, mu, all_states, 0, 0), all_states[0][0]]]    # returns [index, [dedt, dldt, dcdt], r0, y, v, q, e, outer_turn, inner_turn, compErr, timestamp]
  
  #r0 = con_derv[0][2]
  #con_derv[0][-1] = np.pi*(r0**(3/2) + a)/2
  #timestamp is set to a quarter-period to keep the half-orbit forced update from happening at max or min r
  #forced update at max/min r causes problem, near r0 is better
  #doing this because more circular orbits are wonky
  #but this might break stuff for non-circular orbits??
  #problem for later
  
  
  compErr = 0
  milestone = 0
  rmin = all_states[0][1]
  orbitCount = np.round(20/(rmin))
  issues = []
  checker = []
  orbitside = np.sign(all_states[0][1] - con_derv[0][2])
  in_disk = 0
  #Main Loop
  while (all_states[i][0] < max_time and i < 50*max_time):
    try:
      update = False
      first = True
      loop2 = False
      roche = ((1/mu)**(1/3)) * (472393*mu)
          # (472393*mu) is a rough approx of 1 solar radius for a given mu

      #Grab the current state
      state = all_states[i]  
      r0 = con_derv[-1][2]      

      #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
      if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001) :
        break
      elif state[1] >= state[1]*1000:    #Currently based on event horizon, might change it to be based on r0
        break
        
      #Runge-Kutta update using geodesic
      while (err_calc >= err_target) or (first == True):
        step_check = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, mass, a)
        new_step = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, mass, a) 
        mod_step = mm.np.append(step_check[0:6], mm.fix_sin(step_check[6])*mm.fix_cos(step_check[7]))
        mod_new = np.append(new_step[0:6], mm.fix_sin(new_step[6])*mm.fix_cos(new_step[7]))
        mod_state = np.append(state[0:6], mm.fix_sin(state[6])*mm.fix_cos(state[7]))
        mod_pre = mod_step - mod_new
        mod_pre[:4] = mod_pre[:4]/np.linalg.norm(mod_state[:4]) 
        mod_pre[4:] = mod_pre[4:]/np.linalg.norm(mod_state[4:])
        err_calc = np.linalg.norm(mod_pre)
        old_dTau, dTau = dTau, min(0.95 * dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
        first = False

      #constant modifying section
      '''
      #First check: is it inside the accretion disk? (Annulus from 3rs (6gu) to ??? with some (small) thickness)
      if ((new_step[1] >= 6) and (new_step[1] <= 10)) and (abs(new_step[2] - np.pi/2) <= 0.05):
          r_star = 472393*mu                 #approx 1 solar radius
          density = 2.29*(10**(-9))*(mu**2)    #approx surface density of 1 solar mass spread between 6gu and 10gu (Assumed constant)
          md = np.pi*(r_star**2)*density     #mass accreted by star passing through disk (estimated by "hole punch method")
          pos = new_step[:4]                 #position
          print(pos, "pos")
          print(new_step[4:], "vel")
          disk_state = np.array(mm.set_u_kerr(new_step, 1.0, a, True, 90, 90, special="circle"))
                                             #4-velocity of disk at given r
          pd = md * disk_state[4:]
          pstar = mu * new_step[4:]   #4-momenta of star and disk segment
          pnew = pd + pstar                  #combined 4-momentum
          print(pnew, "pnew")
          mu = (-mm.check_interval(mm.kerr, [*pos, *pnew], 1.0, 0.0))**(1/2)
          new_state = np.array([*pos, *(pnew/mu)])

          new_con = mm.find_constants(new_state, 1.0, a)
          update = True
        
      #Second check: is it at/within the tidal Roche limit?
      #if (new_step[1] <= roche):
          #in_disk = 3
          #tidal disrupt function
      #'''
      #Whenever you pass from one side of r0 to the other, mess with the effective potential.
      if (( np.sign(new_step[1] - r0) != orbitside )):  # or (flip)):  #(new_step[0] - con_derv[-1][10]) > np.pi*(r0**(3/2) + a) or
        orbitside = np.sign(new_step[1] - r0)
        update = True
        con_derv.append([i, *mm.peters_integrate(constants[-1], a, mu, all_states, con_derv[-1][0], i), new_step[0]])
        new_con = constants[-1] + con_derv[-1][1]                                   #Figure out how the constants have changed!
        dir1 = new_step[5]
        new_step = mm.new_recalc_state(new_con, new_step, mass, a)                           #Change the velocity so it actually makes sense with those constants
        test = mm.check_interval(mm.kerr, new_step, mass, a)
        if abs(test+1)>(err_target):
          print("HEY SOMETHING'S UP")
          print(i)
          print("behold")
          print(test)
          print(new_step)
          print(constants[-1])
          print("press enter to continue")
          #input()
          loop2 = True
          #break
        if dir1 != new_step[5]:
          checker.append([dir1, new_step[5], con_derv[-1][1]  ])
      #Initializing for the next step
      #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
      #Only happens the step before the derivatives are recalculated.
      
      #new interval thing
      trial = 0
      while loop2 == True:
        ut, up = new_step[4], new_step[7]
        err = mm.check_interval(mm.kerr, new_step, mass, a)
        metric = mm.kerr(new_step, mass, a)[0]
        gtt, gtp = metric[0][0], metric[0][3]
        disc = np.sqrt(ut**2 + ((gtp/gtt)**2)*(up**2) + 2*(gtp/gtt)*ut*up - (1.0 + err)/gtt)
        int_steps = [np.copy(new_step), np.copy(new_step)]
        int_steps[0][4] += -ut - (gtp/gtt)*up + disc
        int_steps[1][4] += -ut - (gtp/gtt)*up - disc
        checks = [(mm.check_interval(mm.kerr, int_steps[0], mass, a)+1)**2, (mm.check_interval(mm.kerr, int_steps[1], mass, a)+1)**2]
        test = mm.check_interval(mm.kerr, int_steps[checks.index(min(checks))], mass, a)
        trial += 1
        if (abs(test+1)<=(err_target)) and (int_steps[checks.index(min(checks))][4] > 0):
          new_step = int_steps[checks.index(min(checks))]
          loop2 = False
        if trial >= 10:
          print("giving up!", trial, test)
          break
              
      #Update stuff!
      if update == True:
        constants.append(new_con)
        qarter.append(new_con[2] + (a*new_con[0] - new_con[1])**2)
        #print(constants[-1], a, mu, all_states, con_derv[-1], i)
        
        if con_derv[-1][9] == True:
          compErr += 1
          issues.append((i, new_step[0]))

      interval.append(mm.check_interval(mm.kerr, new_step, mass, a))
      dTau_change.append(old_dTau)
      #true_u_t.append(set_u_kerr(new_step, mass, a, timelike, 0, 0, special = "circle")[4])
      metric = mm.kerr(new_step, mass, a)[0]
      all_states.append(new_step )    #update position and velocity
      #phys_avgs.append([np.average(all_states[:][1]),
      #                  np.average(all_states[:][2]),
      #                  np.average(all_states[:][3])])
      i += 1

      if new_step[1] < rmin:
        rmin = new_step[1]
      progress = max(new_step[0]/max_time, i/(50*max_time)) * 100
      if verbose == True:
          if (progress >= milestone):
            print("Program has completed " + str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
            milestone += 1
          if (new_step[3] >= orbitCount*2*np.pi):
            print("Number of orbits = " + str(round(new_step[3]/(2*np.pi))) + ", t = " + str(new_step[0]) + ", r_min = " + str(rmin))
            orbitCount += np.round(20/rmin)

    #Lets you end the program before the established end without breaking anything
    except KeyboardInterrupt:
      print("Ending program")
      cap = len(all_states) - 1
      all_states = all_states[:cap]
      interval = interval[:cap]
      dTau_change = dTau_change[:cap]
      true_u_t = true_u_t[:cap]
      constants = constants[:cap]
      qarter = qarter[:cap]
      #phys_avgs = phys_avgs[:cap]
      break

  r = all_states[0][1]
  constants = np.array(constants)
  qarter = np.array(qarter)
  con_derv = np.array(con_derv, dtype=object)
  interval = np.array(interval)
  dTau_change = np.array(dTau_change)
  true_u_t = np.array(true_u_t)
  all_states = np.array(all_states)
  #phys_avgs = np.array(phys_avgs)
  if verbose == True:
      print("There were " + str(compErr) + " issues with complex roots/turning points.")
      for i in issues:
        print("Complex value at index " + str(i[0]) + ", t = " + str(i[1]))
      #print("Here's some data!")
      #print(checker)
  final = {"name": label,
           "raw": all_states,
           "inputs": inputs,
           "dTau": dTau,
           #"r_av": phys_avgs[:,0],
           #"theta_av": phys_avgs[:,1],
           #"phi_av": phys_avgs[:,2],
           "pos": all_states[:,1:4],
           "all_vel": all_states[:,4:], 
           "time": all_states[:,0],
           "interval": interval,
           "vel": (np.square(all_states[:,5]) + np.square(all_states[:,1]) * (np.square(all_states[:,6]) + np.square(all_states[:,7])))**(0.5),
           "dTau_change": dTau_change,
           "ut check": true_u_t,
           "energy": constants[:, 0],
           "phi_momentum": constants[:, 1],
           "qarter":qarter,
           "carter": constants[:, 2],
           "spin": a,
           "freqs": np.array([((r**(3/2) + pro*a)**(-1))*np.sqrt(1 - (6/r) + pro*(8*a*(r**(-3/2))) - (3*((a/r)**(2)))),
                              ((r**(3/2) + pro*a)**(-1))*np.sqrt(1 - pro*(4*a*(r**(-3/2))) + (3*((a/r)**(2)))),
                              ((r**(3/2) + pro*a)**(-1))]),
           "r0": con_derv[:,2],
           "y": con_derv[:,3],
           "v": con_derv[:,4],
           "q": con_derv[:,5],
           "e": con_derv[:,6],
           "ot": con_derv[:,7],
           "it": con_derv[:,8],
           "tracktime": con_derv[:,10]}
  return final

#version that uses integrated peters formulas, updates through lagrange multipliers+constants
def inspiral_long3(state, mass, a, mu, vstep, max_time, dTau, timelike, err_target, eta, xi, label, spec = False, verbose=True):  #basically the same as schwarz, except references to schwarz changed to kerr
  inputs = (mass, mu, vstep, max_time, timelike, err_target, spec)          #Grab initial input in case you want to run the continue function
  all_states = [np.copy(state)[0]]                                                   #Grab that initial state         
  err_calc = 1                                                                  #initialize calculated error
  pro = np.sign(float(eta))**(np.sign(float(eta)))                              # +1 if prograde/polar, -1 if retrograde
  i = 0                                                                         #initialize step counter

  print("Normalizing initial state")
  all_states[0] = mm.set_u_kerr(all_states[0], mass, a, timelike, eta*np.pi/180, xi*np.pi/180, special = spec)      #normalize initial state so it's actually physical

  interval = [mm.check_interval(mm.kerr, all_states[0], mass, a)]           #create interval tracker
  dTau_change = [dTau]                                                #create dTau tracker
  true_u_t = [all_states[0][4]]                                         #create u_t tracker (what it should be, not what it is)
  metric = mm.kerr(all_states[0], mass, a)[0]                                      #initial metric
  initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
  initLz = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
  initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], mass, a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
  initC = initQ - (a*initE - initLz)**2                                          #initial adjusted Carter constant     
  
  # Adding a thing in here to just have something for all components of L
  xyz = np.array([all_states[0][1]*mm.fix_sin(all_states[0][2])*mm.fix_cos(all_states[0][3]),
                  all_states[0][1]*mm.fix_sin(all_states[0][2])*mm.fix_sin(all_states[0][3]),
                  all_states[0][1]*mm.fix_cos(all_states[0][2]) ])
  v_xyz = np.array([all_states[0][5]*mm.fix_sin(all_states[0][2])*mm.fix_cos(all_states[0][3]) + all_states[0][1]*all_states[0][6]*mm.fix_cos(all_states[0][2])*mm.fix_cos(all_states[0][3]) - all_states[0][1]*all_states[0][7]*mm.fix_sin(all_states[0][2])*mm.fix_sin(all_states[0][3]),
                    all_states[0][5]*mm.fix_sin(all_states[0][2])*mm.fix_sin(all_states[0][3]) + all_states[0][1]*all_states[0][6]*mm.fix_cos(all_states[0][2])*mm.fix_cos(all_states[0][3]) + all_states[0][1]*all_states[0][7]*mm.fix_sin(all_states[0][2])*mm.fix_cos(all_states[0][3]),
                    all_states[0][5]*mm.fix_cos(all_states[0][2]) - all_states[0][1]*all_states[0][6]*mm.fix_sin(all_states[0][2])])
  L_prop = np.cross(xyz, v_xyz)
  initLx, initLy = (initLz/L_prop[2])*L_prop[0], (initLz/L_prop[2])*L_prop[1]                   
  constants = [ np.array([initE,      #energy   
                          initLx,
                          initLy,
                          initLz,      #angular momentum
                          initC]) ]    #Carter constant (C)
  qarter = [initQ]           #Carter constant (Q)
  '''
  various = np.array([[0,             #r0 - Semimajor axis of the orbit, equal to the radius if circular
                       0,             #y - Carter constant C divided by angular momentum squared
                       0,             #v - square root of mass divided by r0
                       0,             #q - spin divided by mass
                       0,             #e - eccentricity of the orbit
                       0,             #ot - apoapsis of the orbit, the "outer turn"
                       0,             #it - periapsis of the orbit, the "inner turn"
                       0]])           #tracktime - timestamp
  '''

  con_derv = [[0, *mm.peters_integrate2(constants[0], a, mu, all_states, 0, 0), all_states[0][0]]]    
  # returns [index, [dedt, dlxdt, dlydt, dlzdt, dcdt], r0, y, v, q, e, outer_turn, inner_turn, compErr, timestamp]
  
  #r0 = con_derv[0][2]
  #con_derv[0][-1] = np.pi*(r0**(3/2) + a)/2
  #timestamp is set to a quarter-period to keep the half-orbit forced update from happening at max or min r
  #forced update at max/min r causes problem, near r0 is better
  #doing this because more circular orbits are wonky
  #but this might break stuff for non-circular orbits??
  #problem for later
  
  
  compErr = 0
  milestone = 0
  rmin = all_states[0][1]
  orbitCount = np.round(20/(rmin))
  issues = []
  checker = []
  orbitside = np.sign(all_states[0][1] - con_derv[0][2])
  in_disk = 0
  #Main Loop
  while (all_states[i][0] < max_time and i < 50*max_time):
    try:
      update = False
      first = True
      loop2 = False
      roche = ((1/mu)**(1/3)) * (472393*mu)
          # (472393*mu) is a rough approx of 1 solar radius for a given mu

      #Grab the current state
      state = all_states[i]  
      r0 = con_derv[-1][2]      

      #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
      if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001) :
        break
      elif state[1] >= state[1]*1000:    #Currently based on event horizon, might change it to be based on r0
        break
        
      #Runge-Kutta update using geodesic
      while (err_calc >= err_target) or (first == True):
        step_check = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, mass, a)
        new_step = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, mass, a) 
        mod_step = mm.np.append(step_check[0:6], mm.fix_sin(step_check[6])*mm.fix_cos(step_check[7]))
        mod_new = np.append(new_step[0:6], mm.fix_sin(new_step[6])*mm.fix_cos(new_step[7]))
        mod_state = np.append(state[0:6], mm.fix_sin(state[6])*mm.fix_cos(state[7]))
        mod_pre = mod_step - mod_new
        mod_pre[:4] = mod_pre[:4]/np.linalg.norm(mod_state[:4]) 
        mod_pre[4:] = mod_pre[4:]/np.linalg.norm(mod_state[4:])
        err_calc = np.linalg.norm(mod_pre)
        old_dTau, dTau = dTau, min(0.95 * dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
        first = False

      #constant modifying section
      '''
      #First check: is it inside the accretion disk? (Annulus from 3rs (6gu) to ??? with some (small) thickness)
      if ((new_step[1] >= 6) and (new_step[1] <= 10)) and (abs(new_step[2] - np.pi/2) <= 0.05):
          r_star = 472393*mu                 #approx 1 solar radius
          density = 2.29*(10**(-9))*(mu**2)    #approx surface density of 1 solar mass spread between 6gu and 10gu (Assumed constant)
          md = np.pi*(r_star**2)*density     #mass accreted by star passing through disk (estimated by "hole punch method")
          pos = new_step[:4]                 #position
          print(pos, "pos")
          print(new_step[4:], "vel")
          disk_state = np.array(mm.set_u_kerr(new_step, 1.0, a, True, 90, 90, special="circle"))
                                             #4-velocity of disk at given r
          pd = md * disk_state[4:]
          pstar = mu * new_step[4:]   #4-momenta of star and disk segment
          pnew = pd + pstar                  #combined 4-momentum
          print(pnew, "pnew")
          mu = (-mm.check_interval(mm.kerr, [*pos, *pnew], 1.0, 0.0))**(1/2)
          new_state = np.array([*pos, *(pnew/mu)])

          new_con = mm.find_constants(new_state, 1.0, a)
          update = True
        
      #Second check: is it at/within the tidal Roche limit?
      #if (new_step[1] <= roche):
          #in_disk = 3
          #tidal disrupt function
      #'''
      #Whenever you pass from one side of r0 to the other, mess with the effective potential.
      if (( np.sign(new_step[1] - r0) != orbitside )):  # or (flip)):  #(new_step[0] - con_derv[-1][10]) > np.pi*(r0**(3/2) + a) or
        orbitside = np.sign(new_step[1] - r0)
        update = True
        con_derv.append([i, *mm.peters_integrate2(constants[-1], a, mu, all_states, con_derv[-1][0], i), new_step[0]])
        new_con = constants[-1] + con_derv[-1][1]                                   #Figure out how the constants have changed!
        dir1 = new_step[5]
        new_step = mm.new_recalc_state2(constants[-1], con_derv[-1][1], new_step, mu, mass, a)                           #Change the velocity so it actually makes sense with those constants
        test = mm.check_interval(mm.kerr, new_step, mass, a)
        if abs(test+1)>(err_target):
          print("HEY SOMETHING'S UP")
          print(i)
          print("behold")
          print(test)
          print(new_step)
          #print(constants[-2])
          print(constants[-1])
          print("press enter to continue")
          #input()
          loop2 = True
          #break
        if dir1 != new_step[5]:
          checker.append([dir1, new_step[5], con_derv[-1][1]  ])
      #Initializing for the next step
      #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
      #Only happens the step before the derivatives are recalculated.
      
      #new interval thing
      trial = 0
      while loop2 == True:
        ut, up = new_step[4], new_step[7]
        err = mm.check_interval(mm.kerr, new_step, mass, a)
        metric = mm.kerr(new_step, mass, a)[0]
        gtt, gtp = metric[0][0], metric[0][3]
        disc = np.sqrt(ut**2 + ((gtp/gtt)**2)*(up**2) + 2*(gtp/gtt)*ut*up - (1.0 + err)/gtt)
        int_steps = [np.copy(new_step), np.copy(new_step)]
        int_steps[0][4] += -ut - (gtp/gtt)*up + disc
        int_steps[1][4] += -ut - (gtp/gtt)*up - disc
        checks = [(mm.check_interval(mm.kerr, int_steps[0], mass, a)+1)**2, (mm.check_interval(mm.kerr, int_steps[1], mass, a)+1)**2]
        test = mm.check_interval(mm.kerr, int_steps[checks.index(min(checks))], mass, a)
        trial += 1
        if (abs(test+1)<=(err_target)) and (int_steps[checks.index(min(checks))][4] > 0):
          new_step = int_steps[checks.index(min(checks))]
          loop2 = False
        if trial >= 10:
          print("giving up!", trial, test)
          break
              
      #Update stuff!
      if update == True:
        constants.append(new_con)
        qarter.append(new_con[4] + (a*new_con[0] - new_con[3])**2)
        #print(constants[-1], a, mu, all_states, con_derv[-1], i)
        
        if con_derv[-1][9] == True:
          compErr += 1
          issues.append((i, new_step[0]))

      interval.append(mm.check_interval(mm.kerr, new_step, mass, a))
      dTau_change.append(old_dTau)
      #true_u_t.append(set_u_kerr(new_step, mass, a, timelike, 0, 0, special = "circle")[4])
      metric = mm.kerr(new_step, mass, a)[0]
      all_states.append(new_step )    #update position and velocity
      #phys_avgs.append([np.average(all_states[:][1]),
      #                  np.average(all_states[:][2]),
      #                  np.average(all_states[:][3])])
      i += 1

      if new_step[1] < rmin:
        rmin = new_step[1]
      progress = max(new_step[0]/max_time, i/(50*max_time)) * 100
      if verbose == True:
          if (progress >= milestone):
            print("Program has completed " + str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
            milestone += 1
          if (new_step[3] >= orbitCount*2*np.pi):
            print("Number of orbits = " + str(round(new_step[3]/(2*np.pi))) + ", t = " + str(new_step[0]) + ", r_min = " + str(rmin))
            orbitCount += np.round(20/rmin)

    #Lets you end the program before the established end without breaking anything
    except KeyboardInterrupt:
      print("Ending program")
      cap = len(all_states) - 1
      all_states = all_states[:cap]
      interval = interval[:cap]
      dTau_change = dTau_change[:cap]
      true_u_t = true_u_t[:cap]
      constants = constants[:cap]
      qarter = qarter[:cap]
      #phys_avgs = phys_avgs[:cap]
      break

  r = all_states[0][1]
  constants = np.array(constants)
  qarter = np.array(qarter)
  con_derv = np.array(con_derv, dtype=object)
  interval = np.array(interval)
  dTau_change = np.array(dTau_change)
  true_u_t = np.array(true_u_t)
  all_states = np.array(all_states)
  #phys_avgs = np.array(phys_avgs)
  if verbose == True:
      print("There were " + str(compErr) + " issues with complex roots/turning points.")
      for i in issues:
        print("Complex value at index " + str(i[0]) + ", t = " + str(i[1]))
      #print("Here's some data!")
      #print(checker)
  final = {"name": label,
           "raw": all_states,
           "inputs": inputs,
           "dTau": dTau,
           #"r_av": phys_avgs[:,0],
           #"theta_av": phys_avgs[:,1],
           #"phi_av": phys_avgs[:,2],
           "pos": all_states[:,1:4],
           "all_vel": all_states[:,4:], 
           "time": all_states[:,0],
           "interval": interval,
           "vel": (np.square(all_states[:,5]) + np.square(all_states[:,1]) * (np.square(all_states[:,6]) + np.square(all_states[:,7])))**(0.5),
           "dTau_change": dTau_change,
           "ut check": true_u_t,
           "energy": constants[:, 0],
           "phi_momentum": constants[:, 3],
           "Lx_momentum": constants[:, 1],
           "Ly_momentum": constants[:, 2],
           "qarter":qarter,
           "carter": constants[:, 4],
           "spin": a,
           "freqs": np.array([((r**(3/2) + pro*a)**(-1))*np.sqrt(1 - (6/r) + pro*(8*a*(r**(-3/2))) - (3*((a/r)**(2)))),
                              ((r**(3/2) + pro*a)**(-1))*np.sqrt(1 - pro*(4*a*(r**(-3/2))) + (3*((a/r)**(2)))),
                              ((r**(3/2) + pro*a)**(-1))]),
           "r0": con_derv[:,2],
           "y": con_derv[:,3],
           "v": con_derv[:,4],
           "q": con_derv[:,5],
           "e": con_derv[:,6],
           "ot": con_derv[:,7],
           "it": con_derv[:,8],
           "tracktime": con_derv[:,10]}
  return final

#both of these are used for versions after this
def getEnergy(state, mass, a):
    metric, chris = mm.kerr(state, mass, a)
    stuff = np.matmul(metric, state[4:])
    ene = -stuff[0]
    #print(stuff)
    return ene
    
def getLs(state, mu):
    t, r, theta, phi, vel4 = *state[:4], state[4:]
    sint, cost = mm.fix_sin(theta), mm.fix_cos(theta)
    sinp, cosp = mm.fix_sin(phi), mm.fix_cos(phi)
    sph2cart = np.array([[1, 0,         0,           0           ],
                         [0, sint*cosp, r*cost*cosp, -r*sint*sinp],
                         [0, sint*sinp, r*cost*sinp, r*sint*cosp ],
                         [0, cost,      -r*sint,     0           ]])
    vel4cart = np.matmul(sph2cart, vel4)
    vel3cart = vel4cart[1:4]
    pos3cart = np.array([r*sint*cosp, r*sint*sinp, r*cost])
    Lmom = np.cross(pos3cart, vel3cart)
    return Lmom

#version that uses integrated peters formulas, updates through least squares+constants
def inspiral_long4(state, mass, a, mu, vstep, max_time, dTau, timelike, err_target, eta, xi, label, spec = False, verbose=True):  #basically the same as schwarz, except references to schwarz changed to kerr
  inputs = (mass, mu, vstep, max_time, timelike, err_target, spec)          #Grab initial input in case you want to run the continue function
  all_states = [np.copy(state)[0]]                                                   #Grab that initial state         
  err_calc = 1                                                                  #initialize calculated error
  pro = np.sign(float(eta))**(np.sign(float(eta)))                              # +1 if prograde/polar, -1 if retrograde
  i = 0                                                                         #initialize step counter

  print("Normalizing initial state")
  all_states[0] = mm.set_u_kerr(all_states[0], mass, a, timelike, eta*np.pi/180, xi*np.pi/180, special = spec)      #normalize initial state so it's actually physical

  interval = [mm.check_interval(mm.kerr, all_states[0], mass, a)]           #create interval tracker
  dTau_change = [dTau]                                                #create dTau tracker
  true_u_t = [all_states[0][4]]                                         #create u_t tracker (what it should be, not what it is)
  metric = mm.kerr(all_states[0], mass, a)[0]                                      #initial metric
  initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
  initLz = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
  initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], mass, a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
  initC = initQ - (a*initE - initLz)**2                                          #initial adjusted Carter constant     
  
  # Adding a thing in here to just have something for all components of L
  xyz = np.array([all_states[0][1]*mm.fix_sin(all_states[0][2])*mm.fix_cos(all_states[0][3]),
                  all_states[0][1]*mm.fix_sin(all_states[0][2])*mm.fix_sin(all_states[0][3]),
                  all_states[0][1]*mm.fix_cos(all_states[0][2]) ])
  v_xyz = np.array([all_states[0][5]*mm.fix_sin(all_states[0][2])*mm.fix_cos(all_states[0][3]) + all_states[0][1]*all_states[0][6]*mm.fix_cos(all_states[0][2])*mm.fix_cos(all_states[0][3]) - all_states[0][1]*all_states[0][7]*mm.fix_sin(all_states[0][2])*mm.fix_sin(all_states[0][3]),
                    all_states[0][5]*mm.fix_sin(all_states[0][2])*mm.fix_sin(all_states[0][3]) + all_states[0][1]*all_states[0][6]*mm.fix_cos(all_states[0][2])*mm.fix_cos(all_states[0][3]) + all_states[0][1]*all_states[0][7]*mm.fix_sin(all_states[0][2])*mm.fix_cos(all_states[0][3]),
                    all_states[0][5]*mm.fix_cos(all_states[0][2]) - all_states[0][1]*all_states[0][6]*mm.fix_sin(all_states[0][2])])
  L_prop = np.cross(xyz, v_xyz)
  initLx, initLy = (initLz/L_prop[2])*L_prop[0], (initLz/L_prop[2])*L_prop[1]                   
  constants = [ np.array([initE,      #energy   
                          initLx,
                          initLy,
                          initLz,      #angular momentum
                          initC]) ]    #Carter constant (C)
  qarter = [initQ]           #Carter constant (Q)
  etemp = getEnergy(all_states[0], mass, a)
  ltemp = getLs(all_states[0], mu)
  r, theta = all_states[0][1], all_states[0][2]
  constants2 = [np.array([etemp, *ltemp, ((ltemp[2] - a*etemp)/mm.fix_sin(theta))**2 + (a*mm.fix_cos(theta))**2 + ((r**2 + (a*mm.fix_cos(theta))**2)*all_states[0][6])**2 - (a*etemp - ltemp[2])**2])]
  #constants = constants2.copy()
  '''
  various = np.array([[0,             #r0 - Semimajor axis of the orbit, equal to the radius if circular
                       0,             #y - Carter constant C divided by angular momentum squared
                       0,             #v - square root of mass divided by r0
                       0,             #q - spin divided by mass
                       0,             #e - eccentricity of the orbit
                       0,             #ot - apoapsis of the orbit, the "outer turn"
                       0,             #it - periapsis of the orbit, the "inner turn"
                       0]])           #tracktime - timestamp
  '''

  con_derv = [[0, *mm.peters_integrate2(constants[0], a, mu, all_states, 0, 0), all_states[0][0]]]    
  # returns [index, [dedt, dlxdt, dlydt, dlzdt, dcdt], r0, y, v, q, e, outer_turn, inner_turn, compErr, timestamp]
  
  #r0 = con_derv[0][2]
  #con_derv[0][-1] = np.pi*(r0**(3/2) + a)/2
  #timestamp is set to a quarter-period to keep the half-orbit forced update from happening at max or min r
  #forced update at max/min r causes problem, near r0 is better
  #doing this because more circular orbits are wonky
  #but this might break stuff for non-circular orbits??
  #problem for later
  
  
  compErr = 0
  milestone = 0
  rmin = all_states[0][1]
  orbitCount = np.round(20/(rmin))
  issues = []
  checker = []
  orbitside = np.sign(all_states[0][1] - con_derv[0][2])
  in_disk = 0
  #Main Loop
  while (all_states[i][0] < max_time and i < 50*max_time):
    try:
      update = False
      first = True
      loop2 = False
      #roche = ((1/mu)**(1/3)) * (472393*mu)
          # (472393*mu) is a rough approx of 1 solar radius for a given mu

      #Grab the current state
      state = all_states[i]  
      r0 = con_derv[-1][2]   
      #print("r0=",r0)

      #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
      if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001) :
        break
      elif state[1] >= state[1]*1000:    #Currently based on event horizon, might change it to be based on r0
        break
        
      #Runge-Kutta update using geodesic
      while (err_calc >= err_target) or (first == True):
        step_check = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, mass, a)
        new_step = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, mass, a) 
        mod_step = mm.np.append(step_check[0:6], mm.fix_sin(step_check[6])*mm.fix_cos(step_check[7]))
        mod_new = np.append(new_step[0:6], mm.fix_sin(new_step[6])*mm.fix_cos(new_step[7]))
        mod_state = np.append(state[0:6], mm.fix_sin(state[6])*mm.fix_cos(state[7]))
        mod_pre = mod_step - mod_new
        mod_pre[:4] = mod_pre[:4]/np.linalg.norm(mod_state[:4]) 
        mod_pre[4:] = mod_pre[4:]/np.linalg.norm(mod_state[4:])
        err_calc = np.linalg.norm(mod_pre)
        old_dTau, dTau = dTau, min(0.95 * dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
        first = False

      #constant modifying section
      '''
      #First check: is it inside the accretion disk? (Annulus from 3rs (6gu) to ??? with some (small) thickness)
      if ((new_step[1] >= 6) and (new_step[1] <= 10)) and (abs(new_step[2] - np.pi/2) <= 0.05):
          r_star = 472393*mu                 #approx 1 solar radius
          density = 2.29*(10**(-9))*(mu**2)    #approx surface density of 1 solar mass spread between 6gu and 10gu (Assumed constant)
          md = np.pi*(r_star**2)*density     #mass accreted by star passing through disk (estimated by "hole punch method")
          pos = new_step[:4]                 #position
          print(pos, "pos")
          print(new_step[4:], "vel")
          disk_state = np.array(mm.set_u_kerr(new_step, 1.0, a, True, 90, 90, special="circle"))
                                             #4-velocity of disk at given r
          pd = md * disk_state[4:]
          pstar = mu * new_step[4:]   #4-momenta of star and disk segment
          pnew = pd + pstar                  #combined 4-momentum
          print(pnew, "pnew")
          mu = (-mm.check_interval(mm.kerr, [*pos, *pnew], 1.0, 0.0))**(1/2)
          new_state = np.array([*pos, *(pnew/mu)])

          new_con = mm.find_constants(new_state, 1.0, a)
          update = True
        
      #Second check: is it at/within the tidal Roche limit?
      #if (new_step[1] <= roche):
          #in_disk = 3
          #tidal disrupt function
      '''
      #Whenever you pass from one side of r0 to the other, mess with the effective potential.
      if (( np.sign(new_step[1] - r0) != orbitside ) and (mu != 0.0)):  # or (flip)):  #(new_step[0] - con_derv[-1][10]) > np.pi*(r0**(3/2) + a) or
        #print("orbitside=",orbitside)
        #print("actual=", np.sign(new_step[1] - r0))
        #print(r0)
        #print(new_step)
        #print(state)
        orbitside = np.sign(new_step[1] - r0)
        update = True
        con_derv.append([i, *mm.peters_integrate2(constants[-1], a, mu, all_states, con_derv[-1][0], i), new_step[0]])
        new_con = constants[-1] + con_derv[-1][1]                                   #Figure out how the constants have changed!
        #print(con_derv[-1][1][0:4])
        dir1 = new_step[5]
        new_step = mm.new_recalc_state3(con_derv[-1][1], new_step, mu, mass, a)                           #Change the velocity so it actually makes sense with those constants
        #print("new new")
        #print(new_step)
        test = mm.check_interval(mm.kerr, new_step, mass, a)
        if abs(test+1)>(err_target):
          print("HEY SOMETHING'S UP")
          print(i)
          print("behold")
          print(test)
          print(new_step)
          print(constants[-1])
          print("press enter to continue")
          #input()
          loop2 = True
          #break
        if dir1 != new_step[5]:
          checker.append([dir1, new_step[5], con_derv[-1][1]  ])
      #Initializing for the next step
      #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
      #Only happens the step before the derivatives are recalculated.
      
      #new interval thing
      trial = 0
      while loop2 == True:
        ut, up = new_step[4], new_step[7]
        err = mm.check_interval(mm.kerr, new_step, mass, a)
        metric = mm.kerr(new_step, mass, a)[0]
        gtt, gtp = metric[0][0], metric[0][3]
        disc = np.sqrt(ut**2 + ((gtp/gtt)**2)*(up**2) + 2*(gtp/gtt)*ut*up - (1.0 + err)/gtt)
        int_steps = [np.copy(new_step), np.copy(new_step)]
        int_steps[0][4] += -ut - (gtp/gtt)*up + disc
        int_steps[1][4] += -ut - (gtp/gtt)*up - disc
        checks = [(mm.check_interval(mm.kerr, int_steps[0], mass, a)+1)**2, (mm.check_interval(mm.kerr, int_steps[1], mass, a)+1)**2]
        test = mm.check_interval(mm.kerr, int_steps[checks.index(min(checks))], mass, a)
        trial += 1
        if (abs(test+1)<=(err_target)) and (int_steps[checks.index(min(checks))][4] > 0):
          new_step = int_steps[checks.index(min(checks))]
          loop2 = False
        if trial >= 10:
          print("giving up!", trial, test)
          break
      
      #Update stuff!
      if update == True:
        constants.append(new_con)
        qarter.append(new_con[4] + (a*new_con[0] - new_con[3])**2)
        #print(constants[-1], a, mu, all_states, con_derv[-1], i)
        
        if con_derv[-1][9] == True:
          compErr += 1
          issues.append((i, new_step[0]))
    
      interval.append(mm.check_interval(mm.kerr, new_step, mass, a))
      #TEMPORARY
      etemp = getEnergy(new_step, mass, a)
      ltemp = getLs(new_step, mu)
      r, theta = new_step[1], new_step[2]
      constants2.append(np.array([etemp, *ltemp, ((ltemp[2] - a*etemp)/mm.fix_sin(theta))**2 + (a*mm.fix_cos(theta))**2 + ((r**2 + (a*mm.fix_cos(theta))**2)*new_step[6])**2 - (a*etemp - ltemp[2])**2]))
      #TEMPORARY
      #constants.append(np.array([etemp, *ltemp, ltemp[0]**2 + ltemp[1]**2]))
      dTau_change.append(old_dTau)
      #true_u_t.append(set_u_kerr(new_step, mass, a, timelike, 0, 0, special = "circle")[4])
      metric = mm.kerr(new_step, mass, a)[0]
      all_states.append(new_step )    #update position and velocity
      #phys_avgs.append([np.average(all_states[:][1]),
      #                  np.average(all_states[:][2]),
      #                  np.average(all_states[:][3])])
      i += 1

      if new_step[1] < rmin:
        rmin = new_step[1]
      progress = max(new_step[0]/max_time, i/(50*max_time)) * 100
      if verbose == True:
          if (progress >= milestone):
            print("Program has completed " + str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
            milestone += 1
          if (new_step[3] >= orbitCount*2*np.pi):
            print("Number of orbits = " + str(round(new_step[3]/(2*np.pi))) + ", t = " + str(new_step[0]) + ", r_min = " + str(rmin))
            orbitCount += np.round(20/rmin)

    #Lets you end the program before the established end without breaking anything
    except KeyboardInterrupt:
      print("Ending program")
      cap = len(all_states) - 1
      all_states = all_states[:cap]
      interval = interval[:cap]
      dTau_change = dTau_change[:cap]
      true_u_t = true_u_t[:cap]
      constants = constants[:cap]
      qarter = qarter[:cap]
      #phys_avgs = phys_avgs[:cap]
      break

  r = all_states[0][1]
  constants = np.array(constants)
  constants2 = np.array(constants2)
  qarter = np.array(qarter)
  con_derv = np.array(con_derv, dtype=object)
  interval = np.array(interval)
  dTau_change = np.array(dTau_change)
  true_u_t = np.array(true_u_t)
  all_states = np.array(all_states)
  #phys_avgs = np.array(phys_avgs)
  if verbose == True:
      print("There were " + str(compErr) + " issues with complex roots/turning points.")
      for i in issues:
        print("Complex value at index " + str(i[0]) + ", t = " + str(i[1]))
      #print("Here's some data!")
      #print(checker)
  final = {"name": label,
           "raw": all_states,
           "inputs": inputs,
           "dTau": dTau,
           #"r_av": phys_avgs[:,0],
           #"theta_av": phys_avgs[:,1],
           #"phi_av": phys_avgs[:,2],
           "pos": all_states[:,1:4],
           "all_vel": all_states[:,4:], 
           "time": all_states[:,0],
           "interval": interval,
           "vel": (np.square(all_states[:,5]) + np.square(all_states[:,1]) * (np.square(all_states[:,6]) + np.square(all_states[:,7])))**(0.5),
           "dTau_change": dTau_change,
           "ut check": true_u_t,
           "energy": constants[:, 0],
           "phi_momentum": constants[:, 3],
           "Lx_momentum": constants[:, 1],
           "Ly_momentum": constants[:, 2],
           "qarter":qarter,
           "carter": constants[:, 4],
           "energy2": constants2[:, 0],
           "phi_momentum2": constants2[:, 3],
           "Lx_momentum2": constants2[:, 1],
           "Ly_momentum2": constants2[:, 2],
           "carter2": constants2[:, 4],
           "spin": a,
           "freqs": np.array([((r**(3/2) + pro*a)**(-1))*np.sqrt(1 - (6/r) + pro*(8*a*(r**(-3/2))) - (3*((a/r)**(2)))),
                              ((r**(3/2) + pro*a)**(-1))*np.sqrt(1 - pro*(4*a*(r**(-3/2))) + (3*((a/r)**(2)))),
                              ((r**(3/2) + pro*a)**(-1))]),
           "r0": con_derv[:,2],
           "y": con_derv[:,3],
           "v": con_derv[:,4],
           "q": con_derv[:,5],
           "e": con_derv[:,6],
           "ot": con_derv[:,7],
           "it": con_derv[:,8],
           "tracktime": con_derv[:,10]}
  return final

#version that uses integrated peters formulas, updates through least squares+psuedoconstants, no update for mu = 0.0
def inspiral_long5(state, mass, a, mu, vstep, max_time, dTau, timelike, err_target, eta, xi, label, spec = False, verbose=True):  #basically the same as schwarz, except references to schwarz changed to kerr
  inputs = (mass, a, mu, vstep, max_time, timelike, err_target, spec)          #Grab initial input in case you want to run the continue function
  all_states = [np.copy(state)[0]]                                                   #Grab that initial state         
  err_calc = 1                                                                  #initialize calculated error
  pro = np.sign(float(eta))**(np.sign(float(eta)))                              # +1 if prograde/polar, -1 if retrograde
  i = 0                                                                         #initialize step counter

  print("Normalizing initial state")
  all_states[0] = mm.set_u_kerr(all_states[0], mass, a, timelike, eta*np.pi/180, xi*np.pi/180, special = spec)      #normalize initial state so it's actually physical

  interval = [mm.check_interval(mm.kerr, all_states[0], mass, a)]           #create interval tracker
  dTau_change = [dTau]                                                #create dTau tracker
  metric = mm.kerr(all_states[0], mass, a)[0]                                      #initial metric
  initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
  initLz = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
  initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], mass, a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
  initC = initQ - (a*initE - initLz)**2                                          #initial adjusted Carter constant                      
  constants = [ np.array([initE,      #energy   
                          initLz,      #angular momentum (axial)
                          initC]) ]    #Carter constant (C)
  qarter = [initQ]           #Carter constant (Q)

  false_constants = [np.array([getEnergy(all_states[0], mass, a), *getLs(all_states[0], mu)])]  #Cartesian approximation of L vector
  
  '''
  various = np.array([[0,             #r0 - Semimajor axis of the orbit, equal to the radius if circular
                       0,             #y - Carter constant C divided by angular momentum squared
                       0,             #v - square root of mass divided by r0
                       0,             #q - spin divided by mass
                       0,             #e - eccentricity of the orbit
                       0,             #ot - apoapsis of the orbit, the "outer turn"
                       0,             #it - periapsis of the orbit, the "inner turn"
                       0]])           #tracktime - timestamp
  '''

  con_derv = [[0, *mm.peters_integrate3(constants[0], a, mu, all_states, 0, 0), 0.0]]    
  # returns [index, [dedt, dlxdt, dlydt, dlzdt], r0, y, v, q, e, outer_turn, inner_turn, compErr, timestamp]
  
  #r0 = con_derv[0][2]
  #con_derv[0][-1] = np.pi*(r0**(3/2) + a)/2
  #timestamp is set to a quarter-period to keep the half-orbit forced update from happening at max or min r
  #forced update at max/min r causes problem, near r0 is better
  #doing this because more circular orbits are wonky
  #but this might break stuff for non-circular orbits??
  #problem for later
  
  
  compErr = 0
  milestone = 0
  rmin = all_states[0][1]
  issues = []
  checker = []
  orbitside = np.sign(all_states[0][1] - con_derv[0][2])
  orbitCount = 0
  tracktime = [all_states[0][0]]
  #Main Loop
  while (all_states[i][0] < max_time and i < 50*max_time):
    try:
      update = False
      condate = False
      first = True
      loop2 = False

      #Grab the current state
      state = all_states[i]  
      r0 = con_derv[-1][2]   
      #print("r0=",r0)

      #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
      if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001) :
        break
      elif (state[1] >= (1 + np.sqrt(1 - a**2))*1000):    #Currently based on event horizon, might change it to be based on r0
        break
        
      #Runge-Kutta update using geodesic
      while (err_calc >= err_target) or (first == True):
        step_check = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, mass, a)
        new_step = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, mass, a) 
        mod_step = np.append(step_check[0:6], mm.fix_sin(step_check[6])*mm.fix_cos(step_check[7]))
        mod_new = np.append(new_step[0:6], mm.fix_sin(new_step[6])*mm.fix_cos(new_step[7]))
        mod_state = np.append(state[0:6], mm.fix_sin(state[6])*mm.fix_cos(state[7]))
        mod_pre = mod_step - mod_new
        mod_pre[:4] = mod_pre[:4]/np.linalg.norm(mod_state[:4]) 
        mod_pre[4:] = mod_pre[4:]/np.linalg.norm(mod_state[4:])
        err_calc = np.linalg.norm(mod_pre)
        #print(err_calc)
        old_dTau, dTau = dTau, min(0.95 * dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
        first = False

      #constant modifying section
      #Whenever you pass from one side of r0 to the other, mess with the effective potential.
      if ( np.sign(new_step[1] - r0) != orbitside ):  # or (flip)):  #(new_step[0] - con_derv[-1][10]) > np.pi*(r0**(3/2) + a) or
          update = True
          orbitside = np.sign(new_step[1] - r0)
          if mu != 0.0:
              condate = True
              con_derv.append([i, *mm.peters_integrate3(constants[-1], a, mu, all_states, con_derv[-1][0], i), new_step[0]])
              dir1 = new_step[5]
              new_step = mm.new_recalc_state3(con_derv[-1][1], new_step, mu, mass, a)                           #Change the velocity so it actually makes sense with those constants
              test = mm.check_interval(mm.kerr, new_step, mass, a)
              if abs(test+1)>(err_target):
                  print("Error on PN Update")
                  print("Index:", i)
                  print(new_step)
                  print("Previous constants", constants[-1])
                  loop2 = True
                  #break
              if dir1 != new_step[5]:
                  checker.append([dir1, new_step[5], con_derv[-1][1]  ])
      #Initializing for the next step
      #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
      #Only happens the step before the derivatives are recalculated.
      
      #new interval thing
      trial = 0
      og_new_step = np.copy(new_step)
      while loop2 == True:
        ut, up = new_step[4], new_step[7]
        err = mm.check_interval(mm.kerr, new_step, mass, a)
        metric = mm.kerr(new_step, mass, a)[0]
        gtt, gtp = metric[0][0], metric[0][3]
        disc = np.sqrt(ut**2 + ((gtp/gtt)**2)*(up**2) + 2*(gtp/gtt)*ut*up - (1.0 + err)/gtt)
        int_steps = [np.copy(new_step), np.copy(new_step)]
        int_steps[0][4] += -ut - (gtp/gtt)*up + disc
        int_steps[1][4] += -ut - (gtp/gtt)*up - disc
        checks = [(mm.check_interval(mm.kerr, int_steps[0], mass, a)+1)**2, (mm.check_interval(mm.kerr, int_steps[1], mass, a)+1)**2]
        test = mm.check_interval(mm.kerr, int_steps[checks.index(min(checks))], mass, a)
        trial += 1
        new_step = int_steps[checks.index(min(checks))]
        if (abs(test+1)<=(err_target)) and (int_steps[checks.index(min(checks))][4] > 0):
          loop2 = False
        if trial >= 10:
          print("giving up!", trial, test)
          new_step = np.copy(og_new_step)
          break
      
      #Update stuff!
      if update == True:
         metric = mm.kerr(new_step, mass, a)[0]
         newE = -np.matmul(new_step[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
         newLz = np.matmul(new_step[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
         newQ = np.matmul(np.matmul(mm.kill_tensor(new_step, mass, a), new_step[4:]), new_step[4:])    #new Carter constant Q
         newC = newQ - (a*newE - newLz)**2                                                             #initial adjusted Carter constant                      
         constants.append([newE, newLz, newC])
         qarter.append(newQ)
         if condate == False:
             con_derv.append([i, *con_derv[-1][1:10], new_step[0]])
             
         tracktime.append(new_step[0])
         
      if con_derv[-1][9] == True:
         compErr += 1
         issues.append((i, new_step[0]))
         
      interval.append(mm.check_interval(mm.kerr, new_step, mass, a))
      false_constants.append([getEnergy(new_step, mass, a), *getLs(new_step, mu)])
      dTau_change.append(old_dTau)
      all_states.append(new_step )    #update position and velocity
      i += 1

      if new_step[1] < rmin:
        rmin = new_step[1]
      progress = max(new_step[0]/max_time, i/(50*max_time)) * 100
      if verbose == True:
          if (progress >= milestone):
            print("Program has completed " + str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
            if (new_step[3]//(2*np.pi)) != orbitCount:
                print("Number of orbits = " + str(round(new_step[3]/(2*np.pi), 2)) + ", t = " + str(new_step[0]) + ", r_min = " + str(rmin))
                orbitCount = round(new_step[3]/(2*np.pi))
            milestone += 1

    #Lets you end the program before the established end without breaking anything
    except KeyboardInterrupt:
      print("Ending program")
      cap = len(all_states) - 1
      all_states = all_states[:cap]
      interval = interval[:cap]
      dTau_change = dTau_change[:cap]
      constants = constants[:cap]
      qarter = qarter[:cap]
      break

  r = all_states[0][1]
  constants = np.array(constants)
  false_constants = np.array(false_constants)
  qarter = np.array(qarter)
  con_derv = np.array(con_derv, dtype=object)
  interval = np.array(interval)
  dTau_change = np.array(dTau_change)
  all_states = np.array(all_states)
  tracktime = np.array(tracktime)
  if verbose == True:
      print("There were " + str(compErr) + " issues with complex roots/turning points.")
      for i in issues:
        print("Complex value at index " + str(i[0]) + ", t = " + str(i[1]))
      #print("Here's some data!")
      #print(checker)
  final = {"name": label,
           "raw": all_states,
           "inputs": inputs,
           "dTau": dTau,
           "pos": all_states[:,1:4],
           "all_vel": all_states[:,4:], 
           "time": all_states[:,0],
           "interval": interval,
           "vel": (np.square(all_states[:,5]) + np.square(all_states[:,1]) * (np.square(all_states[:,6]) + np.square(all_states[:,7])))**(0.5),
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
                              ((r**(3/2) + pro*a)**(-1))]),
           "r0": con_derv[:,2],
           "y": con_derv[:,3],
           "v": con_derv[:,4],
           "q": con_derv[:,5],
           "e": con_derv[:,6],
           "ot": con_derv[:,7],
           "it": con_derv[:,8],
           "tracktime": con_derv[:,10]}
  return final

def clean_inspiral(mass, a, mu, max_time, err_target, label, cons=False, velorient=False, vel4=False, params=False, pos=False, veltrue=False, verbose=False):
  #basically the same as schwarz, except references to schwarz changed to kerr
  inputs = [mass, a, mu, max_time, err_target, label, cons, velorient, vel4, params, pos]          #Grab initial input in case you want to run the continue function
  all_states = [[np.zeros(8)]]                                                  #Grab that initial state         
  err_calc = 1                                                                  #initialize calculated error
  i = 0                                                                         #initialize step counter

  if (np.shape(veltrue) == (4,)) and (np.shape(pos) == (4,)):
      all_states[0] = [*pos, *veltrue]
  else:
      print("Normalizing initial state")
      all_states[0] = mm.set_u_kerr2(mass, a, cons, velorient, vel4, params, pos)      #normalize initial state so it's actually physical
  pro = np.sign(all_states[0][7])

  interval = [mm.check_interval(mm.kerr, all_states[0], mass, a)]           #create interval tracker
  dTau = 0.1
  dTau_change = [dTau]                                                #create dTau tracker
  metric = mm.kerr(all_states[0], mass, a)[0]                                      #initial metric
  initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
  initLz = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
  initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], mass, a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
  initC = initQ - (a*initE - initLz)**2                                          #initial adjusted Carter constant                      
  constants = [ np.array([initE,      #energy   
                          initLz,      #angular momentum (axial)
                          initC]) ]    #Carter constant (C)
  qarter = [initQ]           #Carter constant (Q)

  false_constants = [np.array([getEnergy(all_states[0], mass, a), *getLs(all_states[0], mu)])]  #Cartesian approximation of L vector

  con_derv = [[0, *mm.peters_integrate3(constants[0], a, mu, all_states, 0, 0), 0.0]]    
  
  compErr = 0
  milestone = 0
  rmin = all_states[0][1]
  issues = []
  checker = []
  orbitside = np.sign(all_states[0][1] - con_derv[0][2])
  orbitCount = 0
  tracktime = [all_states[0][0]]
  stop = False
  #Main Loop
  while (all_states[i][0] < max_time and i < 50*max_time):
    try:
      update = False
      condate = False
      first = True
      loop2 = False

      #Grab the current state
      state = all_states[i]  
      r0 = con_derv[-1][2]   
      #print("r0=",r0)

      #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
      if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001) :
        break
      #elif (state[1] >= (1 + np.sqrt(1 - a**2))*(10**6)):    #Currently based on event horizon, might change it to be based on r0
      #  break
        
      #Runge-Kutta update using geodesic
      while (err_calc >= err_target) or (first == True):
        step_check = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, mass, a)
        new_step = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, mass, a) 
        mod_step = np.append(step_check[0:6], mm.fix_sin(step_check[6])*mm.fix_cos(step_check[7]))
        mod_new = np.append(new_step[0:6], mm.fix_sin(new_step[6])*mm.fix_cos(new_step[7]))
        mod_state = np.append(state[0:6], mm.fix_sin(state[6])*mm.fix_cos(state[7]))
        mod_pre = mod_step - mod_new
        mod_pre[:4] = mod_pre[:4]/np.linalg.norm(mod_state[:4]) 
        mod_pre[4:] = mod_pre[4:]/np.linalg.norm(mod_state[4:])
        err_calc = np.linalg.norm(mod_pre)
        #print(err_calc)
        old_dTau, dTau = dTau, min(0.95 * dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
        first = False

      #constant modifying section
      #Whenever you pass from one side of r0 to the other, mess with the effective potential.
      if ( np.sign(new_step[1] - r0) != orbitside ):  # or (flip)):  #(new_step[0] - con_derv[-1][10]) > np.pi*(r0**(3/2) + a) or
          update = True
          orbitside = np.sign(new_step[1] - r0)
          if mu != 0.0:
              condate = True
              con_derv.append([i, *mm.peters_integrate3(constants[-1], a, mu, all_states, con_derv[-1][0], i), new_step[0]])
              dir1 = new_step[5]
              new_step = mm.new_recalc_state3(con_derv[-1][1], new_step, mu, mass, a)                           #Change the velocity so it actually makes sense with those constants
              test = mm.check_interval(mm.kerr, new_step, mass, a)
              if abs(test+1)>(err_target):
                  print("Error on PN Update")
                  print("Index:", i)
                  print(new_step)
                  print("Previous constants", constants[-1])
                  loop2 = True
                  #break
              if dir1 != new_step[5]:
                  checker.append([dir1, new_step[5], con_derv[-1][1]  ])
      #Initializing for the next step
      #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
      #Only happens the step before the derivatives are recalculated.
      
      #new interval thing
      trial = 0
      og_new_step = np.copy(new_step)
      while loop2 == True:
        ut, up = new_step[4], new_step[7]
        err = mm.check_interval(mm.kerr, new_step, mass, a)
        metric = mm.kerr(new_step, mass, a)[0]
        gtt, gtp = metric[0][0], metric[0][3]
        disc = np.sqrt(ut**2 + ((gtp/gtt)**2)*(up**2) + 2*(gtp/gtt)*ut*up - (1.0 + err)/gtt)
        int_steps = [np.copy(new_step), np.copy(new_step)]
        int_steps[0][4] += -ut - (gtp/gtt)*up + disc
        int_steps[1][4] += -ut - (gtp/gtt)*up - disc
        checks = [(mm.check_interval(mm.kerr, int_steps[0], mass, a)+1)**2, (mm.check_interval(mm.kerr, int_steps[1], mass, a)+1)**2]
        test = mm.check_interval(mm.kerr, int_steps[checks.index(min(checks))], mass, a)
        trial += 1
        new_step = int_steps[checks.index(min(checks))]
        if (abs(test+1)<=(err_target)) and (int_steps[checks.index(min(checks))][4] > 0):
          loop2 = False
        if trial >= 10:
          print("giving up!", trial, test)
          new_step = np.copy(og_new_step)
          break
      
      #Update stuff!
      if update == True:
         metric = mm.kerr(new_step, mass, a)[0]
         newE = -np.matmul(new_step[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
         newLz = np.matmul(new_step[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
         newQ = np.matmul(np.matmul(mm.kill_tensor(new_step, mass, a), new_step[4:]), new_step[4:])    #new Carter constant Q
         newC = newQ - (a*newE - newLz)**2                                                             #initial adjusted Carter constant                      
         constants.append([newE, newLz, newC])
         qarter.append(newQ)
         if condate == False:
             con_derv.append([i, *con_derv[-1][1:10], new_step[0]])
             
         tracktime.append(new_step[0])
         
      if con_derv[-1][9] == True:
         compErr += 1
         issues.append((i, new_step[0]))
         
      interval.append(mm.check_interval(mm.kerr, new_step, mass, a))
      false_constants.append([getEnergy(new_step, mass, a), *getLs(new_step, mu)])
      dTau_change.append(old_dTau)
      all_states.append(new_step )    #update position and velocity
      i += 1

      if new_step[1] < rmin:
        rmin = new_step[1]
      progress = max(new_step[0]/max_time, i/(50*max_time)) * 100
      if verbose == True:
          if (progress >= milestone):
            print("Program has completed " + str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
            if (new_step[3]//(2*np.pi)) != orbitCount:
                print("Number of orbits = " + str(round(new_step[3]/(2*np.pi), 2)) + ", t = " + str(new_step[0]) + ", r_min = " + str(rmin))
                orbitCount = round(new_step[3]/(2*np.pi))
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

  r = all_states[0][1]
  constants = np.array(constants)
  false_constants = np.array(false_constants)
  qarter = np.array(qarter)
  con_derv = np.array(con_derv, dtype=object)
  interval = np.array(interval)
  dTau_change = np.array(dTau_change)
  all_states = np.array(all_states)
  tracktime = np.array(tracktime)
  if verbose == True:
      print("There were " + str(compErr) + " issues with complex roots/turning points.")
      for i in issues:
        print("Complex value at index " + str(i[0]) + ", t = " + str(i[1]))
      #print("Here's some data!")
      #print(checker)
  final = {"name": label,
           "raw": all_states,
           "inputs": inputs,
           "pos": all_states[:,1:4],
           "all_vel": all_states[:,4:], 
           "time": all_states[:,0],
           "interval": interval,
           "vel": (np.square(all_states[:,5]) + np.square(all_states[:,1]) * (np.square(all_states[:,6]) + np.square(all_states[:,7])))**(0.5),
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
                              ((r**(3/2) + pro*a)**(-1))]),
           "r0": con_derv[:,2],
           "y": con_derv[:,3],
           "v": con_derv[:,4],
           "q": con_derv[:,5],
           "e": con_derv[:,6],
           "ot": con_derv[:,7],
           "it": con_derv[:,8],
           "tracktime": tracktime,
           "stop": stop}
  return final

def clean_continue(data, max_time=False, label=False, verbose=False):   
    inputs = np.copy(data["inputs"])
    posf, velf = data["raw"][-1,:4], data["raw"][-1,4:]
    inputs[3] = (data["raw"][-1,0])*2 if max_time==False else max_time
    inputs[5] = inputs[5] + " extend" if label==False else label 
    new_data = clean_inspiral(*inputs[:6], pos=posf, veltrue=velf, verbose=verbose)
    final = {"name": data["name"],
             "raw": np.concatenate((data["raw"][:-1], new_data["raw"])),
             "inputs": inputs,
             "pos": np.concatenate((data["pos"][:-1], new_data["pos"])),
             "all_vel": np.concatenate((data["all_vel"][:-1], new_data["all_vel"])), 
             "time": np.concatenate((data["time"][:-1], new_data["time"])),
             "interval": np.concatenate((data["interval"][:-1], new_data["interval"])),
             "vel": np.concatenate((data["vel"][:-1], new_data["vel"])),
             "dTau_change": np.concatenate((data["dTau_change"][:-1], new_data["dTau_change"])),
             "energy": np.concatenate((data["energy"], new_data["energy"])),
             "phi_momentum": np.concatenate((data["phi_momentum"], new_data["phi_momentum"])),
             "qarter":np.concatenate((data["qarter"], new_data["qarter"])),
             "carter": np.concatenate((data["carter"], new_data["carter"])),
             "energy2": np.concatenate((data["energy2"][:-1], new_data["energy2"])),
             "Lx_momentum": np.concatenate((data["Lx_momentum"][:-1], new_data["Lx_momentum"])),
             "Ly_momentum": np.concatenate((data["Ly_momentum"][:-1], new_data["Ly_momentum"])),
             "Lz_momentum": np.concatenate((data["Lz_momentum"][:-1], new_data["Lz_momentum"])),
             "spin": data["spin"],
             "freqs": data["freqs"],
             "r0": np.concatenate((data["r0"], new_data["r0"])),
             "y": np.concatenate((data["y"], new_data["y"])),
             "v": np.concatenate((data["v"], new_data["v"])),
             "q": np.concatenate((data["q"], new_data["q"])),
             "e": np.concatenate((data["e"], new_data["e"])),
             "ot": np.concatenate((data["ot"], new_data["ot"])),
             "it": np.concatenate((data["it"], new_data["it"])),
             "tracktime": np.concatenate((data["tracktime"], new_data["tracktime"])),
             "stop": data["stop"]}
    return final
    
def clean_inspiral2(mass, a, mu, endflag, err_target, label, cons=False, velorient=False, vel4=False, params=False, pos=False, veltrue=False, units="grav", verbose=False):
  #basically the same as schwarz, except references to schwarz changed to kerr
  
  termdict = {"time": "all_states[i][0]",
              "phi_orbit": "abs(all_states[i][3]/(2*np.pi))",
              "rad_orbit": "orbitCount"}
  
  try:
      terms = endflag.split(" ")
      newflag = termdict[terms[0]] + terms[1] + terms[2]
  except:
      print("Endflag should be a valid variable name, comparison operator, and numerical value, all separated by spaces")
      return 0
  
  inputs = [mass, a, mu, endflag, err_target, label, cons, velorient, vel4, params, pos, units]          #Grab initial input in case you want to run the continue function
  all_states = [[np.zeros(8)]]                                                  #Grab that initial state         
  err_calc = 1 
  err_calc2 = 1                                                                 #initialize calculated error
  i = 0                                                                         #initialize step counter
  
  if (np.shape(veltrue) == (4,)) and (np.shape(pos) == (4,)):
      all_states[0] = [*pos, *veltrue]
  else:
      print("Normalizing initial state")
      all_states[0] = mm.set_u_kerr2(mass, a, cons, velorient, vel4, params, pos, units)      #normalize initial state so it's actually physical
  pro = np.sign(all_states[0][7])
  M = mass
  mass = 1.0

  interval = [mm.check_interval(mm.kerr, all_states[0], mass, a)]           #create interval tracker
  dTau = 0.1
  dTau_change = [dTau]                                                #create dTau tracker
  metric = mm.kerr(all_states[0], mass, a)[0]                                      #initial metric

  if np.shape(cons) == (3,):
      initE, initLz, initC = cons
      initQ = initC + (a*initE - initLz)**2
  else:
      initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
      initLz = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
      initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], mass, a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
      initC = initQ - (a*initE - initLz)**2                                          #initial adjusted Carter constant                      
  constants = [ np.array([initE,      #energy   
                          initLz,      #angular momentum (axial)
                          initC]) ]    #Carter constant (C)
  qarter = [initQ]           #Carter constant (Q)
  
  false_constants = [np.array([getEnergy(all_states[0], mass, a), *getLs(all_states[0], mu)])]  #Cartesian approximation of L vector

  con_derv = [[0, *mm.peters_integrate3(constants[0], a, mu, all_states, 0, 0), 0.0]]    
  #contains: [[index, np.array([dedt, dlxdt, dlydt, dlzdt]),      r0,         y,   v,            q,   e,   outer_turn, inner_turn, compErr, timestamp]]
             #[1,     np.array([(c**5)/(G*M), c**2, c**2, c**2]), G*M/(c**2), 1.0, c/np.sqrt(G), 1/M, 1.0, G*M/(c**2), G*M/(c**2), 1      , G*M/(c**3)] 
  
  compErr = 0
  milestone = 0
  rmin = all_states[0][1]
  issues = []
  checker = []
  orbitside = np.sign(all_states[0][1] - con_derv[0][2])
  if orbitside == 0:
      orbitside = -1
  print(con_derv[0][2])
  #print(constants)
  r = con_derv[0][2]
  #print(r, pro, a)
  #print(((r**(3/2) + pro*a)**(-1)), np.sqrt(1 - (6/r) + pro*(8*a*(r**(-3/2))) - (3*((a/r)**(2)))))
  orbitCount = all_states[-1][0]/(2*np.pi/(((r**(3/2) + pro*a)**(-1))*(1 - (6/r) + pro*(8*a*(r**(-3/2))) - (3*((a/r)**(2))))**(0.5)))
  tracktime = [all_states[0][0]]
  stop = False
  
  def sph2cartconv(state):
      cart_pos = np.array([state[0], state[1]*np.sin(state[2])*np.cos(state[3]), state[1]*np.sin(state[2])*np.sin(state[3]), state[1]*np.cos(state[2])])
      r, cost, sint, cosp, sinp = state[1], np.cos(state[2]), np.sin(state[2]), np.cos(state[3]), np.sin(state[3])
      sph2cart = np.array([[sint*cosp, r*cost*cosp, -r*sint*sinp],
                           [sint*sinp, r*cost*sinp, r*sint*cosp ],
                           [cost,      -r*sint,     0           ]])
      cart_vel = np.array([state[4], *np.dot(sph2cart, state[5:])])
      return cart_pos, cart_vel
  
  #Main Loop
  print(constants[0], all_states[0], mass, a)
  while (not(eval(newflag)) and i < 10**7):
    try:
      update = False
      condate = False
      first = True
      loop2 = False

      #Grab the current state
      state = all_states[i]  
      r0 = con_derv[-1][2]   
      #print("r0=",r0)

      #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
      if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001) :
        break
      #elif (state[1] >= (1 + np.sqrt(1 - a**2))*(10**6)):    #Currently based on event horizon, might change it to be based on r0
      #  break
        
      #Runge-Kutta update using geodesic
      old_dTau = dTau
      skip = False
      #print("new step")
      while ((err_calc >= err_target) or (first == True)) and (skip == False):
        step_check = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, mass, a)
        new_step = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, mass, a) 
        mod_step = np.append(step_check[0:6], np.sign(step_check[6])*np.sign(step_check[7])*(step_check[6]**2 + (np.cos(step_check[2])*step_check[7])**2)**0.5)
        mod_new = np.append(new_step[0:6], np.sign(new_step[6])*np.sign(new_step[7])*(new_step[6]**2 + (np.cos(new_step[2])*new_step[7])**2)**0.5)
        mod_state = np.append(state[0:6], np.sign(state[6])*np.sign(state[7])*(state[6]**2 + (np.cos(state[2])*state[7])**2)**0.5)
        mod_pre = mod_step - mod_new
        mod_pre[:4] = mod_pre[:4]/np.linalg.norm(mod_state[:4]) 
        mod_pre[4:] = mod_pre[4:]/np.linalg.norm(mod_state[4:])
        err_calc = np.linalg.norm(mod_pre)
        '''
        #print(mod_step)
        #print(mod_new)
        #print(mod_pre)
        #print(err_calc)
        cart_step_pos, cart_step_vel = sph2cartconv(step_check)
        cart_new_pos,  cart_new_vel  = sph2cartconv(new_step)
        err_calc_pos = 1 - (np.dot(cart_step_pos, cart_new_pos) / (np.linalg.norm(cart_step_pos)*np.linalg.norm(cart_new_pos)))
        err_calc_vel = 1 - (np.dot(cart_step_vel, cart_new_vel) / (np.linalg.norm(cart_step_vel)*np.linalg.norm(cart_new_vel)))
        err_calc2 = (err_calc_pos**2 + err_calc_vel**2)**0.5
        #print(err_calc2, new_step[0] - state[0])
        #print(" ")
        #3.141592653589793
        #6.283185307179586
        '''
        old_dTau, dTau = dTau, min(dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
        first = False
      #print(" ")
      #print(new_step[2]/np.pi, new_step[0] - state[0])
      metric = mm.kerr(new_step, mass, a)[0]
      newE = -np.matmul(new_step[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
      newLz = np.matmul(new_step[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
      newQ = np.matmul(np.matmul(mm.kill_tensor(new_step, mass, a), new_step[4:]), new_step[4:])    #new Carter constant Q
      newC = newQ - (a*newE - newLz)**2   
      
      #if np.linalg.norm(constants[-1] - np.array([newE, newLz, newC])) > err_target:
          #print(i, "YOU FOOL", np.linalg.norm(constants[-1] - np.array([newE, newLz, newC])))

      #constant modifying section
      #Whenever you pass from one side of r0 to the other, mess with the effective potential.
      #print("yo?")
      #print(new_step)
      #print("dude", i)
      #print("Y", mm.check_interval(mm.kerr, new_step, mass, a))
      if abs(mm.check_interval(mm.kerr, new_step, mass, a)+1)>(err_target):
          #print("bad?")
          #print(new_step)
          hold = mm.recalc_state(constants[-1], new_step, mass, a)
          #print(hold)
          #print(new_step - hold)
          check = mm.check_interval(mm.kerr, hold, mass, a)
          #print("old", mm.check_interval(mm.kerr, new_step, mass, a)+1)
          #print("new", check+1)
      if ( np.sign(new_step[1] - r0) != orbitside) or ((new_step[3] - all_states[con_derv[-1][0]][3] > np.pi*(3/2)) and (np.std([state[1] for state in all_states[con_derv[-1][0]:]]) < 0.01*np.mean([state[1] for state in all_states[con_derv[-1][0]:]]))):
          '''
          print( np.sign(new_step[1] - r0) != orbitside)
          print(new_step[1])
          print(r0)
          print(new_step[3] - all_states[con_derv[-1][0]][3] > np.pi)
          print(np.std([state[1] for state in all_states[con_derv[-1][0]:]]) < 0.01*np.mean([state[1] for state in all_states[con_derv[-1][0]:]]))
          '''
          #print("pop")
          #print(all_states[-5:])
          #print(new_step)
          #print("party time", i, r0, np.sign(new_step[1] - r0), orbitside)
          testTau = old_dTau
          '''
          while ( np.abs(new_step[1] - r0) > 10**(-8)): #Changes the previous step so it lands exactly on r0, or close enough to not matter
              #print("fix")
              testTau = testTau*(np.abs(state[1] - r0)/np.abs(state[1] - new_step[1]))
              new_step = mm.gen_RK(mm.ck5, mm.kerr, state, testTau, mass, a)
              #print(i, testTau, (np.abs(state[1] - r0)/np.abs(state[1] - new_step[1])))
          '''
          #new_step = mm.recalc_state(constants[-1], new_step, mass, a)
          #This bit refits the new_step to match E,L,C
          #Turned off now, not sure how much it actually helps?
          
          update = True
          if ( np.sign(new_step[1] - r0) != orbitside):
              orbitside *= -1
          orbitCount += 0.5
          #print(orbitCount)
          #print("X", mm.check_interval(mm.kerr, new_step, mass, a))
          if mu != 0.0 and i - con_derv[-1][0] > 2:
              condate = True
              #print(all_states)
              #print(con_derv[-1][0], i)
              #print(mm.peters_integrate4(constants[-1], a, mu, all_states, con_derv[-1][0], i))
              con_derv.append([i, *mm.peters_integrate4(constants[-1], a, mu, all_states, con_derv[-1][0], i), new_step[0]])
              dir1 = new_step[5]
              if "new" not in label:
                  new_step = mm.new_recalc_state4(con_derv[-1][1], new_step, mu, mass, a)                           #Change the velocity so it actually makes sense with those constants
              else:
                  new_step, ch_cons = mm.new_recalc_state5(constants[-1], con_derv[-1][1], new_step, mu, mass, a) 
                  constants.append(ch_cons)
                  qarter.append(ch_cons[2] + (a*ch_cons[0] - ch_cons[1])**2 )
                  tracktime.append(new_step[0])
              test = mm.check_interval(mm.kerr, new_step, mass, a)
              #print("O", test)
              print("___________________________________")
              #print(ch_cons, new_step, mass, a)
              if abs(test+1)>(err_target):
                  #print(test, "woo")
        
                  print("Error on PN Update")
                  '''
                  print("Index:", i)
                  print(new_step)
                  print("Previous constants", constants[-1])
                  '''
                  loop2 = True
                  #break
              if dir1 != new_step[5]:
                  checker.append([dir1, new_step[5], con_derv[-1][1]  ])
      #Initializing for the next step
      #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
      #Only happens the step before the derivatives are recalculated.
      
      #new interval thing
      trial = 0
      og_new_step = np.copy(new_step)
      while loop2 == True:
        ut, up = new_step[4], new_step[7]
        err = mm.check_interval(mm.kerr, new_step, mass, a)
        metric = mm.kerr(new_step, mass, a)[0]
        gtt, gtp = metric[0][0], metric[0][3]
        disc = np.sqrt(ut**2 + ((gtp/gtt)**2)*(up**2) + 2*(gtp/gtt)*ut*up - (1.0 + err)/gtt)
        int_steps = [np.copy(new_step), np.copy(new_step)]
        int_steps[0][4] += -ut - (gtp/gtt)*up + disc
        int_steps[1][4] += -ut - (gtp/gtt)*up - disc
        checks = [(mm.check_interval(mm.kerr, int_steps[0], mass, a)+1)**2, (mm.check_interval(mm.kerr, int_steps[1], mass, a)+1)**2]
        test = mm.check_interval(mm.kerr, int_steps[checks.index(min(checks))], mass, a)
        trial += 1
        new_step = int_steps[checks.index(min(checks))]
        if (abs(test+1)<=(err_target)) and (int_steps[checks.index(min(checks))][4] > 0):
          #print("X", test)
          print("fixed!")
          #print(new_step - og_new_step)
          loop2 = False
        if trial >= 10:
          print("giving up!", trial, test)
          new_step = np.copy(og_new_step)
          break
      
      #Update stuff!
      if (update == True) and ("new" not in label):
         #print("Constant Change!")
         metric = mm.kerr(new_step, mass, a)[0]
         newE = -np.matmul(new_step[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
         newLz = np.matmul(new_step[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
         newQ = np.matmul(np.matmul(mm.kill_tensor(new_step, mass, a), new_step[4:]), new_step[4:])    #new Carter constant Q
         newC = newQ - (a*newE - newLz)**2                                                             #initial adjusted Carter constant  
         #print(newE, newLz, newC, "FROM MAIN")
         coeff = np.array([newE**2 - 1, 2*mass, (a**2)*(newE**2 - 1) - newLz**2 - newC, 2*mass*((a*newE - newLz)**2 + newC), -newC*(a**2)])
         coeff2 = np.array([4*(newE**2 - 1), 3*2*mass, 2*((a**2)*(newE**2 - 1) - newLz**2 - newC), 2*mass*((a*newE - newLz)**2 + newC)])
         #print(coeff2)
         ro = np.max(np.roots(coeff2))
         #if np.sum(coeff*np.array([ro**4, ro**3, ro**2, ro, 1.0])) < 0.0:
             #print(metric)
             #print("Bad Constants??", np.sum(coeff*np.array([ro**4, ro**3, ro**2, ro, 1.0])))
             #print(newE, newLz, newC)
             #print("Bad Constants??", new_step[0], newE, newLz, newC, np.sum(coeff*np.array([ro**4, ro**3, ro**2, ro, 1.0])))
         constants.append([newE, newLz, newC])
         qarter.append(newQ)
         if condate == False:
             con_derv.append([i, *con_derv[-1][1:10], new_step[0]])
             
         tracktime.append(new_step[0])
         
      if con_derv[-1][9] == True:
         compErr += 1
         issues.append((i, new_step[0]))
         
      interval.append(mm.check_interval(mm.kerr, new_step, mass, a))
      false_constants.append([getEnergy(new_step, mass, a), *getLs(new_step, mu)])
      dTau_change.append(old_dTau)
      all_states.append(new_step )    #update position and velocity
      i += 1

      progress = max(1 - abs(eval(terms[2]) - eval(termdict[terms[0]]))/eval(terms[2]), i/(10**7) ) * 100
      if verbose == True:
          if (progress >= milestone):
            print("Program has completed " + str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
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
  
  #unit conversion stuff
  if units == "mks":
      G, c = 6.67*(10**-11), 3*(10**8)
  elif units == "cgs":
      G, c = 6.67*(10**-8),  3*(10**10)
  else:
      G, M, c = 1.0, 1.0, 1.0
  print(len(issues), len(all_states))
  constants = np.array([entry*np.array([M*mu*(c**2), M*M*mu*G/c, (M*M*mu*G/c)**2]) for entry in np.array(constants)], dtype=np.float64)
  false_constants = np.array(false_constants) #np.array([entry*np.array([c**2, G/(M*c), G/(M*c), G/(M*c)]) for entry in np.array(false_constants)])
  qarter = np.array(qarter) #np.array([entry * (G/c)**2 for entry in qarter])
  interval = np.array(interval)
  dTau_change = np.array([entry * (G*M)/(c**3) for entry in dTau_change])
  all_states = np.array([entry*np.array([(G*M)/(c**3), (G*M)/(c**2), 1.0, 1.0, 1.0, c, (c**3)/(G*M), (c**3)/(G*M)]) for entry in np.array(all_states)]) 
  #print(all_states[0])
  tracktime = np.array([entry * (G*M)/(c**3) for entry in tracktime])
  con_derv = np.array([entry * np.array([1, np.array([(c**5)/(G*M), c**2, c**2, c**2]), G*M/(c**2), 1.0, c/np.sqrt(G), 1/M, 1.0, G*M/(c**2), G*M/(c**2), 1, G*M/(c**3)]) for entry in con_derv])
  
  r = all_states[0][1]
  ind = argrelmin(all_states[:,1])[0]
  omega, otime = np.diff(all_states[:,2][ind]) - 2*np.pi, np.diff(all_states[:,0][ind])
  
  if verbose == True:
      print("There were " + str(compErr) + " issues with complex roots/turning points.")
      for i in issues:
        print("Complex value at index " + str(i[0]) + ", t = " + str(i[1]))
      #print("Here's some data!")
      #print(checker)
  final = {"name": label,
           "raw": all_states,
           "inputs": inputs,
           "pos": all_states[:,1:4],
           "all_vel": all_states[:,4:], 
           "time": all_states[:,0],
           "interval": interval,
           "vel": (np.square(all_states[:,5]) + np.square(all_states[:,1]) * (np.square(all_states[:,6]) + np.square(all_states[:,7])))**(0.5),
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
                              ((r**(3/2) + pro*a)**(-1))]) * (c**3)/(G*M),
           "r0": con_derv[:,2],
           "y": con_derv[:,3],
           "v": con_derv[:,4],
           "q": con_derv[:,5],
           "e": con_derv[:,6],
           "ot": con_derv[:,7],
           "it": con_derv[:,8],
           "tracktime": tracktime,
           "omegadot": omega/otime,
           "otime": all_states[:,0][ind][1:],
           "stop": stop}
  return final

def clean_continue2(data, endflag=False, label=False, verbose=False):   
    inputs = np.copy(data["inputs"])
    posf, velf = data["raw"][-1,:4], data["raw"][-1,4:]
    if endflag == False:
        terms = inputs[3].split(" ")
        inputs[3] = " ".join([terms[0], terms[1], str(2*eval(terms[2]))])
    else:
        inputs[3] = endflag
    inputs[5] = inputs[5] + " extend" if label==False else label 
    new_data = clean_inspiral2(*inputs[:6], pos=posf, veltrue=velf, units=inputs[-1], verbose=verbose)
    final = {"name": data["name"],
             "raw": np.concatenate((data["raw"][:-1], new_data["raw"])),
             "inputs": inputs,
             "pos": np.concatenate((data["pos"][:-1], new_data["pos"])),
             "all_vel": np.concatenate((data["all_vel"][:-1], new_data["all_vel"])), 
             "time": np.concatenate((data["time"][:-1], new_data["time"])),
             "interval": np.concatenate((data["interval"][:-1], new_data["interval"])),
             "vel": np.concatenate((data["vel"][:-1], new_data["vel"])),
             "dTau_change": np.concatenate((data["dTau_change"][:-1], new_data["dTau_change"])),
             "energy": np.concatenate((data["energy"], new_data["energy"])),
             "phi_momentum": np.concatenate((data["phi_momentum"], new_data["phi_momentum"])),
             "qarter":np.concatenate((data["qarter"], new_data["qarter"])),
             "carter": np.concatenate((data["carter"], new_data["carter"])),
             "energy2": np.concatenate((data["energy2"][:-1], new_data["energy2"])),
             "Lx_momentum": np.concatenate((data["Lx_momentum"][:-1], new_data["Lx_momentum"])),
             "Ly_momentum": np.concatenate((data["Ly_momentum"][:-1], new_data["Ly_momentum"])),
             "Lz_momentum": np.concatenate((data["Lz_momentum"][:-1], new_data["Lz_momentum"])),
             "spin": data["spin"],
             "freqs": data["freqs"],
             "r0": np.concatenate((data["r0"], new_data["r0"])),
             "y": np.concatenate((data["y"], new_data["y"])),
             "v": np.concatenate((data["v"], new_data["v"])),
             "q": np.concatenate((data["q"], new_data["q"])),
             "e": np.concatenate((data["e"], new_data["e"])),
             "ot": np.concatenate((data["ot"], new_data["ot"])),
             "it": np.concatenate((data["it"], new_data["it"])),
             "tracktime": np.concatenate((data["tracktime"], new_data["tracktime"])),
             "omegadot": np.concatenate((data["omegadot"], new_data["omegadot"])),
             "otime": np.concatenate((data["otime"], new_data["otime"])),
             "stop": data["stop"]}
    return final

def clean_inspiral2b(mass, a, mu, endflag, err_target, label, cons=False, velorient=False, vel4=False, params=False, pos=False, veltrue=False, units="grav", verbose=False):
  #basically the same as schwarz, except references to schwarz changed to kerr
  
  termdict = {"time": "all_states[i][0]",
              "phi_orbit": "abs(all_states[i][3]/(2*np.pi))",
              "rad_orbit": "orbitCount"}
  
  try:
      terms = endflag.split(" ")
      newflag = termdict[terms[0]] + terms[1] + terms[2]
  except:
      print("Endflag should be a valid variable name, comparison operator, and numerical value, all separated by spaces")
      return 0
  
  inputs = [mass, a, mu, endflag, err_target, label, cons, velorient, vel4, params, pos, units]          #Grab initial input in case you want to run the continue function
  all_states = [[np.zeros(8)]]                                                  #Grab that initial state         
  err_calc = 1 
  err_calc2 = 1                                                                 #initialize calculated error
  i = 0                                                                         #initialize step counter
  
  if (np.shape(veltrue) == (4,)) and (np.shape(pos) == (4,)):
      all_states[0] = [*pos, *veltrue]
  else:
      print("Normalizing initial state")
      all_states[0] = mm.set_u_kerr2(mass, a, cons, velorient, vel4, params, pos, units)      #normalize initial state so it's actually physical
  pro = np.sign(all_states[0][7])
  M = mass
  mass = 1.0

  interval = [mm.check_interval(mm.kerr, all_states[0], mass, a)]           #create interval tracker
  dTau = 0.1
  dTau_change = [dTau]                                                #create dTau tracker
  metric = mm.kerr(all_states[0], mass, a)[0]                                      #initial metric

  if np.shape(cons) == (3,):
      initE, initLz, initC = cons
      initQ = initC + (a*initE - initLz)**2
  else:
      initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
      initLz = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
      initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], mass, a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
      initC = initQ - (a*initE - initLz)**2                                          #initial adjusted Carter constant                      
  constants = [ np.array([initE,      #energy   
                          initLz,      #angular momentum (axial)
                          initC]) ]    #Carter constant (C)
  qarter = [initQ]           #Carter constant (Q)
  
  false_constants = [np.array([getEnergy(all_states[0], mass, a), *getLs(all_states[0], mu)])]  #Cartesian approximation of L vector

  con_derv = [[0, *mm.peters_integrate3(constants[0], a, mu, all_states, 0, 0), 0.0]]    
  #contains: [[index, np.array([dedt, dlxdt, dlydt, dlzdt]),      r0,         y,   v,            q,   e,   outer_turn, inner_turn, compErr, timestamp]]
             #[1,     np.array([(c**5)/(G*M), c**2, c**2, c**2]), G*M/(c**2), 1.0, c/np.sqrt(G), 1/M, 1.0, G*M/(c**2), G*M/(c**2), 1      , G*M/(c**3)] 
  
  compErr = 0
  milestone = 0
  rmin = all_states[0][1]
  issues = []
  checker = []
  orbitside = np.sign(all_states[0][1] - con_derv[0][2])
  if orbitside == 0:
      orbitside = -1
  print(con_derv[0][2])
  r = con_derv[0][2]
  orbitCount = all_states[-1][0]/(2*np.pi/(((r**(3/2) + pro*a)**(-1))*(1 - (6/r) + pro*(8*a*(r**(-3/2))) - (3*((a/r)**(2))))**(0.5)))
  tracktime = [all_states[0][0]]
  stop = False
  
  def sph2cartconv(state):
      cart_pos = np.array([state[0], state[1]*np.sin(state[2])*np.cos(state[3]), state[1]*np.sin(state[2])*np.sin(state[3]), state[1]*np.cos(state[2])])
      r, cost, sint, cosp, sinp = state[1], np.cos(state[2]), np.sin(state[2]), np.cos(state[3]), np.sin(state[3])
      sph2cart = np.array([[sint*cosp, r*cost*cosp, -r*sint*sinp],
                           [sint*sinp, r*cost*sinp, r*sint*cosp ],
                           [cost,      -r*sint,     0           ]])
      cart_vel = np.array([state[4], *np.dot(sph2cart, state[5:])])
      return cart_pos, cart_vel
  
  def viable_cons(constants, state, mass, a):
      energy, lz, cart = constants
      coeff = np.array([energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*mass*((a*energy - lz)**2 + cart), -cart*(a**2)])
      coeff2 = np.array([(energy**2 - 1)*4, (2.0*mass)*3, ((a**2)*(energy**2 - 1) - lz**2 - cart)*2, 2*mass*((a*energy - lz)**2) + 2*mass*cart])
      r0 = max(np.roots(coeff2))
      potential_min = np.polyval(coeff, r0)
      return potential_min, r0
    
  #Main Loop
  print(constants[0], all_states[0], mass, a)
  while (not(eval(newflag)) and i < 10**7):
    try:
      update = False
      condate = False
      first = True
      loop2 = False

      #Grab the current state
      state = all_states[i]  
      r0 = con_derv[-1][2]   
      #print("r0=",r0)

      #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
      if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001) :
        break
      #elif (state[1] >= (1 + np.sqrt(1 - a**2))*(10**6)):    #Currently based on event horizon, might change it to be based on r0
      #  break
        
      #Runge-Kutta update using geodesic
      old_dTau = dTau
      skip = False
      #print("new step")
      while ((err_calc >= err_target) or (first == True)) and (skip == False):
        step_check = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, mass, a)
        new_step = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, mass, a) 
        mod_step = np.append(step_check[0:6], np.sign(step_check[6])*np.sign(step_check[7])*(step_check[6]**2 + (np.cos(step_check[2])*step_check[7])**2)**0.5)
        mod_new = np.append(new_step[0:6], np.sign(new_step[6])*np.sign(new_step[7])*(new_step[6]**2 + (np.cos(new_step[2])*new_step[7])**2)**0.5)
        mod_state = np.append(state[0:6], np.sign(state[6])*np.sign(state[7])*(state[6]**2 + (np.cos(state[2])*state[7])**2)**0.5)
        mod_pre = mod_step - mod_new
        mod_pre[:4] = mod_pre[:4]/np.linalg.norm(mod_state[:4]) 
        mod_pre[4:] = mod_pre[4:]/np.linalg.norm(mod_state[4:])
        err_calc = np.linalg.norm(mod_pre)
        old_dTau, dTau = dTau, min(dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
        first = False

      metric = mm.kerr(new_step, mass, a)[0]
      newE = -np.matmul(new_step[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
      newLz = np.matmul(new_step[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
      newQ = np.matmul(np.matmul(mm.kill_tensor(new_step, mass, a), new_step[4:]), new_step[4:])    #new Carter constant Q
      newC = newQ - (a*newE - newLz)**2   
      
      #if np.linalg.norm(constants[-1] - np.array([newE, newLz, newC])) > err_target:
          #print(i, "YOU FOOL", np.linalg.norm(constants[-1] - np.array([newE, newLz, newC])))

      #constant modifying section
      #Whenever you pass from one side of r0 to the other, mess with the effective potential.
      if ( np.sign(new_step[1] - r0) != orbitside) or ((new_step[3] - all_states[con_derv[-1][0]][3] > np.pi*(3/2)) and (np.std([state[1] for state in all_states[con_derv[-1][0]:]]) < 0.01*np.mean([state[1] for state in all_states[con_derv[-1][0]:]]))):
          testTau = old_dTau
          update = True
          if ( np.sign(new_step[1] - r0) != orbitside):
              orbitside *= -1
          orbitCount += 0.5
          if mu != 0.0 and i - con_derv[-1][0] > 2:
              condate = True
              con_derv.append([i, *mm.peters_integrate4(constants[-1], a, mu, all_states, con_derv[-1][0], i), new_step[0]])
              dir1 = new_step[5]
              if "new" in label:
                  print("new")
                  new_step, ch_cons = mm.new_recalc_state5(constants[-1], con_derv[-1][1], new_step, mu, mass, a) 
                  constants.append(ch_cons)
                  qarter.append(ch_cons[2] + (a*ch_cons[0] - ch_cons[1])**2 )
                  tracktime.append(new_step[0])
              elif "kachow" in label:
                  print("kachow")
                  new_step, ch_cons = mm.new_recalc_state6(constants[-1], con_derv[-1][1], new_step, mu, mass, a) 
                  pot_min = viable_cons(ch_cons, new_step, mass, a)
                  if pot_min[0] < -err_target:
                      print("please?")
                      Lphi, Cart, ro = *ch_cons[1:], pot_min[1]
                      ch_cons[0] = (4*a*Lphi*r0 + ((4*a*Lphi*ro)**2 - 4*(ro**4 + 2*ro*(a**2))*((a*Lphi)**2 - (ro**2 - 2*ro + a**2)*(ro**2 + Lphi**2 + Cart)))**(0.5))/(2*(ro**4 + 2*ro*(a**2)))
                      #new_step = mm.recalc_state(ch_cons, new_step, mass, a)
                  constants.append(ch_cons)
                  qarter.append(ch_cons[2] + (a*ch_cons[0] - ch_cons[1])**2 )
                  tracktime.append(new_step[0])
              else:
                  print("neither")
                  new_step = mm.new_recalc_state4(con_derv[-1][1], new_step, mu, mass, a)                           #Change the velocity so it actually makes sense with those constants
                  
              test = mm.check_interval(mm.kerr, new_step, mass, a)
              #print("O", test)
              print("___________________________________")
              #print(ch_cons, new_step, mass, a)
              if abs(test+1)>(err_target):
                  #print(test, "woo")
        
                  print("Error on PN Update")
                  '''
                  print("Index:", i)
                  print(new_step)
                  print("Previous constants", constants[-1])
                  '''
                  loop2 = True
                  #break
              if dir1 != new_step[5]:
                  checker.append([dir1, new_step[5], con_derv[-1][1]  ])
      #Initializing for the next step
      #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
      #Only happens the step before the derivatives are recalculated.
      
      #new interval thing
      trial = 0
      og_new_step = np.copy(new_step)
      while loop2 == True:
        ut, up = new_step[4], new_step[7]
        err = mm.check_interval(mm.kerr, new_step, mass, a)
        metric = mm.kerr(new_step, mass, a)[0]
        gtt, gtp = metric[0][0], metric[0][3]
        disc = np.sqrt(ut**2 + ((gtp/gtt)**2)*(up**2) + 2*(gtp/gtt)*ut*up - (1.0 + err)/gtt)
        int_steps = [np.copy(new_step), np.copy(new_step)]
        int_steps[0][4] += -ut - (gtp/gtt)*up + disc
        int_steps[1][4] += -ut - (gtp/gtt)*up - disc
        checks = [(mm.check_interval(mm.kerr, int_steps[0], mass, a)+1)**2, (mm.check_interval(mm.kerr, int_steps[1], mass, a)+1)**2]
        test = mm.check_interval(mm.kerr, int_steps[checks.index(min(checks))], mass, a)
        trial += 1
        new_step = int_steps[checks.index(min(checks))]
        if (abs(test+1)<=(err_target)) and (int_steps[checks.index(min(checks))][4] > 0):
          #print("X", test)
          print("fixed!")
          #print(new_step - og_new_step)
          loop2 = False
        if trial >= 10:
          print("giving up!", trial, test)
          new_step = np.copy(og_new_step)
          break
      
      #Update stuff!
      if (update == True) and ("new" not in label) and ("kachow" not in label):
         #print("Constant Change!")
         metric = mm.kerr(new_step, mass, a)[0]
         newE = -np.matmul(new_step[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
         newLz = np.matmul(new_step[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
         newQ = np.matmul(np.matmul(mm.kill_tensor(new_step, mass, a), new_step[4:]), new_step[4:])    #new Carter constant Q
         newC = newQ - (a*newE - newLz)**2                                                             #initial adjusted Carter constant  
         #print(newE, newLz, newC, "FROM MAIN")
         coeff = np.array([newE**2 - 1, 2*mass, (a**2)*(newE**2 - 1) - newLz**2 - newC, 2*mass*((a*newE - newLz)**2 + newC), -newC*(a**2)])
         coeff2 = np.array([4*(newE**2 - 1), 3*2*mass, 2*((a**2)*(newE**2 - 1) - newLz**2 - newC), 2*mass*((a*newE - newLz)**2 + newC)])
         #print(coeff2)
         ro = np.max(np.roots(coeff2))
         #if np.sum(coeff*np.array([ro**4, ro**3, ro**2, ro, 1.0])) < 0.0:
             #print(metric)
             #print("Bad Constants??", np.sum(coeff*np.array([ro**4, ro**3, ro**2, ro, 1.0])))
             #print(newE, newLz, newC)
             #print("Bad Constants??", new_step[0], newE, newLz, newC, np.sum(coeff*np.array([ro**4, ro**3, ro**2, ro, 1.0])))
         constants.append([newE, newLz, newC])
         qarter.append(newQ)
         if condate == False:
             con_derv.append([i, *con_derv[-1][1:10], new_step[0]])
             
         tracktime.append(new_step[0])
         
      if con_derv[-1][9] == True:
         compErr += 1
         issues.append((i, new_step[0]))
         
      interval.append(mm.check_interval(mm.kerr, new_step, mass, a))
      false_constants.append([getEnergy(new_step, mass, a), *getLs(new_step, mu)])
      dTau_change.append(old_dTau)
      all_states.append(new_step )    #update position and velocity
      i += 1

      progress = max(1 - abs(eval(terms[2]) - eval(termdict[terms[0]]))/eval(terms[2]), i/(10**7) ) * 100
      if verbose == True:
          if (progress >= milestone):
            print("Program has completed " + str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
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
  
  #unit conversion stuff
  if units == "mks":
      G, c = 6.67*(10**-11), 3*(10**8)
  elif units == "cgs":
      G, c = 6.67*(10**-8),  3*(10**10)
  else:
      G, M, c = 1.0, 1.0, 1.0
  print(len(issues), len(all_states))
  constants = np.array([entry*np.array([M*mu*(c**2), M*M*mu*G/c, (M*M*mu*G/c)**2]) for entry in np.array(constants)], dtype=np.float64)
  false_constants = np.array(false_constants) #np.array([entry*np.array([c**2, G/(M*c), G/(M*c), G/(M*c)]) for entry in np.array(false_constants)])
  qarter = np.array(qarter) #np.array([entry * (G/c)**2 for entry in qarter])
  interval = np.array(interval)
  dTau_change = np.array([entry * (G*M)/(c**3) for entry in dTau_change])
  all_states = np.array([entry*np.array([(G*M)/(c**3), (G*M)/(c**2), 1.0, 1.0, 1.0, c, (c**3)/(G*M), (c**3)/(G*M)]) for entry in np.array(all_states)]) 
  #print(all_states[0])
  tracktime = np.array([entry * (G*M)/(c**3) for entry in tracktime])
  con_derv = np.array([entry * np.array([1, np.array([(c**5)/(G*M), c**2, c**2, c**2]), G*M/(c**2), 1.0, c/np.sqrt(G), 1/M, 1.0, G*M/(c**2), G*M/(c**2), 1, G*M/(c**3)]) for entry in con_derv])
  
  r = all_states[0][1]
  ind = argrelmin(all_states[:,1])[0]
  omega, otime = np.diff(all_states[:,2][ind]) - 2*np.pi, np.diff(all_states[:,0][ind])
  
  if verbose == True:
      print("There were " + str(compErr) + " issues with complex roots/turning points.")
      for i in issues:
        print("Complex value at index " + str(i[0]) + ", t = " + str(i[1]))
      #print("Here's some data!")
      #print(checker)
  final = {"name": label,
           "raw": all_states,
           "inputs": inputs,
           "pos": all_states[:,1:4],
           "all_vel": all_states[:,4:], 
           "time": all_states[:,0],
           "interval": interval,
           "vel": (np.square(all_states[:,5]) + np.square(all_states[:,1]) * (np.square(all_states[:,6]) + np.square(all_states[:,7])))**(0.5),
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
                              ((r**(3/2) + pro*a)**(-1))]) * (c**3)/(G*M),
           "r0": con_derv[:,2],
           "y": con_derv[:,3],
           "v": con_derv[:,4],
           "q": con_derv[:,5],
           "e": con_derv[:,6],
           "ot": con_derv[:,7],
           "it": con_derv[:,8],
           "tracktime": tracktime,
           "omegadot": omega/otime,
           "otime": all_states[:,0][ind][1:],
           "stop": stop}
  return final

def clean_inspiral3(mass, a, mu, endflag, err_target, label="default", cons=False, velorient=False, vel4=False, params=False, pos=False, veltrue=False, units="grav", verbose=False):
  #basically the same as schwarz, except references to schwarz changed to kerr
  
  def sph2cartconv(state):
      cart_pos = np.array([state[0], state[1]*np.sin(state[2])*np.cos(state[3]), state[1]*np.sin(state[2])*np.sin(state[3]), state[1]*np.cos(state[2])])
      r, cost, sint, cosp, sinp = state[1], np.cos(state[2]), np.sin(state[2]), np.cos(state[3]), np.sin(state[3])
      sph2cart = np.array([[sint*cosp, r*cost*cosp, -r*sint*sinp],
                           [sint*sinp, r*cost*sinp, r*sint*cosp ],
                           [cost,      -r*sint,     0           ]])
      cart_vel = np.array([state[4], *np.dot(sph2cart, state[5:])])
      return cart_pos, cart_vel
  
  def viable_cons(constants, state, mass, a):
      energy, lz, cart = constants
      coeff = np.array([energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*mass*((a*energy - lz)**2 + cart), -cart*(a**2)])
      coeff2 = np.array([(energy**2 - 1)*4, (2.0*mass)*3, ((a**2)*(energy**2 - 1) - lz**2 - cart)*2, 2*mass*((a*energy - lz)**2) + 2*mass*cart])
      r0 = max(np.roots(coeff2))
      potential_min = np.polyval(coeff, r0)
      return potential_min, r0
  
  termdict = {"time": "all_states[i][0]",
              "phi_orbit": "abs(all_states[i][3]/(2*np.pi))",
              "rad_orbit": "orbitCount"}
  
  try:
      terms = endflag.split(" ")
      newflag = termdict[terms[0]] + terms[1] + terms[2]
  except:
      print("Endflag should be a valid variable name, comparison operator, and numerical value, all separated by spaces")
      return 0
  
  inputs = [mass, a, mu, endflag, err_target, label, cons, velorient, vel4, params, pos, units]          #Grab initial input in case you want to run the continue function
  all_states = [[np.zeros(8)]]                                                  #Grab that initial state         
  err_calc = 1 
  err_calc2 = 1                                                                 #initialize calculated error
  i = 0                                                                         #initialize step counter
  
  if (np.shape(veltrue) == (4,)) and (np.shape(pos) == (4,)):
      all_states[0] = [*pos, *veltrue]
  else:
      print("Normalizing initial state")
      all_states[0], cons = mm.set_u_kerr2(mass, a, cons, velorient, vel4, params, pos, units)      #normalize initial state so it's actually physical
  pro = np.sign(all_states[0][7])
  M = mass
  mass = 1.0

  interval = [mm.check_interval(mm.kerr, all_states[0], mass, a)]           #create interval tracker
  metric = mm.kerr(all_states[0], mass, a)[0]                                      #initial metric

  if np.shape(cons) == (3,):
      initE, initLz, initC = cons
      print(cons)
      initQ = initC + (a*initE - initLz)**2
      pot_min = viable_cons([initE, initLz, initC], all_states[0], mass, a)
  else:
      initE = -np.matmul(all_states[0][4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
      initLz = np.matmul(all_states[0][4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
      initQ = np.matmul(np.matmul(mm.kill_tensor(all_states[0], mass, a), all_states[0][4:]), all_states[0][4:])    #initial Carter constant Q
      initC = initQ - (a*initE - initLz)**2                                          #initial adjusted Carter constant 
      pot_min = viable_cons([initE, initLz, initC], all_states[0], mass, a)
      if pot_min[0] < 0.0:
          print("RAGH")
          return False
              
  coeff = np.array([initE**2 - 1, 2.0*mass, (a**2)*(initE**2 - 1) - initLz**2 - initC, 2*mass*((a*initE - initLz)**2) + 2*mass*initC, -initC*(a**2)])
  coeff2 = np.array([(initE**2 - 1)*4, (2.0*mass)*3, ((a**2)*(initE**2 - 1) - initLz**2 - initC)*2, 2*mass*((a*initE - initLz)**2) + 2*mass*initC])
  r0, inner_turn, outer_turn = np.sort(np.roots(coeff2))[-1], *np.sort(np.real(np.roots(coeff)))[-2:]
  e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
  tracker = [[r0, e, inner_turn, outer_turn, all_states[0][0], 0]]
  if True in np.iscomplex(tracker[0]):
      #this = ((r0**2 + a**2)*initE - a*initLz)**2 - (r0**2 - 2*r0 + a**2)*(r0**2 + (initLz - a*initE)**2 + initC)
      #initE += (-this)*((2*r0*((r0**3 + r0*(a**2) + 2*(a**2))*initE - 2*initLz*a))**(-1))
      initE = (4*a*initLz*r0 + ((4*a*initLz*r0)**2 - 4*(r0**4 + 2*r0*(a**2))*((a*initLz)**2 - (r0**2 - 2*r0 + a**2)*(r0**2 + initLz**2 + initC)))**(0.5))/(2*(r0**4 + 2*r0*(a**2)))
      
      coeff = np.array([initE**2 - 1, 2.0*mass, (a**2)*(initE**2 - 1) - initLz**2 - initC, 2*mass*((a*initE - initLz)**2) + 2*mass*initC, -initC*(a**2)])
      coeff2 = np.array([(initE**2 - 1)*4, (2.0*mass)*3, ((a**2)*(initE**2 - 1) - initLz**2 - initC)*2, 2*mass*((a*initE - initLz)**2) + 2*mass*initC])
      r0, inner_turn, outer_turn = np.sort(np.roots(coeff2))[-1], *np.sort(np.roots(coeff))[-2:]
      e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
      tracker = [[r0, e, inner_turn, outer_turn, all_states[0][0], 0]]
  constants = [ np.array([initE,      #energy   
                          initLz,      #angular momentum (axial)
                          initC]) ]    #Carter constant (C)
  qarter = [initQ]           #Carter constant (Q)
  
  false_constants = [np.array([getEnergy(all_states[0], mass, a), *getLs(all_states[0], mu)])]  #Cartesian approximation of L vector

  #con_derv = [[0, *mm.peters_integrate5(all_states, a, mu, all_states, 0, 0), 0.0]]    
  #contains: [[index, np.array([dedt, dlxdt, dlydt, dlzdt]),      r0,         y,   v,            q,   e,   outer_turn, inner_turn, compErr, timestamp]]
             #[1,     np.array([(c**5)/(G*M), c**2, c**2, c**2]), G*M/(c**2), 1.0, c/np.sqrt(G), 1/M, 1.0, G*M/(c**2), G*M/(c**2), 1      , G*M/(c**3)] 
             
  tracker = [[r0, e, inner_turn, outer_turn, all_states[0][0], 0]]
  #r0, eccentricity, turning points, timestamp, index
  
  compErr = 0
  milestone = 0
  rmin = all_states[0][1]
  issues = []
  #checker = []
  orbitside = np.sign(all_states[0][1] - r0)
  if orbitside == 0:
      orbitside = -1

  orbitCount = all_states[-1][0]/(2*np.pi/(((r0**(3/2) + pro*a)**(-1))*(1 - (6/r0) + pro*(8*a*(r0**(-3/2))) - (3*((a/r0)**(2))))**(0.5)))
  stop = False

  if label == "default":
      A = (a**2)*(1 - initE**2)
      z2 = ((A + initLz**2 + initC) - ((A + initLz**2 + initC)**2 - 4*A*initC)**(1/2))/(2*A)
      inc = np.arccos(np.sqrt(z2))/np.pi
      label = "r" + str(r0) + "e" + str(e) + "zU+03C0" + str(inc) + "mu" + str(mu) + "a" + str(a)

  #Main Loop
  #print(constants[0], all_states[0], mass, a)
  dTau = np.abs(np.real((inner_turn/200)**(2)))
  dTau_change = [dTau]                                                #create dTau tracker
  borken = 0
  while (not(eval(newflag)) and i < 10**7):
    try:
      update = False
      condate = False
      first = True
      loop2 = False
      thunk = False

      #Grab the current state
      state = all_states[i]  
      r0 = tracker[-1][0]   
      #print("r0=",r0)

      #Break if you fall inside event horizon, or if you get really far away (orbit is unbound)
      if (state[1] <= (1 + np.sqrt(1 - a**2))*1.0001) :
        break
      #elif (state[1] >= (1 + np.sqrt(1 - a**2))*(10**6)):    #Currently based on event horizon, might change it to be based on r0
      #  break
        
      #Runge-Kutta update using geodesic
      old_dTau = dTau
      skip = False
      #print("new step")
      #print("thingy", i)
      pom = False
      while ((err_calc >= err_target) or (first == True)) and (skip == False) :
            new_step = mm.gen_RK(mm.ck4, mm.kerr, state, dTau, mass, a)
            step_check = mm.gen_RK(mm.ck5, mm.kerr, state, dTau, mass, a) 
            mod_step = np.append(step_check[0:6], np.sign(step_check[6])*np.sign(step_check[7])*(step_check[6]**2 + (np.cos(step_check[2])*step_check[7])**2)**0.5)
            mod_new = np.append(new_step[0:6], np.sign(new_step[6])*np.sign(new_step[7])*(new_step[6]**2 + (np.cos(new_step[2])*new_step[7])**2)**0.5)
            mod_state = np.append(state[0:6], np.sign(state[6])*np.sign(state[7])*(state[6]**2 + (np.cos(state[2])*state[7])**2)**0.5)
            mod_pre = mod_step - mod_new
            mod_pre[:4] = mod_pre[:4]/np.linalg.norm(mod_state[:4]) 
            mod_pre[4:] = mod_pre[4:]/np.linalg.norm(mod_state[4:])
            err_calc = np.linalg.norm(mod_pre)
            #err_calc = np.linalg.norm(new_step - step_check)
            err_calc = abs(1 - np.dot(new_step, step_check)/np.dot(new_step, new_step))
            #print(err_calc, "sph")
            if "bark" in label:
                new_cart = np.array([new_step[0],
                                     new_step[1]*np.sin(new_step[2])*np.cos(new_step[3]),
                                     new_step[1]*np.sin(new_step[2])*np.sin(new_step[3]),
                                     new_step[1]*np.cos(new_step[2]),
                                     new_step[4],
                                     new_step[5]*np.sin(new_step[2])*np.cos(new_step[3]) + new_step[1]*new_step[6]*np.cos(new_step[2])*np.cos(new_step[3]) - new_step[1]*new_step[7]*np.sin(new_step[2])*np.sin(new_step[3]),
                                     new_step[5]*np.sin(new_step[2])*np.sin(new_step[3]) + new_step[1]*new_step[6]*np.cos(new_step[2])*np.sin(new_step[3]) + new_step[1]*new_step[7]*np.sin(new_step[2])*np.cos(new_step[3]),
                                     new_step[5]*np.cos(new_step[2]) - new_step[1]*new_step[6]*np.sin(new_step[2])])
                cart_check=np.array([step_check[0],
                                     step_check[1]*np.sin(step_check[2])*np.cos(step_check[3]),
                                     step_check[1]*np.sin(step_check[2])*np.sin(step_check[3]),
                                     step_check[1]*np.cos(step_check[2]),
                                     step_check[4],
                                     step_check[5]*np.sin(step_check[2])*np.cos(step_check[3]) + step_check[1]*step_check[6]*np.cos(step_check[2])*np.cos(step_check[3]) - step_check[1]*step_check[7]*np.sin(step_check[2])*np.sin(step_check[3]),
                                     step_check[5]*np.sin(step_check[2])*np.sin(step_check[3]) + step_check[1]*step_check[6]*np.cos(step_check[2])*np.sin(step_check[3]) + step_check[1]*step_check[7]*np.sin(step_check[2])*np.cos(step_check[3]),
                                     step_check[5]*np.cos(step_check[2]) - step_check[1]*step_check[6]*np.sin(step_check[2])])
                err_calc = abs(1 - np.dot(new_cart, cart_check)/np.dot(new_cart, new_cart))
                #print(np.linalg.norm(new_cart - cart_check), "othing")
            #print(0.5*np.pi*(1 - np.abs(state[2]%2 -1)))
            E, L, C = constants[-1]
            def anglething(angle):
                return 0.5*np.pi - np.abs(angle%np.pi - np.pi/2)
            #print(np.pi/2 - np.arccos(L/np.sqrt(L**2 + C)), "inc")
            #print(dTau, "dTau")
            #print(new_step[2], new_step[6], "anglestuff")
            #print(np.sign(new_step[6])*(np.ceil(new_step[2]/np.pi) + np.floor(new_step[2]/np.pi)) < -2*np.sign(new_step[6])*new_step[2]/np.pi)
            #if abs(np.pi/2 - np.arccos(L/np.sqrt(L**2 + C))) < 0.1 and ((anglething(new_step[2])<1e-8 and np.sign(new_step[6])*(np.ceil(new_step[2]/np.pi) + np.floor(new_step[2]/np.pi)) < 2*np.sign(new_step[6])*new_step[2]/np.pi) or (dTau < 0.001*np.mean(dTau_change) and all([anglething(all_states[i][2]) < anglething(all_states[i-1][2]) for i in np.arange(-10, 0)]))):  #("brook" in label and 0.5*np.pi*(1 - np.abs(state[2]%2 -1)) < 0.01) and :
                # if (high inclination) AND ((very close to pole AND approaching pole) OR (dTau is very small AND dTau is monotonically non-increasing))
            if np.sign(new_step[6])*(np.pi/2 - new_step[2]%np.pi) <= -1.55 and np.mean(dTau_change[-10:]) <= 0.001*np.mean(dTau_change):
                print("POW, ind=", i)
                print(new_step)
                new_step[0] += ((new_step[0] - state[0])/abs(new_step[2] - state[2]))*(2*anglething(new_step[2]))
                #new_step[1] += ((new_step[0] - state[0])/abs(new_step[2] - state[2]))*new_step[5]
                new_step[3] += 2*np.arccos(np.sin(abs(np.pi/2 - np.arccos(L/np.sqrt(L**2 + C))))/ np.sin(new_step[2]))
                new_step[6] = -new_step[6]
                print(new_step)
                break
            #print(anglething(new_step[2]), new_step[2])
    
            #print(err_calc, "ap")
            #print(dTau, "hah!")
            #print(np.linalg.norm(new_step - step_check), "thing")
            old_dTau, dTau = dTau, min(dTau * abs(err_target / (err_calc + (err_target/100)))**(0.2), 2*np.pi*(state[1]**(1.5))*0.04)
            if dTau <= 0.0:
                dTau = old_dTau
            first = False
      #print("check", np.sign(new_step[6])*(np.pi/2 - new_step[2]%np.pi), np.mean(dTau_change[-10:]), 0.001*np.mean(dTau_change))
      #print(new_step[0], anglething(new_step[2]), np.mean(dTau_change[-10:]),  0.001*np.mean(dTau_change))
      #print(old_dTau)
      #print("____________")
      #print(new_step[4:], old_dTau, i)
      #print("-")
      #print(new_step)
      if new_step[4] < 0.0:
          print("why??", i)
          thunk = True
          print(new_step)
      metric = mm.kerr(new_step, mass, a)[0]
      #newE = -np.matmul(new_step[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
      #newLz = np.matmul(new_step[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
      #newQ = np.matmul(np.matmul(mm.kill_tensor(new_step, mass, a), new_step[4:]), new_step[4:])    #new Carter constant Q
      #newC = newQ - (a*newE - newLz)**2   
      #'''
      test = mm.check_interval(mm.kerr, new_step, mass, a)
      if thunk == True:
          print(test)
      if abs(test+1)>(err_target):
          #print(test+1)
          #print(new_step)
          #print(new_step, "hey")
          borken = borken + 1
          #print(borken, i+1, borken/(i+1))
          #print("Error on PN Update")
          '''
          #print("Index:", i)
          #print(new_step)
          #print("Previous constants", constants[-1])
      #'''
          bad_int = True
          trial = 0
          og_new_step = np.copy(new_step)
          while bad_int == True:
            if "blee" in label:
                t, r, theta, phi = new_step[:4]
                sine, cosi = np.sin(theta), np.cos(theta)
                rs, sig = 2, r**2 + (a**2)*np.cos(theta)
                delta = r**2 + a**2 - r*rs
                err = mm.check_interval(mm.kerr, new_step, mass, a) + 1
                jumpscale = 0.5
                while abs(err) > 10**(-15) and trial <25:
                    #print(err)
                    old_err = err
                    ut, ur, uth, uphi = new_step[4:]
                    grad = np.array([-(2/sig)*((sig-r*rs)*ut + a*r*rs*uphi*(sine**2)),
                                    2*sig*ur/delta,
                                    2*sig*uth,
                                    (2*(sine**2)/sig)*((r**2)*sig*uphi - a*r*rs*ut + (a**2)*(r*    rs*uphi*(sine**2) - sig*uphi))])
                    thing = np.copy(new_step)
                    #for i in range(8):
                        #print(new_step[i] - thing[i])
                    new_step[4:] += -jumpscale*err*grad
                    err = mm.check_interval(mm.kerr, new_step, mass, a) + 1
                    #print(err)
                    #print("---")
                    trial += 1
                    if abs(err/old_err) > 1:
                        #print("big overshoot")
                        new_step = thing
                        trial -= 1
                        jumpscale *= 0.1
                    elif np.sign(err/old_err) == -1:
                        #print("small overshoot")
                        jumpscale *= 0.5
                    else:
                        jumpscale *= 1.5
                test = mm.check_interval(mm.kerr, new_step, mass, a)
                final_ut = new_step[4]
            else:
                ut, up = new_step[4], new_step[7]
                err = mm.check_interval(mm.kerr, new_step, mass, a)
                metric = mm.kerr(new_step, mass, a)[0]
                gtt, gtp = metric[0][0], metric[0][3]
                A = gtt
                B = (2*gtt*ut + 2*gtp*up)
                C = -1 - err
                int_steps = [np.copy(new_step), np.copy(new_step), np.copy(new_step), np.copy(new_step)]
                print(A)
                print(B)
                print(C)
                print(B**2 - 4*A*C)
                print("ya")
                for num1 in range(4): 
                    try:
                        int_steps[num1][4] += ((-1)**(np.arange(4)//2))*((-B + ((-1)**(np.arange(4)%2))*np.sqrt(B**2 - 4*A*C))/(2*A))
                    except:
                        print("skip", num1)
                        pass
                checks = [(mm.check_interval(mm.kerr, int_steps[0], mass, a)+1)**2,
                          (mm.check_interval(mm.kerr, int_steps[1], mass, a)+1)**2,
                          (mm.check_interval(mm.kerr, int_steps[2], mass, a)+1)**2,
                          (mm.check_interval(mm.kerr, int_steps[3], mass, a)+1)**2]
                test = mm.check_interval(mm.kerr, int_steps[checks.index(min(checks))], mass, a)
                trial += 1
                new_step = int_steps[checks.index(min(checks))]
                final_ut = int_steps[checks.index(min(checks))][4]
            if (abs(test+1)<=(err_target)) and (final_ut > 0):
              
              #print("fixed!---")
              #'''
              #if trial <=3:
              #    print("first!")
      #'''
              bad_int = False
            elif trial >= 10:
              #print("giving up!", trial, test+1, final_ut)
              if abs(mm.check_interval(mm.kerr, new_step, mass, a) + 1) >= abs(mm.check_interval(mm.kerr, og_new_step, mass, a) + 1) or new_step[4] < 0.0:
                  new_step = np.copy(og_new_step)
                  #print("fix!", mm.check_interval(mm.kerr, new_step, mass, a))
              break
      #print(new_step, "warp")
      if thunk == True:
          print(new_step)
      if new_step[4] < 0.0:
          print("you realize this isn't better??")
      #'''
      #constant modifying section
      #Whenever you pass from one side of r0 to the other, mess with the effective potential.
      if ( np.sign(new_step[1] - r0) != orbitside) or ((new_step[3] - all_states[tracker[-1][-1]][3] > np.pi*(3/2)) and (np.std([state[1] for state in all_states[tracker[-1][-1]:]]) < 0.01*np.mean([state[1] for state in all_states[tracker[-1][-1]:]]))):
      #if (np.sign(new_step[5]) != np.sign(all_states[-1][5])):
          if (i - tracker[-1][-1] > 2):
              #print("HEY")
              #print( np.sign(new_step[1] - r0) != orbitside)
              #print(new_step[3] - all_states[tracker[-1][-1]][3] > np.pi*(3/2))
              #print(np.std([state[1] for state in all_states[tracker[-1][-1]:]]) < 0.01*np.mean([state[1] for state in all_states[tracker[-1][-1]:]]))
              
              update = True
              if ( np.sign(new_step[1] - r0) != orbitside):
                  orbitside *= -1
              orbitCount += 0.5
              if mu != 0.0:
                  #print(i)
                  #print(new_step[4:])
                  condate = True
                  #print("__________")
                  dcons = mm.peters_integrate5(all_states, a, mu, tracker[-1][-1], i)
                  #print("secthingy", i)
                  new_step, ch_cons = mm.new_recalc_state6(constants[-1], dcons, new_step, mu, mass, a)
                  
                  pot_min = viable_cons(ch_cons, new_step, mass, a)
                  #print(new_step, "wap")
                  while pot_min[0] < -err_target:
                      print("tick?")
                      Lphi, Cart, ro = *ch_cons[1:], pot_min[1]
                      #print("please?", ch_cons[0], (-pot_min[0])*((2*ro*((ro**3 + ro*(a**2) + 2*(a**2))*ch_cons[0] - 2*Lphi*a))**(-1)))
                      #ch_cons[0] = (4*a*Lphi*r0 + ((4*a*Lphi*ro)**2 - 4*(ro**4 + 2*ro*(a**2))*((a*Lphi)**2 - (ro**2 - 2*ro + a**2)*(ro**2 + Lphi**2 + Cart)))**(0.5))/(2*(ro**4 + 2*ro*(a**2)))
                      ch_cons[0] += max(10**(-16), 2*(-pot_min[0])*((2*ro*((ro**3 + ro*(a**2) + 2*(a**2))*ch_cons[0] - 2*Lphi*a))**(-1)))
                      #ch_cons[0] += 2*(-pot_min[0])*((2*ro*((ro**3 + ro*(a**2) + 2*(a**2))*ch_cons[0] - 2*Lphi*a))**(-1))
                      #print("       ", ch_cons[0])
                      new_step = mm.recalc_state(ch_cons, new_step, mass, a)
                      pot_min = viable_cons(ch_cons, new_step, mass, a)
                  #print(new_step[4:])
                  #print("_____")
              
                
      #Initializing for the next step
      #Updates the constants based on the calculated derivatives, then updates the state velocities based on the new constants.
      #Only happens the step before the derivatives are recalculated.
      
      #Update stuff!
      #print(update)
      if (update == True):
         #print("Constant Change!")
         if condate == False:
             metric = mm.kerr(new_step, mass, a)[0]
             newE = -np.matmul(new_step[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
             newLz = np.matmul(new_step[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
             newQ = np.matmul(np.matmul(mm.kill_tensor(new_step, mass, a), new_step[4:]), new_step[4:])    #new Carter constant Q
             newC = newQ - (a*newE - newLz)**2                                                             #initial adjusted Carter constant  
             #print(newE, newLz, newC, "FROM MAIN")
             coeff = np.array([newE**2 - 1, 2*mass, (a**2)*(newE**2 - 1) - newLz**2 - newC, 2*mass*((a*newE - newLz)**2 + newC), -newC*(a**2)])
             coeff2 = np.array([4*(newE**2 - 1), 3*2*mass, 2*((a**2)*(newE**2 - 1) - newLz**2 - newC), 2*mass*((a*newE - newLz)**2 + newC)])
             r0, inner_turn, outer_turn = max(np.roots(coeff2)), *np.sort(np.roots(coeff))[-2:]
             e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
             tracker.append([r0, e, inner_turn, outer_turn, new_step[0], i])
             #print(np.sum(coeff*np.array([r0**4, r0**3, r0**2, r0, 1.0])), "thing")
             constants.append([newE, newLz, newC])
             qarter.append(newQ)
             #print("Are we in here somehow?")
         else:
            constants.append(ch_cons)
            qarter.append(ch_cons[2] + (a*ch_cons[0] - ch_cons[1])**2)
            coeff = np.array([ch_cons[0]**2 - 1, 2*mass, (a**2)*(ch_cons[0]**2 - 1) - ch_cons[1]**2 - ch_cons[2], 2*mass*((a*ch_cons[0] - ch_cons[1])**2 + ch_cons[2]), -ch_cons[2]*(a**2)])
            coeff2 = np.array([4*(ch_cons[0]**2 - 1), 3*2*mass, 2*((a**2)*(ch_cons[0]**2 - 1) - ch_cons[1]**2 - ch_cons[2]), 2*mass*((a*ch_cons[0] - ch_cons[1])**2 + ch_cons[2])])
            #print("hewwo??", *np.real(np.sort(np.roots(coeff))[-2:]))
            r0, inner_turn, outer_turn = max(np.roots(coeff2)), *np.sort(np.roots(coeff))[-2:]
            inner_turn, outer_turn = np.real(inner_turn), np.real(outer_turn)
            e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
            tracker.append([r0, e, inner_turn, outer_turn, new_step[0], i])

         if True in np.iscomplex(tracker[-1]):
             compErr += 1
             print("skring")
             print(tracker[-1])
             #tracker[-1] = np.real(tracker[-1])
             issues.append((i, new_step[0]))  
      interval.append(mm.check_interval(mm.kerr, new_step, mass, a))
      false_constants.append([getEnergy(new_step, mass, a), *getLs(new_step, mu)])
      dTau_change.append(old_dTau)
      all_states.append(new_step )    #update position and velocity
      i += 1

      progress = max(1 - abs(eval(terms[2]) - eval(termdict[terms[0]]))/eval(terms[2]), i/(10**7) ) * 100
      if verbose == True:
          if (progress >= milestone):
            print("Program has completed " + str(round(eval(termdict[terms[0]]), 2)), ",", str(round(progress, 4)) + "% of maximum run: Index = " + str(i))
            milestone = int(progress) + 1

    #Lets you end the program before the established end without breaking anything
    #except KeyboardInterrupt:
  #'''
    except:
      print("Ending program")
      stop = True
      cap = len(all_states) - 1
      all_states = all_states[:cap]
      interval = interval[:cap]
      dTau_change = dTau_change[:cap]
      constants = constants[:cap]
      qarter = qarter[:cap]
      break
  #'''
  print(len(issues), len(all_states))
  #unit conversion stuff
  if units == "mks":
      G, c = 6.67*(10**-11), 3*(10**8)
  elif units == "cgs":
      G, c = 6.67*(10**-8),  3*(10**10)
  else:
      G, M, c = 1.0, 1.0, 1.0
  if mu == 0.0:
      truemu = 0.0
      mu = 1.0
      
  constants = np.array([entry*np.array([M*mu*(c**2), M*M*mu*G/c, (M*M*mu*G/c)**2]) for entry in np.array(constants)], dtype=np.float64)
  false_constants = np.array(false_constants)
  qarter = np.array(qarter)
  interval = np.array(interval)
  dTau_change = np.array([entry * (G*M)/(c**3) for entry in dTau_change])
  all_states = np.array([entry*np.array([(G*M)/(c**3), (G*M)/(c**2), 1.0, 1.0, 1.0, c, (c**3)/(G*M), (c**3)/(G*M)]) for entry in np.array(all_states)]) 
  tracker = np.array([entry*np.array([(G*M)/(c**2), 1.0, (G*M)/(c**2), (G*M)/(c**2), (G*M)/(c**3), 1]) for entry in tracker])
  print(np.array([(G*M)/(c**3), (G*M)/(c**2), 1.0, 1.0, 1.0, c, (c**3)/(G*M), (c**3)/(G*M)]))
  r = all_states[0][1]
  ind = argrelmin(all_states[:,1])[0]
  omega, otime = np.diff(all_states[:,2][ind]) - 2*np.pi, np.diff(all_states[:,0][ind])
  
  if verbose == True:
      print("There were " + str(compErr) + " issues with complex roots/turning points.")
      #for i in issues:
       # print("Complex value at index " + str(i[0]) + ", t = " + str(i[1]))
      #print("Here's some data!")
      #print(checker)
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
                              ((r**(3/2) + pro*a)**(-1))]) * (c**3)/(G*M),
           "r0": tracker[:,0],
           "e": tracker[:,1],
           "it": tracker[:,2],
           "ot": tracker[:,3],
           "tracktime": tracker[:,4],
           "trackix": tracker[:,5],
           "omegadot": omega/otime,
           "otime": all_states[:,0][ind][1:],
           "stop": stop,
           "issues": issues}
  return final

def wrapsnap(data):
    be, me = np.polynomial.polynomial.polyfit(list(data["tracktime"]), data["energy"], 1)
    bl, ml = np.polynomial.polynomial.polyfit(list(data["tracktime"]), data["phi_momentum"], 1)
    bc, mc = np.polynomial.polynomial.polyfit(list(data["tracktime"]), data["carter"], 1)
    #print("E slope:", me)
    #print("L slope:", ml)
    #print("C slope:", mc)
    return me, ml, mc

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
                test = clean_inspiral(1.0, spin, mu, t*4, 0.1, 10**(-13), "test", params = [rm, e0, inc*np.pi/2], verbose=False)
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

def dicttotxt(dic, filename):
    current = txttodict(filename)
    current.update(dic)
    with open(filename, 'w') as f: 
        for key, value in current.items(): 
            f.write('%s:%s\n' % (key, value))
    return 0

def txttodict(filename):
    newdict = {}
    try:
        file1 = open(filename, 'r')
        Lines = file1.readlines()
        for line in Lines:
            key, value = line.strip().split(":")
            newdict[literal_eval(key)] = literal_eval(value)
    except:
        pass
    return newdict

def txttodict2(filename):
    newdict = {}
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    for line in Lines:
        key, value = line.strip().split(":")
        newdict[literal_eval(key)] = literal_eval(value)
    return newdict

def dict_saver(data, filename):
    np.save(filename, data) 
    return True
        
def dict_from_file(filename):
    if ".npy" not in filename:
        filename = filename+".npy"
    data = np.load(filename, allow_pickle='TRUE').item()
    #print(read_dictionary['hello']) # displays "world"
    return data

def fillholes(filename):
    rerun = False
    dctkeys = list(txttodict(filename).keys())
    params = [[], [], [], [], []]
    for key in dctkeys:
        for i in range(5):
            if key[i] not in params[i]:
                params[i].append(key[i])
    print(params)
    biglist = [(i, j, k, l, m) for i, j, k, l, m in product(params[0], params[1], params[2], params[3], params[4]) if (i, j, k, l, m) not in dctkeys]
    print(len(biglist))
    while len(biglist) > 300:
        rerun = True
        print("Shrinking set to cap runtime")
        ind = random.randint(0,4)
        while len(params[ind]) <= 1:
            print(params[ind])
            print("Too short!")
            ind = random.randint(0,4)
        params[ind].pop(random.randint(0, len(params[ind])-1))
        print(params)
        biglist = [(i, j, k, l, m) for i, j, k, l, m in product(params[0], params[1], params[2], params[3], params[4]) if (i, j, k, l, m) not in dctkeys]
        print(len(biglist))
    condt, skip = wrapwrap(*params, skip=dctkeys)
    if rerun == True:
        print("Rerun to fill skipped areas.")
    return condt, skip

def fillholes2(filename, numadd):
    dctkeys = list(txttodict(filename).keys())
    params = [[], [], [], [], []]
    for key in dctkeys:
        for i in range(5):
            if key[i] not in params[i]:
                params[i].append(key[i])
    print(params)
    biglist = [(i, j, k, l, m) for i, j, k, l, m in product(params[0], params[1], params[2], params[3], params[4]) if (i, j, k, l, m) not in dctkeys]
    smallist = biglist[:numadd]
    condt = {}
    for index in smallist:
        rm, e0, inc, spin, mu = index
        t = np.sqrt(4*(np.pi**2)*(rm**3))
        try:
            test = clean_inspiral(1.0, spin, mu, t*4, 0.1, 10**(-13), "test", params = [rm, e0, inc*np.pi/2], verbose=False)
            if test["time"][-1] < t*4:
                if test["stop"] == True:
                    smallist.append(False)
                    break
                else:
                    print("Something's wonky.")
            else:
                condt[rm, e0, inc, spin, mu] = wrapsnap(test)
        except:
            print("Didn't work?")
        print("---")
    if len(biglist) > 300:
        print("Rerun to fill skipped areas.")
    return condt, smallist

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
'''
e_val, f_val, g_val, orb = np.zeros((7, 50)), np.zeros((7,50)), np.zeros((7,50)), np.zeros((7, 50))
for ene in enlist:
    small, big = l_locator(ene, 0.0, 10)
    xval = np.linspace(small, big)
    for l in xval:
        t = np.sqrt(4*(np.pi**2)*((1/(1-ene**2))**3))
        test = clean_inspiral(1.0, 0.0, 0.0, t*10, 0.1, 10**(-12), "test", cons=[ene, l, 0.0], verbose=False)
        e, f, g = eandf(test)
        o = (max(test["pos"][:,2])/(2*np.pi))/t
        e_val[enlist.index(ene), np.where(xval == l)[0][0]] = e
        f_val[enlist.index(ene), np.where(xval == l)[0][0]] = f
        g_val[enlist.index(ene), np.where(xval == l)[0][0]] = g
        orb[enlist.index(ene), np.where(xval == l)[0][0]] = o
'''