# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:14:26 2022

@author: hepiz
"""

import numpy as np
import MetricMath as mm




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
          print(constants)
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
        if abs(test+1)<=(err_target):
          new_step = int_steps[checks.index(min(checks))]
          loop2 = False
        if trial >= 10:
          print("giving up!", trial, test)  
              
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
      phys_avgs = phys_avgs[:cap]
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
