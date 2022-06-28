# -*- coding: utf-8 -*-
"""
Metric Math stuff
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

g = 1
#whenever present, mass is usually set equal to 1

def find_rmb(spin):
    if spin >= 0.0:
        pro = 1.0
    else:
        pro = -1.0
    def pro_min(x, a):
        if a >= 0.0:
            ang = (a - np.sqrt(a**2 - (a**2 + x**2)*(1 - (x/2))))/(1- (x/2))
        else:
            a = abs(a)
            ang = -(a + np.sqrt(a**2 - (a**2 + x**2)*(1 - (x/2))))/(1- (x/2))
        return ang
    
    data = optimize.minimize(pro_min, 4, args=(spin), bounds=( (1+np.sqrt(1-spin**2), 10), ))
    l_mb, r_mb = pro*data['fun'][0], data['x'][0]
    return l_mb, r_mb

def fix_sin(value):
    '''
    fix_sin function smooths values from numpy.sin,
    setting values below a certain threshold to zero.

    Parameters
    ----------
    value : int/float/etc.

    Returns
    -------
    num : float

    '''
    num = np.sin(value)
    if abs(num) <= 10**(-15):
        num = 0.0
    return num

def fix_cos(value):
    '''
    fix_cos function smooths values from numpy.cos,
    setting values below a certain threshold to zero.

    Parameters
    ----------
    value : int/float/etc.

    Returns
    -------
    num : float

    '''
    num = np.cos(value)
    if abs(num) <= 10**(-15):
        num = 0.0
    return num

def mink(state, mass):
    '''
    mink function generates metric and christoffel symbols
    for the minkowski metric. Mostly for testing purposes

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    mass : int/float
        mass of black hole in arbitrary units. Generally set to 1

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
    chris = {}
    return (metric, chris)

# schwarz function computes the metric and christoffel connection terms for a given state
# specific to schwarzchild orbit and may be phased out once kerr is complete
def schwarz(state, mass):
    '''
    schwarz function generates metric and christoffel symbols
    for the schwarzschild metric.

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    mass : int/float
        mass of black hole in arbitrary units. Generally set to 1

    Returns
    -------
    metric : 4x4 list
        Spacetime metric in terms of coordinate directions
    chris : dictionary {string: float}
        List of connection terms between coordinate directions

    '''
    r, theta = state[1], state[2]
    metric = [[-(1-((2*g*mass)/r)), 0,                        0,    0],
              [0,                   (1-((2*g*mass)/r))**(-1), 0,    0],
              [0,                   0,                        r**2, 0],
              [0,                   0,                        0,    (r**2) * (fix_sin(theta))**2]]
    chris = {"001": (g * mass) / (r * (r - 2 * g * mass)),
             "010": (g * mass) / (r * (r - 2 * g * mass)),
             "100": g * (mass / r**3) * (r - 2 * g * mass),
             "111": -(g * mass) / (r * (r - 2 * g * mass)),
             "122": -(r - 2 * g * mass),
             "133": -(r - 2 * g * mass) * fix_sin(theta)**2,
             "212": 1 / r,
             "221": 1 / r,
             "233": -fix_sin(theta) * fix_cos(theta),
             "313": 1 / r,
             "323": fix_cos(theta) / fix_sin(theta),
             "331": 1 / r,
             "332": fix_cos(theta) / fix_sin(theta)}
    return (metric, chris)

# set_u function normalizes a given initial state to maintain a proper
# spacetime interval: -1 for timelike paths, 0 for null
# converts from a locally defined tetrad frame to more generally defined black hole frame
# can be made to default to a circular orbit

# specific to schwarzchild orbit and may be phased out once kerr is complete
#timelike - whether the orbit is for a massive particle or not (True or False)
#circular - makes initialization default to circular orbit
#eta - angle between launch and a line parallel to black hole axis
#xi - angle between launch and a line drawn between black hole and launch point
def set_u(state, mass, timelike, circular, eta, xi):
    '''
    set_u function normalizes an input state vector according to schwarzschild
    metric

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    mass : int/float
        mass of black hole in arbitrary units. Generally set to 1
    timelike : boolean
        specifies type of spacetime trajectory. Generally set to True
    circular : boolean
        can be toggled to normalize state vector to a circular orbit with
        radius determined by 4-position
    eta : float
        angular displacement between desired trajectory and positive r direction,
        measured in radians
    xi : float
        angular displacement between desired trajectory and positive phi direction,
        measured in radians

    Returns
    -------
    new : 8 element numpy array
        4-position and 4-velocity of the test particle at a particular moment

    '''
    r, theta = state[1], state[2]
    metric, chris = schwarz(state, mass)
    value = (-1 if timelike == True else 0)

    new = np.copy(state)
    tilde = np.copy(state)
    tetrad_matrix = [[(1-((2*g*mass)/r))**(-1/2), 0,                         0,   0],
                     [0,                          (1-((2*g*mass)/r))**(1/2), 0,   0],
                     [0,                          0,                         1/r, 0],
                     [0,                          0,                         0,   1/(r * fix_sin(theta))]]
    if circular == True:
        #calculates trajectory directly from known L for equicircular orbit
        u_t = (1 - (2 + fix_sin(theta)**2)/r)**(-1/2)
        ang_vel = u_t * r**(-3/2)
        gamma = u_t * (1 - (2/r))**(1/2)
        beta = ang_vel * r / gamma
    else:
        u_t = ((-1/metric[0][0])*(metric[1][1]*state[5]**2 + metric[2][2]*state[6]**2 + metric[3][3]*state[7]**2 - value))**(1/2)
        gamma = u_t * (1 - (2/r))**(1/2)
        beta = (1 - gamma**(-2))**(1/2)
    tilde[4:] = gamma, gamma*beta*fix_sin(eta)*fix_cos(xi), -gamma*beta*fix_cos(eta), gamma*beta*fix_sin(eta)*fix_sin(xi)
    final = np.matmul(tetrad_matrix, tilde[4:])
    new[4:] = final
    return new

def kerr(state, mass, a):
    '''
    kerr function generates metric and christoffel symbols
    for the kerr metric. should be identical to schwarz for a=0

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    mass : int/float
        mass of black hole in arbitrary units. Generally set to 1
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    metric : 4x4 list
        Spacetime metric in terms of coordinate directions
    chris : dictionary {string: float}
        List of connection terms between coordinate directions

    '''
    r, theta, s = state[1], state[2], 2*mass
    sine, cosi = fix_sin(theta), fix_cos(theta)
    #various defined values that make math easier
    rho2, tri = r**2 + (a*cosi)**2, r**2 - s*r + a**2
    al2, w = (rho2*tri)/(rho2*tri + 2*mass*r*(a**2 + r**2)), (2*mass*r*a)/(rho2*tri + 2*mass*r*(a**2 + r**2))
    wu2 = ((rho2*tri + 2*mass*r*(a**2 + r**2))/(rho2))*sine**2
    bigA = (r**2 + a**2)**2 - tri*(a*sine)**2
    metric = [[-al2 + wu2*(w**2), 0,             0,    -w*wu2 ],
              [0,                 rho2/tri,      0,    0      ],
              [0,                 0,             rho2, 0      ],
              [-w*wu2,            0,             0,    wu2    ]]
    chris = {"001": s*(r**2 + a**2)*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2)),
             "002": -s*(a**2)*r*sine*cosi/(rho2**2),
             "010": s*(r**2 + a**2)*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2)), #=001
             "013": s*a*(sine**2)*(((a*cosi)**2)*(a**2 - r**2) - (r**2)*(a**2 + 3*(r**2)))/(2*(rho2**2)*tri),
             "020": -s*(a**2)*r*sine*cosi/(rho2**2), #=002
             "023": s*r*cosi*((a*sine)**3)/(rho2**2),
             "031": s*a*(sine**2)*(((a*cosi)**2)*(a**2 - r**2) - (r**2)*(a**2 + 3*(r**2)))/(2*(rho2**2)*tri), #=013
             "032": s*r*cosi*((a*sine)**3)/(rho2**2), #=023
             "100": s*tri*(r**2 - (a*cosi)**2)/(2*(rho2**3)),
             "103": -tri*s*a*(sine**2)*(r**2 - (a*cosi)**2)/(2*(rho2**3)),
             "111": (2*r*((a*sine)**2) - s*(r**2 - (a*cosi)**2))/(2*rho2*tri),
             "112": -(a**2)*sine*cosi/rho2,
             "121": -(a**2)*sine*cosi/rho2, #=112
             "122": -r*tri/rho2,
             "130": -tri*s*a*(sine**2)*(r**2 - (a*cosi)**2)/(2*(rho2**3)), #=103
             "133": (tri*(sine**2)/(2*(rho2**3)))*(-2*r*(rho2**2) + s*((a*sine)**2)*(r**2 - (a*cosi)**2)),
             "200": -s*(a**2)*r*sine*cosi/(rho2**3),
             "203": s*a*r*(r**2 + a**2)*sine*cosi/(rho2**3),
             "211": (a**2)*sine*cosi/(rho2*tri),
             "212": r/rho2,
             "221": r/rho2, #=212
             "222": -(a**2)*sine*cosi/rho2,
             "230": s*a*r*(r**2 + a**2)*sine*cosi/(rho2**3), #=203
             "233": -(sine*cosi/(rho2**3))*(bigA*rho2 + (r**2 + a**2)*s*r*((a*sine)**2)),
             "301": s*a*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2)),
             "302": -s*a*r*(cosi/sine)/(rho2**2),
             "310": s*a*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2)), #=301
             "313": (2*r*(rho2**2) + s*(((a**2)*sine*cosi)**2 - (r**2)*(rho2 + r**2 + a**2)))/(2*tri*(rho2**2)),
             "320": -s*a*r*(cosi/sine)/(rho2**2), #=302
             "323": ((cosi/sine)/(rho2**2))*((rho2**2) + s*r*((a*sine)**2)),
             "331": (2*r*(rho2**2) + s*(((a**2)*sine*cosi)**2 - (r**2)*(rho2 + r**2 + a**2)))/(2*tri*(rho2**2)), #=313
             "332": ((cosi/sine)/(rho2**2))*((rho2**2) + s*r*((a*sine)**2))} #=323
    return (metric, chris)

#check_interval function returns the spacetime interval of a given state
#should be -1 for timelike orbits (or 0 for null), otherwise something is wrong
def check_interval(solution, state, *args):
    '''
    check_interval function returns the spacetime interval for a particular
    state vector given a particular spacetime soultion. 
    All timelike intervals should = -1

    Parameters
    ----------
    solution : function
        One of the solution functions mink, schwarz, or kerr
    state : 8 element list/numpy array
        DESCRIPTION.
    *args : int/float
        args required for different solutions, depends on specific function.
        generally mass and possibly spin

    Returns
    -------
    interval : float
        spacetime interval

    '''
    metric, chris = solution(state, *args)
    interval = 0
    for u in range(4):
        for v in range(4):
            interval += metric[u][v] * state[4+u] * state[4+v]
    return interval

def set_u_kerr(state, mass, a, timelike, eta, xi, special=False):
    '''
    set_u function normalizes an input state vector according to kerr metric

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
            OR
            7 element list/numpy array
        4-position, Energy, Phi-momentum, and Carter constant (C) of the test particle at a particular moment
    mass : int/float
        mass of black hole in arbitrary units. Generally set to 1
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    timelike : boolean
        specifies type of spacetime trajectory. Generally set to True
    eta : float
        angular displacement between desired trajectory and positive r direction,
        measured in radians
    xi : float
        angular displacement between desired trajectory and positive phi direction,
        measured in radians
    special : boolean/string
        if false, run normalization with data as given
        if "circle", modify parameters to create a circular orbit
            at the given radius
        if "zoom", modify parameters to create a 'zoom-whirl' orbit
            (in progress)

    Returns
    -------
    new : 8 element numpy array
        4-position and 4-velocity of the test particle at a particular moment

    '''

    if len(state) == 7 and special == False:
        print("Constants given, defaulting to Sago calculation.")
        cons = state[4:]
        new = recalc_state(cons, state, mass, a)
        return new
    

    r, theta = state[1], state[2]
    metric, chris = kerr(state, mass, a)
    #various defined values that make math easier
    rho2 = r**2 + (a**2)*(fix_cos(theta)**2)
    tri = r**2 - 2*mass*r + a**2
    al2 = (rho2*tri)/(rho2*tri + 2*mass*r*((a**2) + (r**2)))
    w = (2*mass*r*a)/(rho2*tri + 2*mass*r*((a**2) + (r**2)))
    wu2 = ((rho2*tri + 2*mass*r*((a**2) + (r**2)))/rho2)*(fix_sin(theta)**2)
    plusmin = np.sign(fix_sin(eta))
    new = np.copy(state)
    tetrad_matrix = np.array([[1/(np.sqrt(al2)), 0,                 0,               0],
                              [0,                np.sqrt(tri/rho2), 0,               0],
                              [0,                0,                 1/np.sqrt(rho2), 0],
                              [w/np.sqrt(al2),   0,                 0,               1/np.sqrt(wu2)]])

    if special == "circle":
        #calculates trajectory directly from known E and L for equicircular orbit
        ene = (r**2 - 2*mass*r + plusmin*a*np.sqrt(mass*r))/(r*np.sqrt(r**2 - 3*mass*r + plusmin*2*a*np.sqrt(mass*r)))
        lel = plusmin*np.sqrt(mass*r)*(r**2 - plusmin*2*a*np.sqrt(mass*r) + a**2)/(r*np.sqrt(r**2 - 3*mass*r + plusmin*2*a*np.sqrt(mass*r)))
        u_down = np.array([-ene, 0, 0, lel])
        final = np.matmul(np.linalg.inv(metric), u_down)
       
    elif special == "zoom":
        '''
        photon sphere stuff
        radius of photon sphere for kerr
        r = r_s*(1 + cos((2/3)*arccos(+/|a|/M)))
        where r_s is schwarz radius, +- is for prograde vs retrograde
        stick with prograde, remember M=1, r_s=1 and a >=0 , left with
        r = 1 + cos((2/3)*arccos(a))

        calculate E and L for a circular orbit at this radius, then 
        plug that into u_down thing for metric at totally different radius

        gives reliable zoom-whirls???
        
        E+L for IBCO not the same as zoomwhirl by a longshot, would actually be less?? more??
        ask Jeremy
        '''
        r_p = (r + 3)/2.0
        ene = ((r_p)**2 - 2*mass*(r_p) + plusmin*a*np.sqrt(mass*(r_p)) )/((r_p)*np.sqrt((r_p)**2 - 3*mass*(r_p) + plusmin*2*a*np.sqrt(mass*(r_p))))
        lel = plusmin*np.sqrt(mass*(r_p))*((r_p)**2 - plusmin*2*a*np.sqrt(mass*(r_p)) + a**2)/((r_p)*np.sqrt((r_p)**2 - 3*mass*(r_p) + plusmin*2*a*np.sqrt(mass*(r_p))))
        lel = np.sqrt(2*mass*3*(1 + ene*3))
        u_down = np.array([-ene, 0, 0, lel])
        print(u_down)
        final = np.matmul(np.linalg.inv(tetrad_matrix), u_down)
    else:
        rdot, thetadot, phidot = state[5]/state[4], state[6]/state[4], state[7]/state[4]
        vel_2 = (rdot**2 + (r * thetadot)**2 + (r * fix_sin(theta) * phidot)**2)
        beta = np.sqrt(vel_2)
        if (beta >= 1):
            print("Velocity greater than or equal to speed of light. Setting beta to 0.99")
            beta = 0.99
        gamma = 1/np.sqrt(1 - beta**2)
        tilde = np.array([gamma, gamma*beta*fix_cos(eta), -gamma*beta*fix_sin(eta)*fix_cos(xi), gamma*beta*fix_sin(eta)*fix_sin(xi)])
        test = np.copy(state)
        test[4:] = tilde
        final = np.matmul(tetrad_matrix, tilde)
    new[4:] = final
    return new

# kill_tensor function defines the Kerr killing tensor for a given state and spin parameter
def kill_tensor(state, mass, a):
    '''
    kill_tensor function calculates killing Kerr killing tensor for a given system

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    mass : int/float
        mass of black hole in arbitrary units. Generally set to 1
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    ktens : 4 element numpy array
        Describes symmetries in spacetime

    '''
    r, theta = state[1], state[2]
    metric, chris = kerr(state, mass, a)
    rho2, tri = r**2 + (a*fix_cos(theta))**2, r**2 - 2*mass*r + a**2
    l_up = np.array([(r**2 + a**2)/tri, 1.0, 0.0, a/tri])
    n_up = np.array([(r**2 + a**2)/(2*rho2), -tri/(2*rho2), 0.0, a/(2*rho2)])
    l = np.matmul(metric, l_up)
    n = np.matmul(metric, n_up)
    l_n = np.outer(l, n)
    ktens = 2 * rho2 * 0.5*(l_n + np.transpose(l_n)) + (r**2) * np.array(metric)
    return ktens

# gr_diff_eq function calculates the time differential of a given state
# normal state is [position, velocity], this outputs [velocity, acceleration]
# very important, do not break
def gr_diff_eq(solution, state, *args):
    '''
    gr_diff_eq function calculates the instantaneous proper time derivative for
    a given state in a given system

    Parameters
    ----------
    solution : function
        One of the solution functions mink, schwarz, or kerr
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    *args : int/float
        args required for different solutions, depends on specific function.
        generally mass and possibly spin.

    Returns
    -------
    d_state: 8 element numpy array
        4-velocity and 4-acceleration for the test particle at a particular moment

    '''

    d_state = np.array([0,0,0,0,0,0,0,0], dtype=float)                                         #create empty array to be the derivative of the state
    d_state[0:4] = state[4:]                                                      #derivative of position is velocity
    metric, chris = solution(state, *args)
    for i in range(4):
        u = 0                                                                       #Last four entries are velocities
        for j in range(4):
            for k in range(4):                                                        #loop through indices to retrieve each Christoffel symbol
                index = str(i) + str(j) + str(k)
                if index in chris.keys():
                    u -= chris[index] * state[j + 4] * state[k + 4]
        d_state[4+i] = u                                                            #assign derivatives of velocity
    return d_state                                                                #return derivative of state

rk4 = {"label": "Standard RK4",
       "nodes": [1/2, 1/2, 1],
       "weights": [1/6, 1/3, 1/3, 1/6],
       "coeff": [[1/2], 
                 [0, 1/2],
                 [0, 0, 1]]}                                                    #Butcher table for standard RK4 method

ck5 = {"label": "Cash-Karp 5th Order",
       "nodes": [1/5, 3/10, 3/5, 1, 7/8],
       "weights": [37/378, 0, 250/621, 125/594, 0, 512/1771],
       "coeff": [[1/5],
                 [3/40, 9/40], 
                 [3/10, -9/10, 6/5], 
                 [-11/54, 5/2, -70/27, 35/27],
                 [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]]}     #Butcher table for 5th order Cash-Karp method

ck4 = {"label": "Cash-Karp 4th Order",
       "nodes": [1/5, 3/10, 3/5, 1, 7/8],
       "weights": [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4],
       "coeff": [[1/5],
                 [3/40, 9/40], 
                 [3/10, -9/10, 6/5], 
                 [-11/54, 5/2, -70/27, 35/27],
                 [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]]}     #Butcher table for 4th order Cash-Karp method

#gen_RK function applies a given Runge-Kutta method to calculate whatever the new state
#of an orbit will be after some amount of proper time
#butcher - one of the runge-kutta dictionaries
#solution - either schwarz or kerr function defined above
#state - state of the system
#dTau - proper time between current state and state-to-be-calculated
#args - required info for solution
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
            #print(param)
        k.append(gr_diff_eq(solution, param, *args))                          #which is then used to find the next k value
        #print(k)
    new_state = np.copy(state)
    for val in range(len(k)):                                                     #another for loop to add all the weights and find the final state
        new_state += k[val] * butcher["weights"][val] * dTau                        #can probably be simplified but this works for now
    return new_state



def constant_recalc(test, mass, a):
    r, theta = test[1], test[2]
    nsin, ncos = fix_sin(theta), fix_cos(theta)
    rho2, tri = r**2 + (a*ncos)**2, r**2 - 2*mass*r + a**2
    kill = kill_tensor(test, mass, a)
    metric = kerr(test, mass, a)[0]
    #print("get info")

    ene = -np.matmul(test[4:], np.matmul(metric, [1, 0, 0, 0]))
    lel = np.matmul(test[4:], np.matmul(metric, [0, 0, 0, 1]))
    qrt = np.matmul(np.matmul(kill, test[4:]), test[4:])
    cqrt = qrt - (a*ene - lel)**2 
    #print(ene, qrt)
    #print("get constants")
    
    p_r = ene*(r**2 + a**2) - a*lel
    r_r = (p_r)**2 - tri*(r**2 + (a*ene - lel)**2 + cqrt)
    #print(r_r, "rr")
    #print( (ene**2 - 1)*(r**4) + (2*mass - qrt)*(r**2) + 2*mass*r*qrt, "rr guess")
    the_the = cqrt - (cqrt + (a**2)*(1 - ene**2) + lel**2)*(ncos**2) + (a**2)*(1 - ene**2)*(ncos**4)
    #print(the_the, "thethe")
    #print("state precursors")
    
    tlam = (-a*(a*ene*(nsin**2) - lel) + (p_r)*(r**2 + a**2)/tri) * (1/rho2)
    rlam = np.sqrt(abs(r_r)) * (1/rho2)
    cothelam = np.sqrt(abs(the_the)) * (-1/(rho2*nsin))
    philam = (-(a*ene - lel/(nsin**2)) + (p_r)*a/tri) * (1/rho2)
    #print("state stuff")
    
    #sign correction
    rlam = rlam * np.sign(test[5]) * np.sign(rlam)
    cothelam = cothelam * np.sign(test[6]) * np.sign(cothelam)
    #print("sign correction")
    
    new_state = np.copy(test)
    new_state[4:] = np.array([tlam, rlam, cothelam, philam])
    return new_state

def find_constants(test, mass, a):
  r, theta = test[1], test[2]
  rho2, tri = r**2 + (a*fix_cos(theta))**2, r**2 - 2*mass*r + a**2
  kill = kill_tensor(test, mass, a)
  metric = kerr(test, mass, a)[0]

  ene = -np.matmul(test[4:], np.matmul(metric, [1, 0, 0, 0]))
  lel = np.matmul(test[4:], np.matmul(metric, [0, 0, 0, 1]))
  qrt = np.matmul(np.matmul(kill, test[4:]), test[4:])
  cqrt = qrt - (a*ene - lel)**2 
  return np.array([ene, lel, cqrt])

def rrfunc(x, ene, lel, cqrt, mass, a):
  at = ene**2 - 1
  bt = 2*mass
  ct = (a**2)*(ene**2 - 1) - lel**2 - cqrt
  dt = 2*mass*((a*ene - lel)**2) + 2*mass*cqrt
  et = 2*((a*ene - lel)**2)*(a**2) + cqrt*(a**2)
  return at*(x**4) + bt*(x**3) + ct*(x**2) + dt*x + et

def nrrfunc(x, ene, lel, cqrt, mass, a):
  at = ene**2 - 1
  bt = 2*mass
  ct = (a**2)*(ene**2 - 1) - lel**2 - cqrt
  dt = 2*mass*((a*ene - lel)**2) + 2*mass*cqrt
  et = 2*((a*ene - lel)**2)*(a**2) + cqrt*(a**2)
  return -(at*(x**4) + bt*(x**3) + ct*(x**2) + dt*x + et)

def nrrfunc2(x, ene, lel, cqrt, mass, a):
  p = ene*((x**2) + (a**2)) - a*lel
  tri = (x**2) - 2*mass*x + (a**2)
  return (p**2) - tri*((x**2) + ((a*ene - lel)**2) + cqrt)

def thetafunc(theta, ene, lel, cqrt, mass, a):
  x = np.cos(theta)
  return -(cqrt - (cqrt + (a**2)*(1-(ene**2)) + (lel**2))*(x**2) + (a**2)*(1-(ene**2))*(x**4))

def nrrfuncdet(x, ene, lel, cqrt, mass, a):
  at = ene**2 - 1
  bt = 2*mass
  ct = (a**2)*(ene**2 - 1) - lel**2 - cqrt
  dt = 2*mass*((a*ene - lel)**2) + 2*mass*cqrt
  et = 2*((a*ene - lel)**2)*(a**2) + cqrt*(a**2)
  return [-at, -bt, -ct, -dt, -et]

def find_ro(ene, lel, cqrt, mass, a):
  at = ene**2 - 1
  bt = 2*mass
  ct = (a**2)*(ene**2 - 1) - lel**2 - cqrt
  dt = 2*mass*((a*ene - lel)**2) + 2*mass*cqrt
  slope = [-4*at, -3*bt, -2*ct, -dt]
  zeros = np.roots(slope)
  curve = [-(12*at*(x**2) + 6*bt*x + 2*ct) for x in zeros]
  #print(zeros)
  #print(curve)
  r0 = 0
  for i in range(len(zeros)):
    if (curve[i] > 0) and (zeros[i] > 1 + np.sqrt(1-a**2)):
      r0 = zeros[i]
  if r0 == 0:
    r0 = max(zeros)
  return r0

def constant_derivatives(constants, mass, a, mu, rad):
  energy, lmom, cart = constants[0], constants[1], constants[2] 
  coeff = [energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lmom**2 - cart, 2*mass*((a*energy - lmom)**2) + 2*mass*cart, 2*((a*energy - lmom)**2)*(a**2) + cart*(a**2)]
  roots = np.sort(np.roots(coeff))
  r0 = find_ro(energy, lmom, cart, mass, a)
  if (True in np.iscomplex([r0, *roots])):
    r0 = r0.real
    roots = roots.real
  outer_turn, inner_turn = roots[-1], roots[-2]
  y, v, q, e = cart/(lmom**2), np.sqrt(mass/r0), a/mass, (outer_turn/r0) - 1
  dedt = -(32/5)*((mu/mass)**2)*(v**10)*(1 - (1247/336)*(v**2) - ((73/12)*q - 4*np.pi)*(v**3) - ((44711/9072) - (33/16)*(q**2))*(v**4) + ((3749/336)*q - (8191/672)*np.pi)*(v**5) + ((277/24) - (4001/84)*(v**2) + ((3583/48)*np.pi - (457/4)*q)*(v**3) + (42*(q**2) - (1091291/9072))*(v**4) + ((58487/672)*q - (364337/1344)*np.pi)*(v**5))*(e**2) + ((73/24) - (527/96)*(q*v) - (3749/672)*(v**2))*(q*(v**3)*y) + ((457/8) - (5407/48)*(q*v) - (58487/1344)*(v**2))*(q*(v**3)*(e**2)*y) )
  dldt = -(32/5)*((mu**2)/mass)*(v**7)*(1 - (1247/336)*(v**2) - ((61/12)*q - 4*np.pi)*(v**3) - ((44711/9072) - (33/16)*(q**2))*(v**4) + ((417/56)*q - (8191/672)*np.pi)*(v**5) + ((51/8) - (17203/672)*(v**2) + (-(781/12)*q + (369/8)*np.pi)*(v**3) + ((929/32)*(q**2) - (1680185/18144))*(v**4) + ((1809/224)*q - (48373/336)*np.pi)*(v**5))*(e**2) + (-(1/2) + (1247/672)*(v**2) + ((61/8)*q - 2*np.pi)*(v**3) - ((213/32)*(q**2) - (44711/18144))*(v**4) - ((4301/224)*q - (8191/1344)*np.pi)*(v**5))*y + (-(51/16) + (17203/1344)*(v**2) + ((1513/16)*q - (369/16)*np.pi)*(v**3) + ((1680185/36288) - (5981/64)*(q**2))*(v**4) - (168*q - (48373/672)*np.pi)*(v**5))*((e**2)*y))
  dqdt = -(64/5)*(mu**3)*(v**6)*(1 - q*v - (743/336)*(v**2) - ((1637/336)*q - 4*np.pi)*(v**3) + ((439/48)*(q**2) - (129193/18144) - 4*np.pi*q)*(v**4) + ((151765/18144)*q - (4159/672)*np.pi - (33/16)*(q**3))*(v**5) + ((43/8) - (51/8)*q*v - (2425/224)*(v**2) - ((14869/224)*q - (337/8)*np.pi)*(v**3) - ((453601/4536) - (3631/32)*(q**2) + (369/8)*np.pi*q)*(v**4) + ((141049/9072)*q - (38029/672)*np.pi - (929/32)*(q**3))*(v**5))*(e**2) + ((1/2)*q*v + (1637/672)*q*(v**3) - ((1355/96)*(q**2) - 2*np.pi*q)*(v**4) - ((151765/36288)*q - (213/32)*(q**3))*(v**5))*y + ((51/16)*q*v + (14869/448)*q*(v**3) + ((369/16)*np.pi*q - (33257/192)*(q**2))*(v**4) + (-(141049/18144)*q + (5981/64)*(q**3))*(v**5))*(e**2)*y )
  dcdt = -(64/5)*(mu**3)*(v**6)*y*(1-(743/336)*(v**2) - ((85/8)*q - 4*np.pi)*(v**3) - ((129193/18144) - (307/96)*(q**2))*(v**4) + ((2553/224)*q - (4159/672)*np.pi)*(v**5) + ((43/8) - (2425/224)*(v**2) + ((337/8)*np.pi - (1793/16)*q)*(v**3) - ((453601/4536) - (7849/192)*(q**2))*(v**4) + ((3421/224)*q - (38029/672)*np.pi)*(v**5))*(e**2))
  return (np.array([dedt, dldt, dcdt]), r0, y, v, q, e, outer_turn, inner_turn)

def test_stuff(constants, mass, a, mu, rad):
  energy, lmom, cart = constants[0], constants[1], constants[2] 
  coeff = [energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lmom**2 - cart, 2*mass*((a*energy - lmom)**2) + 2*mass*cart, 2*((a*energy - lmom)**2)*(a**2) + cart*(a**2)]
  lst = np.roots([energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lmom**2 - cart, 2*mass*((a*energy - lmom)**2) + 2*mass*cart, 2*((a*energy - lmom)**2)*(a**2) + cart*(a**2) + energy/2])
  lst.sort()
  outer_turn = lst[-1]
  inner_turn = lst[-2]
  e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
  return (lst, outer_turn, inner_turn, e)

def constant_derivatives_long(constants, mass, a, mu):
  compErr = False
  energy, lmom, cart = constants[0], constants[1], constants[2]
  coeff = [energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lmom**2 - cart, 2*mass*((a*energy - lmom)**2) + 2*mass*cart, 2*((a*energy - lmom)**2)*(a**2) + cart*(a**2)]
  roots = np.sort(np.roots(coeff))
  r0 = find_ro(energy, lmom, cart, mass, a)
  if (True in np.iscomplex([r0, *roots])):
    compErr = True
    r0 = r0.real
    roots = roots.real
  outer_turn, inner_turn = roots[-1], roots[-2]
  y, v, q, e = cart/(lmom**2), np.sqrt(mass/r0), a/mass, (outer_turn/r0) - 1
  dedt = -(32/5)*((mu/mass)**2)*(v**10)*(1 - (1247/336)*(v**2) - ((73/12)*q - 4*np.pi)*(v**3) - ((44711/9072) - (33/16)*(q**2))*(v**4) + ((3749/336)*q - (8191/672)*np.pi)*(v**5) + ((277/24) - (4001/84)*(v**2) + ((3583/48)*np.pi - (457/4)*q)*(v**3) + (42*(q**2) - (1091291/9072))*(v**4) + ((58487/672)*q - (364337/1344)*np.pi)*(v**5))*(e**2) + ((73/24) - (527/96)*(q*v) - (3749/672)*(v**2))*(q*(v**3)*y) + ((457/8) - (5407/48)*(q*v) - (58487/1344)*(v**2))*(q*(v**3)*(e**2)*y) )
  dldt = -(32/5)*((mu**2)/mass)*(v**7)*(1 - (1247/336)*(v**2) - ((61/12)*q - 4*np.pi)*(v**3) - ((44711/9072) - (33/16)*(q**2))*(v**4) + ((417/56)*q - (8191/672)*np.pi)*(v**5) + ((51/8) - (17203/672)*(v**2) + (-(781/12)*q + (369/8)*np.pi)*(v**3) + ((929/32)*(q**2) - (1680185/18144))*(v**4) + ((1809/224)*q - (48373/336)*np.pi)*(v**5))*(e**2) + (-(1/2) + (1247/672)*(v**2) + ((61/8)*q - 2*np.pi)*(v**3) - ((213/32)*(q**2) - (44711/18144))*(v**4) - ((4301/224)*q - (8191/1344)*np.pi)*(v**5))*y + (-(51/16) + (17203/1344)*(v**2) + ((1513/16)*q - (369/16)*np.pi)*(v**3) + ((1680185/36288) - (5981/64)*(q**2))*(v**4) - (168*q - (48373/672)*np.pi)*(v**5))*((e**2)*y))
  dcdt = -(64/5)*(mu**3)*(v**6)*y*(1-(743/336)*(v**2) - ((85/8)*q - 4*np.pi)*(v**3) - ((129193/18144) - (307/96)*(q**2))*(v**4) + ((2553/224)*q - (4159/672)*np.pi)*(v**5) + ((43/8) - (2425/224)*(v**2) + ((337/8)*np.pi - (1793/16)*q)*(v**3) - ((453601/4536) - (7849/192)*(q**2))*(v**4) + ((3421/224)*q - (38029/672)*np.pi)*(v**5))*(e**2))
  return (np.array([dedt, dldt, dcdt]), r0, y, v, q, e, outer_turn, inner_turn, compErr)

def constant_derivatives_long2(constants, mass, a, mu):
  compErr = False
  energy, lmom, cart = constants[0], constants[1], constants[2]
  coeff = [energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lmom**2 - cart, 2*mass*((a*energy - lmom)**2) + 2*mass*cart, 2*((a*energy - lmom)**2)*(a**2) + cart*(a**2)]
  roots = np.sort(np.roots(coeff))
  r0 = find_ro(energy, lmom, cart, mass, a)
  if (True in np.iscomplex([r0, *roots])):
    compErr = True
    r0 = r0.real
    roots = roots.real
  outer_turn, inner_turn = roots[-1], roots[-2]
  y, v, q, e = cart/(lmom**2), np.sqrt(mass/r0), a/mass, (outer_turn - inner_turn)/(outer_turn + inner_turn)
  dedt = -(32/5)*((mu/mass)**2)*(v**10)*(1 - (1247/336)*(v**2) - ((73/12)*q - 4*np.pi)*(v**3) - ((44711/9072) - (33/16)*(q**2))*(v**4) + ((3749/336)*q - (8191/672)*np.pi)*(v**5) + ((277/24) - (4001/84)*(v**2) + ((3583/48)*np.pi - (457/4)*q)*(v**3) + (42*(q**2) - (1091291/9072))*(v**4) + ((58487/672)*q - (364337/1344)*np.pi)*(v**5))*(e**2) + ((73/24) - (527/96)*(q*v) - (3749/672)*(v**2))*(q*(v**3)*y) + ((457/8) - (5407/48)*(q*v) - (58487/1344)*(v**2))*(q*(v**3)*(e**2)*y) )
  dldt = -(32/5)*((mu**2)/mass)*(v**7)*(1 - (1247/336)*(v**2) - ((61/12)*q - 4*np.pi)*(v**3) - ((44711/9072) - (33/16)*(q**2))*(v**4) + ((417/56)*q - (8191/672)*np.pi)*(v**5) + ((51/8) - (17203/672)*(v**2) + (-(781/12)*q + (369/8)*np.pi)*(v**3) + ((929/32)*(q**2) - (1680185/18144))*(v**4) + ((1809/224)*q - (48373/336)*np.pi)*(v**5))*(e**2) + (-(1/2) + (1247/672)*(v**2) + ((61/8)*q - 2*np.pi)*(v**3) - ((213/32)*(q**2) - (44711/18144))*(v**4) - ((4301/224)*q - (8191/1344)*np.pi)*(v**5))*y + (-(51/16) + (17203/1344)*(v**2) + ((1513/16)*q - (369/16)*np.pi)*(v**3) + ((1680185/36288) - (5981/64)*(q**2))*(v**4) - (168*q - (48373/672)*np.pi)*(v**5))*((e**2)*y))
  dcdt = -(64/5)*(mu**3)*(v**6)*y*(1-(743/336)*(v**2) - ((85/8)*q - 4*np.pi)*(v**3) - ((129193/18144) - (307/96)*(q**2))*(v**4) + ((2553/224)*q - (4159/672)*np.pi)*(v**5) + ((43/8) - (2425/224)*(v**2) + ((337/8)*np.pi - (1793/16)*q)*(v**3) - ((453601/4536) - (7849/192)*(q**2))*(v**4) + ((3421/224)*q - (38029/672)*np.pi)*(v**5))*(e**2))
  return (np.array([dedt, dldt, dcdt]), r0, y, v, q, e, outer_turn, inner_turn, compErr)

def recalc_state(constants, state, mass, a):
  energy, lmom, cart = constants[0], constants[1], constants[2]
  rad, theta = state[1], state[2]
  sig, tri = rad**2 + (a**2)*(fix_cos(theta)**2), rad**2 - 2*mass*rad + a**2

  p_r = energy*(rad**2 + a**2) - a*lmom
  r_r = (p_r)**2 - tri*(rad**2 + (a*energy - lmom)**2 + cart)
  the_the = cart - (cart + (a**2)*(1 - energy**2) + lmom**2)*(fix_cos(theta)**2) + (a**2)*(1 - energy**2)*(fix_cos(theta)**4)

  tlam = -a*(a*energy*(fix_sin(theta)**2) - lmom) + ((rad**2 + a**2)/tri)*p_r
  rlam_2 = r_r
  rlam = np.sqrt(abs(rlam_2))
  cothelam_2 = the_the
  cothelam = np.sqrt(abs(cothelam_2))
  thelam = (-1/fix_sin(theta))*cothelam
  philam = -( a*energy - ( lmom/(fix_sin(theta)**2) ) ) + (a/tri)*p_r
  
  ttau = tlam/sig
  rtau = rlam/sig
  thetau = thelam/sig
  phitau = philam/sig
  
  #sign correction
  if len(state) == 7:
    rtau = abs(rtau) * -1
  else:
    rtau = abs(rtau) * np.sign(state[5]) 
  thetau = abs(thetau) * np.sign(state[6])

  if len(state) == 8:
    new_state = np.copy(state)
  else:
    new_state = np.zeros(8)
    new_state[:4] = state[:4]
  new_state[4:] = np.array([ttau, rtau, thetau, phitau])
  return new_state
