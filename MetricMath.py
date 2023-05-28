# -*- coding: utf-8 -*-
"""
Metric Math stuff
"""

import numpy as np
from scipy import optimize
import scipy.interpolate as spi
import least_squares as ls

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
    if abs(num) <= 10**(-8):
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
    if abs(num) <= 10**(-8):
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
    sine, cosi = np.sin(theta), np.cos(theta)
    #various defined values that make math easier
    rho2, tri = r**2 + (a*cosi)**2, r**2 - s*r + a**2
    al2, w = (rho2*tri)/(rho2*tri + 2*mass*r*(a**2 + r**2)), (2*mass*r*a)/(rho2*tri + 2*mass*r*(a**2 + r**2))
    wu2 = ((rho2*tri + 2*mass*r*(a**2 + r**2))/(rho2))*sine**2
    bigA = (r**2 + a**2)**2 - tri*(a*sine)**2
    metric = [[-al2 + wu2*(w**2), 0.0,             0.0,    -w*wu2 ],
              [0.0,               rho2/tri,        0.0,    0.0    ],
              [0.0,               0.0,             rho2,   0.0    ],
              [-w*wu2,            0.0,             0.0,    wu2    ]]
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
    
    Note: applying this to a 4-momentum gives -m^2

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
    #print(interval, "woo")
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
    new = np.array([*state[:4], 0, 0, 0, 0])
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

def set_u_kerr2(mass, a, cons=False, velorient=False, vel4=False, params=False, pos=False, units="grav"):
    if units == "mks BLAM":
        G, M, c = 6.67*(10**-11), mass, 3*(10**8)
    elif units == "cgs BLAM":
        G, M, c = 6.67*(10**-8), mass, 3*(10**10)
    else:
        G, M, c = 1.0, 1.0, 1.0
    mass = 1.0
    if np.shape(cons) == (3,):
        print("Calculating initial velocity from constants E,L,C")
        #print("cons given")
        #print(cons)
        #cons = list(np.array(cons) / np.array([c**2, G*M/c, ((G*M)/c)**2]))
        #print("cons taken")
        #print(cons)
        #print(" ")
        if np.shape(pos) == (4,):
            pos = list(np.array(pos) / np.array([(G*M)/(c**3), (G*M)/(c**2), 1.0, 1.0]))
            new = recalc_state(cons, pos, mass, a)
        else:
            E, L, C = cons
            #print(E, L, C, a)
            Rdco = [4*(E**2 - 1), 6, 2*((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C)]
            #print(Rdco)
            flats = np.roots(Rdco)
            print(flats)
            pos = [0.0, flats[0], np.pi/2, 0.0]
            #pos = list(np.array(pos) / np.array([(G*M)/(c**3), (G*M)/(c**2), 1.0, 1.0]))
            new = recalc_state(cons, pos, mass, a)
    elif (np.shape(velorient) == (3,)):
        print("Calculating intial velocity from tetrad velocity and orientation")
        velorient = list(np.array(velorient) / np.array([c, 1.0, 1.0]))
        beta, eta, xi = velorient
        #eta is radial angle - 0 degrees is radially outwards, 90 degrees is no radial component
        #xi is up/down - 0 degrees is along L vector, 90 degrees is no up/down component
        eta, xi = eta*np.pi/180, xi*np.pi/180
        if (beta >= 1):
            print("Velocity greater than or equal to speed of light. Setting beta to 0.05")
            beta = 0.05
        gamma = 1/np.sqrt(1 - beta**2)
        
        if np.shape(pos) != (4,):
            r, theta = (1/beta*np.sin(eta)*np.sin(xi))**2, np.pi/2
            pos = [0.0, r, theta, 0.0]
            pos = list(np.array(pos) / np.array([(G*M)/(c**3), (G*M)/(c**2), 1.0, 1.0]))
        else:
            pos = list(np.array(pos) / np.array([(G*M)/(c**3), (G*M)/(c**2), 1.0, 1.0]))
            r, theta = pos[1], pos[2]
            
        #metric, chris = kerr([0.0, r, theta, 0.0], mass, a)
        #various defined values that make math easier
        rho2 = r**2 + (a**2)*(fix_cos(theta)**2)
        tri = r**2 - 2*mass*r + a**2
        al2 = (rho2*tri)/(rho2*tri + 2*mass*r*((a**2) + (r**2)))
        w = (2*mass*r*a)/(rho2*tri + 2*mass*r*((a**2) + (r**2)))
        wu2 = ((rho2*tri + 2*mass*r*((a**2) + (r**2)))/rho2)*(fix_sin(theta)**2)
        tetrad_matrix = np.array([[1/(np.sqrt(al2)), 0,                 0,               0],
                                  [0,                np.sqrt(tri/rho2), 0,               0],
                                  [0,                0,                 1/np.sqrt(rho2), 0],
                                  [w/np.sqrt(al2),   0,                 0,               1/np.sqrt(wu2)]])
        
        tilde = np.array([gamma, gamma*beta*fix_cos(eta), -gamma*beta*fix_sin(eta)*fix_cos(xi), gamma*beta*fix_sin(eta)*fix_sin(xi)])
        new = np.matmul(tetrad_matrix, tilde)
        new = np.array([[*pos, *new]])
    elif (np.shape(vel4) == (4,)) and np.shape(pos) == (4,):
        print("Calculating initial velocity from tetrad component velocities")
        vel4 = list(np.array(vel4) / np.array([1.0, c, (c**3)/(G*M), (c**3)/(G*M)]))
        pos = list(np.array(pos) / np.array([(G*M)/(c**3), (G*M)/(c**2), 1.0, 1.0]))
        r, theta = pos[1], pos[2]
        metric, chris = kerr(pos, mass, a)
        #various defined values that make math easier
        rho2 = r**2 + (a**2)*(fix_cos(theta)**2)
        tri = r**2 - 2*mass*r + a**2
        al2 = (rho2*tri)/(rho2*tri + 2*mass*r*((a**2) + (r**2)))
        w = (2*mass*r*a)/(rho2*tri + 2*mass*r*((a**2) + (r**2)))
        wu2 = ((rho2*tri + 2*mass*r*((a**2) + (r**2)))/rho2)*(fix_sin(theta)**2)
        tetrad_matrix = np.array([[1/(np.sqrt(al2)), 0,                 0,               0],
                                  [0,                np.sqrt(tri/rho2), 0,               0],
                                  [0,                0,                 1/np.sqrt(rho2), 0],
                                  [w/np.sqrt(al2),   0,                 0,               1/np.sqrt(wu2)]])
        rdot, thetadot, phidot = vel4[1]/vel4[0], vel4[2]/vel4[0], vel4[3]/vel4[0]
        vel_2 = (rdot**2 + (r * thetadot)**2 + (r * fix_sin(theta) * phidot)**2)
        beta = np.sqrt(vel_2)
        gamma = 1/np.sqrt(1 - vel_2)
        eta = np.arccos(np.sqrt((r * fix_sin(theta) * phidot)**2)/beta)
        xi = np.arccos(np.sqrt(rdot**2)/(beta*np.sin(eta)))
        tilde = np.array([gamma, gamma*beta*fix_cos(eta), -gamma*beta*fix_sin(eta)*fix_cos(xi), gamma*beta*fix_sin(eta)*fix_sin(xi)])
        new = np.matmul(tetrad_matrix, tilde)
    elif np.shape(params) == (3,):
        params = list(np.array(params) / np.array([(G*M)/(c**2), 1.0, 1.0]))
        print("Calculating initial velocity from orbital parameters r0, e, i (WIP)")
        new, newpams = ls.leastsquaresparam(*params, a)
        newpams = newpams * np.array([(G*M)/(c**2), 1.0, 1.0])
        print("Actual parameters:")
        print(newpams[0], newpams[1], newpams[2])
        print(new)
    else:
        print("Insufficent information provided, begone")
        new = np.array([0.0, 2000000.0, np.pi/2, 0.0, 7.088812050083354, -0.99, 0.0, 0.0])
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

def gen_RK2(butcher, solution, state, dTau, *args):
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
    
    metric, chris = kerr(state, 1.0, 0.0)
    oldE = -np.matmul(state[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
    oldLz = np.matmul(state[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
    oldQ = np.matmul(np.matmul(kill_tensor(state, 1.0, 0.0), state[4:]), state[4:])    #new Carter constant Q
    oldC = oldQ - (0.0*oldE - oldLz)**2  
    
    k = [gr_diff_eq(solution, state, *args)]                                      #start with k1, based on initial conditions
    for i in range(len(butcher["nodes"])):                                        #iterate through each non-zero node as defined by butcher table
        param = np.copy(state)                                                      #start with the basic state, then
        for j in range(len(butcher["coeff"][i])):                                   #interate through each coeffiecient
            param += np.array(butcher["coeff"][i][j] * dTau * k[j])                   #in order to obtain the approximated state based on previously defined k values
            print(param, i, j)
        k.append(gr_diff_eq(solution, param, *args))                          #which is then used to find the next k value
        #print(k)
    new_state = np.copy(state)
    for val in range(len(k)):                                                     #another for loop to add all the weights and find the final state
        new_state += k[val] * butcher["weights"][val] * dTau                        #can probably be simplified but this works for now
    
    metric, chris = kerr(new_state, 1.0, 0.0)
    newE = -np.matmul(new_state[4:], np.matmul(metric, [1, 0, 0, 0]))                              #new energy
    newLz = np.matmul(new_state[4:], np.matmul(metric, [0, 0, 0, 1]))                              #new angular momentum (axial)
    newQ = np.matmul(np.matmul(kill_tensor(new_state, 1.0, 0.0), new_state[4:]), new_state[4:])    #new Carter constant Q
    newC = newQ - (0.0*newE - newLz)**2  
    
    print("ignore")
    print(oldE, oldLz, oldC)
    print(newE, newLz, newC)
    
    print(np.linalg.norm(np.array([oldE, oldLz, oldC]) - np.array([newE, newLz, newC]))/np.linalg.norm(np.array([oldE, oldLz, oldC])) )
    
    #This is consistently giving me changes to E/L/C - should I correct that manually?
    
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

  '''
  ene = -np.matmul(test[4:], np.matmul(metric, [1, 0, 0, 0]))
  lel = np.matmul(test[4:], np.matmul(metric, [0, 0, 0, 1]))
  qrt = np.matmul(np.matmul(kill, test[4:]), test[4:])
  '''
  ene = (1 - (2*r/rho2))*test[4] + (2*a*r*(fix_sin(theta)**2)/rho2)*test[7]
  lel = -(2*a*r*(fix_sin(theta)**2)/rho2)*test[4] + ((r**2 + a**2)**2 - tri*((a*fix_sin(theta))**2))*(fix_sin(theta)**2)*test[7]/rho2
  qrt = ((lel - a*ene*(fix_sin(theta)**2))**2)/(fix_sin(theta)**2) + (a*fix_cos(theta))**2 + (rho2*test[6])**2
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

def constant_derivatives2(constants, mass, a, mu, rad):
    #peters version
  energy, lmom, cart = constants[0], constants[1], constants[2] 
  coeff = [energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lmom**2 - cart, 2*mass*((a*energy - lmom)**2) + 2*mass*cart, 2*((a*energy - lmom)**2)*(a**2) + cart*(a**2)]
  roots = np.sort(np.roots(coeff))
  r0 = find_ro(energy, lmom, cart, mass, a)
  if (True in np.iscomplex([r0, *roots])):
    r0 = r0.real
    roots = roots.real
  outer_turn, inner_turn = roots[-1], roots[-2]
  y, v, q, e = cart/(lmom**2), np.sqrt(mass/r0), a/mass, (outer_turn/r0) - 1
  dedt = -(32/5)*(mu**2)*((1+mu)/((r0**5)*((1-(e**2))**(7/2))))*(1 + (73/24)*(e**2) + (37/96)*(e**4))
  dldt = -(32/5)*(mu**2)*( np.sqrt(1+mu)/( (r0**(7/2)) * (1-(e**2))**2 ) )*(1 + (7/8)*(e**2))
  #dqdt = -(64/5)*(mu**3)*(v**6)*(1 - q*v - (743/336)*(v**2) - ((1637/336)*q - 4*np.pi)*(v**3) + ((439/48)*(q**2) - (129193/18144) - 4*np.pi*q)*(v**4) + ((151765/18144)*q - (4159/672)*np.pi - (33/16)*(q**3))*(v**5) + ((43/8) - (51/8)*q*v - (2425/224)*(v**2) - ((14869/224)*q - (337/8)*np.pi)*(v**3) - ((453601/4536) - (3631/32)*(q**2) + (369/8)*np.pi*q)*(v**4) + ((141049/9072)*q - (38029/672)*np.pi - (929/32)*(q**3))*(v**5))*(e**2) + ((1/2)*q*v + (1637/672)*q*(v**3) - ((1355/96)*(q**2) - 2*np.pi*q)*(v**4) - ((151765/36288)*q - (213/32)*(q**3))*(v**5))*y + ((51/16)*q*v + (14869/448)*q*(v**3) + ((369/16)*np.pi*q - (33257/192)*(q**2))*(v**4) + (-(141049/18144)*q + (5981/64)*(q**3))*(v**5))*(e**2)*y )
  #dcdt = -(64/5)*(mu**3)*(v**6)*y*(1-(743/336)*(v**2) - ((85/8)*q - 4*np.pi)*(v**3) - ((129193/18144) - (307/96)*(q**2))*(v**4) + ((2553/224)*q - (4159/672)*np.pi)*(v**5) + ((43/8) - (2425/224)*(v**2) + ((337/8)*np.pi - (1793/16)*q)*(v**3) - ((453601/4536) - (7849/192)*(q**2))*(v**4) + ((3421/224)*q - (38029/672)*np.pi)*(v**5))*(e**2))
  dcdt = 0
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
    #alternate e
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

def constant_derivatives_long3(constants, mass, a, mu):
    #peters version
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
  dedt = -(32/5)*(mu**2)*((1+mu)/((r0**5)*((1-(e**2))**(7/2))))*(1 + (73/24)*(e**2) + (37/96)*(e**4))
  dldt = -(32/5)*(mu**2)*( np.sqrt(1+mu)/( (r0**(7/2)) * (1-(e**2))**2 ) )*(1 + (7/8)*(e**2))
  dcdt = 0
  return (np.array([dedt, dldt, dcdt]), r0, y, v, q, e, outer_turn, inner_turn, compErr)

def constant_derivatives_long4(constants, mass, a, mu):
    #peters new e version
  compErr = False
  energy, lmom, cart = constants[0], constants[1], constants[2]
  coeff = [energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lmom**2 - cart, 2*mass*((a*energy - lmom)**2) + 2*mass*cart, 2*((a*energy - lmom)**2)*(a**2) + cart*(a**2)]
  roots = np.sort(np.roots(coeff))
  #print(roots)
  r0 = 0 #find_ro(energy, lmom, cart, mass, a)
  if (True in np.iscomplex([r0, *roots])):
    compErr = True
    r0 = r0.real
    roots = roots.real
    
  y, q = cart/(lmom**2), a/mass
  if len(roots) == 4:
    outer_turn, inner_turn = roots[-1], roots[-2]
    e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
    r0 = inner_turn/(1-e)
  else:
    outer_turn, inner_turn = (10**25), roots[-1]
    e = 1.0 - (10**(-16))
    r0 = (outer_turn + inner_turn)/2
    
  #print(r0, e)
  v = np.sqrt(mass/r0)
  #print(mu, r0, e)
  #print((1 + (7/8)*(e**2)))
  dedt = -(32/5)*(mu**2)*((1+mu)/((r0**5)*((1-(e**2))**(7/2))))*(1 + (73/24)*(e**2) + (37/96)*(e**4))
  dldt = -(32/5)*(mu**2)*( np.sqrt(1+mu)/( (r0**(7/2)) * (1-(e**2))**2 ) )*(1 + (7/8)*(e**2))
  dcdt = 0
  return (np.array([dedt, dldt, dcdt]), r0, y, v, q, e, outer_turn, inner_turn, compErr)

def constant_derivatives_long5(constants, mass, a, mu):
    #ryan new e version
  compErr = False
  energy, lmom, cart = constants[0], constants[1], constants[2]
  qart = cart + (a*energy - lmom)**2
  coeff = [energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lmom**2 - cart, 2*mass*((a*energy - lmom)**2) + 2*mass*cart, 2*((a*energy - lmom)**2)*(a**2) + cart*(a**2)]
  roots = np.sort(np.roots(coeff))
  #print(roots)
  r0 = 0 #find_ro(energy, lmom, cart, mass, a)
  if (True in np.iscomplex([r0, *roots])):
    compErr = True
    r0 = r0.real
    roots = roots.real
    
  y, q = cart/(lmom**2), a/mass
  if len(roots) == 4:
    outer_turn, inner_turn = roots[-1], roots[-2]
    e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
    r0 = inner_turn/(1-e)
  else:
    outer_turn, inner_turn = (10**25), roots[-1]
    e = 1.0 - (10**(-16))
    r0 = (outer_turn + inner_turn)/2
    
  v = np.sqrt(mass/r0)
  
  ci = lmom/((cart + lmom**(2))**(1/2))
  si = np.sqrt(1.0 - ci**2)
  psi0 = 0

  #print(mu, r0, e)
  #print(ci)
  #from Ryan Paper
  #print(( ci*(1 + (7/8)*(e**2)) + a*((1/(r0*(1-(e**2))))**(3/2))*( ((61/24) + (63/8)*(e**2) + (95/64)*(e**4)) - (ci**2)*((61/8) + (109/4)*(e**2) + (293/64)*(e**4)) - np.cos(2*psi0)*(si**2)*((5/4)*(e**2) + (13/16)*(e**4)) )  ))
  dedt = -(32/5)* (mu**2) * (1/(r0**5)) * ((1/(1-(e**2)))**(7/2)) * ((1 + (73/24)*(e**2) + (37/96)*(e**4)) - a*((1/(r0*(1-(e**2))))**(3/2))*ci*((73/12) + (1211/24)*(e**2) + (3143/96)*(e**4) + (66/65)*(e**6)))
  dldt = -(32/5)*(mu**2)*( 1/( (r0**(7/2)) * (1-(e**2))**2 ) ) * ( ci*(1 + (7/8)*(e**2)) + a*((1/(r0*(1-(e**2))))**(3/2))*( ((61/24) + (63/8)*(e**2) + (95/64)*(e**4)) - (ci**2)*((61/8) + (109/4)*(e**2) + (293/64)*(e**4)) - np.cos(2*psi0)*(si**2)*((5/4)*(e**2) + (13/16)*(e**4)) )  )
  dcdt = 0
  #print(dedt, dldt)
  return (np.array([dedt, dldt, dcdt]), r0, y, v, q, e, outer_turn, inner_turn, compErr)

def recalc_state(constants, state, mass, a):
  energy, lmom, cart = constants[0], constants[1], constants[2]
  rad, theta = state[1], state[2]
  sig, tri = rad**2 + (a**2)*(fix_cos(theta)**2), rad**2 - 2*mass*rad + a**2

  p_r = energy*(rad**2 + a**2) - a*lmom
  r_r = (p_r)**2 - tri*(rad**2 + (a*energy - lmom)**2 + cart)
  #print(r_r)
  the_the = cart - (cart + (a**2)*(1 - energy**2) + lmom**2)*(fix_cos(theta)**2) + (a**2)*(1 - energy**2)*(fix_cos(theta)**4)

  tlam = -a*(a*energy*(np.sin(theta)**2) - lmom) + ((rad**2 + a**2)/tri)*p_r
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
  
  #sign correction and initialization
  if (len(state) != 8):
    rtau = abs(rtau) * -1
    thetau = abs(thetau) 
    new_state = np.zeros(8)
    new_state[:4] = state[:4]
  else:
    rtau = abs(rtau) * np.sign(state[5]) 
    thetau = abs(thetau) * np.sign(state[6])
    new_state = np.copy(state)

  new_state[4:] = np.array([ttau, rtau, thetau, phitau])
  return new_state


#interpolate function takes calculated orbit and recalculates values for regular time intervals
def interpolate(data, time):
  data = np.array(data)
  #print(data[0] )
  new_time = np.linspace(time[0], time[-1], max(len(time), int(time[-1] - time[0])))
  #new_time = np.linspace(time[0], time[-1], min(len(time), 10**5))
  r_poly = spi.CubicSpline(time, data[:,0])
  theta_poly = spi.CubicSpline(time, data[:,1])
  phi_poly = spi.CubicSpline(time, data[:,2])
  new_data = np.array([[r_poly(i), theta_poly(i), phi_poly(i)] for i in new_time])
  return new_data, new_time

#sphr2quad function finds the quadrupole moment of a specific position in spherical coordinates
def sphr2quad(pos):
  x = pos[0] * np.sin(pos[1]) * np.cos(pos[2])
  y = pos[0] * np.sin(pos[1]) * np.sin(pos[2])
  z = pos[0] * np.cos(pos[1])
  qmom = np.array([[2*x*x - (y**2 + z**2), 3*x*y,                 3*x*z],
                   [3*y*x,                 2*y*y - (x**2 + z**2), 3*y*z],
                   [3*z*x,                 3*z*y,                 2*z*z - (x**2 + y**2)    ]], dtype=np.float64)
  return qmom

#matrix_derive function calculates the nth time derivative of a series of 3x3 matrices,
#where n is determined by degree
def matrix_derive(data, time, degree):
  polys = [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]]
  for i in range(3):
    for j in range(3):
      polys[i][j] = spi.CubicSpline(time, data[:,i,j])
  new_data = np.array([[[polys[0][0](i, degree), polys[0][1](i, degree), polys[0][2](i, degree)],
                        [polys[1][0](i, degree), polys[1][1](i, degree), polys[1][2](i, degree)],
                        [polys[2][0](i, degree), polys[2][1](i, degree), polys[2][2](i, degree)]] for i in time])
  return new_data

#gwaves function calculates gravitational wave data based on 
#quadrupole moments, time data, and some given distance from the source
def gwaves(quad_moment, time, distance):
  der_2 = matrix_derive(quad_moment, time, 2)
  waves = np.array([(2/distance) * entry for entry in der_2])
  return waves

#full_transform combines all previous functions in this block in order to
#produce gravitational wave data from the orbits calculated by one of the
#main orbit functions, like schw_orbit or kerr_orbit
def full_transform(data, distance):    #defunctish??
  sphere, time = data["pos"], data["time"]
  int_sphere, int_time = interpolate(sphere, time)
  quad = np.array([sphr2quad(pos) for pos in int_sphere])
  waves = gwaves(quad, int_time, distance)
  return waves, int_time

#sphr2quad function finds the quadrupole moment of a specific position in spherical coordinates
def ortholize(pos, mu):
  x = pos[0] * np.sin(pos[1]) * np.cos(pos[2])
  y = pos[0] * np.sin(pos[1]) * np.sin(pos[2])
  z = pos[0] * np.cos(pos[1])

  qmom = np.array([[3*x*x - pos[0]**2,       3*x*y,             3*x*z      ],
                   [      3*y*x,       3*y*y - pos[0]**2,       3*y*z      ],
                   [      3*z*x,             3*z*y,       3*z*z - pos[0]**2]], dtype=np.float64)
  qmom = qmom*mu
  return qmom

def peters_integrate(constants, a, mu, states, ind1, ind2):
    compErr = False
    mass = 1
    energy, lmom, cart = constants[0], constants[1], constants[2]
    coeff = [energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lmom**2 - cart, 2*mass*((a*energy - lmom)**2) + 2*mass*cart, 2*((a*energy - lmom)**2)*(a**2) + cart*(a**2)]
    roots = np.sort(np.roots(coeff))
    if (True in np.iscomplex(roots)):
      compErr = True
      roots = roots.real    
    y, q = cart/(lmom**2), a/mass
    if len(roots) == 4:
      outer_turn, inner_turn = roots[-1], roots[-2]
      e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
      r0 = inner_turn/(1-e)
    else:
      outer_turn, inner_turn = (10**25), roots[-1]
      e = 1.0 - (10**(-16))
      r0 = (outer_turn + inner_turn)/2
    v = np.sqrt(mass/r0)
    dedt, dldt, dcdt = 0, 0, 0
    
    if ind1 != ind2:
        states = np.array(states)
        sphere, time = states[ind1:ind2][:,1:4], states[ind1:ind2][:,0]
        int_sphere, int_time = interpolate(sphere, time)
        div = int_time[1]-int_time[0]
        quad = np.array([ortholize(pos, mu) for pos in int_sphere])
        dt2 = matrix_derive(quad, int_time, 2)
        dt3 = matrix_derive(quad, int_time, 3)
    
        
        for i in range(3):
            for j in range(3):
                dedt += sum( ((dt3[:,i,j])**2 - (1/3)*(dt3[:,i,i])*(dt3[:,j,j]))*div )
                if (i+j == 1):
                    for m in range(3):
                        dldt += (j-i)*sum( (dt2[:,i,m]*dt3[:,j,m])*div )  
            
    dedt = (-1/5)*dedt
    dldt = (-2/5)*dldt
    dcdt = 0
    #print(dedt, dldt)
    return (np.array([dedt, dldt, dcdt]), r0, y, v, q, e, outer_turn, inner_turn, compErr)

def levi_civita(i,j,k):
    pos = [[0,1,2], [1,2,0], [2,0,1]]
    neg = [[0,2,1], [1,0,2], [2,1,0]]
    order = [i,j,k]
    if order in pos:
        return 1
    elif order in neg:
        return -1
    else:
        return 0

def peters_integrate2(constants, a, mu, states, ind1, ind2):
    #print("t1")
    compErr = False
    mass = 1
    energy, lx, ly, lz, cart = constants[0], constants[1], constants[2], constants[3], constants[4]
    coeff = [energy**2 - 1, 2*mass, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*mass*((a*energy - lz)**2) + 2*mass*cart, 2*((a*energy - lz)**2)*(a**2) + cart*(a**2)]
    roots = np.sort(np.roots(coeff))
    if (True in np.iscomplex(roots)):
      compErr = True
      roots = roots.real    
    y, q = cart/(lz**2), a/mass
    if len(roots) == 4:
      outer_turn, inner_turn = roots[-1], roots[-2]
      e = (outer_turn - inner_turn)/(outer_turn + inner_turn)
      r0 = inner_turn/(1-e)
    else:
      outer_turn, inner_turn = (10**25), roots[-1]
      e = 1.0 - (10**(-16))
      r0 = (outer_turn + inner_turn)/2
    v = np.sqrt(mass/r0)
    dedt, dldt, dcdt = 0, np.array([0.0, 0.0, 0.0]), 0
    
    if ind1 != ind2:
        states = np.array(states)
        sphere, time = states[ind1:ind2][:,1:4], states[ind1:ind2][:,0]
        #print(sphere)
        #print(time)
        int_sphere, int_time = interpolate(sphere, time)
        div = int_time[1]-int_time[0]
        quad = np.array([ortholize(pos, mu) for pos in int_sphere])
        dt2 = matrix_derive(quad, int_time, 2)
        dt3 = matrix_derive(quad, int_time, 3)
        #print((int_time[-1]-int_time[0]))
    
        
        for i in range(3):
            for j in range(3):
                dedt += np.sum( ( (dt3[:,i,j])**2 - (1/3)*(dt3[:,i,i])*(dt3[:,j,j]) )*div)
                dldt[i] += np.sum(dt2[:,(i+1)%3,j]*dt3[:,(i+2)%3, j]*div)
                dldt[i] -= np.sum(dt2[:,(i-1)%3,j]*dt3[:,(i-2)%3, j]*div)

    dedt = (-1/5)*dedt
    #print(dldt)
    dlxdt, dlydt, dlzdt = (-2/5)*dldt
    dcdt = 2*lx*dlxdt + 2*ly*dlydt  #only true if a=0
    #print(states[0])
    theta = states[0][2]
    dcdt = 2*((fix_sin(theta)**(-2))*(lz - a*energy*(fix_sin(theta)**(2)))*(dlzdt - a*dedt*(fix_sin(theta)**(2))) + (a*energy - lz)*(a*dedt - dlzdt))
    #print("SURPRISE", dedt, dlxdt, dlydt, dlzdt, dcdt)
    #print("why")
    #print(dedt, dlxdt, dlydt, dlzdt)
    return (np.array([dedt, dlxdt, dlydt, dlzdt, dcdt]), r0, y, v, q, e, outer_turn, inner_turn, compErr)

def peters_integrate3(constants, a, mu, states, ind1, ind2):
    compErr = False
    mass = 1
    energy, lz, cart = constants[0], constants[1], constants[2]
    coeff = np.array([energy**2 - 1, 2.0*mass, (a**2)*(energy**2 - 1) - lz**2 - cart, 2*mass*((a*energy - lz)**2) + 2*mass*cart, -cart*(a**2)])
    coeff2 = np.array([(energy**2 - 1)*4, (2.0*mass)*3, ((a**2)*(energy**2 - 1) - lz**2 - cart)*2, 2*mass*((a*energy - lz)**2) + 2*mass*cart])

    def quad(a, b, c):
        sol1 = (-b + (b**2 - 4*a*c)**(0.5))/(2*a)
        sol2 = (-b - (b**2 - 4*a*c)**(0.5))/(2*a)
        #if sol1 == np.sqrt(-1):
        #    print("WHY")
        return sol1, sol2

    #cubic part
    def cubic_solver(coeffs):
        a0, a1, a2 = coeffs[3]/coeffs[0], coeffs[2]/coeffs[0], coeffs[1]/coeffs[0]
        q = (1/3)*a1 - (1/9)*(a2**2)
        r = (1/6)*(a1*a2 - 3*a0) - (1/27)*(a2**3)
        s1, s2 = (r + np.sqrt(q**3 + r**2 + 0j))**(1/3),  (r - np.sqrt(q**3 + r**2 + 0j))**(1/3)
        roots = [(s1 + s2) - a2/3, 
                 -0.5*(s1 + s2) - a2/3 + 0.5*((-3)**(1/2))*(s1 - s2),
                 -0.5*(s1 + s2) - a2/3 - 0.5*((-3)**(1/2))*(s1 - s2)]
        # get rid of tiny imaginary bits. For a 10^6 solar mass BH, 10**(-10) distance units is ~15cm
        roots = np.where(np.imag(roots) < 10**(-10), np.real(roots), roots)
        return roots
    
    #quartic part
    def quartic_solver(coeffs):
        a3, a2, a1, a0 = coeffs[1]/coeffs[0], coeffs[2]/coeffs[0], coeffs[3]/coeffs[0], coeffs[4]/coeffs[0]
        #print(a3, a2, a1, a0, "coeffs!")
        a, b, c, d = 1.0, -a2, a1*a3 - 4*a0, -(a1**2 + a0*a3*a3 - 4*a0*a2)
        #print(a, b, c, d, "lil nums!")
        A = -(b*b*b)/(27*a*a*a) + (b*c)/(6*a*a) - d/(2*a)
        B = c/(3*a) - (b*b)/(9*a*a)
        C = b/(3*a)
        #print(A, B, C, "nums!")
        u1 = (A + (A*A + B*B*B)**0.5)**(1.0/3.0) + (A - (A*A + B*B*B)**0.5)**(1.0/3.0) - C
        #print(u1, "u1!")
        sols = cubic_solver([1.0, -a2, a1*a3 - 4*a0, -(a1*a1 + a0*a3*a3 - 4*a0*a2)])
        sols = np.where(np.imag(sols) < 10**(-14), np.real(sols), sols)
        #print(sols, "is there a real boy here?")
        u2 = sols[np.where(np.iscomplex(sols) == False)[0][0]]
        #print(u2, "it's this guy!")
    
        #print(1.0, 0.5*a3 - (0.25*a3*a3 + u1 -a2)**0.5, 0.5*u1 - (0.25*u1*u1 - a0)**0.5)
        #print(1.0, 0.5*a3 + (0.25*a3*a3 + u1 -a2)**0.5, 0.5*u1 + (0.25*u1*u1 - a0)**0.5)
        #print(1.0, 0.5*a3 - (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 - (0.25*u2*u2 - a0)**0.5)
        #print(1.0, 0.5*a3 + (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 + (0.25*u2*u2 - a0)**0.5)
        roots = [*quad(1.0, 0.5*a3 - (0.25*a3*a3 + u1 -a2)**0.5, 0.5*u1 - (0.25*u1*u1 - a0)**0.5),
                 *quad(1.0, 0.5*a3 + (0.25*a3*a3 + u1 -a2)**0.5, 0.5*u1 + (0.25*u1*u1 - a0)**0.5)]
        roots = np.where(np.imag(roots) < 10**(-14), np.real(roots), roots)
        roots2 = [*quad(1.0, 0.5*a3 - (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 - (0.25*u2*u2 - a0)**0.5),
                  *quad(1.0, 0.5*a3 + (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 + (0.25*u2*u2 - a0)**0.5)]
        roots2 = np.where(np.imag(roots) < 10**(-14), np.real(roots), roots)
        roots3 = [*np.roots([1.0, 0.5*a3 - (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 - (0.25*u2*u2 - a0)**0.5]),
                  *np.roots([1.0, 0.5*a3 + (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 + (0.25*u2*u2 - a0)**0.5])]
        roots3 = np.where(np.imag(roots) < 10**(-14), np.real(roots), roots)
        #x0, y0, z0 = 1.0, 0.5*a3 - (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 - (0.25*u2*u2 - a0)**0.5
        #x1, y1, z1 = 1.0, 0.5*a3 + (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 + (0.25*u2*u2 - a0)**0.5
        r1, r2 = quad(1.0, 0.5*a3 - (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 - (0.25*u2*u2 - a0)**0.5)
        r3, r4 = quad(1.0, 0.5*a3 + (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 + (0.25*u2*u2 - a0)**0.5)
        #print(quad(1.0, 0.5*a3 - (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 - (0.25*u2*u2 - a0)**0.5), "maybe dumb?")
        #print([*quad(1.0, 0.5*a3 - (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 - (0.25*u2*u2 - a0)**0.5),
        #          *quad(1.0, 0.5*a3 + (0.25*a3*a3 + u2 -a2)**0.5, 0.5*u2 + (0.25*u2*u2 - a0)**0.5)], "no way")
        #print(roots, "old")
        #print(roots2, "new")
        #print(roots3, "cheat")
        #print(r1, r2, r3, r4)
        return np.array([r1, r2, r3, r4])
    
    #turns = np.sort(quartic_solver(coeff))
    flats = cubic_solver(coeff2)
    turns = np.sort(np.roots(coeff))
    #flats = np.roots(coeff2)
    r0 = np.max(flats)
    rc = (1 + np.sqrt(1 + a))**2
    if (True in np.iscomplex(turns)):
      #compErr = True
      #print("DAMN")
      #print(turns)
      #print("is it though?? this should be >= 0")
      this = np.sum(np.array([r0**4, r0**3, r0**2, r0, 1.0])*coeff)
      #print(this)
      if this < 0.0:
          compErr = True
      #turns = np.real(turns)  
    y, q = cart/(lz**2), a/mass
    outer_turn, inner_turn = turns[-1], turns[-2]
    e = (outer_turn - inner_turn)/(outer_turn + inner_turn - 2*rc)
    v = np.sqrt(mass/r0)
    dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[ind1:ind2][:,1:4], states[ind1:ind2][:,0]
        #print(sphere)
        #print(time)
        #print( np.diff(time))
        int_sphere, int_time = interpolate(sphere, time)
        div = int_time[1]-int_time[0]
        quad = np.array([ortholize(pos, mu) for pos in int_sphere])
        dt2 = matrix_derive(quad, int_time, 2)
        dt3 = matrix_derive(quad, int_time, 3)

        for i in range(3):
            for j in range(3):
                dedt += np.sum( ( (dt3[:,i,j])**2 - (1/3)*(dt3[:,i,i])*(dt3[:,j,j]) )*div)
                dldt[i] += np.sum(dt2[:,(i+1)%3,j]*dt3[:,(i+2)%3, j]*div)
                dldt[i] -= np.sum(dt2[:,(i-1)%3,j]*dt3[:,(i-2)%3, j]*div)

    dedt = (-1/5)*dedt
    dlxdt, dlydt, dlzdt = (-2/5)*dldt
    theta = states[0][2]
    dcdt = 2*((fix_sin(theta)**(-2))*(lz - a*energy*(fix_sin(theta)**(2)))*(dlzdt - a*dedt*(fix_sin(theta)**(2))) + (a*energy - lz)*(a*dedt - dlzdt))
    return (np.array([dedt, dlxdt, dlydt, dlzdt]), r0, y, v, q, e, outer_turn, inner_turn, compErr)

def new_recalc_state(constants, state, mass, a):
  energy, lmom, cart = constants[0], constants[1], constants[2]
  rad, theta = state[1], state[2]
  sig, tri = rad**2 + (a**2)*(fix_cos(theta)**2), rad**2 - 2*mass*rad + a**2
  
  facA = 1 - (2*mass*rad)/sig
  facB = 2*mass*a*rad*(fix_sin(theta)**2)/sig
  facC = (fix_sin(theta)**2)*((rad**2 + a**2)**2 - tri*((a*fix_sin(theta))**2))/sig
  
  ut = (energy*facC - lmom*facB)/(facA*facC - facB**2)
  uphi = (energy*facB + lmom*facA)/(facA*facC - facB**2)
  
  #ASSUMED PLANAR ORBIT
  utheta = 0
  
  metric, chris = kerr(state, mass, a)
  ur = ((-1/metric[1][1])*(1 + metric[0][0]*(ut**2) + metric[2][2]*(utheta**2) + metric[3][3]*(uphi**2) + 2*metric[0][3]*ut*uphi))**(1/2)
  
  #sign correction
  if (len(state) == 7):
    ur = abs(ur) * -1
  else:
    ur = abs(ur) * np.sign(state[5]) 
  utheta = abs(utheta) * np.sign(state[6])

  if len(state) == 8:
    new_state = np.copy(state)
  else:
    new_state = np.zeros(8)
    new_state[:4] = state[:4]
  new_state[4:] = np.array([ut, ur, utheta, uphi])
  return new_state

def new_recalc_state2(constants, con_derv, state, mu, mass, a):
  ene, lmom, cart = constants[0], np.array([constants[1], constants[2], constants[3]]), constants[4]
  ened, lmomd, cartd = con_derv[0], np.array([con_derv[1], con_derv[2], con_derv[3]]), con_derv[4]
  #print("change", lmomd)
  full_v = np.sqrt(state[5]**2 + (state[1]*state[6])**2 + (state[1]*fix_cos(state[2])*state[7])**2)
  orthostate = np.array([state[0], 
                         state[1]*fix_sin(state[2])*fix_cos(state[3]),
                         state[1]*fix_sin(state[2])*fix_sin(state[3]),
                         state[1]*fix_cos(state[2]),
                         state[4],
                         state[5]*fix_sin(state[2])*fix_cos(state[3]) + state[1]*state[6]*fix_cos(state[2])*fix_cos(state[3]) - state[1]*state[7]*fix_sin(state[2])*fix_sin(state[3]),
                         state[5]*fix_sin(state[2])*fix_sin(state[3]) + state[1]*state[6]*fix_cos(state[2])*fix_cos(state[3]) + state[1]*state[7]*fix_sin(state[2])*fix_cos(state[3]),
                         state[5]*fix_cos(state[2]) - state[1]*state[6]*fix_sin(state[2])])
  pos = orthostate[1:4]
  print(pos)
  old_3vel = state[5:]/state[4]
  print(old_3vel)
  ortho_3vel = orthostate[5:]/orthostate[4]
  print(ortho_3vel)
  #new_ortho_3vel = (1/mu) * np.dot(lmom, lmomd) * np.cross(lmom, pos) / (np.linalg.norm(np.cross(lmom, pos))**2) + ortho_3vel
  new_ortho_3vel = (1/(mu*(state[1]**2))) * np.dot(lmom, lmomd) * np.cross(lmom, pos) / np.dot(lmom, lmom) + ortho_3vel
  print(new_ortho_3vel)
  new_full_v = np.linalg.norm(new_ortho_3vel)
  rho, rho_v = np.linalg.norm(pos[:2]), np.linalg.norm(new_ortho_3vel[:2])
  new_3vel = np.array([np.dot(pos, new_ortho_3vel)/state[1],
                      (-1/np.sqrt(1 - (pos[2]/state[1])**2))*(new_ortho_3vel[2]*state[1] - (np.dot(pos, new_ortho_3vel)/state[1])*pos[2])/state[1]**2,
                      (1/((pos[1]/pos[0])**2 + 1))*(new_ortho_3vel[1]*pos[0] - new_ortho_3vel[0]*pos[1])/pos[0]**2  ])
  print(new_3vel)
  print("my vels")
  metric, chris = kerr(state, mass, a)
  #print(metric)
  #print((metric[0][0] + metric[1][1]*(new_3vel[0])**2 + metric[2][2]*(new_3vel[1])**2 + metric[3][3]*(new_3vel[2])**2 + 2*metric[0][3]*new_3vel[2]))
  new_ut = np.sqrt(abs(-1/(metric[0][0] + metric[1][1]*(new_3vel[0])**2 + metric[2][2]*(new_3vel[1])**2 + metric[3][3]*(new_3vel[2])**2 + 2*metric[0][3]*new_3vel[2]) ))
  # new and BROKEN - new_ut = (-metric[0][0] - np.dot(pos, ortho_3vel)/(state[1]-2) - (state[1]*new_ortho_3vel[2] - (pos[2]/state[1])*np.dot(pos, ortho_3vel))*(1 - (pos[2]/pos[1])**2)**(-1/2) + ((state[1]*fix_sin(state[2]))**2)*(lmom[2]/(pos[1]**2 + 1))*((1/mu)*(np.dot(lmom, lmomd)/np.dot(lmom, lmom) + 1)))**(-1/2)
  #false_4vec = np.array([1, *new_3vel])
  #new_ut = np.sqrt(-1/np.matmul(np.matmul(metric, false_4vec), false_4vec))
  rad, theta = state[1], state[2]
  sig, tri = rad**2 + (a**2)*(fix_cos(theta)**2), rad**2 - 2*mass*rad + a**2
  
  facA = 1 - (2*mass*rad)/sig
  facB = 2*mass*a*rad*(fix_sin(theta)**2)/sig
  facC = (fix_sin(theta)**2)*((rad**2 + a**2)**2 - tri*((a*fix_sin(theta))**2))/sig
  
  ut = ((ene+ened)*facC - lmom[2]*facB)/(facA*facC - facB**2)
  
  #print(ut, new_ut)
  print("DIFFERENCE")
  #ut is old version, straight up doesn't work
  new_vel = np.array([new_ut, *(new_3vel*ut)])
  
  new_state = np.zeros(8)
  new_state[:4] = state[:4]
  new_state[4:] = new_vel
  return new_state

def getEnergy(state, mass, a):
    metric, chris = kerr(state, mass, a)
    stuff = np.matmul(metric, state[4:])
    ene = -stuff[0]
    #print(stuff[3])
    return ene

def getLs(state, mu):
    t, r, theta, phi, vel4 = *state[:4], state[4:]
    sint, cost = fix_sin(theta), fix_cos(theta)
    #print(sint, cost)
    #print(np.sin(theta), np.cos(theta))
    sinp, cosp = fix_sin(phi), fix_cos(phi)
    sph2cart = np.array([[1, 0,         0,           0           ],
                         [0, sint*cosp, r*cost*cosp, -r*sint*sinp],
                         [0, sint*sinp, r*cost*sinp, r*sint*cosp ],
                         [0, cost,      -r*sint,     0           ]])
    #print(sph2cart)
    vel4cart = np.matmul(sph2cart, vel4)
    vel3cart = vel4cart[1:4]
    pos3cart = np.array([r*sint*cosp, r*sint*sinp, r*cost])
    #print(pos3cart)
    #print(vel3cart)
    Lmom = np.cross(pos3cart, vel3cart)
    return Lmom

def getLs2(state, mu, a):
    r, theta, phi = state[1:4]
    mass = 1.0
    sint, cost = fix_sin(theta), fix_cos(theta)
    sinp, cosp = fix_sin(phi), fix_cos(phi)
    rho2, tri = r**2 + (a*cost)**2, r**2 - 2*mass*r + a**2
    al2, w = (rho2*tri)/(rho2*tri + 2*mass*r*(a**2 + r**2)), (2*mass*r*a)/(rho2*tri + 2*mass*r*(a**2 + r**2))
    wu2 = ((rho2*tri + 2*mass*r*(a**2 + r**2))/(rho2))*sint**2
    
    cartpos = np.array([r*sint*cosp, r*sint*sinp, r*cost])
    tet2cor = np.array([[1/np.sqrt(al2), 0.0,               0.0,             0.0             ],
                        [0.0,            np.sqrt(tri/rho2), 0.0,             0.0             ],
                        [0.0,            0.0,               1/np.sqrt(rho2), 0.0             ],
                        [w/np.sqrt(al2), 0.0,               0.0,             1/np.sqrt(wu2)  ]])
    cor2tet = np.linalg.inv(tet2cor)
    tet = np.matmul(cor2tet, state[4:])
    gamma, sphvel = tet[0], tet[1:]
    cartvel = np.matmul(np.array([[sint*cosp, r*cost*cosp, -r*sinp],
                                  [sint*sinp, r*cost*sinp,  r*cosp],
                                  [     cosp,     -r*sint,     0.0]]), sphvel)
    print(cartpos)
    print(cartvel)
    Lmom = mu*(1/gamma) * np.cross(cartpos, cartvel)
    return Lmom
                     
def new_recalc_state3(con_derv, state, mu, mass, a, trial=0, old_diff=False):
    t, r, theta, phi, vel4 = *state[:4], state[4:]
    sint, cost = fix_sin(theta), fix_cos(theta)
    sinp, cosp = fix_sin(phi), fix_cos(phi)
    rho2, tri = r**2 + (a*cost)**2, r**2 - 2*mass*r + a**2
    al2, w = (rho2*tri)/(rho2*tri + 2*mass*r*(a**2 + r**2)), (2*mass*r*a)/(rho2*tri + 2*mass*r*(a**2 + r**2))
    wu2 = ((rho2*tri + 2*mass*r*(a**2 + r**2))/(rho2))*sint**2
    
    '''
    #schwarz tetrad
    evhor = (r-2)/r
    tet2cor = np.array([[evhor**(-1/2),   0.0,            0.0,   0.0         ],
                        [0.0,             evhor**(1/2),   0.0,   0.0         ],
                        [0.0,             0.0,            1/r,   0.0         ],
                        [0.0,             0.0,            0.0,   1/(r*sint)  ]])
    '''
    #kerr tetrad
    tet2cor = np.array([[1/np.sqrt(al2), 0.0,               0.0,             0.0             ],
                        [0.0,            np.sqrt(tri/rho2), 0.0,             0.0             ],
                        [0.0,            0.0,               1/np.sqrt(rho2), 0.0             ],
                        [w/np.sqrt(al2), 0.0,               0.0,             1/np.sqrt(wu2)  ]])
    cor2tet = np.linalg.inv(tet2cor)
    sph2cart = np.array([[1.0, 0.0,       0.0,       0.0  ],
                         [0.0, sint*cosp, cost*cosp, -sinp],
                         [0.0, sint*sinp, cost*sinp, cosp ],
                         [0.0, cost,      -sint,     0.0  ]])
    cart2sph = np.linalg.inv(sph2cart)
    #metric, chris = kerr(state, mass, a)
    bigA = np.zeros([4,3])
    
    tet_state = np.matmul(cor2tet, vel4)
    cart_tet_state = np.matmul(sph2cart, np.matmul(cor2tet, vel4))
    strip_ct_state = (cart_tet_state[1:4])/(cart_tet_state[0])

    new_strip_ct_state = np.array([1.0, 1.0, 1.0])
    counter = 0
    while (np.linalg.norm(new_strip_ct_state) >= 1) and (counter <= 4.0):
        new_strip_ct_state = np.array([1.0, 1.0, 1.0])
        for i in range(3):
            dvel = np.array([0.0, 0.0, 0.0])
            dvel[i] = 10**(-(6 + counter))
            new_strip_ct_state = strip_ct_state + dvel
            newgamma = (1 - np.linalg.norm(new_strip_ct_state)**2)**(-1/2)
            new_ct_state = newgamma*np.array([1, *new_strip_ct_state])
            new_vel = np.matmul(tet2cor, np.matmul(cart2sph, new_ct_state))
            new_state = np.array([*state[:4], *new_vel])
            old_cons = np.array([getEnergy(state, mass, a), *getLs(state, mu)])
            new_cons = np.array([getEnergy(new_state, mass, a), *getLs(new_state, mu)])
            del_cons = new_cons - old_cons                                         #Using newcons - oldcons instead of assuming delcons is linear like Jeremy said
                                                                                   #That's what linear means you dip
            bigA[:, i] = del_cons/dvel[i]
      
        dcons = con_derv[0:4]
        bigAt = np.transpose(bigA)
        block = np.linalg.inv(np.matmul(bigAt, bigA))
        dvel = np.matmul(block, np.matmul(bigAt, dcons))

        new_strip_ct_state = strip_ct_state + dvel
        counter += 0.5
    
    if np.linalg.norm(new_strip_ct_state) >= 1:
        print("It's still screwed up???")
        return state
    newgamma = 1/np.sqrt(1 - np.dot(new_strip_ct_state, new_strip_ct_state))
    new_ct_state = newgamma*np.array([1, *new_strip_ct_state])
    new_vel = np.matmul(tet2cor, np.matmul(cart2sph, new_ct_state))
    new_state = np.array([*state[:4], *new_vel])
    
    actuals = con_derv[0:4]
    calcs = np.array([getEnergy(new_state, mass, a), *getLs(new_state, mu)]) - np.array([getEnergy(state, mass, a), *getLs(state, mu)])
    if (trial<1):
        no_change = np.array([0.0, 0.0, 0.0, 0.0])
        diff = np.linalg.norm(actuals - no_change)
    diff = np.linalg.norm(actuals - calcs)
    
    print("Trial ", trial, ": Diff = ", diff)
    if diff > 10**(-15) or True in (np.sign(state) != np.sign(new_state)):
        if trial < 25:
            #print("Failure")
            #print("interval")
            #print(check_interval(kerr, new_state, mass, a))
            #print("actual vs calculated vs true actual??")
            #print(actuals)
            #print(calcs)
            '''
            if old_diff != False:
                if abs((diff-old_diff)/old_diff) <= 0.1:
                    #print("hovering, skip")
                    trial = 2*trial
            '''
            #print(np.array([getEnergy(state, mass, a), *getLs(state, mu)]))
            new_trial = trial + 1
            #should_be = np.array([getEnergy(state, mass, a), *getLs(state, mu)]) + actuals
            new_state = new_recalc_state3(actuals - calcs, new_state, mu, mass, a, trial = new_trial, old_diff=diff)
    new_state[4:] = np.abs(new_state[4:])*np.sign(state[4:])
    return new_state

def new_recalc_state4(con_derv, state, mu, mass, a):
    true_state = np.copy(state)
    t, r, theta, phi = state[:4]
    sint, cost = fix_sin(theta), fix_cos(theta)
    sinp, cosp = fix_sin(phi), fix_cos(phi)
    rho2, tri = r**2 + (a*cost)**2, r**2 - 2*mass*r + a**2
    al2, w = (rho2*tri)/(rho2*tri + 2*mass*r*(a**2 + r**2)), (2*mass*r*a)/(rho2*tri + 2*mass*r*(a**2 + r**2))
    wu2 = ((rho2*tri + 2*mass*r*(a**2 + r**2))/(rho2))*sint**2
    
    #kerr tetrad
    tet2cor = np.array([[1/np.sqrt(al2), 0.0,               0.0,             0.0             ],
                        [0.0,            np.sqrt(tri/rho2), 0.0,             0.0             ],
                        [0.0,            0.0,               1/np.sqrt(rho2), 0.0             ],
                        [w/np.sqrt(al2), 0.0,               0.0,             1/np.sqrt(wu2)  ]])
    cor2tet = np.linalg.inv(tet2cor)
    sph2cart = np.array([[1.0, 0.0,       0.0,       0.0  ],
                         [0.0, sint*cosp, cost*cosp, -sinp],
                         [0.0, sint*sinp, cost*sinp, cosp ],
                         [0.0, cost,      -sint,     0.0  ]])
    cart2sph = np.linalg.inv(sph2cart)
    #metric, chris = kerr(state, mass, a)
    bigA = np.zeros([4,3])
    
    diff = 100.0
    count = 0
    new_state = np.copy(state)
    delcons = con_derv[:4]
    while (diff > 10**(-14)) and (count < 25):
        vel4 = new_state[4:]
        cart_tet_state = np.matmul(sph2cart, np.matmul(cor2tet, vel4))
        strip_ct_state = (cart_tet_state[1:4])/(cart_tet_state[0])
        new_strip_ct_state = np.array([1.0, 1.0, 1.0])
    
        counter = 0
        while (np.linalg.norm(new_strip_ct_state) >= 1):
            new_strip_ct_state = np.array([1.0, 1.0, 1.0])
            for i in range(3):
                dvel = np.array([0.0, 0.0, 0.0])
                dvel[i] = 10**(-(8 + counter))
                new_strip_ct_state = strip_ct_state + dvel
                newgamma = (1 - np.linalg.norm(new_strip_ct_state)**2)**(-1/2)
                new_ct_state = newgamma*np.array([1, *new_strip_ct_state])
                new_vel = np.matmul(tet2cor, np.matmul(cart2sph, new_ct_state))
                new_state = np.array([*state[:4], *new_vel])
                old_cons = np.array([getEnergy(state, mass, a), *getLs(state, mu)])
                new_cons = np.array([getEnergy(new_state, mass, a), *getLs(new_state, mu)])
                del_cons = new_cons - old_cons                                         #Using newcons - oldcons instead of assuming delcons is linear like Jeremy said
                                                                                       #That's what linear means you dip
                bigA[:, i] = del_cons/dvel[i]
          
            dcons = con_derv[0:4]
            bigAt = np.transpose(bigA)
            block = np.linalg.inv(np.matmul(bigAt, bigA))
            dvel = np.matmul(block, np.matmul(bigAt, dcons))
    
            new_strip_ct_state = strip_ct_state + dvel
            counter += 0.5
        '''
        if np.linalg.norm(new_strip_ct_state) >= 1:
            print("It's still screwed up???")
            return state
        '''
        newgamma = 1/np.sqrt(1 - np.dot(new_strip_ct_state, new_strip_ct_state))
        new_ct_state = newgamma*np.array([1, *new_strip_ct_state])
        new_vel = np.matmul(tet2cor, np.matmul(cart2sph, new_ct_state))
        new_state = np.array([*state[:4], *new_vel])
        
        actuals = delcons
        calcs = np.array([getEnergy(new_state, mass, a), *getLs(new_state, mu)]) - np.array([getEnergy(state, mass, a), *getLs(state, mu)])
        calcs = np.where(np.abs(calcs) < 10**(-18), actuals, calcs)
        #print(count)
        #print(actuals)
        #print(calcs)
        diff = np.linalg.norm((actuals - calcs)/actuals)
        state = np.copy(new_state)
        delcons = delcons - (actuals - calcs)
        count += 1
    
    if diff > 10**(-14):
        print("damn it's still screwed up")
        print(actuals)
        print(calcs)
        print(diff)
    new_state[4:] = np.abs(new_state[4:])*np.sign(true_state[4:])
    return new_state

def new_recalc_state5(cons, con_derv, state, mu, mass, a):
    # Step 1
    E0, Lphi0, C0 = cons
    state_false = recalc_state([E0, Lphi0, C0], state, mass, a)

    # Step 2
    if a == 0:
        z2 = C0/(Lphi0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2pl = ((A + Lphi0**2 + C0) + ((A + Lphi0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
        z2mn = ((A + Lphi0**2 + C0) - ((A + Lphi0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
        if z2pl > 1.0:
            z2 = z2mn
        else:
            z2 = z2pl
            
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    
    # Step 4
    t, r, theta, phi, vel4 = *state[:4], state[4:]
    sint, cost = fix_sin(theta), fix_cos(theta)
    sinp, cosp = fix_sin(phi), fix_cos(phi)
    sph2cart = np.array([[1.0, 0.0,       0.0,         0.0    ],
                         [0.0, sint*cosp, r*cost*cosp, -r*sinp],
                         [0.0, sint*sinp, r*cost*sinp, r*cosp ],
                         [0.0, cost,      -r*sint,     0.0    ]])
    vel4cart = np.matmul(sph2cart, vel4)
    vel3cart = vel4cart[1:4]
    pos3cart = np.array([r*sint*cosp, r*sint*sinp, r*cost])
    L_vec = np.linalg.norm(np.cross(pos3cart, vel3cart))

    # Step 5
    E, Lphi = E0 + dE, Lphi0 + (dL_vec/L_vec)*(Lphi0)
    
    # Step 6
    C = z2*((a**2)*(1 - E**2) + (Lphi**2)/(1 - z2))

    
    # Step 7
    new_state = recalc_state([E, Lphi, C], state, mass, a)

    return new_state, [E, Lphi, C]
