# -*- coding: utf-8 -*-
"""
Metric Math stuff
"""

import numpy as np
from scipy import optimize
from scipy.signal import argrelmin
import scipy.interpolate as spi
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sympy as sp

def find_rmb(spin):
    '''
    Brief calculations for marginally bound orbits

    Parameters
    ----------
    spin : float
        Dimensionless spin constant of black hole, between -1 and 1 inclusive

    Returns
    -------
    r_mb: float
        Periapse of an equatorial orbit at the marginally bound orbit
    '''

    return (1 + np.sqrt(1 - spin))**2

def find_rms(spin):
    '''
    Brief calculations for marginally stable orbits (ISCO)

    Parameters
    ----------
    spin : float
        Dimensionless spin constant of black hole, between -1 and 1 inclusive

    Returns
    -------
    r_ms: float
        Radius of an equatorial orbit at the marginally stable bound orbit
    '''
    if spin >= 0.0:
        pro = 1.0
    else:
        pro = -1.0
        
    z1 = 1 + ((1 - spin**2)**(1/3))*(((1 + spin)**(1/3)) + ((1 - spin)**(1/3)))
    z2 = np.sqrt(3*(spin**2) + z1**2)
    r_ms = 3 + z2 - pro*np.sqrt((3 - z1)*(3 + z1 + 2*z2))
    return r_ms

def mink(state):
    '''
    mink function generates metric and christoffel symbols
    for the minkowski metric. Mostly for testing purposes

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment

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
    chris_tens = np.zeros((4,4,4))
    return (metric, chris_tens)

def schwarz(state):
    '''
    schwarz function generates metric and christoffel symbols
    for the schwarzschild metric.

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment

    Returns
    -------
    metric : 4x4 list
        Spacetime metric in terms of coordinate directions
    chris : 4x4x4 numpy array
        List of connection terms between coordinate directions

    '''
    r, theta = state[1], state[2]
    metric = np.array([[-(1-(2/r)),          0,                        0,    0],
                       [0,                   (1-((2)/r))**(-1),        0,    0],
                       [0,                   0,                        r**2, 0],
                       [0,                   0,                        0,    (r**2) * (np.sin(theta))**2]])
    chris_tens = np.zeros((4,4,4))
    chris_tens[0,0,1] = 1 / (r * (r - 2))
    chris_tens[0,1,0] = 1 / (r * (r - 2))
    chris_tens[1,0,0] = (1 / r**3) * (r - 2)
    chris_tens[1,1,1] = -1 / (r * (r - 2))
    chris_tens[1,2,2] = -(r - 2)
    chris_tens[1,3,3] = -(r - 2) * np.sin(theta)**2
    chris_tens[2,1,2] = 1/r
    chris_tens[2,2,1] = 1/r
    chris_tens[2,3,3] = -np.sin(theta) * np.cos(theta)
    chris_tens[3,1,3] = 1/r
    chris_tens[3,3,1] = 1/r
    chris_tens[3,2,3] = np.cos(theta) / np.sin(theta)
    chris_tens[3,3,2] = np.cos(theta) / np.sin(theta)
    return (metric, chris_tens)

def kerr(state, a):
    '''
    kerr function generates metric and christoffel symbols
    for the kerr metric. should be identical to schwarz for a=0

    Parameters
    ----------
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : float
        dimensionless spin constant of black hole, between -1 and 1 inclusive

    Returns
    -------
    metric : 4x4 numpy array
        Spacetime metric in terms of coordinate directions t, r, theta, phi
    chris : 4x4x4 numpy array
        List of connection terms between coordinate directions

    '''
    r, theta = state[1], state[2]
    sine, cosi = np.sin(theta), np.cos(theta)
    #various defined values that make math easier
    rho2, tri = r**2 + (a*cosi)**2, r**2 - 2*r + a**2
    al2, w = (rho2*tri)/(rho2*tri + 2*r*(a**2 + r**2)), (2*r*a)/(rho2*tri + 2*r*(a**2 + r**2))
    wu2 = ((rho2*tri + 2*r*(a**2 + r**2))/(rho2))*sine**2
    bigA = (r**2 + a**2)**2 - tri*(a*sine)**2
    metric = np.array([[-al2 + wu2*(w**2), 0.0,             0.0,    -w*wu2 ],
                       [0.0,               rho2/tri,        0.0,    0.0    ],
                       [0.0,               0.0,             rho2,   0.0    ],
                       [-w*wu2,            0.0,             0.0,    wu2    ]])
    chris_tens = np.zeros((4,4,4))
    chris_tens[0,0,1] = 2*(r**2 + a**2)*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2))
    chris_tens[0,1,0] = 2*(r**2 + a**2)*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2))
    chris_tens[0,0,2] = -2*(a**2)*r*sine*cosi/(rho2**2)
    chris_tens[0,2,0] = -2*(a**2)*r*sine*cosi/(rho2**2)
    chris_tens[0,1,3] = 2*a*(sine**2)*(((a*cosi)**2)*(a**2 - r**2) - (r**2)*(a**2 + 3*(r**2)))/(2*(rho2**2)*tri)
    chris_tens[0,3,1] = 2*a*(sine**2)*(((a*cosi)**2)*(a**2 - r**2) - (r**2)*(a**2 + 3*(r**2)))/(2*(rho2**2)*tri)
    chris_tens[0,2,3] = 2*r*cosi*((a*sine)**3)/(rho2**2)
    chris_tens[0,3,2] = 2*r*cosi*((a*sine)**3)/(rho2**2)
    chris_tens[1,0,0] = 2*tri*(r**2 - (a*cosi)**2)/(2*(rho2**3))
    chris_tens[1,0,3] = -tri*2*a*(sine**2)*(r**2 - (a*cosi)**2)/(2*(rho2**3))
    chris_tens[1,3,0] = -tri*2*a*(sine**2)*(r**2 - (a*cosi)**2)/(2*(rho2**3))
    chris_tens[1,1,1] = (2*r*((a*sine)**2) - 2*(r**2 - (a*cosi)**2))/(2*rho2*tri)
    chris_tens[1,1,2] = -(a**2)*sine*cosi/rho2
    chris_tens[1,2,1] = -(a**2)*sine*cosi/rho2
    chris_tens[1,2,2] = -r*tri/rho2
    chris_tens[1,3,3] = (tri*(sine**2)/(2*(rho2**3)))*(-2*r*(rho2**2) + 2*((a*sine)**2)*(r**2 - (a*cosi)**2))
    chris_tens[2,0,0] = -2*(a**2)*r*sine*cosi/(rho2**3)
    chris_tens[2,0,3] = 2*a*r*(r**2 + a**2)*sine*cosi/(rho2**3)
    chris_tens[2,3,0] = 2*a*r*(r**2 + a**2)*sine*cosi/(rho2**3)
    chris_tens[2,1,1] = (a**2)*sine*cosi/(rho2*tri)
    chris_tens[2,1,2] = r/rho2
    chris_tens[2,2,1] = r/rho2
    chris_tens[2,2,2] = -(a**2)*sine*cosi/rho2
    chris_tens[2,3,3] = -(sine*cosi/(rho2**3))*(bigA*rho2 + (r**2 + a**2)*2*r*((a*sine)**2))
    chris_tens[3,0,1] = 2*a*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2))
    chris_tens[3,1,0] = 2*a*(r**2 - (a*cosi)**2)/(2*tri*(rho2**2))
    chris_tens[3,0,2] = -2*a*r*(cosi/sine)/(rho2**2)
    chris_tens[3,2,0] = -2*a*r*(cosi/sine)/(rho2**2)
    chris_tens[3,1,3] = (2*r*(rho2**2) + 2*(((a**2)*sine*cosi)**2 - (r**2)*(rho2 + r**2 + a**2)))/(2*tri*(rho2**2))
    chris_tens[3,3,1] = (2*r*(rho2**2) + 2*(((a**2)*sine*cosi)**2 - (r**2)*(rho2 + r**2 + a**2)))/(2*tri*(rho2**2))
    chris_tens[3,2,3] = ((cosi/sine)/(rho2**2))*((rho2**2) + 2*r*((a*sine)**2))
    chris_tens[3,3,2] = ((cosi/sine)/(rho2**2))*((rho2**2) + 2*r*((a*sine)**2))
    return (metric, chris_tens)

def check_interval(solution, state, *args):
    '''
    Returns the spacetime interval for a state vector given a particular spacetime solution. 

    Parameters
    ----------
    solution : function
        One of the solution functions mink, schwarz, or kerr
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    *args : int/float
        args required for different solutions, depends on specific function.

    Returns
    -------
    interval : float
        spacetime interval. Returns -1 for velocities, -m^2 for 4-momenta
    '''
    metric, chris = solution(state, *args)
    interval = np.einsum("ij,i,j -> ", metric, state[4:], state[4:])
    return interval

def set_u_kerr(a, cons=False, velorient=False, vel4=False, params=False, pos=False):
    '''
    Creates and normalizes an input state vector according to kerr metric given a variety of inputs

    Parameters
    ----------
    a : float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    cons : 3-element list/array of floats
        energy, angular momentum, and carter constant per unit mass
    velorient : 3-element list/array of floats
        ratio of velocity/speed of light (beta), angle between r-hat and trajectory (eta - radians), angle between phi hat and trajectory (xi - radians)
    vel4 : 4-element list/array of floats
        tetrad component velocities [t, r, theta, phi]
    params : 3-element list/array of floats
        minimum of effective radial potential (GU distance), eccentricity, inclination (pi/2 as equatorial, negative values correspond to retrograde motion)
    pos : 4-element list/array of floats
        initial 4-position of particle
    Returns
    -------
    new : 8 element numpy array
        4-position and 4-velocity of the test particle at a particular moment
    cons : 3-element list/array of floats
        energy, angular momentum, and carter constant per unit mass
    '''
    if np.shape(cons) == (3,):
        #print("Calculating initial velocity from constants E,L,C")
        if np.shape(pos) == (4,):
            new = recalc_state(cons, pos, a)
        else:
            E, L, C = cons
            Rdco = [4*(E**2 - 1), 6, 2*((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C)]
            flats = np.roots(Rdco)
            flats = np.sort(flats.real[abs(flats.imag)<1e-14])
            pos = [0.0, flats[-1], np.pi/2, 0.0]
            new = recalc_state(cons, pos, a)
    elif (np.shape(velorient) == (3,)) and np.shape(pos) == (4,):
        #print("Calculating intial velocity from tetrad velocity and orientation")
        beta, eta, xi = velorient
        #eta is radial angle - 0 degrees is radially outwards, 90 degrees is in the phi direction
        #xi is up/down - 0 degrees is in the theta , 90 degrees is no up/down component
        eta, xi = eta*np.pi/180, xi*np.pi/180
        if (beta > 1):
            #print("Tetrad velocity exceeds c. Normalizing to 0.05")
            beta = 0.05
        gamma = 1/np.sqrt(1 - beta**2)
        r, theta = pos[1], pos[2]
            
        #various defined values that make math easier
        rho2 = r**2 + (a**2)*(np.cos(theta)**2)
        tri = r**2 - 2*r + a**2
        al2 = (rho2*tri)/(rho2*tri + 2*r*((a**2) + (r**2)))
        w = (2*r*a)/(rho2*tri + 2*r*((a**2) + (r**2)))
        wu2 = ((rho2*tri + 2*r*((a**2) + (r**2)))/rho2)*(np.sin(theta)**2)
        tetrad_matrix = np.array([[1/(np.sqrt(al2)), 0,                 0,               0],
                                  [0,                np.sqrt(tri/rho2), 0,               0],
                                  [0,                0,                 1/np.sqrt(rho2), 0],
                                  [w/np.sqrt(al2),   0,                 0,               1/np.sqrt(wu2)]])
        
        tilde = np.array([gamma, gamma*beta*np.cos(eta), -gamma*beta*np.sin(eta)*np.cos(xi), gamma*beta*np.sin(eta)*np.sin(xi)])
        new = np.matmul(tetrad_matrix, tilde)
        new = np.array([*pos, *new])
    elif (np.shape(vel4) == (4,)) and np.shape(pos) == (4,):
        #print("Calculating initial velocity from tetrad component velocities")
        r, theta = pos[1], pos[2]
        metric, chris = kerr(pos, a)
        #various defined values that make math easier
        rho2 = r**2 + (a**2)*(np.cos(theta)**2)
        tri = r**2 - 2*r + a**2
        al2 = (rho2*tri)/(rho2*tri + 2*r*((a**2) + (r**2)))
        w = (2*r*a)/(rho2*tri + 2*r*((a**2) + (r**2)))
        wu2 = ((rho2*tri + 2*r*((a**2) + (r**2)))/rho2)*(np.sin(theta)**2)
        tetrad_matrix = np.array([[1/(np.sqrt(al2)), 0,                 0,               0],
                                  [0,                np.sqrt(tri/rho2), 0,               0],
                                  [0,                0,                 1/np.sqrt(rho2), 0],
                                  [w/np.sqrt(al2),   0,                 0,               1/np.sqrt(wu2)]])
        rdot, thetadot, phidot = vel4[1]/vel4[0], vel4[2]/vel4[0], vel4[3]/vel4[0]
        vel_2 = (rdot**2 + thetadot**2 + phidot**2)
        beta = np.sqrt(vel_2)
        if beta > 1.0:
            #print("Tetrad velocity exceeds c, Normalizing to 0.05")
            rdot, thetadot, phidot = np.array([rdot, thetadot, phidot])*(0.05/beta)
            vel_2 = (rdot**2 + thetadot**2 + phidot**2)
            beta = np.sqrt(vel_2)
        gamma = 1/np.sqrt(1 - vel_2)
        #eta = np.arccos(np.sqrt((r * np.sin(theta) * phidot)**2)/beta)
        #xi = np.arccos(np.sqrt(rdot**2)/(beta*np.sin(eta)))
        #tilde = np.array([gamma, gamma*beta*np.cos(eta), -gamma*beta*np.sin(eta)*np.cos(xi), gamma*beta*np.sin(eta)*np.sin(xi)])
        tilde = np.array([gamma, gamma*rdot, gamma*thetadot, gamma*phidot])
        new = np.matmul(tetrad_matrix, tilde)
    elif np.shape(params) == (3,):
        #print("Calculating initial velocity from orbital parameters r0, e, i (WIP)")
        cons = schmidtparam3(*params, a)
        if cons == False:
            print("Non-viable parameters")
        if np.shape(pos) != (4,):
            pos = [0.0, params[0], np.pi/2, 0.0]
        new = recalc_state(cons, pos, a)
    else:
        print("Insufficent information provided, begone")
        new = np.array([0.0, 2000000.0, np.pi/2, 0.0, 7.088812050083354, -0.99, 0.0, 0.0])
    if cons == False:
        metric, chris = kerr(new, a)
        energy = -np.matmul(new[4:], np.matmul(metric, [1, 0, 0, 0]))        #initial energy
        lz = np.matmul(new[4:], np.matmul(metric, [0, 0, 0, 1]))         #initial angular momentum
        qarter = np.matmul(np.matmul(kill_tensor(new, a), new[4:]), new[4:])    #initial Carter constant Q
        cart = qarter - (a*energy - lz)**2                                          #initial adjusted Carter constant 
        cons = [energy, lz, cart]
    return new, cons

def schmidtparam3(r0, e, i, a):
    '''
    Returns characteristic constants of an orbit given observable parameters + spin
    Cannot generate unbound orbits (e >= 1.0)

    Parameters
    ----------
    r0 : int/float
        semi-major axis of the approximate Keplerian orbit
    e : int/float
        eccentricity of approximate Keplerian orbit, between 0 and 1
    i : int/float
        inclination of orbit w.r.t. angular momentum of black hole, where pi/2 is a prograde equatorial orbit, 0 is a northward polar orbit, and -pi/2 is a retrograde equatorial orbit
        parameter space extends in both directions s.t. any i + 2*n*pi is equivalent for any integer n
    a : int/float
        dimensionlesss spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    cons : 3-element list/array of floats
        energy, angular momentum, and carter constant per unit mass
    '''
    #sympy doesn't like tiny e?? idk what the issue is
    e = 0.0 if e < 1e-15 else e
    p = r0*(1 - (e**2))  #p is semi-latus rectum 
    rp, ra = p/(1 + e), p/(1 - e)
    polar = False
    j = i
    if i == 0.0 or i == np.pi:
        vals = np.transpose([[inc, *schmidtparam3(r0, e, inc, a)] for inc in np.linspace(np.pi/2, i, endpoint=False)])
        Efit, Lfit, Cfit = np.polyfit(vals[0], vals[1], 10), np.polyfit(vals[0], vals[2], 10), np.polyfit(vals[0], vals[3], 10)
        E, L, C = np.polyval(Efit, i), np.polyval(Lfit, i), np.polyval(Cfit, i)
        coeff = np.array([E**2 - 1.0, 2.0, (a**2)*(E**2 - 1.0) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
        coeff2 = np.polyder(coeff)
        turns = np.roots(coeff)
        flats = np.roots(coeff2)
        r02, e2 = (turns[0] + turns[1])/2.0, (turns[0] - turns[1])/(turns[0] + turns[1])
        r_err, e_err = ((r0 - r02)/r0)*100, ((e2 - e)/(2-e))*100
        r_err *= np.sign(r_err)
        e_err *= np.sign(e_err)
        return [E, 0.0, C]
    z = abs(np.cos(j))
    
    def rfuncs(r):
        tri = r**2 - 2*r + a**2
        f = r**4 + (a**2)*(r*(r + 2) + tri*(z**2))
        g = 2*a*r
        h = r*(r - 2) + tri*(z**2)/(max(1e-15, 1 - z**2)) #include a bias for divide by zero error
        d = (r**2 + (a*z)**2)*tri
        return f, g, h, d
    def r_funcs(r):
        f_ = 4*(r**3) + 2*(a**2)*((1 + z**2)*r + (1 - z**2))
        g_ = 2*a
        h_ = 2*(r - 1)/(1 - z**2)
        d_ = 2*(2*r - 3)*(r**2) + 2*(a**2)*((1 + z**2)*r - z**2)
        return f_, g_, h_, d_   
    
    if e == 0.0:
        f1, g1, h1, d1 = rfuncs(p)
        f2, g2, h2, d2 = r_funcs(p)
    else:
        f1, g1, h1, d1 = rfuncs(rp)
        f2, g2, h2, d2 = rfuncs(ra)
    
    def newC(E, L, a, z):
        return (z**2)*((a**2)*(1 - E**2) + (L**2)/(max(1e-15, 1 - z**2)))
    
    x, y = sp.symbols("x y", real=True)
    eq1 = sp.Eq(f1*(x**2) - 2*g1*x*y - h1*(y**2), d1)
    eq2 = sp.Eq(f2*(x**2) - 2*g2*x*y - h2*(y**2), d2)
    symsols = sp.solve([eq1, eq2])

    full_sols = []
    E, L, C = [np.sqrt(1-((1-e**2)/p)), (1 - z**2)*p, p*(z**2)]
    for thing in symsols:
        ene, lel = np.array([thing[x], thing[y]]).astype(float)
        if ene > 0.0 and ene < 1.0: 
            full_sols.append([ene, lel, newC(ene, lel, a, z)])
            E, L, C = [ene, lel, newC(ene, lel, a, z)]
            if np.prod(np.sign([E, L, C])) == np.sign(np.sin(j)):
                break
            else:
                E, L, C = (1 - (1 - e**2)/p)**0.55, ((1 - z**2)*p)**(0.5), p*(z**2)
                
    coeff, ro, count = [-1], 1, 0
    while np.polyval(coeff, ro) < -1e-12 and count < 20:
        E += 10**(-16)
        coeff = np.array([E**2 - 1.0, 2.0, (a**2)*(E**2 - 1.0) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
        coeff2 = np.polyder(coeff)
        turns = np.roots(coeff)
        flats = np.roots(coeff2)
        turns = turns.real[abs(turns.imag)<(1e-6)*r0]
        flats = flats.real[abs(flats.imag)<(1e-6)*r0]
        try:
            ro = max(flats)
        except:
            ro = np.max(np.roots(coeff2)).real

    turns = np.sort(turns)
    r02, e2 = (turns[-1] + turns[-2])/2.0, (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
    r_err, e_err = np.abs((r0 - r02)/r0)*100, np.abs((e2 - e)/(2-e))*100
    
    if r_err < 1e-6 and e_err < 1e-6:
        if (np.sqrt(L**2 + C) - np.abs(L))/np.abs(L) < 1e-15:
            C = 0.0
        return [E, L, C]
    else:
        def r__funcs(r):
            f__ = 12*(r**2) + 2*a*(1 + z**2)
            g__ = 0.0
            h__ = 2/(1 - z**2)
            d__ = 4*(r - 1)*(r + 3) + 2*(a**2)*(1 + z**2)
            return f__, g__, h__, d__  
        
        if False not in np.isreal([r_err, e_err]):
            f1, g1, h1, d1 = r_funcs(rp)
        else:
            f1, g1, h1, d1 = r__funcs(rp)

        f2, g2, h2, d2 = rfuncs(ra)
        
        eq1 = sp.Eq(f1*(x**2) - 2*g1*x*y - h1*(y**2), d1)
        eq2 = sp.Eq(f2*(x**2) - 2*g2*x*y - h2*(y**2), d2)

        symsols = sp.solve([eq1, eq2])
        
        full_sols = []
        for thing in symsols:
            ene, lel = np.array([thing[x], thing[y]]).astype(float)
            if ene > 0.0: 
                full_sols.append([ene, lel, newC(ene, lel, a, z)])
        
        for solution in full_sols:
            if (np.prod(np.sign(solution)) == np.sign(np.sin(j))):
                E, L, C = solution
                break     
        if polar == True:
            L, C = 0.0, C/(z**2) - (a**2)*(1 - E**2)
            
        if (np.sqrt(L**2 + C) - np.abs(L))/np.abs(L) < 1e-15:
            C = 0.0
        return [E, L, C]

def kill_tensor(state, a):
    '''
    kill_tensor function calculates killing Kerr killing tensor for a given system

    Parameters
    ----------
    state : 8 element list/numpy array
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    ktens : 4 element numpy array
        Describes symmetry in spacetime associated with the Carter Constant

    '''
    r, theta = state[1], state[2]
    metric, chris = kerr(state, a)
    rho2, tri = r**2 + (a*np.cos(theta))**2, r**2 - 2*r + a**2
    l_up = np.array([(r**2 + a**2)/tri, 1.0, 0.0, a/tri])
    n_up = np.array([(r**2 + a**2)/(2*rho2), -tri/(2*rho2), 0.0, a/(2*rho2)])
    l = np.matmul(metric, l_up)
    n = np.matmul(metric, n_up)
    l_n = np.outer(l, n)
    ktens = 2 * rho2 * 0.5*(l_n + np.transpose(l_n)) + (r**2) * np.array(metric)
    return ktens

def gr_diff_eq(solution, state, *args):
    '''
    gr_diff_eq function calculates the instantaneous proper time derivative for
    a given state in a given system

    Parameters
    ----------
    solution : function
        One of the GR solution functions (mink, schwarz, or kerr)
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    *args : int/float
        args required for different solutions, depends on specific function.
        generally mass and possibly spin.

    Returns
    -------
    d_state: 8 element numpy array
        4-velocity and 4-acceleration for the test particle at a particular moment

    '''

    d_state = np.zeros((8), dtype=float)                                         #create empty array to be the derivative of the state
    d_state[0:4] = state[4:]                                                      #derivative of position is velocity
    metric, chris = solution(state, *args)
    d_state[4:] = -np.einsum("ijk, j, k -> i", chris, state[4:], state[4:])
    return d_state                                                                #return derivative of state

#Butcher table for standard RK4 method
rk4 = {"label": "Standard RK4",
       "nodes": [1/2, 1/2, 1],
       "weights": [1/6, 1/3, 1/3, 1/6],
       "coeff": [[1/2], 
                 [0, 1/2],
                 [0, 0, 1]]}                                                    

#Butcher table for 5th order Cash-Karp method
ck5 = {"label": "Cash-Karp 5th Order",
       "nodes": [1/5, 3/10, 3/5, 1, 7/8],
       "weights": [37/378, 0, 250/621, 125/594, 0, 512/1771],
       "coeff": [[1/5],
                 [3/40, 9/40], 
                 [3/10, -9/10, 6/5], 
                 [-11/54, 5/2, -70/27, 35/27],
                 [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]]} 

#Butcher table for 4th order Cash-Karp method    
ck4 = {"label": "Cash-Karp 4th Order",
       "nodes": [1/5, 3/10, 3/5, 1, 7/8],
       "weights": [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4],
       "coeff": [[1/5],
                 [3/40, 9/40], 
                 [3/10, -9/10, 6/5], 
                 [-11/54, 5/2, -70/27, 35/27],
                 [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]]}     

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
        k.append(gr_diff_eq(solution, param, *args))                          #which is then used to find the next k value
    new_state = np.copy(state)
    for val in range(len(k)):                                                     #another for loop to add all the weights and find the final state
        new_state += k[val] * butcher["weights"][val] * dTau                        #can probably be simplified but this works for now
    return new_state

def recalc_state(constants, state, a):
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

def interpolate(data, time, supress=True):
    '''
    interpolates coordinate data to be evenly spaced in coordinate time
    Fail state plots time against index twice?

    Parameters
    ----------
    data : N x 3 numpy array of floats
        r, theta, and phi position of test particle
    time : N element numpy array of floats
        coordinate time of test particle
    suppress : bool, defaults to True
        limits the size of the final array to 10000 entries or ~20 samples per phase (assuming circular orbit), whichever is greater

    Returns
    -------
    new_data : M x 3 numpy array of floats
        r, theta, and phi position of test particle, interpolated to be evenly spaced along new_time
    new_time : M element numpy array of floats
        coordinate time of test particle, interpolated to be evenly spaced
        M is maximum of the length of the original time array or the integerized number of time units that have passed
    '''
    data = np.array(data)
    if supress == True:
        try:
            rad = data[argrelmin(data[:,0])[0][0], 0]
        except:
            rad = data[0,0]
        new_time = np.arange(time[0], time[-1], min(2*np.pi*np.sqrt(rad**3)/20, (time[-1] - time[0])/10000))
    else:
        #print("\n", time)
        #print(time[0])
        #print(time[-1])
        #print("hahshdh")
        #print(time[0], time[-1], max(len(time), int(time[-1] - time[0])))
        new_time = np.linspace(time[0], time[-1], max(len(time), 10*int(time[-1] - time[0])))
    try:
        r_poly = spi.CubicSpline(time, data[:,0])
        theta_poly = spi.CubicSpline(time, data[:,1])
        phi_poly = spi.CubicSpline(time, data[:,2])
        new_data = np.transpose(np.array([r_poly(new_time), theta_poly(new_time), phi_poly(new_time)]))
        return new_data, new_time
    except ValueError:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        ax1.plot(time)
        ax2.plot(time)
        return False

def sphr2quad(pos):
    '''
    Calculates quadrupole moment of a test particle

    Parameters
    ----------
    pos : 3-element numpy array of floats
        r, theta, and phi position of test particle

    Returns
    -------
    qmom : 3 x 3 numpy array of floats
        quadrupole moment of test particle
    '''
    x = pos[0] * np.sin(pos[1]) * np.cos(pos[2])
    y = pos[0] * np.sin(pos[1]) * np.sin(pos[2])
    z = pos[0] * np.cos(pos[1])
    qmom = np.array([[2*x*x - (y**2 + z**2), 3*x*y,                 3*x*z],
                     [3*y*x,                 2*y*y - (x**2 + z**2), 3*y*z],
                     [3*z*x,                 3*z*y,                 2*z*z - (x**2 + y**2)]], dtype=np.float64)
    return qmom

def matrix_derive(data, time, degree):
    '''
    Calculates degree-th time derivative of a series of 3x3 matrices, assuming they are interpolated across time

    Parameters
    ----------
    data : N x 3 x 3 numpy array of floats
        x, y, z quadrupole moment of test particle, assumed interpolated
    time : N element numpy array of floats
        coordinate time of test particle, assumed interpolated
    degree : int
        desired degree of the resulting derivative

    Returns
    -------
    new_data : N x 3 x 3 numpy array of floats
        degree-th derivative of quadrupole moment
    '''
    polys = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            polys[i][j] = spi.CubicSpline(time, data[:,i,j])
    new_data = np.transpose(np.array([[polys[0][0](time, degree), polys[0][1](time, degree), polys[0][2](time, degree)],
                                      [polys[1][0](time, degree), polys[1][1](time, degree), polys[1][2](time, degree)],
                                      [polys[2][0](time, degree), polys[2][1](time, degree), polys[2][2](time, degree)]]))
    return new_data

def matrix_derive2(data, old_time, time, degree):
    '''
    Calculates degree-th time derivative of a series of 3x3 matrices, assuming they are interpolated across time

    Parameters
    ----------
    data : N x 3 x 3 numpy array of floats
        x, y, z quadrupole moment of test particle, assumed interpolated
    time : N element numpy array of floats
        coordinate time of test particle, assumed interpolated
    degree : int
        desired degree of the resulting derivative

    Returns
    -------
    new_data : N x 3 x 3 numpy array of floats
        degree-th derivative of quadrupole moment
    '''
    polys = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            polys[i][j] = spi.CubicSpline(old_time, data[:,i,j])
    new_data = np.transpose(np.array([[polys[0][0](time, degree), polys[0][1](time, degree), polys[0][2](time, degree)],
                                      [polys[1][0](time, degree), polys[1][1](time, degree), polys[1][2](time, degree)],
                                      [polys[2][0](time, degree), polys[2][1](time, degree), polys[2][2](time, degree)]]))
    return new_data

def matrix_derive3(data, old_time, time):
    '''
    fhdjknslnfj
    Parameters
    ----------
    data : N x 3 x 3 numpy array of floats
        x, y, z quadrupole moment of test particle, assumed interpolated
    time : N element numpy array of floats
        coordinate time of test particle, assumed interpolated
    degree : int
        desired degree of the resulting derivative

    Returns
    -------
    new_data : N x 3 x 3 numpy array of floats
        degree-th derivative of quadrupole moment
    '''
    polysd2 = [[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]
    polysd3 = [[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]
    dt = np.mean(np.diff(time))
    for i in range(3):
        for j in range(3):
            u = spi.CubicSpline(old_time, data[:,i,j])
            polysd2[i][j] = (-u(time + 2*dt) + 16*u(time + dt) - 30*u(time) + 16*u(time - dt) - u(time - 2*dt))/(12*dt*dt)
            polysd3[i][j] = (-u(time + 3*dt) + 8*u(time + 2*dt) - 13*u(time + dt) + 13*u(time - dt) - 8*u(time - 2*dt) + u(time - 3*dt))/(8*dt*dt*dt)
    polysd2 = np.transpose(polysd2)
    polysd3 = np.transpose(polysd3)
    return polysd2, polysd3

def gwaves(quad_moment, time, distance):
    '''
    Calculates gravitational wave moment from quadrupole moment and distance

    Parameters
    ----------
    quad_moment : N x 3 x 3 numpy array of floats
        x, y, z quadrupole moment of test particle, assumed interpolated
    time : N element numpy array of floats
        coordinate time of test particle, assumed interpolated
    distance : float
        distance from GW source in geometric units

    Returns
    -------
    waves : N x 3 x 3 numpy array of floats
        GW moment over time; waves[:,0,0] is h+ polarization, waves[:,0,1]=waves[:,1,0] is hx polarization
    '''
    der_2 = matrix_derive(quad_moment, time, 2)
    waves = np.array([(2/distance) * entry for entry in der_2])
    return waves

def full_transform(data, distance, supress=True):    #defunctish??
    '''
    Calculates gravitational wave moment from orbit dictionary

    Parameters
    ----------
    data : 30 element dictionary
        full data package of an orbit given by clean_inspiral
    distance : float
        distance from GW source in geometric units

    Returns
    -------
    waves : N x 3 x 3 numpy array of floats
        GW moment in cartesian coords over interpolated time; waves[:,0,0] is h+ polarization, waves[:,0,1]=waves[:,1,0] is hx polarization
    int_time : N element numpy array of floats
        coordinate time of test particle, interpolated to be evenly spaced
        N is maximum of the length of the original time array or the integerized number of time units that have passed
    '''
    sphere, time = data["pos"], data["time"]
    int_sphere, int_time = interpolate(sphere, time, supress)
    quad = np.array([sphr2quad(pos) for pos in int_sphere])
    waves = gwaves(quad, int_time, distance)
    return waves, int_time

def trace_ortholize_old(pos_list):
    '''
    Calculates quadrupole moment in cartesian coords from position in spherical coords 

    Parameters
    ----------
    pos : N x 3 numpy array of floats
        full data package of an orbit given by clean_inspiral, assumed interpolated to be evenly spaced across time

    Returns
    -------
    qmom : N x 3 x 3 numpy array of floats
        quadrupole moment of test particle per unit mass
    '''
    x = pos_list[:,0] * np.sin(pos_list[:,1]) * np.cos(pos_list[:,2])
    y = pos_list[:,0] * np.sin(pos_list[:,1]) * np.sin(pos_list[:,2])
    z = pos_list[:,0] * np.cos(pos_list[:,1]) 
    
    qmom = np.transpose(np.array([[x*x, x*y, x*z],
                                  [y*x, y*y, y*z],
                                  [z*x, z*y, z*z]]))
    return qmom


def trace_ortholize(pos_list, a=None):
    '''
    Calculates quadrupole moment in cartesian coords from position in spherical coords 

    Parameters
    ----------
    pos : N x 3 numpy array of floats
        full data package of an orbit given by clean_inspiral, assumed interpolated to be evenly spaced across time

    Returns
    -------
    qmom : N x 3 x 3 numpy array of floats
        quadrupole moment of test particle per unit mass
    '''
    if a == None:
        print('old')
        return trace_ortholize_old(pos_list)
    x = np.sqrt(pos_list[:,0]**2 + a**2) * np.sin(pos_list[:,1]) * np.cos(pos_list[:,2])
    y = np.sqrt(pos_list[:,0]**2 + a**2) * np.sin(pos_list[:,1]) * np.sin(pos_list[:,2])
    z = pos_list[:,0] * np.cos(pos_list[:,1]) 
    
    qmom = np.transpose(np.array([[x*x, x*y, x*z],
                                  [y*x, y*y, y*z],
                                  [z*x, z*y, z*z]]))
    return qmom

def peters_integrate6(states, a, mu, ind1, ind2):
    '''
    Calculates change in characteristic orbital values from path of test particle through space 

    Parameters
    ----------
    states : N x 8 numpy array of floats
        list of state vectors - [4-position, 4-velocity] in geometric units
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    mu : float
        mass ratio of test particle to central body
    ind1 : int
        index value of the first entry in states relative to the master state list in clean_inspiral
    ind2 : int
        index value of the last entry in states relative to the master state list in clean_inspiral

    Returns
    -------
     4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    '''
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(int_sphere)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad - (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2 = matrix_derive(coolquad, int_time, 2)
        dt3 = matrix_derive(coolquad, int_time, 3)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*np.einsum('ijk,ijk ->i', dt3, dt3)
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
        #print(len(time), time[-1]-time[0], dE, np.linalg.norm([dLx, dLy, dLz]))
    return np.array([dE, dLx, dLy, dLz])
    #return quad

def peters_integrate6_2(states, a, mu, ind1, ind2):
    '''
    Calculates change in characteristic orbital values from path of test particle through space 

    Parameters
    ----------
    states : N x 8 numpy array of floats
        list of state vectors - [4-position, 4-velocity] in geometric units
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    mu : float
        mass ratio of test particle to central body
    ind1 : int
        index value of the first entry in states relative to the master state list in clean_inspiral
    ind2 : int
        index value of the last entry in states relative to the master state list in clean_inspiral

    Returns
    -------
     4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    '''
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(int_sphere)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = np.copy(quad)#3*quad - np.array([delta*r*r for r,th,ph in int_sphere])
        dt2 = matrix_derive(coolquad, int_time, 2)
        dt3 = matrix_derive(coolquad, int_time, 3)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
        #print(len(time), time[-1]-time[0], dE, np.linalg.norm([dLx, dLy, dLz]))
    return np.array([dE, dLx, dLy, dLz])
    #return quad

def peters_integrate6_3(states, a, mu, ind1, ind2):
    '''
    Calculates change in characteristic orbital values from path of test particle through space 

    Parameters
    ----------
    states : N x 8 numpy array of floats
        list of state vectors - [4-position, 4-velocity] in geometric units
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    mu : float
        mass ratio of test particle to central body
    ind1 : int
        index value of the first entry in states relative to the master state list in clean_inspiral
    ind2 : int
        index value of the last entry in states relative to the master state list in clean_inspiral

    Returns
    -------
     4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    '''
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(sphere)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad - (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2 = matrix_derive2(coolquad, time, int_time, 2)
        dt3 = matrix_derive2(coolquad, time, int_time, 3)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
        #print(len(time), time[-1]-time[0], dE, np.linalg.norm([dLx, dLy, dLz]))
    return np.array([dE, dLx, dLy, dLz])
    #return quad

def peters_integrate6_4(states, a, mu, ind1, ind2):
    '''
    Calculates change in characteristic orbital values from path of test particle through space 

    Parameters
    ----------
    states : N x 8 numpy array of floats
        list of state vectors - [4-position, 4-velocity] in geometric units
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    mu : float
        mass ratio of test particle to central body
    ind1 : int
        index value of the first entry in states relative to the master state list in clean_inspiral
    ind2 : int
        index value of the last entry in states relative to the master state list in clean_inspiral

    Returns
    -------
     4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    '''
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad - (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2, dt3 = matrix_derive3(coolquad, time, int_time)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu
        #print([dE, dLx, dLy, dLz])
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
        #print(len(time), time[-1]-time[0], dE, np.linalg.norm([dLx, dLy, dLz]))
    return np.array([dE, dLx, dLy, dLz])
    #return quad

def peters_integrate6_5(states, a, mu, ind1, ind2):
    '''
    Calculates change in characteristic orbital values from path of test particle through space 

    Parameters
    ----------
    states : N x 8 numpy array of floats
        list of state vectors - [4-position, 4-velocity] in geometric units
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    mu : float
        mass ratio of test particle to central body
    ind1 : int
        index value of the first entry in states relative to the master state list in clean_inspiral
    ind2 : int
        index value of the last entry in states relative to the master state list in clean_inspiral

    Returns
    -------
     4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    '''
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1 - 10) > 2:
        states = np.array(states)
        sphere, time = states[5:-5, 1:4], states[5:-5, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(sphere, a)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad - (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2, dt3 = matrix_derive3(coolquad, time, int_time)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu*(states[-1,0] - states[0,0])/(int_time[-1] - int_time[0])
            #scale both changes to make up for the bits that got cut off
        #print([dE, dLx, dLy, dLz])
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
        #print(len(time), time[-1]-time[0], dE, np.linalg.norm([dLx, dLy, dLz]))
        return np.array([dE, dLx, dLy, dLz])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0])
    #return quad

def peters_integrate7(states, a, mu, ind1, ind2):
    '''
    Calculates change in characteristic orbital values from path of test particle through space 

    Parameters
    ----------
    states : N x 8 numpy array of floats
        list of state vectors - [4-position, 4-velocity] in geometric units
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    mu : float
        mass ratio of test particle to central body
    ind1 : int
        index value of the first entry in states relative to the master state list in clean_inspiral
    ind2 : int
        index value of the last entry in states relative to the master state list in clean_inspiral

    Returns
    -------
     4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    '''
    path = np.array(states)
    #print(path)
    int_time = np.arange(int(path[0, 0]), int(path[-1, 0] + 1))
    #print(int_time)
    path = np.transpose([np.interp(int_time, path[:,0], path[:,1]),
                         np.interp(int_time, path[:,0], path[:,2]),
                         np.interp(int_time, path[:,0], path[:,3])])
    cartpath = np.transpose([path[:,0]*np.sin(path[:,1])*np.cos(path[:,2]), 
                             path[:,0]*np.sin(path[:,1])*np.sin(path[:,2]),
                             path[:,0]*np.cos(path[:,1])])
    quad = mu*np.einsum("ij, ik -> ijk", cartpath, cartpath)
    dt2 = matrix_derive(quad, int_time, 2)
    dt3 = matrix_derive(quad, int_time, 3)
    levciv = np.array([[[0, 0, 0],   #Levi-civita tensor - np.array([[[int(not((i+1)*(j+1)*(k+1)-6))*(int(j-i==1)*2-1) for k in range(3)] for j in range(3)] for i in range(3)])
                        [0, 0, 1],
                        [0, -1, 0]],
                       [[0, 0, -1],
                        [0, 0, 0],
                        [1, 0, 0]],
                       [[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 0]]])
    #dE = -(1/2)*np.sum(np.transpose([[dt3[:,i,j]**2 - (1/3)*dt3[:,i,i]*dt3[:,j,j] for j in range(3)] for i in range(3)]))
    #dL = np.transpose([[[[levciv[i,j,k]*dt2[:,j,m]*dt3[:,k,m] for m in range(3)] for k in range(3)] for j in range(3)] for i in range(3)])
    dE = (-1/5)*(np.einsum("ijk, ijk", dt3, dt3) - (1/3)*np.einsum("ijj, ikk", dt3, dt3))
    dL = -(2/5)*(np.einsum("ijk, ljm, lkm -> i", levciv, dt2, dt3))
    return np.array([dE, dL[0], dL[1], dL[2]])
    #return quad
    
def peters_integrate8(states, a, mu, ind1, ind2):
    '''
    Calculates change in characteristic orbital values from path of test particle through space 

    Parameters
    ----------
    states : N x 8 numpy array of floats
        list of state vectors - [4-position, 4-velocity] in geometric units
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    mu : float
        mass ratio of test particle to central body
    ind1 : int
        index value of the first entry in states relative to the master state list in clean_inspiral
    ind2 : int
        index value of the last entry in states relative to the master state list in clean_inspiral

    Returns
    -------
     4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    '''
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = mu*trace_ortholize(int_sphere)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad #- (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2 = matrix_derive(coolquad, int_time, 2)
        dt3 = matrix_derive(coolquad, int_time, 3)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*(np.einsum('ijk,ijk -> i', dt3, dt3) - (1/3)*np.einsum('ijj, ikk -> i', dt3, dt3))
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)
    return np.array([dE, dLx, dLy, dLz])

def mat_derv2(data, time, degree):
    new = np.copy(data)
    devtime = np.copy(time)
    for i in range(degree):
        new = np.diff(new, axis = 0)
        devtime = 0.5*(devtime[:-1] + devtime[1:])
    final = np.transpose([[np.interp(time, devtime, new[:,a,b]) for b in range(3)] for a in range(3)])
    #print(data)
    #print(final)
    #print(np.shape(data))
    #print(np.shape(final))
    return final

def peters_integrate9(states, a, mu, ind1, ind2):
    '''
    Calculates change in characteristic orbital values from path of test particle through space 

    Parameters
    ----------
    states : N x 8 numpy array of floats
        list of state vectors - [4-position, 4-velocity] in geometric units
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive
    mu : float
        mass ratio of test particle to central body
    ind1 : int
        index value of the first entry in states relative to the master state list in clean_inspiral
    ind2 : int
        index value of the last entry in states relative to the master state list in clean_inspiral

    Returns
    -------
     4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    '''
    #dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time, False)
        div = np.mean(np.diff(int_time))
        quad = trace_ortholize(int_sphere)
        delta = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        coolquad = quad - (1/3)*np.einsum('i, jk -> ijk', np.einsum('ijj -> i', quad), delta)
        dt2 = mat_derv2(coolquad, int_time, 2)
        dt3 = mat_derv2(coolquad, int_time, 3)
        levciv = np.array([[[0, 0, 0],   #Levi-civita tensor
                            [0, 0, 1],
                            [0, -1, 0]],
                           [[0, 0, -1],
                            [0, 0, 0],
                            [1, 0, 0]],
                           [[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 0]]])
        dedt = (-1/5)*np.einsum('ijk,ijk ->i', dt3, dt3)
        dldt = (-2/5)*np.einsum("ijk, ljm, lkm -> li", levciv, dt2, dt3)
        dE = np.sum(dedt*div)*mu
        dLx, dLy, dLz = np.sum(dldt*div, axis=0)*mu
        #print(dE, np.sqrt(dLx**2 + dLy**2 + dLz**2))
    return np.array([dE, dLx, dLy, dLz])
    #return quad

def new_recalc_state5(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
            
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    #dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    dC = 2*z2*(L0*dLz/(1-z2) - (a**2)*E0*dE)  
    if np.isnan(dC):
        dC = -2*z2*(a**2)*E0*dE 

    # Step 4
    E, L = E0 + dE, L0 + dLz*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
    
    # Step 5
    C = C0 + dC
    #print(dE, dLz, dC)
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    test = max(np.roots(np.polyder(potent)))
    count = 0
    #while (np.polyval(potent, test) < 0.0):
    #    count += 1
    #    dR = -np.polyval(potent, test)
    #    E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
    #    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    #    test = max(np.roots(np.polyder(potent)))
    #print(count)
    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state6(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
            
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    #dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    dC = 2*z2*(L0*dLz/(1-z2) - (a**2)*E0*dE)  
    if np.isnan(dC):
        dC = -2*z2*(a**2)*E0*dE 

    # Step 4
    E, L = E0 + dE, L0 + dLz*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
    
    # Step 5
    C = C0 + dC
    #print(dE, dLz, dC)
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    test = max(np.roots(np.polyder(potent)))
    count = 0
    #while (np.polyval(potent, test) < 0.0):
    #    count += 1
    #    dR = -np.polyval(potent, test)
    #    E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
    #    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    #    test = max(np.roots(np.polyder(potent)))
    #print(count)
    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state7(cons, con_derv, state, a, loop=0):
    metric, chris = kerr(state, a)
    E0 = -np.matmul(metric, state[4:])[0]
    r, theta, phi = state[1:4]
    sint, cost, sinp, cosp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    rho2, tri = r**2 + (a**2)*(cost**2), r**2 - 2*r + a**2
    al2 = (rho2*tri)/(rho2*tri + 2*r*(a**2 + r**2))
    w = (2*r*a)/(rho2*tri + 2*r*(a**2 + r**2))
    wbar2 = ((rho2*tri + 2*r*(a**2 + r**2))/rho2)*(sint**2)
    tet2kerr = np.array([[1/np.sqrt(al2), 0.0,               0.0,             0.0],
                         [0.0,            np.sqrt(tri/rho2), 0.0,             0.0],
                         [0.0,            0.0,               1/np.sqrt(rho2), 0.0],
                         [w/np.sqrt(al2), 0.0,               0.0,             1/np.sqrt(wbar2)]])
    tetrad = np.linalg.solve(tet2kerr, state[4:])
    tetcart = np.array([*tetrad[:2], tetrad[3], -tetrad[2]])
    vel, cartpos = tetcart[1:]/tetcart[0], np.array([r*sint*cosp, r*sint*sinp, r*cost])
    L0 = np.cross(cartpos, vel)
    A, eps = np.zeros((4,3)), np.max(abs(con_derv))
    #print("---")
    #print(con_derv)
    #print(eps)
    if eps == 0.0:
        eps = 1e-7
    #print(eps)
    
    def getNewCons(j):
        intvel = np.array([0.0,0.0,0.0])
        intvel[j] += eps
        intL, gamma = np.cross(cartpos, intvel+vel), 1/np.sqrt(1 - np.linalg.norm(intvel + vel)**2)
        inttetrad = gamma*np.array([1, intvel[0], -intvel[2], intvel[1]])
        intkerr = np.matmul(tet2kerr, inttetrad)
        intE = -np.matmul(metric, intkerr)[0]
        return np.array([intE - E0, *(intL - L0)])/eps
    
    A[:,0], A[:,1], A[:,2] = getNewCons(0), getNewCons(1), getNewCons(2)
    try:
        #print("org")
        bigD = np.linalg.inv(np.matmul(np.transpose(A), A))
        dvel = np.matmul(bigD, np.matmul(np.transpose(A), con_derv[:4]))
    except:
        #print(A)
        #print(np.matmul(np.transpose(A), A))
        dvel = np.linalg.solve(np.matmul(np.transpose(A), A), np.matmul(np.transpose(A), con_derv[:4]))
    #print(dvel)
    newvel = vel + dvel
    gamma = 1/np.sqrt(1 - np.linalg.norm(newvel)**2)
    newtetrad = gamma*np.array([1, newvel[0], -newvel[2], newvel[1]])
    newkerr = np.matmul(tet2kerr, newtetrad)
    holdstate = np.array([*state[0:4], *newkerr])
    newE = -np.matmul(metric, newkerr)[0]        #initial energy
    newLz = np.matmul(metric, newkerr)[3]        #initial angular momentum
    newQ = np.matmul(np.matmul(kill_tensor(holdstate, a), newkerr), newkerr)
    newC = newQ - (a*newE - newLz)**2  
    #print(con_derv)
    #print([newE - E0, *(np.cross(cartpos, newvel) - L0)])
    #print(con_derv)
    #print(con_derv - np.array([newE - E0, *(np.cross(cartpos, newvel) - L0)]))
    #print(np.linalg.norm(con_derv - np.array([newE - E0, *(np.cross(cartpos, newvel) - L0)])))
    #print([newE, newLz, newC])
    #print("-----")
    #print(np.linalg.norm(con_derv - np.array([newE - E0, *(np.cross(cartpos, newvel) - L0)])))
    #if loop < 10 and np.linalg.norm(con_derv - np.array([newE - E0, *(np.cross(cartpos, newvel) - L0)])) >= 1e-15:
        #print("HELLO WAIT WHAT")
        #print(np.linalg.norm(con_derv - np.array([newE - E0, *(np.cross(cartpos, newvel) - L0)])))
        #holdstate, [newE, newLz, newC] = new_recalc_state7([newE, newLz, newC], np.array([newE - E0, *(np.cross(cartpos, newvel) - L0)]), holdstate, a, loop=loop+1)
    return holdstate, [newE, newLz, newC]

def new_recalc_state8(cons, con_derv, state, a):
    metric, chris = kerr(state, a)
    E0 = -np.matmul(metric, state[4:])[0]
    
    r, theta, phi = state[1:4]
    sint, cost, sinp, cosp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    rho2, tri = r**2 + (a**2)*(cost**2), r**2 - 2*r + a**2
    al2 = (rho2*tri)/(rho2*tri + 2*r*(a**2 + r**2))
    w = (2*r*a)/(rho2*tri + 2*r*(a**2 + r**2))
    wbar2 = ((rho2*tri + 2*r*(a**2 + r**2))/rho2)*(sint**2)
    tet2kerr = np.array([[1/np.sqrt(al2), 0.0,               0.0,             0.0],
                         [0.0,            np.sqrt(tri/rho2), 0.0,             0.0],
                         [0.0,            0.0,               1/np.sqrt(rho2), 0.0],
                         [w/np.sqrt(al2), 0.0,               0.0,             1/np.sqrt(wbar2)]])
    
    tetrad = np.linalg.solve(tet2kerr, state[4:])
    tetcart = np.array([*tetrad[:2], tetrad[3], -tetrad[2]])
    vel, cartpos = tetcart[1:]/tetcart[0], np.array([r*sint*cosp, r*sint*sinp, r*cost])
    hold = con_derv
    L0 = np.cross(cartpos, vel)
    A = np.zeros((4,3))
    loop = -1
    #print(con_derv)
    sol_list = [vel]
    hold_derv = con_derv
    target = np.array([E0, *L0]) + con_derv
    thing = 1e10
    tog = 0
    #print(target)
    while np.linalg.norm(hold) > 6e-15:
        thing = np.linalg.norm(hold)
        eps = 1e-8 #np.sqrt(abs(max(hold)))

        def getNewCons(j):
            intvel = np.array([0.0,0.0,0.0])
            intvel[j] += eps
            intL, gamma = np.cross(cartpos, intvel+vel), 1/np.sqrt(1 - np.linalg.norm(intvel + vel)**2)
            inttetrad = gamma*np.array([1, intvel[0], -intvel[2], intvel[1]])
            intkerr = np.matmul(tet2kerr, inttetrad)
            intE = -np.matmul(metric, intkerr)[0]
            return np.array([intE - E0, *(intL - L0)])/eps
    
        A[:,0], A[:,1], A[:,2] = getNewCons(0), getNewCons(1), getNewCons(2)
        dvel = np.linalg.solve(np.matmul(np.transpose(A), A), np.matmul(np.transpose(A), hold_derv[:4]))
        #print(dvel)
        vel = vel + dvel + tog*np.random.randn(3)*1e-17
        sol_list.append(vel)
        L1 = np.cross(cartpos, vel)
        gamma = 1/np.sqrt(1 - np.linalg.norm(vel)**2)
        newtetrad = gamma*np.array([1, vel[0], -vel[2], vel[1]])
        newkerr = np.matmul(tet2kerr, newtetrad)
        E1 = -np.matmul(metric, newkerr)[0]
        hold = (np.array([E1 - E0, *(L1 - L0)]) - con_derv)
        loop += 1
        #print(hold, np.linalg.norm(hold))
        hold_derv = target - np.array([E1, *L1])
        #if np.linalg.norm(hold) > thing:
        #    tog = 1
            #print("KACHOW")
        #else:
        #    tog = 0
        if loop > 100:
            #print("oop!")
            #print(con_derv)
            #print(np.array([E1 - E0, *(L1 - L0)]))
            #print(hold)
            break
    #print(np.array([E1 - E0, *(L1 - L0)]))
    #print(100*(np.array([E1 - E0, *(L1 - L0)]) - con_derv)/con_derv)
    newstate = np.array([*state[0:4], *newkerr])
    newLz = np.matmul(metric, newkerr)[3]        #initial angular momentum
    newQ = np.matmul(np.matmul(kill_tensor(newstate, a), newkerr), newkerr)
    newC = newQ - (a*E1 - newLz)**2 
    return newstate, [E1, newLz, newC]

def new_recalc_state9(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    theta = state[2]
    fmom = np.matmul(metric, state[4:])
    L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
    cosz, sinz = np.sqrt(min(1.0, np.abs(z2))), np.sqrt(1 - min(1.0, np.abs(z2)))
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    #dC = 2*z2*(L0*dLz/(1-z2) - (a**2)*E0*dE)  
    #if np.isnan(dC):
    #    dC = -2*z2*(a**2)*E0*dE 
    dC = 2*(L*dL_vec - L0*dLz - ((a*np.cos(state[2]))**2)*(1 - E0**2)*dE) if C0 != 0 else 0.0
        #From glamp A3, thetadot term goes away because I don't change position!

    #dC = 2*(L*dL_vec - L0*dLz - (a**2)*cosz*(sinz*state[6]*(1 - E0**2) + cosz*E0*dE))
    #dC = 2*(L0 - a*E0*np.sin(theta)**2)*(dLz - a*dE*np.sin(theta)**2 - 2*a*E0*np.sin(theta)*np.cos(theta)*state[6])*np.sin(theta)**2
    #dC -= 2*np.sin(theta)*np.cos(theta)*(L0 - a*E0*np.sin(theta)**2)*state[6]
    #dC /= np.sin(theta)**4
    #dC -=2*(a**2)*np.cos(theta)*np.sin(theta)*state[6]
    #dC += 2*(state[1]**2 + (a*np.cos(theta))**2)*(2*state[1]*state[5] - 2*(a**2)*np.sin(theta)*np.cos(theta)*state[6])*(state[6]**2)

    # Step 4
    E, L = E0 + dE, L0 + dLz    #*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
                                        #I actually don't think I need to make that correction
    
    # Step 5
    C = C0 + dC
    
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    test = max(np.roots(np.polyder(potent)))
    count = 0
    #Make sure point we're AT is viable, the rest is maybe lame?
    '''
    while (np.polyval(potent, state[1]) < 0.0):
        dR = -np.polyval(potent, test)
        #scale all the variables
        #E -= dE*(1e-5)
        L += dLz*(1e-5)
        C += dC*(1e-5)
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
        count += 1
        if count == 1000:
            print("gotdamn")
        if count >= 1e6:
            break
    if count > 0:
        print("L, C adjust: %s, %s (%s times)"%( count*dLz*1e-5, count*dC*1e-5, count))
    '''
    # Step 6
    #print(E, L, C)
    #print(dE, dLz*np.sign(L0), dC)
    #print(cosz, sinz, np.abs(z2))
    #print(E0, L0, C0)
    #print(E, L, C)
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state9a(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    theta = state[2]
    thedot = state[6]
    #thedotdot = -((thedot/np.sqrt(1 - theta**2))**2)*theta  #an approximation, check that
    fmom = np.matmul(metric, state[4:])
    L = np.sqrt(fmom[2]**2 + (fmom[3]**2)/(np.sin(theta)**2))
    Lx, Ly = fmom[2]*np.sin(state[3]), -L*np.cos(state[3])
    # Step 2
    if a == 0:
        z2 = C0/(L0**2 + C0)
    else:
        A = (a**2)*(1 - E0**2)
        z2 = ((A + L0**2 + C0) - ((A + L0**2 + C0)**2 - 4*A*C0)**(1/2))/(2*A)
    cosz, sinz = np.sqrt(np.abs(z2)), np.sqrt(1 - np.abs(z2))
    sint, cost = np.sin(theta), np.cos(theta)
    # Step 3
    dE, dLx, dLy, dLz = con_derv[:4]
    dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    def get_Ls(state):
        x, y, z = state[1]*np.sin(state[2])*np.cos(state[3]), state[1]*np.sin(state[2])*np.sin(state[3]), state[1]*np.cos(state[2])
        vx = state[5]*np.sin(state[2])*np.cos(state[3]) + state[1]*state[6]*np.cos(state[2])*np.cos(state[3]) - state[1]*state[7]*np.sin(state[2])*np.sin(state[3]) 
        vy = state[5]*np.sin(state[2])*np.sin(state[3]) + state[1]*state[6]*np.cos(state[2])*np.sin(state[3]) + state[1]*state[7]*np.sin(state[2])*np.cos(state[3]) 
        vz = state[5]*np.cos(state[2]) - state[1]*state[6]*np.sin(state[2])
        return np.cross([x,y,z], [vx, vy, vz])
    Lx, Ly, Lz = get_Ls(state)
    #print("WHAT IS HAPPENING")
    #print(dLx, dLy, dLz)
    #print(Lx, Ly, L0)
    #print(np.sqrt(Lx**2 + Ly**2 + L0**2))
    #print(L)
    #print(Lx**2 + Ly**2)
    #print(C0)
    #dC = 2*(L*dL_vec - L0*dLz - (a**2)*cosz*(sinz*state[6]*(1 - E0**2) + cosz*E0*dE))
    #dC = 2*L*dL_vec - 2*L0*dLz - 2*(a**2)*(thedot*np.sin(theta)*np.cos(theta)*(1 - E0**2) + (np.cos(theta)**2)*E0*dE)
    dC = 2*Lx*dLx + 2*Ly*dLy - 2*(a**2)*(thedot*np.sin(theta)*np.cos(theta)*(1 - E0**2) + (np.cos(theta)**2)*E0*dE)
    #print(2*Lx*dLx + 2*Ly*dLy)
    #print(dC)
    #sig = state[1]**2 + (a*cost)**2
    #sig_dot = 2*state[1]*state[5] - 2*(a**2)*thedot*sint*cost
    #dQ = 2*(L0 - a*E0*(sint**2))*(dLz - a*(dE*(sint**2) + 2*E0*thedot*sint*cost))/(sint**2)
    #dQ += -((L0 - a*E0*(sint**2))**2)*(2*cost/(sint**3))*thedot
    #dQ += -2*(a**2)*thedot*sint*cost + 2*sig*sig_dot*(thedot**2) #+ 2*(sig**2)*thedot*thedotdot second derivative, real small
    #dC = dQ - 2*(a*E0 - L0)*(a*dE - dLz)

    # Step 4
    E, L = E0 + dE, L0 + dLz*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
    
    # Step 5
    C = C0 + dC
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    test = max(np.roots(np.polyder(potent)))
    '''
    while (np.polyval(potent, test) < 0.0):
        #print("WEIRD THING:", np.polyval(potent, test))
        dR = -np.polyval(potent, test)
        E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
        #C -= 0.0001*C
        #L -= 0.01*L
        #potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        #test = max(np.roots(np.polyder(potent)))
    '''
    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    #print(E0 - E)
    #print(dE)
    #print(L0 - L)
    #print(dLz)
    #print(C0 - C)
    #print(dC)
    return new_state, [E, L, C]

def new_recalc_state10(cons, con_derv, state, a):
    metric, chris = kerr(state, a)
    E0 = -np.matmul(metric, state[4:])[0]
    r, theta, phi = state[1:4]
    sint, cost, sinp, cosp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    rho2, tri = r**2 + (a**2)*(cost**2), r**2 - 2*r + a**2
    al2 = (rho2*tri)/(rho2*tri + 2*r*(a**2 + r**2))
    w = (2*r*a)/(rho2*tri + 2*r*(a**2 + r**2))
    wbar2 = ((rho2*tri + 2*r*(a**2 + r**2))/rho2)*(sint**2)
    tet2kerr = np.array([[1/np.sqrt(al2), 0.0,               0.0,             0.0],
                         [0.0,            np.sqrt(tri/rho2), 0.0,             0.0],
                         [0.0,            0.0,               1/np.sqrt(rho2), 0.0],
                         [w/np.sqrt(al2), 0.0,               0.0,             1/np.sqrt(wbar2)]])
    tetrad = np.linalg.solve(tet2kerr, state[4:])
    tetcart = np.array([*tetrad[:2], tetrad[3], -tetrad[2]])
    vel, cartpos = tetcart[1:]/tetcart[0], np.array([r*sint*cosp, r*sint*sinp, r*cost])
    L0 = np.cross(cartpos, vel)
    A, eps = np.zeros((4,3)), 1e-7
    loop, err, target = 0, 100, np.array([E0, *L0]) + con_derv
    #print(con_derv)
    #print(np.linalg.norm(con_derv)**2)
    #print(np.array([E0, *L0]))
    diff = con_derv
    
    def getNewCons(j, vel, eps):
        intvel = np.array([0.0,0.0,0.0])
        intvel[j] += eps
        intL, gamma = np.cross(cartpos, intvel+vel), 1/np.sqrt(1 - np.linalg.norm(intvel + vel)**2)
        inttetrad = gamma*np.array([1, intvel[0], -intvel[2], intvel[1]])
        intkerr = np.matmul(tet2kerr, inttetrad)
        intE = -np.matmul(metric, intkerr)[0]
        return np.array([intE - E0, *(intL - L0)])/eps
    
    while err > 1e-5 and loop < 100:
        A[:,0], A[:,1], A[:,2] = getNewCons(0, vel, eps), getNewCons(1, vel, eps), getNewCons(2, vel, eps)
        dvel = np.linalg.solve(np.matmul(np.transpose(A), A), np.matmul(np.transpose(A), diff))
        vel = vel + dvel
        gamma, newL = 1/np.sqrt(1 - np.linalg.norm(vel)**2), np.cross(cartpos, vel)
        newtetrad = gamma*np.array([1, vel[0], -vel[2], vel[1]])
        newkerr = np.matmul(tet2kerr, newtetrad)
        newE = -np.matmul(metric, newkerr)[0]        #initial energy
        err = 100*np.linalg.norm((np.array([newE, *newL]) - target)/target)
        #print(np.array([newE, *newL]), err)
        loop += 1
        diff = target - np.array([newE, *newL])
    #print("___")
    #print(target)
    #print(np.array([newE, *newL]))
    #print(np.linalg.norm(con_derv)**2)
    holdstate = np.array([*state[0:4], *newkerr])
    newLz = np.matmul(metric, newkerr)[3]        #initial angular momentum
    newQ = np.matmul(np.matmul(kill_tensor(holdstate, a), newkerr), newkerr)
    newC = newQ - (a*newE - newLz)**2  

    return holdstate, [newE, newLz, newC]

def new_recalc_state11(cons, con_derv, state, a, mu, path):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons

    cosi = L0/np.sqrt(L0**2 + C0)
    path = np.array(path)
    e = 1 - min(path[:,1])/max(path[:,1]) if path[0,5] < 0 else max(path[:,1])/min(path[:,1]) - 1
    r0 = max(path[:,1]) if path[0,5] < 0 else min(path[:,1])
    p = r0*(1 - e**2)
    R = lambda r: (E0**2 - 1.0)*(r**4) + 2.0*(r**3) + ((a**2)*(E0**2 - 1.0) - L0**2 - C0)*(r**2) + 2*((a*E0 - L0)**2 + C0)*r - C0*(a**2)
    turns = optimize.fsolve(R, [(a**2)*C0, (0.3*(a**2)*C0 + 0.7*p/(1 + e)), p/(1 + e), p/(1 - e)])
    e = (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
    p = np.sqrt(turns[-1]*turns[-2]*(1 - e**2))
    f1 = lambda x: 1 + (73/24)*(x**2) + (37/96)*(x**4)
    f2 = lambda x: 73/12 + (823/24)*(x**2) + (949/32)*(x**4) + (491/192)*(x**6)
    f3 = lambda x: 1 + (7/8)*(x**2)
    f4 = lambda x: 61/24 + (63/8)*(x**2) + (94/64)*(x**4)
    f5 = lambda x: 61/8 + (91/4)*(x**2) + (461/64)*(x**4)
    f6 = lambda x: 97/12 + (37/2)*(x**2) + (211/32)*(x**4)
    
    r0 = p/(1 - e**2)
    
    dEdt = ((-32/5)*(mu**2)*(p**(-5))*((1 - e**2)**(3/2))*(f1(e) - a*(p**(-3/2))*cosi*f2(e)))
    dLdt = ((-32/5)*(mu**2)*(p**(-7/2))*((1 - e**2)**(3/2))*(cosi*f3(e) + a*(p**(-3/2))*(f4(e) - (cosi**2)*f5(e))))
    dQdt = ((-64/5)*(mu**3)*(p**(-3))*((1 - e**2)**(3/2))*(f3(e) - a*(p**(-3/2))*cosi*f6(e)))
    dCdt = dQdt - 2*L0*dLdt
    dt = path[-1,0] - path[0,0]
    #print(dQ, dC)
    #print(dC, dC + 2*L0*dL, dC - 2*L0*dL)
    
    #print(path[-1,0] - path[0,0])
    #print("no")
    #print(r0*(1-e), r0*(1+e), r0, e)
    E, L, C = E0 + dEdt*dt, L0 + dLdt*dt, C0 + 0*dCdt*dt
    #print("HEY", E0, L0, C0)
    #print("HEY", E, L, C)
    #print(E)
    '''
    R2 = lambda r: (E**2 - 1.0)*(r**4) + 2.0*(r**3) + ((a**2)*(E**2 - 1.0) - L**2 - C)*(r**2) + 2*((a*E - L)**2 + C)*r - C*(a**2)
    test = optimize.fsolve(R2, turns)
    while R2(max(test)) < 0.0:
        dR = -np.polyval(potent, test)
        E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
    '''
    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state12(cons, con_derv, state, a, mu, path):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    
    roots = root_getter(E0, L0, C0, a)[0]
    e = (roots[-1] - roots[-2])/(roots[-1] + roots[-2])
    p = 0.5*(roots[-1] + roots[-2])*(1 - e**2)
    cosi = L0/np.sqrt(L0**2 + C0)

    f1 = lambda x: 1 + (73/24)*(x**2) + (37/96)*(x**4)
    f2 = lambda x: 73/12 + (823/24)*(x**2) + (949/32)*(x**4) + (491/192)*(x**6)
    f3 = lambda x: 1 + (7/8)*(x**2)
    f4 = lambda x: 61/24 + (63/8)*(x**2) + (94/64)*(x**4)
    f5 = lambda x: 61/8 + (91/4)*(x**2) + (461/64)*(x**4)
    f6 = lambda x: 97/12 + (37/2)*(x**2) + (211/32)*(x**4)
    
    dEdt = ((-32/5)*(mu**2)*(p**(-5))*((1 - e**2)**(3/2))*(f1(e) - a*(p**(-3/2))*cosi*f2(e)))
    dLdt = ((-32/5)*(mu**2)*(p**(-7/2))*((1 - e**2)**(3/2))*(cosi*f3(e) + a*(p**(-3/2))*(f4(e) - (cosi**2)*f5(e))))
    dQdt = ((-64/5)*(mu**3)*(p**(-3))*((1 - e**2)**(3/2))*(f3(e) - a*(p**(-3/2))*cosi*f6(e)))
    dCdt = dQdt - 2*L0*dLdt
    dt = path[-1][0] - path[0][0]
    #print(dQ, dC)
    #print(dC, dC + 2*L0*dL, dC - 2*L0*dL)
    
    #print(path[-1,0] - path[0,0])
    #print("no")
    #print(r0*(1-e), r0*(1+e), r0, e)

    E, L, C = E0 + dEdt*dt/mu, L0 + dLdt*dt/mu, C0 + 0*dCdt*dt/mu
    #print("HEY", E0, L0, C0)
    #print("HEY", E, L, C)
    #print(E)
    '''
    R2 = lambda r: (E**2 - 1.0)*(r**4) + 2.0*(r**3) + ((a**2)*(E**2 - 1.0) - L**2 - C)*(r**2) + 2*((a*E - L)**2 + C)*r - C*(a**2)
    test = optimize.fsolve(R2, turns)
    while R2(max(test)) < 0.0:
        dR = -np.polyval(potent, test)
        E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
    '''
    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state13(cons, con_derv, state, a, mu, path):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    
    roots = root_getter(E0, L0, C0, a)[0]
    e = (roots[-1] - roots[-2])/(roots[-1] + roots[-2])
    p = 0.5*(roots[-1] + roots[-2])*(1 - e**2)
    cosi = L0/np.sqrt(L0**2 + C0)

    f1 = lambda x: 1 + (73/24)*(x**2) + (37/96)*(x**4)
    f2 = lambda x: 73/12 + (823/24)*(x**2) + (949/32)*(x**4) + (491/192)*(x**6)
    f3 = lambda x: 1 + (7/8)*(x**2)
    f4 = lambda x: 61/24 + (63/8)*(x**2) + (94/64)*(x**4)
    f5 = lambda x: 61/8 + (91/4)*(x**2) + (461/64)*(x**4)
    f6 = lambda x: 97/12 + (37/2)*(x**2) + (211/32)*(x**4)
    
    dEdt = ((-32/5)*(mu**2)*(p**(-5))*((1 - e**2)**(3/2))*(f1(e) - a*(p**(-3/2))*cosi*f2(e)))
    dLdt = ((-32/5)*(mu**2)*(p**(-7/2))*((1 - e**2)**(3/2))*(cosi*f3(e) + a*(p**(-3/2))*(f4(e) - (cosi**2)*f5(e))))
    #dQdt = ((-64/5)*(mu**3)*(p**(-3))*((1 - e**2)**(3/2))*(f3(e) - a*(p**(-3/2))*cosi*f6(e)))
    #dCdt = dQdt - 2*L0*dLdt
    dCdt = 2*(C0/L0)*dLdt
    dt = path[-1][0] - path[0][0]
    #print(path[0], "PATH")
    #print(dQ, dC)
    #print(dC, dC + 2*L0*dL, dC - 2*L0*dL)
    
    #print(path[-1,0] - path[0,0])
    #print("no")
    #print(r0*(1-e), r0*(1+e), r0, e)
    yomp = 1
    if path[0][0] == 0.0:
        yomp = 0

    #print(dEdt, dLdt, dQdt, yomp*dCdt, L0, mu*np.sqrt(p), "AUGH")
    E, L, C = E0 + dEdt*dt/mu, L0 + dLdt*dt/mu, C0 + dCdt*dt/mu
    #print("HEY", E0, L0, C0)
    #print("HEY", E, L, C)
    #print(E)
    '''
    R2 = lambda r: (E**2 - 1.0)*(r**4) + 2.0*(r**3) + ((a**2)*(E**2 - 1.0) - L**2 - C)*(r**2) + 2*((a*E - L)**2 + C)*r - C*(a**2)
    test = optimize.fsolve(R2, turns)
    while R2(max(test)) < 0.0:
        dR = -np.polyval(potent, test)
        E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
    '''
    # Step 6
    new_state = recalc_state([E, L, C], state, a)
    #print(len(path), path[-1][0] - path[0][0], dEdt*dt/mu, (2*L0*dLdt*dt/mu + dCdt*dt/mu)/(2*np.sqrt(L0**2 + C)))
    return new_state, [E, L, C]

def new_recalc_state14(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
    r, theta, phi, vel4 = *state[1:4], state[4:]
    sint, cost = np.sin(theta), np.cos(theta)
    sinp, cosp = np.sin(phi), np.cos(phi)
    sph2cart = np.array([[1, 0,         0,           0           ],
                         [0, (r/np.sqrt(r**2 + a**2))*sint*cosp, np.sqrt(r**2 + a**2)*cost*cosp, -np.sqrt(r**2 + a**2)*sint*sinp],
                         [0, (r/np.sqrt(r**2 + a**2))*sint*sinp, np.sqrt(r**2 + a**2)*cost*sinp, np.sqrt(r**2 + a**2)*sint*cosp ],
                         [0,                             r*cost,                        -r*sint, 0                              ]])
    vel4cart = np.matmul(sph2cart, vel4)
    vel3cart = vel4cart[1:4]
    pos3cart = np.array([np.sqrt(r**2 + a**2)*sint*cosp, np.sqrt(r**2 + a**2)*sint*sinp, r*cost])
    Lx, Ly, Lz = np.cross(pos3cart, vel3cart)
    dE, dLx, dLy, dLz = con_derv[:4]
    dC = 2*(Lx*dLx + Ly*dLy)
    E, L, C = E0 + dE, L0 + dLz, C0 + dC
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def new_recalc_state15(cons, con_derv, state, a):
    '''
    Calculates new state vector from current state and change in orbital constants

    Parameters
    ----------
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass
    con_derv : 4-element numpy array of floats
        change in orbital characteristics (energy, cartesian components of L) per unit mass 
    state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment
    a : int/float
        dimensionless spin constant of black hole, between 0 and 1 inclusive

    Returns
    -------
    new_state : 8 element numpy array of floats
        4-position and 4-velocity of the test particle at a particular moment after correction
    cons : 3-element array of floats
        energy, azimuthal angular momentum, and Carter constant per unit mass after correction
    '''
    # Step 1
    E0, L0, C0 = cons
    metric, chris = kerr(state, a)
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
    Lx, Ly, Lz = np.cross(pos3cart, vel3cart)
    dE, dLx, dLy, dLz = con_derv[:4]
    dC = 2*(C0/L0)*dLz
    #dC = 2*(Lx*dLx + Ly*dLy)
    #dC = 2*(Lx*dLx + Ly*dLy)
    E, L, C = E0 + dE, L0 + dLz, C0 + dC
    new_state = recalc_state([E, L, C], state, a)
    return new_state, [E, L, C]

def Jfunc(x, r0, e, i, a, E, L, C):
    '''
    Supplementary function for freqs_finder. Separated because it uses recursion to iterate

    Parameters
    ----------
    x : float
        Radius in terms of gravitational radii
    r0 : float
        Semimajor axis corresponding to given orbit, in terms of gravitational radii
    e : float
        Eccentrity of orbit
    i : float
        Inclination of orbit, with pi/2 as equatorial
    a : float
        Dimensionless spin of central black hole
    E : float
        Specific energy of orbit
    L : float
        Specific angular momentum of orbit
    C : float
        Specific Carter constant of orbit

    Returns
    -------
    float
        Intermediate value used for calculating orbital frequencies

    '''
    #E, L, C = schmidtparam3(r0, e, i, a)
    p = r0*(1 - e**2)
    J = lambda x: (1-E**2)*(1-e**2)+2*(1-E**2-1/r0)*(1+e*np.cos(x))+((1-E**2)*(3+e**2)/(1-e**2)-(4/p)+((a**2)*(1-E**2)+L**2+C)*(1/(r0*p)))*((1+e*np.cos(x))**2)
    z1 = 1 + ((1+a)**(1/3) + (1-a)**(1/3))*(1 - a**2)**(1/3)
    z2 = np.sqrt(3*(a**2) + z1**2)
    rms = 3 + z2 - np.sign(a)*np.sqrt((3-z1)*(3 + z1 + 2*z2))
    if J(0) < 0.0 and r0*(1-e) > rms:
        E1, L1, C1 = schmidtparam3(r0/10, e, i, a)
        return Jfunc(x, r0/10, e, i, a, E1, L1, C1)/10
    else:
        return J(x)
    
def freqs_finder(E, L, C, a):
    '''
    Calculates characteristic frequencies of a given orbit

    Parameters
    ----------
    E : float
        Specific energy of orbit
    L : float
        Specific angular momentum of orbit
    C : float
        Specific Carter constant of orbit
    a : float
        Dimensionless spin of central black hole

    Returns
    -------
    3-element numpy array
        [Radial frequency, theta frequency, phi frequency] in geometric units

    '''
    B2 = (a**2)*(1 - E**2)
    roots = np.round(np.sort(np.roots([B2, 0, -(B2 + C + L**2), 0, C])), 15)
    if len(roots) == 4:
        zm, zp = roots[-2], roots[-1]
    else:
        zm, zp = roots[-1], 1e151
    k, i = (zm**2)/(zp**2), np.arccos(zm)
    
    Rcoeff = np.array([E**2 - 1.0, 2.0, (a**2)*(E**2 - 1.0) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
    ri, ro = np.sort(np.roots(Rcoeff))[-2:]
    if ri.imag/ri.real > 1e-8:
        return np.array([np.nan, np.nan, np.nan])

    r0, e = 0.5*(ro + ri), (ro - ri)/(ro + ri)
    p = r0*(1 - e**2)

    J2 = lambda x: np.real(Jfunc(x, r0, e, i, a, E, L, C))
    H = lambda x: np.real(1 - (2/p)*(1 + e*np.cos(x)) + ((a/p)**2)*(1 + e*np.cos(x))**2)
    G = lambda x: np.real(L - 2*(L - a*E)*(1 + e*np.cos(x))/p)
    F = lambda x: np.real(E + ((a/p)**2)*E*((1 + e*np.cos(x))**2) - 2*a*(L - a*E)*((1 + e*np.cos(x))/p)**3)
    
    Xt = integrate.quad(lambda x: 1/(J2(x)**0.5), 0.0, np.pi)[0]
    Yt = integrate.quad(lambda x: (p**2)/(((1+e*np.cos(x))**2)*(J2(x)**0.5)), 0.0, np.pi)[0]
    Zt = integrate.quad(lambda x: G(x)/(H(x)*(J2(x)**0.5)), 0.0, np.pi)[0]
    Wt = integrate.quad(lambda x: (p**2)*F(x)/(((1 + e*np.cos(x))**2)*H(x)*(J2(x)**0.5)), 0.0, np.pi)[0]

    Kk = integrate.quad(lambda p: 1/np.sqrt(1 - k*(np.sin(p)**2)), 0, np.pi/2)[0]
    Ek = integrate.quad(lambda p: np.sqrt(1 - k*(np.sin(p)**2)), 0, np.pi/2)[0]
    Pk = integrate.quad(lambda p: 1/((1-(zm*np.sin(p))**2)*np.sqrt(1 - k*(np.sin(p)**2))), 0, np.pi/2)[0]

    Lam = (Yt + Xt*(a*zp)**2)*Kk - Xt*Ek*(a*zp)**2
    wr, wt, wp = np.pi*p*Kk/((1-e**2)*Lam), np.pi*(B2**0.5)*zp*Xt/(2*Lam), (1/Lam)*((Zt - L*Xt)*Kk + L*Xt*Pk)
    g = (1/Lam)*((Wt + E*Xt*(a*zp)**2)*Kk - E*Xt*Ek*(a*zp)**2)
    wr, wt, wp, g = np.where(np.array([wr, wt, wp, g]).imag < 1e-11, np.array([wr, wt, wp, g]).real, np.array([wr, wt, wp, g]))
    if a == 0.0:
        wt = wp
    return np.array([wr, wt, wp])/g

def seper_locator(r0, inc, a):
    '''
    Locates seperatrix for a given semimajor axis, inclination, and black hole spin

    Parameters
    ----------
    r0 : float
        Semimajor axis in gravitational units
    inc : float
        Inclination of orbit, with pi/2 as equatorial
    a : float
        Dimensionless spin of central black hole

    Returns
    -------
    3-element numpy array
        [Specific Energy, Specific Angular Momentum, Specific Carter constant] of seperatrix orbit
    float
        Eccentricity of orbit

    '''
    r2, r3 = 1, 0
    rmb = find_rmb(a)
    e = (1 - (rmb/r0))*0.5
    e_list = [e]
    loops = 1
    while (r2 - r3 > 1e-11 or r2 - r3 < 0) and loops < 100:
        r1, r2 = r0*(1 + e), r0*(1 - e)
        E, L, C = schmidtparam3(r0, e, inc, a)
        A_B = 2/(1 - E**2) - (r1 + r2)
        AB = (a**2)*C/((1 - E**2)*r1*r2)
        r3 = (A_B + np.sqrt(A_B**2 - 4*AB))/2.0
        #print(r2, r3, r2-r3, e)
        if r2 - r3 > 1e-11:
            new_e = e*(1 + (np.abs(r2 - r3)/np.abs(r1 - r3))**1.0)
            if new_e == e:
                new_e += 5e-13
        elif r2 - r3 < 0:
            if r1 < r3:
                print("r0 gives plunge for all values of e")
                return [False, False, False]
            else:
                new_e = e*(1 + (r2 - r3)/10)
        e = new_e
        e_list.append(e)
        loops += 1
    return [E, L, C], e

def root_getter(E, L, C, spin):
    a, b, c, d, e = np.array([(E**2 - 1.0), 2.0, ((spin**2)*(E**2 - 1.0) - L**2 - C),  (2*((L - spin*E)**2) + 2*C), -(spin**2)*C]).astype(complex)
    p1 = 2*(c**3) - 9*b*c*d + 27*a*(d**2) + 27*(b**2)*e - 72*a*c*e
    p2 = p1 + (-4*(c**2 - 3*b*d + 12*a*e)**3 + p1**2)**0.5
    p3 = (c**2 - 3*b*d + 12*a*e)/(3*a*((0.5*p2)**(1/3))) + ((0.5*p2)**(1/3))/(3*a)
    p4 = ((b**2)/(4*(a**2)) - (2*c)/(3*a) + p3)**(0.5)
    p5 = (b**2)/(2*(a**2)) - (4*c)/(3*a) - p3
    p6 = (-(b**3)/(a**3) + (4*b*c)/(a**2) - 8*d/a)/(4*p4)
    x1 = -b/(4*a) - p4/2 - 0.5*((p5 - p6)**0.5)
    x2 = -b/(4*a) - p4/2 + 0.5*((p5 - p6)**0.5)
    x3 = -b/(4*a) + p4/2 - 0.5*((p5 + p6)**0.5)
    x4 = -b/(4*a) + p4/2 + 0.5*((p5 + p6)**0.5) 
    turns = np.array([np.real(num) if np.abs(np.imag(num)) < 1e-12 else num for num in [x1, x2, x3, x4]]).astype(complex)
    
    flats = np.roots([4*a,3*b,2*c,d])
    flats = np.array([np.real(num) if np.abs(np.imag(num)) < 1e-12 else num for num in flats]).astype(complex)
    
    a, b, c, d, e = np.array([(a**2)*(1 - E**2), 0.0, -(C + (a**2)*(1 - E**2) + L**2), 0.0, C]).astype(complex)
    p1 = 2*(c**3) - 9*b*c*d + 27*a*(d**2) + 27*(b**2)*e - 72*a*c*e
    p2 = p1 + (-4*(c**2 - 3*b*d + 12*a*e)**3 + p1**2)**0.5
    p3 = (c**2 - 3*b*d + 12*a*e)/(3*a*((0.5*p2)**(1/3))) + ((0.5*p2)**(1/3))/(3*a)
    p4 = ((b**2)/(4*(a**2)) - (2*c)/(3*a) + p3)**(0.5)
    p5 = (b**2)/(2*(a**2)) - (4*c)/(3*a) - p3
    p6 = (-(b**3)/(a**3) + (4*b*c)/(a**2) - 8*d/a)/(4*p4)
    x1 = -b/(4*a) - p4/2 - 0.5*((p5 - p6)**0.5)
    x2 = -b/(4*a) - p4/2 + 0.5*((p5 - p6)**0.5)
    x3 = -b/(4*a) + p4/2 - 0.5*((p5 + p6)**0.5)
    x4 = -b/(4*a) + p4/2 + 0.5*((p5 + p6)**0.5) 
    zs = np.array([np.real(num) if np.abs(np.imag(num)) < 1e-5 else num for num in [x1, x2, x3, x4]]).astype(complex)
    
    return np.sort(turns), np.sort(flats), np.sort(zs)