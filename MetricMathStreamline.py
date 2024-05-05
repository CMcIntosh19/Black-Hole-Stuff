# -*- coding: utf-8 -*-
"""
Metric Math stuff
"""

import numpy as np
from scipy import optimize
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
    l_mb: float
        Specific angular momentum of an equatorial orbit at the marginally bound orbit
    r_mb: float
        Periapse of an equatorial orbit at the marginally bound orbit
    '''
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
    #print(data)
    l_mb, r_mb = pro*data['fun'], data['x'][0]
    return l_mb, r_mb

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
        print("Calculating initial velocity from constants E,L,C")
        if np.shape(pos) == (4,):
            new = recalc_state(cons, pos, a)
        else:
            E, L, C = cons
            Rdco = [4*(E**2 - 1), 6, 2*((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C)]
            flats = np.roots(Rdco)
            flats = np.sort(flats.real[abs(flats.imag)<1e-14])
            pos = [0.0, flats[-1], np.pi/2, 0.0]
            new = recalc_state(cons, pos, a)
    elif (np.shape(velorient) == (3,)):
        print("Calculating intial velocity from tetrad velocity and orientation")
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
        else:
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
        new = np.array([[*pos, *new]])
    elif (np.shape(vel4) == (4,)) and np.shape(pos) == (4,):
        print("Calculating initial velocity from tetrad component velocities")
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
        vel_2 = (rdot**2 + (r * thetadot)**2 + (r * np.sin(theta) * phidot)**2)
        beta = np.sqrt(vel_2)
        gamma = 1/np.sqrt(1 - vel_2)
        eta = np.arccos(np.sqrt((r * np.sin(theta) * phidot)**2)/beta)
        xi = np.arccos(np.sqrt(rdot**2)/(beta*np.sin(eta)))
        tilde = np.array([gamma, gamma*beta*np.cos(eta), -gamma*beta*np.sin(eta)*np.cos(xi), gamma*beta*np.sin(eta)*np.sin(xi)])
        new = np.matmul(tetrad_matrix, tilde)
    elif np.shape(params) == (3,):
        print("Calculating initial velocity from orbital parameters r0, e, i (WIP)")
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
        h = r*(r - 2) + tri*(z**2)/(1 - z**2)
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
        return (z**2)*((a**2)*(1 - E**2) + (L**2)/(1 - z**2))
    
    x, y = sp.symbols("x y", real=True)
    eq1 = sp.Eq(f1*(x**2) - 2*g1*x*y - h1*(y**2), d1)
    eq2 = sp.Eq(f2*(x**2) - 2*g2*x*y - h2*(y**2), d2)
    symsols = sp.solve([eq1, eq2])

    full_sols = []
    for thing in symsols:
        ene, lel = np.array([thing[x], thing[y]]).astype(float)
        if ene > 0.0 and ene < 1.0: 
            full_sols.append([ene, lel, newC(ene, lel, a, z)])
            E, L, C = [ene, lel, newC(ene, lel, a, z)]
            if np.product(np.sign([E, L, C])) == np.sign(np.sin(j)):
                break
            else:
                E, L, C = (1 - (1 - e**2)/p)**0.55, ((1 - z**2)*p)**(0.5), p*(z**2)
                
    coeff, ro = [-1], 1
    while np.polyval(coeff, ro) < -1e-12:
        E += 10**(-16)
        coeff = np.array([E**2 - 1.0, 2.0, (a**2)*(E**2 - 1.0) - L**2 - C, 2*((a*E - L)**2 + C), -C*(a**2)])
        coeff2 = np.polyder(coeff)
        turns = np.roots(coeff)
        flats = np.roots(coeff2)
        turns = turns.real[abs(turns.imag)<(1e-6)*r0]
        flats = flats.real[abs(flats.imag)<(1e-6)*r0]
        ro = max(flats)
    turns = np.sort(turns)
    r02, e2 = (turns[-1] + turns[-2])/2.0, (turns[-1] - turns[-2])/(turns[-1] + turns[-2])
    r_err, e_err = np.abs((r0 - r02)/r0)*100, np.abs((e2 - e)/(2-e))*100
    
    if r_err < 1e-6 and e_err < 1e-6:
        if (np.sqrt(L**2 + C) - np.abs(L))/L < 1e-15:
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
            if (np.product(np.sign(solution)) == np.sign(np.sin(j))):
                E, L, C = solution
                break     
        if polar == True:
            L, C = 0.0, C/(z**2) - (a**2)*(1 - E**2)
        #print(flats)
        if (np.sqrt(L**2 + C) - np.abs(L))/L < 1e-15:
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
        #If current radius is between the inner and outer turning points, maintain direction
        if (rad - roots[-2])*(roots[-1] - rad) > 0:
            direc = np.sign(state[5])
        #If current radius is somehow outside that range, follow the potential to go back in
        else:
            direc = np.sign(np.polyval(np.polyder(r_r), rad))
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
        limits the size of the final array to 10000 entries

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
        new_time = np.linspace(time[0], time[-1], min(10000, max(len(time), int(time[-1] - time[0]))))
    else:
        new_time = np.linspace(time[0], time[-1], max(len(time), int(time[-1] - time[0])))
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

def trace_ortholize(pos_list):
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
    dedt, dldt = 0, np.array([0.0, 0.0, 0.0])
    if (ind2 - ind1) > 2:
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0]
        int_sphere, int_time = interpolate(sphere, time)
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
    return np.array([dE, dLx, dLy, dLz])

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
    dL_vec = -np.linalg.norm([dLx, dLy, dLz])
    if z2 != 1.0 and L0 != 0.0:
        dC = 2*z2*(L0*dLz/(1-z2) - (a**2)*E0*dE)  
    else:
        #Lz/(1-z2) seems to be approximately the total angular momentum, so replace Lz/(1-z2) with sqrt(C) to account for polar orbits
        dC = 2*z2*(np.sqrt(C0)*dL_vec - (a**2)*E0*dE)  

    # Step 4
    E, L = E0 + dE, L0 + dLz*np.sign(L0) #make sure L0 is going towards 0, not becoming increasingly negative if retrograde
    
    # Step 5
    C = C0 + dC
    potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
    test = max(np.roots(np.polyder(potent)))
    while (np.polyval(potent, test) < 0.0):
        dR = -np.polyval(potent, test)
        E += max(dR*(( 2*test*((test**3 + (a**2)*test + 2*(a**2))*E - 2*L*a))**(-1)), 10**(-16))
        potent = np.array([(E**2 - 1), 2, ((a**2)*(E**2 - 1) - L**2 - C), 2*((a*E - L)**2 + C), -C*(a**2)])
        test = max(np.roots(np.polyder(potent)))
    
    # Step 6
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
    #print(ro, ri)
    r0, e = 0.5*(ro + ri), (ro - ri)/(ro + ri)
    p = r0*(1 - e**2)
    #print(r0, e, i)
    
    J2 = lambda x: Jfunc(x, r0, e, i, a, E, L, C)
    H = lambda x: 1 - (2/p)*(1 + e*np.cos(x)) + ((a/p)**2)*(1 + e*np.cos(x))**2
    G = lambda x: L - 2*(L - a*E)*(1 + e*np.cos(x))/p
    F = lambda x: E + ((a/p)**2)*E*((1 + e*np.cos(x))**2) - 2*a*(L - a*E)*((1 + e*np.cos(x))/p)**3
    
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
    rmb = find_rmb(a)[1]
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