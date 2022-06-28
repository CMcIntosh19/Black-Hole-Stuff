# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:42:29 2022

@author: hepiz
"""

import numpy as np
import MetricMath as mm
import MainLoops as ml
import matplotlib.pyplot as plt
import OrbitPlotter as op

# Program is meant to put together orbits that approximate what's seen in ASASSN-14ko
# specifically in the approximation of a repeated partial tidal disruption event as noted in https://doi.org/10.3847/1538-4357/abe38d


# Approximated data (non_specific to stellar mass)


msun = 1.989*(10**30)             # kg

period = 114.2 * 24 * 60 * 60     # seconds
G = 6.6743 * (10**(-11))          # Newton * kg / s^2
c = 299792458.0                   # speed of light in m/s
            

# Convert some things for use in program (c=G=1)



def make_scale(mbh):
    mbh = mbh * msun
    semi_m = (period/(2*np.pi) * np.sqrt(G*mbh) )**(2/3)
                                  # semimajor axis in meters
                                  
    rs = 2*G*mbh / (c**2)         # schwarzschild radius (zero-spin)
    gu = rs/2.0                   # 1/2 rs = 1 of whatever distance unit the program uses
    semi = semi_m / gu            # semimajor axis in program distance units
    
    return semi, gu

# Specifics of orbit are determined by properties of star
# Tidal disruption radius (aka periapsis of orbit) is roughly equal to r_star * (mbh/mstar)^(1/3)
# r_star for main sequence has a direct relation, but off-main sequence stars with different radii do exist
# still need to convert values to units the program will use

rsun = 696.34 * (10**6)           # meters

def wheres_peri(mbh, mstar, rad=None):  # mass and radius are in units of stellar mass/radius; rad=None means main sequence
    semi, gu = make_scale(mbh)
    if rad == None:               
        if mstar >= 1.0:
            xi = 0.57
        else:
            xi = 0.8
        rad = mstar**(xi)          # determines radius for a main sequence star
    
    rad = rad * rsun
    mstar = mstar
    mbh = mbh
    
    peri_m = rad * ((mbh/mstar)**(1/3))
                                  # perihelion in meters
    peri = peri_m / gu
    
    return peri, rad


# Then you need to actually calculate the orbit
# Energy is easy - just plug in values for a circular orbit using the given data
# Carter constant isn't necessary, since at the moment I'm assuming equitorial orbits
    # Technically that's probably definitely wrong since it would kool-aid man through the accretion disk
    # but it's also assuming zero-spin, so same difference
# Angular momentum is the part that depends on periapsis. Maybe there's some easy L(energy, periapsis) formula
# but for now I'll just use brute force bisection

def per_diff(upp, low):
    return 2*abs(upp-low)/abs(upp+low) * 100


def find_orbit(mbh, mstar, rad=None):      # masses and radius are in units of stellar mass/radius; rad=None means main sequence
    semi, gu = make_scale(mbh)
    peri, rad = wheres_peri(mbh, mstar, rad)
    mu = mstar/mbh
    
    circle_launch  = np.array([ [0.00, semi, np.pi/2, 0.00, 1.0, 5.0, 0.0, 1.1] ])  #ELC values don't matter since this is circular
    circle_orb = ml.inspiral_long(circle_launch, 1.0, 0.0, mu, 1, 45000, 0.1, True, 10**(-10), 90, 90, 'circle', spec='circle',verbose=False)
                                     # Run for a little over a full orbit, just to be sure everything settles out nicely
                          
    ene, lel = circle_orb["energy"][-1], circle_orb["phi_momentum"][-1]
    small = min(circle_orb["pos"][:,0])
    lel_min, lel_max = 0.0, lel
    plunge = 0
    
    
    while (per_diff(lel_min, lel_max) >= 10**(-8)) and (plunge <= 10):
        if small < 2.5:
            plunge += 1
            
        if small < peri:
            lel_min = lel
            lel = (lel_min + lel_max)/2.0
        else:
            best = lel
            lel_max = lel
            lel = (lel_min + lel_max)/2.0
        
        
        print(lel_min)
        print(lel_max)
        print(peri)
        print(small)
        print(plunge)
        lab = "M=" + str(mstar) + ", R=" + str(rad/rsun)
        launch  = np.array([ [0.00, semi, np.pi/2, 0.00, ene, lel, 0.0] ])  #ELC values don't matter since this is circular
        orb = ml.inspiral_long(launch, 1.0, 0.0, mu, 1, 45000, 0.1, True, 10**(-10), 90, 90, lab, verbose=False)
        small = min(orb["pos"][:,0])
    
    if plunge > 9:
        print("\nParameters might be non-physical, plotting best guess.")
        launch  = np.array([ [0.00, semi, np.pi/2, 0.00, ene, best, 0.0] ])  #ELC values don't matter since this is circular
        orb = ml.inspiral_long(launch, 1.0, 0.0, mu, 1, 45000, 0.1, True, 10**(-10), 90, 90, lab, verbose=False)
        
    return orb
    
        
        
    




'''
#Calculate some real values

initial  = np.array([ [0.00, rad, np.pi/2, 0.00, 1.0, 4.0, 0.0] ])
                    # These E,L,C values don't really matter since the next line makes this a circular orbit
                    # (Values will be modified to accomodate)
test0 = ml.inspiral_long(initial, 1.0, 0.0, mu, 1, 20000, 0.1, True, 10**(-15), 90, 90, 'circle', verbose=False)
                    # Run for a little over a full orbit, just to be sure everything settles out nicely

ene, lel, car = test0["energy"][-1], test0["phi_momentum"][-1], test0["carter"][-1]
                    # Get the E,L,C values! Only the energy is truly vital, but keeping L is useful for comparison
                    # C is 0.0 by default, but we might as well grab all three
                    

# Tidal disruption radius is equal to r_star*(mbh/mstar)^(1/3)
# Star mass is already set to 





[-0.99854886  0.          0.         18.62317351]
'''
















