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



def make_scale(mbh):              #mbh is mass of black hole in solar masses
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
    rad = rad / gu
    
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
    circle_orb = ml.inspiral_long(circle_launch, 1.0, 0.0, mu, 1, 50000, 0.1, True, 10**(-14), 90, 90, 'circle', spec='circle',verbose=False)
                                     # Run for a little over a full orbit, just to be sure everything settles out nicely
                          
    ene, lel = circle_orb["energy"][-1], circle_orb["phi_momentum"][-1]
    small = min(circle_orb["pos"][:,0])
    lel_min, lel_max = 0.0, lel
    
    
    while (per_diff(lel_min, lel_max) >= 10**(-8)):
        if small < peri:
            lel_min = lel
            lel = (lel_min + lel_max)/2.0
        else:
            best = lel
            lel_max = lel
            lel = (lel_min + lel_max)/2.0
        
        
        lab = "M=" + str(mstar) + ", R=" + str(rad/rsun)
        launch  = np.array([ [0.00, semi, np.pi/2, 0.00, ene, lel, 0.0] ])  #ELC values don't matter since this is circular
        orb = ml.inspiral_long(launch, 1.0, 0.0, mu, 1, 50000, 0.1, True, 10**(-14), 90, 90, lab, verbose=False)
        small = min(orb["pos"][:,0])
        print("Target Periapsis: " + str(peri))
        print("Actual Periapsis: " + str(small) + "\n")
        
    
    if per_diff(peri, small) > 0.1:
        print("\nCalculated periapsis is a plunge orbit, returning closest guess.\n")
        
    print("Schwarzschild radius is:   " + str(gu/500.0) + "km")
    print("Marginally bound orbit is: " + str(gu/250.0) + "km")
    print("Calculated Periapsis is:   " + str(peri*gu/1000.0) + "km")
    print("Equivalent to " + str(peri) + " in program units.")
    
    return orb
    
        
def collision(state, mu, mass, a, r0, e, density=2.95*(10**(-7)), rad=False):
    peri, rad = wheres_peri(mass, mass*mu, rad)
    
    md = np.pi*(rad**2) * density
        #assumes disk has constant surface density, star impacts fully within it
        #10^(-4) of the star's mass spread in a disk between 3R_s and 6R_s is roughly 2.95*(10^-7)
        #will probably need to modify this to account for distance stuff or whatever
    thing = np.array(mm.set_u_kerr(state, 1.0, a, True, 90, 90, special="circle"))
    pd = md * thing[4:]    #4 momentum of disk segment
    pstar = mu * state[4:]  #4 momentum of star
    pos = state[:4]        #position data
    pnew = pd + pstar      #4 momentum of modified star
    new_mu = (-mm.check_interval(mm.kerr, [*pos, *pnew], 1.0, a))**(1/2)
        #applying check_interval to a 4-momentum gives -m^2
    new_state = np.array([*pos, *(pnew/new_mu)])
    #since mu changes, I'm gonna have to do the constant update right before the change is applied, i think
    #Which means I update it THREE times per orbit at the very least, since there's a chance it could hit the disk multiple times
    return new_state, new_mu
    
def PTDE(state, a, mu, r0):
    ene, lel, cart = mm.find_constants(state, 1.0, 0.0)
    incline = lel / np.sqrt(cart + lel**2)
    r, theta = state[1], state[2]
    period = 2*np.pi*(r0**(3/2))
    r0_dot = (-0.0017)*(2/3)*(1/(4*(np.pi**2)*period))**(1/3)
    dEdr0 = (r0**(7/2) - 6*(r0**(5/2)) + 8*a*(r0**2) - 3*(a**2)*(r0**(3/2)))/(2*(r0**(5/2))*((r0**2 - 3*r0 + a*np.sqrt(r0))**(3/2)))
    dEdt = dEdr0*r0_dot
    ene_new = ene + dEdt*period
    full_mom = (1/(r-2))*(-2*a*ene_new + np.sqrt( (r-2)*(4*a*ene*lel + (lel**2)*(r-2) + (r**3)*(ene_new**2 - ene**2)) + (a**2)*((ene_new*r)**2 - (ene**2)*(r**2 - 4))))
    lel_new, cart_new = full_mom*incline, full_mom*(1 - incline**2)
    star_state = mm.recalc_state([ene_new, lel_new, cart_new], state, 1.0, a)
    A, B = (1 - (2*r)/(r**2 + (a*mm.fix_cos(theta))**2)), 2*a*r*(mm.fix_sin(theta)**2)/(r**2 + (a*mm.fix_cos(theta))**2)
    dm = mu*( (A*(state[4] - star_state[4]) + B*(state[7] - star_state[7])) /  (1 - A*star_state[4] - B*star_state[7]) )
    p_full, p_star = mu*state[4:], (mu - dm)*star_state[4:]
    p_dm = p_full - p_star
    dm_state = np.array([*state[:4], *(p_dm/dm)])

    return ((mu-dm, star_state), (dm, dm_state))

def rel_E_disk(state, a, mu):
    d_state = np.array(mm.set_u_kerr(state, 1.0, a, True, 90, 90, special="circle"))      #position, velocity of disk at given radius
    disk_v2 = ((d_state[5]/d_state[4])**2 + (d_state[1]*d_state[6]/d_state[4])**2 + (d_state[1]*np.sin(d_state[2])*d_state[7]/d_state[4])**2)   #v^2 of disk
    disk_gamma = 1/np.sqrt(1-disk_v2)     #gamma of disk
    boost = np.array([[disk_gamma,     -np.sqrt(disk_v2)*disk_gamma,     0,     0],
                      [-np.sqrt(disk_v2)*disk_gamma,     disk_gamma,     0,     0],
                      [0,               0,                       0, 0],
                      [0,               0,                       0, 0]])       #boost matrix into disk frame
    print(boost)
    cart_state = np.array([state[4], state[5], state[1]*state[7], -state[1]*state[6]])   #convert object velocity to cartesian
    print(cart_state)
    boost_state = np.matmul(boost, cart_state)      #apply boost to get object velocity in disk frame
    object_v2 = ((boost_state[1]/boost_state[0])**2 + (boost_state[2]/boost_state[0])**2 + (boost_state[3]/boost_state[0])**2) #object v2 in disk frame
    object_gamma = 1/np.sqrt(1-object_v2)     #gamma of object in disk frame
    object_ke = (object_gamma - 1)*mu
    return object_ke

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






