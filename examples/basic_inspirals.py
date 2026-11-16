# Basic Inspiral Example

import sys
from pathlib import Path

# Add the repository root to Python's import path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from MainLoopStreamline import EMRIGenerator

# EMRIGenerator is the primary interface for generating inspiral data. 
# It can be initialized with various inputs, such as:

# Keplerian parameters,
p = 12                       # Semilatus Rectum (in geometric distance units)
e = 0.35                     # Eccentricity
inc = np.pi/6                # Signed minimum polar angle (radians), NOT conventional orbital inclination
                             #  pi/2 = prograde equatorial, 0 = polar, -pi/2 = retrograde equatorial
orbit1 = EMRIGenerator(
    a=0.8, 
    mu=1e-4,
    params=[p, e, inc],
    endflag="rad_orbit > 25",
    err_target=1e-12,
    label="example orbit 1"
)

# or Characteristic values, among other methods
# These values taken from orbit1 will give us an essentially identical result.
E = orbit1["energy"][0]            # Specific Energy
L = orbit1["phi_momentum"][0]      # Specific Axial Angular Momentum (negative for retrograde orbits)
C = orbit1["carter"][0]            # Specific Carter Constant (always non-negative, where 0 corresponds to 
                                   #    equatorial orbits)
orbit2 = EMRIGenerator(
    a=0.8, 
    mu=1e-4,
    cons=[E, L, C],
    endflag="rad_orbit > 25",
    err_target=1e-12,
    label="example orbit 2"
)

# It is also possible to initialize an inspiral with a 4-position and 4-velocity 
#   in Boyer-Lindquist coordinates, as long as they are on the mass-shell.
# Here, we take the initial state from our first orbit, since we already know it works.
# This will also give us an identical result to orbit1.
position = orbit1["raw"][0, :4]                   # BL position [time, radius, polar coord., azimuthal coord]
velocity = orbit1["raw"][0, 4:]                   # BL velocity 

orbit3 = EMRIGenerator(
    a=0.8, 
    mu=1e-4,
    pos = position,
    veltrue = velocity,
    endflag="rad_orbit > 25",
    err_target=1e-12,
    label="example orbit 3"
)

print("Available data:", list(orbit1.keys()))
print("Number of integration steps:", len(orbit1["time"]))
print("Number of orbital updates:", len(orbit1["energy"]))

fig1, ax1 = plt.subplots()
ax1.plot(orbit1["time"], orbit1["raw"][:, 1])
ax1.set_xlabel("Coordinate time (M)")
ax1.set_ylabel("Radial coordinate (M)")
plt.show()

# There's also a native plotting function

from OrbitPlotter import plotvalues

fig2, ax2 = plotvalues([orbit1, orbit2, orbit3], "radius", "theta")
plt.show()