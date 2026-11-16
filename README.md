# EMRICode (name pending)
## A Python Toolkit for generating and displaying generic Extreme Mass Ratio Inspirals
Extreme Mass Ratio Inspirals are binary systems consisting of a massive or supermassive black hole orbited by a much smaller object, such that the mass ratio between them is $\leq$ 10<sup>-4</sup>. 
EMRI systems are expected to be potent emitters of gravitational radiation as well as a thorough test of General Relativity in regions of extreme gravity, 
which makes them of great interest to astronomers and astrophysicists in the field of multi-messenger astronomy.

As part of the ongoing effort to study EMRI systems, this package is being developed as a tool to generate orbits across as much of the parameter space as possible. 
The project is currently incomplete.

Packages Used:
- Numpy
- Scipy
- Numba
- Sympy
- tqdm
- Matplotlib
- IPython
- h5py
- filelock
- PyYAML
- psutil

## Future Functionality Goals
- Full/Partial Tidal Disruption of smaller body
- Accretion Disk interaction
- Higher order corrections to energy loss calculations
- Effects of non-zero smaller body spin




# EMRICode (name pending)

### A Python Toolkit for Generating and Visualizing Generic Extreme Mass-Ratio Inspirals

Extreme mass-ratio inspirals (EMRIs) are binary systems consisting of a massive or supermassive black hole orbited by a much smaller compact object, with a mass ratio typically \(\lesssim 10^{-4}\). These systems are expected to be important sources of gravitational radiation and provide opportunities to test General Relativity in the strong-field regime.

EMRICode is an ongoing project intended to provide a flexible framework for simulating EMRI orbital dynamics across a broad range of orbital parameters. The long-term goal is to support the inclusion of additional physical effects beyond vacuum geodesic motion and gravitational radiation reaction.

**This project is under active development.** Some functionality is incomplete or subject to change.

## Current Functionality

EMRICode currently supports:

- Numerical integration of timelike trajectories in Kerr spacetime, including eccentric and inclined orbits.
- Evolution of orbital parameters under an approximate gravitational radiation-reaction prescription.
- Calculation and tracking of orbital quantities, including conserved quantities and orbital frequencies.
- Visualization and analysis of generated trajectories.
- Saving and loading simulation data using HDF5 files.
- Running multiple simulations through a multiprocessing worker.

## Installation

EMRICode has been installation-tested with Python 3.12, 3.13, and 3.14.

Clone the repository:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
```

Create and activate a virtual environment:

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick Start

## Quick Start

The primary interface for generating EMRI trajectories is `EMRIGenerator`, located in `MainLoopStreamline.py`.

A simulation can be initialized using orbital parameters, constants of motion, or initial position and velocity. The following example generates a short eccentric, inclined trajectory around a rotating black hole, then plots it radial motion against time and a 3D trajectory.

```python
import numpy as np
from MainLoopStreamline import EMRIGenerator

orbit = EMRIGenerator(
    a=0.8,
    mu=1e-4,
    params=[12, 0.35, -np.pi / 6],
    endflag="rad_orbit > 25",
    err_target=1e-12,
    label="example_orbit"
)

# Coordinate trajectory: [t, r, theta, phi]
positions = orbit["pos"]

# Coordinate time and radial position
time = orbit["time"]
radius = orbit["raw"][:, 1]

print("Number of integration steps:", len(time))
print("Final coordinate time:", time[-1])
print("Final radial position:", radius[-1])
print("Simulation stopped early:", orbit["stop"])
print("Simulation continued to plunge:", orbit["plunge"])
```

The resulting dictionary contains the full numerical trajectory, orbital parameters, and diagnostic quantities. The example above uses geometrized units with (G=c=M=1), where (M) is the mass of the central black hole.

For a complete example, see `examples/basic_inspiral.py` (once added).


## Inputs and Outputs

## Configuring Simulations

`EMRIGenerator` accepts several methods for specifying initial orbital conditions. The most accessible is the `params` argument, which takes three values:

```python
params = [p, e, inclination]
```

- `p`: Semilatus rectum, in units of the central black hole's gravitational radius \(GM/c^2\).
- `e`: Orbital eccentricity.
- `inclination`: Orbital inclination in radians. In the current convention, \(\pi/2\) corresponds to equatorial prograde motion, \(0\) to polar motion, and \(-\pi/2\) to equatorial retrograde motion.

Other important arguments include:

| Argument | Description |
|---|---|
| `a` | Dimensionless central black hole spin, between -1 and 1 |
| `mu` | Secondary-to-primary mass ratio |
| `endflag` | Condition specifying when integration should stop |
| `err_target` | Numerical error tolerance for the geodesic integrator (currently tested up to 1e-15)|
| `label` | Identifier for the simulation |
| `verbose` | Progress reporting: 0 (silent), 1 (progress bar), or 2 (detailed) |

For alternative initialization methods, including constants of motion and local velocity components, see the `EMRIGenerator` docstring.

### Stopping Conditions

Simulation duration is controlled by the `endflag` argument, which accepts a comparison expression as a string.

For example:

```python
endflag="rad_orbit > 5"       # Stop after approximately five radial orbits
endflag="elapsed_time > 1000" # Stop after 1000 geometric time units
endflag="index > 100000"      # Stop after approximately 100,000 integration steps
```

Other supported stopping variables include orbital radius, eccentricity, inclination, and the number of azimuthal orbits.

### Simulation Output

`EMRIGenerator` returns a dictionary containing trajectory data, evolving orbital quantities, and diagnostic information.

The most useful entries for getting started are:

| Key | Description |
|---|---|
| `raw` | Full \(N\times8\) trajectory array, ordered as \([t,r,\theta,\phi,u^t,u^r,u^\theta,u^\phi]\) |
| `pos` | \(N\times4\) position array |
| `vel` | \(N\times4\) four-velocity array |
| `time` | Coordinate time at each integration step |
| `proper_time` | Proper time along the trajectory |
| `energy` | Specific orbital energy, tracked once per radial orbit |
| `phi_momentum` | Specific axial angular momentum, tracked once per radial orbit |
| `carter` | Specific Carter-related constant, tracked once per radial orbit |
| `p`, `e`, `inc` | Evolving Keplerian orbital parameters, tracked once per radial orbit|
| `tracktime` | Coordinate time at each radial orbit update |
| `trackix` | Indices connecting orbital-parameter updates to the full trajectory |
| `stop` | Indicates whether the simulation stopped before satisfying the requested stopping condition |
| `plunge` | Indicates whether the trajectory ended in a plunge |
| `unbind` | Indicates whether the secondary exceeded the code's unbound-distance threshold |

Arrays such as `raw` and `time` contain values at each integration step. Arrays such as `energy`, `p`, and `e` are recorded at radial-orbit boundaries and therefore generally have fewer entries.

See the `EMRIGenerator` docstring for the complete output dictionary.

## Current Limitations

- Radiation reaction is treated approximately rather than through a complete gravitational self-force calculation.
- Environmental interactions and secondary-body spin effects are not yet implemented in the released code.
- Numerical accuracy and computational cost depend on the chosen orbital parameters and integration settings.
- The software remains under development, and its interfaces and numerical methods may change.

## Future Functionality Goals

- Full and partial tidal disruption of the smaller body.
- Accretion disk interactions.
- Higher-order corrections to orbital evolution and energy-loss calculations.
- Effects of nonzero secondary-body spin.
- Continued improvements to numerical accuracy, performance, and parameter-space coverage.

## Citation

If you use EMRICode in published research, please cite the following paper:

**[Courtney McIntosh, Jeremy Schnittman]**, *A Multipole Technique for the Simulation and Analysis of Gravitational Waveforms from Extreme Mass Ratio Inspirals*, arXiv:2610.27153 (2026).

[\[Link to arXiv preprint\]](https://doi.org/10.48550/arXiv.2610.27153)

This citation will be updated when the journal version becomes available.

## License

???