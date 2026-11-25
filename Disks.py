# Disks.py

import numpy as np
import MetricMathStreamline as mm
from scipy.integrate import quad
from scipy.special import gamma

# Define the basic Accretion disk object
# Give it a mass, a radial extent, an aspect ratio, and an orientation (default that to align with spin)
class AccretionDisk():
    def __init__(self, r_inner, r_outer, aspect_ratio, mass, surface_profile=None, orientation=None, precess_rate=0.0):
        # Define the bounds of the disk
        self.r_inner = r_inner
        if r_outer > r_inner:
            self.r_outer = r_outer
        else:
            self.r_outer = 10 * r_inner
        if aspect_ratio > 0:
            self.aspect_ratio = aspect_ratio
        else:
            self.aspect_ratio = 0.05

        # Define the mass-distribution stuff
        self.mass = mass
        if surface_profile is None:
            self.surface_profile = lambda r: (r / 10.0)**(-1.0)
        else:
            self.surface_profile = surface_profile
        integral, _ = quad(lambda r: 2 * np.pi * r * self.surface_profile(r), 
                           r_inner,
                           r_outer)
        self.sigma0 = mass / integral

        # Define the orientation stuff
        if orientation is None:
            self.orientation = np.array([0.0, 0.0, 1.0])
        else:
            self.orientation = np.array(orientation)/np.linalg.norm(orientation)
        self.precess_rate = precess_rate

    def surface_density(self, r):
        r = np.asarray(r)
        sigma = self.sigma0 * self.surface_profile(r)
        return np.where((r >= self.r_inner) & (r <= self.r_outer), sigma, 0.0)

    def volume_density(self, trajectory, radii, n=2):
        sigma = self.surface_density(radii)
        dot = np.vecdot(trajectory[:, 1:4], self.get_orientation(trajectory[:, 0]))
        H = self.aspect_ratio * radii
        rho = (sigma/H) * (gamma(n + 1.5)/(np.pi * gamma(n + 1))) * (1 - (dot/H)**2)**n
        return rho

    def get_orientation(self, elapsed_time):
        angle = np.asarray(elapsed_time) * self.precess_rate
        c, s = np.cos(angle), np.sin(angle)
        nx, ny, nz = self.orientation
        return np.stack([nx*c - ny*s, nx*s + ny*c, np.broadcast_to(nz, np.shape(angle))], axis=-1)

    def find_crossings(self, trajectory):
        # Trajectory is Nx8 cartesian position + velocity
        # Assume I can treat the crossings as linear for the moment, that's what cool kids do 
        times = trajectory[:, 0]
        dot = np.vecdot(trajectory[:, 1:4], self.get_orientation(times))
        top_bot = np.sign(dot)
        ixs = np.where(top_bot[:-1] + top_bot[1:] == 0)[0]
        m_dot = ((dot[ixs + 1] - dot[ixs])/(times[ixs + 1] - times[ixs]))
        b_dot = dot[ixs] - m_dot * times[ixs]
        cross_times = - b_dot / m_dot
        m = ((trajectory[ixs + 1] - trajectory[ixs]).T/(times[ixs + 1] - times[ixs])).T
        b = trajectory[ixs] - (m.T * times[ixs]).T
        cross_states = (m.T * cross_times).T + b
        return cross_states
        
    def isinDisk(self, trajectory, radii):
        # Trajectory is Nx8 cartesian position + velocity
        # radii are the radii, obviously - default is in radii, so why bother recalculating, just gimme
        times = trajectory[:, 0]
        ratios = np.vecdot(trajectory[:, 1:4], self.get_orientation(times))/radii
        # taking the dot product gives the cosine of the angle between orientation and point
        # which is the sine of the angle between the midplane and the point, which is the same as the aspect ratio
        return (np.abs(ratios) <= self.aspect_ratio) & (radii >= self.r_inner) & (radii <= self.r_outer)

def flatdisk_interaction(disk, trajectory, a, mu, cons):
    carts = mm.sph2cart_vec(trajectory, a)
    cross_carts = disk.find_crossings(carts)
    if len(cross_carts) == 0:
        return np.zeros(3)
    crossings = mm.cart2sph_vec(cross_carts, a)

    densities = disk.surface_density(crossings[:, 1])
    dm = densities * 16.0*np.pi*(mu**2)
    dm_total = np.sum(dm)
    if dm_total <= 0:
        return np.zeros(3)
    M_new = mu + dm_total

    oris = disk.get_orientation(crossings[:, 0])
    oris = oris / np.linalg.norm(oris, axis=1)[:, None]
    incs = np.arctan2(np.hypot(oris[:, 0], oris[:, 1]), oris[:, 2])
    E, L, C = mm.schmidtcircle(crossings[:, 1], incs, np.full_like(incs, a))

    # Changes in specific E and Lz
    dE = (np.dot(dm, E) - dm_total*cons[0]) / M_new
    dLz = (np.dot(dm, L) - dm_total*cons[1]) / M_new

    # Approximate disk angular momentum vectors
    Leff = np.sqrt(L**2 + C)
    L_disk_vec = np.sum(oris * (Leff*dm)[:, None], axis=0)

    # Estimate secondary angular momentum direction
    pos = carts[-1, 1:4]
    vel = carts[-1, 5:8]
    L_old_dir = np.cross(pos, vel)
    L_old_dir /= np.linalg.norm(L_old_dir)
    Leff_old = np.sqrt(cons[1]**2 + cons[2]) * L_old_dir 
    Leff_new = np.linalg.norm((mu*Leff_old + L_disk_vec) / M_new)
    dC = Leff_new**2 - (cons[1] + dLz)**2 - cons[2]

    return np.array([dE, dLz, dC])

def thickdisk_interaction(disk, trajectory, a, mu, cons, n=2):
    radii = trajectory[:, 1]
    carts = mm.sph2cart_vec(trajectory, a)
    diskmask = disk.isinDisk(carts, radii)
    dists = np.linalg.norm(np.diff(carts, axis=0)[:, 1:4], axis=-1)[diskmask]
    densities = disk.volume_density(carts[diskmask], radii[diskmask], n)
    dm = densities * 16.0*np.pi*(mu**2) * dists
    dm_total = np.sum(dm)
    if dm_total <= 0:
        return np.zeros(3)
    M_new = mu + dm_total

    oris = np.cross(carts[diskmask, 1:4], carts[diskmask, 5:8])
    oris = oris / np.linalg.norm(oris, axis=1)[:, None]
    incs = np.arctan2(np.hypot(oris[:, 0], oris[:, 1]), oris[:, 2])
    E, L, C = mm.schmidtcircle(radii[diskmask], incs, np.full_like(incs, a))

    # Changes in specific E and Lz
    dE = (np.dot(dm, E) - dm_total*cons[0]) / M_new
    dLz = (np.dot(dm, L) - dm_total*cons[1]) / M_new

    # Approximate disk angular momentum vectors
    Leff = np.sqrt(L**2 + C)
    L_disk_vec = np.sum(oris * (Leff*dm)[:, None], axis=0)

    # Estimate secondary angular momentum direction
    pos = carts[-1, 1:4]
    vel = carts[-1, 5:8]
    L_old_dir = oris[-1]
    L_old_dir /= np.linalg.norm(L_old_dir)
    Leff_old = np.sqrt(cons[1]**2 + cons[2]) * L_old_dir 
    Leff_new = np.linalg.norm((mu*Leff_old + L_disk_vec) / M_new)
    dC = Leff_new**2 - (cons[1] + dLz)**2 - cons[2]

    return np.array([dE, dLz, dC])
