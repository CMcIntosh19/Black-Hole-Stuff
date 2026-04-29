import MetricMathStreamline as mm
import MainLoopStreamline as ml
import OrbitPlotter as op
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import warnings
import importlib
from tqdm import tqdm
importlib.reload(ml)
importlib.reload(mm)
importlib.reload(op)
import time
from multiprocessing import Process
from itertools import product
import os
from multiprocessing import Pool, cpu_count
import pickle

def all_dcons(states, a, mu, cons, ind1, ind2):
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
    if (ind2 - ind1 - 10) > 2:
        E0, L0, C0 = cons
        states = np.array(states)
        sphere, time = states[:, 1:4], states[:, 0] - states[0,0]
        int_sphere, int_time = mm.interpolate3(sphere, time)
        div = np.mean(np.diff(int_time))
        quad = mm.trace_ortholize_njit(int_sphere, a)
        coolquad = mm.traceless_quad(quad)
        dt = np.mean(np.diff(int_time))
        dt2, dt3 = mm.matrix_derive3_numba(coolquad, dt) 
        ref_r = int_sphere[-5, 0]
        drdt = np.zeros_like(int_time)
        drdt[1:-1] = (int_sphere[2:, 0] - int_sphere[:-2, 0]) / (int_time[2:] - int_time[:-2])
        ref_drdt = drdt[-5]
        r_scale = np.std(int_sphere[:, 0])
        v_scale = np.std(drdt)
        phase_space_diffs = ((int_sphere[:, 0] - ref_r)/r_scale)**2 + ((drdt - ref_drdt)/v_scale)**2

        N = 12
        sorted_ix = np.argsort(phase_space_diffs)   # Sort the indices of the smallest diff values (smallest values first)
        candidates = sorted_ix[:N]                  # Grab the first 12   (best 12 matches)
        ref_ix = np.min(candidates)                 # Grab the first of the best

        dt2 = dt2[ref_ix:-5]
        dt3 = dt3[ref_ix:-5]
        int_time = int_time[ref_ix:-5]
        int_sphere = int_sphere[ref_ix:-5]
        dedt = (-1/5)*(np.einsum('ijk,ijk ->i', dt3, dt3) - (1/3)*np.einsum('ijj,ikk ->i', dt3, dt3))
        dldt = mm.compute_dldt(dt2, dt3)
        if a == 0:
            z2 = C0/(L0**2 + C0)
        else:
            A = (a**2)*(1 - E0**2)
            sig = A + L0**2 + C0
            z2 = (sig - (sig**2 - 4*A*C0)**(1/2))/(2*A)
        
        dE = mu*mu*np.sum(dedt*div)
        dLx, dLy, dLz = mu*mu*np.sum(dldt*div, axis=0)

        P = E0*(int_sphere[:, 0]**2 + a*a) - a*L0
        D = int_sphere[:, 0]*(int_sphere[:, 0] - 2) + a*a
        dcdt_polar = (-2*a*a*z2*E0*dedt + 2*z2*L0*dldt[:, 2]/(1 - z2))*div*mu*mu
        dcdt_radial = ((2*P*(int_sphere[:, 0]**2 + a*a)/D - 2*a*(a*E0 - L0))*dedt + (-2*P*a/D + 2*(a*E0 - L0))*dldt[:, 2])*div*mu*mu
        return np.array([dE, dLx, dLy, dLz, np.sum(dcdt_polar), np.sum(dcdt_radial)])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
def get_dC_stuff2(a, p, e, inc):
    skip = False

    try:
        r_start, the_start = np.random.uniform(p/(1 + e), p/(1 - e)), np.random.uniform(abs(inc), np.pi - abs(inc))
        data = ml.EMRIGenerator(a, 0.0, "rad_orbit > 10", params=[p/(1 - e**2), e, inc], pos=[0.0, r_start, the_start, 0.0], verbose=0, label="mark5")
        E0, L0, C0 = data["energy"][0], data["phi_momentum"][0], data["carter"][0]
        if data["plunge"] == True or data["stop"] == True:
            skip = True
    except:
        skip = True
        
    if not skip:
        # Calculate dE, dL_vec, and dC for the EPTD integrated along the orbit
        start, end = 0, data["trackix"][-1]
        dE, dLx, dLy, dLz, dC_polar_int, dC_radial_int = all_dcons(data["raw"][start:end], a, 1e-5, [E0, L0, C0], start, end)

        # Calculate dC for LS method
        _, new_cons = mm.new_recalc_state8d([E0, L0, C0], [dE, dLx, dLy, dLz], data["raw"][-1], a)
        dC_least_squares = new_cons[-1] - C0

        # Calculate dC for MLS method   THE DIFFERENT LEAST SQUARES STUFF??
        _, new_cons = mm.new_recalc_state8d([E0, L0, C0], [dE, dLx, dLy, dLz], data["raw"][-1], a, svd=True)
        dC_mod_least_squares = new_cons[-1] - C0

        # Calculate dC for GD method
        _, new_cons = mm.new_recalc_state9l([E0, L0, C0], [dE, dLx, dLy, dLz], data["raw"][-1], a)
        dC_mod_Glamp_derv = new_cons[-1] - C0

        # Calculate dC for MGD method
        _, new_cons = mm.new_recalc_state9j([E0, L0, C0], [dE, dLx, dLy, dLz], data["raw"][-1], a)
        dC_Glamp_derv = new_cons[-1] - C0

        # Calculate z = cos(theta_min) and x = sin(theta_min)
        if a == 0:
            z2 = C0/(L0**2 + C0)
        else:
            A = (a**2)*(1 - E0**2)
            sig = A + L0**2 + C0
            z2 = (sig - (sig**2 - 4*A*C0)**(1/2))/(2*A)
        z = np.sqrt(min(1.0, np.abs(z2)))
        x2 = 1 - z2
        x = np.sign(L0) * np.sqrt(x2)

        #Calculate dC from Polar EPTD
        dC_dE = -2*a*a*z2*E0
        dC_dLz = 2*z2*L0/(1 - z2)
        dC_dx = - 2*(a*a*x*(1 - E0*E0) + L0*L0/(x**3))
        dC0 = dC_dE*dE + dC_dLz*dLz
        dx = dC0 + 2*(1 - x2)*a*a*E0*dE - 2*L0*dLz*(1 - x2)/x2
        dx /= -2*(x*(a*a*(1 - E0*E0) + L0*L0/x2) + L0*L0*(1 - x2)/(x**3))
        dC_polar = dC0 + dC_dx*dx

        #Calculate dC from Polar EPTD (explicit difference)
        dC_dE = -2*a*a*z2*E0
        dC_dLz = 2*z2*L0/(1 - z2)
        dC0 = dC_dE*dE + dC_dLz*dLz
        A2 = (a**2)*(1 - (E0 + dE)**2)
        sig2 = A2 + (L0 + dLz)**2 + C0 + dC0
        z2_2 = (sig2 - (sig2**2 - 4*A2*(C0 + dC0))**(1/2))/(2*A2)
        dx = np.sign(L0 + dLz)*np.sqrt(1 - z2_2) - x
        dC_polar_diff = dC0 + dC_dx*dx

        #Calculate dC from Radial EPTD (at potential minimum)
        poly = np.array([4*(E0**2 - 1), 3*2, 2*((a**2)*(E0**2 - 1) - L0**2 - C0), 2*((a*E0 - L0)**2 + C0)])
        r = np.real(max(np.roots(poly)))
        P, D = E0*(r*r + a*a) - a*L0, r*r - 2*r + a*a
        dC_dE = 2*P*(r*r + a*a)/D - 2*a*(a*E0 - L0)
        dC_dLz = -2*a*P/D + 2*(a*E0 - L0)
        dC0 = dC_dE*dE + dC_dLz*dLz
        dC_dr = (4*E0*r*P - P*P*(2*r - 2))/(D*D) - 2*r
        poly2 = np.array([4*((E0 + dE)**2 - 1), 3*2, 2*((a**2)*((E0 + dE)**2 - 1) - (L0 + dLz)**2 - (C0 + dC0)), 2*((a*(E0 + dE) - (L0 + dLz))**2 + (C0 + dC0))])
        dr = np.real(max(np.roots(poly2))) - r
        dC_radial_pot_min = dC0 + dC_dr*dr

        #Calculate dC from Radial EPTD (at apoapse)
        poly = np.array([(E0**2 - 1), 2, ((a**2)*(E0**2 - 1) - L0**2 - C0), 2*((a*E0 - L0)**2 + C0), -C0*(a**2)])
        r = np.real(max(np.roots(poly)))
        P, D = E0*(r*r + a*a) - a*L0, r*r - 2*r + a*a
        dC_dE = 2*P*(r*r + a*a)/D - 2*a*(a*E0 - L0)
        dC_dLz = -2*a*P/D + 2*(a*E0 - L0)
        dC0 = dC_dE*dE + dC_dLz*dLz
        dC_dr = (4*E0*r*P - P*P*(2*r - 2))/(D*D) - 2*r
        poly2 = np.array([((E0 + dE)**2 - 1), 2, ((a**2)*((E0 + dE)**2 - 1) - (L0 + dLz)**2 - (C0 + dC0)), 2*((a*(E0 + dE) - (L0 + dLz))**2 + (C0 + dC0)), -(C0 + dC0)*(a**2)])
        dr = np.real(max(np.roots(poly2))) - r
        dC_radial_apo = dC0 + dC_dr*dr

        #Calculate dC from Radial EPTD (at periapse)
        poly = np.array([(E0**2 - 1), 2, ((a**2)*(E0**2 - 1) - L0**2 - C0), 2*((a*E0 - L0)**2 + C0), -C0*(a**2)])
        r = np.sort(np.real(np.roots(poly)))[-2]
        P, D = E0*(r*r + a*a) - a*L0, r*r - 2*r + a*a
        dC_dE = 2*P*(r*r + a*a)/D - 2*a*(a*E0 - L0)
        dC_dLz = -2*a*P/D + 2*(a*E0 - L0)
        dC0 = dC_dE*dE + dC_dLz*dLz
        dC_dr = (4*E0*r*P - P*P*(2*r - 2))/(D*D) - 2*r
        poly2 = np.array([((E0 + dE)**2 - 1), 2, ((a**2)*((E0 + dE)**2 - 1) - (L0 + dLz)**2 - (C0 + dC0)), 2*((a*(E0 + dE) - (L0 + dLz))**2 + (C0 + dC0)), -(C0 + dC0)*(a**2)])
        dr = np.sort(np.real(np.roots(poly)))[-2] - r
        dC_radial_peri = dC0 + dC_dr*dr

        #Calculate dC from Radial EPTD (at semilatus rectum)
        poly = np.array([(E0**2 - 1), 2, ((a**2)*(E0**2 - 1) - L0**2 - C0), 2*((a*E0 - L0)**2 + C0), -C0*(a**2)])
        rp, ra = np.sort(np.real(np.roots(poly)))[-2:]
        r = 2 * rp * ra / (rp + ra)
        P, D = E0*(r*r + a*a) - a*L0, r*r - 2*r + a*a
        dC_dE = 2*P*(r*r + a*a)/D - 2*a*(a*E0 - L0)
        dC_dLz = -2*a*P/D + 2*(a*E0 - L0)
        dC0 = dC_dE*dE + dC_dLz*dLz
        dC_dr = (4*E0*r*P - P*P*(2*r - 2))/(D*D) - 2*r
        poly2 = np.array([((E0 + dE)**2 - 1), 2, ((a**2)*((E0 + dE)**2 - 1) - (L0 + dLz)**2 - (C0 + dC0)), 2*((a*(E0 + dE) - (L0 + dLz))**2 + (C0 + dC0)), -(C0 + dC0)*(a**2)])
        rp, ra = np.sort(np.real(np.roots(poly2)))[-2:]
        r2 = 2 * rp * ra / (rp + ra)
        dr = r2 - r
        dC_radial_semilat = dC0 + dC_dr*dr

        del data
        return [E0, L0, C0,                 # initial constants (0-2)
                dE, dLx, dLy, dLz,          # changes calculated from Peters (1964)  (3 - 6)
                dC_least_squares,           # 7
                dC_mod_least_squares,       # 8
                dC_Glamp_derv,              # 9
                dC_mod_Glamp_derv,          # 10
                dC_polar,                   # 11
                dC_polar_diff,              # 12
                dC_polar_int,               # 13
                dC_radial_pot_min,          # 14
                dC_radial_apo,              # 15
                dC_radial_peri,             # 16
                dC_radial_semilat,          # 17
                dC_radial_int]              # 18
    else:
        return np.nan*np.arange(19)


# Generate list of inclinations of length val
val = 50
incs = np.linspace(np.pi/2, -np.pi/2, val)
a_s = [0.1, 0.3, 0.5, 0.7, 0.9]
e_s = [0.9, 0.5, 0.1, 1e-4]

'''
sep_cache = {}
# Create dict
big_dict = {}
# Save it all special like
big_dict["incs"] = incs


pairs = list(product(a_s, e_s))

for a, e in pairs:
    # If we already have this one, skip it
    p_grid = [None] * val 
    dC_grid = [[None] * val for _ in range(val)]
    for i, inc in enumerate(incs):
        # get_sep_inc generates lists of semilatus recta, eccentricities, and radii corresponding to Kerr separatrix for a given spin and inclination
        key = (a, inc)
        if key not in sep_cache:
            sep_cache[key] = mm.get_sep_inc(a, inc)
        p_sep, e_sep, _ = sep_cache[key]
        # Get the semilatus value that corresponds to the given eccentricity! Values of p smaller than this plunge, so we don't include them
        p_min = p_sep[op.get_index(e_sep, e)]
        # For a central body M = 10^6 M_sun, maximum radius/semilatus rectum in LISA range is ~75 r_g
        # Make it denser lower in cause interesting stuff happens there
        p_row = np.geomspace(p_min, 75, val)
        p_grid[i] = p_row
        for j, p in enumerate(p_row):
            # get_dC_stuff2 returns a 19 element list that's either a bunch of nans (if the simulated orbit fails for whatever reason)
            #  OR constants of motion(E, L_z, C), dE and dL_vec as calculated by Peters, and a bunch of different ways to calculate dC
            dC_grid[i][j] = get_dC_stuff2(a, p, e, inc)
    
    # Turn them into numpy arrays!
    p_grid = np.array(p_grid)
    dC_grid = np.array(dC_grid)
    # Save to the big dict!
    big_dict[a, e] = [p_grid, dC_grid]
    # Get rid of arrays to save space while we keep going! 
    del p_grid, dC_grid
'''

def compute_pair(args):
    a, e, incs, val = args
    p_grid = [None] * val
    dC_grid = [[None] * val for _ in range(val)]

    for i, inc in enumerate(incs):
        sep = mm.get_sep_inc(a, inc)
        p_min = sep["p"][op.get_index(sep["e"], e)]

        p_row = np.geomspace(p_min + 0.1, 75, val)
        p_grid[i] = p_row

        for j, p in enumerate(p_row):
            dC_grid[i][j] = get_dC_stuff2(a, p, e, inc)

    return (a, e, np.array(p_grid), np.array(dC_grid))

def compute_point(args):
    a, e, inc = args

    # separatrix
    sep = mm.get_sep_inc(a, inc)
    p_min = sep["p"][op.get_index(sep["e"], e)]

    p_row = np.geomspace(p_min + 0.1, 75, val)
    print(f"Running point a={a:.4f}, e={e:.4f}, inc={inc/np.pi:4f}", flush=True)
    dC_row = [get_dC_stuff2(a, p, e, inc) for p in p_row]
    print(f"-------------- Finished point a={a:.4f}, e={a:.4f}, inc={inc/np.pi:4f}", flush=True)

    return a, e, inc, p_row, dC_row

'''
def main():

    pairs = [(a, e, incs, val) for a in a_s for e in e_s]
    max_workers = min(len(pairs), os.cpu_count() - 1)

    print("Preparing workers.", flush=True)

    processes = []
    for pair in pairs:
        proc = Process(target=compute_pair, args=pair)
        processes.append(proc)
        proc.start()

    try:
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        for proc in processes:
            proc.join()
'''

'''
def main():
    pairs = [(a, e, incs, val) for a in a_s for e in e_s]

    with Pool(cpu_count()) as pool:
        results = pool.map(compute_pair, pairs)

    big_dict = {"incs": incs}

    for a, e, p_grid, dC_grid in results:
        big_dict[(a, e)] = [p_grid, dC_grid]
'''

def main_old():
    start = time.perf_counter()
    tasks = [(a, e, inc) for a, e, inc in product(a_s, e_s, incs)]

    with Pool(cpu_count()) as pool:
        results = pool.map(compute_point, tasks)

    big_dict = {"incs": incs}

    # organize results
    temp = {}

    for a, e, inc, p_row, dC_row in results:
        key = (a, e)
        if key not in temp:
            temp[key] = {}

        temp[key][inc] = (p_row, dC_row)

    # rebuild arrays in correct order
    for (a, e), inc_data in temp.items():
        p_grid = []
        dC_grid = []

        for inc in incs:  # preserves ordering
            p_row, dC_row = inc_data[inc]
            p_grid.append(p_row)
            dC_grid.append(dC_row)

        big_dict[(a, e)] = [np.array(p_grid), np.array(dC_grid)]

    end = time.perf_counter()
    
    for key in big_dict.keys():
        print(key)
        if type(key) != str:
            print("---", np.shape(big_dict[key][1]))

    print(f"runtime: {end - start}")

def main():

    run_count, tick = 0, 0

    start = time.perf_counter()

    tasks = [(a, e, inc) for a, e, inc in product(a_s, e_s, incs)]

    big_dict = {"incs": incs}
    temp = {}

    completed_pairs = set()

    n_workers = max(1, cpu_count() - 2)
    with Pool(n_workers) as pool:
        for a, e, inc, p_row, dC_row in pool.imap_unordered(compute_point, tasks, chunksize=2):

            key = (a, e)
            run_count += 1

            if key not in temp:
                temp[key] = {}

            temp[key][inc] = (p_row, dC_row)

            if 100*run_count/len(tasks) - tick >= 1:
                tick += int(100*run_count/len(tasks) - tick)
                print(f"{tick}% of tasks completed")

            if len(temp[key]) == len(incs) and key not in completed_pairs:
                p_grid = []
                dC_grid = []

                for inc_val in incs:  # preserve ordering
                    p_r, dC_r = temp[key][inc_val]
                    p_grid.append(p_r)
                    dC_grid.append(dC_r)

                p_grid = np.array(p_grid)
                dC_grid = np.array(dC_grid)

                big_dict[key] = [p_grid, dC_grid]

                completed_pairs.add(key)

                # free memory
                del temp[key]

                # 💾 Periodic full checkpoint
                if len(completed_pairs) % 3 == 0:
                    with open("D:/EMRIData/checkpoint.pkl", "wb") as f:
                        pickle.dump(big_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Checkpoint saved: {len(completed_pairs)} of {len(a_s)*len(e_s)} keys completed")

    # Final save
    with open("D:/EMRIData/final_big_dict.pkl", "wb") as f:
        pickle.dump(big_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    end = time.perf_counter()

    print(f"runtime: {end - start}")

if __name__ == "__main__":
    main()

