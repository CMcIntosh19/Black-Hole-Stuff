import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal import find_peaks
import MetricMathStreamline as mm
import MainLoopStreamline as ml
import OrbitPlotter as op
import time
from numba import njit
import importlib
importlib.reload(ml)
importlib.reload(mm)
importlib.reload(op)
import cProfile
import pstats
import matplotlib.colors as mcolors






# ==============================
# INPUTS
# ==============================
gorf = ml.load_index()
data = []
seps = []
cutoff = 150
for key, val in gorf.items():
    try:
        if "Quasi-Circular" in val["Label"]:
            print(val["Label"])
            if len(data) == 0:
                spin = val["Spin"]
            #if len(data) >= 10:
            #    break
            try:
                num = int(val["Label"].split()[-1])
            except:
                num = 1
            if cutoff - num >= 0:
                hold = ml.load_emri_data(key, reconstruct=False)
                #print("a1")
                seps.append(hold["raw"][-1,0])
                #print("a2")
                if num == 1 or num == cutoff:
                    #print("a3")
                    op.justfourier(hold, m_bh=1e7)
                    #print("a4")
                data.append(hold["raw"][:, :4])
                #print(np.shape(data))
                #print("a9")
                del hold
    except Exception as e:
        print(e)
        break
del gorf
#op.justfourier(data)
#op.justfourier(data, m_bh=1e7)
data = np.concatenate(data)

def generate_strain_from_bl(bl_data, spin, distance, M_BH=1e7, fs=False):
    """
    Generate GW strain from Boyer-Lindquist geodesic data.

    Parameters
    ----------
    bl_data : (N,8) array
        Columns: [t, r, theta, phi, u_t, u_r, u_theta, u_phi]
    spin : float
        Dimensionless spin of central body
    distance : float
        Observer distance (geometric units)
    M_BH : float
        Large body mass (Solar Masses)

    Returns
    -------
    dt: ee
    T_r: tgyhfdj
    t_uniform : (M,)
    h_plus : (M,)
    h_cross : (M,)
    """
    bl_data = np.asarray(bl_data)
    M_sun_seconds = 4.92550613054e-6
    M_sun_meters  = 1.47662958977e3

    # Extract columns and scale to units
    t_arr = bl_data[:,0] * M_sun_seconds * M_BH
    r_arr = bl_data[:,1] * M_sun_meters * M_BH
    th_arr = bl_data[:,2]
    ph_arr = bl_data[:,3]
    distance = distance * M_sun_meters * M_BH

    # Ensure strictly increasing time
    sort_idx = np.argsort(t_arr)
    t_arr = t_arr[sort_idx]
    r_arr = r_arr[sort_idx]
    th_arr = th_arr[sort_idx]
    ph_arr = ph_arr[sort_idx]

    # Radial Frequency
    if np.allclose(r_arr, r_arr[0]):
        f_r = 0.0
        T_r = np.inf
    else:
        peaks_r, _ = find_peaks(r_arr)
        T_r = np.mean(np.diff(t_arr[peaks_r]))
        f_r = 1 / T_r
    # Theta Frequency
    if np.allclose(th_arr, th_arr[0]):
        f_theta = 0.0
        T_theta = np.inf
    else:
        peaks_th, _ = find_peaks(th_arr)
        T_theta = np.mean(np.diff(t_arr[peaks_th]))
        f_theta = 1 / T_theta
    # Phi Frequency
    f_phi = np.abs(ph_arr[-1] - ph_arr[0]) / (2*np.pi*(t_arr[-1] - t_arr[0]))
    T_phi = 1/f_phi
    f_gw_max = 2 * f_phi + 2 * f_theta
    fs_target = 8 * f_gw_max
    dt = 1 / fs if fs else 1 / fs_target
    t_uniform = np.arange(t_arr[0], t_arr[-1], dt)

    M = len(t_uniform)
    h_plus = np.zeros(M)
    h_cross = np.zeros(M)

    j = np.searchsorted(t_arr, t_uniform, side='right') - 1
    w = (t_uniform - t_arr[j]) / (t_arr[j + 1] - t_arr[j])

    # Linear interpolation in BL coords
    r  = r_arr[j]  + w*(r_arr[j+1]  - r_arr[j])
    th = th_arr[j] + w*(th_arr[j+1] - th_arr[j])
    ph = ph_arr[j] + w*(ph_arr[j+1] - ph_arr[j])

    # Convert BL -> Cartesian (flat-space embedding)
    sin_th = np.sin(th)
    cos_th = np.cos(th)
    sin_ph = np.sin(ph)
    cos_ph = np.cos(ph)

    x = np.sqrt(r*r + spin*spin) * sin_th * cos_ph
    y = np.sqrt(r*r + spin*spin) * sin_th * sin_ph
    z = r * cos_th

    # Quadrupole second derivative (leading order)
    r2 = x*x + y*y + z*z
    qmom = np.transpose([[3*x*x - r2, 3*x*y,      3*x*z],
                         [3*y*x,      3*y*y - r2, 3*y*z],
                         [3*z*x,      3*z*y,      3*z*z - r2]])
    
    qmom_d2 = np.zeros_like(qmom)
    for i in range(3):
        for j in range(3):
            f = qmom[:, i, j]
            for k in range(3, M-3):  # Avoid boundaries for 7-point stencil
                qmom_d2[k, i, j] = (-f[k+2] + 16*f[k+1] - 30*f[k] + 16*f[k-1] - f[k-2]) / (12 * dt**2)

    h_plus  = (qmom_d2[:, 0, 0] - qmom_d2[:, 1, 1]) / distance
    h_cross = (2*qmom_d2[:, 0, 1]) / distance

    # fast-scale lower bound
    T_min = 5 * T_phi
    # modulation upper bound
    T_max = 0.5 * T_r if np.isfinite(T_r) else 10 * T_phi

    return dt, min(max(T_min, 0.1*(t_arr[-1] - t_arr[0])), T_max), t_uniform, h_plus, h_cross

dt, T_window, t_geom, h, h2 = generate_strain_from_bl(data, spin, 10000, fs=5e-3)

# Sampling frequency
fs = 1.0 / dt
print(f"Sampling freq = {fs:.4e} Hz")

# ==============================
# Compute spectrogram
# ==============================


nper = int(10 * T_window * fs)
nper = 2**int(np.log2(nper))

f, t_spec, Sxx_plus = spectrogram(
    h,
    fs=fs,
    window='hann',
    nperseg=nper,
    noverlap=int(0.85 * nper),
    detrend='constant',
    mode='psd'
)

_, _, Sxx_cross = spectrogram(
    h2,
    fs=fs,
    window='hann',
    nperseg=nper,
    noverlap=int(0.85 * nper),
    detrend='constant',
    mode='psd'
)
t_spec += t_geom[0] - t_spec[0]

print(f"data times: {data[0,0]:.5e} -> {data[-1,0]:.5e}")
print(f"t_geom times: {t_geom[0]:.5e} -> {t_geom[-1]:.5e}")
print(f"t_spec times: {t_spec[0]:.5e} -> {t_spec[-1]:.5e}")

# ==============================
# Plot
# ==============================
df = f[1] - f[0]
print("Frequency resolution:", df)



'''plt.figure(figsize=(8,6))
#plt.pcolormesh(t_spec, f, Sxx_plus, shading='gouraud')
plt.imshow(Sxx_plus, aspect='auto', origin='lower', extent=[t_spec[0], t_spec[-1], f[0], f[-1]])
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.title(f"Gravitational Wave Spectrogram")
#plt.colorbar(label="Strain amplitude")
#plt.ylim(0, 1024)   # adjust as needed
[plt.axvline(thing["time"][-1]*4.92550613054e-6*1e7, color="k") for thing in data]
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(8,6))
#plt.pcolormesh(t_spec, f, np.log10(Sxx_plus), shading='gouraud')
plt.imshow(np.log10(Sxx_plus), aspect='auto', origin='lower', extent=[t_spec[0], t_spec[-1], f[0], f[-1]])
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.title(f"Gravitational Wave Spectrogram")
plt.colorbar(label="Strain amplitude")
#plt.ylim(0, 1024)   # adjust as needed
[plt.axvline(thing["time"][-1]*4.92550613054e-6*1e7, color="k") for thing in data]
plt.tight_layout()
plt.show(block=False)'''

S_plus = np.log10(Sxx_plus + 1e-20)
S_cross = np.log10(Sxx_cross + 1e-20)

'''def normalize(X):
    X = X - np.min(X)
    X = X / np.max(X)
    return X

R = normalize(S_plus)
B = normalize(S_cross)
G = normalize(np.log10(Sxx_plus + Sxx_cross + 1e-20))

rgb = np.zeros((R.shape[0], R.shape[1], 3))
rgb[:,:,0] = R   # Red = h+
rgb[:,:,2] = B   # Blue = hx
rgb[:,:,1] = G   # Green = total power

plt.figure(figsize=(8,6))
plt.imshow(
    rgb,
    aspect='auto',
    origin='lower',
    extent=[t_spec[0], t_spec[-1], f[0], f[-1]]
)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("Polarization-Colored Spectrogram")
#[plt.axvline(thing["time"][-1]*4.92550613054e-6*1e7, color="k") for thing in data]
plt.colorbar()  # optional, though RGB doesn't map linearly
plt.tight_layout()
plt.show(block=False)'''

def normalize_percentile(X, low=5, high=99):
    lo = np.percentile(X, low)
    hi = np.percentile(X, high)
    X = np.clip(X, lo, hi)
    return (X - lo) / (hi - lo)

'''R = normalize_percentile(S_plus, 5, 99)
B = normalize_percentile(S_cross, 5, 99)
G = normalize_percentile(np.log10(Sxx_plus + Sxx_cross + 1e-20), 5, 99)

rgb = np.zeros((R.shape[0], R.shape[1], 3))
rgb[:,:,0] = R   # Red = h+
rgb[:,:,2] = B   # Blue = hx
rgb[:,:,1] = G   # Green = total power

plt.figure(figsize=(8,6))
plt.imshow(
    rgb,
    aspect='auto',
    origin='lower',
    extent=[t_spec[0], t_spec[-1], f[0], f[-1]]
)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("Polarization-Colored Spectrogram (Percentile)")
#[plt.axvline(thing["time"][-1]*4.92550613054e-6*1e7, color="k") for thing in data]
plt.colorbar()  # optional, though RGB doesn't map linearly
plt.tight_layout()
plt.show(block = False)'''

Sxx = Sxx_plus + Sxx_cross

Sxx_log = np.log10(Sxx + 1e-20)

def normalize_floor(X, floor_db=50):
    X_max = np.max(X)
    X_min = X_max - floor_db
    X = np.clip(X, X_min, X_max)
    return (X - X_min) / (X_max - X_min)

V = normalize_percentile(Sxx_log)
psi = np.arctan2(Sxx_cross, Sxx_plus)*(240/355)
H = 2 * psi / np.pi
print("psi and H")
print(Sxx_plus.min(), Sxx_plus.max())
print(Sxx_cross.min(), Sxx_cross.max())
print(psi.min(), psi.max())
print(H.min(), H.max())
pol_fraction = np.abs(Sxx_plus - Sxx_cross) / (Sxx + 1e-20)
S = pol_fraction

hsv = np.zeros((H.shape[0], H.shape[1], 3))
hsv[:,:,0] = H   # Hue
hsv[:,:,1] = S   # Saturation
hsv[:,:,2] = V   # Brightness

rgb = mcolors.hsv_to_rgb(hsv)

plt.figure(figsize=(8,6))
plt.imshow(
    rgb,
    aspect='auto',
    origin='lower',
    extent=[t_spec[0], t_spec[-1], f[0], f[-1]]
)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("Polarization-Colored Spectrogram (HSV)")
#[plt.axvline(sep*4.92550613054e-6*1e7, color="red") for sep in seps]
plt.colorbar()  # optional, though RGB doesn't map linearly
plt.tight_layout()
plt.show(block=False)

V = normalize_percentile(Sxx)
psi = np.arctan2(Sxx_cross, Sxx_plus)*(240/355)
H = 2 * psi / np.pi
pol_fraction = np.abs(Sxx_plus - Sxx_cross) / (Sxx + 1e-20)
S = pol_fraction

hsv = np.zeros((H.shape[0], H.shape[1], 3))
hsv[:,:,0] = H   # Hue
hsv[:,:,1] = S   # Saturation
hsv[:,:,2] = V   # Brightness

rgb = mcolors.hsv_to_rgb(hsv)

plt.figure(figsize=(8,6))
plt.imshow(
    rgb,
    aspect='auto',
    origin='lower',
    extent=[t_spec[0], t_spec[-1], f[0], f[-1]]
)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("Polarization-Colored Spectrogram (HSV) (non-log)")
#[plt.axvline(thing["time"][-1]*4.92550613054e-6*1e7, color="k") for thing in data]
plt.colorbar()  # optional, though RGB doesn't map linearly
plt.tight_layout()
plt.show()