import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.signal import find_peaks
import scipy.interpolate as spi
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

def generate_strain_from_bl(bl_data, spin, distance, M_BH=1e7, e_r=None, fs=False):
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
    G, c = 6.67e-11, 3e8
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
    fs_target = max(20 * f_gw_max, 10 / T_phi)
    dt = 1 / fs if fs else 1 / fs_target
    print(f"Default sampling frequency = {fs_target}; manual sampling freqency = {fs}")
    t_uniform = np.arange(t_arr[0], t_arr[-1], dt)

    M = len(t_uniform)
    h_plus = np.zeros(M)
    h_cross = np.zeros(M)

    j = np.searchsorted(t_arr, t_uniform, side='right') - 1
    j = np.clip(j, 0, len(t_arr)-2)
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

    carts = np.zeros((len(t_uniform), 3))
    carts[:, 0] = np.sqrt(r*r + spin*spin) * sin_th * cos_ph
    carts[:, 1] = np.sqrt(r*r + spin*spin) * sin_th * sin_ph
    carts[:, 2] = r * cos_th
    r2 = np.sum(carts * carts, axis=1)

    # Quadrupole second derivative
    qmom = carts[:, :, None] * carts[:, None, :]
    qmom -= np.eye(3)[None,:,:] * (r2[:,None,None]/3)
    
    qmom_d2 = np.zeros_like(qmom)
    for i in range(3):
        for j in range(3):
            spline = spi.CubicSpline(t_uniform, qmom[:,i,j])
            qmom_d2[:, i, j] = spline(t_uniform, 2)

    if e_r is None:
        e_r = [np.sin(np.pi/3), np.cos(np.pi/3), 0]
    e_r = np.array(e_r) / np.linalg.norm(e_r)
    # choose any vector not parallel to e_r
    if np.allclose(e_r, [0,0,1]):
        ref = np.array([0,1,0])
    else:
        ref = np.array([0,0,1])
    e_th = ref - np.dot(ref, e_r)*e_r
    e_th /= np.linalg.norm(e_th)
    e_ph = np.cross(e_th, e_r)
    P = np.eye(3) - np.outer(e_r, e_r)

    term = P @ qmom_d2 @ P        # works in NumPy ≥1.16
    trace = np.einsum('ij,kji->k', P, qmom_d2)
    h_TT = term - 0.5 * trace[:, None, None] * P
    h_TT *= 2/distance

    h_plus = 0.5 * (np.einsum("i, j, kij -> k", e_th, e_th, h_TT) - np.einsum("i, j, kij -> k", e_ph, e_ph, h_TT))
    h_cross = 0.5 * (np.einsum("i, j, kij -> k", e_th, e_ph, h_TT) + np.einsum("i, j, kij -> k", e_ph, e_th, h_TT))

    # fast-scale lower bound
    T_min = 5 * T_phi
    # modulation upper bound
    T_max = 0.5 * T_r if np.isfinite(T_r) else 10 * T_phi

    return dt, min(max(T_min, 0.1*(t_arr[-1] - t_arr[0])), T_max), t_uniform, h_plus, h_cross

gorf = ml.load_index()

spin = None
mass = 1e7 # solar masses
cutoff = 100
f_big, t_spec_big, Sxx_plus_big, Sxx_cross_big = [], [], [], []
for key, val in gorf.items():
    try:
        if "Near-Polar" not in val["Label"]:
            continue

        if spin is None:
            spin = val["Spin"]

        try:
            num = int(val["Label"].split()[-1])
        except:
            num = 1

        print(val["Label"])

        if cutoff - num < 0:
            continue

        hold = ml.load_emri_data(key, quiet=True, reconstruct=False)

        raw = hold["raw"][:, :4]

        if num == 1 or num == cutoff:
            op.justfourier(hold, m_bh=mass)
        del hold
        dt, T_window, t_geom, h_plus, h_cross = generate_strain_from_bl(raw, spin, 10000, M_BH = mass)
        del raw
        fs = 1.0 / dt
        nper = int(10 * T_window * fs)
        nper = 2**int(np.log2(nper))
        f, t_spec, Sxx_plus = spectrogram(
            h_plus,
            fs=fs,
            window='hann',
            nperseg=nper,
            noverlap=int(0.85 * nper),
            detrend='constant',
            mode='psd'
        )
        print(np.shape(Sxx_plus))
        t_spec += t_geom[0] - t_spec[0]
        f_big.append(f)
        t_spec_big.append(t_spec)
        Sxx_plus_big.append(Sxx_plus)
        del f, t_spec, Sxx_plus, t_geom, h_plus

        _, _, Sxx_cross = spectrogram(
            h_cross,
            fs=fs,
            window='hann',
            nperseg=nper,
            noverlap=int(0.85 * nper),
            detrend='constant',
            mode='psd'
        )
        Sxx_cross_big.append(Sxx_cross)
        del Sxx_cross, h_cross

    except Exception as e:
        print(e)
        break

del gorf
f_big = np.concatenate(f_big)
t_spec_big = np.concatenate(t_spec_big)
Sxx_plus_big = np.concatenate(Sxx_plus_big, axis=1)
Sxx_cross_big = np.concatenate(Sxx_cross_big, axis=1)

# Sampling frequency
df = f_big[1] - f_big[0]
print("Frequency resolution:", df)

S_plus = np.log10(Sxx_plus_big + 1e-20)
S_cross = np.log10(Sxx_cross_big + 1e-20)

def normalize_percentile(X, low=5, high=99):
    lo = np.percentile(X, low)
    hi = np.percentile(X, high)
    X = np.clip(X, lo, hi)
    return (X - lo) / (hi - lo)

V = normalize_percentile(np.log10(Sxx_plus_big + Sxx_cross_big + 1e-20))
psi = np.arctan2(Sxx_cross_big, Sxx_plus_big)*(240/355)
H = 2 * psi / np.pi
S = np.abs(Sxx_plus_big - Sxx_cross_big) / (Sxx_plus_big + Sxx_cross_big + 1e-20)

hsv = np.zeros((H.shape[0], H.shape[1], 3))
hsv[:,:,0] = H   # Hue
hsv[:,:,1] = S   # Saturation
hsv[:,:,2] = V   # Brightness

rgb = mcolors.hsv_to_rgb(hsv)

plt.figure(figsize=(8,6))
plt.imshow(
    rgb,  # Convert to rgb
    aspect='auto',
    origin='lower',
    extent=[t_spec_big[0], t_spec_big[-1], f_big[0], f_big[-1]]
)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("Polarization-Colored Spectrogram (HSV)")
plt.colorbar()  # optional, though RGB doesn't map linearly
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(8,6))
bit = np.random.randint(0, len(t_spec_big))
plt.loglog(f_big, Sxx_plus_big[:, bit], label="h+")
plt.loglog(f_big, Sxx_cross_big[:, bit], label="hx")
plt.title(f"Random Slice at t={t_spec_big[bit]:.4e}")
plt.tight_layout()
plt.grid()
plt.show()