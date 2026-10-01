# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:27:21 2025

@author: TJ Hammond
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed

# %% Main loop
cc = 3e8
'''Parameters'''
a = 2*np.pi/800e-9*3/8*1 / \
    1.73     # Coupling constant k*3/8/n0, where n0 is the index of refraction
C = 3.2e-22  # Nonlinear X^(3) susceptibility tensor X^(3)_xxxx
# Cross-phase modulation coefficient, X^(3)_xxyy, relative to X^(3)_xxxx
P = C/1.85
Y0 = 6e9    # Amplitude of initial Gaussian envelope
# tau = 400e-15   # Width of the Gaussian envelope
FWHM = 120e-15  # Pulse duration
tau = FWHM/(2*(2*np.log(2))**0.5)  # ~50.9 fs
# print(tau)
wp = 2*np.pi*cc/800e-9    # Frequency wp (pump)
ws = 2*np.pi*cc/600e-9    # Frequency ws (signal)
B = 0.05  # Angle for X(t) initial condition
s = 1e-7  # Relative initial field amplitude for signal
phi = 0.0  # Phase shift for X(t)


# Define the time array
Nt = 301
# Time array for the initial conditions
t_array = np.linspace(-100e-15, 100e-15, Nt)
dt = t_array[1]-t_array[0]

w = 2*np.pi * np.fft.fftfreq(Nt, d=dt)
w = np.fft.fftshift(w)

bandwidth = 4/tau   # ~4/tau is a generous cutoff covering the pulse spectrum
# Gaussian filter centered on pump
mask = np.exp(-(w - wp)**2 / (2 * bandwidth**2))

'''
t_array being too large (> +/-350 fs) results in an extremely noisy
polarization angle vs time plot, remove noise by limiting to -350 to 350 or
smaller
'''

# Define the initial conditions as functions of time


def Y_t(t):
    return Y0 * np.cos(B) * np.exp(-t**2 / (2 * tau**2)) * (np.exp(1j * wp * t) + s*np.exp(1j * ws * t))


def X_t(t):
    return Y0 * np.sin(B) * np.exp(1j * phi) * np.exp(-t**2 / (2 * tau**2)) * np.exp(1j * wp * t)


# Compute the initial arrays for X and Y
X0_array = X_t(t_array)
Y0_array = Y_t(t_array)

# Define the coupled differential equations


def coupled_equations(z, state):
    # Reshape the state array into X and Y arrays (complex)
    X = state[:Nt] + 1j*state[Nt:2*Nt]        # re + i*im → complex X
    Y = state[2*Nt:3*Nt] + 1j*state[3*Nt:]    # re + i*im → complex Y

    # Compute magnitudes
    X_abs2, Y_abs2 = np.abs(X)**2, np.abs(Y)**2

    # Define the derivatives (Nonlinear step)
    # dX_dz = -1j*a * (C*X_abs2 + 2*P*Y_abs2) * X - 1j*a*P*Y**2*np.conj(X) # Equation 2a
    # dY_dz = -1j*a * (C*Y_abs2 + 2*P*X_abs2) * Y - 1j*a*P*X**2*np.conj(Y) # Equation 2b
    '''
        First term = self-phase modulation
        Second term = cross-phase modulation
        Third term = degenerate four-wave mixing
    '''

    ''' OPTICAL SHOCK ADDITION  '''
    # Existing nonlinear terms
    NLX = (C*X_abs2 + 2*P*Y_abs2)*X + P*Y**2*np.conj(X)
    NLY = (C*Y_abs2 + 2*P*X_abs2)*Y + P*X**2*np.conj(Y)

    # Shock terms — time derivative of the nonlinear polarization
    shock_coeff = 1/wp   # τ_shock = 1/ω₀

    dNLX_dt = np.gradient(np.abs(NLX), dt)
    dNLY_dt = np.gradient(np.abs(NLY), dt)

    dX_dz = -1j*a * (NLX + shock_coeff * dNLX_dt)
    dY_dz = -1j*a * (NLY + shock_coeff * dNLY_dt)

    # Flatten and return real and imaginary parts of the derivatives
    return np.hstack([dX_dz.real, dX_dz.imag, dY_dz.real, dY_dz.imag])


# Integration range
z_span = (0, 500e-6)  # Range of z (start, end) (0.5mm crystal thickness)
z_eval = np.linspace(*z_span, 101)  # Points at which to evaluate
dz = z_eval[1] - z_eval[0]


'''############################################################################
    NEW PROPAGATION CODE 
    (With Spatial Dispersion + Optical Shock [in function])
############################################################################'''

beta2 = 84.868e-30 / 1e-3   # From refractiveindex.info

phase = np.exp(-1j * 0.5 * beta2 * (w - wp)**2 * dz)


def propagate(Y0_in, tau_in, B_in, z_eval_in=None, dz_in=None, phase_in=None):
    if z_eval_in is None:
        z_eval_in = z_eval
    if dz_in is None:
        dz_in = dz
    if phase_in is None:
        phase_in = phase

    # Recompute initial conditions with new parameters
    def Y_t_local(t):
        return Y0_in * np.cos(B_in) * np.exp(-t**2 / (2 * tau_in**2)) * (np.exp(1j * wp * t) + s*np.exp(1j * ws * t))

    def X_t_local(t):
        return Y0_in * np.sin(B_in) * np.exp(1j * phi) * np.exp(-t**2 / (2 * tau_in**2)) * np.exp(1j * wp * t)

    X = X_t_local(t_array)
    Y = Y_t_local(t_array)

    Xt_values = np.zeros((Nt, len(z_eval_in)), dtype=complex)
    Yt_values = np.zeros((Nt, len(z_eval_in)), dtype=complex)

    # store initial condition (z = 0)
    Xt_values[:, 0] = X
    Yt_values[:, 0] = Y

    # Main propagation loop over z
    for i in range(len(z_eval_in) - 1):

        '''
        (1) NONLINEAR STEP (time domain)

            Reuses existing coupled_equations() function, which computes dX/dz
            and dY/dz from nonlinear effects (like in the paper) but with RK4 
            instead of solve_ivp()
        '''
        # Pack complex fields into real-valued state vector
        # (because function expects real + imaginary parts separately)
        state = np.hstack([X.real, X.imag, Y.real, Y.imag])

        ''' ---- RK4 integration ---- '''
        # k1: slope at current point
        k1 = coupled_equations(0, state)

        # k2: slope halfway using k1
        k2 = coupled_equations(0, state + 0.5 * dz_in * k1)

        # k3: another midpoint slope using k2
        k3 = coupled_equations(0, state + 0.5 * dz_in * k2)

        # k4: slope at end of step using k3
        k4 = coupled_equations(0, state + dz_in * k3)

        # Combine slopes to get next state (RK4 formula)
        solution = state + (dz_in / 6) * (k1 + 2*k2 + 2*k3 + k4)

        # Unpack back into complex fields
        X = solution[:Nt] + 1j * solution[Nt:2*Nt]
        Y = solution[2*Nt:3*Nt] + 1j * solution[3*Nt:]

        # If this value prints "nan nan" the step blew up, verifies optical
        # shock works properly
        # print(i, np.max(np.abs(X)), np.max(np.abs(Y)))
        if math.isnan(np.max(np.abs(X))) and math.isnan(np.max(np.abs(Y))):
            print(f"Blew up at {i}")
            return None, None

        '''
        (2) DISPERSION STEP (frequency domain)
        
            This is where the derived phase factor is applied: 
                phase = exp(-i * 0.5 * β_2 * (ω - ω_p)^2 * dz)
                
            Takes X and Y field values, transforms into frequency domaain, 
            imposes dispersion, and transforms back to the time domain
        '''
        # Transform fields to frequency domain
        Xw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(X), axis=0))
        Yw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Y), axis=0))

        # Apply dispersion phase shift to each frequency component
        Xw *= phase_in
        Yw *= phase_in

        # Transform back to time domain
        X = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Xw), axis=0))
        Y = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Yw), axis=0))

        Xt_values[:, i+1] = X
        Yt_values[:, i+1] = Y

    return Xt_values, Yt_values


Xt_out, Yt_out = propagate(Y0, tau, B)


def stokes_parameters(Ex, Ey):
    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2*np.real(Ex*np.conj(Ey))
    S3 = -2*np.imag(Ex*np.conj(Ey))     # sign convention may vary
    return S0, S1, S2, S3


def polarization_orientation_and_ellipticity(Ex, Ey, degrees=True):
    S0, S1, S2, S3 = stokes_parameters(Ex, Ey)
    psi = 0.5*np.arctan2(S2, S1)
    chi = 0.5*np.arcsin(S3/S0)
    if degrees:
        psi = np.rad2deg(psi)
        chi = np.rad2deg(chi)
    return psi, chi
# %% Intensity


'''
Pulse intensity vs time plots

X axis: time (fs)
Y axis: |field|² (intensity)

Shows how the Gaussian pulse envelope changes shape as it propagates. 
Y is the dominant (pump-like) polarization. X starts tiny (flat) and then may 
grow. The four plots show pump depletion and signal growth in time.
'''
plt.figure()
# [:,0] means first element of every row (start)
plt.plot(t_array*1e15, np.abs(Yt_out[:, 0])**2, label='Y(t) at 0 µm')
# [:,-1] means last element of every row (end)
plt.plot(t_array*1e15, np.abs(Yt_out[:, -1])**2, label='Y(t) at 500 µm')
plt.plot(t_array*1e15, np.abs(Xt_out[:, 0])**2, label='X(t) at 0 µm')
plt.plot(t_array*1e15, np.abs(Xt_out[:, -1])**2, label='X(t) at 500 µm')
# plt.title("Dispersion amplified by 500x")
plt.xlabel('Time (fs)')
plt.ylabel('Pulse Intensity (arb. units)')
plt.legend()

'''
Polarization angle vs time plot

X axis: time (fs)
Y axis: angle (degrees)0

arctan2(|X|, |Y|) at the exit of the crystal (z = 500 µm), evaluated at every 
time point. 0° means purely Y-polarized (no rotation); 45° means equal X and Y; 
90° means fully X-polarized. Shows how much the polarization rotates across the 
pulse.
'''
plt.figure()
angs = np.arctan2(np.abs(Xt_out[:, -1]), np.abs(Yt_out[:, -1]))
plt.plot(t_array*1e15, np.real(angs)*180/np.pi)
plt.xlabel('Time (fs)')
plt.ylabel('Polarization Angle (degrees)')

# %% Chirp
# Plot real part of Y field to show chirp
plt.figure()
plt.plot(t_array*1e15, np.real(Yt_out[:, 0]), label='Y(t) at 0 µm', alpha=0.7)
plt.plot(t_array*1e15, np.real(Yt_out[:, -1]),
         label='Y(t) at 500 µm', alpha=0.7)
plt.xlabel('Time (fs)')
plt.ylabel('Real field amplitude (arb. units)')
plt.legend()
# %% Pulse Duration Scan
fwhm_values = np.linspace(50e-15, 400e-15, 20)
output_fwhms = []

iteration = 0

for fwhm in fwhm_values:
    tau_in = fwhm/(2*(2*np.log(2))**0.5)
    Xt_out, Yt_out = propagate(Y0, tau_in, B)

    if Xt_out is None:
        output_fwhms.append(np.nan)
        print(f"Ran iteration {fwhm+1} (blew up, skipped)")
        continue

    intensity = np.abs(Xt_out[:, -1])**2
    if np.max(intensity) == 0 or np.any(np.isnan(intensity)):
        output_fwhms.append(np.nan)
        print(f"Ran iteration {fwhm+1} (zero/nan intensity, skipped)")
        continue

    I_max = np.max(intensity)
    if not np.isfinite(I_max) or I_max < 1e-12:
        output_fwhms.append(np.nan)
        continue

    half_max = np.max(intensity)/2
    indices = np.where(intensity >= half_max)[0]

    print(f"max I = {np.max(intensity)}, min I = {np.min(intensity)}")
    if len(indices) < 2:
        # No valid FWHM
        output_fwhms.append(np.nan)
        print(f"Invalid FWHM at Y0={Y0}")
        continue

    fwhm_out = t_array[indices[-1]] - t_array[indices[0]]
    output_fwhms.append(fwhm_out)

    iteration += 1
    print(f"Ran iteration {iteration}")

output_fwhms = np.array(output_fwhms)

plt.figure()
plt.plot(fwhm_values*1e15, output_fwhms*1e15, label='Output X FWHM')
plt.plot(fwhm_values*1e15, fwhm_values*1e15/np.sqrt(3),
         'k--', label='1/√3 theoretical limit')
plt.axvline(x=FWHM*1e15, ymin=0, color='Red', label='120 fs FWHM')
plt.xlabel('Input FWHM (fs)')
plt.ylabel('Output X FWHM (fs)')
plt.legend()

# %% Intensity Scan
Y0_values = np.linspace(1e9, 8e9, 20)  # vary amplitude
output_fwhms = []
input_fwhm = FWHM  # fixed at 120 fs

for Y0_scan in Y0_values:
    Xt_out, Yt_out = propagate(Y0_scan, tau, B)

    intensity = np.abs(Xt_out[:, -1])**2
    if np.max(intensity) == 0 or np.any(np.isnan(intensity)):
        output_fwhms.append(np.nan)
        continue

    half_max = np.max(intensity) / 2
    indices = np.where(intensity >= half_max)[0]
    fwhm_out = t_array[indices[-1]] - t_array[indices[0]]
    output_fwhms.append(fwhm_out)

    print(f"Ran iteration {Y0_scan+1}")

output_fwhms = np.array(output_fwhms)

plt.figure()
plt.plot(Y0_values, output_fwhms*1e15, label='Output X FWHM')
plt.axhline(y=input_fwhm*1e15/np.sqrt(3), color='k',
            linestyle='--', label='1/√3 theoretical limit')
plt.xlabel('Y0 (field amplitude)')
plt.ylabel('Output X FWHM (fs)')
plt.legend()

min_index = np.argmin(output_fwhms*1e15)
x_min = Y0_values[min_index]
y_min = (output_fwhms*1e15)[min_index]
print(f"Minima point: ({x_min}, {y_min})")
# %% Pulse evolution
z_indices = [0, len(z_eval)//2, -1]
z_labels = ['0 µm', '250 µm', '500 µm']

fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for col, (zi, zl) in enumerate(zip(z_indices, z_labels)):

    X_intensity = np.abs(Xt_out[:, zi])**2
    Y_intensity = np.abs(Yt_out[:, zi])**2

    # Normalize each to its own peak
    X_norm = X_intensity / np.max(X_intensity)
    Y_norm = Y_intensity / np.max(Y_intensity)

    axes[0, col].plot(t_array*1e15, X_norm)
    axes[0, col].set_title(f'X(t) at {zl}')
    axes[0, col].set_xlabel('Time (fs)')
    axes[0, col].set_ylabel('Normalized Intensity')

    axes[1, col].plot(t_array*1e15, Y_norm)
    axes[1, col].set_title(f'Y(t) at {zl}')
    axes[1, col].set_xlabel('Time (fs)')
    axes[1, col].set_ylabel('Normalized Intensity')

plt.tight_layout()

# %% Polarization rotation vs pump angle B (Across whole pulse, bell shape at high)
B_values = np.linspace(0, np.pi, 36)  # 0 to 180 degrees, 36 points
output_angles_low = []
output_angles_high = []

Y0_low = 2e9   # low intensity
Y0_high = 6e9  # high intensity (your current working value)

iteration = 0

for B_scan in B_values:
    # Low intensity
    Xt_low, Yt_low = propagate(Y0_low, tau, B_scan)
    angle_low = np.arctan2(np.abs(Xt_low[:, -1]), np.abs(Yt_low[:, -1]))
    output_angles_low.append(np.mean(angle_low) * 180/np.pi)

    iteration += 1
    print(f"Ran iteration {iteration}")

    # High intensity
    Xt_high, Yt_high = propagate(Y0_high, tau, B_scan)
    angle_high = np.arctan2(np.abs(Xt_high[:, -1]), np.abs(Yt_high[:, -1]))
    output_angles_high.append(np.mean(angle_high) * 180/np.pi)
    
    iteration += 1
    print(f"Ran iteration {iteration}")

plt.figure()
plt.plot(B_values*180/np.pi, output_angles_low,
         label=f'Low intensity (Y0={Y0_low:.0e})')
plt.plot(B_values*180/np.pi, output_angles_high,
         label=f'High intensity (Y0={Y0_high:.0e})')
plt.xlabel('Input pump angle B (degrees)')
plt.ylabel('Output polarization angle (degrees)')
plt.legend()

# %% Polarization rotation vs pump angle B (At peak of pulse, crown shape at high)
B_values = np.linspace(0, np.pi, 36)  # 0 to 180 degrees, 36 points
output_angles_low = []
output_angles_high = []

Y0_low = 2e9   # low intensity
Y0_high = 6e9  # high intensity (your current working value)

iteration = 0

for B_scan in B_values:
    # Low intensity
    Xt_low, Yt_low = propagate(Y0_low, tau, B_scan)
    peak_idx = np.argmax(np.abs(Xt_low[:, -1])**2)
    angle_low = np.arctan2(
        np.abs(Xt_low[peak_idx, -1]), np.abs(Yt_low[peak_idx, -1]))
    output_angles_low.append(angle_low * 180/np.pi)

    iteration += 1
    print(f"Ran iteration {iteration}")

    # High intensity
    Xt_high, Yt_high = propagate(Y0_high, tau, B_scan)
    peak_idx = np.argmax(np.abs(Xt_high[:, -1])**2)
    angle_high = np.arctan2(
        np.abs(Xt_high[peak_idx, -1]), np.abs(Yt_high[peak_idx, -1]))
    output_angles_high.append(angle_high * 180/np.pi)

    iteration += 1
    print(f"Ran iteration {iteration}")

plt.figure()
plt.plot(B_values*180/np.pi, output_angles_low,
         label=f'Low intensity (Y0={Y0_low:.0e})')
plt.plot(B_values*180/np.pi, output_angles_high,
         label=f'High intensity (Y0={Y0_high:.0e})')
plt.xlabel('Input pump angle B (degrees)')
plt.ylabel('Output polarization angle (degrees)')
plt.legend()

# %% Polarization Direction
B_values = np.linspace(0, np.pi/2, 90)  # Controls resolution

angle_map = np.zeros((len(B_values), Nt))  # rows = B, cols = time

for i, B_scan in enumerate(B_values):
    Xt_out, Yt_out = propagate(Y0, tau, B_scan)

    if Xt_out is None:
        angle_map[i, :] = np.nan
        continue

    X_exit = Xt_out[:, -1]
    Y_exit = Yt_out[:, -1]

    angle_map[i, :] = np.arctan2(np.abs(X_exit), np.abs(Y_exit)) * 180/np.pi

    print(f"Ran iteration {i+1}")

# Meshgrid for plotting
T, Bgrid = np.meshgrid(t_array * 1e15, B_values * 180/np.pi)

plt.figure()
plt.pcolormesh(T, Bgrid, angle_map, shading='auto', cmap='seismic')
plt.colorbar(label='Resulting angle (deg)')

plt.xlabel('Time (fs)')
plt.ylabel('Initial angle (deg)')

#%% Polarization Direction + Ellipticity
B_values = np.linspace(0, np.pi/2, 90)

angle_map = np.zeros((len(B_values), Nt))      # azimuth / direction
ellip_map = np.zeros((len(B_values), Nt))      # ellipticity

tau_in = 140e-15/(2*(2*np.log(2))**0.5)

for i, B_scan in enumerate(B_values):
    Xt_out, Yt_out = propagate(4.75e9, tau_in, B_scan)

    if Xt_out is None:
        angle_map[i, :] = np.nan
        ellip_map[i, :] = np.nan
        continue

    X_exit = Xt_out[:, -1]
    Y_exit = Yt_out[:, -1]

    # Match professor's convention: EoutY → Ex, EoutX → Ey
    psi, chi = polarization_orientation_and_ellipticity(Y_exit, X_exit)

    angle_map[i, :] = psi   # ellipse orientation (0-90°)
    ellip_map[i, :] = chi   # ellipticity (-45 to +45°)

    print(f"Ran iteration {i+1}")

# Meshgrid for plotting
T, Bgrid = np.meshgrid(t_array * 1e15, B_values * 180/np.pi)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].pcolormesh(T, Bgrid, angle_map, shading='auto', cmap='seismic')
fig.colorbar(im0, ax=axes[0], label='Ellipse Orientation (deg)')
axes[0].set_xlabel('Time (fs)')
axes[0].set_ylabel('Initial Angle (deg)')
axes[0].set_title('Ellipse Orientation')

im1 = axes[1].pcolormesh(T, Bgrid, ellip_map, shading='auto', cmap='seismic')
fig.colorbar(im1, ax=axes[1], label='Ellipticity Angle (deg)')
axes[1].set_xlabel('Time (fs)')
axes[1].set_ylabel('Initial Angle (deg)')
axes[1].set_title('Ellipticity (0=linear, ±45=circular)')

plt.tight_layout()

# %% Critical I·L Scan (2D: Intensity vs Crystal Length)
Y0_values = np.linspace(1e9, 8e9, 20)
L_values = np.linspace(50e-6, 1000e-6, 20)
B_fixed = np.deg2rad(15)
chi_thresh = 10.0
psi_thresh = 5.0

eps0 = 8.854e-12
I_values = 0.5 * eps0 * cc * Y0_values**2
IL_map = np.outer(L_values, I_values)

max_chi_map = np.full((len(L_values), len(Y0_values)), np.nan)
max_psi_map = np.full((len(L_values), len(Y0_values)), np.nan)


def scan_single(li, yi):
    L = L_values[li]
    Y0_s = Y0_values[yi]
    dz_s = 0.1e-6
    z_eval_s = np.arange(0, L + dz_s, dz_s)
    phase_s = np.exp(-1j * 0.5 * beta2 * (w - wp)**2 * dz_s)

    Xt, Yt = propagate(Y0_s, tau, B_fixed, z_eval_s, dz_s, phase_s)

    if Xt is None:
        return li, yi, np.nan, np.nan

    X_exit, Y_exit = Xt[:, -1], Yt[:, -1]
    psi, chi = polarization_orientation_and_ellipticity(Y_exit, X_exit)
    mask = (np.abs(Y_exit)**2 + np.abs(X_exit)**2) > 1e-3 * np.max(np.abs(Y_exit)**2 + np.abs(X_exit)**2)

    max_chi = np.max(np.abs(chi[mask]))
    max_psi = np.max(np.abs(psi[mask] - np.rad2deg(B_fixed)))
    return li, yi, max_chi, max_psi


indices = [(li, yi) for li in range(len(L_values)) for yi in range(len(Y0_values))]

total = len(indices)
print(f"Starting {total} propagations across {len(L_values)} lengths × {len(Y0_values)} intensities")

results = Parallel(n_jobs=4, verbose=10)(delayed(scan_single)(li, yi) for li, yi in indices)

# --- Unpack results ---
for li, yi, max_chi, max_psi in results:
    max_chi_map[li, yi] = max_chi
    max_psi_map[li, yi] = max_psi

print("Done.")

# --- Plotting ---
I_plot = I_values * 1e-16
L_plot = L_values * 1e6
IL_plot = IL_map * 1e-10

I_fine = np.linspace(I_plot.min(), I_plot.max(), 300)
L_fine = np.linspace(L_plot.min(), L_plot.max(), 300)
I_fine_grid, L_fine_grid = np.meshgrid(I_fine, L_fine)
IL_fine = np.outer(L_fine, I_fine) * 1e-10


# Interpolate the grid to smooth data
def smooth_map(data_map):
    filled = np.where(np.isnan(data_map), -1, data_map)
    interp = RegularGridInterpolator((L_plot, I_plot), filled, method='linear', bounds_error=False, fill_value=-1)
    pts = np.column_stack([L_fine_grid.ravel(), I_fine_grid.ravel()])
    result = interp(pts).reshape(L_fine_grid.shape)
    return np.where(result < 0, np.nan, result)


chi_smooth = smooth_map(max_chi_map)
psi_smooth = smooth_map(max_psi_map)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
IL_levels = np.linspace(np.nanmin(IL_plot), np.nanmax(IL_plot), 8)

for ax, data_smooth, data_map, thresh, label in [
        
    (axes[0], chi_smooth, max_chi_map, chi_thresh, 'Max Ellipticity |χ|'),
    (axes[1], psi_smooth, max_psi_map, psi_thresh,
     'Max Orientation Deviation |Δψ|'), ]:
    
    im = ax.pcolormesh(I_fine, L_fine, data_smooth, shading='auto', cmap='hot_r', vmin=0, vmax=np.nanmax(data_map))
    fig.colorbar(im, ax=ax, label=f'{label} (deg)')

    cs = ax.contour(I_fine, L_fine, IL_fine, levels=IL_levels, colors='cyan', linewidths=0.8, alpha=0.6)
    ax.clabel(cs, fmt='%.1f', fontsize=7)

    ax.contour(I_fine, L_fine, data_smooth, levels=[thresh], colors='lime', linewidths=2)

    nan_mask = np.isnan(data_smooth).astype(float)
    ax.contourf(I_fine, L_fine, nan_mask, levels=[0.5, 1.5], hatches=['////'], colors='none', edgecolors='gray', alpha=0.3)

    ax.set_xlabel('Peak Intensity (10¹⁶ W/m²)')
    ax.set_ylabel('Crystal Length (µm)')
    ax.set_title(f'{label}  [green = {thresh}° threshold]')

plt.tight_layout()

# --- Report critical I·L ---
print("\n--- Critical I·L estimates ---")
for label, data_map, thresh in [("chi", max_chi_map, chi_thresh), ("psi", max_psi_map, psi_thresh)]:
    exceeded = data_map >= thresh
    if exceeded.any():
        li_idx, yi_idx = np.where(exceeded)
        critical_IL = IL_map[li_idx, yi_idx]
        print(f"  {label} > {thresh}°: min I·L = {np.min(critical_IL):.3e} W/m")
    else:
        print(f"  {label}: threshold never exceeded in this range")
        
#Critical I·L vs Threshold
thresh_values = np.linspace(1, 90, 720)  # degrees
critical_IL_chi = []
critical_IL_psi = []

for thresh in thresh_values:
    # Chi
    exceeded = max_chi_map >= thresh
    if exceeded.any():
        li_idx, yi_idx = np.where(exceeded)
        critical_IL_chi.append(np.min(IL_map[li_idx, yi_idx]))
    else:
        critical_IL_chi.append(np.nan)  # threshold never reached

    # Psi
    exceeded = max_psi_map >= thresh
    if exceeded.any():
        li_idx, yi_idx = np.where(exceeded)
        critical_IL_psi.append(np.min(IL_map[li_idx, yi_idx]))
    else:
        critical_IL_psi.append(np.nan)

critical_IL_chi = np.array(critical_IL_chi)
critical_IL_psi = np.array(critical_IL_psi)

plt.figure(figsize=(8, 5))
plt.plot(thresh_values, critical_IL_chi * 1e-12, label='Ellipticity |χ|', color='royalblue')
plt.plot(thresh_values, critical_IL_psi * 1e-12, label='Orientation |Δψ|', color='darkorange')
plt.xlabel('Threshold Angle (deg)')
plt.ylabel('Critical I·L (10¹² W/m)')
plt.title('Minimum I·L to Exceed Polarization Threshold')
plt.legend()
plt.grid(alpha=0.3)

# Mark where NaN starts (threshold never exceeded) 
chi_limit = thresh_values[np.where(np.isnan(critical_IL_chi))[0][0]] if np.any(np.isnan(critical_IL_chi)) else None
psi_limit = thresh_values[np.where(np.isnan(critical_IL_psi))[0][0]] if np.any(np.isnan(critical_IL_psi)) else None

if chi_limit: plt.axvline(chi_limit, color='royalblue', linestyle='--', alpha=0.5, label=f'χ max = {chi_limit:.1f}°')
if psi_limit: plt.axvline(psi_limit, color='darkorange', linestyle='--', alpha=0.5, label=f'ψ max = {psi_limit:.1f}°')
plt.legend()

#%% 2D Gaussian Beam — Spatial Polarization Variation

# --- Parameters ---
L_beam = 500e-6                             # Fixed crystal length (m)
B_beam = np.deg2rad(15)                     # Fixed input angle
w0_values = [10e-6, 50e-6, 100e-6]          # Beam waists to scan (m)
Nr = 30                                     # Number of radial points
I_peak = 0.5 * 8.854e-12 * cc * Y0**2       # Peak intensity from Y0=6e9

# Radial grid (0 to 2*w0_max, fine enough for all beam sizes)
r_max = 2 * max(w0_values)
r_values = np.linspace(0, r_max, Nr)

# Precompute z/dz/phase for fixed L
dz_beam = 0.1e-6
z_eval_beam = np.arange(0, L_beam + dz_beam, dz_beam)
phase_beam = np.exp(-1j * 0.5 * beta2 * (w - wp)**2 * dz_beam)

def beam_scan_single(ri, w0):
    # Gaussian intensity profile
    r = r_values[ri]
    I_r = I_peak * np.exp(-2 * r**2 / w0**2)
    Y0_r = np.sqrt(2 * I_r / (8.854e-12 * cc))   # invert I = 0.5*eps0*c*|E|^2

    if Y0_r < 1e6:   # essentially zero field, skip
        return ri, 45.0, 0.0   # unpolarized limit -> return neutral values
    
    
    # Propagation and Stokes extraction
    Xt, Yt = propagate(Y0_r, tau, B_beam, z_eval_beam, dz_beam, phase_beam)

    if Xt is None:
        return ri, np.nan, np.nan

    X_exit, Y_exit = Xt[:, -1], Yt[:, -1]
    psi, chi = polarization_orientation_and_ellipticity(Y_exit, X_exit)


    total_I = np.abs(Y_exit)**2 + np.abs(X_exit)**2
    mask    = total_I > 1e-3 * np.max(total_I)

    # Return peak-weighted average over pulse
    weights = total_I[mask]
    psi_wavg = np.average(psi[mask], weights=weights)
    chi_wavg = np.average(chi[mask], weights=weights)

    return ri, psi_wavg, chi_wavg


r_um = r_values * 1e6  # for plotting

# Store results: dict keyed by w0
psi_results = {}
chi_results = {}

# Run the 30-point radial scan in parallel for each beam waist separately
for w0 in w0_values:
    print(f"\nRunning w0 = {w0*1e6:.0f} µm ...")
    indices_beam = list(range(Nr))

    results_beam = Parallel(n_jobs=4, verbose=5)(
        delayed(beam_scan_single)(ri, w0) for ri in indices_beam
    )
    results_beam.sort(key=lambda x: x[0])  # sort by ri

    psi_arr = np.array([r[1] for r in results_beam])
    chi_arr = np.array([r[2] for r in results_beam])

    psi_results[w0] = psi_arr
    chi_results[w0] = chi_arr


# Plot 1: Line plots of psi and chi vs radius
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['royalblue', 'darkorange', 'green']

for (w0, col) in zip(w0_values, colors):
    lbl = f'w₀ = {w0*1e6:.0f} µm'
    axes[0].plot(r_um, psi_results[w0], color=col, label=lbl)
    axes[1].plot(r_um, chi_results[w0], color=col, label=lbl)

    # Mark 1/e^2 radius for each w0
    axes[0].axvline(w0*1e6, color=col, linestyle='--', alpha=0.4)
    axes[1].axvline(w0*1e6, color=col, linestyle='--', alpha=0.4)

axes[0].set_xlabel('Radius (µm)')
axes[0].set_ylabel('Ellipse Orientation ψ (deg)')
axes[0].set_title('Polarization Orientation vs Beam Radius')
axes[0].legend()

axes[1].set_xlabel('Radius (µm)')
axes[1].set_ylabel('Ellipticity χ (deg)')
axes[1].set_title('Ellipticity vs Beam Radius')
axes[1].legend()

plt.tight_layout()


# Plot 2: 2D beam cross-section heatmaps
# Build 2D (x, y) grids for each w0
fig2, axes2 = plt.subplots(2, len(w0_values), figsize=(5*len(w0_values), 9))

for col_idx, w0 in enumerate(w0_values):
    xy_max  = 2 * w0
    xy_vals = np.linspace(-xy_max, xy_max, 200)
    XX, YY  = np.meshgrid(xy_vals, xy_vals)
    RR      = np.sqrt(XX**2 + YY**2)

    # Interpolate 1D radial results onto 2D grid
    psi_2d = np.interp(RR.ravel(), r_values, psi_results[w0]).reshape(RR.shape)
    chi_2d = np.interp(RR.ravel(), r_values, chi_results[w0]).reshape(RR.shape)

    # Gaussian intensity envelope for overlay
    I_2d = I_peak * np.exp(-2 * RR**2 / w0**2)
    I_norm = I_2d / I_peak

    # Mask very low intensity regions (beam edge)
    beam_mask = I_norm < 0.01
    psi_2d[beam_mask] = np.nan
    chi_2d[beam_mask] = np.nan

    xy_um = xy_vals * 1e6

    im0 = axes2[0, col_idx].pcolormesh(xy_um, xy_um, psi_2d,
                                        shading='auto', cmap='seismic',
                                        vmin=0, vmax=90)
    fig2.colorbar(im0, ax=axes2[0, col_idx], label='ψ (deg)')
    # Overlay 1/e² beam radius circle
    theta_c = np.linspace(0, 2*np.pi, 200)
    axes2[0, col_idx].plot(w0*1e6*np.cos(theta_c), w0*1e6*np.sin(theta_c),
                            'k--', linewidth=1, label='1/e² radius')
    axes2[0, col_idx].set_title(f'ψ — w₀={w0*1e6:.0f}µm')
    axes2[0, col_idx].set_xlabel('x (µm)')
    axes2[0, col_idx].set_ylabel('y (µm)')
    axes2[0, col_idx].legend(fontsize=7)

    im1 = axes2[1, col_idx].pcolormesh(xy_um, xy_um, chi_2d,
                                        shading='auto', cmap='seismic',
                                        vmin=-25, vmax=25)
    fig2.colorbar(im1, ax=axes2[1, col_idx], label='χ (deg)')
    axes2[1, col_idx].plot(w0*1e6*np.cos(theta_c), w0*1e6*np.sin(theta_c),
                            'k--', linewidth=1, label='1/e² radius')
    axes2[1, col_idx].set_title(f'χ — w₀={w0*1e6:.0f}µm')
    axes2[1, col_idx].set_xlabel('x (µm)')
    axes2[1, col_idx].set_ylabel('y (µm)')
    axes2[1, col_idx].legend(fontsize=7)

plt.suptitle('2D Beam Cross-Section: Polarization State', fontsize=13)
plt.tight_layout()