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

#%% Parameters
cc = 3e8
a = 2*np.pi/800e-9*3/8*1/1.73       # Coupling constant k*3/8/n0, where n0 is the index of refraction
C = 3.2e-22                         # Nonlinear X^(3) susceptibility tensor X^(3)_xxxx
P = C/1.85                          # Cross-phase modulation coefficient, X^(3)_xxyy, relative to X^(3)_xxxx
# Y0 = 4.75e9                       # Amplitude of initial Gaussian envelope
Y0 = 6e9
# Y0 = np.sqrt(2*377*0.3e17)
# tau = 400e-15                     # Width of the Gaussian envelope
FWHM = 140e-15                      # Pulse duration
tau = FWHM/(2*(2*np.log(2))**0.5)   # ~50.9 fs
# print(tau)
wp = 2*np.pi*cc/800e-9              # Frequency wp (pump)
# ws = 2*np.pi*cc/600e-9            # Frequency ws (signal)
# B = 0.05                          # Angle for X(t) initial condition
# s = 1e-7                          # Relative initial field amplitude for signal
phi = 0.0                           # Phase shift for X(t)

# Angle array for X(t) initial condition
NB = 91
Bvec = np.linspace(0,np.pi/2,NB)

# Define the time array
Nt = 301
t_array = np.linspace(-100e-15, 100e-15, Nt)
dt = t_array[1] - t_array[0]

w = 2*np.pi * np.fft.fftfreq(Nt, d=dt)
w = np.fft.fftshift(w)
'''
t_array being too large (> +/-350 fs) results in an extremely noisy
polarization angle vs time plot, remove noise by limiting to -350 to 350 or
smaller
'''

z_span = (0,500e-6)  # Crystal thickness
dz = 5e-6            # Step size in crystal
z_eval = np.linspace(*z_span, 101)

beta2 = 84.868e-30 / 1e-3   # From refractiveindex.info
phase = np.exp(-1j * 0.5 * beta2 * (w - wp*0)**2 * dz)

#%% Functions

def propagate(Y0_in, tau_in, B_in, z_eval_in=None, dz_in=None, phase_in=None, a_in=None, dispersion=True, optical_shock=True):
    # Unless specified, use default values
    if z_eval_in is None:
        z_eval_in = z_eval
    if dz_in is None:
        dz_in = dz
    if phase_in is None:
        phase_in = phase
    if a_in is None:
        a_in = a

    '''
    def Y_t(t):
        return Y0_in * np.cos(B_in) * np.exp(-t**2 / (2 * tau_in**2)) * (np.exp(1j * wp * t) + s*np.exp(1j * ws * t))

    def X_t(t):
        return Y0_in * np.sin(B_in) * np.exp(1j * phi) * np.exp(-t**2 / (2 * tau_in**2)) * np.exp(1j * wp * t)
    '''
    # Define the initial conditions as functions of time
    def Y_t(t):
        return Y0_in * np.cos(B_in) * np.exp(-t**2 / (2 * tau_in**2)) * np.exp(0*1j * wp * t) 
    
    def X_t(t):
        return Y0_in * np.sin(B_in) * np.exp(-t**2 / (2 * tau_in**2)) * np.exp(0*1j * wp * t) * np.exp(1j * phi)
        
    # Compute the initial arrays for X and Y
    X = X_t(t_array)
    Y = Y_t(t_array)
    
    # Define the coupled differential equations
    def coupled_equations(z, state):
        # Reshape the state array into X and Y arrays (complex)
        X = state[:Nt] + 1j*state[Nt:2*Nt]        # re + i*im → complex X
        Y = state[2*Nt:3*Nt] + 1j*state[3*Nt:]    # re + i*im → complex Y

        # Compute magnitudes
        X_abs2, Y_abs2 = np.abs(X)**2, np.abs(Y)**2
        
        if optical_shock == False: 
            # Define the derivatives (Nonlinear step)
            dX_dz = -1j*a_in * (C*X_abs2 + 2*P*Y_abs2) * X - 1j*a_in*P*Y**2*np.conj(X) # Equation 2a
            dY_dz = -1j*a_in * (C*Y_abs2 + 2*P*X_abs2) * Y #- 1j*a_in*P*X**2*np.conj(Y) # Equation 2b
            '''
                First term = self-phase modulation
                Second term = cross-phase modulation
                Third term = degenerate four-wave mixing
            '''
        elif optical_shock == True:
            # Existing nonlinear terms
            NLX = (C*X_abs2 + 2*P*Y_abs2)*X + P*Y**2*np.conj(X)
            NLY = (C*Y_abs2 + 2*P*X_abs2)*Y + P*X**2*np.conj(Y)
            
            # Shock terms — time derivative of the nonlinear polarization
            shock_coeff = 1/wp   # τ_shock = 1/ω₀
            
            dNLX_dt = np.gradient((NLX), dt)
            dNLY_dt = np.gradient((NLY), dt)
            
            #dNLX_dt = np.gradient((C*X_abs2 + 2*P*Y_abs2)*X + P*Y**2*np.conj(X), dt)
            # dNLY_dt = np.gradient((C*Y_abs2 + 2*P*X_abs2)*Y + P*X**2*np.conj(Y), dt)
            
            dX_dz = -1j*a_in * (NLX - 1j*shock_coeff * dNLX_dt)
            dY_dz = -1j*a_in * (NLY - 1j*shock_coeff * dNLY_dt)

        # Flatten and return real and imaginary parts of the derivatives
        return np.hstack([dX_dz.real, dX_dz.imag, dY_dz.real, dY_dz.imag])
    
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
        if dispersion == True:
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

'''for countB in range(0,NB):
    B = Bvec[countB]
    Xt_out, Yt_out = propagate(Y0, tau, B)
    print(countB)'''
    
def stokes_parameters(Ex, Ey):
    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2*np.real(Ex*np.conj(Ey))
    S3 = -2*np.imag(Ex*np.conj(Ey))     # sign convention may vary
    return S0, S1, S2, S3

def polarization_orientation_and_ellipticity(Ex, Ey, degrees=True):
    S0, S1, S2, S3 = stokes_parameters(Ex, Ey)
    psi = 0.5*np.unwrap(np.arctan2(S2, S1))
    # psi = 0.5*np.arctan2(S2, S1)
    chi = 0.5*np.arcsin(S3/S0)
    if degrees:
        psi = np.rad2deg(psi)
        chi = np.rad2deg(chi)
    return psi, chi

#%% Polarization Direction + Ellipticity

# Create new arrays of zeroes
angle_map = np.zeros((len(Bvec), Nt))      # azimuth / direction
ellip_map = np.zeros((len(Bvec), Nt))      # ellipticity

for i, B_scan in enumerate(Bvec):
    Xt_out, Yt_out = propagate(4.75e9, tau, B_scan)
    
    # Handles areas where pulse too weak or doesn't exist
    if Xt_out is None:
        angle_map[i, :] = np.nan
        ellip_map[i, :] = np.nan
        continue
    
    X_exit = Xt_out[:, -1]
    Y_exit = Yt_out[:, -1]

    psi, chi = polarization_orientation_and_ellipticity(Y_exit, X_exit)

    angle_map[i, :] = psi   # ellipse orientation (0-90°)
    ellip_map[i, :] = chi   # ellipticity (-45 to +45°)

    print(f"Ran iteration {i+1}")

#%% Polarization Plot
# Meshgrid for plotting
T, Bgrid = np.meshgrid(t_array * 1e15, Bvec * 180/np.pi)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].pcolormesh(T, Bgrid, angle_map, shading='auto', cmap='seismic', vmin=0)
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

#%% Critical I·L Scan (2D: Intensity vs Crystal Length)
'''
Sweeps over a 20x20 grid of peak intensities and crystal lengths, propagates
each combination, and records the worst-case chi and psi reached anywhere in
the pulse. Goal is to find the minimum I·L product needed to push either
parameter past a given threshold — answers when polarization becomes an issue.
'''

# Scan range: field amplitudes 1–8 GV/m, crystal lengths 50–1000 µm
Y0_values = np.linspace(1e9, 8e9, 20)
L_values = np.linspace(50e-6, 1000e-6, 20)
B_fixed = np.deg2rad(15)  # Fixed input angle for all scans
chi_thresh = 10.0         # Reference threshold for the 2D map contour (deg)
psi_thresh = 5.0          # Reference threshold for the 2D map contour (deg)

# Convert field amplitudes to intensities, then compute I·L for every (L, I) pair
eps0 = 8.854e-12
I_values = 0.5 * eps0 * cc * Y0_values**2
IL_map = np.outer(L_values, I_values)  # shape: (len(L), len(I))

# Output arrays — filled with NaN so unfinished/blown-up runs are obvious
max_chi_map = np.full((len(L_values), len(Y0_values)), np.nan)
max_psi_map = np.full((len(L_values), len(Y0_values)), np.nan)

def scan_single(li, yi):
    '''Propagates one (L, Y0) pair and returns the max chi and psi over the pulse'''
    L = L_values[li]
    Y0_s = Y0_values[yi]
    # Use finer z steps than the default since some L values are very short
    dz_s = 0.1e-6
    z_eval_s = np.arange(0, L + dz_s, dz_s)
    phase_s = np.exp(-1j * 0.5 * beta2 * (w - wp*0)**2 * dz_s)

    Xt, Yt = propagate(Y0_s, tau, B_fixed, z_eval_s, dz_s, phase_s)

    if Xt is None:
        return li, yi, np.nan, np.nan

    X_exit, Y_exit = Xt[:, -1], Yt[:, -1]
    psi, chi = polarization_orientation_and_ellipticity(Y_exit, X_exit)

    # Only look at parts of the pulse above 0.1% of peak power — ignores noise at the wings
    mask = (np.abs(Y_exit)**2 + np.abs(X_exit)**2) > 1e-3 * np.max(np.abs(Y_exit)**2 + np.abs(X_exit)**2)

    max_chi = np.max(np.abs(chi[mask]))
    max_psi = np.max(np.abs(psi[mask] - np.rad2deg(B_fixed)))  # deviation from input angle
    return li, yi, max_chi, max_psi


# Build flat list of all (li, yi) index pairs to feed into joblib
indices = [(li, yi) for li in range(len(L_values)) for yi in range(len(Y0_values))]

total = len(indices)
print(f"Starting {total} propagations across {len(L_values)} lengths × {len(Y0_values)} intensities")

# Run all 400 propagations in parallel across 4 workers
results = Parallel(n_jobs=4, verbose=10)(delayed(scan_single)(li, yi) for li, yi in indices)

# --- Unpack results ---
for li, yi, max_chi, max_psi in results:
    max_chi_map[li, yi] = max_chi
    max_psi_map[li, yi] = max_psi

print("Done.")

#%% 2D I·L Heatmap Plot
# Rescale axes to nicer units for display
I_plot = I_values * 1e-16   # W/m² → 10¹⁶ W/m²
L_plot = L_values * 1e6     # m → µm
IL_plot = IL_map * 1e-10    # W/m → 10¹⁰ W/m (for contour labels)

# Build fine interpolation grids for smooth plotting
I_fine = np.linspace(I_plot.min(), I_plot.max(), 300)
L_fine = np.linspace(L_plot.min(), L_plot.max(), 300)
I_fine_grid, L_fine_grid = np.meshgrid(I_fine, L_fine)
IL_fine = np.outer(L_fine, I_fine) * 1e-10  # I·L on the fine grid, same units


# Smooth the 20x20 scan data onto the fine grid for display.
# NaN entries (blown-up runs) are temporarily replaced with -1 so the
# interpolator doesn't propagate NaNs, then masked back out afterward
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

# Plot both chi and psi maps with the same structure
for ax, data_smooth, data_map, thresh, label in [
    (axes[0], chi_smooth, max_chi_map, chi_thresh, 'Max Ellipticity |χ|'),
    (axes[1], psi_smooth, max_psi_map, psi_thresh, 'Max Orientation Deviation |Δψ|'),]:

    # Background heatmap of the polarization parameter
    im = ax.pcolormesh(I_fine, L_fine, data_smooth, shading='auto', cmap='hot_r', vmin=0, vmax=np.nanmax(data_map))
    fig.colorbar(im, ax=ax, label=f'{label} (deg)')

    # Cyan contour lines showing constant I·L — helps read off the critical product
    cs = ax.contour(I_fine, L_fine, IL_fine, levels=IL_levels, colors='cyan', linewidths=0.8, alpha=0.6)
    ax.clabel(cs, fmt='%.1f', fontsize=7)

    # Green contour marking where the threshold is first crossed
    ax.contour(I_fine, L_fine, data_smooth, levels=[thresh], colors='lime', linewidths=2)

    # Hatch over NaN regions (where propagation blew up)
    nan_mask = np.isnan(data_smooth).astype(float)
    ax.contourf(I_fine, L_fine, nan_mask, levels=[0.5, 1.5], hatches=['////'], colors='none', edgecolors='gray', alpha=0.3)

    ax.set_xlabel('Peak Intensity (10¹⁶ W/m²)')
    ax.set_ylabel('Crystal Length (µm)')
    ax.set_title(f'{label}  [green = {thresh}° threshold]')

plt.tight_layout()

# --- Report critical I·L as a single number for each threshold ---
print("\n--- Critical I·L estimates ---")
for label, data_map, thresh in [("chi", max_chi_map, chi_thresh), ("psi", max_psi_map, psi_thresh)]:
    exceeded = data_map >= thresh
    if exceeded.any():
        li_idx, yi_idx = np.where(exceeded)
        critical_IL = IL_map[li_idx, yi_idx]
        print(f"  {label} > {thresh}°: min I·L = {np.min(critical_IL):.3e} W/m")
    else:
        print(f"  {label}: threshold never exceeded in this range")

#%% Critical I·L vs Threshold Plot
# Instead of reporting for just one threshold, sweep from 1° to 90° and find
# the minimum I·L needed to cross each threshold. Where the curve goes NaN,
# that threshold is physically unreachable given the nonlinear ceiling.
thresh_values = np.linspace(1, 90, 720)  # degrees
critical_IL_chi = []
critical_IL_psi = []

for thresh in thresh_values:
    # Chi: find the lowest I·L in the scan that exceeded this threshold
    exceeded = max_chi_map >= thresh
    if exceeded.any():
        li_idx, yi_idx = np.where(exceeded)
        critical_IL_chi.append(np.min(IL_map[li_idx, yi_idx]))
    else:
        critical_IL_chi.append(np.nan)  # threshold physically unreachable

    # Psi: same logic
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

# Mark the physical maxima — where the curve terminates is the ceiling set by
# the nonlinear dynamics, not by the scan range
chi_limit = thresh_values[np.where(np.isnan(critical_IL_chi))[0][0]] if np.any(np.isnan(critical_IL_chi)) else None
psi_limit = thresh_values[np.where(np.isnan(critical_IL_psi))[0][0]] if np.any(np.isnan(critical_IL_psi)) else None

if chi_limit: plt.axvline(chi_limit, color='royalblue', linestyle='--', alpha=0.5, label=f'χ max = {chi_limit:.1f}°')
if psi_limit: plt.axvline(psi_limit, color='darkorange', linestyle='--', alpha=0.5, label=f'ψ max = {psi_limit:.1f}°')
plt.legend()

#%% Normalized Threshold Plot
# Normalized plot: x-axis as fraction of physical maximum
# Replot the same curves but normalize each threshold to its own physical max.
# This puts chi and psi on the same x-axis scale so their sensitivities can be
# compared directly — otherwise the different ceilings (22° vs 63°) make it
# hard to tell which parameter requires less I·L for the same relative effect.
chi_max_physical = chi_limit if chi_limit is not None else thresh_values[-1]
psi_max_physical = psi_limit if psi_limit is not None else thresh_values[-1]

# Trim each array to only the region below its own physical max
thresh_frac_chi = thresh_values[thresh_values <= chi_max_physical] / chi_max_physical
thresh_frac_psi = thresh_values[thresh_values <= psi_max_physical] / psi_max_physical
IL_chi_norm     = critical_IL_chi[:len(thresh_frac_chi)]
IL_psi_norm     = critical_IL_psi[:len(thresh_frac_psi)]

plt.figure(figsize=(8, 5))
plt.plot(thresh_frac_chi * 100, IL_chi_norm * 1e-12, label='Ellipticity |χ|', color='royalblue')
plt.plot(thresh_frac_psi * 100, IL_psi_norm * 1e-12, label='Orientation |Δψ|', color='darkorange')
plt.xlabel('Threshold (% of physical maximum)')
plt.ylabel('Critical I·L (10¹² W/m)')
plt.title('Minimum I·L to Exceed Polarization Threshold\n(normalized to physical max: '
          f'χ_max={chi_max_physical:.1f}°, ψ_max={psi_max_physical:.1f}°)')
plt.legend()
plt.grid(alpha=0.3)
# 50% of max is a natural midpoint reference — if the curves cross here that
# tells you which parameter responds more aggressively at moderate distortion
plt.axvline(50, color='gray', linestyle='--', alpha=0.5, label='50% of max')
plt.legend()

#%% 2D Gaussian Beam — Spatial Polarization Variation

# --- Parameters ---
L_beam = 500e-6                             # Fixed crystal length (m)
B_beam = np.deg2rad(15)                     # Fixed input angle
w0_values = [50e-6, 75e-6, 100e-6]          # Beam waists to scan (m)
Nr = 30                                     # Number of radial points
I_peak = 0.5 * 8.854e-12 * cc * Y0**2       # Peak intensity from Y0=6e9

# Radial grid (0 to 2*w0_max, fine enough for all beam sizes)
r_max = 2 * max(w0_values)
r_values = np.linspace(0, r_max, Nr)

# Precompute z/dz/phase for fixed L
dz_beam = 0.1e-6
z_eval_beam = np.arange(0, L_beam + dz_beam, dz_beam)
phase_beam = np.exp(-1j * 0.5 * beta2 * (w - wp*0)**2 * dz_beam)

def beam_scan_single(ri, w0, B_local):
    # Gaussian intensity profile
    r = r_values[ri]
    I_r = I_peak * np.exp(-2 * r**2 / w0**2)
    Y0_r = np.sqrt(2 * I_r / (8.854e-12 * cc))   # invert I = 0.5*eps0*c*|E|^2

    if Y0_r < 1e6:   # essentially zero field, skip
        return ri, B_local, 0.0   # unpolarized limit -> return neutral values
    
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
        delayed(beam_scan_single)(ri, w0, np.rad2deg(B_beam)) for ri in indices_beam
    )
    results_beam.sort(key=lambda x: x[0])  # sort by ri

    psi_arr = np.array([r[1] for r in results_beam])
    chi_arr = np.array([r[2] for r in results_beam])

    psi_results[w0] = psi_arr
    chi_results[w0] = chi_arr

#%% Gaussian Beam Plots
# Plot 1: Line plots of psi and chi vs radius
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['royalblue', 'darkorange', 'green']

for (w0, col) in zip(w0_values, colors):
    lbl = f'w₀ = {w0*1e6:.0f} µm'
    axes[0].plot(r_um, psi_results[w0], color=col, label=lbl)
    axes[1].plot(r_um, chi_results[w0], color=col, label=lbl)

    # Mark 1/e^2 radius for each w0
    axes[0].axvline(w0*1e6, color=col, linestyle='--', alpha=1)
    axes[1].axvline(w0*1e6, color=col, linestyle='--', alpha=1)

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
    axes2[0, col_idx].grid(alpha=1)

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
    axes2[1, col_idx].grid(alpha=1)

plt.suptitle('2D Beam Cross-Section: Polarization State', fontsize=13)
plt.tight_layout()

# Plot 3: Beam area fraction exceeding threshold
chi_thresh_range = np.linspace(0, 25, 300)  # degrees, up to chi saturation limit
psi_thresh_range = np.linspace(0, 65, 300)  # degrees, up to psi saturation limit

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['royalblue', 'darkorange', 'green']

for w0, col in zip(w0_values, colors):
    # Build fine 2D grid for this beam waist
    xy_max  = 2 * w0
    xy_vals = np.linspace(-xy_max, xy_max, 500)
    XX, YY  = np.meshgrid(xy_vals, xy_vals)
    RR      = np.sqrt(XX**2 + YY**2)

    # Gaussian intensity weight at each pixel
    I_2d   = np.exp(-2 * RR**2 / w0**2)  # normalized to peak=1

    # Interpolate radial psi/chi results onto 2D grid
    chi_2d = np.interp(RR.ravel(), r_values, chi_results[w0]).reshape(RR.shape)
    psi_2d = np.interp(RR.ravel(), r_values, psi_results[w0]).reshape(RR.shape)

    # Mask outside 3*w0 (negligible power)
    outer_mask = RR > 3 * w0
    I_2d[outer_mask]   = 0
    chi_2d[outer_mask] = 0
    psi_2d[outer_mask] = 0

    total_power = np.sum(I_2d)

    chi_fractions = []
    psi_fractions = []

    for thresh in chi_thresh_range:
        chi_exceeded = np.abs(chi_2d) > thresh

        chi_fractions.append(np.sum(I_2d[chi_exceeded]) / total_power * 100)
        
    for thresh in psi_thresh_range:
        psi_exceeded = np.abs(psi_2d - np.rad2deg(B_beam)) > thresh

        psi_fractions.append(np.sum(I_2d[psi_exceeded]) / total_power * 100)

    lbl = f'w₀ = {w0*1e6:.0f} µm'
    axes[0].plot(chi_thresh_range, chi_fractions, color=col, label=lbl)
    axes[1].plot(psi_thresh_range, psi_fractions, color=col, label=lbl)

# Reference lines
for ax in axes:
    ax.axhline(50, color='black', linestyle='--', alpha=1, label='50% of power')
    ax.axhline(10, color='black', linestyle=':',  alpha=1, label='10% of power')
    ax.set_xlabel('Threshold Angle (deg)')
    ax.set_ylabel('Beam Power Fraction (%)')
    ax.legend()
    ax.grid(alpha=1)

axes[0].set_title('Fraction of Beam Power with |χ| > Threshold')
axes[1].set_title('Fraction of Beam Power with |Δψ| > Threshold')

plt.tight_layout()

# --- Print summary table ---
print(f"\n{'w0 (µm)':<12} {'chi>5° (%)':<15} {'chi>10° (%)':<15} {'psi>5° (%)':<15} {'psi>10° (%)'}")
print("-" * 70)
for w0, col in zip(w0_values, colors):
    xy_max  = 2 * w0
    xy_vals = np.linspace(-xy_max, xy_max, 500)
    XX, YY  = np.meshgrid(xy_vals, xy_vals)
    RR      = np.sqrt(XX**2 + YY**2)
    I_2d    = np.exp(-2 * RR**2 / w0**2)
    chi_2d  = np.interp(RR.ravel(), r_values, chi_results[w0]).reshape(RR.shape)
    psi_2d  = np.interp(RR.ravel(), r_values, psi_results[w0]).reshape(RR.shape)
    outer_mask = RR > 3 * w0
    I_2d[outer_mask] = chi_2d[outer_mask] = psi_2d[outer_mask] = 0
    total_power = np.sum(I_2d)

    def frac(data, t): return np.sum(I_2d[np.abs(data) > t]) / total_power * 100

    print(f"{w0*1e6:<12.0f} {frac(chi_2d,5):<15.1f} {frac(chi_2d,10):<15.1f} "f"{frac(psi_2d - np.rad2deg(B_beam),5):<15.1f} {frac(psi_2d - np.rad2deg(B_beam),10):<15.1f}")
    
#%% Parameter Sweep — Polarization vs Beam Parameters

# --- Base Parameters ---
Y0_base = 6e9
FWHM_base = 140e-15
Tau_base = FWHM_base/(2*(2*np.log(2))**0.5)         # Width of gaussian (s)
L_base = 500e-6
Lambda_base = 800e-9
wp_base = 2*np.pi*cc/Lambda_base                    # Pump wavelength (rad/s)
B_base = np.deg2rad(15)
eps0 = 8.854e-12                                    # Permittivity of f.s.

# --- Changing Parameters ---
NP = 100                                            # Number of points
Y0_sweep = np.linspace(1e9, 10e9, NP)               # Field amplitude (V/m)
FWHM_sweep = np.linspace(50e-15, 500e-15, NP)       # Pulse duration (s)
L_sweep = np.linspace(50e-6, 2000e-6, NP)           # Crystal length (m)
Lambda_sweep = np.linspace(600e-9, 1200e-9, NP)     # Wavelength (m)
B_sweep = np.linspace(0, np.pi/2, 91)               # Input angle (rad)

def sweep_1D_params(param_val, param_name):
    
    # Initialize the y lists
    chi_out, psi_out = [], []
    
    # If needed, precompute the array of indices and beta2 values so that they
    # can be indexed in the for loop because they are different for each new
    # wavelength
    if param_name == 'Wavelength':
        n_sweep = (2.956362+0.02195770/(param_val**2-0.01428322)-0.01062387*param_val**2-0.0000204968*param_val**4)**5
        beta2_sweep = np.gradient(np.gradient(n_sweep, param_val), param_val) * param_val**2/(2*np.pi*cc**2)
    
    # Scan through the parameter arrays (x axis), define new parameters and
    # call propagate()
    for i, val in enumerate(param_val):
        if param_name == 'Wavelength':
            ''' 
            Must manually solve for the new phase since β2 is different at each
            wavelength. Take sellmeier equation for n(λ) and then use the
            fact that β2 is the second derivative of k with respect to ω, and turn
            into β2​ = λ^2/(2πc^2) * d^2/dλ^2[n(λ)]
            '''
            wp_scan = 2*np.pi*cc/val
            n_scan = n_sweep[i]
            beta2_scan = beta2_sweep[i]
            a_scan = 2*np.pi/val * 3/8 * 1/n_scan
            phase_scan = np.exp(-1j * 0.5 * beta2_scan * (w - wp_scan*0)**2 * dz)
            # Make the scan
            Xt_scan, Yt_scan = propagate(Y0_base, Tau_base, B_base, None, None, phase_scan, a_scan)
            
        if param_name == 'Amplitude':
            # Make the scan
            Xt_scan, Yt_scan = propagate(val, Tau_base, B_base)
            
        if param_name == 'Duration':
            Tau_scan = val/(2*(2*np.log(2))**0.5)
            # Make the scan
            Xt_scan, Yt_scan = propagate(Y0_base, Tau_scan, B_base)
            
        if param_name == 'Length':
            dz_scan     = 5e-6
            z_eval_scan = np.arange(0, val + dz_scan, dz_scan)
            phase_scan = np.exp(-1j * 0.5 * beta2 * (w - wp*0)**2 * dz_scan)
            # Make the scan
            Xt_scan, Yt_scan = propagate(Y0_base, Tau_base, B_base, z_eval_scan, dz_scan, phase_scan)
            
        if param_name == 'Angle':
            # Make the scan
            Xt_scan, Yt_scan = propagate(Y0_base, Tau_base, val)
            
        # Return nothing if there is no field at that point
        if Xt_scan is None:
            return np.nan, np.nan
        
        # Define the output fields to be the last part of the scanned arrays
        # and then pass them through polarization_orientation_and_ellipticity()
        # to get the psi and chi values at that point along x
        X_exit, Y_exit = Xt_scan[:, -1], Yt_scan[:, -1]
        psi, chi = polarization_orientation_and_ellipticity(Y_exit, X_exit)
        
        # Find the max of psi and chi values (most meaningful) after applying
        # the mask, only subtract the angle (regular formula) if parameter is
        # not the varying input angle
        
        if param_name == 'Angle':
            max_chi  = chi[Nt//2]
            max_dpsi = psi[Nt//2] - np.rad2deg(val)
        else:
            max_chi  = chi[Nt//2]
            max_dpsi = psi[Nt//2] - np.rad2deg(B_base)
            
        # Append to new lists
        chi_out.append(max_chi)
        psi_out.append(max_dpsi)
        
        print(f"{[param_name]} Ran iteration {i+1}")
        
    return np.array(chi_out), np.array(psi_out)

# Get the new arrays for each parameter for plotting
chi_Y0, psi_Y0 = sweep_1D_params(Y0_sweep, 'Amplitude')
chi_FWHM, psi_FWHM = sweep_1D_params(FWHM_sweep, 'Duration')
chi_L, psi_L = sweep_1D_params(L_sweep, 'Length')
chi_Lambda, psi_Lambda = sweep_1D_params(Lambda_sweep, 'Wavelength')
chi_B, psi_B = sweep_1D_params(B_sweep, 'Angle')

#%% 1D Sweep Plots

I_sweep = 0.5 * eps0 * cc * Y0_sweep**2    # Convert to intensity for x-axis

# All the data comprised in one data list to be easily referenced when plotting
sweep_data = [
    (I_sweep * 1e-16,      'Peak Intensity (10¹⁶ W/m²)',  chi_Y0,   psi_Y0),
    (FWHM_sweep * 1e15,    'Pulse FWHM (fs)',             chi_FWHM, psi_FWHM),
    (L_sweep * 1e6,        'Crystal Length (µm)',         chi_L,    psi_L),
    (Lambda_sweep * 1e9,   'Wavelength (nm)',             chi_Lambda, psi_Lambda),
    (np.rad2deg(B_sweep),  'Input Angle (deg)',           chi_B,    psi_B),
]

# Plot data on however many plots
for (xvals, xlabel, chi_vals, psi_vals) in sweep_data:
    fig1D, ax_row = plt.subplots(1, 2, figsize=(14, 5))
    
    ax_row[0].plot(xvals, chi_vals, color='royalblue', ms=4)
    ax_row[0].set_xlabel(xlabel)
    ax_row[0].set_ylabel('χ (deg)')
    # ax_row[0].set_ylabel('χ (deg)')
    ax_row[0].set_title(f'Ellipticity vs {xlabel.split("(")[0].strip()}')
    ax_row[0].grid(alpha=1)

    ax_row[1].plot(xvals, psi_vals, color='darkorange', ms=4)
    ax_row[1].set_xlabel(xlabel)
    ax_row[1].set_ylabel('Δψ (deg)')
    # ax_row[1].set_ylabel('ψ (deg)')
    ax_row[1].set_title(f'Orientation Deviation vs {xlabel.split("(")[0].strip()}')
    ax_row[1].grid(alpha=1)

plt.tight_layout()
print("1D plots done.")

#%% Pulse Duration + Input Angle

output_fwhms = []

# --- Base Parameters ---
Y0_base = 6e9
FWHM_base = 140e-15
Tau_base = FWHM_base/(2*(2*np.log(2))**0.5)         # Width of gaussian (s)
L_base = 500e-6
Lambda_base = 800e-9
wp_base = 2*np.pi*cc/Lambda_base                    # Pump wavelength (rad/s)
eps0 = 8.854e-12                                    # Permittivity of f.s.

dz_base = L_base/100
z_eval_base = np.arange(0, L_base, dz_base)
n_base = (2.956362 + 0.02195770/(Lambda_base**2 - 0.01428322) - 0.01062387*Lambda_base**2 - 0.0000204968*Lambda_base**4)**0.5
beta2_base = 84.868e-30 / 1e-3   # From refractiveindex.info
phase_base = np.exp(-1j * 0.5 * beta2_base * (w - wp_base*0)**2 * dz_base)
a_base = 2*np.pi/Lambda_base * 3/8 * 1/n_base


# --- Changing Parameters ---
NP = 20                                             # Number of points
Y0_sweep = np.linspace(1e9, 10e9, NP)               # Field amplitude (V/m)
L_sweep = np.linspace(50e-6, 2000e-6, NP)           # Crystal length (m)
Lambda_sweep = np.linspace(600e-9, 1200e-9, NP)     # Wavelength (m)
B_sweep = np.linspace(0, np.pi/2, NP)               # Input angle (rad)

I_sweep = 0.5 * eps0 * cc * Y0_sweep**2    # Convert to intensity

def sweep_2d_params(param_val, param_name):

    result_map = np.full((len(param_val), len(B_sweep)), np.nan)
    
    if param_name == 'Wavelength':
        n_arr = (2.956362 + 0.02195770/(param_val**2 - 0.01428322) - 0.01062387*param_val**2 - 0.0000204968*param_val**4)**0.5
        beta2_arr = np.gradient(np.gradient(n_arr, param_val), param_val) * param_val**2 / (2*np.pi*cc**2)
    
    for i, val in enumerate(param_val):
        if param_name == 'Wavelength':
            beta2_arr = np.gradient(np.gradient(n_arr, param_val), param_val) * param_val**2 / (2*np.pi*cc**2)
            n_scan = n_arr[i]
            beta2_scan = beta2_arr[i]
            wp_scan = 2*np.pi*cc / val
            a_scan = 2*np.pi/val * 3/8 * 1/n_scan
            phase_scan = np.exp(-1j * 0.5 * beta2_scan * (w - wp_scan*0)**2 * dz)
            
        elif param_name == 'Length':
            dz_scan = 5e-6
            z_eval_scan = np.arange(0, val + dz_scan, dz_scan)
            phase_scan_L = np.exp(-1j * 0.5 * beta2_base * (w - wp_base*0)**2 * dz_scan)
                
            
        for j, angle_val in enumerate(B_sweep):
            if param_name == 'Wavelength':
                Xt_scan, Yt_scan = propagate(Y0_base, Tau_base, angle_val, z_eval_base, dz_base, phase_scan, a_scan, False, False)

            elif param_name == 'Amplitude':
                Xt_scan, Yt_scan = propagate(val, Tau_base, angle_val, z_eval_base, dz_base, phase_base, a_base, False, False)

            elif param_name == 'Length':
                Xt_scan, Yt_scan = propagate(Y0_base, Tau_base, angle_val, z_eval_scan, dz_scan, phase_scan_L, a_base, False, False)

            # Check if the field is valid
            if Xt_scan is None:
                print(f"  [{param_name}] i={i+1}/{len(param_val)}, j={j+1}/{len(B_sweep)} — blew up")
                continue

            intensity = np.abs(Xt_scan[:, -1])**2
            if np.max(intensity) == 0 or np.any(np.isnan(intensity)) or not np.isfinite(np.max(intensity)):
                print(f"  [{param_name}] i={i+1}/{len(param_val)}, j={j+1}/{len(B_sweep)} — zero/nan intensity")
                fwhm_out = 0 # Intensity is zero or nan, zero X output
                result_map[i, j] = fwhm_out
                continue

            half_max = np.max(intensity) / 2
            indices_hw = np.where(intensity >= half_max)[0]

            if len(indices_hw) < 2:
                print(f"  [{param_name}] i={i+1}/{len(param_val)}, j={j+1}/{len(B_sweep)} — invalid FWHM")
                continue

            fwhm_out = t_array[indices_hw[-1]] - t_array[indices_hw[0]]
            result_map[i, j] = fwhm_out
            print(f"  [{param_name}] i={i+1}/{len(param_val)}, j={j+1}/{len(B_sweep)} — FWHM out = {fwhm_out*1e15:.1f} fs")

    return result_map

Y0_output_fwhm = np.array(sweep_2d_params(Y0_sweep, 'Amplitude'))

L_output_fwhm = np.array(sweep_2d_params(L_sweep, 'Length'))

Lambda_output_fwhm = np.array(sweep_2d_params(Lambda_sweep, 'Wavelength'))

#%% Plot

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Output Pulse Duration vs Input FWHM', fontsize=14, fontweight='bold')

angle_deg = np.rad2deg(B_sweep)

sweep_data = [
    (I_sweep * 1e-16,      'Peak Intensity (10¹⁶ W/m²)',    Y0_output_fwhm   * 1e15),
    (L_sweep * 1e6,        'Crystal Length (µm)',           L_output_fwhm    * 1e15),
    (Lambda_sweep * 1e9,   'Wavelength (nm)',               Lambda_output_fwhm * 1e15)
]

for ax, (y_vals, y_label, fwhm_map) in zip(axes.flat, sweep_data):
    im = ax.pcolormesh(angle_deg, y_vals, fwhm_map, cmap='inferno', shading='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Output FWHM (fs)', fontsize=10)
    ax.set_xlabel('Input Angle (deg)', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f'Output FWHM vs Input Angle\n& {y_label}', fontsize=10)

plt.tight_layout()

#%%
'''
intensity = 0.45 or 3.1 or 12.2 W/m^2 (200 fs)
    Y0 = 
length = ? (leave as 0.5 mm)
wavelength = 681 nm (200 fs)
angle = 5.5 or 57 or 77.5 deg (200 fs)
'''



#%% Base parameters for 1D sweeps
Y0_base = 6e9                                       # Field amplitude (V/m)
FWHM_base = 140e-15                                 # Pulse Duration (fs)
tau_base = FWHM_base/(2*(2*np.log(2))**0.5)         # Width of gaussian (fs)
L_base = 500e-6                                     # Crystal length (m)
Lambda_base = 800e-9                                # Wavelength (m)
wp_base = 2*np.pi*cc/Lambda_base                    # Pump frequency (rad/s)
B_base = np.deg2rad(15)                             # Input angle (rad)
eps0 = 8.854e-12                                    # Permittivity of f.s.
           
z_span_base = (0,L_base)                            # Crystal thickness
dz_base = L_base/100                                # Step size in crystal
z_eval_base = np.linspace(*z_span_base, 101)        # Evaluation range

beta2_base = 84.868e-30 / 1e-3                      # From refractiveindex.info
phase_base = np.exp(-1j * 0.5 * beta2_base * (w - wp_base*0)**2 * dz_base)

n_base = (2.956362+0.02195770/(Lambda_base**2-0.01428322)-0.01062387*Lambda_base**2-0.0000204968*Lambda_base**4)**0.5
a_base = 2*np.pi/Lambda_base * 3/8 * 1/n_base

def find_crossings_variable_target(x_vals, y_vals, target_vals):
    """Same as find_crossings, but target is itself an array (e.g. x/sqrt(3))."""
    x_vals = np.asarray(x_vals)
    y_vals = np.asarray(y_vals)
    target_vals = np.asarray(target_vals)
    
    diff = y_vals - target_vals
    crossings = []
    for i in range(len(diff) - 1):
        d1, d2 = diff[i], diff[i+1]
        if np.isnan(d1) or np.isnan(d2):
            continue
        if d1 == 0:
            crossings.append(x_vals[i])
        elif d1 * d2 < 0:
            frac = -d1 / (d2 - d1)
            x_cross = x_vals[i] + frac * (x_vals[i+1] - x_vals[i])
            crossings.append(x_cross)
    return crossings

def find_crossings(x_vals, y_vals, target):
    """
    Finds x-values where y_vals crosses a horizontal target line.
    Returns a list of interpolated x-crossing points.
    Skips any NaN points (blown-up runs) cleanly.
    """
    x_vals = np.asarray(x_vals)
    y_vals = np.asarray(y_vals)
    
    crossings = []
    for i in range(len(y_vals) - 1):
        y1, y2 = y_vals[i], y_vals[i+1]
        x1, x2 = x_vals[i], x_vals[i+1]

        # Skip if either point is NaN (blown-up run)
        if np.isnan(y1) or np.isnan(y2):
            continue

        # Check if target lies between y1 and y2 (sign change)
        if (y1 - target) == 0:
            crossings.append(x1)
        elif (y1 - target) * (y2 - target) < 0:
            # Linear interpolation for the exact crossing point
            frac = (target - y1) / (y2 - y1)
            x_cross = x1 + frac * (x2 - x1)
            crossings.append(x_cross)

    return crossings

#%% Pulse duration scan
fwhm_values = np.linspace(50e-15, 400e-15, 1000)
output_fwhms = []

iteration = 0

for fwhm_scan in fwhm_values:
    tau_scan = fwhm_scan/(2*(2*np.log(2))**0.5)
    Xt_out, Yt_out = propagate(Y0_base, tau_scan, B_base, z_eval_base, dz_base, phase_base, a_base, dispersion=False, optical_shock=False)

    intensity = np.abs(Xt_out[:, -1])**2
    if np.max(intensity) == 0 or np.any(np.isnan(intensity)):
        output_fwhms.append(np.nan)
        continue

    half_max = np.max(intensity)/2
    indices = np.where(intensity >= half_max)[0]
    fwhm_out = t_array[indices[-1]] - t_array[indices[0]]
    output_fwhms.append(fwhm_out)

    iteration += 1
    print(f"Ran iteration {iteration}")

output_fwhms = np.array(output_fwhms)

plt.figure()
plt.plot(fwhm_values*1e15, output_fwhms*1e15, label='Output X FWHM')
plt.plot(fwhm_values*1e15, fwhm_values*1e15, color='red', label='No broadening')
plt.plot(fwhm_values*1e15, fwhm_values*1e15/np.sqrt(3), 'k--', label='1/√3 theoretical limit')
plt.xlabel('Input FWHM (fs)')
plt.ylabel('Output X FWHM (fs)')
plt.legend()


# Pulse duration scan — target is a line (fwhm_values / sqrt(3)), not constant
pd_crossings = find_crossings_variable_target(
    fwhm_values*1e15, output_fwhms*1e15, fwhm_values*1e15/np.sqrt(3)
)
print(f"Pulse duration scan crosses 1/√3 at input FWHM = {pd_crossings} fs")

#%% Amplitude Scan
Y0_values = np.linspace(1e9, 10e9, 1000)
output_fwhms = []

iteration = 0

for Y0_scan in Y0_values:
    Xt_out, Yt_out = propagate(Y0_scan, tau_base, B_base, z_eval_base, dz_base, phase_base, a_base, dispersion=False, optical_shock=False)

    intensity = np.abs(Xt_out[:, -1])**2
    if np.max(intensity) == 0 or np.any(np.isnan(intensity)):
        output_fwhms.append(np.nan)
        continue

    half_max = np.max(intensity) / 2
    indices = np.where(intensity >= half_max)[0]
    fwhm_out = t_array[indices[-1]] - t_array[indices[0]]
    output_fwhms.append(fwhm_out)

    iteration += 1
    print(f"Ran iteration {iteration}")

output_fwhms = np.array(output_fwhms)

plt.figure()
plt.plot(Y0_values*1e-9, output_fwhms*1e15, label='Output X FWHM')
plt.axhline(y=FWHM_base*1e15, color='red', label='No broadening')
plt.axhline(y=FWHM_base*1e15/np.sqrt(3), color='k', linestyle='--', label='1/√3 theoretical limit')
plt.xlabel('Y0 (V/m)')
plt.ylabel('Output X FWHM (fs)')
plt.legend()

# Amplitude scan — target is constant (FWHM_base/sqrt(3))
y0_crossings = find_crossings(
    Y0_values*1e-9, output_fwhms*1e15, FWHM_base*1e15/np.sqrt(3)
)
print(f"Amplitude scan crosses 1/√3 at Y0 = {y0_crossings} GV/m")

#%% Length scan
L_values = np.linspace(50e-6, 2000e-6, 1000)
output_fwhms = []

iteration = 0

for L_scan in L_values:
    dz_scan     = 5e-6
    z_eval_scan = np.arange(0, L_scan + dz_scan, dz_scan)
    phase_scan = np.exp(-1j * 0.5 * beta2_base * (w - wp_base*0)**2 * dz_scan)
    
    Xt_out, Yt_out = propagate(Y0_base, tau_base, B_base, z_eval_scan, dz_scan, phase_scan, a_base, dispersion=False, optical_shock=False)

    intensity = np.abs(Xt_out[:, -1])**2
    if np.max(intensity) == 0 or np.any(np.isnan(intensity)):
        output_fwhms.append(np.nan) 
        continue

    half_max = np.max(intensity) / 2
    indices = np.where(intensity >= half_max)[0]
    fwhm_out = t_array[indices[-1]] - t_array[indices[0]]
    output_fwhms.append(fwhm_out)

    iteration += 1
    print(f"Ran iteration {iteration}")

output_fwhms = np.array(output_fwhms)

plt.figure()
plt.plot(L_values*1e6, output_fwhms*1e15, label='Output X FWHM')
plt.axhline(y=FWHM_base*1e15, color='red', label='No broadening')
plt.axhline(y=FWHM_base*1e15/np.sqrt(3), color='k', linestyle='--', label='1/√3 theoretical limit')
plt.xlabel('Crystal Thickness (µm)')
plt.ylabel('Output X FWHM (fs)')
plt.legend()

# Length scan
L_crossings = find_crossings(
    L_values*1e6, output_fwhms*1e15, FWHM_base*1e15/np.sqrt(3)
)
print(f"Length scan crosses 1/√3 at L = {L_crossings} µm")

#%% Wavelength scan

Lambda_values = np.linspace(600e-9, 1200e-9, 1000)
output_fwhms = []

n_values = (2.956362+0.02195770/(Lambda_values**2-0.01428322)-0.01062387*Lambda_values**2-0.0000204968*Lambda_values**4)**5
beta2_values = np.gradient(np.gradient(n_values, Lambda_values), Lambda_values) * Lambda_values**2/(2*np.pi*cc**2)

iteration = 0

for i, Lambda_scan in enumerate(Lambda_values):
    wp_scan = 2*np.pi*cc/Lambda_scan
    n_scan = n_values[i]
    beta2_scan = beta2_values[i]
    a_scan = 2*np.pi/Lambda_scan * 3/8 * 1/n_scan
    phase_scan = np.exp(-1j * 0.5 * beta2_scan * (w - wp_scan*0)**2 * dz_base)
    
    Xt_out, Yt_out = propagate(Y0_base, tau_base, B_base, z_eval_base, dz_base, phase_scan, a_scan, dispersion=False, optical_shock=False)

    intensity = np.abs(Xt_out[:, -1])**2
    if np.max(intensity) == 0 or np.any(np.isnan(intensity)):
        output_fwhms.append(np.nan)
        continue

    half_max = np.max(intensity) / 2
    indices = np.where(intensity >= half_max)[0]
    fwhm_out = t_array[indices[-1]] - t_array[indices[0]]
    output_fwhms.append(fwhm_out)

    iteration += 1
    print(f"Ran iteration {iteration}")

output_fwhms = np.array(output_fwhms)

plt.figure()
plt.plot(Lambda_values*1e9, output_fwhms*1e15, label='Output X FWHM')
plt.axhline(y=FWHM_base*1e15, color='red', label='No broadening')
plt.axhline(y=FWHM_base*1e15/np.sqrt(3), color='k', linestyle='--', label='1/√3 theoretical limit')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Output X FWHM (fs)')
plt.legend()

# Wavelength scan
lambda_crossings = find_crossings(
    Lambda_values*1e9, output_fwhms*1e15, FWHM_base*1e15/np.sqrt(3)
)
print(f"Wavelength scan crosses 1/√3 at λ = {lambda_crossings} nm")

#%% Angle scan

B_values = np.linspace(0, np.pi/2, 900)
output_fwhms = []

iteration = 0

for B_scan in B_values:
    Xt_out, Yt_out = propagate(Y0_base, tau_base, B_scan, z_eval_base, dz_base, phase_base, a_base, dispersion=False, optical_shock=False)

    intensity = np.abs(Xt_out[:, -1])**2
    if np.max(intensity) == 0 or np.any(np.isnan(intensity)):
        output_fwhms.append(np.nan)
        continue

    half_max = np.max(intensity) / 2
    indices = np.where(intensity >= half_max)[0]
    fwhm_out = t_array[indices[-1]] - t_array[indices[0]]
    output_fwhms.append(fwhm_out)

    iteration += 1
    print(f"Ran iteration {iteration}")

output_fwhms = np.array(output_fwhms)

plt.figure()
plt.plot(np.rad2deg(B_values), output_fwhms*1e15, label='Output X FWHM')
plt.axhline(y=FWHM_base*1e15, color='red', label='No broadening')
plt.axhline(y=FWHM_base*1e15/np.sqrt(3), color='k', linestyle='--', label='1/√3 theoretical limit')
plt.xlabel('Input Angle (deg)')
plt.ylabel('Output X FWHM (fs)')
plt.legend()

# Angle scan
angle_crossings = find_crossings(
    np.rad2deg(B_values), output_fwhms*1e15, FWHM_base*1e15/np.sqrt(3)
)
print(f"Angle scan crosses 1/√3 at angle = {angle_crossings} deg")

#%%
'''
Amplitude scan crosses 1/√3 at Y0 = [np.float64(2.4448488444602643), np.float64(6.501097101485683)] GV/m

Length scan crosses 1/√3 at L = [np.float64(128.81634873215629), np.float64(935.447915532108)] µm

Wavelength scan crosses 1/√3 at λ = [] nm

Angle scan crosses 1/√3 at angle = [np.float64(20.18460516556392), np.float64(54.09793098571528), np.float64(57.87202500364649)] deg

Pulse duration scan crosses 1/√3 at input FWHM = [] fs
'''

#%%

# --- Decoupled test: B=90° puts everything in X, Y=0 ---
B_test = 0
Y0_test = 6e9
FWHM_test = 140e-15
tau_test = FWHM_test / (2*(2*np.log(2))**0.5)

Xt_out, Yt_out = propagate(
    Y0_test, tau_test, B_test,
    z_eval_base, dz_base, phase_base, a_base,
    dispersion=False, optical_shock=False
)

print("Max |Y| anywhere:", np.max(np.abs(Yt_out)))   # should be ~0
print("Max |X| at z=0:  ", np.max(np.abs(Xt_out[:, 0])))
print("Max |X| at z=L:  ", np.max(np.abs(Xt_out[:, -1])))

diff = np.abs(Xt_out[:, -1]) - np.abs(Xt_out[:, 0])
print("Max pointwise |X| change:", np.max(np.abs(diff)))

def get_fwhm(field_1d):
    intensity = np.abs(field_1d)**2
    half_max = np.max(intensity) / 2
    idx = np.where(intensity >= half_max)[0]
    return t_array[idx[-1]] - t_array[idx[0]]

fwhm_initial = get_fwhm(Xt_out[:, 0])
fwhm_final   = get_fwhm(Xt_out[:, -1])
print(f"\nInput FWHM:  {fwhm_initial*1e15:.4f} fs")
print(f"Output FWHM: {fwhm_final*1e15:.4f} fs")