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
a = 2*np.pi/800e-9*3/8*1/1.73     # Coupling constant k*3/8/n0, where n0 is the index of refraction
C = 3.2e-22  # Nonlinear X^(3) susceptibility tensor X^(3)_xxxx
P = C/1.85  # Cross-phase modulation coefficient, X^(3)_xxyy, relative to X^(3)_xxxx
# Y0 = 4.75e9    # Amplitude of initial Gaussian envelope
Y0 = 6e9
# Y0 = np.sqrt(2*377*0.3e17)
# tau = 400e-15   # Width of the Gaussian envelope
FWHM = 140e-15  # Pulse duration
tau = FWHM/(2*(2*np.log(2))**0.5)  # ~50.9 fs
# print(tau)
wp = 2*np.pi*cc/800e-9    # Frequency wp (pump)
# ws = 2*np.pi*cc/600e-9    # Frequency ws (signal)
# B = 0.05  # Angle for X(t) initial condition
# s = 1e-7  # Relative initial field amplitude for signal
phi = 0.0  # Phase shift for X(t)

'''Change from True to False to disable shock'''
optical_shock = True    

# Angle array for X(t) initial condition
NB = 91
Bvec = np.linspace(0,np.pi/2,NB)

# Define the time array
Nt = 301
t_array = np.linspace(-100e-15, 100e-15, Nt)
dt = t_array[1] - t_array[0]

w = 2*np.pi * np.fft.fftfreq(Nt, d=dt)
w = np.fft.fftshift(w)

bandwidth = 4/tau   # ~4/tau is a generous cutoff covering the pulse spectrum
# Gaussian filter centered on pump
mask = np.exp(-(w - wp*0)**2 / (2 * bandwidth**2))

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

def propagate(Y0_in, tau_in, B_in, z_eval_in=None, dz_in=None, phase_in=None):
    # Unless specified, use default values
    if z_eval_in is None:
        z_eval_in = z_eval
    if dz_in is None:
        dz_in = dz
    if phase_in is None:
        phase_in = phase

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
            dX_dz = -1j*a * (C*X_abs2 + 2*P*Y_abs2) * X - 1j*a*P*Y**2*np.conj(X) # Equation 2a
            dY_dz = -1j*a * (C*Y_abs2 + 2*P*X_abs2) * Y - 1j*a*P*X**2*np.conj(Y) # Equation 2b
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
            
            dX_dz = -1j*a * (NLX - 1j*shock_coeff * dNLX_dt)
            dY_dz = -1j*a * (NLY - 1j*shock_coeff * dNLY_dt)

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
    # psi = 0.5*np.unwrap(np.arctan2(S2, S1))
    psi = 0.5*np.arctan2(S2, S1)
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

# --- Plotting ---
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

# --- Critical I·L vs Threshold ---
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

# --- Normalized plot: x-axis as fraction of physical maximum ---
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

#%% Beam Area Fraction Exceeding Threshold

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
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% of power')
    ax.axhline(10, color='gray', linestyle=':',  alpha=0.5, label='10% of power')
    ax.set_xlabel('Threshold Angle (deg)')
    ax.set_ylabel('Beam Power Fraction (%)')
    ax.legend()
    ax.grid(alpha=0.3)

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