# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:27:21 2025

@author: TJ Hammond
"""
#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import math
from inferno_white_cmap import inferno_white
from inferno_white_cmap import inverse_inferno_white
from pyhank import HankelTransform
from scipy.interpolate import interp1d

#%% Parameters

# --- Constants ---
cc = 3e8                            # Speed of light
C = 3.2e-22                         # Nonlinear X^(3) susceptibility tensor X^(3)_xxxx
P = C/1.85                          # Cross-phase modulation coefficient, X^(3)_xxyy, relative to X^(3)_xxxx
eps0 = 8.854e-12                    # Permittivity of free space

# ws = 2*np.pi*cc/600e-9            # Frequency ws (signal)
# B = 0.05                          # Angle for X(t) initial condition
# s = 1e-7                          # Relative initial field amplitude for signal
phi = 0.0                           # Phase shift for X(t)


# Angle array for X(t) initial condition
NB = 91
Bvec = np.linspace(0,np.pi/2,NB)

# Define the time array
Nt = 1001
t_array = np.linspace(-200e-15, 200e-15, Nt)
dt = t_array[1] - t_array[0]

'''
t_array being too large (> +/-350 fs) results in an extremely noisy
polarization angle vs time plot, remove noise by limiting to -350 to 350 or
smaller
'''

# Central frequency
w = 2*np.pi * np.fft.fftfreq(Nt, d=dt)  
w = np.fft.fftshift(w)


# --- Changing Parameters ---
NP = 512                                            # Number of points
Y0_sweep = np.linspace(1e9, 10e9, NP)               # Field amplitude (V/m)
L_sweep = np.linspace(50e-6, 2000e-6, NP)           # Crystal length (m)
Lambda_sweep = np.linspace(600e-9, 1200e-9, NP)     # Wavelength (m)
#Lambda_sweep = np.linspace(1600e-9, 2200e-9, NP)    # Wavelength (m)
B_sweep = np.linspace(0, np.pi/2, NP)               # Input angle (rad)
I_sweep = 0.5 * eps0 * cc * Y0_sweep**2             # Intensity (W/m^2)


# --- Base Parameters ---
Y0_base = 6e9
#I_base = 9e16
#Y0_base = (I_base/(0.5*eps0*cc))**0.5
FWHM_base = 100e-15                                 # Pulse duration (s)                 
Tau_base = FWHM_base/(2*(np.log(2))**0.5)           # Width of gaussian (s)
L_base = 500e-6
Lambda_base = 785e-9
Lambda_base_micron = Lambda_base*1e6
wp_base = 2*np.pi*cc/Lambda_base                    # Pump wavelength (rad/s)
B_base = np.deg2rad(2)
beta2_base = 87.149e-27                             # From refractiveindex.info
#beta2_base = -27.138e-27


# --- Necessary Calculations ---
dz_base = L_base/100
z_eval_base = np.arange(0, L_base, dz_base)
n_base = (2.956362 + 0.02195770/(Lambda_base_micron**2 - 0.01428322) - 0.01062387*Lambda_base_micron**2 - 0.0000204968*Lambda_base_micron**4)**0.5
phase_base = np.exp(-1j * 0.5 * beta2_base * (w - wp_base*0)**2 * dz_base)
a_base = 2*np.pi/Lambda_base * 3/8 * 1/n_base       # Coupling constant k*3/8/n0, where n0 is the index of refraction

#%% Functions

def propagate(Y0_in, tau_in, B_in, z_eval_in=None, dz_in=None, phase_in=None, a_in=None, dispersion=True, optical_shock=True, spm=True, xpm=True, fwm=True):
    # Unless specified, use default values
    if z_eval_in is None:
        z_eval_in = z_eval_base
    if dz_in is None:
        dz_in = dz_base
    if phase_in is None:
        phase_in = phase_base
    if a_in is None:
        a_in = a_base

    '''
    def Y_t(t):
        return Y0_in * np.cos(B_in) * np.exp(-t**2 / (2 * tau_in**2)) * (np.exp(1j * wp * t) + s*np.exp(1j * ws * t))

    def X_t(t):
        return Y0_in * np.sin(B_in) * np.exp(1j * phi) * np.exp(-t**2 / (2 * tau_in**2)) * np.exp(1j * wp * t)
    '''
    # Define the initial conditions as functions of time
    def Y_t(t):
        return Y0_in * np.cos(B_in) * np.exp(-t**2 / (2 * tau_in**2)) * np.exp(1j * 0*wp_base * t) 
    
    def X_t(t):
        return Y0_in * np.sin(B_in) * np.exp(-t**2 / (2 * tau_in**2)) * np.exp(1j * 0*wp_base * t) * np.exp(1j * phi)
        
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
        
        spm_term_X = C * X_abs2 if spm else 0
        spm_term_Y = C * Y_abs2 if spm else 0
        xpm_term_X = 2*P * Y_abs2 if xpm else 0
        xpm_term_Y = 2*P * X_abs2 if xpm else 0
        fwm_term_X = P * Y**2 * np.conj(X) if fwm else 0
        fwm_term_Y = P * X**2 * np.conj(Y) if fwm else 0

        if optical_shock == False:
            dX_dz = -1j*a_in * (spm_term_X + xpm_term_X) * X - 1j*a_in*fwm_term_X
            dY_dz = -1j*a_in * (spm_term_Y + xpm_term_Y) * Y - 1j*a_in*fwm_term_Y
        else:
            NLX = (spm_term_X + xpm_term_X)*X + fwm_term_X
            NLY = (spm_term_Y + xpm_term_Y)*Y + fwm_term_Y
            shock_coeff = 1/wp_base
            dNLX_dt = np.gradient(NLX, dt)
            dNLY_dt = np.gradient(NLY, dt)
            dX_dz = -1j*a_in * (NLX - 1j*shock_coeff * dNLX_dt)
            dY_dz = -1j*a_in * (NLY - 1j*shock_coeff * dNLY_dt)

        '''
        if optical_shock == False: 
            # Define the derivatives (Nonlinear step)
            #dX_dz = -1j*a_in * (C*X_abs2 + 2*P*Y_abs2) * X - 1j*a_in*P*Y**2*np.conj(X) # Equation 2a
            #dY_dz = -1j*a_in * (C*Y_abs2 + 2*P*X_abs2) * Y - 1j*a_in*P*X**2*np.conj(Y) # Equation 2b
            dX_dz = -1j*a_in * (2*P*Y_abs2) * X #- 1j*a_in*P*Y**2*np.conj(X) # Equation 2a
            dY_dz = -1j*a_in * (2*P*X_abs2) * Y #- 1j*a_in*P*X**2*np.conj(Y) # Equation 2b
            
                First term = self-phase modulation
                Second term = cross-phase modulation
                Third term = degenerate four-wave mixing
                
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
            '''
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
    S3 = 2*np.imag(Ex*np.conj(Ey))     # sign convention may vary
    return S0, S1, S2, S3

def polarization_orientation_and_ellipticity(Ex, Ey, degrees=True):
    S0, S1, S2, S3 = stokes_parameters(Ex, Ey)
    psi = 0.5*np.unwrap(np.arctan2(S2, S1))
    #psi = 0.5*np.arctan2(S2, S1)
    chi = 0.5*np.arcsin(S3/S0)
    if degrees:
        psi = np.rad2deg(psi)
        chi = np.rad2deg(chi)
    return psi, chi
'''
def fwhm_from_intensity(t, I):
    half_max = np.max(I) / 2
    peak_idx = np.argmax(I)

    # walk left from the peak until dropping below half-max
    left = peak_idx
    while left > 0 and I[left] > half_max:
        left -= 1
    # walk right from the peak until dropping below half-max
    right = peak_idx
    while right < len(I) - 1 and I[right] > half_max:
        right += 1

    if left == peak_idx or right == peak_idx:
        return np.nan  # peak at array edge, can't bracket it

    # linear interpolation for sub-sample precision
    t_left = t[left] + (half_max - I[left]) / (I[left+1] - I[left]) * (t[left+1] - t[left])
    t_right = t[right-1] + (half_max - I[right-1]) / (I[right] - I[right-1]) * (t[right] - t[right-1])

    return t_right - t_left
'''

def fwhm_from_intensity(t, I):
    half_max = np.max(I) / 2
    crossings = []

    for i in range(len(I) - 1):
        if (I[i] - half_max) == 0:
            crossings.append(t[i])
        elif (I[i] - half_max) * (I[i+1] - half_max) < 0:
            # linear interpolation
            frac = (half_max - I[i]) / (I[i+1] - I[i])
            crossings.append(t[i] + frac * (t[i+1] - t[i]))

    if len(crossings) >= 2:
        return crossings[-1] - crossings[0]
    return np.nan

def polarizer_jones(theta):
    '''Linear polarizer Jones matrix, axis at angle theta (rad) from lab X-axis.'''
    c, s = np.cos(theta), np.sin(theta)
    R       = np.array([[c, -s], [s,  c]])   # rotate lab frame -> polarizer frame
    R_inv   = np.array([[c,  s], [-s, c]])   # rotate polarizer frame -> lab frame (R(-theta))
    P       = np.array([[1, 0], [0, 0]])     # pass polarizer-frame "x", block polarizer-frame "y"
    return R_inv @ P @ R

def apply_polarizer(X_exit, Y_exit, theta):
    '''
    Applies a linear polarizer at angle theta to the exit fields.
    X_exit, Y_exit: 1D complex arrays over time (e.g. Xt_out[:, -1], Yt_out[:, -1])
    Returns X_pol, Y_pol — fields AFTER the polarizer, back in the lab (X, Y) frame.
    '''
    M = polarizer_jones(theta)
    E_in = np.vstack([X_exit, Y_exit])      # shape (2, Nt)
    E_out = M @ E_in
    return E_out[0], E_out[1]
'''
def find_eliminating_angle(X_exit, Y_exit):
    
    Returns the polarizer angle (rad) that BLOCKS the dominant ellipse axis,
    i.e. passes only the orthogonal component — or pass theta_psi directly
    to instead PASS the dominant axis (eliminate the orthogonal one).

    Uses psi at the pulse's actual peak (which shifts away from t=0 once
    dispersion + optical shock are on), not a fixed t=0 index.
    
    total_intensity = np.abs(X_exit)**2 + np.abs(Y_exit)**2
    peak_idx = np.argmax(total_intensity)

    psi, _ = polarization_orientation_and_ellipticity(Y_exit, X_exit)
    theta_psi = np.deg2rad(psi[peak_idx])
    return theta_psi
'''

def find_eliminating_angle(X_exit, Y_exit):
    """
    Returns the polarizer angle (rad) that lands closest to the true
    extrema of the transmission curve. Uses power-INTEGRATED Stokes
    parameters (summed over the whole pulse), matching how
    transmitted_fraction is computed — NOT the instantaneous polarization
    state at a single time sample, which drifts across the pulse and
    gives a biased angle.
    """
    S1_tot = np.sum(np.abs(X_exit)**2 - np.abs(Y_exit)**2)
    S2_tot = np.sum(2*np.real(X_exit * np.conj(Y_exit)))
    theta_psi = 0.5 * np.arctan2(S2_tot, S1_tot)
    if theta_psi < 0:
        theta_psi += np.pi
    return theta_psi

def is_single_lobe(I, energy_threshold=0.9):
    half_max = np.max(I) / 2
    peak_idx = np.argmax(I)
    left = peak_idx
    while left > 0 and I[left] > half_max:
        left -= 1
    right = peak_idx
    while right < len(I) - 1 and I[right] > half_max:
        right += 1
    lobe_energy = np.sum(I[left:right+1])
    return (lobe_energy / np.sum(I)) > energy_threshold

#%% Intensity

'''
Pulse intensity vs time plots

X axis: time (fs)
Y axis: |field|² (intensity)

Shows how the Gaussian pulse envelope changes shape as it propagates. 
Y is the dominant (pump-like) pol arization. X starts tiny (flat) and then may 
grow. The four plots show pump depletion and signal growth in time.
'''
Xt_out, Yt_out = propagate(Y0_base, Tau_base, B_base, None, None, None, None, True, True)

plt.figure()
# [:,0] means first element of every row (start)
plt.plot(t_array*1e15, np.abs(Yt_out[:, 0])**2, label='Y(t) at 0 µm')
# [:,-1] means last element of every row (end)
plt.plot(t_array*1e15, np.abs(Yt_out[:, -1])**2, label='Y(t) at 500 µm')
plt.plot(t_array*1e15, np.abs(Xt_out[:, 0])**2, label='X(t) at 0 µm')
plt.plot(t_array*1e15, np.abs(Xt_out[:, -1])**2, label='X(t) at 500 µm')
plt.xlabel('Time (fs)')
plt.ylabel('Pulse Intensity (arb. units)')
plt.legend()

#%% Pulse Duration + Input Angle Parameters Sweep

'''# a smooth reference: analytic-ish via much finer finite difference per-point
def beta2_fd(lam0, dlam=1e-11):
    lam = np.array([lam0-dlam, lam0, lam0+dlam])
    n = (2.956362 + 0.02195770/((lam*1e6)**2 - 0.01428322) - 0.01062387*(lam*1e6)**2 - 0.0000204968*(lam*1e6)**4)**0.5
    d2n = (n[0]-2*n[1]+n[2])/dlam**2
    return lam0**3/(2*np.pi*cc**2)*d2n
'''

def sellmeier_n(lam):
    """Sellmeier index for MgO, lam in meters."""
    lam_um = lam * 1e6
    return (2.956362 + 0.02195770/(lam_um**2 - 0.01428322)
            - 0.01062387*lam_um**2 - 0.0000204968*lam_um**4)**0.5

def beta2_fd(lam0, dlam=1e-11):
    """Fine centered finite-difference d^2n/dlambda^2 -> beta2, evaluated
    at a single wavelength with a small, convergence-tested step size —
    independent of the coarse sweep grid, so it isn't polluted by the
    100-point spacing."""
    n_minus = sellmeier_n(lam0 - dlam)
    n0      = sellmeier_n(lam0)
    n_plus  = sellmeier_n(lam0 + dlam)
    d2n = (n_minus - 2*n0 + n_plus) / dlam**2
    return lam0**3 / (2*np.pi*cc**2) * d2n


def sweep_2d_params(param_val, param_name):

    result_map = np.full((len(param_val), len(B_sweep)), np.nan)
    
    '''
    if param_name == 'Wavelength':
        n_arr = (2.956362 + 0.02195770/((param_val*1e6)**2 - 0.01428322) - 0.01062387*(param_val*1e6)**2 - 0.0000204968*(param_val*1e6)**4)**0.5
        beta2_arr = np.gradient(np.gradient(n_arr, param_val), param_val) * param_val**3 / (2*np.pi*cc**2)
        #beta2_arr = np.array([beta2_fd(l) for l in param_val])
    ''' 
    
    if param_name == 'Wavelength':
        n_arr = np.array([sellmeier_n(l) for l in param_val])
        beta2_arr = np.array([beta2_fd(l) for l in param_val])   # fine FD, not np.gradient twice

    for i, val in enumerate(param_val):
        if param_name == 'Wavelength':
            n_scan = n_arr[i]
            beta2_scan = beta2_arr[i]
            wp_scan = 2*np.pi*cc / val
            a_scan = 2*np.pi/val * 3/8 * 1/n_scan
            phase_scan = np.exp(-1j * 0.5 * beta2_scan * (w - wp_scan*0)**2 * dz_base)
            
        elif param_name == 'Length':
            dz_scan = 5e-6
            z_eval_scan = np.arange(0, val + dz_scan, dz_scan)
            phase_scan_L = np.exp(-1j * 0.5 * beta2_base * (w - wp_base*0)**2 * dz_scan)
                
            
        for j, angle_val in enumerate(B_sweep):
            if param_name == 'Wavelength':
                Xt_scan, Yt_scan = propagate(Y0_base, Tau_base, angle_val, z_eval_base, dz_base, phase_scan, a_scan, True, False)

            elif param_name == 'Amplitude':
                Xt_scan, Yt_scan = propagate(val, Tau_base, angle_val, z_eval_base, dz_base, phase_base, a_base, True, False)

            elif param_name == 'Length':
                Xt_scan, Yt_scan = propagate(Y0_base, Tau_base, angle_val, z_eval_scan, dz_scan, phase_scan_L, a_base, True, False)

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
            
            #if not is_single_lobe(intensity):
                #result_map[i, j] = np.nan   # or a sentinel, so it's excluded from the 1/√3 contour entirely
            
            fwhm_out = fwhm_from_intensity(t_array, intensity)
            result_map[i, j] = fwhm_out
            print(f"  [{param_name}] i={i+1}/{len(param_val)}, j={j+1}/{len(B_sweep)} — FWHM out = {fwhm_out*1e15:.1f} fs")

    return result_map

Y0_output_fwhm = np.array(sweep_2d_params(Y0_sweep, 'Amplitude'))
L_output_fwhm = np.array(sweep_2d_params(L_sweep, 'Length'))
Lambda_output_fwhm = np.array(sweep_2d_params(Lambda_sweep, 'Wavelength'))

#%% Sweep plot

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Output Pulse Duration vs Input Angle', fontsize=14, fontweight='bold')

angle_deg = np.rad2deg(B_sweep)
threshold_fs = FWHM_base * 1e15 / np.sqrt(3)   # 1/√3 compression limit, in fs

sweep_data = [
    (I_sweep * 1e-16,      'Peak Intensity (10¹⁶ W/m²)',    Y0_output_fwhm   * 1e15),
    (L_sweep * 1e6,        'Crystal Length (µm)',           L_output_fwhm    * 1e15),
    (Lambda_sweep * 1e9,   'Wavelength (nm)',               Lambda_output_fwhm * 1e15)
]

for ax, (y_vals, y_label, fwhm_map) in zip(axes.flat, sweep_data):
    im = ax.pcolormesh(angle_deg, y_vals, fwhm_map, cmap='inferno', shading='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Output FWHM (fs)', fontsize=10)
    
    # 1/√3 theoretical compression limit, overlaid as a contour
    cs = ax.contour(angle_deg, y_vals, fwhm_map, levels=[threshold_fs], colors='cyan', linewidths=1.5)
    ax.clabel(cs, fmt=lambda v: '1/√3 limit', fontsize=8)
    
    ax.set_xlabel('Input Angle (deg)', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f'Output FWHM vs Input Angle\n& {y_label}', fontsize=10)

plt.tight_layout()

#%% Sanity check polarizer matrix

test_X = np.array([1.0+0j])
test_Y = np.array([0.0+0j])

for theta_deg in [0, 45, 90]:
    Xp, Yp = apply_polarizer(test_X, test_Y, np.deg2rad(theta_deg))
    print(f"theta={theta_deg:3d}°  ->  X_out={Xp[0]:.3f}  Y_out={Yp[0]:.3f}  |E_out|={np.hypot(abs(Xp[0]), abs(Yp[0])):.3f}")
    
#%% Sweep polarizer angle independently of crystal angle B_base

# Get fields from passing through an MgO crystal
Xt_out, Yt_out = propagate(Y0_base, Tau_base, B_base, z_eval_base, dz_base,
                            phase_base, a_base, dispersion=True, optical_shock=True)

# Get exit fields after the crystal
X_exit, Y_exit = Xt_out[:, -1], Yt_out[:, -1]

# Total power going INTO the polarizer
total_power_in = np.sum(np.abs(X_exit)**2 + np.abs(Y_exit)**2)

# Polarizer angle, independent of B_base (crystal angle)
theta_sweep = np.linspace(0, np.pi, 181)  
 
# Empty lists
transmitted_fraction = []
output_fwhm_X = []

# Rotate the polarizer
for theta in theta_sweep:   
    # Finds the fields AFTER the polarizer has been applied
    X_pol, Y_pol = apply_polarizer(X_exit, Y_exit, theta)
    
    # Transmitted power
    power_out = np.sum(np.abs(X_pol)**2 + np.abs(Y_pol)**2)
    
    # Turn into fraction for percentage
    transmitted_fraction.append(power_out / total_power_in)

    # Intensity
    intensity_X = np.abs(X_pol)**2
    
    # Check if the intensity is always above zero
    if np.max(intensity_X) > 0:
        output_fwhm_X.append(fwhm_from_intensity(t_array, intensity_X))
    else:
        output_fwhm_X.append(np.nan)
        
# Arrays
transmitted_fraction = np.array(transmitted_fraction)
output_fwhm_X = np.array(output_fwhm_X)

# Angle that eliminates the orthogonal component (passes only the dominant ellipse axis)
theta_eliminate = find_eliminating_angle(X_exit, Y_exit)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(np.rad2deg(theta_sweep), transmitted_fraction*100)
axes[0].axvline(np.rad2deg(theta_eliminate), color='red', linestyle='--', label=f'ψ (dominant axis) = {np.rad2deg(theta_eliminate):.1f}°')
axes[0].set_xlabel('Polarizer Angle (deg)')
axes[0].set_ylabel('Transmitted Power (%)')
axes[0].set_title('Polarizer Transmission vs Angle')
axes[0].legend()

axes[1].plot(np.rad2deg(theta_sweep), output_fwhm_X*1e15)
axes[1].axvline(np.rad2deg(theta_eliminate), color='red', linestyle='--', label=f'ψ (dominant axis) = {np.rad2deg(theta_eliminate):.1f}°')
axes[1].set_xlabel('Polarizer Angle (deg)')
axes[1].set_ylabel('Output X FWHM (fs)')
axes[1].set_title('Post-Polarizer Pulse Duration vs Angle')
axes[1].legend()

plt.tight_layout()

# Max and min points + point where dominant axis vert. line meets function
power_min = np.min(transmitted_fraction)
power_max = np.max(transmitted_fraction)
power_axis = np.interp(theta_eliminate, theta_sweep, transmitted_fraction)

angle_axis = np.rad2deg(theta_eliminate)
angle_min = np.where(transmitted_fraction == power_min)[0][0]
angle_max = np.where(transmitted_fraction == power_max)[0][0]

print(f"\n{'ψ measurement':^24} | {'angle (deg)':^24} | {'transmission at that angle':^36}")
print(f"{'ψ at peak of pulse':^24} | {angle_axis:^24.2f} | {power_axis:^36.2%}")
print(f"{'numerical minimum':^24} | {angle_min:^24.2f} | {power_min:^36.2%}")
print(f"{'numerical maximum':^24} | {angle_max:^24.2f} | {power_max:^36.2%}")

#%% Pulse Duration/Transmission Polarizer Angle Parameters Sweep (Polaarizer)

# Polarizer angle, independent of B_base (crystal angle)
theta_sweep = np.linspace(0, np.pi, NP)  

def polarizer_sweep_2d_params(param_val, param_name):
    
    result_map_trans = np.full((len(param_val), len(theta_sweep)), np.nan)
    result_map_fwhm = np.full((len(param_val), len(theta_sweep)), np.nan)
    single_lobe_mask = np.full((len(param_val), len(theta_sweep)), True)   # NEW

    for i, val in enumerate(param_val):
        for j, theta_val in enumerate(theta_sweep):
            if param_name == 'Input Angle':
                Xt_scan, Yt_scan = propagate(Y0_base, Tau_base, val, z_eval_base, dz_base, phase_base, a_base, True, True)
            elif param_name == 'Amplitude':
                Xt_scan, Yt_scan = propagate(val, Tau_base, B_base, z_eval_base, dz_base, phase_base, a_base, True, True)

            X_exit, Y_exit = Xt_scan[:, -1], Yt_scan[:, -1]
            total_power_in_local = np.sum(np.abs(X_exit)**2 + np.abs(Y_exit)**2)
            X_pol, Y_pol = apply_polarizer(X_exit, Y_exit, theta_val)
            power_out = np.sum(np.abs(X_pol)**2 + np.abs(Y_pol)**2)
            result_map_trans[i, j] = power_out / total_power_in_local

            intensity_X = np.abs(X_pol)**2
            if np.max(intensity_X) > 0:
                result_map_fwhm[i, j] = fwhm_from_intensity(t_array, intensity_X)
                single_lobe_mask[i, j] = is_single_lobe(intensity_X)   # NEW
            else:
                result_map_fwhm[i, j] = 0
                single_lobe_mask[i, j] = True   # NEW (zero-field edge case, not a fragmentation issue)

            print(f"  [{param_name}] i={i+1}/{len(param_val)}, j={j+1}/{len(theta_sweep)}")

    return result_map_trans, result_map_fwhm, single_lobe_mask   # NEW return

Y0_trans_frac, Y0_output_fwhm, Y0_single_lobe = polarizer_sweep_2d_params(Y0_sweep, 'Amplitude')
B_trans_frac, B_output_fwhm, B_single_lobe = polarizer_sweep_2d_params(B_sweep, 'Input Angle')

#%% Sweep plot 2

fig, axes = plt.subplots(2, 2, figsize=(12, 14))
fig.suptitle('Polarizer Transmission/Post-Polarizer Pulse Duration vs Angle', fontsize=14, fontweight='bold')

theta_deg = np.rad2deg(theta_sweep)

sweep_data = [
    (I_sweep * 1e-16,       'Peak Intensity (10¹⁶ W/m²)',   Y0_trans_frac,  Y0_output_fwhm*1e15,  Y0_single_lobe),
    (np.rad2deg(B_sweep),   'Crystal Input Angle (deg)',    B_trans_frac,   B_output_fwhm*1e15,   B_single_lobe)
]

for row, (y_vals, y_label, trans_map, fwhm_map, lobe_mask) in enumerate(sweep_data):
    ax_trans = axes[row, 0]
    ax_fwhm  = axes[row, 1]
    X, Y = np.meshgrid(theta_deg, y_vals)

    pcm1 = ax_trans.pcolormesh(X, Y, trans_map * 100, shading='auto', cmap='viridis')
    fig.colorbar(pcm1, ax=ax_trans).set_label('Transmitted Power (%)', fontsize=10)

    pcm2 = ax_fwhm.pcolormesh(X, Y, fwhm_map, shading='auto', cmap='inferno')
    fig.colorbar(pcm2, ax=ax_fwhm).set_label('Output X FWHM (fs)', fontsize=10)

    # NEW: outline the single-lobe / fragmented boundary directly on top
    ax_fwhm.contour(X, Y, lobe_mask.astype(float), levels=[0.5], colors='cyan', linewidths=1.5)

    ax_trans.set_xlabel('Polarizer Angle (deg)', fontsize=11)
    ax_trans.set_ylabel(y_label, fontsize=11)
    ax_trans.set_title(f'Polarizer Transmission vs Polarizer Angle\n& {y_label}', fontsize=10)
    
    
    ax_fwhm.set_xlabel('Polarizer Angle (deg)', fontsize=11)
    ax_fwhm.set_ylabel(y_label, fontsize=11)
    ax_fwhm.set_title(f'Post-Polarizer Pulse Duration vs Polarizer Angle\n& {y_label}', fontsize=10)

plt.tight_layout()
plt.show()

#%% Stage 1 — Radial beam profile, no diffraction coupling

# --- Radial grid ---
Nr = 1024
w0_base = 100e-6                     # 1/e^2 beam waist radius (m) — pick a physically reasonable value for your setup
r_max = 4 * w0_base                  # go out to 4 waists so the Gaussian tail is negligible at the edge
r_array = np.linspace(-1*r_max, r_max, Nr)  # uniform for now; will move to QDHT-appropriate (Bessel-zero) grid in Stage 2

def spatial_profile(r):
    """Gaussian spatial envelope, amplitude (not intensity) form:
    E(r) = E(0) * exp(-r^2 / w0^2)."""
    return np.exp(-r**2 / w0_base**2)

R_weight = spatial_profile(r_array)   # shape (Nr,), radial amplitude scaling

# --- Propagate independently at each radius ---
# Xt_out_2d, Yt_out_2d: shape (Nt, Nr, len(z_eval)) would be memory-heavy for
# full z-history at every radius; for Stage 1 we only need the FIELD, not the
# full z-history, at every radius, so just keep the final z output plus z=0.
Xt_r0 = np.zeros((Nt, Nr), dtype=complex)   # field at z = 0, all radii
Yt_r0 = np.zeros((Nt, Nr), dtype=complex)
Xt_rL = np.zeros((Nt, Nr), dtype=complex)   # field at z = L_base, all radii
Yt_rL = np.zeros((Nt, Nr), dtype=complex)

for ir, r_val in enumerate(r_array):
    Y0_r = Y0_base * R_weight[ir]     # scale the on-axis field amplitude by the local radial weight
    Xt_out_r, Yt_out_r = propagate(Y0_r, Tau_base, B_base, None, None, None, None,
                                    dispersion=True, optical_shock=True)
    if Xt_out_r is None:
        print(f"  r={r_val*1e6:.1f} µm — blew up, skipping")
        continue
    Xt_r0[:, ir] = Xt_out_r[:, 0]
    Yt_r0[:, ir] = Yt_out_r[:, 0]
    Xt_rL[:, ir] = Xt_out_r[:, -1]
    Yt_rL[:, ir] = Yt_out_r[:, -1]
    print(f"  r={r_val*1e6:.1f} µm done ({ir+1}/{Nr})")

#%% Stage 1 — sanity check plots

# (t, r) intensity map at crystal exit — should look like a smooth radially-
# decaying version of your existing on-axis Y(t)/X(t) plot, with no radial
# mixing (each column is literally an independent 1D-time simulation).
intensity_Y_rL = np.abs(Yt_rL)**2
intensity_X_rL = np.abs(Xt_rL)**2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].pcolormesh(r_array*1e6, t_array*1e15, intensity_Y_rL, shading='auto', cmap=inferno_white)
axes[0].set_xlabel('Radius (µm)')
axes[0].set_ylabel('Time (fs)')
axes[0].set_title('Y(t,r) Intensity at z = L (no diffraction)')
fig.colorbar(im0, ax=axes[0])

im1 = axes[1].pcolormesh(r_array*1e6, t_array*1e15, intensity_X_rL, shading='auto', cmap=inferno_white)
axes[1].set_xlabel('Radius (µm)')
axes[1].set_ylabel('Time (fs)')
axes[1].set_title('X(t,r) Intensity at z = L (no diffraction)')
fig.colorbar(im1, ax=axes[1])
plt.tight_layout()

# Peak intensity vs radius, at z=0 and z=L — should each be a clean Gaussian
# (z=0 is exactly Gaussian by construction; z=L will show some narrowing from
# SPM at high radial amplitude, but still monotonically decaying, no rings
# yet since there's no diffraction to redistribute energy between radii)
plt.figure()
plt.plot(r_array*1e6, np.max(np.abs(Yt_r0)**2, axis=0), label='Y peak intensity, z=0')
plt.plot(r_array*1e6, np.max(np.abs(Yt_rL)**2, axis=0), label='Y peak intensity, z=L')
plt.xlabel('Radius (µm)')
plt.ylabel('Peak Intensity (arb. units)')
plt.legend()
plt.title('Radial Peak Intensity Profile (no diffraction)')

#%% Stage 2 — Hankel/QDHT diffraction only (no nonlinearity yet), via PyHank

w0_base = 100e-6   # beam waist radius (m)

# --- Set up the Hankel transform grid ---
# order=0 for a radially-symmetric (no orbital angular momentum) Gaussian beam.
# max_radius sets how far out the grid goes; n_points is the radial resolution.
n_points = 512
r_max = 6 * w0_base   # generous margin so the field ~0 at the boundary (QDHT accuracy requirement)
H = HankelTransform(order=0, max_radius=r_max, n_points=n_points)

r_array = H.r     # non-uniform, Bessel-zero-based radial grid (not linspace)

# --- Initial field, sampled on an arbitrary convenient grid then resampled ---
# PyHank wants the field defined on H.r directly, or resampled from another
# grid via H.to_transform_r(). Easiest is to just evaluate directly on H.r:
def gaussian_field(r, w0):
    return np.exp(-r**2 / w0**2)

E_r = gaussian_field(r_array, w0_base).astype(complex)

# --- Propagation setup ---
n_base_785 = n_base
k0 = 2*np.pi*n_base_785 / Lambda_base

zR = np.pi * w0_base**2 * n_base_785 / Lambda_base
# z_steps = np.linspace(0, 5*zR, 200)   # go out to 5 Rayleigh ranges instead of 500 µm

z_steps = np.linspace(0, 10*L_base, 512)
dz_diff = z_steps[1] - z_steps[0]   # now consistent with the actual z range


beam_width_vs_z = []

for z in z_steps:
    Ekr = H.qdht(E_r)
    diffraction_phase = np.exp(-1j * H.kr**2 * dz_diff / (2*k0))
    Ekr = Ekr * diffraction_phase
    E_r = H.iqdht(Ekr)

    I_r = np.abs(E_r)**2
    if np.max(I_r) > 0:
        target = np.max(I_r) / np.e**2
        idx_above = np.where(I_r >= target)[0]
        if len(idx_above) > 0 and idx_above[-1] < len(r_array)-1:
            i0 = idx_above[-1]
            # linear interpolation between i0 and i0+1 for the true 1/e^2 crossing
            r0, r1 = r_array[i0], r_array[i0+1]
            I0, I1 = I_r[i0], I_r[i0+1]
            frac = (target - I0) / (I1 - I0) if I1 != I0 else 0
            w_est = r0 + frac*(r1 - r0)
        else:
            w_est = r_array[idx_above[-1]] if len(idx_above) > 0 else np.nan
    else:
        w_est = np.nan
    beam_width_vs_z.append(w_est)
    
beam_width_vs_z = np.array(beam_width_vs_z)

# --- Analytic comparison ---
zR = np.pi * w0_base**2 * n_base_785 / Lambda_base
w_analytic = w0_base * np.sqrt(1 + (z_steps/zR)**2)

plt.figure()
plt.plot(z_steps*1e6, beam_width_vs_z*1e6, label='PyHank QDHT numerical')
plt.plot(z_steps*1e6, w_analytic*1e6, '--', label='Analytic Gaussian diffraction')
plt.xlabel('z (µm)')
plt.ylabel('Beam radius (µm)')
plt.title('Diffraction-only beam spreading: PyHank vs analytic')
plt.legend()


plt.figure()
plt.plot(r_array*1e6, np.abs(gaussian_field(r_array, w0_base))**2, label='Initial |E|²')
plt.plot(r_array*1e6, np.abs(E_r)**2 / np.max(np.abs(E_r)**2) * np.max(np.abs(gaussian_field(r_array, w0_base))**2), label='Final |E|² (renormalized shape)')
plt.xlabel('Radius (µm)')
plt.ylabel('Intensity (arb.)')
plt.title(f'Radial profile at z=0 vs z={z_steps[-1]*1e6:.0f} µm')
plt.legend()

#%% Grid validation: round-trip fidelity + energy conservation, WITH resampling stress test

def test_grid_fidelity2(w0, n_points, r_max_factor=6, sharpen_power=1, uniform_n=2000):
    r_max = r_max_factor * w0
    H = HankelTransform(order=0, max_radius=r_max, n_points=n_points)

    # Field defined on an independent uniform grid, NOT on H.r
    r_uniform = np.linspace(0, r_max, uniform_n)
    E_uniform = (np.exp(-(r_uniform**2 / w0**2)))**sharpen_power
    E_uniform = E_uniform.astype(complex)

    # Manually resample onto the QDHT grid (equivalent to what
    # to_transform_r() does internally, but works regardless of how H was built)
    interp_re = interp1d(r_uniform, E_uniform.real, kind='cubic', bounds_error=False, fill_value=0.0)
    interp_im = interp1d(r_uniform, E_uniform.imag, kind='cubic', bounds_error=False, fill_value=0.0)
    E_on_H = interp_re(H.r) + 1j*interp_im(H.r)

    # Round-trip through the transform
    E_kr = H.qdht(E_on_H)
    E_out = H.iqdht(E_kr)

    # Pointwise fidelity (on H.r grid)
    rel_error = np.abs(E_out - E_on_H) / np.max(np.abs(E_on_H))
    max_rel_error = np.max(rel_error)

    # Energy conservation check
    energy_before = np.trapz(np.abs(E_uniform)**2 * r_uniform, r_uniform)
    energy_after  = np.trapz(np.abs(E_out)**2 * H.r, H.r)
    energy_rel_change = (energy_after - energy_before) / energy_before

    return H.r, E_on_H, E_out, max_rel_error, energy_rel_change

# --- Sweep n_points, both for a plain Gaussian and a sharpened one ---
w0_test = w0_base
n_points_list = [64, 128, 256, 512, 1024, 2048]

print(f"{'n_points':>10} | {'sharpen=1 (plain)':>27} | {'sharpen=3 (peaked)':>27}")
print(f"{'':>10} | {'max err':>13} {'ΔE/E':>13} | {'max err':>13} {'ΔE/E':>13}")
for n_pts in n_points_list:
    _, _, _, err1, dE1 = test_grid_fidelity2(w0_test, n_pts, sharpen_power=1)
    _, _, _, err3, dE3 = test_grid_fidelity2(w0_test, n_pts, sharpen_power=3)
    print(f"{n_pts:10d} | {err1:13.2e} {dE1:13.2%} | {err3:13.2e} {dE3:13.2%}")
    
#%% Stage 3 — Full split-step: nonlinearity + temporal dispersion + radial diffraction

# --- Radial grid (update R_max/n_points per your round-trip validation) ---
w0_base = 100e-6
n_points_r = 512
R_max = 500e-6          # generously > expected beam growth over your 5 cm test range -- adjust after validating

def coupled_equations_2d(state, Nt_, Nr_, spm=True, xpm=True, fwm=True):
    n_tot = Nt_ * Nr_
    X = state[:n_tot].reshape(Nt_, Nr_) + 1j*state[n_tot:2*n_tot].reshape(Nt_, Nr_)
    Y = state[2*n_tot:3*n_tot].reshape(Nt_, Nr_) + 1j*state[3*n_tot:].reshape(Nt_, Nr_)

    X_abs2, Y_abs2 = np.abs(X)**2, np.abs(Y)**2
    spm_term_X = C * X_abs2 if spm else 0
    spm_term_Y = C * Y_abs2 if spm else 0
    xpm_term_X = 2*P * Y_abs2 if xpm else 0
    xpm_term_Y = 2*P * X_abs2 if xpm else 0
    fwm_term_X = P * Y**2 * np.conj(X) if fwm else 0
    fwm_term_Y = P * X**2 * np.conj(Y) if fwm else 0

    NLX = (spm_term_X + xpm_term_X)*X + fwm_term_X
    NLY = (spm_term_Y + xpm_term_Y)*Y + fwm_term_Y
    shock_coeff = 1/wp_base
    dNLX_dt = np.gradient(NLX, dt, axis=0)
    dNLY_dt = np.gradient(NLY, dt, axis=0)
    dX_dz = -1j*a_base * (NLX - 1j*shock_coeff * dNLX_dt)
    dY_dz = -1j*a_base * (NLY - 1j*shock_coeff * dNLY_dt)

    return np.hstack([dX_dz.real.ravel(), dX_dz.imag.ravel(),
                       dY_dz.real.ravel(), dY_dz.imag.ravel()])


def propagate_2d(Y0, tau, B, z_eval, dz, H, r_array, Nr, dispersion=True, diffraction=True, spm=True, xpm=True, fwm=True):
    Nt_ = Nt   # temporal grid size stays global (unchanged across this sweep)
    k0_local = 2*np.pi * n_base / Lambda_base

    T = np.exp(-t_array**2/(2*tau**2))[:, None]
    R = np.exp(-r_array**2/w0_base**2)[None, :]
    Y_field = (Y0*np.cos(B)) * T * R + 0j
    X_field = (Y0*np.sin(B)) * T * R + 0j

    for i in range(len(z_eval) - 1):
        state = np.hstack([X_field.real.ravel(), X_field.imag.ravel(),
                            Y_field.real.ravel(), Y_field.imag.ravel()])
        k1 = coupled_equations_2d(state, Nt_, Nr, spm, xpm, fwm)
        k2 = coupled_equations_2d(state + 0.5*dz*k1, Nt_, Nr, spm, xpm, fwm)
        k3 = coupled_equations_2d(state + 0.5*dz*k2, Nt_, Nr, spm, xpm, fwm)
        k4 = coupled_equations_2d(state + dz*k3, Nt_, Nr, spm, xpm, fwm)
        sol = state + (dz/6)*(k1 + 2*k2 + 2*k3 + k4)

        n_tot = Nt_*Nr
        X_field = sol[:n_tot].reshape(Nt_, Nr) + 1j*sol[n_tot:2*n_tot].reshape(Nt_, Nr)
        Y_field = sol[2*n_tot:3*n_tot].reshape(Nt_, Nr) + 1j*sol[3*n_tot:].reshape(Nt_, Nr)

        if np.any(np.isnan(X_field)) or np.any(np.isnan(Y_field)):
            print(f"Blew up at step {i}")
            return None, None

        if dispersion:
            Xw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(X_field, axes=0), axis=0), axes=0)
            Yw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Y_field, axes=0), axis=0), axes=0)
            Xw *= phase_base[:, None]
            Yw *= phase_base[:, None]
            X_field = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Xw, axes=0), axis=0), axes=0)
            Y_field = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Yw, axes=0), axis=0), axes=0)

        if diffraction:
            Xkr = H.qdht(X_field, axis=1)
            Ykr = H.qdht(Y_field, axis=1)
            diff_phase = np.exp(-1j * H.kr**2 * dz / (2*k0_local))[None, :]
            Xkr *= diff_phase
            Ykr *= diff_phase
            X_field = H.iqdht(Xkr, axis=1)
            Y_field = H.iqdht(Ykr, axis=1)
        print(f"Propagate Iteration {i+1}")
    return X_field, Y_field

#%% Stage 3 presentation plot: on-axis intensity, diffraction on vs off

X_diff_off, Y_diff_off = propagate_2d(Y0_base, Tau_base, B_base, z_eval_base, dz_base,
                                        H, r_array, Nr, dispersion=True, diffraction=False)
X_diff_on, Y_diff_on = propagate_2d(Y0_base, Tau_base, B_base, z_eval_base, dz_base,
                                      H, r_array, Nr, dispersion=True, diffraction=True)

ir0 = 0  # radius closest to axis

plt.figure()
plt.plot(t_array*1e15, np.abs(Y_diff_off[:, ir0])**2, label='Y(t), diffraction OFF')
plt.plot(t_array*1e15, np.abs(Y_diff_on[:, ir0])**2, '--', label='Y(t), diffraction ON')
plt.xlabel('Time (fs)')
plt.ylabel('Intensity (arb. units)')
plt.title(f'On-axis (r={r_array[ir0]*1e6:.2f} µm) output, z = L')
plt.legend()

#%% Stage 3 presentation plot: radial intensity slice at fixed time, diffraction on vs off

# Use the pulse's temporal peak (on-axis) as the fixed time slice, so the
# comparison is at the moment where the nonlinear/diffraction coupling is strongest
peak_t_idx_off = np.argmax(np.abs(Y_diff_off[:, 0])**2)
peak_t_idx_on  = np.argmax(np.abs(Y_diff_on[:, 0])**2)
# these may differ slightly since diffraction can shift the peak in time;
# use the diffraction-OFF peak index as the shared reference time for a fair comparison
t_idx_fixed = peak_t_idx_off

plt.figure()
plt.plot(r_array*1e6, np.abs(Y_diff_off[t_idx_fixed, :])**2, label='Y(r), diffraction OFF')
plt.plot(r_array*1e6, np.abs(Y_diff_on[t_idx_fixed, :])**2, '--', label='Y(r), diffraction ON')
plt.xlabel('Radius (µm)')
plt.ylabel('Intensity (arb. units)')
plt.title(f'Radial profile at t = {t_array[t_idx_fixed]*1e15:.1f} fs, z = L')
plt.legend()
plt.xlim(0, 300)   # zoom to where the beam actually has meaningful intensity

#%% Stage 3 sanity check: diffraction=False should reproduce Stage 1

dz_test = dz_base/2
z_eval_test = np.arange(0, L_base, dz_test)

X_2d, Y_2d = propagate_2d(Y0_base, Tau_base, B_base, z_eval_test, dz_test,
                           dispersion=True, diffraction=True)

# Pick a few radii to spot-check against independent 1D propagate() calls
check_indices = [0, len(r_array)//4, len(r_array)//2, 3*len(r_array)//4]

for ir in check_indices:
    r_val = r_array[ir]
    R_weight_val = np.exp(-r_val**2 / w0_base**2)   # matches the R(r) built inside propagate_2d
    Y0_r = Y0_base * R_weight_val

    Xt_1d, Yt_1d = propagate(Y0_r, Tau_base, B_base, None, None, None, None,
                              dispersion=True, optical_shock=True)

    Y_2d_slice = Y_2d[:, ir]          # propagate_2d's Y(t) at this radius, z = L
    Y_1d_slice = Yt_1d[:, -1]         # independent 1D run's Y(t) at z = L

    rel_diff = np.max(np.abs(Y_2d_slice - Y_1d_slice)) / np.max(np.abs(Y_1d_slice))
    print(f"r={r_val*1e6:6.1f} µm | Y0_r={Y0_r:.3e} | max rel diff = {rel_diff:.2e}")
    
#%% Isolate: pure diffraction only inside propagate_2d, compare to Stage 2 analytic

X_diff_only, Y_diff_only = propagate_2d(Y0_base, Tau_base, B_base, z_eval_test, dz_test,
                                          dispersion=False, diffraction=True,
                                          spm=False, xpm=False, fwm=False)

# Compare the Y(t=peak, r) beam profile at z=L against the analytic Gaussian
# diffraction prediction over the same L, using your already-validated zR formula
L_total = z_eval_test[-1]
zR = np.pi * w0_base**2 * n_base / Lambda_base
w_analytic_L = w0_base * np.sqrt(1 + (L_total/zR)**2)

peak_t_idx = Nt // 2   # or np.argmax over the whole field if pulse has shifted
I_r_final = np.abs(Y_diff_only[peak_t_idx, :])**2

target = np.max(I_r_final) / np.e**2
idx_above = np.where(I_r_final >= target)[0]
w_numerical_L = r_array[idx_above[-1]] if len(idx_above) > 0 else np.nan

print(f"Analytic w(L)  = {w_analytic_L*1e6:.3f} µm")
print(f"Numerical w(L) = {w_numerical_L*1e6:.3f} µm")
print(f"Relative diff  = {(w_numerical_L-w_analytic_L)/w_analytic_L:.2%}")

#%% Round-trip energy check using the REAL post-nonlinear field, not a synthetic proxy

# Grab the field after just ONE nonlinear RK4 step (nonlinearity on, dispersion/diffraction off)
# so we isolate exactly what the nonlinear step hands to the diffraction step.
T = np.exp(-t_array**2/(2*Tau_base**2))[:, None]
R = np.exp(-r_array**2/w0_base**2)[None, :]
Y_field_0 = (Y0_base*np.cos(B_base)) * T * R + 0j
X_field_0 = (Y0_base*np.sin(B_base)) * T * R + 0j

state0 = np.hstack([X_field_0.real.ravel(), X_field_0.imag.ravel(),
                     Y_field_0.real.ravel(), Y_field_0.imag.ravel()])

k1 = coupled_equations_2d(state0, True, True, True)
k2 = coupled_equations_2d(state0 + 0.5*dz_test*k1, True, True, True)
k3 = coupled_equations_2d(state0 + 0.5*dz_test*k2, True, True, True)
k4 = coupled_equations_2d(state0 + dz_test*k3, True, True, True)
sol1 = state0 + (dz_base/6)*(k1 + 2*k2 + 2*k3 + k4)

n_tot = Nt*Nr
Y_after_NL = sol1[2*n_tot:3*n_tot].reshape(Nt, Nr) + 1j*sol1[3*n_tot:].reshape(Nt, Nr)

# Round-trip THIS real, nonlinearly-evolved field through the Hankel transform
Y_kr = H.qdht(Y_after_NL, axis=1)
Y_roundtrip = H.iqdht(Y_kr, axis=1)

# Energy check per time-slice (cylindrical measure), summed over the pulse
energy_before = np.trapz(np.abs(Y_after_NL)**2 * r_array[None, :], r_array, axis=1)
energy_after  = np.trapz(np.abs(Y_roundtrip)**2 * r_array[None, :], r_array, axis=1)

rel_energy_change = (energy_after - energy_before) / (energy_before + 1e-300)  # avoid /0 in the wings
peak_t = np.argmax(np.sum(np.abs(Y_after_NL)**2, axis=1))
print(f"Energy rel. change at pulse peak in time: {rel_energy_change[peak_t]:.2%}")
print(f"Max |energy rel. change| over all t:      {np.max(np.abs(rel_energy_change)):.2%}")

plt.figure()
plt.plot(r_array*1e6, np.abs(Y_after_NL[peak_t,:])**2, label='Before round-trip')
plt.plot(r_array*1e6, np.abs(Y_roundtrip[peak_t,:])**2, '--', label='After round-trip')
plt.xlabel('Radius (µm)')
plt.ylabel('|Y|² at pulse temporal peak')
plt.legend()
plt.title('Real nonlinear field: before vs after Hankel round-trip')

#%% Convergence check: does the diffraction+nonlinear result converge as dz shrinks?

for dz_factor in [1, 2, 4]:
    dz_test = dz_base / dz_factor
    z_eval_test = np.arange(0, L_base, dz_test)   # rebuilt correctly for each dz_test

    X_2d_test, Y_2d_test = propagate_2d(Y0_base, Tau_base, B_base, z_eval_test, dz_test,
                                         dispersion=True, diffraction=True)

    # Same comparison as before, at r=0.7 µm (the most sensitive/high-field radius)
    ir = 0
    r_val = r_array[ir]
    R_weight_val = np.exp(-r_val**2 / w0_base**2)
    Y0_r = Y0_base * R_weight_val
    Xt_1d, Yt_1d = propagate(Y0_r, Tau_base, B_base, None, None, None, None,
                              dispersion=True, optical_shock=True)

    rel_diff = np.max(np.abs(Y_2d_test[:, ir] - Yt_1d[:, -1])) / np.max(np.abs(Yt_1d[:, -1]))
    print(f"dz = dz_base/{dz_factor}  |  rel diff at r=0.7µm = {rel_diff:.2%}")
    
#%% n_points convergence check: does the ~17% discrepancy hold as radial grid refines?

# Reference: independent 1D propagate() at r=0.7 µm-equivalent field amplitude
ir_test_radius = 0.7e-6   # approx r value used in earlier tests; recomputed per grid below
R_weight_val = np.exp(-ir_test_radius**2 / w0_base**2)
Y0_r_ref = Y0_base * R_weight_val
Xt_1d, Yt_1d = propagate(Y0_r_ref, Tau_base, B_base, None, None, None, None,
                          dispersion=True, optical_shock=True)

print(f"{'n_points':>10} | {'r used (µm)':>12} | {'rel diff':>10}")
for n_pts_test in [256, 512, 1024]:
    H_test = HankelTransform(order=0, max_radius=R_max, n_points=n_pts_test)
    r_array_test = H_test.r
    Nr_test = n_pts_test

    z_eval_test = np.arange(0, L_base, dz_base)

    X_2d_test, Y_2d_test = propagate_2d(Y0_base, Tau_base, B_base, z_eval_test, dz_base,
                                         H_test, r_array_test, Nr_test,
                                         dispersion=True, diffraction=True)

    ir = 0   # first grid point, closest to axis for this grid's own spacing
    r_val = r_array_test[ir]
    Y0_r = Y0_base * np.exp(-r_val**2/w0_base**2)
    Xt_1d_i, Yt_1d_i = propagate(Y0_r, Tau_base, B_base, None, None, None, None,
                                  dispersion=True, optical_shock=True)

    rel_diff = np.max(np.abs(Y_2d_test[:, ir] - Yt_1d_i[:, -1])) / np.max(np.abs(Yt_1d_i[:, -1]))
    print(f"{n_pts_test:10d} | {r_val*1e6:12.3f} | {rel_diff:10.2%}")
    
#%% Stage 4 — Polarization/Stokes analysis in 2D

def apply_polarizer_2d(X_exit, Y_exit, theta):
    """
    2D-aware version of apply_polarizer for (Nt, Nr) fields.
    Uses the same Jones matrix as apply_polarizer/polarizer_jones,
    applied elementwise instead of via vstack+matmul (which only
    works for 1D (Nt,) inputs).
    """
    M = polarizer_jones(theta)   # (2,2), reuse existing helper unchanged
    X_pol = M[0,0]*X_exit + M[0,1]*Y_exit
    Y_pol = M[1,0]*X_exit + M[1,1]*Y_exit
    return X_pol, Y_pol

# Run the full physics: nonlinearity + dispersion + diffraction, at your validated grid
n_points_r = 512
H = HankelTransform(order=0, max_radius=R_max, n_points=n_points_r)
r_array = H.r
Nr = n_points_r

z_eval = np.arange(0, L_base, dz_base)
X_out, Y_out = propagate_2d(Y0_base, Tau_base, B_base, z_eval, dz_base,
                             H, r_array, Nr, dispersion=True, diffraction=True)

# --- ψ(t,r), χ(t,r): existing function already works on 2D input via broadcasting ---
psi_tr, chi_tr = polarization_orientation_and_ellipticity(Y_out, X_out)   # shape (Nt, Nr)

# --- Radial "ring" maps at the pulse's temporal peak (matches your earlier bullseye plots) ---
total_intensity_tr = np.abs(X_out)**2 + np.abs(Y_out)**2
peak_t_idx = np.argmax(np.sum(total_intensity_tr, axis=1))   # peak of the radially-integrated pulse

psi_at_peak_r = psi_tr[peak_t_idx, :]
chi_at_peak_r = chi_tr[peak_t_idx, :]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(r_array*1e6, psi_at_peak_r)
axes[0].set_xlabel('Radius (µm)')
axes[0].set_ylabel('ψ (deg)')
axes[0].set_title('Polarization orientation vs radius, at pulse peak')

axes[1].plot(r_array*1e6, chi_at_peak_r)
axes[1].set_xlabel('Radius (µm)')
axes[1].set_ylabel('χ (deg)')
axes[1].set_title('Ellipticity angle vs radius, at pulse peak')
plt.tight_layout()

# --- Full (t, r) polarization maps — this is the real bullseye/ring test now,
#     since diffraction is actually active and radii are coupled ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].pcolormesh(r_array*1e6, t_array*1e15, psi_tr, shading='auto', cmap=inferno_white)
fig.colorbar(im0, ax=axes[0], label='ψ (deg)')
axes[0].set_xlabel('Radius (µm)')
axes[0].set_ylabel('Time (fs)')
axes[0].set_title('ψ(t,r)')

im1 = axes[1].pcolormesh(r_array*1e6, t_array*1e15, chi_tr, shading='auto', cmap=inferno_white)
fig.colorbar(im1, ax=axes[1], label='χ (deg)')
axes[1].set_xlabel('Radius (µm)')
axes[1].set_ylabel('Time (fs)')
axes[1].set_title('χ(t,r)')
plt.tight_layout()

# --- Radius-resolved polarizer sweep: apply_polarizer already works per-radius via broadcasting ---
theta_sweep_2d = np.linspace(0, np.pi, 91)
trans_vs_r_theta = np.zeros((Nr, len(theta_sweep_2d)))

# total power in, per radius, integrated over the pulse (cylindrical-consistent weighting optional;
# using flat sum over t here since we're comparing polarizer performance per radius, not total energy)
power_in_per_r = np.sum(np.abs(X_out)**2 + np.abs(Y_out)**2, axis=0)   # shape (Nr,)

for j, theta in enumerate(theta_sweep_2d):
    X_pol, Y_pol = apply_polarizer_2d(X_out, Y_out, theta)
    power_out_per_r = np.sum(np.abs(X_pol)**2 + np.abs(Y_pol)**2, axis=0)
    trans_vs_r_theta[:, j] = power_out_per_r / (power_in_per_r + 1e-300)
    print(f"Loop 1: iteration {j+1}")

plt.figure()
im = plt.pcolormesh(np.rad2deg(theta_sweep_2d), r_array*1e6, trans_vs_r_theta*100,
                     shading='auto', cmap='viridis')
plt.colorbar(im, label='Transmitted Power (%)')
plt.xlabel('Polarizer Angle (deg)')
plt.ylabel('Radius (µm)')
plt.title('Polarizer transmission vs angle & radius')

# --- Radius-resolved eliminating angle: find_eliminating_angle needs a per-radius loop
#     since it does a scalar arctan2 on summed Stokes params, not a broadcastable op ---
theta_eliminate_vs_r = np.zeros(Nr)
for ir in range(Nr):
    theta_eliminate_vs_r[ir] = find_eliminating_angle(X_out[:, ir], Y_out[:, ir])
    print(f"Loop 2: iteration {ir+1}")

plt.figure()
plt.plot(r_array*1e6, np.rad2deg(theta_eliminate_vs_r))
plt.xlabel('Radius (µm)')
plt.ylabel('ψ dominant-axis angle (deg)')
plt.title('Eliminating polarizer angle vs radius')


#%% Polarization orientation vs B_sweep (HELP)

# Run the full physics: nonlinearity + dispersion + diffraction, at your validated grid
n_points_r = 512
H = HankelTransform(order=0, max_radius=R_max, n_points=n_points_r)
r_array = H.r
Nr = n_points_r

z_eval = np.arange(0, L_base, dz_base)
X_out, Y_out = propagate_2d(Y0_base, Tau_base, B_base, z_eval, dz_base,
                             H, r_array, Nr, dispersion=True, diffraction=True)

# --- ψ(t,r), χ(t,r): existing function already works on 2D input via broadcasting ---
psi_tr, chi_tr = polarization_orientation_and_ellipticity(Y_out, X_out)   # shape (Nt, Nr)

# --- Radial "ring" maps at the pulse's temporal peak (matches your earlier bullseye plots) ---
total_intensity_tr = np.abs(X_out)**2 + np.abs(Y_out)**2
peak_t_idx = np.argmax(np.sum(total_intensity_tr, axis=1))   # peak of the radially-integrated pulse

psi_at_peak_r = psi_tr[peak_t_idx, :]
chi_at_peak_r = chi_tr[peak_t_idx, :]

#%%
plt.figure()
plt.plot(np.rad2deg(B_sweep), psi_at_peak_r)
plt.xlabel('Input angle (deg)')
plt.ylabel('ψ (deg)')

#%% Stage 4 addition — radially-integrated polarizer efficiency

# Cylindrically-correct weight: power in an annulus ~ r dr, not a flat sum over r
r_weight = r_array  # np.trapz will handle the dr spacing from r_array itself

def radially_integrated_transmission(X_out, Y_out, theta):
    """
    Total transmitted power fraction through a polarizer at angle theta,
    integrated over the whole beam (all r), using cylindrical weighting.
    X_out, Y_out: (Nt, Nr) fields BEFORE the polarizer.
    """
    X_pol, Y_pol = apply_polarizer_2d(X_out, Y_out, theta)

    # power(t) per radius -> integrate over r (cylindrical) -> integrate over t
    power_in_tr  = np.abs(X_out)**2 + np.abs(Y_out)**2
    power_out_tr = np.abs(X_pol)**2 + np.abs(Y_pol)**2

    power_in_t  = np.trapz(power_in_tr  * r_weight[None, :], r_array, axis=1)
    power_out_t = np.trapz(power_out_tr * r_weight[None, :], r_array, axis=1)

    total_in  = np.sum(power_in_t)
    total_out = np.sum(power_out_t)
    return total_out / (total_in + 1e-300)

theta_sweep_r = np.linspace(0, np.pi, 181)
trans_vs_theta_integrated = np.array(
    [radially_integrated_transmission(X_out, Y_out, th) for th in theta_sweep_r]
)

plt.figure()
plt.plot(np.rad2deg(theta_sweep_r), trans_vs_theta_integrated * 100)
plt.xlabel('Polarizer Angle (deg)')
plt.ylabel('Transmitted Power (%)')
plt.title('Radially-integrated polarizer transmission vs angle')

#%% Stage 4 addition — efficiency vs propagation length

# Sweep crystal length L, at fixed polarizer angle(s); rerun propagate_2d for
# each length (analogous to the 'Length' branch in sweep_2d_params for 1D).
L_sweep_stage4 = np.linspace(50e-6, 500e-6, 10)   # keep coarse-ish, this is expensive
theta_fixed = find_eliminating_angle(X_out[:, 0], Y_out[:, 0])  # or pick a fixed angle of interest

trans_vs_L = np.zeros(len(L_sweep_stage4))
for iL, L_val in enumerate(L_sweep_stage4):
    dz_L = L_val / 100
    z_eval_L = np.arange(0, L_val, dz_L)
    X_L, Y_L = propagate_2d(Y0_base, Tau_base, B_base, z_eval_L, dz_L,
                             H, r_array, Nr, dispersion=True, diffraction=True)
    if X_L is None:
        trans_vs_L[iL] = np.nan
        continue
    trans_vs_L[iL] = radially_integrated_transmission(X_L, Y_L, theta_fixed)
    print(f"L = {L_val*1e6:.0f} um done ({iL+1}/{len(L_sweep_stage4)})")

plt.figure()
plt.plot(L_sweep_stage4*1e6, trans_vs_L*100, 'o-')
plt.xlabel('Crystal Length (µm)')
plt.ylabel('Transmitted Power (%)')
plt.title(f'Radially-integrated transmission vs propagation length\n(polarizer fixed at {np.rad2deg(theta_fixed):.1f}°)')