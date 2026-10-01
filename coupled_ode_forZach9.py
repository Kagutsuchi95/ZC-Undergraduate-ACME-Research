# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:27:21 2025

@author: TJ Hammond
"""

import numpy as np
import matplotlib.pyplot as plt
import math

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
tau = FWHM/(2*(np.log(2))**0.5)   # ~50.9 fs
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
Nt = 1001
t_array = np.linspace(-200e-15, 200e-15, Nt)
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

def propagate(Y0_in, tau_in, B_in, z_eval_in=None, dz_in=None, phase_in=None, a_in=None, dispersion=True, optical_shock=True, spm=True, xpm=True, fwm=True):
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
        return Y0_in * np.cos(B_in) * np.exp(-t**2 / (2 * tau_in**2)) * np.exp(1j * 0*wp * t) 
    
    def X_t(t):
        return Y0_in * np.sin(B_in) * np.exp(-t**2 / (2 * tau_in**2)) * np.exp(1j * 0*wp * t) * np.exp(1j * phi)
        
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
            shock_coeff = 1/wp
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
    """Linear polarizer Jones matrix, axis at angle theta (rad) from lab X-axis."""
    c, s = np.cos(theta), np.sin(theta)
    R       = np.array([[c, -s], [s,  c]])   # rotate lab frame -> polarizer frame
    R_inv   = np.array([[c,  s], [-s, c]])   # rotate polarizer frame -> lab frame (R(-theta))
    P       = np.array([[1, 0], [0, 0]])     # pass polarizer-frame "x", block polarizer-frame "y"
    return R_inv @ P @ R

def apply_polarizer(X_exit, Y_exit, theta):
    """
    Applies a linear polarizer at angle theta to the exit fields.
    X_exit, Y_exit: 1D complex arrays over time (e.g. Xt_out[:, -1], Yt_out[:, -1])
    Returns X_pol, Y_pol — fields AFTER the polarizer, back in the lab (X, Y) frame.
    """
    M = polarizer_jones(theta)
    E_in = np.vstack([X_exit, Y_exit])      # shape (2, Nt)
    E_out = M @ E_in
    return E_out[0], E_out[1]

def find_eliminating_angle(X_exit, Y_exit):
    """
    Returns the polarizer angle (rad) that BLOCKS the dominant ellipse axis,
    i.e. passes only the orthogonal component — or pass theta_psi directly
    to instead PASS the dominant axis (eliminate the orthogonal one).
    Uses psi at pulse center, matching your existing convention.
    """
    psi, _ = polarization_orientation_and_ellipticity(Y_exit, X_exit)
    theta_psi = np.deg2rad(psi[Nt//2])
    return theta_psi

# %% Intensity

Xt_out, Yt_out = propagate(Y0, tau, 5, None, None, None, None, False, False)
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
# Corrected way (gives ~1)
I = np.exp(-t_array**2 / (2*tau**2)) # Test Gaussian
fwhm_out = fwhm_from_intensity(t_array, I)
'''

# Old way (gives ~1/sqrt(2))
intensity = np.abs(Xt_out[:, -1])**2
fwhm_out = fwhm_from_intensity(t_array, intensity)


print(fwhm_out/FWHM)

#%% Pulse Duration + Input Angle

output_fwhms = []

# --- Base Parameters ---
Y0_base = 6e9
FWHM_base = 100e-15
Tau_base = FWHM_base/(2*(np.log(2))**0.5)         # Width of gaussian (s)
L_base = 500e-6
Lambda_base = 785e-9
Lambda_base_micron = Lambda_base*1e6
wp_base = 2*np.pi*cc/Lambda_base                    # Pump wavelength (rad/s)
B_base = np.deg2rad(15)
eps0 = 8.854e-12                                    # Permittivity of f.s.

dz_base = L_base/100
z_eval_base = np.arange(0, L_base, dz_base)
n_base = (2.956362 + 0.02195770/(Lambda_base_micron**2 - 0.01428322) - 0.01062387*Lambda_base_micron**2 - 0.0000204968*Lambda_base_micron**4)**0.5
beta2_base = 84.868e-27   # From refractiveindex.info
phase_base = np.exp(1j * 0.5 * beta2_base * (w - wp_base*0)**2 * dz_base)
a_base = 2*np.pi/Lambda_base * 3/8 * 1/n_base

# --- Changing Parameters ---
NP = 20                                            # Number of points
Y0_sweep = np.linspace(1e9, 10e9, NP)                # Field amplitude (V/m)
L_sweep = np.linspace(50e-6, 2000e-6, NP)           # Crystal length (m)
Lambda_sweep = np.linspace(600e-9, 1200e-9, NP)     # Wavelength (m)
B_sweep = np.linspace(0, np.pi/2, NP)               # Input angle (rad)

I_sweep = 0.5 * eps0 * cc * Y0_sweep**2    # Convert to intensity

I_base = 10e16
Y0_base = (I_base/(0.5*eps0*cc))**0.5
#%% 
def sweep_2d_params(param_val, param_name):

    result_map = np.full((len(param_val), len(B_sweep)), np.nan)
    
    if param_name == 'Wavelength':
        n_arr = (2.956362 + 0.02195770/((param_val*1e6)**2 - 0.01428322) - 0.01062387*(param_val*1e6)**2 - 0.0000204968*(param_val*1e6)**4)**0.5
        beta2_arr = np.gradient(np.gradient(n_arr, param_val), param_val) * param_val**3 / (2*np.pi*cc**2)
    
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
            
            fwhm_out = fwhm_from_intensity(t_array, intensity)
            result_map[i, j] = fwhm_out
            print(f"  [{param_name}] i={i+1}/{len(param_val)}, j={j+1}/{len(B_sweep)} — FWHM out = {fwhm_out*1e15:.1f} fs")

    return result_map

Y0_output_fwhm = np.array(sweep_2d_params(Y0_sweep, 'Amplitude'))

L_output_fwhm = np.array(sweep_2d_params(L_sweep, 'Length'))

Lambda_output_fwhm = np.array(sweep_2d_params(Lambda_sweep, 'Wavelength'))

#%% Plot

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

#%% Isolation test with and without dispersion

test_cases_disp = [
    ('XPM only, no disp',       dict(spm=False, xpm=True,  fwm=False), False),
    ('XPM only, with disp',     dict(spm=False, xpm=True,  fwm=False), True),
    ('FWM only, no disp',       dict(spm=False, xpm=False, fwm=True),  False),
    ('FWM only, with disp',     dict(spm=False, xpm=False, fwm=True),  True),
    ('XPM+FWM, no disp',        dict(spm=False, xpm=True,  fwm=True),  False),
    ('XPM+FWM, with disp',      dict(spm=False, xpm=True,  fwm=True),  True),
    ('All three, no disp',      dict(spm=True,  xpm=True,  fwm=True),  False),
    ('All three, with disp',    dict(spm=True,  xpm=True,  fwm=True),  True),
]

print(f"\n{'Label':<28} {'FWHM (fs)':>10} {'ratio':>8}   (1/√3 = {1/np.sqrt(3):.4f})")
print("-" * 55)
for label, flags, disp in test_cases_disp:
    Xt_out, Yt_out = propagate(Y0_base, Tau_base, B_base, z_eval_base, dz_base,
                                phase_base, a_base,
                                dispersion=disp, optical_shock=False, **flags)
    if Xt_out is None:
        print(f"{label:<28} {'blew up':>10}")
        continue
    intensity = np.abs(Xt_out[:, -1])**2
    fwhm_out = fwhm_from_intensity(t_array, intensity)
    ratio = fwhm_out / FWHM_base
    print(f"{label:<28} {fwhm_out*1e15:>10.2f} {ratio:>8.4f}")
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
tau_base = FWHM_base/(2*(np.log(2))**0.5)         # Width of gaussian (fs)
L_base = 500e-6                                     # Crystal length (m)
Lambda_base = 800e-9                                # Wavelength (m)
wp_base = 2*np.pi*cc/Lambda_base                    # Pump frequency (rad/s)
B_base = np.deg2rad(15)                             # Input angle (rad)
eps0 = 8.854e-12                                    # Permittivity of f.s.

Y0_base = (4e16/(0.5*cc*eps0))**0.5 
L_base = 500e-6
Lambda_base = 1000
B_base = np.deg2rad(5)
           
z_span_base = (0,L_base)                            # Crystal thickness
dz_base = L_base/100                                # Step size in crystal
z_eval_base = np.linspace(*z_span_base, 101)        # Evaluation range

beta2_base = 84.868e-30 / 1e-3                      # From refractiveindex.info
phase_base = np.exp(-1j * 0.5 * beta2_base * (w - wp_base*0)**2 * dz_base)

n_base = (2.956362+0.02195770/((Lambda_base*1e6)**2-0.01428322)-0.01062387*(Lambda_base*1e6)**2-0.0000204968*(Lambda_base*1e6)**4)**0.5
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
fwhm_values = np.linspace(50e-15, 400e-15, 100)
output_fwhms = []

iteration = 0

for fwhm_scan in fwhm_values:
    tau_scan = fwhm_scan/(2*(np.log(2))**0.5)
    Xt_out, Yt_out = propagate(Y0_base, tau_scan, B_base, z_eval_base, dz_base, phase_base, a_base, dispersion=False, optical_shock=False)

    intensity = np.abs(Xt_out[:, -1])**2
    if np.max(intensity) == 0 or np.any(np.isnan(intensity)):
        output_fwhms.append(np.nan)
        continue

    fwhm_out = fwhm_from_intensity(t_array, intensity)
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
Y0_values = np.linspace(1e9, 8e9, 100)
output_fwhms = []

iteration = 0

for Y0_scan in Y0_values:
    Xt_out, Yt_out = propagate(Y0_scan, tau_base, B_base, z_eval_base, dz_base, phase_base, a_base, dispersion=False, optical_shock=False)

    intensity = np.abs(Xt_out[:, -1])**2
    if np.max(intensity) == 0 or np.any(np.isnan(intensity)):
        output_fwhms.append(np.nan)
        continue

    fwhm_out = fwhm_from_intensity(t_array, intensity)
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
L_values = np.linspace(50e-6, 2000e-6, 100)
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

    fwhm_out = fwhm_from_intensity(t_array, intensity)
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

Lambda_values = np.linspace(600e-9, 1200e-9, 100)
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

    fwhm_out = fwhm_from_intensity(t_array, intensity)
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

B_values = np.linspace(0, np.pi/2, 100)
output_fwhms = []

iteration = 0

for B_scan in B_values:
    Xt_out, Yt_out = propagate(Y0_base, tau_base, B_scan, z_eval_base, dz_base, phase_base, a_base, dispersion=False, optical_shock=False)

    intensity = np.abs(Xt_out[:, -1])**2
    if np.max(intensity) == 0 or np.any(np.isnan(intensity)):
        output_fwhms.append(np.nan)
        continue

    fwhm_out = fwhm_from_intensity(t_array, intensity)
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

#%% Sanity check polarizer matrix

test_X = np.array([1.0+0j])
test_Y = np.array([0.0+0j])

for theta_deg in [0, 45, 90]:
    Xp, Yp = apply_polarizer(test_X, test_Y, np.deg2rad(theta_deg))
    print(f"theta={theta_deg:3d}°  ->  X_out={Xp[0]:.3f}  Y_out={Yp[0]:.3f}  |E_out|={np.hypot(abs(Xp[0]), abs(Yp[0])):.3f}")