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
NP = 100                                            # Number of points
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
    R       = np.array([[c, -s], [s,  c]])   # R(theta): rotate polarizer frame -> lab frame
    R_inv   = np.array([[c,  s], [-s, c]])   # R(-theta): rotate lab frame -> polarizer frame
    P       = np.array([[1, 0], [0, 0]])     # pass polarizer-frame "x", block polarizer-frame "y"
    return R @ P @ R_inv                     # = [[c^2, cs], [cs, s^2]], passes axis (cos theta, sin theta)

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

#%% Stage 1 — Radial beam profile, no diffraction coupling                                          -> ~1 min run time

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

#%% Stage 2 — Hankel/QDHT diffraction only (no nonlinearity yet), via PyHank                        -> instant run time

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
    
#%% Stage 3 — Full split-step: nonlinearity + temporal dispersion + radial diffraction              -> instant run time

# --- Radial grid (update R_max/n_points per your round-trip validation) ---
w0_base = 100e-6
n_points_r = 512
R_max = 500e-6          # generously > expected beam growth over your 5 cm test range -- adjust after validating

def propagate_2d(Y0, tau, B, z_eval, dz, H, r_array, dispersion=True, optical_shock=True, diffraction=True, spm=True, xpm=True, fwm=True):
    Nt_ = Nt   # temporal grid size stays global (unchanged across this sweep)
    k0_local = 2*np.pi * n_base / Lambda_base
    Nr = len(r_array)
    
    T = np.exp(-t_array**2/(2*tau**2))[:, None]
    R = np.exp(-r_array**2/w0_base**2)[None, :]
    Y_field = (Y0*np.cos(B)) * T * R + 0j
    X_field = (Y0*np.sin(B)) * T * R + 0j

    def coupled_equations_2d(state, Nt_, r_array, spm=True, xpm=True, fwm=True):
        
        Nr_ = len(r_array)
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



        if optical_shock == False:
            dX_dz = -1j*a_base * (spm_term_X + xpm_term_X) * X - 1j*a_base*fwm_term_X
            dY_dz = -1j*a_base * (spm_term_Y + xpm_term_Y) * Y - 1j*a_base*fwm_term_Y
        else:
            NLX = (spm_term_X + xpm_term_X)*X + fwm_term_X
            NLY = (spm_term_Y + xpm_term_Y)*Y + fwm_term_Y
            shock_coeff = 1/wp_base
            dNLX_dt = np.gradient(NLX, dt, axis=0)
            dNLY_dt = np.gradient(NLY, dt, axis=0)
            dX_dz = -1j*a_base * (NLX - 1j*shock_coeff * dNLX_dt)
            dY_dz = -1j*a_base * (NLY - 1j*shock_coeff * dNLY_dt)

        return np.hstack([dX_dz.real.ravel(), dX_dz.imag.ravel(),
                           dY_dz.real.ravel(), dY_dz.imag.ravel()])
    
    for i in range(len(z_eval) - 1):
        state = np.hstack([X_field.real.ravel(), X_field.imag.ravel(),
                            Y_field.real.ravel(), Y_field.imag.ravel()])
        k1 = coupled_equations_2d(state, Nt_, r_array, spm, xpm, fwm)
        k2 = coupled_equations_2d(state + 0.5*dz*k1, Nt_, r_array, spm, xpm, fwm)
        k3 = coupled_equations_2d(state + 0.5*dz*k2, Nt_, r_array, spm, xpm, fwm)
        k4 = coupled_equations_2d(state + dz*k3, Nt_, r_array, spm, xpm, fwm)
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

def propagate_2d_batched(Y0, tau, B_values, z_eval, dz, H, r_array, dispersion=True, optical_shock=True, diffraction=True, spm=True, xpm=True, fwm=True):
    """Same physics as propagate_2d, but B_values is an array — all angles
    propagated simultaneously by folding B into the radial axis."""
    Nt_ = Nt
    Nr = len(r_array)
    NB_local = len(B_values)  # Use a local unique variable name
    k0_local = 2*np.pi * n_base / Lambda_base

    T = np.exp(-t_array**2/(2*tau**2))[:, None, None]         # (Nt,1,1)
    R = np.exp(-r_array**2/w0_base**2)[None, :, None]         # (1,Nr,1)
    cosB = np.cos(B_values)[None, None, :]                    # (1,1,NB_local)
    sinB = np.sin(B_values)[None, None, :]

    Y_field = (Y0*cosB) * T * R + 0j   # (Nt, Nr, NB_local)
    X_field = (Y0*sinB) * T * R + 0j

    # FIX: Pass NB_local explicitly into the helper function to avoid global scope confusion
    def coupled_equations_2d(state, NB_local=NB_local):
        n_tot = Nt_ * Nr * NB_local
        X = state[:n_tot].reshape(Nt_, Nr, NB_local) + 1j*state[n_tot:2*n_tot].reshape(Nt_, Nr, NB_local)
        Y = state[2*n_tot:3*n_tot].reshape(Nt_, Nr, NB_local) + 1j*state[3*n_tot:].reshape(Nt_, Nr, NB_local)

        X_abs2, Y_abs2 = np.abs(X)**2, np.abs(Y)**2
        spm_term_X = C * X_abs2 if spm else 0
        spm_term_Y = C * Y_abs2 if spm else 0
        xpm_term_X = 2*P * Y_abs2 if xpm else 0
        xpm_term_Y = 2*P * X_abs2 if xpm else 0
        fwm_term_X = P * Y**2 * np.conj(X) if fwm else 0
        fwm_term_Y = P * X**2 * np.conj(Y) if fwm else 0

        if not optical_shock:
            dX_dz = -1j*a_base * (spm_term_X + xpm_term_X) * X - 1j*a_base*fwm_term_X
            dY_dz = -1j*a_base * (spm_term_Y + xpm_term_Y) * Y - 1j*a_base*fwm_term_Y
        else:
            NLX = (spm_term_X + xpm_term_X)*X + fwm_term_X
            NLY = (spm_term_Y + xpm_term_Y)*Y + fwm_term_Y
            shock_coeff = 1/wp_base
            dNLX_dt = np.gradient(NLX, dt, axis=0)
            dNLY_dt = np.gradient(NLY, dt, axis=0)
            dX_dz = -1j*a_base * (NLX - 1j*shock_coeff * dNLX_dt)
            dY_dz = -1j*a_base * (NLY - 1j*shock_coeff * dNLY_dt)

        return np.hstack([dX_dz.real.ravel(), dX_dz.imag.ravel(),
                           dY_dz.real.ravel(), dY_dz.imag.ravel()])
    
    for i in range(len(z_eval) - 1):
        state = np.hstack([X_field.real.ravel(), X_field.imag.ravel(),
                            Y_field.real.ravel(), Y_field.imag.ravel()])
        k1 = coupled_equations_2d(state)
        k2 = coupled_equations_2d(state + 0.5*dz*k1)
        k3 = coupled_equations_2d(state + 0.5*dz*k2)
        k4 = coupled_equations_2d(state + dz*k3)
        sol = state + (dz/6)*(k1 + 2*k2 + 2*k3 + k4)

        n_tot = Nt_*Nr*NB_local
        X_field = sol[:n_tot].reshape(Nt_, Nr, NB_local) + 1j*sol[n_tot:2*n_tot].reshape(Nt_, Nr, NB_local)
        Y_field = sol[2*n_tot:3*n_tot].reshape(Nt_, Nr, NB_local) + 1j*sol[3*n_tot:].reshape(Nt_, Nr, NB_local)

        if np.any(np.isnan(X_field)) or np.any(np.isnan(Y_field)):
            print(f"Blew up at step {i}")
            return None, None

        if dispersion:
            Xw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(X_field, axes=0), axis=0), axes=0)
            Yw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Y_field, axes=0), axis=0), axes=0)
            Xw *= phase_base[:, None, None]
            Yw *= phase_base[:, None, None]
            X_field = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Xw, axes=0), axis=0), axes=0)
            Y_field = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Yw, axes=0), axis=0), axes=0)

        if diffraction:
            # 1. Transform directly along axis 1 (Nr). pyhank automatically broadcasts over axis 0 (Nt) and axis 2 (NB)
            Xkr = H.qdht(X_field, axis=1)
            Ykr = H.qdht(Y_field, axis=1)
            
            # 2. Reshape H.kr from (Nr,) to (1, Nr, 1) to match (Nt, Nr, NB) broadcasting
            diff_phase = np.exp(-1j * (H.kr**2)[None, :, None] * dz / (2 * k0_local))
            
            # 3. Multiply and inverse transform back
            Xkr *= diff_phase
            Ykr *= diff_phase
            X_field = H.iqdht(Xkr, axis=1)
            Y_field = H.iqdht(Ykr, axis=1)
        
        print(f"Propagate Iteration {i+1}")
            
    return X_field, Y_field

#%% Stage 4.0 — Setup: grid, angle convention, polarizer helpers                                    -> instant run time
#
# Angle convention (used everywhere in Stage 4):
#   - The crystal axes X, Y are the lab frame and never rotate. Only the input
#     polarization rotates: the field is projected onto the crystal axes as
#     Y ∝ cos(B), X ∝ sin(B), so B is measured from vertical (Y) toward X.
#   - The polarizer angle theta is measured the same way (from vertical, toward X).
#     Vertical in -> max transmission at theta = 0; orthogonal to the input at B + 90°.
#   - polarizer_zero_offset converts theta to the lab dial reading (dial = theta + offset).
#     Leave at 0 unless the polarizer mount's 0 mark isn't vertical.

polarizer_zero_offset = np.deg2rad(0.0)

# --- Radial grid ---
w0_base = 100e-6
R_max = 500e-6
n_points_r = 100
H = HankelTransform(order=0, max_radius=R_max, n_points=n_points_r)
r_array = H.r
Nr = n_points_r

z_eval = np.arange(0, L_base + dz_base/2, dz_base)   # include z = L so the full crystal length is propagated

def integrated_stokes(X, Y, r_arr=None):
    '''
    Stokes parameters S0, S1, S2 in the vertical-referenced frame (S1 > 0 means
    vertical), integrated over time (axis 0) and, if r_arr is given, over radius
    (axis 1) with r dr weighting. Any trailing axes (e.g. the B axis) are kept.
    '''
    S0 = np.abs(X)**2 + np.abs(Y)**2
    S1 = np.abs(Y)**2 - np.abs(X)**2
    S2 = 2*np.real(Y*np.conj(X))
    out = []
    for S in (S0, S1, S2):
        if r_arr is not None:
            w = r_arr.reshape((1, -1) + (1,)*(S.ndim - 2))
            S = np.trapezoid(S*w, r_arr, axis=1)
        out.append(np.sum(S, axis=0)*dt)
    return out

def transmission(S0, S1, S2, theta_dial):
    '''
    Fraction of the energy passed by a polarizer at dial angle theta_dial (rad).
    Identical to ∫∫ r|E_pass|² dr dt / ∫∫ r|E|² dr dt with E_pass = sinθ·X + cosθ·Y
    (field projection onto the polarizer axis), written via the integrated Stokes
    parameters so it is cheap to sweep. Broadcasts over S and theta_dial.
    '''
    th = theta_dial - polarizer_zero_offset
    return 0.5*(1 + (S1*np.cos(2*th) + S2*np.sin(2*th)) / (S0 + 1e-300))

def output_angle(S1, S2):
    '''Major axis of the polarization ellipse (rad from vertical), wrapped into [-45°, 135°).'''
    return np.mod(0.5*np.arctan2(S2, S1) + np.pi/4, np.pi) - np.pi/4

def to_dial(theta):
    '''Polarizer angle from vertical (rad) -> dial reading in [0, 180°) (rad).'''
    return np.mod(theta + polarizer_zero_offset, np.pi)

#%% Stage 4.1 — Single run at B_base: polarization across the beam                                  -> ~15 sec run time

print(f"B_base = {np.rad2deg(B_base):.2f} deg")
X_out, Y_out = propagate_2d(Y0_base, Tau_base, B_base, z_eval, dz_base,
                             H, r_array, dispersion=True, diffraction=True)

# --- ψ(t,r), χ(t,r), measured from vertical; blanked where the beam is negligible ---
S0_tr, S1_tr, S2_tr, S3_tr = stokes_parameters(Y_out, X_out)   # Ex = Y, Ey = X -> angles from vertical
beam_mask = S0_tr > 1e-3*np.max(S0_tr)
psi_tr = np.where(beam_mask, np.rad2deg(output_angle(S1_tr, S2_tr)), np.nan)
chi_tr = np.where(beam_mask, np.rad2deg(0.5*np.arcsin(np.clip(S3_tr/(S0_tr + 1e-300), -1, 1))), np.nan)

peak_t_idx = np.argmax(np.sum(S0_tr, axis=1))   # peak of the radially-summed pulse

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(r_array*1e6, psi_tr[peak_t_idx, :])
axes[0].set_xlabel('Radius (µm)')
axes[0].set_ylabel('ψ from vertical (deg)')
axes[0].set_title('Polarization orientation vs radius, at pulse peak')
axes[1].plot(r_array*1e6, chi_tr[peak_t_idx, :])
axes[1].set_xlabel('Radius (µm)')
axes[1].set_ylabel('χ (deg)')
axes[1].set_title('Ellipticity angle vs radius, at pulse peak')
plt.tight_layout()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].pcolormesh(r_array*1e6, t_array*1e15, psi_tr, shading='auto', cmap=inferno_white)
fig.colorbar(im0, ax=axes[0], label='ψ from vertical (deg)')
axes[0].set_xlabel('Radius (µm)')
axes[0].set_ylabel('Time (fs)')
axes[0].set_title('ψ(t,r)')
im1 = axes[1].pcolormesh(r_array*1e6, t_array*1e15, chi_tr, shading='auto', cmap=inferno_white)
fig.colorbar(im1, ax=axes[1], label='χ (deg)')
axes[1].set_xlabel('Radius (µm)')
axes[1].set_ylabel('Time (fs)')
axes[1].set_title('χ(t,r)')
plt.tight_layout()

# --- Polarizer transmission: per radius (time-integrated) and whole beam (r dr, t) ---
theta_dial = np.deg2rad(np.linspace(0, 180, 181))
crossed_dial = to_dial(B_base + np.pi/2)   # orthogonal to the input polarization

S0_r, S1_r, S2_r = integrated_stokes(X_out, Y_out)            # shape (Nr,)
trans_r_theta = transmission(S0_r[:, None], S1_r[:, None], S2_r[:, None], theta_dial[None, :])

S0_i, S1_i, S2_i = integrated_stokes(X_out, Y_out, r_array)   # whole beam, scalars
trans_theta = transmission(S0_i, S1_i, S2_i, theta_dial)

plt.figure()
im = plt.pcolormesh(np.rad2deg(theta_dial), r_array*1e6, trans_r_theta*100,
                     shading='auto', cmap='viridis')
plt.colorbar(im, label='Transmitted Power (%)')
plt.axvline(np.rad2deg(crossed_dial), color='w', ls='--', label='Orthogonal to input')
plt.xlabel('Polarizer dial angle (deg)')
plt.ylabel('Radius (µm)')
plt.title('Polarizer transmission vs angle & radius')
plt.legend()

plt.figure()
plt.plot(np.rad2deg(theta_dial), trans_theta*100)
plt.axvline(np.rad2deg(crossed_dial), color='k', ls='--', label='Orthogonal to input')
plt.xlabel('Polarizer dial angle (deg)')
plt.ylabel('Transmitted Power (%)')
plt.title('Radially-integrated polarizer transmission vs angle')
plt.legend()

print(f"Output angle (whole beam): {np.rad2deg(output_angle(S1_i, S2_i)):.2f} deg from vertical")
print(f"Transmission orthogonal to input (cross-pol efficiency): "
      f"{transmission(S0_i, S1_i, S2_i, crossed_dial)*100:.4f}%")
print(f"Minimum transmission (best extinction): {np.min(trans_theta)*100:.4f}% "
      f"at {np.rad2deg(theta_dial[np.argmin(trans_theta)]):.1f} deg")

#%% Stage 4.2 — Output angle vs input angle (all B in one batched run)                              -> ~40 min run time

NB = 100                                  # memory scales with Nt*Nr*NB; reduce if needed
B_sweep = np.linspace(0, np.pi/2, NB)

X_all, Y_all = propagate_2d_batched(Y0_base, Tau_base, B_sweep, z_eval, dz_base,
                                     H, r_array, dispersion=True, diffraction=True)

# Whole beam (r dr, t integrated) — this is what the polarizer/power meter sees
S0_B, S1_B, S2_B = integrated_stokes(X_all, Y_all, r_array)   # shape (NB,)
out_angle_int = output_angle(S1_B, S2_B)

# On-axis at the pulse peak — closest to the 1D (no beam shape) picture
pk_idx = np.argmax(np.abs(X_all[:, 0, :])**2 + np.abs(Y_all[:, 0, :])**2, axis=0)
b_indices = np.arange(NB)
_, S1_pk, S2_pk, _ = stokes_parameters(Y_all[pk_idx, 0, b_indices], X_all[pk_idx, 0, b_indices])
out_angle_peak = output_angle(S1_pk, S2_pk)

plt.figure()
plt.plot(np.rad2deg(B_sweep), np.rad2deg(out_angle_int), label='Whole beam (∫∫ r|E|² dr dt)')
plt.plot(np.rad2deg(B_sweep), np.rad2deg(out_angle_peak), label='On-axis, pulse peak')
plt.plot(np.rad2deg(B_sweep), np.rad2deg(B_sweep), 'k:', label='Output = input')
plt.xlabel('Input angle B (deg from vertical)')
plt.ylabel('Output angle ψ (deg from vertical)')
plt.title('Output polarization angle vs input angle')
plt.legend()

#%% Stage 4.3 — Lab bench test: rotate input polarization and polarizer independently               -> instant run time
#
# Reuses X_all, Y_all from 4.2. Crystal fixed in the lab frame; the input
# polarization angle B and the polarizer dial angle are swept independently,
# and transmission is the whole-beam ∫∫ r|E|² dr dt fraction.

theta_dial_abs = np.deg2rad(np.linspace(0, 180, 181))

# transmission_map[i, j] = transmission at input angle B_sweep[i], polarizer theta_dial_abs[j]
transmission_map = transmission(S0_B[:, None], S1_B[:, None], S2_B[:, None], theta_dial_abs[None, :])

theta_min_measured = theta_dial_abs[np.argmin(transmission_map, axis=1)]   # (NB,)
theta_orth_input = to_dial(B_sweep + np.pi/2)                               # orthogonal to input

plt.figure(figsize=(8, 6))
im = plt.pcolormesh(np.rad2deg(theta_dial_abs), np.rad2deg(B_sweep), transmission_map*100,
                     shading='auto', cmap='viridis')
plt.colorbar(im, label='Transmitted Power (%)')
# markers instead of lines so the 0°/180° wrap doesn't draw a line across the map
plt.plot(np.rad2deg(theta_orth_input), np.rad2deg(B_sweep), 'w.', ms=4, label='Orthogonal to input (B + 90°)')
plt.plot(np.rad2deg(theta_min_measured), np.rad2deg(B_sweep), 'r.', ms=3, label='Measured minimum')
plt.xlabel('Polarizer dial angle (deg)')
plt.ylabel('Input angle B (deg from vertical)')
plt.title('Transmission vs independently rotated input & polarizer')
plt.legend(loc='upper right', fontsize=8)

# Transmission % and efficiency vs input angle
trans_orth = transmission(S0_B, S1_B, S2_B, theta_orth_input)                 # cross-pol efficiency
trans_min = 0.5*(1 - np.sqrt(S1_B**2 + S2_B**2)/(S0_B + 1e-300))              # best extinction
plt.figure()
plt.semilogy(np.rad2deg(B_sweep), trans_orth*100, label='Polarizer orthogonal to input')
plt.semilogy(np.rad2deg(B_sweep), trans_min*100, label='Polarizer at measured minimum')
plt.xlabel('Input angle B (deg from vertical)')
plt.ylabel('Transmitted Power (%)')
plt.title('Polarizer transmission vs input angle')
plt.legend()

# How far the measured extinction sits from orthogonal-to-input (= nonlinear rotation)
extinction_shift_deg = np.rad2deg(np.mod(theta_min_measured - theta_orth_input + np.pi/2, np.pi) - np.pi/2)
plt.figure()
plt.plot(np.rad2deg(B_sweep), extinction_shift_deg)
plt.axhline(0, color='k', lw=0.5)
plt.xlabel('Input angle B (deg from vertical)')
plt.ylabel('Measured minimum − (B + 90°) (deg)')
plt.title('Extinction angle shift from orthogonal-to-input')

#%% Stage 4.4 — Low-intensity sanity check (no nonlinear rotation expected)                         -> ~1 min run time
#
# At very low intensity the crystal shouldn't rotate anything: vertical in should
# peak at a vertical polarizer, and the minimum should sit exactly at B + 90°
# (e.g. B = 15° -> 105° from vertical, plus polarizer_zero_offset on the dial).

Y0_low = 1e6                              # V/m — nonlinear phase ~1e-6 of the Y0_base run
B_check = np.deg2rad([0, 2, 15, 45])
X_low, Y_low = propagate_2d_batched(Y0_low, Tau_base, B_check, z_eval, dz_base,
                                     H, r_array, dispersion=True, diffraction=True)
S0_low, S1_low, S2_low = integrated_stokes(X_low, Y_low, r_array)
trans_low = transmission(S0_low[:, None], S1_low[:, None], S2_low[:, None], theta_dial_abs[None, :])

for ib, B_val in enumerate(B_check):
    print(f"B = {np.rad2deg(B_val):5.1f} deg: "
          f"max at {np.rad2deg(theta_dial_abs[np.argmax(trans_low[ib])]):6.1f} deg "
          f"(expect {np.rad2deg(to_dial(B_val)):6.1f}), "
          f"min at {np.rad2deg(theta_dial_abs[np.argmin(trans_low[ib])]):6.1f} deg "
          f"(expect {np.rad2deg(to_dial(B_val + np.pi/2)):6.1f}), "
          f"min transmission {np.min(trans_low[ib])*100:.2e}%")

#%% Stage 4.5 — Cross-polarization efficiency vs propagation length                                 -> ~1 min run time

L_sweep_stage4 = np.linspace(50e-6, 500e-6, 10)   # keep coarse-ish, this is expensive
theta_fixed = to_dial(B_base + np.pi/2)           # orthogonal to input -> measures generated orthogonal light

trans_vs_L = np.zeros(len(L_sweep_stage4))
for iL, L_val in enumerate(L_sweep_stage4):
    # Keep dz = dz_base for every length: propagate_2d always applies phase_base,
    # which is built for dz_base. L values here are integer multiples of dz_base.
    z_eval_L = np.arange(0, L_val + dz_base/2, dz_base)
    X_L, Y_L = propagate_2d(Y0_base, Tau_base, B_base, z_eval_L, dz_base,
                             H, r_array, dispersion=True, diffraction=True)
    if X_L is None:
        trans_vs_L[iL] = np.nan
        continue
    trans_vs_L[iL] = transmission(*integrated_stokes(X_L, Y_L, r_array), theta_fixed)
    print(f"L = {L_val*1e6:.0f} um done ({iL+1}/{len(L_sweep_stage4)})")

plt.figure()
plt.plot(L_sweep_stage4*1e6, trans_vs_L*100, 'o-')
plt.xlabel('Crystal Length (µm)')
plt.ylabel('Transmitted Power (%)')
plt.title(f'Radially-integrated transmission vs propagation length\n'
          f'(polarizer orthogonal to input, dial {np.rad2deg(theta_fixed):.1f}°)')
plt.show()
