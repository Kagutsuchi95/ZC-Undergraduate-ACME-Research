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

#%% Stage 5.0 — Pulse compression: transmission AND output duration over (B, dial)                    -> ~10 sec run time
#
# Reuses X_all, Y_all from 4.2 (run 4.2 first). The polarizer is just post-processing,
# so every dial angle comes for free from the one batched propagation.
#
# Idea: the peak of the pulse rotates more than the wings, so a polarizer set to pass the
# rotated light lets the peak through and blocks the wings -> shorter pulse out. The catch is
# that the shortest pulses happen where hardly anything gets through, so we map transmission
# and duration together and look for the best compression we can get at a given efficiency.
#
# Duration = FWHM of the power a fast detector behind the polarizer would see over the whole
# beam: P(t) = ∫ r|E_pass(r,t)|² dr, with E_pass = sinθ·X + cosθ·Y (same projection as transmission()).

trans_floor = 0.10          # optimum pick only: shortest pulse with at least this fraction of the energy through
plot_floor = 1e-6           # map shows everything down to here; below this it's just numerical leakage at exact extinction
lobe_floor = 0.60           # main lobe must hold this much of the energy (a clean Gaussian has 76% inside its FWHM)

def fwhm_main_lobe(t, I):
    '''
    Vectorized FWHM of the lobe holding the peak, for I with time on axis 0 (any trailing axes).
    Walks out from the peak to the first half-max crossing on each side, so side lobes don't
    stretch the width like fwhm_from_intensity's outermost crossings do. Also returns the
    fraction of energy inside that lobe, so multi-lobe/broken-up pulses can be masked.
    (is_single_lobe uses 0.9, but a perfect Gaussian only has 76% inside its FWHM, so it
    would reject every clean pulse, hence lobe_floor instead.)
    '''
    Nt_ = I.shape[0]
    idx = np.arange(Nt_).reshape((-1,) + (1,)*(I.ndim - 1))
    peak = np.argmax(I, axis=0)
    half = np.max(I, axis=0)/2
    below = I <= half
    left = np.max(np.where(below & (idx < peak), idx, -1), axis=0)        # last sample below half-max before the peak
    right = np.min(np.where(below & (idx > peak), idx, Nt_), axis=0)      # first sample below half-max after the peak
    ok = (left >= 0) & (right < Nt_)                                      # lobe has to fit inside the time window
    l0, r0 = np.clip(left, 0, Nt_ - 2), np.clip(right, 1, Nt_ - 1)

    def take(i):
        return np.take_along_axis(I, i[None, ...], axis=0)[0]

    # linear interpolation between the samples straddling half-max (sub-sample precision)
    t_left = t[l0] + (half - take(l0))/(take(l0 + 1) - take(l0) + 1e-300)*dt
    t_right = t[r0 - 1] + (half - take(r0 - 1))/(take(r0) - take(r0 - 1) - 1e-300)*dt
    width = np.where(ok, t_right - t_left, np.nan)

    in_lobe = (idx > left) & (idx < right)
    lobe_frac = np.sum(I*in_lobe, axis=0)/(np.sum(I, axis=0) + 1e-300)
    return width, lobe_frac

# Radially integrated pieces of |E_pass|², so the dial sweep is just a weighted sum:
#   |E_pass|² = sin²θ|X|² + cos²θ|Y|² + 2 sinθ cosθ Re(X Y*)
w_r = r_array[None, :, None]
Pxx = np.trapezoid(np.abs(X_all)**2*w_r, r_array, axis=1)                 # (Nt, NB)
Pyy = np.trapezoid(np.abs(Y_all)**2*w_r, r_array, axis=1)
Pxy = np.trapezoid(np.real(X_all*np.conj(Y_all))*w_r, r_array, axis=1)

th = (theta_dial_abs - polarizer_zero_offset)[None, None, :]
P_pass = (np.sin(th)**2*Pxx[:, :, None] + np.cos(th)**2*Pyy[:, :, None]
          + np.sin(2*th)*Pxy[:, :, None])                                 # (Nt, NB, Ndial)

fwhm_map, lobe_map = fwhm_main_lobe(t_array, P_pass)                      # (NB, Ndial)

# Reference: input pulse through the same whole-beam measurement (r and t are separable at
# the input, so this is just FWHM_base, but measure it the same way to keep things consistent)
P_in = np.exp(-t_array**2/Tau_base**2)
fwhm_in, _ = fwhm_main_lobe(t_array, P_in)
ratio_map = fwhm_map/fwhm_in                                              # < 1 means compressed

# Don't hide low transmission: compression matters even when only a few % gets through.
# Only blank broken-up pulses and the exact-extinction points where there's basically no light.
clean = (lobe_map >= lobe_floor) & np.isfinite(ratio_map)
ratio_shown = np.where(clean & (transmission_map >= plot_floor), ratio_map, np.nan)
good = clean & (transmission_map >= trans_floor)                          # used for the optimum pick
ratio_masked = np.where(good, ratio_map, np.nan)

# --- Pick the optimum: shortest clean pulse with at least trans_floor of the energy ---
i_opt, j_opt = np.unravel_index(np.nanargmin(ratio_masked), ratio_masked.shape)
B_opt = B_sweep[i_opt]
dial_opt = theta_dial_abs[j_opt]
print(f"Input FWHM: {fwhm_in*1e15:.1f} fs")
print(f"Optimum (transmission >= {trans_floor*100:.0f}%): B = {np.rad2deg(B_opt):.1f} deg, "
      f"dial = {np.rad2deg(dial_opt):.0f} deg -> FWHM {fwhm_map[i_opt, j_opt]*1e15:.1f} fs "
      f"(ratio {ratio_map[i_opt, j_opt]:.3f}), transmission {transmission_map[i_opt, j_opt]*100:.1f}%")

# --- Maps: transmission and duration side by side, same axes as the 4.3 colormap ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im0 = axes[0].pcolormesh(np.rad2deg(theta_dial_abs), np.rad2deg(B_sweep), transmission_map*100,
                          shading='auto', cmap='viridis')
fig.colorbar(im0, ax=axes[0], label='Transmitted Power (%)')
axes[0].set_title('Transmission')
axes[1].set_facecolor('0.6')                                              # grey = blanked, so it can't be confused with data
im1 = axes[1].pcolormesh(np.rad2deg(theta_dial_abs), np.rad2deg(B_sweep), ratio_shown,
                          shading='auto', cmap=inverse_inferno_white)
fig.colorbar(im1, ax=axes[1], label='FWHM out / FWHM in')
cs = axes[1].contour(np.rad2deg(theta_dial_abs), np.rad2deg(B_sweep), transmission_map*100,
                     levels=[1, 10, 20, 30, 50, 80], colors='k', linewidths=0.7)
axes[1].clabel(cs, fmt='%d%%', fontsize=8)
axes[1].set_title('Output duration (grey: broken pulse or no light)')
for ax in axes:
    ax.plot(np.rad2deg(dial_opt), np.rad2deg(B_opt), 'c*', ms=14, mec='k', label='Optimum')
    ax.set_xlabel('Polarizer dial angle (deg)')
    ax.set_ylabel('Input angle B (deg from vertical)')
    ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()

# --- Trade-off: every (B, dial) as one point, plus the best compression at each efficiency ---
shown = np.isfinite(ratio_shown)
T_flat = transmission_map[shown]
R_flat = ratio_map[shown]
order = np.argsort(-T_flat)                       # walk from high to low transmission
front_T, front_R = [], []
best = np.inf
for k in order:
    if R_flat[k] < best:                          # only keep points that beat everything more efficient
        best = R_flat[k]
        front_T.append(T_flat[k])
        front_R.append(R_flat[k])

plt.figure()
plt.scatter(T_flat*100, R_flat, s=3, c='0.7', label='All (B, dial)')
plt.plot(np.array(front_T)*100, front_R, 'r.-', label='Best compression at that efficiency')
plt.plot(transmission_map[i_opt, j_opt]*100, ratio_map[i_opt, j_opt], 'c*', ms=14, mec='k', label='Optimum')
plt.axhline(1, color='k', lw=0.5)
plt.xscale('log')                                 # low-transmission points matter now, so spread them out
plt.xlabel('Transmitted Power (%)')
plt.ylabel('FWHM out / FWHM in')
plt.title('Compression vs efficiency')
plt.legend()

# Table for the meeting: best compression available above a few efficiency floors
for floor in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
    r_floor = np.where(clean & (transmission_map >= floor), ratio_map, np.nan)
    if np.all(np.isnan(r_floor)):
        print(f">= {floor*100:3.0f}% transmitted: nothing clean")
        continue
    i, j = np.unravel_index(np.nanargmin(r_floor), r_floor.shape)
    print(f">= {floor*100:3.0f}% transmitted: ratio {ratio_map[i, j]:.3f} "
          f"({fwhm_map[i, j]*1e15:.1f} fs) at B = {np.rad2deg(B_sweep[i]):.1f} deg, "
          f"dial = {np.rad2deg(theta_dial_abs[j]):.0f} deg, T = {transmission_map[i, j]*100:.1f}%")

# Pulse shapes at the optimum, so we can see it's really shorter and not just a spike on a pedestal
plt.figure()
plt.plot(t_array*1e15, P_in/np.max(P_in), 'k:', label=f'Input ({fwhm_in*1e15:.0f} fs)')
P_nopol = Pxx[:, i_opt] + Pyy[:, i_opt]
plt.plot(t_array*1e15, P_nopol/np.max(P_nopol), label='Output, no polarizer')
plt.plot(t_array*1e15, P_pass[:, i_opt, j_opt]/np.max(P_pass[:, i_opt, j_opt]),
         label=f'After polarizer ({fwhm_map[i_opt, j_opt]*1e15:.0f} fs)')
plt.xlabel('Time (fs)')
plt.ylabel('Normalized power (∫ r|E|² dr)')
plt.title(f'Pulse at optimum: B = {np.rad2deg(B_opt):.1f}°, dial = {np.rad2deg(dial_opt):.0f}°')
plt.legend()

#%% Stage 5.1 — Redo 4.5 at the optimal dial angle: efficiency AND duration vs crystal length          -> ~2 min run time
#
# Same length sweep as 4.5, but now at the (B, dial) picked in 5.0 instead of B_base with the
# polarizer orthogonal to the input. The old 4.5 setting is run alongside in the same batch
# so the two curves can be compared directly. Dial is held FIXED at its 5.0 value for every L,
# like in the lab where we wouldn't re-tune the polarizer for each crystal.

B_pair = np.array([B_base, B_opt])
dial_pair = np.array([to_dial(B_base + np.pi/2), dial_opt])
labels_pair = [f'Old 4.5: B = {np.rad2deg(B_base):.0f}°, dial {np.rad2deg(dial_pair[0]):.0f}° (orthogonal)',
               f'Optimal: B = {np.rad2deg(B_opt):.1f}°, dial {np.rad2deg(dial_opt):.0f}°']

trans_vs_L_opt = np.full((len(L_sweep_stage4), 2), np.nan)
fwhm_vs_L_opt = np.full((len(L_sweep_stage4), 2), np.nan)
for iL, L_val in enumerate(L_sweep_stage4):
    z_eval_L = np.arange(0, L_val + dz_base/2, dz_base)    # dz stays dz_base, see 4.5
    X_L, Y_L = propagate_2d_batched(Y0_base, Tau_base, B_pair, z_eval_L, dz_base,
                                     H, r_array, dispersion=True, diffraction=True)
    if X_L is None:
        continue
    S0_L, S1_L, S2_L = integrated_stokes(X_L, Y_L, r_array)               # (2,)
    trans_vs_L_opt[iL] = transmission(S0_L, S1_L, S2_L, dial_pair)
    th_L = (dial_pair - polarizer_zero_offset)[None, None, :]
    E_pass_L = np.sin(th_L)*X_L + np.cos(th_L)*Y_L                       # (Nt, Nr, 2)
    P_L = np.trapezoid(np.abs(E_pass_L)**2*r_array[None, :, None], r_array, axis=1)
    fwhm_vs_L_opt[iL] = fwhm_main_lobe(t_array, P_L)[0]
    print(f"L = {L_val*1e6:.0f} um done ({iL+1}/{len(L_sweep_stage4)})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for k in range(2):
    axes[0].plot(L_sweep_stage4*1e6, trans_vs_L_opt[:, k]*100, 'o-', label=labels_pair[k])
    axes[1].plot(L_sweep_stage4*1e6, fwhm_vs_L_opt[:, k]*1e15, 'o-', label=labels_pair[k])
axes[1].axhline(fwhm_in*1e15, color='k', ls=':', label='Input FWHM')
axes[0].set_xlabel('Crystal Length (µm)')
axes[0].set_ylabel('Transmitted Power (%)')
axes[0].set_title('Radially-integrated transmission vs propagation length')
axes[1].set_xlabel('Crystal Length (µm)')
axes[1].set_ylabel('Transmitted pulse FWHM (fs)')
axes[1].set_title('Transmitted pulse duration vs propagation length')
for ax in axes:
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

#%% Stage 7.0 — Setup: h-cut crystal, wavelength-dependent MgO, self-contained 2D propagator       -> instant run time
#
# Hammond's ask (meeting 2026-11-26): put the h-cut (holographic, (011)/(110) cut) MgO crystal
# into the full 2D (time + radius, Hankel) simulation with a pulsed beam, and see if it gives
# better pulse compression. Also scale the wavelength to see whether anomalous dispersion
# (MgO's zero-dispersion point is around 1.55-1.6 um) makes the pulse compress itself.
# Plan, in order:
#   7.1  scalar case (no polarization, just one field): find the I0, w0, tau and crystal length
#        (1-3 mm) where the pulse compresses AND the wings of I(t) stay small
#   7.2  wavelength sweep at that setting: normal vs anomalous dispersion
#   7.3  put polarization back in (z-cut vs h-cut + a polarizer) to clean the wings further
#   7.4  one summary figure for the meeting
#
# Why a new propagator instead of reusing propagate_2d_batched: that one reads a_base, n_base,
# phase_base, wp_base, w0_base and H straight from globals, which are all built for 785 nm and
# w0 = 100 um. Sweeping wavelength and w0 with it would silently use stale values (Spyder keeps
# every global around). So everything Stage 7 needs gets passed in, and every Stage 7 name ends
# in 7 so nothing here overwrites what Stages 1-5 use.
#
# H-cut physics (from the 2026-11-12 write-up, cross-checked with GPT): with the crystal's [100]
# axis vertical (our Y) and [01-1] horizontal (our X), the coupled equations are exactly Hammond's
# Eqs 2a/2b, except the SPM coefficient for the X field becomes (C + 3P)/2 instead of C.
# XPM (2P) and FWM (P) don't change. B is still measured from vertical = from [100], and it maps to
# Jullien/Canova's angle as B = 90 deg - beta_Jullien.

C7 = 3.2e-22                      # chi_xxxx for MgO (same as Parameters). Redefined here on purpose: if the
P7 = C7/1.85                      # BaF2 Jullien file was run in the same console, C and P could be stale
C7_hcut = (C7 + 3*P7)/2           # SPM coefficient along [01-1] in the h-cut (= 1.31*C for MgO)

def n_mgo(lam):
    '''MgO refractive index from the same Sellmeier as Parameters, lam in metres.'''
    lu = lam*1e6
    return np.sqrt(2.956362 + 0.02195770/(lu**2 - 0.01428322) - 0.01062387*lu**2 - 0.0000204968*lu**4)

def mgo_dispersion(lam):
    '''
    beta2 (s^2/m) and beta3 (s^3/m) at wavelength lam, straight from the Sellmeier using
    finite differences of k(w) = n(w) w / c with a fixed step in w. Doing it this way (instead of
    np.gradient twice on a wavelength grid) avoids the noisy beta2 from the old wavelength sweep.
    Check: 785 nm gives +87.0 fs^2/mm and 1785 nm gives -27.1 fs^2/mm, matching beta2_base.
    beta3 matters here because near the zero-dispersion point beta2 alone goes to zero.
    '''
    w0_ = 2*np.pi*cc/lam
    h = 2e-3*w0_
    k = [n_mgo(2*np.pi*cc/(w0_ + j*h))*(w0_ + j*h)/cc for j in (-2, -1, 0, 1, 2)]
    beta2 = (k[3] - 2*k[2] + k[1])/h**2
    beta3 = (k[4] - 2*k[3] + 2*k[1] - k[0])/(2*h**3)
    return beta2, beta3

def E_from_I(I, lam):
    '''Peak field (V/m) from peak intensity (W/m^2) INSIDE the crystal: I = 0.5 n0 eps0 c |E|^2.
    (Note: earlier stages used I = 0.5 eps0 c |E|^2 without n0, so Stage 7 intensities are
    a factor n0 ~ 1.7 higher than the old convention for the same field.)'''
    return np.sqrt(2*I/(n_mgo(lam)*eps0*cc))

def n2_mgo(lam):
    '''Kerr index (m^2/W) implied by our equations: phase rate k*(3/8n0)*C*|E|^2 = k*n2*I.
    C is assumed wavelength-independent (no Miller's-rule correction).'''
    return 3*C7/(4*n_mgo(lam)**2*eps0*cc)

def P_critical(lam):
    '''Critical power for self-focusing (Marburger, Gaussian beam): 3.77 lam^2 / (8 pi n0 n2).'''
    return 3.77*lam**2/(8*np.pi*n_mgo(lam)*n2_mgo(lam))

# --- Stage 7 time grid: wider and coarser than t_array, since the pulses here are short and the
# dispersed light from self-compression spreads out in time. Even Nt keeps the FFTs fast.
Nt7 = 1024
t7 = np.linspace(-300e-15, 300e-15, Nt7, endpoint=False)
dt7 = t7[1] - t7[0]
W7 = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Nt7, d=dt7))     # angular frequency offset from the carrier

def propagate_7(I0, fwhm, B, w0, lam, L, dz, hcut,
                Nr=128, rmax_factor=4.0, rec_every=5, collapse_factor=4.0,
                dispersion=True, diffraction=True, optical_shock=True, verbose=True):
    '''
    Full 2D split-step (RK4 nonlinear step -> dispersion -> Hankel diffraction), like
    propagate_2d_batched, but with every physical parameter passed in.

    I0, fwhm, B, lam, hcut can each be a scalar or an array; they're broadcast together and every
    combination runs at the same time as one batch (last axis). w0 is one value per call, because
    the radial grid is scaled to it (R_max = rmax_factor * w0) so every beam size gets the same
    relative resolution.
      I0    peak intensity inside the crystal (W/m^2)
      fwhm  intensity FWHM of the input pulse (s), field is Gaussian like the rest of the code
      B     input polarization angle from vertical/[100] (rad). B = 0 is the scalar case (X stays 0)
      lam   wavelength (m), sets n0, k, beta2, beta3, the shock time and the coupling constant
      hcut  False = z-cut (001) crystal, True = h-cut with [100] vertical

    Instead of storing the whole field at every z (way too much memory), at every rec_every steps
    it stores the radially integrated pieces of the power, P(t) = 2 pi ∫ r I(r,t) dr, for |Y|^2,
    |X|^2 and Re(X Y*), which is all a polarizer + fast detector needs. Also stores the on-axis
    intensity and the peak intensity anywhere in the beam.

    There's no plasma or multiphoton absorption in this model, so once the beam self-focuses
    (peak intensity > collapse_factor * I0) the results aren't trustworthy anymore. Those z's are
    flagged in out['collapsed'] and the metrics later blank them out.
    '''
    I0, fwhm, B, lam, hcut = [np.atleast_1d(np.asarray(v, dtype=float)) for v in (I0, fwhm, B, lam, hcut)]
    I0, fwhm, B, lam, hcut = np.broadcast_arrays(I0, fwhm, B, lam, hcut)
    Nb = I0.size
    I0, fwhm, B, lam, hcut = [v.ravel() for v in (I0, fwhm, B, lam, hcut)]

    # Per-run constants, shaped (1, 1, Nb) so they broadcast over (time, radius, run)
    n0 = n_mgo(lam)
    k0 = 2*np.pi*n0/lam
    a = (2*np.pi/lam*3/8/n0)[None, None, :]                       # coupling constant k*3/8/n0, same as a_base
    shock = (lam/(2*np.pi*cc))[None, None, :]                     # 1/w0
    CX = np.where(hcut > 0.5, C7_hcut, C7)[None, None, :]         # SPM on X: the only h-cut change
    CY = C7
    b2, b3 = mgo_dispersion(lam)
    disp_phase = np.exp(-1j*(0.5*b2[None, :]*W7[:, None]**2 + b3[None, :]*W7[:, None]**3/6)*dz)[:, None, :]

    # Radial grid scaled to this beam
    H_ = HankelTransform(order=0, max_radius=rmax_factor*w0, n_points=Nr)
    r_ = H_.r
    diff_phase = np.exp(-1j*(H_.kr**2)[None, :, None]*dz/(2*k0[None, None, :]))

    # Input: Gaussian in time (field, so tau = fwhm/(2 sqrt(ln2)) like Tau_base) and in radius
    E0 = E_from_I(I0, lam)
    tau = fwhm/(2*np.sqrt(np.log(2)))
    T = np.exp(-t7[:, None, None]**2/(2*tau[None, None, :]**2))
    R = np.exp(-r_[None, :, None]**2/w0**2)
    Y = (E0*np.cos(B))[None, None, :]*T*R + 0j
    X = (E0*np.sin(B))[None, None, :]*T*R + 0j

    def nl_rhs(X, Y):
        X2, Y2 = np.abs(X)**2, np.abs(Y)**2
        NLX = (CX*X2 + 2*P7*Y2)*X + P7*Y**2*np.conj(X)            # Eq 2a (SPM coefficient CX)
        NLY = (CY*Y2 + 2*P7*X2)*Y + P7*X**2*np.conj(Y)            # Eq 2b
        if optical_shock:
            NLX = NLX - 1j*shock*np.gradient(NLX, dt7, axis=0)
            NLY = NLY - 1j*shock*np.gradient(NLY, dt7, axis=0)
        return -1j*a*NLX, -1j*a*NLY

    I_conv = (0.5*n0*eps0*cc)[None, None, :]                      # |E|^2 -> W/m^2
    wr = (2*np.pi*r_)[None, :, None]

    def record():
        return (np.trapezoid(np.abs(Y)**2*I_conv*wr, r_, axis=1),
                np.trapezoid(np.abs(X)**2*I_conv*wr, r_, axis=1),
                np.trapezoid(np.real(X*np.conj(Y))*I_conv*wr, r_, axis=1),
                (np.abs(X[:, 0, :])**2 + np.abs(Y[:, 0, :])**2)*I_conv[:, 0, :],
                np.max((np.abs(X)**2 + np.abs(Y)**2)*I_conv, axis=(0, 1)))

    Nz = int(round(L/dz))
    recs = [record()]
    z_rec = [0.0]
    for i in range(Nz):
        # (1) nonlinear step, RK4 (same as before, just on complex arrays directly)
        k1x, k1y = nl_rhs(X, Y)
        k2x, k2y = nl_rhs(X + 0.5*dz*k1x, Y + 0.5*dz*k1y)
        k3x, k3y = nl_rhs(X + 0.5*dz*k2x, Y + 0.5*dz*k2y)
        k4x, k4y = nl_rhs(X + dz*k3x, Y + dz*k3y)
        X = X + dz/6*(k1x + 2*k2x + 2*k3x + k4x)
        Y = Y + dz/6*(k1y + 2*k2y + 2*k3y + k4y)
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(Y)):
            print(f"Blew up at step {i}, z = {i*dz*1e3:.3f} mm")
            break
        # (2) dispersion (beta2 + beta3 at each run's own wavelength)
        if dispersion:
            X = np.fft.ifft(np.fft.ifftshift(disp_phase, axes=0)*np.fft.fft(X, axis=0), axis=0)
            Y = np.fft.ifft(np.fft.ifftshift(disp_phase, axes=0)*np.fft.fft(Y, axis=0), axis=0)
        # (3) diffraction
        if diffraction:
            X = H_.iqdht(H_.qdht(X, axis=1)*diff_phase, axis=1)
            Y = H_.iqdht(H_.qdht(Y, axis=1)*diff_phase, axis=1)
        if (i + 1) % rec_every == 0 or i == Nz - 1:
            recs.append(record())
            z_rec.append((i + 1)*dz)
            if verbose:
                print(f"  w0 = {w0*1e6:.0f} um: z = {(i+1)*dz*1e3:.2f} / {L*1e3:.2f} mm")
            if np.all(recs[-1][4] > collapse_factor*I0):
                print(f"  every run has self-focused by z = {(i+1)*dz*1e3:.2f} mm, stopping early")
                break

    # If we stopped early, pad the record out to the full length (copies of the last record, flagged
    # as collapsed below) so every call has the same z axis and the sweeps line up
    z_full = [0.0] + [(j + 1)*dz for j in range(Nz) if (j + 1) % rec_every == 0 or j == Nz - 1]
    while len(z_rec) < len(z_full):
        z_rec.append(z_full[len(z_rec)])
        last = list(recs[-1])
        last[4] = np.full(Nb, np.inf)                              # marks these z's as collapsed
        recs.append(tuple(last))
    Pyy, Pxx, Pxy, I_axis, I_peak = [np.array(v) for v in zip(*recs)]
    collapsed = np.maximum.accumulate(I_peak > collapse_factor*I0[None, :], axis=0)
    return dict(z=np.array(z_rec), Pyy=Pyy, Pxx=Pxx, Pxy=Pxy, I_axis=I_axis, I_peak=I_peak,
                collapsed=collapsed, X=X, Y=Y, r=r_, I0=I0, fwhm=fwhm, B=B, lam=lam, hcut=hcut, w0=w0)

def pulse_metrics(P, t=None):
    '''
    For P with time on axis 0 (any trailing axes): FWHM of the main lobe (walking out from the
    peak, same idea as fwhm_main_lobe in 5.0), the WING fraction and the peak.
    Wing fraction = energy more than one FWHM away from the peak. That's how I'm putting a number
    on "clean up the wings": a perfect Gaussian has only 1.9% out there, so anything well above
    ~2% is pedestal/side lobes.
    '''
    if t is None:
        t = t7
    dt_ = t[1] - t[0]
    Nt_ = P.shape[0]
    idx = np.arange(Nt_).reshape((-1,) + (1,)*(P.ndim - 1))
    pk = np.argmax(P, axis=0)
    half = np.max(P, axis=0)/2
    below = P <= half
    left = np.max(np.where(below & (idx < pk), idx, -1), axis=0)
    right = np.min(np.where(below & (idx > pk), idx, Nt_), axis=0)
    ok = (left >= 0) & (right < Nt_)
    l0, r0 = np.clip(left, 0, Nt_ - 2), np.clip(right, 1, Nt_ - 1)
    take = lambda i: np.take_along_axis(P, i[None, ...], axis=0)[0]
    tl = t[l0] + (half - take(l0))/(take(l0 + 1) - take(l0) + 1e-300)*dt_
    tr = t[r0 - 1] + (half - take(r0 - 1))/(take(r0) - take(r0 - 1) - 1e-300)*dt_
    width = np.where(ok, tr - tl, np.nan)
    t_pk = 0.5*(tl + tr)
    far = np.abs(t.reshape(idx.shape) - t_pk[None, ...]) > width[None, ...]
    wing = np.sum(P*far, axis=0)/(np.sum(P, axis=0) + 1e-300)
    return width, np.where(ok, wing, np.nan), np.max(P, axis=0)

def passed_power(out, dial):
    '''Whole-beam P(t) after a polarizer at dial angle(s) from vertical. Shape (Nz, Nt, Nb, Ndial).
    Same projection as Stage 5: E_pass = sin(th) X + cos(th) Y.'''
    th = np.atleast_1d(dial)[None, None, None, :]
    return (np.sin(th)**2*out['Pxx'][..., None] + np.cos(th)**2*out['Pyy'][..., None]
            + np.sin(2*th)*out['Pxy'][..., None])

# Quick look at the numbers that set the physics, so the sweeps below make sense
for lam_chk in [800e-9, 1300e-9, 1550e-9, 1785e-9, 2000e-9]:
    b2_chk, b3_chk = mgo_dispersion(lam_chk)
    print(f"{lam_chk*1e9:6.0f} nm: n0 = {n_mgo(lam_chk):.4f}, beta2 = {b2_chk*1e27:+7.2f} fs^2/mm, "
          f"beta3 = {b3_chk*1e42:6.1f} fs^3/mm, n2 = {n2_mgo(lam_chk)*1e20:.2f}e-20 m^2/W, "
          f"P_cr = {P_critical(lam_chk)*1e-6:.1f} MW")

#%% Stage 7.1 — Scalar (no polarization) self-compression at 1785 nm: tune I0, tau, w0, z            -> ~17 min run time
#
# B = 0 so only the Y field exists (FWM needs both fields, so X stays exactly 0): pure SPM +
# dispersion + diffraction, i.e. the "no polarization" case. Same for z-cut and h-cut since
# along [100] both have SPM coefficient C.
#
# Why this should compress: at 1785 nm MgO has anomalous dispersion (beta2 < 0). SPM puts red
# light on the front of the pulse and blue on the back, and anomalous dispersion makes blue
# travel faster, so the back catches up with the front -> the pulse squeezes itself
# (soliton-like self-compression). In normal dispersion (785 nm) the same chirp just spreads it.
# Rough scales: soliton order N^2 = L_D/L_NL, best compression around z ~ L_D*(0.32/N + 1.1/N^2).
#
# The catch in 2D: the beam also self-focuses (P >> P_cr), and the edges of the beam see less
# intensity, so they don't compress. A power meter / autocorrelator sees the whole beam, so the
# uncompressed edges show up as WINGS on I(t). That's what we're trying to clean up.
# What we tune: I0 and tau are batched together; w0 is looped (radial grid scales with it);
# z comes for free since every run is recorded along the crystal.

lam7 = 1785e-9
L7 = 3e-3                                     # look out to 3 mm, the record gives every z on the way
dz7 = 10e-6                                   # checked: 5 um gives the same ratios to 0.001
I0_list7 = np.array([0.5, 1, 2, 3, 4])*1e16   # W/m^2 inside the crystal (= 0.5 to 4 TW/cm^2)
fwhm_list7 = np.array([30, 50, 100])*1e-15
w0_list7 = np.array([100, 200, 400])*1e-6

I0_g7, fwhm_g7 = np.meshgrid(I0_list7, fwhm_list7, indexing='ij')     # (NI, Ntau)
scan7 = []
for w0_val in w0_list7:
    scan7.append(propagate_7(I0_g7, fwhm_g7, 0.0, w0_val, lam7, L7, dz7, hcut=False, verbose=False))
    print(f"w0 = {w0_val*1e6:.0f} um done")
z7 = scan7[0]['z']

# Metrics along z for every run: whole-beam pulse (what a detector sees) and on-axis
NI7, Ntau7, Nw7 = len(I0_list7), len(fwhm_list7), len(w0_list7)
ratio7 = np.full((Nw7, len(z7), NI7*Ntau7), np.nan)
wing7 = np.full_like(ratio7, np.nan)
ratio7_axis = np.full_like(ratio7, np.nan)
for iw, out in enumerate(scan7):
    P_beam = np.moveaxis(out['Pyy'], 1, 0)                  # (Nt, Nz, Nb)
    width, wing, _ = pulse_metrics(P_beam)
    width_ax, _, _ = pulse_metrics(np.moveaxis(out['I_axis'], 1, 0))
    bad = out['collapsed']
    ratio7[iw] = np.where(bad, np.nan, width/out['fwhm'][None, :])
    wing7[iw] = np.where(bad, np.nan, wing)
    ratio7_axis[iw] = np.where(bad, np.nan, width_ax/out['fwhm'][None, :])
ratio7 = ratio7.reshape(Nw7, len(z7), NI7, Ntau7)
wing7 = wing7.reshape(Nw7, len(z7), NI7, Ntau7)
ratio7_axis = ratio7_axis.reshape(Nw7, len(z7), NI7, Ntau7)

# --- Pick the best clean setting: shortest whole-beam pulse whose wings are at most wing_max7 ---
wing_max7 = 0.10                              # 10% of the energy outside +/- 1 FWHM (Gaussian = 1.9%)
score7 = np.where(wing7 <= wing_max7, ratio7, np.nan)
iw7, iz7, iI7, itau7 = np.unravel_index(np.nanargmin(score7), score7.shape)
w0_best7, z_best7 = w0_list7[iw7], z7[iz7]
I0_best7, fwhm_best7 = I0_list7[iI7], fwhm_list7[itau7]
print(f"\nBest clean scalar compression (wings <= {wing_max7*100:.0f}%): "
      f"w0 = {w0_best7*1e6:.0f} um, I0 = {I0_best7*1e-16:.1f} TW/cm^2, input FWHM = {fwhm_best7*1e15:.0f} fs, "
      f"z = {z_best7*1e3:.2f} mm -> ratio {ratio7[iw7, iz7, iI7, itau7]:.3f} "
      f"({ratio7[iw7, iz7, iI7, itau7]*fwhm_best7*1e15:.1f} fs), wings {wing7[iw7, iz7, iI7, itau7]*100:.1f}%")

# Table for the meeting: best (shortest) clean pulse per (w0, tau), over all I0 and z
print("\nBest whole-beam ratio with wings <= 10%  [I0 (TW/cm^2), z (mm), wings]:")
for iw in range(Nw7):
    for it in range(Ntau7):
        s = score7[iw, :, :, it]
        if np.all(np.isnan(s)):
            print(f"  w0 = {w0_list7[iw]*1e6:3.0f} um, {fwhm_list7[it]*1e15:3.0f} fs: nothing clean")
            continue
        jz, jI = np.unravel_index(np.nanargmin(s), s.shape)
        print(f"  w0 = {w0_list7[iw]*1e6:3.0f} um, {fwhm_list7[it]*1e15:3.0f} fs: ratio {s[jz, jI]:.3f} "
              f"[{I0_list7[jI]*1e-16:.1f}, {z7[jz]*1e3:.2f}, {wing7[iw, jz, jI, it]*100:.1f}%]   "
              f"(on-axis ratio there {ratio7_axis[iw, jz, jI, it]:.3f})")

# --- FWHM ratio and wings vs z, at the best tau and w0, one line per I0 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for iI in range(NI7):
    lab = f'{I0_list7[iI]*1e-16:.1f} TW/cm²'
    axes[0].plot(z7*1e3, ratio7[iw7, :, iI, itau7], label=lab)
    axes[1].plot(z7*1e3, wing7[iw7, :, iI, itau7]*100, label=lab)
    axes[2].plot(z7*1e3, ratio7_axis[iw7, :, iI, itau7], label=lab)
axes[0].set_ylabel('Whole-beam FWHM out / in')
axes[1].set_ylabel('Wing energy (% outside ±1 FWHM)')
axes[1].axhline(1.9, color='k', ls=':', lw=0.8, label='Gaussian (1.9%)')
axes[2].set_ylabel('On-axis FWHM out / in')
for ax in axes:
    ax.set_xlabel('z (mm)')
    ax.legend(fontsize=7)
fig.suptitle(f'Scalar, {lam7*1e9:.0f} nm, w0 = {w0_best7*1e6:.0f} µm, input {fwhm_best7*1e15:.0f} fs '
             f'(lines end where the beam self-focuses past {4}x I0)')
plt.tight_layout()

# --- Map: best clean ratio over z, for every (I0, tau), one panel per w0 ---
fig, axes = plt.subplots(1, Nw7, figsize=(5*Nw7, 4.2))
for iw in range(Nw7):
    best_map = np.nanmin(np.where(np.isnan(score7[iw]), np.inf, score7[iw]), axis=0)
    best_map[np.isinf(best_map)] = np.nan
    axes[iw].set_facecolor('0.6')
    im = axes[iw].pcolormesh(fwhm_list7*1e15, I0_list7*1e-16, best_map, shading='nearest',
                             cmap=inverse_inferno_white, vmin=0.2, vmax=1.0)
    for iI in range(NI7):
        for it in range(Ntau7):
            if np.isfinite(best_map[iI, it]):
                axes[iw].text(fwhm_list7[it]*1e15, I0_list7[iI]*1e-16, f'{best_map[iI, it]:.2f}',
                              ha='center', va='center', fontsize=8)
    axes[iw].set_xlabel('Input FWHM (fs)')
    axes[iw].set_ylabel('I0 (TW/cm²)')
    axes[iw].set_title(f'w0 = {w0_list7[iw]*1e6:.0f} µm')
    fig.colorbar(im, ax=axes[iw], label='best FWHM ratio (wings ≤ 10%)')
fig.suptitle('Scalar self-compression at 1785 nm, best over z ≤ 3 mm (grey: nothing clean before collapse)')
plt.tight_layout()

# --- Best pulse, linear and log scale (the log plot is where the wings really show) ---
best_run7 = scan7[iw7]
ib7 = iI7*Ntau7 + itau7
P_best_out7 = best_run7['Pyy'][iz7, :, ib7]
P_best_in7 = best_run7['Pyy'][0, :, ib7]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax in axes:
    ax.plot(t7*1e15, P_best_in7/np.max(P_best_in7), 'k:', label=f'Input ({fwhm_best7*1e15:.0f} fs)')
    ax.plot(t7*1e15, P_best_out7/np.max(P_best_out7), label=f'z = {z_best7*1e3:.2f} mm')
    ax.set_xlabel('Time (fs)')
    ax.set_xlim(-3*fwhm_best7*1e15, 3*fwhm_best7*1e15)
axes[0].set_ylabel('Normalized whole-beam power')
axes[1].set_yscale('log')
axes[1].set_ylim(1e-4, 1.5)
axes[0].legend()
fig.suptitle('Best clean scalar pulse (whole beam)')
plt.tight_layout()

#%% Stage 7.2 — Wavelength scaling: does anomalous dispersion compress the pulse by itself?       -> ~3 min run time
#
# Same crystal, same beam, same peak intensity and input duration as the best 7.1 point, only the
# wavelength changes. Below ~1.55 um MgO has normal dispersion, above it anomalous.
# Expectation: normal dispersion -> SPM chirp + dispersion stretch the pulse (ratio > 1);
# anomalous -> self-compression (ratio < 1), and it gets stronger as |beta2| grows with lambda.
# Keep in mind the critical power goes like lambda^2, so at short wavelengths the same intensity
# is many more P_cr and the beam collapses sooner (that's where the lines end).
# Note: run 7.1 first (uses w0_best7, I0_best7, fwhm_best7).

lam_list7 = np.array([800, 1000, 1300, 1550, 1785, 2000, 2200])*1e-9
wl7 = propagate_7(I0_best7, fwhm_best7, 0.0, w0_best7, lam_list7, L7, dz7, hcut=False, verbose=False)
width_wl7, wing_wl7, peak_wl7 = pulse_metrics(np.moveaxis(wl7['Pyy'], 1, 0))     # (Nz, Nlam)
ratio_wl7 = np.where(wl7['collapsed'], np.nan, width_wl7/fwhm_best7)
wing_wl7 = np.where(wl7['collapsed'], np.nan, wing_wl7)
peakgain_wl7 = np.where(wl7['collapsed'], np.nan, peak_wl7/peak_wl7[0][None, :])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for il, lam_val in enumerate(lam_list7):
    b2_val = mgo_dispersion(lam_val)[0]
    lab = f'{lam_val*1e9:.0f} nm (β2 {b2_val*1e27:+.0f} fs²/mm)'
    axes[0].plot(wl7['z']*1e3, ratio_wl7[:, il], label=lab)
    axes[1].plot(wl7['z']*1e3, wing_wl7[:, il]*100, label=lab)
    axes[2].plot(wl7['z']*1e3, peakgain_wl7[:, il], label=lab)
axes[0].axhline(1, color='k', lw=0.5)
axes[0].set_ylabel('Whole-beam FWHM out / in')
axes[1].set_ylabel('Wing energy (%)')
axes[2].set_ylabel('Peak power / input peak power')
for ax in axes:
    ax.set_xlabel('z (mm)')
    ax.legend(fontsize=7)
fig.suptitle(f'Wavelength scaling: I0 = {I0_best7*1e-16:.1f} TW/cm², {fwhm_best7*1e15:.0f} fs, w0 = {w0_best7*1e6:.0f} µm')
plt.tight_layout()

print("\nShortest whole-beam pulse before collapse, per wavelength:")
for il, lam_val in enumerate(lam_list7):
    if np.all(np.isnan(ratio_wl7[:, il])):
        print(f"  {lam_val*1e9:5.0f} nm: collapsed straight away")
        continue
    jz = np.nanargmin(ratio_wl7[:, il])
    print(f"  {lam_val*1e9:5.0f} nm: ratio {ratio_wl7[jz, il]:.3f} at z = {wl7['z'][jz]*1e3:.2f} mm, "
          f"wings {wing_wl7[jz, il]*100:.1f}%, P/P_cr = "
          f"{np.pi*w0_best7**2*I0_best7/2/P_critical(lam_val):.0f}")

#%% Stage 7.3 — Add polarization: z-cut vs h-cut + polarizer to clean the wings                      -> ~22 min run time
#
# Same beam as the best 7.1 point, but now the input is rotated by B from [100] so the X field
# is driven, and a polarizer after the crystal picks out part of the light. The polarizer works
# as an intensity filter: the peak of the pulse (and the centre of the beam) rotates more than the
# low-intensity wings and edges, so the right dial angle passes the compressed core and blocks
# the uncompressed wings. That's the same trick as XPW filtering (Jullien) and Stage 5, but now on
# top of anomalous-dispersion self-compression.
# z-cut and h-cut run in the same batch. In the h-cut B = 0 is along [100] (same as z-cut) and
# B = 90 is along [01-1] (SPM 1.31x stronger), and it isn't symmetric about 45 deg like the z-cut.
# Note: run 7.1 first.

#B_list7 = np.deg2rad([0, 2, 5, 10, 15, 20, 25.5, 30, 40, 50, 60, 70, 80, 85, 88, 90])
B_list7 = np.deg2rad([0, 10, 80])
cut_list7 = np.array([0, 1])                                   # 0 = z-cut, 1 = h-cut
B_g7, cut_g7 = np.meshgrid(B_list7, cut_list7, indexing='ij')  # (NB, 2)
pol7 = propagate_7(I0_best7, fwhm_best7, B_g7, w0_best7, lam7, L7, dz7, hcut=cut_g7, verbose=False)

dial7 = np.deg2rad(np.linspace(0, 180, 181))
trans_floor7 = 0.01                                            # need at least 1% of the energy through

P_pass7 = passed_power(pol7, dial7)                            # (Nz, Nt, Nb, Ndial)
E_tot7 = np.sum(pol7['Pyy'] + pol7['Pxx'], axis=1)             # (Nz, Nb)
T7 = np.sum(P_pass7, axis=1)/E_tot7[..., None]                 # transmission (Nz, Nb, Ndial)
w_p7, wing_p7, pk_p7 = pulse_metrics(np.moveaxis(P_pass7, 1, 0))
ok7 = (~pol7['collapsed'])[..., None] & (T7 >= trans_floor7)
ratio_p7 = np.where(ok7, w_p7/fwhm_best7, np.nan)
wing_p7 = np.where(ok7, wing_p7, np.nan)
# A single number to rank by: compressed AND clean. Ratio x (1 + wings/Gaussian-wings) would be
# fussy; simpler to ask for the shortest pulse whose wings beat the scalar best point.
wing_target7 = wing7[iw7, iz7, iI7, itau7]

cut_names7 = ['z-cut', 'h-cut']
best_pol7 = {}
print(f"\nScalar reference: ratio {ratio7[iw7, iz7, iI7, itau7]:.3f}, wings {wing_target7*100:.1f}% "
      f"at z = {z_best7*1e3:.2f} mm")
for ic in range(2):
    sel = (cut_g7.ravel() == ic)
    s = np.where(sel[None, :, None] & (wing_p7 <= wing_target7), ratio_p7, np.nan)
    if np.all(np.isnan(s)):
        print(f"{cut_names7[ic]}: nothing beats the scalar wings with >= {trans_floor7*100:.0f}% through")
        continue
    jz, jb, jd = np.unravel_index(np.nanargmin(s), s.shape)
    best_pol7[ic] = (jz, jb, jd)
    print(f"{cut_names7[ic]}: B = {np.rad2deg(pol7['B'][jb]):.1f} deg, dial = {np.rad2deg(dial7[jd]):.0f} deg, "
          f"z = {pol7['z'][jz]*1e3:.2f} mm -> ratio {ratio_p7[jz, jb, jd]:.3f} "
          f"({ratio_p7[jz, jb, jd]*fwhm_best7*1e15:.1f} fs), wings {wing_p7[jz, jb, jd]*100:.1f}%, "
          f"T = {T7[jz, jb, jd]*100:.1f}%")

# Same comparison at a fixed length (the scalar best z), so the crystal is the same for both cuts
jz_fix7 = np.argmin(np.abs(pol7['z'] - z_best7))
print(f"\nAt z = {pol7['z'][jz_fix7]*1e3:.2f} mm, best (ratio, wings, T) above {trans_floor7*100:.0f}% transmission:")
for ic in range(2):
    sel = (cut_g7.ravel() == ic)
    for wing_cap in [0.05, wing_target7]:
        s = np.where(sel[:, None] & (wing_p7[jz_fix7] <= wing_cap), ratio_p7[jz_fix7], np.nan)
        if np.all(np.isnan(s)):
            print(f"  {cut_names7[ic]}, wings <= {wing_cap*100:.1f}%: nothing")
            continue
        jb, jd = np.unravel_index(np.nanargmin(s), s.shape)
        print(f"  {cut_names7[ic]}, wings <= {wing_cap*100:.1f}%: ratio {s[jb, jd]:.3f}, wings "
              f"{wing_p7[jz_fix7, jb, jd]*100:.1f}%, T {T7[jz_fix7, jb, jd]*100:.1f}% "
              f"(B = {np.rad2deg(pol7['B'][jb]):.1f}, dial {np.rad2deg(dial7[jd]):.0f})")

# Maps at the fixed length: transmission, ratio and wings over (B, dial), z-cut on top, h-cut below
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ic in range(2):
    sel = np.where(cut_g7.ravel() == ic)[0]
    Bdeg = np.rad2deg(pol7['B'][sel])
    for k, (data, lab, cmap, vr) in enumerate([
            (T7[jz_fix7][sel]*100, 'Transmission (%)', 'viridis', (0, 100)),
            (ratio_p7[jz_fix7][sel], 'FWHM out / in', inverse_inferno_white, (0.2, 1.2)),
            (wing_p7[jz_fix7][sel]*100, 'Wing energy (%)', 'magma_r', (0, 25))]):
        ax = axes[ic, k]
        ax.set_facecolor('0.6')
        im = ax.pcolormesh(np.rad2deg(dial7), Bdeg, data, shading='nearest', cmap=cmap, vmin=vr[0], vmax=vr[1])
        fig.colorbar(im, ax=ax, label=lab)
        ax.set_xlabel('Polarizer dial angle (deg)')
        ax.set_ylabel('Input angle B from [100] (deg)')
        ax.set_title(f'{cut_names7[ic]}: {lab}')
fig.suptitle(f'z = {pol7["z"][jz_fix7]*1e3:.2f} mm, {lam7*1e9:.0f} nm, I0 = {I0_best7*1e-16:.1f} TW/cm², '
             f'{fwhm_best7*1e15:.0f} fs, w0 = {w0_best7*1e6:.0f} µm (grey: < {trans_floor7*100:.0f}% through or collapsed)')
plt.tight_layout()

#%% Stage 7.4 — Summary figure for the meeting: input vs scalar vs z-cut vs h-cut                   -> instant run time
# Run 7.1-7.3 first. Log scale on the right is the one to look at for the wings.

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
curves7 = [(P_best_in7, 'k:', f'Input ({fwhm_best7*1e15:.0f} fs)'),
           (P_best_out7, 'C0-', f'Scalar, z = {z_best7*1e3:.2f} mm '
                                f'({ratio7[iw7, iz7, iI7, itau7]*fwhm_best7*1e15:.0f} fs, wings {wing_target7*100:.0f}%)')]
for ic, col in zip(range(2), ['C1-', 'C3-']):
    if ic not in best_pol7:
        continue
    jz, jb, jd = best_pol7[ic]
    curves7.append((P_pass7[jz, :, jb, jd], col,
                    f'{cut_names7[ic]} + polarizer, z = {pol7["z"][jz]*1e3:.2f} mm, B = {np.rad2deg(pol7["B"][jb]):.1f}° '
                    f'({w_p7[jz, jb, jd]*1e15:.0f} fs, wings {wing_p7[jz, jb, jd]*100:.0f}%, T {T7[jz, jb, jd]*100:.0f}%)'))
for ax in axes:
    for P_c, sty, lab in curves7:
        ax.plot(t7*1e15, P_c/np.max(P_c), sty, label=lab)
    ax.set_xlabel('Time (fs)')
    ax.set_xlim(-2.5*fwhm_best7*1e15, 2.5*fwhm_best7*1e15)
axes[0].set_ylabel('Normalized whole-beam power')
axes[1].set_yscale('log')
axes[1].set_ylim(1e-4, 1.5)
axes[0].legend(fontsize=7, loc='upper left')
fig.suptitle(f'MgO at {lam7*1e9:.0f} nm: self-compression, then polarization cleaning')
plt.tight_layout()
plt.show()
