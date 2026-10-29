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
eps0 = 8.854e-12                    # Permittivity of free space

# --- Crystal: BaF2 (Jullien et al., Opt. Lett. 33, 2353, 2008) ---
# Only the crystal (and their 800 nm / 25 fs pulse) changes from Zach16; the beam stays the same Gaussian.
# MgO values are kept commented out so we can switch back.
#C = 3.2e-22                        # MgO X^(3)_xxxx
#P = C/1.85                         # MgO X^(3)_xxyy (anisotropy sigma = 1 - 3/1.85 = -0.62)
def n_BaF2(lam):
    '''Malitson (1964) Sellmeier for BaF2, lam in m.'''
    l2 = (lam*1e6)**2
    return np.sqrt(1 + 0.643356*l2/(l2 - 0.057789**2) + 0.506762*l2/(l2 - 0.10968**2)
                   + 3.8261*l2/(l2 - 46.3864**2))
n2_BaF2 = 2.0e-20                   # Nonlinear index (m^2/W); doesn't matter for the Jullien test since we set the field from S
sigma_BaF2 = -1.2                   # Anisotropy (X_xxxx - 3X_xxyy)/X_xxxx, Minkovski et al. JOSA B 2004 -- still need to check with Hammond
n0_BaF2 = n_BaF2(800e-9)
C = 4/3 * n0_BaF2**2 * eps0 * cc * n2_BaF2         # X^(3)_xxxx from n2
P = C*(1 - sigma_BaF2)/3                            # X^(3)_xxyy, P/C = 0.733 (MgO was 0.54)

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
#FWHM_base = 100e-15                                # MgO runs
FWHM_base = 25e-15                                  # Pulse duration (s), Jullien's laser
Tau_base = FWHM_base/(2*(np.log(2))**0.5)           # Width of gaussian (s)
#L_base = 500e-6                                    # MgO runs
L_base = 2e-3                                       # Jullien's 2 mm BaF2
#Lambda_base = 785e-9                               # MgO runs
Lambda_base = 800e-9
Lambda_base_micron = Lambda_base*1e6
wp_base = 2*np.pi*cc/Lambda_base                    # Pump wavelength (rad/s)
B_base = np.deg2rad(2)
#beta2_base = 87.149e-27                            # MgO at 785 nm, from refractiveindex.info
#beta2_base = -27.138e-27
# BaF2 beta2 from a fine central difference of k(w) on the Sellmeier (~38 fs^2/mm at 800 nm)
dw_fd = wp_base*1e-3
k_of_w = lambda om: n_BaF2(2*np.pi*cc/om)*om/cc
beta2_base = (k_of_w(wp_base + dw_fd) - 2*k_of_w(wp_base) + k_of_w(wp_base - dw_fd))/dw_fd**2


# --- Necessary Calculations ---
dz_base = L_base/100
z_eval_base = np.arange(0, L_base, dz_base)
#n_base = (2.956362 + 0.02195770/(Lambda_base_micron**2 - 0.01428322) - 0.01062387*Lambda_base_micron**2 - 0.0000204968*Lambda_base_micron**4)**0.5   # MgO
n_base = n_BaF2(Lambda_base)
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

#%% Stage 6 — Jullien cross-check: BaF2 XPW efficiency, spectrum and shortening vs S                -> ~15 min run time
#
# Needs the Parameters cell (now BaF2), Functions, Stage 3 and Stage 4.0 (grid + helpers). Nothing from 4.1-5.1.
#
# Bug test against Jullien et al., Opt. Lett. 33, 2353 (2008): 2 mm BaF2, 25 fs, 800 nm, crossed polarizers.
# Beam is OUR Gaussian (w0_base = 100 um), not their super-Gaussian, so their efficiencies should come out
# higher than ours -- the point is to see if the physics lines up, not to match a different beam.
#
# They plot everything against S = (2pi/lambda) n2 I0 L. In our equations the SPM phase is a*C*E0^2*L,
# and with C = (4/3) n0^2 eps0 c n2 and I0 = 1/2 n0 eps0 c E0^2 that's exactly S. So set E0 from S
# and n2 cancels out -- the comparison doesn't depend on which n2 value we believe.
#
# Their numbers to compare against: Fig. 3 efficiency vs S (up to 28%, and they say Gaussian beams top out
# around 15%), Fig. 4b spectral width ~sqrt3 below 15% efficiency up to ~2.5x at 28%, and Fig. 5b 2.2x
# shortening at 23% efficiency after removing the leftover spectral phase.

S_sweep_J = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
B_J_deg = np.array([10.0, 15.0, 20.0, 22.5, 25.0, 30.0])     # XPW peaks near 22.5 deg from the crystal axis, so scan around it

SS_J, BB_J = np.meshgrid(S_sweep_J, np.deg2rad(B_J_deg), indexing='ij')
SS_J, BB_J = SS_J.ravel(), BB_J.ravel()
E0_J = np.sqrt(SS_J/(a_base*C*L_base))                       # on-axis peak field for each S

# propagate_2d_batched broadcasts Y0 against B, so every (S, B) pair goes in one batch
z_eval_J = np.arange(0, L_base + dz_base/2, dz_base)
X_J, Y_J = propagate_2d_batched(E0_J, Tau_base, BB_J, z_eval_J, dz_base,
                                 H, r_array, dispersion=True, diffraction=True)

# Efficiency = whole-beam energy through a polarizer crossed with the input (dial = B + 90)
S0_J, S1_J, S2_J = integrated_stokes(X_J, Y_J, r_array)
eff_J = transmission(S0_J, S1_J, S2_J, to_dial(BB_J + np.pi/2))

# The XPW field itself (same projection as transmission()), and its whole-beam power vs time
th_J = (BB_J + np.pi/2 - polarizer_zero_offset)[None, None, :]
E_xpw_J = np.sin(th_J)*X_J + np.cos(th_J)*Y_J
w_r_J = r_array[None, :, None]
P_xpw_J = np.trapezoid(np.abs(E_xpw_J)**2*w_r_J, r_array, axis=1)          # (Nt, N)

# Spectra: zero-pad 8x because a 25 fs pulse is only ~7 frequency bins wide on the plain grid.
# Pad evenly on both sides in time, otherwise the pulse gets split in two and the spectrum gets fake fringes.
Npad_J = 8*Nt
w_pad_J = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Npad_J, d=dt))
t_pad_J = (np.arange(Npad_J) - Npad_J//2)*dt
def padded_spectrum(E):
    pad = [(0, 0)]*E.ndim
    pad[0] = ((Npad_J - E.shape[0])//2, Npad_J - E.shape[0] - (Npad_J - E.shape[0])//2)
    return np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(np.pad(E, pad), axes=0), axis=0), axes=0))**2

Spec_xpw_J = np.trapezoid(padded_spectrum(E_xpw_J)*w_r_J, r_array, axis=1)  # whole-beam spectrum, like their spectrometer
dw_in_J = fwhm_from_intensity(w_pad_J, padded_spectrum(np.exp(-t_array**2/(2*Tau_base**2))))

spec_J = np.zeros(len(SS_J))
raw_J = np.zeros(len(SS_J))
tl_J = np.zeros(len(SS_J))
for k in range(len(SS_J)):
    spec_J[k] = fwhm_from_intensity(w_pad_J, Spec_xpw_J[:, k])/dw_in_J
    raw_J[k] = fwhm_from_intensity(t_array, P_xpw_J[:, k])                  # straight out of the crystal, no compressor
    # Transform limit = same spectrum with flat phase (an ideal compressor, like them removing the residual phase)
    E_tl = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(np.sqrt(Spec_xpw_J[:, k]))))
    tl_J[k] = fwhm_from_intensity(t_pad_J, np.abs(E_tl)**2)

shape_J = (len(S_sweep_J), len(B_J_deg))
eff_J, spec_J, raw_J, tl_J = (x.reshape(shape_J) for x in (eff_J, spec_J, raw_J, tl_J))

# They'd have tuned the crystal for the most XPW, so keep the best input angle at each S
j_best = np.argmax(eff_J, axis=1)
i_S = np.arange(len(S_sweep_J))
eff_b, spec_b, raw_b, tl_b = eff_J[i_S, j_best], spec_J[i_S, j_best], raw_J[i_S, j_best], tl_J[i_S, j_best]

print(f"BaF2: n0 = {n_base:.4f}, beta2 = {beta2_base*1e27:.1f} fs^2/mm, P/C = {P/C:.3f}")
print(" S   |  eff   best B | spectrum/input | raw FWHM | TL FWHM | 25 fs / TL")
for k in range(len(S_sweep_J)):
    print(f"{S_sweep_J[k]:4.1f} | {eff_b[k]*100:5.1f}%  {B_J_deg[j_best[k]]:4.1f} |      {spec_b[k]:.2f}      |"
          f" {raw_b[k]*1e15:5.1f} fs | {tl_b[k]*1e15:5.1f} fs |   {FWHM_base/tl_b[k]:.2f}")

# Jullien's data, digitized off a 600 dpi render of the paper (pixel positions -> axis values,
# so good to roughly +/- 0.03 in S, +/- 0.3% efficiency, +/- 0.02 in spectral ratio).
# The old Fig. 3 list was read by eye and was off by up to ~2x at low S -- these replace it.

# Fig. 3: measured XPW efficiency (%) vs S (filled circles)
S_exp_J = np.array([0.80, 0.99, 1.14, 1.41, 1.47, 1.75, 1.88, 2.19, 2.41, 2.72,
                    2.77, 2.88, 3.12, 3.35, 3.50, 3.67, 3.91, 4.13])
eff_exp_J = np.array([2.5, 3.7, 4.4, 6.9, 7.7, 10.6, 10.9, 14.5, 17.2, 19.2,
                      20.2, 20.8, 23.2, 23.5, 25.1, 26.8, 27.8, 28.3])
# Fig. 3: their model (dashed curve, super-Gaussian beam, [011] cut)
S_th_J = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.6])
eff_th_J = np.array([0.7, 2.8, 6.5, 11.1, 15.8, 22.4, 29.3, 34.4, 39.7])

# Fig. 4b: measured XPW spectral width / laser spectral width vs XPW efficiency (diamonds)
eff_spec_exp_J = np.array([2.2, 6.9, 7.5, 10.6, 11.0, 14.6, 17.4, 19.1, 20.3, 22.4, 23.5, 26.8, 28.0, 28.6])
spec_exp_J = np.array([1.75, 1.70, 1.61, 1.64, 1.73, 1.94, 2.13, 2.20, 2.22, 2.37, 2.41, 2.48, 2.53, 2.85])
# Fig. 4b: their model (solid curve)
eff_spec_th_J = np.array([1.0, 7.5, 15.0, 20.0, 27.0, 30.0])
spec_th_J = np.array([1.74, 1.84, 2.00, 2.17, 2.55, 2.90])

#%%
plt.figure()
plt.plot(S_exp_J, eff_exp_J, 'ko', label='Jullien 2008 data (Fig. 3)')
plt.plot(S_th_J, eff_th_J, 'k--', lw=1, label='Jullien 2008 model (super-Gaussian)')
plt.plot(S_sweep_J, eff_b*100, 'o-', label='Our model (Gaussian beam)')
plt.axhline(15, color='0.6', ls='--', lw=0.8, label='Their quoted Gaussian-beam limit (15%)')
plt.xlabel('S = (2π/λ) n₂ I₀ L')
plt.ylabel('XPW efficiency (%)')
plt.title('Efficiency vs S (Jullien Fig. 3)')
plt.legend()

plt.figure()
plt.plot(eff_spec_exp_J, spec_exp_J, 'kD', ms=5, label='Jullien 2008 data (Fig. 4b)')
plt.plot(eff_spec_th_J, spec_th_J, 'k--', lw=1, label='Jullien 2008 model')
plt.plot(eff_b*100, spec_b, 'o-', label='Our model (Gaussian beam)')
plt.axhline(np.sqrt(3), color='k', ls=':', label='√3')
plt.xlabel('XPW efficiency (%)')
plt.ylabel('XPW spectral width / input width')
plt.title('Spectral broadening (Jullien Fig. 4b)')
plt.legend()

#%% Stage 6.1 — Jullien Fig. 5b equivalent: XPW pulse in time + spectrum and phase                  -> ~1 min run time
#
# Needs Parameters (BaF2), Functions, Stage 3 and Stage 4.0. Doesn't need the Stage 6 sweep.
#
# Their Fig. 5b is the XPW pulse at 23% efficiency after they computationally removed the leftover
# spectral phase (11.5 fs), plus that phase. Our Gaussian beam tops out at 19.5% (S = 4.5, best angle 20 deg),
# so this is the closest point we have. Transform-limited = same spectrum with flat phase (ideal compressor).

S_5b = 4.5
B_5b = np.deg2rad(20.0)
E0_5b = np.sqrt(S_5b/(a_base*C*L_base))                 # same S -> E0 conversion as Stage 6
z_5b = np.arange(0, L_base + dz_base/2, dz_base)
X_5b, Y_5b = propagate_2d_batched(np.array([E0_5b]), Tau_base, np.array([B_5b]), z_5b, dz_base,
                                   H, r_array, dispersion=True, diffraction=True)

# XPW = component through the polarizer crossed with the input (dial = B + 90)
th_5b = B_5b + np.pi/2 - polarizer_zero_offset
E_xpw_5b = (np.sin(th_5b)*X_5b + np.cos(th_5b)*Y_5b)[:, :, 0]          # (Nt, Nr)
S0_5b, S1_5b, S2_5b = integrated_stokes(X_5b, Y_5b, r_array)
eff_5b = transmission(S0_5b, S1_5b, S2_5b, to_dial(np.array([B_5b + np.pi/2])))[0]

# Zero-pad 8x in time (evenly on both sides) so the 25 fs spectrum isn't just a few bins
Npad_5b = 8*Nt
pad_5b = ((Npad_5b - Nt)//2, Npad_5b - Nt - (Npad_5b - Nt)//2)
t_pad_5b = (np.arange(Npad_5b) - Npad_5b//2)*dt
w_pad_5b = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Npad_5b, d=dt))
Ew_5b = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(np.pad(E_xpw_5b, (pad_5b, (0, 0))), axes=0), axis=0), axes=0)

Spec_5b = np.trapezoid(np.abs(Ew_5b)**2*r_array[None, :], r_array, axis=1)   # whole-beam spectrum
phase_5b = np.unwrap(np.angle(Ew_5b[:, 0]))                                   # phase on axis, where most of the energy is

# Pulse straight out of the crystal (whole-beam power vs time) and the transform limit
P_raw_5b = np.trapezoid(np.abs(E_xpw_5b)**2*r_array[None, :], r_array, axis=1)
E_tl_5b = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(np.sqrt(Spec_5b))))
I_tl_5b = np.abs(E_tl_5b)**2
fwhm_tl_5b = fwhm_from_intensity(t_pad_5b, I_tl_5b)
fwhm_raw_5b = fwhm_from_intensity(t_array, P_raw_5b)

# Envelope frequency -> wavelength (our w is relative to the carrier wp_base)
lam_nm_5b = 2*np.pi*cc/(wp_base + w_pad_5b)*1e9
in_band = Spec_5b > 0.02*np.max(Spec_5b)                  # only show phase where there's actually light

# Take out the constant and linear parts of the phase (just an overall phase and a time shift),
# like a SPIDER retrieval would, so only the chirp is left
ph_5b = phase_5b - phase_5b[np.argmax(Spec_5b)]
lin_fit = np.polyfit(w_pad_5b[in_band], ph_5b[in_band], 1)
ph_5b = ph_5b - np.polyval(lin_fit, w_pad_5b)

Spec_in_5b = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(
    np.pad(np.exp(-t_array**2/(2*Tau_base**2)), pad_5b)))))**2
in_band_input = Spec_in_5b > 1e-3*np.max(Spec_in_5b)

print(f"S = {S_5b}, B = {np.rad2deg(B_5b):.1f} deg: efficiency {eff_5b*100:.1f}%, "
      f"raw XPW {fwhm_raw_5b*1e15:.1f} fs, transform-limited {fwhm_tl_5b*1e15:.1f} fs "
      f"({FWHM_base/fwhm_tl_5b:.2f}x shorter than the input)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(t_pad_5b*1e15, I_tl_5b/np.max(I_tl_5b), 'k-', label=f'Transform-limited ({fwhm_tl_5b*1e15:.1f} fs)')
axes[0].plot(t_array*1e15, P_raw_5b/np.max(P_raw_5b), 'C1--', label=f'Straight out of crystal ({fwhm_raw_5b*1e15:.1f} fs)')
axes[0].plot(t_array*1e15, np.exp(-t_array**2/Tau_base**2), 'C0:', label=f'Input ({FWHM_base*1e15:.0f} fs)')
axes[0].set_xlim(-50, 50)
axes[0].set_xlabel('Time (fs)')
axes[0].set_ylabel('Temporal intensity')
axes[0].set_title(f'XPW pulse, S = {S_5b}, B = {np.rad2deg(B_5b):.0f}°, efficiency {eff_5b*100:.1f}%\n'
                  f'(Jullien Fig. 5b: 11.5 fs at 23%)')
axes[0].legend(fontsize=8)

axes[1].plot(lam_nm_5b[in_band], Spec_5b[in_band]/np.max(Spec_5b), 'k-', label='XPW spectrum')
axes[1].plot(lam_nm_5b[in_band_input], Spec_in_5b[in_band_input]/np.max(Spec_in_5b), 'k:', label='Input spectrum')
axes[1].set_xlim(700, 900)
axes[1].set_xlabel('Wavelength (nm)')
axes[1].set_ylabel('Spectral intensity')
axes[1].legend(loc='upper left', fontsize=8)
ax_ph = axes[1].twinx()
ax_ph.plot(lam_nm_5b[in_band], ph_5b[in_band], 'C3--')
ax_ph.set_ylim(-5, 5)
ax_ph.set_ylabel('Spectral phase (rad), on axis', color='C3')
axes[1].set_title('Spectrum and phase (compare Jullien Fig. 5a / 5b dashed)')
plt.tight_layout()
plt.show()

