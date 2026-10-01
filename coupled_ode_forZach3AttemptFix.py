# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:27:21 2025

@author: TJ Hammond
"""

import numpy as np
import matplotlib.pyplot as plt
import math
plt.close('all')

#%% Main loop
cc = 3e8
'''Parameters'''
a = 2*np.pi/800e-9*3/8*1/1.73     # Coupling constant k*3/8/n0, where n0 is the index of refraction
C = 3.2e-22 # Nonlinear X^(3) susceptibility tensor X^(3)_xxxx
P = C/1.85  # Cross-phase modulation coefficient, X^(3)_xxyy, relative to X^(3)_xxxx
Y0 = 1.0*6e9    # Amplitude of initial Gaussian envelope
# tau = 400e-15   # Width of the Gaussian envelope
FWHM = 120e-15 # Pulse duration
tau = FWHM/(2*(2*np.log(2))**0.5) # ~50.9 fs
# print(tau)
wp = 2*np.pi*cc/800e-9    # Frequency wp (pump)
ws = 2*np.pi*cc/600e-9    # Frequency ws (signal)
B = 0.05  # Angle for X(t) initial condition
s = 1e-7 # Relative initial field amplitude for signal
phi = 0.0  # Phase shift for X(t)


# Define the time array
Nt = 10001
t_array = np.linspace(-200e-15, 200e-15, Nt)  # Time array for the initial conditions
dt = t_array[1]-t_array[0]

w = 2*np.pi * np.fft.fftfreq(Nt, d=dt)
w = np.fft.fftshift(w)

bandwidth = 4/tau   # ~4/tau is a generous cutoff covering the pulse spectrum
mask = np.exp(-(w - wp)**2 / (2 * bandwidth**2))  # Gaussian filter centered on pump

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
    shock_coeff = 0.1/wp   # τ_shock = 1/ω₀
    
    dNLX_dt = np.gradient(NLX, dt, edge_order=2)
    dNLY_dt = np.gradient(NLY, dt, edge_order=2)
    
    dX_dz = -1j*a * (NLX + shock_coeff * dNLX_dt)
    dY_dz = -1j*a * (NLY + shock_coeff * dNLY_dt)
    
    # Flatten and return real and imaginary parts of the derivatives
    return np.hstack([dX_dz.real, dX_dz.imag, dY_dz.real, dY_dz.imag])

# Integration range
z_span = (0, 500e-6)  # Range of z (start, end) (0.5mm crystal thickness)
z_eval = np.linspace(*z_span, 1001)  # Points at which to evaluate
dz = z_eval[1] - z_eval[0]


'''############################################################################
    NEW PROPAGATION CODE 
    (With Spatial Dispersion + Optical Shock [in function])
############################################################################'''

beta2 = 84.868e-30 / 1e-3   # From refractiveindex.info

phase = np.exp(-1j * 0.5 * beta2 * (w - wp)**2 * dz)

def propagate(Y0_in, tau_in):
    
    # Recompute initial conditions with new parameters
    def Y_t_local(t):
        return Y0_in * np.cos(B) * np.exp(-t**2 / (2 * tau_in**2)) * (np.exp(1j * wp * t) + s*np.exp(1j * ws * t))

    def X_t_local(t):
        return Y0_in * np.sin(B) * np.exp(1j * phi) * np.exp(-t**2 / (2 * tau_in**2)) * np.exp(1j * wp * t)
   
    X = X_t_local(t_array)
    Y = Y_t_local(t_array)
   
    Xt_values = np.zeros((Nt, len(z_eval)), dtype=complex)
    Yt_values = np.zeros((Nt, len(z_eval)), dtype=complex)
   
    # store initial condition (z = 0)
    Xt_values[:, 0] = X
    Yt_values[:, 0] = Y
    
    # Main propagation loop over z
    for i in range(len(z_eval) - 1):
        
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
        k2 = coupled_equations(0, state + 0.5 * dz * k1)
        
        # k3: another midpoint slope using k2
        k3 = coupled_equations(0, state + 0.5 * dz * k2)
        
        # k4: slope at end of step using k3
        k4 = coupled_equations(0, state + dz * k3)
        
        # Combine slopes to get next state (RK4 formula)
        solution = state + (dz / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
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
                phase = exp(i * (k(ω) - k_p) * dz)
                
            Takes X and Y field values, transforms into frequency domaain, 
            imposes dispersion, and transforms back to the time domain
        '''
        # Transform fields to frequency domain
        Xw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(X),axis=0))
        Yw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Y),axis=0))
        
        # Apply dispersion phase shift to each frequency component
        Xw *= phase
        Yw *= phase
        
        # Transform back to time domain
        X = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Xw),axis=0))
        Y = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Yw),axis=0))
        
        Xt_values[:, i+1] = X
        Yt_values[:, i+1] = Y
        
    return Xt_values, Yt_values

Xt_out, Yt_out = propagate(Y0, tau)

#%%

'''
Pulse intensity vs time plots

X axis: time (fs)
Y axis: |field|² (intensity)

Shows how the Gaussian pulse envelope changes shape as it propagates. 
Y is the dominant (pump-like) polarization. X starts tiny (flat) and then may 
grow. The four plots show pump depletion and signal growth in time.
'''
plt.figure()
plt.plot(t_array*1e15,np.abs(Yt_out[:,0])**2, label='Y(t) at 0 µm') # [:,0] means first element of every row (start)
plt.plot(t_array*1e15,np.abs(Yt_out[:,-1])**2, label='Y(t) at 500 µm') # [:,-1] means last element of every row (end)
plt.plot(t_array*1e15,np.abs(Xt_out[:,0])**2, label='X(t) at 0 µm')
plt.plot(t_array*1e15,np.abs(Xt_out[:,-1])**2, label='X(t) at 500 µm')
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
angs = np.arctan2(np.abs(Xt_out[:,-1]),np.abs(Yt_out[:,-1]))
plt.plot(t_array*1e15,np.real(angs)*180/np.pi)
plt.xlabel('Time (fs)')
plt.ylabel('Polarization Angle (degrees)')

#%% If you want to plot 3D plot of the data; I recommend making these matrices smaller
# plt.figure()
# plt.pcolormesh(z_values*1e6,w/wp,np.log10(np.abs(Yw)**2),cmap='jet')
# plt.colorbar()

#%%
'''
Signal growth vs crystal length (log scale)

X axis: z (µm)
Y axis: normalized intensity (log scale)

Works in the frequency domain. Finds the 600 nm signal bin (sigpos), sums ±10 
bins around it to get Xwarr and Ywarr, then plots three curves:
    |Xwarr|² —> the cross-polarized (rotated) signal component
    |Ywarr|² —> the co-polarized (original) signal component
    |Xwarr|² + |Ywarr|² —> total signal power
All normalized to the total initial signal power.
'''

if np.any(np.isnan(Xt_out)) or np.any(np.isnan(Yt_out)):
    print("NaNs detected before FFT — aborting signal growth plot")

# Transform fields to frequency domain
Xw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Xt_out),axis=0))
Yw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Yt_out),axis=0))

plt.figure()
sigpos = np.argmin(np.abs(w-2*np.pi*cc/600e-9))
# Xwarr = Xw[sigpos,:]
# Ywarr = Yw[sigpos,:]
Xwarr = np.sum(Xw[sigpos-10:sigpos+10,:],axis=0)
Ywarr = np.sum(Yw[sigpos-10:sigpos+10,:],axis=0)
Norm = np.abs(Xwarr[0])**2 + np.abs(Ywarr[0])**2
plt.plot(z_eval*1e6,np.abs(Xwarr)**2/(np.abs(Xwarr[0])**2 + np.abs(Ywarr[0])**2), label='Rotated signal component (Xwarr)')
plt.plot(z_eval*1e6,np.abs(Ywarr)**2/(np.abs(Xwarr[0])**2 + np.abs(Ywarr[0])**2), label='Original signal component (Ywarr)')
plt.plot(z_eval*1e6,(np.abs(Xwarr)**2 + np.abs(Ywarr)**2)/(np.abs(Xwarr[0])**2 + np.abs(Ywarr[0])**2), label='Total signal power')
plt.yscale('log')
plt.ylim((0.9,1e4))
plt.xlabel('z (µm)')
plt.ylabel('Normalized Intensity')
plt.legend()

#%%
'''
Polarization angle and ellipticity vs z

X axis: z (µm)
Y axis: angle (degrees)

Also uses Xwarr and Ywarr, the 600nm spectral slice. The two curves, anger and 
ellip, tell you if the output is linearly polarized (ellip ≈ 0° or 180°) or 
elliptical.
'''
plt.figure()
anger = np.arctan2(np.abs(Xwarr),np.abs(Ywarr))*180/np.pi
ellip = (np.unwrap(np.angle(Ywarr)) - np.unwrap(np.angle(Xwarr)))*180/np.pi
# anger = np.arctan(np.abs(Xwarr)**2/np.abs(Ywarr)**2)*180/np.pi

# SOP = -(np.abs(Xwarr)**2 - np.abs(Ywarr)**2)/((np.abs(Xwarr)**2 + np.abs(Ywarr)**2))
plt.plot(z_eval*1e6,anger, label='Polarization rotation angle (anger)')
plt.plot(z_eval*1e6,ellip, label='Phase difference (ellip)')
plt.xlabel('z (µm)')
plt.ylabel('Polarization Angle and Ellipticity')
plt.legend()

#%% Chirp
# Plot real part of Y field to show chirp
plt.figure()
plt.plot(t_array*1e15, np.real(Yt_out[:,0]), label='Y(t) at 0 µm', alpha=0.7)
plt.plot(t_array*1e15, np.real(Yt_out[:,-1]), label='Y(t) at 500 µm', alpha=0.7)
plt.xlabel('Time (fs)')
plt.ylabel('Real field amplitude (arb. units)')
plt.legend()

#%% 
''' PULSE DURATION/INTENSITY ANALYSIS'''
#%% Pulse Duration Scan
fwhm_values = np.linspace(50e-15, 400e-15, 20)
output_fwhms = []

iteration = 0

for fwhm in fwhm_values:
    tau_in = fwhm/(2*(2*np.log(2))**0.5)
    Xt_out, Yt_out = propagate(Y0, tau_in)
    
    if Xt_out is None:
        output_fwhms.append(np.nan)
        iteration += 1
        print(f"Ran iteration {iteration} (blew up, skipped)")
        continue
  
    intensity = np.abs(Xt_out[:,-1])**2
    if np.max(intensity) == 0 or np.any(np.isnan(intensity)):
        output_fwhms.append(np.nan)
        iteration += 1
        print(f"Ran iteration {iteration} (zero/nan intensity, skipped)")
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
plt.plot(fwhm_values*1e15, fwhm_values*1e15/np.sqrt(3), 'k--', label='1/√3 theoretical limit')
plt.axvline(x=FWHM*1e15, ymin=0, color='Red', label='120 fs FWHM')
plt.xlabel('Input FWHM (fs)')
plt.ylabel('Output X FWHM (fs)')
plt.legend()

#%% Intensity Scan
Y0_values = np.linspace(1e9, 8e9, 20)  # vary amplitude
output_fwhms = []
input_fwhm = FWHM  # fixed at 120 fs

iteration = 0

for Y0_scan in Y0_values:
    Xt_out, Yt_out = propagate(Y0_scan, tau)
    
    intensity = np.abs(Xt_out[:,-1])**2
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
plt.plot(Y0_values, output_fwhms*1e15, label='Output X FWHM')
plt.axhline(y=input_fwhm*1e15/np.sqrt(3), color='k', linestyle='--', label='1/√3 theoretical limit')
plt.xlabel('Y0 (field amplitude)')
plt.ylabel('Output X FWHM (fs)')
plt.legend()

min_index = np.argmin(output_fwhms*1e15)
x_min = Y0_values[min_index]
y_min = (output_fwhms*1e15)[min_index]
print(f"Minima point: ({x_min}, {y_min})")
#%% Pulse evolution
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