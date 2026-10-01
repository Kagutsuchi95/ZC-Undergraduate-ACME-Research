# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:27:21 2025

@author: TJ Hammond
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
plt.close('all')

#%%
cc = 3e8
# Parameters
a = 2*np.pi/800e-9*3/8*1/1.73     # Coupling constant k*3/8/n0, where n0 is the index of refraction
C = 3.2e-22 # Nonlinear X^(3) susceptibility tensor X^(3)_xxxx
P = C/1.85  # Cross-phase modulation coefficient, X^(3)_xxyy, relative to X^(3)_xxxx
Y0 = 1.0*6e9    # Amplitude of initial Gaussian envelope
tau = 400e-15   # Width of the Gaussian envelope
wp = 2*np.pi*cc/800e-9    # Frequency wp (pump)
ws = 2*np.pi*cc/600e-9    # Frequency ws (signal)
B = 0.05  # Angle for X(t) initial condition
s = 1e-7 # Relative initial field amplitude for signal
phi = 0.0  # Phase shift for X(t)

# Define the time array
Nt = 10001
t_array = np.linspace(-2000e-15, 2000e-15, Nt)  # Time array for the initial conditions
dt = t_array[1]-t_array[0]

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
    dX_dz = 1j*a * (C*X_abs2 + 2*P*Y_abs2) * X + 1j*a*P*Y**2*np.conj(X) # Equation 2a
    dY_dz = 1j*a * (C*Y_abs2 + 2*P*X_abs2) * Y + 1j*a*P*X**2*np.conj(Y) # Equation 2b
    '''
        First term = self-phase modulation
        Second term = cross-phase moddulation
        Third term = degenerate four-wave mixing
    '''

    # Flatten and return real and imaginary parts of the derivatives
    return np.hstack([dX_dz.real, dX_dz.imag, dY_dz.real, dY_dz.imag])

# Integration range
z_span = (0, 500e-6)  # Range of z (start, end) (0.5mm crystal thickness)
z_eval = np.linspace(*z_span, 1001)  # Points at which to evaluate
dz = z_eval[1] - z_eval[0]

'''
OLD INTEGRATION

# Flatten the initial state
initial_state = np.hstack([
    X0_array.real, X0_array.imag, Y0_array.real, Y0_array.imag
])

# Solve the system
solution = solve_ivp(
    coupled_equations,
    z_span,
    initial_state,
    t_eval=z_eval,
    method="RK45"
)

# Extract solutions and reshape
z_values = solution.t
sol = solution.y
Xt_values = sol[:Nt, :] + 1j * sol[Nt:2*Nt, :]
Yt_values = sol[2*Nt:3*Nt, :] + 1j * sol[3*Nt:, :]

# Convert time-domain fields, Xt and Yt, into frequency domain to look at specific wavelengths
Xw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Xt_values),axis=0))
Yw = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Yt_values),axis=0))
'''


'''
NEW INTEGRATION (With Dispersion)

Finding Dispersion Phase From GVD

Uses taylor expanded formula for the dispersion phase, effect from beta2 is
dominant so others are ignoredd

'''

w = 2*np.pi * np.fft.fftfreq(Nt, d=dt)
w = np.fft.fftshift(w)

beta2 = 84.868e-30 / 1e-3   # From refractiveindex.info

phase = np.exp(1j * 0.5 * beta2 * (w - wp)**2 * dz)
# phase = np.exp(1j * 0.5 * 500 * beta2 * (w - wp)**2 * dz) # Amplified dispersion

'''
NEW PROPAGATION CODE

Solves the problem of a solution without dispersion, improves integration via 
RK4 (more accurate, tested at bottom of code)
'''
# Initialize fields at z = 0
X = X0_array.copy()
Y = Y0_array.copy()

Xt_values = np.zeros((Nt, len(z_eval)), dtype=complex)
Yt_values = np.zeros((Nt, len(z_eval)), dtype=complex)

# store initial condition (z = 0)
Xt_values[:, 0] = X
Yt_values[:, 0] = Y

# Main propagation loop over z
for i in range(len(z_eval) - 1):

    '''
    (1) NONLINEAR STEP (time domain)
    
    Reuses existing coupled_equations() function, which computes dX/dz and 
    dY/dz from nonlinear effects (like in the paper) but with RK4 instead
    of solve_ivp()
    '''
    # Pack complex fields into real-valued state vector (because function expects real + imaginary parts separately)
    state = np.hstack([X.real, X.imag, Y.real, Y.imag])

    '''---- RK4 integration ----'''
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

    '''
    (2) DISPERSION STEP (frequency domain)
    
    This is where the derived phase factor is applied: 
        phase = exp(i * (k(ω) - k_p) * dz)
        
    Takes X and Y field values, transforms into frequency domaain, imposes 
    dispersion, and transforms back to the time domain
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

#%%

'''
Pulse intensity vs time plots

X axis: time (fs)
Y axis: |field|² (intensity)

Shows how the Gaussian pulse envelope changes shape as it propagates. 
Y is the dominant (pump-like) polarization. X starts tiny (flat) and then may grow. 
The four plots show pump depletion and signal growth in time.
'''
plt.figure()
plt.plot(t_array*1e15,np.abs(Yt_values[:,0])**2, label='Y(t) at 0 µm') # [:,0] means first element of every row (start)
plt.plot(t_array*1e15,np.abs(Yt_values[:,-1])**2, label='Y(t) at 500 µm') # [:,-1] means last element of every row (end)
plt.plot(t_array*1e15,np.abs(Xt_values[:,0])**2, label='X(t) at 0 µm')
plt.plot(t_array*1e15,np.abs(Xt_values[:,-1])**2, label='X(t) at 500 µm')
# plt.title("Dispersion amplified by 500x")
plt.xlabel('Time (fs)')
plt.ylabel('Pulse Intensity (arb. units)')
plt.legend()

'''
Polarization angle vs time plot

X axis: time (fs)
Y axis: angle (degrees)

arctan2(|X|, |Y|) at the exit of the crystal (z = 500 µm), evaluated at every time point.
0° means purely Y-polarized (no rotation); 45° means equal X and Y; 90° means fully X-polarized.
Shows how much the polarization rotates across the pulse.
'''
plt.figure()
angs = np.arctan2(np.abs(Xt_values[:,-1]),np.abs(Yt_values[:,-1]))
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

Works in the frequency domain. Finds the 600 nm signal bin (sigpos), sums ±10 bins around it to get
Xwarr and Ywarr, then plots three curves:
    |Xwarr|² —> the cross-polarized (rotated) signal component
    |Ywarr|² —> the co-polarized (original) signal component
    |Xwarr|² + |Ywarr|² —> total signal power
All normalized to the total initial signal power.
'''
print("Xt_values:", Xt_values.shape)
print("Yt_values:", Yt_values.shape)
print("Xw before slicing:", Xw.shape)
plt.figure()
sigpos = np.argmin(np.abs(w-2*np.pi*cc/600e-9))
# Xwarr = Xw[sigpos,:]
# Ywarr = Yw[sigpos,:]
Xwarr = np.sum(Xw[sigpos-10:sigpos+10,:],axis=0)
Ywarr = np.sum(Yw[sigpos-10:sigpos+10,:],axis=0)
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

Also uses Xwarr and Ywarr, the 600nm spectral slice. The two curves, anger and ellip, 
tell you if the output is linearly polarized (ellip ≈ 0° or 180°) or elliptical.
'''
plt.figure()
anger = np.arctan2(np.abs(Xwarr)**1,np.abs(Ywarr)**1)*180/np.pi
ellip = (np.unwrap(np.angle(Ywarr)) - np.unwrap(np.angle(Xwarr)))*180/np.pi
# anger = np.arctan(np.abs(Xwarr)**2/np.abs(Ywarr)**2)*180/np.pi

# SOP = -(np.abs(Xwarr)**2 - np.abs(Ywarr)**2)/((np.abs(Xwarr)**2 + np.abs(Ywarr)**2))
plt.plot(z_eval*1e6,anger, label='Polarization rotation angle (anger)')
plt.plot(z_eval*1e6,ellip, label='Phase difference (ellip)')
plt.xlabel('z (µm)')
plt.ylabel('Polarization Angle and Ellipticity')
plt.legend()
#%%
print((np.abs(Xwarr[-1])**2 + np.abs(Ywarr[-1])**2)/(np.abs(Xwarr[0])**2 + np.abs(Ywarr[0])**2))

#%% Compare RK4 split step method to RK45 solve_ivp() method (TEST)

'''
ACHIEVED RESULTS:
    
    RK4 convergence error: 7.353627181998753e-09
    RK45 convergence error: 4.25742057094363e-06

CONCLUSION: Manual RK4 is more accurate for this purpose than solve_ivp() RK45
'''

'''
def compare(X1, X2):
    # relative L2 error
    return np.linalg.norm(X1 - X2) / np.linalg.norm(X2)

def run_rk4(dz):
    X = X0_array.copy()
    Y = Y0_array.copy()

    Nz = int(z_span[1] / dz)

    for i in range(Nz):

        # nonlinear step (your RK4 function or Euler version)
        state = np.hstack([X.real, X.imag, Y.real, Y.imag])

        k1 = coupled_equations(0, state)
        k2 = coupled_equations(0, state + 0.5*dz*k1)
        k3 = coupled_equations(0, state + 0.5*dz*k2)
        k4 = coupled_equations(0, state + dz*k3)

        state = state + (dz/6)*(k1 + 2*k2 + 2*k3 + k4)

        X = state[:Nt] + 1j*state[Nt:2*Nt]
        Y = state[2*Nt:3*Nt] + 1j*state[3*Nt:]

    return X

X_rk4_1 = run_rk4(dz)
X_rk4_2 = run_rk4(dz/2)

err_rk4 = compare(X_rk4_2, X_rk4_1)
print("RK4 convergence error:", err_rk4)

def run_rk45(rtol):
    solution = solve_ivp(
        coupled_equations,
        z_span,
        initial_state,
        t_eval=[z_span[1]],  # only final point
        method="RK45",
        rtol=rtol,
        atol=1e-9
    )

    sol = solution.y

    X = sol[:Nt,-1] + 1j*sol[Nt:2*Nt,-1]
    return X

X_rk45_1 = run_rk45(1e-6)
X_rk45_2 = run_rk45(1e-9)

err_rk45 = compare(X_rk45_2, X_rk45_1)
print("RK45 convergence error:", err_rk45)
'''

#%% TESTING DISPERSION
'''
# FFT over ALL z positions
Xw = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(Xt_values, axes=0), axis=0),
        axes=0
     )

plt.figure()

# phase at z = 0
plt.plot(w, np.unwrap(np.angle(Xw[:, 0])), label='z = 0')
    
# phase at final z
plt.plot(w, np.unwrap(np.angle(Xw[:, -1])), label='z = L')

plt.legend()


phi = np.unwrap(np.angle(Xw[:, -1]))
plt.figure()
plt.plot(w, np.gradient(np.gradient(phi, w)))

phi0 = np.unwrap(np.angle(Xw[:, 0]))
phiL = np.unwrap(np.angle(Xw[:, -1]))

plt.figure()
plt.plot(w, phiL - phi0)

fit = np.polyfit(w - wp, phiL - phi0, 2)
print(fit[0])
'''