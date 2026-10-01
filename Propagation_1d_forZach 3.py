# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:46:00 2022

@author: drtjh
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft, ifft, fftshift, ifftshift,fft2,ifft2

#%% Constants
c = 2.9979e8
mu0 = 4*np.pi*1e-7
eps0 = 1/(c**2*mu0)
pi = np.pi
Z0 = np.sqrt(mu0/eps0)
#%% Setting up grid in time and frequency
Nt = 5001
t = np.linspace(-500e-15,500e-15,Nt)
dt = t[1]-t[0]
dw = 2*np.pi/(Nt*dt)
w = np.arange(-(Nt-1)*dw/2,Nt*dw/2,dw)

#%% Setting up the propagation grid
Lz = 1e-3
# Nz = 1001
# z = np.linspace(0,Lz,Nz)
# dz = z[1]-z[0]
dz = 0.5e-6
z = np.arange(0,Lz,dz)
Nz = len(z)
#%% Setting up the field
tFWHM = 100e-15 #pulse duration FWHM, s
lambda0 = 800e-9 #central wavelength, m
I0 = 50e15 #peak intensity, W/m^2

w0 = 2*np.pi*c/lambda0
w0pos = np.argmin(np.abs(w-w0))
tau = tFWHM/(2*np.sqrt(np.log(2)))
E0 = np.sqrt(2*377*I0)

Et = E0*np.exp(-t**2/(2*tau**2))*np.exp(1j*w0*t)
Et0 = Et/np.max(np.abs(Et))*E0
Ew = dt/np.sqrt(2*pi)*fftshift(fft(ifftshift(Et)))
Ew0 = Ew

#%% Index of material

x = 2*np.pi*c/w*1e6 #index often given in microns
# nsq = (1+0.6961663/(1-(0.0684043/x)**2)+0.4079426/(1-(0.1162414/x)**2)+0.8974794/(1-(9.896161/x)**2)) #fused silica
nsq = 1+1.4313493/(1-(0.0726631/x)**2)+0.65054713/(1-(0.1193242/x)**2)+5.3414021/(1-(18.028251/x)**2) #sapphire no
# nsq = 1+1.5039759/(1-(0.0740288/x)**2)+0.55069141/(1-(0.1216529/x)**2)+6.5927379/(1-(20.072248/x)**2) #sapphire ne
# nsq = 3.778790+0.070479/(x**2-0.045731)-0.009701*x**2 #YVO4 no
#nsq = 4.607200+0.108087/(x**2-0.052495)-0.014305*x**2 #YVO4 ne
#nsq = 1+0.28604141+1.07044083/(1-1.00585997e-2/x**2)+1.10202242/(1-100/x**2) #quartz no
#nsq = 1+0.28851804+1.09509924/(1-1.02101864e-2/x**2)+1.15662475/(1-100/x**2) #quartz ne


nsq[nsq<0.5]=0.5
nsq[nsq>6]=6
n = np.sqrt(nsq)
# n(L) = A + B/L**2 + C*L**2 + D/L**4
# A10 = 1.43206155
# B10 = 4111.26
# C10 = -3.4693e-9
# D10 = 1
# n = A10 + B10/(2*pi*c/w*1e9)**2 + C10*(2*pi*c/w*1e9)**2 + D10/(2*pi*c/w*1e9)**4

# n[n<0.5] = 0.5
# n[n>3] = 3
# n[np.isnan(n)] = 0.5


# n = np.sqrt(nsq)

ng = n + w*np.gradient(n,dw)
ng0 = ng[w0pos]

# n2 = 2.2e-20 #nonlinear index of material, in m^2/W
n2 = 3.2e-20 #nonlinear Kerr coefficient for sapphire
# n2 = 1.5e-19 #nonlinear Kerr coefficient for YVO4
# n2 = 2.3e-20 #nonlinear Kerr coefficient for quartz
n4 = 0;

# n2 = 2.45e-20 #nonlinear index of refraction, in m^2/W
# n4 = 7.1e-35
# n6 = -2.5e-50*0
#%% The business
n0 = n[w0pos]
k0 = w0*n0/c
X3 = 4/3*eps0*c*n0**2*n2
# X3 = 4/3*n2*np.sqrt(eps0/mu0)*n0**2
wL = w0/4 #lower frequency cutoff
wH = w0*4 #higher frequency cutoff; unbounded leads to runaway

#the derivative function
def dEbdz(Ewold):
    Etold = np.sqrt(2*pi)/dt*fftshift(ifft(ifftshift(Ewold)))
    PNL3told = 6/8*np.abs(Etold)**2*Etold
    PNL3wold = dt/np.sqrt(2*pi)*fftshift(fft(ifftshift(PNL3told)))
    PNLwold = eps0*X3*PNL3wold
    dEwbdz_new = -1j*(n - ng0)*(w/c)*Ewold - 1j*mu0*c*w/(2*n)*PNLwold
    return dEwbdz_new*np.heaviside((w-wL),1)*(1-np.heaviside((w-wH),1))

def dAbdz(Atold):
    const = 3*X3/(8*(-1j)*k0*c**2)
    At3 = np.abs(Atold)**2*Atold
    spm = const*(-w0**2*At3) # return all 3 commented
    grad1 = np.gradient(At3,dt)
    steep = const*(-2*(-1j)*w0*grad1) # multiply wwith 0 when returning
    grad2 = np.gradient(grad1,dt)
    fwm = const*grad2 # multiply wwith 0 when returning
    
    return spm + steep + fwm
    # return +1j*n2*k0*1/(2*Z0)*(Asq*Atold + 0*1j/w0*grad)
    # return -1j*n2*w0/c*1/(2*Z0)*Asq*Atold

Ewold = Ew0
Atold = Et0/2
#propagation
for countZ in range(0,Nz):
    #RK4 for FME
    k1 = dz*dEbdz(Ewold)
    k2 = dz*dEbdz(Ewold+k1/2)
    k3 = dz*dEbdz(Ewold+k2/2)
    k4 = dz*dEbdz(Ewold+k3)

    Ewnew = Ewold + 1/6*(k1+2*k2+2*k3+k4)
    Ewold = Ewnew
    
    #RK4 for NLSE
    k1 = dz*dAbdz(Atold)
    k2 = dz*dAbdz(Atold+k1/2)
    k3 = dz*dAbdz(Atold+k2/2)
    k4 = dz*dAbdz(Atold+k3)

    Atnew = Atold + 1/6*(k1+2*k2+2*k3+k4)
    Awnew = dt/np.sqrt(2*pi)*fftshift(fft(ifftshift(Atnew)))*np.exp(-1j*w/c*(n-ng0)*dz) # dispersion part
    Atold = np.sqrt(2*pi)/dt*fftshift(ifft(ifftshift(Awnew)))
    
    print(f"Iteration {countZ}/{Nz}")
    
    
#%%
plt.figure()
plt.plot(t*1e15,np.real(Et))
Etnew = np.sqrt(2*pi)/dt*fftshift(ifft(ifftshift(Ewnew)))
plt.plot(t*1e15,np.real(Etnew))
plt.plot(t*1e15,np.real(Atold),':')
plt.xlabel('Time (fs)')
plt.ylabel('Field (V/m)')

plt.figure()
waveL = 2*np.pi*c/w #convert to wavelengths, as seen by our spectrometer
plt.semilogy(waveL*1e9,np.abs(Ew0)**2,label='Input')
plt.semilogy(waveL*1e9,np.abs(Ewnew)**2,label='FME')
plt.semilogy(waveL*1e9,np.abs(2*Awnew)**2,label='NLSE')
plt.xlim((300,1200))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (arb u)')
plt.legend()

#%% Rewritten varying pulse duration (UNFINISHED)
tFWHM_values = [50e-15, 100e-15, 200e-15, 400e-15]

fig, axes = plt.subplots(1, len(tFWHM_values), figsize=(16, 5), sharey=False)

for ax, tFWHM in zip(axes, tFWHM_values): # Parallel iteration
    time_window = max(500e-15, tFWHM * 10) # Take the bigger time window
    Nt = int(5000 * time_window / 500e-15) # Number of points to take, minimum 5000
    
    if Nt % 2 != 1: # Check if the number of points is 
        Nt += 1
    
    time = np.linspace(-time_window, time_window, Nt) # Time array for the initial conditions
    dt = t[1] - t[0] # small step in time
    df = 1/(Nt*dt) # small step in frequency
    dw = 2*np.pi*df # small step in angular frequency
    
    w = np.arange(-(Nt-1)*dw/2, Nt*dw/2, dw) # Return angular frequency values for the entire range
    w0 = 2*np.pi*c*lambda0 # initial angular frequency of the field
    w0pos = np.argmin(np.abs(w-w0)) # Return indices of the minimum values angular frequency, position of the initial angular frequency from the initial field
    
    x = 2*np.pi*c/w*1e6 # position (x) value
    
    ## COPIED CODE FROM INDEX SECTION WITH NEW X VALUE
    nsq = 1+1.4313493/(1-(0.0726631/x)**2)+0.65054713/(1-(0.1193242/x)**2)+5.3414021/(1-(18.028251/x)**2)
    
    nsq[nsq<0.5]=0.5
    nsq[nsq>6]=6
    n = np.sqrt(nsq)

    ng = n + w*np.gradient(n,dw)
    ng0 = ng[w0pos]
    n2 = 3.2e-20
    ##################################################
    
    ## COPIED CODE FROM THE BUSINESS SECTION TO USE UPDATED VARIABLES
    n0 = n[w0pos]
    k0 = w0*n0/c
    X3 = 4/3*eps0*c*(n0**2)*n2
    wL = w0/4 # lower frequency cutoff 
    wH = w0*4 # higher frequency cutoff; unbounded leads to runaway
    
    #the derivative function
    def dEbdz(Ewold):
        Etold = np.sqrt(2*pi)/dt*fftshift(ifft(ifftshift(Ewold)))
        PNL3told = 6/8*np.abs(Etold)**2*Etold
        PNL3wold = dt/np.sqrt(2*pi)*fftshift(fft(ifftshift(PNL3told)))
        PNLwold = eps0*X3*PNL3wold
        dEwbdz_new = -1j*(n - ng0)*(w/c)*Ewold - 1j*mu0*c*w/(2*n)*PNLwold
        return dEwbdz_new*np.heaviside((w-wL),1)*(1-np.heaviside((w-wH),1))

    def dAbdz(Atold):
        const = 3*X3/(8*(-1j)*k0*c**2)
        At3 = np.abs(Atold)**2*Atold
        spm = const*(-w0**2*At3) # return all 3 commented
        grad1 = np.gradient(At3,dt)
        steep = const*(-2*(-1j)*w0*grad1) # multiply wwith 0 when returning
        grad2 = np.gradient(grad1,dt)
        fwm = const*grad2 # multiply wwith 0 when returning
        
        return spm + steep + fwm

        Ewold = Ew0
        Atold = Et0/2
        #propagation
        for countZ in range(0,Nz):
            #RK4 for FME
            k1 = dz*dEbdz(Ewold)
            k2 = dz*dEbdz(Ewold+k1/2)
            k3 = dz*dEbdz(Ewold+k2/2)
            k4 = dz*dEbdz(Ewold+k3)
        
            Ewnew = Ewold + 1/6*(k1+2*k2+2*k3+k4)
            Ewold = Ewnew
            
            #RK4 for NLSE
            k1 = dz*dAbdz(Atold)
            k2 = dz*dAbdz(Atold+k1/2)
            k3 = dz*dAbdz(Atold+k2/2)
            k4 = dz*dAbdz(Atold+k3)
        
            Atnew = Atold + 1/6*(k1+2*k2+2*k3+k4)
            Awnew = dt/np.sqrt(2*pi)*fftshift(fft(ifftshift(Atnew)))*np.exp(-1j*w/c*(n-ng0)*dz) # dispersion part
            Atold = np.sqrt(2*pi)/dt*fftshift(ifft(ifftshift(Awnew)))
        #############################################################
        
#%% Varying pulse duration - self contained
tFWHM_values = [50e-15, 100e-15, 200e-15, 400e-15]

fig, axes = plt.subplots(1, len(tFWHM_values), figsize=(16, 5), sharey=False)

for ax, tFWHM in zip(axes, tFWHM_values):
    t_window = max(500e-15, tFWHM * 10)
    Nt_local = int(5001 * t_window / 500e-15)
    Nt_local = Nt_local if Nt_local % 2 == 1 else Nt_local + 1

    t_local = np.linspace(-t_window, t_window, Nt_local)
    dt_local = t_local[1] - t_local[0]
    dw_local = 2*np.pi/(Nt_local*dt_local)
    w_local = np.arange(-(Nt_local-1)*dw_local/2, Nt_local*dw_local/2, dw_local)
    w0 = 2*np.pi*c/lambda0
    w0pos_local = np.argmin(np.abs(w_local - w0))

    x_local = 2*np.pi*c/w_local*1e6
    nsq_local = 1+1.4313493/(1-(0.0726631/x_local)**2)+0.65054713/(1-(0.1193242/x_local)**2)+5.3414021/(1-(18.028251/x_local)**2)
    nsq_local[nsq_local<0.5] = 0.5
    nsq_local[nsq_local>6] = 6
    n_local = np.sqrt(nsq_local)
    ng_local = n_local + w_local*np.gradient(n_local, dw_local)
    ng0_local = ng_local[w0pos_local]
    n0_local = n_local[w0pos_local]
    k0_local = w0*n0_local/c
    X3_local = 4/3*eps0*c*n0_local**2*n2
    wL_local = w0/4
    wH_local = w0*4

    def dEbdz_local(Ewold):
        Etold = np.sqrt(2*pi)/dt_local*fftshift(ifft(ifftshift(Ewold)))
        PNL3told = 6/8*np.abs(Etold)**2*Etold
        PNL3wold = dt_local/np.sqrt(2*pi)*fftshift(fft(ifftshift(PNL3told)))
        PNLwold = eps0*X3_local*PNL3wold
        dEwbdz_new = -1j*(n_local - ng0_local)*(w_local/c)*Ewold - 1j*mu0*c*w_local/(2*n_local)*PNLwold
        return dEwbdz_new*np.heaviside((w_local-wL_local),1)*(1-np.heaviside((w_local-wH_local),1))

    def dAbdz_local(Atold):
        const = 3*X3_local/(8*(-1j)*k0_local*c**2)
        At3 = np.abs(Atold)**2*Atold
        spm = const*(-w0**2*At3)
        grad1 = np.gradient(At3, dt_local)
        steep = const*(-2*(-1j)*w0*grad1)
        grad2 = np.gradient(grad1, dt_local)
        fwm = const*grad2
        return spm + steep + fwm

    tau_local = tFWHM/(2*np.sqrt(np.log(2)))
    E0_local = np.sqrt(2*377*I0)
    Et_local = E0_local*np.exp(-t_local**2/(2*tau_local**2))*np.exp(1j*w0*t_local)
    Et0_local = Et_local/np.max(np.abs(Et_local))*E0_local
    Ew_local = dt_local/np.sqrt(2*pi)*fftshift(fft(ifftshift(Et_local)))
    Ew0_local = Ew_local.copy()

    Ewold = Ew0_local.copy()
    Atold = Et0_local/2

    for countZ in range(0, Nz):
        k1 = dz*dEbdz_local(Ewold)
        k2 = dz*dEbdz_local(Ewold+k1/2)
        k3 = dz*dEbdz_local(Ewold+k2/2)
        k4 = dz*dEbdz_local(Ewold+k3)
        Ewnew = Ewold + 1/6*(k1+2*k2+2*k3+k4)
        Ewold = Ewnew

        k1 = dz*dAbdz_local(Atold)
        k2 = dz*dAbdz_local(Atold+k1/2)
        k3 = dz*dAbdz_local(Atold+k2/2)
        k4 = dz*dAbdz_local(Atold+k3)
        Atnew = Atold + 1/6*(k1+2*k2+2*k3+k4)
        Awnew = dt_local/np.sqrt(2*pi)*fftshift(fft(ifftshift(Atnew)))*np.exp(-1j*w_local/c*(n_local-ng0_local)*dz)
        Atold = np.sqrt(2*pi)/dt_local*fftshift(ifft(ifftshift(Awnew)))

    waveL_local = 2*np.pi*c/w_local
    ax.semilogy(waveL_local*1e9, np.abs(Ew0_local)**2, label='Input')
    ax.semilogy(waveL_local*1e9, np.abs(Ewnew)**2, label='FME')
    ax.semilogy(waveL_local*1e9, np.abs(2*Awnew)**2, label='NLSE')
    ax.set_xlim((300, 1200))
    ax.set_xlabel('Wavelength (nm)')
    ax.set_title(f'tFWHM = {tFWHM*1e15:.0f} fs')
    print(f'tFWHM = {tFWHM*1e15:.0f} fs done, Nt = {Nt_local}')

axes[0].set_ylabel('Intensity (arb u)')
axes[0].legend()
plt.tight_layout()
plt.show()

#%% Time window sanity check
window_values = [500e-15, 1000e-15, 2000e-15, 4000e-15]
tFWHM = 100e-15

fig, axes = plt.subplots(1, len(window_values), figsize=(16, 5), sharey=False)

for ax, t_window in zip(axes, window_values):
    Nt_local = int(5001 * t_window / 500e-15)
    Nt_local = Nt_local if Nt_local % 2 == 1 else Nt_local + 1

    t_local = np.linspace(-t_window, t_window, Nt_local)
    dt_local = t_local[1] - t_local[0]
    dw_local = 2*np.pi/(Nt_local*dt_local)
    w_local = np.arange(-(Nt_local-1)*dw_local/2, Nt_local*dw_local/2, dw_local)
    w0 = 2*np.pi*c/lambda0
    w0pos_local = np.argmin(np.abs(w_local - w0))

    x_local = 2*np.pi*c/w_local*1e6
    nsq_local = 1+1.4313493/(1-(0.0726631/x_local)**2)+0.65054713/(1-(0.1193242/x_local)**2)+5.3414021/(1-(18.028251/x_local)**2)
    nsq_local[nsq_local<0.5] = 0.5
    nsq_local[nsq_local>6] = 6
    n_local = np.sqrt(nsq_local)
    ng_local = n_local + w_local*np.gradient(n_local, dw_local)
    ng0_local = ng_local[w0pos_local]
    n0_local = n_local[w0pos_local]
    k0_local = w0*n0_local/c
    X3_local = 4/3*eps0*c*n0_local**2*n2
    wL_local = w0/4
    wH_local = w0*4

    def dEbdz_local(Ewold):
        Etold = np.sqrt(2*pi)/dt_local*fftshift(ifft(ifftshift(Ewold)))
        PNL3told = 6/8*np.abs(Etold)**2*Etold
        PNL3wold = dt_local/np.sqrt(2*pi)*fftshift(fft(ifftshift(PNL3told)))
        PNLwold = eps0*X3_local*PNL3wold
        dEwbdz_new = -1j*(n_local - ng0_local)*(w_local/c)*Ewold - 1j*mu0*c*w_local/(2*n_local)*PNLwold
        return dEwbdz_new*np.heaviside((w_local-wL_local),1)*(1-np.heaviside((w_local-wH_local),1))

    def dAbdz_local(Atold):
        const = 3*X3_local/(8*(-1j)*k0_local*c**2)
        At3 = np.abs(Atold)**2*Atold
        spm = const*(-w0**2*At3)
        grad1 = np.gradient(At3, dt_local)
        steep = const*(-2*(-1j)*w0*grad1)
        grad2 = np.gradient(grad1, dt_local)
        fwm = const*grad2
        return spm + steep + fwm

    tau_local = tFWHM/(2*np.sqrt(np.log(2)))
    E0_local = np.sqrt(2*377*I0)
    Et_local = E0_local*np.exp(-t_local**2/(2*tau_local**2))*np.exp(1j*w0*t_local)
    Et0_local = Et_local/np.max(np.abs(Et_local))*E0_local
    Ew_local = dt_local/np.sqrt(2*pi)*fftshift(fft(ifftshift(Et_local)))
    Ew0_local = Ew_local.copy()

    Ewold = Ew0_local.copy()
    Atold = Et0_local/2

    for countZ in range(0, Nz):
        k1 = dz*dEbdz_local(Ewold)
        k2 = dz*dEbdz_local(Ewold+k1/2)
        k3 = dz*dEbdz_local(Ewold+k2/2)
        k4 = dz*dEbdz_local(Ewold+k3)
        Ewnew = Ewold + 1/6*(k1+2*k2+2*k3+k4)
        Ewold = Ewnew

        k1 = dz*dAbdz_local(Atold)
        k2 = dz*dAbdz_local(Atold+k1/2)
        k3 = dz*dAbdz_local(Atold+k2/2)
        k4 = dz*dAbdz_local(Atold+k3)
        Atnew = Atold + 1/6*(k1+2*k2+2*k3+k4)
        Awnew = dt_local/np.sqrt(2*pi)*fftshift(fft(ifftshift(Atnew)))*np.exp(-1j*w_local/c*(n_local-ng0_local)*dz)
        Atold = np.sqrt(2*pi)/dt_local*fftshift(ifft(ifftshift(Awnew)))

    waveL_local = 2*np.pi*c/w_local
    ax.semilogy(waveL_local*1e9, np.abs(Ew0_local)**2, label='Input')
    ax.semilogy(waveL_local*1e9, np.abs(Ewnew)**2, label='FME')
    ax.semilogy(waveL_local*1e9, np.abs(2*Awnew)**2, label='NLSE')
    ax.set_xlim((300, 1200))
    ax.set_xlabel('Wavelength (nm)')
    ax.set_title(f'Window = {t_window*1e15:.0f} fs')
    print(f'Window = {t_window*1e15:.0f} fs done, Nt = {Nt_local}')

axes[0].set_ylabel('Intensity (arb u)')
axes[0].legend()
plt.tight_layout()
plt.show()