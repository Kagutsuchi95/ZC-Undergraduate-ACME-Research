# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 20:32:08 2025

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt

c = 2.9979e8
π = np.pi

# n2 = -1/3 * 3.2e-20 #Sapphire
# n2 = -1/3 * 1.5e-19 #YVO4
n2 = -1/3 * 2.3e-20 #Quartz

Ip = 1.5e17 # 1.5e17
λp = 800e-9

def index_o(λμm):
    
    # return (1+1.4313493/(1-(0.0726631/λμm)**2)+0.65054713/(1-(0.1193242/λμm)**2)+5.3414021/(1-(18.028251/λμm)**2))**.5 #Sapphire
    # return (3.778790+0.070479/(λμm**2-0.045731)-0.009701*λμm**2)**.5 #YVO4
    return (1+0.28604141+1.07044083/(1-1.00585997e-2/λμm**2)+1.10202242/(1-100/λμm**2))**.5 #Quartz

def index_e(λμm):
    
    # return (1+1.5039759/(1-(0.0740288/λμm)**2)+0.55069141/(1-(0.1216529/λμm)**2)+6.5927379/(1-(20.072248/λμm)**2))**.5 #Sapphire
    # return (4.607200+0.108087/(λμm**2-0.052495)-0.014305*λμm**2)**.5 #YVO4
    return (1+0.28851804+1.09509924/(1-1.02101864e-2/λμm**2)+1.15662475/(1-100/λμm**2))**.5 #Quartz

def index_e_theta(λμm,θ):
    ne = index_e(λμm)
    no = index_o(λμm)
    return ((np.cos(θ)**2)/(no**2) + (np.sin(θ)**2)/(ne**2))**(-0.5)

DN = -0.0 #this is the birefringence, can be positive or negative, difference between n(e) and n(o) on refractiveindex.info

def Δk(λp,λs,Ip,θ):
    ωp = 2*π*c/λp
    ωs = 2*π*c/λs
    ωi = 2*ωp-ωs
    n_p = index_o(λp*1e6)
    ns = index_e_theta(λs*1e6,θ)
    ni = index_e_theta(2*π*c/ωi*1e6,θ)
    # return 2/c*(n_p*ωp*(1 + 1/3*n2*Ip)) - 1/c*(ns*ωs + ni*ωi) #from Cohen PRL 2009, for orthogonal polarization
    return 1/c*(ns*ωs + ni*ωi + 2*DN*ωp - 2*n_p*ωp) #From Agrawal, eq 10.2.18, for same polarization
#this seems to be correct. So you cannot get perfect phase matching on axis in the normal dispersion regime by tuning the pump power
#but you can in the anomalous dispersion regime
        
ωp = 2*π*c/λp
n_p = index_o(λp*1e6)
kp = ωp/c*n_p
Nω = 1001
ωvec = np.linspace(0,2*(2*π*c/λp),Nω)

def gg(λp,λs,Ip,θ):
    ωp = 2*π*c/λp
    n_p = index_o(λp*1e6)
    alpha = ωp/c*n2*Ip/n_p
    out = alpha**2 - (alpha + (Δk(λp,(2*π*c/ωvec),Ip,θ)))**2
    return (out)**0.5

Nθ = 1001
θvec = np.linspace(0, np.pi/2, Nθ)  # polarization angle: 0 = ordinary, π/2 = extraordinary

peakgainmat = np.zeros((Nθ,Nω),dtype='complex')
for count in range(0,Nθ):
    peakgainmat[count,:] = gg(λp,2*π*c/ωvec,Ip,θvec[count])

plt.figure()
plt.pcolormesh(ωvec/ωp, np.degrees(θvec), np.abs(peakgainmat)*1e-3, cmap='CMRmap', vmin=0, vmax=6)
plt.ylim((0,90))
plt.xlim((0,2)) 
plt.xlabel('$\omega/\omega_p$')
plt.ylabel('Polarization Angle θ')
cbar = plt.colorbar()
cbar.set_label('Gain (mm$^{-1})$')

#%% DN as a function of wavelength
λvec = np.linspace(400e-9,1600e-9,1000)

DNvec = index_e(λvec*1e6) - index_o(λvec*1e6)

plt.figure()
plt.plot(λvec*1e9, DNvec)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Birefringence (n_e - n_o)')
plt.title('Birefringence Spectrum')

#%% Angle vs index
θ = np.linspace(0,np.pi/2,500)
plt.figure()
plt.plot(θ*180/np.pi,index_e_theta(0.8,θ))
plt.xlabel('Polarization Angle θ')
plt.ylabel('Index n')
plt.axhline(y=index_e(0.8), color='black', linestyle='dotted', label='n(e)')
plt.axhline(y=index_o(0.8), color='red', linestyle='dotted', label='n(o)')
plt.legend()