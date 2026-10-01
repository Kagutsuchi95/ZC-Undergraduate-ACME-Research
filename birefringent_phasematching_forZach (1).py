# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 20:32:08 2025

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt

c = 2.9979e8
π = np.pi

n2 = -1/3*4e-20 #MgO #factor of -1/3 is for orthogonally polarized beams
# n2 = -1/3*6e-20 #YAG
# P = 5; n2 = -1/3*P*10e-24#Argon with pressure P
# n2 = -1/3*12e-20 #YSZ
# n2 = 2e-20 #CaF2

Ip = 1.5e17
λp = 800e-9


def index(λμm):
    return (2.956362+0.02195770/(λμm**2-0.01428322)-0.01062387*λμm**2-0.0000204968*λμm**4)**.5 #MgO
    # return (1+2.28200/(1-0.01185/λμm**2)+3.27644/(1-282.734/λμm**2))**.5 #YAG
    # return (1+1.347091/(1-(0.062543/λμm)**2)+2.117788/(1-(0.166739/λμm)**2)+9.452943/(1-(24.320570/λμm)**2))**.5 #Yttrium stabilized cubic zirconium
    # return (1+0.5675888/(1-(0.050263605/λμm)**2)+0.4710914/(1-(0.1003909/λμm)**2)+3.8484723/(1-(34.649040/λμm)**2))**.5 #CaF2
    # return 1+P*(6.432135E-5+2.8606021E-2/(144-λμm**-2)) #Argon with pressure P

DN = -0.0 #this is the birefringence

def Δk(λp,λs,Ip):
    ωp = 2*π*c/λp
    ωs = 2*π*c/λs
    ωi = 2*ωp-ωs
    n_p = index(λp*1e6)
    ns = index(λs*1e6)
    ni = index(2*π*c/ωi*1e6)
    # return 2/c*(n_p*ωp*(1 + 1/3*n2*Ip)) - 1/c*(ns*ωs + ni*ωi) #from Cohen PRL 2009, for orthogonal polarization
    return +1/c*(ns*ωs + ni*ωi + 2*DN*ωp - 2*n_p*ωp ) #From Agrawal, eq 10.2.18, for same polarization
#this seems to be correct. So you cannot get perfect phase matching on axis in the normal dispersion regime by tuning the pump power
#but you can in the anomalous dispersion regime
    
phasematch = Δk(λp,600e-9,Ip)*1e-3


t = np.linspace(-200,200,1001)
tau = 50
Iplaser = Ip*np.exp(-t**2/(2*tau**2))
λsvec = np.linspace(500e-9,1600e-9,1001)


Nω = 1001
ωvec = np.linspace(0,2*(2*π*c/λp),Nω)

def gg(λp,λs,Ip,kperp):
    ωp = 2*π*c/λp
    ωs = 2*π*c/λs
    ωi = 2*ωp-ωs
    n_p = index(λp*1e6)
    ns = index(λs*1e6)
    ni = index(2*π*c/ωi*1e6)
    kp = ωp/c*n_p*(1-n2*Ip/n_p**2)**0.5
    ks = ωvec/c*index(2*π*c/ωvec*1e6)
    ki = ωi/c*ni
    knsq = ωp**2/c**2*n2*Ip
    alpha = ωp/c*n2*Ip/n_p
    out = alpha**2 - (alpha + 1*(1*Δk(λp,(2*π*c/ωvec),Ip) - 1/2*kperp**2*(1/ks + 1/ki)))**2.0
    return (out)**0.5


def ksperpmax(λp,λs,Ip):
    ωp = 2*π*c/λp
    ωs = 2*π*c/λs
    ωi = 2*ωp-ωs
    n_p = index(λp*1e6)
    ns = index(λs*1e6)
    ni = index(2*π*c/ωi*1e6)
    kp = ωp/c*n_p*(1-n2*Ip/n_p**2)**0.5
    ks = ωvec/c*index(2*π*c/ωvec*1e6)
    ki = ωi/c*ni
    costhetas = (4*kp**2 + ks**2 - ki**2)/(4*kp*ks)
    # return (1/2*(ks**2 - ki**2)*(1 + (ks**2 - ki**2)/(2*kp)))**0.5
    return ks*(1-costhetas**2)**0.5

# λp = 785e-9
ωp = 2*π*c/λp
n_p = index(λp*1e6)
kp = ωp/c*n_p

Nk = 1001
kperpvec = np.linspace(0,0.2*kp,Nk)
peakgainmat = np.zeros((Nk,Nω),dtype='complex')
for count in range(0,Nk):
    peakgainmat[count,:] = gg(λp,2*π*c/ωvec,Ip,kperpvec[count])

plt.figure()
plt.pcolormesh(ωvec/ωp,kperpvec/kp,np.abs(peakgainmat)*1e-3,cmap = 'CMRmap')

# plt.figure()
plt.plot(ωvec/ωp,ksperpmax(λp,2*π*c/ωvec,Ip)/kp,'--k')
plt.ylim((0,0.12))
plt.xlim((0,2))
plt.xlabel('$\omega/\omega_p$')
plt.ylabel('$K_{\perp}/k_p$')
cbar = plt.colorbar()
cbar.set_label('Gain (mm$^{-1})$')

#%%
plt.figure(99)
L = 1e-3 #length of the crystal for amplification
plt.plot(2*π*c/ωvec*1e9,peakgainmat[0,:]*L)
plt.xlim((300,1500))

plt.figure()
plt.plot(2*π*c/ωvec*1e9,1/4*np.exp(peakgainmat[0,:]*L))
plt.xlim((300,1500))
