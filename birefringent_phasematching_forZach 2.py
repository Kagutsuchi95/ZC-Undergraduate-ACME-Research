# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 20:32:08 2025

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt

c = 2.9979e8
π = np.pi

# n2 = -1/3 * 2.5e-20 #Fused Siica
# n2 = -1/3 * 4e-20 #MgO #factor of -1/3 is for orthogonally polarized beams
# n2 = -1/3 * 6e-20 #YAG
# n2 = -1/3 * 2.3e-20 #Quartz
n2 = -1/3 * 3.2e-20 #Sapphire
# n2 = -1/3 * 1.5e-19 #YVO4  #bf of 
# n2 = -1/3 * 4e-19 #LiNbO3
# n2 = -1/3 * 30e-20 #TeO2
# n2 = -1/3 * 2.4e-19 #KTP
# n2 = -1/3 * -8e-19 #PMMA
# P = 20; n2 = -1/3*P*10e-24 #Argon with pressure P  #Do for 0.5, 5, 10, 20 atm pressure
# n2 = -1/3*12e-20 #YSZ
# n2 = 2e-20 #CaF2

Ip = 1.5e17 # 1.5e17
# Argon intensity = 2e17 and 4e17
λp = 800e-9

# Explore argon results if things are interesting
def index_o(λμm):
    
    # return (1+0.9310/(1-(0.079/λμm)**2)+0.1735/(1-(0.130/λμm)**2)+2.1121/(1-(14.918/λμm)**2))**.5 #Fused Silica
    # return (2.956362+0.02195770/(λμm**2-0.01428322)-0.01062387*λμm**2-0.0000204968*λμm**4)**.5 #MgO
    # return (1.882+1.404*λμm**2/(λμm**2-0.1338**2)-0.0137*λμm**2)**.5 #YAG
    # return (1+0.28604141+1.07044083/(1-1.00585997e-2/λμm**2)+1.10202242/(1-100/λμm**2))**.5 #Quartz
    return (1+1.4313493/(1-(0.0726631/λμm)**2)+0.65054713/(1-(0.1193242/λμm)**2)+5.3414021/(1-(18.028251/λμm)**2))**.5 #Sapphire
    # return (3.778790+0.070479/(λμm**2-0.045731)-0.009701*λμm**2)**.5 #YVO4
    # return (1+2.6734/(1-0.01764/λμm**2)+1.2290/(1-0.05914/λμm**2)+12.614/(1-474.60/λμm**2))**.5 #LiNbO3
    # return (1+2.584/(1-(0.1342/λμm)**2)+1.157/(1-(0.2638/λμm)**2))**.5 #TeO2
    # return (3.29100+0.04140/(λμm**2-0.03978)+9.35522/(λμm**2-31.45571))**.5 #KTP
    # return (1+1.1819/(1-0.011313/λμm**2))**.5 #PMMA
    # return (1+1.347091/(1-(0.062543/λμm)**2)+2.117788/(1-(0.166739/λμm)**2)+9.452943/(1-(24.320570/λμm)**2))**.5 #Yttrium stabilized cubic zirconium
    # return (1+0.5675888/(1-(0.050263605/λμm)**2)+0.4710914/(1-(0.1003909/λμm)**2)+3.8484723/(1-(34.649040/λμm)**2))**.5 #CaF2
    # return 1+P*(6.432135E-5+2.8606021E-2/(144-λμm**-2)) #Argon with pressure P


def index_e(λμm):
    
    return (1+1.5039759/(1-(0.0740288/λμm)**2)+0.55069141/(1-(0.1216529/λμm)**2)+6.5927379/(1-(20.072248/λμm)**2))**.5 #Sapphire

# turn DN into a spectrum, DN for every wavelength
DN = -0.0 #this is the birefringence, can be positive or negative, difference between n(e) and n(o) on refractiveindex.info
# DN = 1.5534-1.5443
# DN = 1.5472-1.5383 #Quartz #positively uniaxiial
DN = 1.7522-1.7601 #Sapphire #negatively uniiaxial
# DN = 2.1868-1.9726 #YVO4
# DN = 2.1755-2.2553 #LiNbO3
# DN = 1.8446-1.7565 #KTP
# DN = 2.3736-2.2264 #TeO2

def Δk(λp,λs,Ip):
    ωp = 2*π*c/λp
    ωs = 2*π*c/λs
    ωi = 2*ωp-ωs
    # Mess with these 3 values, change to e and o, mix between 2 o and 2 e to find what is best
    n_p = index_o(λp*1e6) # this one is always either o or e, other 2 are the same, compare plot with original
    ns = index_e(λs*1e6) 
    ni = index_e(2*π*c/ωi*1e6) 
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
    n_p = index_o(λp*1e6)
    ns = index_e(λs*1e6)
    ni = index_e(2*π*c/ωi*1e6)
    kp = ωp/c*n_p*(1-n2*Ip/n_p**2)**0.5
    ks = ωvec/c*index_e(2*π*c/ωvec*1e6)
    ki = ωi/c*ni
    knsq = ωp**2/c**2*n2*Ip
    alpha = ωp/c*n2*Ip/n_p
    out = alpha**2 - (alpha + 1*(1*Δk(λp,(2*π*c/ωvec),Ip) - 1/2*kperp**2*(1/ks + 1/ki)))**2.0
    return (out)**0.5


def ksperpmax(λp,λs,Ip):
    ωp = 2*π*c/λp
    ωs = 2*π*c/λs
    ωi = 2*ωp-ωs
    n_p = index_o(λp*1e6)
    ns = index_e(λs*1e6)
    ni = index_e(2*π*c/ωi*1e6)
    kp = ωp/c*n_p*(1-n2*Ip/n_p**2)**0.5
    ks = ωvec/c*index_e(2*π*c/ωvec*1e6)
    ki = ωi/c*ni
    costhetas = (4*kp**2 + ks**2 - ki**2)/(4*kp*ks)
    # return (1/2*(ks**2 - ki**2)*(1 + (ks**2 - ki**2)/(2*kp)))**0.5
    return ks*(1-costhetas**2)**0.5

# λp = 785e-9
ωp = 2*π*c/λp
n_p = index_o(λp*1e6)
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
