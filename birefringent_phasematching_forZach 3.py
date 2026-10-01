# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 20:32:08 2025

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt

c = 2.9979e8
π = np.pi

n2 = -1/3 * 3.2e-20 #Sapphire


Ip = 1.5e17 # 1.5e17
λp = 800e-9

def index_o(λμm):

    return (1+1.4313493/(1-(0.0726631/λμm)**2)+0.65054713/(1-(0.1193242/λμm)**2)+5.3414021/(1-(18.028251/λμm)**2))**.5 #Sapphire



def index_e(λμm):
    
    return (1+1.5039759/(1-(0.0740288/λμm)**2)+0.55069141/(1-(0.1216529/λμm)**2)+6.5927379/(1-(20.072248/λμm)**2))**.5 #Sapphire

# turn DN into a spectrum, DN for every wavelength
DN = -0.0 #this is the birefringence, can be positive or negative, difference between n(e) and n(o) on refractiveindex.info
# DN = 1.7522-1.7601 #Sapphire #negatively uniaxial

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

def angle(kperp, λp):
    ωp = 2*π*c/λp
    n_p = index_o(λp*1e6)
    kp = ωp/c*n_p
    return np.arcsin(kperp / kp)


# λp = 785e-9
ωp = 2*π*c/λp
n_p = index_o(λp*1e6)
kp = ωp/c*n_p

Nk = 1001
# kperpvec = np.linspace(0,0.2*kp,Nk)
thetavec = np.linspace(0,0.2,Nk)
peakgainmat = np.zeros((Nk,Nω),dtype='complex')
for count in range(0,Nk):
    kperp = kp * np.sin(thetavec[count])
    peakgainmat[count,:] = gg(λp,2*π*c/ωvec,Ip,kperp)

plt.figure()
plt.pcolormesh(ωvec/ωp,thetavec*180/np.pi,np.abs(peakgainmat)*1e-3,cmap = 'CMRmap')

# plt.figure()
plt.plot(ωvec/ωp,angle(ksperpmax(λp,2*π*c/ωvec,Ip), λp)*180/np.pi,'--k')
plt.ylim((0,10))
plt.xlim((0,2))
plt.xlabel('$\omega/\omega_p$')
plt.ylabel('Angle (degrees)')
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
