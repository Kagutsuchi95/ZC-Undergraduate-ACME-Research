# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:01:38 2026

@author: zajc9
"""
#%%
# convert frequency to pump wavelength (x axis)
    # common: 800 nm, 1050ish
    # use nm
# look at 1-2 w/wp range
# maybe try yvo4 next?
# maybe quartz?
# look into tunable intensity #1
# 1d simulation
    # orthogonal polarization
    # self phase modulation mentioned?
    # higher order terms give correct spectral broadening
    # green is one orange is the other (methods)
    # forward maxwell vs nonlinear schrodinger
    # can we limit spectral broadening but keep high gain
        # delta omega is prop to intensiity/pulse time^2
        # dont change I too much, change t instead
        # want long pulse and high intensity but damage is too much
        
#%%
# Lower intensity for YVO4 and maybe quartz (maybe try higher too?) (optional)
# 1d simulation
    # background in right 2 plots is cause time domain isnt large enough
        # double peak, 4x peak for last 
        # increase time window, might have to increase NT
        # usee FWHM of 400, keep pulse duration at 100 ms, then compare (should be the same)
            # put in 400 ns pulse duration after
    # I(t) = e^(-t^2/tau^2)
    # Second peak in right 2 plots is SHG
    # frequency doubling to OPA (1% efficiency) [Why we do intensity tuning]
    # wondering if can instead amplify OPA meethod to create spectral broadening from 700nm to 900nm
# Calculate EXTERNAL angle, not internal angle (use snell's law)
    # TIR probably at 40 degrees
    # start from 0 degrees go up to where TIR occurs
    # stop and start from 45 degrees, add +/- as much as you can (idk what he means here)
        # try 30 degrees as well, all crystal cuts (0, 30, 45, 60, 90), see how wide you can tune it
    # look at both signal and idler side separately
    # basically try to get the gain broaadening as wide as possible

#%%
# birefringence in nonlinear index of refraction
# self-phase modulation and coupled phase modulation
# nonlinear polarization rotation
# new code works up until about 1/3 of experimental intensity
# Can we add dispersion in the propagation?
    # take x and y E values, take ft, impose dispersion, do inverse ft
    
#%%
# 120 fs pulse duration (FWHM, not tau)
# optical shock

#%%
# as B rotates, plot how field rotates with polarization rotation
    # for loop over 180 degrees (B rotation)
    # low intensity, should be normal, higher intensities starts getting funky
    # anisotrophies at high freequencies, can we model HHG?
# efficiency, how short of a pulse can we create with MgO crystal?

#%%
# five-photon process (ignore for now)
# what happens to polarization rotaation as you paraameterize the beam?
    # intensity, pump duraation, crystal length, wavelength, etc.
    # What happens? How does the beam compress? What is the efficiency? 
    
#%%
# Manually test parameters to see how pulse changes (we want smaller pulse duration)
# Test without dispersion, self phase modulation, optical shock to HOPEFULLY get the sqrt(3) ideal shorter pulse duration
# Figure out how to add a BEAM SPLITTING POLARIZER after the crystal to get the individual components
    # Look at Jones matrices basically
# THEN figure out how to ROTATE the polarizer independently of the crystal to change the components
    # Want to eliminate orthogonal components (i think that's what he said)
    # will have to put rotation and anti-rotation matrix on either side of the polarizer matrix
    # Need to rotate, get data, then rotate BACK
    
#Test anomalous dispersion 1785 nm

#%%
# Full beam propagation
# Hankel transforms (bessel functions)
    # PYHANK HAS AN INTERPEROLATER BUILT INTO IT
        # CAN GIVE BAD RESULTS AT HIGH INTENSITIES
            # GIVES NONLINEAR ABSORPTION
# Look at polarization and ellipticity at 1-2 degrees
# For PyHank, try doing the hankel transform of the input gaussian, if the gaussian after the transform isn't the same gaussiaan then increase/decrease grid size
# Polarization into 2D 
  # Take beam shape into account
  
# Vary input angle and intensity (parameter sweeps for polarizer setup)

#%%
# Plot efficiency of stage 4 polarizer
    # Mentioned integrating radially and a function of propagation length
# Compare Stage 4 to 1D propagation code
    # Fix stage 4
        # A function of input theta (x) and output theta (y) seems to shoot up to 60 degrees at small angles experimentally, plateau until ~40 deg input, then input=output at ~45 deg
        # Comparing vs 1D code could provide insight why at 2 deg input, output ~9 deg and not ~600 deg but at 15 deg its ~60 (correct for 15, not for 2)
        
#%%
# Integrate over time and radius r|E(r)|^2 to rotate crystal and polarizer independently to see results from the lab
    # Check transmission % and effiency
    # Draw line to show where its orthogonal to
# Crystal is set to xy frame (lab frame, never changing)
    # if no anistrophy, if input polarization is vertical, max transmission at vertical, if 15 deg B then 135 deg polarizer angle should output min transmission (REALLY LOW INTENSITIES)
    # Rotate polarization, not crystal instead of keeping the polarization fixed and rotating the crystal
        # Take the field projection instead of the tensor projection    
