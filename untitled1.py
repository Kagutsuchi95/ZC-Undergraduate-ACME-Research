import numpy as np
import matplotlib.pyplot as plt

c = 2.9979e8
π = np.pi
n2 = -1/3 * 3.2e-20  # Sapphire
Ip = 1.5e17
λp = 800e-9

def index_o(λμm):
    return (1 + 1.4313493/(1-(0.0726631/λμm)**2)
              + 0.65054713/(1-(0.1193242/λμm)**2)
              + 5.3414021/(1-(18.028251/λμm)**2))**0.5  # Sapphire ordinary

def index_e(λμm):
    return (1 + 1.5039759/(1-(0.0740288/λμm)**2)
              + 0.55069141/(1-(0.1216529/λμm)**2)
              + 6.5927379/(1-(20.072248/λμm)**2))**0.5  # Sapphire extraordinary

# -----------------------------------------------------------------------
# Angle-dependent effective extraordinary index (from BBO solution eq. 4)
# n_eff(θ) is the refractive index seen by a wave whose polarization makes
# angle θ with the ordinary axis of the crystal:
#
#   1/n_eff(θ)^2 = cos²(θ)/n_o² + sin²(θ)/n_e²
#
# θ = 0  → pure ordinary polarization  → n_eff = n_o
# θ = π/2 → pure extraordinary polarization → n_eff = n_e
# -----------------------------------------------------------------------
def index_eff(λμm, θ):
    """
    Effective refractive index for polarization angle θ relative to ordinary axis.
    θ is a scalar or array (radians).
    λμm is wavelength in microns (scalar or broadcastable array).
    """
    no = index_o(λμm)
    ne = index_e(λμm)
    return 1.0 / np.sqrt(np.cos(θ)**2 / no**2 + np.sin(θ)**2 / ne**2)

DN = -0.0  # birefringence offset (keep for compatibility)

# -----------------------------------------------------------------------
# Phase mismatch Δk as a function of pump wavelength, signal wavelength,
# pump intensity, and polarization angle φ of the signal/idler.
#
# The pump is taken as ordinary (θ_pump = 0).
# Signal and idler share the same polarization angle φ (Type-I like).
# -----------------------------------------------------------------------
def Δk(λp, λs, Ip, φ):
    ωp = 2*π*c / λp
    ωs = 2*π*c / λs
    ωi = 2*ωp - ωs

    n_p  = index_o(λp * 1e6)                          # pump: ordinary
    ns   = index_eff(λs * 1e6, φ)                     # signal: polarization angle φ
    ni   = index_eff(2*π*c/ωi * 1e6, φ)               # idler:  same polarization angle

    return +1/c * (ns*ωs + ni*ωi + 2*DN*ωp - 2*n_p*ωp)

# -----------------------------------------------------------------------
# OPA gain g(ω, φ)
#
# Original formula used kperp to account for non-collinear geometry:
#   g² = α² − (α + Δk/2 − kperp²/(2ks) − kperp²/(2ki))²
#
# Now φ is the polarization mixing angle. The projection of the pump
# polarization onto the crystal axes modifies the effective indices and
# thus Δk directly. There is no separate kperp term — the "walk-off"
# effect is captured entirely through n_eff(φ).
#
# Reference: BBO solution sheet eq. (4) for n_eff(θ),
#            Agrawal eq. 10.2.18 for the gain formula structure.
# -----------------------------------------------------------------------
Nω  = 1001
ωp  = 2*π*c / λp
n_p = index_o(λp * 1e6)
kp  = ωp/c * n_p

ωvec = np.linspace(1e-3 * ωp, 2*ωp, Nω)   # avoid ω=0 for index eval

Nφ   = 401
# Scan polarization angle from 0 (pure ordinary) to π/2 (pure extraordinary)
φvec = np.linspace(0, π/2, Nφ)

peakgainmat = np.zeros((Nφ, Nω), dtype='complex')

for i, φ in enumerate(φvec):
    ωs  = ωvec
    ωi  = 2*ωp - ωs

    # Effective indices at signal and idler frequencies for this φ
    λs_μm = 2*π*c / ωs * 1e6
    λi_μm = 2*π*c / ωi * 1e6

    # Guard: idler wavelength must be positive (signal < 2ωp)
    valid = ωi > 0
    λi_μm_safe = np.where(valid, λi_μm, λs_μm)  # dummy value where invalid

    ns = index_eff(λs_μm, φ)
    ni = index_eff(λi_μm_safe, φ)

    dk = +1/c * (ns*ωs + ni*ωi + 2*DN*ωp - 2*n_p*ωp)

    # Gain coefficient (Agrawal 10.2.18 structure)
    # α = pump nonlinear coupling coefficient
    α = ωp/c * n2 * Ip / n_p

    out = α**2 - (α + dk)**2
    out = np.where(valid, out, 0.0)
    peakgainmat[i, :] = np.sqrt(out.astype(complex))

# -----------------------------------------------------------------------
# Plot: gain as function of (ω/ωp, φ in degrees)
# -----------------------------------------------------------------------
φ_deg = np.degrees(φvec)

fig, ax = plt.subplots(figsize=(8, 5))
pcm = ax.pcolormesh(ωvec/ωp, φ_deg, np.abs(peakgainmat)*1e-3,
                    cmap='CMRmap', shading='auto')

# Mark Type-I phase matching angle from BBO solution (eq. 4):
# tan(θ_pm) = (n_e(2ω)/n_o(2ω)) * sqrt((n_o²(ω) − n_o²(2ω)) / (n_e²(2ω) − n_o²(ω)))
# Here we solve numerically for each signal frequency
θ_pm_vec = []
for ω in ωvec:
    λs_μm = 2*π*c / ω * 1e6
    λi_μm_safe = 2*π*c / max(2*ωp - ω, 1e10) * 1e6
    # Find angle where Δk = 0 by bisection on φ
    try:
        from scipy.optimize import brentq
        def dk_of_phi(phi):
            ns = index_eff(λs_μm, phi)
            ωi = 2*ωp - ω
            ni = index_eff(2*π*c/ωi*1e6, phi) if ωi > 0 else ns
            return +1/c*(ns*ω + ni*ωi + 2*DN*ωp - 2*n_p*ωp)
        if dk_of_phi(0)*dk_of_phi(π/2) < 0:
            θ_pm = brentq(dk_of_phi, 0, π/2)
        else:
            θ_pm = np.nan
    except Exception:
        θ_pm = np.nan
    θ_pm_vec.append(np.degrees(θ_pm))

ax.plot(ωvec/ωp, θ_pm_vec, '--k', lw=1.5, label='Phase-match angle $\\phi_{pm}$')

ax.set_xlabel('$\\omega / \\omega_p$', fontsize=13)
ax.set_ylabel('Polarization angle $\\phi$ (degrees)', fontsize=13)
ax.set_xlim((0, 2))
ax.set_ylim((0, 90))
ax.legend(fontsize=11)

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label('Gain (mm$^{-1}$)', fontsize=12)
ax.set_title('OPA Gain vs. Frequency and Polarization Angle (Sapphire)', fontsize=13)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/opa_gain_polarization_angle.png', dpi=150)
plt.show()
print("Done. Plot saved.")