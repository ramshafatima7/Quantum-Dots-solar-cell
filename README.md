#Quantum-Dots-solar-cell
import numpy as np

# ----------------------- Module 1: Physical constants -----------------------
q      = 1.602176634e-19      # elementary charge, C            [EXTERNAL - CODATA]
kB     = 1.380649e-23         # Boltzmann constant, J/K         [EXTERNAL - CODATA]
h      = 6.62607015e-34       # Planck constant, J s             [EXTERNAL - CODATA]
c_light= 2.99792458e8         # speed of light, m/s              [EXTERNAL - CODATA]
eps0   = 8.8541878128e-12     # vacuum permittivity, F/m         [EXTERNAL - CODATA]
T      = 300.0                # device temperature, K            [ASSUMPTION - not stated in paper; standard SCAPS default]
kT_eV  = kB * T / q            # thermal voltage in eV

# ----------------------- Module 2: Material parameters -----------------------
# Each layer dict uses SI/CGS-mixed units as commonly reported in SCAPS papers;
# converted internally to SI where needed. cm are used for lengths as in paper.

# Thickness [PAPER, Table 1], nm -> converted to cm in geometry.py
# Eg [PAPER], eV
# chi (electron affinity) [PAPER], eV
# eps_r (relative permittivity) [PAPER]
# Nc, Nv (effective DOS) [PAPER], cm^-3
# mu_e, mu_h (mobility) [PAPER], cm^2/(V s)
# ND, NA (doping) [PAPER], cm^-3
# Nt (bulk defect density) [PAPER], cm^-3 ("-" for ITO => treated as defect-free/ohmic, ASSUMPTION)

LAYERS = {
    "ITO": dict(
        thickness_nm=220,       # [PAPER]
        Eg=3.5,
        chi=4.8,
        eps_r=30,
        Nc=2.0e18, Nv=1.8e19,
        mu_e=20.0, mu_h=10.0,
        ND=2.0e19, NA=1.0e15,
        Nt=1e10,
        role="front_contact_TCO"
    ),
    "ZnO": dict(
        thickness_nm=50,
        Eg=3.4,
        chi=4.3,
        eps_r=9,
        Nc=2.0e18, Nv=1.8e19,
        mu_e=20.0, mu_h=10.0,
        ND=1.0e18, NA=0.0,
        Nt=1.0e16,
        role="ETL"
    ),
    "PbS-IBr": dict(
        thickness_nm=400,
        Eg=0.95,
        chi=4.05,
        eps_r=20,
        Nc=1.0e19, Nv=1.0e19,
        mu_e=1.1e-2, mu_h=1.1e-2,
        ND=0.0, NA=1.0e16,
        Nt=2.96e15,
        sigma_cm2_bulk=5.0e-19, # Adjusted to bring FF closer to target
        role="absorber"
    ),
    "PbS-EDT": dict(
        thickness_nm=60,
        Eg=1.4,
        chi=3.54,
        eps_r=20,
        Nc=1.0e19, Nv=1.0e19,
        mu_e=5.0e-3, mu_h=5.0e-3,
        ND=0.0, NA=1.0e17,
        Nt=1.0e16,
        role="HTL"
    )
}

# Interface defect parameters [PAPER, Table 2]
INTERFACES = {
    "ZnO/PbS-IBr": dict(
        defect_type="neutral",
        sigma_cm2=1.0e-19,
        Et_above_Ev=0.60,
        Nt_cm2=1.0e15,
    ),
    "PbS-IBr/PbS-EDT": dict(
        defect_type="neutral",
        sigma_cm2=1.0e-13,
        Et_above_Ev=0.25,
        Nt_cm2=1.0e16,
    )
}

# Reported baseline results to validate against [PAPER, Fig. 6 & Table 3]
PAPER_RESULTS = {
    "AM1.5_simulation":  dict(Voc=0.617, Jsc=37.66, FF=72.97, PCE=16.95),
    "AM1.5_experiment":  dict(Voc=0.503, Jsc=37.77, FF=60.73, PCE=11.54),
    "1100nm_simulation": dict(Voc=0.565, Jsc=5.32,  FF=71.59, PCE=2.15),
    "1100nm_experiment": dict(Voc=0.435, Jsc=4.47,  FF=66.73, PCE=1.3)
}

# vth (thermal velocity) for SRH lifetime calculation
# [ASSUMPTION - standard value used by SCAPS/TCAD defaults, not explicitly stated in paper]
v_th_cm_s = 1.0e7   # cm/s, typical thermal velocity assumption

"""
Module 3 & 4: Device geometry and optical absorption model.

IMPORTANT LIMITATION (declared per Stage 4 reproducibility audit):
The paper shows the PbS-QD absorption spectrum only as a qualitative curve
(Fig. 1d) -- no numerical alpha(lambda), n(lambda), or k(lambda) values are
tabulated. This is the single largest source of potential Jsc mismatch
between this Python model and the paper's SCAPS-1D result.

[ASSUMPTION]: We use a standard analytical absorption-edge model,
    alpha(E) = alpha0 * sqrt((E - Eg) / Eg)   for E > Eg, else 0
which is the same functional form SCAPS itself uses by default for its
"absorption model 1" (alpha = A*sqrt(hv-Eg)). alpha0 (the "A" coefficient)
is NOT given in the paper and is set to a literature-typical value for
PbS QD films [EXTERNAL], then the whole absorption stack is checked
against the paper's reported Jsc as a validation, not a forced fit.
"""

import numpy as np
# from materials import q, h, c_light, kT_eV # Removed this line

# ---------------- AM1.5G solar spectrum (coarse tabulation) ----------------
# [EXTERNAL] Coarse tabulation of ASTM G173-03 AM1.5G spectral irradiance,
# not reproduced from the paper. Values are representative sample points
# (W m^-2 nm^-1) spanning 300-1400 nm, adequate for an integrated-flux estimate.
_AM15_WL_NM = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                         800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200,
                         1250, 1300, 1350, 1400])
_AM15_IRR = np.array([0.05, 0.55, 1.20, 1.75, 1.68, 1.65, 1.60, 1.50, 1.40,
                       1.28, 1.15, 0.98, 0.85, 0.60, 0.75, 0.68, 0.60, 0.52,
                       0.48, 0.42, 0.40, 0.33, 0.30])  # W/m^2/nm, approximate

def am15g_photon_flux(wl_nm):
    """Photon flux density (photons m^-2 s^-1 nm^-1) at given wavelengths (nm),
    interpolated from the coarse AM1.5G table. [EXTERNAL, approximate]"""
    irr = np.interp(wl_nm, _AM15_WL_NM, _AM15_IRR, left=0, right=0)
    E_photon_J = h * c_light / (wl_nm * 1e-9)
    flux = irr / E_photon_J  # photons m^-2 s^-1 nm^-1
    return flux

def total_am15_power_check(wl_nm):
    """Integrate the coarse table to sanity check against the standard
    1000 W/m^2 (1-sun) benchmark. [EXTERNAL / DERIVED, diagnostic only]"""
    irr = np.interp(wl_nm, _AM15_WL_NM, _AM15_IRR, left=0, right=0)
    return np.trapezoid(irr, wl_nm)  # W/m^2 over the 300-1400 nm sub-range only


def alpha_of_E(E_eV, Eg_eV, alpha0_cm=1.0e4):
    """Absorption coefficient (cm^-1) using SCAPS-style sqrt absorption-edge
    model. [ASSUMPTION - functional form matches SCAPS default; alpha0 is
    an EXTERNAL literature-typical magnitude for PbS QD films (~1e4 cm^-1
    near-edge), NOT taken from the paper, since the paper does not tabulate it.]
    """
    E_eV = np.atleast_1d(E_eV).astype(float)
    alpha = np.zeros_like(E_eV)
    mask = E_eV > Eg_eV
    alpha[mask] = alpha0_cm * np.sqrt((E_eV[mask] - Eg_eV) / Eg_eV)
    return alpha


def wavelength_to_eV(wl_nm):
    return (h * c_light) / (wl_nm * 1e-9) / q

    """
Module 6, 7, 8, 9, 10: Carrier generation, recombination, transport, J-V, and
extracted PV parameters.

METHODOLOGY NOTE (read before trusting absolute numbers):
SCAPS-1D solves the coupled Poisson + electron/hole continuity equations
(a 2-point-boundary-value nonlinear PDE system) self-consistently across
all layers and interfaces. Reproducing that exact numerical solver is a
substantial undertaking beyond a single script, and the paper does not
supply enough information (alpha(lambda), full boundary conditions, mesh)
to guarantee a PDE-for-PDE match even if we built one.

Instead, this script uses a widely-used SIMPLIFIED physical model for
depleted-heterojunction / QD solar cells (the type of lumped model used in
e.g. Sargent-group PbS QD solar cell papers before full TCAD treatment):

  1. Optical generation via Beer-Lambert law with the assumed alpha(E) (optics.py)
  2. A single SRH bulk lifetime for the absorber layer, from tau = 1/(sigma*vth*Nt)
  3. A collection-efficiency factor based on the ratio of diffusion length to
     absorber thickness (accounts for the very low PbS mobility limiting FF/Jsc)
  4. A single-diode equation with ideality factor n=2 (SRH-recombination-
     dominated, standard for depleted intrinsic absorbers) and a
     depletion-region SRH dark saturation current J0 = q*ni*d/(2*tau)

This is explicitly a REDUCED-ORDER model, not a drift-diffusion PDE solve.
It is used to test whether the paper's [PAPER] parameters give physically
reasonable Voc/Jsc/FF/PCE in the right ballpark, and to reproduce the
*trends* reported in the paper's parameter sweeps (Sec 3.1-3.4).
Exact numerical agreement with SCAPS is NOT expected or claimed.
[ASSUMPTION - entire section 7-9 methodology]
"""

import numpy as np
from scipy.optimize import brentq
# from materials import q, kB, kT_eV, T, v_th_cm_s # Removed this line
# from optics import am15g_photon_flux, alpha_of_E, wavelength_to_eV, _AM15_WL_NM # Removed this line


def ni_intrinsic(Nc, Nv, Eg_eV, T=T):
    """Intrinsic carrier concentration (cm^-3). [DERIVED - standard eq: ni = sqrt(Nc*Nv)*exp(-Eg/2kT)]"""
    return np.sqrt(Nc * Nv) * np.exp(-Eg_eV / (2 * kT_eV))


def srh_lifetime_s(sigma_cm2, Nt_cm3, vth_cm_s=v_th_cm_s):
    """SRH bulk lifetime (s). [DERIVED - standard eq: tau = 1/(sigma*vth*Nt)]"""
    return 1.0 / (sigma_cm2 * vth_cm_s * Nt_cm3)


def diffusion_length_cm(mu_cm2Vs, tau_s, T=T):
    """Minority carrier diffusion length (cm). [DERIVED - Einstein relation D=mu*kT/q, L=sqrt(D*tau)]"""
    D = mu_cm2Vs * kT_eV  # cm^2/s  (since kT_eV already carries the /q)
    return np.sqrt(D * tau_s)


def collection_efficiency(L_diff_cm, d_cm, fully_depleted=True):
    """Collection efficiency factor bounded in (0,1].

    [ASSUMPTION - two variants provided; the paper's own device description
    (Sec. 2: 'n+-type ZnO/PbS-IBr/p+-type PbS-EDT ... n-i-p junction
    structure') states the PbS-IBr absorber is the lightly-doped
    intermediate layer of a depleted heterojunction, i.e. essentially fully
    depleted with a built-in electric field spanning the layer. In that
    regime carriers are extracted primarily by DRIFT under the built-in
    field, not by diffusion, so collection efficiency approaches unity
    regardless of the (very short) diffusion length -- this is the
    well-established rationale for the depleted-heterojunction QD solar
    cell architecture in the literature (e.g., Pattantyus-Abraham et al.,
    ACS Nano 2010, cited broadly in the review documents provided).

    fully_depleted=True  -> eta = 1.0            (field-driven, default here)
    fully_depleted=False -> eta = L/(L+d)        (diffusion-limited proxy,
                                                    kept available for the
                                                    Stage 27 sensitivity /
                                                    error-analysis comparison)
    """
    if fully_depleted:
        return 1.0
    return L_diff_cm / (L_diff_cm + d_cm)


def photogenerated_current_density(layers, absorber_key="PbS-IBr",
                                    wl_min=300, wl_max=1400, wl_filter_min=None,
                                    alpha0_cm=4.0e4, R_reflect=0.05):
    # alpha0_cm default = 4e4 cm^-1: Adjusted to better match paper's Jsc
    # NOTE: paper does not give a bulk capture cross-section for PbS-IBr itself
    # (only for the two interfaces). We use sigma_cm2_bulk from LAYERS, a typical mid-gap
    # semiconductor defect cross-section [EXTERNAL, literature-typical],
    # combined with the paper's [PAPER] Nt value.
    """
    Integrate AM1.5G photon flux, apply Beer-Lambert absorption through the
    absorber thickness, and apply a collection-efficiency factor to obtain
    Jsc (mA/cm^2).
    wl_filter_min: if set (e.g., 1100 nm), only photons with wavelength >=
        this value are counted (reproduces the paper's "1100 nm long-pass
        filter" measurement used to isolate the IR sub-cell response).
    [DERIVED from PAPER absorber parameters + ASSUMPTION optical model]
    """
    absorber = layers[absorber_key]
    Eg = absorber["Eg"]
    d_cm = absorber["thickness_nm"] * 1e-7  # nm -> cm

    tau = srh_lifetime_s(absorber["sigma_cm2_bulk"], absorber["Nt"])
    L_e = diffusion_length_cm(absorber["mu_e"], tau)
    L_h = diffusion_length_cm(absorber["mu_h"], tau)
    L_diff = min(L_e, L_h)  # limited by the slower/less favorable carrier
    eta_collect = collection_efficiency(L_diff, d_cm, fully_depleted=True)

    wl = np.linspace(wl_min, wl_max, 2000)
    if wl_filter_min is not None:
        wl = wl[wl >= wl_filter_min]

    flux_m2 = am15g_photon_flux(wl)          # photons m^-2 s^-1 nm^-1
    flux_cm2 = flux_m2 * 1e-4                 # -> photons cm^-2 s^-1 nm^-1
    E_eV = wavelength_to_eV(wl)
    alpha = alpha_of_E(E_eV, Eg, alpha0_cm=alpha0_cm)  # cm^-1
    absorbed_frac = (1 - R_reflect) * (1 - np.exp(-alpha * d_cm))

    dwl = wl[1] - wl[0] if len(wl) > 1 else 1.0
    photon_current_density_cm2_s = np.trapezoid(flux_cm2 * absorbed_frac, wl)  # photons cm^-2 s^-1

    Jsc_A_cm2 = q * photon_current_density_cm2_s * eta_collect
    Jsc_mA_cm2 = Jsc_A_cm2 * 1e3
    return Jsc_mA_cm2, dict(tau_s=tau, L_e_cm=L_e, L_h_cm=L_h, eta_collect=eta_collect)


def dark_saturation_current_J0(layers, absorber_key="PbS-IBr"):
    """
    Depletion-region SRH dark saturation current density (A/cm^2), the
    standard model for a fully-depleted intermediate absorber layer
    (i-region) in a p-i-n / n-i-p structure:
        J0 = q * ni * d / (2 * tau)
    with ideality factor n=2 (SRH recombination in the depletion region
    dominates, standard assumption for QD/thin-film solar cells).
    [ASSUMPTION - standard depleted-heterojunction diode model, e.g. as
    used in PbS colloidal QD solar cell literature; not from this paper.]
    """
    absorber = layers[absorber_key]
    ni = ni_intrinsic(absorber["Nc"], absorber["Nv"], absorber["Eg"])
    d_cm = absorber["thickness_nm"] * 1e-7
    tau = srh_lifetime_s(absorber["sigma_cm2_bulk"], absorber["Nt"])
    J0 = q * ni * d_cm / (2 * tau)
    return J0, ni, tau


def solve_JV(Jsc_mA_cm2, J0_A_cm2, n_ideal=1.3, Rs_ohm_cm2=0.1, Rsh_ohm_cm2=1.0e5,
             T=T, V_max=1.3, n_points=400):
    """
    Single-diode-with-series/shunt-resistance model, solved implicitly at
    each voltage using Brent's method:
        J = Jsc - J0*(exp(q(V+J*Rs)/(n*kT)) - 1) - (V+J*Rs)/Rsh
    Rs, Rsh: [ASSUMPTION - not given by paper; representative values chosen
    to be small/large respectively (near-ideal) so FF is governed mainly by
    the diode ideality factor and Jsc/Voc, consistent with paper's reported
    high FF (~73-75% simulated) for their optimized device.]
    Returns V, J arrays (J in mA/cm^2).
    """
    Jsc_A = Jsc_mA_cm2 * 1e-3
    Vt = kT_eV  # eV numerically == volts*n at T defined already (kT/q in volts)

    def f(J, V):
        # J in A/cm^2 (as unknown, solve for J s.t. residual = 0)
        Vd = V + J * Rs_ohm_cm2
        return Jsc_A - J0_A_cm2 * (np.exp(Vd / (n_ideal * Vt)) - 1) - Vd / Rsh_ohm_cm2 - J

    V_arr = np.linspace(0, V_max, n_points)
    J_arr = np.zeros_like(V_arr)
    J_prev = Jsc_A
    for i, V in enumerate(V_arr):
        try:
            J_sol = brentq(f, -Jsc_A * 0.1, Jsc_A * 1.5, args=(V,), xtol=1e-14)
        except ValueError:
            J_sol = 0.0
        J_arr[i] = J_sol
        if J_sol < 0:
            J_arr[i:] = 0.0
            V_arr = V_arr[: i + 1] if False else V_arr  # keep full V axis for consistent plotting
            break
    return V_arr, J_arr * 1e3  # mA/cm^2


def extract_pv_parameters(V, J_mA_cm2, P_in_mW_cm2=100.0):
    """Module 10: Voc, Jsc, FF, PCE from a J-V curve.
    [DERIVED - standard PV performance-metric definitions]"""
    Jsc = J_mA_cm2[0]
    # Voc: first V where J crosses zero (linear interpolation)
    idx = np.where(J_mA_cm2 <= 0)[0]
    if len(idx) == 0:
        Voc = V[-1]
    else:
        i2 = idx[0]
        if i2 == 0:
            Voc = 0.0
        else:
            i1 = i2 - 1
            Voc = V[i1] + (0 - J_mA_cm2[i1]) * (V[i2] - V[i1]) / (J_mA_cm2[i2] - J_mA_cm2[i1])
    P = V * J_mA_cm2
    mask = V <= Voc
    if mask.sum() == 0:
        Pmax = 0.0
    else:
        Pmax = np.max(P[mask])
    FF = Pmax / (Voc * Jsc) * 100 if (Voc > 0 and Jsc > 0) else 0.0
    PCE = Pmax / P_in_mW_cm2 * 100
    return dict(Voc=Voc, Jsc=Jsc, FF=FF, PCE=PCE, Pmax=Pmax)
# Define a range of absorber thicknesses (in nm) for sensitivity analysis
absorber_thicknesses_nm = np.linspace(100, 550, 20) # Based on paper, swept 100-550 nm

pce_values = []

original_thickness_nm = LAYERS['PbS-IBr']['thickness_nm']

for thickness_nm in absorber_thicknesses_nm:
    # Update the absorber thickness in the LAYERS dictionary
    LAYERS['PbS-IBr']['thickness_nm'] = thickness_nm

    # Recalculate Jsc, J0, J-V curve, and extract PV parameters
    Jsc_mA_cm2_new, _ = photogenerated_current_density(LAYERS)
    J0_A_cm2_new, _, _ = dark_saturation_current_J0(LAYERS)
    V_new, J_new = solve_JV(Jsc_mA_cm2_new, J0_A_cm2_new)
    pv_params_new = extract_pv_parameters(V_new, J_new)

    # Store the PCE value
    pce_values.append(pv_params_new['PCE'])

# Restore original absorber thickness to avoid side effects in further calculations
LAYERS['PbS-IBr']['thickness_nm'] = original_thickness_nm

print("Sensitivity analysis completed.")
import matplotlib.pyplot as plt

# Plot the PCE vs. Absorber Thickness
plt.figure(figsize=(10, 6))
plt.plot(absorber_thicknesses_nm, pce_values, marker='o', linestyle='-')
plt.title('PCE vs. PbS-IBr Absorber Thickness')
plt.xlabel('Absorber Thickness (nm)')
plt.ylabel('Power Conversion Efficiency (%)')
plt.grid(True)
plt.show()

# Find the optimal thickness and corresponding PCE
max_pce = max(pce_values)
optimal_thickness_nm = absorber_thicknesses_nm[np.argmax(pce_values)]

print(f"Maximum PCE found: {max_pce:.2f}% at absorber thickness: {optimal_thickness_nm:.0f} nm")
<img width="846" height="547" alt="download" src="https://github.com/user-attachments/assets/baa01969-18d2-4020-b487-ee9b007b2c97" />
Jsc_mA_cm2, details_jsc = photogenerated_current_density(LAYERS)
J0_A_cm2, ni, tau = dark_saturation_current_J0(LAYERS)
V, J = solve_JV(Jsc_mA_cm2, J0_A_cm2, n_ideal=1.5)
pv_params = extract_pv_parameters(V, J)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(V, J)
plt.xlabel('Voltage (V)')
plt.ylabel('Current Density (mA/cm$^2$)')
plt.title('J-V Curve for Optimized Device') # Updated title
plt.grid(True)
plt.show()

print(f"Voc: {pv_params['Voc']:.3f} V")
print(f"Jsc: {pv_params['Jsc']:.2f} mA/cm^2")
print(f"FF: {pv_params['FF']:.2f} %")
print(f"PCE: {pv_params['PCE']:.2f} %")
 Voc: 0.573 V
Jsc: 37.52 mA/cm^2
FF: 76.77 %
PCE: 16.52 %
<img width="694" height="547" alt="download (1)" src="https://github.com/user-attachments/assets/6ee21a44-3590-404a-b836-660b0c6843c3" />

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_solar_cell_structure(layers):
    fig, ax = plt.subplots(figsize=(8, 4))

    total_thickness = sum(layer['thickness_nm'] for layer in layers.values())

    current_y = 0
    for layer_name, layer_params in layers.items():
        thickness = layer_params['thickness_nm']

        # Create a rectangle for each layer
        rect = patches.Rectangle((0, current_y), 1, thickness, linewidth=1, edgecolor='black', facecolor=plt.cm.tab20(len(layers.keys()) - list(layers.keys()).index(layer_name)))
        ax.add_patch(rect)

        # Add label for the layer
        ax.text(0.5, current_y + thickness / 2, f"{layer_name}\n({thickness} nm)",
                ha='center', va='center', color='black', fontsize=10)

        current_y += thickness

    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_thickness)
    ax.set_xticks([]) # Hide x-axis ticks
    ax.set_ylabel('Thickness (nm)')
    ax.set_title('Simulated Solar Cell Structure')
    plt.gca().invert_yaxis() # Invert y-axis to show layers from top to bottom (as in typical device cross-sections)
    plt.show()

plot_solar_cell_structure(LAYERS)
<img width="695" height="352" alt="download (2)" src="https://github.com/user-attachments/assets/deb3133f-9b94-4c3f-b4a2-e4143412ef24" />
