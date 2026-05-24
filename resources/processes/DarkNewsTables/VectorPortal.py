"""
Vector-portal dark matter processes for SIREN.

Implements the physics from Dutta, Kim, Thompson, Thornton, Van de Water,
PRL 129, 111803 (2022) [arXiv:2110.11944]:

    Production:  pi/K -> l nu V1  (three-body meson decay)
    Decay:       V1 -> chi chi'   (dark photon to DM pair)
    Scattering:  chi N -> chi' N  (upscattering via t-channel V2)
    Decay:       chi' -> chi V1   (de-excitation)
    Decay:       V1 -> e+ e-      (visible signal)

All classes are self-contained with no DarkNews imports.
Cross-section classes implement the ups_case duck-type interface
expected by PyDarkNewsCrossSection.  Decay classes inherit directly
from siren.interactions.DarkNewsDecay.
"""

import os
import math
import numpy as np
import scipy.integrate as _integrate

from siren.interactions import DarkNewsDecay
from siren import dataclasses
from siren.dataclasses import Particle

_ALPHA_EM = 1.0 / 137.036
_GEV2_TO_CM2 = 3.8938e-28
_M_ELECTRON = 0.000511  # GeV


# ---------------------------------------------------------------------------
# Lightweight particle / target stubs
# ---------------------------------------------------------------------------

class _Stub:
    def __init__(self, pdgid, mass=0.0, name=""):
        self.pdgid = pdgid
        self.mass = mass
        self.name = name

    def __eq__(self, other):
        if isinstance(other, int):
            return self.pdgid == other
        return self.pdgid == getattr(other, "pdgid", None)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __int__(self):
        return self.pdgid

    def __hash__(self):
        return hash(self.pdgid)

    def __repr__(self):
        return f"_Stub({self.pdgid}, {self.name!r})"


# ---------------------------------------------------------------------------
# Kinematic helpers (inlined from DarkNews phase_space)
# ---------------------------------------------------------------------------

def _Q2max(E, m_ups, M):
    s = 2.0 * E * M + M**2
    kallen_arg = M**4 + (m_ups**2 - s)**2 - 2.0 * M**2 * (m_ups**2 + s)
    kallen_sqrt = math.sqrt(max(kallen_arg, 0.0))
    return 0.5 / s * (
        M**4 - m_ups**2 * s + s**2
        - M**2 * (m_ups**2 + 2.0 * s)
        + (-M**2 + s) * kallen_sqrt
    )


def _Q2min(E, m_ups, M):
    s = 2.0 * E * M + M**2
    r = m_ups / math.sqrt(s) if s > 0 else 0.0
    m = M / math.sqrt(s) if s > 0 else 0.0

    if r < 1e-3:
        return m**2 / (1.0 - m**2)**2 * r**4 * s if (1.0 - m**2) != 0 else 0.0

    kallen_arg = (1.0 - m**2)**2 - 2.0 * (1.0 + m**2) * r**2 + r**4
    kallen_sqrt = math.sqrt(max(kallen_arg, 0.0))

    return 0.5 * s * (
        1.0 + m**2 - r**2 - kallen_sqrt
        + m**2 * (-2.0 - r**2 + kallen_sqrt)
    )


def _two_body_p_cm(M, m1, m2):
    arg = (M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)
    if arg <= 0.0:
        return 0.0
    return math.sqrt(arg) / (2.0 * M)


def _boost_to_lab(P_parent, p_cm, cos_theta, phi, m_daughter):
    E_parent = P_parent[0]
    p_parent = P_parent[1:]
    M_parent_sq = max(E_parent**2 - np.dot(p_parent, p_parent), 0.0)
    M_parent = math.sqrt(M_parent_sq)

    sin_theta = math.sqrt(max(1.0 - cos_theta**2, 0.0))
    E_rf = math.sqrt(p_cm**2 + m_daughter**2)
    p_rf = np.array([
        p_cm * sin_theta * math.cos(phi),
        p_cm * sin_theta * math.sin(phi),
        p_cm * cos_theta,
    ])

    p_mag = np.linalg.norm(p_parent)
    if p_mag < 1e-12 or M_parent < 1e-12:
        return np.array([E_rf, *p_rf])

    beta = p_mag / E_parent
    gamma = E_parent / M_parent
    beta_hat = p_parent / p_mag

    p_par = np.dot(p_rf, beta_hat)
    p_perp = p_rf - p_par * beta_hat

    E_lab = gamma * (E_rf + beta * p_par)
    p_par_lab = gamma * (p_par + beta * E_rf)
    return np.array([E_lab, *(p_par_lab * beta_hat + p_perp)])


# ---------------------------------------------------------------------------
# Helm nuclear form factor
# ---------------------------------------------------------------------------

def _helm_F2(Q2, A):
    Q = math.sqrt(max(Q2, 0.0))
    Qfm = Q / 0.197327
    s = 0.9
    r0sq = max((1.2 * A**(1.0 / 3.0))**2 - 5.0 * s**2, 0.0)
    Qr = Qfm * math.sqrt(r0sq)
    if Qr < 1e-6:
        j1_over_Qr = 1.0 / 3.0
    else:
        j1_over_Qr = (math.sin(Qr) - Qr * math.cos(Qr)) / Qr**3
    return (3.0 * j1_over_Qr)**2 * math.exp(-(Qfm * s)**2)


# ===================================================================
#  VectorPortalUpsCase  --  chi N -> chi' N  upscattering
# ===================================================================

class VectorPortalUpsCase:
    """
    Dark-matter upscattering chi N -> chi' N via t-channel dark photon V2.

    Implements the duck-type interface expected by PyDarkNewsCrossSection:
        .nu_projectile, .nu_upscattered, .nuclear_target  (with .pdgid, .mass)
        .MA, .m_ups, .h_upscattered, .Ethreshold, .scattering_regime
        .total_xsec(E), .diff_xsec_Q2(E, Q2)
    """

    def __init__(
        self,
        m_chi,
        m_chi_prime,
        m_V,
        g_D,
        epsilon,
        *,
        pdgid_chi=5917,
        pdgid_chi_prime=5918,
        nuclear_pdgid=1000180400,
        nuclear_mass=37.215,
        nuclear_name="Ar40",
        A=40,
        Z=18,
        scattering_regime="coherent",
    ):
        self.m_chi = m_chi
        self.m_chi_prime = m_chi_prime
        self.m_V = m_V
        self.g_D = g_D
        self.epsilon = epsilon
        self.A = A
        self.Z = Z

        self.nu_projectile = _Stub(pdgid_chi, m_chi, "chi")
        self.nu_upscattered = _Stub(pdgid_chi_prime, m_chi_prime, "chi_prime")
        self.nuclear_target = _Stub(nuclear_pdgid, nuclear_mass, nuclear_name)
        self.nuclear_target.Z = Z
        self.nuclear_target.N = A - Z

        self.MA = nuclear_mass
        self.m_ups = m_chi_prime
        self.h_upscattered = 1
        self.scattering_regime = scattering_regime

        self.Ethreshold = ((m_chi_prime + nuclear_mass)**2
                          - m_chi**2 - nuclear_mass**2) / (2.0 * nuclear_mass)

    def _dsigma_dQ2(self, E, Q2):
        m1 = self.m_chi
        m3 = self.m_chi_prime
        M = self.MA
        mV = self.m_V

        s = m1**2 + M**2 + 2.0 * M * E
        flux_sq = (s - M**2)**2
        if flux_sq <= 0.0:
            return 0.0

        delta_m2 = m3**2 - m1**2
        numerator = 2.0 * M**2 * (2.0 * E * M - Q2 - delta_m2)
        if numerator <= 0.0:
            return 0.0

        propagator = 1.0 / (Q2 + mV**2)**2
        M2 = self.g_D**2 * 4.0 * math.pi * _ALPHA_EM * self.epsilon**2 * numerator * propagator
        F2 = _helm_F2(Q2, self.A)
        Q_eff_sq = (self.Z * self.epsilon)**2 if self.scattering_regime == "coherent" else 1.0

        dsig = M2 * F2 / (16.0 * math.pi * flux_sq)
        return max(0.0, dsig) * _GEV2_TO_CM2

    def diff_xsec_Q2(self, E, Q2):
        return np.array(self._dsigma_dQ2(E, Q2))

    def total_xsec(self, E):
        q2min = _Q2min(E, self.m_ups, self.MA)
        q2max = _Q2max(E, self.m_ups, self.MA)
        if q2max <= q2min:
            return 0.0
        result, _ = _integrate.quad(
            lambda q2: self._dsigma_dQ2(E, q2),
            q2min, q2max,
            limit=80, epsrel=1e-4,
        )
        return max(0.0, result)


# ===================================================================
#  ChiPrimeDecay  --  chi' -> chi + V1
# ===================================================================

class ChiPrimeDecay(DarkNewsDecay):
    """
    Two-body decay chi' -> chi + V1.
    Width: Gamma = (g_D^2 / 48 pi) m_chi' lambda^{3/2}(1, r_chi^2, r_V^2)
    """

    def __init__(
        self,
        m_chi,
        m_chi_prime,
        m_V1,
        g_D,
        *,
        pdgid_chi_prime=5918,
        pdgid_chi=5917,
        pdgid_V1=5922,
        table_dir=None,
    ):
        DarkNewsDecay.__init__(self)
        self.m_chi = m_chi
        self.m_chi_prime = m_chi_prime
        self.m_V1 = m_V1
        self.g_D = g_D

        self.pdgid_chi_prime = pdgid_chi_prime
        self.pdgid_chi = pdgid_chi
        self.pdgid_V1 = pdgid_V1

        self.table_dir = table_dir or "."
        os.makedirs(self.table_dir, exist_ok=True)
        self._total_width = self._compute_width()

    def _compute_width(self):
        p = _two_body_p_cm(self.m_chi_prime, self.m_chi, self.m_V1)
        if p <= 0.0:
            return 0.0
        return self.g_D**2 * p**3 / (6.0 * math.pi * self.m_chi_prime**2)

    def GetPossibleSignatures(self):
        sig = dataclasses.InteractionSignature()
        sig.primary_type = Particle.ParticleType(self.pdgid_chi_prime)
        sig.target_type = Particle.ParticleType.Decay
        sig.secondary_types = [
            Particle.ParticleType(self.pdgid_chi),
            Particle.ParticleType(self.pdgid_V1),
        ]
        return [sig]

    def GetPossibleSignaturesFromParent(self, primary_type):
        if int(primary_type) == self.pdgid_chi_prime:
            return self.GetPossibleSignatures()
        return []

    def TotalDecayWidth(self, arg1):
        if isinstance(arg1, dataclasses.InteractionRecord):
            primary = arg1.signature.primary_type
        else:
            primary = arg1
        if int(primary) != self.pdgid_chi_prime:
            return 0.0
        return self._total_width

    def TotalDecayWidthForFinalState(self, record):
        if int(record.signature.primary_type) != self.pdgid_chi_prime:
            return 0.0
        return self._total_width

    def DifferentialDecayWidth(self, record):
        if int(record.signature.primary_type) != self.pdgid_chi_prime:
            return 0.0
        return self._total_width / (4.0 * math.pi)

    def save_to_table(self, table_subdir=None):
        pass

    def SampleFinalState(self, record, random):
        P_parent = np.array(record.primary_momentum)
        p_cm = _two_body_p_cm(self.m_chi_prime, self.m_chi, self.m_V1)

        cos_theta = random.Uniform(-1.0, 1.0)
        phi = random.Uniform(0.0, 2.0 * math.pi)

        P_chi = _boost_to_lab(P_parent, p_cm, cos_theta, phi, self.m_chi)
        P_V1 = _boost_to_lab(P_parent, p_cm, -cos_theta, phi + math.pi, self.m_V1)

        for sec in record.get_secondary_particle_records():
            if int(sec.type) == self.pdgid_chi:
                sec.four_momentum = P_chi
                sec.mass = self.m_chi
            elif int(sec.type) == self.pdgid_V1:
                sec.four_momentum = P_V1
                sec.mass = self.m_V1
        return record


# ===================================================================
#  DarkPhotonDecay  --  V1 -> e- e+
# ===================================================================

class DarkPhotonDecay(DarkNewsDecay):
    """
    Two-body decay V1 -> e- e+.
    Width: Gamma = (alpha epsilon^2 m_V / 3) sqrt(1 - 4 m_e^2/m_V^2) (1 + 2 m_e^2/m_V^2)
    Angular distribution: dGamma/d(cos theta) ~ 1 + beta^2 cos^2(theta)
    """

    def __init__(
        self,
        m_V1,
        epsilon,
        *,
        pdgid_V1=5922,
        table_dir=None,
    ):
        DarkNewsDecay.__init__(self)
        self.m_V1 = m_V1
        self.epsilon = epsilon
        self.pdgid_V1 = pdgid_V1

        self.table_dir = table_dir or "."
        os.makedirs(self.table_dir, exist_ok=True)
        self._total_width = self._compute_width()

    def _compute_width(self):
        mV = self.m_V1
        me = _M_ELECTRON
        if mV < 2.0 * me:
            return 0.0
        beta = math.sqrt(max(1.0 - (2.0 * me / mV)**2, 0.0))
        return (_ALPHA_EM * self.epsilon**2 * mV / 3.0) * beta * (1.0 + 2.0 * me**2 / mV**2)

    def GetPossibleSignatures(self):
        sig = dataclasses.InteractionSignature()
        sig.primary_type = Particle.ParticleType(self.pdgid_V1)
        sig.target_type = Particle.ParticleType.Decay
        sig.secondary_types = [
            Particle.ParticleType.EMinus,
            Particle.ParticleType.EPlus,
        ]
        return [sig]

    def GetPossibleSignaturesFromParent(self, primary_type):
        if int(primary_type) == self.pdgid_V1:
            return self.GetPossibleSignatures()
        return []

    def TotalDecayWidth(self, arg1):
        if isinstance(arg1, dataclasses.InteractionRecord):
            primary = arg1.signature.primary_type
        else:
            primary = arg1
        if int(primary) != self.pdgid_V1:
            return 0.0
        return self._total_width

    def TotalDecayWidthForFinalState(self, record):
        if int(record.signature.primary_type) != self.pdgid_V1:
            return 0.0
        return self._total_width

    def DifferentialDecayWidth(self, record):
        if int(record.signature.primary_type) != self.pdgid_V1:
            return 0.0
        return self._total_width / (4.0 * math.pi)

    def save_to_table(self, table_subdir=None):
        pass

    def SampleFinalState(self, record, random):
        me = _M_ELECTRON
        p_cm = _two_body_p_cm(self.m_V1, me, me)
        beta = p_cm / math.sqrt(p_cm**2 + me**2) if p_cm > 0 else 0.0

        while True:
            cos_theta = random.Uniform(-1.0, 1.0)
            u = random.Uniform(0.0, 1.0 + beta**2)
            if u <= 1.0 + beta**2 * cos_theta**2:
                break

        phi = random.Uniform(0.0, 2.0 * math.pi)

        P_parent = np.array(record.primary_momentum)
        P_eminus = _boost_to_lab(P_parent, p_cm, cos_theta, phi, me)
        P_eplus = _boost_to_lab(P_parent, p_cm, -cos_theta, phi + math.pi, me)

        for sec in record.get_secondary_particle_records():
            if sec.type == Particle.ParticleType.EMinus:
                sec.four_momentum = P_eminus
                sec.mass = me
            elif sec.type == Particle.ParticleType.EPlus:
                sec.four_momentum = P_eplus
                sec.mass = me
        return record


# ===================================================================
#  Flux construction
# ===================================================================

def compute_chi_flux(
    m_meson,
    m_lepton,
    m_V1,
    m_chi,
    m_chi_prime,
    g_D,
    epsilon,
    flux_tag,
    min_energy,
    max_energy,
    physically_normalized=True,
):
    """
    Construct the chi (dark matter) flux at the detector by folding:
        neutrino flux -> parent meson energy -> three-body BR(meson -> l nu V1)
        -> V1 -> chi chi' kinematics -> chi energy spectrum.

    Returns a siren.distributions.TabulatedFluxDistribution.
    """
    import siren

    raw_flux = siren.utilities.load_flux(
        "PionKaon",
        tag=flux_tag,
        physically_normalized=physically_normalized,
    )

    available = m_meson - m_lepton
    if available <= m_V1:
        raise RuntimeError(
            "Channel kinematically forbidden: "
            f"m_meson={m_meson*1e3:.1f} MeV, m_lepton={m_lepton*1e3:.1f} MeV, "
            f"m_V1={m_V1*1e3:.1f} MeV"
        )

    E_nu_rf = (m_meson**2 - m_lepton**2) / (2.0 * m_meson)
    nu_to_meson = m_meson / E_nu_rf

    alpha_D = g_D**2 / (4.0 * math.pi)
    x = m_lepton / m_meson
    y = m_V1 / m_meson

    if (1.0 - x - y) <= 0:
        br_ratio = 0.0
    else:
        num = (1.0 - y**2)**2 * (1.0 + 2.0 * y**2)
        den = (1.0 - x**2)**2
        g_ps = num / den if den > 0 else 0.0
        br_ratio = 2.0 * (alpha_D / _ALPHA_EM) * epsilon**2 * g_ps

    E_chi_rf = (m_V1**2 + m_chi**2) / (2.0 * m_V1) if m_V1 > 0 else 0.0
    E_V_rest = (m_meson**2 + m_V1**2 - m_lepton**2) / (2.0 * m_meson)

    nu_energies = list(raw_flux.GetEnergyNodes())
    chi_energies = []
    chi_flux_vals = []

    for E_nu in nu_energies:
        E_meson = E_nu * nu_to_meson
        if E_meson < m_meson:
            continue

        gamma_meson = E_meson / m_meson
        E_V_lab = gamma_meson * E_V_rest

        gamma_V1 = E_V_lab / m_V1 if m_V1 > 0 else 1.0
        E_chi = gamma_V1 * E_chi_rf

        if E_chi < min_energy or E_chi > max_energy:
            continue

        nu_flux_at_E = raw_flux.EvaluatePDF(E_nu)
        chi_flux_at_E = nu_flux_at_E * br_ratio * 0.5

        chi_energies.append(E_chi)
        chi_flux_vals.append(chi_flux_at_E)

    if not chi_energies:
        chi_energies = [min_energy, max_energy]
        chi_flux_vals = [0.0, 0.0]

    return siren.distributions.TabulatedFluxDistribution(
        min_energy, max_energy, chi_energies, chi_flux_vals, physically_normalized
    )


def compute_chi_flux_from_dk2nu(
    dk2nu_data,
    parent_pdg,
    m_meson,
    m_lepton,
    m_V1,
    m_chi,
    m_chi_prime,
    g_D,
    epsilon,
    min_energy,
    max_energy,
    n_bins=100,
    physically_normalized=True,
):
    """
    Construct the chi flux from dk2nu parent meson data.

    Instead of using tabulated neutrino flux tables, this reads the
    parent meson energies and weights directly from dk2nu simulation
    output, giving a more accurate flux prediction that properly
    accounts for the beamline geometry and focusing.

    Parameters
    ----------
    dk2nu_data : dict
        Output of Dk2nuReader.read_dk2nu().
    parent_pdg : int
        PDG code of parent meson (211 for pi+, 321 for K+, etc.)
    m_meson, m_lepton, m_V1, m_chi, m_chi_prime : float
        Masses in GeV.
    g_D, epsilon : float
        Dark coupling and kinetic mixing.
    min_energy, max_energy : float
        Energy range for the output flux [GeV].
    n_bins : int
        Number of bins in the output flux table.

    Returns
    -------
    siren.distributions.TabulatedFluxDistribution
    """
    import siren

    mask = dk2nu_data["ptype"] == parent_pdg
    E_meson = dk2nu_data["E"][mask]
    weights = dk2nu_data["nimpwt"][mask]

    available = m_meson - m_lepton
    if available <= m_V1 or len(E_meson) == 0:
        energies = [min_energy, max_energy]
        flux_arr = [0.0, 0.0]
        return siren.distributions.TabulatedFluxDistribution(
            min_energy, max_energy, energies, flux_arr, physically_normalized
        )

    alpha_D = g_D**2 / (4.0 * math.pi)
    x = m_lepton / m_meson
    y = m_V1 / m_meson
    if (1.0 - x - y) <= 0:
        br_ratio = 0.0
    else:
        num = (1.0 - y**2)**2 * (1.0 + 2.0 * y**2)
        den = (1.0 - x**2)**2
        g_ps = num / den if den > 0 else 0.0
        br_ratio = 2.0 * (alpha_D / _ALPHA_EM) * epsilon**2 * g_ps

    E_chi_rf = (m_V1**2 + m_chi**2) / (2.0 * m_V1) if m_V1 > 0 else 0.0
    E_V_rest = (m_meson**2 + m_V1**2 - m_lepton**2) / (2.0 * m_meson)

    gamma_mesons = E_meson / m_meson
    E_V_lab = gamma_mesons * E_V_rest
    gamma_V1 = E_V_lab / m_V1 if m_V1 > 0 else np.ones_like(E_V_lab)
    E_chi = gamma_V1 * E_chi_rf

    chi_weights = weights * br_ratio * 0.5

    E_edges = np.linspace(min_energy, max_energy, n_bins + 1)
    hist, _ = np.histogram(E_chi, bins=E_edges, weights=chi_weights)
    dE = E_edges[1] - E_edges[0]
    E_centers = 0.5 * (E_edges[:-1] + E_edges[1:])

    pot = dk2nu_data.get("pot", 0.0)
    if pot > 0:
        flux_vals = hist / (dE * pot)
    else:
        flux_vals = hist / dE

    return siren.distributions.TabulatedFluxDistribution(
        min_energy, max_energy,
        list(E_centers), list(flux_vals),
        physically_normalized,
    )
