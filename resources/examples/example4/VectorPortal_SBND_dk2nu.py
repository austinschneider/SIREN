"""
Vector-portal dark matter at SBND, starting from pions.

Injection chain (pion-first):
    1. Read pi+ kinematics and decay vertices from BNB dk2nu ROOT files
    2. Inject pi+ at its decay vertex via PrimaryExternalDistribution
    3. pi+ -> mu+ nu_mu V1   (three-body decay, Carlson-Rislow matrix element)
    4. V1 -> chi chi'         (dark photon decay to DM pair)
    5. chi propagates to SBND
    6. chi + Ar -> chi' + Ar  (coherent upscattering)
    7. chi' -> chi + V1       (de-excitation)
    8. V1 -> e+ e-            (visible signal)

This preserves the full pion kinematics and angular correlations from
the beamline simulation, rather than pre-computing a chi energy spectrum.

Reference: Dutta et al., PRL 129, 111803 (2022) [arXiv:2110.11944]
"""

import os
import sys
import glob

import numpy as np
import siren
from siren import utilities
from siren._util import GenerateEvents

# ---------------------------------------------------------------------------
# Import model classes via SIREN's module loader
# ---------------------------------------------------------------------------
from siren import _util as _siren_util

_dt_base = os.path.join(
    _siren_util.resource_package_dir(), "processes", "DarkNewsTables",
)

_mod_mp = _siren_util.load_module(
    "siren.resources.processes.DarkNewsTables.MesonProduction",
    os.path.join(_dt_base, "MesonProduction.py"),
)
_mod_vp = _siren_util.load_module(
    "siren.resources.processes.DarkNewsTables.VectorPortal",
    os.path.join(_dt_base, "VectorPortal.py"),
)
_mod_xs = _siren_util.load_module(
    "siren.resources.processes.DarkNewsTables.DarkNewsCrossSection",
    os.path.join(_dt_base, "DarkNewsCrossSection.py"),
)
_mod_dk = _siren_util.load_module(
    "siren.resources.processes.DarkNewsTables.Dk2nuReader",
    os.path.join(_dt_base, "Dk2nuReader.py"),
)

MesonThreeBodySIRENDecay = _mod_mp.MesonThreeBodySIRENDecay
VectorPortalUpsCase = _mod_vp.VectorPortalUpsCase
ChiPrimeDecay = _mod_vp.ChiPrimeDecay
DarkPhotonDecay = _mod_vp.DarkPhotonDecay
PyDarkNewsCrossSection = _mod_xs.PyDarkNewsCrossSection
read_dk2nu = _mod_dk.read_dk2nu
dk2nu_to_primary_distribution = _mod_dk.dk2nu_to_primary_distribution
print_summary = _mod_dk.print_summary
PTYPE_PIPLUS = _mod_dk.PTYPE_PIPLUS

# ---------------------------------------------------------------------------
# Model parameters (Dutta et al. Table I, double-mediator)
# ---------------------------------------------------------------------------
M_CHI       = 8e-3    # GeV
M_CHI_PRIME = 50e-3   # GeV
M_V1        = 17e-3   # GeV  (light dark photon)
M_V2        = 200e-3  # GeV  (heavy upscattering mediator)
G_D         = 1.0
EPSILON_1   = 7e-5    # kinetic mixing for V1
EPSILON_2   = 1e-4    # kinetic mixing for V2
G_MU        = 1e-3    # Yukawa coupling of V1/phi to muon

M_PION = 0.13957039
M_MUON = 0.10565837

PDGID_PION      = 211
PDGID_MUPLUS    = -13
PDGID_NUMU      = 14
PDGID_V1        = 5922
PDGID_CHI       = 5917
PDGID_CHI_PRIME = 5918

events_to_inject = 1000

# ---------------------------------------------------------------------------
# 1. Read dk2nu files and convert to CSV
# ---------------------------------------------------------------------------
dk2nu_dir = os.environ.get(
    "DK2NU_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "G4BNB"),
)
dk2nu_files = sorted(glob.glob(os.path.join(dk2nu_dir, "*dk2nu*.root")))
if not dk2nu_files:
    dk2nu_files = sorted(glob.glob(os.path.join(dk2nu_dir, "nubeam*.root")))

if not dk2nu_files:
    print(f"No dk2nu files found in {dk2nu_dir}")
    print("Set DK2NU_DIR environment variable or place files in sources/G4BNB/")
    sys.exit(1)

print(f"Reading {len(dk2nu_files)} dk2nu file(s) ...")
dk2nu_data = read_dk2nu(dk2nu_files, parent_pdg=[PTYPE_PIPLUS])
print_summary(dk2nu_data)

# ---------------------------------------------------------------------------
# 2. Load SBND detector (needed before CSV so we know the coordinate transform)
# ---------------------------------------------------------------------------
print("\nLoading SBND detector model ...")
detector_model = utilities.load_detector("SBN", detector="SBND")

# dk2nu positions are in BNB (geometry) coordinates (cm).
# SIREN's injector works in detector-local coordinates (m).
pion_type      = siren.dataclasses.Particle.ParticleType(PDGID_PION)
v1_type        = siren.dataclasses.Particle.ParticleType(PDGID_V1)
chi_type       = siren.dataclasses.Particle.ParticleType(PDGID_CHI)
chi_prime_type = siren.dataclasses.Particle.ParticleType(PDGID_CHI_PRIME)

# ---------------------------------------------------------------------------
# 3. Set up processes
# ---------------------------------------------------------------------------
print("Setting up processes ...")

# Primary: pion three-body decay  pi+ -> mu+ nu_mu V1
pion_decay = MesonThreeBodySIRENDecay(
    m_meson=M_PION,
    m_lepton=M_MUON,
    m_mediator=M_V1,
    g_mu=G_MU,
    mediator_type="scalar",
    pdgid_meson=PDGID_PION,
    pdgid_lepton=PDGID_MUPLUS,
    pdgid_neutrino=PDGID_NUMU,
    pdgid_mediator=PDGID_V1,
)
print(f"  Pion 3-body width: {pion_decay._total_width:.4e} GeV")

# V1 -> chi chi' (if kinematically allowed) or V1 -> e+e-
v1_to_chi = None
if M_V1 > M_CHI + M_CHI_PRIME:
    v1_to_chi = ChiPrimeDecay(
        M_CHI, M_CHI_PRIME, M_V1, G_D,
        pdgid_chi_prime=PDGID_CHI_PRIME,
        pdgid_chi=PDGID_CHI,
        pdgid_V1=PDGID_V1,
    )
v1_to_ee = DarkPhotonDecay(M_V1, EPSILON_1, pdgid_V1=PDGID_V1)

# chi N -> chi' N upscattering
ups_case = VectorPortalUpsCase(
    m_chi=M_CHI,
    m_chi_prime=M_CHI_PRIME,
    m_V=M_V2,
    g_D=G_D,
    epsilon=EPSILON_2,
    nuclear_pdgid=1000180400,
    nuclear_mass=37.215,
    nuclear_name="Ar40",
    A=40, Z=18,
)
xs = PyDarkNewsCrossSection(ups_case, always_interpolate=True)
print(f"  chi upscattering threshold: {ups_case.Ethreshold:.4f} GeV")

# chi' -> chi V1
chi_prime_decay = ChiPrimeDecay(
    M_CHI, M_CHI_PRIME, M_V1, G_D,
    pdgid_chi_prime=PDGID_CHI_PRIME,
    pdgid_chi=PDGID_CHI,
    pdgid_V1=PDGID_V1,
)
print(f"  chi' decay width: {chi_prime_decay._total_width:.4e} GeV")

# Assemble process collections
primary_processes = {pion_type: [pion_decay]}

secondary_processes = {
    v1_type: [v1_to_ee],
    chi_type: [xs],
    chi_prime_type: [chi_prime_decay],
}
if v1_to_chi is not None:
    secondary_processes[v1_type].insert(0, v1_to_chi)

# ---------------------------------------------------------------------------
# 4. Distributions
# ---------------------------------------------------------------------------
primary_dist = dk2nu_to_primary_distribution(dk2nu_data, detector_model, parent_pdg=[PTYPE_PIPLUS])
print(f"  Loaded {primary_dist.GetPhysicalNumEvents()} pion events")

primary_injection_distributions = [primary_dist]
primary_physical_distributions = [primary_dist]

secondary_injection_distributions = {}
for sec_type in secondary_processes.keys():
    secondary_injection_distributions[sec_type] = [
        siren.distributions.SecondaryBoundedVertexDistribution()
    ]

# ---------------------------------------------------------------------------
# 5. Stopping condition
# ---------------------------------------------------------------------------
def stop(datum, i):
    secondary_type = datum.record.signature.secondary_types[i]
    if secondary_type == siren.dataclasses.Particle.ParticleType.EMinus:
        return True
    if secondary_type == siren.dataclasses.Particle.ParticleType.EPlus:
        return True
    if secondary_type == siren.dataclasses.Particle.ParticleType.MuPlus:
        return True
    if secondary_type == siren.dataclasses.Particle.ParticleType.MuMinus:
        return True
    if secondary_type == siren.dataclasses.Particle.ParticleType(PDGID_NUMU):
        return True
    return False

# ---------------------------------------------------------------------------
# 6. Run
# ---------------------------------------------------------------------------
print(f"\nInjecting {events_to_inject} events ...")

injector = siren.injection.Injector()
injector.number_of_events = events_to_inject
injector.detector_model = detector_model
injector.primary_type = pion_type
injector.primary_interactions = primary_processes[pion_type]
injector.primary_injection_distributions = primary_injection_distributions
injector.secondary_interactions = secondary_processes
injector.secondary_injection_distributions = secondary_injection_distributions
injector.stopping_condition = stop

events, gen_times = GenerateEvents(injector)
print(f"Generated {len(events)} events in {sum(gen_times):.1f} s")
