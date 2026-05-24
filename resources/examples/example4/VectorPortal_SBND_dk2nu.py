"""
Vector-portal dark matter at SBND using dk2nu pion flux.

Full injection chain:
    1. Read pi+ kinematics from BNB dk2nu ROOT files
    2. Compute chi flux via pi+ -> mu+ nu V1, V1 -> chi chi'
    3. Inject chi into SBND detector
    4. chi + Ar -> chi' + Ar  (coherent upscattering)
    5. chi' -> chi + V1       (de-excitation)
    6. V1 -> e+ e-            (visible signal)

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
# Import Dutta-Kim model classes from DarkNewsTables
# ---------------------------------------------------------------------------
from siren.resources.processes.DarkNewsTables.VectorPortal import (
    VectorPortalUpsCase,
    ChiPrimeDecay,
    DarkPhotonDecay,
    compute_chi_flux_from_dk2nu,
)
from siren.resources.processes.DarkNewsTables.DarkNewsCrossSection import (
    PyDarkNewsCrossSection,
)
from siren.resources.processes.DarkNewsTables.Dk2nuReader import (
    read_dk2nu,
    print_summary,
    PTYPE_PIPLUS,
)

# ---------------------------------------------------------------------------
# Model parameters (Dutta et al. Table I, double-mediator)
# ---------------------------------------------------------------------------
M_CHI       = 8e-3    # GeV  chi   (DM ground state)
M_CHI_PRIME = 50e-3   # GeV  chi'  (DM excited state)
M_V1        = 17e-3   # GeV  V1    (light dark photon, production + decay)
M_V2        = 200e-3  # GeV  V2    (heavy mediator, upscattering)
G_D         = 1.0
EPSILON_1   = 7e-5    # kinetic mixing for V1
EPSILON_2   = 1e-4    # kinetic mixing for V2

PDGID_CHI       = 5917
PDGID_CHI_PRIME = 5918
PDGID_V1        = 5922

# Production channel: pi+ -> mu+ nu_mu V1
M_PION = 0.13957     # GeV
M_MUON = 0.10566     # GeV

events_to_inject = 1000

# ---------------------------------------------------------------------------
# 1. Read dk2nu files
# ---------------------------------------------------------------------------
dk2nu_dir = os.environ.get(
    "DK2NU_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "G4BNB"),
)
dk2nu_files = sorted(glob.glob(os.path.join(dk2nu_dir, "*.dk2nu.root")))
if not dk2nu_files:
    dk2nu_files = sorted(glob.glob(os.path.join(dk2nu_dir, "nubeam*.root")))

if not dk2nu_files:
    print(f"No dk2nu files found in {dk2nu_dir}")
    print("Set DK2NU_DIR environment variable or place files in sources/G4BNB/")
    sys.exit(1)

print(f"Reading {len(dk2nu_files)} dk2nu file(s) from {dk2nu_dir} ...")
dk2nu_data = read_dk2nu(dk2nu_files, parent_pdg=[PTYPE_PIPLUS])
print_summary(dk2nu_data)

# ---------------------------------------------------------------------------
# 2. Load SBN/SBND detector
# ---------------------------------------------------------------------------
print("\nLoading SBND detector model ...")
detector_model = utilities.load_detector("SBN", detector="SBND")

chi_type       = siren.dataclasses.Particle.ParticleType(PDGID_CHI)
chi_prime_type = siren.dataclasses.Particle.ParticleType(PDGID_CHI_PRIME)
v1_type        = siren.dataclasses.Particle.ParticleType(PDGID_V1)

# ---------------------------------------------------------------------------
# 3. Build chi flux from dk2nu pion spectrum
# ---------------------------------------------------------------------------
print("Computing chi flux from dk2nu pion spectrum ...")

# Upscattering threshold determines minimum chi energy
ups_threshold = ((M_CHI_PRIME + 37.215)**2 - M_CHI**2 - 37.215**2) / (2.0 * 37.215)

chi_flux = compute_chi_flux_from_dk2nu(
    dk2nu_data=dk2nu_data,
    parent_pdg=PTYPE_PIPLUS,
    m_meson=M_PION,
    m_lepton=M_MUON,
    m_V1=M_V1,
    m_chi=M_CHI,
    m_chi_prime=M_CHI_PRIME,
    g_D=G_D,
    epsilon=EPSILON_1,
    min_energy=ups_threshold,
    max_energy=3.0,
    n_bins=100,
)

# ---------------------------------------------------------------------------
# 4. Set up cross sections (chi + Ar40 -> chi' + Ar40)
# ---------------------------------------------------------------------------
print("Setting up upscattering cross sections ...")

ups_case = VectorPortalUpsCase(
    m_chi=M_CHI,
    m_chi_prime=M_CHI_PRIME,
    m_V=M_V2,
    g_D=G_D,
    epsilon=EPSILON_2,
    nuclear_pdgid=1000180400,
    nuclear_mass=37.215,
    nuclear_name="Ar40",
    A=40,
    Z=18,
)
xs = PyDarkNewsCrossSection(ups_case, always_interpolate=True)

primary_processes = {chi_type: [xs]}

# ---------------------------------------------------------------------------
# 5. Set up decay chain
# ---------------------------------------------------------------------------
chi_prime_decay = ChiPrimeDecay(
    M_CHI, M_CHI_PRIME, M_V1, G_D,
    pdgid_chi_prime=PDGID_CHI_PRIME,
    pdgid_chi=PDGID_CHI,
    pdgid_V1=PDGID_V1,
)
v1_decay = DarkPhotonDecay(M_V1, EPSILON_1, pdgid_V1=PDGID_V1)

print(f"chi' decay width: {chi_prime_decay._total_width:.4e} GeV")
print(f"V1 decay width:   {v1_decay._total_width:.4e} GeV")

secondary_processes = {
    chi_prime_type: [chi_prime_decay],
    v1_type: [v1_decay],
}

# ---------------------------------------------------------------------------
# 6. Distributions
# ---------------------------------------------------------------------------
mass_dist = siren.distributions.PrimaryMass(M_CHI)
direction_dist = siren.distributions.FixedDirection(
    siren.math.Vector3D(0, 0, 1.0)
)

chi_flux_gen = compute_chi_flux_from_dk2nu(
    dk2nu_data=dk2nu_data,
    parent_pdg=PTYPE_PIPLUS,
    m_meson=M_PION,
    m_lepton=M_MUON,
    m_V1=M_V1,
    m_chi=M_CHI,
    m_chi_prime=M_CHI_PRIME,
    g_D=G_D,
    epsilon=EPSILON_1,
    min_energy=ups_threshold,
    max_energy=3.0,
    n_bins=100,
    physically_normalized=False,
)

DN_min_decay_width = min(chi_prime_decay._total_width, v1_decay._total_width)

decay_range_func = siren.distributions.DecayRangeFunction(
    M_CHI, DN_min_decay_width, 3, 110  # SBND baseline ~110 m from BNB target
)
position_dist = siren.distributions.DecayRangePositionDistribution(
    2.0, 2.0, decay_range_func,
)

primary_injection_distributions = [
    mass_dist,
    chi_flux_gen,
    direction_dist,
    position_dist,
]

primary_physical_distributions = [
    chi_flux,
    direction_dist,
]

secondary_injection_distributions = {}
for sec_type in secondary_processes.keys():
    secondary_injection_distributions[sec_type] = [
        siren.distributions.SecondaryBoundedVertexDistribution()
    ]

# ---------------------------------------------------------------------------
# 7. Stopping condition
# ---------------------------------------------------------------------------
def stop(datum, i):
    secondary_type = datum.record.signature.secondary_types[i]
    if secondary_type == siren.dataclasses.Particle.ParticleType.EMinus:
        return True
    if secondary_type == siren.dataclasses.Particle.ParticleType.EPlus:
        return True
    if secondary_type == chi_type:
        return True
    return False

# ---------------------------------------------------------------------------
# 8. Run injector
# ---------------------------------------------------------------------------
print(f"\nInjecting {events_to_inject} events ...")

injector = siren.injection.Injector()
injector.number_of_events = events_to_inject
injector.detector_model = detector_model
injector.primary_type = chi_type
injector.primary_interactions = primary_processes[chi_type]
injector.primary_injection_distributions = primary_injection_distributions
injector.secondary_interactions = secondary_processes
injector.secondary_injection_distributions = secondary_injection_distributions
injector.stopping_condition = stop

events, gen_times = GenerateEvents(injector)
print(f"Generated {len(events)} events in {sum(gen_times):.1f} s")

# ---------------------------------------------------------------------------
# 9. Print summary
# ---------------------------------------------------------------------------
if events:
    print(f"\nFirst event primary energy: {events[0].record.primary_momentum[0]:.4f} GeV")
    print(f"Number of secondaries: {len(events[0].record.secondary_momenta)}")
