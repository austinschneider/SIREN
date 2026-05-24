"""
Backward-compatibility shim for siren.SIREN_DarkNews.

The canonical implementations now live in:
    siren.resources.processes.DarkNewsTables.DarkNewsCrossSection
    siren.resources.processes.DarkNewsTables.DarkNewsDecay
    siren.resources.processes.DarkNewsTables.VectorPortal
    siren.resources.processes.DarkNewsTables.MesonProduction
    siren.resources.processes.DarkNewsTables.processes

This module re-exports the key classes so that existing scripts
using `from siren.SIREN_DarkNews import ...` continue to work.
"""

import os
from siren import _util

_base = os.path.join(
    _util.resource_package_dir(), "processes", "DarkNewsTables"
)

_mod_xs = _util.load_module(
    "siren.resources.processes.DarkNewsTables.DarkNewsCrossSection",
    os.path.join(_base, "DarkNewsCrossSection.py"),
)
_mod_dec = _util.load_module(
    "siren.resources.processes.DarkNewsTables.DarkNewsDecay",
    os.path.join(_base, "DarkNewsDecay.py"),
)
_mod_proc = _util.load_module(
    "siren.resources.processes.DarkNewsTables.processes",
    os.path.join(_base, "processes.py"),
)

PyDarkNewsCrossSection = _mod_xs.PyDarkNewsCrossSection
PyDarkNewsDecay = _mod_dec.PyDarkNewsDecay

# Re-export the legacy interaction-collection class for old scripts
# that used the SIREN_Controller workflow.
try:
    from siren.DNModelContainer import ModelContainer
    from DarkNews.nuclear_tools import NuclearTarget

    cross_section_kwarg_keys = [
        "tolerance",
        "interp_tolerance",
        "always_interpolate",
    ]


    class PyDarkNewsInteractionCollection:
        """Legacy wrapper -- prefer DarkNewsTables.processes.load_processes()."""

        def __init__(self, table_dir=None, **kwargs):
            self.table_dir = table_dir
            self.models = ModelContainer(**kwargs)
            self.cross_sections = {}
            self.decays = {}

        def GenerateCrossSections(self, use_pickles=False, **kwargs):
            xs_kwargs = {
                k: v for k, v in kwargs.items() if k in cross_section_kwarg_keys
            }
            for ups_key, ups_case in self.models.ups_cases.items():
                self.cross_sections[ups_key] = PyDarkNewsCrossSection(
                    ups_case, **xs_kwargs
                )
            return self.cross_sections

        def GenerateDecays(self, use_pickles=False, **kwargs):
            for dec_key, dec_case in self.models.dec_cases.items():
                self.decays[dec_key] = PyDarkNewsDecay(
                    dec_case, table_dir=self.table_dir
                )
            return self.decays

except ImportError:
    pass

# Vector-portal and meson-production re-exports
try:
    _mod_vp = _util.load_module(
        "siren.resources.processes.DarkNewsTables.VectorPortal",
        os.path.join(_base, "VectorPortal.py"),
    )
    VectorPortalUpsCase = _mod_vp.VectorPortalUpsCase
    ChiPrimeDecay = _mod_vp.ChiPrimeDecay
    DarkPhotonDecay = _mod_vp.DarkPhotonDecay
    compute_chi_flux = _mod_vp.compute_chi_flux
except Exception:
    pass

try:
    _mod_mp = _util.load_module(
        "siren.resources.processes.DarkNewsTables.MesonProduction",
        os.path.join(_base, "MesonProduction.py"),
    )
    MesonThreeBodyDecay = _mod_mp.MesonThreeBodyDecay
    MesonSimpleDecay = _mod_mp.MesonSimpleDecay
    build_phi_flux = _mod_mp.build_phi_flux
except Exception:
    pass
