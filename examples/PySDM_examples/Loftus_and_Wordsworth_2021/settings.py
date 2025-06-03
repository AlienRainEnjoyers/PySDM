#Planetary Properties, Loftus and Wordsworth 2021 Table 1
from PySDM.physics import constants_defaults, si

import numpy as np
from pystrict import strict
from types import SimpleNamespace

from PySDM import Formulae
from PySDM.dynamics import condensation
from PySDM.physics.constants import si
from PySDM_examples.Loftus_and_Wordsworth_2021.planet import Planet


@strict
class Settings:
    def __init__(
        self,
        updraft_velocity: float,
        r_dry: float,
        r_wet: float,
        formulae: Formulae,
        planet: Planet,
    ):
        self.formulae = formulae or Formulae()
        const = self.formulae.constants
        self.planet = planet
        self.p0 = planet.p_STP
        self.RH0 = planet.RH_zref
        self.kappa = 0.2
        self.T0 = planet.T_STP

        pvs = self.formulae.saturation_vapour_pressure.pvs_water(self.T0)
        self.initial_water_vapour_mixing_ratio = const.eps / (
            self.p0 / self.RH0 / pvs - 1
        )
        self.r_dry = r_dry
        self.initial_radius = r_wet
        self.n_output = 500

        self.rtol_x = condensation.DEFAULTS.rtol_x
        self.rtol_thd = condensation.DEFAULTS.rtol_thd
        self.coord = "volume logarithm"
        self.dt_cond_range = condensation.DEFAULTS.cond_range
        self.w = updraft_velocity
