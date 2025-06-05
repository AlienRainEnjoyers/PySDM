#Planetary Properties, Loftus and Wordsworth 2021 Table 1
from PySDM.physics import constants_defaults, si

import numpy as np
from pystrict import strict
from types import SimpleNamespace

from PySDM import Formulae
from PySDM.dynamics import condensation
from PySDM.physics.constants import si
from PySDM_examples.Loftus_and_Wordsworth_2021.planet import Planet
from scipy.optimize import fsolve


@strict
class Settings:
    def __init__(
        self,
        # w_avg: float,
        r_wet: float,
        mass_of_dry_air: float,
        planet: Planet,
        coord: str = "WaterMassLogarithm",
        formulae: Formulae = None,
    ):
        self.formulae = formulae or Formulae(
            saturation_vapour_pressure="AugustRocheMagnus",
            diffusion_coordinate=coord,
        )
        const = self.formulae.constants
        self.p0 = planet.p_STP
        self.RH0 = planet.RH_zref
        self.kappa = 0.2
        self.T0 = planet.T_STP
        self.z_half = 150 * si.metres
        self.dt = 1 *si.second

        pvs = self.formulae.saturation_vapour_pressure.pvs_water(self.T0)
        self.initial_water_vapour_mixing_ratio = const.eps / (
            self.p0 / self.RH0 / pvs - 1
        )

        Rair = (self.formulae.constants.Rv / (1 / self.initial_water_vapour_mixing_ratio + 1)
                + self.formulae.constants.Rd / (1 + self.initial_water_vapour_mixing_ratio)
            )
        c_p = self.formulae.constants.c_pv / (1 / self.initial_water_vapour_mixing_ratio + 1) + self.formulae.constants.c_pd / (
                    1 + self.initial_water_vapour_mixing_ratio
                )
        def f(x):
            # return x -273
            return self.initial_water_vapour_mixing_ratio/(self.initial_water_vapour_mixing_ratio+ const.eps)*self.p0*(x/self.T0)**(c_p/Rair
            ) - self.formulae.saturation_vapour_pressure.pvs_water(x)
        tdews = (fsolve(f, [150,300]))
        self.Tcloud = np.max(tdews)
        self.Zcloud = (self.T0-self.Tcloud)*c_p/self.formulae.constants.g_std
        thstd =self.formulae.trivia.th_std(self.p0, self.T0)

        self.pcloud = self.formulae.hydrostatics.p_of_z_assuming_const_th_and_initial_water_vapour_mixing_ratio(
            self.p0, thstd, self.initial_water_vapour_mixing_ratio, self.Zcloud)

        np.testing.assert_approx_equal(
            actual=self.pcloud*(self.initial_water_vapour_mixing_ratio/(self.initial_water_vapour_mixing_ratio + const.eps))/
              self.formulae.saturation_vapour_pressure.pvs_water(self.Tcloud),
            desired=1,
            significant=4
        )

        # self.w_avg = w_avg
        self.r_wet = r_wet
        # self.N_STP = N_STP
        # self.n_in_dv = N_STP / const.rho_STP * mass_of_dry_air
        self.mass_of_dry_air = mass_of_dry_air
        self.n_output = 500

        self.rtol_x = condensation.DEFAULTS.rtol_x
        self.rtol_thd = condensation.DEFAULTS.rtol_thd
        self.coord = "volume logarithm"
        self.dt_cond_range = condensation.DEFAULTS.cond_range