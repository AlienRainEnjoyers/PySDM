#Planetary Properties, Loftus and Wordsworth 2021 Table 1
from PySDM.physics import constants_defaults, si


import numpy as np
from pystrict import strict
from types import SimpleNamespace

from PySDM import Formulae
from PySDM.dynamics import condensation
from PySDM.physics.constants import si


@strict
class Settings:
    def __init__(
        self,
        updraft_velocity: float,
        r_dry: float,
        r_wet: float,
        formulae: Formulae,
        Planet: dict,

    ):
        self.formulae = formulae or Formulae()
        const = self.formulae.constants
        self.p0 = Planet["p_STP"]
        self.RH0 = Planet["RH_zref"]
        self.kappa = 0.2
        self.T0 = Planet["T_STP"]

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








EARTH_LIKE_CONSTS = {

    "T_STP":300 * si.kelvin, #surf or ref
    "p_STP": 1.01325 * 10e5 * si.pascal,
    "RH_zref": 0.75,
    "dry_molar_conc_H2":0, #mol/mol dry air
    "dry_molar_conc_He":0, #mol/mol dry air
    "dry_molar_conc_N2":1, #mol/mol dry air
    "dry_molar_conc_O2":0, #mol/mol dry air
    "dry_molar_conc_CO2":0, #mol/mol dry air
    "H_LCL": 8.97 * si.kilometre, 
}

EARTH_CONSTS = {
    "g_std": 9.82 * si.metre / si.second**2,
    "T_STP":290 * si.kelvin, #surf or ref
    "p_STP": 1.01325 * 10e5 * si.pascal,
    "RH_zref": 0.75,
    "dry_molar_conc_H2":0, #mol/mol dry air
    "dry_molar_conc_He":0, #mol/mol dry air
    "dry_molar_conc_N2":0.8, #mol/mol dry air
    "dry_molar_conc_O2":0.2, #mol/mol dry air
    "dry_molar_conc_CO2":0, #mol/mol dry air
    "H_LCL": 8.41 * si.kilometre, 

}


EARLY_MARS_CONSTS = {
    "g_std": 3.71 * si.metre / si.second**2,
    "T_STP":290 * si.kelvin, #surf or ref
    "p_STP": 2 * 10e5 * si.pascal,
    "RH_zref": 0.75,
    "dry_molar_conc_H2":0, #mol/mol dry air
    "dry_molar_conc_He":0, #mol/mol dry air
    "dry_molar_conc_N2":0, #mol/mol dry air
    "dry_molar_conc_O2":0, #mol/mol dry air
    "dry_molar_conc_CO2":1, #mol/mol dry air
    "H_LCL": 14.5 * si.kilometre, 
}

JUPITER_CONSTS = {
    "g_std": 24.84 * si.metre / si.second**2,
    "T_STP": 274 * si.kelvin, #surf or ref
    "p_STP": 4.85 * 10e5 * si.pascal,
    "RH_zref": 1,
    "dry_molar_conc_H2":0.864, #mol/mol dry air
    "dry_molar_conc_He":0.136, #mol/mol dry air
    "dry_molar_conc_N2":0, #mol/mol dry air
    "dry_molar_conc_O2":0, #mol/mol dry air
    "dry_molar_conc_CO2":0, #mol/mol dry air
    "H_LCL": 39.8 * si.kilometre, 


}

SATURN_CONSTS = {
    "g_std": 10.47 * si.metre / si.second**2,
    "T_STP": 284 * si.kelvin, #surf or ref
    "p_STP": 10.4 * 10e5 * si.pascal,
    "RH_zref": 1,
    "dry_molar_conc_H2":0.88, #mol/mol dry air
    "dry_molar_conc_He":0.12, #mol/mol dry air
    "dry_molar_conc_N2":0, #mol/mol dry air
    "dry_molar_conc_O2":0, #mol/mol dry air
    "dry_molar_conc_CO2":0, #mol/mol dry air
    "H_LCL": 99.2 * si.kilometre, 

}

K2_18B_CONSTS = {
    "g_std": 12.44 * si.metre / si.second**2,
    "T_STP": 275 * si.kelvin, #surf or ref
    "p_STP": 0.1 * 10e5 * si.pascal,
    "RH_zref": 1,
    "dry_molar_conc_H2":0.9, #mol/mol dry air
    "dry_molar_conc_He":0.1, #mol/mol dry air
    "dry_molar_conc_N2":0, #mol/mol dry air
    "dry_molar_conc_O2":0, #mol/mol dry air
    "dry_molar_conc_CO2":0, #mol/mol dry air
    "H_LCL": 56.6 * si.kilometre, 

}

COMPOSITION_CONSTS = {
    "g_std": 9.82 * si.metre / si.second**2,
    "T_STP": 275 * si.kelvin,  # surf or ref
    "p_STP": 0.75 * 10e5 * si.pascal,
    "RH_zref": 1,
    "dry_molar_conc_H2":0.1, #mol/mol dry air
    "dry_molar_conc_He":0.1, #mol/mol dry air
    "dry_molar_conc_N2":0.1, #mol/mol dry air
    "dry_molar_conc_O2":0.1, #mol/mol dry air
    "dry_molar_conc_CO2":0.1, #mol/mol dry air
}


USER_PLANET_BROAD_CONSTS = {
}