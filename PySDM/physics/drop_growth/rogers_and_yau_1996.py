"""
Computes the rate of change of raindrop equivalent radius (dreq/dt) based on the formulation
from Rogers & Yau (1996), Chapter 7.
"""

from PySDM.physics.saturation_vapour_pressure import Bolton1980


class RogersAndYau1996:
    def __init__(self,_):
        pass

    @staticmethod
    def dreq_dt(const, req, rho_c_l, mu_c, T_air, T_drop, RH, D_c_air, fv_mol):
        """
        Calculate the drop growth rate based on the Lofus et al. (2021) model.

        Parameters:
        f_v_mol (float): Vapor molar fraction.
        D_c_air (float): Diffusion coefficient of water vapor in air.
        mu_c (float): Dynamic viscosity of the droplet.
        r_eq (float): Equilibrium radius of the droplet.
        ro_c_l (float): Density of liquid water.
        R (float): Universal gas constant.
        RH (float): Relative humidity.
        T_air (float): Temperature of the air.
        T_drop (float): Temperature of the droplet.

        Returns:
        float: The drop growth rate.
        """
        p_sat_air = Bolton1980.pvs_water(const, T_air)
        p_sat_drop = Bolton1980.pvs_water(const, T_drop)

        numerator = fv_mol * D_c_air * mu_c
        denominator = req * rho_c_l * const.R_str

        vapor_gradient = RH * (p_sat_air / T_air) - (p_sat_drop / T_drop)

        return (numerator / denominator) * vapor_gradient
    