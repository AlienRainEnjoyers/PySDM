import numpy as np
from PySDM.physics.constants import TWO_THIRDS

# eq. 8 at https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020JE006653
class LofusEtAl2021:
    @staticmethod
    def terminal_velocity(rho_c_l, rho_air, g, C_D, b_a_ratio, r_eq):
        """
        Calculates terminal velocity based on Lofus et al. 2021, eq. 8.

        Parameters:
            rho_c_l (float): Density of condensed water (liquid or ice) [kg/m^3]
            rho_air (float): Density of air [kg/m^3]
            g (float): Acceleration due to gravity [m/s^2]
            C_D (float): Drag coefficient (dimensionless)
            b_a_ratio (float): Ratio of semi-minor to semi-major axes (b/a) (dimensionless)
            r_eq (float): Equivalent radius [m]

        Returns:
            float: Terminal velocity [m/s] (negative sign indicates downward motion)
        """
        
        return -np.sqrt(
            (8/3) *
            (rho_c_l - rho_air) / rho_air *
            const.g_std / C_D *
            b_a_ratio**TWO_THIRDS *
            r_eq
        )
    
    def v_term(const, radius):
        return np.where(
            radius < const.ROGERS_YAU_TERM_VEL_SMALL_R_LIMIT,
            const.ROGERS_YAU_TERM_VEL_SMALL_K * radius**const.TWO,
            np.where(
                radius < const.ROGERS_YAU_TERM_VEL_MEDIUM_R_LIMIT,
                const.ROGERS_YAU_TERM_VEL_MEDIUM_K * radius,
                const.ROGERS_YAU_TERM_VEL_LARGE_K * radius**const.ONE_HALF,
            ),
        )