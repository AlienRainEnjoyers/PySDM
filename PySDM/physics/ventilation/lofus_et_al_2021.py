import numpy as np
from PySDM.physics.constants import PI_4_3, ONE_HALF,THREE,TWO

# eq. 3 and 4 at https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020JE006653
class LofusEtAl2021:
    @staticmethod
    def gravity_force(r_eq, ro_c_l, ro_air, g):
        """
        Calculate the gravity force on a particle.

        Parameters:
        r_eq (float): Equivalent radius of the particle.
        ro_c_l (float): Density of the particle.
        ro_air (float): Density of the air.
        g (float): Gravitational acceleration.

        Returns:
        float: The calculated gravity force.
        """
        return PI_4_3 * r_eq**THREE * (ro_c_l - ro_air) * g
    
    @staticmethod
    def drag_force(C_d, A, ro_air, v):
        """
        Calculate the drag force on a particle.

        Parameters:
        C_d (float): Coefficient of the raindrop.
        A (float): Cros-sectional area of the raindrop.
        ro_air (float): Local density of the air.
        v (float): Relative velocity of the particle.

        Returns:
        float: The calculated drag force.
        """
        return ONE_HALF * C_d * A * ro_air * v**TWO
    