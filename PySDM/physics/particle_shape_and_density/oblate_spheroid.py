import numpy as np
from PySDM.physics.constants import PI_4_3

class OblateSpheroid:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def equivalent_radius(a: float, b: float, sigma_c_air: float, g: float, rho_c_l: float, rho_air: float) -> float: # Removed self, added a, b
        """
        Calculates an equivalent radius based on a formula involving surface tension,
        gravity, and densities.

        The formula is:
        r_eq = sqrt( (sigma_c_air / (g * (rho_c_l - rho_air))) *
                     (b/a)^(1/6) * sqrt( (b/a)^(-2) - 2 * (b/a)^(1/3) ) + 1 )
        
        Args:
            a (float): Semi-major axis (equatorial radius) [m]. Must be > 0.
            b (float): Semi-minor axis (polar radius) [m]. Must be > 0 and <= a.
            sigma_c_air (float): Surface tension between condensed phase and air [N/m].
            g (float): Acceleration due to gravity [m/s^2].
            rho_c_l (float): Density of the condensed phase (liquid/ice) [kg/m^3].
            rho_air (float): Density of air [kg/m^3].

        Returns:
            float: Equivalent radius [m].
        """
        ratio_b_a = b / a

        return ratio_b_a **(1/6) * np.sqrt((sigma_c_air / g * (rho_c_l - rho_air)) * 
                                           (ratio_b_a**(-2) - 2 * ratio_b_a**(1/3)) + 1)
            
    @staticmethod
    def volume(a: float, b: float) -> float: # Removed self, added a, b
        """
        Calculates the volume of the oblate spheroid.
        V = (4/3) * pi * a^2 * b
        
        Args:
            a (float): Semi-major axis (equatorial radius) [m].
            b (float): Semi-minor axis (polar radius) [m].
        Returns:
            float: Volume of the spheroid [m^3].
        """
        return PI_4_3 * (a**2) * b

    @staticmethod
    def reynolds_number(a: float, b: float, velocity_wrt_air: float, dynamic_viscosity: float, air_density: float, sigma_c_air: float, g: float, rho_c_l: float) -> float: # Removed self, added a, b
        """
        Calculates the Reynolds number for the oblate spheroid.
        Re = (air_density * velocity_wrt_air * L) / dynamic_viscosity
        The characteristic length L is taken as 2 * equivalent_radius().
        The equivalent_radius is calculated using the formula involving surface tension,
        gravity, and densities.

        Args:
            a (float): Semi-major axis (equatorial radius) [m].
            b (float): Semi-minor axis (polar radius) [m].
            velocity_wrt_air (float): Velocity of the particle relative to air [m/s].
            dynamic_viscosity (float): Dynamic viscosity of air [Pa*s or kg/(m*s)], also denoted eta_air.
            air_density (float): Density of air [kg/m^3], also denoted rho_air.
            sigma_c_air (float): Surface tension between condensed phase and air [N/m].
            g (float): Acceleration due to gravity [m/s^2].
            rho_c_l (float): Density of the condensed phase (liquid/ice) [kg/m^3].

        Returns:
            float: Reynolds number (dimensionless).
        """
        r_eq = OblateSpheroid.equivalent_radius(a=a, b=b, sigma_c_air=sigma_c_air, g=g, rho_c_l=rho_c_l, rho_air=air_density)
        characteristic_length = 2 * r_eq # L in the formula
        
        if dynamic_viscosity == 0: # avoid division by zero
            return np.inf if air_density * velocity_wrt_air * characteristic_length != 0 else 0.0

        return (air_density * velocity_wrt_air * characteristic_length) / dynamic_viscosity
