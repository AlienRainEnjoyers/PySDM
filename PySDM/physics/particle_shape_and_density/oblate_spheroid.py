import numpy as np

from PySDM.physics.constants import PI_4_3


#For particle_shape and density, we will need to create a new class for liquid oblate spheroids
#under PySDM/particle_shape_and_density
#this needs to include eqn 2 about the equivalent radius, Reynolds number eqn (5)... 
#look at LiquidSphere class for reference
class OblateSpheroid:
    def __init__(self, a, b):
        """
        Initializes an oblate spheroid with semi-axes a and b.
        'a' is the semi-major axis (equatorial radius).
        'b' is the semi-minor axis (polar radius).
        For an oblate spheroid, a >= b.
        """
        if not (isinstance(a, (int, float)) and a > 0):
            raise ValueError("Semi-major axis 'a' must be a positive number.")
        if not (isinstance(b, (int, float)) and b > 0):
            raise ValueError("Semi-minor axis 'b' must be a positive number.")
        
        if a < b:
            raise ValueError("For an oblate spheroid, semi-major axis 'a' must be greater than or equal to semi-minor axis 'b'.")
       
        self.a = float(a)
        self.b = float(b)

    
    def equivalent_radius(self, sigma_c_air: float, g: float, rho_c_l: float, rho_air: float) -> float:
        """
        Calculates an equivalent radius based on a formula involving surface tension,
        gravity, and densities.

        The formula is:
        r_eq = sqrt( (sigma_c_air / (g * (rho_c_l - rho_air))) *
                     (b/a)^(1/6) * sqrt( (b/a)^(-2) - 2 * (b/a)^(1/3) ) + 1 )
        
        Args:
            sigma_c_air (float): Surface tension between condensed phase and air [N/m].
            g (float): Acceleration due to gravity [m/s^2].
            rho_c_l (float): Density of the condensed phase (liquid/ice) [kg/m^3].
            rho_air (float): Density of air [kg/m^3].

        Returns:
            float: Equivalent radius [m].
        
        Raises:
            ValueError: If g * (rho_c_l - rho_air) is zero.
                        self.a is guaranteed to be > 0 by __init__.
        """
        ratio_ba = self.b / self.a

        term_in_inner_sqrt = ratio_ba**(-2) - 2 * ratio_ba**(1/3)
        inner_sqrt_val = np.sqrt(term_in_inner_sqrt)

        ratio_ba_pow_1_6 = ratio_ba**(1/6)

        denominator_factor1 = g * (rho_c_l - rho_air)
        if denominator_factor1 == 0:
            raise ValueError("The term g * (rho_c_l - rho_air) cannot be zero for this calculation.")

        factor1 = sigma_c_air / denominator_factor1

        product_term = factor1 * ratio_ba_pow_1_6 * inner_sqrt_val
        
        # Term inside the outer square root: product_term + 1
        # If this term is negative, np.sqrt will correctly produce nan.
        term_in_outer_sqrt = product_term + 1
        
        r_eq = np.sqrt(term_in_outer_sqrt)

        return r_eq

    def volume(self):
        """
        Calculates the volume of the oblate spheroid.
        V = (4/3) * pi * a^2 * b
        where 'a' is the equatorial radius and 'b' is the polar radius.
        Returns:
            float: Volume of the spheroid.
        """
        return PI_4_3 * (self.a**2) * self.b

    def reynolds_number(self, velocity_wrt_air, dynamic_viscosity, air_density):
        """
        Calculates the Reynolds number for the oblate spheroid.
        Re = (air_density * velocity_wrt_air * L) / dynamic_viscosity
        The characteristic length L is taken as 2 * equivalent_radius().

        Args:
            velocity_wrt_air (float): Velocity of the particle relative to air [m/s].
            dynamic_viscosity (float): Dynamic viscosity of air [Pa*s or kg/(m*s)].
            air_density (float): Density of air [kg/m^3].

        Returns:
            float: Reynolds number (dimensionless).
        """
        r_eq = self.equivalent_radius()
        characteristic_length = 2 * r_eq
        
        if dynamic_viscosity == 0: # avoid division by zero
            # if numerator is also zero, Reynolds number is taken as 0.
            # if numerator is non-zero, Reynolds number is infinite.
            return np.inf if characteristic_length * velocity_wrt_air * air_density != 0 else 0.0

        return (air_density * velocity_wrt_air * characteristic_length) / dynamic_viscosity