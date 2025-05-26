import pint
import numpy as np

from PySDM.physics.constants import si

# Physical constants:
R_EARTH = 6.371e6 * si.meter # [m] Earth radius
M_EARTH = 5.972e24 * si.kg # [kg] Earth mass
G = 6.67430e-11  * si.meter^3/si.kg/si.second^2 # [m³/kg/s²] Gravitational constant
R_GAS = 8.314  *si.J /si.mol /si.K # [J/mol/K] Universal gas constant

class Planet:
    def __init__(self, 
                 R_p: pint.Quantity,  # Planet radius in Earth radii
                 T_surf: pint.Quantity,  # Surface temperature [K]
                 p_surf: pint.Quantity,  # Surface pressure [Pa]
                 composition: np.ndarray,  # Atmospheric composition
                 condensible_species: str,  # Name of condensible species
                 RH: pint.Quantity,  # Relative humidity
                 M_p: pint.Quantity):  # Planet mass in Earth masses

        self.R_p = R_p * R_EARTH  # Convert to meters
        self.T_surf = T_surf
        self.p_surf = p_surf
        self.composition = composition
        self.condensible_species = condensible_species
        self.RH = RH
        self.M_p = M_p * M_EARTH  # Convert to kg
        
        # Calculate surface gravity
        self.g_surf = G * self.M_p / (self.R_p ** 2)
        
        # Atmospheric properties
        self.scale_height = self._calc_scale_height()
        
    def _calc_scale_height(self) -> float:
        """Calculate atmospheric scale height."""
        # Assume mean molecular weight of ~29 g/mol for Earth-like atmosphere
        M_atm = 0.029 * si.kg/si.mol 
        return R_GAS * self.T_surf / (M_atm * self.g_surf)
    
    def gravity(self, z: pint.Quantity) -> pint.Quantity:
        """Calculate gravity at altitude z [m]."""
        r = self.R_p + z
        return G * self.M_p / (r ** 2)
    
    def pressure(self, z: pint.Quantity) -> pint.Quantity:
        """Calculate pressure at altitude z [m]."""
        return self.p_surf * np.exp(-z / self.scale_height)
    
    def temperature(self, z: pint.Quantity) -> pint.Quantity:
        """Calculate temperature at altitude z [m]."""
        # Simple isothermal atmosphere
        return self.T_surf
    
    def saturation_vapor_pressure(self, T: pint.Quantity) -> pint.Quantity:
        """Calculate saturation vapor pressure [Pa] using Clausius-Clapeyron."""
        # For water vapor
        if self.condensible_species == 'h2o':
            # Magnus formula for water
            return 611.657 * np.exp(17.502 * (T - 273.15) / (T - 32.18))
        else:
            # Generic approximation
            return 1000 * np.exp(20 - 5000/T)
    
    def vapor_pressure(self, z: pint.Quantity) -> pint.Quantity:
        """Calculate actual vapor pressure at altitude z [m]."""
        T = self.temperature(z)
        p_sat = self.saturation_vapor_pressure(T)
        return self.RH * p_sat
    