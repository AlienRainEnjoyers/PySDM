import numpy as np

G = 6.674e-11 # [Nm2/kg2]
R_gas = 8.31446 #[J/mol/K] ideal gas constant
N_A = 6.022141e23 # [particles/mol] Avogadro's number
# EARTH SPECIFIC VALUES
M_earth = 5.9721986e24 # [kg]
R_earth = 6.371e6 # [m]

class MockPlanet:
    def __init__(self, R_p=1.0, T_surf=300.0, p_surf=1.01325e5, 
                 X_composition=None, condensible="h2o", RH_surf=0.75, M_p=1.0):
        self.R_p = R_p  # Planet radius in Earth radii
        self.T_surf = T_surf  # Surface temperature in K
        self.p_surf = p_surf  # Surface pressure in Pa
        self.RH_surf = RH_surf  # Relative humidity
        self.M_p = M_p  # Planet mass in Earth masses
        self.X_composition = X_composition if X_composition is not None else np.zeros(5)
        self.condensible = condensible
        # attributes accessed by the functions under test
        # TODO: probably will need to introduce Pint (hopefully not)
        self.cp_gas = 1000
        self.L = 2.5e6
        self.D_eff = 2.2e-5
        self.rho_l = 1000
        self.sigma_l = 0.072
        self.mu_gas = 1.8e-5
        self.k_gas = 0.025

    def calc_p_sat(self, T):
        # saturation vapor pressure of water (Magnus-Tetens approximation)
        return 610.78 * np.exp(17.27 * (T - 273.15) / (T - 273.15 + 237.3))

    def calc_rho_gas(self, p, T):
        # gas density
        return p / (R_gas * T)
