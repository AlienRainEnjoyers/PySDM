import unittest
from unittest.mock import patch
import numpy as np
from PySDM import Formulae
from PySDM.physics import si
from scipy.optimize import fsolve

from PySDM_examples.Loftus_and_Wordsworth_2021.planet import EarthLike, Earth, EarlyMars, Jupiter, Saturn, K2_18B
from PySDM_examples.Loftus_and_Wordsworth_2021.simulation import Simulation
from PySDM_examples.Loftus_and_Wordsworth_2021.parcel import AlienParcel
from PySDM_examples.Loftus_and_Wordsworth_2021 import Settings


class TestLoftusWordsworth2021(unittest.TestCase):
    
    def setUp(self):
        self.formulae = Formulae(
            ventilation="PruppacherAndRasmussen1979",
            saturation_vapour_pressure="AugustRocheMagnus",
            diffusion_coordinate="WaterMassLogarithm",
        )
        self.earth_like = EarthLike()
        
    def test_planet_classes(self):
        """Test planet class instantiation and basic properties."""
        planets = [
            EarthLike(),
            Earth(),
            EarlyMars(),
            Jupiter(),
            Saturn(),
            K2_18B()
        ]
        
        for planet in planets:
            self.assertGreater(planet.g_std, 0)
            self.assertGreater(planet.T_STP, 0)
            self.assertGreater(planet.p_STP, 0)
            self.assertGreaterEqual(planet.RH_zref, 0)
            self.assertLessEqual(planet.RH_zref, 1)
            
            # atmospheric composition sums to 1 or less
            total_conc = (planet.dry_molar_conc_H2 + planet.dry_molar_conc_He + 
                         planet.dry_molar_conc_N2 + planet.dry_molar_conc_O2 + 
                         planet.dry_molar_conc_CO2)
            self.assertLessEqual(total_conc, 1.01)
            
        
    def test_water_vapour_mixing_ratio_calculation(self):
        """Test water vapour mixing ratio calculation."""
        const = self.formulae.constants
        planet = EarthLike()
        
        pvs = self.formulae.saturation_vapour_pressure.pvs_water(planet.T_STP)
        initial_water_vapour_mixing_ratio = const.eps / (
            planet.p_STP / planet.RH_zref / pvs - 1
        )
        
        self.assertGreater(initial_water_vapour_mixing_ratio, 0)
        self.assertLess(initial_water_vapour_mixing_ratio, 0.1)  # Should be less than 10%
        
        
    def test_alien_parcel_initialization(self):
        """Test AlienParcel class initialization."""
        parcel = AlienParcel(
            dt=1.0 * si.second,
            mass_of_dry_air=1e5 * si.kg,
            pcloud=90000 * si.pascal,
            initial_water_vapour_mixing_ratio=0.01,
            Tcloud=280 * si.kelvin,
            w=0,
            zcloud=1000 * si.m,
        )

        self.assertTrue(hasattr(parcel, 'advance_parcel_vars'))
        
    def test_simulation_class(self):
        """Test Simulation class initialization and basic functionality."""
        planet = EarthLike()
        
        settings = Settings(
            planet=planet,
            r_wet=1e-4 * si.m,
            mass_of_dry_air=1e5 * si.kg,
            initial_water_vapour_mixing_ratio=0.01,
            pcloud=90000 * si.pascal,
            Zcloud=1000 * si.m,
            Tcloud=280 * si.kelvin,
            formulae=self.formulae,
        )
        
        simulation = Simulation(settings)
        
        self.assertTrue(hasattr(simulation, 'particulator'))
        self.assertTrue(hasattr(simulation, 'run'))
        self.assertTrue(hasattr(simulation, 'save'))
        
        products = simulation.particulator.products
        required_products = ["radius_m1", "z", "RH", "t"]
        for product in required_products:
            self.assertIn(product, products)
            
    def test_simulation_run_basic(self):
        """Test basic simulation run functionality."""
        planet = EarthLike()
        
        settings = Settings(
            planet=planet,
            r_wet=1e-5 * si.m,  # Small droplet for quick evaporation
            mass_of_dry_air=1e5 * si.kg,
            initial_water_vapour_mixing_ratio=0.01,  # Low humidity
            pcloud=90000 * si.pascal,
            Zcloud=100 * si.m,  # Low height
            Tcloud=280 * si.kelvin,
            formulae=self.formulae,
        )
        
        simulation = Simulation(settings)
        output = simulation.run()
        
        # output structure
        self.assertIn('r', output)
        self.assertIn('S', output)
        self.assertIn('z', output)
        self.assertIn('t', output)
        
        # all output arrays have same length
        lengths = [len(output[key]) for key in output.keys()]
        self.assertTrue(all(l == lengths[0] for l in lengths))
        
        # arrays are not empty
        self.assertGreater(len(output['r']), 0)
