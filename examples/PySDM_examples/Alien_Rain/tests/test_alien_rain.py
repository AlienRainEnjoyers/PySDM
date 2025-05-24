# This test file is temporarily placed in this directory for convenience. To be moved to tests/examples once completed
import unittest
from unittest.mock import patch
import numpy as np

from examples.PySDM_examples.Alien_Rain import alien_rain
from examples.PySDM_examples.Alien_Rain.tests import mock_planet
from examples.PySDM_examples.Alien_Rain.alien_rain import Planet


@patch(
    "examples.PySDM_examples.Alien_Rain.alien_rain.Planet", new=mock_planet.MockPlanet
)
class TestAlienRain(unittest.TestCase):
    def test_calculate_raindrop_parameters(self) -> None:
        """Test the calculate_raindrop_parameters function."""
        pl = Planet()
        r_small, r_max, dr = alien_rain.calculate_raindrop_parameters(pl)

        # Test return types and reasonable values
        self.assertIsInstance(r_small, float)
        self.assertIsInstance(r_max, float)
        self.assertIsInstance(dr, float)

        # Test that dr is the expected constant
        self.assertEqual(dr, 1e-6)

        # Test that r_small < r_max (physical constraint)
        self.assertLess(r_small, r_max)

        # Test reasonable ranges
        self.assertGreater(r_small, 1e-6)  # > 1 micrometer
        self.assertLess(r_small, 1e-3)  # < 1 mm
        self.assertGreater(r_max, 1e-4)  # > 0.1 mm
        self.assertLess(r_max, 1e-2)  # < 1 cm

    def test_calculate_raindrop_parameters_different_conditions(self) -> None:
        """Test raindrop parameters under different planetary conditions."""
        # Test low humidity planet
        pl_low_rh = Planet(RH_surf=0.25)
        r_small_low, r_max_low, _ = alien_rain.calculate_raindrop_parameters(pl_low_rh)

        # Test high humidity planet
        pl_high_rh = Planet(RH_surf=0.95)
        r_small_high, r_max_high, _ = alien_rain.calculate_raindrop_parameters(
            pl_high_rh
        )

        # Lower humidity should lead to larger minimum raindrop size
        self.assertGreater(r_small_low, r_small_high)

        # Test different gravity conditions
        pl_high_g = Planet(M_p=2.0, R_p=1.0)  # Higher gravity
        pl_low_g = Planet(M_p=0.5, R_p=1.5)  # Lower gravity

        _, r_max_high_g, _ = alien_rain.calculate_raindrop_parameters(pl_high_g)
        _, r_max_low_g, _ = alien_rain.calculate_raindrop_parameters(pl_low_g)

        # Higher gravity should lead to smaller maximum stable raindrop
        self.assertLess(r_max_high_g, r_max_low_g)

    def test_compute_evaporation_and_min_radius(self):
        """Test the compute_evaporation_and_min_radius function."""
        # Set up test parameters
        pl_initial = Planet()
        R_p = 1.0
        M_p = 1.0
        X_composition = np.zeros(5)
        X_composition[2] = 1.0  # N2 atmosphere
        r_small = 5e-5
        r_max = 2e-3
        dr = 1e-6

        # Call the function
        r0grid, RHgrid, m_frac_evap, RHs, r_mins = (
            alien_rain.compute_evaporation_and_min_radius(
                pl_initial, R_p, M_p, X_composition, r_small, r_max, dr
            )
        )

        # Test output shapes and types
        self.assertIsInstance(r0grid, np.ndarray)
        self.assertIsInstance(RHgrid, np.ndarray)
        self.assertIsInstance(m_frac_evap, np.ndarray)
        self.assertIsInstance(RHs, np.ndarray)
        self.assertIsInstance(r_mins, np.ndarray)

        # Test grid dimensions
        n_r0 = 150
        n_RH = 60  # 30 + 30
        self.assertEqual(r0grid.shape, (n_RH, n_r0))
        self.assertEqual(RHgrid.shape, (n_RH, n_r0))
        self.assertEqual(m_frac_evap.shape, (n_RH, n_r0))
        self.assertEqual(RHs.shape, (n_RH,))
        self.assertEqual(r_mins.shape, (n_RH,))

        # Test RH values range
        self.assertAlmostEqual(np.min(RHs), 0.25, places=2)
        self.assertAlmostEqual(np.max(RHs), 0.99, places=2)

        # Test r0 values are logarithmically spaced
        r0s_first_row = r0grid[0, :]
        self.assertGreater(r0s_first_row[0], r_small / 10)
        self.assertLessEqual(r0s_first_row[-1], r_max)

        # Test mass fraction evaporated values are between 0 and 1
        self.assertTrue(np.all(m_frac_evap >= 0))
        self.assertTrue(np.all(m_frac_evap <= 1))

        # Test that minimum radii decrease with increasing RH
        self.assertGreater(r_mins[0], r_mins[-1])
