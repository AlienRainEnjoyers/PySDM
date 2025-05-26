################################################################
# generate results for LoWo21 Figure 2
# r_min, fraction raindrop mass evaporated as functions of RH
################################################################
import numpy as np
from typing import Tuple, Dict, Any
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns

from planet import Planet
import drop_prop as drop_prop


CONFIG = {
        # Planetary conditions
        'T_surf': 300,  # [K] surface temperature
        'p_surf': 1.01325e5,  # [Pa] surface pressure
        'RH_surf': 0.75,  # [ ] surface relative humidity
        'R_p': 1.0,  # [R_earth] planet radius
        'M_p': 1.0,  # [M_earth] planet mass
        'condensible_species': 'h2o',
        
        # Simulation parameters
        'dr': 1e-6,  # [m] resolution for smallest raindrop calculation
        'n_r0': 150,  # number of initial radius points
        'n_RH1': 30,  # number of RH points in lower range
        'n_RH2': 30,  # number of RH points in upper range
        'RH_range1': (0.25, 0.75),  # lower RH range (more densely spaced)
        'RH_range2': (0.76, 0.99),  # upper RH range
        'rt_instability_factor': np.pi/2.0,  # Rayleigh-Taylor instability factor
        
        # Output
        'output_dir': 'output/fig02/'
    }



def create_planet(config: Dict[str, Any], RH: float) -> Planet:
    """
    Create a Planet object with specified relative humidity.
    
    Args:
        config: Configuration dictionary
        RH: Relative humidity value
        
    Returns:
        Planet object configured with given parameters
    """

    return Planet(
        config['R_p'], 
        config['T_surf'], 
        config['p_surf'], 
        np.array([0.0, 0.0, 1.0, 0.0, 0.0]) ,
        config['condensible_species'], 
        RH, 
        config['M_p']
    )


def generate_radius_grid(planet: Planet, config: Dict[str, Any]) -> np.ndarray:
    """
    Generate logarithmically spaced initial radius grid.
    
    Args:
        planet: Planet object for calculating size limits
        config: Configuration dictionary
        
    Returns:
        Array of initial radius values [m]
    """
    r_small, _ = drop_prop.calc_smallest_raindrop(planet, config['dr'])
    r_max = drop_prop.calc_r_max_RT(planet, config['rt_instability_factor'])
    
    return np.logspace(
        np.log10(r_small) - 1, 
        np.log10(r_max), 
        config['n_r0']
    )


def generate_humidity_grid(config: Dict[str, Any]) -> np.ndarray:
    """
    Generate relative humidity grid with denser spacing at lower values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Array of relative humidity values
    """
    RH_range1 = config['RH_range1']
    RH_range2 = config['RH_range2']
    n_RH1 = config['n_RH1']
    n_RH2 = config['n_RH2']
    
    RHs1 = np.linspace(RH_range1[0], RH_range1[1], n_RH1)
    RHs2 = np.linspace(RH_range2[0], RH_range2[1], n_RH2)
    
    return np.concatenate([RHs1, RHs2])


def calculate_mass_fraction_evaporated(r0: float, r_end: float) -> float:
    """
    Calculate the fraction of raindrop mass that evaporated.
    
    Args:
        r0: Initial radius [m]
        r_end: Final radius [m]
        
    Returns:
        Mass fraction evaporated (0 = no evaporation, 1 = complete evaporation)
    """
    if r_end <= 0:
        return 1.0
    return 1.0 - (r_end**3 / r0**3)


def simulate_raindrop_evaporation(planet: Planet, r0: float, r_min: float) -> float:
    """
    Simulate raindrop fall and calculate evaporation fraction.
    
    Args:
        planet: Planet object with atmospheric conditions
        r0: Initial raindrop radius [m]
        r_min: Minimum stable raindrop radius [m]
        
    Returns:
        Mass fraction evaporated
    """
    if r0 < r_min:
        return 1.0  # Complete evaporation for drops smaller than minimum
    
    sol = drop_prop.integrate_fall(planet, r0)
    z_end = drop_prop.calc_last_z(sol)
    
    # Get the final radius from the solution
    if hasattr(sol.sol, 'sol') and hasattr(sol.sol.sol, '__call__'):
        # Use dense output if available
        r_end = sol.sol.sol(z_end)[0]
    else:
        # Fall back to interpolating from solution points
        import numpy as np
        if len(sol.sol.t) > 0:
            r_end = np.interp(z_end, sol.sol.t, sol.sol.y[0])
        else:
            r_end = r0  # No change if integration failed
    
    return calculate_mass_fraction_evaporated(r0, r_end)


def compute_evaporation_matrix(config: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
    """
    Compute the evaporation matrix for all RH and radius combinations.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple containing (r0grid, RHgrid, m_frac_evap, RHs, r_mins)
    """
    # Create reference planet for initial calculations
    reference_planet = create_planet(config, config['RH_surf'])
    
    # Generate grids
    r0s = generate_radius_grid(reference_planet, config)
    RHs = generate_humidity_grid(config)
    
    # Create meshgrids
    r0grid, RHgrid = np.meshgrid(r0s, RHs)
    
    # Initialize result arrays
    n_RH = len(RHs)
    n_r0 = len(r0s)
    m_frac_evap = np.zeros((n_RH, n_r0))
    r_mins = np.zeros(n_RH)
    
    # Calculate for each RH value
    for j, RH in enumerate(RHs):
        planet = create_planet(config, RH)
        r_mins[j] = drop_prop.calc_smallest_raindrop(planet, config['dr'])[0]
        
        # Calculate evaporation for each initial radius
        for i, r0 in enumerate(r0s):
            m_frac_evap[j, i] = simulate_raindrop_evaporation(
                planet, r0, r_mins[j]
            )
    
    return r0grid, RHgrid, m_frac_evap, RHs, r_mins


def save_results(output_dir: str, **arrays) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, array in arrays.items():
        np.save(output_path / f'{name}.npy', array)

    evaporated_fraction = arrays['m_frac_evap'] * (arrays['m_frac_evap'] >= 0.0)
    plt.figure(figsize=(10, 6))
    # plt.xscale('log')
    # plt.yscale('linear')
    plt.gca().invert_yaxis()
    sns.heatmap(evaporated_fraction, cmap='viridis', cbar_kws={'label': 'Mass Fraction Evaporated'})
    plt.xlabel('Initial Radius (m)')
    plt.ylabel('Relative Humidity')
    plt.title('Evaporation Matrix')
    plt.show()


def main(config: Dict[str, Any] = CONFIG) -> None:
    print("Computing evaporation matrix...")
    r0grid, RHgrid, m_frac_evap, RHs, r_mins = compute_evaporation_matrix(config)
    
    print("Saving results...")
    save_results(
        config['output_dir'],
        r0grid=r0grid,
        RHgrid=RHgrid,
        m_frac_evap=m_frac_evap,
        RHs=RHs,
        r_mins=r_mins
    )
    
    print(f"Results saved to {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
