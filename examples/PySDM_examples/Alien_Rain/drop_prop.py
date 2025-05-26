"""
Raindrop property calculations.
"""
import numpy as np
from typing import TYPE_CHECKING, Tuple
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from planet import Planet

def calc_r_max_RT(planet: 'Planet', instability_factor: float) -> float:
    """
    Calculate maximum stable raindrop radius based on Rayleigh-Taylor instability.
    
    Args:
        planet: Planet object
        instability_factor: Instability factor (typically π/2)
        
    Returns:
        Maximum stable radius [m]
    """
    # Surface tension of water [N/m]
    sigma = 0.0728
    
    # Density of water [kg/m³]
    rho_water = 1000.0
    
    # Air density at surface [kg/m³]
    # Using ideal gas law: ρ = pM/(RT)
    M_air = 0.029  # [kg/mol] molar mass of air
    R_gas = 8.314  # [J/mol/K]
    rho_air = planet.p_surf * M_air / (R_gas * planet.T_surf)
    
    # Rayleigh-Taylor instability criterion
    # r_max ≈ sqrt(σ / (g * Δρ)) * factor
    delta_rho = rho_water - rho_air
    
    r_max = np.sqrt(sigma / (planet.g_surf * delta_rho)) * instability_factor
    
    return r_max

def terminal_velocity(r: float, planet: 'Planet', z: float = 0.0) -> float:
    """
    Calculate terminal velocity of a raindrop.
    
    Args:
        r: Raindrop radius [m]
        planet: Planet object
        z: Altitude [m]
        
    Returns:
        Terminal velocity [m/s]
    """
    # Water density [kg/m³]
    rho_water = 1000.0
    
    # Air density at altitude z
    p = planet.pressure(z)
    T = planet.temperature(z)
    M_air = 0.029  # [kg/mol]
    R_gas = 8.314  # [J/mol/K]
    rho_air = p * M_air / (R_gas * T)
    
    # Dynamic viscosity of air [Pa·s]
    mu_air = 1.8e-5
    
    # Gravity at altitude z
    g = planet.gravity(z)
    
    # For small drops (r < 0.5 mm), use Stokes law
    if r < 5e-4:
        v_term = 2 * g * rho_water * r**2 / (9 * mu_air)
    else:
        # For larger drops, use empirical formula
        # Based on Best (1950) and modified
        v_term = np.sqrt(8 * r * g * rho_water / (3 * rho_air))
        
        # Apply drag coefficient correction
        Re = 2 * r * v_term * rho_air / mu_air
        if Re > 0.1:
            Cd = 24/Re + 6/(1 + np.sqrt(Re)) + 0.4
            v_term = np.sqrt(4 * r * g * (rho_water - rho_air) / (3 * rho_air * Cd))
    
    return v_term


class FallSolution:
    """Wrapper for ODE solution with additional methods."""
    
    def __init__(self, sol):
        self.sol = sol
        self.t = sol.t
        self.y = sol.y


def calc_smallest_raindrop(planet: 'Planet', dr: float) -> Tuple[float, float]:
    """
    Calculate the smallest stable raindrop radius.
    
    Args:
        planet: Planet object
        dr: Resolution for calculation [m]
        
    Returns:
        Tuple of (r_min [m], terminal_velocity [m/s])
    """
    # Start with a very small radius and check when it becomes unstable
    r_test = 1e-6  # Start with 1 μm
    
    # Check evaporation rate vs fall rate
    # Simple criterion: drop is stable if it can fall faster than it evaporates
    while r_test < 1e-3:  # Check up to 1 mm
        v_term = terminal_velocity(r_test, planet, 0.0)
        
        # Evaporation rate estimate (simplified)
        evap_rate = _calc_evaporation_rate(planet, r_test, 0.0)
        
        # If fall time is much less than evaporation time, drop is stable
        fall_time_scale = planet.scale_height / v_term
        evap_time_scale = r_test / evap_rate if evap_rate > 0 else np.inf
        
        if fall_time_scale < evap_time_scale * 0.1:  # Stable criterion
            return r_test, v_term
            
        r_test += dr
    
    # Default fallback
    return 1e-5, terminal_velocity(1e-5, planet, 0.0)


def _calc_evaporation_rate(planet: 'Planet', r: float, z: float) -> float:
    """
    Calculate evaporation rate dr/dt.
    
    Args:
        planet: Planet object
        r: Current radius [m]
        z: Current altitude [m]
        
    Returns:
        Evaporation rate [m/s]
    """
    if r <= 0:
        return 0.0
    
    # Simplified evaporation model
    T = planet.temperature(z)
    p_vapor = planet.vapor_pressure(z)
    p_sat = planet.saturation_vapor_pressure(T)
    
    # If saturated or supersaturated, no evaporation
    if p_vapor >= p_sat:
        return 0.0
    
    # Evaporation rate proportional to undersaturation
    # This is a simplified model - real evaporation is more complex
    D_vapor = 2.5e-5  # [m²/s] water vapor diffusivity in air
    
    # Fick's law approximation
    vapor_deficit = (p_sat - p_vapor) / p_sat
    evap_rate = D_vapor * vapor_deficit / r
    
    return evap_rate


def raindrop_ode(z: float, y: np.ndarray, planet: 'Planet') -> np.ndarray:
    """
    ODE system for raindrop fall with evaporation.
    
    Args:
        z: Altitude [m] (independent variable)
        y: State vector [r] where r is radius [m]
        planet: Planet object
        
    Returns:
        Derivative vector [dr/dz]
    """
    r = y[0]
    
    if r <= 0:
        return np.array([0.0])
    
    # Terminal velocity
    v_term = terminal_velocity(r, planet, z)
    
    # Evaporation rate
    dr_dt = -_calc_evaporation_rate(planet, r, z)  # Negative for evaporation
    
    # Convert time derivative to altitude derivative
    # dr/dz = (dr/dt) / (dz/dt) = (dr/dt) / v_term
    if v_term > 0:
        dr_dz = dr_dt / v_term
    else:
        dr_dz = 0.0
    
    return np.array([dr_dz])


def integrate_fall(planet: 'Planet', r0: float) -> FallSolution:
    """
    Integrate raindrop fall from top of atmosphere to surface.
    
    Args:
        planet: Planet object
        r0: Initial radius [m]
        
    Returns:
        FallSolution object containing integration results
    """
    # Integration limits
    z_start = 10 * planet.scale_height  # Start high in atmosphere
    z_end = 0.0  # Surface
    
    # Initial conditions
    y0 = np.array([r0])
    
    # Integration settings
    rtol = 1e-6
    atol = 1e-9
    
    # Event function to stop if radius becomes too small
    def radius_event(z, y):
        return y[0] - 1e-8  # Stop if radius < 10 nm
    
    radius_event.terminal = True
    radius_event.direction = -1
    
    # Solve ODE
    sol = solve_ivp(
        lambda z, y: raindrop_ode(z, y, planet),
        [z_start, z_end],
        y0,
        method='RK45',
        rtol=rtol,
        atol=atol,
        events=radius_event,
        dense_output=True
    )
    
    return FallSolution(sol)


def calc_last_z(sol: FallSolution) -> float:
    """
    Calculate the final altitude from integration solution.
    
    Args:
        sol: FallSolution object
        
    Returns:
        Final altitude [m]
    """
    if hasattr(sol.sol, 't_events') and len(sol.sol.t_events) > 0 and len(sol.sol.t_events[0]) > 0:
        # Stopped due to event (small radius)
        return sol.sol.t_events[0][0]
    else:
        # Reached surface
        return sol.sol.t[-1]