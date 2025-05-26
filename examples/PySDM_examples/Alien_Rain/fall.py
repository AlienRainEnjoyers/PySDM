"""
Raindrop fall and evaporation simulation.
"""
import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, TYPE_CHECKING
from . import drop_prop

if TYPE_CHECKING:
    from .planet import Planet

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
        v_term = drop_prop.terminal_velocity(r_test, planet, 0.0)
        
        # Evaporation rate estimate (simplified)
        evap_rate = _calc_evaporation_rate(planet, r_test, 0.0)
        
        # If fall time is much less than evaporation time, drop is stable
        fall_time_scale = planet.scale_height / v_term
        evap_time_scale = r_test / evap_rate if evap_rate > 0 else np.inf
        
        if fall_time_scale < evap_time_scale * 0.1:  # Stable criterion
            return r_test, v_term
            
        r_test += dr
    
    # Default fallback
    return 1e-5, drop_prop.terminal_velocity(1e-5, planet, 0.0)

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
    v_term = drop_prop.terminal_velocity(r, planet, z)
    
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
    def radius_event(z, y, planet):
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
    if len(sol.sol.t_events[0]) > 0:
        # Stopped due to event (small radius)
        return sol.sol.t_events[0][0]
    else:
        # Reached surface
        return sol.t[-1]