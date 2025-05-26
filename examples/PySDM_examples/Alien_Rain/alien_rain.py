import numpy as np
from planet import Planet
import fall as fall
import drop_prop as drop_prop


def setup_planet_conditions():
    """Sets up the initial planetary conditions."""
    X = np.zeros(5)  # composition
    X[2] = 1.0  # f_N2  [mol/mol]
    T_surf = 300  # [K]
    p_surf = 1.01325e5  # [Pa]
    RH_surf = 0.75  # [ ]
    R_p = 1.0  # [R_earth]
    M_p = 1.0  # [M_earth]
    pl = Planet(R_p, T_surf, p_surf, X, "h2o", RH_surf, M_p)
    return pl, R_p, M_p, X  # Return R_p, M_p, X as they are used later


def calculate_raindrop_parameters(pl):
    """Calculates initial raindrop size parameters."""
    dr = 1e-6  # [m]
    r_small, _ = fall.calc_smallest_raindrop(pl, dr)
    # maximum stable raindrop size
    # from Rayleigh-Taylor instability, ℓ_RT = 0.5πa
    r_max = drop_prop.calc_r_max_RT(pl, np.pi / 2.0)  # [m]
    return r_small, r_max, dr


def compute_evaporation_and_min_radius(
    pl_initial, R_p, M_p, X_composition, r_small, r_max, dr
):
    """Computes the fraction of mass evaporated and minimum raindrop size for various RH values."""
    n_r0 = 150
    n_RH1 = 30
    n_RH2 = 30
    n_RH = n_RH1 + n_RH2

    r0s = np.logspace(np.log10(r_small) - 1, np.log10(r_max), n_r0)  # [m]
    RHs1 = np.linspace(0.25, 0.75, n_RH1)  # [ ]
    RHs2 = np.linspace(0.76, 0.99, n_RH2)  # [ ]
    RHs = np.zeros(n_RH)
    RHs[:n_RH1] = RHs1
    RHs[n_RH1:] = RHs2
    r0grid, RHgrid = np.meshgrid(r0s, RHs)
    m_frac_evap = np.zeros((n_RH, n_r0))  # [ ]
    r_mins = np.zeros(n_RH)  # [m]

    T_surf = pl_initial.T_surf  # Get T_surf and p_surf from the initial planet object
    p_surf = pl_initial.p_surf

    for j, RH_val in enumerate(RHs):
        pl_current = Planet(R_p, T_surf, p_surf, X_composition, "h2o", RH_val, M_p)
        r_mins[j] = fall.calc_smallest_raindrop(pl_current, dr)[0]  # [m]
        for i, r0 in enumerate(r0s):
            if r0 >= r_mins[j]:  # don't bother to integrate if r0 is smaller than rmin
                sol = fall.integrate_fall(pl_current, r0)
                z_end = fall.calc_last_z(sol)  # [m]
                r_end = sol.sol.__call__(z_end)[0]  # [m]
                m_frac_evap[j, i] = 1 - r_end**3 / r0**3  # [ ]
            else:
                m_frac_evap[j, i] = 1.0  # [ ]
    return r0grid, RHgrid, m_frac_evap, RHs, r_mins


def save_results(r0grid, RHgrid, m_frac_evap, RHs, r_mins):
    """Saves the computed results to .npy files."""
    dir_path = "output/fig02/"
    np.save(dir_path + "r0grid", r0grid)
    np.save(dir_path + "RHgrid", RHgrid)
    np.save(dir_path + "m_frac_evap", m_frac_evap)
    np.save(dir_path + "RHs", RHs)
    np.save(dir_path + "r_mins", r_mins)


def main():
    """Main function to generate results for LoWo21 Figure 2."""
    pl_initial, R_p, M_p, X_composition = setup_planet_conditions()
    r_small, r_max, dr = calculate_raindrop_parameters(pl_initial)
    r0grid, RHgrid, m_frac_evap, RHs, r_mins = compute_evaporation_and_min_radius(
        pl_initial, R_p, M_p, X_composition, r_small, r_max, dr
    )
    save_results(r0grid, RHgrid, m_frac_evap, RHs, r_mins)


if __name__ == "__main__":
    # filepath: /Users/lursz/agh/rainprops/gen_fig02.py
    ################################################################
    # generate results for LoWo21 Figure 2
    # r_min, fraction raindrop mass evaporated as functions of RH
    ################################################################
    main()
