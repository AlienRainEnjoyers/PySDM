from typing import List, Optional
from PySDM.environments.parcel import Parcel


class AlienParcel(Parcel):
    def __init__(
        self,
        dt,
        mass_of_dry_air: float,
        p0: float,
        initial_water_vapour_mixing_ratio: float,
        T0: float,
        w: [float, callable],
        z0: float = 0,
        mixed_phase=False,
        variables: Optional[List[str]] = None,
    ):
        super().__init__(
            dt=dt,
            mass_of_dry_air=mass_of_dry_air,
            p0=p0,
            initial_water_vapour_mixing_ratio=initial_water_vapour_mixing_ratio,
            T0=T0,
            w=w,
            z0=z0,
            mixed_phase=mixed_phase,
            variables=variables
        )

    def advance_parcel_vars(self):
        """
        Compute new values of displacement, dry-air density and volume,
        and write them to self._tmp and self.mesh.dv
        """
        dt = self.particulator.dt
        formulae = self.particulator.formulae
        T = self["T"][0]
        p = self["p"][0]

        dz_dt = self.w((self.particulator.n_steps + 1 / 2) * dt) + self["terminal_velocity"][0]
        water_vapour_mixing_ratio = (
            self["water_vapour_mixing_ratio"][0]
            - self.delta_liquid_water_mixing_ratio / 2
        )

        # derivative evaluated at p_old, T_old, mixrat_mid, w_mid
        drho_dz = formulae.hydrostatics.drho_dz(
            p=p,
            T=T,
            water_vapour_mixing_ratio=water_vapour_mixing_ratio,
            lv=formulae.latent_heat_vapourisation.lv(T),
            d_liquid_water_mixing_ratio__dz=(
                self.delta_liquid_water_mixing_ratio / dz_dt / dt
            ),
        )
        drhod_dz = drho_dz  # TODO #407

        self.particulator.backend.explicit_euler(self._tmp["z"], dt, dz_dt)
        self.particulator.backend.explicit_euler(
            self._tmp["rhod"], dt, dz_dt * drhod_dz
        )

        self.mesh.dv = formulae.trivia.volume_of_density_mass(
            (self._tmp["rhod"][0] + self["rhod"][0]) / 2, self.mass_of_dry_air
        )