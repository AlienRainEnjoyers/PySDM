import numpy as np

import PySDM.products as PySDM_products
from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.physics import constants as const


## General simulation from Arabas and Shima 2017, also looking at Graf et al. 2019
#Need to edit Parcel in here to change dz into w +terminalv (should this be a w function? an option?)
# Some of this is probably not needed, not sure what yet

class Simulation:
    def __init__(self, settings, backend=CPU):

        dt_output = settings.dt
        self.n_substeps = 1
        while dt_output / self.n_substeps >= settings.dt_max:
            self.n_substeps += 1

        builder = Builder(
            backend=backend(
                formulae=settings.formulae,
                **(
                    {"override_jit_flags": {"parallel": False}}
                    if backend == CPU
                    else {}
                ),
            ),
            n_sd=1,
            environment=Parcel(
                dt=dt_output / self.n_substeps,
                mass_of_dry_air=settings.mass_of_dry_air,
                p0=settings.p0,
                initial_water_vapour_mixing_ratio=settings.initial_water_vapour_mixing_ratio,
                T0=settings.T0,
                w=settings.w,
            ),
        )

        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(
            Condensation(
                rtol_x=settings.rtol_x,
                rtol_thd=settings.rtol_thd,
                dt_cond_range=settings.dt_cond_range,
            )
        )
        attributes = {}
        r_dry = np.array([settings.r_dry])
        attributes["dry volume"] = settings.formulae.trivia.volume(radius=r_dry)
        attributes["kappa times dry volume"] = attributes["dry volume"] * settings.kappa
        attributes["multiplicity"] = np.array([1])
        environment = builder.particulator.environment
        attributes["volume"] = settings.formulae.trivia.volume(radius=settings.initial_radius)
        
        products = [
            PySDM_products.MeanRadius(name="radius_m1", unit="um"),
            PySDM_products.ParcelDisplacement(name="z"),
            PySDM_products.AmbientRelativeHumidity(name="RH", unit="%"),
            PySDM_products.Time(name="t"),
        ]

        self.particulator = builder.build(attributes, products)

        self.n_output = settings.n_output

    def save(self, output):
        cell_id = 0
        output["r"].append(
            self.particulator.products["radius_m1"].get(unit=const.si.m)[cell_id]
        )
        output["z"].append(self.particulator.products["z"].get()[cell_id])
        output["S"].append(self.particulator.products["RH"].get()[cell_id] / 100 - 1)
        output["t"].append(self.particulator.products["t"].get())


    def run(self):
        output = {
            "r": [],
            "S": [],
            "z": [],
            "t": [],
        }

        self.save(output)
        for _ in range(self.n_output):
            self.particulator.run(self.n_substeps)
            self.save(output)

        return output