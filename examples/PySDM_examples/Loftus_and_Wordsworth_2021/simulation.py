import numpy as np

import PySDM.products as PySDM_products
from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.physics import constants as const
from PySDM_examples.Loftus_and_Wordsworth_2021.parcel import AlienParcel


## General simulation from Arabas and Shima 2017, also looking at Graf et al. 2019
#Need to edit Parcel in here to change dz into w +terminalv (should this be a w function? an option?)
# Some of this is probably not needed, not sure what yet
class Simulation:
    def __init__(self, settings, backend=CPU):
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
            environment=AlienParcel(
                dt=settings.dt, #dt_output / self.n_substeps,
                mass_of_dry_air=settings.mass_of_dry_air,
                pcloud=settings.pcloud,
                zcloud= settings.Zcloud,
                initial_water_vapour_mixing_ratio=settings.initial_water_vapour_mixing_ratio,
                Tcloud=settings.Tcloud,
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
        builder.request_attribute("terminal velocity")

        attributes = {}
        r_dry = 1e-10 #np.array([settings.r_dry])
        attributes["dry volume"] = settings.formulae.trivia.volume(radius=r_dry)
        attributes["kappa times dry volume"] = attributes["dry volume"] * settings.kappa
        attributes["multiplicity"] = np.array([1], dtype=np.int64)
        # attributes["terminal velocity"] = np.array([0.0])
        environment = builder.particulator.environment
        r_wet = settings.r_wet
        attributes["volume"] = settings.formulae.trivia.volume(radius=r_wet)
        products = [
            PySDM_products.MeanRadius(name="radius_m1", unit="um"),
            PySDM_products.ParcelDisplacement(name="z"),
            PySDM_products.AmbientRelativeHumidity(name="RH", unit="%"),
            PySDM_products.Time(name="t"),
        ]

        self.particulator = builder.build(attributes, products)


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
        while self.particulator.environment["z"][0] >0 and output["r"][-1] > 1e-6:
            # print(self.particulator.environment["z"][0])
            self.particulator.run(1)
            self.save(output)

        return output