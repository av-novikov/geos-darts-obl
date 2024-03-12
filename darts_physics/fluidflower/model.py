import numpy as np
from dataclasses import dataclass, field

from darts.models.darts_model import DartsModel
from darts.engines import value_vector, index_vector, sim_params, conn_mesh

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, CapillaryPressure, RockEnergyEvaluator, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.flash import ConstantK

from dartsflash.libflash import NegativeFlash, FlashParams, InitialGuess
from dartsflash.libflash import CubicEoS, AQEoS
from dartsflash.components import CompData, EnthalpyIdeal
from dartsflash.eos_properties import EoSDensity, EoSEnthalpy


# region Dataclasses
@dataclass
class Corey:
    nw: float
    ng: float
    swc: float
    sgc: float
    krwe: float
    krge: float
    labda: float
    p_entry: float
    def modify(self, std, mult):
        i = 0
        for attr, value in self.__dict__.items():
            if attr != 'type':
                setattr(self, attr, value * (1 + mult[i] * float(getattr(std, attr))))
            i += 1

    def random(self, std):
        for attr, value in self.__dict__.items():
            if attr != 'type':
                std_in = value * float(getattr(std, attr))
                param = np.random.normal(value, std_in)
                if param < 0:
                    param = 0
                setattr(self, attr, param)


@dataclass
class PorPerm:
    type: str
    poro: float
    perm: float
    anisotropy: list = None
    hcap: float = 2200
    rcond: float = 181.44

# endregion

class Model(DartsModel):
    def __init__(self, n_points):
        # Call base class constructor
        super().__init__()

        self.n_obl_points = n_points
        self.zero = 1e-8
        corey = {
            0: Corey(nw=2.0, ng=1.5, swc=0.11, sgc=0.06, krwe=0.80, krge=0.85, labda=2., p_entry=0.),
            1: Corey(nw=2.0, ng=1.5, swc=0.12, sgc=0.08, krwe=0.93, krge=0.95, labda=2., p_entry=1e-3),
            2: Corey(nw=2.5, ng=2.0, swc=0.14, sgc=0.10, krwe=0.93, krge=0.95, labda=2., p_entry=3e-3),
            3: Corey(nw=2.5, ng=2.0, swc=0.32, sgc=0.14, krwe=0.71, krge=0.75, labda=2., p_entry=15e-3)
        }
        self.set_physics(corey=corey, zero=self.zero, temperature=296., n_points=self.n_obl_points)
        self.physics.init_physics()

    def set_physics(self, corey: dict = {}, zero: float = 1e-8, temperature: float = None, n_points: int = 10001):
        """Physical properties"""
        # Fluid components, ions and solid
        components = ["H2O", "CO2"]
        phases = ["V", "Aq"]
        comp_data = CompData(components, setprops=True)

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        pr = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, AQEoS.Ziabakhsh2012)

        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        if temperature is None:  # if None, then thermal=True
            thermal = True
        else:
            thermal = False

        min_t = 273.15 if temperature is None else None
        max_t = 373.15 if temperature is None else None
        self.physics = Compositional(components, phases, timer=self.timer,
                                     n_points=n_points, min_p=1.0, max_p=500.0,
                                     min_z=zero / 10, max_z=1 - zero / 10,
                                     min_t=min_t, max_t=max_t,
                                     thermal=thermal, cache=False)

        for i, (region, corey_params) in enumerate(corey.items()):
            diff = 0.  #8.64e-6
            property_container = PropertyContainer(components_name=components, phases_name=phases, Mw=comp_data.Mw,
                                                   min_z=zero / 10, diff_coef=diff, temperature=temperature)

            # property_container.flash_ev = ConstantK(nc=2, ki=[0.001, 100])
            property_container.flash_ev = NegativeFlash(flash_params, ["PR", "AQ"], [InitialGuess.Henry_VA])
            property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                                  ('Aq', Garcia2001(components)), ])
            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(components)), ])

            if thermal:
                # property_container.enthalpy_ev = dict([('V', Enthalpy(hcap=0.035)),
                #                                        ('L', Enthalpy(hcap=0.035)),
                #                                        ('Aq', Enthalpy(hcap=0.00418*18.015)),]
                h_ideal = EnthalpyIdeal(components)
                property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr, h_ideal=h_ideal)),
                                                       ('Aq', EoSEnthalpy(eos=aq, h_ideal=h_ideal)), ])

                property_container.conductivity_ev = dict([('V', ConstFunc(0.)),
                                                           ('Aq', ConstFunc(0.)), ])

            property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey_params, 'V')),
                                                   ('Aq', ModBrooksCorey(corey_params, 'Aq'))])
            property_container.capillary_pressure_ev = ModCapillaryPressure(corey_params)

            property_container.output_props = {"satV": lambda: property_container.sat[0],
                                               "xCO2": lambda: property_container.x[1, 1],
                                               "rhoV": lambda: property_container.dens[0],
                                               "rho_mA": lambda: property_container.dens_m[1]}

            self.physics.add_property_region(property_container, i)

    def set_well_controls(self):
        self.reservoir.wells[0].control = self.physics.new_rate_inj(0, self.inj_stream, 0)
        self.reservoir.wells[1].control = self.physics.new_rate_inj(0, self.inj_stream, 0)
        self.reservoir.wells[2].control = self.physics.new_bhp_prod(self.p_prod)
        return

class ModBrooksCorey:
    def __init__(self, corey, phase):
        self.k_rw_e = corey.krwe
        self.k_rg_e = corey.krge

        self.swc = corey.swc
        self.sgc = corey.sgc

        self.nw = corey.nw
        self.ng = corey.ng

        self.phase = phase

    def evaluate(self, sat):
        if self.phase == "Aq":
            Se = (sat - self.swc)/(1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rw_e * Se ** self.nw
        else:
            Se = (sat - self.sgc) / (1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rg_e * Se ** self.ng

        return k_r


class ModCapillaryPressure:
    def __init__(self, corey):
        self.swc = corey.swc
        self.p_entry = corey.p_entry
        self.labda = corey.labda
        # self.labda = 3
        self.eps = 1e-3

    def evaluate(self, sat):
        sat_w = sat[1]
        Se = (sat_w - self.swc)/(1 - self.swc)
        if Se < self.eps:
            Se = self.eps
        pc = self.p_entry * self.eps ** (1/self.labda) * Se ** (-1/self.labda)  # for p_entry to non-wetting phase
        # if Se > 1 - self.eps:
        #     pc = 0

        #pc = self.p_entry
        pc = self.p_entry
        Pc = np.array([0, pc], dtype=object)  # V, Aq
        return Pc

class BrooksCorey:
    def __init__(self, wetting: bool):
        self.sat_wr = 0.15
        # self.sat_nwr = 0.1

        self.lambda_w = 4.2
        self.lambda_nw = 3.7

        self.wetting = wetting

    def evaluate(self, sat_w):
        # From Brooks-Corey (1964)
        Se = (sat_w - self.sat_wr)/(1-self.sat_wr)
        if Se > 1:
            Se = 1
        elif Se < 0:
            Se = 0

        if self.wetting:
            k_r = Se**((2+3*self.lambda_w)/self.lambda_w)
        else:
            k_r = (1-Se)**2 * (1-Se**((2+self.lambda_nw)/self.lambda_nw))

        if k_r > 1:
            k_r = 1
        elif k_r < 0:
            k_r = 0

        return k_r
