# simulation/time_evolution.py
import numpy as np
import pandas as pd
from reactor.core import ReactorCore
from reactor.fuel import Pebble, FuelManagement
from reactor.neutronics import NeutronicsModel
from utils.constants import *


class ReactorSimulation:
    """Class for simulating reactor operation over time"""

    def __init__(
        self,
        core_radius=DEFAULT_CORE_RADIUS,
        core_height=DEFAULT_CORE_HEIGHT,
        enrichment=DEFAULT_FUEL_ENRICHMENT,
        power=DEFAULT_POWER,
    ):
        # Initialize reactor components
        self.core = ReactorCore(radius=core_radius, height=core_height)
        self.fuel = FuelManagement(self.core)
        self.neutronics = NeutronicsModel()

        # Set up simulation parameters
        self.power = power  # W
        self.power_density = power / self.core.volume  # W/cm^3
        self.time = 0.0  # days
        self.time_step = 1.0  # days

        # Calculate initial k_eff
        self.k_eff = self.neutronics.calculate_k_eff(self.core.B_g_squared)

        # Calculate initial flux
        self.phi_0 = self.neutronics.calculate_phi_0(self.power, self.core.volume)
        self.phi_avg = self.core.average_flux(self.phi_0)

        # Initialize concentrations for burnup calculations
        # Converting enrichment to atom densities
        u_density = self.fuel.pebbles_in_core * 7.0 / self.core.fuel_volume  # g/cm^3
        u235_density = u_density * enrichment / 100
        u238_density = u_density * (1 - enrichment / 100)

        # Convert to atoms/cm^3
        u235_atoms_per_cm3 = u235_density * AVOGADRO_NUMBER / 235.0
        u238_atoms_per_cm3 = u238_density * AVOGADRO_NUMBER / 238.0

        self.initial_concentrations = {
            "U235": u235_atoms_per_cm3,
            "U238": u238_atoms_per_cm3,
            "Pu239": 0.0,
            "I135": 0.0,
            "Xe135": 0.0,
            "Pm149": 0.0,
            "Sm149": 0.0,
        }

        # Initialize results storage
        self.results = {
            "time": [self.time],
            "k_eff": [self.k_eff],
            "phi_avg": [self.phi_avg],
            "burnup_avg": [0.0],
            "U235": [u235_atoms_per_cm3],
            "U238": [u238_atoms_per_cm3],
            "Pu239": [0.0],
            "Xe135": [0.0],
            "Sm149": [0.0],
            "fresh_pebbles_added": [0],
            "pebbles_discharged": [0],
        }

    def step(self):
        """Advance simulation by one time step"""
        self.time += self.time_step

        # Simulate refueling
        refueling_results = self.fuel.examine_pebbles(
            self.fuel.pebbles_in_core, self.time_step
        )

        # Simple burnup calculation based on power
        # This is a very simplified model - in reality, a detailed burnup code would be used
        power_per_u235 = (
            self.power / self.initial_concentrations["U235"]
        )  # W per U-235 atom

        # Calculate U-235 depletion
        u235_depletion_rate = power_per_u235 / (200 * 1.6e-13)  # atoms per second
        u235_depletion = (
            u235_depletion_rate * self.time_step * 86400
        )  # atoms depleted in this time step

        # Update concentrations
        current_u235 = self.results["U235"][-1]
        new_u235 = max(0, current_u235 - u235_depletion)

        # Simple Pu-239 production model
        # Approximately 0.6 Pu-239 atoms produced per U-235 atom depleted
        pu239_production = 0.6 * (current_u235 - new_u235)
        new_pu239 = self.results["Pu239"][-1] + pu239_production

        # Update U-238 (consumption due to Pu-239 production)
        new_u238 = self.results["U238"][-1] - pu239_production

        # Simple model for Xe-135 and Sm-149 buildup and equilibrium
        # Xe-135 reaches equilibrium quickly (within days)
        if self.time < 5:
            new_xe135 = (
                self.results["Xe135"][-1] + (2.0e17 - self.results["Xe135"][-1]) * 0.5
            )
        else:
            new_xe135 = 2.0e17  # Equilibrium value

        # Sm-149 builds up more slowly
        if self.time < 100:
            new_sm149 = (
                self.results["Sm149"][-1] + (5.0e17 - self.results["Sm149"][-1]) * 0.01
            )
        else:
            new_sm149 = 5.0e17  # Equilibrium value

        # Recalculate k_eff based on new composition
        # This is a simplified model - in reality, detailed transport calculations would be performed
        k_eff_contribution_u235 = 1.5  # Arbitrary number for illustration
        k_eff_contribution_pu239 = 1.8  # Higher than U-235 due to higher nu-bar
        k_eff_decrement_xe135 = 0.03 * (new_xe135 / 2.0e17)  # Xenon worth
        k_eff_decrement_sm149 = 0.01 * (new_sm149 / 5.0e17)  # Samarium worth

        # Calculate new k_eff based on current composition
        new_k_eff = (
            current_u235 / self.initial_concentrations["U235"]
        ) * k_eff_contribution_u235
        new_k_eff += (
            new_pu239 / self.initial_concentrations["U235"]
        ) * k_eff_contribution_pu239
        new_k_eff -= k_eff_decrement_xe135
        new_k_eff -= k_eff_decrement_sm149

        # Calculate burnup
        energy_released = self.power * self.time_step * 86400  # J
        uranium_mass = self.fuel.pebbles_in_core * 7 / 1e6  # tonnes
        burnup_increment = energy_released / uranium_mass * 1e-9  # GWd/tHM
        new_burnup = self.results["burnup_avg"][-1] + burnup_increment

        # Store results
        self.results["time"].append(self.time)
        self.results["k_eff"].append(new_k_eff)
        self.results["phi_avg"].append(self.phi_avg)
        self.results["burnup_avg"].append(new_burnup)
        self.results["U235"].append(new_u235)
        self.results["U238"].append(new_u238)
        self.results["Pu239"].append(new_pu239)
        self.results["Xe135"].append(new_xe135)
        self.results["Sm149"].append(new_sm149)
        self.results["fresh_pebbles_added"].append(refueling_results["fresh_added"])
        self.results["pebbles_discharged"].append(refueling_results["discharged"])

        return self.results

    def run_simulation(self, duration):
        """Run simulation for specified duration in days"""
        steps = int(duration / self.time_step)

        for _ in range(steps):
            self.step()

        return pd.DataFrame(self.results)

    def calculate_equilibrium_params(self):
        """Calculate equilibrium parameters for the reactor"""
        # This would use a more detailed calculation in a real model

        # Calculate equilibrium xenon and samarium concentrations
        xenon_eq = self.neutronics.calculate_xenon_equilibrium(self.phi_avg)

        # Calculate equilibrium burnup distribution
        # In a real pebble bed reactor, this would be a complex calculation
        # based on pebble flow patterns and discharge criteria

        # For this simplified model, use some reasonable assumptions
        equilibrium_burnup = 90.0  # GWd/tHM
        days_to_equilibrium = (
            equilibrium_burnup
            * 1e9
            * self.fuel.pebbles_in_core
            * 7
            / (self.power * 86400)
        )

        return {
            "xenon_equilibrium": xenon_eq,
            "equilibrium_burnup": equilibrium_burnup,
            "days_to_equilibrium": days_to_equilibrium,
        }
