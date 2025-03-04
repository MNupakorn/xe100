# reactor/fuel.py
import math
from utils.constants import *


class Pebble:
    """Class representing a TRISO fuel pebble"""

    def __init__(self, enrichment=DEFAULT_FUEL_ENRICHMENT):
        self.diameter = 6.0  # cm
        self.radius = 3.0  # cm
        self.graphite_shell_thickness = 0.5  # cm
        self.fuel_zone_radius = 2.5  # cm

        # Calculate volumes
        self.total_volume = (4 / 3) * math.pi * (self.radius**3)  # cm^3
        self.fuel_zone_volume = (4 / 3) * math.pi * (self.fuel_zone_radius**3)  # cm^3
        self.graphite_volume = self.total_volume - self.fuel_zone_volume  # cm^3

        # Fuel properties
        self.enrichment = enrichment  # %
        self.uranium_per_pebble = 7.0  # g
        self.u235_per_pebble = self.uranium_per_pebble * (self.enrichment / 100)  # g
        self.u238_per_pebble = self.uranium_per_pebble * (
            1 - self.enrichment / 100
        )  # g

        # TRISO particles
        self.triso_particles = 15000  # per pebble

        # Burnup properties
        self.burnup = 0.0  # GWd/tHM
        self.cycles_completed = 0
        self.residence_time = 0.0  # days

    def update_burnup(self, power_density, time_step):
        """Update burnup based on power density and time"""
        # power_density in W/cm^3, time_step in days

        # Convert power density to power per pebble
        power_per_pebble = power_density * self.fuel_zone_volume  # W

        # Calculate energy released in this time step
        energy_released = power_per_pebble * time_step * 86400  # W * s = J

        # Convert energy to burnup increment (GWd/tHM)
        burnup_increment = (
            energy_released / (self.uranium_per_pebble * 1e-6) * 1e-9 / 86400
        )

        self.burnup += burnup_increment
        self.residence_time += time_step

        return self.burnup


class FuelManagement:
    """Class for managing fuel loading, circulation, and extraction"""

    def __init__(self, core, target_burnup=90.0):
        self.core = core
        self.target_burnup = target_burnup  # GWd/tHM

        # Calculate number of pebbles in core
        pebble = Pebble()
        self.pebbles_in_core = int(self.core.fuel_volume / pebble.total_volume)

        # Refueling parameters
        self.daily_pebble_examination_rate = 1450  # pebbles/day
        self.daily_fresh_pebble_rate = 325  # pebbles/day
        self.daily_discharge_rate = 325  # pebbles/day

        # Pebble distribution tracking
        self.burnup_distribution = {
            "0-15": 0.15,
            "15-30": 0.20,
            "30-45": 0.22,
            "45-60": 0.18,
            "60-75": 0.15,
            "75-90": 0.10,
        }

    def examine_pebbles(self, current_pebbles, time_step):
        """Simulate pebble examination and recycling process"""
        # This is a simplified model

        # Number of pebbles to examine in this time step
        examination_count = int(self.daily_pebble_examination_rate * time_step)

        # Pebbles to discharge (those exceeding target burnup)
        discharge_count = int(self.daily_discharge_rate * time_step)

        # New pebbles to add
        fresh_pebble_count = int(self.daily_fresh_pebble_rate * time_step)

        # Update pebble counts
        return {
            "examined": examination_count,
            "discharged": discharge_count,
            "recycled": examination_count - discharge_count,
            "fresh_added": fresh_pebble_count,
        }

    def calculate_average_burnup(self, power, time):
        """Calculate average burnup after operating at given power for specified time"""
        # power in W, time in days

        # Calculate energy released
        energy_released = power * time * 86400  # J

        # Total uranium mass in core
        pebble = Pebble()
        total_uranium_mass = (
            self.pebbles_in_core * pebble.uranium_per_pebble * 1e-6
        )  # tonnes

        # Calculate burnup
        burnup = energy_released / total_uranium_mass * 1e-9  # GWd/tHM

        return burnup

    def estimate_cycle_time(self):
        """Estimate average time for a pebble to complete one cycle through the core"""
        # Based on pebble flow rate and core dimensions
        # Simplified calculation

        # Total number of pebbles in core
        # Average transit time (days) = pebbles in core / daily examination rate
        transit_time = self.pebbles_in_core / self.daily_pebble_examination_rate

        return transit_time  # days
