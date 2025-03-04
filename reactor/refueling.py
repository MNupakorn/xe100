# reactor/refueling.py
import numpy as np
import random
from utils.constants import *


class PebbleFlow:
    """Class to simulate pebble flow in the reactor"""

    def __init__(
        self, core_radius=DEFAULT_CORE_RADIUS, core_height=DEFAULT_CORE_HEIGHT
    ):
        self.core_radius = core_radius  # cm
        self.core_height = core_height  # cm

        # Flow parameters
        self.avg_flow_speed = 3.0  # mm/hour
        self.flow_variance = 0.5  # mm/hour (standard deviation)

        # Calculate average transit time
        self.avg_transit_time = (self.core_height * 10) / self.avg_flow_speed  # hours

        # Setup flow channels
        self.setup_flow_channels()

    def setup_flow_channels(self, num_channels=5):
        """Setup virtual flow channels in the core"""
        self.num_channels = num_channels

        # Define channel radii (dividing core into concentric regions)
        self.channel_radii = [
            self.core_radius * (i / num_channels) for i in range(1, num_channels + 1)
        ]

        # Define channel areas
        self.channel_areas = [
            np.pi * (r**2 - (self.channel_radii[i - 1] if i > 0 else 0) ** 2)
            for i, r in enumerate(self.channel_radii)
        ]

        # Define flow speed variation across channels
        # Center flows slightly faster than outer regions
        self.channel_speeds = [
            self.avg_flow_speed * (1 + 0.2 * (1 - i / num_channels))
            for i in range(num_channels)
        ]

    def calculate_transit_time(self, radial_position):
        """Calculate transit time for a pebble at a given radial position"""
        # Find which channel this position belongs to
        channel_idx = next(
            (i for i, r in enumerate(self.channel_radii) if radial_position <= r),
            len(self.channel_radii) - 1,
        )

        # Get base transit time for this channel
        base_transit_time = (
            self.core_height * 10 / self.channel_speeds[channel_idx]
        )  # hours

        # Add random variation
        variation = random.normalvariate(0, self.flow_variance * 10)  # hours

        return base_transit_time + variation

    def simulate_pebble_path(self, initial_position):
        """Simulate path of a pebble from initial position"""
        r, theta, z = initial_position

        # Calculate transit time
        transit_time = self.calculate_transit_time(r)

        # Create path points (simplified)
        num_points = 20
        time_points = np.linspace(0, transit_time, num_points)
        z_points = self.core_height * (1 - time_points / transit_time)

        # Add some random lateral movement (very simplified)
        r_points = [r + random.normalvariate(0, 0.2) for _ in range(num_points)]
        theta_points = [
            theta + random.normalvariate(0, 0.05) for _ in range(num_points)
        ]

        # Ensure pebble stays within core and follows general downward flow
        r_points = [min(max(r, 0), self.core_radius - 0.5) for r in r_points]
        z_points = np.clip(z_points, 0, self.core_height)

        return {
            "time": time_points,
            "r": r_points,
            "theta": theta_points,
            "z": z_points,
            "transit_time": transit_time,
        }


class RefuelingSystem:
    """Class to simulate the continuous refueling system"""

    def __init__(self, core, target_burnup=90.0):
        self.core = core
        self.target_burnup = target_burnup  # GWd/tHM

        # Pebble parameters
        self.pebble_diameter = 6.0  # cm
        self.pebble_volume = (4 / 3) * np.pi * (self.pebble_diameter / 2) ** 3  # cm^3

        # Calculate number of pebbles in core
        self.packing_fraction = 0.61
        self.core_volume = np.pi * (self.core.radius**2) * self.core.height  # cm^3
        self.fuel_volume = self.core_volume * self.packing_fraction
        self.num_pebbles = int(self.fuel_volume / self.pebble_volume)

        # Refueling parameters
        self.daily_extraction_rate = 1450  # pebbles/day
        self.daily_loading_rate = 325  # pebbles/day

        # Burnup distribution (initialize)
        self.initialize_burnup_distribution()

        # Storage capacities
        self.fresh_storage_capacity = 15000  # pebbles
        self.spent_storage_capacity = 15000  # pebbles

        # Current storage
        self.fresh_storage = self.fresh_storage_capacity  # start full
        self.spent_storage = 0  # start empty

    def initialize_burnup_distribution(self):
        """Initialize the burnup distribution of pebbles in the core"""
        # For a new core, all pebbles have zero burnup
        # For an operating core, distribute according to expected distribution

        # Typical distribution after some time of operation
        self.burnup_bins = [0, 15, 30, 45, 60, 75, 90]

        # Fraction of pebbles in each bin (should sum to 1)
        self.burnup_distribution = [0.15, 0.20, 0.22, 0.18, 0.15, 0.10]

        # Calculate number of pebbles in each bin
        self.pebbles_per_bin = [
            int(frac * self.num_pebbles) for frac in self.burnup_distribution
        ]

        # Adjust to ensure total matches num_pebbles
        while sum(self.pebbles_per_bin) < self.num_pebbles:
            idx = random.randint(0, len(self.pebbles_per_bin) - 1)
            self.pebbles_per_bin[idx] += 1

    def step_refueling(self, time_step, burnup_increment):
        """Simulate refueling for a time step"""
        # time_step in days
        # burnup_increment - average burnup increase per time step

        # Calculate number of pebbles to examine
        pebbles_to_examine = int(self.daily_extraction_rate * time_step)

        # Limit by what's actually in the core
        pebbles_to_examine = min(pebbles_to_examine, self.num_pebbles)

        # Determine how many pebbles from each burnup bin to extract
        # This is a simplified approach - in reality, extraction would be more complex
        extracted_per_bin = [
            int(pebbles_to_examine * frac) for frac in self.burnup_distribution
        ]

        # Adjust to ensure total matches pebbles_to_examine
        while sum(extracted_per_bin) < pebbles_to_examine:
            idx = random.randint(0, len(extracted_per_bin) - 1)
            extracted_per_bin[idx] += 1

        # Calculate how many pebbles reach discharge burnup
        discharged_pebbles = extracted_per_bin[
            -1
        ]  # Assume last bin is at discharge burnup

        # Calculate how many pebbles are recycled
        recycled_pebbles = pebbles_to_examine - discharged_pebbles

        # Calculate how many fresh pebbles to add
        fresh_pebbles = min(self.daily_loading_rate * time_step, self.fresh_storage)

        # Update storage
        self.fresh_storage -= fresh_pebbles
        self.spent_storage += discharged_pebbles

        # Ensure spent storage doesn't exceed capacity
        self.spent_storage = min(self.spent_storage, self.spent_storage_capacity)

        # Update burnup distribution due to burnup increment
        self.update_burnup_distribution(
            burnup_increment, recycled_pebbles, fresh_pebbles
        )

        return {
            "examined": pebbles_to_examine,
            "discharged": discharged_pebbles,
            "recycled": recycled_pebbles,
            "fresh_added": fresh_pebbles,
            "fresh_storage": self.fresh_storage,
            "spent_storage": self.spent_storage,
            "burnup_distribution": list(
                zip(self.burnup_bins[:-1], self.pebbles_per_bin)
            ),
        }

    def update_burnup_distribution(
        self, burnup_increment, recycled_pebbles, fresh_pebbles
    ):
        """Update the burnup distribution after a time step"""
        # Move pebbles up bins based on burnup increment
        # This is a simplified model - in reality, burnup distribution would be continuous

        # Calculate fraction of pebbles that move to next bin
        bin_width = (
            self.burnup_bins[1] - self.burnup_bins[0]
        )  # Assume uniform bin width
        fraction_to_move = burnup_increment / bin_width

        # Move pebbles up bins (from highest to lowest to avoid double-counting)
        for i in range(len(self.pebbles_per_bin) - 1, 0, -1):
            pebbles_to_move = int(self.pebbles_per_bin[i - 1] * fraction_to_move)
            self.pebbles_per_bin[i] += pebbles_to_move
            self.pebbles_per_bin[i - 1] -= pebbles_to_move

        # Add recycled pebbles (distribute them back across bins)
        # In reality, recycled pebbles would go back into specific burnup ranges
        for i in range(len(self.pebbles_per_bin) - 1):  # Don't add to the highest bin
            self.pebbles_per_bin[i] += int(
                recycled_pebbles
                * self.burnup_distribution[i]
                / sum(self.burnup_distribution[:-1])
            )

        # Add fresh pebbles to the lowest bin
        self.pebbles_per_bin[0] += fresh_pebbles

        # Normalize distribution to match total pebbles in core
        total_pebbles = sum(self.pebbles_per_bin)
        if total_pebbles != self.num_pebbles:
            # Adjust to ensure total matches num_pebbles
            diff = self.num_pebbles - total_pebbles
            # Distribute the difference proportionally
            for i in range(len(self.pebbles_per_bin)):
                self.pebbles_per_bin[i] += int(
                    diff * self.pebbles_per_bin[i] / total_pebbles
                )

            # Ensure exact match by adjusting the largest bin
            diff = self.num_pebbles - sum(self.pebbles_per_bin)
            idx = self.pebbles_per_bin.index(max(self.pebbles_per_bin))
            self.pebbles_per_bin[idx] += diff

    def calculate_refueling_impact(self, k_eff):
        """Calculate impact of refueling on reactivity"""
        # Simplified model for reactivity impact of refueling

        # Calculate reactivity of current core
        reactivity_current = (k_eff - 1) / k_eff

        # Calculate average burnup of core (weighted average)
        avg_burnup = (
            sum(
                self.burnup_bins[i] * self.pebbles_per_bin[i]
                for i in range(len(self.pebbles_per_bin))
            )
            / self.num_pebbles
        )

        # Estimate reactivity change from refueling
        # Simplified model: assume linear relationship between burnup and reactivity
        # Fresh fuel adds positive reactivity, removing spent fuel removes negative reactivity
        fresh_fuel_effect = 0.0005  # delta_k/k per day of fresh fuel addition
        spent_fuel_effect = 0.0003  # delta_k/k per day of spent fuel removal

        # Daily reactivity change
        daily_reactivity_change = (self.daily_loading_rate * fresh_fuel_effect) + (
            self.daily_extraction_rate * spent_fuel_effect
        )

        return {
            "current_reactivity": reactivity_current,
            "average_burnup": avg_burnup,
            "daily_reactivity_change": daily_reactivity_change,
        }
