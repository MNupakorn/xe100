# reactor/core.py
import numpy as np
import math
from utils.constants import *


class ReactorCore:
    """Class representing the XE-100 SMR reactor core"""

    def __init__(
        self,
        radius=DEFAULT_CORE_RADIUS,
        height=DEFAULT_CORE_HEIGHT,
        packing_fraction=DEFAULT_PACKING_FRACTION,
    ):
        self.radius = radius  # cm
        self.height = height  # cm
        self.packing_fraction = packing_fraction

        # Calculate core volume
        self.volume = math.pi * (self.radius**2) * self.height  # cm^3
        self.fuel_volume = self.volume * self.packing_fraction  # cm^3

        # Geometric buckling
        self.alpha_1 = 2.405  # First zero of Bessel function J0
        self.gamma_1 = self.alpha_1 / self.radius  # cm^-1
        self.mu_1 = math.pi / self.height  # cm^-1
        self.B_g_squared = (self.gamma_1**2) + (self.mu_1**2)  # cm^-2

        # Flux shape
        # phi(r,z) = phi_0 * J0(gamma_1 * r) * sin(mu_1 * z)

    def flux_distribution(self, r, z, phi_0):
        """Calculate neutron flux at position (r,z)"""
        from scipy.special import j0

        if r > self.radius or z < 0 or z > self.height:
            return 0.0

        return phi_0 * j0(self.gamma_1 * r) * math.sin(self.mu_1 * z)

    def average_flux(self, phi_0):
        """Calculate average flux in the core"""
        # This is an approximation based on integration of the flux shape
        return phi_0 * 0.353

    def peak_to_average_ratio(self):
        """Calculate peak-to-average flux ratio in the core"""
        # Peak flux occurs at r=0, z=H/2
        peak_location_factor = 1.0 * math.sin(math.pi / 2)  # J0(0) = 1, sin(pi/2) = 1
        avg_factor = 0.353

        return peak_location_factor / avg_factor  # Approximately 2.83

    def calculate_volume_fraction(self, r_min, r_max, z_min, z_max):
        """Calculate the fraction of core volume in a specified region"""
        if r_min < 0 or r_max > self.radius or z_min < 0 or z_max > self.height:
            raise ValueError("Region boundaries must be within core dimensions")

        total_volume = self.volume
        region_volume = math.pi * (r_max**2 - r_min**2) * (z_max - z_min)

        return region_volume / total_volume
