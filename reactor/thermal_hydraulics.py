# reactor/thermal_hydraulics.py
import numpy as np
import math
from utils.constants import *


class ThermalHydraulicsModel:
    """Class for thermal-hydraulics calculations of the XE-100 SMR"""

    def __init__(
        self,
        core_radius=DEFAULT_CORE_RADIUS,
        core_height=DEFAULT_CORE_HEIGHT,
        power=DEFAULT_POWER,
        inlet_temp=260,
        outlet_temp=750,
    ):
        self.core_radius = core_radius  # cm
        self.core_height = core_height  # cm
        self.power = power  # W
        self.inlet_temp = inlet_temp  # °C
        self.outlet_temp = outlet_temp  # °C

        # Calculate core volume
        self.core_volume = math.pi * (self.core_radius**2) * self.core_height  # cm^3

        # Power density
        self.power_density = self.power / self.core_volume  # W/cm^3

        # Helium coolant properties at average temperature
        self.avg_temp = (self.inlet_temp + self.outlet_temp) / 2  # °C
        self.coolant_pressure = 7.0  # MPa
        self.calculate_helium_properties()

        # Calculate mass flow rate
        self.calculate_mass_flow_rate()

    def calculate_helium_properties(self):
        """Calculate helium properties at average temperature and pressure"""
        T_K = self.avg_temp + 273.15  # K
        P_MPa = self.coolant_pressure  # MPa

        # Helium properties (approximate correlations)
        # Density (kg/m^3)
        self.density = 48.14 * (P_MPa / T_K)  # kg/m^3

        # Specific heat capacity (J/kg·K)
        self.cp = 5193.0  # J/kg·K (nearly constant for helium)

        # Thermal conductivity (W/m·K)
        self.conductivity = 0.002682 * (T_K**0.71)  # W/m·K

        # Dynamic viscosity (Pa·s)
        self.viscosity = 3.674e-7 * (T_K**0.7)  # Pa·s

    def calculate_mass_flow_rate(self):
        """Calculate required mass flow rate for the specified power and temperature rise"""
        delta_T = self.outlet_temp - self.inlet_temp  # °C
        self.mass_flow_rate = self.power / (self.cp * delta_T)  # kg/s

    def calculate_temperature_distribution(self, power_distribution):
        """Calculate temperature distribution given power distribution"""
        # This is a simplified model using 1D axial temperature distribution
        # power_distribution should be a function that takes z and returns relative power

        # Discretize the core height
        z_points = np.linspace(0, self.core_height, 100)
        temperatures = np.zeros_like(z_points)

        # Calculate temperatures
        for i, z in enumerate(z_points):
            # Normalized axial position
            z_norm = z / self.core_height

            # Get relative power at this position
            rel_power = power_distribution(z_norm)

            # Simplified temperature calculation
            # T(z) = Tin + ΔT * Integral(0 to z) of rel_power
            if i == 0:
                temperatures[i] = self.inlet_temp
            else:
                # Trapezoidal integration of power up to this point
                power_integral = (
                    np.trapz(
                        [
                            power_distribution(zp / self.core_height)
                            for zp in z_points[: i + 1]
                        ],
                        z_points[: i + 1],
                    )
                    / self.core_height
                )

                temperatures[i] = (
                    self.inlet_temp
                    + (self.outlet_temp - self.inlet_temp) * power_integral
                )

        return z_points, temperatures

    def calculate_pebble_temperature(self, local_power_density, coolant_temp):
        """Calculate temperature distribution in a fuel pebble"""
        # Pebble parameters
        pebble_radius = 3.0  # cm
        fuel_zone_radius = 2.5  # cm
        graphite_shell_thickness = 0.5  # cm

        # Graphite thermal conductivity (W/cm·K)
        k_graphite = 0.3  # W/cm·K

        # Heat transfer coefficient (approximate) (W/cm^2·K)
        h = 0.1  # W/cm^2·K

        # Pebble power (W)
        pebble_volume = (4 / 3) * math.pi * (fuel_zone_radius**3)  # cm^3
        pebble_power = local_power_density * pebble_volume  # W

        # Temperature difference from surface to center (°C)
        # Using simplified spherical conduction equation
        delta_T_conduction = pebble_power / (4 * math.pi * k_graphite * pebble_radius)

        # Temperature difference from coolant to surface (°C)
        delta_T_convection = pebble_power / (4 * math.pi * pebble_radius**2 * h)

        # Temperature at pebble surface (°C)
        surface_temp = coolant_temp + delta_T_convection

        # Temperature at pebble center (°C)
        center_temp = surface_temp + delta_T_conduction

        return {
            "center_temperature": center_temp,
            "surface_temperature": surface_temp,
            "coolant_temperature": coolant_temp,
            "delta_T_conduction": delta_T_conduction,
            "delta_T_convection": delta_T_convection,
        }

    def safety_analysis(self):
        """Perform basic safety analysis"""
        # Maximum allowable fuel temperature
        max_fuel_temp = 1250  # °C

        # Estimate peak fuel temperature
        # Using simplified model - peak is typically at the center of hottest pebble
        peak_power_density = (
            self.power_density * 2.5
        )  # Assume peak-to-average ratio of 2.5
        coolant_temp_at_peak = (
            self.outlet_temp - 50
        )  # Estimate coolant temp at peak location

        peak_temps = self.calculate_pebble_temperature(
            peak_power_density, coolant_temp_at_peak
        )

        # Safety margins
        temp_margin = max_fuel_temp - peak_temps["center_temperature"]

        return {
            "peak_fuel_temperature": peak_temps["center_temperature"],
            "temperature_margin": temp_margin,
            "is_safe": temp_margin > 0,
            "detailed_temps": peak_temps,
        }

    def calculate_pressure_drop(self):
        """Calculate pressure drop across the core"""
        # Constants for the Ergun equation for packed beds
        alpha = 150  # Viscous term constant
        beta = 1.75  # Inertial term constant

        # Pebble diameter in m
        pebble_diameter = 0.06  # m

        # Convert core dimensions to m
        core_height_m = self.core_height / 100  # m
        core_radius_m = self.core_radius / 100  # m

        # Core cross-sectional area
        core_area = math.pi * core_radius_m**2  # m^2

        # Superficial velocity (m/s)
        superficial_velocity = self.mass_flow_rate / (self.density * core_area)  # m/s

        # Void fraction (porosity)
        void_fraction = 1 - DEFAULT_PACKING_FRACTION

        # Calculate pressure drop using Ergun equation
        viscous_term = (
            alpha
            * (1 - void_fraction) ** 2
            / void_fraction**3
            * self.viscosity
            * superficial_velocity
            / pebble_diameter**2
        )
        inertial_term = (
            beta
            * (1 - void_fraction)
            / void_fraction**3
            * self.density
            * superficial_velocity**2
            / pebble_diameter
        )

        pressure_drop = (viscous_term + inertial_term) * core_height_m  # Pa

        return pressure_drop / 1e5  # bar
