# simulation/steady_state.py
import numpy as np
from utils.constants import *


class SteadyStateCalculator:
    """Class for performing steady-state calculations for the reactor"""

    def __init__(self, core, neutronics):
        """
        Initialize steady-state calculator

        Parameters:
        -----------
        core : ReactorCore
            Instance of ReactorCore class

        neutronics : NeutronicsModel
            Instance of NeutronicsModel class
        """
        self.core = core
        self.neutronics = neutronics

    def calculate_critical_enrichment(
        self, min_enrichment=3.0, max_enrichment=20.0, tolerance=0.001
    ):
        """
        Calculate the critical enrichment for the given core configuration

        Parameters:
        -----------
        min_enrichment : float
            Minimum enrichment to consider (%)

        max_enrichment : float
            Maximum enrichment to consider (%)

        tolerance : float
            Tolerance for k_eff - 1.0

        Returns:
        --------
        dict
            Dictionary with critical enrichment and corresponding neutronic parameters
        """

        # Function to calculate k_eff for a given enrichment
        def calculate_k_eff(enrichment):
            # Scale cross sections based on enrichment
            # This is a simplified model - in reality, the relationship is more complex

            # Base cross section values (at 4.5% enrichment)
            base_enrichment = 4.5

            # Adjust absorption cross section
            sigma_a_base = self.neutronics.sigma_a

            # Adjust fission cross section
            sigma_f_base = self.neutronics.sigma_f

            # Scale cross sections (linear approximation)
            sigma_a = sigma_a_base * (0.8 + 0.2 * enrichment / base_enrichment)
            sigma_f = sigma_f_base * (enrichment / base_enrichment)

            # Create new neutronics model with adjusted cross sections
            new_neutronics = type(self.neutronics)(
                D=self.neutronics.D,
                sigma_a=sigma_a,
                sigma_f=sigma_f,
                nu=self.neutronics.nu,
            )

            # Calculate k_eff
            return new_neutronics.calculate_k_eff(self.core.B_g_squared)

        # Use bisection method to find critical enrichment
        a, b = min_enrichment, max_enrichment
        k_a, k_b = calculate_k_eff(a), calculate_k_eff(b)

        # Check if solution exists in the range
        if (k_a - 1.0) * (k_b - 1.0) > 0:
            # Both have same sign relative to 1.0, no solution in range
            if abs(k_a - 1.0) < abs(k_b - 1.0):
                return {
                    "enrichment": a,
                    "k_eff": k_a,
                    "converged": False,
                    "message": f"No solution found in range. Closest at enrichment={a}%, k_eff={k_a}",
                }
            else:
                return {
                    "enrichment": b,
                    "k_eff": k_b,
                    "converged": False,
                    "message": f"No solution found in range. Closest at enrichment={b}%, k_eff={k_b}",
                }

        # Bisection method
        iterations = 0
        max_iterations = 100

        while abs(b - a) > 0.01 and iterations < max_iterations:
            c = (a + b) / 2
            k_c = calculate_k_eff(c)

            if abs(k_c - 1.0) < tolerance:
                # Found solution within tolerance
                return {
                    "enrichment": c,
                    "k_eff": k_c,
                    "converged": True,
                    "iterations": iterations,
                    "message": f"Converged to k_eff={k_c} at enrichment={c}%",
                }

            if (k_c - 1.0) * (k_a - 1.0) < 0:
                b, k_b = c, k_c
            else:
                a, k_a = c, k_c

            iterations += 1

        # Use linear interpolation for final result
        if iterations == max_iterations:
            # Didn't converge within max iterations, but still approximate
            result_enrichment = a + (1.0 - k_a) * (b - a) / (k_b - k_a)
            result_k_eff = calculate_k_eff(result_enrichment)
        else:
            result_enrichment = (a + b) / 2
            result_k_eff = calculate_k_eff(result_enrichment)

        return {
            "enrichment": result_enrichment,
            "k_eff": result_k_eff,
            "converged": iterations < max_iterations,
            "iterations": iterations,
            "message": f"{'Converged' if iterations < max_iterations else 'Did not converge'} to k_eff={result_k_eff} at enrichment={result_enrichment}%",
        }

    def calculate_critical_radius(
        self, min_radius=100, max_radius=300, tolerance=0.001
    ):
        """
        Calculate the critical radius for the given neutronic parameters

        Parameters:
        -----------
        min_radius : float
            Minimum radius to consider (cm)

        max_radius : float
            Maximum radius to consider (cm)

        tolerance : float
            Tolerance for k_eff - 1.0

        Returns:
        --------
        dict
            Dictionary with critical radius and corresponding k_eff
        """

        # Function to calculate k_eff for a given radius
        def calculate_k_eff(radius):
            # Create a new core with the given radius
            new_core = type(self.core)(
                radius=radius,
                height=self.core.height,
                packing_fraction=self.core.packing_fraction,
            )

            # Calculate k_eff
            return self.neutronics.calculate_k_eff(new_core.B_g_squared)

        # Use bisection method to find critical radius
        a, b = min_radius, max_radius
        k_a, k_b = calculate_k_eff(a), calculate_k_eff(b)

        # Check if solution exists in the range
        if (k_a - 1.0) * (k_b - 1.0) > 0:
            # Both have same sign relative to 1.0, no solution in range
            if abs(k_a - 1.0) < abs(k_b - 1.0):
                return {
                    "radius": a,
                    "k_eff": k_a,
                    "converged": False,
                    "message": f"No solution found in range. Closest at radius={a} cm, k_eff={k_a}",
                }
            else:
                return {
                    "radius": b,
                    "k_eff": k_b,
                    "converged": False,
                    "message": f"No solution found in range. Closest at radius={b} cm, k_eff={k_b}",
                }

        # Bisection method
        iterations = 0
        max_iterations = 100

        while abs(b - a) > 0.1 and iterations < max_iterations:
            c = (a + b) / 2
            k_c = calculate_k_eff(c)

            if abs(k_c - 1.0) < tolerance:
                # Found solution within tolerance
                return {
                    "radius": c,
                    "k_eff": k_c,
                    "converged": True,
                    "iterations": iterations,
                    "message": f"Converged to k_eff={k_c} at radius={c} cm",
                }

            if (k_c - 1.0) * (k_a - 1.0) < 0:
                b, k_b = c, k_c
            else:
                a, k_a = c, k_c

            iterations += 1

        # Use linear interpolation for final result
        if iterations == max_iterations:
            # Didn't converge within max iterations, but still approximate
            result_radius = a + (1.0 - k_a) * (b - a) / (k_b - k_a)
            result_k_eff = calculate_k_eff(result_radius)
        else:
            result_radius = (a + b) / 2
            result_k_eff = calculate_k_eff(result_radius)

        return {
            "radius": result_radius,
            "k_eff": result_k_eff,
            "converged": iterations < max_iterations,
            "iterations": iterations,
            "message": f"{'Converged' if iterations < max_iterations else 'Did not converge'} to k_eff={result_k_eff} at radius={result_radius} cm",
        }

    def calculate_control_rod_worth(self, rod_worth_coefficient=0.005):
        """
        Calculate control rod worth for reactivity control

        Parameters:
        -----------
        rod_worth_coefficient : float
            Coefficient for control rod worth calculation (cm^-1)

        Returns:
        --------
        dict
            Dictionary with control rod worth data
        """
        # This is a simplified model for control rod worth
        # In reality, control rod worth would be calculated with more detailed models

        # Calculate reactivity worth as function of insertion depth
        max_insertion = self.core.height
        insertion_points = np.linspace(0, max_insertion, 20)

        # Calculate reactivity worth at each insertion point
        reactivity_worth = []
        for insertion in insertion_points:
            # Simple S-curve model for reactivity worth vs insertion
            normalized_insertion = insertion / max_insertion
            worth = (1 - np.exp(-4 * normalized_insertion)) / (
                1 + np.exp(-4 * (normalized_insertion - 0.5))
            )
            worth *= rod_worth_coefficient * max_insertion
            reactivity_worth.append(worth)

        # Calculate differential worth (derivative of reactivity worth)
        differential_worth = np.gradient(reactivity_worth, insertion_points)

        return {
            "insertion_points": insertion_points,
            "reactivity_worth": reactivity_worth,
            "differential_worth": differential_worth,
            "total_worth": reactivity_worth[-1],
        }

    def calculate_temperature_coefficients(self, delta_T=50):
        """
        Calculate fuel and moderator temperature coefficients

        Parameters:
        -----------
        delta_T : float
            Temperature difference for coefficient calculation (°C)

        Returns:
        --------
        dict
            Dictionary with temperature coefficient data
        """
        # Baseline k_eff
        baseline_k_eff = self.neutronics.calculate_k_eff(self.core.B_g_squared)

        # Fuel temperature coefficient estimation
        # Increase in fuel temperature primarily affects resonance absorption in U-238
        # which increases the absorption cross section

        # Estimate new sigma_a with increased fuel temperature
        sigma_a_hot = self.neutronics.sigma_a * (
            1 + 0.01 * delta_T / 100
        )  # ~1% increase per 100°C

        # Create new neutronics model with adjusted cross sections
        neutronics_fuel_hot = type(self.neutronics)(
            D=self.neutronics.D,
            sigma_a=sigma_a_hot,
            sigma_f=self.neutronics.sigma_f,
            nu=self.neutronics.nu,
        )

        # Calculate k_eff with hot fuel
        k_eff_fuel_hot = neutronics_fuel_hot.calculate_k_eff(self.core.B_g_squared)

        # Calculate fuel temperature coefficient
        fuel_temp_coeff = (k_eff_fuel_hot - baseline_k_eff) / (delta_T * baseline_k_eff)

        # Moderator temperature coefficient estimation
        # Increase in moderator temperature primarily affects the diffusion coefficient
        # and moderator density

        # Estimate new D with increased moderator temperature
        D_hot = self.neutronics.D * (
            1 + 0.005 * delta_T / 100
        )  # ~0.5% increase per 100°C

        # Create new neutronics model with adjusted diffusion coefficient
        neutronics_mod_hot = type(self.neutronics)(
            D=D_hot,
            sigma_a=self.neutronics.sigma_a,
            sigma_f=self.neutronics.sigma_f,
            nu=self.neutronics.nu,
        )

        # Calculate k_eff with hot moderator
        k_eff_mod_hot = neutronics_mod_hot.calculate_k_eff(self.core.B_g_squared)

        # Calculate moderator temperature coefficient
        mod_temp_coeff = (k_eff_mod_hot - baseline_k_eff) / (delta_T * baseline_k_eff)

        # Combined temperature coefficient
        combined_temp_coeff = fuel_temp_coeff + mod_temp_coeff

        return {
            "baseline_k_eff": baseline_k_eff,
            "k_eff_fuel_hot": k_eff_fuel_hot,
            "k_eff_mod_hot": k_eff_mod_hot,
            "fuel_temp_coeff": fuel_temp_coeff,
            "mod_temp_coeff": mod_temp_coeff,
            "combined_temp_coeff": combined_temp_coeff,
            "delta_T": delta_T,
        }
