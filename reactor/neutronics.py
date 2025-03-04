# reactor/neutronics.py
import numpy as np
from scipy.integrate import solve_ivp
from utils.constants import *


class NeutronicsModel:
    """Class for neutronics calculations"""

    def __init__(
        self,
        D=DEFAULT_DIFFUSION_COEF,
        sigma_a=DEFAULT_SIGMA_A,
        sigma_f=DEFAULT_SIGMA_F,
        nu=DEFAULT_NU,
    ):
        self.D = D  # Diffusion coefficient (cm)
        self.sigma_a = sigma_a  # Absorption cross section (cm^-1)
        self.sigma_f = sigma_f  # Fission cross section (cm^-1)
        self.nu = nu  # Neutrons per fission

        # Derived parameters
        self.nu_sigma_f = nu * sigma_f  # (cm^-1)
        self.L_squared = D / sigma_a  # (cm^2)
        self.L = np.sqrt(self.L_squared)  # (cm)
        self.k_inf = self.nu_sigma_f / self.sigma_a

    def calculate_k_eff(self, B_g_squared):
        """Calculate k_eff using diffusion theory"""
        k_eff = self.k_inf / (1 + self.L_squared * B_g_squared)
        return k_eff

    def calculate_material_buckling(self):
        """Calculate material buckling B_m^2"""
        B_m_squared = (self.k_inf - 1) / self.L_squared
        return B_m_squared

    def calculate_phi_0(self, power, core_volume, avg_flux_factor=0.353):
        """Calculate phi_0 based on reactor power"""
        # P = Sigma_f * phi_avg * E_fission * Volume
        # phi_avg = phi_0 * avg_flux_factor

        energy_per_fission = MEV_PER_FISSION * EV_TO_JOULE  # J

        phi_avg = power / (self.sigma_f * energy_per_fission * core_volume)
        phi_0 = phi_avg / avg_flux_factor

        return phi_0

    def calculate_critical_enrichment(self, initial_enrichment, target_k_eff=1.0):
        """Estimate critical enrichment to achieve target k_eff"""
        # This is a simplified model - in reality, the relationship
        # between enrichment and k_eff is more complex

        # Approximate relationship: k_eff ~ enrichment
        current_k_eff = self.k_inf / (1 + self.L_squared * DEFAULT_B_G_SQUARED)
        ratio = target_k_eff / current_k_eff

        # Adjust enrichment
        critical_enrichment = initial_enrichment * ratio

        # Limit to realistic values
        return max(3.0, min(20.0, critical_enrichment))  # %

    def calculate_control_rod_worth(
        self, insertion_depth, max_insertion_depth, max_worth=0.05
    ):
        """Calculate control rod worth based on insertion depth"""
        # insertion_depth and max_insertion_depth in cm
        # max_worth is the reactivity worth (delta_k/k) at full insertion

        # Simplified S-curve model for rod worth vs. insertion
        if insertion_depth <= 0:
            return 0.0

        normalized_depth = min(1.0, insertion_depth / max_insertion_depth)

        # S-curve formula
        worth = (
            max_worth
            * (1 - np.exp(-4 * normalized_depth))
            / (1 + np.exp(-8 * (normalized_depth - 0.5)))
        )

        return worth  # delta_k/k

    def calculate_xenon_equilibrium(self, phi_avg):
        """Calculate equilibrium Xenon-135 concentration"""
        # phi_avg: average neutron flux (n/cm^2-s)

        # Yield and decay constants
        gamma_i = 0.063  # I-135 yield from fission
        gamma_x = 0.003  # Direct Xe-135 yield from fission
        lambda_i = 2.93e-5  # I-135 decay constant (s^-1)
        lambda_x = 2.11e-5  # Xe-135 decay constant (s^-1)
        sigma_x = 2.65e-18  # Xe-135 microscopic absorption cross section (cm^2)

        # Fission rate
        fission_rate = self.sigma_f * phi_avg  # fissions/cm^3-s

        # Equilibrium I-135 concentration
        I_eq = gamma_i * fission_rate / lambda_i  # atoms/cm^3

        # Equilibrium Xe-135 concentration
        Xe_eq = (gamma_x * fission_rate + lambda_i * I_eq) / (
            lambda_x + sigma_x * phi_avg
        )  # atoms/cm^3

        # Reactivity worth of equilibrium Xenon
        xenon_worth = -(sigma_x * Xe_eq) / self.sigma_a  # delta_k/k

        return {
            "I135_concentration": I_eq,
            "Xe135_concentration": Xe_eq,
            "reactivity_worth": xenon_worth,
        }
