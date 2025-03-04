# simulation/burnup.py
import numpy as np
from scipy.integrate import solve_ivp
from utils.constants import *
import matplotlib.pyplot as plt


class BurnupSimulation:
    """Class for detailed burnup simulations"""

    def __init__(self, initial_composition, neutron_flux, cross_sections=None):
        """
        Initialize burnup simulation

        Parameters:
        -----------
        initial_composition : dict
            Dictionary with isotope names as keys and concentrations (atoms/cm^3) as values

        neutron_flux : float
            Average neutron flux in n/cm^2/s

        cross_sections : dict, optional
            Dictionary with cross section data for each isotope
        """
        self.composition = initial_composition.copy()
        self.flux = neutron_flux

        # Set up default cross sections if not provided
        if cross_sections is None:
            self.cross_sections = {
                "U235": {"capture": 50.0, "fission": 300.0, "n_gamma": 8.7},
                "U238": {"capture": 2.6, "fission": 0.103, "n_gamma": 2.7},
                "Pu239": {"capture": 286.0, "fission": 742.0, "n_gamma": 40.0},
                "Pu240": {"capture": 285.0, "fission": 0.064, "n_gamma": 8.3},
                "Pu241": {"capture": 368.0, "fission": 1010.0, "n_gamma": 14.0},
                "Xe135": {"capture": 2.65e6, "fission": 0.0, "n_gamma": 2.65e6},
                "Sm149": {"capture": 4.15e4, "fission": 0.0, "n_gamma": 4.15e4},
                "I135": {"capture": 7.0, "fission": 0.0, "n_gamma": 7.0},
                "Pm149": {"capture": 1500.0, "fission": 0.0, "n_gamma": 1500.0},
            }
        else:
            self.cross_sections = cross_sections

        # Set up decay constants (in 1/s)
        self.decay_constants = {
            "U235": 3.12e-17,  # ~700 million years
            "U238": 4.92e-18,  # ~4.5 billion years
            "Pu239": 9.11e-13,  # ~24,000 years
            "Pu240": 3.35e-12,  # ~6,560 years
            "Pu241": 1.53e-9,  # ~14.3 years
            "Xe135": 2.11e-5,  # ~9.14 hours
            "I135": 2.93e-5,  # ~6.57 hours
            "Pm149": 3.63e-6,  # ~53.1 hours
        }

        # Set up fission yields
        self.fission_yields = {
            "U235": {"Xe135": 0.002, "I135": 0.063, "Pm149": 0.011},  # direct yield
            "Pu239": {"Xe135": 0.002, "I135": 0.076, "Pm149": 0.0123},  # direct yield
        }

        # Set up neutrons per fission
        self.nu = {
            "U235": 2.43,
            "U238": 2.53,
            "Pu239": 2.88,
            "Pu240": 2.81,
            "Pu241": 2.95,
        }

        # Initialize results storage
        self.results = {"time": [], "composition": [], "k_eff": [], "burnup": []}

    def get_isotope_indices(self, isotope_list):
        """Create mapping between isotope names and indices in the ODE system"""
        return {iso: i for i, iso in enumerate(isotope_list)}

    def differential_equations(self, t, y, isotope_list, isotope_indices):
        """Define the system of differential equations for burnup calculation"""
        dydt = np.zeros_like(y)

        # Extract current concentrations from y vector
        concentrations = {iso: y[isotope_indices[iso]] for iso in isotope_list}

        # Calculate reaction rates and isotope production/destruction
        for i, isotope in enumerate(isotope_list):
            # Decay term
            if isotope in self.decay_constants:
                decay_rate = self.decay_constants[isotope] * concentrations[isotope]
                dydt[i] -= decay_rate

            # Neutron reaction terms
            if isotope in self.cross_sections:
                # Convert cross sections from barns to cm^2
                xs_capture = self.cross_sections[isotope]["capture"] * BARN_TO_CM2
                xs_fission = self.cross_sections[isotope]["fission"] * BARN_TO_CM2

                # Calculate reaction rates
                capture_rate = self.flux * xs_capture * concentrations[isotope]
                fission_rate = self.flux * xs_fission * concentrations[isotope]

                # Depletion due to reactions
                dydt[i] -= capture_rate + fission_rate

            # Production terms

            # U-238 (n,γ) → Np-239 → Pu-239
            if isotope == "Pu239" and "U238" in concentrations:
                xs_n_gamma_U238 = self.cross_sections["U238"]["n_gamma"] * BARN_TO_CM2
                dydt[i] += self.flux * xs_n_gamma_U238 * concentrations["U238"]

            # Fission product yields
            for parent in ["U235", "Pu239"]:
                if parent in concentrations and isotope in ["I135", "Pm149"]:
                    if isotope in self.fission_yields[parent]:
                        xs_fission_parent = (
                            self.cross_sections[parent]["fission"] * BARN_TO_CM2
                        )
                        fission_rate_parent = (
                            self.flux * xs_fission_parent * concentrations[parent]
                        )
                        yield_fraction = self.fission_yields[parent][isotope]
                        dydt[i] += fission_rate_parent * yield_fraction

            # Xenon-135 production from I-135 decay
            if isotope == "Xe135" and "I135" in concentrations:
                dydt[i] += self.decay_constants["I135"] * concentrations["I135"]

            # Sm-149 production from Pm-149 decay
            if isotope == "Sm149" and "Pm149" in concentrations:
                dydt[i] += self.decay_constants["Pm149"] * concentrations["Pm149"]

        return dydt

    def simulate(self, time_span, time_points=100):
        """Run burnup simulation for specified time span"""
        # Convert time_span to seconds (from days)
        t_span = (time_span[0] * 86400, time_span[1] * 86400)

        # Create t_eval points for solution output
        t_eval = np.linspace(t_span[0], t_span[1], time_points)

        # Get list of isotopes in the simulation
        isotope_list = list(self.composition.keys())
        isotope_indices = self.get_isotope_indices(isotope_list)

        # Create initial conditions vector
        y0 = np.array([self.composition[iso] for iso in isotope_list])

        # Solve the ODE system
        solution = solve_ivp(
            lambda t, y: self.differential_equations(
                t, y, isotope_list, isotope_indices
            ),
            t_span,
            y0,
            method="LSODA",
            t_eval=t_eval,
            atol=1e-10,
            rtol=1e-8,
        )

        # Process results
        time_days = solution.t / 86400  # Convert back to days

        # Store composition history
        composition_history = []
        for i in range(len(time_days)):
            comp = {iso: solution.y[isotope_indices[iso]][i] for iso in isotope_list}
            composition_history.append(comp)

        # Calculate k_eff history
        k_eff_history = []
        for comp in composition_history:
            k_eff_history.append(self.calculate_k_eff(comp))

        # Calculate burnup history
        burnup_history = []
        initial_u235 = self.composition["U235"]
        for comp in composition_history:
            u235_consumed = initial_u235 - comp["U235"]
            # Convert to burnup in GWd/tHM
            energy_per_fission = MEV_PER_FISSION * EV_TO_JOULE  # J
            energy_per_atom = energy_per_fission

            # Calculate heavy metal density (approximate)
            hm_density = 0.0
            for iso in ["U235", "U238", "Pu239", "Pu240", "Pu241"]:
                if iso in self.composition:
                    hm_density += self.composition[iso] * (
                        235
                        if iso == "U235"
                        else (
                            238
                            if iso == "U238"
                            else (
                                239
                                if iso == "Pu239"
                                else 240 if iso == "Pu240" else 241
                            )
                        )
                    )  # g/mol

            hm_density /= AVOGADRO_NUMBER  # g/cm^3

            # Calculate burnup
            burnup = (
                u235_consumed * energy_per_atom / (hm_density * 1e-6) * 1e-9
            )  # GWd/tHM
            burnup_history.append(burnup)

        # Store results
        self.results = {
            "time": time_days,
            "composition": composition_history,
            "k_eff": k_eff_history,
            "burnup": burnup_history,
        }

        return self.results

    def calculate_k_eff(self, composition):
        """Calculate k_eff based on current composition"""
        numerator = 0.0  # Production term
        denominator = 0.0  # Absorption term

        for isotope, density in composition.items():
            if isotope in self.cross_sections:
                # Convert cross sections from barns to cm^2
                xs_capture = self.cross_sections[isotope]["capture"] * BARN_TO_CM2
                xs_fission = self.cross_sections[isotope]["fission"] * BARN_TO_CM2

                # Production term (nu * fission)
                if isotope in self.nu and xs_fission > 0:
                    numerator += self.nu[isotope] * xs_fission * density

                # Absorption term (capture + fission)
                denominator += (xs_capture + xs_fission) * density

        # Add leakage term to denominator (simplified)
        # Using a typical buckling value
        buckling = 1.73e-4  # cm^-2
        leakage = DEFAULT_DIFFUSION_COEF * buckling * denominator

        # Calculate k_eff
        k_eff = numerator / (denominator + leakage) if denominator > 0 else 0.0

        return k_eff

    def plot_results(self):
        """Plot key results from the simulation"""
        if len(self.results["time"]) == 0:
            print("No results to plot. Run simulate() first.")
            return

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # Plot k_eff
        ax1.plot(self.results["time"], self.results["k_eff"], "b-")
        ax1.axhline(y=1.0, color="r", linestyle="--")
        ax1.set_ylabel("k_eff")
        ax1.set_title("Effective Multiplication Factor")
        ax1.grid(True)

        # Plot burnup
        ax2.plot(self.results["time"], self.results["burnup"], "g-")
        ax2.set_ylabel("Burnup (GWd/tHM)")
        ax2.set_title("Burnup Evolution")
        ax2.grid(True)

        # Plot isotope concentrations
        isotopes_to_plot = ["U235", "U238", "Pu239", "Xe135", "Sm149"]
        for isotope in isotopes_to_plot:
            if isotope in self.composition:
                values = [comp[isotope] for comp in self.results["composition"]]
                normalization = (
                    self.composition[isotope]
                    if isotope in ["U235", "U238"]
                    else max(values)
                )
                if normalization > 0:
                    normalized_values = [v / normalization for v in values]
                    ax3.plot(self.results["time"], normalized_values, label=isotope)

        ax3.set_xlabel("Time (days)")
        ax3.set_ylabel("Relative Concentration")
        ax3.set_title("Isotope Concentration Evolution")
        ax3.legend(loc="best")
        ax3.grid(True)

        plt.tight_layout()
        return fig

    def get_results_dataframe(self):
        """Convert results to pandas DataFrame"""
        import pandas as pd

        # Create basic dataframe with time, k_eff, and burnup
        df = pd.DataFrame(
            {
                "time": self.results["time"],
                "k_eff": self.results["k_eff"],
                "burnup": self.results["burnup"],
            }
        )

        # Add isotope concentrations
        for isotope in self.composition.keys():
            df[isotope] = [comp[isotope] for comp in self.results["composition"]]

        return df
