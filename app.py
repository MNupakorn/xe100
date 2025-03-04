# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reactor.core import ReactorCore
from reactor.neutronics import NeutronicsModel
from reactor.thermal_hydraulics import ThermalHydraulicsModel
from reactor.fuel import Pebble, FuelManagement
from reactor.refueling import RefuelingSystem
from simulation.time_evolution import ReactorSimulation
from simulation.burnup import BurnupSimulation
from simulation.steady_state import SteadyStateCalculator
from utils.plotting import plot_reactor_parameters, plot_interactive_reactor
from utils.data_manager import DataManager
from utils.constants import *
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(layout="wide", page_title="XE-100 SMR Simulator", page_icon="🔬")

# Initialize data manager
data_manager = DataManager()

# Initialize session state variables if they don't exist
if "results" not in st.session_state:
    st.session_state.results = None
if "core" not in st.session_state:
    st.session_state.core = None
if "neutronics" not in st.session_state:
    st.session_state.neutronics = None
if "thermal" not in st.session_state:
    st.session_state.thermal = None
if "refueling" not in st.session_state:
    st.session_state.refueling = None
if "phi_0" not in st.session_state:
    st.session_state.phi_0 = None
if "simulation_time" not in st.session_state:
    st.session_state.simulation_time = 0

# Title and description
st.title("XE-100 SMR Reactor Simulator")
st.markdown(
    """
This application simulates the operation of an XE-100 Small Modular Reactor (SMR) 
with TRISO fuel pebbles using a pebble bed design.
"""
)

# Create sidebar for inputs
st.sidebar.header("Reactor Parameters")

core_radius = st.sidebar.slider("Core Radius (cm)", 100.0, 300.0, 190.0, 10.0)
core_height = st.sidebar.slider("Core Height (cm)", 500.0, 1200.0, 900.0, 50.0)
enrichment = st.sidebar.slider("Fuel Enrichment (%)", 3.0, 10.0, 4.5, 0.1)
power = (
    st.sidebar.slider("Reactor Power (MWth)", 20.0, 200.0, 80.0, 5.0) * 1e6
)  # Convert to W

# Advanced parameters
st.sidebar.header("Advanced Parameters")
show_advanced = st.sidebar.checkbox("Show Advanced Parameters", False)

if show_advanced:
    packing_fraction = st.sidebar.slider(
        "Pebble Packing Fraction", 0.50, 0.70, 0.61, 0.01
    )
    coolant_inlet_temp = st.sidebar.slider(
        "Coolant Inlet Temperature (°C)", 200.0, 350.0, 260.0, 5.0
    )
    coolant_outlet_temp = st.sidebar.slider(
        "Coolant Outlet Temperature (°C)", 600.0, 950.0, 750.0, 5.0
    )
    target_burnup = st.sidebar.slider(
        "Target Discharge Burnup (GWd/tHM)", 60.0, 120.0, 90.0, 5.0
    )
else:
    packing_fraction = 0.61
    coolant_inlet_temp = 260.0
    coolant_outlet_temp = 750.0
    target_burnup = 90.0

# Simulation parameters
st.sidebar.header("Simulation Parameters")
simulation_time = st.sidebar.slider("Simulation Time (days)", 1, 3650, 365, 1)
time_step = st.sidebar.slider("Time Step (days)", 0.1, 30.0, 1.0, 0.1)

# Run simulation button
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        # Create basic models
        try:
            # Create core model
            core = ReactorCore(
                radius=core_radius,
                height=core_height,
                packing_fraction=packing_fraction,
            )

            # Create neutronics model
            neutronics = NeutronicsModel()

            # Create thermal-hydraulics model
            thermal = ThermalHydraulicsModel(
                core_radius=core_radius,
                core_height=core_height,
                power=power,
                inlet_temp=coolant_inlet_temp,
                outlet_temp=coolant_outlet_temp,
            )

            # Create refueling system
            refueling = RefuelingSystem(core, target_burnup=target_burnup)

            # Create and run simulation
            simulation = ReactorSimulation(
                core_radius=core_radius,
                core_height=core_height,
                enrichment=enrichment,
                power=power,
            )

            # Set time step
            simulation.time_step = time_step

            # Run simulation
            results = simulation.run_simulation(simulation_time)

            # Store results in session state
            st.session_state.results = results
            st.session_state.core = core
            st.session_state.neutronics = neutronics
            st.session_state.thermal = thermal
            st.session_state.refueling = refueling
            st.session_state.phi_0 = simulation.phi_0
            st.session_state.simulation_time = simulation_time

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_manager.save_simulation_results(results, f"simulation_{timestamp}")

            st.success("Simulation completed successfully!")

        except Exception as e:
            st.error(f"Error during simulation: {str(e)}")
            st.exception(e)

# Display results if available
if st.session_state.results is not None:
    st.header("Simulation Results")

    # Display basic metrics
    try:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if hasattr(st.session_state, "core") and st.session_state.core is not None:
                core_volume = st.session_state.core.volume / 1e6  # Convert to m³
                st.metric("Core Volume", f"{core_volume:.2f} m³")
            else:
                st.metric("Core Volume", "N/A")

        with col2:
            try:
                if (
                    "k_eff" in st.session_state.results
                    and isinstance(
                        st.session_state.results["k_eff"], (list, np.ndarray, pd.Series)
                    )
                    and len(st.session_state.results["k_eff"]) > 0
                ):
                    initial_k_eff = (
                        st.session_state.results["k_eff"].iloc[0]
                        if isinstance(st.session_state.results["k_eff"], pd.Series)
                        else st.session_state.results["k_eff"][0]
                    )
                    st.metric("Initial k_eff", f"{initial_k_eff:.4f}")
                else:
                    st.metric("Initial k_eff", "N/A")
            except Exception as e:
                st.metric("Initial k_eff", "Error")

        with col3:
            try:
                if (
                    "k_eff" in st.session_state.results
                    and isinstance(
                        st.session_state.results["k_eff"], (list, np.ndarray, pd.Series)
                    )
                    and len(st.session_state.results["k_eff"]) > 0
                ):
                    final_k_eff = (
                        st.session_state.results["k_eff"].iloc[-1]
                        if isinstance(st.session_state.results["k_eff"], pd.Series)
                        else st.session_state.results["k_eff"][-1]
                    )
                    st.metric("Final k_eff", f"{final_k_eff:.4f}")
                else:
                    st.metric("Final k_eff", "N/A")
            except Exception as e:
                st.metric("Final k_eff", "Error")

        with col4:
            try:
                if (
                    "burnup_avg" in st.session_state.results
                    and isinstance(
                        st.session_state.results["burnup_avg"],
                        (list, np.ndarray, pd.Series),
                    )
                    and len(st.session_state.results["burnup_avg"]) > 0
                ):
                    final_burnup = (
                        st.session_state.results["burnup_avg"].iloc[-1]
                        if isinstance(st.session_state.results["burnup_avg"], pd.Series)
                        else st.session_state.results["burnup_avg"][-1]
                    )
                    st.metric("Final Burnup", f"{final_burnup:.2f} GWd/tHM")
                else:
                    st.metric("Final Burnup", "N/A")
            except Exception as e:
                st.metric("Final Burnup", "Error")
    except Exception as e:
        st.warning(f"Could not display metrics: {str(e)}")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Time Evolution", "Fuel Composition", "Reactor Physics", "Thermal Analysis"]
    )

    with tab1:
        st.subheader("Reactor Parameters Over Time")

        try:
            # Plot interactive charts using custom plotting function
            interactive_chart = plot_interactive_reactor(st.session_state.results)
            st.plotly_chart(interactive_chart, use_container_width=True)

            # Show data table with filtering
            st.subheader("Simulation Data")

            # Check if results is a DataFrame
            if isinstance(st.session_state.results, pd.DataFrame):
                # Allow user to select columns to display
                all_columns = st.session_state.results.columns.tolist()
                default_columns = ["time", "k_eff", "burnup_avg"]

                # Ensure default columns exist in the data
                default_columns = [col for col in default_columns if col in all_columns]

                selected_columns = st.multiselect(
                    "Select columns to display",
                    all_columns,
                    default=(
                        default_columns
                        if default_columns
                        else all_columns[: min(5, len(all_columns))]
                    ),
                )

                if selected_columns:
                    st.dataframe(st.session_state.results[selected_columns])
                else:
                    st.dataframe(st.session_state.results)

                # Download button for data
                csv_data = st.session_state.results.to_csv(index=False)

                st.download_button(
                    label="Download data as CSV",
                    data=csv_data,
                    file_name=f"xe100_simulation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                # Convert to DataFrame if it's a dict
                if isinstance(st.session_state.results, dict):
                    # Convert any arrays or lists in the dict to have the same length
                    max_length = 0
                    for key, value in st.session_state.results.items():
                        if isinstance(value, (list, np.ndarray)):
                            max_length = max(max_length, len(value))

                    # Create a normalized dict for DataFrame conversion
                    df_dict = {}
                    for key, value in st.session_state.results.items():
                        if (
                            isinstance(value, (list, np.ndarray))
                            and len(value) == max_length
                        ):
                            df_dict[key] = value

                    # Convert to DataFrame
                    if df_dict:
                        results_df = pd.DataFrame(df_dict)
                        st.dataframe(results_df)

                        # Download button for data
                        csv_data = results_df.to_csv(index=False)

                        st.download_button(
                            label="Download data as CSV",
                            data=csv_data,
                            file_name=f"xe100_simulation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
                    else:
                        st.write(
                            "Results data structure is not suitable for tabular display."
                        )
                else:
                    st.write("Results not available in a suitable format for display.")
        except Exception as e:
            st.error(f"Error generating time evolution plots: {str(e)}")
            st.exception(e)

    with tab2:
        st.subheader("Fuel Composition Evolution")

        try:
            # Check if we have fuel composition data
            fuel_data_available = False
            required_keys = ["U235", "U238"]
            if isinstance(st.session_state.results, pd.DataFrame):
                fuel_data_available = all(
                    key in st.session_state.results.columns for key in required_keys
                )
            elif isinstance(st.session_state.results, dict):
                fuel_data_available = all(
                    key in st.session_state.results for key in required_keys
                )

            if fuel_data_available:
                # Get initial and final compositions
                if isinstance(st.session_state.results, pd.DataFrame):
                    initial_u235 = st.session_state.results["U235"].iloc[0]
                    initial_u238 = st.session_state.results["U238"].iloc[0]
                    final_u235 = st.session_state.results["U235"].iloc[-1]
                    final_u238 = st.session_state.results["U238"].iloc[-1]
                    final_pu239 = (
                        st.session_state.results["Pu239"].iloc[-1]
                        if "Pu239" in st.session_state.results.columns
                        else 0
                    )
                else:
                    initial_u235 = st.session_state.results["U235"][0]
                    initial_u238 = st.session_state.results["U238"][0]
                    final_u235 = st.session_state.results["U235"][-1]
                    final_u238 = st.session_state.results["U238"][-1]
                    final_pu239 = (
                        st.session_state.results["Pu239"][-1]
                        if "Pu239" in st.session_state.results
                        else 0
                    )

                initial_total = initial_u235 + initial_u238
                final_others = initial_total - final_u235 - final_u238 - final_pu239
                final_others = max(0, final_others)  # Ensure non-negative

                # Create pie charts for initial and final composition
                col1, col2 = st.columns(2)

                with col1:
                    labels = ["U-235", "U-238"]
                    values = [
                        initial_u235 / initial_total * 100,
                        initial_u238 / initial_total * 100,
                    ]

                    fig1 = go.Figure(
                        data=[
                            go.Pie(
                                labels=labels,
                                values=values,
                                hole=0.3,
                                title="Initial Composition (%)",
                            )
                        ]
                    )

                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    labels = ["U-235", "U-238", "Pu-239", "Others"]
                    values = [
                        final_u235 / initial_total * 100,
                        final_u238 / initial_total * 100,
                        final_pu239 / initial_total * 100,
                        final_others / initial_total * 100,
                    ]

                    fig2 = go.Figure(
                        data=[
                            go.Pie(
                                labels=labels,
                                values=values,
                                hole=0.3,
                                title="Final Composition (%)",
                            )
                        ]
                    )

                    st.plotly_chart(fig2, use_container_width=True)

                # Display fuel metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    if "burnup_avg" in st.session_state.results:
                        if isinstance(st.session_state.results, pd.DataFrame):
                            final_burnup = st.session_state.results["burnup_avg"].iloc[
                                -1
                            ]
                        else:
                            final_burnup = st.session_state.results["burnup_avg"][-1]
                        st.metric("Final Burnup", f"{final_burnup:.2f} GWd/tHM")
                    else:
                        st.metric("Final Burnup", "N/A")

                with col2:
                    u235_consumption = (initial_u235 - final_u235) / initial_u235 * 100
                    st.metric("U-235 Consumption", f"{u235_consumption:.2f}%")

                with col3:
                    if final_pu239 > 0:
                        pu_production = (
                            final_pu239 / (initial_u235 * 1000) * 1e6
                        )  # g Pu-239 per kg U-235
                        st.metric(
                            "Pu-239 Production", f"{pu_production:.2f} g/kg U-235"
                        )
                    else:
                        st.metric("Pu-239 Production", "N/A")

                # Plot isotope evolution over time
                st.subheader("Isotope Evolution Over Time")

                fig3 = go.Figure()

                # Function to safely get data and normalize it
                def get_normalized_data(key, normalize_to_initial=False):
                    if isinstance(st.session_state.results, pd.DataFrame):
                        if key in st.session_state.results.columns:
                            data = st.session_state.results[key]
                            if normalize_to_initial:
                                initial_value = data.iloc[0]
                                if initial_value != 0:
                                    return data / initial_value
                            else:
                                max_value = data.max()
                                if max_value != 0:
                                    return data / max_value
                    else:
                        if key in st.session_state.results:
                            data = st.session_state.results[key]
                            if normalize_to_initial:
                                initial_value = data[0]
                                if initial_value != 0:
                                    return np.array(data) / initial_value
                            else:
                                max_value = max(data)
                                if max_value != 0:
                                    return np.array(data) / max_value
                    return None

                # Get time data
                time_data = None
                if isinstance(st.session_state.results, pd.DataFrame):
                    if "time" in st.session_state.results.columns:
                        time_data = st.session_state.results["time"]
                else:
                    if "time" in st.session_state.results:
                        time_data = st.session_state.results["time"]

                if time_data is not None:
                    # Add U-235 trace
                    u235_data = get_normalized_data("U235", True)
                    if u235_data is not None:
                        fig3.add_trace(
                            go.Scatter(
                                x=time_data,
                                y=u235_data,
                                mode="lines",
                                name="U-235 (relative to initial)",
                            )
                        )

                    # Add Pu-239 trace if available
                    pu239_data = get_normalized_data("Pu239")
                    if pu239_data is not None:
                        fig3.add_trace(
                            go.Scatter(
                                x=time_data,
                                y=pu239_data,
                                mode="lines",
                                name="Pu-239 (relative to max)",
                            )
                        )

                    # Add Xe-135 and Sm-149 traces if available
                    for isotope in ["Xe135", "Sm149"]:
                        isotope_data = get_normalized_data(isotope)
                        if isotope_data is not None:
                            fig3.add_trace(
                                go.Scatter(
                                    x=time_data,
                                    y=isotope_data,
                                    mode="lines",
                                    name=f"{isotope} (relative to max)",
                                )
                            )

                    fig3.update_layout(
                        title="Relative Isotope Concentrations Over Time",
                        xaxis_title="Time (days)",
                        yaxis_title="Relative Concentration",
                        legend=dict(x=0, y=1, traceorder="normal"),
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Fuel composition data not available for display.")
        except Exception as e:
            st.error(f"Error generating fuel composition plots: {str(e)}")
            st.exception(e)

    with tab3:
        st.subheader("Reactor Physics Analysis")

        try:
            # Display neutronic parameters
            col1, col2, col3 = st.columns(3)

            with col1:
                if (
                    hasattr(st.session_state, "neutronics")
                    and st.session_state.neutronics is not None
                ):
                    if hasattr(st.session_state.neutronics, "L"):
                        diffusion_length = st.session_state.neutronics.L
                        st.metric("Diffusion Length", f"{diffusion_length:.2f} cm")
                    else:
                        st.metric("Diffusion Length", "N/A")
                else:
                    st.metric("Diffusion Length", "N/A")

            with col2:
                if (
                    hasattr(st.session_state, "core")
                    and st.session_state.core is not None
                ):
                    if hasattr(st.session_state.core, "B_g_squared"):
                        geometric_buckling = st.session_state.core.B_g_squared
                        st.metric(
                            "Geometric Buckling", f"{geometric_buckling:.6f} cm⁻²"
                        )
                    else:
                        st.metric("Geometric Buckling", "N/A")
                else:
                    st.metric("Geometric Buckling", "N/A")

            with col3:
                if (
                    hasattr(st.session_state, "neutronics")
                    and st.session_state.neutronics is not None
                    and hasattr(
                        st.session_state.neutronics, "calculate_material_buckling"
                    )
                ):
                    try:
                        material_buckling = (
                            st.session_state.neutronics.calculate_material_buckling()
                        )
                        st.metric("Material Buckling", f"{material_buckling:.6f} cm⁻²")
                    except Exception:
                        st.metric("Material Buckling", "Error")
                else:
                    st.metric("Material Buckling", "N/A")

            # Display flux distribution if data available
            st.subheader("Neutron Flux Distribution")

            if (
                hasattr(st.session_state, "core")
                and st.session_state.core is not None
                and hasattr(st.session_state.core, "radius")
                and hasattr(st.session_state.core, "height")
                and hasattr(st.session_state.core, "gamma_1")
                and hasattr(st.session_state.core, "mu_1")
                and hasattr(st.session_state, "phi_0")
                and st.session_state.phi_0 is not None
            ):

                # Create 2D flux contour plot
                try:
                    from scipy.special import j0

                    r = np.linspace(0, st.session_state.core.radius, 100)
                    z = np.linspace(0, st.session_state.core.height, 100)
                    R, Z = np.meshgrid(r, z)

                    flux = np.zeros_like(R)
                    for i in range(R.shape[0]):
                        for j in range(R.shape[1]):
                            flux[i, j] = (
                                st.session_state.phi_0
                                * j0(st.session_state.core.gamma_1 * R[i, j])
                                * np.sin(st.session_state.core.mu_1 * Z[i, j])
                            )

                    fig, ax = plt.subplots(figsize=(10, 6))
                    contour = ax.contourf(R, Z, flux, 50, cmap="viridis")
                    ax.set_xlabel("Radius (cm)")
                    ax.set_ylabel("Height (cm)")
                    ax.set_title("Neutron Flux Distribution (r-z plane)")
                    plt.colorbar(contour, ax=ax, label="Flux (n/cm²/s)")
                    plt.tight_layout()

                    st.pyplot(fig)

                    # Display flux profiles along radial and axial directions
                    col1, col2 = st.columns(2)

                    with col1:
                        # Radial flux profile at mid-height
                        fig, ax = plt.subplots(figsize=(8, 4))
                        z_midpoint = st.session_state.core.height / 2
                        radial_flux = (
                            st.session_state.phi_0
                            * j0(st.session_state.core.gamma_1 * r)
                            * np.sin(st.session_state.core.mu_1 * z_midpoint)
                        )
                        ax.plot(r, radial_flux, "b-")
                        ax.set_xlabel("Radius (cm)")
                        ax.set_ylabel("Flux (n/cm²/s)")
                        ax.set_title("Radial Flux Profile at Mid-Height")
                        ax.grid(True)
                        st.pyplot(fig)

                    with col2:
                        # Axial flux profile at center
                        fig, ax = plt.subplots(figsize=(8, 4))
                        axial_flux = (
                            st.session_state.phi_0
                            * j0(0)
                            * np.sin(st.session_state.core.mu_1 * z)
                        )
                        ax.plot(z, axial_flux, "r-")
                        ax.set_xlabel("Height (cm)")
                        ax.set_ylabel("Flux (n/cm²/s)")
                        ax.set_title("Axial Flux Profile at Center")
                        ax.grid(True)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating flux distribution plots: {str(e)}")
            else:
                st.info("Flux distribution data not available.")
        except Exception as e:
            st.error(f"Error in reactor physics analysis: {str(e)}")
            st.exception(e)

    with tab4:
        st.subheader("Thermal-Hydraulic Analysis")

        try:
            if (
                hasattr(st.session_state, "thermal")
                and st.session_state.thermal is not None
            ):
                # Display thermal-hydraulic parameters
                col1, col2, col3 = st.columns(3)

                with col1:
                    if hasattr(st.session_state.thermal, "power_density"):
                        st.metric(
                            "Power Density",
                            f"{st.session_state.thermal.power_density:.2f} W/cm³",
                        )
                    else:
                        st.metric("Power Density", "N/A")

                with col2:
                    if hasattr(st.session_state.thermal, "mass_flow_rate"):
                        st.metric(
                            "Helium Mass Flow Rate",
                            f"{st.session_state.thermal.mass_flow_rate:.2f} kg/s",
                        )
                    else:
                        st.metric("Helium Mass Flow Rate", "N/A")

                with col3:
                    if hasattr(st.session_state.thermal, "coolant_pressure"):
                        st.metric(
                            "Coolant Pressure",
                            f"{st.session_state.thermal.coolant_pressure:.1f} MPa",
                        )
                    else:
                        st.metric("Coolant Pressure", "N/A")

                # Display temperature distribution if method available
                if hasattr(
                    st.session_state.thermal, "calculate_temperature_distribution"
                ):
                    st.subheader("Temperature Distribution")

                    try:
                        # Create simplified power distribution function
                        def power_distribution(z_norm):
                            return 1.5 * np.sin(np.pi * z_norm)

                        # Calculate temperature distribution
                        z_points, temperatures = (
                            st.session_state.thermal.calculate_temperature_distribution(
                                power_distribution
                            )
                        )

                        # Plot temperature distribution
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(z_points, temperatures, "r-")
                        ax.set_xlabel("Axial Position (cm)")
                        ax.set_ylabel("Coolant Temperature (°C)")
                        ax.set_title("Axial Coolant Temperature Distribution")
                        ax.grid(True)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(
                            f"Error calculating temperature distribution: {str(e)}"
                        )

                # Display fuel pebble temperature if method available
                if hasattr(st.session_state.thermal, "calculate_pebble_temperature"):
                    st.subheader("Fuel Pebble Temperature Analysis")

                    try:
                        # Create a set of local power densities
                        power_densities = (
                            np.linspace(0.5, 2.0, 5)
                            * st.session_state.thermal.power_density
                        )
                        coolant_temps = np.linspace(
                            st.session_state.thermal.inlet_temp,
                            st.session_state.thermal.outlet_temp,
                            5,
                        )

                        # Calculate pebble temperatures at various conditions
                        results = []
                        for pd in power_densities:
                            for ct in coolant_temps:
                                temp_data = st.session_state.thermal.calculate_pebble_temperature(
                                    pd, ct
                                )
                                results.append(
                                    {
                                        "Power Density": pd,
                                        "Coolant Temp": ct,
                                        "Surface Temp": temp_data[
                                            "surface_temperature"
                                        ],
                                        "Center Temp": temp_data["center_temperature"],
                                    }
                                )

                        # Create a DataFrame for plotting
                        temp_df = pd.DataFrame(results)

                        # Create 3D surface plot for center temperature
                        fig = go.Figure(
                            data=[
                                go.Surface(
                                    x=temp_df["Power Density"].values.reshape(5, 5),
                                    y=temp_df["Coolant Temp"].values.reshape(5, 5),
                                    z=temp_df["Center Temp"].values.reshape(5, 5),
                                    colorscale="Inferno",
                                )
                            ]
                        )

                        fig.update_layout(
                            title="Fuel Pebble Center Temperature (°C)",
                            scene=dict(
                                xaxis_title="Power Density (W/cm³)",
                                yaxis_title="Coolant Temperature (°C)",
                                zaxis_title="Center Temperature (°C)",
                            ),
                            autosize=False,
                            width=700,
                            height=500,
                            margin=dict(l=65, r=50, b=65, t=90),
                        )

                        st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Error calculating pebble temperatures: {str(e)}")

                # Perform safety analysis if method available
                if hasattr(st.session_state.thermal, "safety_analysis"):
                    st.subheader("Safety Analysis")

                    try:
                        safety_results = st.session_state.thermal.safety_analysis()

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if "peak_fuel_temperature" in safety_results:
                                st.metric(
                                    "Peak Fuel Temperature",
                                    f"{safety_results['peak_fuel_temperature']:.1f} °C",
                                )
                            else:
                                st.metric("Peak Fuel Temperature", "N/A")

                        with col2:
                            if "temperature_margin" in safety_results:
                                st.metric(
                                    "Temperature Margin",
                                    f"{safety_results['temperature_margin']:.1f} °C",
                                )
                            else:
                                st.metric("Temperature Margin", "N/A")

                        with col3:
                            if "is_safe" in safety_results:
                                safety_status = (
                                    "✅ Safe"
                                    if safety_results["is_safe"]
                                    else "❌ Unsafe"
                                )
                                st.metric("Safety Status", safety_status)
                            else:
                                st.metric("Safety Status", "N/A")

                        # Display detailed temperature data
                        if "detailed_temps" in safety_results:
                            with st.expander("Detailed Temperature Data"):
                                st.json(safety_results["detailed_temps"])
                    except Exception as e:
                        st.error(f"Error performing safety analysis: {str(e)}")
            else:
                st.info("Thermal-hydraulic data not available.")
        except Exception as e:
            st.error(f"Error in thermal-hydraulic analysis: {str(e)}")
            st.exception(e)
else:
    # Display welcome screen if no simulation has been run
    st.info("Click 'Run Simulation' button in the sidebar to start the simulation.")

    # Display application description
    st.markdown(
        """
    ## XE-100 SMR Simulator
    
    This application simulates the operation of an XE-100 Small Modular Reactor (SMR) that uses TRISO fuel in a pebble bed configuration.
    
    ### Features:
    
    - Simulate reactor operation over time periods from days to years
    - Analyze neutron flux distribution in the core
    - Track fuel burnup and isotopic composition changes
    - Model the continuous refueling process
    - Evaluate thermal-hydraulic performance and safety margins
    
    ### How to use:
    
    1. Adjust the reactor parameters in the sidebar
    2. Set the simulation time and time step
    3. Click "Run Simulation" button
    4. Explore the results in the different tabs
    
    ### Reactor Physics Model:
    
    The simulator uses:
    - One-group diffusion model with Bessel function and sinusoidal solutions
    - Burnup tracking for key isotopes including U-235, U-238, Pu-239
    - Xenon and Samarium poisoning effects
    - Continuous pebble refueling simulation
    - Thermal-hydraulic analysis with temperature calculations
    """
    )

    # Display sample images
    col1, col2 = st.columns(2)

    with col1:
        st.image(
            "https://images.squarespace-cdn.com/content/v1/5e13e7e8e3a0d42924a7e3e9/1586877919872-KH5V1P9O3KV0ANYPZRB8/graphic-reactor-xe100-cutout.jpg?format=1000w",
            caption="XE-100 SMR Concept (Example)",
            use_column_width=True,
        )

    with col2:
        st.image(
            "https://images.squarespace-cdn.com/content/5e13e7e8e3a0d42924a7e3e9/1587073435468-RAE6ZE1ME423UT8OA1L4/graphic-triso-x-pebble.jpg?content-type=image%2Fjpeg",
            caption="TRISO Fuel Particles (Example)",
            use_column_width=True,
        )

# Footer
st.markdown("---")
st.markdown(
    "XE-100 SMR Simulator - Nuclear Reactor Dynamics Simulation Tool | Last updated: "
    + datetime.now().strftime("%Y-%m-%d")
)
