# utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import j0


def plot_reactor_parameters(results):
    """Plot key reactor parameters over time"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot k_eff
    if "k_eff" in results and len(results["k_eff"]) > 0:
        ax1.plot(results["time"], results["k_eff"], "b-", label="k_eff")
        ax1.axhline(y=1.0, color="r", linestyle="--", label="Critical (k=1)")
        ax1.set_ylabel("k_eff")
        ax1.legend()
        ax1.grid(True)

    # Plot burnup
    if "burnup_avg" in results and len(results["burnup_avg"]) > 0:
        ax2.plot(results["time"], results["burnup_avg"], "g-", label="Average Burnup")
        ax2.set_ylabel("Burnup (GWd/tHM)")
        ax2.legend()
        ax2.grid(True)

    # Plot nuclide evolution
    isotopes = ["U235", "Pu239", "Xe135", "Sm149"]
    for isotope in isotopes:
        if isotope in results and len(results[isotope]) > 0:
            if isotope == "U235":
                # Normalize U-235 to its initial value
                ax3.plot(
                    results["time"],
                    results[isotope] / results[isotope][0],
                    "r-",
                    label=f"{isotope} (rel. to initial)",
                )
            else:
                # Normalize others to their maximum value (if non-zero)
                max_val = max(results[isotope]) if max(results[isotope]) > 0 else 1
                ax3.plot(
                    results["time"],
                    results[isotope] / max_val,
                    label=f"{isotope} (rel. to max)",
                )

    ax3.set_xlabel("Time (days)")
    ax3.set_ylabel("Relative Concentration")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    return fig


def plot_interactive_reactor(results):
    """Create interactive plots with Plotly"""
    # Check if results is a DataFrame or a dict
    is_dataframe = hasattr(results, "columns")

    # Create subplot figure
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Reactivity Parameters",
            "Burnup Evolution",
            "Nuclide Concentrations",
        ),
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    # Function to safely get data
    def get_data(key):
        if is_dataframe:
            return results[key] if key in results.columns else None
        else:
            return results[key] if key in results else None

    # Get time data
    time_data = get_data("time")
    if time_data is None:
        # If no time data, create placeholder
        time_data = list(range(len(next(iter(results.values())))))

    # Plot k_eff
    k_eff_data = get_data("k_eff")
    if k_eff_data is not None:
        fig.add_trace(go.Scatter(x=time_data, y=k_eff_data, name="k_eff"), row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=[1.0] * len(time_data),
                name="Critical (k=1)",
                line=dict(dash="dash", color="red"),
            ),
            row=1,
            col=1,
        )

    # Plot burnup
    burnup_data = get_data("burnup_avg")
    if burnup_data is not None:
        fig.add_trace(
            go.Scatter(x=time_data, y=burnup_data, name="Average Burnup"), row=2, col=1
        )

    # Plot nuclide evolution
    isotopes = ["U235", "Pu239", "Xe135", "Sm149"]
    for isotope in isotopes:
        isotope_data = get_data(isotope)
        if isotope_data is not None:
            if isotope == "U235":
                # Normalize U-235 to its initial value
                initial_value = isotope_data[0]
                if initial_value != 0:
                    fig.add_trace(
                        go.Scatter(
                            x=time_data,
                            y=isotope_data / initial_value,
                            name=f"{isotope} (rel. to initial)",
                        ),
                        row=3,
                        col=1,
                    )
            else:
                # Normalize others to their maximum value (if non-zero)
                max_val = max(isotope_data) if max(isotope_data) > 0 else 1
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=isotope_data / max_val,
                        name=f"{isotope} (rel. to max)",
                    ),
                    row=3,
                    col=1,
                )

    # Update layout
    fig.update_layout(
        height=900,
        width=800,
        title_text="XE-100 SMR Reactor Simulation",
        showlegend=True,
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="k_eff", row=1, col=1)
    fig.update_yaxes(title_text="Burnup (GWd/tHM)", row=2, col=1)
    fig.update_yaxes(title_text="Relative Concentration", row=3, col=1)

    # Update x-axis label
    fig.update_xaxes(title_text="Time (days)", row=3, col=1)

    return fig


def plot_3d_flux_distribution(core, phi_0):
    """Create 3D visualization of neutron flux in the core"""
    import plotly.graph_objects as go

    # Create a structured grid for the cylindrical core
    r = np.linspace(0, core.radius, 20)
    theta = np.linspace(0, 2 * np.pi, 20)
    z = np.linspace(0, core.height, 20)

    r_grid, theta_grid, z_grid = np.meshgrid(r, theta, z)

    # Convert to Cartesian coordinates
    x = r_grid * np.cos(theta_grid)
    y = r_grid * np.sin(theta_grid)

    # Calculate flux at each point
    flux = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                r_val = np.sqrt(x[i, j, k] ** 2 + y[i, j, k] ** 2)
                flux[i, j, k] = (
                    phi_0 * j0(core.gamma_1 * r_val) * np.sin(core.mu_1 * z[i, j, k])
                )

    # Create 3D scatter plot with color based on flux
    points = go.Scatter3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        mode="markers",
        marker=dict(size=3, color=flux.flatten(), colorscale="Viridis", opacity=0.7),
    )

    # Create cylinder outline
    theta_outline = np.linspace(0, 2 * np.pi, 100)
    x_outline = core.radius * np.cos(theta_outline)
    y_outline = core.radius * np.sin(theta_outline)

    # Bottom circle
    bottom_outline = go.Scatter3d(
        x=x_outline,
        y=y_outline,
        z=np.zeros_like(theta_outline),
        mode="lines",
        line=dict(color="black", width=3),
        showlegend=False,
    )

    # Top circle
    top_outline = go.Scatter3d(
        x=x_outline,
        y=y_outline,
        z=np.ones_like(theta_outline) * core.height,
        mode="lines",
        line=dict(color="black", width=3),
        showlegend=False,
    )

    # Vertical lines
    vertical_lines = []
    for i in range(0, len(theta_outline), 10):
        vertical_lines.append(
            go.Scatter3d(
                x=[x_outline[i], x_outline[i]],
                y=[y_outline[i], y_outline[i]],
                z=[0, core.height],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
        )

    # Create figure
    fig = go.Figure(data=[points, bottom_outline, top_outline] + vertical_lines)

    # Update layout
    fig.update_layout(
        title="3D Neutron Flux Distribution",
        scene=dict(
            xaxis_title="X (cm)",
            yaxis_title="Y (cm)",
            zaxis_title="Z (cm)",
            aspectmode="data",
        ),
        width=800,
        height=800,
    )

    return fig
