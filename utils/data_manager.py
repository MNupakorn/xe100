# utils/data_manager.py
import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime


class DataManager:
    """Class for managing simulation data, saving, and loading"""

    def __init__(self, data_dir="simulation_data"):
        """
        Initialize DataManager

        Parameters:
        -----------
        data_dir : str
            Directory to store simulation data
        """
        self.data_dir = data_dir

        # Create directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def save_simulation_results(self, results, name=None):
        """
        Save simulation results to disk

        Parameters:
        -----------
        results : dict or pandas.DataFrame
            Simulation results to save

        name : str, optional
            Name for the saved data file. If None, a timestamp will be used

        Returns:
        --------
        str
            Path to the saved file
        """
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"simulation_{timestamp}"

        # Determine file extension based on data type
        if isinstance(results, pd.DataFrame):
            ext = "csv"
            file_path = os.path.join(self.data_dir, f"{name}.{ext}")
            results.to_csv(file_path, index=False)
        elif isinstance(results, dict):
            # Check if all values are simple types that can be saved as JSON
            simple_types = (int, float, str, bool, type(None))

            # Check if values are arrays that can be converted to lists
            convertible = True
            for v in results.values():
                if isinstance(v, np.ndarray):
                    results[v] = v.tolist()
                elif not (
                    isinstance(v, simple_types)
                    or (
                        isinstance(v, (list, tuple))
                        and all(isinstance(i, simple_types) for i in v)
                    )
                    or (
                        isinstance(v, dict)
                        and all(
                            isinstance(k, simple_types) and isinstance(vv, simple_types)
                            for k, vv in v.items()
                        )
                    )
                ):
                    convertible = False
                    break

            if convertible:
                ext = "json"
                file_path = os.path.join(self.data_dir, f"{name}.{ext}")
                with open(file_path, "w") as f:
                    json.dump(results, f, indent=2)
            else:
                ext = "pkl"
                file_path = os.path.join(self.data_dir, f"{name}.{ext}")
                with open(file_path, "wb") as f:
                    pickle.dump(results, f)
        else:
            # Unknown type, use pickle
            ext = "pkl"
            file_path = os.path.join(self.data_dir, f"{name}.{ext}")
            with open(file_path, "wb") as f:
                pickle.dump(results, f)

        return file_path

    def load_simulation_results(self, file_path):
        """
        Load simulation results from disk

        Parameters:
        -----------
        file_path : str
            Path to the saved data file

        Returns:
        --------
        dict or pandas.DataFrame
            Loaded simulation results
        """
        # Check file extension
        _, ext = os.path.splitext(file_path)

        if ext.lower() == ".csv":
            # Load CSV data
            return pd.read_csv(file_path)
        elif ext.lower() == ".json":
            # Load JSON data
            with open(file_path, "r") as f:
                return json.load(f)
        elif ext.lower() == ".pkl":
            # Load pickle data
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def list_available_simulations(self):
        """
        List all available saved simulations

        Returns:
        --------
        list
            List of saved simulation file paths
        """
        files = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".csv") or file.endswith(".json") or file.endswith(".pkl"):
                files.append(os.path.join(self.data_dir, file))

        return sorted(files)

    def export_results_for_visualization(self, results, file_format="csv"):
        """
        Export simulation results in a format suitable for external visualization

        Parameters:
        -----------
        results : dict or pandas.DataFrame
            Simulation results to export

        file_format : str
            Format to export the data ('csv', 'json', 'excel')

        Returns:
        --------
        str
            Path to the exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"visualization_export_{timestamp}"

        # Convert results to DataFrame if necessary
        if isinstance(results, dict):
            # Create DataFrame from dict
            df = pd.DataFrame()

            # Add time column if available
            if "time" in results:
                df["time"] = results["time"]

            # Add other columns
            for key, value in results.items():
                if (
                    key != "time"
                    and isinstance(value, (list, np.ndarray))
                    and len(value) == len(df)
                    or len(df) == 0
                ):
                    df[key] = value
        else:
            df = results.copy()

        # Export in the specified format
        if file_format.lower() == "csv":
            file_path = os.path.join(self.data_dir, f"{name}.csv")
            df.to_csv(file_path, index=False)
        elif file_format.lower() == "json":
            file_path = os.path.join(self.data_dir, f"{name}.json")
            df.to_json(file_path, orient="records", indent=2)
        elif file_format.lower() == "excel":
            file_path = os.path.join(self.data_dir, f"{name}.xlsx")
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        return file_path

    def compare_simulations(self, file_paths, parameters=None):
        """
        Compare multiple simulations

        Parameters:
        -----------
        file_paths : list
            List of file paths to compare

        parameters : list, optional
            List of parameters to compare. If None, all common parameters will be compared

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the comparison results
        """
        # Load all simulations
        simulations = []
        for file_path in file_paths:
            sim_data = self.load_simulation_results(file_path)
            sim_name = os.path.basename(file_path).split(".")[0]

            if isinstance(sim_data, pd.DataFrame):
                # For DataFrames, extract final state
                final_state = sim_data.iloc[-1].to_dict()
                final_state["name"] = sim_name
                simulations.append(final_state)
            elif isinstance(sim_data, dict):
                # For dictionaries, extract scalar values or final values of arrays
                flat_dict = {"name": sim_name}
                for key, value in sim_data.items():
                    if isinstance(value, (int, float, str, bool, type(None))):
                        flat_dict[key] = value
                    elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        flat_dict[f"{key}_final"] = value[-1]
                        flat_dict[f"{key}_initial"] = value[0]
                        flat_dict[f"{key}_mean"] = np.mean(value)
                        flat_dict[f"{key}_min"] = np.min(value)
                        flat_dict[f"{key}_max"] = np.max(value)

                simulations.append(flat_dict)

        # Create comparison DataFrame
        if not simulations:
            return pd.DataFrame()

        # Find common parameters if not specified
        if parameters is None:
            parameters = set(simulations[0].keys())
            for sim in simulations[1:]:
                parameters &= set(sim.keys())
            parameters = list(parameters)

        # Extract values for comparison
        comparison = {}
        for param in parameters:
            comparison[param] = [sim.get(param, None) for sim in simulations]

        return pd.DataFrame(comparison)
