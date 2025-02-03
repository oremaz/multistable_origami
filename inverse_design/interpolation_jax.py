# This file contains functions to interpolate data from a dataset.
# Copyright (C) 2025 Orel Mazor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from functools import lru_cache

# Local modules
from ...data_processing.load_df_script import load_df
from ...data_processing.process_df_script import preprocess_data_for_accelernum




## Simulation Data Preparation
class DataPreparation:
    def __init__(self, dL_values=None, angles=None):
        if dL_values is None:
            dL_values = np.round(np.arange(0.3, 1, 0.01), decimals=2)
        if angles is None:
            angles = np.arange(-180, 180, 0.5)
        self.dL_values = dL_values
        self.angles = angles
        self.torques_list = []
        self.dataframes = []

    def lists(self):
        """
        Loads data for each d/L, preprocesses it, and interpolates
        over a shared angle grid.
        """
        for dL in self.dL_values:
            df = load_df(dL)  # Existing implementation[1]
            df = preprocess_data_for_accelernum(df)  # Existing implementation[1]
            df = df[["angle", "torque"]]
            self.dataframes.append(df)

        for df in self.dataframes:
            interp_func = interp1d(
                df["angle"], df["torque"], kind="linear", fill_value="extrapolate"
            )
            self.torques_list.append(interp_func(self.angles))

        return self.dataframes, self.torques_list


## Experimental Data Preparation
class ExperimentalData:
    def __init__(self):
        pass

    def read_experiment_spec(self, folder, i=-1, slicing_angle=10, indices=[132, 402]):
        """
        Reads and concatenates data from multiple Excel sheets, returning
        the sliced angle and torque arrays[1].
        """
        Time, Torque, Angle = [], [], []
        columns_to_extract = [0, 3, 4]
        sheet_indices = [2, 3, 4]

        for filename in os.listdir(folder):
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                file_path = os.path.join(folder, filename)
                time_combined, angle_combined, torque_combined = [], [], []
                try:
                    excel_data = pd.ExcelFile(file_path)
                    last_angle = None
                    for sheet_index in sheet_indices:
                        if sheet_index < len(excel_data.sheet_names):
                            df = pd.read_excel(
                                file_path,
                                sheet_name=excel_data.sheet_names[sheet_index],
                                usecols=columns_to_extract,
                            )
                            t = df.iloc[3:, 0].to_numpy()
                            angle = df.iloc[3:, 1].to_numpy()
                            torque = df.iloc[3:, 2].to_numpy()

                            if last_angle is not None:
                                angle += last_angle[-1]

                            time_combined.extend(t)
                            angle_combined.extend(angle)
                            torque_combined.extend(torque)
                            last_angle = angle
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

                time_combined = np.array(time_combined)
                angle_combined = -np.degrees(np.array(angle_combined)) - slicing_angle
                torque_combined = -np.array(torque_combined) * 0.001
                Time.append(time_combined)
                Angle.append(angle_combined)
                Torque.append(torque_combined)

        i0, i1 = indices
        return Angle[i][i0:i1], Torque[i][i0:i1]


class ExperimentalDataPreparation:
    def __init__(self, path="../../Experiments/tests"):
        self.path = path
        self.torques_list = []
        self.base_folders = []

        # Populate base folders during initialization
        for root, dirs, files in os.walk(self.path):
            if "base" in dirs:
                base_folder_path = os.path.join(root, "base")
                self.base_folders.append(base_folder_path)

        self.Edata = ExperimentalData()

    def lists(self):
        """
        Loads data for each d/L, preprocesses it, and interpolates
        over a shared angle grid.
        """

        for folder in self.base_folders:
            Angle, Torque = self.Edata.read_experiment_spec(folder)
            Torque_interpolator = interp1d(
                Angle, Torque, kind="linear", fill_value="extrapolate"
            )
            Angle1 = np.arange(-145, 125.5, 0.5)
            Torque_interp = Torque_interpolator(Angle1)
            self.torques_list.append(Torque_interp)

        return self.torques_list


## Interpolation
class Interpolation2D:
    def __init__(self, method="regulargrid"):
        self.method = method
        self.interpolator = None

    def fine_grid_interpolation_callable(self, dL_values, angles, torque_data):
        """
        Creates a 2D interpolator with the chosen method[1].
        """
        if self.method == "regulargrid":
            interpolator = jax.scipy.interpolate.RegularGridInterpolator(
                (dL_values, angles),
                torque_data,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )

            def interpolator_callable(points):
                return interpolator(points)

            return interpolator_callable

    def build_interpolator(self, dL_values, angles, torque_data):
        """
        Stores the interpolator as a class attribute.
        """
        self.interpolator = self.fine_grid_interpolation_callable(
            dL_values, angles, torque_data
        )

    def interpolate(self, theta, dL):
        """
        Interpolates the provided angle (theta) and d/L.
        """
        if self.interpolator is None:
            raise ValueError("Interpolator has not been created yet.")
        points_to_interpolate = jnp.column_stack((dL, theta))
        return self.interpolator(points_to_interpolate)


## Plotting
class Plotter:
    def __init__(self):
        pass

    def plot_interpolated_data(self, angles, torques_list, dL_values):
        """
        Plots torque vs angle for all d/L values on a single figure[1].
        """
        colors = plt.cm.viridis(np.linspace(0, 1, len(dL_values)))
        for i, torque_interp in enumerate(torques_list):
            plt.plot(angles, torque_interp, color=colors[i])

        sm = plt.cm.ScalarMappable(
            cmap="viridis", norm=plt.Normalize(vmin=min(dL_values), vmax=max(dL_values))
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label("d/L")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Torque (mN.m)")
        plt.grid(True)
        plt.show()


def interpolator_simple(method="regulargrid"):
    """
    Base data interpolation using DataPreparation and Interpolation2D.
    """
    dp = DataPreparation()
    _, torques_list = dp.lists()
    angles = dp.angles
    dL_values = dp.dL_values
    torque_data = jnp.array(torques_list)
    torque_data = jnp.vstack(torque_data)
    i2d = Interpolation2D(method=method)
    i2d.build_interpolator(dL_values, angles, torque_data)

    def interpolate_points(points):
        return i2d.interpolator(points)

    return interpolate_points

@lru_cache(maxsize=None)
def torques_lists_exp():
    edp_exp = ExperimentalDataPreparation()
    torques_list_exp = edp_exp.lists()
    return torques_list_exp

def interpolator_simple_exp(method="regulargrid"):
    """
    Experimental data interpolation using ExperimentalData and Interpolation2D.
    This example uses dummy data for demonstration. Adapt as needed.
    """
    torques_list_exp = torques_lists_exp()
    angles = np.arange(-145, 125.5, 0.5)
    dL_values = jnp.array(
        [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    torque_data = jnp.array(torques_list_exp)
    torque_data = jnp.vstack(torque_data)
    i2d = Interpolation2D(method=method)
    i2d.build_interpolator(dL_values, angles, torque_data)

    def interpolate_points(points):
        return i2d.interpolator(points)

    return interpolate_points

def interpolator2D(theta, dL, method="regulargrid", source: str = None):
    """
    2D interpolation for the provided angle(s) (theta) and d/L value(s).
    source can be:
      - None/'sim': Simulation data
      - 'exp': Experimental data
    """
    if source is None or source == "sim":
        interpolator = interpolator_simple(method=method)
    elif source == "exp":
        interpolator = interpolator_simple_exp(method=method)
    else:
        raise ValueError("Invalid source. Use None or 'exp'.")

    theta = jnp.asarray(theta)
    dL = jnp.asarray(dL)
    points_to_interpolate = jnp.column_stack((dL, theta))
    interpolated_values = interpolator(points_to_interpolate)

    return (
        interpolated_values
        if jnp.size(interpolated_values) > 1
        else interpolated_values[0].astype(float)
    )
