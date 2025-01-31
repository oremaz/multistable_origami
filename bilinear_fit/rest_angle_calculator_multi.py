# This file contains functions to calculate the rest angles of a multistable origami.
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


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt

from .bilinear_fits import bilinear_model_pca, find_zeros_of_bilinear
from .interpolated_data_script import interpolated_fit
from ...data_processing.load_df_script import load_df
from ...data_processing.process_df_script import preprocess_data_for_accelernum


def mirror(f: callable) -> callable:
    return lambda x: -f(-x)


def interpolate_df(df: pd.DataFrame) -> callable:
    interp_func = interp1d(
        df["angle"], df["torque"], kind="linear", fill_value="extrapolate"
    )
    return interp_func


# plot df and interpolated function
def plot_df_and_interpolated(df: pd.DataFrame, mirrored=False):
    interp_func = interpolate_df(df)
    if mirrored:
        interp_func = mirror(interp_func)
        df = mirror_df(df)
    plt.plot(df["angle"], df["torque"], label="Data")
    x = np.linspace(df["angle"].min(), df["angle"].max(), 1000)
    plt.plot(x, interp_func(x), label="Interpolated function")
    plt.xlabel("Angle")
    plt.ylabel("Torque")
    plt.legend()
    plt.show()


def pred_fit(
    dL: float,
    offset: float,
    width: float,
    thickness: float,
    mirrored=False,
    method="PCA",
) -> callable:
    """Helper function to return either the bilinear or mirrored bilinear function."""
    df = load_df(dL, offset, width, thickness)
    bilinear_func = bilinear_model_pca(df, return_params=False, method=method)
    return mirror(bilinear_func) if mirrored else bilinear_func


def pred_sim(
    dL: float,
    offset: float,
    width: float,
    thickness: float,
    mirrored=False,
    process=False,
) -> callable:
    df = load_df(dL, offset, width, thickness)
    if process:
        df = preprocess_data_for_accelernum(df)
    func = interpolate_df(df)
    return mirror(func) if mirrored else func


def pred_interpolated(
    dL: float,
    offset: float,
    width: float,
    thickness: float,
    mirrored=False,
    method="PCA",
) -> callable:
    func = interpolated_fit(dL, offset, width, thickness, method=method)
    return mirror(func) if mirrored else func


def generate_symmetric_configs(N):
    # Génère toutes les combinaisons de miroir et non-miroir
    base_configs = list(product([False, True], repeat=N))

    # Filtre les configurations pour ne garder que la moitié de manière symétrique
    symmetric_configs = []
    for config in base_configs:
        # Crée la configuration miroir en inversant les valeurs
        mirrored_config = tuple(not x for x in config)
        # Ajoute la configuration si ni elle ni son miroir n'est déjà dans la liste
        if config not in symmetric_configs and mirrored_config not in symmetric_configs:
            symmetric_configs.append(config)

    return symmetric_configs


def mirror_df(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the mirrored DataFrame."""
    df = df[["angle", "torque"]].copy()
    df["angle"] = -df["angle"]
    df["torque"] = -df["torque"]
    df["stiffnessnum"] = np.gradient(df["torque"], df["angle"])
    df["accelernum"] = np.gradient(df["stiffnessnum"], df["angle"])
    return df


def sum_multilinear(
    *parameters: Tuple[
        Union[float, int], Union[float, int], Union[float, int], Union[float, int]
    ],
    pred_func: callable,
    config: Optional[Tuple[bool]] = None,
    method="PCA",
    process=False,
):
    """Generates 2^(N-1) unique configurations of multilinear sum."""
    N = len(parameters)
    configs = generate_symmetric_configs(N)

    def create_function(config):
        """Returns a function representing a specific mirrored/non-mirrored configuration."""
        if "method" in pred_func.__code__.co_varnames:
            preds = [
                pred_func(*parameter, mirrored, method=method)
                for parameter, mirrored in zip(parameters, config)
            ]
        elif "process" in pred_func.__code__.co_varnames:
            preds = [
                pred_func(*parameter, mirrored, process=process)
                for parameter, mirrored in zip(parameters, config)
            ]
        else:
            preds = [
                pred_func(*parameter, mirrored)
                for parameter, mirrored in zip(parameters, config)
            ]
        return lambda x: sum(pred(x) for pred in preds)

    if config is not None:
        return create_function(config)
    return [create_function(config) for config in configs]


def sum_funcs(*funcs: callable) -> callable:
    """Returns the sum of all functions."""
    return lambda x: sum(func(x) for func in funcs)


def sum_inverse_multilinear(
    *parameters: Tuple[
        Union[float, int], Union[float, int], Union[float, int], Union[float, int]
    ],
    pred_func: callable,
    config: Optional[Tuple[bool]] = None,
    method: str = "PCA",
    process: bool = False,
):
    """Generates 2^(N-1) unique configurations of the inverse of the sum of the inverses of functions."""

    N = len(parameters)
    configs = generate_symmetric_configs(
        N
    )  # Assuming this function generates the mirrored configurations

    def create_function(config):
        """Returns a function representing a specific mirrored/non-mirrored configuration."""
        if "method" in pred_func.__code__.co_varnames:
            preds = [
                pred_func(*parameter, mirrored, method=method)
                for parameter, mirrored in zip(parameters, config)
            ]
        elif "process" in pred_func.__code__.co_varnames:
            preds = [
                pred_func(*parameter, mirrored, process=process)
                for parameter, mirrored in zip(parameters, config)
            ]
        else:
            preds = [
                pred_func(*parameter, mirrored)
                for parameter, mirrored in zip(parameters, config)
            ]

        # Inverse of the sum of inverses: 1 / (sum(1 / pred(x)) for all predictions)
        return lambda x: 1 / sum(
            1 / pred(x) for pred in preds if pred(x) != 0
        )  # Avoid division by zero

    if config is not None:
        return create_function(config)

    return [create_function(config) for config in configs]


def process_start_value(config_index, config_func):
    """Process a single configuration function and return the rest angle result."""
    rest_angle = find_zeros_of_bilinear(bilinear_func=config_func)
    rest_angle_key = f"rest_angle_config_{config_index}"
    return {rest_angle_key: rest_angle}


def calculate_rest_angles_model_all(list_of_parameters, pred_func):
    """Parallelizes the computation of rest angles for all configurations."""
    # Generate all unique sum configurations
    sum_functions = sum_multilinear(*list_of_parameters, pred_func=pred_func)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_start_value, i, func)
            for i, func in enumerate(sum_functions)
        ]

        # Collect results
        results = {}
        for future in as_completed(futures):
            try:
                result = future.result()  # Get the result from each future
                results.update(result)  # Update the combined result dictionary
            except Exception as e:
                print(f"Error occurred: {e}")

    # Convert the combined results into a DataFrame
    return pd.DataFrame([results])


if __name__ == "__main__":
    # Test code here
    pass
