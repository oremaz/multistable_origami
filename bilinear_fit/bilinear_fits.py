# Implements a bilinear or bi-power fit for a given dataset, and find the root of the fitted function.
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
from scipy.optimize import root_scalar
from sklearn.linear_model import LinearRegression
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Optional, Tuple

from ...data_processing.clustering import clustering
from ...data_processing.load_df_script import load_df
from ..Fits.fits import fit_linear_model, fit_linear_model_with_fixed_point, best_fit


def bilinear_model_pca(
    df: pd.DataFrame, return_params=False, y_axis="torque", method="PCA"
):
    X_positive, X_negative = clustering(df, method=method, y_axis=y_axis)
    # Extracts the 'angle' and 'torque' columns for each group
    x_pos, y_pos = X_positive["angle"].values.reshape(-1, 1), X_positive[y_axis].values
    x_neg, y_neg = X_negative["angle"].values.reshape(-1, 1), X_negative[y_axis].values

    model_pos = LinearRegression().fit(x_pos, y_pos)
    model_neg = LinearRegression().fit(x_neg, y_neg)

    # Gets the parameters of the models
    slope_pos, intercept_pos = (
        model_pos.coef_[0],
        model_pos.intercept_,
    )  # High stiffness
    slope_neg, intercept_neg = model_neg.coef_[0], model_neg.intercept_  # Low stiffness

    # Finds the intersection point (optimal break)
    optimal_break = (intercept_neg - intercept_pos) / (slope_pos - slope_neg)

    # Defines the bilinear function
    def bilinear_func(x_val):
        return np.where(
            x_val <= optimal_break,
            slope_neg * x_val + intercept_neg,
            slope_pos * x_val + intercept_pos,
        )

    if return_params:
        return (
            bilinear_func,
            optimal_break,
            slope_neg,
            intercept_neg,
            slope_pos,
            intercept_pos,
        )
    else:
        return bilinear_func


def find_best_powers(
    dL_values: Union[List[Union[float, int]], np.ndarray],
    y_axis="torque",
    power_values: Union[List[Union[float, int]], np.ndarray] = np.arange(0, 5, 1),
    offset=0.0,
    width=8.0,
    thickness=0.127,
    window: int = None,
    method="PCA",
    fixed_point: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
):
    L_min, L_max = [], []

    def process_dL(dL):
        df = load_df(dL, offset, width, thickness)
        X_positive, X_negative = clustering(df, y_axis=y_axis, method=method)

        if window is not None:
            X_negative = X_negative[:-window]
            X_positive = X_positive[window:]

        x_pos, y_pos = (
            X_positive["angle"].values.reshape(-1, 1),
            X_positive[y_axis].values,
        )
        x_neg, y_neg = (
            X_negative["angle"].values.reshape(-1, 1),
            X_negative[y_axis].values,
        )

        _, _, _, power_pos, _, _, _ = best_fit(
            x_pos,
            y_pos,
            polynomial=False,
            power_values=power_values,
            fixed_point=fixed_point,
        )
        _, _, _, power_neg, _, _, _ = best_fit(
            x_neg,
            y_neg,
            polynomial=False,
            power_values=power_values,
            fixed_point=fixed_point,
        )

        return power_neg, power_pos

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_dL, dL_values)

    for power_neg, power_pos in results:
        L_min.append(power_neg)
        L_max.append(power_pos)

    power_min = Counter(L_min).most_common(1)[0][0]
    power_max = Counter(L_max).most_common(1)[0][0]

    return power_min, power_max


def bilinear_model_pca_fit(
    df: pd.DataFrame,
    return_params=False,
    y_axis="torque",
    power_values: Union[List[Union[float, int]], np.ndarray] = np.arange(0, 5, 1),
    window: int = None,
    method="PCA",
    fixed_point: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
):
    # Splits the data using the PCA function
    X_positive, X_negative = clustering(df, method=method, y_axis=y_axis)
    power_min, power_max = find_best_powers(
        dL_values=np.round(np.arange(0.36, 1.005, 0.005), decimals=3),
        y_axis=y_axis,
        power_values=power_values,
        window=window,
        method=method,
        fixed_point=fixed_point,
    )
    if window is not None:
        X_negative = X_negative[:-window]
        X_positive = X_positive[window:]
    # Extracts the 'angle' and 'torque' columns for each group
    x_pos, y_pos = X_positive["angle"].values.reshape(-1, 1), X_positive[y_axis].values
    x_neg, y_neg = X_negative["angle"].values.reshape(-1, 1), X_negative[y_axis].values

    if fixed_point is not None:
        best_model_pos, _, _, _, _, _, _ = fit_linear_model_with_fixed_point(
            x_pos, y_pos, power_max, fixed_point[0], fixed_point[1]
        )
        best_model_neg, _, _, _, _, _, _ = fit_linear_model_with_fixed_point(
            x_neg, y_neg, power_min, fixed_point[0], fixed_point[1]
        )
    else:
        (
            best_model_pos,
            _,
            _,
        ) = fit_linear_model(x_pos, y_pos, power_max)
        (
            best_model_neg,
            _,
            _,
        ) = fit_linear_model(x_neg, y_neg, power_min)

    # Gets the parameters of the models
    slope_pos, intercept_pos = (
        best_model_pos.coef,
        best_model_pos.intercept,
    )  # High stiffness
    slope_neg, intercept_neg = (
        best_model_neg.coef,
        best_model_neg.intercept,
    )  # Low stiffness

    def find_intersection(
        slope_pos, intercept_pos, power_max, slope_neg, intercept_neg, power_min
    ):
        # Define the equation for the intersection
        def equation(x):
            return (slope_pos * (x**power_max) + intercept_pos) - (
                slope_neg * (x**power_min) + intercept_neg
            )

        # Use root_scalar with an initial guess and a method like 'newton' or 'secant'
        initial_guess = 1.0
        try:
            result = root_scalar(
                equation, x0=initial_guess, method="newton"
            )  # or method='secant'
            if result.converged:
                return result.root
            else:
                print("The method did not converge.")
                return None
        except ValueError as e:
            print(f"Root finding method failed: {e}")
            return None

    optimal_break = find_intersection(
        slope_pos, intercept_pos, power_max, slope_neg, intercept_neg, power_min
    )

    # Defines the bilinear function using models
    def bilinear_func(x_val):
        # Use models to predict values based on x_val and optimal_break
        y_val_neg = best_model_neg.predict(x_val)
        y_val_pos = best_model_pos.predict(x_val)
        return np.where(x_val <= optimal_break, y_val_neg, y_val_pos)

    # Return the function and parameters based on return_params flag
    if return_params:
        return (
            bilinear_func,
            optimal_break,
            slope_neg,
            intercept_neg,
            power_min,
            best_model_neg,
            slope_pos,
            intercept_pos,
            power_max,
            best_model_pos,
        )
    else:
        return bilinear_func


def find_zeros_of_bilinear(
    df: Optional[pd.DataFrame] = None,
    bilinear_func=None,
    limit: Union[float, int] = -180,
    bracket: Union[List[Union[float, int]], np.ndarray] = [-180, 180],
    method="PCA",
    window: int = None,
    fit=False,
    fixed_point: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
):
    if bilinear_func is None:
        if fit:
            bilinear_func = bilinear_model_pca_fit(
                df,
                return_params=False,
                window=window,
                method=method,
                fixed_point=fixed_point,
            )
        else:
            bilinear_func = bilinear_model_pca(df, return_params=False, method=method)

    # Find the zero with the bisection method
    def bilinear_func_2d(x):
        return bilinear_func(np.array([[x]]))

    try:
        # Find the zero with the bisection method using the modified function
        result = root_scalar(bilinear_func_2d, bracket=bracket, method="bisect")
        print(result)
        if result.converged:
            return result.root
    except ValueError as e:
        pass

    return limit  # Return -180 if the root is not found or if f(a) and f(b) are the same sign
