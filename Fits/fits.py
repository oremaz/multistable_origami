# <Use Class_regression.py to optimize a polynomial regression model.>
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
from typing import List, Union, Optional, Tuple

from .Class_regression import *


def fit_linear_model(X, y, b_fixed: Union[float, int]):
    # Si b_fixed == 0, utiliser un modèle constant
    if b_fixed == 0:
        model = ConstantModel()
        model.fit(X, y)
        y_pred = model.predict(X)
        r_squared = model.score(X, y)
        return model, r_squared, y_pred

    # Si b_fixed > 0, utiliser un modèle puissance
    model = PowerModel(b_fixed)
    model.fit(X, y)
    y_pred = model.predict(X)
    r_squared = model.score(X, y)

    return model, r_squared, y_pred


def fit_linear_model_with_fixed_point(X, y, b_fixed, fixed_points):
    # Si b_fixed == 0, utiliser un modèle constant
    if b_fixed == 0:
        model = ConstantModel()
        model.fit(X, y)
        y_pred = model.predict(X)
        r_squared = model.score(X, y)
        return model, r_squared, y_pred

    # Si b_fixed > 0, utiliser un modèle puissance avec un point fixe
    model = PowerModelWithFixedPoints(b_fixed, fixed_points)
    model.fit(X, y)
    y_pred = model.predict(X)
    r_squared = model.score(X, y)

    return model, r_squared, y_pred


def fit_poly(
    X: Union[List[Union[float, int]], np.array],
    y: Union[List[Union[float, int]], np.array],
    deg: int,
):
    """Fit a polynomial regression model of degree deg."""
    model = PolynomialModel(deg)
    model.fit(X, y)
    y_pred = model.predict(X)
    r_squared = model.score(X, y)
    return model, r_squared, y_pred


def best_fit(
    X: Union[List[Union[float, int]], np.array],
    y: Union[List[Union[float, int]], np.array],
    polynomial: bool = False,
    power_values: Union[List[Union[float, int]], np.ndarray] = np.arange(0, 8, 1),
    degrees=[0, 1, 2, 3],
    fixed_point: Optional[List[Tuple[Union[float, int], Union[float, int]]]] = None,
):
    """Find the best model with the highest R² among various options."""

    # Test polynomial models with degrees 0 to 3
    if polynomial:
        best_r_squared = -np.inf  # Initialize best R² as negative infinity
        best_model = None
        best_pred = None
        best_description = ""
        for degree in degrees:
            model, r_squared, y_pred = fit_poly(X, y, degree)
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_model = model
                best_pred = y_pred
                best_description = f"Polynomial model of degree {degree}"
        return best_model, best_r_squared, best_pred, best_description

    elif fixed_point is not None:
        best_r_squared = -np.inf
        best_model = None
        best_pred = None
        best_description = ""
        for power in power_values:
            model, r_squared, y_pred = fit_linear_model_with_fixed_point(
                X, y, power, fixed_points=fixed_point
            )
            if r_squared > best_r_squared:
                best_power = power
                best_r_squared = r_squared
                best_model = model
                best_y_pred = y_pred
                best_slope = model.coef_
                best_intercept = model.intercept_
                best_description = (
                    f"y = {best_slope:.3f} * x^{power} + {best_intercept:.2f}"
                )
        return (
            best_model,
            best_slope,
            best_intercept,
            best_power,
            best_r_squared,
            best_y_pred,
            best_description,
        )
    # Test linear models with powers between 0 and 5,
    else:
        best_r_squared = -np.inf  # Initialize best R² as negative infinity
        best_model = None
        best_power = None
        best_y_pred = None
        best_slope = None
        best_intercept = None
        best_description = ""
        for power in power_values:
            # Ajuster le modèle linéaire avec la valeur b courante
            model, r_squared, y_pred = fit_linear_model(X, y, power)

            # Vérifier si cette valeur de b améliore le R²
            if r_squared > best_r_squared:
                best_power = power
                best_r_squared = r_squared
                best_model = model
                best_y_pred = y_pred
                best_slope = model.coef
                best_intercept = model.intercept
                best_description = (
                    f"y = {best_slope:.3f} * x^{power} + {best_intercept:.2f}"
                )
        return (
            best_model,
            best_slope,
            best_intercept,
            best_power,
            best_r_squared,
            best_y_pred,
            best_description,
        )
