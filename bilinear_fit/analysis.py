# This file contains the functions to extract and fit relevant parameters of a bilinear model extracted from a dataframe.
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


from .bilinear_fits import *
from ...data_processing.clustering import rest_angle_with_simulation
from ...data_processing.load_df_script import load_csv_simulation


def stiffnessdata(
    df: pd.DataFrame,
    fit=False,
    window: int = None,
    method="PCA",
    fixed_point: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
):
    if fit:
        return bilinear_model_pca_fit(
            df,
            return_params=True,
            window=window,
            method=method,
            fixed_point=fixed_point,
        )[2](df["angle"]), bilinear_model_pca_fit(
            df,
            return_params=True,
            window=window,
            method=method,
            fixed_point=fixed_point,
        )[6]
    else:
        return bilinear_model_pca(df, return_params=True, method=method)[
            2
        ], bilinear_model_pca(df, return_params=True, method=method)[4]


def intercept_data(
    df: pd.DataFrame,
    fit=False,
    window: int = None,
    method="PCA",
    fixed_point: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
):
    if fit:
        return bilinear_model_pca_fit(
            df,
            return_params=True,
            window=window,
            method=method,
            fixed_point=fixed_point,
        )[3](df["angle"]), bilinear_model_pca_fit(
            df,
            return_params=True,
            window=window,
            method=method,
            fixed_point=fixed_point,
        )[7]
    else:
        return bilinear_model_pca(df, return_params=True, method=method)[
            3
        ], bilinear_model_pca(df, return_params=True, method=method)[5]


def model_data(
    df: pd.DataFrame,
    window: int = None,
    method="PCA",
    fixed_point: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
):
    return bilinear_model_pca_fit(
        df, return_params=True, window=window, method=method, fixed_point=fixed_point
    )[5], bilinear_model_pca_fit(
        df, return_params=True, window=window, method=method, fixed_point=fixed_point
    )[9]


def errormodel_simulation(
    df: pd.DataFrame,
    method="PCA",
    window: int = None,
    fit=False,
    fixed_point: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
):
    "Parameters : a processed Dataframe"
    rest_angle_simulation = rest_angle_with_simulation(df)
    rest_angle_model = find_zeros_of_bilinear(
        df, method=method, window=window, fit=fit, fixed_point=fixed_point
    )
    return abs(rest_angle_simulation - rest_angle_model)


def analysis_2D(
    dL=[0.88],
    offset=[0],
    width=[8],
    thickness=[0.127],
    colors=None,
    fit=False,
    window: int = None,
    method="PCA",
    fixed_point: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
):
    """
    Computes stiffness, rest_angle, and snap_angle as a function of d/L (dL) for given parameters.

    Parameters:
    - dL: List of dLs (d/L).
    - offsets: List of offsets (default is [0]).
    - width: List for width (default is 9).
    - thickness: List for thickness (optional).
    - colors: List of colors (optional, can be used to determine thickness).
    Be careful: only one of the lists must have a length greater than 1.

    Returns:
    - DataFrame containing the computed results.
    """

    if colors is None:
        assert thickness is not None, (
            "Thickness must be provided if color is not specified."
        )
    else:
        thickness = [
            data_thickness_colors[color] for color in colors
        ]  # Assuming `data_thickness_colors` is defined elsewhere

    results = {}
    dfs = load_csv_simulation(
        (["dL"], [dL]),
        (["offset"], [offset]),
        (["width"], [width]),
        (["thickness"], [thickness]),
    )

    def process_df(index, key, df):
        rest_angle = find_zeros_of_bilinear(
            df=df, window=window, method=method, fit=fit, fixed_point=fixed_point
        )
        if fit:
            var = bilinear_model_pca_fit(
                df,
                return_params=True,
                window=window,
                method=method,
                fixed_point=fixed_point,
            )
            break_angle = var[1]
            k_min = var[2]
            k_max = var[8]
            intercept_min = var[3]
            intercept_max = var[7]
            model = var[0]
            break_torque = model(break_angle)
        else:
            var = bilinear_model_pca(df, return_params=True, method=method)
            k_min = var[2]
            k_max = var[4]
            k_ratio = k_min / k_max
            intercept_min = var[3]
            intercept_max = var[5]
            break_angle = var[1]
            model = var[0]
            break_torque = k_min * break_angle + intercept_min
        if break_angle <= -180 or break_angle >= 180:
            break_angle = np.nan
            break_torque = np.nan
        rest_angle_error = errormodel_simulation(
            df, window=window, method=method, fit=fit
        )
        rest_angle_simulation = rest_angle_with_simulation(df)
        x = df["angle"].values
        y = df["torque"].values
        global_error = np.sqrt(np.mean((y - model(x)) ** 2))
        if df["stiffnessnum"].min() < -2:
            snap_angle = df.angle.loc[df.stiffnessnum == df.stiffnessnum.min()].values[
                0
            ]
        else:
            snap_angle = np.nan
        L = [
            k_min,
            k_max,
            k_ratio,
            intercept_min,
            intercept_max,
            rest_angle,
            break_angle,
            snap_angle,
            rest_angle_error,
            rest_angle_simulation,
            break_torque,
            global_error,
        ]
        return key, L

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_df, i, key, df): i
            for i, (key, df) in enumerate(dfs.items())
        }
        for future in futures:
            key, L = future.result()
            if len(dL) > 1:
                dL_value = dL[futures[future]]
                results[float(dL_value)] = L
            if len(offset) > 1:
                offset_value = offset[futures[future]]
                results[float(offset_value)] = L
            if len(width) > 1:
                width_value = width[futures[future]]
                results[float(width_value)] = L
            if len(thickness) > 1:
                thickness_value = thickness[futures[future]]
                results[float(thickness_value)] = L

    df_tot = pd.DataFrame(results).T
    df_tot.columns = [
        "k_min",
        "k_max",
        "k_ratio",
        "intercept_min",
        "intercept_max",
        "rest_angle",
        "break_angle",
        "snap_angle",
        "rest_angle_error",
        "rest_angle_simulation",
        "break_torque",
        "global_error",
    ]
    df_tot.index.name = ", ".join(
        name
        for condition, name in [
            (len(dL) > 1, "dL"),
            (len(offset) > 1, "offset"),
            (len(width) > 1, "width"),
            (len(thickness) > 1, "thickness"),
        ]
        if condition
    )

    parameters_to_add = {
        "dL": dL,
        "offset": offset,
        "width": width,
        "thickness": thickness,
    }
    for param_name, param_values in parameters_to_add.items():
        if len(param_values) == 1:
            df_tot[param_name] = param_values[0]

    df_tot.sort_index(inplace=True)
    return df_tot


def fit_2D(
    df: pd.DataFrame,
    parameter: str,
    power_values: Union[List[Union[float, int]], np.ndarray] = np.arange(0, 5, 1),
    degrees=[0, 1, 2],
    fixed_point: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
    polynomial=False,
    model=False,
):
    """

    Parameters:
    - df: a dataframe generated by the function analysis_2D
    - parameter : y-axis : in (k_min,k_max,rest_angle,snap_angle,error,rest_angle_simulation)

    """

    # Extraire les colonnes 'd/L' et 'k_min' du DataFrame

    if parameter == "rest_angle" or parameter == "rest_angle_simulation":
        df = df[(df[parameter] > -179.9) & (df[parameter] < 179.9)]
        X = df.index.values.reshape(-1, 1)
        y = df[parameter].values
    else:
        X = df.index.values.reshape(-1, 1)
        y = df[parameter].values

    if model == False:
        if polynomial == False:
            # Variables pour stocker les meilleurs résultats
            (
                best_model,
                best_slope,
                best_intercept,
                best_power,
                best_r_squared,
                best_y_pred,
                best_description,
            ) = best_fit(
                X,
                y,
                polynomial=polynomial,
                power_values=power_values,
                fixed_point=fixed_point,
            )
            # Retourner les meilleurs paramètres et résultats
            return (
                best_model,
                best_slope,
                best_intercept,
                best_power,
                best_r_squared,
                best_y_pred,
                best_description,
            )
        else:
            best_model, best_r_squared, best_y_pred, best_description = best_fit(
                X, y, polynomial=polynomial, fixed_point=fixed_point, degrees=degrees
            )
            return best_model, best_r_squared, best_y_pred, best_description
    else:
        return best_fit(
            X,
            y,
            polynomial=polynomial,
            power_values=power_values,
            fixed_point=fixed_point,
            degrees=degrees,
        )[0]
