# <one line to give the program's name and a brief idea of what it does.>
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

from .analysis import *
from .interpolated_data_script import *
from .complete_df_script import *
from .bilinear_fits import *
from .rest_angle_calculator_multi import *
from ...data_processing.clustering import (
    find_optimal_break_angle,
    rest_angle_with_simulation,
    normalized_errors,
)
from ...data_processing.load_df_script import load_df
from ...data_processing.process_df_script import (
    preprocess_data,
    preprocess_data_for_accelernum,
)

save_dir = "../../Graphs/bilinear_fit"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Plot of simulation data
def plot_simulation(df: pd.DataFrame, rest_angle: Optional[Union[float, int]] = None):
    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(df["angle"], df["torque"], "ro", label="Simulation")

    # Ajouter le rest angle
    if rest_angle:
        rest_angle = rest_angle_with_simulation(df=df)
        plt.axvline(
            x=rest_angle,
            color="y",
            linestyle="--",
            label=f"rest_angle = {rest_angle:.2f}",
        )

    plt.xlabel("Angle")
    plt.ylabel("Torque")
    plt.legend()
    plt.savefig(f"{save_dir}/simulation_torque_vs_angle.pdf")
    plt.show()


# Plot of stiffness and acceleration data


def plot_stiffness_num_multiple(dico_df: dict, parameter="dL"):
    """
    Plot the stiffness number for multiple dataframes.

    Parameters:
    - dico_df: a dictionary containing the dataframes.
    """
    # Create a figure
    plt.figure(figsize=(12, 8))

    # Plot the stiffness number for each dataframe
    for key, df in dico_df.items():
        df = preprocess_data(df)
        stiffness_num_values = df["stiffnessnum"]
        # Je veux un label numérique (extrait dL de la clé sous forme 'data/data_raw-dL0_10-offset0-th0_05-width19_0/data_raw-dL0_10-offset0_0-th0_05-width19_0.csv'
        keylab = key.split(parameter)[1].split("-")[0].replace("_", ".")
        plt.plot(df["angle"], stiffness_num_values, label=keylab)

    # Add labels and title
    plt.xlabel("Angle")
    plt.ylabel("Stiffness")
    plt.legend()
    plt.savefig(f"{save_dir}/stiffness_vs_angle_multiple_{parameter}.pdf")
    plt.show()


def plot_acceler_num_multiple(dico_df: dict, parameter="dL"):
    """
    Plot the acceledLn number for multiple dataframes.

    Parameters:
    - dico_df: a dictionary containing the dataframes.
    """
    # Create a figure
    plt.figure(figsize=(12, 8))

    # Plot the acceledLn number for each dataframe
    for key, df in dico_df.items():
        df = preprocess_data_for_accelernum(df)
        accelern_num_values = df["accelernum"]
        # Je veux un label numérique (extrait dL de la clé sous forme 'data/data_raw-dL0_10-offset0-th0_05-width19_0/data_raw-dL0_10-offset0_0-th0_05-width19_0.csv'
        keylab = key.split(parameter)[1].split("-")[0].replace("_", ".")
        plt.plot(df["angle"], accelern_num_values, label=keylab)

    # Add labels and title
    plt.xlabel("Angle")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.savefig(f"{save_dir}/acceleration_vs_angle_multipledL_{parameter}.pdf")
    plt.show()


# Plot of fitted data
def plot_data_with_fit(
    df: pd.DataFrame,
    rest_angle: Optional[Union[float, int]] = None,
    y_axis="torque",
    process=False,
    fit=False,
    window: int = None,
    method="PCA",
    fixed_point: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
):
    if process:
        df = preprocess_data(df)
    x_pred = np.linspace(-180, 180, 3600)

    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(df["angle"], df[y_axis], "ro", label="Simulation")

    if fit:
        func, optimal_break, _, _, _, _, _, _, _, _ = bilinear_model_pca_fit(
            df,
            return_params=True,
            y_axis=y_axis,
            window=window,
            method=method,
            fixed_point=fixed_point,
        )
    else:
        func, optimal_break, _, _, _, _ = bilinear_model_pca(
            df, return_params=True, y_axis=y_axis, method=method
        )
    y_pred = func(x_pred)

    # Tracer les segments ajustés
    plt.plot(x_pred, y_pred, "b-", label="Modèle")

    if optimal_break >= -180 and optimal_break <= 180:
        # Ajouter le point de rupture
        plt.axvline(
            x=optimal_break,
            color="g",
            linestyle="--",
            label=f"break_angle = {optimal_break:.2f}",
        )

    # Ajouter le rest angle
    if rest_angle:
        rest_angle = find_zeros_of_bilinear(
            df=df, method=method, window=window, fit=fit, fixed_point=fixed_point
        )
        plt.axvline(
            x=rest_angle,
            color="y",
            linestyle="--",
            label=f"rest_angle = {rest_angle:.2f}",
        )

    plt.xlabel("Angle")
    plt.ylabel(f"{y_axis}")
    plt.title(f"{y_axis} vs Angle with bilinear fit using {method}")
    plt.legend()
    plt.legend()
    plt.savefig(f"{save_dir}/torque_vs_angle_fit_{method}.pdf")
    plt.show()


# Plot of global_analysis data


def plot_2D(df: pd.DataFrame, parameter: str):
    """

    Parameters:
    - df: a dataframe generated by the function analysis_2D
    - parameter : y-axis : in (k_min,k_max,rest_angle,break_angle,snap_angle,error,rest_angle_simulation,break_torque,intercept_min,intercept_max)
    """
    last_columns = df.iloc[:, -3:]  # Obtenir les 3 dernières colonnes
    for col in last_columns.columns:
        print(
            f"{col}: {last_columns[col].values[0]}"
        )  # Imprimer la clé et la première valeur
    df.plot(y=parameter, title=f"{parameter} = f({df.index.name})")
    plt.gca().get_legend().remove()
    plt.savefig(f"{save_dir}/{parameter}_vs_{df.index.name}.pdf")


def plot_2D_fit(
    df: pd.DataFrame,
    parameter: str,
    power_values: Union[List[Union[float, int]], np.ndarray] = np.arange(0, 5, 1),
    fixed_point: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
):
    X = df.index.values.reshape(-1, 1)
    y = df[parameter].values
    best_model, best_slope, best_intercept, best_power, _, _, best_description = fit_2D(
        df, parameter, power_values=power_values, fixed_point=fixed_point
    )
    y_pred = best_model.predict(X)
    # Print all the parameters
    last_columns = df.iloc[:, -3:]  # Obtenir les 3 dernières colonnes
    for col in last_columns.columns:
        print(
            f"{col}: {last_columns[col].values[0]}"
        )  # Imprimer la clé et la première valeur
    # Tracé pour paramètre
    plt.scatter(X, y, color="blue", label=f"Data_sim {parameter}")
    plt.plot(X, y_pred, color="red", label=best_description)
    plt.xlabel(f"{df.index.name}")
    plt.ylabel(f"{parameter}")
    plt.legend()
    plt.savefig(f"{save_dir}/{parameter}_vs_{df.index.name}_fit.pdf")


# Plot of interpolated data


def plot_interpolated_fit(
    dL: Union[float, int],
    thickness: Union[float, int],
    offset: Union[float, int],
    width: Union[float, int],
    angles=True,
    method="Optimisation",
):
    data = interpolated_fit(dL, offset, width, thickness, angles=angles, method=method)
    if angles:
        f = data[0]
        rest_angle = data[1]
        break_angle = data[2]
    else:
        f = data
    x_values = np.linspace(-180, 180, 3600)
    y_values = f(x_values)
    plt.plot(x_values, y_values, label=f"Interpolated fit for dL = {dL}")
    plt.xlabel("Angle")
    plt.ylabel("Torque")
    plt.title("Interpolated fit for bilinear model using {method}")
    if angles:
        plt.axvline(
            x=rest_angle,
            color="y",
            linestyle="--",
            label=f"rest_angle = {rest_angle:.2f}",
        )
        plt.axvline(
            x=break_angle,
            color="g",
            linestyle="--",
            label=f"break_angle = {break_angle:.2f}",
        )
    plt.legend()
    plt.savefig(
        f"{save_dir}/interpolated_profile_dL{dL}_thickness{thickness}_offset{offset}_width{width}.pdf"
    )
    plt.show()


# Comparison of different plots for torque vs Angle for different methods


def compare_plots(
    dL: Union[float, int] = 0.88,
    offset: Union[float, int] = 0,
    width: Union[float, int] = 8,
    thickness: Union[float, int] = 0.127,
    methods: List[str] = ["Optimisation"],
):
    df = load_df(dL, offset, width, thickness)
    x_pred = np.linspace(-180, 180, 3600)

    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    plt.plot(df["angle"], df["torque"], "r--", label="Simulation")

    colors = [
        "c",
        "m",
        "y",
        "orange",
        "purple",
    ]  # Cyan, Magenta, Yellow, Orange,Purple for the fits
    colors = colors[: len(methods)]

    for method, color in zip(methods, colors):
        (
            func,
            optimal_break,
            _,
            _,
            _,
            _,
        ) = bilinear_model_pca(df, return_params=True, method=method)
        y_pred = func(x_pred)
        rest_angle_fit = find_zeros_of_bilinear(df=df, method=method)
        break_angle = optimal_break
        plt.axvline(
            x=rest_angle_fit,
            color="y",
            linestyle="--",
            label=f"rest_angle_{method} = {rest_angle_fit:.2f}",
        )
        plt.axvline(
            x=break_angle,
            color="g",
            linestyle="--",
            label=f"break_angle_{method} = {break_angle:.2f}",
        )
        plt.plot(x_pred, y_pred, color=color, label=f"Fit {method}")

    for method in methods:
        # Tracer interpolated fit
        f_dico, rest_angle_dico, break_angle_dico = interpolated_fit_general(
            [dL], [offset], [width], [thickness], angles=True, method=method
        )
        f = list(f_dico.values())[0]
        rest_angle_interpolated = list(rest_angle_dico.values())[0]
        break_angle_interpolated = list(break_angle_dico.values())[0]
        y_pred_interpolated = f(x_pred)
        plt.plot(x_pred, y_pred_interpolated, color="k", label="Interpolated Data")

    # Ajouter les rest angles

    rest_angle_simulation = rest_angle_with_simulation(df)
    plt.axvline(
        x=rest_angle_simulation,
        color="r",
        linestyle="--",
        label=f"rest_angle_simulation = {rest_angle_simulation:.2f}",
    )
    plt.axvline(
        x=rest_angle_interpolated,
        color="b",
        linestyle="--",
        label=f"rest_angle_interpolated = {rest_angle_interpolated:.2f}",
    )

    # Ajouter les break angles
    plt.axvline(
        x=break_angle_interpolated,
        color="purple",
        linestyle="--",
        label=f"break_angle_interpolated = {break_angle_interpolated:.2f}",
    )
    plt.xlabel("Angle")
    plt.ylabel("Torque")
    plt.legend()
    plt.savefig(
        f"{save_dir}/comparison_plots_dL{dL}_offset{offset}_width{width}_thickness{thickness}.pdf"
    )
    plt.show()


# Comparison of rest_angles


def compare_rest_angle(df: pd.DataFrame):
    "Parameters : a dataframe generated by the function dataframe_complete"

    methods = df["methods"].values[0]

    plt.figure(figsize=(10, 6))

    # Plot rest angles from different methods
    plt.plot(
        df.index, df["rest_angle_simulation"], "r--", label="Rest Angle Simulation"
    )
    plt.plot(
        df.index,
        df["rest_angle_interpolated"],
        color="k",
        label="Rest Angle Interpolated",
    )

    colors = [
        "c",
        "m",
        "y",
        "orange",
        "purple",
    ]  # Cyan, Magenta, Yellow, Orange,Purple for the fits
    colors = colors[: len(methods)]

    for method, color in zip(methods, colors):
        plt.plot(
            df.index,
            df[f"rest_angle_{method}"],
            color=color,
            label=f"Rest Angle {method}",
        )

    plt.xlabel("dL")
    plt.ylabel("Rest Angle")
    plt.title("Rest Angle Comparison")

    # Définir le chemin des données
    base_path = "../../Experiments/Image_database/Simple_range"  # Modifier ce chemin
    base_path = os.path.abspath(base_path)

    # Si l'image test est présente, demander à l'utilisateur de dessiner le rectangle de crop
    cropbox = (600, 250, 4500, 3000)  # Valeur par défaut

    dL_values = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
    # Traiter les groupes et analyser les images
    results = image_analysis_module.process_groups(
        base_path,
        pinned_marker_color="green",
        free_marker_color="purple",
        cropbox=cropbox,
    )

    # Tracer le graphique
    image_analysis_module.plot_results(results, dL_values)

    offset = df["offset"].values[0]
    width = df["width"].values[0]
    thickness = df["thickness"].values[0]

    plt.legend(loc="upper left")
    plt.savefig(
        f"{save_dir}/comparison_rest_angles_offset{offset}_width{width}_thickness{thickness}.pdf"
    )
    plt.show()


# Optimisation Analysis


def plot_break_angle(
    dL_list: Union[List[Union[float, int]], np.ndarray] = np.round(
        np.arange(0.36, 1.005, 0.005), decimals=3
    ),
    hyperparam: Union[float, int] = 0,
):
    break_angles = []
    for dL in dL_list:
        df = load_df(dL, offset=0, width=8, thickness=0.127)
        break_angle = find_optimal_break_angle(df, hyperparam)
        break_angles.append(break_angle)
    plt.plot(dL_list, break_angles, "b--", label="Break Angle")
    plt.xlabel("dL")
    plt.ylabel("Break Angle")
    plt.title("Break Angle vs dL")
    plt.legend()
    plt.savefig(f"{save_dir}/break_angle_optimisation_process_vs_dL.pdf")
    plt.show()


def plot_error_hyperparam_bar(
    dL: Union[float, int] = 0.70,
    hyperparam_list: Union[List[Union[float, int]], np.ndarray] = [
        0,
        1e-3,
        1e-2,
        1e-1,
        1,
        10,
        10000,
    ],
):
    errors_data = []
    errors_rest_angle = []
    errors_ratio = []

    # Calcul des erreurs pour chaque hyperparamètre
    for hyperparam in hyperparam_list:
        df = load_df(dL, offset=0, width=8, thickness=0.127)
        break_angle = find_optimal_break_angle(df, hyperparam=hyperparam)
        error_data, error_rest_angle, error_ratio = normalized_errors(break_angle, df)
        errors_data.append(error_data)
        errors_rest_angle.append(error_rest_angle)
        errors_ratio.append(error_ratio)

    # Largeur de chaque groupe de barres et position
    bar_width = 0.2
    indices = np.arange(len(hyperparam_list))

    plt.figure(figsize=(12, 6))

    # Tracé des barres pour chaque type d'erreur
    plt.bar(indices - bar_width, errors_data, bar_width, label="Error_data", color="b")
    plt.bar(indices, errors_rest_angle, bar_width, label="Error_rest_angle", color="r")
    plt.bar(
        indices + bar_width, errors_ratio, bar_width, label="Error_ratio", color="g"
    )

    # Configuration de l'axe x
    plt.xlabel("Hyperparam")
    plt.ylabel("Errors")
    plt.title("Comparison of Errors vs Hyperparam")
    plt.xticks(
        indices, [f"{hp:.0e}" for hp in hyperparam_list]
    )  # Affichage des hyperparamètres
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/error_hyperparam_bar_dL{dL}.pdf")
    plt.show()


def plot_log_error_ratio_multiple_dL(
    dL_list: Union[List[Union[float, int]], np.ndarray],
    hyperparam_list: Union[List[Union[float, int]], np.ndarray] = [
        0,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1,
        10,
        100,
        1000,
    ],
):
    plt.figure(figsize=(10, 6))

    # Boucle sur chaque dL
    for dL in dL_list:
        errors_ratio = []

        # Calcul des erreurs pour chaque hyperparamètre
        for hyperparam in hyperparam_list:
            df = load_df(dL, offset=0, width=8, thickness=0.127)
            break_angle = find_optimal_break_angle(df, hyperparam=hyperparam)
            _, _, error_ratio = normalized_errors(break_angle, df)
            errors_ratio.append(error_ratio)

        # Conversion en logarithme
        log_errors_ratio = np.log(errors_ratio)

        # Tracé de log(error_ratio) pour le dL courant
        plt.plot(hyperparam_list, log_errors_ratio, marker="o", label=f"dL = {dL}")

    # Configuration du graphique
    plt.xlabel("Hyperparam")
    plt.ylabel("log(Error_ratio)")
    plt.title("log(Error_ratio) vs Hyperparam pour différentes valeurs de dL")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loglog_error_ratio_vs_hyperparam_dL.pdf")
    plt.show()
