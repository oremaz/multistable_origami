# This file contains optimization algorithms to inverse designing biststable origami using interpolated data.
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
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Local modules
from ...data_processing.load_df_script import load_df
from .utils import *
from ..Fits import best_fit
from ..bilinear_fit.rest_angle_calculator_multi import (
    sum_multilinear,
    pred_sim,
    mirror,
    mirror_df,
    sum_inverse_multilinear,
)
from .interpolation_jax import interpolator_simple, interpolator2D


def plot_df_with_global_interpolation(
    dL, offset, thickness, width, method="regulargrid", mirrored=False
):
    df = load_df(dL, offset, thickness, width)
    f = interpolated_fit_global(dL, offset, thickness, width, method=method)
    if mirrored:
        f = mirror(f)
        df = mirror_df(df)
    plt.plot(df["angle"], df["torque"], label="Data")
    x = np.linspace(df["angle"].min(), df["angle"].max(), 1000)
    plt.plot(x, f(x), label="Interpolated function")
    plt.xlabel("Angle")
    plt.ylabel("Torque")
    plt.legend()
    plt.show()


def plot_df_with_global_interpolation3d(method="regulargrid"):
    # Interpoler les données
    interp_func = interpolator_simple(method=method)

    # Créer une grille fine pour l'interpolation
    angles_fine = np.arange(-180, 180, 0.5)
    dL_fine = np.linspace(0.6, 0.94, 100)
    angle_grid, dL_grid = np.meshgrid(angles_fine, dL_fine)

    # Points à interpoler
    points_to_interpolate = np.array([dL_grid.ravel(), angle_grid.ravel()]).T

    # Interpolation
    interpolated_torque = interp_func(points_to_interpolate)
    interpolated_torque = interpolated_torque.reshape(dL_grid.shape)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        dL_grid, angle_grid, interpolated_torque, cmap="viridis", edgecolor="none"
    )
    ax.set_xlabel("dL")
    ax.set_ylabel("Angle (degrees)")
    ax.set_zlabel("Torque")
    ax.set_title("Interpolated Torque Surface")

    # Corrected: Using `surf` here after it's defined
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.show()


def interpolator2D_derivative_theta(theta, dL, method="regulargrid", h=1e-5):
    """
    Calcule la dérivée numérique par rapport à theta pour l'interpolation 2D.

    Parameters:
    - theta: Angle (en degrés) autour duquel calculer la dérivée
    - dL: Valeur de dL
    - method: Méthode d'interpolation (par défaut 'regulargrid')
    - h: Pas pour la différence finie

    Returns:
    - Dérivée par rapport à theta
    """

    # Points pour la différence finie centralisée
    theta_plus_h = theta + h
    theta_minus_h = theta - h

    # Calcul des valeurs interpolées pour theta + h et theta - h
    f_theta_plus_h = interpolator2D(theta_plus_h, dL, method)
    f_theta_minus_h = interpolator2D(theta_minus_h, dL, method)

    # Dérivée centralisée par différence finie
    derivative_theta = (f_theta_plus_h - f_theta_minus_h) / (2 * h)

    return derivative_theta


# Derivate a second time
def interpolator2D_derivative_theta2(theta, dL, method="regulargrid", h=1e-5):
    """
    Calcule la dérivée numérique par rapport à theta pour l'interpolation 2D.

    Parameters:
    - theta: Angle (en degrés) autour duquel calculer la dérivée
    - dL: Valeur de dL
    - method: Méthode d'interpolation (par défaut 'regulargrid')
    - h: Pas pour la différence finie

    Returns:
    - Dérivée par rapport à theta
    """

    # Points pour la différence finie centralisée
    theta_plus_h = theta + h
    theta_minus_h = theta - h

    # Calcul des valeurs interpolées pour theta + h et theta - h
    f_theta_plus_h = interpolator2D_derivative_theta(theta_plus_h, dL, method)
    f_theta_minus_h = interpolator2D_derivative_theta(theta_minus_h, dL, method)

    # Dérivée centralisée par différence finie
    derivative_theta = (f_theta_plus_h - f_theta_minus_h) / (2 * h)

    return derivative_theta


# Plotting the derivative as a function of theta for a fixed dL
def plot_derivative_theta(dL, method="regulargrid"):
    theta_values = np.linspace(-180, 180, 100)
    derivative_values = [
        interpolator2D_derivative_theta(theta, dL, method) for theta in theta_values
    ]

    plt.plot(theta_values, derivative_values, label=f"dL = {dL}")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Torque Derivative (dTorque/dTheta)")
    plt.title("Torque Derivative with Respect to Theta")
    plt.legend()
    plt.show()


def plot_derivative_theta2(dL, method="regulargrid"):
    theta_values = np.linspace(-180, 180, 100)
    derivative_values = [
        interpolator2D_derivative_theta2(theta, dL, method) for theta in theta_values
    ]

    plt.plot(theta_values, derivative_values, label=f"dL = {dL}")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Torque Derivative (dTorque/dTheta)")
    plt.title("Torque Derivative with Respect to Theta")
    plt.legend()
    plt.show()


def plot_derivative_theta_3d(method="regulargrid", h=1e-5):
    # Define ranges for theta and dL
    theta_values = np.linspace(-180, 180, 100)
    dL_values = np.linspace(0.6, 0.94, 50)

    # Create mesh grid for theta and dL
    theta_grid, dL_grid = np.meshgrid(theta_values, dL_values)

    # Compute the derivative for each combination of theta and dL
    derivative_values = np.array(
        [
            interpolator2D_derivative_theta(theta, dL, method, h=h)
            for theta, dL in zip(theta_grid.ravel(), dL_grid.ravel())
        ]
    )
    derivative_grid = derivative_values.reshape(theta_grid.shape)

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        theta_grid, dL_grid, derivative_grid, cmap="viridis", edgecolor="none"
    )

    # Labels and title
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("dL")
    ax.set_zlabel("Stiffness")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()


def compute_clusters_and_boundaries(theta_values, dL_values, func, h=1e-5):
    """
    Calcule les clusters et les points de frontière à partir des dérivées et des données d'entrée.

    Parameters:
    - theta_values: Angles (en degrés).
    - dL_values: Valeurs de dL.
    - func: Fonction de couple dépendant de theta et dL.
    - h: Pas pour la dérivée.

    Returns:
    - clusters: Les indices des clusters pour chaque point.
    - boundary_points: Les points aux frontières des clusters (changements de cluster), avec les coordonnées (theta, dL, torque).
    """

    # Créer une grille de (theta, dL) pour l'ACP
    theta_grid, dL_grid = np.meshgrid(theta_values, dL_values)

    derivative_values = np.array(
        [func(theta, dL) for theta, dL in zip(theta_grid.ravel(), dL_grid.ravel())]
    )

    # Créer un tableau de données de (theta, dL, dérivée)
    data_points = np.column_stack(
        (theta_grid.ravel(), dL_grid.ravel(), derivative_values)
    )

    # Normaliser les données
    scaler = StandardScaler()
    data_points_scaled = scaler.fit_transform(data_points)

    # Appliquer l'ACP pour réduire la dimensionnalité à 3
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data_points_scaled).reshape(
        theta_grid.shape[0], theta_grid.shape[1], 3
    )

    # Extraire la première composante principale
    first_component = reduced_data[..., 0]

    # Définir les étiquettes pour séparer les données en fonction du signe de la première composante principale
    labels = (first_component > 0).astype(int)
    # Calculer le torque pour chaque point de frontière (theta, dL, torque)
    boundary_points = []

    # Ajouter le dernier point du premier cluster pour chaque ligne de `dL`
    for i, row in enumerate(labels):
        # Trouver le dernier point du premier cluster (signe positif)
        boundary_index = np.min(np.where(row > 0)[0]) if np.any(row > 0) else None
        if boundary_index is not None:
            theta = theta_values[boundary_index]
            dL = dL_values[i]
            boundary_points.append((theta, dL))

    return labels, boundary_points


def rest_angles_2d(dL_values):
    # Trouver la solution pour chaque dL
    solutions = []
    for dL in dL_values:

        def func(theta):
            return interpolator2D(theta, dL)

        # Fonction d'objectif pour chaque valeur de dL
        theta_solution = find_zeros_scipy(f=func)
        solutions.append((dL, theta_solution))  # (dL, theta_solution)

    return solutions


def plot_clusters_and_boundaries(
    theta_values,
    dL_values,
    func,
    boundary_points,
    rest_angle_bool=False,
    boudary_bound=False,
    zlabel="Stiffness",
):
    """
    Trace les clusters et les points de frontière dans un graphique 3D.

    Parameters:
    - theta_values: Angles (en degrés).
    - dL_values: Valeurs de dL.
    - func: Fonction de couple dépendant de theta et dL.
    - labels: Étiquettes des clusters pour chaque point (générées par `compute_clusters_and_boundaries`).
    - boundary_points: Liste des points de frontière (tuple (theta, dL, torque)).

    Returns:
    - None: Affiche le graphique.
    """
    # Créer une grille de (theta, dL)
    theta_grid, dL_grid = np.meshgrid(theta_values, dL_values)

    # Calculer les valeurs de la fonction de couple
    values = np.array(
        [func(theta, dL) for theta, dL in zip(theta_grid.ravel(), dL_grid.ravel())]
    ).reshape(theta_grid.shape)

    # Créer la figure et un axe 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Tracer la surface
    surf = ax.plot_surface(
        theta_grid, dL_grid, values, cmap="viridis", edgecolor="none", alpha=0.75
    )
    if boudary_bound:
        # Extraire les points de frontière
        boundary_theta = [point[0] for point in boundary_points]
        boundary_dL = [point[1] for point in boundary_points]
        boundary_val = [func(point[0], point[1]) for point in boundary_points]
        # Ajouter un petit décalage au-dessus de la surface pour les points de frontière
        boundary_offset = 0  # Petite valeur pour éviter le recouvrement
        boundary_z = [val + boundary_offset for val in boundary_val]
        ax.scatter(
            boundary_theta,
            boundary_dL,
            boundary_z,
            color="r",
            label="Boundary Points",
            s=50,
        )

    if rest_angle_bool:
        # Calculer les rest_angles pour chaque dL en utilisant la fonction objective_angle
        rest_angles = rest_angles_2d(dL_values)
        boundary_offset = 0.001
        # Ajouter un petit décalage au-dessus de la surface pour les "rest angles"
        rest_z = [func(point[1], point[0]) + boundary_offset for point in rest_angles]
        ax.scatter(
            [point[1] for point in rest_angles],
            [point[0] for point in rest_angles],
            rest_z,
            color="g",
            label="Rest Angles",
            s=50,
        )
    # Ajouter les labels et la légende
    ax.set_xlabel("Theta (degrees)")
    ax.set_ylabel("dL")
    ax.set_zlabel(zlabel)
    ax.view_init(30, 200)  # Ajuste l'angle de vue pour une meilleure visualisation

    # Ajouter une légende
    ax.legend()

    # Afficher la couleur de la surface
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Afficher le graphique
    plt.show()


def stiffness_fit_per_cluster_for_dL_vectorized(
    dL_values, theta_values, func, funcder, h=1e-5
):
    all_cluster_averages = np.zeros(
        (len(dL_values), 2)
    )  # Tableau pour les moyennes de rigidité pour 2 clusters (avant et après le seuil de boundary)
    _, boundary_points = compute_clusters_and_boundaries(
        theta_values, dL_values, funcder, h=h
    )

    for idx, dL_value in enumerate(dL_values):
        # Calculer les points de frontière pour chaque dL

        threshold_theta = boundary_points[idx][0]  # Seuil de frontière pour ce dL
        # Create masks to split theta values into two clusters
        cluster_1_mask = theta_values < threshold_theta
        cluster_2_mask = theta_values >= threshold_theta

        # Calculate cluster 1 and cluster 2 points using the masks
        cluster_1_points = np.column_stack(
            (
                theta_values[cluster_1_mask],
                np.array([dL_value] * np.sum(cluster_1_mask)),
                func(
                    theta_values[cluster_1_mask],
                    np.array([dL_value] * np.sum(cluster_1_mask)),
                ),
            )
        )
        cluster_2_points = np.column_stack(
            (
                theta_values[cluster_2_mask],
                np.array([dL_value] * np.sum(cluster_2_mask)),
                func(
                    theta_values[cluster_2_mask],
                    np.array([dL_value] * np.sum(cluster_2_mask)),
                ),
            )
        )

        # Calculer la moyenne de la rigidité pour chaque cluster en effectuant une régression linéaire
        cluster_averages = []

        for cluster_points in [cluster_1_points, cluster_2_points]:
            X = np.array([point[0] for point in cluster_points]).reshape(
                -1, 1
            )  # Angle (theta)
            y = np.array(
                [point[2] for point in cluster_points]
            )  # Torque (func(theta, dL_value))
            model = LinearRegression().fit(X, y)
            stiffness_mean = model.coef_[
                0
            ]  # Coefficient de régression linéaire (rigidité)
            cluster_averages.append(stiffness_mean)

        all_cluster_averages[idx] = cluster_averages  # Stocker les moyennes pour ce dL

    return all_cluster_averages


def plot_stiffness_by_dL(dL_values, cluster_averages):
    """
    Trace les moyennes de rigidité pour chaque cluster en fonction de dL.

    Parameters:
    - dL_values: Tableau des valeurs de dL.
    - cluster_averages: Tableau des moyennes de rigidité par cluster pour chaque valeur de dL.
    """
    plt.figure(figsize=(10, 6))

    for i in range(cluster_averages.shape[1]):
        plt.plot(dL_values, cluster_averages[:, i], label=f"Cluster {i + 1}")

        plt.xlabel("dL")
        plt.ylabel(f"Stiffness {i}")
        plt.legend()
        plt.show()


def plot_stiffness_with_fit(
    dL_values,
    cluster_averages,
    polynomial=False,
    power_values=np.arange(0, 3, 1),
    degrees=[0, 1, 2, 3],
    fixed_point=None,
):
    """
    Trace les moyennes de rigidité par cluster en fonction de dL et ajoute un fit basé sur la meilleure R².

    Parameters:
    - dL_values: Tableau des valeurs de dL.
    - cluster_averages: Tableau des moyennes de rigidité par cluster pour chaque valeur de dL.
    - polynomial: Si True, utilise un modèle polynomial pour le fit.
    - power_values: Liste des puissances à tester pour le fit linéaire.
    - degrees: Degrés des polynômes à tester.
    - fixed_point: Point fixe pour ajuster les modèles avec un point de référence.
    """
    plt.figure(figsize=(10, 6))

    for i in range(cluster_averages.shape[1]):
        best_model, _, _, _, _, _, best_description = best_fit(
            dL_values,
            cluster_averages[:, i],
            polynomial=polynomial,
            power_values=power_values,
            degrees=degrees,
            fixed_point=fixed_point,
        )
        y_pred = best_model.predict(dL_values.reshape(-1, 1))

        # Tracer la courbe du modèle de fit
        plt.plot(dL_values, cluster_averages[:, i], label=f"Cluster {i + 1}")
        plt.plot(dL_values, y_pred, label=f"{best_description} for Cluster {i + 1}")

        plt.xlabel("dL")
        plt.ylabel("Moyenne de rigidité (stiffness)")
        plt.title("Rigidité moyenne par cluster avec fit")
        plt.legend()
        plt.show()


def plot_derivative_theta2_3d(method="regulargrid", h=1e-5, outlier_threshold=None):
    """
    Plot the 3D surface of the derivative of stiffness with respect to theta,
    with an option to handle outlier values.

    Parameters:
    - method: The interpolation method to use (default 'regulargrid')
    - h: Step size for the derivative calculation (default 1e-5)
    - outlier_threshold: Value to filter out outliers (None for no filtering, otherwise specify a threshold)
    """
    # Define ranges for theta and dL
    theta_values = np.linspace(-180, 180, 100)
    dL_values = np.linspace(0.6, 0.94, 50)

    # Create mesh grid for theta and dL
    theta_grid, dL_grid = np.meshgrid(theta_values, dL_values)

    # Compute the derivative for each combination of theta and dL
    derivative_values = np.array(
        [
            interpolator2D_derivative_theta2(theta, dL, method, h=h)
            for theta, dL in zip(theta_grid.ravel(), dL_grid.ravel())
        ]
    )

    # Optionally filter out outliers based on the provided threshold
    if outlier_threshold is not None:
        derivative_values = np.where(
            np.abs(derivative_values) > outlier_threshold,
            np.nan,  # Replace outliers with NaN for exclusion from the plot
            derivative_values,
        )

    # Reshape the values to match the meshgrid shape
    derivative_grid = derivative_values.reshape(theta_grid.shape)

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        theta_grid, dL_grid, derivative_grid, cmap="viridis", edgecolor="none"
    )

    # Labels and title
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("dL")
    ax.set_zlabel("Stiffness_derivative")
    ax.set_title("3D Surface of Stiffness Derivative with Respect to Theta")

    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.show()


def objective_g(dL, hyperparam, theta1, theta2, hyperparamfixed=1):
    dL_min, dL_max = 0.33, 1.0
    dL1, dL2 = dL
    dL1 = np.clip(dL1, dL_min, dL_max)
    dL2 = np.clip(dL2, dL_min, dL_max)
    # Calculer f1 et f2 avec support vectoriel
    f1_value = hyperparamfixed * interpolator2D(
        theta1, dL1
    ) + hyperparam * interpolator2D(theta1, dL2)
    f2_value = hyperparamfixed * interpolator2D(
        theta2, dL1
    ) - hyperparam * interpolator2D(-theta2, dL2)

    # Retourner la somme des carrés pour chaque paire
    return f1_value**2 + f2_value**2


def plot_objective_surface(
    hyperparam,
    theta1,
    theta2,
    dL1_range=(0.4, 1.0),
    dL2_range=(0.4, 1.0),
    num_points=50,
):
    # Generate a grid of dL1 and dL2 values
    dL1_values = np.linspace(dL1_range[0], dL1_range[1], num_points)
    dL2_values = np.linspace(dL2_range[0], dL2_range[1], num_points)
    dL1_grid, dL2_grid = np.meshgrid(dL1_values, dL2_values)

    # Calculate objective values over the grid
    objective_values = np.array(
        [
            objective_g((dL1, dL2), hyperparam, theta1, theta2)
            for dL1, dL2 in zip(dL1_grid.ravel(), dL2_grid.ravel())
        ]
    ).reshape(dL1_grid.shape)

    # Plot the surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        dL1_grid, dL2_grid, objective_values, cmap="viridis", edgecolor="none"
    )

    # Labeling axes
    ax.set_xlabel("dL1")
    ax.set_ylabel("dL2")
    ax.set_zlabel("Objective Value")
    ax.set_title("Surface Plot of Objective Function")

    plt.show()


def find_dl2_for_dl1(
    dL1,
    target_value,
    hyperparam,
    theta1,
    theta2,
    initial_guesses=np.arange(0.4, 1, 0.05),
    tol_angles=2,
    hyperparamfixed=1,
):
    best_result = None
    lowest_objective_value = float("inf")

    for initial_guess in initial_guesses:

        def objective_minimize(dL2):
            return (
                objective_g(
                    (dL1, dL2[0]),
                    hyperparam,
                    theta1,
                    theta2,
                    hyperparamfixed=hyperparamfixed,
                )
                - target_value
            ) ** 2

        result = minimize(objective_minimize, x0=[initial_guess], bounds=[(0.4, 1.0)])
        if result.success and result.fun < lowest_objective_value:
            best_result = result
            lowest_objective_value = objective_minimize(result.x)
    dL2 = best_result.x[0]
    th2 = 0.127 * (hyperparam) ** (1 / 3)
    tup = ((dL1, 0, 8, 0.127), (dL2, 0, 8, th2))
    theta1_res, theta2_res = find_rest_angles_duo_interpolated(tup)
    error = abs(theta1 - theta1_res) + abs(theta2 - theta2_res)
    if (
        best_result is not None
        and abs(theta1 - theta1_res) < tol_angles
        and abs(theta2 - theta2_res) < tol_angles
    ):
        return best_result.x[0], error
    else:
        return None  # Return None if no good solution is found


def plot_dl2_vs_hyperparam(
    theta1, theta2, dL1, hyperparam_values=np.linspace(0.25, 4, 200), tol_angles=1e-1
):
    # Paramètres
    target_value = 0  # Valeur cible pour l'optimisation, ajustez selon vos besoins

    # Liste pour stocker les résultats
    dL2_values = []
    error_values = []

    # Calcul de dL2 pour chaque valeur de hyperparam
    for hyperparam in hyperparam_values:
        result = find_dl2_for_dl1(
            dL1,
            target_value,
            hyperparam,
            theta1,
            theta2,
            initial_guesses=np.arange(0.4, 1, 0.05),
            tol_angles=tol_angles,
        )
        if result is not None:
            dL2, error = result
            dL2_values.append(dL2)
            error_values.append(error)
        else:
            dL2_values.append(None)
            error_values.append(None)

    # Filtrer les valeurs None pour le tracé
    filtered_hyperparam_values = [
        hp for hp, dl2 in zip(hyperparam_values, dL2_values) if dl2 is not None
    ]
    filtered_dL2_values = [dl2 for dl2 in dL2_values if dl2 is not None]
    filtered_error_values = [err for err in error_values if err is not None]

    # Tracé du graphique
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        filtered_hyperparam_values,
        filtered_dL2_values,
        c=filtered_error_values,
        cmap="viridis",
        marker="o",
    )
    plt.colorbar(sc, label="Error")
    plt.xlabel("hyperparam")
    plt.ylabel("dL2")
    plt.title("Variation de dL2 en fonction de hyperparam")
    plt.grid(True)
    plt.show()


def find_dl2_for_dl1_global(dL1, target_value, hyperparam, theta1, theta2):
    def objective_minimize(dL2):
        return (
            objective_g((dL1, dL2[0]), hyperparam, theta1, theta2) - target_value
        ) ** 2

    result = differential_evolution(objective_minimize, bounds=[(0.4, 1.0)])
    dL2 = result.x[0]

    if result.success and result.fun < 1e-2:
        return result.x[0]
    else:
        return None  # Return None if no suitable solution is found


def plot_dl1_vs_dl2_for_objective(
    target_value,
    hyperparam,
    theta1,
    theta2,
    dL1_range=(0.4, 1.0),
    num_points=200,
    method="multiple_guesses",
    tol_angles=1,
):
    dL1_values = np.linspace(dL1_range[0], dL1_range[1], num_points)
    dL2_solutions = []
    error_values = []

    for dL1 in dL1_values:
        if method == "multiple_guesses":
            result = find_dl2_for_dl1(
                dL1, target_value, hyperparam, theta1, theta2, tol_angles=tol_angles
            )
        elif method == "global":
            result = find_dl2_for_dl1_global(
                dL1, target_value, hyperparam, theta1, theta2
            )

        if result is not None:
            dL2, error = result
            dL2_solutions.append(dL2)
            error_values.append(error)
        else:
            dL2_solutions.append(np.nan)
            error_values.append(np.nan)

    # Plot
    dL1_values = np.array(dL1_values)
    dL2_solutions = np.array(dL2_solutions)
    error_values = np.array(error_values)
    valid_mask = ~np.isnan(dL2_solutions)
    dL1_values = dL1_values[valid_mask]
    dL2_solutions = dL2_solutions[valid_mask]
    error_values = error_values[valid_mask]

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        dL1_values, dL2_solutions, c=error_values, cmap="viridis", marker="o"
    )
    plt.colorbar(sc, label="Error")
    # plt.plot(dL1_values, dL2_solutions, color='blue', label=f'Objective = {target_value}')
    plt.xlabel("dL1")
    plt.ylabel("dL2")
    plt.title("Curve of dL2 as a function of dL1 for target objective value")
    plt.grid()
    plt.legend()
    plt.show()


def plot_surface_dl2_vs_dl1_for_hyperparam_range(
    target_value,
    theta1,
    theta2,
    dL1_range=(0.4, 1.0),
    hyperparam_range=(0.1, 4),
    num_points=50,
    num_hyperparams=39,
    method="multiple_guesses",
    tol_angles=1,
):
    # Discrétiser les valeurs de dL1 et des hyperparamètres
    dL1_values = np.linspace(dL1_range[0], dL1_range[1], num_points)

    # Générer les valeurs de hyperparam entre les limites spécifiées
    hyperparam_values = np.linspace(
        hyperparam_range[0], hyperparam_range[1], num_hyperparams
    )

    # Création de la grille pour stocker les résultats
    dL2_solutions = np.zeros(
        (num_hyperparams, num_points)
    )  # Matrice des solutions de dL2 pour chaque (hyperparam, dL1)

    # Boucle sur chaque valeur de l’hyperparamètre
    for i, hyperparam in enumerate(hyperparam_values):
        for j, dL1 in enumerate(dL1_values):
            if method == "multiple_guesses":
                dL2 = find_dl2_for_dl1(
                    dL1, target_value, hyperparam, theta1, theta2, tol_angles=tol_angles
                )
            elif method == "global":
                dL2 = find_dl2_for_dl1_global(
                    dL1, target_value, hyperparam, theta1, theta2
                )

            # Stocke la solution si elle est valide, sinon met un NaN
            dL2_solutions[i, j] = dL2 if dL2 is not None else np.nan

    # Création de la figure 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Tracer la surface
    dL1_grid, hyperparam_grid = np.meshgrid(dL1_values, hyperparam_values)
    ax.plot_surface(
        dL1_grid,
        hyperparam_grid,
        dL2_solutions,
        cmap="viridis",
        edgecolor="k",
        alpha=0.8,
    )

    # Ajuster les limites des axes pour qu'ils se rejoignent à l'origine
    ax.set_xlim([dL1_range[0], dL1_range[1]])
    ax.set_ylim([hyperparam_range[0], hyperparam_range[1]])
    ax.set_zlim([0.4, np.nanmax(dL2_solutions)])

    # Labels et titres
    ax.set_xlabel("dL1")
    ax.set_ylabel("Hyperparam")
    ax.set_zlabel("dL2")
    ax.set_title("Surface of dL2 as a function of dL1 for different hyperparameters")

    plt.show()


def hyperparam(width, thickness):
    return round(float(np.power(thickness / 0.127, 3) * (width / 8)), 3)


def compute_2_angles(
    theta1,
    theta2,
    method="SLSQP",
    initial_guess=[0.6, 0.85],
    hyperparams=[0.2],
    tol_angles=1,
):
    # Set the valid range for dL1 and dL2
    dL_min = 0.33  # Minimum value for dL1 and dL2
    dL_max = 1  # Maximum value for dL1 and dL2
    best_result = np.inf
    best_hyperparam = None
    dL_solution = None
    theta1_res_best, theta2_res_best = None, None
    results = []
    good_hps = []
    # Loop through hyperparams to find solutions
    for hyperparam in hyperparams:
        # Define the objective functions based on dL1 and dL2

        def objective_dL(dL):
            return objective_g(dL, hyperparam, theta1, theta2)

        bounds = [(dL_min, dL_max), (dL_min, dL_max)]  # Bounds for dL1 and dL2

        # Minimize the objective function
        result = minimize(
            lambda dL: objective_dL(dL), initial_guess, bounds=bounds, method=method
        )
        # and abs(theta1-theta1_res) < tol_angles and abs(theta2-theta2_res) < tol_angles
        if result.success:
            theta1_res, theta2_res = find_rest_angles_duo_interpolated(
                (
                    (result.x[0], 0, 8, 0.127),
                    (result.x[1], 0, 8, 0.127 * (hyperparam) ** (1 / 3)),
                )
            )
            if (
                abs(theta1 - theta1_res) < tol_angles
                and abs(theta2 - theta2_res) < tol_angles
            ):
                results.append((result.x))
                good_hps.append(hyperparam)
                if float(result.fun) < best_result:
                    dL_solution = result.x
                    best_result = float(result.fun)
                    best_hyperparam = hyperparam
                    theta1_res_best, theta2_res_best = theta1_res, theta2_res
            else:
                print(
                    f"Optimization failed for hyperparam {hyperparam}: {result.message}"
                )
        else:
            print(f"Optimization failed for hyperparam {hyperparam}: {result.message}")

    print(f"Best parameters (dL1,dL2): {dL_solution}")
    print(f"Best hyperparam: {best_hyperparam}")
    print(f"Best angles: {theta1_res_best, theta2_res_best}")
    return best_hyperparam, dL_solution, good_hps, results


def plot_dL_vs_hyperparam(
    theta1, theta2, tol_angles=1, hyperparam_values=np.linspace(1, 4, 400)
):
    # Hyperparam range

    # Calculate the objective values for each hyperparam
    _, _, good_hps, results = compute_2_angles(
        theta1, theta2, hyperparams=hyperparam_values, tol_angles=tol_angles
    )

    # Extract dL1, dL2, and hyperparam values from results
    dL1_values = [result[0] for result in results]
    dL2_values = [result[1] for result in results]
    hyperparam_values = good_hps
    # Create a 2D plot for dL1 and dL2 with color based on hyperparam
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        dL1_values, dL2_values, c=hyperparam_values, cmap="viridis", marker="o"
    )
    plt.colorbar(sc, label="λ")

    # Labeling axes
    plt.xlabel("dL1")
    plt.ylabel("dL2")

    plt.show()


def find_best_in_segment(segment, theta1, theta2, target_dL):
    best_value = np.inf
    best_hyperparam = None
    for hyperparam in segment:
        current_value = objective_g(target_dL, hyperparam, theta1, theta2)
        if current_value < best_value:
            best_value = current_value
            best_hyperparam = hyperparam
    return best_hyperparam, best_value


def interpolated_fit_global(dL, offset, width, thickness, method="regulargrid"):
    def f1(theta):
        f = interpolator2D(theta, dL, method=method)
        hp_f1 = hyperparam(width, thickness)
        return hp_f1 * f

    return np.vectorize(f1)


def pred_fit_interpolated_global(
    dL: float,
    offset: float,
    width: float,
    thickness: float,
    mirrored=False,
    method="regulargrid",
):
    func = interpolated_fit_global(dL, offset, width, thickness, method=method)
    return mirror(func) if mirrored else func


def find_rest_angles_duo(tup):
    rest_angle1 = find_zeros_scipy(
        f=sum_multilinear(*tup, pred_func=pred_sim, config=(False, False), process=True)
    )
    rest_angle2 = find_zeros_scipy(
        f=sum_multilinear(*tup, pred_func=pred_sim, config=(False, True), process=True)
    )
    return rest_angle1, rest_angle2


def find_rest_angles_duo_interpolated(tup):
    rest_angle1 = find_zeros_scipy(
        f=sum_multilinear(
            *tup,
            pred_func=pred_fit_interpolated_global,
            config=(False, False),
            process=True,
            method="regulargrid",
        )
    )
    rest_angle2 = find_zeros_scipy(
        f=sum_multilinear(
            *tup,
            pred_func=pred_fit_interpolated_global,
            config=(False, True),
            process=True,
            method="regulargrid",
        )
    )
    return rest_angle1, rest_angle2


def find_rest_angles_duo_interpolated_series(tup):
    rest_angle1 = find_zeros_scipy(
        f=sum_inverse_multilinear(
            *tup,
            pred_func=pred_fit_interpolated_global,
            config=(False, False),
            process=True,
            method="regulargrid",
        )
    )
    rest_angle2 = find_zeros_scipy(
        f=sum_inverse_multilinear(
            *tup,
            pred_func=pred_fit_interpolated_global,
            config=(False, True),
            process=True,
            method="regulargrid",
        )
    )
    return rest_angle1, rest_angle2
