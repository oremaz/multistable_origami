## Using JAX for optimization and interpolation

import jax.scipy.optimize
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .interpolation_jax import (
    ExperimentalDataPreparation,
    interpolator2D,
)
from .optimization_jax import *

# Local modules
from .utils import *

image_analysis_module = dynamic_import(
    os.path.abspath("../../Experiments/Image_analysis/image_analysis_improved.py"),
    "image_analysis_module",
)

# Global instances assumed to be well defined
obj = objectives()
ics = inputs_and_constraints()
po = Postprocessing()
jo = jax_optimizers()
fo = forward_problem()
exp_inst = (
    ExperimentalDataPreparation()
)  # renamed to avoid confusion with function parameters


def plot_colorbar_hp_boundarys(
    boundary_hps: list = [0.02, 0.08],
    tol_angles: int = 1,
    min_angle: float = -180,
    max_angle: float = 0,
) -> None:
    """
    For n=2, plot for each couple of rest angles the hyperparameter value in a colorbar
    and boundary lines for fixed relevant hyperparameters.
    """
    theta_grid = ics.grids_triangle(5, min_angle, max_angle)
    d = jo.optimize_multi_obj_jax_parallel_simplified_gen_proj_opt(
        [0, 0], theta_grid, n2=2, min_hp=0
    )
    d1 = po.extract_grid_results_to_dict_gen_quick(d, theta_grid)

    # Vectorize parameters and compute rest angles
    angles = fo.find_rest_angles_interpolated_jax_parallel(list(d1.values()))
    angles = jnp.array(angles).T

    # Filter valid points based on angle tolerances
    valid_hps = [
        value[2]
        for i, (key, value) in enumerate(d1.items())
        if all(abs(key[j] - angles[i][j]) < tol_angles for j in range(len(key)))
    ]
    valid_keys = [
        key
        for i, key in enumerate(d1.keys())
        if all(abs(key[j] - angles[i][j]) < tol_angles for j in range(len(key)))
    ]

    fig, ax = plt.subplots()
    sc = ax.scatter(
        [key[0] for key in valid_keys],
        [key[1] for key in valid_keys],
        c=valid_hps,
        cmap="coolwarm",
    )
    fig.colorbar(sc, ax=ax)
    ax.set_xlabel("theta1")
    ax.set_ylabel("theta2")

    # Prepare interpolation grid
    points = np.array(list(valid_keys))
    values = np.array(valid_hps)
    grid_x, grid_y = np.mgrid[
        min(points[:, 0]) : max(points[:, 0]) : 100j,
        min(points[:, 1]) : max(points[:, 1]) : 100j,
    ]
    grid_hp = griddata(points, values, (grid_x, grid_y), method="linear")

    # Contour the boundaries
    for boundary_hp in boundary_hps:
        cs = ax.contour(
            grid_x,
            grid_y,
            grid_hp,
            levels=[boundary_hp],
            colors="black",
            linestyles="--",
        )
        ax.clabel(cs, inline=True, fontsize=10, fmt=f"{boundary_hp:.2f}")

    plt.show()


def plot_df_with_global_interpolation_jax(
    dL: list or np.ndarray or jax.Array, offset: float, width: float, thickness: float
) -> None:
    """
    Plots the simulation data and the interpolated function (from experimental data)
    for a given dL value along with the experimental data if available.
    """
    f0 = fo.interpolated_fit_global_jax(dL, offset, width, thickness, source="sim")
    f = fo.interpolated_fit_global_jax(dL, offset, width, thickness, source="exp")
    x = np.arange(-145, 125.5, 0.5)
    plt.plot(x, f0(x), label="Simulation Data")
    plt.plot(x, f(x), label="Interpolated Function (Experimental Data)")

    if dL in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        filename = 'Change'  # Adjust the file path as necessary
        angle, torque = exp_inst.read_experiment_spec(filename)
        plt.plot(angle, torque, "x", label="Experimental Data", color="red")

    plt.xlabel("Angle (degrees)")
    plt.ylabel("Torque (mN.m)")
    plt.legend()
    output_dir = f"./Graphs/inverse_design"  # Adjust the path as necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(
        f"{output_dir}/dL{dL}_offset{offset}_width{width}_thickness{thickness}.png"
    )
    plt.show()


def compare_rest_angle_experiment(save_dir: Optional[str], include_exp: bool = False) -> None:
    """
    Compare the rest angles obtained from the experiment, the interpolated function,
    the simulation and pictures of the samples.
    """
    dL_grid = np.arange(0.33, 1.01, 0.01)
    dL_values = [
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ]

    rest_angles_sim = []
    rest_angles_interp_experiment = []
    for dL in dL_grid:
        f1 = fo.interpolated_fit_global_jax(dL, 0, 8, 0.127)
        rest_angle = find_zeros_scipy(f=f1)
        rest_angles_sim.append(rest_angle)

    dL_grid3 = np.arange(0.3, 1.01, 0.01)
    for dL in dL_grid3:
        f2 = fo.interpolated_fit_global_jax(dL, 0, 8, 0.127, source="exp")
        rest_angle = find_zeros_scipy(f=f2)
        if round(dL, 2) in dL_values:
            print(f"Rest angle for dL = {dL} : {rest_angle}")
        rest_angles_interp_experiment.append(rest_angle)

    plt.plot(dL_grid, rest_angles_sim, label="Rest Angle Simulation", color="black")
    plt.plot(
        dL_grid3,
        rest_angles_interp_experiment,
        label="Rest Angle Interpolated",
        color="blue",
    )

    # Include experimental data if requested
    if include_exp:
        base_path = 'change'  # Adjust the path as needed
        base_path = os.path.abspath(base_path)
        cropbox = (1600, 550, 4500, 4000)  # Default cropbox values

        results = image_analysis_module.process_groups(
            dL_values,
            base_path,
            pinned_marker_color="green",
            free_marker_color="purple",
            cropbox=cropbox,
        )
        image_analysis_module.plot_results(results, dL_values)

    plt.xlabel("dL")
    plt.ylabel("Rest Angle")
    plt.legend(loc="upper left")
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/comparison_rest_angles_experiment.png")
    plt.show()


def interpolator2D_derivative_theta(theta, dL, h=1e-6):
    theta_plus_h = theta + h
    theta_minus_h = theta - h

    f_theta_plus_h = interpolator2D(
        theta_plus_h,
        dL,
    )
    f_theta_minus_h = interpolator2D(theta_minus_h, dL)

    derivative_theta = (f_theta_plus_h - f_theta_minus_h) / (2 * h)

    return derivative_theta


def interpolator2D_stiffness(theta, dL):
    def func(theta):
        return interpolator2D(theta, dL)

    gradient = jax.grad(func)(theta)
    return gradient


def plot_stiffness(dL):
    theta_values = np.linspace(-180, 180, 360)
    derivative_values = [interpolator2D_stiffness(theta, dL) for theta in theta_values]

    plt.plot(theta_values, derivative_values, label=f"dL = {dL}")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Stiffness (dTorque/dTheta)")
    plt.legend()
    plt.show()


def compute_clusters_and_boundaries(theta_values, dL_values, func):
    theta_grid, dL_grid = np.meshgrid(theta_values, dL_values)

    derivative_values = np.array(
        [func(theta, dL) for theta, dL in zip(theta_grid.ravel(), dL_grid.ravel())]
    )

    data_points = np.column_stack(
        (theta_grid.ravel(), dL_grid.ravel(), derivative_values)
    )

    scaler = StandardScaler()
    data_points_scaled = scaler.fit_transform(data_points)

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data_points_scaled).reshape(
        theta_grid.shape[0], theta_grid.shape[1], 3
    )

    first_component = reduced_data[..., 0]

    labels = (first_component > 0).astype(int)
    boundary_points = []

    for i, row in enumerate(labels):
        boundary_index = np.min(np.where(row > 0)[0]) if np.any(row > 0) else None
        if boundary_index is not None:
            theta = theta_values[boundary_index]
            dL = dL_values[i]
            boundary_points.append((theta, dL))

    return labels, boundary_points


def rest_angles_2d(dL_values):
    solutions = []
    for dL in dL_values:

        def func(theta):
            return interpolator2D(theta, dL)

        theta_solution = find_zeros_scipy(f=func)
        solutions.append((dL, theta_solution)) 

    return solutions

def plot_clusters_and_boundaries(
    theta_values,
    dL_values,
    func,
    boundary_points,
    zlabel,
    rest_angle_bool=False,
    boudary_bound=False,
):
    theta_grid, dL_grid = np.meshgrid(theta_values, dL_values)

    values = np.array(
        [func(theta, dL) for theta, dL in zip(theta_grid.ravel(), dL_grid.ravel())]
    ).reshape(theta_grid.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        theta_grid, dL_grid, values, cmap="viridis", edgecolor="none", alpha=0.75
    )
    if boudary_bound:
        boundary_theta = [point[0] for point in boundary_points]
        boundary_dL = [point[1] for point in boundary_points]
        boundary_val = [func(point[0], point[1]) for point in boundary_points]
        # Add a small offset above the surface for the boundary points
        boundary_offset = 1e-4  
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
        rest_angles = rest_angles_2d(dL_values)
        boundary_offset = 1e-4
        rest_z = [func(point[1], point[0]) + boundary_offset for point in rest_angles]
        ax.scatter(
            [point[1] for point in rest_angles],
            [point[0] for point in rest_angles],
            rest_z,
            color="g",
            label="Rest Angles",
            s=50,
        )
    ax.set_xlabel("Theta (degrees)")
    ax.set_ylabel("dL")
    ax.set_zlabel(zlabel)
    ax.view_init(30, 200)  

    ax.legend()

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()


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
            obj.objective_g((dL1, dL2), hyperparam, (theta1, theta2), (0, 0))
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


def compute_2_angles(
    theta1,
    theta2,
    hyperparams,
    initial_guess=[0.5, 0.6],
    tol_angles=1,
):
    # Set the valid range for dL1 and dL2
    dL_min = dLrest(
        -180, interpolator_func=interpolator2D
    )  # Minimum value for dL1 and dL2
    dL_max = 1  # Maximum value for dL1 and dL2
    best_result = np.inf
    best_hyperparam = None
    dL_solution = None
    theta1_res_best, theta2_res_best = None, None
    results = []
    good_hps = []
    for hyperparam in hyperparams:
        # Define the objective functions based on dL1 and dL2

        def objective_dL(dL):
            return obj.objective_g(dL, hyperparam, [theta1, theta2], [0, 0])

        # Compile the gradient computation using JAX's automatic differentiation
        # jax_grad = jax.grad(objective_dL)

        # Create a wrapper function that returns both the function value and the gradient
        # def objective_with_grad(dL):
        # f = objective_dL(dL)
        # g = jax_grad(dL)
        # return f, np.array(g)

        bounds = [(dL_min, dL_max), (dL_min, dL_max)]  # Bounds for dL1 and dL2

        # Minimize the objective function
        result = minimize(
            objective_dL,
            x0=initial_guess,
            bounds=bounds,
            method="Nelder-Mead",
            jac=False,
            options={"maxiter": 50},
        )
        if result.success:
            print(result.x)
            theta1_res, theta2_res = (
                fo.find_rest_angles_interpolated_jax_hp(
                    [result.x[0], result.x[1], hyperparam]
                )[0],
                fo.find_rest_angles_interpolated_jax_hp(
                    [result.x[0], result.x[1], hyperparam]
                )[1],
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
    theta1, theta2, tol_angles, hyperparam_values
):

    _, _, good_hps, results = compute_2_angles(
        theta1, theta2, hyperparams=hyperparam_values, tol_angles=tol_angles
    )

    dL1_values = [result[0] for result in results]
    dL2_values = [result[1] for result in results]
    hyperparam_values = good_hps
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        dL1_values, dL2_values, c=hyperparam_values, cmap="viridis", marker="o"
    )
    plt.colorbar(sc, label="λ")

    plt.xlabel("dL1")
    plt.ylabel("dL2")

    plt.show()
