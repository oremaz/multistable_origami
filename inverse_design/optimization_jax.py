# # This file contains optimization algorithms to inverse designing biststable origami using interpolated data, for any arbitrary dimensions (number of stable states). Implemented with JAX.>
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


from math import log
import os
import pickle

import jax
import jax.numpy as jnp
from jaxopt import LBFGS, LBFGSB, ProjectedGradient
from jaxopt.projection import projection_box
import nlopt
import numpy as np
from typing import Callable, Optional, Tuple, List, Dict, Union


# Local modules
from .interpolation_jax import interpolator2D
from .utils import *


class forward_problem:
    def __init__(self):
        pass

    def interpolated_fit_global_jax(
        self,
        dL: float,
        offset: float,
        width: float,
        thickness: float,
        source: Optional[str] = None,
    ) -> Callable[[jax.Array], jax.Array]:
        """
        Interpolated torque-angle profile for a shim characterized bydL, offset, width and thickness.
        """
        dL1 = dL

        def f1(theta):
            theta = jnp.atleast_1d(theta)
            dL2 = jnp.repeat(dL1, len(theta))
            f = interpolator2D(theta, dL2, source=source)
            hp_f1 = hyperparam_jax(width, thickness)
            return hp_f1 * f

        return f1

    def pred_fit_interpolated_global_hp(
        self,
        dL: float,
        hyperparam: float,
        mirrored: bool = False,
        source: Optional[str] = None,
    ) -> Callable:
        """
        Creates a partial interpolation function with scaled hyperparameters
        and an optional mirror transform.
        """
        func = self.interpolated_fit_global_jax(
            dL, 0, 8 * hyperparam, 0.127, source=source
        )
        return mirror(func) if mirrored else func

    def find_rest_angles_interpolated_jax_hp(
        self,
        params: Union[List[float], np.ndarray, jax.Array],
        source: Optional[str] = None,
    ) -> List[jax.Array]:
        """
        Finds rest angles using a specialized 'pred_fit_interpolated_global_hp'
        for hyperparameter scaling.
        """

        def pred_func(dL_val, hp, mirrored=False):
            return self.pred_fit_interpolated_global_hp(
                dL_val, hp, mirrored=mirrored, source=source
            )

        params = jnp.asarray(params)
        tup = tup_vectorized([params])[0]
        list_func = sum_multilinear_jax(*tup, pred_func=pred_func)
        rest_angles = [find_zeros_jax(func) for func in list_func]
        return rest_angles


    def find_rest_angles_interpolated_jax_parallel(
        self,
        params_list: Union[List[List[float]], np.ndarray, jax.Array],
        source: Optional[str] = None,
    ) -> jax.Array:
        """
        Vectorized version of 'find_rest_angles_interpolated_jax_hp'.
        """
        params_array = jnp.array(params_list)

        def func_to_map(params):
            return self.find_rest_angles_interpolated_jax_hp(params, source=source)

        return jax.vmap(func_to_map)(params_array)


class objectives:
    def __init__(self):
        pass

    def generate_func(
        self,
        bool_val: bool,
        theta: float,
        hyperparam: float,
        dL: float,
        source: Optional[str] = None,
    ) -> Callable[[float], float]:
        """
        Generate the torque-angle profile for mode 0 or mode 1
        """
        # Calcul des deux branches
        positive_branch = hyperparam * interpolator2D(theta, dL, source=source)
        negative_branch = -hyperparam * interpolator2D(-theta, dL, source=source)

        # Utilisation de jnp.where pour sélectionner la branche appropriée
        return jnp.where(bool_val, negative_branch, positive_branch)

    def objective_g(
        self,
        dL: Union[List[float], np.ndarray, jax.Array],
        hyperparam: Union[List[float], np.ndarray, jax.Array],
        theta: Union[List[float], np.ndarray, jax.Array],
        target_value: Union[List[float], np.ndarray, jax.Array],
        hyperparamfixed: float = 1,
        source: Optional[str] = None,
    ) -> float:
        """
        Objective function to minimize in order to find a geometric configuration for a targeted theta

        Parameters:
        - dL: List of n dL values.
        - hyperparam: List of n-1 hyperparameters.
        - theta: List of target angles for each configuration.
        - target_value: List of target torques for each configuration.
        - hyperparamfixed: Fixed hyperparameter value (first shim).
        - source: Source of the interpolation data (exp or sim).
        """
        n = len(dL)
        dL_min, dL_max = 0.33, 1.0
        bools = generate_symmetric_configs(n)
        hp = jnp.insert(jnp.array(hyperparam), 0, hyperparamfixed)
        dL = jnp.clip(jnp.array(dL), dL_min, dL_max)

        f_list = []
        for i, bool_config in enumerate(bools):
            f = 0
            for j, bo in enumerate(bool_config):
                f += self.generate_func(bo, theta[i], hp[j], dL[j], source)
            f_list.append(f - target_value[i])

        return jnp.sum(jnp.power(jnp.array(f_list), 2))

    def combined_objective_gen(
        self,
        params: Union[List[float], np.ndarray, jax.Array],
        target_value: Union[List[float], np.ndarray, jax.Array],
        theta: Union[List[float], np.ndarray, jax.Array],
        n: int = 2,
        source: Optional[str] = None,
    ) -> float:
        """
        Objective function with additional constraints and penalty terms.

        Parameters:
        - params: List of n dL values then n-1 hyperparameters characterizing the origami.
        - target_value: List of target torques for each configuration.
        - theta: List of target angles for each configuration.
        - source: Source of the interpolation data (exp or sim).
        """
        params = jnp.asarray(params)
        assert n == int(0.5 * (len(params) + 1))
        dL = jnp.array(params[:n])

        # define with partial interpolator2D with fixed source
        def interpolator2D_s(theta, dL):
            return interpolator2D(theta, dL, source=source)

        dLrest1 = dLrest(theta[0], interpolator_func=interpolator2D_s)
        hyperparam = jnp.array(params[n:])

        # Objective 1: Minimizing the difference from target_value
        error = (
            self.objective_g(dL, hyperparam, theta, target_value, source=source) ** 2
        )

        # Objective 2: Minimizing an internal difference measure for dL
        diff_dL = jnp.mean(calculate_new_vector(dL))

        # Objective 3: Checking hyperparam bounds [0.02, 1]
        diff_hp = jax.lax.cond(
            jnp.any(jnp.where((1 >= hyperparam) & (hyperparam >= 0.02), 0, 1) == 1),
            lambda _: 1,
            lambda _: 0,
            operand=None,
        )

        # Objective 4: Checking dL is within [dLrest1, 1]
        diff_dL_2 = jax.lax.cond(
            jnp.any(jnp.where((1 >= dL) & (dL >= dLrest1), 0, 1) == 1),
            lambda _: 1,
            lambda _: 0,
            operand=None,
        )

        # Objective 5: L2 penalty on dL
        diff_dL_3 = jnp.sqrt(jnp.sum(jnp.power(dL, 2)))

        # Weights
        w1, w2, w3, w4, w5 = 1.0, 0, 1e5, 1e5, 1e-6
        total_objective = (
            w1 * error
            + w2 * error * diff_dL
            + w3 * error * diff_hp
            + w4 * error * diff_dL_2
            + w5 * error * diff_dL_3
        )

        return jnp.squeeze(total_objective)

    def combined_objective_gen_non_bounded(
        self,
        params: Union[List[float], np.ndarray, jax.Array],
        target_value: Union[List[float], np.ndarray, jax.Array],
        theta: Union[List[float], np.ndarray, jax.Array],
        n: int = 2,
        source: Optional[str] = None,
    ) -> float:
        """
        Similar to combined_objective_gen but without bounding constraints on parameters.
        """
        params = jnp.asarray(params)
        assert n == int(0.5 * (len(params) + 1))
        dL = jnp.array(params[:n])
        hyperparam = jnp.array(params[n:])
        error = (
            self.objective_g(dL, hyperparam, theta, target_value, source=source) ** 2
        )

        # Minimize internal difference for dL
        diff_dL = jnp.mean(calculate_new_vector(dL))

        # L2 penalty on 1-params
        penalty = jnp.sqrt(jnp.sum(jnp.power(1 - params, 2)))

        # L2 penalty on params
        penalty2 = jnp.sqrt(jnp.sum(jnp.power(params, 2)))

        w1 = 1
        w2 = 0
        w5 = 2.5e-6
        w6 = 0
        # Weights (can adjust)
        total_objective = w1 * error + w2 * error * diff_dL + w5 * penalty + w6 * penalty2
        return jnp.squeeze(total_objective)


class inputs_and_constraints:
    "Class for generating grids and bounds for optimization (inputs and constraints)."

    def __init__(self):
        pass

    ## The two functions below are used to generatye a grid of points to test with brute force optimization

    def generate_input(
        self,
        n: int,
        min_dL: float = 0.4,
        max_dL: float = 1,
        min_hp: float = 0.08,
        max_hp: float = 1,
        steps_dL: Optional[List[float]] = None,
        steps_hp: Optional[List[float]] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """Generates input ranges and boundary ranges for optimization."""
        input_ranges = [(min_dL, max_dL) for _ in range(n)] + [
            (min_hp, max_hp) for _ in range(n - 1)
        ]
        if len(steps_dL) == 1:
            steps_dL = [steps_dL[0] for _ in range(n)]
        if len(steps_hp) == 1:
            steps_hp = [steps_hp[0] for _ in range(n - 1)]
        coordinate_ranges = [[(max_dL, steps_dL[i])] for i in range(n)] + [
            [(max_hp, steps_hp[i])] for i in range(n - 1)
        ]
        return input_ranges, coordinate_ranges

    def generate_grids(
        self,
        intervals: List[Tuple[float, float]],
        steps: List[Union[float, List[Tuple[float, float]]]],
    ) -> List[jax.Array]:
        """
        Parameters:
        - intervals: List of tuples representing (min, max) bounds for each dimension.
        - steps: List of either:
            - A single step size (float) for regular grid generation.
            - A list of tuples [(boundary1, step1), (boundary2, step2), ...] for dynamic step sizes.

        Returns:
        - A list of JAX arrays representing the grid points for each dimension.
        """

        def dynamic_grid(interval, step_ranges):
            """Generates a grid with dynamic step sizes based on boundaries."""
            min_val, max_val = interval
            grid = []

            current_val = min_val
            for boundary, step in step_ranges:
                if current_val < boundary:
                    grid.append(jnp.arange(current_val, min(boundary, max_val), step))
                    current_val = min(boundary, max_val)
                if current_val >= max_val:
                    break

            if current_val < max_val:
                last_step = step_ranges[-1][1]
                grid.append(jnp.arange(current_val, max_val, last_step))

            return jnp.concatenate(grid)

        grids = []
        for interval, step in zip(intervals, steps):
            if isinstance(step, list):
                grids.append(dynamic_grid(interval, step))
            else:
                grids.append(jnp.arange(interval[0], interval[1], step))

        return grids

    ## The two functions below are used to generate a grid of input angles for batch optimization with JAX

    def grids_triangle(
        self, step: float, min_val: float, max_val: float, n: int = 2
    ) -> jax.Array:
        "Generates a triangular grid for n dimensions. (number of points = comb(n,N_angles+n-1))"
        ranges = [np.arange(min_val, max_val + 1, step) for _ in range(n)]
        grids = np.meshgrid(*ranges, indexing="ij")
        grids_2d = np.vstack([g.ravel() for g in grids]).T
        valid = np.all(np.diff(grids_2d, axis=1) >= 0, axis=1)
        valid_grids = grids_2d[valid]
        return [jnp.array(valid_grids[:, i]) for i in range(n)]

    def create_random_meshgrid_sorted(
        self, N: int, n: int, interval: Tuple[int, int], seed: int
    ) -> jax.Array:
        """
        Generates N lists of n random integers within a given interval and sorts them in ascending order.
        """
        key = jax.random.PRNGKey(seed=seed)
        minval, maxval = interval
        keys = jax.random.split(key, N)
        sorted_lists = []

        for k in keys:
            random_integers = jax.random.randint(
                k, shape=(n,), minval=minval, maxval=maxval
            )
            sorted_integers = jnp.sort(random_integers)
            sorted_lists.append(sorted_integers)

        return jnp.array(sorted_lists).T

    ## The function below is used to generate bounds for optimization
    def build_bounds(
        self, n: int, theta: float = -180, min_hp: float = 0.08
    ) -> Tuple[jax.Array, jax.Array]:
        "Builds lower and upper bounds for optimization."
        # Explain the bounds theta and min_hp
        dL_min = dLrest(theta, interpolator_func=interpolator2D)
        lower_bounds = jnp.concatenate([jnp.full(n, dL_min), jnp.full(n - 1, min_hp)])
        upper_bounds = jnp.full(2 * n - 1, 1.0)
        return lower_bounds, upper_bounds


class nlopt_optimizers:
    def __init__(self):
        self.objectives = objectives()
        self.ics = inputs_and_constraints()
        self.fo = forward_problem()

    def minimize_combination_with_nlopt(
        self,
        params: Union[List[float], np.ndarray, jax.Array],
        target_value: Union[List[float], np.ndarray, jax.Array],
        theta: Union[List[float], np.ndarray, jax.Array],
        n: int = 2,
        min_hp: float = 0.08,
        min_theta: float = -180,
        source: Optional[str] = None,
        objective: Callable = None,  # New optional parameter
    ) -> np.ndarray:
        """
        Launch optimization via NLopt (LD_MMA) with a provided objective function to minimize.

        Parameters:
        - params: Initial guess for the optimization.
        - target_value: Target torque values for each configuration.
        - theta: Target angle values, negative and by ascending order.
        - n : Number of shims in parallel.
        - min_hp : Minimum hyperparameter value for bounds.
        - source : Source of the interpolation data (exp or sim).
        - min_theta : Set the minimum dL_value for bounds
        - objective : Custom objective function to minimize (defaults to 'combined_objective_gen_non_bounded')
        The objective function should have the signature:
        def objective(params_flat: np.ndarray, target_value: np.ndarray, theta: np.ndarray, source: str) -> float
        """
        # Set default objective if not provided
        if objective is None:
            objective = self.objectives.combined_objective_gen_non_bounded

        # Convert initial parameters to a NumPy array for NLopt
        params = np.asarray(params)

        def objective_with_fixed_args(params_flat):
            # Use the default or provided objective function
            return objective(params_flat, target_value, theta, source)

        def nlopt_objective(params_flat, grad):
            obj_value = objective_with_fixed_args(params_flat)
            if grad.size > 0:
                grad_jax = jax.grad(objective_with_fixed_args)(params_flat)
                grad_concat = np.concatenate([np.array(g).ravel() for g in grad_jax])
                grad[:] = grad_concat
            print(float(obj_value))
            print(params_flat)
            return float(obj_value)

        opt = nlopt.opt(nlopt.LD_MMA, len(params))
        lower_bounds, upper_bounds = self.ics.build_bounds(
            n, theta=min_theta, min_hp=min_hp
        )
        print(lower_bounds, upper_bounds)
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        opt.set_min_objective(nlopt_objective)
        opt.set_stopval(1e-5)

        opt_result = opt.optimize(params)
        return opt_result
    
    # Plot the number of iterations and the objective function value

    def minimize_combination_with_nlopt2(
        self,
        params: Union[List[float], np.ndarray, jax.Array],
        theta: Union[List[float], np.ndarray, jax.Array],
        n: int = 2,
        source: Optional[str] = None,
        max_time_minutes: float = 5.0,
        min_hp: float = 0.08,
        min_theta: float = -180,
        forward: Callable = None,
    ) -> dict:
        """
        Launch optimization via NLopt (G_MLSL_LDS) with LN_BOBYQA as a local solver using a provided forward function.

        Parameters:
        - params: Initial guess for the optimization.
        - theta: Target angle values.
        - n : Number of shims in parallel.
        - min_hp : Minimum hyperparameter value for bounds.
        - source : Source of the interpolation data (exp or sim).
        - min_theta : Set the minimum dL_value for bounds
        - objective : Custom objective function to minimize (defaults to 'combined_objective_gen_non_bounded')
        """
        # Set default objective if not provided
        if forward is None:
            forward = self.fo.find_rest_angles_interpolated_jax_hp

        params = jnp.asarray(params)
        n = int(0.5 * (len(params) + 1))
        print(f"n = {n}")

        # Container to record the best candidate encountered during evaluations
        best_solution = {"x": None, "obj": float("inf")}

        def obj(params):
            rest_angles = forward(params, source=source)
            rest_angles = jnp.array(rest_angles)
            theta_jax = jnp.asarray(theta)
            # Use the provided objective function signature.
            return jnp.linalg.norm(rest_angles - theta_jax) + 1e-4 * jnp.linalg.norm(
                1 - params
            )

        def nlopt_objective(params, grad):
            params = jnp.asarray(params)
            # You could opt to use the passed objective function directly if its signature matches,
            # or use the 'obj' helper; here we keep the original structure to record the best solution.
            obj_value = obj(params)
            obj_value_float = float(obj_value)

            if best_solution["x"] is None or obj_value_float < best_solution["obj"]:
                best_solution["obj"] = obj_value_float
                best_solution["x"] = np.array(params)

            if grad.size > 0:
                grad_jax = jax.grad(obj)(params)
                grad[:] = np.array(grad_jax)

            print(f"Objective: {obj_value_float}")
            print(f"Params: {np.array(params)}")
            return obj_value_float

        opt = nlopt.opt(nlopt.G_MLSL_LDS, len(params))
        lower_bounds, upper_bounds = self.ics.build_bounds(
            n, theta=min_theta, min_hp=min_hp
        )
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        opt.set_min_objective(nlopt_objective)
        opt.set_local_optimizer(nlopt.opt(nlopt.LN_BOBYQA, len(params)))
        opt.set_stopval(0.25 + 0.5 * (n - 1) * (n - 2))
        opt.set_maxtime(max_time_minutes * 60)

        try:
            _ = opt.optimize(np.array(params))
        except Exception as e:
            print("Optimization terminated with exception:", e)

        return best_solution["x"], best_solution["obj"]

    def batch_minimize_combinations(
        self,
        params_list: list,
        theta: Union[List[float], np.ndarray, jax.Array],
        source: Optional[str] = None,
    ) -> dict:
        """
        For each set of initial parameters in params_list, call the existing
        minimize_combination_with_nlopt2 function and collect the results.
        """
        results = {}
        for i, params in enumerate(params_list):
            print(f"\nStarting optimization for parameter set {i}")
            # Ensure params is a numpy array (if not already)
            params_np = np.array(params)
            # Call the original non-batch function
            optimized_params, objective_value = self.minimize_combination_with_nlopt2(
                params_np, theta, source=source
            )

            # Assemble the result. If your non-batch version later returns more than
            # just the optimized parameters (e.g., an objective value or an optimization status),
            # include them here.
            results[i] = {
                "initial_params": params_np,
                "optimized_params": optimized_params,
                "objective": objective_value,  # if available
            }
        return results


class jax_optimizers:
    """
    Optimizations using JAX, referencing 'objectives' and 'inputs_and_constraints'
    classes for underlying logic.
    """

    def __init__(self):
        self.obj = objectives()
        self.ics = inputs_and_constraints()

    def grid_minimize_optimized(
        self,
        func: Callable[[jax.Array], float],
        grids: List[Union[List[float], np.ndarray, jax.Array]],
    ) -> Tuple[float, jax.Array]:
        """
        Brute-force grid search to minimize 'func' using JAX loops over 'grids'.
        """
        mesh_grids = jnp.meshgrid(*grids, indexing="ij")
        points = jnp.column_stack([g.ravel() for g in mesh_grids])
        initial_best = (jnp.inf, jnp.zeros(len(grids)))

        def eval_grid_point(i, best):
            point = points[i]
            value = jnp.squeeze(func(point))
            return jax.lax.cond(
                value < best[0],
                lambda _: (value, point),
                lambda _: best,
                operand=None,
            )

        total_points = points.shape[0]

        def loop_body(i, best):
            return eval_grid_point(i, best)

        best_value, best_point = jax.lax.fori_loop(
            0, total_points, loop_body, initial_best
        )
        return best_value, best_point

    def recursive_optimize_multi_obj_jax_parallel_bruteforce(
        self,
        target_value: Union[List[float], np.ndarray, jax.Array],
        theta_grids: List[Union[List[float], np.ndarray, jax.Array]],
        min_hp_init: float = 0.08,
        min_dL_init: float = 0.4,
        source: Optional[str] = None,
        itnumber: int = 3,
        objective: Optional[Callable] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Executes a brute-force search on multiple grid dimensions.

        Parameters:
        - Target value : A list of targeted torque for each different configuration.
        - Theta grids : A list of lists of angles to optimize over (batch optimization).
        - min_hp_init : Minimum hyperparameter value for bounds.
        - min_dL_init : Minimum dL value for bounds.
        - source : Source of the interpolation data (exp or sim).
        - itnumber : Number of iterations over grid search (dynamic step size and grid adjustment).
        - objective : Custom objective function to minimize (defaults to 'combined_objective_gen_non_bounded')
        """
        if objective is None:
            objective = self.obj.combined_objective_gen_non_bounded

        theta_grids_flat = [grid.flatten() for grid in theta_grids]
        i_thetas_flattened = jnp.stack(theta_grids_flat, axis=-1)
        n2 = len(theta_grids)
        print(f"Number of angles: {n2}")
        step = 0.01
        min_dL = min_dL_init
        max_dL = 1.0
        min_hp = min_hp_init
        max_hp = 1.0
        obj = 0.0
        for _ in range(itnumber):
            intervals, coordinate_ranges = self.ics.generate_input(
                n2,
                min_dL=min_dL,
                max_dL=max_dL,
                min_hp=min_hp,
                max_hp=max_hp,
                steps_dL=[step],
                steps_hp=[step],
            )
            grids = self.ics.generate_grids(intervals, coordinate_ranges)

            def objective_for_grid(params, angles):
                return objective(params, target_value, angles, n=n2, source=source)

            @jax.jit
            def core_function():
                def minimize_for_each_point(i_theta):
                    return self.grid_minimize_optimized(
                        lambda p: objective_for_grid(p, i_theta), grids
                    )

                return jax.vmap(minimize_for_each_point)(i_thetas_flattened)

            result = core_function()
            if result[0] < obj:
                obj = result[0]
                min_dL = max(min(result[1][0], result[1][1]) - 0.05, min_dL_init)
                max_dL = min(max(result[1][0], result[1][1]) + 0.05, 1)
                min_hp = max(result[1][2] - 0.05, min_hp_init)
                max_hp = min(result[1][2] + 0.05, 1)
                step = step / 10
            else:
                break

        return result

    def optimize_multi_obj_jax_parallel_simplified_gen_proj_opt(
        self,
        target_value: Union[List[float], np.ndarray, jax.Array],
        theta_grids: List[Union[List[float], np.ndarray, jax.Array]],
        n2: int,
        min_hp: float,
        min_angle: Optional[float] = None,
        source: Optional[str] = None,
        objective: Optional[Callable] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Implements gradient descent with an optimal step size to optimize the objective
        over provided theta_grids.

        Parameters:
        - Target value : A list of targeted torque for each different combination of shims.
        - Theta grids : A list of lists of angles to optimize over (batch optimization).
        - n2 : Number of shims in parallel.
        - min_hp : Minimum hyperparameter value for bounds.
        - source : Source of the interpolation data (exp or sim).
        - min_angle : Set the minimum dL_value for bounds
        - objective : Custom objective function to minimize (defaults to 'combined_objective_gen_non_bounded')
        - w5 : Weight for the L2 penalty.
        """

        assert len(target_value) == 2 ** (n2 - 1)
        assert len(theta_grids) == 2 ** (n2 - 1)
        if objective is None:
            objective = self.obj.combined_objective_gen_non_bounded

        theta_grids_flat = [grid.flatten() for grid in theta_grids]

        def partial_combined_objective_gen(params, target_value, angles):
            return objective(params, target_value, angles, n2, source=source)

        @jax.jit
        def core_function():
            def optimize_for_angles(angles):
                dL = jnp.ones(n2)
                hyperparam = jnp.ones(n2 - 1) * 0.5
                params = jnp.concatenate([dL, hyperparam])
                angle_input = jnp.where(
    min_angle is None,
    jnp.array(angles.min(), jnp.float32),
    jnp.array(min_angle, jnp.float32)
)

                lower_bounds, upper_bounds = self.ics.build_bounds(
                    n2, angle_input, min_hp
                )
                solver = ProjectedGradient(
                    fun=partial_combined_objective_gen,
                    projection=projection_box,
                    tol=1e-100,
                    maxiter=1000,
                )
                result = solver.run(
                    init_params=params,
                    hyperparams_proj=(lower_bounds, upper_bounds),
                    target_value=target_value,
                    angles=angles,
                )
                value = self.obj.combined_objective_gen(
                    result.params, target_value, angles, n2, source=source
                )
                print(f"Optimization result: {value}")
                return value, result.params

            batched_input = jnp.stack(theta_grids_flat, axis=-1)
            optimize_all_pairs = jax.vmap(optimize_for_angles, in_axes=0)
            return optimize_all_pairs(batched_input)

        return core_function()

    def optimize_multi_obj_jax_parallel_simplified_gen_proj_step(
        self,
        target_value: Union[List[float], np.ndarray, jax.Array],
        theta_grids: List[Union[List[float], np.ndarray, jax.Array]],
        n2: int,
        s: float,
        min_hp: float = 0.08,
        source: Optional[str] = None,
        min_angle: Optional[float] = None,
        objective: Optional[Callable] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Implements gradient descent with fixed stepsize 's' and box constraints to optimize
        the objective over provided theta_grids.
        """
        if objective is None:
            objective = self.obj.combined_objective_gen_non_bounded

        theta_grids_flat = [grid.flatten() for grid in theta_grids]

        def partial_combined_objective_gen(params, target_value, angles):
            return objective(params, target_value, angles, n2, source=source)

        @jax.jit
        def core_function():
            def optimize_for_angles(angles):
                dL = jnp.ones(n2)
                hyperparam = jnp.ones(n2 - 1) * 0.5
                params = jnp.concatenate([dL, hyperparam])
                angle_input = jnp.where(
                    min_angle is None,
                    jnp.array(angles.min(), jnp.float32),
                    jnp.array(min_angle, jnp.float32)
                )

                lower_bounds, upper_bounds = self.ics.build_bounds(
                    n2, angle_input, min_hp
                )
                solver = ProjectedGradient(
                    fun=partial_combined_objective_gen,
                    projection=projection_box,
                    tol=1e-100,
                    maxiter=1000,
                    stepsize=s,
                )
                result = solver.run(
                    init_params=params,
                    hyperparams_proj=(lower_bounds, upper_bounds),
                    target_value=target_value,
                    angles=angles,
                )
                # Evaluate final value using the unconstrained objective.
                value = self.obj.combined_objective_gen(
                    result.params, target_value, angles, n2, source=source
                )
                print(f"Optimization result: {value}")
                return value, result.params

            batched_input = jnp.stack(theta_grids_flat, axis=-1)
            optimize_all_pairs = jax.vmap(optimize_for_angles, in_axes=0)
            return optimize_all_pairs(batched_input)

        return core_function()

    def optimize_multi_obj_jax_parallel_simplified_gen_proj_lbfgsb(
        self,
        target_value: Union[List[float], np.ndarray, jax.Array],
        theta_grids: List[Union[List[float], np.ndarray, jax.Array]],
        n2: int,
        min_angle: Optional[float] = None,
        objective: Optional[Callable] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Uses LBFGSB with box constraints to optimize the objective on the given theta_grids.
        """
        if objective is None:
            objective = self.obj.combined_objective_gen_non_bounded

        theta_grids_flat = [grid.flatten() for grid in theta_grids]

        def partial_combined_objective_gen(params, target_value, angles):
            return objective(params, target_value, angles, n2)

        @jax.jit
        def core_function():
            def optimize_for_angles(angles):
                dL = jnp.ones(n2)
                hyperparam = jnp.ones(n2 - 1) * 0.5
                params = jnp.concatenate([dL, hyperparam])
                angle_input = jnp.where(
    min_angle is None,
    jnp.array(angles.min(), jnp.float32),
    jnp.array(min_angle, jnp.float32)
)

                lower_bounds, upper_bounds = self.ics.build_bounds(n2, angle_input)
                solver = LBFGSB(
                    fun=partial_combined_objective_gen, tol=1e-15, maxiter=500
                )
                result = solver.run(
                    init_params=params,
                    bounds=(lower_bounds, upper_bounds),
                    target_value=target_value,
                    angles=angles,
                )
                value = objective(result.params, target_value, angles, n2)
                print(f"Optimization result: {value}")
                return value, result.params

            batched_input = jnp.stack(theta_grids_flat, axis=-1)
            optimize_all_pairs = jax.vmap(optimize_for_angles, in_axes=0)
            return optimize_all_pairs(batched_input)

        return core_function()

    def optimize_multi_obj_jax_parallel_simplified_gen_lbfgs(
        self,
        target_value: Union[List[float], np.ndarray, jax.Array],
        theta_grids: list,
        n2: int,
        objective: callable = None,
    ) -> tuple:
        """
        Uses LBFGS (unconstrained) to optimize 'combined_objective_gen'
        on the given theta_grids.
        """
        if objective is None:
            objective = self.obj.combined_objective_gen

        theta_grids_flat = [grid.flatten() for grid in theta_grids]

        def partial_combined_objective_gen(params, target_value, angles):
            return objective(params, target_value, angles, n2)

        @jax.jit
        def core_function():
            def optimize_for_angles(angles):
                dL = jnp.ones(n2)
                hyperparam = jnp.ones(n2 - 1) * 0.5
                params = jnp.concatenate([dL, hyperparam])
                solver = LBFGS(
                    fun=partial_combined_objective_gen, tol=1e-15, maxiter=500
                )
                result = solver.run(
                    init_params=params,
                    target_value=target_value,
                    angles=angles,
                )
                value = objective(result.params, target_value, angles, n2)
                print(f"Optimization result: {value}")
                return value, result.params

            batched_input = jnp.stack(theta_grids_flat, axis=-1)
            optimize_all_pairs = jax.vmap(optimize_for_angles, in_axes=0)
            return optimize_all_pairs(batched_input)

        return core_function()


class Postprocessing:
    """
    This class contains postprocessing functions for optimization results.
    """

    def __init__(self):
        self.fo = forward_problem()

    def extract_grid_results_to_dict_gen_quick(
        self,
        d: Tuple[jax.Array, jax.Array],
        theta_grids: List[Union[List[float], np.ndarray, jax.Array]],
    ) -> Dict[Tuple[float, ...], jax.Array]:
        """
        Extracts results from optimization ('d') made over 'theta_grids' to build a dictionary
        mapping each combination of angles in input to the corresponding results. Then saves it as a .pkl file.
        """
        theta_grids_flat = np.array(theta_grids).reshape(
            np.array(theta_grids).shape[0], -1
        )
        keys = np.vstack(theta_grids_flat).T
        keys_tuples = [tuple(key) for key in keys]
        values = d[1]
        results_dict = dict(zip(keys_tuples, values))

        i = 1
        filename_template = "dicts/results_dict_{}.pkl"
        os.makedirs(os.path.dirname(filename_template), exist_ok=True)
        while os.path.exists(filename_template.format(i)):
            i += 1

        filename = filename_template.format(i)
        with open(filename, "wb") as f:
            pickle.dump(results_dict, f)

        return results_dict

    def dict_bools_gen(
        self,
        dictionary: Dict[Tuple[float, ...], Union[List[float], np.ndarray, jax.Array]],
        n: int,
        tol_angles: int = 1,
        threshold: float = 0.08,
    ) -> Dict[Tuple[float, ...], int]:
        """
        Assigns boolean flags to each dictionary entry based on angle differences
        (rest angles vs. dictionary keys) and hyperparam thresholds.
        """
        angles = self.fo.find_rest_angles_interpolated_jax_parallel(
            list(dictionary.values())
        )
        angles = jnp.array(angles).T
        dict_bools_result = {}
        for i, (key, value) in enumerate(dictionary.items()):
            # Compare each angle in 'key' with angles[i]
            if all(abs(key[j] - angles[i][j]) < tol_angles for j in range(len(key))):
                if jnp.all(value[n:] >= threshold):
                    dict_bools_result[key] = 2
                else:
                    dict_bools_result[key] = 1
            else:
                dict_bools_result[key] = 0
        return dict_bools_result

    def ratio_list(
        self,
        d: Tuple[jax.Array, jax.Array],
        theta_grid: Union[List[float], np.ndarray, jax.Array],
        threshold: float = 0.02,
    ) -> List[float]:
        """
        Computes ratios of valid solutions for multiple angle tolerances in dict_bools_gen.
        """
        ratios = []
        n1 = int(log(len(theta_grid)) / log(2)) + 1
        for i in [1, 2, 3, 4, 5]:
            d1 = self.extract_grid_results_to_dict_gen_quick(d, theta_grid)
            d2 = self.dict_bools_gen(d1, n=n1, tol_angles=i, threshold=threshold)
            values = list(d2.values())
            n_values = len(values)
            ratios.append(
                [
                    sum(1 for value in values if value >= 2) / n_values,
                    sum(1 for value in values if value >= 1) / n_values,
                ]
            )
        return ratios
