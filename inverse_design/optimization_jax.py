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
from jax import lax, vmap
from jaxopt import LBFGS, LBFGSB, ProjectedGradient
from jaxopt.projection import projection_box
import nlopt
import numpy as np
from scipy.interpolate import griddata, interp1d
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, List, Dict, Union


# Local modules
from .interpolation_jax import interpolator2D, ExperimentalData
from .utils import *

image_analysis_module = dynamic_import(
    os.path.abspath("../../Experiments\Image_analysis\image_analysis_improved.py"), "image_analysis_module"
)


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
        return lax.cond(
            bool_val,
            lambda: -hyperparam * interpolator2D(-theta, dL, source=source),
            lambda: hyperparam * interpolator2D(theta, dL, source=source),
        )

    def objective_g(
        self,
        dL: Union[List[float], np.ndarray, jax.Array],
        hyperparam: Union[List[float], np.ndarray, jax.Array],
        theta: Union[List[float], np.ndarray, jax.Array],
        hyperparamfixed: float = 1,
        n: int = 2,
        source: Optional[str] = None,
    ) -> float:
        """
        Objective function to minimize in order to find a geometric configuration for a targeted theta
        """
        dL_min, dL_max = 0.33, 1.0
        bools = generate_symmetric_configs(n)
        hp = jnp.insert(jnp.array(hyperparam), 0, hyperparamfixed)
        dL = jnp.clip(jnp.array(dL), dL_min, dL_max)

        f_list = []
        for i, bool_config in enumerate(bools):
            f = 0
            for j, bo in enumerate(bool_config):
                f += self.generate_func(bo, theta[i], hp[j], dL[j], source=source)
            f_list.append(f)

        return jnp.sum(jnp.power(jnp.array(f_list), 2))

    def combined_objective_gen(
        self,
        params: Union[List[float], np.ndarray, jax.Array],
        target_value: float,
        theta: Union[List[float], np.ndarray, jax.Array],
        n: int = 2,
        source: Optional[str] = None,
    ) -> float:
        """
        Objective function with additional constraints and penalty terms.
        """
        dL = jnp.array(params[:n])
        # define with partial interpolator2D with fixed source
        def interpolator2D_s(theta, dL):
            return interpolator2D(theta, dL, source=source)
        dLrest1 = dLrest(theta[0], interpolator_func=interpolator2D_s)
        hyperparam = jnp.array(params[n:])

        # Objective 1: Minimizing the difference from target_value
        error = (
            self.objective_g(dL, hyperparam, theta, n=n, source=source) - target_value
        ) ** 2

        # Objective 2: Minimizing an internal difference measure for dL
        diff_dL = jnp.mean(calculate_new_vector(dL))

        # Objective 3: Checking hyperparam bounds [0.02, 1]
        diff_hp = lax.cond(
            jnp.any(jnp.where((1 >= hyperparam) & (hyperparam >= 0.02), 0, 1) == 1),
            lambda _: 1,
            lambda _: 0,
            operand=None,
        )

        # Objective 4: Checking dL is within [dLrest1, 1]
        diff_dL_2 = lax.cond(
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
        target_value: float,
        theta: Union[List[float], np.ndarray, jax.Array],
        n: int = 2,
        source: Optional[str] = None,
    ) -> float:
        """
        Similar to combined_objective_gen but without bounding constraints on parameters.
        """
        dL = jnp.array(params[:n])
        hyperparam = jnp.array(params[n:])
        error = (
            self.objective_g(dL, hyperparam, theta, n=n, source=source) - target_value
        ) ** 2

        # Minimize internal difference for dL
        diff_dL = jnp.mean(calculate_new_vector(dL))

        # L2 penalty on dL
        diff_dL_3 = jnp.sqrt(jnp.sum(jnp.power(dL, 2)))

        # Weights (can adjust)
        w1, w2, w5 = 1.0, 0, 5e-3
        total_objective = w1 * error + w2 * error * diff_dL + w5 * error * diff_dL_3
        return jnp.squeeze(total_objective)


class grids_and_bounds:
    "Class for generating grids and bounds for optimization (inputs and constraints)."

    def __init__(self):
        pass

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

    def generate_input(
        self, n: int, min_dL: float = 0.32, min_hp: float = 0.08
    ) -> Tuple[jax.Array, jax.Array]:
        "Generates input ranges and boundary ranges for optimization."
        input_ranges = [(min_dL, 1) for _ in range(n)] + [
            (min_hp, 1) for _ in range(n - 1)
        ]
        coordinate_ranges = [[(4e-1, 0.005), (1, 0.01)] for _ in range(n)] + [
            [(4e-2, 1e-3), (8e-2, 2e-3), (1, 5e-3)] for _ in range(n - 1)
        ]
        return input_ranges, coordinate_ranges

    def build_bounds(
        self, n: int, theta: float, min_hp: float = 0.08
    ) -> Tuple[jax.Array, jax.Array]:
        "Builds lower and upper bounds for optimization."
        dL_min = dLrest(theta, interpolator_func=interpolator2D)
        lower_bounds = jnp.concatenate([jnp.full(n, dL_min), jnp.full(n - 1, min_hp)])
        upper_bounds = jnp.full(2 * n - 1, 1.0)
        return lower_bounds, upper_bounds

    def tup_vectorized(
        self, list_params: List[Union[List[float], np.ndarray, jax.Array]]
    ) -> jax.Array:
        n = int(0.5 * (len(list_params[0]) + 1))
        dL = jnp.array(list_params)[:, :n]
        hyperparam = jnp.array(list_params)[:, n:]
        hyperparam = jnp.insert(hyperparam, 0, 1, axis=1)
        tup = jnp.stack([dL, hyperparam], axis=-1)
        return tup


class nlopt_optimizers:
    def __init__(self):
        self.objectives = objectives()
        self.gbs = grids_and_bounds()
        self.po = Postprocessing()

    def minimize_combination_with_nlopt(
        self,
        params: np.ndarray,
        target_value: float,
        theta: Union[List[float], np.ndarray, jax.Array],
        n: int = 2,
        source: Optional[str] = None,
    ) -> np.ndarray:
        """
        Launch optimization via NLopt (LD_MMA) with combined_objective_gen_non_bounded and added constraints.
        """
        # Convertit les paramètres initiaux en tableau NumPy pour NLopt
        params = np.asarray(params)

        def objective_with_fixed_args(params_flat):
            # Appel direct à combined_objective_gen_non_bounded
            return objectives.combined_objective_gen_non_bounded(
                params_flat, target_value, theta, n, source
            )

        def objective(params_flat, grad):
            obj_value = objective_with_fixed_args(params_flat)
            if grad.size > 0:
                grad_jax = jax.grad(objective_with_fixed_args)(params_flat)
                grad_concat = np.concatenate([np.array(g).ravel() for g in grad_jax])
                grad[:] = grad_concat
            print(float(obj_value))
            print(params_flat)
            return float(obj_value)

        opt = nlopt.opt(nlopt.LD_MMA, len(params))
        lower_bounds, upper_bounds = self.gbs.build_bounds(n, -180, min_hp=0.01)
        print(lower_bounds, upper_bounds)
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        opt.set_min_objective(objective)
        opt.set_stopval(1e-5)

        opt_result = opt.optimize(params)
        return opt_result

    def minimize_combination_with_nlopt2(
        self,
        params: np.ndarray,
        theta: Union[List[float], np.ndarray, jax.Array],
        n: int = 2,
        source: Optional[str] = None,
        max_time_minutes: float = 5.0,  # Maximum allowed time in minutes
    ) -> dict:
        """
        Launch optimization via NLopt (G_MLSL_LDS) with LN_BOBYQA as a local solver.
        A maximum time limit (in minutes) is enforced, and the best candidate (i.e.,
        the one with the lowest objective value encountered) is returned in a dict.
        
        The objective function minimizes the difference between computed rest angles
        and target angles, with a small regularization term.
        
        Returns a dict with:
        - "optimized_params": The best candidate parameter vector.
        - "objective": The corresponding objective value.
        """
        params = jnp.asarray(params)
        n = int(0.5 * (len(params) + 1))
        print(f"n = {n}")

        # Container to record the best candidate encountered during evaluations
        best_solution = {"x": None, "obj": float("inf")}

        def obj(params_flat):
            params_flat = jnp.asarray(params_flat)
            tup1 = self.gbs.tup_vectorized([params_flat])[0]
            rest_angles = self.po.find_rest_angles_interpolated_jax_hp(tup1, source=source)
            rest_angles = jnp.array(rest_angles)
            theta_jax = jnp.asarray(theta)
            return jnp.linalg.norm(rest_angles - theta_jax) + 0.001 * jnp.linalg.norm(params_flat)

        def objective(params_flat, grad):
            params_flat = jnp.asarray(params_flat)
            obj_value = obj(params_flat)
            obj_value_float = float(obj_value)
            
            # Record the candidate if it has a better objective value than seen so far
            if best_solution["x"] is None or obj_value_float < best_solution["obj"]:
                best_solution["obj"] = obj_value_float
                best_solution["x"] = np.array(params_flat)
            
            if grad.size > 0:
                grad_jax = jax.grad(obj)(params_flat)
                grad[:] = np.array(grad_jax)
            
            print(f"Objective: {obj_value_float}")
            print(f"Params: {np.array(params_flat)}")
            return obj_value_float

        # Set up NLopt using global optimization method G_MLSL_LDS and LN_BOBYQA as a local solver
        opt = nlopt.opt(nlopt.G_MLSL_LDS, len(params))
        lower_bounds, upper_bounds = self.gbs.build_bounds(n, -180, min_hp=0.001)
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        opt.set_min_objective(objective)
        opt.set_local_optimizer(nlopt.opt(nlopt.LN_BOBYQA, len(params)))
        opt.set_stopval(0.25 + 0.5 * (n - 1) * (n - 2))
        
        # Enforce maximum time (convert minutes to seconds)
        opt.set_maxtime(max_time_minutes * 60)

        # Run the optimization. Note that the candidate returned by NLopt via optimize
        # isn't necessarily the best one encountered.
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

        Returns a dictionary mapping each run index to a dict containing:
        - 'initial_params': the original parameter vector,
        - 'optimized_params': the result from NLopt.
        """
        results = {}
        for i, params in enumerate(params_list):
            print(f"\nStarting optimization for parameter set {i}")
            # Ensure params is a numpy array (if not already)
            params_np = np.array(params)
            # Call the original non-batch function
            optimized_params, objective_value = self.minimize_combination_with_nlopt2(params_np, theta, source=source)
            
            # Assemble the result. If your non-batch version later returns more than 
            # just the optimized parameters (e.g., an objective value or an optimization status),
            # include them here.
            results[i] = {
                "initial_params": params_np,
                "optimized_params": optimized_params,
                "objective": objective_value,        # if available
            }
        return results

class jax_optimizers:
    """
    Optimizations using JAX, referencing 'objectives' and 'grids_and_bounds'
    classes for underlying logic.
    """

    def __init__(self):
        self.obj = objectives()
        self.gb = grids_and_bounds()

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
            return lax.cond(
                value < best[0], lambda _: (value, point), lambda _: best, operand=None
            )

        total_points = points.shape[0]

        def loop_body(i, best):
            return eval_grid_point(i, best)

        best_value, best_point = lax.fori_loop(0, total_points, loop_body, initial_best)
        return best_value, best_point

    def optimize_multi_obj_jax_parallel__bruteforce(
        self,
        target_value: float,
        theta_grids: List[Union[List[float], np.ndarray, jax.Array]],
        min_hp: float = 0.08,
        source: Optional[str] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Executes a brute-force search on multiple grid dimensions, evaluating
        'combined_objective_gen_non_bounded' at each point.
        """
        theta_grids_flat = [grid.flatten() for grid in theta_grids]
        i_thetas_flattened = jnp.stack(theta_grids_flat, axis=-1)
        n2 = len(theta_grids)

        intervals, coordinate_ranges = self.gb.generate_input(n2, min_hp=min_hp)
        grids = self.gb.generate_grids(intervals, coordinate_ranges)

        def objective_for_grid(params, angles):
            return self.obj.combined_objective_gen_non_bounded(
                params, target_value, angles, n=n2, source=source
            )

        @jax.jit
        def core_function():
            def minimize_for_each_point(i_theta):
                return self.grid_minimize_optimized(
                    lambda p: objective_for_grid(p, i_theta), grids
                )

            return jax.vmap(minimize_for_each_point)(i_thetas_flattened)

        return core_function()

    def optimize_multi_obj_jax_parallel_simplified_gen_proj_step(
        self,
        target_value: float,
        theta_grids: List[Union[List[float], np.ndarray, jax.Array]],
        n2: int,
        s: float,
        min_hp: float = 0.08,
        source: Optional[str] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Implementation of Gradient Descent with fixed stepsize 's' and box constraints to optimize
        'combined_objective_gen_non_bounded' over provided theta_grids.
        """
        theta_grids_flat = [grid.flatten() for grid in theta_grids]

        def partial_combined_objective_gen(params, target_value, angles):
            return self.obj.combined_objective_gen_non_bounded(
                params, target_value, angles, n2, source=source
            )

        @jax.jit
        def core_function():
            def optimize_for_angles(angles):
                dL = jnp.ones(n2)
                hyperparam = jnp.ones(n2 - 1) * 0.5
                params = jnp.concatenate([dL, hyperparam])
                lower_bounds, upper_bounds = self.gb.build_bounds(
                    n2, angles.min(), min_hp
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
                value = self.obj.combined_objective_gen(
                    result.params, target_value, angles, n2, source=source
                )
                print(f"Optimization result: {value}")
                return value, result.params

            batched_input = jnp.stack(theta_grids_flat, axis=-1)
            optimize_all_pairs = vmap(optimize_for_angles, in_axes=(0,))
            return optimize_all_pairs(batched_input)

        return core_function()

    def optimize_multi_obj_jax_parallel_simplified_gen_proj_opt(
        self,
        target_value: float,
        theta_grids: List[Union[List[float], np.ndarray, jax.Array]],
        n2: int,
        min_hp: float,
        source: Optional[str] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Implementation of Gradient Descent with optimal stepsize,
        to optimize 'combined_objective_gen_non_bounded' over provided theta_grids.
        """
        theta_grids_flat = [grid.flatten() for grid in theta_grids]

        def partial_combined_objective_gen(params, target_value, angles):
            return self.obj.combined_objective_gen_non_bounded(
                params, target_value, angles, n2, source=source
            )

        @jax.jit
        def core_function():
            def optimize_for_angles(angles):
                dL = jnp.ones(n2)
                hyperparam = jnp.ones(n2 - 1) * 0.5
                params = jnp.concatenate([dL, hyperparam])
                lower_bounds, upper_bounds = self.gb.build_bounds(
                    n2, angles.min(), min_hp
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
            optimize_all_pairs = vmap(optimize_for_angles, in_axes=(0,))
            return optimize_all_pairs(batched_input)

        return core_function()

    def optimize_multi_obj_jax_parallel_simplified_gen_proj_lbfgsb(
        self,
        target_value: float,
        theta_grids: List[Union[List[float], np.ndarray, jax.Array]],
        n2: int,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Uses LBFGSB with box constraints to optimize
        'combined_objective_gen_non_bounded' on the given theta_grids.
        """
        theta_grids_flat = [grid.flatten() for grid in theta_grids]

        def partial_combined_objective_gen(params, target_value, angles):
            return self.obj.combined_objective_gen_non_bounded(
                params, target_value, angles, n2
            )

        @jax.jit
        def core_function():
            def optimize_for_angles(angles):
                dL = jnp.ones(n2)
                hyperparam = jnp.ones(n2 - 1) * 0.5
                params = jnp.concatenate([dL, hyperparam])
                lower_bounds, upper_bounds = self.gb.build_bounds(n2, angles.min())
                solver = LBFGSB(
                    fun=partial_combined_objective_gen, tol=1e-15, maxiter=500
                )
                result = solver.run(
                    init_params=params,
                    bounds=(lower_bounds, upper_bounds),
                    target_value=target_value,
                    angles=angles,
                )
                value = self.obj.combined_objective_gen_non_bounded(
                    result.params, target_value, angles, n2
                )
                print(f"Optimization result: {value}")
                return value, result.params

            batched_input = jnp.stack(theta_grids_flat, axis=-1)
            optimize_all_pairs = vmap(optimize_for_angles, in_axes=(0,))
            return optimize_all_pairs(batched_input)

        return core_function()

    def optimize_multi_obj_jax_parallel_simplified_gen_lbfgs(
        self,
        target_value: float,
        theta_grids: List[Union[List[float], np.ndarray, jax.Array]],
        n2: int,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Uses LBFGS (unconstrained) to optimize 'combined_objective_gen'
        on the given theta_grids.
        """
        theta_grids_flat = [grid.flatten() for grid in theta_grids]

        def partial_combined_objective_gen(params, target_value, angles):
            return self.obj.combined_objective_gen(params, target_value, angles, n2)

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
                value = self.obj.combined_objective_gen(
                    result.params, target_value, angles, n2
                )
                print(f"Optimization result: {value}")
                return value, result.params

            batched_input = jnp.stack(theta_grids_flat, axis=-1)
            optimize_all_pairs = vmap(optimize_for_angles, in_axes=(0,))
            return optimize_all_pairs(batched_input)

        return core_function()


class Postprocessing:
    """
    This class contains postprocessing functions for optimization results.
    """

    def __init__(self):
        self.gbs = grids_and_bounds()

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

    def interpolated_fit_global_jax(
        self,
        dL: float,
        offset: float,
        width: float,
        thickness: float,
        method: str = "regulargrid",
        source: Optional[str] = None,
    ) -> Callable[[jax.Array], jax.Array]:
        """
        Returns a JAX-compatible function that interpolates over theta
        while scaling by a hyperparameter function hyperparam_jax.
        """
        dL1 = dL

        def f1(theta):
            theta = jnp.atleast_1d(theta)
            dL2 = jnp.repeat(dL1, len(theta))
            f = interpolator2D(theta, dL2, method=method, source=source)
            hp_f1 = hyperparam_jax(width, thickness)
            return hp_f1 * f

        return f1

    def pred_fit_interpolated_global_jax(
        self,
        dL: float,
        offset: float,
        width: float,
        thickness: float,
        mirrored: bool = False,
        method: str = "regulargrid",
    ) -> Callable[[jax.Array], jax.Array]:
        """
        Returns a function that optionally mirrors the output of 'interpolated_fit_global'.
        """
        func = self.interpolated_fit_global_jax(
            dL, offset, width, thickness, method=method
        )
        return mirror(func) if mirrored else func

    def find_rest_angles_duo_interpolated_jax(
        self, tup: Union[List[float], np.ndarray, jax.Array]
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Finds two rest angles from a set of parameters using
        different 'mirrored' configurations.
        """
        rest_angle1 = find_zeros_jax(
            sum_multilinear_jax(
                *tup,
                pred_func=self.pred_fit_interpolated_global_jax,
                config=(False, False),
                method="regulargrid",
            )
        )
        rest_angle2 = find_zeros_jax(
            sum_multilinear_jax(
                *tup,
                pred_func=self.pred_fit_interpolated_global_jax,
                config=(False, True),
                method="regulargrid",
            )
        )
        return rest_angle1, rest_angle2

    def find_rest_angles_duo_interpolated_jax_parallel(
        self, tup_list: Union[List[List[float]], np.ndarray, jax.Array]
    ) -> jax.Array:
        """
        Vectorizes 'find_rest_angles_duo_interpolated_jax' across multiple tuples.
        """
        tup_array = jnp.array(tup_list)
        return jax.vmap(self.find_rest_angles_duo_interpolated_jax)(tup_array)

    def rest_angles_result(
        self,
        tups_sampled: Union[List[List[float]], np.ndarray, jax.Array],
        source: Optional[str] = None,
    ) -> List[jax.Array]:
        """
        Collects the computed rest angles for each element in 'tups_sampled'.
        """
        results = []
        for tup in tups_sampled:
            results.append(
                self.find_rest_angles_interpolated_jax_parallel(tup, source=source)
            )
        return results

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
        t = self.gbs.tup_vectorized(list(dictionary.values()))
        angles = self.rest_angles_result([t])
        angles = jnp.array(angles).T
        dict_bools_result = {}
        for i, (key, value) in enumerate(dictionary.items()):
            # Compare each angle in 'key' with angles[i]
            if all(abs(key[j] - angles[i][j]) < tol_angles for j in range(len(key))):
                if jnp.all(value[n:] >= threshold):
                    dict_bools_result[key] = 1
                else:
                    dict_bools_result[key] = 0
            else:
                dict_bools_result[key] = 0
        return dict_bools_result

    def pred_fit_interpolated_global_hp(
        self,
        dL: float,
        hyperparam: float,
        method: str = "regulargrid",
        mirrored: bool = False,
        source: Optional[str] = None,
    ) -> Callable:
        """
        Creates a partial interpolation function with scaled hyperparameters
        and an optional mirror transform.
        """
        func = self.interpolated_fit_global_jax(
            dL, 0, 8 * hyperparam, 0.127, method=method, source=source
        )
        return mirror(func) if mirrored else func

    def find_rest_angles_interpolated_jax_hp(
        self,
        tup: Union[List[float], np.ndarray, jax.Array],
        source: Optional[str] = None,
    ) -> List[jax.Array]:
        """
        Finds rest angles using a specialized 'pred_fit_interpolated_global_hp'
        for hyperparameter scaling.
        """

        def pred_func(dL_val, hp, mirrored=False):
            return self.pred_fit_interpolated_global_hp(
                dL_val, hp, method="regulargrid", mirrored=mirrored, source=source
            )

        list_func = sum_multilinear_jax(*tup, pred_func=pred_func)
        rest_angles = [find_zeros_jax(func) for func in list_func]
        return rest_angles

    def find_rest_angles_interpolated_jax_parallel(
        self,
        tup_list: Union[List[List[float]], np.ndarray, jax.Array],
        source: Optional[str] = None,
    ) -> jax.Array:
        """
        Vectorized version of 'find_rest_angles_interpolated_jax_hp'.
        """
        tup_array = jnp.array(tup_list)

        def func_to_map(tup):
            return self.find_rest_angles_interpolated_jax_hp(tup, source=source)

        return jax.vmap(func_to_map)(tup_array)

    def ratio_list(
        self,
        d: Tuple[jax.Array, jax.Array],
        theta_grid: Union[List[float], np.ndarray, jax.Array],
    ) -> List[float]:
        """
        Computes ratios of valid solutions for multiple angle tolerances in dict_bools_gen.
        """
        ratios = []
        n1 = int(log(len(theta_grid)) / log(2)) + 1
        for i in [1, 2, 3, 4, 5]:
            d1 = self.extract_grid_results_to_dict_gen_quick(d, theta_grid)
            d2 = self.dict_bools_gen(d1, n=n1, tol_angles=i, threshold=0.02)
            values = list(d2.values())
            n_values = len(values)
            ratios.append(sum(values) / n_values)
        return ratios


class plots:
    def __init__(self):
        self.obj = objectives()
        self.gb = grids_and_bounds()
        self.po = Postprocessing()
        self.jo = jax_optimizers()
        self.exp = ExperimentalData()

    def plot_colorbar_hp_boundarys(
        self,
        boundary_hps: List[float] = [0.02, 0.08],
        tol_angles: int = 1,
        min_angle: float = -180,
        max_angle: float = 0,
    ) -> None:
        "For n =2, plot for each couple of rest angles the hyperparameter value in a colorbar and boundary lines for fixed relevant hyperparameters."
        theta_grid = self.gb.grids_triangle(1, min_angle, max_angle)
        d = self.jo.optimize_multi_obj_jax_parallel_simplified_gen_proj_opt(
            0, theta_grid, n2=2, min_hp=0
        )
        d1 = self.po.extract_grid_results_to_dict_gen_quick(d, theta_grid)

        # Vectorize parameters and compute rest angles
        t = self.gb.tup_vectorized(list(d1.values()))
        angles = self.po.rest_angles_result([t])
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
        self,
        dL: Union[List[float], np.ndarray, jax.Array],
        offset: float,
        width: float,
        thickness: float,
        method: str = "regulargrid",
    ) -> None:
        "Plots the simulation data and the interpolated function (from experimental data) for a given dL value + the real experimental data if available."
        f0 = self.po.interpolated_fit_global_jax(
            dL, offset, width, thickness, method=method, source="sim"
        )
        f = self.po.interpolated_fit_global_jax(
            dL, offset, width, thickness, method=method, source="exp"
        )
        x = np.arange(-145, 125.5, 0.5)
        plt.plot(x, f0(x), label="Simulation Data")
        plt.plot(x, f(x), label="Interpolated Function (Experimental Data)")
        if dL in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
            filename = (
                f"../../" # Change the path
            )
            angle, torque = self.exp.read_experiment_spec(filename)
            plt.plot(angle, torque, "x", label="Experimental Data", color="red")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Torque (mN.m)")
        plt.legend()
        output_dir = f"../../Graphs/inverse_design" # Change the path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(
            f"{output_dir}/dL{dL}_offset{offset}_width{width}_thickness{thickness}_method{method}.png"
        )
        plt.show()

    def compare_rest_angle_experiment(self, save_dir: str) -> None:
        "Compare the rest angles obtained from the experiment, the interpolated function, the simulation and pictures of the samples."
        dL_grid = np.arange(0.33, 1.01, 0.01)
        rest_angles_sim = []
        rest_angles_interp_experiment = []
        for dL in dL_grid:
            f1 = self.po.interpolated_fit_global_jax(
                dL, 0, 8, 0.127, method="regulargrid"
            )
            rest_angle = find_zeros_scipy(f=f1)
            rest_angles_sim.append(rest_angle)
        dL_grid3 = np.arange(0.4, 1.01, 0.01)
        for dL in dL_grid3:
            f2 = self.po.interpolated_fit_global_jax(
                dL, 0, 8, 0.127, method="regulargrid", source="exp"
            )
            rest_angle = find_zeros_scipy(f=f2)
            rest_angles_interp_experiment.append(rest_angle)
        plt.plot(dL_grid, rest_angles_sim, label="Rest Angle Simulation", color="black")
        plt.plot(
            dL_grid3,
            rest_angles_interp_experiment,
            label="Rest Angle Interpolated",
            color="blue",
        )
        ## Add experimental data

        # Définir le chemin des données
        base_path = "./"  # Modifier ce chemin
        base_path = os.path.abspath(base_path)

        cropbox = (1600, 550, 4500, 4000)  # Valeur par défaut

        dL_values = [
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
        # Traiter les groupes et analyser les images
        results = image_analysis_module.process_groups(
            base_path,
            pinned_marker_color="green",
            free_marker_color="purple",
            cropbox=cropbox,
        )

        # Tracer le graphique
        image_analysis_module.plot_results(results, dL_values)
        plt.xlabel("dL")
        plt.ylabel("Rest Angle")

        plt.legend(loc="upper left")
        plt.savefig(f"{save_dir}/comparison_rest_angles_experiment.png")
        plt.show()
