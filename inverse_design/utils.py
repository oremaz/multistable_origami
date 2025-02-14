# This files contains useful functions for the inverse design of multistable origami.
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


from jax import lax
import jax.numpy as jnp
import scipy as sp
import importlib.util


def bump_function(x):
    """
    Returns a smooth transition between 0 and 1.
    0 for x < 0, 1 for x >= 1, and a polynomial transition in between.
    """
    return jnp.where(x < 0, 0, jnp.where(x >= 1, 1, x * x * (3 - 2 * x)))


def plateau(x, a, b, c, d):
    """
    Creates a plateau-shaped function that transitions based on bump_function().
    """
    return jnp.where(
        x < 0,
        d,
        jnp.where(
            x < a,
            d + (c - d) * bump_function(x / a),
            jnp.where(
                x <= b,
                c,
                jnp.where(x <= 1, c + (d - c) * bump_function((x - b) / (1 - b)), d),
            ),
        ),
    )


def mirror(f):
    """
    Returns a function that mirrors the input function f around the y-axis.
    """
    return lambda x: -f(-x)


def transpose(lists):
    """
    Transposes a list of lists, assuming each sub-list has equal length.
    """
    return [[row[i] for row in lists] for i in range(len(lists[0]))]


def sample_array(arr, start_indices, distance):
    """
    Returns slices of arr starting at each index in start_indices with given step distance.
    """
    return [arr[start::distance] for start in start_indices]


def hyperparam_jax(width, thickness):
    """
    JAX-compatible version of hyperparam().
    """
    return jnp.round(jnp.power(thickness / 0.127, 3) * (width / 8), 3)


def calculate_new_vector(dL):
    """
    Computes a new vector from dL by summing certain pairwise transformations.
    """
    diff = dL[:, None] - dL[None, :]
    squared_diff = jnp.power(diff, 2)
    plateau_values = plateau(100 * squared_diff, 1e-4, 1 - 1e-4, 0.01, 1)
    return jnp.sum(squared_diff * plateau_values, axis=1)


def generate_symmetric_configs(N):
    """
    Generates 2^(N-1) configurations of booleans for mirrored or non-mirrored states.
    """
    num_configs = 2**N
    configs = jnp.array(jnp.indices((2,) * N).reshape(N, num_configs).T, dtype=bool)
    return configs[: 2 ** (N - 1)]


def find_zeros_jax(f, bounds=(-180, 180), tol=1e-3, max_iter=100):
    """
    Finds a root of function f using a simple bisection approach in [a, b].
    """
    a, b = bounds
    fa, fb = f(a), f(b)
    is_bracket_valid = fa * fb < 0.0

    def cond_fun(state):
        a_, b_, fa_, fb_, iters = state
        return jnp.logical_and(jnp.abs(b_ - a_) > tol, iters < max_iter)

    def body_fun(state):
        a_, b_, fa_, fb_, iters = state
        c = 0.5 * (a_ + b_)
        fc = f(c)
        a_new = jnp.where(fa_ * fc < 0, a_, c)
        b_new = jnp.where(fa_ * fc < 0, c, b_)
        fa_new = jnp.where(fa_ * fc < 0, fa_, fc)
        fb_new = jnp.where(fa_ * fc < 0, fc, fb_)
        return a_new, b_new, fa_new, fb_new, iters + 1

    def nan_return(_):
        return jnp.nan

    def bisection_method(_):
        state = (a, b, fa, fb, 0)
        final_state = lax.while_loop(cond_fun, body_fun, state)
        a_final, b_final, _, _, _ = final_state
        return 0.5 * (a_final + b_final)

    return lax.cond(is_bracket_valid, bisection_method, nan_return, operand=None)


def find_zeros_scipy(f, bounds=(-180, 180), tol=1e-3, max_iter=200):
    # Use SciPy's root sscalr bisection method to find the root
    return sp.optimize.root_scalar(
        f, bracket=bounds, method="bisect", xtol=tol, maxiter=max_iter
    ).root


def sum_multilinear_jax(*parameters, pred_func, config=None):
    """
    Generates 2^(N-1) unique configurations of multilinear sums.
    """
    N = len(parameters)
    configs = generate_symmetric_configs(N)

    def create_function(c):
        preds = []
        for param, mirrored in zip(parameters, c):
            preds.append(pred_func(*param, mirrored=mirrored))
        return lambda x: jnp.sum(jnp.array([jnp.squeeze(p(x)) for p in preds]), axis=0)

    if config is not None:
        return create_function(config)
    return [create_function(c) for c in configs]


def indice_vectorized(list_of_tups, tups_list):
    # Convert the list of tuples into a JAX array
    list_of_tups_jax = jnp.array(
        list_of_tups
    )  # Shape: (N, d), where N is the number of tuples

    # Expand dimensions to broadcast for comparison
    # list_of_tups_jax: (N, d) -> (N, 1, d)
    # tups_list: (M, d) -> (1, M, d)
    expanded_tups = jnp.expand_dims(list_of_tups_jax, axis=1)  # Shape: (N, 1, d)
    expanded_tups_list = jnp.expand_dims(tups_list, axis=0)  # Shape: (1, M, d)

    # Perform element-wise comparison and check if all elements match along the last axis
    comparison = jnp.all(
        jnp.isclose(expanded_tups, expanded_tups_list, atol=1e-5), axis=2
    )  # Shape: (N, M)

    # Find indices where matches occur for each tuple in the list
    indices = jnp.where(
        comparison
    )  # Returns a tuple of arrays (row_indices, col_indices)
    unique_values, indices = jnp.unique(
        indices[1], return_index=True
    )  # Get unique row indices
    return unique_values[
        jnp.argsort(indices)
    ]  # Return only the column indices (matches in tups_list)


def tups_sampled(tups, num_indices, num_samples):
    # Générer des indices régulièrement espacés
    indices = jnp.linspace(0, len(tups) - 1, num=num_indices * num_samples, dtype=int)
    indices = indices.reshape(num_samples, num_indices)
    tups_sampled = tups[indices]
    return tups_sampled


def tups_sampled_conc(tups_sampled):
    # Concaténer les sous-tableaux pour obtenir un tableau 2D
    tups_sampled_conc = jnp.concatenate(tups_sampled, axis=0)
    return tups_sampled_conc


def theta_rest(dL, interpolator_func):
    def f(theta):
        return interpolator_func(theta, dL)

    return find_zeros_jax(f)


def dLrest(theta, interpolator_func):
    """
    Finds the zero of interpolator_func(theta, dL), effectively finding
    the baseline dL for a given theta[1].
    """

    def f(dL):
        return interpolator_func(theta, dL)

    return find_zeros_jax(f)


def dynamic_import(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tup_vectorized(list_params):
    n = int(0.5 * (len(list_params[0]) + 1))
    dL = jnp.array(list_params)[:, :n]
    hyperparam = jnp.array(list_params)[:, n:]
    hyperparam = jnp.insert(hyperparam, 0, 1, axis=1)
    tup = jnp.stack([dL, hyperparam], axis=-1)
    return tup
