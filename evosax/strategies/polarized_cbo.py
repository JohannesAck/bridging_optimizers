"""
Polarized CBO code.

Normal CBO is a special case of Polarized CBO, so you can use it is also implemented here as
an option.

https://arxiv.org/abs/2211.05238
"""

from typing import Dict, Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct

from utils import add_hist_to_info

from ..strategy import Strategy


@struct.dataclass
class EvoState:
    mean: chex.Array
    archive: chex.Array
    fitness: chex.Array
    best_archive: chex.Array
    best_archive_fitness: chex.Array
    best_member: chex.Array
    info_dict: Dict[str, jax.Array]
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0
    step_size: float = 0.5
    sigma: float = 0.1


@struct.dataclass
class EvoParams:
    step_size_init: float = 0.5
    sigma_init: float = 0.1
    beta: float = 1.0
    kernel_size: float = 0.1
    init_min: float = -0.1
    init_max: float = 0.1
    scale_factor: float = 1.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class PolarizedCBO(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        n_devices: Optional[int] = None,
        sigma_init: Optional[float] = 0.03,
        step_size_init: Optional[float] = 0.5,
        scale_factor: float = 1.0,
        whiten_weights: bool = False,
        do_default_cbo: bool = False, # do default CBO update, i.e. replaces kernel with constant
        constant_noise: bool = False,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """
        Polarized CBO, which is a variant of CBO that computes a "local mean" per particle.
        Bungert et al., https://arxiv.org/abs/2211.05238

        If `do_default_cbo` is True, then we actually do default CBO, which replaces the 
        kernel with 1.
        If `constant_noise` is True, then we do not use the distance between consensus point 
        and particle in noise calculation.

        When step_size = 1.0, we always use `do_default_cbo` and become equivalent to
        Optimization via Integration.
        """
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            n_devices=n_devices,
            **fitness_kwargs
        )
        self.sigma_init = sigma_init
        self.step_size_init = step_size_init
        self.scale_factor = scale_factor
        self.do_default_cbo = do_default_cbo
        self.constant_noise = constant_noise
        self.whiten_weights = whiten_weights
        if self.whiten_weights and not self.do_default_cbo:
            print('Warning: whiten_weights is kinda questionable with polarized CBO'
                  ' because whitening is done globally but consensus is calculated locally.')
        self.strategy_name = "PolarizedCBO"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(
            sigma_init=self.sigma_init,
            step_size_init=self.step_size_init,
            scale_factor=self.scale_factor,
        )

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.popsize, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        hist_names = ['kernel', 'step_dist', 'noise_dist', 'fitness_before', 'fitness_after']
        # hist_names = ['step_dist', 'noise_dist', 'fitness_before', 'fitness_after']
        info_dict = {name + '_hist': jnp.zeros(10) for name in hist_names}
        info_dict.update({name + '_bins': jnp.zeros(11) for name in hist_names})
        
        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            # fitness=jnp.zeros(self.popsize) + jnp.finfo(jnp.float32).max,
            fitness=jnp.zeros(self.popsize),
            best_archive=initialization,
            best_archive_fitness=jnp.zeros(self.popsize)
            + jnp.finfo(jnp.float32).max,
            best_member=initialization.mean(axis=0),
            info_dict=info_dict,
            sigma=params.sigma_init,
            step_size=params.step_size_init,
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """
        `ask` for new proposed candidates to evaluate next.
        1. Calculate the local mean for each particle
        2. Update toward local mean
        3. Add noise
        """

        x_next, info_dict = update_step(state.archive, state.fitness, state.sigma, state.step_size, 
                             params, self.do_default_cbo, self.constant_noise, self.whiten_weights, rng)

        x_next_scaled = x_next * params.scale_factor
        
        return x_next_scaled, state.replace(mean=jnp.mean(x_next,0), archive=x_next, info_dict=info_dict)


    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """
        `tell` update to ES state.
        """
        x_unscaled = x / params.scale_factor
        
        # best logging
        best_archive = jnp.where(
            jnp.mean(fitness) < jnp.mean(state.best_archive_fitness),
            x_unscaled,
            state.best_archive,
        )
        best_archive_fitness = jnp.minimum(fitness, state.best_archive_fitness)

        best_member_idx = jnp.argmin(fitness)
        best_member = jnp.where(
            fitness[best_member_idx] < state.best_fitness, 
            x_unscaled[best_member_idx],
            state.best_member
        )
        best_fitness = jnp.minimum(fitness[best_member_idx], state.best_fitness)

        return state.replace(
            fitness=fitness,
            best_archive=best_archive,
            best_archive_fitness=best_archive_fitness,
            best_member=best_member,
            best_fitness=best_fitness,
        )


def update_step(
    x: chex.Array,
    fitness: chex.Array,
    sigma: float,
    step_size: float,
    params: EvoParams,
    do_default_cbo: bool, 
    constant_noise: bool, 
    whiten_weights: bool,
    rng: chex.PRNGKey
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    This update step does both the update of particles towards the local mean and the addition of noise.
    Intuitively, the update should be in `tell` and the noise in `ask`, but splitting it up is a 
    lot less legible in my opinion.
    """

    info_dict = add_hist_to_info(dict(), fitness, 'fitness_before')
    
    if whiten_weights:
        fitness = fitness - jnp.mean(fitness)
        fitness = fitness / jnp.sqrt(jnp.var(fitness) + 1e-9)
    
    info_dict = add_hist_to_info(info_dict, fitness, 'fitness_after')

    # get global mean
    weight = jnp.exp(-params.beta * fitness)
    weight_normed  = weight / (jnp.sum(weight) + 1e-9)
    global_mean = jnp.dot(weight_normed, x)
    
    
    if do_default_cbo:
        local_means = jnp.repeat(jnp.expand_dims(global_mean, axis=0), x.shape[0], axis=0)
        info_dict = add_hist_to_info(info_dict, jnp.ones_like(local_means), 'kernel')  # wrong shape but doesn't matter
    else:
        member_ids = jnp.arange(x.shape[0])
        local_means, kernels = jax.vmap(
            get_local_weighted_mean, in_axes=(None, None, 0, None, None)
        )(x, fitness, member_ids, params.beta, params.kernel_size)
        global_mean_repeat = jnp.repeat(jnp.expand_dims(global_mean, axis=0), x.shape[0], axis=0)
        local_means = jnp.where(step_size == 1.0, global_mean_repeat, local_means)

        info_dict = add_hist_to_info(info_dict, kernels, 'kernel')

    # Update each towards local mean
    x_new = x + step_size * (local_means - x)

    if constant_noise:
        x_new_noisy = x_new + jax.random.normal(rng, x.shape) * sigma 
    else:
        x_new_noisy = x_new + jax.random.normal(rng, x.shape) * sigma * jnp.linalg.norm(local_means - x, axis=1, keepdims=True)
    
    info_dict = add_hist_to_info(info_dict, jnp.linalg.norm(x_new - x, axis=1), 'step_dist')
    info_dict = add_hist_to_info(info_dict, jnp.linalg.norm(x_new_noisy - x_new, axis=1), 'noise_dist')
    
    return x_new_noisy, info_dict


def get_local_weighted_mean(
    particle_locs: chex.Array,
    fitness: chex.Array,
    particle_id: int,
    beta: float,
    kernel_var: float,
) -> jax.Array:
    x = particle_locs[particle_id]
    kernels = jax.vmap(gauss_kernel, in_axes=(None, 0, None))(x, particle_locs, kernel_var)

    weights = kernels * jnp.exp(-beta * fitness)
    weights_normed = weights / (jnp.sum(weights) + 1e-9)
    weighted_mean = weights_normed @ particle_locs
    return weighted_mean, kernels


def gauss_kernel(x: chex.Array, y: chex.Array, kernel_var: float) -> float:
    return jnp.exp(-jnp.sum(jnp.square(x-y)) / (kernel_var + 1e-9))  # TODO check kernel correct
