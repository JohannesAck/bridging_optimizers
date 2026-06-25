"""
Clustered CBO code.

Difference to Polarized CBO:
In Polarized CBO each particle has its own consensus point, based on kernels to other particles.
Thus the algorithm scales quadratically with the number of particles.

In clustered CBO, each cluster is a consensus point and the particles are updated towards a weighted
combination of the cluster consensus points, which is a bit more efficient.
The combination is weighted by a cluster probability, which measures how much each particle
"belongs" to each cluster and is updated during training.

In practice the efficiency argument really does not matter for us.

What does matter is that the cluster assignments become deterministic quickly, leading the algorithm
to essentially perform CBO separately in each cluster, ensuring convergence to multiple optima,
if cluster initialization is lucky.

https://arxiv.org/abs/2211.05238
"""

from functools import partial
from typing import Dict, Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct

from ..strategy import Strategy
from utils import add_hist_to_info



@struct.dataclass
class EvoState:
    mean: chex.Array
    archive: chex.Array
    fitness: chex.Array
    cluster_locs: chex.Array
    cluster_probs: chex.Array
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
    cluster_alpha: float = 1.0
    kernel_size: float = 0.1
    init_min: float = -0.1
    init_max: float = 0.1
    scale_factor: float = 1.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class ClusteredCBO(Strategy):
    def __init__(
        self,
        popsize: int,
        num_clusters: int = 4,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        n_devices: Optional[int] = None,
        sigma_init: float = 0.03,
        step_size_init: float = 0.5,
        scale_factor: float = 1.0,
        cluster_alpha: float = 1.0,
        whiten_weights: bool = False,
        do_default_cbo: bool = False, # do default CBO update, replaces kernel with constant
        constant_noise: bool = False,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """
        Polarized CBO, which is a variant of CBO that computes a "local mean" per particle.
        Bungert et al., https://arxiv.org/abs/2211.05238

        If `do_default_cbo` is True, then we actually do default CBO, which replaces the kernel with 1.
        If `constant_noise` is True, then we do not use the distance between consensus point and particle in noise calculation.

        When step_size = 1.0, we always disable polarization and perform Optimization via Integration.
        """
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            n_devices=n_devices,
            **fitness_kwargs
        )
        self.sigma_init = sigma_init
        self.cluster_alpha = cluster_alpha
        self.num_clusters = num_clusters
        self.step_size_init = step_size_init
        self.scale_factor = scale_factor
        self.do_default_cbo = do_default_cbo
        self.constant_noise = constant_noise
        self.whiten_weights = whiten_weights
        if self.whiten_weights and not self.do_default_cbo:
            print('Warning: whiten_weights is kinda questionable with clustered CBO'
                  ' because whitening is done globally but consensus is calculated locally.')
        self.strategy_name = "ClusteredCBO"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(
            sigma_init=self.sigma_init,
            step_size_init=self.step_size_init,
            scale_factor=self.scale_factor,
            cluster_alpha=self.cluster_alpha,
        )

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        rng, subkey_part, subkey_cluster, subkey_probs = jax.random.split(rng, 4)
        initialization = jax.random.uniform(
            subkey_part,
            (self.popsize, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        hist_names = ['kernel', 'step_dist', 'noise_dist', 'fitness_before', 'fitness_after',
                      'cluster_probs']
        info_dict = {name + '_hist': jnp.zeros(10) for name in hist_names}
        info_dict.update({name + '_bins': jnp.zeros(11) for name in hist_names})

        cluster_probs = jax.random.uniform(subkey_probs, (self.popsize, self.num_clusters))
        cluster_probs = cluster_probs / jnp.sum(cluster_probs, 1)[:,None]
        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            cluster_locs=jax.random.choice(subkey_cluster, initialization, (self.num_clusters,)),
            cluster_probs=cluster_probs,
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

        x_next, cluster_locs, cluster_probs, info_dict = update_step(state.archive, state.fitness,
                                        state.cluster_locs, state.cluster_probs,
                                        state.sigma, state.step_size, params, self.do_default_cbo,
                                        self.constant_noise, self.whiten_weights, rng)

        x_next_scaled = x_next * params.scale_factor

        new_state = state.replace(
            mean=jnp.mean(x_next,0),
            cluster_locs=cluster_locs,
            cluster_probs=cluster_probs,
            archive=x_next,
            info_dict=info_dict
        )
        
        return x_next_scaled, new_state


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
    cluster_locs: chex.Array,
    cluster_probs: chex.Array,
    sigma: float,
    step_size: float,
    params: EvoParams,
    do_default_cbo: bool, 
    constant_noise: bool, 
    whiten_weights: bool,
    rng: chex.PRNGKey
) -> Tuple[jax.Array, jax.Array, jax.Array, Dict[str, jax.Array]]:
    """
    This update step does both the update of particles towards the local mean and the addition of noise.
    Intuitively, the update should be in `tell` and the noise in `ask`, but splitting it up is a 
    lot less legible in my opinion.
    """


    info_dict = add_hist_to_info(dict(), fitness, 'fitness_before')

    rng, subkey_noise, subkey_2, subkey_3 = jax.random.split(rng, 4)
    
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
        new_cluster_locs = cluster_locs
        new_cluster_probs = cluster_probs
    else:
        kernel_fun = partial(gauss_kernel, kernel_var=params.kernel_size)
        kernels = jax.vmap(
            jax.vmap(kernel_fun, in_axes=(None, 0)),
            in_axes=(0, None))(
            x, cluster_locs,
        )  # gives [num_part, num_clust]

        # dotproduct kernel
        # x_normed = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        # cluster_locs_normed = cluster_locs / (jnp.linalg.norm(cluster_locs, axis=1, keepdims=True) + 1e-9)
        # kernels = (jnp.dot(x_normed, cluster_locs_normed.T) + 1.0) / 2.0
        # kernels = ((x_normed @ cluster_locs_normed.T) + 1.0) / 2.0

        # update cluster probabilities of ech particle
        max_probs = jnp.max(cluster_probs, 1)
        ratios = cluster_probs / max_probs[:, None]
        ratios = jnp.pow(ratios, params.cluster_alpha)
        
        kernel_prob_prod = ratios * kernels
        new_cluster_probs = kernel_prob_prod / jnp.sum(kernel_prob_prod + 1e-9, 1)[:,None]

        new_cluster_locs = []
        for idx_cluster in range(cluster_locs.shape[0]):  # TODO vectorize better to get rid of this loop
            weights = new_cluster_probs[:, idx_cluster] * jnp.exp(-params.beta * fitness)
            weights_normed = weights / (jnp.sum(weights, 0) + 1e-9)
            new_cluster = weights_normed @ x
            new_cluster_locs.append(new_cluster)
        new_cluster_locs = jnp.stack(new_cluster_locs)

        local_means = new_cluster_probs @ new_cluster_locs
        
        # if step_size=1.0 do default CBO
        global_mean_repeat = jnp.repeat(jnp.expand_dims(global_mean, axis=0), x.shape[0], axis=0)
        local_means = jnp.where(step_size == 1.0, global_mean_repeat, local_means)
        info_dict = add_hist_to_info(info_dict, kernels, 'kernel')

    # Update each towards local mean
    x_new = x + step_size * (local_means - x)

    if constant_noise:
        x_new_noisy = x_new + jax.random.normal(subkey_noise, x.shape) * sigma 
    else:
        x_new_noisy = x_new + jax.random.normal(subkey_noise, x.shape) * sigma \
            * jnp.linalg.norm(local_means - x, axis=1, keepdims=True)
    

    # if step_size=1 reinitialize clusters, to ensure clusters are started with useful inits
    new_cluster_locs = jnp.where(step_size == 1.0, 
                                jax.random.choice(subkey_2, x_new_noisy, (new_cluster_locs.shape[0],)), 
                                new_cluster_locs)
    random_cluster_probs = jax.random.uniform(subkey_3, (x.shape[0], new_cluster_locs.shape[0]))
    random_cluster_probs = random_cluster_probs / jnp.sum(random_cluster_probs, 1)[:,None]
    new_cluster_probs = jnp.where(step_size == 1.0, 
                                random_cluster_probs,
                                new_cluster_probs)

    # logging
    info_dict = add_hist_to_info(info_dict, jnp.linalg.norm(x_new - x, axis=1), 'step_dist')
    info_dict = add_hist_to_info(info_dict, jnp.linalg.norm(x_new_noisy - x_new, axis=1), 'noise_dist')
    info_dict = add_hist_to_info(info_dict, new_cluster_probs, 'cluster_probs')
    
    return x_new_noisy, new_cluster_locs, new_cluster_probs, info_dict


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
