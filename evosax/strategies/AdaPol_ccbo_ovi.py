"""
A combination of Clustered CBO and Optimization via Integration.
Adaptivity will probably be done similar to SaDE, i.e. track success of each method and 
adaptively choose number of particles per method.

Currently has only two strats:
OVI and Clustered CBO

The Clustered CBO particles use step-size `state.step_size`, the OVI particles use step-size=1
"""

from enum import Enum
from functools import partial
from typing import Dict, List, Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct

from utils import add_hist_to_info

from ..strategy import Strategy


@struct.dataclass
class EvoState:
    mean: jax.Array
    archive: jax.Array
    fitness: jax.Array
    particle_strategies: jax.Array
    success_rate_histories: jax.Array
    cluster_locs: chex.Array
    cluster_probs: chex.Array
    best_archive: jax.Array
    best_archive_fitness: jax.Array
    best_member: jax.Array
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


strat_enum = Enum('Strat', [('ClusteredCBO', 0 ), ('OptimizationViaIntegration', 1)])
class AdaptiveCcboOvi(Strategy):
    strat_names = strat_enum
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
        success_memory_len: int = 15,
        whiten_weights: bool = False,
        do_default_cbo: bool = False, # do default CBO update, i.e. replaces kernel with constant
        constant_noise: bool = True,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """
        Adaptive method that chooses between of Clustered CBO and OVI.

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
        self.num_clusters = num_clusters
        self.sigma_init = sigma_init
        self.step_size_init = step_size_init
        self.scale_factor = scale_factor
        self.do_default_cbo = do_default_cbo
        self.constant_noise = constant_noise
        self.whiten_weights = whiten_weights
        self.success_memory_len = success_memory_len
        if self.whiten_weights and not self.do_default_cbo:
            print('Warning: whiten_weights is kinda questionable with clustered  CBO'
                  ' because whitening is done globally but consensus is calculated locally.')
        self.strategy_name = "AdaptClusterCBO"

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
        rng, subkey_part, subkey_cluster, subkey_probs = jax.random.split(rng, 4)
        initialization = jax.random.uniform(
            subkey_part,
            (self.popsize, self.num_dims),
            minval=params.init_min,
            maxval=params.init_max,
        )
        hist_names = ['kernel', 'fitness_before', 'fitness_after']
        # hist_names = ['kernel', 'step_dist', 'noise_dist', 'fitness_before', 'fitness_after']
        info_dict = {name + '_hist': jnp.zeros(10) for name in hist_names}
        info_dict.update({name + '_bins': jnp.zeros(11) for name in hist_names})
        
        cluster_probs = jax.random.uniform(subkey_probs, (self.popsize, self.num_clusters))
        cluster_probs = cluster_probs / jnp.sum(cluster_probs, 1)[:,None]

        state = EvoState(
            mean=initialization.mean(axis=0),
            archive=initialization,
            cluster_locs=jax.random.choice(subkey_cluster, initialization, (self.num_clusters,)),
            cluster_probs=cluster_probs,
            particle_strategies=jnp.zeros(self.popsize).astype(int),
            success_rate_histories=jnp.ones((len(self.strat_names), self.success_memory_len)) / len(self.strat_names),
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

        x_next, new_state = self.update_step(state, params, self.do_default_cbo, 
                                        self.constant_noise, self.whiten_weights, rng)

        x_next_scaled = x_next * params.scale_factor
        
        return x_next_scaled, new_state

    def tell_strategy(
        self,
        x: jax.Array,
        fitness: jax.Array,
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
        self,
        state: EvoState,
        params: EvoParams,
        do_default_cbo: bool, 
        constant_noise: bool, 
        whiten_weights: bool,
        rng: chex.PRNGKey
    ) -> Tuple[jax.Array, EvoState]:
        """
        This update step does both the update of particles towards the local mean and the addition of noise.
        Intuitively, the update should be in `tell` and the noise in `ask`, but splitting it up is a 
        lot less legible in my opinion.
        """
        reset_ratio =  0.33


        x = state.archive
        fitness = state.fitness

        # update success rates
        success_rate_histories = self.update_success_rates(
            state.success_rate_histories, state.particle_strategies, fitness, 50
        )
        new_strategy_ratios = success_rate_histories.mean(axis=1)
        previous_strategy_counts = jnp.array([jnp.sum(state.particle_strategies == i ) 
                                        for i in range(len(self.strat_names))])

        # if success_rate for any strategy is zero, add some particles to it for exploration
        stagnated_strat_mask = previous_strategy_counts == 0
        new_strategy_ratios = jnp.where(stagnated_strat_mask, 
                                        jnp.sum(new_strategy_ratios) * reset_ratio, new_strategy_ratios)
        # also manipulate success rate to give time to explore
        success_rate_histories = jnp.where(stagnated_strat_mask[:, None],
                                           jnp.mean(success_rate_histories) * reset_ratio, success_rate_histories)

        # normalize ratios
        new_strategy_ratios = new_strategy_ratios / jnp.sum(new_strategy_ratios)
        new_strategy_counts = (new_strategy_ratios * self.popsize).astype(int)
        # prevent rounding issues
        new_strategy_counts = new_strategy_counts.at[-1].add(self.popsize - jnp.sum(new_strategy_counts))

        # reassign particles to different strategies
        strategy_diffs = new_strategy_counts - previous_strategy_counts
        new_strats = reassign_particles(rng, state.particle_strategies, strategy_diffs)

        # if no particles were using CBO, choose random particles as cluster locs and reset cluster probs
        rng, subkey = jax.random.split(rng)
        no_cbo = stagnated_strat_mask[0]
        random_locs = jax.random.choice(subkey, x, (self.num_clusters,), replace=False)
        cluster_locs = jax.lax.select(
            no_cbo, random_locs, state.cluster_locs
        )
        # cluster_probs = jax.lax.select(
        #     no_cbo,
        #     (1 / self.num_clusters) * jnp.ones_like(state.cluster_probs),
        #     state.cluster_probs
        # )
        rng, subkey = jax.random.split(rng)
        cluster_probs_random = jax.random.uniform(subkey, (self.popsize, self.num_clusters))
        cluster_probs_random = cluster_probs_random / jnp.sum(cluster_probs_random, 1, keepdims=True)
        cluster_probs = jax.lax.select(
            no_cbo,
            cluster_probs_random,
            state.cluster_probs
        )


        # shape fitness
        info_dict = add_hist_to_info(dict(), fitness, 'fitness_before')
        if whiten_weights:
            fitness = fitness - jnp.mean(fitness)
            fitness = fitness / jnp.sqrt(jnp.var(fitness) + 1e-9)
        
        info_dict = add_hist_to_info(info_dict, fitness, 'fitness_after')

        ignore_oviparticls_in_cbo = True
        ignore_cboparticles_in_ovi = True

        # get global mean for OVI update
        if ignore_cboparticles_in_ovi: # global mean only over OVI particles
            weight = jnp.exp(-params.beta * fitness)
            weight_masked = jnp.where(state.particle_strategies == 1, weight, 0.0)
            # but if there are no OVI particles, use all particles
            weight = jax.lax.select(jnp.sum(weight_masked) == 0, weight, weight_masked)
            weight_normed  = weight / (jnp.sum(weight) + 1e-9)
            global_mean = jnp.dot(weight_normed, x) 
        else:
            weight = jnp.exp(-params.beta * fitness)
            weight_normed  = weight / (jnp.sum(weight) + 1e-9)
            global_mean = jnp.dot(weight_normed, x)
        
        if do_default_cbo:  # default CBO instead of clustered CBO
            raise NotImplementedError()
            local_means = jnp.repeat(jnp.expand_dims(global_mean, axis=0), x.shape[0], axis=0)
        else:
            # get local means for Clustered CBO update
            kernel_fun = partial(gauss_kernel, kernel_var=params.kernel_size)
            kernels = jax.vmap(
                jax.vmap(kernel_fun, in_axes=(None, 0)),
                in_axes=(0, None))(
                x, cluster_locs,
            )  # gives [num_part, num_clust]
            max_probs = jnp.max(cluster_probs, 1)
            ratios = cluster_probs / max_probs[:, None]
            ratios = jnp.pow(ratios, params.cluster_alpha)
            
            kernel_prob_prod = ratios * (kernels + 1e-9)
            new_cluster_probs = kernel_prob_prod / jnp.sum(kernel_prob_prod, 1)[:,None]

            new_cluster_locs = []
            for idx_cluster in range(self.num_clusters):  # TODO vectorize better to get rid of this loop
                weights = new_cluster_probs[:, idx_cluster] * jnp.exp(-params.beta * fitness)
                if ignore_oviparticls_in_cbo:
                    weights_masked_old = jnp.where(state.particle_strategies == 0, weights, 0.0)
                    weights_masked_new = jnp.where(new_strats == 0, weights, 0.0)
                    # after resetting CBO, we need to mask based on the new particles or everything is zero.
                    weights = jax.lax.select(jnp.sum(weights_masked_old) == 0, weights_masked_new, weights_masked_old)
                weights_normed = weights / (jnp.sum(weights, 0) + 1e-9)
                new_cluster = weights_normed @ x
                new_cluster_locs.append(new_cluster)
            new_cluster_locs = jnp.stack(new_cluster_locs)

            local_means = new_cluster_probs @ new_cluster_locs

            info_dict = add_hist_to_info(info_dict, kernels, 'kernel')

        # Update according to each strategy
        rng, subkey = jax.random.split(rng)
        subkeys = jax.random.split(subkey, self.popsize)
        def cbo_update(x, part_idx, subkey):
            x_new = x + state.step_size * (local_means[part_idx] - x)
            if constant_noise:
                x_new_noisy = x_new + jax.random.normal(subkey, x.shape) * state.sigma 
            else:
                x_new_noisy = x_new + jax.random.normal(subkey, x.shape) * state.sigma * \
                    jnp.linalg.norm(local_means[part_idx] - x)
            return x_new_noisy
        x_cbo = jax.vmap(cbo_update)(x, jnp.arange(self.popsize), subkeys)

        def ovi_update(x, part_idx, subkey):
            x_new = global_mean
            x_new_noisy = x_new + jax.random.normal(subkey, x.shape) * state.sigma 
            return x_new_noisy
        x_ovi = jax.vmap(ovi_update)(x, jnp.arange(self.popsize), subkeys)

        selector = jnp.repeat(new_strats[:, None], x_cbo.shape[1], 1)
        x_new_noisy = jax.lax.select_n(selector, x_cbo, x_ovi)

        # reset cluster probs for particles that are not using CBO
        # new_cluster_probs = jnp.where((new_strats == 1)[:, None], 
        #                               (1 / self.num_clusters) * jnp.ones_like(new_cluster_probs),
        #                               new_cluster_probs)
        

        # info_dict = add_hist_to_info(info_dict, jnp.linalg.norm(x_new - x, axis=1), 'step_dist')
        # info_dict = add_hist_to_info(info_dict, jnp.linalg.norm(x_new_noisy - x_new, axis=1), 'noise_dist')

        new_state = state.replace(
            archive=x_new_noisy,
            mean=x_new_noisy.mean(axis=0),
            particle_strategies=new_strats,
            success_rate_histories=success_rate_histories,
            info_dict=info_dict,
            cluster_locs=new_cluster_locs,
            cluster_probs=new_cluster_probs,
        )
        
        return x_new_noisy, new_state

    def update_success_rates(
            self, 
            success_rate_histories: jax.Array, 
            used_strategies: jax.Array, 
            fitness: jax.Array,
            success_percentile: int = 20
        ) -> jax.Array:
        success_rate_histories = jnp.roll(success_rate_histories, 1, axis=1)

        particle_in_top = jnp.where(fitness <= jnp.percentile(fitness, success_percentile), 1, 0)
        for idx in range(len(self.strat_names)):
            successes = jnp.sum(particle_in_top, where=used_strategies == idx)
            total = jnp.sum(used_strategies == idx)
            new_success_rate = successes / total
            success_rate_histories = success_rate_histories.at[idx, 0].set(new_success_rate)
        success_rate_histories = jnp.where(jnp.isnan(success_rate_histories), 0, success_rate_histories)

        return success_rate_histories


def reassign_particles(rng, old_assignments, delta):
    """
    Reassign particles to different strategies, based on the given delta.

    `old_assignments` : array[int](pop_size,): representing assignment of each particle to a strategy
    `delta` : array[int](n_strategy,): represents desired change in particles per strategy

    e.g. delta=[-10 10] will take 10 random particles from strategy 0 and add them to strategy 1
    """
    n_strats = len(delta)

    transport_plan = get_transport(delta)

    new_assignments = old_assignments
    for idx_from in range(n_strats): 
        # these loops could be scanned but should be fine for now?
        for idx_to in range(n_strats):  
            rng, subkey = jax.random.split(rng)
            new_assignments = update_n_random_matches(
                subkey, new_assignments, 
                transport_plan[idx_from, idx_to],
                idx_from, idx_to
            )
    return new_assignments


def get_transport(delta):
    """
    Returns a transport map of the shape transport[from][to] = count that achieves the deltas in the input.
    Example:
    delta = [ 20  10 -20 -10]
    output:
    [[ 0,  0,  0,  0],
     [ 0,  0,  0,  0],
     [20,  0,  0,  0],
     [ 0, 10,  0,  0]]
     
     This plan can then be executed with the reassign_particles method above.
    """
    n_strats = delta.shape[0]

    def scan_fn(carry, x):
        remaining: jax.Array = carry
        idx_giver = x
        giver_remaining = -jnp.minimum(remaining[idx_giver], 0)

        output = jnp.zeros(n_strats, int)
        for idx_rec in range(n_strats):
            rec_needs = jnp.maximum(remaining[idx_rec], 0)
            n_transfer = jnp.minimum(giver_remaining, rec_needs)
            giver_remaining -= n_transfer
            output = output.at[idx_rec].set(n_transfer)
            remaining = remaining.at[idx_rec].subtract(n_transfer)
            remaining = remaining.at[idx_giver].add(n_transfer)
        return remaining, output

    remainder, plan = jax.lax.scan(scan_fn, delta, jnp.arange(n_strats))
    return plan


def update_n_random_matches(rng, array, num_flip, before_val, after_val):
    """
    Update n randomly chosen elements of array that are `before_val` to `after_val
    """
    match_mask = (array == before_val)    

    scores = jax.random.uniform(rng, shape=(array.shape[0],))
    target_scores = jnp.where(match_mask, scores, jnp.inf)
    sorted_scores = jnp.sort(target_scores)
    threshold = sorted_scores[num_flip]
    change_mask = target_scores < threshold

    return jnp.where(change_mask, after_val, array)


def get_local_weighted_mean(
    particle_locs: chex.Array,
    fitness: chex.Array,
    particle_id: int,
    beta: float,
    kernel_var: float,
    mask: jax.Array,
) -> jax.Array:
    """
    Where mask=False, the particles will be ignored
    """
    x = particle_locs[particle_id]
    kernels = jax.vmap(gauss_kernel, in_axes=(None, 0, None))(x, particle_locs, kernel_var)

    weights = kernels * jnp.exp(-beta * fitness)
    weights = jnp.where(mask, weights, 0.0)
    weights_normed = weights / (jnp.sum(weights) + 1e-9)
    weighted_mean = weights_normed @ particle_locs
    return weighted_mean, kernels


def gauss_kernel(x: chex.Array, y: chex.Array, kernel_var: float) -> float:
    return jnp.exp(-jnp.sum(jnp.square(x-y)) / (kernel_var + 1e-9))  # TODO check kernel correct
