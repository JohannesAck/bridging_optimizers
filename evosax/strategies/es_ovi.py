# Hybrid of OpenES and OVI (ES-OVI)
# this is easily doable because they both use the same sampling distribution, (isometric gaussian)
# Just the update they perform differs
# for this purpose there is a new parameter "alpha" which interpolates between the two updates
# e.g. we calculate two updates \theta_{t+1, OVI} and \theta_{t+1, ES}
# and then we do \theta_{t+1} = \alpha * \theta_{t+1, OVI} + (1 - \alpha) * \theta_{t+1, ES}
# where OVI does log-sum-exp, i.e. approximates the minimum of the samples and ES approximates the mean

from typing import Dict, Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct

from ..strategy import Strategy


@struct.dataclass
class EvoState:
    mean: chex.Array
    archive: chex.Array
    fitness: chex.Array
    sigma: float
    lrate: float
    alpha: float
    best_member: chex.Array
    info_dict: Dict[str, chex.Array]
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    sigma_init: float = 0.3
    scale_factor: float = 1.0
    beta: float = 1.0
    lrate_init: float = 0.01
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class OpenESOVI(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        scale_factor: float = 1.0,
        sigma_init: float = 0.1,
        alpha_init: float = 0.5,
        lrate_init: float = 0.01,
        opt_name: str = "sgd",
        use_quasi_mc: bool = False,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        beta: float = 1.0,
        whiten_weights: bool = False,
        only_combine_angles: bool = True,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Hybrid of OpenES and OVI

        """
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            mean_decay,
            n_devices,
            **fitness_kwargs
        )
        self.strategy_name = "OpenESOVI"
        self.use_quasi_mc = use_quasi_mc
        self.only_combine_angles = only_combine_angles

        assert opt_name == 'sgd', "Only SGD is supported for now"

        # Set core kwargs es_params
        self.scale_factor = scale_factor
        self.sigma_init = sigma_init
        self.alpha_init = alpha_init
        self.lrate_init = lrate_init
        self.beta = beta
        self.whiten_weights = whiten_weights


    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        params = EvoParams(
            sigma_init=self.sigma_init,
            scale_factor=self.scale_factor,
            beta=self.beta,
            lrate_init=self.lrate_init,
        )
        return params

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        # Initialize evolution paths & covariance matrix
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )

        archive = jnp.tile(initialization, (self.popsize, 1))
        state = EvoState(
            mean=initialization,
            archive=archive,
            fitness=jnp.zeros(self.popsize) + jnp.finfo(jnp.float32).max,
            sigma=params.sigma_init,
            lrate=self.lrate_init,
            alpha=self.alpha_init,
            best_member=initialization,
            best_fitness=jnp.finfo(jnp.float32).max,
            info_dict={
                'lambda': 0.0,
                'es_grad_norm': 0.0,
                'ovi_grad_norm': 0.0,
                'es_ovi_grad_cossim': 0.0,
                'comb_grad_norm': 0.0,
            }, # type: ignore
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        
        if self.use_quasi_mc:
            # Quasi-Monte Carlo sampling
            raise NotImplementedError("Quasi-Monte Carlo not implemented yet.")
        else:
            # Monte Carlo sampling
            noise = jax.random.normal(rng, (self.popsize, self.num_dims))
            x = state.mean + state.sigma * noise
        
        # x = jnp.clip(x, params.clip_min, params.clip_max)
        x_scaled = x * params.scale_factor       
        return x_scaled, state

    def tell_strategy(
        self,
        x: jax.Array,
        fitness: jax.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        # moment matching weighted by exponential of fitness
        x_unscaled = x / params.scale_factor
        raw_fitness = fitness
        
        # ovi update
        if self.whiten_weights:
            fitness = fitness - jnp.mean(fitness)
            beta = 1 / jnp.sqrt(jnp.var(fitness) + 1e-9)
        else:
            beta = params.beta
        weight = jnp.exp(-beta * fitness)
        weight_normed = weight / (jnp.sum(weight) + 1e-9)
        new_mean_ovi = jnp.dot(weight_normed, x_unscaled)
        ovi_grad = (new_mean_ovi - state.mean) / state.sigma

        noise = (x - state.mean) / state.sigma
        es_grad = - 1.0 / (self.popsize * state.sigma) * jnp.dot(noise.T, raw_fitness)
        
        if self.only_combine_angles:
            # only combine the angles of the gradients
            es_grad_normed = es_grad / (jnp.linalg.norm(es_grad) + 1e-9)
            ovi_grad_normed = ovi_grad / (jnp.linalg.norm(ovi_grad) + 1e-9)
            comb_grad = state.alpha * ovi_grad_normed + (1 - state.alpha) * es_grad_normed
            comb_grad_renormed = comb_grad / jnp.linalg.norm(comb_grad  + 1e-9)
            comb_grad = comb_grad_renormed * 0.5 * (jnp.linalg.norm(ovi_grad) + jnp.linalg.norm(es_grad))
        else:
            comb_grad = state.alpha * ovi_grad + (1 - state.alpha) * es_grad


        new_mean = state.mean + state.lrate * comb_grad

        es_grad_norm = jnp.linalg.norm(es_grad)
        ovi_grad_norm = jnp.linalg.norm(ovi_grad)
        comb_grad_norm = jnp.linalg.norm(comb_grad)
        es_ovi_grad_cossim = jnp.dot(es_grad, ovi_grad) / (es_grad_norm * ovi_grad_norm + 1e-9)

        new_state = state.replace(
            archive=x_unscaled,
            fitness=raw_fitness,
            mean=new_mean,
            info_dict={
                'lambda': beta,
                'es_grad_norm': es_grad_norm,
                'ovi_grad_norm': ovi_grad_norm,
                'es_ovi_grad_cossim': es_ovi_grad_cossim,
                'comb_grad_norm': comb_grad_norm,
                }
        )
        return new_state

