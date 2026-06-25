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
    best_member: chex.Array
    info_dict: Dict[str, chex.Array]
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    sigma_init: float = 0.3
    scale_factor: float = 1.0
    beta: float = 1.0
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class OptimizationViaIntegration(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        scale_factor: float = 1.0,
        sigma_init: float = 0.1,
        use_quasi_mc: bool = False,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        beta: float = 1.0,
        whiten_weights: bool = False,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Optimization by Integration
        from "Gradient-Free Optimization via Integration"
        Andrieu et al. http://arxiv.org/abs/2408.00888

        `use_quasi_mc` if True, uses quasi-Monte Carlo sampling instead of MC to estimate new mean.
        `whiten_weights` if True, weights are normalized to variance 1.
        Note that setting z_score=True achieves a similar effect, by normalizing and whitening the weights.
        """
        super().__init__(
            popsize,
            num_dims,
            pholder_params,
            mean_decay,
            n_devices,
            **fitness_kwargs
        )
        self.strategy_name = "OptByInt"
        self.use_quasi_mc = use_quasi_mc

        # Set core kwargs es_params
        self.scale_factor = scale_factor
        self.sigma_init = sigma_init
        self.beta = beta
        self.whiten_weights = whiten_weights


    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        params = EvoParams(
            sigma_init=self.sigma_init,
            scale_factor=self.scale_factor,
            beta=self.beta,
        )
        return params

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
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
            best_member=initialization,
            best_fitness=jnp.finfo(jnp.float32).max,
            info_dict={'lambda': 0.0}
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
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        # moment matching weighted by exponential of fitness
        x_unscaled = x / params.scale_factor
        raw_fitness = fitness
        if self.whiten_weights:
            fitness = fitness - jnp.mean(fitness)
            beta = 1 / jnp.sqrt(jnp.var(fitness) + 1e-9)
        else:
            beta = params.beta
        weight = jnp.exp(-beta * fitness)


        weight_normed = weight / (jnp.sum(weight) + 1e-9)
        new_mean = jnp.dot(weight_normed, x_unscaled)
        best_idx = jnp.argmin(raw_fitness)

        new_state = state.replace(
            archive=x_unscaled,
            fitness=raw_fitness,
            mean=new_mean,
            best_member=jnp.where(raw_fitness[best_idx] < state.best_fitness,
                                  x[best_idx], state.best_member),
            best_fitness=jnp.where(raw_fitness[best_idx] < state.best_fitness,
                                raw_fitness[best_idx], state.best_fitness),
            info_dict={'lambda': beta}
        )
        return new_state

