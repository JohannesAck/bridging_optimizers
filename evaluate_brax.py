"""JAX-based Brax policy and observation normalization."""

from typing import Callable, Optional, Tuple
import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import jax_utils


def default_mlp_init(scale=0.05):
    return nn.initializers.uniform(scale)


class BraxMLP(nn.Module):
    num_hidden_units: int
    num_hidden_layers: int
    out_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(
                self.num_hidden_units,
                bias_init=default_mlp_init(),
            )(x)
            if self.activation == "relu":
                x = nn.relu(x)
            elif self.activation == "tanh":
                x = nn.tanh(x)
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")
        x = nn.Dense(
            self.out_dim,
            bias_init=default_mlp_init(),
        )(x)
        x = nn.tanh(x)
        return x


def get_weight_norm(params):
    weights_list = jax.tree.flatten(params['params'])[0]  # FIXME
    weights_flat = jax.tree.map(lambda x: x.flatten(), weights_list)
    weights_vec = jnp.concatenate(weights_flat)
    weight_norm = jnp.mean(jnp.square(weights_vec))  
    return weight_norm

def brax_eval_fn(env, network, max_steps, obs_norm, rng, params, obs_params, weight_decay_scale=0.0):
    brax_state = env.reset(rng)
    valid_mask = jnp.ones((1,))
    acc_return = jnp.array([0.0])

    weight_norm = get_weight_norm(params)

    def env_step(state_input, tmp):
        brax_state, valid_mask, acc_return = state_input
        original_obs = brax_state.obs
        normed_obs = obs_norm(original_obs, obs_params)
        action = network.apply(params, normed_obs)
        brax_state = env.step(brax_state, action)
        acc_return = acc_return + brax_state.reward * valid_mask
        valid_mask = valid_mask * (1 - brax_state.done.ravel())
        # jax.debug.print(f"reward: {brax_state.reward}")
        return (brax_state, valid_mask, acc_return), (valid_mask, original_obs)

    carry_out, scan_out = jax.lax.scan(
        env_step, (brax_state, valid_mask, acc_return), (), max_steps
    )
    cum_reward = carry_out[2]
    mask_buffer = scan_out[0]
    obs_buffer = scan_out[1]

    rew_out = cum_reward.squeeze() - weight_decay_scale * weight_norm
    return rng, rew_out, mask_buffer, obs_buffer

def brax_eval_fn_preproc_state_postproc_act(
        env, 
        network, 
        max_steps, 
        obs_norm, 
        rng, 
        params, 
        obs_params,
        obs_proc_param: Optional[jax.Array] = None,
        act_proc_param: Optional[jax.Array] = None,
        preproc_state_fn: Optional[Callable[[jax.Array, chex.PRNGKey, jax.Array], jax.Array]] = None,
        postproc_action_fn: Optional[Callable[[jax.Array, chex.PRNGKey, jax.Array], jax.Array]]= None,
    ):
    if preproc_state_fn is None:
        preproc_state_fn = lambda x, rng, p: x
    if postproc_action_fn is None:
        postproc_action_fn = lambda x, rng, p: x

    brax_state = env.reset(rng)
    valid_mask = jnp.ones((1,))
    acc_return = jnp.array([0.0])

    def env_step(state_input, tmp):
        brax_state, valid_mask, acc_return, rng = state_input
        rng, subkey_obs, subkey_act = jax.random.split(rng, 3)
        original_obs = brax_state.obs
        normed_obs = obs_norm(original_obs, obs_params)
        normed_obs = preproc_state_fn(normed_obs, subkey_obs, obs_proc_param)
        action = network.apply(params, normed_obs)
        action = postproc_action_fn(action, subkey_act, act_proc_param)
        brax_state = env.step(brax_state, action)
        acc_return = acc_return + brax_state.reward * valid_mask
        valid_mask = valid_mask * (1 - brax_state.done.ravel())
        # jax.debug.print(f"reward: {brax_state.reward}")
        return (brax_state, valid_mask, acc_return, rng), (valid_mask, original_obs)

    carry_out, scan_out = jax.lax.scan(
        env_step, (brax_state, valid_mask, acc_return, rng), (), max_steps
    )
    cum_reward = carry_out[2]
    mask_buffer = scan_out[0]
    obs_buffer = scan_out[1]
    return rng, cum_reward.squeeze(), mask_buffer, obs_buffer


def normalize_obs(
    obs: jnp.ndarray,
    obs_params: jnp.ndarray,
    obs_shape: Tuple,
    clip_value: float = 5.0,
    std_min_value: float = 1e-6,
    std_max_value: float = 1e6,
) -> jnp.ndarray:
    """Normalize the given observation."""

    obs_steps = obs_params[0]
    running_mean, running_var = jnp.split(obs_params[1:], 2)
    running_mean = running_mean.reshape(obs_shape)
    running_var = running_var.reshape(obs_shape)

    variance = running_var / (obs_steps + 1.0)
    variance = jnp.clip(variance, std_min_value, std_max_value)
    return jnp.clip((obs - running_mean) / jnp.sqrt(variance), -clip_value, clip_value)


def update_obs_params(
    obs_buffer: jnp.ndarray, obs_mask: jnp.ndarray, obs_params: jnp.ndarray
) -> jnp.ndarray:
    """Update observation normalization parameters."""

    obs_steps = obs_params[0]
    running_mean, running_var = jnp.split(obs_params[1:], 2)
    if obs_mask.ndim != obs_buffer.ndim:
        obs_mask = obs_mask.reshape(
            obs_mask.shape + (1,) * (obs_buffer.ndim - obs_mask.ndim)
        )

    new_steps = jnp.sum(obs_mask)
    total_steps = obs_steps + new_steps

    input_to_old_mean = (obs_buffer - running_mean) * obs_mask
    mean_diff = jnp.sum(input_to_old_mean / total_steps, axis=(0, 1))
    new_mean = running_mean + mean_diff

    input_to_new_mean = (obs_buffer - new_mean) * obs_mask
    var_diff = jnp.sum(input_to_new_mean * input_to_old_mean, axis=(0, 1))
    new_var = running_var + var_diff

    return jnp.concatenate([jnp.ones(1) * total_steps, new_mean, new_var])


def map_update_obs_params(
    obs_buffer: jnp.ndarray, obs_mask: jnp.ndarray, obs_params: jnp.ndarray
) -> jnp.ndarray:
    obs_steps = obs_params[0]
    running_mean, running_var = jnp.split(obs_params[1:], 2)
    if obs_mask.ndim != obs_buffer.ndim:
        obs_mask = obs_mask.reshape(
            obs_mask.shape + (1,) * (obs_buffer.ndim - obs_mask.ndim)
        )

    new_steps = jnp.sum(obs_mask)
    dev_new_steps = jax.lax.psum(new_steps, "device")
    total_steps = obs_steps + dev_new_steps

    input_to_old_mean = (obs_buffer - running_mean) * obs_mask
    mean_diff = jnp.sum(input_to_old_mean / total_steps, axis=(0, 1))
    dev_mean_diff = jax.lax.psum(mean_diff, "device")
    new_mean = running_mean + dev_mean_diff

    input_to_new_mean = (obs_buffer - new_mean) * obs_mask
    var_diff = jnp.sum(input_to_new_mean * input_to_old_mean, axis=(0, 1))
    dev_var_diff = jax.lax.psum(var_diff, "device")
    new_var = running_var + dev_var_diff

    return jnp.concatenate([jnp.ones(1) * total_steps, new_mean, new_var])


def reshape_buffer(data: jnp.ndarray) -> jnp.ndarray:
    # data.shape = (#device, steps, #jobs/device, *)
    data = data.transpose([1, 0] + [i for i in range(2, data.ndim)])
    return jnp.reshape(data, (data.shape[0], data.shape[1] * data.shape[2], -1))


def gen_obs_norm_fn(norm_option):
    # 3 Norm options = 1. "global", 2. "async", 3. "sync"
    # 1. Collects data from all devices and updates params on a single device
    # 2. Collects data from one devices and updates params on each device
    # 3. Collects data from all devices and updates params on all devices
    if norm_option == "global":

        def obs_fn(obs_buffer, mask_buffer, obs_params):
            re_obs_buffer = reshape_buffer(obs_buffer)
            re_mask_buffer = reshape_buffer(mask_buffer)
            obs_params = jax_utils.unreplicate(obs_params)
            obs_params = update_obs_params(re_obs_buffer, re_mask_buffer, obs_params)
            obs_params = jax_utils.replicate(obs_params)
            return obs_params

    elif norm_option == "async":

        def obs_fn(obs_buffer, mask_buffer, obs_params):
            return jax.pmap(update_obs_params, axis_name="device")(
                obs_buffer, mask_buffer, obs_params
            )

    elif norm_option == "sync":

        def obs_fn(obs_buffer, mask_buffer, obs_params):
            return jax.pmap(map_update_obs_params, axis_name="device")(
                obs_buffer, mask_buffer, obs_params
            )

    else:
        raise ValueError(f"Invalid norm option: {norm_option}")
    return obs_fn
