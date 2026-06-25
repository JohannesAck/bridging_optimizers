import functools
import os
from typing import Dict, List, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np

try :
    from brax.v1.envs import create
except ImportError:
    print('Brax not installed')

from matplotlib import pyplot as plt
import optax
import tqdm

from evaluate_brax import BraxMLP, brax_eval_fn as eval_out_fn, brax_eval_fn_preproc_state_postproc_act, normalize_obs
from evosax.core.reshape import ParameterReshaper




ACTION_REPEAT_ENVS = ['hopper', 'acrobot']


def render_policy(
    args,
    policy_params,
    obs_norm_params: jnp.ndarray,
    rng: chex.PRNGKey,
    output_fp: str = 'output',
    param_name: str = "policy",
):
        from brax.v1.io import html, image
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        os.makedirs(output_fp, exist_ok=True)

        env = create(
            env_name=args.env_name,
            episode_length=args.max_steps,
            legacy_spring=True,
            action_repeat=4 if args.env_name in ACTION_REPEAT_ENVS else 1,
        )
        obs_norm_fn = functools.partial(normalize_obs, obs_shape=(env.observation_size,))
        network = BraxMLP(args.hidden_units, args.hidden_layers, env.action_size)
        pholder_params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, env.observation_size)))
        reshaper = ParameterReshaper(pholder_params)
        policy_params_reshaped = reshaper.reshape(policy_params)
        policy_params_reshaped = jax.tree.map(lambda x: x[0], policy_params_reshaped)

        step_jit = jax.jit(env.step)
        policy_jit = jax.jit(network.apply)
        obs_norm_fn = jax.jit(obs_norm_fn)


        total_reward = 0
        rollout = []

        rng, subkey = jax.random.split(rng)
        brax_state = env.reset(subkey)

        while not brax_state.done:
            original_obs = brax_state.obs
            normed_obs = obs_norm_fn(original_obs, obs_norm_params)
            action = policy_jit(policy_params_reshaped, normed_obs)
            brax_state = step_jit(brax_state, action)

            total_reward = total_reward + brax_state.reward
            rollout.append(brax_state)

        rollout = rollout[:-1]  # remove last state because it's not valid
        print("Cumulative reward:", total_reward)
        html = html.render(env.sys, [s.qp for s in rollout])  # probably should just be (env.sys, rollout)?
        html_fp = os.path.join(output_fp, param_name + f'rew_{total_reward:.0f}' + '.html')
        with open(html_fp, "w") as f:
            f.write(html)
        print(f"Saved policy rollout to {html_fp}")

        rgb_arrays = [image.render_array(env.sys, s.qp, 240, 240) for s in rollout[:100]]
        fps =  int(1 / env.sys.config.dt)
        clip = ImageSequenceClip(rgb_arrays, fps=fps)
        video_fp = os.path.join(output_fp, param_name + f'rew_{total_reward:.0f}' + '.mp4')
        clip.write_videofile(video_fp, logger=None, preset='')
        print(f"Saved policy rollout to {video_fp}")
        video_fp = os.path.join(output_fp, param_name+ f'rew_{total_reward:.0f}' + '.gif')
        clip.write_gif(video_fp, logger=None, fps=fps)
        print(f"Saved policy rollout to {video_fp}")

        positions = np.array([s.qp.pos for s in rollout])[:, :6]
        positions = np.reshape(positions, (positions.shape[0], -1))
        # plot each separately
        fig, axes = plt.subplots(positions.shape[1]// 3, 3, figsize=(10, 10))
        for i in range(positions.shape[1]):
            ax = axes[i // 3, i % 3]
            ax.plot(positions[:, i], marker='x')
        plt.tight_layout()
        fig_fp = os.path.join(output_fp, param_name + '.png')
        plt.savefig(fig_fp)
        plt.close()
        print(f"Saved policy rollout to {fig_fp}")

        # get contact rewards
        if 'reward_contact' in rollout[0].metrics:
            contact_rewards = np.array([s.metrics['reward_contact'] for s in rollout])
            plt.plot(contact_rewards, marker='x')
            contact_fp = os.path.join(output_fp, param_name + '_contact.png')
            plt.savefig(contact_fp)
            plt.xlim(50, 100)
            contact_fp = os.path.join(output_fp, param_name + '_contact_zoom.png')
            plt.savefig(contact_fp)
            plt.close()
            print(f"Saved contact rewards to {contact_fp}")


def tb_add_histogram(writer, hist_name, counts, bins, step):
    if np.isnan(bins).any():
        print(f"skipping histogram for nan in bins. bins={bins}")
        return
    if not np.all(np.diff(bins) > 0):
        print(f"skipping histogram for non-monotonic bins. bins={bins}")
        return
    raw_values = [np.ones(int(counts[i])) * 0.5 * (bins[i] + bins[i + 1]) for i in range(len(counts))]
    raw_values = np.concatenate(raw_values)
    try:
        writer.add_histogram(hist_name, raw_values, step, bins)
    except Exception as e:
        print(f"Failed to add histogram {hist_name} with shape {raw_values.shape}, bins {bins.shape} at step {step}")
        print(e)


def add_hist_to_info(info_dict: Dict[str, jax.Array], values: chex.Array, name: str
                     ) -> Dict[str, jax.Array]:
    hist, bins = jnp.histogram(values)
    info_dict[name + '_hist'] = hist
    info_dict[name + '_bins'] = bins
    return info_dict


def polynomial_schedule(
    init_value: chex.Scalar,
    power: chex.Scalar,
) -> optax.Schedule:
  """
  Polynomial schedule with no specified end value. Just decays on its own.
  """
  def schedule(count):
    return init_value * (1 + count) ** power
  return schedule


# from jax import config
# config.update("jax_debug_nans", True) #slooooooow


class TqdmUpTo(tqdm.tqdm):
    def update_to(self, b):
        delta = b - self.n
        if delta < 1:
            return
        self.update(delta)  # also sets self.n = b * bsize
    def update_and_desc(self, b, desc):
        delta = b - self.n
        if delta < 1:
            return
        self.set_description(desc)
        self.update_to(b)


eval_fns_sharpness = {}
eval_fns_sharpness_norm = {}

def evaluate_solution_sharpness(        
        args, 
        params,
        obs_params, 
        rng: chex.PRNGKey, 
        n_samples: int = 256,
        sigmas: Optional[List[float]] = None,
        normalized_noise: bool = False,
):
    global eval_fns_sharpness, eval_fns_sharpness_norm
    if sigmas is None:
        sigmas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    sigmas = jnp.array(sigmas)

    if args.env_name not in eval_fns_sharpness:
        env = create(
            env_name=args.env_name,
            episode_length=args.max_steps,
            legacy_spring=True,
            action_repeat=4 if args.env_name in ACTION_REPEAT_ENVS else 1,
        )
        normalize = functools.partial(normalize_obs, obs_shape=(env.observation_size,))
        network = BraxMLP(args.hidden_units, args.hidden_layers, env.action_size)
        eval_fn = functools.partial(eval_out_fn, env, network, args.max_steps, normalize)
        pholder_params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, env.observation_size)))
        reshaper = ParameterReshaper(pholder_params)

        n_eval = 8
        def mc_eval_fn(rng, params, obs_params, sigma):
            if not normalized_noise:
                rng, subkey = jax.random.split(rng)
                params = params + sigma * jax.random.normal(subkey, params.shape)

            param_reshaped = reshaper.reshape(params[None])
            param_reshaped = jax.tree.map(lambda x: x[0], param_reshaped)

            if normalized_noise:
                for layer_name in param_reshaped['params']:
                    rng, subkey = jax.random.split(rng)
                    kernel = param_reshaped['params'][layer_name]['kernel']
                    noise_std = sigma * jnp.std(kernel)
                    noise = jax.random.normal(subkey, kernel.shape) * noise_std
                    param_reshaped['params'][layer_name]['kernel'] = kernel + noise

            subkeys = jax.random.split(rng, n_eval)
            fitness = jax.vmap(eval_fn, in_axes=(0, None, None))(subkeys, param_reshaped, obs_params)[1]
            return jnp.mean(fitness)
        
        mc_eval_vmap = jax.vmap(mc_eval_fn, in_axes=(0, None, None, None))
        mc_eval_vmap = jax.jit(jax.vmap(mc_eval_vmap, in_axes=(None, None, None, 0)))
        if normalized_noise:
            eval_fns_sharpness_norm[args.env_name] = mc_eval_vmap
        else:
            eval_fns_sharpness[args.env_name] = mc_eval_vmap
    else:
        if normalized_noise:
            mc_eval_vmap = eval_fns_sharpness_norm[args.env_name]
        else:
            mc_eval_vmap = eval_fns_sharpness[args.env_name]

    fitnesses = []
    fitness_vars = []
    rng, subkey = jax.random.split(rng)
    subkeys = jax.random.split(subkey, n_samples)
    fitness = mc_eval_vmap(subkeys, params, obs_params, sigmas)
    fitnesses.append(fitness)
    fitness_vars.append(jnp.var(fitness))
    
    sigmas = jnp.array(sigmas)
    fitnesses = jnp.array(fitnesses)
    fitness_vars = jnp.array(fitness_vars)
    return sigmas, fitnesses, fitness_vars


eval_fns_robustness = {}
def eval_disturbance_robustness(args, params, obs_params, rng, plot_fp):
    global eval_fns_robustness
    
    env = create(
        env_name=args.env_name,
        episode_length=args.max_steps,
        legacy_spring=True,
        action_repeat=4 if args.env_name in ACTION_REPEAT_ENVS else 1,
    )
    normalize = functools.partial(normalize_obs, obs_shape=(env.observation_size,))
    network = BraxMLP(args.hidden_units, args.hidden_layers, env.action_size)
    print(f'{params.shape=}')
    
    def act_noise_fn(state, rng, act_pre_param):
        noise = jax.random.normal(rng, state.shape) * act_pre_param
        return state + noise

    def obs_noise_fn(state, rng, obs_pre_param):
        noise = jax.random.normal(rng, state.shape) * obs_pre_param
        return state + noise

    eval_fn = functools.partial(brax_eval_fn_preproc_state_postproc_act, 
                                env, 
                                network, 
                                args.max_steps, 
                                normalize,
                                preproc_state_fn=obs_noise_fn,
                                postproc_action_fn=act_noise_fn,
                                )
    pholder_params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, env.observation_size)))
    reshaper = ParameterReshaper(pholder_params)
    params_reshaped = reshaper.reshape(params)
    params_reshaped = jax.tree.map(lambda x: x[0], params_reshaped)
    n_eval = 256
    if args.env_name not in eval_fns_robustness:
        def mc_eval_fn(subkey, params_reshaped, obs_params, obs_proc_param, act_proc_param):
            subkeys = jax.random.split(subkey, n_eval)
            fitness = jax.vmap(eval_fn, in_axes=(0, None, None, None, None))(
                subkeys, params_reshaped, obs_params, obs_proc_param, act_proc_param)[1]
            return fitness

        mc_eval_fn_vmap_proc = jax.vmap(mc_eval_fn, in_axes=(None, None, None, 0, 0))
        eval_fns_robustness[args.env_name] = mc_eval_fn_vmap_proc
    else:
        mc_eval_fn_vmap_proc = eval_fns_robustness[args.env_name]
    rng, subkey = jax.random.split(rng)
    
    act_sigmas = jnp.power(10, np.linspace(-3, 0, 31))
    act_fitnesses = mc_eval_fn_vmap_proc(subkey, params_reshaped, obs_params, 
                                        jnp.zeros_like(act_sigmas), act_sigmas)

    obs_sigmas = jnp.power(10, np.linspace(-3, 0, 31))
    obs_fitnesses = mc_eval_fn_vmap_proc(subkey, params_reshaped, obs_params, 
                                         obs_sigmas, jnp.zeros_like(obs_sigmas))
    
    return act_sigmas, act_fitnesses, obs_sigmas, obs_fitnesses
        