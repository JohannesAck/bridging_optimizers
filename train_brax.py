# import os
# flags = os.environ.get('XLA_FLAGS', '')
# os.environ['XLA_FLAGS'] = flags + " --xla_force_host_platform_device_count=2"

import datetime
import functools
import itertools
import os
from typing import Any, Callable, Dict
from dataclasses import asdict, dataclass
import logging
import uuid

import tyro
import einops
import numpy as np
import jax
import jax.numpy as jnp
import optax
import tqdm
from tensorboardX import SummaryWriter
from brax.v1.envs import create

from evosax import DiffusionEvolution, PolarizedCBO, OpenES, EvoParams, Strategy, \
                   OptimizationViaIntegration, Sep_CMA_ES, CMA_ES, ClusteredCBO
from evosax.strategies.AdaPol_ccbo_ovi import AdaptiveCcboOvi
from evaluate_brax import brax_eval_fn
from evaluate_brax import BraxMLP, normalize_obs, update_obs_params, reshape_buffer
from evosax.strategies.AdaPol_ccbo_ovi import AdaptiveCcboOvi
from evosax.strategies.es_ovi import OpenESOVI
from utils import ACTION_REPEAT_ENVS, TqdmUpTo, render_policy, tb_add_histogram


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class Config:
    exp_name: str = "exp"
    # stratname: str = "AdaCCBOOVI"

    # stratname: str = "OptimizationViaIntegration"
    # stratname: str = "Sep_CMA_ES"
    # stratname: str = "CBO"
    # stratname: str = "PolarizedCBO"
    # stratname: str = "ClusteredCBO"
    stratname: str = "OpenESSGD"

    # stratname: str = "OpenESOVI"
    # stratname: str = "DiffusionEvolution"

    num_generations: int = 500

    # env_name: str = "ant"
    # env_name: str = "hopper"
    # max_steps: int = 1000

    env_name: str = "acrobot"
    max_steps: int = 500

    # env_name: str = "inverted_pendulum"
    # max_steps: int = 100

    mc_evals: int = 8
    popsize: int = 256

    # fitness transformation
    centered_rank: bool = False
    z_score: bool = True
    
    # Diffevo
    fitness_map_temp: float = 1.0

    # OVI/CBO specific
    whiten_weights: bool = False
    kernel_size: float = 10.0
    step_size_init: float = 1.0  
    # step_size_init this is used inconsistently in config but the way it is now
    # is how I ran the hyperparameter-optimization. Change if future experiments

    # OpenESOVI-hybrid
    ovies_alpha: float = 0.75
    only_combine_angles_openesovi: bool = True

    lr: float = 1.0

    hidden_layers: int = 2
    hidden_units: int = 32

    sigma: float = 0.3
    decay_sigma: bool = False
    decay_lr: bool = False          # for OpenES methods
    final_sigma_ratio: float = 0.05
    final_lr_ratio: float = 0.05

    do_hparam_sweep: bool = False

    weight_decay: float = 0.0
    seeds_per_config: int = 10

    seed: int = 1
    num_devices: int = 1



def get_brax_evo_fn(
    env_name,
    strat_partial: Callable[..., Strategy],
    evoparam_updates: Dict[str, Any],
    log_cb: Callable[[int, Dict[str, jax.Array], Dict[str, jax.Array]], None],
    max_steps: int,
    mc_evals: int,
    schedules: Dict[str, optax.Schedule],
    hidden_layers: int,
    hidden_units: int,
    num_generations: int,
    weight_decay: float
):
    env = create(
        env_name=env_name,
        episode_length=max_steps,
        legacy_spring=True,
        action_repeat=4 if env_name in ACTION_REPEAT_ENVS else 1,
    )
    num_obs_dims = int(jnp.prod(jnp.array(env.observation_size)))
    normalize = functools.partial(normalize_obs, obs_shape=(env.observation_size,))
    network = BraxMLP(hidden_units, hidden_layers, env.action_size)
    pholder_params = network.init(
        jax.random.PRNGKey(0), jnp.zeros((1, env.observation_size))
    )
    eval_fn = functools.partial(brax_eval_fn, env, network, max_steps, normalize, weight_decay_scale=weight_decay)
    mc_batch_eval = jax.vmap(eval_fn, in_axes=(0, None, None))
    pop_mc_batch_eval = jax.vmap(mc_batch_eval, in_axes=(None, 0, None))

    def brax_evo_fn(rng, param_update_dict):
        strategy = strat_partial(pholder_params=pholder_params)
        evo_params = strategy.default_params.replace(
            **evoparam_updates,
        )
        evo_params = evo_params.replace(**param_update_dict)
        obs_params = jnp.zeros(1 + num_obs_dims * 2)

        rng, init_rng = jax.random.split(rng)
        evo_state = strategy.initialize(init_rng, evo_params)

        def body_fn(carry, _):
            rng, evo_state, obs_params = carry
            rng, ask_rng, eval_rng = jax.random.split(rng, 3)
            mc_eval_rng = jax.random.split(eval_rng, mc_evals)
            x, evo_state = strategy.ask(ask_rng, evo_state, evo_params)
            _, fitness, mask_buffer, obs_buffer = pop_mc_batch_eval(
                mc_eval_rng, x, obs_params
            )
            mc_mean_fitness = jnp.mean(fitness, axis=1)
            old_mean_param = evo_state.mean
            evo_state = strategy.tell(x, mc_mean_fitness, evo_state, evo_params)
            re_obs_buffer = reshape_buffer(obs_buffer)
            re_mask_buffer = reshape_buffer(mask_buffer)
            new_obs_params = update_obs_params(re_obs_buffer, re_mask_buffer, obs_params)
            mean_fitness = jnp.mean(fitness)
            max_fitness = jnp.max(mc_mean_fitness)
            log_dict = {
                "mean_fitness": mean_fitness,
                "max_fitness": max_fitness,
                "weight_mean": jnp.mean(evo_state.mean),
                "weight_std": jnp.std(evo_state.mean),
                "weight_mean_square": jnp.mean(evo_state.mean ** 2),
                "weight_abs_mean": jnp.mean(jnp.abs(evo_state.mean)),
            }
            # TODO move the following inside the strategies
            if hasattr(evo_state, "cluster_locs"):
                hard_assignments = jnp.argmin(evo_state.cluster_probs, axis=1)
                for i in range(strategy.num_clusters):
                    cluster_fitness = jnp.sum(mc_mean_fitness, where=hard_assignments == i) / (jnp.sum(hard_assignments == i) + 1e-6)
                    cluster_ratio = jnp.mean(hard_assignments == i)
                    log_dict[f'cluster_{i}_fitness'] = cluster_fitness
                    log_dict[f'cluster_{i}_ratio'] = cluster_ratio
            if hasattr(evo_state, 'particle_strategies'):
                for idx, strat_name in enumerate(strategy.strat_names):
                    log_dict[f'stratratio_{strat_name}'] = \
                        jnp.mean(evo_state.particle_strategies == idx)
            for pname, schedule in schedules.items():
                current_p_val = getattr(evo_state, pname)
                init_val = getattr(evo_params, pname + '_init')
                new_p_val = init_val * schedule(evo_state.gen_counter)
                evo_state = evo_state.replace(**{pname: jnp.ones_like(current_p_val) * new_p_val})
                log_dict[pname] = new_p_val

            if hasattr(evo_state, "info_dict"):
                log_dict |= evo_state.info_dict
            jax.experimental.io_callback(log_cb, (), evo_state.gen_counter, log_dict, ordered=False)
            
            best_flat = strategy.param_reshaper.flatten(x)[jnp.argmax(mc_mean_fitness)]
            if hasattr(evo_state, "archive"):
                mean_best_member_obs = (old_mean_param, best_flat, obs_params)
            else:
                mean_best_member_obs = (old_mean_param, best_flat, obs_params)

            return (rng, evo_state, new_obs_params,), (
                mean_fitness,
                max_fitness,
                log_dict,
                mean_best_member_obs,
            )

        rng, eval_rng = jax.random.split(rng)
        (final_state, fitness_stats) = jax.lax.scan(
            body_fn,
            (rng, evo_state, obs_params),
            None,
            length=num_generations,
        )
        _, evo_state, obs_params = final_state
        mean_fitness_history, max_fitness_history, log_dicts, best_member_obs = fitness_stats
        best_member_obs = jax.tree.map(lambda x: x[::10], best_member_obs)
        return mean_fitness_history, max_fitness_history, evo_state, obs_params, evo_params, log_dicts, best_member_obs

    return brax_evo_fn

# @jax.disable_jit()
def main(args: Config, summary_writer, rng):
    strat_partial, schedules, hopt_names, hopt_vals, param_updates = get_strategy(args)

    t = TqdmUpTo(total=args.num_generations, desc=f'{args.stratname} Generations', 
                 leave=True, dynamic_ncols=True)

    def log_cb(gen, log_dict):
        gen = gen.item()
        if gen % 10 != 0:
            return
        mean_rew, max_rew = log_dict["mean_fitness"].item(), log_dict["max_fitness"].item()
        hist_names = [n.removesuffix('_hist') for n in log_dict.keys() if n.endswith('_hist')]
        for hist_name, bins_name in [(n + '_hist', n + '_bins') for n in hist_names]:
            tb_add_histogram(summary_writer, f"live/{hist_name}", log_dict[hist_name], log_dict[bins_name], gen)
            del log_dict[hist_name], log_dict[bins_name]
        for k, v in log_dict.items():
            summary_writer.add_scalar(f"live/{k}", v.item(), gen)
        desc = f'Mean: {mean_rew:.2f}, Max: {max_rew:.2f} W STD: {log_dict["weight_std"]:+.3f}'
        t.update_and_desc(gen, desc)


    brax_evo_fn = get_brax_evo_fn(
        env_name=args.env_name,
        strat_partial=strat_partial,
        evoparam_updates=param_updates,
        log_cb=log_cb, # type: ignore
        max_steps=args.max_steps,
        mc_evals=args.mc_evals,
        schedules=schedules,
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        num_generations=args.num_generations,
        weight_decay=args.weight_decay,
    )

    # for hyperparameter optimization
    param_update_dicts = {
        pname: jnp.array([pval[p_idx] for pval in hopt_vals]) for p_idx, pname in enumerate(hopt_names)  # check this 
    }
    n_configs = len(hopt_vals)
    if not param_update_dicts:
        hopt_names, hopt_vals = zip(*param_updates.items())
        param_update_dicts = {k: jnp.array([v]) for k, v in param_updates.items()}
        n_configs = 1

    n_runs = n_configs * args.seeds_per_config
    print(f'Total number of configs: {n_configs}, with {args.seeds_per_config} seeds each')
    print(f'Total number of configs: {n_configs}, with {args.seeds_per_config} seeds each')

    def split_key(key, num_splits):
        return jax.random.split(key, num_splits)

    vmap_keys = split_key(rng, args.seeds_per_config)

    jit_eval_fn = jax.jit(brax_evo_fn)
    vmap_eval_fn = jax.vmap(jit_eval_fn, in_axes=(None, 0))
    if len(jax.devices()) > 1:
        pmap_eval_fn = jax.pmap(vmap_eval_fn, in_axes=(0, None))
    else:
        pmap_eval_fn = jax.vmap(vmap_eval_fn, in_axes=(0, None))
    

    # with jax.disable_jit():
    mean_fitness_history, max_fitness_history, evo_state, obs_params, evo_params, log_dicts, mean_best_obs_hist = \
        pmap_eval_fn(
            vmap_keys, param_update_dicts,
    )

        

    # log results
    # 1) write to tensorboard
    if len(mean_fitness_history.shape) == 2:
        mean_fitness_history = mean_fitness_history[None]
        max_fitness_history = max_fitness_history[None]

    log_dicts_raw = log_dicts
    log_dicts = jax.tree.map(lambda x: np.array(x), log_dicts)
    log_dicts = jax.tree.map(lambda x: einops.rearrange(x, "d b ... -> (d b) ..."), log_dicts)

    best_run_idx = np.argmax(log_dicts["mean_fitness"][:,-1])

    hist_keys = [n.removesuffix('_hist') for n in log_dicts.keys() if n.endswith('_hist')]
    no_hist_keys = [n for n in log_dicts.keys() if not (n.endswith('_hist') or n.endswith('_bins'))]
    for idx_t in tqdm.trange(args.num_generations, desc="Writing to tensorboard"):
        for run_idx in range(n_runs):
            for key in hist_keys:
                tb_add_histogram(summary_writer, f"run{run_idx}/{key}_hist", 
                                 log_dicts[key + '_hist'][run_idx][idx_t],
                                 log_dicts[key + '_bins'][run_idx][idx_t], idx_t)
            for key in no_hist_keys:
                summary_writer.add_scalar(f"run{run_idx}/{key}", 
                                          log_dicts[key][run_idx][idx_t], idx_t)
        # means
        for key in no_hist_keys:
            summary_writer.add_scalar(f"mean/{key}", 
                                      np.mean(log_dicts[key][:,idx_t]), idx_t)
        for key in no_hist_keys:
            summary_writer.add_scalar(f"best_mean_run/{key}", 
                                      log_dicts[key][best_run_idx, idx_t], idx_t)

    config_final_mean_mean = mean_fitness_history.mean(axis=0)[:,-1]
    config_final_max_mean = max_fitness_history.mean(axis=0)[:,-1]
    mean_fitness_history_raw = mean_fitness_history.copy()
    max_fitness_history_raw = max_fitness_history.copy()
    mean_fitness_history = einops.rearrange(mean_fitness_history, "d b t -> (d b) t")
    max_fitness_history = einops.rearrange(max_fitness_history, "d b t -> (d b) t")

    summary_writer.close()

    # 2) log results with different configs.
    for config_idx in range(n_configs):
        config_str = " ".join([f"{k}: {v[config_idx]:.4f}" for k, v in param_update_dicts.items()])
        print(config_str, f'Final Mean: {config_final_mean_mean[config_idx]:.2f}, Final Max: {config_final_max_mean[config_idx]:.2f}')

    print('Top 5 configs by mean:')
    top5_idx = np.argsort(config_final_mean_mean)[-5:]
    for config_idx in top5_idx[::-1]:
        config_str =  " ".join([f"{k}: {v[config_idx]:.4f}" for k, v in param_update_dicts.items()])
        print(config_str, f'Final Mean: {config_final_mean_mean[config_idx]:.2f}, Final Max: {config_final_max_mean[config_idx]:.2f}')

    print('Top 5 configs by max:')
    top5_idx = np.argsort(config_final_max_mean)[-5:]
    for config_idx in top5_idx[::-1]:
        config_str =  " ".join([f"{k}: {v[config_idx]:.4f}" for k, v in param_update_dicts.items()])
        print(config_str, f'Final Mean: {config_final_mean_mean[config_idx]:.2f}, Final Max: {config_final_max_mean[config_idx]:.2f}')

    # 3) save evo param
    evo_param_rearange = jax.tree.map(lambda x: einops.rearrange(x, "d b ... -> (d b) ..."), evo_params)
    fp = os.path.join(summary_writer.logdir, "evo_params.txt")
    with open(fp, "w") as f:
        for run_idx in range(n_runs):
            conf = asdict(jax.tree.map(lambda x: x[run_idx].item(), evo_param_rearange))
            if 'opt_params' in conf:
                for k, v in conf['opt_params'].items():
                    if v is None:
                        continue
                    conf['opt_params/' + k] = v
                del conf['opt_params']
            config_str = f"Run {run_idx}" + " ".join([f"{k}: {v:.5f}" for k, v in conf.items()])
            f.write(config_str)
            f.write(f' Final Mean: {mean_fitness_history[run_idx][-1]:.2f}, Final Max: {max_fitness_history[run_idx][-1]:.2f}\n')
            f.write("\n")
    print(f"Saved evo params to {fp}")
    t.close()

    # 4) save checkpoints
    for config_idx in range(n_configs):
        for seed_idx in range(args.seeds_per_config):
            checkpoint_dict = {
                "evo_state": jax.tree.map(lambda x: x[seed_idx, config_idx], evo_state),
                "obs_params": jax.tree.map(lambda x: x[seed_idx, config_idx], obs_params),
                "mean_best_obs_hist": jax.tree.map(lambda x: x[seed_idx, config_idx], mean_best_obs_hist),
                "log_dicts": jax.tree.map(lambda x: x[seed_idx, config_idx], log_dicts_raw),
                "evo_params": asdict(jax.tree.map(lambda x: x[seed_idx, config_idx].item(), evo_params)),
                "mean_fitness_history": mean_fitness_history_raw[seed_idx, config_idx],
                "max_fitness_history": max_fitness_history_raw[seed_idx, config_idx],
            }
            make_nparray_if_jax_arry = lambda x: np.array(x) if isinstance(x, jax.Array) else x
            checkpoint_dict = jax.tree.map(make_nparray_if_jax_arry, checkpoint_dict)
            checkpoint_dict['args'] = asdict(args)

            checkpoint_fp = os.path.join(summary_writer.logdir, f"checkpoint_conf{config_idx}_seed{seed_idx}.npy")
            np.save(checkpoint_fp, checkpoint_dict)
            print(f"Saved checkpoint to {checkpoint_fp}")


    
    # 5) render results (sloooow)
    for run_idx in range(min(n_runs, 2)):
        evo_state_curr = jax.tree.map(lambda x: x[run_idx], evo_state)
        obs_params_curr = jax.tree.map(lambda x: x[run_idx], obs_params)
        run_logdir = os.path.join(summary_writer.logdir, str(run_idx))
        os.makedirs(run_logdir, exist_ok=True)


        # render videos of policies
        rng, subkey = jax.random.split(rng)
        render_policy(args, evo_state_curr.mean, obs_params_curr[0], subkey, run_logdir, param_name="final_mean")
        render_policy(args, evo_state_curr.best_member, obs_params_curr[0], subkey, run_logdir, param_name="final_best")
        if hasattr(evo_state, "archive"):
            n_archive = evo_state_curr.archive.shape[1]
            for i in range(min(n_archive, 2)):
                render_policy(args, evo_state_curr.archive[:,i], obs_params_curr[0], subkey, run_logdir, param_name=f"archive_{i}")


        if hasattr(evo_state, "cluster_locs"):
            # pairwise interpolate weights
            for i in range(evo_state_curr.cluster_locs.shape[1]):
                render_policy(args, evo_state_curr.cluster_locs[:,i], obs_params_curr[0], subkey, run_logdir, param_name=f"cluster_{i}")


    print(f"Results saved to {summary_writer.logdir}")

def get_strategy(args: Config):
    param_names = []
    param_vals = []
    schedules = {}

    if args.stratname == "OpenESSGD":
        strat_partial = functools.partial(OpenES,
            popsize=args.popsize,
            maximize=True,
            z_score=args.z_score,
            centered_rank=args.centered_rank,  
            lrate_init=args.lr,
            lrate_decay=1.0,
            lrate_limit=0.001,
            opt_name="sgd",
            use_antithetic_sampling=False
        )
        param_updates = {
            "sigma_init": args.sigma,
            "sigma_decay": 1.0,
            "sigma_limit": 0.00
        }
        if args.decay_sigma:
            schedules['sigma'] = optax.schedules.linear_schedule(1.0, args.final_sigma_ratio, args.num_generations)
        if args.decay_lr:
            schedules['lrate'] = optax.schedules.linear_schedule(1.0, args.final_lr_ratio, args.num_generations)
        if args.do_hparam_sweep:
            param_names = ['sigma_init', 'lrate_init']
            param_vals = list(itertools.product([1e-2, 3e-2, 1e-1, 3e-1], [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]))
    elif args.stratname == "Sep_CMA_ES":
        strat_partial = functools.partial(Sep_CMA_ES,
            popsize=args.popsize,
            maximize=True,
            centered_rank=args.centered_rank,
            z_score=args.z_score,
        )
        param_updates = {
            "sigma_init": args.sigma,
        }
        if args.do_hparam_sweep:
            param_names = ['sigma_init']
            param_vals = np.array([1e-2, 3e-2, 1e-1, 3e-1, 1e0])[:, None]

    elif args.stratname == "DiffusionEvolution":
        strat_partial = functools.partial(DiffusionEvolution,
            popsize=args.popsize,
            maximize=True,
            n_devices=1,
            num_generations=args.num_generations,
            fitness_mapping="energy",
            alpha_schedule="cosine",
            num_latent_dims=16,
            centered_rank=args.centered_rank,
            z_score=args.z_score,
        )
        param_updates = {
            "init_scale": 0.05,
            'fitness_map_temp': args.fitness_map_temp,
            'sigma_init': args.sigma,
        }
        if args.do_hparam_sweep:
            param_names = ['sigma_init', 'fitness_map_temp']
            param_vals = list(itertools.product([1e-2, 3e-2, 1e-1, 3e-1], [0.6, 0.7, 0.8, 0.9, 1.0]))
    elif args.stratname == "CBO":
        strat_partial = functools.partial(PolarizedCBO,
            popsize=args.popsize,
            maximize=True,
            n_devices=1,
            do_default_cbo=True,
            centered_rank=args.centered_rank,
            constant_noise=True,
            whiten_weights=args.whiten_weights,
            z_score=args.z_score,
        )
        param_updates = {
            'scale_factor': 1.0,
            'sigma_init': args.sigma,
            'step_size_init': args.step_size_init,
            'init_min': 0.0,
            'init_max': 0.0,
        }
        # n_optviaint_gens = args.num_generations
        n_optviaint_gens = 0
        # n_optviaint_gens = 3 * args.num_generations // 4
        n_cbo_gens = args.num_generations - n_optviaint_gens
        schedules = {
            'step_size': optax.schedules.constant_schedule(0.3),
        }
        if args.decay_sigma:
            schedules['sigma'] =  optax.schedules.linear_schedule(1.0, 0.0, args.num_generations)
        if args.do_hparam_sweep:
            param_names = ['sigma_init', 'step_size_init']
            param_vals = list(itertools.product([1e-2, 3e-2, 1e-1, 3e-1], [1e-1, 2e-1, 3e-1, 5e-1]))
    elif args.stratname == "PolarizedCBO":
        strat_partial = functools.partial(PolarizedCBO,
            popsize=args.popsize,
            maximize=True,
            n_devices=1,
            do_default_cbo=False,
            centered_rank=args.centered_rank,
            constant_noise=True,
            z_score=args.z_score,
            whiten_weights=args.whiten_weights
        )
        param_updates = {
            'scale_factor': 1.0,
            'sigma_init': args.sigma,
            'init_min': 0.0,
            'kernel_size': args.kernel_size,
        }
        # n_optviaint_gens = args.num_generations
        # n_optviaint_gens = 0
        n_optviaint_gens = 0
        n_cbo_gens = args.num_generations - n_optviaint_gens
        schedules = {
            'step_size': optax.schedules.constant_schedule(0.5)
        }
        if args.decay_sigma:
            schedules['sigma'] = optax.schedules.linear_schedule(1.0, 0.0, n_optviaint_gens)
        if args.do_hparam_sweep:
            param_names = ['sigma_init', 'kernel_size']
            param_vals = list(itertools.product([1e-2, 3e-2, 1e-1, 3e-1], [1e-1, 1e0, 1e1, 1e2]))
    elif args.stratname == "ClusteredCBO":
        strat_partial = functools.partial(ClusteredCBO,
            popsize=args.popsize,
            num_clusters=4,
            maximize=True,
            n_devices=1,
            do_default_cbo=False,
            centered_rank=args.centered_rank,
            constant_noise=True,
            whiten_weights=args.whiten_weights,
            z_score=args.z_score,
        )
        param_updates = {
            'scale_factor': 1.0,
            'sigma_init': args.sigma,
            'step_size_init': args.step_size_init,
            'init_min': 0.0,
            'init_max': 0.0,
            'kernel_size': args.kernel_size,
        }
        n_optviaint_gens = args.num_generations // 2
        schedules = {
            'step_size': optax.schedules.constant_schedule(0.3)
        }
        if args.decay_sigma:
            schedules['sigma'] = optax.schedules.linear_schedule(1.0, 0.0, n_optviaint_gens)
        if args.do_hparam_sweep:
            param_names = ['sigma_init', 'kernel_size']
            param_vals = list(itertools.product([1e-2, 3e-2, 1e-1, 3e-1], [1e-1, 1e0, 1e1, 1e2]))
    elif args.stratname == "OptimizationViaIntegration":
        strat_partial = functools.partial(OptimizationViaIntegration,
            popsize=args.popsize,
            maximize=True,
            centered_rank=args.centered_rank,
            z_score=args.z_score,
            whiten_weights=args.whiten_weights
        )
        schedules = {
            'sigma': optax.schedules.linear_schedule(1.0, 0.0, args.num_generations),
        }
        param_updates = {
            'scale_factor': 1.0,
            'sigma_init': args.sigma,
            # 'sigma_init': 0.03,
        }
        if args.do_hparam_sweep:
            param_names = ['sigma_init']
            param_vals = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0])[:, None]
    elif args.stratname == "OpenESOVI":
        strat_partial = functools.partial(OpenESOVI,
            popsize=args.popsize,
            maximize=True,
            centered_rank=args.centered_rank,
            z_score=args.z_score,
            whiten_weights=args.whiten_weights,
            alpha_init=args.ovies_alpha,
            lrate_init=args.lr,
            only_combine_angles=args.only_combine_angles_openesovi,
        )
        param_updates = {
            'scale_factor': 1.0,
            'sigma_init': args.sigma,
        }
        if args.decay_sigma:
            schedules['sigma'] = optax.schedules.linear_schedule(1.0, args.final_sigma_ratio, args.num_generations)
        if args.decay_lr:
            schedules['lrate'] = optax.schedules.linear_schedule(1.0, args.final_lr_ratio, args.num_generations)
        if args.do_hparam_sweep:
            param_names = ['sigma_init', 'lrate_init']
            param_vals = list(itertools.product([1e-2, 3e-2, 1e-1, 3e-1, 1e0], [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]))
    elif args.stratname == "AdaCCBOOVI":
        strat_partial = functools.partial(AdaptiveCcboOvi,
            popsize=args.popsize,
            maximize=True,
            centered_rank=args.centered_rank,
            z_score=args.z_score,
            whiten_weights=args.whiten_weights,
            success_memory_len=100,
            step_size_init=0.1
        )
        constant_gens = 3 * args.num_generations // 4
        anneal_gens = args.num_generations - constant_gens

        param_updates = {
            'scale_factor': 1.0,
            'sigma_init': args.sigma,
            'kernel_size': args.kernel_size,
        }

        if args.decay_sigma:
            schedules = {
                'sigma': optax.schedules.linear_schedule(1.0, 0.0, args.num_generations),
            }
        if args.do_hparam_sweep:
            param_names = ['sigma_init']
            param_vals = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])[:, None]
    else:
        raise ValueError(f"Unknown strategy name: {args.stratname}")
    return strat_partial, schedules, param_names, param_vals, param_updates
            

if __name__ == "__main__":
    from zoneinfo import ZoneInfo
    args = tyro.cli(Config)

    rng = jax.random.PRNGKey(args.seed)
    current_time = datetime.datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d-%H%M%S")
    current_date = datetime.datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")
    uuid = str(uuid.uuid4())[:6]
    run_name = f"{args.env_name}_{args.stratname}_{args.popsize}_{args.num_generations}_{current_time}_{args.exp_name}_{uuid}"
    print(f"Starting run {run_name}")
    
    log_dir = os.path.join("output", current_date, run_name)
    os.makedirs(log_dir, exist_ok=True)
    # dump config
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(str(args))
    summary_writer = SummaryWriter(log_dir)
    main(args, summary_writer, rng)
