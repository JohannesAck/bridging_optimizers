# import os
# flags = os.environ.get('XLA_FLAGS', '')
# os.environ['XLA_FLAGS'] = flags + " --xla_force_host_platform_device_count=2"

import datetime
import functools
import itertools
import os
import time
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

from evosax import DiffusionEvolution, PolarizedCBO, OpenES, \
                   OptimizationViaIntegration, Sep_CMA_ES, CMA_ES, ClusteredCBO
from evosax.strategies.AdaPol_ccbo_ovi import AdaptiveCcboOvi
from evosax.strategies.es_ovi import OpenESOVI
from evosax.problems.bbob import BBOB_fns, BBOBFitness


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class Config:
    exp_name: str = "exp"
    # stratname: str = "AdaCCBOOVI"

    # stratname: str = "OptimizationViaIntegration"
    # stratname: str = "Sep_CMA_ES"
    stratname: str = "CBO"
    # stratname: str = "PolarizedCBO"
    # stratname: str = "ClusteredCBO"
    # stratname: str = "OpenESAdam"
    # stratname: str = "OpenESSGD"

    # stratname: str = "OpenESOVI"
    # stratname: str = "OpenESSGDNoAntithetic"
    # stratname: str = "DiffusionEvolution"

    num_generations: int = 2000

    problem_name: str = "BBOFuns"


    popsize: int = 256

    cbo_stepsize: float = 0.3


    # fitness transformation
    centered_rank: bool = True
    z_score: bool = False
    
    # OVI/CBO specific
    whiten_weights: bool = False

    # OpenESOVI-hybrid
    ovies_alpha: float = 0.5
    only_combine_angles_openesovi: bool = True


    sigma: float = 0.1
    sigma_end_ratio: float = 0.3  # does not apply for OVI, OVI goes to 0.0 as defined by algorithm

    # lr: float = 0.15
    lr: float = 0.01
    lr_end_ratio: float = 0.3 

    seeds_per_config: int = 20

    seed: int = 1
    num_devices: int = 1



def get_evo_fun(
        problem_name: str,
        strat_partial: Callable,
        evoparam_updates: Dict[str, Any],
        log_cb: Callable,
        schedules: Dict[str, Callable],
        num_generations: int,
        num_dims: int = 2,
    ) -> Callable:
    problem = BBOBFitness(problem_name, num_dims=num_dims)

    def evo_fn(rng, param_update_dict):
        strategy = strat_partial(num_dims=num_dims)
        evo_params = strategy.default_params.replace(
            **evoparam_updates,
        )
        evo_params = evo_params.replace(**param_update_dict)
        
        rng, subkey_strat_init, subkey_problem = jax.random.split(rng, 3)
        evo_state = strategy.initialize(subkey_strat_init, evo_params)
        bbo_r, bbo_q = problem.get_rotation_matrices(subkey_problem)

        def body_fn(carry, _):
            rng, evo_state = carry
            rng, ask_rng, eval_rng = jax.random.split(rng, 3)
            x, evo_state = strategy.ask(ask_rng, evo_state, evo_params)
            fitness = problem.rollout(
                eval_rng, x, bbo_r, bbo_q
            )

            old_mean_param = evo_state.mean
            evo_state = strategy.tell(x, fitness, evo_state, evo_params)

            mean_fitness = jnp.mean(fitness)
            min_fitness = jnp.min(fitness)
            log_dict = {
                "mean_fitness": mean_fitness,
                "min_fitness": min_fitness,
                "weight_mean": jnp.mean(evo_state.mean),
                "weight_std": jnp.std(evo_state.mean),
                "weight_mean_square": jnp.mean(evo_state.mean ** 2),
                "weight_abs_mean": jnp.mean(jnp.abs(evo_state.mean)),
            }
            if hasattr(evo_state, "cluster_locs"):
                hard_assignments = jnp.argmin(evo_state.cluster_probs, axis=1)
                for i in range(strategy.num_clusters):
                    cluster_fitness = jnp.sum(fitness, where=hard_assignments == i) / (jnp.sum(hard_assignments == i) + 1e-6)
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
            # jax.experimental.io_callback(log_cb, (), evo_state.gen_counter, log_dict, ordered=False)
            
            best_member = x[jnp.argmax(fitness)]
            if hasattr(evo_state, "archive"):
                mean_best_member = (old_mean_param, best_member)
            else:
                mean_best_member = (old_mean_param, best_member)

            return (rng, evo_state,), (
                mean_fitness,
                min_fitness,
                log_dict,
                mean_best_member,
            )

        rng, eval_rng = jax.random.split(rng)
        (final_state, fitness_stats) = jax.lax.scan(
            body_fn,
            (rng, evo_state),
            None,
            length=num_generations,
        )
        _, evo_state = final_state
        mean_fitness_history, min_fitness_history, log_dicts, best_member_obs = fitness_stats
        best_member_obs = jax.tree.map(lambda x: x[::10], best_member_obs)
        return mean_fitness_history, min_fitness_history, evo_state, evo_params, log_dicts, best_member_obs
    return evo_fn



def main(args: Config, summary_writer, rng):
    strat_partial, schedules, hopt_names, hopt_vals, param_updates = get_strategy(args)

    def log_cb(gen, log_dict):
        gen = gen.item()
        if gen % 100 != 0:
            return
        mean_rew, min_rew = log_dict["mean_fitness"].item(), log_dict["min_fitness"].item()
        hist_names = [n.removesuffix('_hist') for n in log_dict.keys() if n.endswith('_hist')]
        for hist_name, bins_name in [(n + '_hist', n + '_bins') for n in hist_names]:
            del log_dict[hist_name], log_dict[bins_name]
        for k, v in log_dict.items():
            summary_writer.add_scalar(f"live/{k}", v.item(), gen)
        desc = f'Mean: {mean_rew:.2f}, Min: {min_rew:.2f} W STD: {log_dict["weight_std"]:+.3f}'
        t.update_and_desc(gen, desc)


    bbo_evo_fn = get_evo_fun(
        problem_name=args.problem_name,
        strat_partial=strat_partial,
        evoparam_updates=param_updates,
        log_cb=log_cb,
        schedules=schedules,
        num_generations=args.num_generations,
        num_dims=2
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

    def split_key(key, num_splits):
        return jax.random.split(key, num_splits)

    vmap_keys = split_key(rng, args.seeds_per_config)

    jit_eval_fn = jax.jit(bbo_evo_fn)
    vmap_eval_fn = jax.vmap(jit_eval_fn, in_axes=(None, 0))
    if len(jax.devices()) > 1:
        pmap_eval_fn = jax.pmap(vmap_eval_fn, in_axes=(0, None))
    else:
        pmap_eval_fn = jax.vmap(vmap_eval_fn, in_axes=(0, None))
    

    # with jax.disable_jit():
    mean_fitness_history, min_fitness_history, evo_state, evo_params, log_dicts, mean_best_obs_hist = \
        pmap_eval_fn(
            vmap_keys, param_update_dicts,
    )

        
    return mean_fitness_history, min_fitness_history, param_updates

def get_strategy(args: Config):
    param_names = []
    param_vals = []
    schedules = {}

    if args.stratname == "OpenESAdam":
        strat_partial = functools.partial(OpenES,
            popsize=args.popsize,
            z_score=args.z_score,
            centered_rank=args.centered_rank,  
            lrate_init=args.lr,
            lrate_decay=1.0,
            lrate_limit=0.001,
        )
        schedules = {
            'sigma': optax.schedules.linear_schedule(1.0, args.sigma_end_ratio, args.num_generations),
        }
        param_updates = {
            "sigma_init": args.sigma,
            "sigma_decay": 1.0,
            "sigma_limit": 0.00
        }
        param_names = ['sigma_init', 'lrate_init']
        param_vals = list(itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2,], [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]))

    elif args.stratname == "OpenESSGD":
        strat_partial = functools.partial(OpenES,
            popsize=args.popsize,
            z_score=args.z_score,
            centered_rank=args.centered_rank,  
            lrate_init=args.lr,
            lrate_decay=1.0,
            sigma_decay=1.0,
            opt_name="sgd",
        )
        param_updates = {
            "sigma_init": args.sigma,
        }
        schedules = {
            'sigma': optax.schedules.linear_schedule(1.0, args.sigma_end_ratio, args.num_generations),
            'lrate': optax.schedules.linear_schedule(1.0, args.lr_end_ratio, args.num_generations),
        }
        param_names = ['sigma_init', 'lrate_init']
        param_vals = list(itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2,], [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]))
    elif args.stratname == "CMA_ES":
        strat_partial = functools.partial(CMA_ES,
            popsize=args.popsize,
            centered_rank=args.centered_rank,
            z_score=args.z_score,
        )
        param_updates = {
            "sigma_init": args.sigma,
        }
        param_names = ['sigma_init', 'c_m']
        param_vals = list(itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2,], [0.8, 0.9, 1.0, 1.1, 1.2]))
    elif args.stratname == "Sep_CMA_ES":
        strat_partial = functools.partial(Sep_CMA_ES,
            popsize=args.popsize,
            centered_rank=args.centered_rank,
            z_score=args.z_score,
        )
        param_updates = {
            "sigma_init": args.sigma,
        }
        param_names = ['sigma_init', 'c_m']
        param_vals = list(itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2,], [0.8, 0.9, 1.0, 1.1, 1.2]))
    elif args.stratname == "DiffusionEvolution":
        strat_partial = functools.partial(DiffusionEvolution,
            popsize=args.popsize,
            n_devices=1,
            num_generations=args.num_generations,
            fitness_mapping="energy",
            alpha_schedule="cosine",
            num_latent_dims=None,
            centered_rank=args.centered_rank,
            z_score=args.z_score,
            sigma_init=args.sigma,
        )
        param_updates = {
            "init_scale": 1.0,
        }
        param_names = ['sigma_init', 'fitness_map_temp']
        param_vals = list(itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2,], [0.6, 0.7, 0.8, 0.9, 1.0]))

    elif args.stratname == "CBO":
        strat_partial = functools.partial(PolarizedCBO,
            popsize=args.popsize,
            n_devices=1,
            do_default_cbo=True,
            centered_rank=args.centered_rank,
            constant_noise=False,
            whiten_weights=args.whiten_weights,
            z_score=args.z_score,
        )
        param_updates = {
            'scale_factor': 1.0,
            'sigma_init': args.sigma,
            'step_size_init': args.cbo_stepsize,
        }
        param_names = ['sigma_init', 'beta']
        param_vals = list(itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2,], [0.5, 0.75, 1.0, 1.25, 1.5]))
    elif args.stratname == "CBOConstNoise":
        strat_partial = functools.partial(PolarizedCBO,
            popsize=args.popsize,
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
            'step_size_init': args.cbo_stepsize,
        }
        schedules = {
            'sigma': optax.schedules.linear_schedule(1.0, args.sigma_end_ratio, args.num_generations),
        }
        param_names = ['sigma_init', 'beta']
        param_vals = list(itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2,], [0.5, 0.75, 1.0, 1.25, 1.5]))
    elif args.stratname == "PolarizedCBO":
        strat_partial = functools.partial(PolarizedCBO,
            popsize=args.popsize,
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
            'step_size_init': args.cbo_stepsize,
            'kernel_size': 5.0,
        }
        schedules = {
            'sigma': optax.schedules.linear_schedule(1.0, args.sigma_end_ratio, args.num_generations),
        }
        param_names = ['sigma_init', 'beta', 'kernel_size']
        param_vals = list(itertools.product([1e-4, 1e-3, 1e-2], [0.75, 1.0, 1.25], [1.0, 2.0, 5.0]))
    elif args.stratname == "ClusteredCBO":
        strat_partial = functools.partial(ClusteredCBO,
            popsize=args.popsize,
            num_clusters=4,
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
            'step_size_init': args.cbo_stepsize,
        }
        n_optviaint_gens = args.num_generations // 2
        n_cbo_gens = args.num_generations - n_optviaint_gens
        schedules = {
            'sigma': optax.schedules.linear_schedule(1.0, args.sigma_end_ratio, args.num_generations),
        }
        param_names = ['sigma_init', 'beta', 'kernel_size']
        param_vals = list(itertools.product([1e-4, 1e-3, 1e-2], [0.75, 1.0, 1.25], [1.0, 2.0, 5.0]))
    elif args.stratname == "SchedPolarizedCBOOVI":
        strat_partial = functools.partial(PolarizedCBO,
            popsize=args.popsize,
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
            'step_size_init': 1.0,
            'init_min': 0.0,
            'init_max': 0.0,
            'kernel_size': 10.0,
        }
        n_optviaint_gens = args.num_generations // 2
        n_cbo_gens = args.num_generations - n_optviaint_gens
        schedules = {
            'step_size': optax.schedules.join_schedules(
                [optax.schedules.constant_schedule(1.0), optax.schedules.linear_schedule(0.4, 0.1, n_cbo_gens)],
                [n_optviaint_gens]
            ),
            'sigma': optax.schedules.join_schedules(
                [optax.schedules.linear_schedule(1.0, 0.1, n_optviaint_gens), 
                 optax.schedules.linear_schedule(0.2, 0.00, n_cbo_gens)],
                [n_optviaint_gens]
            )
        }
    elif args.stratname == "SchedClusteredCBOOVI":
        strat_partial = functools.partial(ClusteredCBO,
            popsize=args.popsize,
            num_clusters=4,
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
            'step_size_init': 1.0,
            'init_min': 0.0,
            'init_max': 0.0,
        }
        n_optviaint_gens = args.num_generations // 2
        n_cbo_gens = args.num_generations - n_optviaint_gens
        schedules = {
            'step_size': optax.schedules.join_schedules(
                [optax.schedules.constant_schedule(1.0), optax.schedules.linear_schedule(0.4, 0.1, n_cbo_gens)],
                [n_optviaint_gens]
            ),
            'sigma': optax.schedules.join_schedules(
                [optax.schedules.linear_schedule(1.0, 0.1, n_optviaint_gens), 
                 optax.schedules.constant_schedule(1.5),
                 optax.schedules.linear_schedule(0.3, 0.00, n_cbo_gens)],
                [n_optviaint_gens - 1,  n_optviaint_gens]
            )
        }
    elif args.stratname == "OptimizationViaIntegration":
        strat_partial = functools.partial(OptimizationViaIntegration,
            popsize=args.popsize,
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

        param_names = ['sigma_init', 'beta']
        param_vals = list(itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2,], [0.5, 0.75, 1.0, 1.25, 1.5]))
    elif args.stratname == "OpenESOVI":
        strat_partial = functools.partial(OpenESOVI,
            popsize=args.popsize,
            centered_rank=args.centered_rank,
            z_score=args.z_score,
            whiten_weights=args.whiten_weights,
            alpha_init=args.ovies_alpha,
            lrate_init=args.lr,
            only_combine_angles=args.only_combine_angles_openesovi,
        )
        schedules = {
            'sigma': optax.schedules.linear_schedule(1.0, args.sigma_end_ratio, args.num_generations),
            'lrate': optax.schedules.linear_schedule(1.0, args.lr_end_ratio, args.num_generations),
        }

        param_updates = {
            'scale_factor': 1.0,
            'sigma_init': args.sigma,
        }
        param_names = ['sigma_init', 'lrate_init']
        param_vals = list(itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2,], [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]))
    elif args.stratname == "AdaCCBOOVI":
        strat_partial = functools.partial(AdaptiveCcboOvi,
            popsize=args.popsize,
            centered_rank=args.centered_rank,
            z_score=args.z_score,
            whiten_weights=args.whiten_weights,
            success_memory_len=100,
            step_size_init=0.3
        )
        constant_gens = 3 * args.num_generations // 4

        schedules = {}

        param_updates = {
            'scale_factor': 1.0,
            'sigma_init': 0.03,
            'kernel_size': 10.0,
        }
        param_names = ['sigma_init', 'beta']
        param_vals = list(itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2,], [0.5, 0.75, 1.0, 1.25, 1.5]))
    else:
        raise ValueError(f"Unknown strategy name: {args.stratname}")
    param_updates.update({
        'init_min': -5.0,
        'init_max': 5.0,
        'clip_min': -5.0,
        'clip_max': 5.0,
    })

    return strat_partial, schedules, param_names, param_vals, param_updates
            

if __name__ == "__main__":
    from zoneinfo import ZoneInfo
    import matplotlib.pyplot as plt
    args = tyro.cli(Config)

    rng = jax.random.PRNGKey(args.seed)
    current_time = datetime.datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d-%H%M%S")
    current_date = datetime.datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")
    uuid = str(uuid.uuid4())[:6]
    run_name = f"{args.problem_name}_{args.stratname}_{args.popsize}_{args.num_generations}_{current_time}_{args.exp_name}_{uuid}"
    print(f"Starting run {run_name}")
    
    log_dir = os.path.join("output", current_date, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "param_updates"), exist_ok=True)
    # dump config
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        f.write(str(args))
    summary_writer = SummaryWriter(log_dir)

    problem_res_dict = {}
    for problem_name in tqdm.tqdm(BBOB_fns, desc="Problems", leave=True, total=len(BBOB_fns)):
        if problem_name == "Katsuura": # buggy, skip
            continue
        mean_fits = {}
        min_fits = {}
        strats = ["OpenESSGD", "CBO", "CBOConstNoise", "OptimizationViaIntegration", "DiffusionEvolution", "CMA_ES", "Sep_CMA_ES", "PolarizedCBO", "OpenESOVI"]
        for stratname in tqdm.tqdm(strats, desc="Strategies", leave=True):
            args.stratname = stratname
            args.problem_name = problem_name
            start_time = time.time()
            mean_fitness, min_fitness, param_updates = main(args, summary_writer, rng)
            print(f"Finished {stratname} on {problem_name} in {time.time() - start_time:.2f}s")
            mean_fits[stratname] = mean_fitness
            min_fits[stratname] = min_fitness
            # save parameter updates
            with open(os.path.join(log_dir, 'param_updates', f"{problem_name}_{stratname}_param_updates.txt"), "w") as f:
                f.write(str(param_updates))
        

        problem_res_dict[problem_name] = {
            "mean_fits": mean_fits,
            "min_fits": min_fits,
        }
    
    np.save(os.path.join(log_dir, "problem_res_dict.npy"), problem_res_dict)
    print('Saved problem results to', os.path.join(log_dir, "problem_res_dict.npy"))
    print('done')
