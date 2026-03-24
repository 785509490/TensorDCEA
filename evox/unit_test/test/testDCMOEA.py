import time
import torch
import matplotlib.pyplot as plt
import numpy as np

from evox.algorithms import tensorDCEA
from evox.metrics import igd
from evox.problems.numerical import DCP2, DCP4
from evox.workflows import StdWorkflow, EvalMonitor
from evox.operators.crossover import simulated_binary, DE_crossover, simulated_binaryF

device = "cuda"
torch.set_default_device(device)
print(f"Using device: {torch.get_default_device()}")

# ==================== para ====================
istime = True   # True: real-time mode False: iteration mode
igd_plot_mode = 'continuous'
igd_sample_interval = 10

popSize = 1000

if istime:
    max_gen = 10000 # Maximum number of iterations (to prevent infinite loops)
    taut = 2 # Interval between environmental changes (seconds)
    nt = 10 # Number of environmental changes
    max_time = nt * taut

    print(f"Configuration: TIME-BASED MODE")
    print(f"  - Maximum generations: {max_gen} (backup limit)")
    print(f"  - Environment change interval: {taut}s")
    print(f"  - Number of environment changes: {nt}")
    print(f"  - Maximum runtime: {max_time}s")
else:
    taut = 50 # Algebraic intervals of environmental change
    nt = 10 # Number of environmental changes
    max_gen = taut * nt
    max_time = None

    print(f"Configuration: GENERATION-BASED MODE")
    print(f"  - Maximum generations: {max_gen}")
    print(f"  - Environment change interval: {taut} generations")
    print(f"  - Number of environment changes: {nt}")

print(f"  - IGD plot mode: {igd_plot_mode}")
if igd_plot_mode == 'continuous':
    print(f"  - IGD sample interval: every {igd_sample_interval} generations\n")
else:
    print()

# ==================== init ====================
if istime:
    prob = DCP2(taut=taut, nt=nt, maxG=max_time, d=50)
    algo = tensorDCEA(pop_size=popSize, n_objs=2, lb=prob.lb, ub=prob.ub,
                  max_gen=max_gen, taut=taut, istime=True)
else:
    prob = DCP4(taut=taut, nt=nt, maxG=max_gen, d=50)
    algo = tensorDCEA(pop_size=popSize, n_objs=2, lb=prob.lb, ub=prob.ub,
                  max_gen=max_gen, taut=taut)

pf = prob.pf()
m = prob.m

monitor = EvalMonitor()
workflow = StdWorkflow(algo, prob, monitor)

env_populations = []
env_pfs = []
env_igds = []
env_generations = []
env_runtimes = []

all_igds = []
all_timestamps = []
all_generations = []
all_env_indices = []
env_change_points = []

current_env_idx = 0
env_start_gen = 0
timeout_flag = False

# ==================== strat ====================
workflow.init_step()

print(f"Starting optimization at {time.strftime('%H:%M:%S')}")
print("=" * 80)
start_time = time.time()
last_change_time = start_time

for i in range(max_gen):
    # ==================== real-time mode ====================
    if istime:
        current_time = time.time()
        elapsed_time = current_time - start_time
        time_since_last_change = current_time - last_change_time

        if elapsed_time >= max_time:
            timeout_flag = True
            print(f"\n{'=' * 80}")
            print(f"TIMEOUT: Maximum runtime ({max_time}s) reached at generation {i + 1}")
            print(f"Actual runtime: {elapsed_time:.4f}s")

            fit = workflow.algorithm.fit.clone()
            fit = fit[~torch.isnan(fit).any(dim=1)]
            if current_env_idx < len(pf):
                PF = pf[current_env_idx]
                if len(fit) > 0:
                    current_igd = igd(fit, PF)
                    env_igds.append(current_igd.item() if torch.is_tensor(current_igd) else current_igd)
                    env_populations.append(fit.cpu())
                    env_pfs.append(PF.cpu())
                    env_generations.append(i + 1 - env_start_gen)
                    env_runtimes.append(time_since_last_change)
                    print(f"Environment {current_env_idx} terminated. "
                          f"IGD: {env_igds[-1]:.6f}, Generations: {env_generations[-1]}")

            print(f"{'=' * 80}\n")
            break

        should_change = time_since_last_change >= taut

        if should_change and i > 0:
            fit = workflow.algorithm.fit.clone()
            fit = fit[~torch.isnan(fit).any(dim=1)]
            if current_env_idx < len(pf):
                PF = pf[current_env_idx]
                if len(fit) > 0:
                    current_igd = igd(fit, PF)
                    env_igds.append(current_igd.item() if torch.is_tensor(current_igd) else current_igd)
                    env_populations.append(fit.cpu())
                    env_pfs.append(PF.cpu())
                    env_generations.append(i + 1 - env_start_gen)
                    env_runtimes.append(time_since_last_change)

                    print(f"\n{'=' * 80}")
                    print(f"Environment {current_env_idx} finished after {time_since_last_change:.4f}s")
                    print(f"  - IGD: {env_igds[-1]:.6f}")
                    print(f"  - Total elapsed time: {elapsed_time:.4f}s")
                    print(f"{'=' * 80}\n")

            if igd_plot_mode == 'continuous':
                env_change_points.append(elapsed_time if istime else i + 1)

            current_env_idx += 1
            last_change_time = current_time
            env_start_gen = i + 1

            if current_env_idx >= nt:
                print(f"All {nt} environments completed!")
                break

    # ==================== iteration mode ====================
    else:
        if (i + 1) % taut == 0 and i != 0:
            fit = workflow.algorithm.fit.clone()
            fit = fit[~torch.isnan(fit).any(dim=1)]
            if current_env_idx < len(pf):
                PF = pf[current_env_idx]
                if len(fit) > 0:
                    current_igd = igd(fit, PF)
                    env_igds.append(current_igd.item() if torch.is_tensor(current_igd) else current_igd)
                    env_populations.append(fit.cpu())
                    env_pfs.append(PF.cpu())
                    env_generations.append(i + 1 - env_start_gen)
                    env_runtimes.append(i + 1 - env_start_gen)

                    print(f"\n{'=' * 80}")
                    print(f"Environment {current_env_idx} finished")
                    print(f"  - IGD: {env_igds[-1]:.6f}")
                    print(f"{'=' * 80}\n")

        if i % taut == 0 and i != 0:
            if igd_plot_mode == 'continuous':
                env_change_points.append(i)
            current_env_idx += 1
            env_start_gen = i + 1

    # ==================== work ====================
    if istime:
        workflow.step(elapsed_time, time_since_last_change)
    else:
        workflow.step(i)

    fit = workflow.algorithm.fit
    fit = fit[~torch.isnan(fit).any(dim=1)]

    if len(fit) > 0 and current_env_idx < len(pf) and igd_plot_mode == 'continuous':
        if i % igd_sample_interval == 0:
            PF = pf[current_env_idx]
            current_igd = igd(fit, PF)
            igd_value = current_igd.item() if torch.is_tensor(current_igd) else current_igd

            all_igds.append(igd_value)
            all_generations.append(i + 1)
            all_env_indices.append(current_env_idx)
            if istime:
                all_timestamps.append(elapsed_time)

# ==================== draw ====================
if len(env_populations) > 0:
    print("Generating final result plots...")

    fig_final, ax_final = plt.subplots(figsize=(12, 10))
    offset_step = 0.3
    colors = plt.cm.rainbow(np.linspace(0.0, 0.6, len(env_populations)))

    for env_idx, (pop, pf_data) in enumerate(zip(env_populations, env_pfs)):
        offset_x = env_idx * offset_step
        offset_y = env_idx * offset_step
        pop_np = pop.numpy()
        pf_np = pf_data.numpy()

        pop_shifted = pop_np + np.array([offset_x, offset_y])
        pf_shifted = pf_np + np.array([offset_x, offset_y])

        ax_final.scatter(pf_shifted[:, 0], pf_shifted[:, 1],
                         c='red', marker='o', s=10, alpha=0.3, zorder=1)
        ax_final.scatter(pop_shifted[:, 0], pop_shifted[:, 1],
                         c=[colors[env_idx]], marker='o', s=30, alpha=0.8,
                         edgecolors='black', linewidths=0.5, zorder=2,
                         label=f'Env {env_idx} (IGD: {env_igds[env_idx]:.4f})')

    ax_final.set_xlabel('Objective 1', fontsize=12)
    ax_final.set_ylabel('Objective 2', fontsize=12)
    ax_final.legend(loc='best', fontsize=8, ncol=2)
    ax_final.grid(True, alpha=0.3)
    plt.tight_layout()

    print("Final plots generated successfully!")
    plt.show()
else:
    print("No data to plot - optimization terminated too early.")

print(f"\nScript completed at {time.strftime('%H:%M:%S')}")
