import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import math

from vecsim.vehicle     import Vehicle
from vecsim.edge_server import EdgeServer
from vecsim.agent       import DQNAgent
from vecsim.channel     import WirelessChannel

random.seed(123)
np.random.seed(123)

STEP_DURATION = 0.1

channel = WirelessChannel(
    bandwidth=5_000_000,
    transmission_power=1000,
    noise_factor=1
)

# -----------------------------
# POLICY EVALUATION LOOP
# -----------------------------
def run_policy(policy_fn, num_episodes=100):

    servers = {
        1: EdgeServer(server_id=1, capacity=300, x=10, y=0, coverage_radius=10),
        2: EdgeServer(server_id=2, capacity=300, x=30, y=0, coverage_radius=10),
        3: EdgeServer(server_id=3, capacity=300, x=50, y=0, coverage_radius=10),
    }

    vehicle = Vehicle(vehicle_id=1, x=0, y=0, speed=2, direction=0)

    all_completion_rates = []
    all_drop_rates = []

    for episode in range(num_episodes):

        vehicle.reset(x=0, y=0)
        for server in servers.values():
            server.reset()

        current_server_id = None
        total_generated = 0
        total_dropped = 0

        for step in range(35):

            vehicle.move()
            vehicle.tick(step)

            # -------------------------
            # NETWORK DYNAMICS (FIXED)
            # -------------------------
            for server in servers.values():
                server.transmission_tick()
            for server in servers.values():
                server.backhaul_tick()
            for server in servers.values():
                server.tick(step)

            # -------------------------
            # ACTIVE SERVER
            # -------------------------
            active_server = None
            for server in servers.values():
                if server.in_range(vehicle):
                    active_server = server
                    break

            # -------------------------
            # HANDOVER (FIXED BACKHAUL)
            # -------------------------
            if active_server is not None:
                if current_server_id != active_server.server_id:

                    if current_server_id is not None:
                        old_server = servers[current_server_id]
                        waiting_tasks = old_server.release_queue()

                        num_tasks = len(waiting_tasks)

                        for task in waiting_tasks:
                            shared_rate = active_server.backhaul_bandwidth / max(num_tasks, 1)
                            migration_time = task.data_size / shared_rate
                            delay_steps = math.ceil(migration_time / STEP_DURATION)

                            active_server.accept_backhaul_task(task, delay_steps)

                    current_server_id = active_server.server_id

            # -------------------------
            # TASK GENERATION
            # -------------------------
            tasks = vehicle.generate_tasks(step)
            total_generated += len(tasks)

            # =====================================================
            # CASE 1: NO SERVER
            # =====================================================
            if active_server is None:
                total_dropped += len(tasks)
                continue

            # =====================================================
            # CASE 2: SERVER AVAILABLE
            # =====================================================
            for task in tasks:

                queue_len = len(active_server.task_queue)

                edge_load = (
                    len(active_server.task_queue) +
                    len(active_server.transmission_queue) +
                    len(active_server.backhaul_queue) +
                    (1 if active_server.current_task else 0)
                )
                local_busy = 1 if vehicle.local_task else 0

                state = [
                    edge_load / 50,
                    len(active_server.transmission_queue) / 20,
                    task.cpu_cycles / 1000,
                    task.data_size / 1_000_000,
                    len(vehicle.local_queue) / 20,
                    local_busy
                ]

                action = policy_fn(state)

                # -------------------------
                # LOCAL
                # -------------------------
                if action == 0:
                    vehicle.accept_local_task(task)

                # -------------------------
                # OFFLOAD (WITH CHANNEL MODEL)
                # -------------------------
                else:
                    distance = math.sqrt(
                        (vehicle.x - active_server.x) ** 2 +
                        (vehicle.y - active_server.y) ** 2
                    )

                    delay_sec = channel.compute_delay(task.data_size, distance)
                    delay_steps = math.ceil(delay_sec / STEP_DURATION)

                    task.transmission_remaining = delay_steps
                    active_server.add_to_transmission_queue(task)

        # -------------------------
        # METRICS
        # -------------------------
        total_completed = (
            sum(s.tasks_processed for s in servers.values()) +
            vehicle.local_tasks_processed
        )

        completion_rate = total_completed / total_generated if total_generated > 0 else 0
        drop_rate = total_dropped / total_generated if total_generated > 0 else 0

        all_completion_rates.append(completion_rate)
        all_drop_rates.append(drop_rate)

    return {
        "completion_rate": np.mean(all_completion_rates),
        "drop_rate": np.mean(all_drop_rates)
    }

# -----------------------------
# BASELINES
# -----------------------------
def always_offload(state):
    return 1

def always_local(state):
    return 0

# -----------------------------
# DQN POLICY
# -----------------------------
trained_agent = DQNAgent(state_size=6, action_size=2)
trained_agent.load("results/trained_agent.npy")

trained_agent.epsilon = 0.0

def dqn_policy(state):
    return trained_agent.act(state)

# -----------------------------
# RUN EXPERIMENTS
# -----------------------------
print("Running always-offload baseline...")
offload_results = run_policy(always_offload)

print("Running always-local baseline...")
local_results = run_policy(always_local)

print("Running trained DQN policy...")
dqn_results = run_policy(dqn_policy)

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\n=== Baseline Comparison ===")
print(f"{'Policy':<20} {'Completion Rate':>16} {'Drop Rate':>12}")

print(f"{'Always Offload':<20} {offload_results['completion_rate']:>15.1%} {offload_results['drop_rate']:>12.1%}")
print(f"{'Always Local':<20}   {local_results['completion_rate']:>15.1%} {local_results['drop_rate']:>12.1%}")
print(f"{'DQN Agent':<20}       {dqn_results['completion_rate']:>15.1%} {dqn_results['drop_rate']:>12.1%}")

# -----------------------------
# PLOT
# -----------------------------
policies = ["Always Offload", "Always Local", "DQN Agent"]

completion_rates = [
    offload_results["completion_rate"],
    local_results["completion_rate"],
    dqn_results["completion_rate"]
]

drop_rates = [
    offload_results["drop_rate"],
    local_results["drop_rate"],
    dqn_results["drop_rate"]
]

x = np.arange(len(policies))
width = 0.35

plt.bar(x - width/2, completion_rates, width, label="Completion Rate")
plt.bar(x + width/2, drop_rates, width, label="Drop Rate")

plt.xticks(x, policies)
plt.ylabel("Rate")
plt.title("Policy Comparison — VEC Simulator")
plt.legend()
plt.tight_layout()
plt.savefig("results/comparison.png", dpi=300)
plt.show()