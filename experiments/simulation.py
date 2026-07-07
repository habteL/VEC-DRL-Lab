import csv
import random
import numpy as np
import math

from vecsim.vehicle     import Vehicle
from vecsim.edge_server import EdgeServer
from vecsim.agent       import DQNAgent
from vecsim.channel     import WirelessChannel
# =====================================================
# RANDOM SEED
# =====================================================

random.seed(42)
np.random.seed(42)


# =====================================================
# SYSTEM CONFIGURATION
# =====================================================

servers = {
    1: EdgeServer(
        server_id=1,
        capacity=300,
        x=10,
        y=0,
        coverage_radius=10
    ),

    2: EdgeServer(
        server_id=2,
        capacity=300,
        x=30,
        y=0,
        coverage_radius=10
    ),

    3: EdgeServer(
        server_id=3,
        capacity=300,
        x=50,
        y=0,
        coverage_radius=10
    ),
}


vehicle = Vehicle(
    vehicle_id=1,
    x=0,
    y=0,
    speed=2,
    direction=0
)


# =====================================================
# UPDATED STATE SIZE = 6
# =====================================================

agent = DQNAgent(
    state_size=6,
    action_size=2
)


channel = WirelessChannel(
    bandwidth=5_000_000,
    transmission_power=1000,
    noise_factor=1
)


STEP_DURATION = 0.1


episode_metrics = []


# =====================================================
# TRAINING LOOP
# =====================================================

for episode in range(1000):

    vehicle.reset(x=0, y=0)

    for server in servers.values():
        server.reset()


    # task_id -> experience waiting for completion
    pending_experiences = {}


    current_server_id = None


    total_generated = 0
    total_dropped = 0
    episode_reward = 0


    step_metrics = []


    # =================================================
    # TIME SLOT LOOP
    # =================================================

    for step in range(35):


        # -------------------------------
        # Vehicle movement
        # -------------------------------

        vehicle.move()
        vehicle.tick(step)
        # -------------------------------
        # Network and computation update
        # -------------------------------

        for server in servers.values():
            server.transmission_tick()

        for server in servers.values():
            server.backhaul_tick()

        for server in servers.values():
            server.tick(step)

        # =================================================
        # FIND ACTIVE RSU
        # =================================================


        active_server = None


        for server in servers.values():

            if server.in_range(vehicle):

                active_server = server
                break



        # =================================================
        # HANDOVER WITH BACKHAUL DELAY
        # =================================================


        if active_server is not None:


            if current_server_id != active_server.server_id:


                if current_server_id is not None:


                    old_server = servers[current_server_id]


                    waiting_tasks = old_server.release_queue()


                    num_tasks = len(waiting_tasks)


                    for task in waiting_tasks:


                        shared_rate = (
                            active_server.backhaul_bandwidth /
                            max(num_tasks,1)
                        )


                        migration_time = (
                            task.data_size /
                            shared_rate
                        )


                        delay_steps = math.ceil(
                            migration_time /
                            STEP_DURATION
                        )


                        active_server.accept_backhaul_task(
                            task,
                            delay_steps
                        )


                current_server_id = active_server.server_id
        # =================================================
        # TASK GENERATION
        # =================================================

        tasks = vehicle.generate_tasks(step)

        total_generated += len(tasks)



        # =================================================
        # CASE 1: NO RSU AVAILABLE
        # =================================================

        if active_server is None:


            total_dropped += len(tasks)


            step_metrics.append({

                "episode": episode,

                "step": step,

                "active_server": None,

                "tasks_generated": len(tasks),

                "tasks_dropped": len(tasks),

                "total_queue_length":
                    sum(
                        len(s.task_queue)
                        for s in servers.values()
                    ),

                "local_queue_length":
                    len(vehicle.local_queue),

                "total_processed":
                    sum(
                        s.tasks_processed
                        for s in servers.values()
                    )

            })


            continue



        # =================================================
        # CASE 2: RSU AVAILABLE
        # =================================================


        for task in tasks:



            # -------------------------------------------------
            # EDGE LOAD CALCULATION
            # -------------------------------------------------

            edge_load = (

                len(active_server.task_queue)

                +

                len(active_server.transmission_queue)

                +

                len(active_server.backhaul_queue)

                +

                (1 if active_server.current_task else 0)

            )



            # -------------------------------------------------
            # STATE BEFORE ACTION
            # -------------------------------------------------


            local_busy = (
                1 if vehicle.local_task else 0
            )


            state = [

                edge_load / 50,

                len(active_server.transmission_queue) / 20,

                task.cpu_cycles / 1000,

                task.data_size / 1_000_000,

                len(vehicle.local_queue) / 20,

                local_busy

            ]



            action = agent.act(state)



            # =================================================
            # ACTION 0: LOCAL EXECUTION
            # =================================================


            if action == 0:  # local
                accepted = vehicle.accept_local_task(task)
                local_pressure = (
                    len(vehicle.local_queue) +
                    (1 if vehicle.local_task else 0)
                ) / 5.0
                immediate_reward = 0.2 - local_pressure

            else:  # offload
                distance = math.sqrt(
                    (vehicle.x - active_server.x)**2 +
                    (vehicle.y - active_server.y)**2
                )
                delay_sec = channel.compute_delay(task.data_size, distance)
                delay_steps = math.ceil(delay_sec / STEP_DURATION)
                task.transmission_remaining = delay_steps
                active_server.add_to_transmission_queue(task)

                edge_pressure = edge_load / 10.0
                channel_cost = delay_steps / 3.0
                immediate_reward = 0.2 - edge_pressure - channel_cost



            # =================================================
            # STORE EXPERIENCE WAITING FOR OUTCOME
            # =================================================


            pending_experiences[task.task_id] = {


                "state": state,


                "action": action,


                "immediate_reward":
                    immediate_reward


            }


            episode_reward += immediate_reward


        # =================================================
        # DELIVER COMPLETION REWARDS
        # =================================================


        # -------- Edge completed tasks --------

        for server in servers.values():

            for task in server.completed_tasks:

                if task.task_id in pending_experiences:

                    exp = pending_experiences.pop(task.task_id)


                    next_state = [

                        len(server.task_queue) / 50,

                        len(server.transmission_queue) / 20,

                        task.cpu_cycles / 1000,

                        task.data_size / 1_000_000,

                        len(vehicle.local_queue) / 20,

                        1 if vehicle.local_task else 0
                    ]


                    final_reward = (
                        exp["immediate_reward"]
                        + 1.0
                    )


                    agent.remember(
                        exp["state"],
                        exp["action"],
                        final_reward,
                        next_state
                    )

                    agent.learn()

                    episode_reward += 1.0


            # completion buffer
            server.completed_tasks.clear()
            
        # -------- Local completed tasks --------

        for task in vehicle.local_completed:

            if task.task_id in pending_experiences:

                exp = pending_experiences.pop(task.task_id)


                next_state = [

                    0,
                    0,

                    task.cpu_cycles / 1000,

                    task.data_size / 1_000_000,

                    len(vehicle.local_queue) / 20,

                    1 if vehicle.local_task else 0
                ]


                final_reward = (
                    exp["immediate_reward"]
                    + 1.0
                )


                agent.remember(
                    exp["state"],
                    exp["action"],
                    final_reward,
                    next_state
                )


                agent.learn()

                episode_reward += 1.0


        vehicle.local_completed.clear()
        # =================================================
        # STEP METRICS
        # =================================================


        step_metrics.append({

            "episode": episode,

            "step": step,

            "active_server":
                active_server.server_id,


            "tasks_generated":
                len(tasks),


            "tasks_dropped":
                0,


            "total_queue_length":

                sum(

                    len(s.task_queue)

                    for s in servers.values()

                ),


            "local_queue_length":

                len(vehicle.local_queue),


            "total_processed":

                sum(

                    s.tasks_processed

                    for s in servers.values()

                )

        })



    # =====================================================
    # END OF EPISODE PENALTY
    # =====================================================


    total_unfinished = 0


    for server in servers.values():


        total_unfinished += (

            len(server.task_queue)

            +

            len(server.transmission_queue)

            +

            len(server.backhaul_queue)

            +

            (1 if server.current_task else 0)

        )



    total_unfinished += (

        len(vehicle.local_queue)

        +

        (1 if vehicle.local_task else 0)

    )



    episode_end_penalty = (

        -0.5 *
        total_unfinished

    )


    episode_reward += episode_end_penalty



    # Remove unfinished experiences

    pending_experiences.clear()



    # =====================================================
    # EPISODE METRICS
    # =====================================================


    all_completed = (

        sum(
            s.tasks_processed
            for s in servers.values()
        )

        +

        vehicle.local_tasks_processed

    )



    completion_rate = (

        all_completed /
        total_generated

        if total_generated > 0
        else 0

    )



    episode_metrics.append({

        "episode":
            episode,


        "completion_rate":
            completion_rate,


        "episode_reward":
            episode_reward,


        "unfinished_tasks":
            total_unfinished,


        "agent_epsilon":
            agent.epsilon

    })



# =====================================================
# SAVE MODEL
# =====================================================

agent.save("results/trained_agent.npy")


print(
    "Agent weights saved"
)



# =====================================================
# SAVE EPISODE METRICS
# =====================================================
with open("results/episode_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["episode", "completion_rate", "episode_reward",
                    "unfinished_tasks", "agent_epsilon"]
    )
    writer.writeheader()
    writer.writerows(episode_metrics)

# =====================================================
# SAVE STEP METRICS
# =====================================================
with open("results/step_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=step_metrics[0].keys())
    writer.writeheader()
    writer.writerows(step_metrics)



# =====================================================
# FINAL REPORT
# =====================================================


print("\n=== Training Complete ===")

print(
    f"Episodes run          : {len(episode_metrics)}"
)

print(
    f"Final epsilon         : {agent.epsilon:.4f}"
)

print(
    f"Final memory size     : {len(agent.memory)}"
)



last = episode_metrics[-1]


print(
    f"Last episode reward   : {last['episode_reward']:.2f}"
)


print(
    f"Last completion rate  : "
    f"{last['completion_rate']*100:.1f}%"
)


print(
    f"Unfinished tasks      : "
    f"{last['unfinished_tasks']}"
)


print("Episode metrics saved : results/episode_metrics.csv")
print("Step metrics saved    : results/step_metrics.csv")

server_completed = sum(s.tasks_processed for s in servers.values())
local_completed  = vehicle.local_tasks_processed

print(f"Server completed      : {server_completed}")
print(f"Local completed       : {local_completed}")
if all_completed > 0:
    print(f"Edge utilization      : "
          f"{server_completed/all_completed*100:.1f}%")
    print(f"Local utilization     : "
          f"{local_completed/all_completed*100:.1f}%")
    
