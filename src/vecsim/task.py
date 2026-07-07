class Task:
    def __init__(self, task_id, owner_id, cpu_cycles, deadline):
        self.task_id = task_id
        self.owner_id = owner_id
        self.cpu_cycles = cpu_cycles
        self.remaining_cycles = cpu_cycles
        self.deadline = deadline
        self.created_at = None    # set when generated
        self.completed_at = None  # set when finished
        self.latency = None       # computed after completion
        self.data_size = cpu_cycles * 1000  # bytes
        self.transmission_delay = 0         # steps needed to transmit
        self.transmission_remaining = 0     # steps still left to transmit
        self.status = "pending"             # pending → transmitting → queued → processing → completed
        self.backhaul_remaining = 0
    def __repr__(self):
        return (f"Task {self.task_id} | owner: Vehicle {self.owner_id} | "
                f"cpu: {self.cpu_cycles} | deadline: {self.deadline} | "
                f"status: {self.status}")
if __name__ == "__main__":
            
    t1 = Task(task_id=1, owner_id=1, cpu_cycles=500, deadline=10)
    t2 = Task(task_id=2, owner_id=1, cpu_cycles=1200, deadline=5)

    print(t1)
    print(t2)

    print(t1.status)
    t1.status = "completed"
    print(t1.status)
    print(t2.status)