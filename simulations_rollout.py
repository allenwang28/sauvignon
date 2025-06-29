import multiprocessing as mp
import random
import time

# --- Policy Worker ---
def policy_worker(input_queue, worker_id, response_queues):
    print(f"[PolicyWorker-{worker_id}] Started")
    while True:
        item = input_queue.get()
        if item is None:
            print(f"[PolicyWorker-{worker_id}] Shutting down")
            break
        obs, req_id, rollout_id = item
        print(f"[PolicyWorker-{worker_id}] Received obs: {obs} (req_id={req_id}, rollout_id={rollout_id})")
        # Simulate work
        time.sleep(random.uniform(0.05, 0.2))
        action = f"action_{worker_id}_{random.randint(0, 100)}"
        print(f"[PolicyWorker-{worker_id}] Sending action: {action} (req_id={req_id}, rollout_id={rollout_id})")
        response_queues[rollout_id].put((req_id, action))

# --- Policy Router ---
def policy_router(request_queue, num_workers, response_queues):
    print(f"[PolicyRouter] Starting with {num_workers} workers")
    worker_queues = [mp.Queue() for _ in range(num_workers)]
    workers = [
        mp.Process(target=policy_worker, args=(worker_queues[i], i, response_queues))
        for i in range(num_workers)
    ]
    for w in workers:
        w.start()

    worker_idx = 0
    while True:
        item = request_queue.get()
        if item is None:
            print(f"[PolicyRouter] Shutting down workers")
            for q in worker_queues:
                q.put(None)
            break
        obs, req_id, rollout_id = item
        print(f"[PolicyRouter] Received request for obs: {obs} (req_id={req_id}, rollout_id={rollout_id})")
        target_queue = worker_queues[worker_idx]
        print(f"[PolicyRouter] Routing to worker {worker_idx}")
        worker_idx = (worker_idx + 1) % num_workers
        target_queue.put((obs, req_id, rollout_id))

    for w in workers:
        w.join()
    print(f"[PolicyRouter] All workers shut down")

# --- Rollout Process ---
def rollout_process(policy_router_queue, response_queue, num_steps, rollout_id):
    print(f"[Rollout-{rollout_id}] Started")
    trajectory = []
    obs = f"obs_{rollout_id}_0"
    for t in range(num_steps):
        req_id = f"{rollout_id}_{t}"
        print(f"[Rollout-{rollout_id}] Sending obs: {obs} to policy router (req_id={req_id})")
        policy_router_queue.put((obs, req_id, rollout_id))
        # Wait for response with matching req_id
        while True:
            resp_id, action = response_queue.get()
            if resp_id == req_id:
                break
        print(f"[Rollout-{rollout_id}] Got action: {action} (req_id={req_id})")
        # Simulate environment step
        reward = random.random()
        obs = f"obs_{rollout_id}_{t+1}"
        trajectory.append((obs, action, reward))
    print(f"[Rollout-{rollout_id}] Finished. Trajectory: {trajectory}")

# --- Main ---
def main():
    num_policy_workers = 2
    num_rollouts = 3
    num_steps = 5

    # Queues
    policy_router_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(num_rollouts)]

    # Start policy router
    router = mp.Process(target=policy_router, args=(policy_router_queue, num_policy_workers, response_queues))
    router.start()

    # Start rollouts
    rollouts = [
        mp.Process(target=rollout_process, args=(policy_router_queue, response_queues[i], num_steps, i))
        for i in range(num_rollouts)
    ]
    for r in rollouts:
        r.start()
    for r in rollouts:
        r.join()

    # Shutdown router (which will shutdown workers)
    policy_router_queue.put(None)
    router.join()

if __name__ == "__main__":
    main()
