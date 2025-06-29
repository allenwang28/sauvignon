import multiprocessing as mp
import random
import time
import json
import os
from collections import defaultdict
from simulations_util import DeepResearchRequest, SimulationConfig
import matplotlib.pyplot as plt


def verifier_worker(input_queue, trace_queue, component_name, worker_id, step_low, step_high, tid, response_queues):
    while True:
        idle_start = time.time()
        item = input_queue.get()
        idle_end = time.time()
        if item is None:
            break
        idle_dur = idle_end - idle_start
        trace_queue.put({
            "name": f"{component_name}[{worker_id}]::idle",
            "ph": "X",
            "ts": int(idle_start * 1e6),
            "dur": int(idle_dur * 1e6),
            "pid": 1,
            "tid": tid,
            "args": {},
        })
        req, t0, req_id, rollout_id = item
        t1 = time.time()
        time.sleep(step_low + (step_high - step_low) * random.random())
        req.latest_turn += 1
        req.events.append(component_name)
        req.action = f'{component_name}_action'
        t2 = time.time()
        trace_queue.put({
            "name": f"{component_name}[{worker_id}]::step",
            "ph": "X",
            "ts": int(t1 * 1e6),
            "dur": int((t2 - t1) * 1e6),
            "pid": 1,
            "tid": tid,
            "args": {"request": req.summary()},
        })
        # Return to rollout via its response queue
        response_queues[rollout_id].put((req, t2, req_id))


def rollout_process(verifier_queues, trace_queue, replay_buffer, prompts, rollout_id, config, tid, response_queue):
    trajectory = []
    for idx, prompt in enumerate(prompts):
        t0 = time.time()
        req = DeepResearchRequest(initial_prompt=prompt, rank=idx)
        done = False
        obs = prompt
        step_count = 0
        while not done and step_count < config.max_steps:
            # Route to a random verifier
            route = random.choice(["coding_verifier", "math_verifier", "judger"])
            q = verifier_queues[route]
            req_id = f"{rollout_id}_{idx}_{req.latest_turn}_{route}_{random.randint(0, 1e9)}"
            q.put((req, t0, req_id, rollout_id))
            # Wait for verifier to process
            while True:
                req2, t2, resp_id = response_queue.get()
                if resp_id == req_id:
                    break
            # Emit rollout step event
            t3 = time.time()
            trace_queue.put({
                "name": f"rollout[{rollout_id}]::step",
                "ph": "X",
                "ts": int(t2 * 1e6),
                "dur": int((t3 - t2) * 1e6),
                "pid": 1,
                "tid": tid,
                "args": {"request": req2.summary()},
            })
            trajectory.append((obs, req2.action))
            step_count += 1
            obs = f"obs_{rollout_id}_{req2.latest_turn}"
        # Place trajectory in replay buffer
        replay_buffer.put((trajectory, time.time()))


def train_loop(replay_buffer, trace_queue, batch_size, max_steps, tid):
    steps = 0
    while steps < max_steps:
        batch = []
        idle_start = time.time()
        for _ in range(batch_size):
            item = replay_buffer.get()
            idle_end = time.time()
            idle_dur = idle_end - idle_start
            trace_queue.put({
                "name": f"trainer[0]::idle",
                "ph": "X",
                "ts": int(idle_start * 1e6),
                "dur": int(idle_dur * 1e6),
                "pid": 1,
                "tid": tid,
                "args": {},
            })
            if item is None:
                return
            batch.append(item)
            idle_start = time.time()
        t1 = time.time()
        time.sleep(0.1)
        t2 = time.time()
        for traj, _ in batch:
            trace_queue.put({
                "name": f"trainer[0]::step",
                "ph": "X",
                "ts": int(t1 * 1e6),
                "dur": int((t2 - t1) * 1e6),
                "pid": 1,
                "tid": tid,
                "args": {"trajectory_len": len(traj)},
            })
        steps += 1


def metrics_from_trace(trace_events):
    metrics = defaultdict(lambda: {'processed': 0, 'latencies': [], 'idle_time': 0.0})
    for event in trace_events:
        name = event['name']
        component = name.split('[')[0]
        if '::step' in name:
            dur = event['dur'] / 1e6
            metrics[component]['processed'] += 1
            metrics[component]['latencies'].append(dur)
        elif '::idle' in name:
            dur = event['dur'] / 1e6
            metrics[component]['idle_time'] += dur
    return metrics


def main():
    print("Starting rollout simulation...")
    config = SimulationConfig(
        num_coding_verifiers=2,
        num_math_verifiers=2,
        num_llm_judge_judgers=1,
        batch_size=4,
        max_steps=2,
        coding_verifier_step_low=0.2,
        coding_verifier_step_high=0.5,
        math_verifier_step_low=0.3,
        math_verifier_step_high=0.7,
        llm_judge_judger_step_low=0.1,
        llm_judge_judger_step_high=0.3,
        trace_output="trace.json",
        num_envs=4,
    )
    prompts = [f"prompt_{i}" for i in range(config.num_envs * config.max_steps)]
    print(f"Created {len(prompts)} prompts")
    # Queues
    verifier_queues = {
        "coding_verifier": mp.Queue(),
        "math_verifier": mp.Queue(),
        "judger": mp.Queue(),
    }
    trace_queue = mp.Queue()
    replay_buffer = mp.Queue()
    num_rollouts = config.num_envs
    response_queues = [mp.Queue() for _ in range(num_rollouts)]
    print(f"Created {num_rollouts} rollouts")
    # Start verifier workers
    tid = 0
    workers = []
    for i in range(config.num_coding_verifiers):
        workers.append(mp.Process(target=verifier_worker, args=(verifier_queues["coding_verifier"], trace_queue, "coding_verifier", i, config.coding_verifier_step_low, config.coding_verifier_step_high, tid, response_queues)))
        tid += 1
    for i in range(config.num_math_verifiers):
        workers.append(mp.Process(target=verifier_worker, args=(verifier_queues["math_verifier"], trace_queue, "math_verifier", i, config.math_verifier_step_low, config.math_verifier_step_high, tid, response_queues)))
        tid += 1
    for i in range(config.num_llm_judge_judgers):
        workers.append(mp.Process(target=verifier_worker, args=(verifier_queues["judger"], trace_queue, "judger", i, config.llm_judge_judger_step_low, config.llm_judge_judger_step_high, tid, response_queues)))
        tid += 1
    print(f"Starting {len(workers)} verifier workers")
    for w in workers:
        w.start()
    # Start rollouts
    rollout_procs = []
    rollout_chunk = (len(prompts) + num_rollouts - 1) // num_rollouts
    for i in range(num_rollouts):
        chunk = prompts[i*rollout_chunk:(i+1)*rollout_chunk]
        rollout_procs.append(mp.Process(target=rollout_process, args=(verifier_queues, trace_queue, replay_buffer, chunk, i, config, tid, response_queues[i])))
        tid += 1
    print(f"Starting {len(rollout_procs)} rollout processes")
    for p in rollout_procs:
        p.start()
    # Start trainer
    trainer = mp.Process(target=train_loop, args=(replay_buffer, trace_queue, config.batch_size, config.max_steps, tid))
    trainer.start()
    print("Started trainer")
    # Wait for rollouts to finish
    print("Waiting for rollouts to finish...")
    for p in rollout_procs:
        p.join()
    print("All rollouts finished")
    # Signal verifiers to shutdown
    for q in verifier_queues.values():
        for _ in range(config.num_coding_verifiers + config.num_math_verifiers + config.num_llm_judge_judgers):
            q.put(None)
    for w in workers:
        w.join()
    print("All verifier workers finished")
    # Signal trainer to shutdown
    replay_buffer.put(None)
    trainer.join()
    print("Trainer finished")
    # Collect traces
    traces = []
    while not trace_queue.empty():
        traces.append(trace_queue.get())
    with open(config.trace_output, "w") as f:
        json.dump({"traceEvents": traces}, f)
    print(f"Perfetto trace written to {os.path.abspath(config.trace_output)}")
    print("To view, open https://ui.perfetto.dev and load the trace file.")
    # Print summary
    metrics = metrics_from_trace(traces)
    components = list(metrics.keys())
    throughput = [
        metrics[c]['processed'] / max(1e-9, sum(metrics[c]['latencies']) + metrics[c]['idle_time'])
        for c in components
    ]
    avg_latency = [
        sum(metrics[c]['latencies']) / len(metrics[c]['latencies']) if metrics[c]['latencies'] else 0
        for c in components
    ]
    idle_time = [metrics[c]['idle_time'] for c in components]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    axs[0].bar(components, throughput)
    axs[0].set_title('Throughput (events/sec)')
    axs[1].bar(components, avg_latency)
    axs[1].set_title('Average Latency (s)')
    axs[2].bar(components, idle_time)
    axs[2].set_title('Total Idle Time (s)')
    plt.tight_layout()
    plt.show()
    print("\nComponent Summary:")
    print(f"{'Component':>16} | {'Throughput (ev/s)':>16} | {'Avg Latency (s)':>16} | {'Idle Time (s)':>16}")
    print("-" * 70)
    for c in components:
        print(f"{c:>16} | {throughput[components.index(c)]:>16.2f} | {avg_latency[components.index(c)]:>16.4f} | {idle_time[components.index(c)]:>16.2f}")

if __name__ == "__main__":
    main()
