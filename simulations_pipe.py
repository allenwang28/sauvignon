"""Simulating the performance of a pipelined approach."""

import asyncio
from simulations_util import DeepResearchRequest, SimulationConfig
from dataclasses import dataclass, field
import random
import time
import functools
import matplotlib.pyplot as plt
import multiprocessing as mp
import json
import os
from typing import List, Optional
from collections import Counter


@dataclass
class DeepResearchRequest:
    initial_prompt: str
    rank: int = 0
    events: List[str] = field(default_factory=list)
    latest_turn: int = 0
    action: Optional[str] = None
    score: int = 0

    def summary(self):
        return {
            "prompt": self.initial_prompt,
            "rank": self.rank,
            "turn": self.latest_turn,
            "events": list(self.events),
            "action": self.action,
            "score": self.score,
        }

@dataclass
class SimulationConfig:
    num_policys: int = 2
    num_coding_verifiers: int = 2
    num_math_verifiers: int = 2
    num_llm_judge_judgers: int = 1
    num_preprocess: int = 1
    num_prompt_loaders: int = 1
    batch_size: int = 4
    max_steps: int = 4
    policy_step_low: float = 0.5
    policy_step_high: float = 1.0
    coding_verifier_step_low: float = 0.2
    coding_verifier_step_high: float = 0.5
    math_verifier_step_low: float = 0.3
    math_verifier_step_high: float = 0.7
    llm_judge_judger_step_low: float = 0.1
    llm_judge_judger_step_high: float = 0.3
    preprocess_step_low: float = 0.05
    preprocess_step_high: float = 0.2
    trace_output: str = "trace.json"


# Worker function for each replica
def worker_loop(input_queue, output_queue, trace_queue, component_name, worker_idx, step_low, step_high, tid, policy_queue=None):
    while True:
        idle_start = time.time()
        item = input_queue.get()
        idle_end = time.time()
        if item is None:
            break
        # Emit idle event
        idle_dur = idle_end - idle_start
        trace_queue.put({
            "name": f"{component_name}[{worker_idx}]::idle",
            "ph": "X",
            "ts": int(idle_start * 1e6),
            "dur": int(idle_dur * 1e6),
            "pid": 1,
            "tid": tid,
            "args": {},
        })
        req, t0 = item
        t1 = time.time()
        # Simulate work
        time.sleep(step_low + (step_high - step_low) * random.random())
        req.latest_turn += 1
        req.events.append(component_name)
        req.action = f'{component_name}_action'
        t2 = time.time()
        trace_queue.put({
            "name": f"{component_name}[{worker_idx}]::step",
            "ph": "X",
            "ts": int(t1 * 1e6),
            "dur": int((t2 - t1) * 1e6),
            "pid": 1,
            "tid": tid,
            "args": {"request": req.summary()},
        })
        # Routing logic
        if component_name == "policy":
            route = random.choice(["coding_verifier", "math_verifier", "judger", "preprocess"])
            if route == "coding_verifier":
                output_queue["coding_verifier"].put((req, t0))
            elif route == "math_verifier":
                output_queue["math_verifier"].put((req, t0))
            elif route == "judger":
                output_queue["judger"].put((req, t0))
            else:
                output_queue["preprocess"].put((req, t0))
        elif component_name in ["coding_verifier", "math_verifier", "judger"]:
            policy_queue.put((req, t0))
        elif component_name == "preprocess":
            output_queue.put((req, t0))

# Trainer worker (batches)
def trainer_loop(input_queue, trace_queue, batch_size, max_steps, tid):
    steps = 0
    while steps < max_steps:
        batch = []
        idle_start = time.time()
        for _ in range(batch_size):
            item = input_queue.get()
            idle_end = time.time()
            # Emit idle event for trainer
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
            req, t0 = item
            batch.append((req, t0))
            idle_start = time.time()  # For next idle period
        t1 = time.time()
        time.sleep(0.1)
        t2 = time.time()
        for req, t0 in batch:
            req.latest_turn += 1
            req.events.append('trainer')
            req.action = 'train_action'
            trace_queue.put({
                "name": f"trainer[0]::step",
                "ph": "X",
                "ts": int(t1 * 1e6),
                "dur": int((t2 - t1) * 1e6),
                "pid": 1,
                "tid": tid,
                "args": {"request": req.summary()},
            })
        steps += 1
    # Signal done
    return

def prompt_loader_worker(policy_queue, trace_queue, prompts, loader_idx, tid):
    import time
    for idx, prompt in enumerate(prompts):
        t0 = time.time()
        req = DeepResearchRequest(initial_prompt=prompt, rank=idx)
        policy_queue.put((req, t0))
        trace_queue.put({
            "name": f"prompt_loader[{loader_idx}]::step",
            "ph": "X",
            "ts": int(t0 * 1e6),
            "dur": 0,
            "pid": 1,
            "tid": tid,
            "args": {"prompt": prompt, "rank": idx},
        })

class MultiprocessingRunner:
    def __init__(self, config):
        self.config = config
        # Queues
        self.policy_queue = mp.Queue()
        self.coding_verifier_queue = mp.Queue()
        self.math_verifier_queue = mp.Queue()
        self.judger_queue = mp.Queue()
        self.preprocess_queue = mp.Queue()
        self.trainer_queue = mp.Queue()
        self.trace_queue = mp.Queue()
        # Worker process lists
        self.policy_workers = []
        self.coding_verifier_workers = []
        self.math_verifier_workers = []
        self.judger_workers = []
        self.preprocess_workers = []
        self.trainer = None
        self.prompt_loaders = []
        self.next_tid = 0

    def start_workers(self, prompts):
        cfg = self.config
        # Prompt loaders
        n = cfg.num_prompt_loaders
        chunk_size = (len(prompts) + n - 1) // n
        for i in range(n):
            chunk = prompts[i*chunk_size:(i+1)*chunk_size]
            tid = self.next_tid
            self.next_tid += 1
            p = mp.Process(target=prompt_loader_worker, args=(self.policy_queue, self.trace_queue, chunk, i, tid))
            p.start()
            self.prompt_loaders.append(p)
        # Policy workers
        for i in range(cfg.num_policys):
            tid = self.next_tid
            self.next_tid += 1
            p = mp.Process(target=worker_loop, args=(self.policy_queue, {
                "coding_verifier": self.coding_verifier_queue,
                "math_verifier": self.math_verifier_queue,
                "judger": self.judger_queue,
                "preprocess": self.preprocess_queue
            }, self.trace_queue, "policy", i, cfg.policy_step_low, cfg.policy_step_high, tid))
            p.start()
            self.policy_workers.append(p)
        # Coding verifier workers
        for i in range(cfg.num_coding_verifiers):
            tid = self.next_tid
            self.next_tid += 1
            p = mp.Process(target=worker_loop, args=(self.coding_verifier_queue, None, self.trace_queue, "coding_verifier", i, cfg.coding_verifier_step_low, cfg.coding_verifier_step_high, tid, self.policy_queue))
            p.start()
            self.coding_verifier_workers.append(p)
        # Math verifier workers
        for i in range(cfg.num_math_verifiers):
            tid = self.next_tid
            self.next_tid += 1
            p = mp.Process(target=worker_loop, args=(self.math_verifier_queue, None, self.trace_queue, "math_verifier", i, cfg.math_verifier_step_low, cfg.math_verifier_step_high, tid, self.policy_queue))
            p.start()
            self.math_verifier_workers.append(p)
        # Judger workers
        for i in range(cfg.num_llm_judge_judgers):
            tid = self.next_tid
            self.next_tid += 1
            p = mp.Process(target=worker_loop, args=(self.judger_queue, None, self.trace_queue, "judger", i, cfg.llm_judge_judger_step_low, cfg.llm_judge_judger_step_high, tid, self.policy_queue))
            p.start()
            self.judger_workers.append(p)
        # Preprocess workers
        for i in range(cfg.num_preprocess):
            tid = self.next_tid
            self.next_tid += 1
            p = mp.Process(target=worker_loop, args=(self.preprocess_queue, self.trainer_queue, self.trace_queue, "preprocess", i, cfg.preprocess_step_low, cfg.preprocess_step_high, tid))
            p.start()
            self.preprocess_workers.append(p)
        # Trainer
        tid = self.next_tid
        self.next_tid += 1
        self.trainer = mp.Process(target=trainer_loop, args=(self.trainer_queue, self.trace_queue, cfg.batch_size, cfg.max_steps, tid))
        self.trainer.start()

    def shutdown_workers(self):
        cfg = self.config
        for _ in range(cfg.num_policys):
            self.policy_queue.put(None)
        for _ in range(cfg.num_coding_verifiers):
            self.coding_verifier_queue.put(None)
        for _ in range(cfg.num_math_verifiers):
            self.math_verifier_queue.put(None)
        for _ in range(cfg.num_llm_judge_judgers):
            self.judger_queue.put(None)
        for _ in range(cfg.num_preprocess):
            self.preprocess_queue.put(None)
        self.trainer_queue.put(None)
        for p in self.policy_workers + self.coding_verifier_workers + self.math_verifier_workers + self.judger_workers + self.preprocess_workers:
            p.join()
        if self.trainer:
            self.trainer.join()
        for p in self.prompt_loaders:
            p.join()

    def run(self, prompts):
        self.start_workers(prompts)
        # Wait for trainer to finish
        self.trainer.join()
        self.shutdown_workers()
        # Collect traces
        traces = []
        while not self.trace_queue.empty():
            traces.append(self.trace_queue.get())
        with open(self.config.trace_output, "w") as f:
            json.dump({"traceEvents": traces}, f)
        print(f"Perfetto trace written to {os.path.abspath(self.config.trace_output)}")
        print("To view, open https://ui.perfetto.dev and load the trace file.")


if __name__ == "__main__":
    config = SimulationConfig(
        num_policys=16,
        num_coding_verifiers=2,
        num_math_verifiers=2,
        num_llm_judge_judgers=1,
        num_preprocess=1,
        num_prompt_loaders=2,
        batch_size=4,
        max_steps=2,
        policy_step_low=0.5,
        policy_step_high=1.0,
        coding_verifier_step_low=0.2,
        coding_verifier_step_high=0.5,
        math_verifier_step_low=0.3,
        math_verifier_step_high=0.7,
        llm_judge_judger_step_low=0.1,
        llm_judge_judger_step_high=0.3,
        preprocess_step_low=0.05,
        preprocess_step_high=0.2,
        trace_output="trace_pipeline.json",
    )
    prompts = [f"prompt_{i}" for i in range(config.max_steps * config.batch_size)]
    runner = MultiprocessingRunner(config)
    try:
        runner.run(prompts)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        import sys
        sys.exit(0)

    # Load trace and compute metrics
    import json
    import matplotlib.pyplot as plt
    from collections import defaultdict

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

    with open(config.trace_output) as f:
        trace_data = json.load(f)
    metrics = metrics_from_trace(trace_data['traceEvents'])

    components = list(metrics.keys())
    # Throughput: processed / (total active time)
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

    # Print summary table
    print("\nComponent Summary:")
    print(f"{'Component':>16} | {'Throughput (ev/s)':>16} | {'Avg Latency (s)':>16} | {'Idle Time (s)':>16}")
    print("-" * 70)
    for c in components:
        print(f"{c:>16} | {throughput[components.index(c)]:>16.2f} | {avg_latency[components.index(c)]:>16.4f} | {idle_time[components.index(c)]:>16.2f}")
