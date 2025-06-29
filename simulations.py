"""Simulating the performance of different system approaches.

Focus only on "data collection" in async RL.
"""

import asyncio
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
    batch_size: int = 4
    max_steps: int = 2
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


class Entity:
    def __init__(self, name: str, step_low: float, step_high: float):
        self.step_low = step_low
        self.step_high = step_high
        self.name = name

    async def step(self, data):
        sleep_time = self.step_low + (self.step_high - self.step_low) * random.random()
        await asyncio.sleep(sleep_time)
        return data


class Controller:
    """A controller for a group of entities, with round-robin load balancing."""
    def __init__(self, num_entities: int, name: str, step_low: float, step_high: float):
        self.num_entities = num_entities
        self.entities = [
            Entity(name=f"{name}_{i}", step_low=step_low, step_high=step_high) for i in range(num_entities)
        ]
        self._rr_idx = 0

    async def step(self, data):
        # Round-robin selection
        entity = self.entities[self._rr_idx]
        self._rr_idx = (self._rr_idx + 1) % self.num_entities
        return await entity.step(data)



class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'policy': {'processed': 0, 'latencies': [], 'idle_time': 0.0},
            'coding_verifier': {'processed': 0, 'latencies': [], 'idle_time': 0.0},
            'math_verifier': {'processed': 0, 'latencies': [], 'idle_time': 0.0},
            'judger': {'processed': 0, 'latencies': [], 'idle_time': 0.0},
            'preprocess': {'processed': 0, 'latencies': [], 'idle_time': 0.0},
            'trainer': {'processed': 0, 'latencies': [], 'idle_time': 0.0},
        }
        self.start_time = time.time()

    def record(self, component, latency):
        self.metrics[component]['processed'] += 1
        self.metrics[component]['latencies'].append(latency)

    def record_idle(self, component, idle_time):
        self.metrics[component]['idle_time'] += idle_time

    def report(self):
        elapsed = time.time() - self.start_time
        print("--- Metrics Report ---")
        for component, data in self.metrics.items():
            processed = data['processed']
            throughput = processed / elapsed if elapsed > 0 else 0
            avg_latency = sum(data['latencies']) / len(data['latencies']) if data['latencies'] else 0
            idle = data['idle_time']
            print(f"{component:>12}: Throughput: {throughput:.2f}/s, Avg Latency: {avg_latency:.2f}s, Idle: {idle:.2f}s, Processed: {processed}")
        print("----------------------")


class PipelinedRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        # Queues between pipeline stages
        self.policy_queue = asyncio.Queue()
        self.coding_verifier_queue = asyncio.Queue()
        self.math_verifier_queue = asyncio.Queue()
        self.judger_queue = asyncio.Queue()
        self.preprocess_queue = asyncio.Queue()
        self.replay_buffer = asyncio.Queue()
        self.weight_store = asyncio.Queue()
        # Controllers for each component
        self.policy_controller = Controller(
            num_entities=config.num_policys,
            name="policy",
            step_low=config.policy_step_low,
            step_high=config.policy_step_high,
        )
        self.coding_verifier_controller = Controller(
            num_entities=config.num_coding_verifiers,
            name="coding_verifier",
            step_low=config.coding_verifier_step_low,
            step_high=config.coding_verifier_step_high,
        )
        self.math_verifier_controller = Controller(
            num_entities=config.num_math_verifiers,
            name="math_verifier",
            step_low=config.math_verifier_step_low,
            step_high=config.math_verifier_step_high,
        )
        self.judger_controller = Controller(
            num_entities=config.num_llm_judge_judgers,
            name="llm_judge_judger",
            step_low=config.llm_judge_judger_step_low,
            step_high=config.llm_judge_judger_step_high,
        )
        self.preprocess_controller = Controller(
            num_entities=1,
            name="preprocess",
            step_low=config.preprocess_step_low,
            step_high=config.preprocess_step_high,
        )
        # Metrics
        self.metrics = MetricsCollector()
        # Shutdown event
        self.shutdown_event = asyncio.Event()

    async def prompt_loader(self, prompts):
        for prompt in prompts:
            if self.shutdown_event.is_set():
                break
            t0 = time.time()
            await self.policy_queue.put((prompt, t0))

    async def policy_loop(self):
        while not self.shutdown_event.is_set():
            idle_start = time.time()
            try:
                data, t0 = await asyncio.wait_for(self.policy_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            idle_end = time.time()
            self.metrics.record_idle('policy', idle_end - idle_start)
            t1 = time.time()
            action = await self.policy_controller.step(data)
            t2 = time.time()
            self.metrics.record('policy', t2 - t1)
            route = random.choice(["coding", "math", "judger", "preprocess"])
            if route == "coding":
                await self.coding_verifier_queue.put((action, t0))
            elif route == "math":
                await self.math_verifier_queue.put((action, t0))
            elif route == "judger":
                await self.judger_queue.put((action, t0))
            else:
                await self.preprocess_queue.put((action, t0))

    async def verifier_loop(self, queue, controller, metrics_name):
        while not self.shutdown_event.is_set():
            idle_start = time.time()
            try:
                action, t0 = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            idle_end = time.time()
            self.metrics.record_idle(metrics_name, idle_end - idle_start)
            t1 = time.time()
            verification = await controller.step(action)
            t2 = time.time()
            self.metrics.record(metrics_name, t2 - t1)
            await self.policy_queue.put((verification, t0))

    async def preprocess_loop(self):
        while not self.shutdown_event.is_set():
            idle_start = time.time()
            try:
                action, t0 = await asyncio.wait_for(self.preprocess_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            idle_end = time.time()
            self.metrics.record_idle('preprocess', idle_end - idle_start)
            t1 = time.time()
            preprocessed = await self.preprocess_controller.step(action)
            t2 = time.time()
            self.metrics.record('preprocess', t2 - t1)
            await self.replay_buffer.put((preprocessed, t0))

    async def train_loop(self):
        steps = 0
        while steps < self.config.max_steps:
            batch = []
            idle_start = time.time()
            for _ in range(self.config.batch_size):
                preprocessed, t0 = await self.replay_buffer.get()
                batch.append((preprocessed, t0))
            idle_end = time.time()
            self.metrics.record_idle('trainer', idle_end - idle_start)
            t1 = time.time()
            await asyncio.sleep(0.1)
            t2 = time.time()
            for _, t0 in batch:
                self.metrics.record('trainer', t2 - t1)
            await self.weight_store.put("weights")
            steps += 1
        # Signal shutdown to all other loops
        self.shutdown_event.set()

    async def run(self, prompts, report_interval=5):
        tasks = [
            asyncio.create_task(self.prompt_loader(prompts)),
            asyncio.create_task(self.policy_loop()),
            asyncio.create_task(functools.partial(self.verifier_loop, self.coding_verifier_queue, self.coding_verifier_controller, 'coding_verifier')()),
            asyncio.create_task(functools.partial(self.verifier_loop, self.math_verifier_queue, self.math_verifier_controller, 'math_verifier')()),
            asyncio.create_task(functools.partial(self.verifier_loop, self.judger_queue, self.judger_controller, 'judger')()),
            asyncio.create_task(self.preprocess_loop()),
            asyncio.create_task(self.train_loop()),
        ]
        async def reporter():
            while not self.shutdown_event.is_set():
                await asyncio.sleep(report_interval)
                self.metrics.report()
        tasks.append(asyncio.create_task(reporter()))
        # Wait for train_loop to finish
        await tasks[-2]
        # Cancel all other tasks
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class RolloutRunner:
    pass

# Worker function for each replica
def worker_loop(input_queue, output_queue, trace_queue, component_name, worker_idx, step_low, step_high, policy_queue=None):
    while True:
        item = input_queue.get()
        if item is None:
            break
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
            "pid": os.getpid(),
            "tid": worker_idx,
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
            # Multi-turn: send back to policy
            policy_queue.put((req, t0))
        elif component_name == "preprocess":
            output_queue.put((req, t0))
        # Trainer handled separately

# Trainer worker (batches)
def trainer_loop(input_queue, trace_queue, batch_size, max_steps):
    steps = 0
    while steps < max_steps:
        batch = []
        for _ in range(batch_size):
            item = input_queue.get()
            if item is None:
                return
            req, t0 = item
            batch.append((req, t0))
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
                "pid": os.getpid(),
                "tid": 0,
                "args": {"request": req.summary()},
            })
        steps += 1
    # Signal done
    return

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

    def start_workers(self):
        cfg = self.config
        # Policy workers
        for i in range(cfg.num_policys):
            p = mp.Process(target=worker_loop, args=(self.policy_queue, {
                "coding_verifier": self.coding_verifier_queue,
                "math_verifier": self.math_verifier_queue,
                "judger": self.judger_queue,
                "preprocess": self.preprocess_queue
            }, self.trace_queue, "policy", i, cfg.policy_step_low, cfg.policy_step_high))
            p.start()
            self.policy_workers.append(p)
        # Coding verifier workers
        for i in range(cfg.num_coding_verifiers):
            p = mp.Process(target=worker_loop, args=(self.coding_verifier_queue, None, self.trace_queue, "coding_verifier", i, cfg.coding_verifier_step_low, cfg.coding_verifier_step_high, self.policy_queue))
            p.start()
            self.coding_verifier_workers.append(p)
        # Math verifier workers
        for i in range(cfg.num_math_verifiers):
            p = mp.Process(target=worker_loop, args=(self.math_verifier_queue, None, self.trace_queue, "math_verifier", i, cfg.math_verifier_step_low, cfg.math_verifier_step_high, self.policy_queue))
            p.start()
            self.math_verifier_workers.append(p)
        # Judger workers
        for i in range(cfg.num_llm_judge_judgers):
            p = mp.Process(target=worker_loop, args=(self.judger_queue, None, self.trace_queue, "judger", i, cfg.llm_judge_judger_step_low, cfg.llm_judge_judger_step_high, self.policy_queue))
            p.start()
            self.judger_workers.append(p)
        # Preprocess workers
        for i in range(cfg.num_preprocess):
            p = mp.Process(target=worker_loop, args=(self.preprocess_queue, self.trainer_queue, self.trace_queue, "preprocess", i, cfg.preprocess_step_low, cfg.preprocess_step_high))
            p.start()
            self.preprocess_workers.append(p)
        # Trainer
        self.trainer = mp.Process(target=trainer_loop, args=(self.trainer_queue, self.trace_queue, cfg.batch_size, cfg.max_steps))
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

    def run(self, prompts):
        self.start_workers()
        # Feed prompts
        for idx, prompt in enumerate(prompts):
            req = DeepResearchRequest(initial_prompt=prompt, rank=idx)
            self.policy_queue.put((req, time.time()))
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
        trace_output="trace.json",
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
