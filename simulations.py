"""Simulating the performance of different system approaches.

Focus only on "data collection" in async RL.
"""

import asyncio
from dataclasses import dataclass, field
import random
import time
import functools
import matplotlib.pyplot as plt
import json
import os
from typing import List, Optional


@dataclass
class SimulationConfig:
    num_policys: int = 1
    num_coding_verifiers: int = 1
    num_math_verifiers: int = 1
    num_llm_judge_judgers: int = 1
    batch_size: int = 4
    max_steps: int = 100  # Number of batches to train before stopping
    trace_output: str = "trace.json"  # Output file for Perfetto trace

    # Step time ranges for each component (in seconds)
    policy_step_low: float = 1.0
    policy_step_high: float = 2.0

    coding_verifier_step_low: float = 0.5
    coding_verifier_step_high: float = 1.0

    math_verifier_step_low: float = 0.8
    math_verifier_step_high: float = 1.5

    llm_judge_judger_step_low: float = 0.3
    llm_judge_judger_step_high: float = 0.7

    preprocess_step_low: float = 0.2
    preprocess_step_high: float = 0.5


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
        idx = self._rr_idx
        self._rr_idx = (self._rr_idx + 1) % self.num_entities
        result = await entity.step(data)
        return result, idx


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
        self.trace_events = []  # For Perfetto
        self.pid_map = {k: i for i, k in enumerate(self.metrics.keys())}
        self.tid_counter = 0

    def record(self, component, latency, t_start=None, t_end=None, event_type="step", extra_args=None, request=None, replica_idx=0):
        self.metrics[component]['processed'] += 1
        self.metrics[component]['latencies'].append(latency)
        if t_start is not None and t_end is not None:
            trace_args = extra_args or {}
            trace_args['component'] = component
            rank = getattr(request, 'rank', None)
            if request is not None:
                trace_args['request'] = request.summary() if hasattr(request, 'summary') else str(request)
            event_name = f"{component}[{replica_idx}]::{event_type}"
            self.trace_events.append({
                "name": event_name,
                "ph": "X",
                "ts": int(t_start * 1e6),
                "dur": int((t_end - t_start) * 1e6),
                "pid": self.pid_map[component],
                "tid": self.pid_map[component],
                "args": trace_args,
            })

    def record_idle(self, component, idle_time, t_start=None, t_end=None, request=None, replica_idx=0):
        self.metrics[component]['idle_time'] += idle_time
        if t_start is not None and t_end is not None:
            trace_args = {"component": component}
            rank = getattr(request, 'rank', None)
            if request is not None:
                trace_args['request'] = request.summary() if hasattr(request, 'summary') else str(request)
            event_name = f"{component}[{replica_idx}]::idle"
            self.trace_events.append({
                "name": event_name,
                "ph": "X",
                "ts": int(t_start * 1e6),
                "dur": int((t_end - t_start) * 1e6),
                "pid": self.pid_map[component],
                "tid": self.pid_map[component],
                "args": trace_args,
            })

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
        for idx, prompt in enumerate(prompts):
            if self.shutdown_event.is_set():
                break
            t0 = time.time()
            req = DeepResearchRequest(initial_prompt=prompt, rank=idx)
            await self.policy_queue.put((req, t0))

    async def policy_loop(self):
        while not self.shutdown_event.is_set():
            idle_start = time.time()
            try:
                req, t0 = await asyncio.wait_for(self.policy_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            idle_end = time.time()
            self.metrics.record_idle('policy', idle_end - idle_start, t_start=idle_start, t_end=idle_end, request=req, replica_idx=0)
            t1 = time.time()
            req.latest_turn += 1
            req.events.append('policy')
            req.action = 'policy_action'
            action, worker_idx = await self.policy_controller.step(req)
            t2 = time.time()
            self.metrics.record('policy', t2 - t1, t_start=t1, t_end=t2, event_type="step", extra_args={"input": str(req.initial_prompt)}, request=req, replica_idx=worker_idx)
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
                req, t0 = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            idle_end = time.time()
            self.metrics.record_idle(metrics_name, idle_end - idle_start, t_start=idle_start, t_end=idle_end, request=req, replica_idx=0)
            t1 = time.time()
            req.latest_turn += 1
            req.events.append(metrics_name)
            req.action = f'{metrics_name}_action'
            verification, worker_idx = await controller.step(req)
            t2 = time.time()
            self.metrics.record(metrics_name, t2 - t1, t_start=t1, t_end=t2, event_type="step", extra_args={"input": str(req.initial_prompt)}, request=req, replica_idx=worker_idx)
            await self.policy_queue.put((verification, t0))

    async def preprocess_loop(self):
        while not self.shutdown_event.is_set():
            idle_start = time.time()
            try:
                req, t0 = await asyncio.wait_for(self.preprocess_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            idle_end = time.time()
            self.metrics.record_idle('preprocess', idle_end - idle_start, t_start=idle_start, t_end=idle_end, request=req, replica_idx=0)
            t1 = time.time()
            req.latest_turn += 1
            req.events.append('preprocess')
            req.action = 'preprocess_action'
            preprocessed, worker_idx = await self.preprocess_controller.step(req)
            t2 = time.time()
            self.metrics.record('preprocess', t2 - t1, t_start=t1, t_end=t2, event_type="step", extra_args={"input": str(req.initial_prompt)}, request=req, replica_idx=worker_idx)
            await self.replay_buffer.put((preprocessed, t0))

    async def train_loop(self):
        steps = 0
        while steps < self.config.max_steps:
            batch = []
            idle_start = time.time()
            for _ in range(self.config.batch_size):
                req, t0 = await self.replay_buffer.get()
                batch.append((req, t0))
            idle_end = time.time()
            self.metrics.record_idle('trainer', idle_end - idle_start, t_start=idle_start, t_end=idle_end, request=batch[0][0] if batch else None, replica_idx=0)
            t1 = time.time()
            await asyncio.sleep(0.1)
            t2 = time.time()
            for req, t0 in batch:
                req.latest_turn += 1
                req.events.append('trainer')
                req.action = 'train_action'
                self.metrics.record('trainer', t2 - t1, t_start=t1, t_end=t2, event_type="step", extra_args={}, request=req, replica_idx=0)
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

if __name__ == "__main__":
    import sys
    import asyncio

    # Example prompts
    prompts = [f"prompt_{i}" for i in range(100)]

    # Example config (customize as needed)
    config = SimulationConfig(
        num_policys=2,
        num_coding_verifiers=2,
        num_math_verifiers=2,
        num_llm_judge_judgers=1,
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

    runner = PipelinedRunner(config)
    try:
        asyncio.run(runner.run(prompts, report_interval=5))
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        sys.exit(0)

    # Plot results
    metrics = runner.metrics.metrics
    components = list(metrics.keys())
    throughput = []
    avg_latency = []
    idle_time = []
    for comp in components:
        processed = metrics[comp]['processed']
        elapsed = time.time() - runner.metrics.start_time
        throughput.append(processed / elapsed if elapsed > 0 else 0)
        avg_latency.append(sum(metrics[comp]['latencies']) / len(metrics[comp]['latencies']) if metrics[comp]['latencies'] else 0)
        idle_time.append(metrics[comp]['idle_time'])

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].bar(components, throughput)
    axs[0].set_title('Throughput (processed/s)')
    axs[1].bar(components, avg_latency)
    axs[1].set_title('Average Latency (s)')
    axs[2].bar(components, idle_time)
    axs[2].set_title('Total Idle Time (s)')
    plt.tight_layout()
    plt.show()

    # Write Perfetto trace
    with open(config.trace_output, "w") as f:
        json.dump({"traceEvents": runner.metrics.trace_events}, f)
    print(f"Perfetto trace written to {os.path.abspath(config.trace_output)}")
    print("To view, open https://ui.perfetto.dev and load the trace file.")
